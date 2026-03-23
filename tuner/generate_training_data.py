"""
從 PGN 文件生成訓練數據集的腳本。(已整合 Stockfish 數據清洗功能，支援多進程並行處理)

這個腳本的目的是為 `tuner.py` 調優腳本創建一個數據集，此版本專為
Texel's Tuning Method 設計。它通過以下步驟工作：
1.  讀取 PGN 文件並解析對局結果。
2.  將對局分配給多個工作進程 (Worker Process)。
3.  每個 Worker 使用獨立的 Stockfish 引擎實例進行並行分析與清洗。
4.  主進程收集結果並寫入 JSON Lines 文件。

數據清洗邏輯更新 (Continuous Sigmoid Labeling):
使用 Stockfish 的 centipawn 分數轉換為 Sigmoid 值作為連續標籤：
- Sigmoid(score) = 1 / (1 + 10^(-score/400))
- 過濾掉 |score| > max_cp 的局面（預設 200cp = ±2.0 分）
- 過濾掉雙方 ELO 低於門檻的對局（預設 2400）
- 跳過前 N 步開局（預設 12 步）
"""
import chess
import chess.pgn
import chess.engine
import json
import sys
import os
import argparse
import shutil
import math
import concurrent.futures
import multiprocessing

# --- 全局變數 ---
# 這些變數現在只在 worker 進程中被使用
STOCKFISH_ENGINE = None

def parse_result(result_str: str) -> float:
    """解析 PGN 結果字串。保留以防需要過濾未完成的對局。"""
    if result_str == '1-0':
        return 1.0
    elif result_str == '0-1':
        return 0.0
    elif result_str == '1/2-1/2':
        return 0.5
    else:
        return None

def should_skip_position(board):
    if board.is_check():
        return True
    
    # 過濾明顯材質失衡的局面（超過 2 分差距，約等於兩個兵）
    material_diff = 0
    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    for piece_type, value in piece_values.items():
        material_diff += len(board.pieces(piece_type, chess.WHITE)) * value
        material_diff -= len(board.pieces(piece_type, chess.BLACK)) * value
    
    if abs(material_diff) > 2:
        return True

    return False

def is_endgame_position(board: chess.Board) -> bool:
    no_queens = not board.pieces(chess.QUEEN, chess.WHITE) and not board.pieces(chess.QUEEN, chess.BLACK)
    if no_queens:
        return True

    white_pieces = len(board.pieces(chess.ROOK, chess.WHITE)) * 2 + len(board.pieces(chess.KNIGHT, chess.WHITE)) + len(board.pieces(chess.BISHOP, chess.WHITE))
    black_pieces = len(board.pieces(chess.ROOK, chess.BLACK)) * 2 + len(board.pieces(chess.KNIGHT, chess.BLACK)) + len(board.pieces(chess.BISHOP, chess.BLACK))

    if (white_pieces + black_pieces) <= 7:
        return True

    return False

def setup_stockfish_engine(path: str):
    global STOCKFISH_ENGINE
    if path == 'stockfish' or path == 'stockfish.exe':
        detected = shutil.which('stockfish')
        if detected:
            path = detected
            
    try:
        STOCKFISH_ENGINE = chess.engine.SimpleEngine.popen_uci(path)
    except FileNotFoundError:
        print(f"Error: Stockfish executable not found at the specified path: {path}", file=sys.stderr)
        sys.exit(1)

def get_stockfish_sigmoid_label(board: chess.Board, depth: int, max_cp: int = 200) -> float:
    """
    使用 Stockfish 分析局面並返回連續 Sigmoid 標籤。
    
    過濾掉 |score| > max_cp 的局面（太過懸殊，對訓練無益）。
    
    Returns:
        Sigmoid(score) 介於 0.0~1.0 之間的連續值
        None if |score| > max_cp, analysis fails, or mate detected
    """
    if STOCKFISH_ENGINE is None:
        raise RuntimeError("Stockfish engine is not initialized.")

    try:
        info = STOCKFISH_ENGINE.analyse(board, chess.engine.Limit(depth=depth))
        score_obj = info["score"].white()
        
        if score_obj.is_mate():
            # Mate 局面一定超過 max_cp 閾值，直接過濾
            return None
        
        score = score_obj.score()
        
        # 過濾掉過於懸殊的局面
        if abs(score) > max_cp:
            return None
        
        # 轉換為 Sigmoid: 1 / (1 + 10^(-score/400))
        sigmoid = 1.0 / (1.0 + math.pow(10.0, -score / 400.0))
        return sigmoid
            
    except (chess.engine.EngineTerminatedError, chess.engine.EngineError):
        return None

def check_elo(headers, min_elo):
    """
    檢查雙方 ELO 是否都高於門檻。
    如果 PGN 沒有 ELO 資訊，則放行（回傳 True）。
    
    Args:
        headers: PGN game headers dict
        min_elo: 最低 ELO 門檻（0 表示不篩選）
    
    Returns:
        True if both players meet the ELO threshold
    """
    if min_elo <= 0:
        return True
    
    try:
        white_elo = int(headers.get("WhiteElo", "0"))
        black_elo = int(headers.get("BlackElo", "0"))
    except (ValueError, TypeError):
        # 無法解析 ELO，放行
        return True
    
    # 如果兩方都是 0（無 ELO 資訊），放行
    if white_elo == 0 and black_elo == 0:
        return True
    
    return white_elo >= min_elo and black_elo >= min_elo

# --- Worker Function for Parallel Processing ---

def process_game_batch(games_data, stockfish_path, depth, max_cp, min_move, max_move):
    """
    Worker function to process a batch of games.
    games_data: list of (game_headers, moves_uci_list) tuples.
    """
    # Initialize engine for this worker
    setup_stockfish_engine(stockfish_path)
    
    mg_results = []
    eg_results = []
    discarded = 0
    filtered_by_score = 0
    
    try:
        for headers, moves_uci in games_data:
            # Reconstruct board
            board = chess.Board() 
            if "FEN" in headers:
                board.set_fen(headers["FEN"])

            for i, move_uci in enumerate(moves_uci):
                try:
                    move = chess.Move.from_uci(move_uci)
                    if move not in board.legal_moves:
                        break
                except ValueError:
                    break 
                
                # Check quiescence before push
                is_capture_or_promotion = board.is_capture(move) or move.promotion
                
                board.push(move)
                
                # Filter Quiescence (skip positions resulting from capture)
                if is_capture_or_promotion:
                    continue

                if min_move <= board.fullmove_number <= max_move:
                    if not board.is_checkmate() and not board.is_stalemate() and not board.is_insufficient_material() and not should_skip_position(board):
                        
                        # Use Stockfish to generate the continuous sigmoid label
                        label = get_stockfish_sigmoid_label(board, depth, max_cp)
                        
                        if label is not None:
                            fen = board.fen()
                            data = json.dumps({"fen": fen, "result": round(label, 6)})
                            
                            if is_endgame_position(board):
                                eg_results.append(data)
                            else:
                                mg_results.append(data)
                        else:
                            filtered_by_score += 1
                    else:
                        discarded += 1
    finally:
        if STOCKFISH_ENGINE:
            STOCKFISH_ENGINE.quit()
            
    return mg_results, eg_results, discarded, filtered_by_score

def generate_data_parallel(pgn_file, output_mg, output_eg, min_move, max_move, games_limit, depth, max_cp, min_elo, stockfish_path, num_workers=None):
    """
    Main driver for parallel generation.
    """
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1) 
        
    print(f"Starting parallel generation with {num_workers} workers...", file=sys.stderr)
    print(f"Settings: min_move={min_move}, max_move={max_move}, max_cp={max_cp}, min_elo={min_elo}, depth={depth}", file=sys.stderr)
    
    BATCH_SIZE = 50 
    
    games_read = 0
    games_skipped_elo = 0
    mg_total = 0
    eg_total = 0
    discarded_total = 0
    filtered_by_score_total = 0
    
    current_batch = []
    futures = []
    
    with open(pgn_file, encoding='utf-8') as pgn, \
         open(output_mg, 'w', encoding='utf-8') as f_mg, \
         open(output_eg, 'w', encoding='utf-8') as f_eg, \
         concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        
        while True:
            if games_limit and games_read >= games_limit:
                break
                
            try:
                game = chess.pgn.read_game(pgn)
            except ValueError:
                continue
                
            if game is None:
                break
            
            headers = dict(game.headers)
            
            # ELO 篩選
            if not check_elo(headers, min_elo):
                games_skipped_elo += 1
                continue
                
            games_read += 1
            
            # 使用 UCI 格式（與 process_game_batch 中的 Move.from_uci 對應）
            moves = [move.uci() for move in game.mainline_moves()] 
            
            current_batch.append((headers, moves))
            
            if len(current_batch) >= BATCH_SIZE:
                future = executor.submit(process_game_batch, current_batch, stockfish_path, depth, max_cp, min_move, max_move)
                futures.append(future)
                current_batch = []
                
        if current_batch:
            future = executor.submit(process_game_batch, current_batch, stockfish_path, depth, max_cp, min_move, max_move)
            futures.append(future)
            
        print(f"All games read ({games_read} accepted, {games_skipped_elo} skipped by ELO). Waiting for {len(futures)} tasks to complete...", file=sys.stderr)
        
        for future in concurrent.futures.as_completed(futures):
            try:
                mg_res, eg_res, discarded, filtered_by_score = future.result()
                
                for line in mg_res:
                    f_mg.write(line + '\n')
                for line in eg_res:
                    f_eg.write(line + '\n')
                    
                mg_total += len(mg_res)
                eg_total += len(eg_res)
                discarded_total += discarded
                filtered_by_score_total += filtered_by_score
                
                total_kept = mg_total + eg_total
                print(f"\rKept: {total_kept} (MG: {mg_total} | EG: {eg_total}) | Filtered(score): {filtered_by_score_total} | Discarded: {discarded_total}", end="", file=sys.stderr)
                     
            except Exception as e:
                print(f"\nWorker exception: {e}", file=sys.stderr)

    print(f"\n--- Data Generation Complete ---", file=sys.stderr)
    print(f"Total Games Read: {games_read} (Skipped by ELO: {games_skipped_elo})", file=sys.stderr)
    print(f"Middlegame positions saved: {mg_total}", file=sys.stderr)
    print(f"Endgame positions saved: {eg_total}", file=sys.stderr)
    print(f"Positions filtered by score (|cp| > {max_cp}): {filtered_by_score_total}", file=sys.stderr)
    print(f"Positions discarded (other): {discarded_total}", file=sys.stderr)

if __name__ == '__main__':
    multiprocessing.freeze_support()

    default_stockfish = shutil.which('stockfish')
    if not default_stockfish:
        if os.path.exists("./stockfish/stockfish-windows-x86-64-avx2.exe"):
             default_stockfish = "./stockfish/stockfish-windows-x86-64-avx2.exe"
        else:
             default_stockfish = "stockfish"

    default_pgn = "tuner/twic1613.pgn"
    if not os.path.exists(default_pgn) and os.path.exists(os.path.join("tuner", "twic1613.pgn")):
        default_pgn = os.path.join("tuner", "twic1613.pgn")

    parser = argparse.ArgumentParser(
        description="Generate cleaned chess training data (Parallel Split MG/EG) using Stockfish Sigmoid Labeling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--pgn", type=str, default=default_pgn, help="Input PGN file.")
    parser.add_argument("--output-mg", type=str, default="tuner/training_data_middlegame_cleaned.jsonl", help="Output MG file.")
    parser.add_argument("--output-eg", type=str, default="tuner/training_data_endgame_cleaned.jsonl", help="Output EG file.")
    parser.add_argument("--min-move", type=int, default=12, help="Start move number (skip opening).")
    parser.add_argument("--max-move", type=int, default=200, help="End move number.")
    parser.add_argument("--games", type=int, default=0, help="Games limit (0 for all).")
    
    # Stockfish 相關
    parser.add_argument("--stockfish-path", type=str, default=default_stockfish, help="Stockfish executable path.")
    parser.add_argument("--stockfish-depth", type=int, default=12, help="Stockfish analysis depth.")
    
    # 篩選相關
    parser.add_argument("--max-cp", type=int, default=200, help="Maximum absolute centipawn score to keep (filters extreme positions). 200 = ±2.0 pawns.")
    parser.add_argument("--min-elo", type=int, default=2400, help="Minimum ELO for both players (0 to disable).")
    
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes (default: CPU count - 1).")

    args = parser.parse_args()

    if not os.path.exists(args.pgn):
        print(f"Error: PGN file not found at '{args.pgn}'", file=sys.stderr)
        sys.exit(1)

    if not shutil.which(args.stockfish_path) and not os.path.exists(args.stockfish_path):
         print(f"Warning: Stockfish path '{args.stockfish_path}' might be invalid.", file=sys.stderr)

    generate_data_parallel(
        pgn_file=args.pgn,
        output_mg=args.output_mg,
        output_eg=args.output_eg,
        min_move=args.min_move,
        max_move=args.max_move,
        games_limit=args.games if args.games > 0 else None,
        depth=args.stockfish_depth,
        max_cp=args.max_cp,
        min_elo=args.min_elo,
        stockfish_path=args.stockfish_path,
        num_workers=args.workers
    )
