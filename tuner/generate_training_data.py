"""
從 PGN 文件生成訓練數據集的腳本。(已整合 Stockfish 數據清洗功能，支援多進程並行處理)

這個腳本的目的是為 `tuner.py` 調優腳本創建一個數據集，此版本專為
Texel's Tuning Method 設計。它通過以下步驟工作：
1.  讀取 PGN 文件並解析對局結果。
2.  將對局分配給多個工作進程 (Worker Process)。
3.  每個 Worker 使用獨立的 Stockfish 引擎實例進行並行分析與清洗。
4.  主進程收集結果並寫入 JSON Lines 文件。

數據清洗邏輯更新 (Score-based Labeling):
不依賴 PGN 的原始結果，而是完全根據 Stockfish 的靜態評分來賦予標籤：
- Score > 100 cp: Result = 1.0 (白勝)
- Score < -100 cp: Result = 0.0 (黑勝)
- -100 <= Score <= 100 cp: Result = 0.5 (和棋)
這能讓引擎學會識別明顯的勝勢。
"""
import chess
import chess.pgn
import chess.engine
import json
import sys
import os
import argparse
import shutil
import concurrent.futures
import multiprocessing

# --- 全局變數 ---
# 這些變數現在只在 worker 進程中被使用
STOCKFISH_ENGINE = None

def parse_result(result_str: str) -> float:
    # 雖然新的邏輯不依賴 PGN 結果進行驗證，但我們仍然需要解析它來過濾掉未完成的比賽嗎？
    # 不，新的邏輯是 "Label Override"，只要局面合法且 Stockfish 能給分，我們就用 Stockfish 的分。
    # 但為了避免解析到 header 損壞的局，還是保留基本的解析，或者直接忽略。
    # 原代碼用它來做初步過濾，我們這裡可以放寬，但保留此函數以防萬一需要。
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
    
    # 這裡的材質檢查是用於過濾 "明顯材質失衡" 的局面嗎？
    # 原代碼 logic: absolute diff > 5 pawn values -> skip?
    # 這似乎是用來過濾掉已經太過懸殊的局面，避免過擬合？
    # 或者是在靜態評估訓練中，我們希望專注於材質相對平衡但有位置優勢的局面？
    # 用戶指示 "保留過濾條件"。
    material_diff = 0
    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    for piece_type, value in piece_values.items():
        material_diff += len(board.pieces(piece_type, chess.WHITE)) * value
        material_diff -= len(board.pieces(piece_type, chess.BLACK)) * value
    
    if abs(material_diff) > 5:
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

def get_stockfish_label(board: chess.Board, depth: int) -> float:
    """
    使用 Stockfish 分析局面並返回強制標籤 (Label Override)。
    
    Returns:
        1.0 if score > 100 cp (White winning)
        0.0 if score < -100 cp (Black winning)
        0.5 if -100 <= score <= 100 (Draw/Equal)
        None if analysis fails
    """
    if STOCKFISH_ENGINE is None:
        raise RuntimeError("Stockfish engine is not initialized.")

    try:
        info = STOCKFISH_ENGINE.analyse(board, chess.engine.Limit(depth=depth))
        score_obj = info["score"].white()
        
        if score_obj.is_mate():
            # Mate is always a win (> 100) or loss (< -100)
            score = 30000 if score_obj.mate() > 0 else -30000
        else:
            score = score_obj.score()

        # Score-based Labeling Logic
        if score > 100:
            return 1.0
        elif score < -100:
            return 0.0
        else:
            return 0.5
            
    except (chess.engine.EngineTerminatedError, chess.engine.EngineError):
        return None

# --- Worker Function for Parallel Processing ---

def process_game_batch(games_data, stockfish_path, depth, threshold, min_move, max_move):
    """
    Worker function to process a batch of games.
    games_data: list of (game_headers, moves_list) tuples.
    """
    # Initialize engine for this worker
    setup_stockfish_engine(stockfish_path)
    
    mg_results = []
    eg_results = []
    discarded = 0
    
    try:
        for headers, moves_san in games_data:
            # We don't strictly need the PGN result anymore for validation,
            # but we still might want to ensure the game has a valid result structure?
            # For now, we process all games provided.
            
            # Reconstruct board
            board = chess.Board() 
            if "FEN" in headers:
                board.set_fen(headers["FEN"])

            for i, move_san in enumerate(moves_san):
                try:
                    move = board.parse_san(move_san)
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
                        
                        # Use Stockfish to generate the label directly
                        label = get_stockfish_label(board, depth)
                        
                        if label is not None:
                            fen = board.fen()
                            data = json.dumps({"fen": fen, "result": label})
                            
                            if is_endgame_position(board):
                                eg_results.append(data)
                            else:
                                mg_results.append(data)
                        else:
                            discarded += 1
    finally:
        if STOCKFISH_ENGINE:
            STOCKFISH_ENGINE.quit()
            
    return mg_results, eg_results, discarded

def generate_data_parallel(pgn_file, output_mg, output_eg, min_move, max_move, games_limit, depth, threshold, stockfish_path, num_workers=None):
    """
    Main driver for parallel generation.
    """
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1) 
        
    print(f"Starting parallel generation with {num_workers} workers...", file=sys.stderr)
    
    BATCH_SIZE = 50 
    
    games_read = 0
    mg_total = 0
    eg_total = 0
    discarded_total = 0
    
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
                
            games_read += 1
            
            headers = dict(game.headers)
            moves = [move.uci() for move in game.mainline_moves()] 
            
            current_batch.append((headers, moves))
            
            if len(current_batch) >= BATCH_SIZE:
                future = executor.submit(process_game_batch, current_batch, stockfish_path, depth, threshold, min_move, max_move)
                futures.append(future)
                current_batch = []
                
        if current_batch:
            future = executor.submit(process_game_batch, current_batch, stockfish_path, depth, threshold, min_move, max_move)
            futures.append(future)
            
        print(f"All games read. Waiting for {len(futures)} tasks to complete...", file=sys.stderr)
        
        for future in concurrent.futures.as_completed(futures):
            try:
                mg_res, eg_res, discarded = future.result()
                
                for line in mg_res:
                    f_mg.write(line + '\n')
                for line in eg_res:
                    f_eg.write(line + '\n')
                    
                mg_total += len(mg_res)
                eg_total += len(eg_res)
                discarded_total += discarded
                
                total_kept = mg_total + eg_total
                print(f"\rProcessed Positions (Kept+Discarded): {total_kept + discarded_total} | MG: {mg_total} | EG: {eg_total}", end="", file=sys.stderr)
                     
            except Exception as e:
                print(f"\nWorker exception: {e}", file=sys.stderr)

    print(f"\n--- Data Generation Complete ---", file=sys.stderr)
    print(f"Total Games Read: {games_read}", file=sys.stderr)
    print(f"Middlegame positions saved: {mg_total}", file=sys.stderr)
    print(f"Endgame positions saved: {eg_total}", file=sys.stderr)
    print(f"Positions discarded: {discarded_total}", file=sys.stderr)

if __name__ == '__main__':
    multiprocessing.freeze_support()

    default_stockfish = shutil.which('stockfish')
    if not default_stockfish:
        if os.path.exists("./stockfish/stockfish-windows-x86-64-avx2.exe"):
             default_stockfish = "./stockfish/stockfish-windows-x86-64-avx2.exe"
        else:
             default_stockfish = "stockfish"

    default_pgn = "twic1613.pgn"
    if not os.path.exists(default_pgn) and os.path.exists(os.path.join("tuner", "twic1613.pgn")):
        default_pgn = os.path.join("tuner", "twic1613.pgn")

    parser = argparse.ArgumentParser(
        description="Generate cleaned chess training data (Parallel Split MG/EG) using Stockfish Labeling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--pgn", type=str, default=default_pgn, help="Input PGN file.")
    parser.add_argument("--output-mg", type=str, default="training_data_middlegame_cleaned.jsonl", help="Output MG file.")
    parser.add_argument("--output-eg", type=str, default="training_data_endgame_cleaned.jsonl", help="Output EG file.")
    parser.add_argument("--min-move", type=int, default=8, help="Start move number.")
    parser.add_argument("--max-move", type=int, default=200, help="End move number.")
    parser.add_argument("--games", type=int, default=1000, help="Games limit (0 for all).")
    
    parser.add_argument("--stockfish-path", type=str, default=default_stockfish, help="Stockfish executable path.")
    parser.add_argument("--stockfish-depth", type=int, default=10, help="Stockfish analysis depth.")
    parser.add_argument("--consistency-threshold", type=int, default=30, help="Draw consistency threshold (cp). (Deprecated in score-based labeling)")
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
        threshold=args.consistency_threshold,
        stockfish_path=args.stockfish_path,
        num_workers=args.workers
    )
