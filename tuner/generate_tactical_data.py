"""
從 PGN 文件生成「戰術/殺王」專用數據集的腳本。(多進程並行處理)

這個腳本的過濾邏輯（黃金條件）：
1. 雙方總材質分差 (Material Difference) <= 100cp (約 1 兵以內)。
2. Stockfish 預測分數 >= 200cp 或 <= -200cp，或是強制將殺 (Mate)。
3. 每盤棋最多只抽取 2 個符合條件的局面，避免過度擬合同一局。

這個資料集專門用於解鎖並優化高度非線性的「國王安全參數」。
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

STOCKFISH_ENGINE = None

def should_skip_position(board):
    if board.is_check():
        return True
    
    # 過濾材質失衡的局面（我們只要分差 <= 100cp 的均勢局面）
    material_diff = 0
    piece_values = {chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 300, chess.ROOK: 500, chess.QUEEN: 900}
    for piece_type, value in piece_values.items():
        material_diff += len(board.pieces(piece_type, chess.WHITE)) * value
        material_diff -= len(board.pieces(piece_type, chess.BLACK)) * value
    
    if abs(material_diff) > 100:
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

def get_stockfish_tactical_label(board: chess.Board, depth: int, min_cp: int = 200) -> float:
    """
    使用 Stockfish 分析局面並返回連續 Sigmoid 標籤。
    只保留具有強烈戰術/殺王潛力的局面。
    
    Returns:
        Sigmoid(score) 介於 0.0~1.0 之間的連續值
        None 如果 |score| < min_cp (平淡局面)
    """
    if STOCKFISH_ENGINE is None:
        raise RuntimeError("Stockfish engine is not initialized.")

    try:
        info = STOCKFISH_ENGINE.analyse(board, chess.engine.Limit(depth=depth))
        score_obj = info["score"].white()
        
        if score_obj.is_mate():
            mate_moves = score_obj.mate()
            if mate_moves > 0:
                return 1.0
            else:
                return 0.0
        
        score = score_obj.score()
        
        # 戰術過濾：我們只想要分數懸殊的局面
        if abs(score) < min_cp:
            return None
            
        # 防止指數溢出
        score = max(min(score, 2000), -2000)
        
        sigmoid = 1.0 / (1.0 + math.pow(10.0, -score / 400.0))
        return sigmoid
            
    except (chess.engine.EngineTerminatedError, chess.engine.EngineError):
        return None

def check_elo(headers, min_elo):
    if min_elo <= 0:
        return True
    try:
        white_elo = int(headers.get("WhiteElo", "0"))
        black_elo = int(headers.get("BlackElo", "0"))
    except (ValueError, TypeError):
        return True
    if white_elo == 0 and black_elo == 0:
        return True
    return white_elo >= min_elo and black_elo >= min_elo

def process_game_batch(games_data, stockfish_path, depth, min_cp, min_move, max_move, max_per_game):
    setup_stockfish_engine(stockfish_path)
    
    results = []
    discarded = 0
    filtered_by_score = 0
    
    try:
        for headers, moves_uci in games_data:
            board = chess.Board() 
            if "FEN" in headers:
                board.set_fen(headers["FEN"])

            positions_extracted = 0

            for i, move_uci in enumerate(moves_uci):
                if positions_extracted >= max_per_game:
                    break
                    
                try:
                    move = chess.Move.from_uci(move_uci)
                    if move not in board.legal_moves:
                        break
                except ValueError:
                    break 
                
                is_capture_or_promotion = board.is_capture(move) or move.promotion
                board.push(move)
                
                if is_capture_or_promotion:
                    continue

                if min_move <= board.fullmove_number <= max_move:
                    if not board.is_checkmate() and not board.is_stalemate() and not board.is_insufficient_material() and not should_skip_position(board):
                        
                        label = get_stockfish_tactical_label(board, depth, min_cp)
                        
                        if label is not None:
                            fen = board.fen()
                            data = json.dumps({"fen": fen, "result": round(label, 6)})
                            results.append(data)
                            positions_extracted += 1
                        else:
                            filtered_by_score += 1
                    else:
                        discarded += 1
    finally:
        if STOCKFISH_ENGINE:
            STOCKFISH_ENGINE.quit()
            
    return results, discarded, filtered_by_score

def generate_data_parallel(pgn_file, output_file, min_move, max_move, games_limit, depth, min_cp, min_elo, stockfish_path, max_per_game, num_workers=None):
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1) 
        
    print(f"Starting tactical generation with {num_workers} workers...", file=sys.stderr)
    print(f"Settings: min_move={min_move}, min_cp={min_cp}, max_per_game={max_per_game}", file=sys.stderr)
    
    BATCH_SIZE = 50 
    
    games_read = 0
    games_skipped_elo = 0
    total_kept = 0
    discarded_total = 0
    filtered_total = 0
    
    current_batch = []
    futures = []
    
    with open(pgn_file, encoding='utf-8') as pgn, \
         open(output_file, 'w', encoding='utf-8') as f_out, \
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
            if not check_elo(headers, min_elo):
                games_skipped_elo += 1
                continue
                
            games_read += 1
            moves = [move.uci() for move in game.mainline_moves()] 
            
            current_batch.append((headers, moves))
            
            if len(current_batch) >= BATCH_SIZE:
                future = executor.submit(process_game_batch, current_batch, stockfish_path, depth, min_cp, min_move, max_move, max_per_game)
                futures.append(future)
                current_batch = []
                
        if current_batch:
            future = executor.submit(process_game_batch, current_batch, stockfish_path, depth, min_cp, min_move, max_move, max_per_game)
            futures.append(future)
            
        print(f"All games read ({games_read} accepted). Waiting for {len(futures)} tasks to complete...", file=sys.stderr)
        
        for future in concurrent.futures.as_completed(futures):
            try:
                res, discarded, filtered = future.result()
                
                for line in res:
                    f_out.write(line + '\n')
                    
                total_kept += len(res)
                discarded_total += discarded
                filtered_total += filtered
                
                print(f"\rTactical Positions Kept: {total_kept} | Filtered(flat score): {filtered_total} | Discarded(material/checks): {discarded_total}", end="", file=sys.stderr)
                     
            except Exception as e:
                print(f"\nWorker exception: {e}", file=sys.stderr)

    print(f"\n--- Tactical Data Generation Complete ---", file=sys.stderr)
    print(f"Total Games Read: {games_read}", file=sys.stderr)
    print(f"Total Tactical Positions Saved to {output_file}: {total_kept}", file=sys.stderr)

if __name__ == "__main__":

    default_stockfish = shutil.which('stockfish')
    if not default_stockfish:
        if os.path.exists("./stockfish/stockfish-windows-x86-64-avx2.exe"):
             default_stockfish = "./stockfish/stockfish-windows-x86-64-avx2.exe"
        else:
             default_stockfish = "stockfish"

    parser = argparse.ArgumentParser(description="Generate tactical chess dataset for King Safety tuning.")
    parser.add_argument("--pgn", type=str, default="tuner/twic1613.pgn", help="Input PGN file")
    parser.add_argument("--output", type=str, default="tuner/tactical_data.jsonl", help="Output JSONL file")
    parser.add_argument("--games", type=int, default=50000, help="Number of games to parse")
    parser.add_argument("--min-move", type=int, default=12, help="Minimum fullmove number")
    parser.add_argument("--max-move", type=int, default=100, help="Maximum fullmove number")
    parser.add_argument("--stockfish-path", type=str, default=default_stockfish, help="Path to Stockfish")
    parser.add_argument("--stockfish-depth", type=int, default=12, help="Analysis depth")
    parser.add_argument("--min-cp", type=int, default=200, help="Minimum abs(score) to consider it a tactical position")
    parser.add_argument("--min-elo", type=int, default=2400, help="Minimum ELO for both players")
    parser.add_argument("--max-per-game", type=int, default=4, help="Max tactical positions extracted per game")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers")

    args = parser.parse_args()

    if not os.path.exists(args.pgn):
        print(f"Error: PGN file not found: {args.pgn}")
        sys.exit(1)

    generate_data_parallel(
        pgn_file=args.pgn,
        output_file=args.output,
        min_move=args.min_move,
        max_move=args.max_move,
        games_limit=args.games,
        depth=args.stockfish_depth,
        min_cp=args.min_cp,
        min_elo=args.min_elo,
        stockfish_path=args.stockfish_path,
        max_per_game=args.max_per_game,
        num_workers=args.workers
    )
