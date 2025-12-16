"""
從 PGN 文件生成訓練數據集的腳本。(已整合 Stockfish 數據清洗功能，支援多進程並行處理)

這個腳本的目的是為 `tuner.py` 調優腳本創建一個數據集，此版本專為
Texel's Tuning Method 設計。它通過以下步驟工作：
1.  讀取 PGN 文件並解析對局結果。
2.  將對局分配給多個工作進程 (Worker Process)。
3.  每個 Worker 使用獨立的 Stockfish 引擎實例進行並行分析與清洗。
4.  主進程收集結果並寫入 JSON Lines 文件。

這個經過清洗的數據集將更客觀，有助於訓練出一個評估更理性、棋力更強的引擎。
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

def is_evaluation_consistent(board: chess.Board, result: float, depth: int, threshold_cp: int) -> bool:
    if STOCKFISH_ENGINE is None:
        raise RuntimeError("Stockfish engine is not initialized.")

    try:
        info = STOCKFISH_ENGINE.analyse(board, chess.engine.Limit(depth=depth))
        score_obj = info["score"].white()
        
        if score_obj.is_mate():
            score = 30000 if score_obj.mate() > 0 else -30000
        else:
            score = score_obj.score()

        WIN_TOLERANCE = 30 
        
        if result == 1.0: 
            return score > -WIN_TOLERANCE
        elif result == 0.0: 
            return score < WIN_TOLERANCE
        elif result == 0.5: 
            return abs(score) <= threshold_cp 
            
    except (chess.engine.EngineTerminatedError, chess.engine.EngineError):
        return False

    return True

# --- Worker Function for Parallel Processing ---

def process_game_batch(games_data, stockfish_path, depth, threshold, min_move, max_move):
    """
    Worker function to process a batch of games.
    games_data: list of (game_headers, moves_list) tuples.
    This avoids pickling the entire game object or engine.
    """
    # Initialize engine for this worker
    setup_stockfish_engine(stockfish_path)
    
    mg_results = []
    eg_results = []
    discarded = 0
    
    try:
        for headers, moves_san in games_data:
            result = parse_result(headers.get("Result", "*"))
            if result is None: continue
            
            # Reconstruct board
            board = chess.Board() # Standard start pos
            # If FEN is in headers (e.g. from position), handle it? Assuming standard startpos PGNs for now.
            if "FEN" in headers:
                board.set_fen(headers["FEN"])

            # Replay moves
            # Optimisation: We only need to check positions within range.
            # But we must play all moves to get there.
            
            for i, move_san in enumerate(moves_san):
                try:
                    move = board.parse_san(move_san)
                except ValueError:
                    break # Invalid move in PGN
                
                # Check quiescence before push (is_capture uses current board)
                is_capture_or_promotion = board.is_capture(move) or move.promotion
                
                board.push(move)
                
                # Filter Quiescence (skip positions resulting from capture)
                if is_capture_or_promotion:
                    continue

                if min_move <= board.fullmove_number <= max_move:
                    if not board.is_checkmate() and not board.is_stalemate() and not board.is_insufficient_material() and not should_skip_position(board):
                        
                        if is_evaluation_consistent(board, result, depth, threshold):
                            fen = board.fen()
                            data = json.dumps({"fen": fen, "result": result})
                            
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
        num_workers = max(1, multiprocessing.cpu_count() - 1) # Leave 1 core for OS/Main
        
    print(f"Starting parallel generation with {num_workers} workers...", file=sys.stderr)
    
    # Reading PGN is sequential, analysis is parallel.
    # Strategy: Read chunks of games, submit to pool.
    
    BATCH_SIZE = 50 # Games per worker task
    
    positions_count = 0 # Cannot track total positions easily without parsing moves in main, skipping for perf.
    games_read = 0
    mg_total = 0
    eg_total = 0
    discarded_total = 0
    
    # Buffer for batching
    current_batch = []
    futures = []
    
    with open(pgn_file, encoding='utf-8') as pgn, \
         open(output_mg, 'w', encoding='utf-8') as f_mg, \
         open(output_eg, 'w', encoding='utf-8') as f_eg, \
         concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        
        while True:
            # Check for game limit
            if games_limit and games_read >= games_limit:
                break
                
            try:
                game = chess.pgn.read_game(pgn)
            except ValueError:
                continue
                
            if game is None:
                break
                
            games_read += 1
            
            # Extract minimal data for worker: Headers dict and list of moves (SAN strings)
            # Passing SAN strings is safer than move objects across processes
            headers = dict(game.headers)
            moves = [move.uci() for move in game.mainline_moves()] # Using UCI string is slightly safer/faster to re-parse? SAN is smaller but requires context. 
            # Board.parse_uci vs parse_san. parse_uci is faster. 
            # But wait, read_game gives us Move objects.
            # Ideally we pass UCI strings.
            # Re-parsing UCI is very fast.
            
            # Wait, main moves are Move objects.
            # board.push_uci(uci_str) works.
            
            current_batch.append((headers, moves))
            
            if len(current_batch) >= BATCH_SIZE:
                # Submit batch
                future = executor.submit(process_game_batch, current_batch, stockfish_path, depth, threshold, min_move, max_move)
                futures.append(future)
                current_batch = []
                
                # Check for completed futures to keep memory low
                # (Simple polling or just let them pile up? Pile up is bad for memory if many results.
                # But futures list grows. We should reap.)
                
                # Optimization: Reap completed futures periodically
                # Instead of waiting only at the end, verify completion as we go to show progress
                # But we can't block here or we lose read speed.
                # A simple way is to check done futures from the list.
                # But futures list might be huge if not reaped.
                # Let's keep it simple: We iterate as_completed *after* reading all games?
                # No, if 1M games, we run OOM.
                # We must yield results as we go.
                pass

        # Submit remaining
        if current_batch:
            future = executor.submit(process_game_batch, current_batch, stockfish_path, depth, threshold, min_move, max_move)
            futures.append(future)
            
        print(f"All games read. Waiting for {len(futures)} tasks to complete...", file=sys.stderr)
        
        # Collect results
        # Use as_completed to print progress as tasks finish
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
                
                # Progress update (rough)
                total_kept = mg_total + eg_total
                # Print every time a batch returns (or every few batches)
                # Just print always for responsiveness, using carriage return
                print(f"\rProcessed Positions (Kept+Discarded): {total_kept + discarded_total} | MG: {mg_total} | EG: {eg_total}", end="", file=sys.stderr)
                     
            except Exception as e:
                print(f"\nWorker exception: {e}", file=sys.stderr)

    print(f"\n--- Data Generation Complete ---", file=sys.stderr)
    print(f"Total Games Read: {games_read}", file=sys.stderr)
    print(f"Middlegame positions saved: {mg_total}", file=sys.stderr)
    print(f"Endgame positions saved: {eg_total}", file=sys.stderr)
    print(f"Positions discarded: {discarded_total}", file=sys.stderr)

if __name__ == '__main__':
    # Fix for Windows Multiprocessing
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
        description="Generate cleaned chess training data (Parallel Split MG/EG) using Stockfish referee.",
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
    parser.add_argument("--consistency-threshold", type=int, default=30, help="Draw consistency threshold (cp).")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes (default: CPU count - 1).")

    args = parser.parse_args()

    if not os.path.exists(args.pgn):
        print(f"Error: PGN file not found at '{args.pgn}'", file=sys.stderr)
        sys.exit(1)

    # In worker model, setup happens in worker.
    # But check if stockfish path is valid once here.
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
