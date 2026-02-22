
import argparse
import subprocess
import os
import sys
import random
import time
import math
import chess
import chess.engine
import chess.pgn
import threading
import concurrent.futures
import traceback
from tune_search.param_config import SEARCH_PARAMS, get_bounds, get_step, get_default_params

# Engine command template
ENGINE_CMD_TEMPLATE = "python tune_search/engine_adapter.py"

def get_engine_cmd_list(params):
    """
    Constructs the command line list for the engine with specific parameters.
    params: dict of {PARAM_NAME: value}
    """
    cmd = [sys.executable, "-u", "tune_search/engine_adapter.py"]
    for k, v in params.items():
        cmd.append(f"--{k}")
        cmd.append(str(v))
    return cmd

def warmup_engine(engine, name="Engine"):
    """
    Runs a shallow search to trigger Numba JIT compilation.
    This prevents the first actual game move from timing out.
    """
    try:
        # Search Kiwipete position for depth 1 to trigger JIT compilation on more complex positions
        print(f"[{name}] Warming up with Kiwipete depth 1... (May take ~1 min)")
        kiwipete = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
        engine.play(chess.Board(kiwipete), chess.engine.Limit(depth=1))
        print(f"[{name}] Warmup complete.")
    except Exception as e:
        print(f"[{name}] Warmup failed: {e}")

def play_game_reused(engine1, engine2, time_limit, increment, opening_fen=None):
    """
    Plays a single game between two ALREADY RUNNING engines.
    engine1: White
    engine2: Black
    Returns score for Engine 1 (1.0 win, 0.5 draw, 0.0 loss).
    """
    board = chess.Board(opening_fen) if opening_fen else chess.Board()
    
    # Reset engines for new game
    # python-chess engine.configure() fails for 'ucinewgame' because main.py doesn't report it as an option.
    # We send it directly as a command to ensure internal state (Hash, History) is cleared.
    try:
        if engine1.protocol: engine1.protocol.send_line("ucinewgame")
        if engine2.protocol: engine2.protocol.send_line("ucinewgame")
        # Small sleep to ensure engines process it, though UCI doesn't strictly require wait
        time.sleep(0.05)
    except Exception as e:
        print(f"Error sending ucinewgame command: {e}")

    # Parse time control
    # Note: 'time' in Limit is generally "time limit for this move" if used this way,
    # or base time if handling clock. python-chess SimpleEngine.play with Limit(time=X) 
    # usually treats X as movetime (seconds per move).
    # Increment is not a standard argument for fixed-movetime Limit constructor.
    # If the user provides "0.1+0.01", we treat it as movetime=0.1s and ignore increment for now
    # to avoid TypeError.
    limit = chess.engine.Limit(time=time_limit)
    
    # Game Loop
    try:
        while not board.is_game_over():
            if board.turn == chess.WHITE:
                result = engine1.play(board, limit)
            else:
                result = engine2.play(board, limit)
                
            if result.move is None:
                print(f"Engine ({'White' if board.turn == chess.WHITE else 'Black'}) returned no move. Claiming loss.")
                return 0.0 if board.turn == chess.WHITE else 1.0
                    
            board.push(result.move)
    except Exception as e:
        # Check if it's a timeout error (wrapped by concurrent.futures or asyncio)
        msg = str(e)
        if "TimeoutError" in msg or "timeout" in msg.lower():
            # If engine times out, it loses.
            print(f"Engine ({'White' if board.turn == chess.WHITE else 'Black'}) timed out (Loss)")
            if board.turn == chess.WHITE:
                return 0.0 # White timed out, Black wins (Score for Engine 1 is 0.0)
            else:
                return 1.0 # Black timed out, White wins (Score for Engine 1 is 1.0)
        
        print(f"Game error: {e}")
        traceback.print_exc()
        return 0.5

    outcome = board.outcome()
    if outcome.winner == chess.WHITE:
        return 1.0
    elif outcome.winner == chess.BLACK:
        return 0.0
    else:
        return 0.5

def run_match_python(base_params, test_params, num_games=10, tc="0.1+0.01", concurrency=1, opening_book="tuner/twic1613.pgn"):
    """
    Runs a match using python-chess, REUSING engine processes to avoid recompilation overhead.
    """
    
    # Parse TC
    if '+' in tc:
        parts = tc.split('+')
        base_time = float(parts[0])
        inc = float(parts[1])
    else:
        base_time = float(tc)
        inc = 0.0
        
    # Load Openings
    openings = []
    if opening_book and os.path.exists(opening_book):
        try:
            _, ext = os.path.splitext(opening_book)
            if ext == '.bin':
                import chess.polyglot as pg
                print(f"Generating openings from Polyglot book: {opening_book}")
                with pg.open_reader(opening_book) as reader:
                    for _ in range(num_games):
                        board = chess.Board()
                        for _ in range(8):
                            try:
                                entry = reader.weighted_choice(board)
                                board.push(entry.move)
                            except IndexError:
                                break
                        openings.append(board.fen())
            else:
                with open(opening_book) as f:
                    for _ in range(num_games * 2):
                        game = chess.pgn.read_game(f)
                        if game is None: break
                        board = game.board()
                        for i, move in enumerate(game.mainline_moves()):
                            if i >= 8: break
                            board.push(move)
                        openings.append(board.fen())
        except Exception as e:
            print(f"Error reading opening book: {e}")
    
    if not openings:
        openings = [chess.STARTING_FEN]

    # --- Start Engines ONCE ---
    # We create two engines: 'Base' and 'Test'.
    # Note: If concurrency > 1, we would need a pool of engines.
    # For now, let's assume concurrency=1 or create multiple pairs.
    
    # To support concurrency properly with reused engines, we need a list of engine pairs.
    # num_pairs = concurrency
    
    engine_pairs = []
    
    print(f"Starting {concurrency} engine pair(s)...")
    
    # Use ThreadPoolExecutor to start engines and warmup in parallel
    # This avoids sequential waiting for each engine to compile
    
    def start_engine_pair(index):
        try:
            # Base Engine
            cmd_base = get_engine_cmd_list(base_params)
            engine_base = chess.engine.SimpleEngine.popen_uci(cmd_base, timeout=120.0)
            
            # Test Engine
            cmd_test = get_engine_cmd_list(test_params)
            engine_test = chess.engine.SimpleEngine.popen_uci(cmd_test, timeout=120.0)
            
            # Warmup
            # Warmup both engines in this thread (still sequential per pair, but pairs are parallel)
            warmup_engine(engine_base, f"Base-{index}")
            warmup_engine(engine_test, f"Test-{index}")
            
            return (engine_base, engine_test)
        except Exception as e:
            print(f"Error starting engine pair {index}: {e}")
            return None

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(start_engine_pair, i) for i in range(concurrency)]
            
            for future in concurrent.futures.as_completed(futures):
                pair = future.result()
                if pair:
                    engine_pairs.append(pair)
                else:
                    raise Exception("Failed to start one or more engine pairs")
            
    except Exception as e:
        print(f"Error starting engines: {e}")
        for b, t in engine_pairs:
            b.quit()
            t.quit()
        return 0.5

    score_test = 0.0
    games_played = 0
    results = {"win": 0, "loss": 0, "draw": 0}
    
    # We distribute the 'pairs of games' (A vs B, B vs A) among the engine workers
    total_pairs = num_games // 2
    
    # Queue of work: each item is an opening FEN
    work_queue = [random.choice(openings) for _ in range(total_pairs)]
    
    lock = threading.Lock()
    
    def worker(pair_idx):
        """
        Worker function using a specific engine pair (engine_base, engine_test).
        Process games from the queue until empty.
        """
        my_base, my_test = engine_pairs[pair_idx]
        
        while True:
            try:
                # Pop work
                with lock:
                    if not work_queue:
                        return
                    opening = work_queue.pop()
            except IndexError:
                return

            # Play Game 1: Test (White) vs Base (Black)
            # Use 'my_test' as White, 'my_base' as Black
            s1 = play_game_reused(my_test, my_base, base_time, inc, opening)
            
            res_str1 = "Draw"
            if s1 == 1.0: res_str1 = "Test Won"
            elif s1 == 0.0: res_str1 = "Base Won"
            
            # Play Game 2: Base (White) vs Test (Black)
            # Use 'my_base' as White, 'my_test' as Black
            s2 = play_game_reused(my_base, my_test, base_time, inc, opening)
            
            # s2 is score for Base (White). Test score is 1.0 - s2
            test_score_g2 = 1.0 - s2
            
            res_str2 = "Draw"
            if s2 == 1.0: res_str2 = "Base Won"
            elif s2 == 0.0: res_str2 = "Test Won"

            pair_score = s1 + test_score_g2
            
            with lock:
                nonlocal score_test, games_played
                score_test += pair_score
                games_played += 2
                
                # Update stats
                for s in [s1, test_score_g2]:
                    if s == 1.0: results["win"] += 1
                    elif s == 0.0: results["loss"] += 1
                    else: results["draw"] += 1
                
                print(f"Game Pair Finished: {res_str1} | {res_str2} (Test Score: {score_test}/{games_played})")

    # Start threads
    threads = []
    # import threading # Removed as it's already imported at top
    for i in range(concurrency):
        t = threading.Thread(target=worker, args=(i,))
        t.start()
        threads.append(t)
        
    for t in threads:
        t.join()

    # Cleanup Engines
    print("Closing engines...")
    for b, t in engine_pairs:
        try:
            b.quit()
        except Exception:
            b.close()
        try:
            t.quit()
        except Exception:
            t.close()

    final_score = score_test / games_played if games_played > 0 else 0.5
    print(f"Match Finished. Test Score: {final_score:.3f} ({results['win']} W - {results['loss']} L - {results['draw']} D)")
    return final_score

class SPSAOptimizer:
    def __init__(self, param_names, initial_params):
        self.param_names = param_names
        self.current_params = initial_params.copy()
        
        # SPSA Hyperparameters
        self.a = 200.0  # Learning rate scaling (larger because params are integers like 500)
        self.c = 20.0   # Perturbation scaling
        self.A = 20.0   # Stability constant
        self.alpha = 0.602
        self.gamma = 0.101
        
        self.iteration = 0

    def step(self, match_runner_func):
        self.iteration += 1
        k = self.iteration
        
        ak = self.a / ((k + self.A) ** self.alpha)
        ck = self.c / (k ** self.gamma)
        
        # 1. Generate Perturbation Vector (Bernoulli +/- 1)
        delta = {}
        for name in self.param_names:
            delta[name] = 1 if random.random() < 0.5 else -1
            
        # 2. Create Two Parameter Sets
        theta_plus = {}
        theta_minus = {}
        
        for name in self.param_names:
            current_val = self.current_params[name]
            perturbation = int(round(ck * delta[name]))
            
            # Apply bounds
            min_val, max_val = get_bounds(name)
            
            val_plus = current_val + perturbation
            val_minus = current_val - perturbation
            
            # Clip
            if min_val is not None:
                val_plus = max(min_val, min(max_val, val_plus))
                val_minus = max(min_val, min(max_val, val_minus))
            
            theta_plus[name] = int(val_plus)
            theta_minus[name] = int(val_minus)
            
        # 3. Evaluate
        print(f"Iter {k}: Playing Match (Plus vs Minus)...")
        score_plus_vs_minus = match_runner_func(theta_minus, theta_plus) # Returns score for Plus (Test)
        
        result_score = score_plus_vs_minus # 0.0 to 1.0 (Test/Plus score)
        win_signal = result_score - 0.5 # Positive if Plus won
        
        print(f"Iter {k}: Score(Plus vs Minus) = {result_score:.3f}. Signal = {win_signal:.3f}")
        
        for name in self.param_names:
            step = ak * win_signal * delta[name]
            
            # Update
            new_val = self.current_params[name] + step
            
            # Clip
            min_val, max_val = get_bounds(name)
            if min_val is not None:
                new_val = max(min_val, min(max_val, new_val))
            
            self.current_params[name] = int(round(new_val))
            
        # Print changes
        print(f"Updated Params: {self.current_params}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=10, help="Games per iteration")
    parser.add_argument("--concurrency", type=int, default=2, help="Concurrency")
    parser.add_argument("--tc", type=str, default="0.1+0.01", help="Time control (e.g. 0.1+0.01)")
    parser.add_argument("--iter", type=int, default=100, help="Iterations")
    
    args = parser.parse_args()
    
    # Check for Opening Book (PGN or Polyglot)
    # Prioritize argument if I add one later, but for now check standard locations
    book_path = "chess_engine/polyglot.bin" # Default engine book
    
    if not os.path.exists(book_path):
        candidates = [
            "polyglot.bin",
            "tuner/twic1613.pgn",
            "twic1613.pgn",
            "openings.pgn"
        ]
        found = False
        for c in candidates:
            if os.path.exists(c):
                book_path = c
                found = True
                break
        
        if not found:
            print(f"Warning: No opening book found (checked polyglot.bin, twic1613.pgn, etc.). Matches will start from startpos.")
            book_path = None
    
    if book_path:
        print(f"Using opening book: {book_path}")
            
    initial_params = get_default_params()
    param_names = list(initial_params.keys())
    
    optimizer = SPSAOptimizer(param_names, initial_params)
    
    print(f"Starting tuning for {args.iter} iterations.")
    print(f"Params to tune: {param_names}")
    
    for i in range(args.iter):
        optimizer.step(
            lambda p_base, p_test: run_match_python(
                p_base, p_test, 
                num_games=args.games, 
                tc=args.tc, 
                concurrency=args.concurrency,
                opening_book=book_path if book_path else "tuner/twic1613.pgn"
            )
        )
        
        # Save current best/state
        with open("tune_search/current_params.txt", "w") as f:
            f.write(str(optimizer.current_params))

if __name__ == "__main__":
    main()
