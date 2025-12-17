
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
    cmd = [sys.executable, "tune_search/engine_adapter.py"]
    for k, v in params.items():
        cmd.append(f"--{k}")
        cmd.append(str(v))
    return cmd

def warmup_engine(engine):
    """
    Runs a shallow search to trigger Numba JIT compilation.
    This prevents the first actual game move from timing out.
    """
    try:
        # Search startpos for 1ms or depth 1
        engine.play(chess.Board(), chess.engine.Limit(depth=1))
    except Exception as e:
        print(f"Warmup failed: {e}")

def play_game(params1, params2, time_limit, increment, opening_fen=None):
    """
    Plays a single game between Engine 1 (White) and Engine 2 (Black).
    Returns score for Engine 1 (1.0 win, 0.5 draw, 0.0 loss).
    """
    
    # Start Engine 1
    cmd1 = get_engine_cmd_list(params1)
    try:
        # Increase timeout to allow for Numba JIT compilation
        engine1 = chess.engine.SimpleEngine.popen_uci(cmd1, timeout=60.0)
        warmup_engine(engine1)
    except Exception as e:
        print(f"Error starting engine 1: {e}")
        traceback.print_exc()
        return 0.5

    # Start Engine 2
    cmd2 = get_engine_cmd_list(params2)
    try:
        engine2 = chess.engine.SimpleEngine.popen_uci(cmd2, timeout=60.0)
        warmup_engine(engine2)
    except Exception as e:
        print(f"Error starting engine 2: {e}")
        traceback.print_exc()
        engine1.quit()
        return 0.5

    board = chess.Board(opening_fen) if opening_fen else chess.Board()
    
    # Parse time control
    limit = chess.engine.Limit(time=time_limit)
    
    # Game Loop
    try:
        while not board.is_game_over():
            if board.turn == chess.WHITE:
                # Engine 1
                result = engine1.play(board, limit)
            else:
                # Engine 2
                result = engine2.play(board, limit)
                
            if result.move is None:
                # Resignation or crash?
                print("Engine returned no move. Claiming loss.")
                if board.turn == chess.WHITE:
                    # White (Engine 1) crashed -> Loss
                    engine1.quit()
                    engine2.quit()
                    return 0.0
                else:
                    # Black (Engine 2) crashed -> Win for Engine 1
                    engine1.quit()
                    engine2.quit()
                    return 1.0
                    
            board.push(result.move)
    except Exception as e:
        print(f"Game error: {e}")
        traceback.print_exc()
        engine1.quit()
        engine2.quit()
        return 0.5

    engine1.quit()
    engine2.quit()
    
    outcome = board.outcome()
    if outcome.winner == chess.WHITE:
        return 1.0
    elif outcome.winner == chess.BLACK:
        return 0.0
    else:
        return 0.5

def run_match_python(base_params, test_params, num_games=10, tc="0.1+0.01", concurrency=1, opening_book="tuner/twic1613.pgn"):
    """
    Runs a match using python-chess.
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
            # Check file extension to decide how to load
            _, ext = os.path.splitext(opening_book)
            
            if ext == '.bin':
                # Polyglot Book - Generate openings via Random Walk
                import chess.polyglot
                print(f"Generating openings from Polyglot book: {opening_book}")
                
                with chess.polyglot.open_reader(opening_book) as reader:
                    for _ in range(num_games): # One opening per game pair is enough
                        board = chess.Board()
                        # Walk 8 plies (4 moves) or until out of book
                        for _ in range(8):
                            try:
                                entry = reader.weighted_choice(board)
                                board.push(entry.move)
                            except IndexError:
                                # No moves in book for this position
                                break
                        openings.append(board.fen())
                        
            else:
                # PGN Book
                with open(opening_book) as f:
                    # Read first N games as openings
                    for _ in range(num_games * 2): # Read enough
                        game = chess.pgn.read_game(f)
                        if game is None: break
                        # Play out first 8 plies (4 moves) to get a position
                        board = game.board()
                        for i, move in enumerate(game.mainline_moves()):
                            if i >= 8: break
                            board.push(move)
                        openings.append(board.fen())
                        
        except Exception as e:
            print(f"Error reading opening book: {e}")
    
    if not openings:
        openings = [chess.STARTING_FEN]

    score_test = 0.0
    games_played = 0
    
    # Pairwise games (A vs B, then B vs A from same opening)
    pairs = num_games // 2
    
    def play_pair(idx):
        opening = random.choice(openings)
        
        # Game 1: Test (White) vs Base (Black)
        s1 = play_game(test_params, base_params, base_time, inc, opening)
        
        # Game 2: Base (White) vs Test (Black)
        s2 = play_game(base_params, test_params, base_time, inc, opening)
        
        # s2 is score for Base. Test score is 1.0 - s2
        return s1 + (1.0 - s2)

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(play_pair, i) for i in range(pairs)]
        
        for future in concurrent.futures.as_completed(futures):
            pair_score = future.result()
            score_test += pair_score
            games_played += 2
            # print(f"Games: {games_played}/{num_games}, Test Score: {score_test}")

    return score_test / games_played

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
