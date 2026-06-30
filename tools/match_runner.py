import argparse
import chess
import chess.pgn
import math
import os
import subprocess
import sys
import time
from datetime import datetime

def parse_info_line(line):
    if not line:
        return ""
    
    tokens = line.split()
    info_dict = {}
    
    i = 0
    while i < len(tokens):
        if tokens[i] == "depth" and i + 1 < len(tokens):
            info_dict["depth"] = tokens[i+1]
            i += 2
        elif tokens[i] == "score" and i + 2 < len(tokens):
            score_type = tokens[i+1]  # "cp" or "mate"
            score_val = tokens[i+2]
            if score_type == "cp":
                try:
                    val = int(score_val)
                    info_dict["score"] = f"{val/100:+.2f}"
                except ValueError:
                    info_dict["score"] = f"cp {score_val}"
            elif score_type == "mate":
                info_dict["score"] = f"#{score_val}"
            else:
                info_dict["score"] = f"{score_type} {score_val}"
            i += 3
        elif tokens[i] == "nodes" and i + 1 < len(tokens):
            info_dict["nodes"] = tokens[i+1]
            i += 2
        elif tokens[i] == "time" and i + 1 < len(tokens):
            info_dict["time"] = tokens[i+1] + "ms"
            i += 2
        elif tokens[i] == "pv":
            info_dict["pv"] = " ".join(tokens[i+1:])
            break
        else:
            i += 1
            
    parts = []
    if "depth" in info_dict: parts.append(f"d={info_dict['depth']}")
    if "score" in info_dict: parts.append(f"eval={info_dict['score']}")
    if "nodes" in info_dict: parts.append(f"n={info_dict['nodes']}")
    if "time" in info_dict: parts.append(f"t={info_dict['time']}")
    
    return ", ".join(parts)

class Engine:
    def __init__(self, name, path, limits=None, cache_dir=None, verbose=True):
        self.name = name
        self.path = os.path.abspath(path)
        self.process = None
        self.working_dir = os.path.dirname(self.path)
        self.verbose = verbose
        self.cache_dir = cache_dir
        self.limits = limits or {"time": 1000, "depth": None, "nodes": None}

    def start(self):
        if self.path.endswith(".py"):
            cmd = [sys.executable, "-u", self.path]
        else:
            cmd = [self.path]

        try:
            env_cache_dir = None
            if self.path.endswith(".py"):
                import tempfile
                import random
                
                if self.cache_dir:
                    env_cache_dir = self.cache_dir
                else:
                    self.unique_cache_dir = os.path.join(
                        tempfile.gettempdir(), 
                        f"numba_cache_{self.name}_{time.time_ns()}_{random.randint(0, 100000)}"
                    )
                    os.makedirs(self.unique_cache_dir, exist_ok=True)
                    env_cache_dir = self.unique_cache_dir

            env = os.environ.copy()
            if env_cache_dir:
                env["NUMBA_CACHE_DIR"] = env_cache_dir

            # Redirect stderr to stdout to prevent deadlocks and capture logs
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=self.working_dir,
                universal_newlines=True,
                bufsize=1,
                env=env
            )
            self.send_command("uci")
            self.wait_for_ready()
        except Exception as e:
            print(f"Error starting engine {self.name}: {e}")
            raise

    def terminate(self):
        if self.process:
            self.send_command("quit")
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
            
            # Clean up the unique cache directory only if it was dynamically created
            if getattr(self, "unique_cache_dir", None) and os.path.exists(self.unique_cache_dir):
                import shutil
                try:
                    shutil.rmtree(self.unique_cache_dir)
                except Exception:
                    pass

    def send_command(self, command):
        if self.process:
            try:
                self.process.stdin.write(command + "\n")
                self.process.stdin.flush()
            except BrokenPipeError:
                pass

    def wait_for_ready(self):
        self.send_command("isready")
        while True:
            line = self.read_line()
            if line is None:
                raise RuntimeError(f"Engine {self.name} terminated unexpectedly during initialization")
            if line == "readyok":
                break
            else:
                if self.verbose:
                    print(f"[{self.name} init] {line}")

    def read_line(self):
        if not self.process:
            return None
        line = self.process.stdout.readline()
        if not line:
            return None
        return line.strip()

    def warm_up(self):
        # Only Python engines need JIT warm-up/compilation
        if not self.path.endswith(".py"):
            return

        if self.verbose:
            print(f"[{self.name}] Warming up...")
        self.send_command("setoption name OwnBook value false")
        self.send_command("position kiwipete")
        self.send_command("go depth 14")
        while True:
            line = self.read_line()
            if line is None:
                raise RuntimeError(f"Engine {self.name} terminated during warm-up")
            if self.verbose:
                print(f"[{self.name} warmup] {line}")
            if line.startswith("bestmove"):
                break

    def get_move(self, moves_history, start_fen=None):
        if start_fen:
            cmd = f"position fen {start_fen}"
        else:
            cmd = "position startpos"

        if moves_history:
            cmd += " moves " + " ".join(moves_history)

        self.send_command(cmd)
        
        nodes = self.limits.get("nodes")
        depth = self.limits.get("depth")
        time_limit_ms = self.limits.get("time")

        if nodes is not None:
            self.send_command(f"go nodes {nodes}")
        elif depth is not None:
            self.send_command(f"go depth {depth}")
        else:
            self.send_command(f"go movetime {time_limit_ms}")

        best_move = None
        last_info = ""
        while True:
            line = self.read_line()
            if line is None:
                print(f"[{self.name}] Engine crashed or closed connection")
                break

            if line.startswith("info"):
                if "depth" in line and "score" in line:
                    last_info = line

            if line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        best_move = chess.Move.from_uci(parts[1])
                    except ValueError:
                        print(f"[{self.name}] Invalid move received: {parts[1]}")
                break

        info_str = parse_info_line(last_info)
        return best_move, info_str

class SPRTTest:
    def __init__(self, elo0=0.0, elo1=5.0, alpha=0.05, beta=0.05):
        self.elo0 = elo0
        self.elo1 = elo1
        self.a = math.log((1 - beta) / alpha)
        self.b = math.log(beta / (1 - alpha))
        self.pair_scores = []
        
    def add_pair_result(self, score):
        self.pair_scores.append(score)
        
    def check_status(self):
        n = len(self.pair_scores)
        if n < 5:
            return "Continue", 0.0
            
        mean = sum(self.pair_scores) / n
        mean_sq = sum(x**2 for x in self.pair_scores) / n
        variance = mean_sq - mean**2
        
        if variance <= 1e-10:
            return "Continue", 0.0
            
        mu0_game = 1.0 / (1.0 + 10.0 ** (-self.elo0 / 400.0))
        mu1_game = 1.0 / (1.0 + 10.0 ** (-self.elo1 / 400.0))
        
        mu0_pair = 2.0 * mu0_game
        mu1_pair = 2.0 * mu1_game
        
        sum_scores = sum(self.pair_scores)
        llr = (mu1_pair - mu0_pair) / variance * (sum_scores - n * (mu0_pair + mu1_pair) / 2.0)
        
        if llr >= self.a:
            return "H1 Accepted (Engine improved)", llr
        elif llr <= self.b:
            return "H0 Accepted (No improvement)", llr
        else:
            return "Continue", llr

def play_game(white_engine, black_engine, game_number, start_fen=None, verbose=True):
    if start_fen:
        board = chess.Board(start_fen)
    else:
        board = chess.Board()

    # Send new game command
    white_engine.send_command("ucinewgame")
    black_engine.send_command("ucinewgame")
    white_engine.wait_for_ready()
    black_engine.wait_for_ready()

    pgn_game = chess.pgn.Game()
    if start_fen:
        pgn_game.setup(start_fen)
        
    pgn_game.headers["Event"] = "Engine vs Stockfish Match"
    pgn_game.headers["Site"] = "Local"
    pgn_game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    pgn_game.headers["Round"] = str(game_number)
    pgn_game.headers["White"] = white_engine.name
    pgn_game.headers["Black"] = black_engine.name

    node = pgn_game

    log_lines = []
    def log(msg, end="\n"):
        if verbose:
            print(msg, end=end)
        log_lines.append(msg + end)

    log(f"Game {game_number}: {white_engine.name} (White) vs {black_engine.name} (Black)", end="")
    if start_fen:
        log(" (from custom opening)")
    else:
        log("")

    while not board.is_game_over(claim_draw=True):
        mover = white_engine if board.turn == chess.WHITE else black_engine

        try:
            moves_history = [m.uci() for m in board.move_stack]
            move, info_str = mover.get_move(moves_history, start_fen)
        except Exception as e:
            log(f"Error getting move from {mover.name}: {e}")
            break

        if move is None:
            log(f"No move returned by {mover.name}. Forfeiting.")
            result = "0-1" if board.turn == chess.WHITE else "1-0"
            pgn_game.headers["Result"] = result
            return result, pgn_game, "".join(log_lines)

        if move not in board.legal_moves:
            log(f"Illegal move from {mover.name}: {move}")
            result = "0-1" if board.turn == chess.WHITE else "1-0"
            pgn_game.headers["Result"] = result
            return result, pgn_game, "".join(log_lines)

        board.push(move)
        node = node.add_variation(move)
        if info_str:
            node.comment = info_str

    result = board.result(claim_draw=True)
    pgn_game.headers["Result"] = result
    log(f" Result: {result}")

    return result, pgn_game, "".join(log_lines)

def calculate_elo_statistics(wins, draws, losses):
    total_games = wins + draws + losses
    if total_games == 0:
        return 0.0, 0.0

    score = wins + 0.5 * draws
    p = score / total_games

    if p <= 0:
        return -float('inf'), 0.0
    if p >= 1:
        return float('inf'), 0.0

    elo_diff = -400 * math.log10(1 / p - 1)

    mean_sq = (wins * 1.0 + draws * 0.25) / total_games
    variance = mean_sq - (p ** 2)
    
    if variance < 0: variance = 0
    
    sigma = math.sqrt(variance)
    se = sigma / math.sqrt(total_games)

    z_score = 1.96
    gradient = 400 / (math.log(10) * p * (1 - p))
    error_margin = z_score * se * gradient

    return elo_diff, error_margin

def count_result(result_str, is_white):
    if result_str == "1-0": return 1.0 if is_white else 0.0
    if result_str == "0-1": return 0.0 if is_white else 1.0
    return 0.5 

def run_match_tournament(engine_path, sf_path, games_count, engine_limits, sf_limits, engine_name="MyEngine", sf_name="Stockfish", concurrency=1):
    openings = [None]
    openings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "openings.epd")
    if os.path.exists(openings_path):
        with open(openings_path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
            if lines: openings = lines

    import random
    random.shuffle(openings)

    scores = {engine_name: 0.0, sf_name: 0.0}
    engine_stats = {"wins": 0, "draws": 0, "losses": 0}

    pgn_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tournament_analysis")
    pgn_file = os.path.join(pgn_dir, "tournament_results.pgn")
    os.makedirs(pgn_dir, exist_ok=True)
    with open(pgn_file, "w") as f:
        pass
        
    sprt = SPRTTest(elo0=0.0, elo1=10.0, alpha=0.05, beta=0.05)
    
    pairs_count = games_count // 2
    status, llr = "Continue", 0.0
    total_matches_played = 0

    import queue
    import threading

    pair_queue = queue.Queue()
    for pair_i in range(1, pairs_count + 1):
        pair_queue.put(pair_i)

    state_lock = threading.Lock()
    sprt_stopped = threading.Event()

    actual_concurrency = min(concurrency, pairs_count)
    if actual_concurrency < 1:
        actual_concurrency = 1

    compilation_barrier_1 = threading.Barrier(actual_concurrency)

    def worker_thread(worker_id):
        nonlocal total_matches_played, status, llr
        
        verbose_worker = (worker_id == 0)
        e1 = Engine(engine_name, engine_path, limits=engine_limits, cache_dir=None, verbose=verbose_worker)
        e2 = Engine(sf_name, sf_path, limits=sf_limits, cache_dir=None, verbose=verbose_worker)
        
        try:
            # Stage 1: Start MyEngine and warm up
            e1.start()
            e1.warm_up()
            try:
                compilation_barrier_1.wait()
            except threading.BrokenBarrierError:
                return

            # Stage 2: Start Stockfish
            e2.start()
            
            while not sprt_stopped.is_set():
                try:
                    pair_i = pair_queue.get_nowait()
                except queue.Empty:
                    break
                
                opening_fen = openings[(pair_i - 1) % len(openings)]
                pair_verbose = verbose_worker and (pair_i == 1)
                
                # Game A: MyEngine plays White
                game_num_a = 2 * pair_i - 1
                res_a, pgn_a, log_a = play_game(e1, e2, game_num_a, opening_fen, verbose=pair_verbose)
                
                # Game B: MyEngine plays Black
                game_num_b = 2 * pair_i
                res_b, pgn_b, log_b = play_game(e2, e1, game_num_b, opening_fen, verbose=pair_verbose)
                
                with state_lock:
                    if sprt_stopped.is_set():
                        pair_queue.task_done()
                        break
                    
                    if not pair_verbose:
                        print(log_a + log_b, end="", flush=True)
                        
                    score_a_for_e1 = count_result(res_a, is_white=True)
                    score_b_for_e1 = count_result(res_b, is_white=False)
                    pair_score = score_a_for_e1 + score_b_for_e1
                    
                    sprt.add_pair_result(pair_score)
                    scores[engine_name] += pair_score
                    scores[sf_name] += (2.0 - pair_score)
                    
                    for s in [score_a_for_e1, score_b_for_e1]:
                        if s == 1.0: engine_stats["wins"] += 1
                        elif s == 0.0: engine_stats["losses"] += 1
                        else: engine_stats["draws"] += 1
                        
                    with open(pgn_file, "a") as f_pgn:
                        f_pgn.write(str(pgn_a) + "\n\n")
                        f_pgn.write(str(pgn_b) + "\n\n")
                        
                    status, llr = sprt.check_status()
                    total_matches_played += 2
                    
                    print(f"\nAfter {len(sprt.pair_scores)} pairs ({len(sprt.pair_scores)*2} games):", flush=True)
                    print(f"{engine_name}: {scores[engine_name]} ({engine_stats['wins']}W - {engine_stats['draws']}D - {engine_stats['losses']}L)", flush=True)
                    print(f"{sf_name}: {scores[sf_name]}", flush=True)
                    print(f"SPRT LLR: {llr:.2f} bounds: [{sprt.b:.2f}, {sprt.a:.2f}]", flush=True)
                    print(f"SPRT Status: {status}", flush=True)
                    print("-" * 30, flush=True)
                    
                    if status != "Continue":
                        print(f"SPRT bounds triggered. Stopping test early.")
                        sprt_stopped.set()
                        pair_queue.task_done()
                        break
                        
                pair_queue.task_done()
        except Exception as e:
            import traceback
            print(f"Worker {worker_id} generated an exception: {e}\n{traceback.format_exc()}")
            compilation_barrier_1.abort()
        finally:
            e1.terminate()
            e2.terminate()

    threads = []
    print(f"Starting match tournament with concurrency={actual_concurrency} (parallel threads)...")
    for worker_id in range(actual_concurrency):
        t = threading.Thread(target=worker_thread, args=(worker_id,))
        t.start()
        threads.append(t)
        
    for t in threads:
        t.join()

    print("\n=== Match Tournament Finished ===")
    print(f"Total Matches Played: {total_matches_played}")
    print(f"Final Score {engine_name}: {scores[engine_name]} ({engine_stats['wins']}W - {engine_stats['draws']}D - {engine_stats['losses']}L)")
    print(f"Final Score {sf_name}: {scores[sf_name]}")
    print(f"SPRT Status: {status} (LLR = {llr:.2f})")

    elo_diff, error_margin = calculate_elo_statistics(engine_stats["wins"], engine_stats["draws"], engine_stats["losses"])
    if math.isinf(elo_diff):
         print(f"ELO Difference: > +/- 800 (Perfect score or zero score)")
    else:
         print(f"Estimated ELO Difference ({engine_name} - {sf_name}): {elo_diff:+.2f} [+/- {error_margin:.2f}]")
         print(f"95% Confidence Interval: {elo_diff - error_margin:+.2f} to {elo_diff + error_margin:+.2f}")

    print(f"PGN saved to {pgn_file}")

def clear_numba_cache(root_dir):
    import shutil
    from pathlib import Path
    print("Clearing Numba __pycache__ directories...")
    p = Path(root_dir)
    for cache_dir in p.rglob("__pycache__"):
        if cache_dir.is_dir():
            try:
                shutil.rmtree(cache_dir)
            except Exception:
                pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a match between MyEngine and Stockfish.")
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    default_engine = os.path.join(root_dir, "main.py")
    if os.name == 'nt':
        default_sf = os.path.join(root_dir, "external", "stockfish", "stockfish-windows-x86-64-avx2.exe")
    else:
        default_sf = os.path.join(root_dir, "external", "stockfish", "src", "stockfish")
        
    parser.add_argument("--engine", default=default_engine, help="Path to your engine's main.py")
    parser.add_argument("--sf", default=default_sf, help="Path to the Stockfish executable")
    parser.add_argument("--engine_name", default="MyEngine", help="Name of your engine")
    parser.add_argument("--sf_name", default="Stockfish", help="Name of Stockfish")
    parser.add_argument("--games", type=int, default=100, help="Number of games to play (default: 100)")
    parser.add_argument("--concurrency", "-c", type=int, default=1, help="Number of parallel game processes (concurrency)")
    
    # Engine limits
    parser.add_argument("--engine_time", type=int, default=1000, help="MyEngine time per move in ms (default: 1000)")
    parser.add_argument("--engine_depth", type=int, default=None, help="MyEngine fixed search depth per move (overrides --engine_time if set)")
    parser.add_argument("--engine_nodes", type=int, default=None, help="MyEngine fixed nodes limit per move (overrides depth and time if set)")
    
    # Stockfish limits
    parser.add_argument("--sf_time", type=int, default=1000, help="Stockfish time per move in ms (default: 1000)")
    parser.add_argument("--sf_depth", type=int, default=None, help="Stockfish fixed search depth per move (overrides --sf_time if set)")
    parser.add_argument("--sf_nodes", type=int, default=None, help="Stockfish fixed nodes limit per move (overrides depth and time if set)")

    parser.add_argument("--clear-cache", action="store_true", help="Force clear Numba cache before starting")

    args = parser.parse_args()

    if not os.path.exists(args.engine):
        print(f"Error: Engine path not found: {args.engine}")
        sys.exit(1)
    if not os.path.exists(args.sf):
        print(f"Error: Stockfish path not found: {args.sf}")
        sys.exit(1)

    if args.clear_cache:
        clear_numba_cache(root_dir)

    engine_limits = {
        "time": args.engine_time,
        "depth": args.engine_depth,
        "nodes": args.engine_nodes
    }
    
    sf_limits = {
        "time": args.sf_time,
        "depth": args.sf_depth,
        "nodes": args.sf_nodes
    }

    print("=== Configuration ===")
    print(f"MyEngine ({args.engine_name}): {args.engine}")
    if engine_limits['nodes'] is not None:
        print(f"  Limits: Fixed Nodes = {engine_limits['nodes']}")
    elif engine_limits['depth'] is not None:
        print(f"  Limits: Fixed Depth = {engine_limits['depth']}")
    else:
        print(f"  Limits: Fixed Time = {engine_limits['time']} ms/move")
        
    print(f"Stockfish ({args.sf_name}): {args.sf}")
    if sf_limits['nodes'] is not None:
        print(f"  Limits: Fixed Nodes = {sf_limits['nodes']}")
    elif sf_limits['depth'] is not None:
        print(f"  Limits: Fixed Depth = {sf_limits['depth']}")
    else:
        print(f"  Limits: Fixed Time = {sf_limits['time']} ms/move")
    print(f"Games to Play: {args.games}")
    print(f"Concurrency: {args.concurrency}")
    print("=====================")

    run_match_tournament(
        args.engine, 
        args.sf, 
        args.games, 
        engine_limits, 
        sf_limits, 
        engine_name=args.engine_name, 
        sf_name=args.sf_name, 
        concurrency=args.concurrency
    )
