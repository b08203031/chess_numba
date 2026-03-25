import argparse
import chess
import chess.pgn
import math
import os
import subprocess
import sys
import time
from datetime import datetime

class Engine:
    def __init__(self, name, script_path):
        self.name = name
        self.script_path = os.path.abspath(script_path)
        self.process = None
        self.working_dir = os.path.dirname(self.script_path)

    def start(self):
        cmd = [sys.executable, "-u", self.script_path]

        try:
            # Redirect stderr to stdout to prevent deadlocks and capture logs
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=self.working_dir,
                universal_newlines=True,
                bufsize=1
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
                # 打印啟動期間所有的消息，包括 Numba 編譯警告，避免看起來像當機
                print(f"[{self.name} init] {line}")

    def read_line(self):
        if not self.process:
            return None
        line = self.process.stdout.readline()
        if not line:
            return None
        return line.strip()

    def warm_up(self):
        # Trigger JIT compilation
        print(f"[{self.name}] Warming up...")
        self.send_command("setoption name OwnBook value false")
        self.send_command("position startpos")
        self.send_command("go depth 2")
        while True:
            line = self.read_line()
            if line is None:
                raise RuntimeError(f"Engine {self.name} terminated during warm-up")
            # 打印 warmup 期間所有的消息
            print(f"[{self.name} warmup] {line}")
            if line.startswith("bestmove"):
                break

    def get_move(self, moves_history, time_limit_ms, start_fen=None):
        if start_fen:
            cmd = f"position fen {start_fen}"
        else:
            cmd = "position startpos"

        if moves_history:
            cmd += " moves " + " ".join(moves_history)

        self.send_command(cmd)
        self.send_command(f"go movetime {time_limit_ms}")

        best_move = None
        while True:
            line = self.read_line()
            if line is None:
                print(f"[{self.name}] Engine crashed or closed connection")
                break

            # print(f"[{self.name}] {line}") # Debug output

            if line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        best_move = chess.Move.from_uci(parts[1])
                    except ValueError:
                        print(f"[{self.name}] Invalid move received: {parts[1]}")
                break

        return best_move

class SPRTTest:
    def __init__(self, elo0=0.0, elo1=5.0, alpha=0.05, beta=0.05):
        self.elo0 = elo0
        self.elo1 = elo1
        self.a = math.log((1 - beta) / alpha)
        self.b = math.log(beta / (1 - alpha))
        self.pair_scores = [] # stores scores of Engine 1 in each pair (0 to 2)
        
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
            
        # Expected scores for a single game based on Elo difference
        mu0_game = 1.0 / (1.0 + 10.0 ** (-self.elo0 / 400.0))
        mu1_game = 1.0 / (1.0 + 10.0 ** (-self.elo1 / 400.0))
        
        # Expected scores for a PAIR of games
        mu0_pair = 2.0 * mu0_game
        mu1_pair = 2.0 * mu1_game
        
        # Calculate Sequential Probability Ratio Test (SPRT) Log-Likelihood Ratio (LLR)
        # using the normal approximation of the score sum
        sum_scores = sum(self.pair_scores)
        llr = (mu1_pair - mu0_pair) / variance * (sum_scores - n * (mu0_pair + mu1_pair) / 2.0)
        
        if llr >= self.a:
            return "H1 Accepted (Engine improved)", llr
        elif llr <= self.b:
            return "H0 Accepted (No improvement)", llr
        else:
            return "Continue", llr

def play_game(white_engine, black_engine, time_limit_ms, game_number, start_fen=None):
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
        
    pgn_game.headers["Event"] = "Engine Tournament"
    pgn_game.headers["Site"] = "Local"
    pgn_game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    pgn_game.headers["Round"] = str(game_number)
    pgn_game.headers["White"] = white_engine.name
    pgn_game.headers["Black"] = black_engine.name

    node = pgn_game

    print(f"Game {game_number}: {white_engine.name} (White) vs {black_engine.name} (Black)", end="")
    if start_fen:
        print(" (from custom opening)")
    else:
        print()

    while not board.is_game_over(claim_draw=True):
        mover = white_engine if board.turn == chess.WHITE else black_engine

        try:
            moves_history = [m.uci() for m in board.move_stack]
            move = mover.get_move(moves_history, time_limit_ms, start_fen)
        except Exception as e:
            print(f"Error getting move from {mover.name}: {e}")
            break

        if move is None:
            print(f"No move returned by {mover.name}. Forfeiting.")
            result = "0-1" if board.turn == chess.WHITE else "1-0"
            pgn_game.headers["Result"] = result
            return result, pgn_game

        if move not in board.legal_moves:
            print(f"Illegal move from {mover.name}: {move}")
            result = "0-1" if board.turn == chess.WHITE else "1-0"
            pgn_game.headers["Result"] = result
            return result, pgn_game

        board.push(move)
        node = node.add_variation(move)

    result = board.result(claim_draw=True)
    pgn_game.headers["Result"] = result
    print(f" Result: {result}")

    return result, pgn_game

def calculate_elo_statistics(wins, draws, losses):
    """
    計算 Elo 分差與 95% 信賴區間的誤差。
    """
    total_games = wins + draws + losses
    if total_games == 0:
        return 0.0, 0.0

    # 1. 計算得分率 (Score Rate, p)
    score = wins + 0.5 * draws
    p = score / total_games

    # 處理邊界情況
    if p <= 0:
        return -float('inf'), 0.0
    if p >= 1:
        return float('inf'), 0.0

    # 2. 計算 Elo 分差
    elo_diff = -400 * math.log10(1 / p - 1)

    # 3. 計算標準誤 (Standard Error, SE)
    # E[x^2] = (wins * 1^2 + draws * 0.5^2 + losses * 0^2) / N
    mean_sq = (wins * 1.0 + draws * 0.25) / total_games
    variance = mean_sq - (p ** 2)
    
    if variance < 0: variance = 0
    
    sigma = math.sqrt(variance)
    se = sigma / math.sqrt(total_games)

    # 4. 計算誤差範圍 (95% CI)
    z_score = 1.96
    gradient = 400 / (math.log(10) * p * (1 - p))
    error_margin = z_score * se * gradient

    return elo_diff, error_margin

def count_result(result_str, is_white):
    if result_str == "1-0": return 1.0 if is_white else 0.0
    if result_str == "0-1": return 0.0 if is_white else 1.0
    return 0.5 

def run_tournament(engine1_path, engine2_path, games_count, time_ms, engine1_name="Engine_A", engine2_name="Engine_B"):

    e1 = Engine(engine1_name, engine1_path)
    e2 = Engine(engine2_name, engine2_path)
    
    # Read openings if available
    openings = [None]
    if os.path.exists("openings.epd"):
        with open("openings.epd", "r") as f:
            lines = [l.strip() for l in f if l.strip()]
            if lines: openings = lines

    import random
    random.shuffle(openings)

    try:
        print("Starting engines...")
        e1.start()
        e2.start()

        e1.warm_up()
        e2.warm_up()

        scores = {engine1_name: 0.0, engine2_name: 0.0}
        e1_stats = {"wins": 0, "draws": 0, "losses": 0}

        pgn_file = "tournament_analysis/tournament_results.pgn"
        os.makedirs(os.path.dirname(pgn_file), exist_ok=True)
        with open(pgn_file, "w") as f:
            pass
            
        # Target Elo bounds: Test if we gained 10 Elo or 0 Elo
        sprt = SPRTTest(elo0=0.0, elo1=10.0, alpha=0.05, beta=0.05)
        
        pairs_count = games_count // 2
        game_num = 1
        
        status, llr = "Continue", 0.0
        
        for pair_i in range(1, pairs_count + 1):
            opening_fen = openings[(pair_i - 1) % len(openings)]
            
            # Game A: Engine 1 plays White
            res_a, pgn_a = play_game(e1, e2, time_ms, game_num, opening_fen)
            score_a_for_e1 = count_result(res_a, is_white=True)
            game_num += 1
            
            # Game B: Engine 1 plays Black
            res_b, pgn_b = play_game(e2, e1, time_ms, game_num, opening_fen)
            score_b_for_e1 = count_result(res_b, is_white=False)
            game_num += 1
            
            pair_score = score_a_for_e1 + score_b_for_e1
            sprt.add_pair_result(pair_score)
            scores[e1.name] += pair_score
            scores[e2.name] += (2.0 - pair_score)
            
            # Record individual stats for ELO calculation
            for s in [score_a_for_e1, score_b_for_e1]:
                if s == 1.0: e1_stats["wins"] += 1
                elif s == 0.0: e1_stats["losses"] += 1
                else: e1_stats["draws"] += 1

            with open(pgn_file, "a") as f:
                print(pgn_a, file=f, end="\n\n")
                print(pgn_b, file=f, end="\n\n")

            status, llr = sprt.check_status()
            
            print(f"\nAfter {pair_i} pairs ({(pair_i)*2} games):")
            print(f"{engine1_name}: {scores[engine1_name]}")
            print(f"{engine2_name}: {scores[engine2_name]}")
            print(f"SPRT LLR: {llr:.2f} bounds: [{sprt.b:.2f}, {sprt.a:.2f}]")
            print(f"SPRT Status: {status}")
            print("-" * 30)
            
            if status != "Continue":
                print(f"SPRT bounds triggered. Stopping test early.")
                break

        print("\n=== Tournament Finished ===")
        print(f"Total Matches Played: {game_num - 1}")
        print(f"Final Score {engine1_name}: {scores[engine1_name]} ({e1_stats['wins']}W - {e1_stats['draws']}D - {e1_stats['losses']}L)")
        print(f"Final Score {engine2_name}: {scores[engine2_name]}")
        print(f"SPRT Status: {status} (LLR = {llr:.2f})")

        elo_diff, error_margin = calculate_elo_statistics(e1_stats["wins"], e1_stats["draws"], e1_stats["losses"])
        if math.isinf(elo_diff):
             print(f"ELO Difference: > +/- 800 (Perfect score or zero score)")
        else:
             print(f"Estimated ELO Difference ({engine1_name} - {engine2_name}): {elo_diff:+.2f} [+/- {error_margin:.2f}]")
             print(f"95% Confidence Interval: {elo_diff - error_margin:+.2f} to {elo_diff + error_margin:+.2f}")

        print(f"PGN saved to {pgn_file}")

    finally:
        print("Terminating engines...")
        e1.terminate()
        e2.terminate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a tournament between two versions of the engine.")
    parser.add_argument("--engine1", default="chess_engine_v2/main.py", help="Path to the first engine's main.py (default: chess_engine_v2/main.py)")
    parser.add_argument("--engine2", default="main.py", help="Path to the second engine's main.py (default: main.py)")
    parser.add_argument("--name1", default="chess_engine_v2", help="Name of the first engine (default: chess_engine_v2)")
    parser.add_argument("--name2", default="chess_engine", help="Name of the second engine (default: chess_engine)")
    parser.add_argument("--games", type=int, default=250, help="Number of games to play (default: 500)")
    parser.add_argument("--time", type=int, default=500, help="Time per move in ms (default: 500)")

    args = parser.parse_args()

    if not os.path.exists(args.engine1):
        print(f"Error: Engine 1 path not found: {args.engine1}")
        sys.exit(1)
    if not os.path.exists(args.engine2):
        print(f"Error: Engine 2 path not found: {args.engine2}")
        sys.exit(1)

    run_tournament(args.engine1, args.engine2, args.games, args.time, args.name1, args.name2)