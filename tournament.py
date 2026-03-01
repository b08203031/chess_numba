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
        self.send_command("position startpos")
        self.send_command("go depth 2")
        while True:
            line = self.read_line()
            if line is None:
                raise RuntimeError(f"Engine {self.name} terminated during warm-up")
            # print(f"[{self.name} warm-up] {line}")
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

def play_game(white_engine, black_engine, time_limit_ms, game_number):
    board = chess.Board()

    # Send new game command
    white_engine.send_command("ucinewgame")
    black_engine.send_command("ucinewgame")
    white_engine.wait_for_ready()
    black_engine.wait_for_ready()

    pgn_game = chess.pgn.Game()
    pgn_game.headers["Event"] = "Engine Tournament"
    pgn_game.headers["Site"] = "Local"
    pgn_game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    pgn_game.headers["Round"] = str(game_number)
    pgn_game.headers["White"] = white_engine.name
    pgn_game.headers["Black"] = black_engine.name

    node = pgn_game

    print(f"Game {game_number}: {white_engine.name} (White) vs {black_engine.name} (Black)")

    while not board.is_game_over(claim_draw=True):
        if board.turn == chess.WHITE:
            mover = white_engine
        else:
            mover = black_engine

        try:
            moves_history = [m.uci() for m in board.move_stack]
            move = mover.get_move(moves_history, time_limit_ms)
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

        # Simple progress indicator
        # print(".", end="", flush=True)

    # print() # Newline after dots
    result = board.result(claim_draw=True)
    pgn_game.headers["Result"] = result
    print(f"Result: {result}")

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
    
    # 確保變異數非負 (浮點誤差可能導致極小負值)
    if variance < 0: variance = 0
    
    sigma = math.sqrt(variance)
    se = sigma / math.sqrt(total_games)

    # 4. 計算誤差範圍 (95% CI)
    z_score = 1.96
    # Gradient approximation: d(Elo)/dp
    gradient = 400 / (math.log(10) * p * (1 - p))
    error_margin = z_score * se * gradient

    return elo_diff, error_margin

def run_tournament(engine1_path, engine2_path, games_count, time_ms, engine1_name="Engine_A", engine2_name="Engine_B"):

    e1 = Engine(engine1_name, engine1_path)
    e2 = Engine(engine2_name, engine2_path)

    try:
        print("Starting engines...")
        e1.start()
        e2.start()

        e1.warm_up()
        e2.warm_up()

        scores = {engine1_name: 0.0, engine2_name: 0.0}
        
        # 追蹤 Engine 1 的詳細戰績以計算 Elo 誤差
        e1_stats = {"wins": 0, "draws": 0, "losses": 0}

        pgn_file = "tournament_analysis/tournament_results.pgn"
        # Clear existing PGN file
        with open(pgn_file, "w") as f:
            pass

        for i in range(1, games_count + 1):
            # Alternate colors
            if i % 2 != 0:
                white, black = e1, e2
            else:
                white, black = e2, e1

            result, pgn_game = play_game(white, black, time_ms, i)

            with open(pgn_file, "a") as f:
                print(pgn_game, file=f, end="\n\n")

            if result == "1-0":
                scores[white.name] += 1.0
                # 更新 Engine 1 戰績
                if white.name == engine1_name:
                    e1_stats["wins"] += 1
                else:
                    e1_stats["losses"] += 1
                    
            elif result == "0-1":
                scores[black.name] += 1.0
                # 更新 Engine 1 戰績
                if black.name == engine1_name:
                    e1_stats["wins"] += 1
                else:
                    e1_stats["losses"] += 1
            else:
                scores[white.name] += 0.5
                scores[black.name] += 0.5
                # 更新 Engine 1 戰績
                e1_stats["draws"] += 1

            # Print intermediate stats
            print(f"After {i} games:")
            print(f"{engine1_name}: {scores[engine1_name]}")
            print(f"{engine2_name}: {scores[engine2_name]}")
            print("-" * 30)

        # Final Stats
        print("\n=== Tournament Finished ===")
        print(f"Total Games: {games_count}")
        print(f"Final Score {engine1_name}: {scores[engine1_name]} ({e1_stats['wins']}W - {e1_stats['draws']}D - {e1_stats['losses']}L)")
        print(f"Final Score {engine2_name}: {scores[engine2_name]}")

        # 計算 Elo 與誤差
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
    parser.add_argument("--engine1", default="main.py", help="Path to the first engine's main.py (default: main.py)")
    parser.add_argument("--engine2", default="chess_engine_v2/main.py", help="Path to the second engine's main.py (default: chess_engine_v2/main.py)")
    parser.add_argument("--name1", default="chess_engine", help="Name of the first engine (default: chess_engine)")
    parser.add_argument("--name2", default="chess_engine_v2", help="Name of the second engine (default: chess_engine_v2)")
    parser.add_argument("--games", type=int, default=20, help="Number of games to play (default: 20)")
    parser.add_argument("--time", type=int, default=100, help="Time per move in ms (default: 100)")

    args = parser.parse_args()

    if not os.path.exists(args.engine1):
        print(f"Error: Engine 1 path not found: {args.engine1}")
        sys.exit(1)
    if not os.path.exists(args.engine2):
        print(f"Error: Engine 2 path not found: {args.engine2}")
        sys.exit(1)

    run_tournament(args.engine1, args.engine2, args.games, args.time, args.name1, args.name2)