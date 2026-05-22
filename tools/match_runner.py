
import subprocess
import chess
import chess.pgn
import sys
import time
import re
import os

# Configuration
_tools_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_tools_dir)
if os.name == 'nt':
    STOCKFISH_PATH = os.path.join(_root_dir, "external", "stockfish", "stockfish-windows-x86-64-avx2.exe")
    MY_ENGINE_CMD = [sys.executable, "-u", os.path.join(_root_dir, "main.py")]
else:
    STOCKFISH_PATH = os.path.join(_root_dir, "external", "stockfish", "src", "stockfish")
    MY_ENGINE_CMD = ["python3", "-u", os.path.join(_root_dir, "main.py")]

STOCKFISH_TIME_MS = 1
MY_ENGINE_TIME_MS = 10000

def run_match():
    # Initialize engines
    print("Starting engines...")
    try:
        sf_process = subprocess.Popen(
            [STOCKFISH_PATH],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        my_engine_process = subprocess.Popen(
            MY_ENGINE_CMD,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
    except Exception as e:
        print(f"Error starting engines: {e}")
        return

    # Initialize UCI
    print("Initializing Stockfish...")
    send_uci_command(sf_process, "uci")
    wait_for_ready(sf_process, "Stockfish")
    
    print("Initializing MyEngine...")
    send_uci_command(my_engine_process, "uci")
    wait_for_ready(my_engine_process, "MyEngine")
    
    # Warm-up Phase
    warm_up_engine(my_engine_process)

    # Reset for game
    send_uci_command(my_engine_process, "ucinewgame")
    send_uci_command(sf_process, "ucinewgame")
    wait_for_ready(my_engine_process, "MyEngine")
    wait_for_ready(sf_process, "Stockfish")

    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "Engine Match"
    game.headers["White"] = "MyEngine"
    game.headers["Black"] = "Stockfish"
    
    my_engine_color = chess.WHITE
    
    node = game
    
    pv_logs = []
    
    print("Starting game...")
    
    try:
        while not board.is_game_over():
            fen = board.fen()
            print(f"Move {board.fullmove_number}: {fen}")
            
            if board.turn == my_engine_color:
                # My Engine's turn
                print("MyEngine thinking...")
                best_move, best_pv_info = play_my_engine_move(my_engine_process, board, MY_ENGINE_TIME_MS)
                
                # Verify PV
                if best_pv_info:
                    print(f"MyEngine PV found: depth={best_pv_info['depth']}")
                    validation_result = verify_pv(board, best_pv_info['pv'])
                    pv_logs.append({
                        'fen': fen,
                        'depth': best_pv_info['depth'],
                        'pv': best_pv_info['pv'],
                        'valid': validation_result['valid'],
                        'error': validation_result.get('error'),
                        'move_number': board.fullmove_number
                    })
                else:
                    print("MyEngine No PV found")
                    pv_logs.append({
                        'fen': fen,
                        'depth': 0,
                        'pv': [],
                        'valid': True, 
                        'error': "No PV returned",
                        'move_number': board.fullmove_number
                    })
                    
            else:
                # Stockfish's turn
                print("Stockfish thinking...")
                best_move = play_stockfish_move(sf_process, board, STOCKFISH_TIME_MS)
            
            print(f"Move played: {best_move}")
            board.push(best_move)
            node = node.add_variation(best_move)
            
            # Save intermediate results
            save_results(game, pv_logs)
            
    except KeyboardInterrupt:
        print("Match interrupted")
    except Exception as e:
        print(f"Error in game loop: {e}")
        
    print("Game Over")
    print("Result:", board.result())
    game.headers["Result"] = board.result()
    
    save_results(game, pv_logs)

    # Cleanup
    send_uci_command(sf_process, "quit")
    send_uci_command(my_engine_process, "quit")
    sf_process.wait()
    my_engine_process.wait()

def save_results(game, pv_logs):
    # Save Game
    with open("game_history.pgn", "w") as f:
        print(game, file=f, end="\n\n")
        
    # Save PV Analysis
    with open("pv_analysis.txt", "w") as f:
        for log in pv_logs:
            f.write(f"Move {log['move_number']} (MyEngine):\n")
            f.write(f"  FEN: {log['fen']}\n")
            f.write(f"  Reported PV: {' '.join(log['pv'])}\n")
            f.write(f"  Depth: {log['depth']}\n")
            if log['valid']:
                f.write(f"  Status: VALID\n")
            else:
                f.write(f"  Status: INVALID - {log['error']}\n")
            f.write("-" * 40 + "\n")

def send_uci_command(process, command):
    # print(f"> {command}")
    process.stdin.write(command + "\n")
    process.stdin.flush()

def wait_for_ready(process, name):
    send_uci_command(process, "isready")
    while True:
        line = process.stdout.readline()
        if not line:
             print(f"{name} closed stdout unexpectedly")
             break
        line = line.strip()
        # print(f"{name}: {line}")
        if line == "readyok":
            return

def warm_up_engine(process):
    print("Warming up MyEngine (JIT compilation)...")
    send_uci_command(process, "position startpos")
    send_uci_command(process, "go depth 2")
    
    while True:
        line = process.stdout.readline()
        if not line: break
        line = line.strip()
        if line.startswith("bestmove"):
            break
    print("Warm-up complete.")

def play_stockfish_move(process, board, time_ms):
    send_uci_command(process, f"position fen {board.fen()}")
    send_uci_command(process, f"go movetime {time_ms}")
    
    best_move = None
    while True:
        line = process.stdout.readline()
        if not line: break
        line = line.strip()
        if line.startswith("bestmove"):
            parts = line.split()
            best_move = chess.Move.from_uci(parts[1])
            break
    return best_move

def play_my_engine_move(process, board, time_ms):
    send_uci_command(process, f"position fen {board.fen()}")
    send_uci_command(process, f"go movetime {time_ms}")
    
    best_move = None
    deepest_pv = None
    max_depth = -1
    
    while True:
        line = process.stdout.readline()
        if not line: break
        line = line.strip()
        print(f"MyEngine: {line}")
        
        if line.startswith("info"):
            if "pv" in line:
                try:
                    parts = line.split()
                    
                    if "depth" in parts:
                        depth_idx = parts.index("depth")
                        depth = int(parts[depth_idx + 1])
                    else:
                        depth = 0

                    pv_idx = parts.index("pv")
                    pv_moves = parts[pv_idx + 1:]
                    
                    # Update if this is deeper OR same depth (sometimes later output is better)
                    if depth >= max_depth:
                        max_depth = depth
                        deepest_pv = {
                            'depth': depth,
                            'pv': pv_moves
                        }
                except ValueError:
                    pass
                    
        elif line.startswith("bestmove"):
            parts = line.split()
            best_move = chess.Move.from_uci(parts[1])
            break
            
    return best_move, deepest_pv

def verify_pv(board, pv_moves):
    temp_board = board.copy()
    
    for move_uci in pv_moves:
        try:
            move = chess.Move.from_uci(move_uci)
        except ValueError:
             return {'valid': False, 'error': f"Malformed UCI move: {move_uci}"}
            
        if move not in temp_board.legal_moves:
             return {'valid': False, 'error': f"Illegal move {move_uci} in position {temp_board.fen()}"}
        
        temp_board.push(move)
        
    return {'valid': True}

if __name__ == "__main__":
    run_match()
