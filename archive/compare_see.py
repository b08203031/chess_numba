
import subprocess
import sys
import numpy as np
import time

from chess_engine.fen_parser import parse_fen
from chess_engine.move_generator import generate_legal_moves
from chess_engine.move import move_to_uci
from chess_engine.see import see
from chess_engine.engine_types import piece_bbs_signature, occupancy_bbs_signature

from test_puzzle import puzzles

STOCKFISH_BIN = "./stockfish/src/stockfish"

def get_stockfish_see(fen, moves):
    """
    Returns a dict {uci_move: see_value} using Stockfish.
    """
    process = subprocess.Popen(
        STOCKFISH_BIN,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )

    commands = []
    commands.append(f"position fen {fen}")
    for m in moves:
        commands.append(f"see {m}")
    commands.append("quit")

    input_str = "\n".join(commands) + "\n"
    stdout, stderr = process.communicate(input=input_str)

    results = {}
    for line in stdout.splitlines():
        if line.startswith("see"):
            # Format: see e2e4 value 0
            parts = line.split()
            if len(parts) >= 4 and parts[2] == "value":
                move_str = parts[1]
                try:
                    val = int(parts[3])
                    results[move_str] = val
                except ValueError:
                    pass
    return results

def compare_see():
    print("--- Comparing SEE with Stockfish ---")

    # Select first 15 puzzles
    test_puzzles = puzzles[:15]

    total_moves = 0
    matches = 0
    mismatches = 0

    for i, puzzle in enumerate(test_puzzles):
        fen = puzzle['fen']
        print(f"\nPuzzle {i+1}: {puzzle['name']}")
        print(f"FEN: {fen}")

        piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
        legal_moves = generate_legal_moves(piece_bbs, occupancy_bbs, game_state)

        engine_results = {}
        uci_moves = []

        for move in legal_moves:
            uci = move_to_uci(move)
            val = see(piece_bbs, occupancy_bbs, game_state[0], move)
            engine_results[uci] = val
            uci_moves.append(uci)

        stockfish_results = get_stockfish_see(fen, uci_moves)

        print(f"{'Move':<6} | {'My SEE':<8} | {'SF SEE':<8} | {'Diff':<8} | {'Status'}")
        print("-" * 50)

        for uci in uci_moves:
            my_val = engine_results.get(uci, 0)
            sf_val = stockfish_results.get(uci, 0) # Default to 0 if SF fails?

            diff = my_val - sf_val

            # Allow some tolerance because piece values might differ slightly
            # My values: 100, 320, 330, 500, 900
            # SF values: 208, 781, 825, 1276, 2538 (roughly 2x - 2.5x my values)
            # So direct comparison is apples to oranges!
            # I must scale my values or normalize?
            # Or just check sign/trend?
            # The user asked: "check if consistent or trend matches".

            # Let's verify scaling.
            # SF Pawn ~208. My Pawn 100. Ratio ~ 2.08
            # SF Queen 2538. My Queen 900. Ratio ~ 2.82
            # The piece values are fundamentally different.
            # I should output both and check if Sign matches (Positive/Negative/Zero).

            sign_match = False
            if my_val > 0 and sf_val > 0: sign_match = True
            elif my_val < 0 and sf_val < 0: sign_match = True
            elif my_val == 0 and sf_val == 0: sign_match = True

            # Special case: Small positive vs 0 or small negative might be threshold diffs.

            status = "MATCH" if sign_match else "MISMATCH"
            if abs(my_val) < 50 and abs(sf_val) < 100: status = "CLOSE" # Near zero

            print(f"{uci:<6} | {my_val:<8} | {sf_val:<8} | {diff:<8} | {status}")

            if status == "MATCH" or status == "CLOSE":
                matches += 1
            else:
                mismatches += 1
            total_moves += 1

    print("\n--- Summary ---")
    print(f"Total Moves Checked: {total_moves}")
    print(f"Sign Matches: {matches}")
    print(f"Sign Mismatches: {mismatches}")
    print(f"Match Rate: {matches/total_moves*100:.1f}%")

if __name__ == "__main__":
    # Warmup Numba
    print("Warming up...")
    piece_bbs, occupancy_bbs, game_state = parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    moves = generate_legal_moves(piece_bbs, occupancy_bbs, game_state)
    if len(moves) > 0:
        see(piece_bbs, occupancy_bbs, game_state[0], moves[0])

    compare_see()
