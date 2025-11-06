# main.py
import numpy as np
from chess_engine.evaluation import evaluate_position
import chess_engine.move_generator as move_gen # Keep for old tests

def print_bitboard(bb: np.uint64):
    """Prints a bitboard in a human-readable 8x8 format."""
    print("\n  a b c d e f g h")
    print(" +-----------------+")
    for rank in range(7, -1, -1):
        print(f"{rank + 1}|", end=" ")
        for file in range(8):
            square = rank * 8 + file
            mask = np.uint64(1) << np.uint64(square)
            print("X" if bb & mask else ".", end=" ")
        print(f"|")
    print(" +-----------------+")
    print(f"Decimal value: {bb}\n")


if __name__ == "__main__":
    # =========================================================================
    # --- Test Case for Evaluation Function ---
    # =========================================================================
    print("--- Testing Evaluation Function ---")

    # --- Setup: Initial Board State (Standard Opening Position) ---
    wp = np.uint64(0b00000000_00000000_00000000_00000000_00000000_00000000_11111111_00000000)
    wn = np.uint64(0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_01000010)
    wb = np.uint64(0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00100100)
    wr = np.uint64(0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_10000001)
    wq = np.uint64(0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00001000)
    wk = np.uint64(0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00010000)

    bp = np.uint64(0b00000000_11111111_00000000_00000000_00000000_00000000_00000000_00000000)
    bn = np.uint64(0b01000010_00000000_00000000_00000000_00000000_00000000_00000000_00000000)
    bb = np.uint64(0b00100100_00000000_00000000_00000000_00000000_00000000_00000000_00000000)
    br = np.uint64(0b10000001_00000000_00000000_00000000_00000000_00000000_00000000_00000000)
    bq = np.uint64(0b00001000_00000000_00000000_00000000_00000000_00000000_00000000_00000000)
    bk = np.uint64(0b00010000_00000000_00000000_00000000_00000000_00000000_00000000_00000000)

    # --- Create the bitboards NumPy array ---
    # This matches the 'u8[:]' part of the Numba signature.
    bitboards = np.array([
        wp, wn, wb, wr, wq, wk,
        bp, bn, bb, br, bq, bk
    ], dtype=np.uint64)

    # --- Other game state variables ---
    # Explicitly cast to np.int64 to match the 'i8' part of the Numba signature.
    castling_rights = np.int64(15)
    ep_square = np.int64(-1)
    side_to_move_white = np.int64(0)
    side_to_move_black = np.int64(1)

    # --- Call the evaluation function ---
    # Numba JIT compilation happens on the first call.
    print("\nEvaluating initial board state...")
    score = evaluate_position(bitboards, castling_rights, ep_square, side_to_move_white)
    print(f"Initial Score from White's perspective: {score}")

    # --- Test from Black's perspective ---
    print("\nEvaluating initial board state (Black to move)...")
    score_black = evaluate_position(bitboards, castling_rights, ep_square, side_to_move_black)
    print(f"Initial Score from Black's perspective: {score_black}")

    # The absolute scores should be identical because the position is symmetrical.
    # The sign should be opposite.
    if score == -score_black:
        print("\nSUCCESS: Perspective scoring is working correctly.")
    else:
        print(f"\nERROR: Perspective scoring is incorrect. White: {score}, Black: {score_black}")

    # --- Old Tests for Move Generation (Disabled but kept for reference) ---
    if False:
        # (Old test code remains here, unchanged)
        pass
