# main.py
import numpy as np
import time

# Import the new search and move modules
from chess_engine.search import find_best_move
from chess_engine.move import get_from_square, get_to_square

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

def square_to_algebraic(sq):
    """Converts a square index (0-63) to algebraic notation (e.g., 'e4')."""
    file = 'abcdefgh'[sq % 8]
    rank = str((sq // 8) + 1)
    return f"{file}{rank}"

def pretty_print_move(move):
    """Converts an encoded move to a human-readable string."""
    if move == 0:
        return "NULL_MOVE"
    from_sq = get_from_square(move)
    to_sq = get_to_square(move)
    return f"{square_to_algebraic(from_sq)}{square_to_algebraic(to_sq)}"


if __name__ == "__main__":
    # =========================================================================
    # --- Test Case for the Full Search Engine ---
    # =========================================================================
    print("--- Testing Full Alpha-Beta Search Engine ---")

    # --- Setup: Initial Board State (Standard Opening Position) ---
    # This matches the tuple structure expected by our Numba functions.
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

    white_pieces_bb = wp | wn | wb | wr | wq | wk
    black_pieces_bb = bp | bn | bb | br | bq | bk
    all_pieces_bb = white_pieces_bb | black_pieces_bb

    initial_board_state = (
        wp, wn, wb, wr, wq, wk,
        bp, bn, bb, br, bq, bk,
        white_pieces_bb, black_pieces_bb, all_pieces_bb,
        np.uint8(15),  # Castling rights (WK, WQ, BK, BQ)
        np.int8(-1),   # En passant square (-1 for none)
        np.uint8(0),   # Side to move (0 for White)
        np.uint8(0)    # Halfmove clock
    )

    print("\nInitial board state:")
    print_bitboard(all_pieces_bb)

    # --- Run the search ---
    search_depth = 3 # A reasonable depth for a quick test
    print(f"Searching for the best move at depth {search_depth}...")

    # The first run will include Numba's JIT compilation time.
    start_time_first_run = time.time()
    best_move_encoded = find_best_move(initial_board_state, search_depth)
    end_time_first_run = time.time()

    print(f"\nFirst run (including JIT compilation) took: {end_time_first_run - start_time_first_run:.4f} seconds.")
    print(f"Engine's choice: {pretty_print_move(best_move_encoded)}")

    # --- Run the search again to measure pure performance ---
    print("\nRunning search again to measure performance without compilation overhead...")
    start_time_second_run = time.time()
    best_move_encoded_2 = find_best_move(initial_board_state, search_depth)
    end_time_second_run = time.time()

    print(f"\nSecond run took: {end_time_second_run - start_time_second_run:.4f} seconds.")
    print(f"Engine's choice (should be the same): {pretty_print_move(best_move_encoded_2)}")

    if best_move_encoded == best_move_encoded_2:
        print("\nSUCCESS: The search is deterministic and produces consistent results.")
    else:
        print("\nERROR: The search produced different results on subsequent runs.")
