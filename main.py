# main.py
import numpy as np
import time

from chess_engine.move import get_from_square, get_to_square, encode_move, NORMAL_MOVE
from chess_engine.board_operations import make_move, unmake_move
from chess_engine.zobrist import compute_initial_hash

def square_to_algebraic(sq):
    """Converts a square index (0-63) to algebraic notation."""
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

def test_make_unmake(piece_bbs, occupancy_bbs, game_state):
    """
    Tests the make_move and unmake_move functions with the new separated state structure.
    """
    print("\n--- Testing Make/Unmake Move Logic (Refactored) ---")

    from_sq, to_sq = 12, 28 # e2 -> e4
    pawn_double_move = encode_move(from_sq, to_sq, 0, NORMAL_MOVE)

    print(f"Original Board State Hash: {game_state[-1]}")
    print(f"Testing move: {pretty_print_move(pawn_double_move)}")

    new_piece_bbs, new_occupancy_bbs, new_game_state, unmake_info = make_move(
        piece_bbs, occupancy_bbs, game_state, pawn_double_move
    )

    incremental_key = new_game_state[-1]
    recalculated_key = compute_initial_hash(new_piece_bbs, new_game_state)

    print(f"Incrementally Updated Hash: {incremental_key}")
    print(f"Recalculated Hash on New State: {recalculated_key}")

    if incremental_key == recalculated_key:
        print("SUCCESS: Zobrist key was updated correctly.")
    else:
        print("ERROR: Zobrist key mismatch!")
        return False

    restored_piece_bbs, restored_occupancy_bbs, restored_game_state = unmake_move(
        new_piece_bbs, new_occupancy_bbs, new_game_state, pawn_double_move, unmake_info
    )

    if restored_piece_bbs == piece_bbs and restored_occupancy_bbs == occupancy_bbs and restored_game_state == game_state:
        print("SUCCESS: Board state was perfectly restored after unmake_move.")
    else:
        print("ERROR: Board state mismatch after unmake_move!")
        if restored_piece_bbs != piece_bbs:
            print("  - Mismatch in piece_bbs")
        if restored_occupancy_bbs != occupancy_bbs:
            print("  - Mismatch in occupancy_bbs")
        if restored_game_state != game_state:
            print("  - Mismatch in game_state")
            for i in range(len(game_state)):
                if restored_game_state[i] != game_state[i]:
                    print(f"    - At index {i}: Original={game_state[i]}, Restored={restored_game_state[i]}")
        return False

    print("--- Make/Unmake Test Passed ---")
    return True


if __name__ == "__main__":
    # =========================================================================
    # --- Setup: Initial Board State (Refactored Structure) ---
    # =========================================================================
    wp, wn, wb, wr, wq, wk = (np.uint64(0x000000000000FF00), np.uint64(0x0000000000000042),
                             np.uint64(0x0000000000000024), np.uint64(0x0000000000000081),
                             np.uint64(0x0000000000000008), np.uint64(0x0000000000000010))
    bp, bn, bb, br, bq, bk = (np.uint64(0x00FF000000000000), np.uint64(0x4200000000000000),
                             np.uint64(0x2400000000000000), np.uint64(0x8100000000000000),
                             np.uint64(0x0800000000000000), np.uint64(0x1000000000000000))

    initial_piece_bbs = (wp, wn, wb, wr, wq, wk, bp, bn, bb, br, bq, bk)

    white_pieces_bb = wp | wn | wb | wr | wq | wk
    black_pieces_bb = bp | bn | bb | br | bq | bk
    initial_occupancy_bbs = (white_pieces_bb, black_pieces_bb, white_pieces_bb | black_pieces_bb)

    game_state_no_hash = (
        np.uint8(0),   # Side to move (0 for White)
        np.uint8(15),  # Castling rights
        np.int8(-1),   # En passant square
        np.uint8(0),   # Halfmove clock
        np.uint64(0)   # Zobrist key placeholder
    )

    initial_zobrist_key = compute_initial_hash(initial_piece_bbs, game_state_no_hash)
    initial_game_state = game_state_no_hash[:-1] + (initial_zobrist_key,)

    # =========================================================================
    # --- Run Tests ---
    # =========================================================================
    test_make_unmake(initial_piece_bbs, initial_occupancy_bbs, initial_game_state)
