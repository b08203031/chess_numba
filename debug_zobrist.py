# debug_zobrist.py
# import numpy as np
#
# # Temporarily add chess_engine to path for imports
# import sys
# import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
#
# from chess_engine.board import Board
# from chess_engine.move import encode_move
# from chess_engine.zobrist import compute_initial_hash, debug_recalculate_hash
# from chess_engine.board_operations import make_move
# from chess_engine.constants import PAWN, WHITE, BLACK, ALL_CASTLING_RIGHTS
#
# def run_zobrist_debug():
#     """
#     Initializes a board, performs a single move, and compares the
#     incrementally updated Zobrist key with a fully recalculated one.
#     """
#     # 1. Initialize Board and construct the state tuples manually
#     board = Board()
#
#     # piece_bbs: (wP, wN, wB, wR, wQ, wK, bP, bN, bB, bR, bQ, bK)
#     piece_bbs = (
#         board.P, board.N, board.B, board.R, board.Q, board.K,
#         board.p, board.n, board.b, board.r, board.q, board.k
#     )
#
#     # occupancy_bbs: (white_pieces, black_pieces, all_pieces)
#     white_occupancy = board.white_pieces
#     black_occupancy = board.black_pieces
#     occupancy_bbs = (white_occupancy, black_occupancy, white_occupancy | black_occupancy)
#
#     # game_state: (side_to_move, castling_rights, en_passant_square, halfmove_clock, zobrist_key)
#     # Start with a fresh board state
#     side_to_move = WHITE
#     castling_rights = ALL_CASTLING_RIGHTS # WK_CASTLE | WQ_CASTLE | BK_CASTLE | BQ_CASTLE
#     en_passant_square = -1
#     halfmove_clock = 0
#
#     # Compute initial hash from the constructed state
#     temp_game_state_for_hash = (side_to_move, castling_rights, en_passant_square, halfmove_clock, 0)
#     initial_key = compute_initial_hash(piece_bbs, temp_game_state_for_hash)
#
#     game_state = (side_to_move, castling_rights, en_passant_square, halfmove_clock, initial_key)
#
#     print(f"Initial Board Hash: {initial_key}")
#
#     # 2. Define and make a simple move (e.g., e2e4)
#     # from_square=12 (e2), to_square=28 (e4), piece=PAWN, flag=NORMAL
#     move = encode_move(from_square=12, to_square=28, promotion_piece=0, special_flag=0)
#
#     print("\n=============================================")
#     print("           PERFORMING MOVE e2e4")
#     print("=============================================\n")
#
#     # 3. Call the modified make_move
#     # It will print the verbose incremental update log
#     new_piece_bbs, new_occupancy_bbs, new_game_state, _ = make_move(
#         piece_bbs, occupancy_bbs, game_state, move
#     )
#
#     incremental_key = new_game_state[4]
#
#     # 4. Call the new debug_recalculate_hash function
#     # It will print the verbose recalculation log on the *new* board state
#     recalculated_key = debug_recalculate_hash(new_piece_bbs, new_game_state)
#
#     # 5. Compare the final keys
#     print("\n--- FINAL COMPARISON ---")
#     print(f"  Incremental Key: {incremental_key}")
#     print(f"Recalculated Key: {recalculated_key}")
#     if incremental_key == recalculated_key:
#         print("✅ SUCCESS: Keys match!")
#     else:
#         print("❌ FAILURE: Keys DO NOT match!")
#     print("------------------------")
#
#
# if __name__ == "__main__":
#     print("Starting Zobrist key debug harness...")
#     run_zobrist_debug()
