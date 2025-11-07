# chess_engine/engine_types.py
import numba
import numpy as np

# --- Numba Type Signatures for the Refactored Board State ---

piece_bbs_signature = numba.types.UniTuple(numba.uint64, 12)
"""
Numba type signature for the piece bitboards.
"""

occupancy_bbs_signature = numba.types.UniTuple(numba.uint64, 3)
"""
Numba type signature for the occupancy bitboards.
"""

game_state_signature = numba.types.Tuple([
    numba.uint8,   # side_to_move
    numba.uint8,   # castling_rights
    numba.int8,    # en_passant_square
    numba.uint8,   # halfmove_clock
    numba.uint64   # zobrist_key
])
"""
Numba type signature for the core game state.
"""

board_state_flat_signature = numba.types.Tuple(
    list(piece_bbs_signature.types) +
    list(occupancy_bbs_signature.types) +
    [numba.uint8, numba.int8, numba.uint8]
)
"""
Numba type signature for the flattened board state.
"""

unmake_info_signature = numba.types.Tuple([
    numba.int8,    # captured_piece_type
    numba.uint8,   # old_castling_rights
    numba.int8,    # old_en_passant_square
    numba.uint8    # old_halfmove_clock
])
"""
Numba type signature for the unmake_info tuple.
"""
