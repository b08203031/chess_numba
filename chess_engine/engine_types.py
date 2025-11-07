# chess_engine/engine_types.py
import numba as nb
import numpy as np

# --- Numba Type Signatures for the Refactored Board State ---

piece_bbs_signature = nb.types.UniTuple(nb.uint64, 12)
"""
Numba type signature for the piece bitboards.
"""

occupancy_bbs_signature = nb.types.UniTuple(nb.uint64, 3)
"""
Numba type signature for the occupancy bitboards.
"""

game_state_signature = nb.types.Tuple([
    nb.uint8,   # side_to_move
    nb.uint8,   # castling_rights
    nb.int8,    # en_passant_square
    nb.uint8,   # halfmove_clock
    nb.uint64   # zobrist_key
])
"""
Numba type signature for the core game state.
"""

board_state_flat_signature = nb.types.Tuple(
    list(piece_bbs_signature.types) +
    list(occupancy_bbs_signature.types) +
    [nb.uint8, nb.int8, nb.uint8]
)
"""
Numba type signature for the flattened board state.
"""

unmake_info_signature = nb.types.Tuple([
    nb.int8,    # captured_piece_type
    nb.uint8,   # old_castling_rights
    nb.int8,    # old_en_passant_square
    nb.uint8    # old_halfmove_clock
])
"""
Numba type signature for the unmake_info tuple.
"""
