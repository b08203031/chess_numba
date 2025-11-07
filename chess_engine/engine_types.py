# chess_engine/types.py
import numba as nb
import numpy as np

# --- Numba Type Signatures for the Refactored Board State ---
# This provides a single source of truth for how Numba should interpret the board_state tuple.

# 1. Piece Bitboards: A uniform tuple of 12 uint64s. Numba can index this with a variable.
piece_bbs_signature = nb.types.UniTuple(nb.uint64, 12)

# 2. Occupancy Bitboards: A uniform tuple of 3 uint64s.
occupancy_bbs_signature = nb.types.UniTuple(nb.uint64, 3)

# 3. Core Game State: A tuple with mixed, but well-defined, types.
game_state_signature = nb.types.Tuple([
    nb.uint8,   # side_to_move
    nb.uint8,   # castling_rights
    nb.int8,    # en_passant_square
    nb.uint8,   # halfmove_clock
    nb.uint64   # zobrist_key
])

# 4. Flattened Board State for Move Generation: A heterogeneous tuple.
# (12 piece_bbs, 3 occupancy_bbs, castling_rights, en_passant_square, side_to_move)
board_state_flat_signature = nb.types.Tuple(
    list(piece_bbs_signature.types) +       # 12 x uint64
    list(occupancy_bbs_signature.types) +   # 3 x uint64
    [nb.uint8, nb.int8, nb.uint8]          # castling, ep_square, side_to_move
)

# --- Official Numba Type Signature for unmake_info ---
# Corrected the type of old_en_passant_square to int8 to handle -1.
unmake_info_signature = nb.types.Tuple([
    nb.int8,    # captured_piece_type
    nb.uint8,   # old_castling_rights
    nb.int8,    # old_en_passant_square
    nb.uint8    # old_halfmove_clock
])
