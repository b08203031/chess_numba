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
    numba.uint8,   # old_halfmove_clock
    numba.uint64   # old_zobrist_key
])
"""
Numba type signature for the unmake_info tuple.
"""

# --- Search Context ---
from numba.experimental import jitclass
from chess_engine.transposition_table import numba_tt_entry_type

# Define the specification for the SearchContext jitclass
search_context_spec = [
    ('transposition_table', numba.types.Array(numba_tt_entry_type, 1, 'C')),
    ('killer_moves', numba.uint16[::1]),
    ('pv_table', numba.uint16[:, :]),
    ('history_table', numba.int32[:, :]),
]

@jitclass(search_context_spec)
class SearchContext:
    def __init__(self, transposition_table, killer_moves, pv_table, history_table):
        self.transposition_table = transposition_table
        self.killer_moves = killer_moves
        self.pv_table = pv_table
        self.history_table = history_table

# Create the Numba type from the class
search_context_type = SearchContext.class_type.instance_type