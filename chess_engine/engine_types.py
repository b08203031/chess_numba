# chess_engine/engine_types.py
import numba
import numpy as np

# --- Numba Type Signatures for the Refactored Board State ---

# The board state is now represented by 1D NumPy arrays of uint64
# This allows for MUTABLE (in-place) modifications within Numba functions.
piece_bbs_signature = numba.uint64[::1]
"""
Numba type signature for the piece bitboards (as a NumPy array).
"""

occupancy_bbs_signature = numba.uint64[::1]
"""
Numba type signature for the occupancy bitboards (as a NumPy array).
"""

game_state_signature = numba.uint64[::1]
"""
Numba type signature for the core game state (as a NumPy array).
All elements are uint64 for type simplicity in Numba.
- Index 0: side_to_move
- Index 1: castling_rights
- Index 2: en_passant_square (64 for none)
- Index 3: halfmove_clock
- Index 4: zobrist_key
"""

# The unmake_info tuple remains unchanged, as it's a simple data carrier.
unmake_info_signature = numba.types.Tuple([
    numba.int8,    # captured_piece_type
    numba.uint8,   # old_castling_rights
    numba.uint8,   # old_en_passant_square (64 for none)
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
