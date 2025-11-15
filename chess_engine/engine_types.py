# chess_engine/engine_types.py
import numba
import numpy as np

# --- Color Constants ---
WHITE, BLACK = 0, 1

# --- Piece Type Constants ---
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 0, 1, 2, 3, 4, 5


# --- Numba Type Signatures for the Refactored Board State ---
piece_bbs_signature = numba.uint64[::1]
occupancy_bbs_signature = numba.uint64[::1]
game_state_signature = numba.uint64[::1]

unmake_info_signature = numba.types.Tuple([
    numba.int8, numba.uint8, numba.uint8, numba.uint8, numba.uint64
])

# --- Search Context ---
from numba.experimental import jitclass
from chess_engine.transposition_table import numba_tt_entry_type

search_context_spec = [
    ('transposition_table', numba.types.Array(numba_tt_entry_type, 1, 'C')),
    ('killer_moves', numba.uint16[::1]),
    ('pv_table', numba.uint16[:, :]),
    ('history_table', numba.int32[:, :]),
    ('nodes_searched', numba.uint64),
]

@jitclass(search_context_spec)
class SearchContext:
    def __init__(self, transposition_table, killer_moves, pv_table, history_table):
        self.transposition_table = transposition_table
        self.killer_moves = killer_moves
        self.pv_table = pv_table
        self.history_table = history_table
        self.nodes_searched = np.uint64(0)

search_context_type = SearchContext.class_type.instance_type
