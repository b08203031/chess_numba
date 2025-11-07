# chess_engine/transposition_table.py
import numpy as np
import numba as nb
from chess_engine.constants import TT_SIZE_MB

# 1. Define constants for TT entry flags
TT_FLAG_NONE = 0
TT_FLAG_EXACT = 1  # Exact score (score is between alpha and beta)
TT_FLAG_ALPHA = 2  # Upper bound (score <= alpha), an ALL-node
TT_FLAG_BETA = 3   # Lower bound (score >= beta), a CUT-node

# 2. Define the data type (dtype) for a transposition table entry
tt_entry_dtype = np.dtype([
    ('key', np.uint64),
    ('score', np.int16),
    ('depth', np.uint8),
    ('flag', np.uint8),
    ('best_move', np.uint16)
])

# Convert the NumPy dtype to a Numba-compatible type
numba_tt_entry_type = nb.from_dtype(tt_entry_dtype)

# Create a global, "empty" entry to return on a TT miss for Numba type stability
_EMPTY_TT_ENTRY = np.zeros(1, dtype=tt_entry_dtype)[0]
_EMPTY_TT_ENTRY['flag'] = TT_FLAG_NONE

def create_transposition_table(size_mb):
    """Initializes the transposition table based on the specified size in MB."""
    entry_size_bytes = tt_entry_dtype.itemsize
    num_entries = (size_mb * 1024 * 1024) // entry_size_bytes
    transposition_table = np.zeros(num_entries, dtype=tt_entry_dtype)
    return transposition_table

@nb.njit(cache=True)
def clear_transposition_table(tt):
    """Clears all entries in the transposition table by setting each field to zero."""
    for i in range(len(tt)):
        tt[i]['key'] = np.uint64(0)
        tt[i]['score'] = np.int16(0)
        tt[i]['depth'] = np.uint8(0)
        tt[i]['flag'] = np.uint8(0)
        tt[i]['best_move'] = np.uint16(0)

@nb.njit(cache=True)
def probe_tt(tt, zobrist_key):
    """Looks up an entry in the transposition table. Returns the entry if the key matches."""
    index = zobrist_key % len(tt)
    entry = tt[index]
    if entry['key'] == zobrist_key:
        return entry
    else:
        return _EMPTY_TT_ENTRY

@nb.njit(cache=True)
def store_tt(tt, zobrist_key, depth, score, flag, best_move):
    """Stores an entry in the transposition table using a depth-first replacement strategy."""
    index = zobrist_key % len(tt)
    existing_entry = tt[index]

    if depth >= existing_entry['depth']:
        tt[index]['key'] = zobrist_key
        tt[index]['depth'] = np.uint8(depth)
        tt[index]['score'] = np.int16(score)
        tt[index]['flag'] = np.uint8(flag)
        tt[index]['best_move'] = np.uint16(best_move)
