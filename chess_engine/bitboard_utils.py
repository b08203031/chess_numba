
import numpy as np
import numba

@numba.njit(numba.int32(numba.uint64), cache=True)
def count_bits(bb):
    """
    Counts the number of set bits in a bitboard (popcount).
    This is a Numba-jitted function.
    """
    c = 0
    while bb > 0:
        bb &= (bb - np.uint64(1))
        c += 1
    return c

@numba.njit(numba.uint8(numba.uint64), cache=True)
def get_ls1b_index(bb):
    """
    Gets the index of the least significant 1st bit (LSB).
    This is a Numba-jitted function.
    """
    if bb == 0:
        return np.uint8(64) # Should not happen, but as a safeguard
    return np.uint8(np.log2(bb & -bb))
