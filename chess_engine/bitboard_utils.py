
import numpy as np
import numba
from chess_engine.constants import BB_SQUARES
from chess_engine.engine_types import piece_bbs_signature

@numba.njit(numba.int8(piece_bbs_signature, numba.uint8), cache=True)
def find_piece_type_on_square(piece_bbs, square):
    """
    Finds the piece type (0-11) on a given square.
    Returns -1 if no piece is found.
    """
    bb_square = BB_SQUARES[square]
    for piece_type in range(12):
        if piece_bbs[piece_type] & bb_square:
            return numba.int8(piece_type)
    return numba.int8(-1)

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
