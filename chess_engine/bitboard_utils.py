import numpy as np
import numba
from chess_engine.constants import BB_SQUARES
from chess_engine.engine_types import piece_bbs_signature

def _init_king_attack_zones():
    """
    Precomputes 5x5 king attack zones for each square on the board.
    A 5x5 zone is centered around the king.
    """
    zones = np.zeros(64, dtype=np.uint64)
    for sq in range(64):
        zone_bb = np.uint64(0)
        rank, file = sq // 8, sq % 8

        for r in range(rank - 2, rank + 3):
            for f in range(file - 2, file + 3):
                if 0 <= r < 8 and 0 <= f < 8:
                    target_sq = r * 8 + f
                    zone_bb |= BB_SQUARES[target_sq]
        
        # Exclude the king's own square
        zone_bb &= ~BB_SQUARES[sq]
        zones[sq] = zone_bb
    return zones

KING_ATTACK_ZONES = _init_king_attack_zones()

def _init_file_masks():
    """
    Precomputes bitboard masks for each file.
    """
    masks = np.zeros(8, dtype=np.uint64)
    for f in range(8):
        mask = np.uint64(0)
        for r in range(8):
            mask |= BB_SQUARES[r * 8 + f]
        masks[f] = mask
    return masks

FILE_MASKS = _init_file_masks()

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

