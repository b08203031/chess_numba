import numpy as np
import numba as nb
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

def _init_color_specific_king_zones(is_white):
    """
    Precomputes color-specific king zones (king moves + 3 squares forward).
    """
    zones = np.zeros(64, dtype=np.uint64)
    for sq in range(64):
        zone_bb = np.uint64(0)
        rank, file = sq // 8, sq % 8

        # King's standard 3x3 zone
        for r in range(rank - 1, rank + 2):
            for f in range(file - 1, file + 2):
                if 0 <= r < 8 and 0 <= f < 8:
                    zone_bb |= BB_SQUARES[r * 8 + f]

        # Add 3 squares in front, depending on color
        if is_white:
            if rank < 5: # White king on rank 1-6
                for i in range(1, 4):
                    if 0 <= file - 1 < 8: zone_bb |= BB_SQUARES[sq + i*8 - 1]
                    zone_bb |= BB_SQUARES[sq + i*8]
                    if 0 <= file + 1 < 8: zone_bb |= BB_SQUARES[sq + i*8 + 1]
        else: # Black
            if rank > 2: # Black king on rank 8-3
                 for i in range(1, 4):
                    if 0 <= file - 1 < 8: zone_bb |= BB_SQUARES[sq - i*8 - 1]
                    zone_bb |= BB_SQUARES[sq - i*8]
                    if 0 <= file + 1 < 8: zone_bb |= BB_SQUARES[sq - i*8 + 1]

        zones[sq] = zone_bb
    return zones

WHITE_KING_ZONES = _init_color_specific_king_zones(is_white=True)
BLACK_KING_ZONES = _init_color_specific_king_zones(is_white=False)

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

@nb.njit(nb.int8(piece_bbs_signature, nb.uint8), cache=True)
def find_piece_type_on_square(piece_bbs, square):
    """
    Finds the piece type (0-11) on a given square.
    Returns -1 if no piece is found.
    """
    bb_square = BB_SQUARES[square]
    for piece_type in range(12):
        if piece_bbs[piece_type] & bb_square:
            return nb.int8(piece_type)
    return nb.int8(-1)

@nb.njit(nb.int32(nb.uint64), cache=True)
def count_bits(bb: np.uint64) -> np.int32:
    """
    Counts the number of set bits in a uint64 bitboard using SWAR.
    This is an O(1) constant-time operation.
    """
    # A series of parallel bitwise operations
    bb = bb - ((bb >> np.uint64(1)) & np.uint64(0x5555555555555555))
    bb = (bb & np.uint64(0x3333333333333333)) + ((bb >> np.uint64(2)) & np.uint64(0x3333333333333333))
    bb = (bb + (bb >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return np.int32((bb * np.uint64(0x0101010101010101)) >> np.uint64(56))
