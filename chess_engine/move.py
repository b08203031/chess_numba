# chess_engine/move.py

import numba
import numpy as np

# --- Constants for Encoding/Decoding ---
# Using 16-bit unsigned integers for moves
# Bits 0-5:   From square (0-63)
# Bits 6-11:  To square (0-63)
# Bits 12-13: Promotion piece type (00: None, 01: N, 10: B, 11: R)
# Bits 14-15: Special move flags (00: Normal, 01: Promotion, 10: En Passant, 11: Castling)

FROM_MASK = np.uint16(0x003F)      # 0b0000000000111111
TO_MASK = np.uint16(0x0FC0)       # 0b0000111111000000
PROMO_PIECE_MASK = np.uint16(0x3000)   # 0b0011000000000000
FLAG_MASK = np.uint16(0xC000)    # 0b1100000000000000

# Promotion Piece Types (encoded in bits 12-13)
PROMO_TYPE_NONE = np.uint16(0)
PROMO_TYPE_KNIGHT = np.uint16(1)
PROMO_TYPE_BISHOP = np.uint16(2)
PROMO_TYPE_ROOK = np.uint16(3)
# Note: Queen promotion is handled by a combination of the promotion flag and this piece type.
# For simplicity, we can use one of the flags to signify Queen, e.g. PROMO_TYPE_ROOK + FLAG_PROMOTION might signify queen
# Let's adjust the flags to be more specific.

# Special Move Flags (encoded in bits 14-15)
FLAG_NORMAL = np.uint16(0)
FLAG_PROMOTION = np.uint16(1)
FLAG_EN_PASSANT = np.uint16(2)
FLAG_CASTLING = np.uint16(3)

# Let's refine the promotion logic. A better way is to use the promo piece bits for all four pieces.
# We need 2 bits for 4 pieces. Let's make it simpler.
# Bits 12-13 for promotion piece type
# 00 -> Knight, 01 -> Bishop, 10 -> Rook, 11 -> Queen
# We will use the promotion FLAG to indicate if it's a promotion move.

PROMO_KNIGHT = np.uint16(0)
PROMO_BISHOP = np.uint16(1)
PROMO_ROOK = np.uint16(2)
PROMO_QUEEN = np.uint16(3)


# --- Numba JIT'd Helper Functions ---

@numba.jit(nopython=True, inline='always')
def encode_move(from_sq, to_sq, flag, promo_piece=PROMO_TYPE_NONE):
    """
    Encodes a move into a 16-bit integer.
    - from_sq: The starting square (0-63).
    - to_sq: The destination square (0-63).
    - flag: The move type (FLAG_NORMAL, FLAG_PROMOTION, etc.).
    - promo_piece: The piece to promote to (PROMO_KNIGHT, etc.).
    """
    return np.uint16(from_sq | (to_sq << 6) | (promo_piece << 12) | (flag << 14))

@numba.jit(nopython=True, inline='always')
def get_from_square(move):
    """Extracts the starting square from an encoded move."""
    return np.uint8(move & FROM_MASK)

@numba.jit(nopython=True, inline='always')
def get_to_square(move):
    """Extracts the destination square from an encoded move."""
    return np.uint8((move & TO_MASK) >> 6)

@numba.jit(nopython=True, inline='always')
def get_promotion_piece(move):
    """Extracts the promotion piece type from an encoded move."""
    return np.uint8((move & PROMO_PIECE_MASK) >> 12)

@numba.jit(nopython=True, inline='always')
def get_move_flag(move):
    """Extracts the special move flag from an encoded move."""
    return np.uint8((move & FLAG_MASK) >> 14)

@numba.jit(nopython=True, inline='always')
def is_promotion(move):
    """Checks if the move is a promotion."""
    return get_move_flag(move) == FLAG_PROMOTION

@numba.jit(nopython=True, inline='always')
def is_en_passant(move):
    """Checks if the move is an en passant capture."""
    return get_move_flag(move) == FLAG_EN_PASSANT

@numba.jit(nopython=True, inline='always')
def is_castling(move):
    """Checks if the move is a castling move."""
    return get_move_flag(move) == FLAG_CASTLING
