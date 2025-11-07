# chess_engine/move.py
import numpy as np
import numba

# --- Move Encoding Blueprint ---
# A move is encoded as a 16-bit unsigned integer.
# Bits | 15, 14       | 13, 12           | 11, 10, 9, 8, 7, 6  | 5, 4, 3, 2, 1, 0
# Use  | Special Flag | Promotion Piece  | To Square           | From Square

FROM_SQUARE_MASK = np.uint16(0x003F)
"""
A bitmask to extract the from_square from a move.
"""
TO_SQUARE_MASK = np.uint16(0x0FC0)
"""
A bitmask to extract the to_square from a move.
"""
PROMOTION_PIECE_MASK = np.uint16(0x3000)
"""
A bitmask to extract the promotion piece from a move.
"""
SPECIAL_MOVE_MASK = np.uint16(0xC000)
"""
A bitmask to extract the special move flag from a move.
"""

TO_SQUARE_SHIFT = 6
"""
The number of bits to shift to get the to_square.
"""
PROMOTION_PIECE_SHIFT = 12
"""
The number of bits to shift to get the promotion piece.
"""
SPECIAL_MOVE_SHIFT = 14
"""
The number of bits to shift to get the special move flag.
"""

SPECIAL_MOVE_FLAG_NORMAL = np.uint16(0)
"""
A flag that indicates a normal move.
"""
SPECIAL_MOVE_FLAG_PROMOTION = np.uint16(1)
"""
A flag that indicates a promotion move.
"""
SPECIAL_MOVE_FLAG_EN_PASSANT = np.uint16(2)
"""
A flag that indicates an en passant move.
"""
SPECIAL_MOVE_FLAG_CASTLING = np.uint16(3)
"""
A flag that indicates a castling move.
"""

PROMO_KNIGHT, PROMO_BISHOP, PROMO_ROOK, PROMO_QUEEN = 0, 1, 2, 3
"""
Constants for promotion pieces.
"""

@numba.jit(nopython=True, inline='always')
def encode_move(from_square: int, to_square: int, promotion_piece: int, special_flag: int) -> np.uint16:
    """
    Encodes move information into a 16-bit unsigned integer.

    Args:
        from_square: The square the piece is moving from.
        to_square: The square the piece is moving to.
        promotion_piece: The piece to promote to.
        special_flag: The special move flag.

    Returns:
        A 16-bit unsigned integer representing the move.
    """
    move = np.uint16(from_square)
    move |= np.uint16(to_square << TO_SQUARE_SHIFT)
    move |= np.uint16(promotion_piece << PROMOTION_PIECE_SHIFT)
    move |= np.uint16(special_flag << SPECIAL_MOVE_SHIFT)
    return move

@numba.jit(nopython=True, inline='always')
def get_from_square(move: np.uint16) -> int:
    """
    Extracts the from_square from a move.

    Args:
        move: The move to extract the from_square from.

    Returns:
        The from_square of the move.
    """
    return int(move & FROM_SQUARE_MASK)

@numba.jit(nopython=True, inline='always')
def get_to_square(move: np.uint16) -> int:
    """
    Extracts the to_square from a move.

    Args:
        move: The move to extract the to_square from.

    Returns:
        The to_square of the move.
    """
    return int((move & TO_SQUARE_MASK) >> TO_SQUARE_SHIFT)

@numba.jit(nopython=True, inline='always')
def get_promotion_piece(move: np.uint16) -> int:
    """
    Extracts the promotion piece type from a move.

    Args:
        move: The move to extract the promotion piece from.

    Returns:
        The promotion piece of the move.
    """
    return int((move & PROMOTION_PIECE_MASK) >> PROMOTION_PIECE_SHIFT)

@numba.jit(nopython=True, inline='always')
def get_special_move_flag(move: np.uint16) -> int:
    """
    Extracts the special move flag from a move.

    Args:
        move: The move to extract the special move flag from.

    Returns:
        The special move flag of the move.
    """
    return int((move & SPECIAL_MOVE_MASK) >> SPECIAL_MOVE_SHIFT)
