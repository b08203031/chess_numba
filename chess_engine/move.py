# chess_engine/move.py
import numpy as np
import numba as nb

# --- Move Encoding Blueprint ---
# A move is encoded as a 16-bit unsigned integer.
# Bits | 15, 14       | 13, 12           | 11, 10, 9, 8, 7, 6  | 5, 4, 3, 2, 1, 0
# Use  | Special Flag | Promotion Piece  | To Square           | From Square

# --- Constants for Bit Manipulation ---
FROM_SQUARE_MASK = np.uint16(0x003F)  # 0000 0000 0011 1111
TO_SQUARE_MASK = np.uint16(0x0FC0)    # 0000 1111 1100 0000
PROMOTION_PIECE_MASK = np.uint16(0x3000) # 0011 0000 0000 0000
SPECIAL_MOVE_MASK = np.uint16(0xC000) # 1100 0000 0000 0000

TO_SQUARE_SHIFT = 6
PROMOTION_PIECE_SHIFT = 12
SPECIAL_MOVE_SHIFT = 14

# --- Constants for Special Move Flags ---
NORMAL_MOVE = np.uint16(0)
PROMOTION = np.uint16(1)
EN_PASSANT = np.uint16(2)
CASTLING = np.uint16(3)

# --- Constants for Promotion Pieces ---
# Note: These values (0-3) map directly to the piece types offset.
# e.g., KNIGHT = 0, so promotion to knight adds 0 to the piece type base.
KNIGHT, BISHOP, ROOK, QUEEN = 0, 1, 2, 3

@nb.jit(nopython=True, inline='always')
def encode_move(from_square: int, to_square: int, promotion_piece: int, special_flag: int) -> np.uint16:
    """
    Encodes move information into a 16-bit unsigned integer.
    """
    move = np.uint16(from_square)
    move |= np.uint16(to_square << TO_SQUARE_SHIFT)
    move |= np.uint16(promotion_piece << PROMOTION_PIECE_SHIFT)
    move |= np.uint16(special_flag << SPECIAL_MOVE_SHIFT)
    return move

@nb.jit(nopython=True, inline='always')
def get_from_square(move: np.uint16) -> int:
    """Extracts the from_square from a move."""
    return int(move & FROM_SQUARE_MASK)

@nb.jit(nopython=True, inline='always')
def get_to_square(move: np.uint16) -> int:
    """Extracts the to_square from a move."""
    return int((move & TO_SQUARE_MASK) >> TO_SQUARE_SHIFT)

@nb.jit(nopython=True, inline='always')
def get_promotion_piece(move: np.uint16) -> int:
    """Extracts the promotion piece type from a move."""
    return int((move & PROMOTION_PIECE_MASK) >> PROMOTION_PIECE_SHIFT)

@nb.jit(nopython=True, inline='always')
def get_special_move_flag(move: np.uint16) -> int:
    """Extracts the special move flag from a move."""
    return int((move & SPECIAL_MOVE_MASK) >> SPECIAL_MOVE_SHIFT)
