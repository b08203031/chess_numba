# chess_engine/move.py
import numpy as np
import numba

# --- Move Encoding Blueprint / 移動編碼藍圖 ---
# A move is encoded as a 16-bit unsigned integer.
# 移動被編碼為一個 16 位無符號整數。
# Bits | 15, 14       | 13, 12           | 11, 10, 9, 8, 7, 6  | 5, 4, 3, 2, 1, 0
# Use  | Special Flag | Promotion Piece  | To Square           | From Square

FROM_SQUARE_MASK = np.uint16(0x003F)
"""
用於從移動中提取「起始方格」的位元掩碼。
"""
TO_SQUARE_MASK = np.uint16(0x0FC0)
"""
用於從移動中提取「目標方格」的位元掩碼。
"""
PROMOTION_PIECE_MASK = np.uint16(0x3000)
"""
用於從移動中提取「升變棋子」的位元掩碼。
"""
SPECIAL_MOVE_MASK = np.uint16(0xC000)
"""
用於從移動中提取「特殊移動標誌」的位元掩碼。
"""

TO_SQUARE_SHIFT = 6
"""
獲取「目標方格」所需的位移位數。
"""
PROMOTION_PIECE_SHIFT = 12
"""
獲取「升變棋子」所需的位移位數。
"""
SPECIAL_MOVE_SHIFT = 14
"""
獲取「特殊移動標誌」所需的位移位數。
"""

SPECIAL_MOVE_FLAG_NORMAL = np.uint16(0)
"""
表示正常移動的標誌。
"""
SPECIAL_MOVE_FLAG_PROMOTION = np.uint16(1)
"""
表示升變移動的標誌。
"""
SPECIAL_MOVE_FLAG_EN_PASSANT = np.uint16(2)
"""
表示吃過路兵移動的標誌。
"""
SPECIAL_MOVE_FLAG_CASTLING = np.uint16(3)
"""
表示王車易位移動的標誌。
"""

PROMO_KNIGHT, PROMO_BISHOP, PROMO_ROOK, PROMO_QUEEN = 0, 1, 2, 3
"""
升變棋子的常量定義。
"""

@numba.jit(nopython=True, inline='always')
def encode_move(from_square: int, to_square: int, promotion_piece: int, special_flag: int) -> np.uint16:
    """
    將移動信息編碼為一個 16 位無符號整數。

    Args:
        from_square (int): 棋子移動的起始方格 (0-63)。
        to_square (int): 棋子移動的目標方格 (0-63)。
        promotion_piece (int): 升變後的棋子類型 (0-3: N, B, R, Q)，如果沒有升變則忽略。
        special_flag (int): 特殊移動標誌 (0: 正常, 1: 升變, 2: 吃過路兵, 3: 易位)。

    Returns:
        np.uint16: 代表該移動的 16 位無符號整數。
    """
    move = np.uint16(from_square)
    move |= np.uint16(to_square << TO_SQUARE_SHIFT)
    move |= np.uint16(promotion_piece << PROMOTION_PIECE_SHIFT)
    move |= np.uint16(special_flag << SPECIAL_MOVE_SHIFT)
    return move

@numba.jit(nopython=True, inline='always')
def get_from_square(move: np.uint16) -> int:
    """
    從編碼的移動中提取起始方格。

    Args:
        move (np.uint16): 編碼的移動。

    Returns:
        int: 起始方格索引 (0-63)。
    """
    return int(move & FROM_SQUARE_MASK)

@numba.jit(nopython=True, inline='always')
def get_to_square(move: np.uint16) -> int:
    """
    從編碼的移動中提取目標方格。

    Args:
        move (np.uint16): 編碼的移動。

    Returns:
        int: 目標方格索引 (0-63)。
    """
    return int((move & TO_SQUARE_MASK) >> TO_SQUARE_SHIFT)

@numba.jit(nopython=True, inline='always')
def get_promotion_piece(move: np.uint16) -> int:
    """
    從編碼的移動中提取升變棋子類型。

    Args:
        move (np.uint16): 編碼的移動。

    Returns:
        int: 升變棋子類型 (0-3)，需要映射到實際的棋子代碼。
    """
    return int((move & PROMOTION_PIECE_MASK) >> PROMOTION_PIECE_SHIFT)

@numba.jit(nopython=True, inline='always')
def get_special_move_flag(move: np.uint16) -> int:
    """
    從編碼的移動中提取特殊移動標誌。

    Args:
        move (np.uint16): 編碼的移動。

    Returns:
        int: 特殊移動標誌 (0-3)。
    """
    return int((move & SPECIAL_MOVE_MASK) >> SPECIAL_MOVE_SHIFT)

def move_to_uci(move: np.uint16) -> str:
    """
    將內部移動編碼轉換為 UCI 格式的字串 (例如 "e2e4", "a7a8q")。
    
    Args:
        move (np.uint16): 編碼的移動。
        
    Returns:
        str: UCI 格式的移動字串。
    """
    from .core import SQUARE_TO_ALGEBRAIC
    
    from_sq = get_from_square(move)
    to_sq = get_to_square(move)
    
    uci = SQUARE_TO_ALGEBRAIC[from_sq] + SQUARE_TO_ALGEBRAIC[to_sq]
    
    if get_special_move_flag(move) == SPECIAL_MOVE_FLAG_PROMOTION:
        promo_piece = get_promotion_piece(move)
        if promo_piece == PROMO_QUEEN:
            uci += 'q'
        elif promo_piece == PROMO_ROOK:
            uci += 'r'
        elif promo_piece == PROMO_BISHOP:
            uci += 'b'
        elif promo_piece == PROMO_KNIGHT:
            uci += 'n'
            
    return uci
