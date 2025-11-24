# chess_engine/zobrist.py
import numpy as np
import numba
from chess_engine.engine_types import piece_bbs_signature, game_state_signature

BLACK = 1
"""
代表黑方的常量。
"""

np.random.seed(123456789)
PIECE_SQUARE_KEYS = np.random.randint(1, 2**64 - 1, (12, 64), dtype=np.uint64)
"""
用於每個方格上每個棋子的 Zobrist 鍵值。
是一個 12x64 的二維陣列。
"""
SIDE_TO_MOVE_KEY = np.random.randint(1, 2**64 - 1, dtype=np.uint64)
"""
用於當前行棋方的 Zobrist 鍵值（如果輪到黑方走棋，則 XOR 此值）。
"""
EN_PASSANT_FILE_KEYS = np.random.randint(1, 2**64 - 1, 8, dtype=np.uint64)
"""
用於每個吃過路兵直線（File）的 Zobrist 鍵值。
是一個長度為 8 的一維陣列。
"""
CASTLING_RIGHTS_KEYS = np.random.randint(1, 2**64 - 1, 16, dtype=np.uint64)
"""
用於每種王車易位權限組合的 Zobrist 鍵值。
是一個長度為 16 的一維陣列（對應 4 個位元的組合）。
"""

from chess_engine.constants import DE_BRUIJN_INDEX, DE_BRUIJN_SEQUENCE

@numba.jit(numba.int8(numba.uint64), nopython=True, inline='always')
def get_lsb_index(bitboard: np.uint64) -> int:
    """
    使用 De Bruijn 序列位掃描（Bitscan）技術尋找最低有效位（LSB）的索引。
    這是一個非常快速的 O(1) 操作。

    Args:
        bitboard (np.uint64): 要尋找 LSB 的位元棋盤。

    Returns:
        int: LSB 的索引 (0-63)。如果位元棋盤為空，則返回 -1。
    """
    bitboard = np.uint64(bitboard)
    if bitboard == 0:
        return -1
    lsb = bitboard & (-bitboard)
    # 使用 De Bruijn 序列進行哈希映射
    index = (lsb * DE_BRUIJN_SEQUENCE) >> np.uint64(58)
    return DE_BRUIJN_INDEX[index]

@numba.jit(numba.uint64(piece_bbs_signature, game_state_signature), nopython=True)
def compute_initial_hash(piece_bbs: np.ndarray, game_state: np.ndarray) -> np.uint64:
    """
    從頭計算當前局面的 Zobrist 哈希值。
    通常僅在初始化或除錯時使用，後續更新應使用增量更新。

    Args:
        piece_bbs (np.ndarray): 12 個棋子的位元棋盤。
        game_state (np.ndarray): 遊戲狀態陣列（包含行棋方、易位權限等）。

    Returns:
        np.uint64: 計算出的 Zobrist 哈希值。
    """
    zobrist_key = np.uint64(0)
    side_to_move = game_state[0]
    castling_rights = game_state[1]
    en_passant_square = game_state[2]

    # XOR 所有棋子的鍵值
    for piece_type in range(12):
        bb = piece_bbs[piece_type]
        while bb != 0:
            square = get_lsb_index(bb)
            zobrist_key ^= PIECE_SQUARE_KEYS[piece_type, square]
            bb &= np.uint64(bb - 1) # 清除 LSB

    # XOR 易位權限鍵值
    zobrist_key ^= CASTLING_RIGHTS_KEYS[castling_rights]

    # XOR 吃過路兵直線鍵值（如果有）
    if en_passant_square != 64:
        ep_file = en_passant_square % 8
        zobrist_key ^= EN_PASSANT_FILE_KEYS[ep_file]

    # 如果是黑方走棋，XOR 行棋方鍵值
    if side_to_move == BLACK:
        zobrist_key ^= SIDE_TO_MOVE_KEY

    return zobrist_key
