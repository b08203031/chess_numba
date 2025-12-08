import numpy as np
import numba as nb
from chess_engine.constants import BB_SQUARES
from chess_engine.engine_types import piece_bbs_signature

def _init_king_attack_zones():
    """
    預計算棋盤上每個方格的 5x5 王的攻擊區域。
    5x5 區域以王為中心。

    Returns:
        np.array: 大小為 64 的 uint64 陣列，每個元素代表對應方格的王周圍的攻擊區域位元棋盤。
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
        
        # Exclude the king's own square / 排除王所在的方格
        zone_bb &= ~BB_SQUARES[sq]
        zones[sq] = zone_bb
    return zones

KING_ATTACK_ZONES = _init_king_attack_zones()

def _init_file_masks():
    """
    預計算每條直線（File）的位元棋盤掩碼。

    Returns:
        np.array: 大小為 8 的 uint64 陣列，每個元素代表一條直線的掩碼。
    """
    masks = np.zeros(8, dtype=np.uint64)
    for f in range(8):
        mask = np.uint64(0)
        for r in range(8):
            mask |= BB_SQUARES[r * 8 + f]
        masks[f] = mask
    return masks

FILE_MASKS = _init_file_masks()

def _init_between_bb():
    """
    預計算兩點之間的連線（Ray）。
    如果兩點不在同一條線（橫、直、斜）上，則為 0。
    不包含端點。
    """
    table = np.zeros((64, 64), dtype=np.uint64)
    for sq1 in range(64):
        for sq2 in range(64):
            if sq1 == sq2:
                continue

            # 檢查是否共線
            r1, f1 = sq1 // 8, sq1 % 8
            r2, f2 = sq2 // 8, sq2 % 8

            dr = r2 - r1
            df = f2 - f1

            # 判斷方向
            step_r, step_f = 0, 0

            if dr == 0: # 同 Rank
                step_f = 1 if df > 0 else -1
            elif df == 0: # 同 File
                step_r = 1 if dr > 0 else -1
            elif abs(dr) == abs(df): # 同 Diagonal
                step_r = 1 if dr > 0 else -1
                step_f = 1 if df > 0 else -1
            else:
                continue # 不共線

            # 生成 Ray
            bb = np.uint64(0)
            curr_r, curr_f = r1 + step_r, f1 + step_f
            while curr_r != r2 or curr_f != f2:
                bb |= BB_SQUARES[curr_r * 8 + curr_f]
                curr_r += step_r
                curr_f += step_f

            table[sq1][sq2] = bb
    return table

BETWEEN_BB = _init_between_bb()

@nb.njit(nb.int8(piece_bbs_signature, nb.uint8), cache=True)
def find_piece_type_on_square(piece_bbs, square):
    """
    在給定的方格上尋找棋子類型（0-11）。
    
    Args:
        piece_bbs (np.uint64[::1]): 12 個棋子位元棋盤的陣列。
        square (int): 方格索引 (0-63)。

    Returns:
        int: 棋子類型索引 (0-11)，如果在該方格上沒有找到棋子，則返回 -1。
    """
    bb_square = BB_SQUARES[square]
    for piece_type in range(12):
        if piece_bbs[piece_type] & bb_square:
            return nb.int8(piece_type)
    return nb.int8(-1)

@nb.njit(nb.int32(nb.uint64), cache=True)
def count_bits(bb: np.uint64) -> np.int32:
    """
    使用 SWAR (SIMD within a register) 技術計算 uint64 位元棋盤中設置為 1 的位元數量。
    這是一個 O(1) 的常數時間操作，不依賴於設置位元的數量。
    
    Args:
        bb (np.uint64): 需要計算位元數的位元棋盤。
        
    Returns:
        np.int32: 設置為 1 的位元數量。
    """
    # A series of parallel bitwise operations / 一系列並行的位元運算
    bb = bb - ((bb >> np.uint64(1)) & np.uint64(0x5555555555555555))
    bb = (bb & np.uint64(0x3333333333333333)) + ((bb >> np.uint64(2)) & np.uint64(0x3333333333333333))
    bb = (bb + (bb >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return np.int32((bb * np.uint64(0x0101010101010101)) >> np.uint64(56))

@nb.njit(nb.uint64(nb.uint8, nb.uint8), cache=True)
def get_between_bb(sq1, sq2):
    return BETWEEN_BB[sq1, sq2]
