import numpy as np
import numba as nb
from numba.extending import intrinsic
from numba import types
from chess_engine.constants import BB_SQUARES
from chess_engine.engine_types import piece_bbs_signature

# hardware intrinsic for trailing zeros
@intrinsic
def count_trailing_zeros(typingctx, val):
    def codegen(context, builder, signature, args):
        val_arg = args[0]
        # llvm.cttz.i64(i64 <src>, i1 <is_zero_undef>)
        # We pass False for is_zero_undef, so it is defined for 0 (returns 64).
        return builder.cttz(val_arg, context.get_constant(types.boolean, False))

    sig = types.int64(types.uint64)
    return sig, codegen

@nb.njit(nb.int8(nb.uint64), cache=True, inline='always')
def get_lsb_index(bitboard: np.uint64) -> int:
    """
    Uses hardware CTZ (Count Trailing Zeros) intrinsic via LLVM to find LSB index.
    """
    if bitboard == 0:
        return -1
    return nb.int8(count_trailing_zeros(bitboard))

def _init_king_attack_zones():
    """
    預計算棋盤上每個方格的 3x3 王的攻擊區域再加上前面一排 1x3 的區域。
    3x3 區域以王為中心。

    Returns:
        np.array: 大小為 64 的 uint64 陣列，每個元素代表對應方格的王周圍的攻擊區域位元棋盤。
    """
    zones = np.zeros(64, dtype=np.uint64)
    for sq in range(64):
        zone_bb = np.uint64(0)
        rank, file = sq // 8, sq % 8

        for r in range(rank - 1, rank + 3):
            for f in range(file - 1, file + 2):
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
