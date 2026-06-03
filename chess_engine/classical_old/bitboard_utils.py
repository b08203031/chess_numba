import numpy as np
import numba as nb
from numba.extending import intrinsic
from numba import types
from chess_engine.classical_old.constants import BB_SQUARES
from chess_engine.classical_old.engine_types import piece_bbs_signature

# hardware intrinsic for leading zeros
@intrinsic
def count_leading_zeros(typingctx, val):
    def codegen(context, builder, signature, args):
        val_arg = args[0]
        # llvm.ctlz.i64(i64 <src>, i1 <is_zero_undef>)
        # We pass False for is_zero_undef, so it is defined for 0 (returns 64).
        return builder.ctlz(val_arg, context.get_constant(types.boolean, False))

    sig = types.int64(types.uint64)
    return sig, codegen

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

# hardware intrinsic for set bits (population count)
@intrinsic
def count_set_bits(typingctx, val):
    def codegen(context, builder, signature, args):
        res = builder.ctpop(args[0])
        # Return as int32
        return builder.trunc(res, context.get_data_type(types.int32))
    sig = types.int32(types.uint64)
    return sig, codegen

@nb.njit(nb.int8(nb.uint64), cache=True, inline='always')
def get_lsb_index(bitboard: np.uint64) -> int:
    """
    Uses hardware CTZ (Count Trailing Zeros) intrinsic via LLVM to find LSB index.
    """
    if bitboard == 0:
        return -1
    return nb.int8(count_trailing_zeros(bitboard))

@nb.njit(nb.int8(nb.uint64), cache=True, inline='always')
def get_msb_index(bitboard: np.uint64) -> int:
    """
    Uses hardware CLZ (Count Leading Zeros) intrinsic via LLVM to find MSB index.
    """
    if bitboard == 0:
        return -1
    return nb.int8(63 - count_leading_zeros(bitboard))

def _init_king_attack_zones(color):
    """
    Precomputes the king attack zone for each square on the board, following SF11 logic.
    The king's position is clamped to B2-G7 to ensure a consistent 3x3 neighborhood.
    預計算棋盤上每個方格的王攻擊區域，遵循 SF11 邏輯。
    將王位限制在 B2-G7 以確保穩定的 3x3 鄰域格。

    Args:
        color (int): Ignored for geometry (kept for signature compatibility).

    Returns:
        np.array: Array of 64 uint64 bitboards.
    """
    zones = np.zeros(64, dtype=np.uint64)
    for sq in range(64):
        rank, file = sq // 8, sq % 8
        
        # SF11 Logic: clamp rank and file to [1, 6] (B2 to G7)
        clamped_rank = max(1, min(6, rank))
        clamped_file = max(1, min(6, file))
        
        zone_bb = np.uint64(0)
        # Generate 3x3 neighborhood around the clamped coordinate
        for r in range(clamped_rank - 1, clamped_rank + 2):
            for f in range(clamped_file - 1, clamped_file + 2):
                zone_bb |= BB_SQUARES[r * 8 + f]
        
        # SF11's kingRing includes the king's square itself. 
        # No excluding BB_SQUARES[sq] here.
        zones[sq] = zone_bb
    return zones

WHITE_KING_ZONES = _init_king_attack_zones(0)
BLACK_KING_ZONES = _init_king_attack_zones(1)

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

def _init_squares_between():
    """
    Pre-computes squares between any two squares on the same rank, file, or diagonal.
    """
    sq_between = np.zeros((64, 64), dtype=np.uint64)
    for s1 in range(64):
        for s2 in range(64):
            if s1 == s2: continue
            
            r1, f1 = s1 // 8, s1 % 8
            r2, f2 = s2 // 8, s2 % 8
            
            if r1 == r2: # Same rank
                df = 1 if f2 > f1 else -1
                for f in range(f1 + df, f2, df):
                    sq_between[s1, s2] |= BB_SQUARES[r1 * 8 + f]
            elif f1 == f2: # Same file
                dr = 1 if r2 > r1 else -1
                for r in range(r1 + dr, r2, dr):
                    sq_between[s1, s2] |= BB_SQUARES[r * 8 + f1]
            elif abs(r1 - r2) == abs(f1 - f2): # Same diagonal
                dr = 1 if r2 > r1 else -1
                df = 1 if f2 > f1 else -1
                for i in range(1, abs(r1 - r2)):
                    sq_between[s1, s2] |= BB_SQUARES[(r1 + i * dr) * 8 + (f1 + i * df)]
    return sq_between

SQUARES_BETWEEN = _init_squares_between()

def _init_rays():
    """
    Pre-computes orthogonal and diagonal rays for each square.
    """
    rook_rays = np.zeros(64, dtype=np.uint64)
    bishop_rays = np.zeros(64, dtype=np.uint64)
    for sq in range(64):
        r, f = sq // 8, sq % 8
        # Rook rays
        for r_curr in range(r + 1, 8): rook_rays[sq] |= BB_SQUARES[r_curr * 8 + f]
        for r_curr in range(r - 1, -1, -1): rook_rays[sq] |= BB_SQUARES[r_curr * 8 + f]
        for f_curr in range(f + 1, 8): rook_rays[sq] |= BB_SQUARES[r * 8 + f_curr]
        for f_curr in range(f - 1, -1, -1): rook_rays[sq] |= BB_SQUARES[r * 8 + f_curr]
        # Bishop rays
        for i in range(1, 8):
            if r+i < 8 and f+i < 8: bishop_rays[sq] |= BB_SQUARES[(r+i)*8 + (f+i)]
            if r+i < 8 and f-i >= 0: bishop_rays[sq] |= BB_SQUARES[(r+i)*8 + (f-i)]
            if r-i >= 0 and f+i < 8: bishop_rays[sq] |= BB_SQUARES[(r-i)*8 + (f+i)]
            if r-i >= 0 and f-i >= 0: bishop_rays[sq] |= BB_SQUARES[(r-i)*8 + (f-i)]
    return rook_rays, bishop_rays

ROOK_RAYS, BISHOP_RAYS = _init_rays()

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

@nb.njit(nb.int8(piece_bbs_signature, nb.uint8, nb.uint8), cache=True, inline='always')
def find_piece_type_on_square_side(piece_bbs, square, side):
    """
    Finds the piece type on a square, but only for the given side (0=White, 1=Black).
    Reduces the search space from 12 to 6 bitboards.
    """
    bb_square = BB_SQUARES[square]
    start = 0 if side == 0 else 6
    for piece_type in range(start, start + 6):
        if piece_bbs[piece_type] & bb_square:
            return nb.int8(piece_type)
    return nb.int8(-1)

@nb.njit(nb.int32(nb.uint64), cache=True)
def count_bits(bb: np.uint64) -> np.int32:
    """
    Uses hardware POPCNT intrinsic via LLVM to find the number of set bits.
    """
    return count_set_bits(bb)
