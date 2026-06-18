# chess_engine/classical_old/pawns.py

import numba
import numpy as np

from chess_engine.classical_old.constants import *
from chess_engine.classical_old.bitboard_utils import get_lsb_index, get_msb_index, count_bits, FILE_MASKS
from chess_engine.classical_old.move_generator import PAWN_ATTACKS
from chess_engine.classical_old.engine_types import (
    piece_bbs_signature, piece_counts_signature
)

NOT_A_FILE = ~np.uint64(0x0101010101010101)
NOT_H_FILE = ~np.uint64(0x8080808080808080)
CENTER_FILES = np.uint64(0x3C3C3C3C3C3C3C3C)  # Files C, D, E, F

# --- Pre-computed Manhattan Distance Table / 預計算曼哈頓距離表 ---
def _create_manhattan_distance_table():
    table = np.zeros((64, 64), dtype=np.int32)
    for sq1 in range(64):
        rank1, file1 = sq1 // 8, sq1 % 8
        for sq2 in range(64):
            rank2, file2 = sq2 // 8, sq2 % 8
            table[sq1, sq2] = abs(rank1 - rank2) + abs(file1 - file2)
    return table

MANHATTAN_DISTANCE = _create_manhattan_distance_table()

def _create_chebyshev_distance_table():
    table = np.zeros((64, 64), dtype=np.int32)
    for sq1 in range(64):
        rank1, file1 = sq1 // 8, sq1 % 8
        for sq2 in range(64):
            rank2, file2 = sq2 // 8, sq2 % 8
            table[sq1, sq2] = max(abs(rank1 - rank2), abs(file1 - file2))
    return table

CHEBYSHEV_DISTANCE = _create_chebyshev_distance_table()


# --- Pre-computed Masks for Pawn Structure Evaluation / 兵型評估的預計算掩碼 ---
ADJACENT_FILES_MASKS = np.array([
    FILE_MASKS[1],  # File A
    FILE_MASKS[0] | FILE_MASKS[2],  # File B
    FILE_MASKS[1] | FILE_MASKS[3],  # File C
    FILE_MASKS[2] | FILE_MASKS[4],  # File D
    FILE_MASKS[3] | FILE_MASKS[5],  # File E
    FILE_MASKS[4] | FILE_MASKS[6],  # File F
    FILE_MASKS[5] | FILE_MASKS[7],  # File G
    FILE_MASKS[6],  # File H
], dtype=np.uint64)

WHITE_FORWARD_RANKS = np.zeros(64, dtype=np.uint64)
BLACK_FORWARD_RANKS = np.zeros(64, dtype=np.uint64)

for sq in range(64):
    rank = sq // 8
    # White forward: ranks > rank
    mask = np.uint64(0)
    for r in range(rank + 1, 8):
        mask |= RANK_MASKS[r]
    WHITE_FORWARD_RANKS[sq] = mask

    # Black forward: ranks < rank
    mask = np.uint64(0)
    for r in range(0, rank):
        mask |= RANK_MASKS[r]
    BLACK_FORWARD_RANKS[sq] = mask


def _create_passed_pawn_masks():
    white_masks = np.zeros(64, dtype=np.uint64)
    black_masks = np.zeros(64, dtype=np.uint64)
    for sq in range(64):
        file_idx = sq % 8
        rank_idx = sq // 8
        
        path_mask = FILE_MASKS[file_idx] | ADJACENT_FILES_MASKS[file_idx]
        
        white_front_span = np.uint64(0)
        for r in range(rank_idx + 1, 8):
            white_front_span |= (path_mask & (np.uint64(0xFF) << np.uint64(r * 8)))
        white_masks[sq] = white_front_span

        black_front_span = np.uint64(0)
        for r in range(rank_idx - 1, -1, -1):
            black_front_span |= (path_mask & (np.uint64(0xFF) << np.uint64(r * 8)))
        black_masks[sq] = black_front_span
        
    return white_masks, black_masks

WHITE_PASSED_PAWN_MASKS, BLACK_PASSED_PAWN_MASKS = _create_passed_pawn_masks()

# Precompute forward file masks
WHITE_FORWARD_FILE_MASKS = np.zeros(64, dtype=np.uint64)
BLACK_FORWARD_FILE_MASKS = np.zeros(64, dtype=np.uint64)

for sq in range(64):
    file_idx = sq % 8
    WHITE_FORWARD_FILE_MASKS[sq] = WHITE_FORWARD_RANKS[sq] & FILE_MASKS[file_idx]
    BLACK_FORWARD_FILE_MASKS[sq] = BLACK_FORWARD_RANKS[sq] & FILE_MASKS[file_idx]


# --- Pre-computed Pawn Structure Masks ---
PHALANX_MASK = np.zeros(64, dtype=np.uint64)
WHITE_SUPPORT_MASK = np.zeros(64, dtype=np.uint64)
BLACK_SUPPORT_MASK = np.zeros(64, dtype=np.uint64)
WHITE_BACKWARD_TEST_MASK = np.zeros(64, dtype=np.uint64)
BLACK_BACKWARD_TEST_MASK = np.zeros(64, dtype=np.uint64)

for sq in range(64):
    file_idx = sq % 8
    rank = sq // 8
    
    # Phalanx
    PHALANX_MASK[sq] = ADJACENT_FILES_MASKS[file_idx] & RANK_MASKS[rank]
    
    # Support
    if rank > 0:
        WHITE_SUPPORT_MASK[sq] = ADJACENT_FILES_MASKS[file_idx] & RANK_MASKS[rank - 1]
    if rank < 7:
        BLACK_SUPPORT_MASK[sq] = ADJACENT_FILES_MASKS[file_idx] & RANK_MASKS[rank + 1]
        
    # Backward Test Masks
    if sq < 56:
        WHITE_BACKWARD_TEST_MASK[sq] = ADJACENT_FILES_MASKS[file_idx] & BLACK_FORWARD_RANKS[sq + 8]
    if sq >= 8:
        BLACK_BACKWARD_TEST_MASK[sq] = ADJACENT_FILES_MASKS[file_idx] & WHITE_FORWARD_RANKS[sq - 8]


@numba.njit(cache=True, boundscheck=False, fastmath=True)
def evaluate_shelter_aligned(piece_bbs, king_sq, color):
    """
    評估國王前方的兵盾（Friendly Shelter）與敵兵風暴（Enemy Storm）。
    使用 for 循環，LLVM 會自動展開以提高效能與可讀性。
    """
    mg_score = SHELTER_BASE_MG
    eg_score = SHELTER_BASE_EG
    king_rank = king_sq >> 3
    if color == 0:
        rank_mask = WHITE_FORWARD_RANKS[king_sq] | RANK_MASKS[king_rank]
    else:
        rank_mask = BLACK_FORWARD_RANKS[king_sq] | RANK_MASKS[king_rank]

    friendly_pawns = (piece_bbs[0] if color == 0 else piece_bbs[6]) & rank_mask
    enemy_pawns = (piece_bbs[6] if color == 0 else piece_bbs[0]) & rank_mask
    
    # 決定國王所在的中心直行 (B-G 線)
    king_file = king_sq & 7
    center_file = max(1, min(6, king_file))

    # 迴圈遍歷三個直線：center_file - 1, center_file, center_file + 1
    for f in range(center_file - 1, center_file + 2):
        file_mask = FILE_MASKS[f]
        
        our_pawns_on_file = friendly_pawns & file_mask
        our_rank = 0
        if our_pawns_on_file:
            pawn_sq = get_lsb_index(our_pawns_on_file) if color == 0 else get_msb_index(our_pawns_on_file)
            our_rank = (pawn_sq >> 3) if color == 0 else (7 - (pawn_sq >> 3))
            
        their_pawns_on_file = enemy_pawns & file_mask
        their_rank = 0
        if their_pawns_on_file:
            pawn_sq = get_lsb_index(their_pawns_on_file) if color == 0 else get_msb_index(their_pawns_on_file)
            their_rank = (pawn_sq >> 3) if color == 0 else (7 - (pawn_sq >> 3))
            
        d = f if f < 4 else 7 - f
        mg_score += SHELTER_STRENGTH[d, our_rank]
        
        if our_rank > 0 and our_rank == their_rank - 1:
            if their_rank == 2:
                mg_score += BLOCKED_STORM
                eg_score += BLOCKED_STORM_EG
        else:
            mg_score += UNBLOCKED_STORM[d, their_rank]
            
    return mg_score, eg_score


@numba.njit(cache=True, boundscheck=False, fastmath=True)
def evaluate_pawn_structure(piece_bbs, castling_rights):
    """
    評估雙方的兵型結構（通路兵、孤兵、重疊兵、後兵、連結兵、弱對決、開放線弱兵）。
    同時計算並返回兵盾與兵風暴評估分數（已併入國王安全）。
    
    Args:
        piece_bbs (np.ndarray): 12 個棋子的位元棋盤。
        castling_rights (np.uint8): 易位權限位元遮罩。
        
    Returns:
        tuple: (mg_score, eg_score, 
                white_mg_shield, white_eg_shield, black_mg_shield, black_eg_shield,
                w_passed, b_passed,
                white_pawn_attacks_span, black_pawn_attacks_span) 從白方視角。
    """
    mg_score = np.int32(0)
    eg_score = np.int32(0)
    
    white_pawns = piece_bbs[0]
    black_pawns = piece_bbs[6]
    
    white_king_sq = get_lsb_index(piece_bbs[5])
    black_king_sq = get_lsb_index(piece_bbs[11])
    
    w_passed = np.uint64(0)
    b_passed = np.uint64(0)
    
    # Precompute base pawn attacks and spans
    white_pawn_attacks = ((white_pawns & NOT_A_FILE) << np.uint64(7)) | ((white_pawns & NOT_H_FILE) << np.uint64(9))
    black_pawn_attacks = ((black_pawns & NOT_H_FILE) >> np.uint64(7)) | ((black_pawns & NOT_A_FILE) >> np.uint64(9))
    white_pawn_attacks_span = white_pawn_attacks
    black_pawn_attacks_span = black_pawn_attacks
 
    # Precompute double attacks needed for passed pawn condition C
    white_double_attacks = ((white_pawns & NOT_A_FILE) << np.uint64(7)) & ((white_pawns & NOT_H_FILE) << np.uint64(9))
    black_double_attacks = ((black_pawns & NOT_A_FILE) >> np.uint64(9)) & ((black_pawns & NOT_H_FILE) >> np.uint64(7))
    
    # --- 1. Iterate White Pawns ---
    temp_wp = white_pawns
    while temp_wp:
        sq = get_lsb_index(temp_wp)
        rank = sq >> 3
        file_idx = sq & 7
        mask_sq = BB_SQUARES[sq]

        # Extract bits for backward and lever checks
        lever = black_pawns & PAWN_ATTACKS[0, sq]
        lever_push = black_pawns & PAWN_ATTACKS[0, sq + 8]
        blocked = black_pawns & BB_SQUARES[sq + 8]
        
        phalanx = white_pawns & PHALANX_MASK[sq]
        support = white_pawns & WHITE_SUPPORT_MASK[sq]
        doubled = (white_pawns & BB_SQUARES[sq - 8]) != 0

        # Backward Pawn Detection
        backward = False
        if not (white_pawns & WHITE_BACKWARD_TEST_MASK[sq]) and (lever_push | blocked):
            backward = True

        # Project attacks span forward if pawn is not backward nor blocked
        if not backward and not blocked:
            white_pawn_attacks_span |= WHITE_FORWARD_RANKS[sq] & ADJACENT_FILES_MASKS[file_idx]

        opposed = (black_pawns & WHITE_FORWARD_FILE_MASKS[sq]) != 0
        
        # Evaluate Passed Pawn Conditions (Stopper/Lever/Support logic)
        stoppers = black_pawns & WHITE_PASSED_PAWN_MASKS[sq]
        cond_a = (stoppers ^ lever) == 0
        cond_b = False
        if (stoppers ^ lever_push) == 0:
            cond_b = count_bits(phalanx) >= count_bits(lever_push)
            
        cond_c = False
        if stoppers == blocked and rank >= 4:
            shifted_support = support << np.uint64(8)
            if (shifted_support & ~(black_pawns | black_double_attacks)) != 0:
                cond_c = True

        is_passed = cond_a or cond_b or cond_c

        if is_passed:
            w_passed |= mask_sq

        # Score calculations
        if support or phalanx:
            # Connected Bonus (using distinct MG/EG bonuses and weights for proper scaling)
            v_mg = CONNECTED_BONUS[rank] * (2 + (1 if phalanx else 0) - (1 if opposed else 0)) + CONNECTED_SUPPORT_WEIGHT * count_bits(support)
            v_eg = CONNECTED_BONUS_EG[rank] * (2 + (1 if phalanx else 0) - (1 if opposed else 0)) + CONNECTED_SUPPORT_WEIGHT_EG * count_bits(support)
            mg_score += v_mg
            eg_score += np.int32((v_eg * (rank - 2)) / 4)
        elif not (white_pawns & ADJACENT_FILES_MASKS[file_idx]):
            # Isolated
            mg_score += ISOLATED_PAWN_PENALTY[0]
            eg_score += ISOLATED_PAWN_PENALTY[1]
            if not opposed:
                mg_score += WEAK_UNOPPOSED_PENALTY[0]
                eg_score += WEAK_UNOPPOSED_PENALTY[1]
        elif backward:
            # Backward
            mg_score += BACKWARD_PAWN_PENALTY[0]
            eg_score += BACKWARD_PAWN_PENALTY[1]
            if not opposed:
                mg_score += WEAK_UNOPPOSED_PENALTY[0]
                eg_score += WEAK_UNOPPOSED_PENALTY[1]

        if not support:
            if doubled:
                mg_score += DOUBLED_PAWN_PENALTY[0]
                eg_score += DOUBLED_PAWN_PENALTY[1]
            if count_bits(lever) > 1:
                mg_score += WEAK_LEVER_PENALTY[0]
                eg_score += WEAK_LEVER_PENALTY[1]

        temp_wp &= temp_wp - np.uint64(1)

    # --- 2. Iterate Black Pawns ---
    temp_bp = black_pawns
    while temp_bp:
        sq = get_lsb_index(temp_bp)
        rank = sq >> 3
        relative_rank = 7 - rank
        file_idx = sq & 7
        mask_sq = BB_SQUARES[sq]

        # Extract bits for backward and lever checks
        lever = white_pawns & PAWN_ATTACKS[1, sq]
        lever_push = white_pawns & PAWN_ATTACKS[1, sq - 8]
        blocked = white_pawns & BB_SQUARES[sq - 8]
        
        phalanx = black_pawns & PHALANX_MASK[sq]
        support = black_pawns & BLACK_SUPPORT_MASK[sq]
        doubled = (black_pawns & BB_SQUARES[sq + 8]) != 0

        # Backward Pawn Detection
        backward = False
        if not (black_pawns & BLACK_BACKWARD_TEST_MASK[sq]) and (lever_push | blocked):
            backward = True

        # Project attacks span forward if pawn is not backward nor blocked
        if not backward and not blocked:
            black_pawn_attacks_span |= BLACK_FORWARD_RANKS[sq] & ADJACENT_FILES_MASKS[file_idx]

        opposed = (white_pawns & BLACK_FORWARD_FILE_MASKS[sq]) != 0
        
        # Evaluate Passed Pawn Conditions (Stopper/Lever/Support logic)
        stoppers = white_pawns & BLACK_PASSED_PAWN_MASKS[sq]
        cond_a = (stoppers ^ lever) == 0
        cond_b = False
        if (stoppers ^ lever_push) == 0:
            cond_b = count_bits(phalanx) >= count_bits(lever_push)
            
        cond_c = False
        if stoppers == blocked and relative_rank >= 4:
            shifted_support = support >> np.uint64(8)
            if (shifted_support & ~(white_pawns | white_double_attacks)) != 0:
                cond_c = True

        is_passed = cond_a or cond_b or cond_c

        if is_passed:
            b_passed |= mask_sq

        # Score calculations
        if support or phalanx:
            # Connected Bonus (using distinct MG/EG bonuses and weights for proper scaling)
            v_mg = CONNECTED_BONUS[relative_rank] * (2 + (1 if phalanx else 0) - (1 if opposed else 0)) + CONNECTED_SUPPORT_WEIGHT * count_bits(support)
            v_eg = CONNECTED_BONUS_EG[relative_rank] * (2 + (1 if phalanx else 0) - (1 if opposed else 0)) + CONNECTED_SUPPORT_WEIGHT_EG * count_bits(support)
            mg_score -= v_mg
            eg_score -= np.int32((v_eg * (relative_rank - 2)) / 4)
        elif not (black_pawns & ADJACENT_FILES_MASKS[file_idx]):
            # Isolated
            mg_score -= ISOLATED_PAWN_PENALTY[0]
            eg_score -= ISOLATED_PAWN_PENALTY[1]
            if not opposed:
                mg_score += WEAK_UNOPPOSED_PENALTY[0]
                eg_score += WEAK_UNOPPOSED_PENALTY[1]
        elif backward:
            # Backward
            mg_score -= BACKWARD_PAWN_PENALTY[0]
            eg_score -= BACKWARD_PAWN_PENALTY[1]
            if not opposed:
                mg_score += WEAK_UNOPPOSED_PENALTY[0]
                eg_score += WEAK_UNOPPOSED_PENALTY[1]

        if not support:
            if doubled:
                mg_score -= DOUBLED_PAWN_PENALTY[0]
                eg_score -= DOUBLED_PAWN_PENALTY[1]
            if count_bits(lever) > 1:
                mg_score -= WEAK_LEVER_PENALTY[0]
                eg_score -= WEAK_LEVER_PENALTY[1]

        temp_bp &= temp_bp - np.uint64(1)

    # --- 3. Evaluate Pawn Shield & Storm (Virtualized Castling aligned) ---
    # White Shield Evaluation
    white_mg_shield, white_eg_shield = evaluate_shelter_aligned(piece_bbs, white_king_sq, 0)
    if (castling_rights & 1) != 0:
        mg, eg = evaluate_shelter_aligned(piece_bbs, 6, 0) # G1
        if mg > white_mg_shield:
            white_mg_shield, white_eg_shield = mg, eg
    if (castling_rights & 2) != 0:
        mg, eg = evaluate_shelter_aligned(piece_bbs, 2, 0) # C1
        if mg > white_mg_shield:
            white_mg_shield, white_eg_shield = mg, eg

    # Black Shield Evaluation
    black_mg_shield, black_eg_shield = evaluate_shelter_aligned(piece_bbs, black_king_sq, 1)
    if (castling_rights & 4) != 0:
        mg, eg = evaluate_shelter_aligned(piece_bbs, 62, 1) # G8
        if mg > black_mg_shield:
            black_mg_shield, black_eg_shield = mg, eg
    if (castling_rights & 8) != 0:
        mg, eg = evaluate_shelter_aligned(piece_bbs, 58, 1) # C8
        if mg > black_mg_shield:
            black_mg_shield, black_eg_shield = mg, eg

    return (mg_score, eg_score, 
            white_mg_shield, white_eg_shield, black_mg_shield, black_eg_shield,
            w_passed, b_passed,
            white_pawn_attacks_span, black_pawn_attacks_span)


@numba.njit(cache=True, boundscheck=False, fastmath=True)
def evaluate_pawn_structure_cached(piece_bbs, pawn_key, castling_rights, search_context):
    """
    Cached version of evaluate_pawn_structure using Pawn Hash Table.
    """
    idx = int(pawn_key & np.uint64(0x3FFFF))
    white_king_sq = get_lsb_index(piece_bbs[5])
    black_king_sq = get_lsb_index(piece_bbs[11])
    
    # Verify pawn key, king positions, and castling rights to validate caches
    if (search_context.pawn_table_keys[idx] == pawn_key and 
        search_context.pawn_table_w_king_sq[idx] == white_king_sq and 
        search_context.pawn_table_b_king_sq[idx] == black_king_sq and
        search_context.pawn_table_castling_rights[idx] == castling_rights):
        
        return (
            search_context.pawn_table_mg[idx],
            search_context.pawn_table_eg[idx],
            search_context.pawn_table_w_shield_mg[idx],
            search_context.pawn_table_w_shield_eg[idx],
            search_context.pawn_table_b_shield_mg[idx],
            search_context.pawn_table_b_shield_eg[idx],
            search_context.pawn_table_w_passed[idx],
            search_context.pawn_table_b_passed[idx],
            search_context.pawn_table_w_attacks_span[idx],
            search_context.pawn_table_b_attacks_span[idx]
        )
        
    res = evaluate_pawn_structure(piece_bbs, castling_rights)
    
    # Store computed values in cache (write-always)
    search_context.pawn_table_keys[idx] = pawn_key
    search_context.pawn_table_w_king_sq[idx] = np.int8(white_king_sq)
    search_context.pawn_table_b_king_sq[idx] = np.int8(black_king_sq)
    search_context.pawn_table_castling_rights[idx] = castling_rights
    search_context.pawn_table_mg[idx] = res[0]
    search_context.pawn_table_eg[idx] = res[1]
    search_context.pawn_table_w_shield_mg[idx] = res[2]
    search_context.pawn_table_w_shield_eg[idx] = res[3]
    search_context.pawn_table_b_shield_mg[idx] = res[4]
    search_context.pawn_table_b_shield_eg[idx] = res[5]
    search_context.pawn_table_w_passed[idx] = res[6]
    search_context.pawn_table_b_passed[idx] = res[7]
    search_context.pawn_table_w_attacks_span[idx] = res[8]
    search_context.pawn_table_b_attacks_span[idx] = res[9]
    
    return res


@numba.njit(numba.types.UniTuple(numba.int32, 2)(piece_bbs_signature, piece_counts_signature), cache=True, boundscheck=False, fastmath=True)
def evaluate_piece_coordination(piece_bbs, piece_counts):
    """
    評估棋子協同性特徵（雙象、車在開放線）。
    
    Args:
        piece_bbs (np.ndarray): 12 個棋子的位元棋盤。
        piece_counts (np.ndarray): 12 個棋子的數量。
        
    Returns:
        tuple: (mg_score, eg_score) 從白方視角。
    """
    mg_score = np.int32(0)
    eg_score = np.int32(0)

    white_pawns = piece_bbs[0]
    white_rooks = piece_bbs[3]
    black_pawns = piece_bbs[6]
    black_rooks = piece_bbs[9]

    # --- 1. Bishop Pair / 雙象 ---
    # NOTE: Bishop pair is now handled by the material imbalance polynomial matrix
    # (compute_imbalance). The standalone bonus has been removed to avoid double-counting.
    # See SF11 material.cpp QuadraticOurs[0][0] = 1438.

    # --- 2. Rooks on Open and Semi-Open Files / 車在開放線和半開放線 ---
    for f in range(8):
        file_mask = FILE_MASKS[f]
        
        white_pawns_on_file = (white_pawns & file_mask) != 0
        black_pawns_on_file = (black_pawns & file_mask) != 0

        # White rooks
        if (white_rooks & file_mask):
            if not white_pawns_on_file:
                if not black_pawns_on_file:
                    # Open file for White / 白方開放線
                    mg_score += ROOK_ON_OPEN_FILE_BONUS[0]
                    eg_score += ROOK_ON_OPEN_FILE_BONUS[1]
                else:
                    # Semi-open file for White / 白方半開放線
                    mg_score += ROOK_ON_SEMI_OPEN_FILE_BONUS[0]
                    eg_score += ROOK_ON_SEMI_OPEN_FILE_BONUS[1]
        
        # Black rooks
        if (black_rooks & file_mask):
            if not black_pawns_on_file:
                if not white_pawns_on_file:
                    # Open file for Black / 黑方開放線
                    mg_score -= ROOK_ON_OPEN_FILE_BONUS[0]
                    eg_score -= ROOK_ON_OPEN_FILE_BONUS[1]
                else:
                    # Semi-open file for Black / 黑方半開放線
                    mg_score -= ROOK_ON_SEMI_OPEN_FILE_BONUS[0]
                    eg_score -= ROOK_ON_SEMI_OPEN_FILE_BONUS[1]
    
    # --- 3. Rooks on 7th Rank / 車在第 7 橫排 ---
    # White Rooks on Rank 7 (Index 6)
    white_rooks_on_7th = white_rooks & RANK_MASKS[6]
    if white_rooks_on_7th:
        count = count_bits(white_rooks_on_7th)
        mg_score += ROOK_ON_SEVENTH_BONUS[0] * count
        eg_score += ROOK_ON_SEVENTH_BONUS[1] * count

    # Black Rooks on Rank 2 (Index 1) - Relative 7th for Black
    black_rooks_on_7th = black_rooks & RANK_MASKS[1]
    if black_rooks_on_7th:
        count = count_bits(black_rooks_on_7th)
        mg_score -= ROOK_ON_SEVENTH_BONUS[0] * count
        eg_score -= ROOK_ON_SEVENTH_BONUS[1] * count

    return mg_score, eg_score
