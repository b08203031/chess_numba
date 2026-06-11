# chess_engine/evaluation.py

import numba
import numpy as np

from chess_engine.classical.constants import *

from chess_engine.classical.bitboard_utils import get_lsb_index, get_msb_index, count_bits, WHITE_KING_ZONES, BLACK_KING_ZONES, FILE_MASKS, SQUARES_BETWEEN
from chess_engine.classical.engine_types import (
    piece_bbs_signature, occupancy_bbs_signature, game_state_signature, piece_counts_signature
)
from chess_engine.classical.move_generator import (
    get_bishop_attacks, get_rook_attacks, get_queen_attacks, KNIGHT_ATTACKS, PAWN_ATTACKS, KING_ATTACKS,
    get_pinned_pieces
)

NOT_A_FILE = ~np.uint64(0x0101010101010101)
NOT_H_FILE = ~np.uint64(0x8080808080808080)

# --- Pre-computed Manhattan Distance Table / 預計算曼哈頓距離表 ---
def _create_manhattan_distance_table():
    """
    Pre-computes a 64x64 lookup table for the manhattan distance between any two squares.
    Distance = abs(rank1 - rank2) + abs(file1 - file2).
    預計算任意兩個方格之間的曼哈頓距離。
    """
    table = np.zeros((64, 64), dtype=np.int32)
    for sq1 in range(64):
        rank1, file1 = sq1 // 8, sq1 % 8
        for sq2 in range(64):
            rank2, file2 = sq2 // 8, sq2 % 8
            table[sq1, sq2] = abs(rank1 - rank2) + abs(file1 - file2)
    return table

MANHATTAN_DISTANCE = _create_manhattan_distance_table()

def _create_chebyshev_distance_table():
    """
    Pre-computes a 64x64 lookup table for the Chebyshev distance (King distance) between any two squares.
    Distance = max(abs(rank1 - rank2), abs(file1 - file2)).
    預計算任意兩個方格之間的切比雪夫距離（國王距離）。
    """
    table = np.zeros((64, 64), dtype=np.int32)
    for sq1 in range(64):
        rank1, file1 = sq1 // 8, sq1 % 8
        for sq2 in range(64):
            rank2, file2 = sq2 // 8, sq2 % 8
            table[sq1, sq2] = max(abs(rank1 - rank2), abs(file1 - file2))
    return table

CHEBYSHEV_DISTANCE = _create_chebyshev_distance_table()


# --- Pre-computed Masks for Pawn Structure Evaluation / 兵型評估的預計算掩碼 ---

# Masks for adjacent files (e.g., for file B, it's file A and C)
# 相鄰直線的掩碼（例如 B 線的相鄰線是 A 和 C）
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

# Precomputed masks for forward ranks (relative to white)
# White pawn at rank r: forward mask includes ranks > r
# Black pawn at rank r: forward mask includes ranks < r (relative to board)
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
    """
    預計算通路兵掩碼。通路兵是指前方沒有敵方兵阻擋（包括相鄰直線）。
    """
    white_masks = np.zeros(64, dtype=np.uint64)
    black_masks = np.zeros(64, dtype=np.uint64)
    for sq in range(64):
        file_idx = sq % 8
        rank_idx = sq // 8
        
        # Mask includes the pawn's own file and adjacent files
        # 掩碼包括兵自己的直線和相鄰直線
        path_mask = FILE_MASKS[file_idx] | ADJACENT_FILES_MASKS[file_idx]
        
        # White passed pawn: no black pawns in front on the path
        # 白方通路兵：前方路徑上沒有黑兵
        white_front_span = np.uint64(0)
        for r in range(rank_idx + 1, 8):
            white_front_span |= (path_mask & (np.uint64(0xFF) << np.uint64(r * 8)))
        white_masks[sq] = white_front_span

        # Black passed pawn: no white pawns in front on the path
        # 黑方通路兵：前方路徑上沒有白兵
        black_front_span = np.uint64(0)
        for r in range(rank_idx - 1, -1, -1):
            black_front_span |= (path_mask & (np.uint64(0xFF) << np.uint64(r * 8)))
        black_masks[sq] = black_front_span
        
    return white_masks, black_masks

WHITE_PASSED_PAWN_MASKS, BLACK_PASSED_PAWN_MASKS = _create_passed_pawn_masks()




@numba.njit(numba.types.UniTuple(numba.int32, 6)(piece_bbs_signature), cache=True, boundscheck=False, fastmath=True)
def evaluate_pawn_structure(piece_bbs):
    """
    評估雙方的兵型結構（通路兵、孤兵、重疊兵、後兵、連結兵）。
    以及兵風暴（Pawn Storm）懲罰。
    
    Args:
        piece_bbs (np.ndarray): 12 個棋子的位元棋盤。
        
    Returns:
        tuple: (mg_score, eg_score, white_pawn_tropism, black_pawn_tropism, white_pawn_storm, black_pawn_storm) 從白方視角。
    """
    mg_score = np.int32(0)
    eg_score = np.int32(0)
    
    white_pawn_tropism = np.int32(0)
    black_pawn_tropism = np.int32(0)
    
    # Pawn Storm Accumulators (Negative values, penalties)
    white_pawn_storm = np.int32(0)
    black_pawn_storm = np.int32(0)
    
    white_pawns = piece_bbs[0]
    black_pawns = piece_bbs[6]
    


    white_king_sq = get_lsb_index(piece_bbs[5])
    black_king_sq = get_lsb_index(piece_bbs[11])
    
    white_king_file = white_king_sq % 8
    black_king_file = black_king_sq % 8

    # --- 1. Iterate White Pawns ---
    temp_wp = white_pawns
    while temp_wp:
        sq = get_lsb_index(temp_wp)
        rank = sq // 8
        file_idx = sq % 8

        adjacent_pawns = ADJACENT_FILES_MASKS[file_idx] & white_pawns

        # Tropism
        dist = MANHATTAN_DISTANCE[sq, black_king_sq]
        white_pawn_tropism += KING_TROPISM_WEIGHTS[0] * (KING_TROPISM_MAX_DISTANCE - dist)
        
        # Pawn Storm (White Pawn attacking Black King)
        if abs(file_idx - black_king_file) <= 1:
            black_pawn_storm -= PAWN_STORM_PENALTY_BY_RANK[7 - rank]

        # Isolated Pawn
        if not adjacent_pawns:
            mg_score += ISOLATED_PAWN_PENALTY[0]
            eg_score += ISOLATED_PAWN_PENALTY[1]

        # Doubled Pawn
        if (WHITE_FORWARD_RANKS[sq] & white_pawns & FILE_MASKS[file_idx]):
            mg_score += DOUBLED_PAWN_PENALTY[0]
            eg_score += DOUBLED_PAWN_PENALTY[1]

        # Determine if passed or candidate for backward pawn exclusion
        is_passed = not (WHITE_PASSED_PAWN_MASKS[sq] & black_pawns)
        is_candidate = False
        if not is_passed and not (WHITE_FORWARD_RANKS[sq] & FILE_MASKS[file_idx] & black_pawns):
            enemy_blockers = (WHITE_PASSED_PAWN_MASKS[sq] & black_pawns) | (RANK_MASKS[rank] & ADJACENT_FILES_MASKS[file_idx] & black_pawns)
            num_blockers = 0
            temp_blockers = enemy_blockers
            while temp_blockers:
                num_blockers += 1
                temp_blockers &= temp_blockers - np.uint64(1)
            if num_blockers == 1:
                is_candidate = True

        # Backward Pawn Logic (Only for non-passed and non-candidate pawns)
        if not is_passed and not is_candidate:
             support_mask = BLACK_FORWARD_RANKS[sq] | RANK_MASKS[rank]
             has_support = (adjacent_pawns & support_mask) != 0
             if not has_support:
                 stop_sq = sq + 8
                 if stop_sq < 64:
                     if (PAWN_ATTACKS[1, stop_sq] & black_pawns):
                         mg_score += BACKWARD_PAWN_PENALTY[0]
                         eg_score += BACKWARD_PAWN_PENALTY[1]

        # --- SF11 Connected Pawn Bonus (all pawns: phalanx or supported) ---
        phalanx = adjacent_pawns & RANK_MASKS[rank]
        support_mask_behind = RANK_MASKS[max(0, rank - 1)] & ADJACENT_FILES_MASKS[file_idx]
        support = white_pawns & support_mask_behind
        if phalanx or support:
            opposed = np.int32(1) if (WHITE_FORWARD_RANKS[sq] & FILE_MASKS[file_idx] & black_pawns) else np.int32(0)
            ph = np.int32(1) if phalanx else np.int32(0)
            sup_cnt = count_bits(support)
            v = CONNECTED_BONUS[rank] * (np.int32(2) + ph - opposed) + CONNECTED_SUPPORT_WEIGHT * sup_cnt
            mg_score += v
            if is_passed:
                eg_score += (v * (np.int32(rank) + np.int32(1))) // np.int32(4)
            else:
                eg_score += v // 2

        temp_wp &= temp_wp - np.uint64(1)

    # --- 2. Iterate Black Pawns ---
    temp_bp = black_pawns
    while temp_bp:
        sq = get_lsb_index(temp_bp)
        rank = sq // 8 # 0-7
        relative_rank = 7 - rank
        file_idx = sq % 8

        adjacent_pawns = ADJACENT_FILES_MASKS[file_idx] & black_pawns

        # Tropism
        dist = MANHATTAN_DISTANCE[sq, white_king_sq]
        black_pawn_tropism += KING_TROPISM_WEIGHTS[0] * (KING_TROPISM_MAX_DISTANCE - dist)
        
        # Pawn Storm (Black Pawn attacking White King)
        if abs(file_idx - white_king_file) <= 1:
            white_pawn_storm -= PAWN_STORM_PENALTY_BY_RANK[rank]

        # Isolated Pawn
        if not adjacent_pawns:
            mg_score -= ISOLATED_PAWN_PENALTY[0]
            eg_score -= ISOLATED_PAWN_PENALTY[1]

        # Doubled Pawn
        if (BLACK_FORWARD_RANKS[sq] & black_pawns & FILE_MASKS[file_idx]):
             mg_score -= DOUBLED_PAWN_PENALTY[0]
             eg_score -= DOUBLED_PAWN_PENALTY[1]

        # Determine if passed or candidate for backward pawn exclusion
        is_passed = not (BLACK_PASSED_PAWN_MASKS[sq] & white_pawns)
        is_candidate = False
        if not is_passed and not (BLACK_FORWARD_RANKS[sq] & FILE_MASKS[file_idx] & white_pawns):
            enemy_blockers = (BLACK_PASSED_PAWN_MASKS[sq] & white_pawns) | (RANK_MASKS[rank] & ADJACENT_FILES_MASKS[file_idx] & white_pawns)
            num_blockers = 0
            temp_blockers = enemy_blockers
            while temp_blockers:
                num_blockers += 1
                temp_blockers &= temp_blockers - np.uint64(1)
            if num_blockers == 1:
                is_candidate = True

        # Backward Pawn Logic (Only for non-passed and non-candidate pawns)
        if not is_passed and not is_candidate:
             support_mask = WHITE_FORWARD_RANKS[sq] | RANK_MASKS[rank]
             has_support = (adjacent_pawns & support_mask) != 0

             if not has_support:
                 stop_sq = sq - 8
                 if stop_sq >= 0:
                     if (PAWN_ATTACKS[0, stop_sq] & white_pawns):
                         mg_score -= BACKWARD_PAWN_PENALTY[0]
                         eg_score -= BACKWARD_PAWN_PENALTY[1]

        # --- SF11 Connected Pawn Bonus (all pawns: phalanx or supported) ---
        phalanx = adjacent_pawns & RANK_MASKS[rank]
        support_mask_ahead = RANK_MASKS[min(7, rank + 1)] & ADJACENT_FILES_MASKS[file_idx]
        support = black_pawns & support_mask_ahead
        if phalanx or support:
            opposed = np.int32(1) if (BLACK_FORWARD_RANKS[sq] & FILE_MASKS[file_idx] & white_pawns) else np.int32(0)
            ph = np.int32(1) if phalanx else np.int32(0)
            sup_cnt = count_bits(support)
            v = CONNECTED_BONUS[relative_rank] * (np.int32(2) + ph - opposed) + CONNECTED_SUPPORT_WEIGHT * sup_cnt
            mg_score -= v
            if is_passed:
                eg_score -= (v * (np.int32(relative_rank) + np.int32(1))) // np.int32(4)
            else:
                eg_score -= v // 2

        temp_bp &= temp_bp - np.uint64(1)

    return mg_score, eg_score, white_pawn_tropism, black_pawn_tropism, white_pawn_storm, black_pawn_storm


@numba.njit(cache=True, boundscheck=False, fastmath=True)
def evaluate_pawn_structure_cached(piece_bbs, pawn_key, search_context):
    """
    Cached version of evaluate_pawn_structure using Pawn Hash Table.
    """
    idx = int(pawn_key & np.uint64(0xFFFF))
    white_king_sq = get_lsb_index(piece_bbs[5])
    black_king_sq = get_lsb_index(piece_bbs[11])
    
    # Verify pawn key and king positions to validate king safety / tropism / storm caches
    if (search_context.pawn_table_keys[idx] == pawn_key and 
        search_context.pawn_table_w_king_sq[idx] == white_king_sq and 
        search_context.pawn_table_b_king_sq[idx] == black_king_sq):
        
        return (
            search_context.pawn_table_mg[idx],
            search_context.pawn_table_eg[idx],
            search_context.pawn_table_w_tropism[idx],
            search_context.pawn_table_b_tropism[idx],
            search_context.pawn_table_w_storm[idx],
            search_context.pawn_table_b_storm[idx]
        )
        
    mg, eg, w_trop, b_trop, w_storm, b_storm = evaluate_pawn_structure(piece_bbs)
    
    # Store computed values in cache (write-always)
    search_context.pawn_table_keys[idx] = pawn_key
    search_context.pawn_table_w_king_sq[idx] = np.int8(white_king_sq)
    search_context.pawn_table_b_king_sq[idx] = np.int8(black_king_sq)
    search_context.pawn_table_mg[idx] = mg
    search_context.pawn_table_eg[idx] = eg
    search_context.pawn_table_w_tropism[idx] = w_trop
    search_context.pawn_table_b_tropism[idx] = b_trop
    search_context.pawn_table_w_storm[idx] = w_storm
    search_context.pawn_table_b_storm[idx] = b_storm
    
    return mg, eg, w_trop, b_trop, w_storm, b_storm


@numba.njit(cache=True, boundscheck=False, fastmath=True)
def evaluate_passed_pawns(piece_bbs, occupancy_bbs, white_attacks, black_attacks):
    """
    Evaluates passed pawns and candidate passed pawns dynamically.
    Incorporates blockade scaling, Tarrasch rook/queen behind passed pawn support,
    and King proximity logic.
    """
    mg_score = np.int32(0)
    eg_score = np.int32(0)

    white_pawns = piece_bbs[0]
    black_pawns = piece_bbs[6]

    # Count non-pawns for material-based scaling
    w_non_pawns = count_bits(piece_bbs[1] | piece_bbs[2] | piece_bbs[3] | piece_bbs[4])
    b_non_pawns = count_bits(piece_bbs[7] | piece_bbs[8] | piece_bbs[9] | piece_bbs[10])
    
    w_minors = count_bits(piece_bbs[1] | piece_bbs[2])
    b_minors = count_bits(piece_bbs[7] | piece_bbs[8])
    
    w_scale = np.int32(4)
    if w_non_pawns < b_non_pawns:
        w_scale = np.int32(2)
        if w_minors == 0 and b_minors >= 1:
            w_scale = np.int32(1)
            
    b_scale = np.int32(4)
    if b_non_pawns < w_non_pawns:
        b_scale = np.int32(2)
        if b_minors == 0 and w_minors >= 1:
            b_scale = np.int32(1)

    white_king_sq = get_lsb_index(piece_bbs[5])
    black_king_sq = get_lsb_index(piece_bbs[11])

    all_pieces = occupancy_bbs[2]

    # --- 1. Iterate White Pawns ---
    temp_wp = white_pawns
    while temp_wp:
        sq = get_lsb_index(temp_wp)
        rank = sq // 8
        file_idx = sq % 8

        is_passed = not (WHITE_PASSED_PAWN_MASKS[sq] & black_pawns)
        is_candidate = False
        if not is_passed and not (WHITE_FORWARD_RANKS[sq] & FILE_MASKS[file_idx] & black_pawns):
            enemy_blockers = (WHITE_PASSED_PAWN_MASKS[sq] & black_pawns) | (RANK_MASKS[rank] & ADJACENT_FILES_MASKS[file_idx] & black_pawns)
            num_blockers = 0
            temp_blockers = enemy_blockers
            while temp_blockers:
                num_blockers += 1
                temp_blockers &= temp_blockers - np.uint64(1)
            if num_blockers == 1:
                is_candidate = True

        if is_passed:
            p_bonus_mg = PASSED_PAWN_BONUS[rank, 0]
            p_bonus_eg = PASSED_PAWN_BONUS[rank, 1]

            # PassedFile adjustment
            edge_dist = np.int32(min(file_idx, 7 - file_idx))
            p_bonus_mg += PASSED_FILE_BONUS[0] * edge_dist
            p_bonus_eg += PASSED_FILE_BONUS[1] * edge_dist

            # Blockade and Path Safety Scaling
            block_sq = sq + 8
            is_blocked = (all_pieces & (np.uint64(1) << np.uint64(block_sq))) != 0
            block_attacked = (black_attacks & (np.uint64(1) << np.uint64(block_sq))) != 0
            block_defended = (white_attacks & (np.uint64(1) << np.uint64(block_sq))) != 0

            path_mask = WHITE_FORWARD_RANKS[sq] & FILE_MASKS[file_idx]
            path_attacked = (black_attacks & path_mask) != 0
            path_occupied = (all_pieces & path_mask) != 0

            if is_blocked:
                p_bonus_mg //= 4
                p_bonus_eg //= 4
            elif block_attacked and not block_defended:
                p_bonus_mg //= 2
                p_bonus_eg //= 2
            elif block_attacked and block_defended:
                p_bonus_mg = (p_bonus_mg * 3) // 4
                p_bonus_eg = (p_bonus_eg * 3) // 4
            elif path_attacked or path_occupied:
                p_bonus_mg = (p_bonus_mg * 3) // 4
                p_bonus_eg = (p_bonus_eg * 3) // 4

            # Tarrasch support from behind (Rook/Queen)
            behind_mask = BLACK_FORWARD_RANKS[sq] & FILE_MASKS[file_idx]
            has_support_behind = ((piece_bbs[3] | piece_bbs[4]) & behind_mask) != 0
            if has_support_behind:
                p_bonus_mg += 15
                p_bonus_eg += 30

            # Scale passed pawn bonus
            p_bonus_mg = (p_bonus_mg * w_scale) // 4
            p_bonus_eg = (p_bonus_eg * w_scale) // 4

            mg_score += p_bonus_mg
            eg_score += p_bonus_eg

            # King Proximity Logic
            if block_sq < 64:
                r = np.int32(rank) - np.int32(1)
                rr = r * (r - np.int32(1))
                if rr > 0:
                    dist_friendly = CHEBYSHEV_DISTANCE[white_king_sq, block_sq]
                    dist_enemy = CHEBYSHEV_DISTANCE[black_king_sq, block_sq]

                    dist_friendly_2nd = np.int32(0)
                    if (block_sq // 8) != 7:
                        dist_friendly_2nd = CHEBYSHEV_DISTANCE[white_king_sq, block_sq + 8]

                    proximity_bonus = (dist_enemy * 5 - dist_friendly * 2 - dist_friendly_2nd) * rr
                    proximity_bonus = max(-150, min(150, proximity_bonus))
                    proximity_bonus = (proximity_bonus * w_scale) // 4
                    eg_score += proximity_bonus

        elif is_candidate:
            c_bonus_mg = CANDIDATE_PASSED_PAWN_BONUS[rank, 0]
            c_bonus_eg = CANDIDATE_PASSED_PAWN_BONUS[rank, 1]

            # Blockade and Path Safety Scaling (applied to candidates similarly)
            block_sq = sq + 8
            is_blocked = (all_pieces & (np.uint64(1) << np.uint64(block_sq))) != 0
            block_attacked = (black_attacks & (np.uint64(1) << np.uint64(block_sq))) != 0
            block_defended = (white_attacks & (np.uint64(1) << np.uint64(block_sq))) != 0

            path_mask = WHITE_FORWARD_RANKS[sq] & FILE_MASKS[file_idx]
            path_attacked = (black_attacks & path_mask) != 0
            path_occupied = (all_pieces & path_mask) != 0

            if is_blocked:
                c_bonus_mg //= 4
                c_bonus_eg //= 4
            elif block_attacked and not block_defended:
                c_bonus_mg //= 2
                c_bonus_eg //= 2
            elif block_attacked and block_defended:
                c_bonus_mg = (c_bonus_mg * 3) // 4
                c_bonus_eg = (c_bonus_eg * 3) // 4
            elif path_attacked or path_occupied:
                c_bonus_mg = (c_bonus_mg * 3) // 4
                c_bonus_eg = (c_bonus_eg * 3) // 4

            c_bonus_mg = (c_bonus_mg * w_scale) // 4
            c_bonus_eg = (c_bonus_eg * w_scale) // 4

            mg_score += c_bonus_mg
            eg_score += c_bonus_eg

        temp_wp &= temp_wp - np.uint64(1)

    # --- 2. Iterate Black Pawns ---
    temp_bp = black_pawns
    while temp_bp:
        sq = get_lsb_index(temp_bp)
        rank = sq // 8
        relative_rank = 7 - rank
        file_idx = sq % 8

        is_passed = not (BLACK_PASSED_PAWN_MASKS[sq] & white_pawns)
        is_candidate = False
        if not is_passed and not (BLACK_FORWARD_RANKS[sq] & FILE_MASKS[file_idx] & white_pawns):
            enemy_blockers = (BLACK_PASSED_PAWN_MASKS[sq] & white_pawns) | (RANK_MASKS[rank] & ADJACENT_FILES_MASKS[file_idx] & white_pawns)
            num_blockers = 0
            temp_blockers = enemy_blockers
            while temp_blockers:
                num_blockers += 1
                temp_blockers &= temp_blockers - np.uint64(1)
            if num_blockers == 1:
                is_candidate = True

        if is_passed:
            p_bonus_mg = PASSED_PAWN_BONUS[relative_rank, 0]
            p_bonus_eg = PASSED_PAWN_BONUS[relative_rank, 1]

            # PassedFile adjustment
            edge_dist = np.int32(min(file_idx, 7 - file_idx))
            p_bonus_mg += PASSED_FILE_BONUS[0] * edge_dist
            p_bonus_eg += PASSED_FILE_BONUS[1] * edge_dist

            # Blockade and Path Safety Scaling
            block_sq = sq - 8
            is_blocked = (all_pieces & (np.uint64(1) << np.uint64(block_sq))) != 0
            block_attacked = (white_attacks & (np.uint64(1) << np.uint64(block_sq))) != 0
            block_defended = (black_attacks & (np.uint64(1) << np.uint64(block_sq))) != 0

            path_mask = BLACK_FORWARD_RANKS[sq] & FILE_MASKS[file_idx]
            path_attacked = (white_attacks & path_mask) != 0
            path_occupied = (all_pieces & path_mask) != 0

            if is_blocked:
                p_bonus_mg //= 4
                p_bonus_eg //= 4
            elif block_attacked and not block_defended:
                p_bonus_mg //= 2
                p_bonus_eg //= 2
            elif block_attacked and block_defended:
                p_bonus_mg = (p_bonus_mg * 3) // 4
                p_bonus_eg = (p_bonus_eg * 3) // 4
            elif path_attacked or path_occupied:
                p_bonus_mg = (p_bonus_mg * 3) // 4
                p_bonus_eg = (p_bonus_eg * 3) // 4

            # Tarrasch support from behind (Rook/Queen)
            behind_mask = WHITE_FORWARD_RANKS[sq] & FILE_MASKS[file_idx]
            has_support_behind = ((piece_bbs[9] | piece_bbs[10]) & behind_mask) != 0
            if has_support_behind:
                p_bonus_mg += 15
                p_bonus_eg += 30

            # Scale passed pawn bonus
            p_bonus_mg = (p_bonus_mg * b_scale) // 4
            p_bonus_eg = (p_bonus_eg * b_scale) // 4

            mg_score -= p_bonus_mg
            eg_score -= p_bonus_eg

            # King Proximity Logic
            if block_sq >= 0:
                r = np.int32(relative_rank) - np.int32(1)
                rr = r * (r - np.int32(1))
                if rr > 0:
                    dist_friendly = CHEBYSHEV_DISTANCE[black_king_sq, block_sq]
                    dist_enemy = CHEBYSHEV_DISTANCE[white_king_sq, block_sq]

                    dist_friendly_2nd = np.int32(0)
                    if (block_sq // 8) != 0:
                        dist_friendly_2nd = CHEBYSHEV_DISTANCE[black_king_sq, block_sq - 8]

                    proximity_bonus = (dist_enemy * 5 - dist_friendly * 2 - dist_friendly_2nd) * rr
                    proximity_bonus = max(-150, min(150, proximity_bonus))
                    proximity_bonus = (proximity_bonus * b_scale) // 4
                    eg_score -= proximity_bonus

        elif is_candidate:
            c_bonus_mg = CANDIDATE_PASSED_PAWN_BONUS[relative_rank, 0]
            c_bonus_eg = CANDIDATE_PASSED_PAWN_BONUS[relative_rank, 1]

            # Blockade and Path Safety Scaling (applied to candidates similarly)
            block_sq = sq - 8
            is_blocked = (all_pieces & (np.uint64(1) << np.uint64(block_sq))) != 0
            block_attacked = (white_attacks & (np.uint64(1) << np.uint64(block_sq))) != 0
            block_defended = (black_attacks & (np.uint64(1) << np.uint64(block_sq))) != 0

            path_mask = BLACK_FORWARD_RANKS[sq] & FILE_MASKS[file_idx]
            path_attacked = (white_attacks & path_mask) != 0
            path_occupied = (all_pieces & path_mask) != 0

            if is_blocked:
                c_bonus_mg //= 4
                c_bonus_eg //= 4
            elif block_attacked and not block_defended:
                c_bonus_mg //= 2
                c_bonus_eg //= 2
            elif block_attacked and block_defended:
                c_bonus_mg = (c_bonus_mg * 3) // 4
                c_bonus_eg = (c_bonus_eg * 3) // 4
            elif path_attacked or path_occupied:
                c_bonus_mg = (c_bonus_mg * 3) // 4
                c_bonus_eg = (c_bonus_eg * 3) // 4

            c_bonus_mg = (c_bonus_mg * b_scale) // 4
            c_bonus_eg = (c_bonus_eg * b_scale) // 4

            mg_score -= c_bonus_mg
            eg_score -= c_bonus_eg

        temp_bp &= temp_bp - np.uint64(1)

    return mg_score, eg_score


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
    # white_bishops = piece_bbs[2] # Not needed for iteration, count from piece_counts
    white_rooks = piece_bbs[3]
    black_pawns = piece_bbs[6]
    # black_bishops = piece_bbs[8] # Not needed for iteration
    black_rooks = piece_bbs[9]

    # --- 1. Bishop Pair / 雙象 ---
    # A bonus is awarded if a side has two or more bishops.
    # 擁有雙象給予獎勵。
    if piece_counts[2] >= 2:
        mg_score += BISHOP_PAIR_BONUS[0]
        eg_score += BISHOP_PAIR_BONUS[1]
    if piece_counts[8] >= 2:
        mg_score -= BISHOP_PAIR_BONUS[0]
        eg_score -= BISHOP_PAIR_BONUS[1]

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
        # Check if Black King is on Rank 8 (Index 7)
        # Note: We can also award bonus if there are pawns on rank 7, but King on 8th is classic.
        # Simple implementation: Just bonus for Rook on 7th.
        # Ideally, we should check if it confines the king or attacks pawns.
        # Simplified: Bonus if on 7th.
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


@numba.njit(numba.int32(numba.int32, numba.uint64, numba.uint64, numba.int32), cache=True, boundscheck=False, fastmath=True)
def _evaluate_pawn_shield_for_color(king_sq, friendly_pawns, enemy_pawns, color):
    """
    Evaluates the pawn shield in front of the king for a single color.
    Only active when the king is near the back rank (SF11-style guard).
    評估單一方國王前方的兵盾。只在王靠近底線時有效。
    """
    score = np.int32(0)
    king_rank = king_sq // 8
    king_file = king_sq % 8

    # Only evaluate pawn shield when king is near back rank
    # White: rank 0-2, Black: rank 5-7
    if color == 0 and king_rank > 2:
        return score
    if color == 1 and king_rank < 5:
        return score

    original_rank = 1 if color == 0 else 6
    one_step_rank = 2 if color == 0 else 5

    for f in range(max(0, king_file - 1), min(7, king_file + 1) + 1):
        file_mask = FILE_MASKS[f]
        pawns_on_file = friendly_pawns & file_mask

        if not pawns_on_file:
            score -= PAWN_SHIELD_MISSING_PENALTY
            # Open/semi-open file penalty (only when no friendly pawn)
            if not (enemy_pawns & file_mask):
                score -= KING_OPEN_FILE_PENALTY
            else:
                score -= KING_SEMI_OPEN_FILE_PENALTY
        else:
            if color == 0:
                pawn_sq = get_lsb_index(pawns_on_file)
            else:
                pawn_sq = get_msb_index(pawns_on_file)
            pawn_rank = pawn_sq // 8

            if pawn_rank == original_rank:
                score += PAWN_SHIELD_INTACT_BONUS
            elif pawn_rank == one_step_rank:
                score += PAWN_SHIELD_ADVANCED_BONUS
            else:
                score -= PAWN_SHIELD_PUSHED_PENALTY

    return score

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def _evaluate_king_attackers(king_sq, color, piece_bbs, occupancy_bbs, enemy_attacks_bb, friendly_attacks_bb, king_zone):
    """
    Calculates king danger using a multi-indicator linear formula inspired by SF11,
    then converts via quadratic transformation.
    使用受 SF11 啟發的多指標線性公式計算王的危險度，再透過二次轉換得到分數。

    Indicators:
      1. zone_attack_units  — piece attacks in king zone (weighted)
      2. safe_check_units   — safe check squares
      3. weak_sq_count      — weak squares in king zone
      4. unsafe_check_count — unsafe check squares
      5. king_attacks_count — attacks on squares adjacent to king
      6. queen absence      — flat deduction when no enemy queen
    """
    # Optimization: If no enemy piece attacks the king zone, we can skip.
    if not (enemy_attacks_bb & king_zone):
        return np.int32(0)

    zone_attack_units = np.int32(0)
    attacker_count = np.int32(0)
    king_attacks_count = np.int32(0)

    enemy_start_idx = 6 if color == 0 else 0
    (en_p, en_n, en_b, en_r, en_q, _) = piece_bbs[enemy_start_idx : enemy_start_idx + 6]

    all_pieces_occupancy = occupancy_bbs[2]

    # Friendly defense mask excluding king (king can't defend against contact attacks)
    friendly_attacks_without_king = friendly_attacks_bb & ~KING_ATTACKS[king_sq]

    # King adjacent squares (the 8 squares around the king)
    king_adjacent = KING_ATTACKS[king_sq]

    # Collect combined attacks per piece type for check detection
    en_n_attacks_all = np.uint64(0)
    en_b_attacks_all = np.uint64(0)
    en_r_attacks_all = np.uint64(0)
    en_q_attacks_all = np.uint64(0)

    # Knights
    temp_bb = en_n
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        attacks = KNIGHT_ATTACKS[sq]
        en_n_attacks_all |= attacks
        attacks_in_zone = attacks & king_zone
        if attacks_in_zone:
            zone_attack_units += KING_SAFETY_ATTACK_UNITS[1]
            attacker_count += 1
        king_attacks_count += count_bits(attacks & king_adjacent)
        temp_bb &= temp_bb - np.uint64(1)

    # Bishops
    temp_bb = en_b
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        attacks = get_bishop_attacks(sq, all_pieces_occupancy)
        en_b_attacks_all |= attacks
        attacks_in_zone = attacks & king_zone
        if attacks_in_zone:
            zone_attack_units += KING_SAFETY_ATTACK_UNITS[2]
            attacker_count += 1
        king_attacks_count += count_bits(attacks & king_adjacent)
        temp_bb &= temp_bb - np.uint64(1)

    # Rooks
    temp_bb = en_r
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        attacks = get_rook_attacks(sq, all_pieces_occupancy)
        en_r_attacks_all |= attacks
        attacks_in_zone = attacks & king_zone
        if attacks_in_zone:
            zone_attack_units += KING_SAFETY_ATTACK_UNITS[3]
            attacker_count += 1
        king_attacks_count += count_bits(attacks & king_adjacent)
        temp_bb &= temp_bb - np.uint64(1)

    # Queens
    temp_bb = en_q
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        attacks = get_queen_attacks(sq, all_pieces_occupancy)
        en_q_attacks_all |= attacks
        attacks_in_zone = attacks & king_zone
        if attacks_in_zone:
            zone_attack_units += KING_SAFETY_ATTACK_UNITS[4]
            attacker_count += 1
        king_attacks_count += count_bits(attacks & king_adjacent)
        temp_bb &= temp_bb - np.uint64(1)

    # --- Safe Check Detection (Inspired by Stockfish 11) ---
    friendly_start_idx = 0 if color == 0 else 6
    friendly_queen = piece_bbs[friendly_start_idx + 4]
    occ_xray = all_pieces_occupancy ^ friendly_queen

    rook_check_rays = get_rook_attacks(king_sq, occ_xray)
    bishop_check_rays = get_bishop_attacks(king_sq, occ_xray)

    attacker_occ = occupancy_bbs[1] if color == 0 else occupancy_bbs[0]
    safe = ~attacker_occ & ~friendly_attacks_bb

    safe_check_units = np.int32(0)
    unsafe_checks = np.uint64(0)

    # Rook safe checks
    rook_safe_checks = rook_check_rays & safe & en_r_attacks_all
    if rook_safe_checks:
        safe_check_units += SAFE_CHECK_ROOK
    else:
        unsafe_checks |= rook_check_rays & en_r_attacks_all

    # Queen safe checks
    queen_safe_checks = (rook_check_rays | bishop_check_rays) & safe & en_q_attacks_all & ~rook_safe_checks
    if queen_safe_checks:
        safe_check_units += SAFE_CHECK_QUEEN

    # Bishop safe checks
    bishop_safe_checks = bishop_check_rays & safe & en_b_attacks_all & ~queen_safe_checks
    if bishop_safe_checks:
        safe_check_units += SAFE_CHECK_BISHOP
    else:
        unsafe_checks |= bishop_check_rays & en_b_attacks_all

    # Knight safe checks
    knight_check_squares = KNIGHT_ATTACKS[king_sq] & en_n_attacks_all
    if knight_check_squares & safe:
        safe_check_units += SAFE_CHECK_KNIGHT
    else:
        unsafe_checks |= knight_check_squares

    # Weak squares: in king zone, attacked by enemy, not defended by us (except king)
    weak_in_zone = king_zone & enemy_attacks_bb & ~friendly_attacks_without_king
    weak_sq_count = count_bits(weak_in_zone)

    # Unsafe check count
    unsafe_check_count = count_bits(unsafe_checks)

    # --- Attacker count gating (softened from hard cutoff) ---
    if attacker_count == 0:
        return np.int32(0)

    # --- Combine all indicators into kingDanger ---
    kingDanger = (
          zone_attack_units
        + safe_check_units
        + KING_DANGER_WEAK_SQ * weak_sq_count
        + KING_DANGER_UNSAFE_CHECK * unsafe_check_count
        + KING_DANGER_ATTACK_ON_KING_SQ * king_attacks_count
    )

    # Queen absence: flat deduction
    if not en_q:
        kingDanger = max(np.int32(0), kingDanger - KING_DANGER_NO_QUEEN)

    # Single attacker: halve the final danger score (softened threat)
    if attacker_count == 1:
        kingDanger = kingDanger // 2

    # Quadratic transformation (equivalent to old KING_SAFETY_TABLE[i] = i²/2)
    if kingDanger > 0:
        penalty = kingDanger * kingDanger // KING_DANGER_DIVISOR
        # Cap at 1000 to avoid extreme values
        penalty = min(penalty, np.int32(1000))
        return -penalty
    else:
        return np.int32(0)

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def evaluate_king_safety(piece_bbs, occupancy_bbs, white_attacks, black_attacks, white_tropism, black_tropism, white_pawn_storm_score, black_pawn_storm_score, piece_counts, white_attacks2, black_attacks2, pinned_white, pinned_black):
    """
    King Safety evaluation. Combines pawn shield, king attackers, tropism, and pawn storm.
    Material-based scaling is handled by tapered eval phase interpolation.
    國王安全評估。結合兵盾、攻擊者、向心性和兵風暴。材質縮放由 根據對手材質（馬、象、車、后）進行縮放處理。
    """
    (wp_bb, wn_bb, wb_bb, wr_bb, wq_bb, wk_bb,
    bp_bb, bn_bb, bb_bb, br_bb, bq_bb, bk_bb) = piece_bbs

    white_king_sq = get_lsb_index(wk_bb)
    black_king_sq = get_lsb_index(bk_bb)

    # --- Calculate raw scores for each component ---
    white_shield = _evaluate_pawn_shield_for_color(white_king_sq, wp_bb, bp_bb, 0)
    white_attackers = _evaluate_king_attackers(white_king_sq, 0, piece_bbs, occupancy_bbs, black_attacks, white_attacks, WHITE_KING_ZONES[white_king_sq])

    black_shield = _evaluate_pawn_shield_for_color(black_king_sq, bp_bb, wp_bb, 1)
    black_attackers = _evaluate_king_attackers(black_king_sq, 1, piece_bbs, occupancy_bbs, white_attacks, black_attacks, BLACK_KING_ZONES[black_king_sq])

    # --- Phase 4: Pinned pieces penalty (SF11-inspired) ---
    white_pinned_penalty = np.int32(count_bits(pinned_white)) * KING_DANGER_PINNED
    black_pinned_penalty = np.int32(count_bits(pinned_black)) * KING_DANGER_PINNED

    # --- Phase 5: Scaling based on enemy material (Crucial for Endgame/Strength) ---
    # Sum raw scores first (pinned penalty hurts the side whose pieces are pinned)
    white_raw_safety = white_shield + white_attackers + white_tropism + white_pawn_storm_score - white_pinned_penalty
    black_raw_safety = black_shield + black_attackers + black_tropism + black_pawn_storm_score - black_pinned_penalty

    # Penalty for white king is scaled by black's potential attacking material.
    black_material_for_scaling = (piece_counts[7] * SCALING_WEIGHTS[1] +
                                 piece_counts[8] * SCALING_WEIGHTS[2] +
                                 piece_counts[9] * SCALING_WEIGHTS[3] +
                                 piece_counts[10] * SCALING_WEIGHTS[4])
    white_scaling_factor = black_material_for_scaling / MAX_SCALING_MATERIAL
    white_safety_scaled = np.int32(white_raw_safety * white_scaling_factor)

    # Penalty for black king is scaled by white's potential attacking material.
    white_material_for_scaling = (piece_counts[1] * SCALING_WEIGHTS[1] +
                                 piece_counts[2] * SCALING_WEIGHTS[2] +
                                 piece_counts[3] * SCALING_WEIGHTS[3] +
                                 piece_counts[4] * SCALING_WEIGHTS[4])
    black_scaling_factor = white_material_for_scaling / MAX_SCALING_MATERIAL
    black_safety_scaled = np.int32(black_raw_safety * black_scaling_factor)

    mg_safety_score = white_safety_scaled - black_safety_scaled

    # Endgame: king safety is less important. Tapered eval handles the main reduction;
    # we apply a further scale for the eg component.
    eg_safety_score = mg_safety_score * EG_SAFETY_SCALE

    return mg_safety_score, eg_safety_score


@numba.njit(cache=True, boundscheck=False, fastmath=True)
def evaluate_attacks_mobility_threats(piece_bbs, occupancy_bbs):
    (wp_bb, wn_bb, wb_bb, wr_bb, wq_bb, wk_bb,
     bp_bb, bn_bb, bb_bb, br_bb, bq_bb, bk_bb) = piece_bbs

    white_occupancy = occupancy_bbs[0]
    black_occupancy = occupancy_bbs[1]
    all_occupancy = occupancy_bbs[2]

    # Extract King Squares for Tropism
    white_king_sq = get_lsb_index(wk_bb)
    black_king_sq = get_lsb_index(bk_bb)

    # --- Pawn Attacks ---
    white_pawn_attacks = ((wp_bb & NOT_A_FILE) << np.uint64(7)) | ((wp_bb & NOT_H_FILE) << np.uint64(9))
    black_pawn_attacks = ((bp_bb & NOT_H_FILE) >> np.uint64(7)) | ((bp_bb & NOT_A_FILE) >> np.uint64(9))

    white_attacks = white_pawn_attacks | KING_ATTACKS[white_king_sq]
    black_attacks = black_pawn_attacks | KING_ATTACKS[black_king_sq]

    # --- Mobility Area (SF11-inspired) ---
    # Exclude: enemy pawn attacks, blocked low pawns, friendly queen, friendly king
    LOW_RANKS_WHITE = np.uint64(0x0000000000FFFF00)  # rank2 | rank3
    LOW_RANKS_BLACK = np.uint64(0x00FFFF0000000000)  # rank6 | rank7
    # Simpler: find white pawns on low ranks whose advance square (sq+8) is occupied
    white_blocked_low = np.uint64(0)
    temp = wp_bb & LOW_RANKS_WHITE
    while temp:
        sq = get_lsb_index(temp)
        if all_occupancy & (np.uint64(1) << np.uint64(sq + 8)):
            white_blocked_low |= np.uint64(1) << np.uint64(sq)
        temp &= temp - np.uint64(1)

    black_blocked_low = np.uint64(0)
    temp = bp_bb & LOW_RANKS_BLACK
    while temp:
        sq = get_lsb_index(temp)
        if sq >= 8 and (all_occupancy & (np.uint64(1) << np.uint64(sq - 8))):
            black_blocked_low |= np.uint64(1) << np.uint64(sq)
        temp &= temp - np.uint64(1)

    # --- Pin Calculation (must precede mobility area) ---
    pinned_white = get_pinned_pieces(piece_bbs, occupancy_bbs, np.uint64(0))  # WHITE
    pinned_black = get_pinned_pieces(piece_bbs, occupancy_bbs, np.uint64(1))  # BLACK

    white_mobility_area = ~(black_pawn_attacks | white_blocked_low | wq_bb | wk_bb | pinned_white)
    black_mobility_area = ~(white_pawn_attacks | black_blocked_low | bq_bb | bk_bb | pinned_black)

    # --- Accumulators ---
    mg_mobility = np.int32(0)
    eg_mobility = np.int32(0)
    
    white_knight_attacks = np.uint64(0)
    white_bishop_attacks = np.uint64(0)
    white_rook_attacks = np.uint64(0)
    white_queen_attacks = np.uint64(0)
    white_minor_attacks = np.uint64(0)   # N+B combined (kept for king safety)
    white_attacks2 = np.uint64(0)        # Squares attacked by 2+ white pieces
    
    black_knight_attacks = np.uint64(0)
    black_bishop_attacks = np.uint64(0)
    black_rook_attacks = np.uint64(0)
    black_queen_attacks = np.uint64(0)
    black_minor_attacks = np.uint64(0)   # N+B combined (kept for king safety)
    black_attacks2 = np.uint64(0)        # Squares attacked by 2+ black pieces

    white_piece_tropism = np.int32(0)
    black_piece_tropism = np.int32(0)

    # --- Outpost Accumulators ---
    mg_outpost = np.int32(0)
    eg_outpost = np.int32(0)

    # --- White Pieces ---
    temp_bb = wn_bb
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        att = KNIGHT_ATTACKS[sq]
        white_attacks2 |= (white_attacks & att)
        white_attacks |= att
        white_knight_attacks |= att
        white_minor_attacks |= att
        moves = count_bits(att & ~white_occupancy & white_mobility_area)
        clamped = min(moves, 8)
        mg_mobility += KNIGHT_MOBILITY_BONUS[clamped, 0]
        eg_mobility += KNIGHT_MOBILITY_BONUS[clamped, 1]
        
        # KingProtector: penalty per Chebyshev distance from own king
        mg_mobility -= KING_PROTECTOR[0] * CHEBYSHEV_DISTANCE[sq, white_king_sq]
        eg_mobility -= KING_PROTECTOR[1] * CHEBYSHEV_DISTANCE[sq, white_king_sq]


        
        # Tropism: Distance to Black King
        distance = MANHATTAN_DISTANCE[sq, black_king_sq]
        white_piece_tropism += KING_TROPISM_WEIGHTS[1] * (KING_TROPISM_MAX_DISTANCE - distance)
        
        # Outpost Logic (White Knight)
        if not (black_pawn_attacks & BB_SQUARES[sq]):
            if (PAWN_ATTACKS[WHITE, sq] & wp_bb):
                rank = sq // 8
                mg_outpost += OUTPOST_BONUS_KNIGHT[rank, 0]
                eg_outpost += OUTPOST_BONUS_KNIGHT[rank, 1]
                file_idx = sq % 8
                if not (ADJACENT_FILES_MASKS[file_idx] & WHITE_FORWARD_RANKS[sq] & bp_bb):
                     mg_outpost += OUTPOST_HOLE_BONUS[0]
                     eg_outpost += OUTPOST_HOLE_BONUS[1]

        temp_bb &= temp_bb - np.uint64(1)

    temp_bb = wb_bb
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        # X-Ray: bishop attacks through all queens
        att = get_bishop_attacks(sq, all_occupancy ^ wq_bb ^ bq_bb)
        white_attacks2 |= (white_attacks & att)
        white_attacks |= att
        white_bishop_attacks |= att
        white_minor_attacks |= att
        moves = count_bits(att & ~white_occupancy & white_mobility_area)
        clamped = min(moves, 13)
        mg_mobility += BISHOP_MOBILITY_BONUS[clamped, 0]
        eg_mobility += BISHOP_MOBILITY_BONUS[clamped, 1]
        
        # KingProtector: penalty per Chebyshev distance from own king
        mg_mobility -= KING_PROTECTOR[0] * CHEBYSHEV_DISTANCE[sq, white_king_sq]
        eg_mobility -= KING_PROTECTOR[1] * CHEBYSHEV_DISTANCE[sq, white_king_sq]



        # BishopPawns: penalty per own pawn on same color square as bishop
        # Square color: (rank + file) % 2, where rank = sq//8, file = sq%8
        bishop_color = (sq // 8 + sq % 8) % 2
        bishop_color_mask = np.uint64(0xAA55AA55AA55AA55) if bishop_color == 1 else np.uint64(0x55AA55AA55AA55AA)
        same_color_pawns = count_bits(wp_bb & bishop_color_mask)
        mg_mobility -= BISHOP_PAWNS_PENALTY[0] * same_color_pawns
        eg_mobility -= BISHOP_PAWNS_PENALTY[1] * same_color_pawns
        
        # Tropism
        distance = MANHATTAN_DISTANCE[sq, black_king_sq]
        white_piece_tropism += KING_TROPISM_WEIGHTS[2] * (KING_TROPISM_MAX_DISTANCE - distance)

        # Outpost Logic (White Bishop)
        if not (black_pawn_attacks & BB_SQUARES[sq]):
            if (PAWN_ATTACKS[WHITE, sq] & wp_bb):
                rank = sq // 8
                mg_outpost += OUTPOST_BONUS_BISHOP[rank, 0]
                eg_outpost += OUTPOST_BONUS_BISHOP[rank, 1]
                file_idx = sq % 8
                if not (ADJACENT_FILES_MASKS[file_idx] & WHITE_FORWARD_RANKS[sq] & bp_bb):
                     mg_outpost += OUTPOST_HOLE_BONUS[0]
                     eg_outpost += OUTPOST_HOLE_BONUS[1]

        temp_bb &= temp_bb - np.uint64(1)

    temp_bb = wr_bb
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        # X-Ray: rook attacks through all queens + own rooks
        att = get_rook_attacks(sq, all_occupancy ^ wq_bb ^ bq_bb ^ wr_bb)
        white_attacks2 |= (white_attacks & att)
        white_attacks |= att
        white_rook_attacks |= att
        moves = count_bits(att & ~white_occupancy & white_mobility_area)
        clamped = min(moves, 14)
        mg_mobility += ROOK_MOBILITY_BONUS[clamped, 0]
        eg_mobility += ROOK_MOBILITY_BONUS[clamped, 1]
        
        # TrappedRook: SF11 bilateral detection
        # Penalty when rook is between king and the nearest edge on either side
        if moves <= 3:
            king_file = white_king_sq % 8
            rook_file = sq % 8
            if (king_file < 4) == (rook_file < king_file):
                mg_mobility -= TRAPPED_ROOK[0]
                eg_mobility -= TRAPPED_ROOK[1]
        
        # Tropism
        distance = MANHATTAN_DISTANCE[sq, black_king_sq]
        white_piece_tropism += KING_TROPISM_WEIGHTS[3] * (KING_TROPISM_MAX_DISTANCE - distance)

        temp_bb &= temp_bb - np.uint64(1)

    temp_bb = wq_bb
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        att = get_queen_attacks(sq, all_occupancy)
        white_attacks2 |= (white_attacks & att)
        white_attacks |= att
        white_queen_attacks |= att
        moves = count_bits(att & ~white_occupancy & white_mobility_area)
        clamped = min(moves, 27)
        mg_mobility += QUEEN_MOBILITY_BONUS[clamped, 0]
        eg_mobility += QUEEN_MOBILITY_BONUS[clamped, 1]
        
        # Tropism
        distance = MANHATTAN_DISTANCE[sq, black_king_sq]
        white_piece_tropism += KING_TROPISM_WEIGHTS[4] * (KING_TROPISM_MAX_DISTANCE - distance)

        temp_bb &= temp_bb - np.uint64(1)

    # --- Black Pieces ---
    temp_bb = bn_bb
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        att = KNIGHT_ATTACKS[sq]
        black_attacks2 |= (black_attacks & att)
        black_attacks |= att
        black_knight_attacks |= att
        black_minor_attacks |= att
        moves = count_bits(att & ~black_occupancy & black_mobility_area)
        clamped = min(moves, 8)
        mg_mobility -= KNIGHT_MOBILITY_BONUS[clamped, 0]
        eg_mobility -= KNIGHT_MOBILITY_BONUS[clamped, 1]
        
        # KingProtector: penalty per Chebyshev distance from own king
        mg_mobility += KING_PROTECTOR[0] * CHEBYSHEV_DISTANCE[sq, black_king_sq]
        eg_mobility += KING_PROTECTOR[1] * CHEBYSHEV_DISTANCE[sq, black_king_sq]


        
        # Tropism: Distance to White King
        distance = MANHATTAN_DISTANCE[sq, white_king_sq]
        black_piece_tropism += KING_TROPISM_WEIGHTS[1] * (KING_TROPISM_MAX_DISTANCE - distance)

        # Outpost Logic (Black Knight)
        if not (white_pawn_attacks & BB_SQUARES[sq]):
            if (PAWN_ATTACKS[BLACK, sq] & bp_bb):
                rank = sq // 8
                rel_rank = 7 - rank
                mg_outpost -= OUTPOST_BONUS_KNIGHT[rel_rank, 0]
                eg_outpost -= OUTPOST_BONUS_KNIGHT[rel_rank, 1]
                file_idx = sq % 8
                if not (ADJACENT_FILES_MASKS[file_idx] & BLACK_FORWARD_RANKS[sq] & wp_bb):
                     mg_outpost -= OUTPOST_HOLE_BONUS[0]
                     eg_outpost -= OUTPOST_HOLE_BONUS[1]

        temp_bb &= temp_bb - np.uint64(1)

    temp_bb = bb_bb
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        # X-Ray: bishop attacks through all queens
        att = get_bishop_attacks(sq, all_occupancy ^ wq_bb ^ bq_bb)
        black_attacks2 |= (black_attacks & att)
        black_attacks |= att
        black_bishop_attacks |= att
        black_minor_attacks |= att
        moves = count_bits(att & ~black_occupancy & black_mobility_area)
        clamped = min(moves, 13)
        mg_mobility -= BISHOP_MOBILITY_BONUS[clamped, 0]
        eg_mobility -= BISHOP_MOBILITY_BONUS[clamped, 1]
        
        # KingProtector: penalty per Chebyshev distance from own king
        mg_mobility += KING_PROTECTOR[0] * CHEBYSHEV_DISTANCE[sq, black_king_sq]
        eg_mobility += KING_PROTECTOR[1] * CHEBYSHEV_DISTANCE[sq, black_king_sq]



        # BishopPawns: penalty per own pawn on same color square as bishop
        # Square color: (rank + file) % 2, where rank = sq//8, file = sq%8
        bishop_color = (sq // 8 + sq % 8) % 2
        bishop_color_mask = np.uint64(0xAA55AA55AA55AA55) if bishop_color == 1 else np.uint64(0x55AA55AA55AA55AA)
        same_color_pawns = count_bits(bp_bb & bishop_color_mask)
        mg_mobility += BISHOP_PAWNS_PENALTY[0] * same_color_pawns
        eg_mobility += BISHOP_PAWNS_PENALTY[1] * same_color_pawns
        
        # Tropism
        distance = MANHATTAN_DISTANCE[sq, white_king_sq]
        black_piece_tropism += KING_TROPISM_WEIGHTS[2] * (KING_TROPISM_MAX_DISTANCE - distance)

        # Outpost Logic (Black Bishop)
        if not (white_pawn_attacks & BB_SQUARES[sq]):
            if (PAWN_ATTACKS[BLACK, sq] & bp_bb):
                rank = sq // 8
                rel_rank = 7 - rank
                mg_outpost -= OUTPOST_BONUS_BISHOP[rel_rank, 0]
                eg_outpost -= OUTPOST_BONUS_BISHOP[rel_rank, 1]
                file_idx = sq % 8
                if not (ADJACENT_FILES_MASKS[file_idx] & BLACK_FORWARD_RANKS[sq] & wp_bb):
                     mg_outpost -= OUTPOST_HOLE_BONUS[0]
                     eg_outpost -= OUTPOST_HOLE_BONUS[1]

        temp_bb &= temp_bb - np.uint64(1)

    temp_bb = br_bb
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        # X-Ray: rook attacks through all queens + own rooks
        att = get_rook_attacks(sq, all_occupancy ^ wq_bb ^ bq_bb ^ br_bb)
        black_attacks2 |= (black_attacks & att)
        black_attacks |= att
        black_rook_attacks |= att
        moves = count_bits(att & ~black_occupancy & black_mobility_area)
        clamped = min(moves, 14)
        mg_mobility -= ROOK_MOBILITY_BONUS[clamped, 0]
        eg_mobility -= ROOK_MOBILITY_BONUS[clamped, 1]
        
        # TrappedRook: SF11 bilateral detection
        if moves <= 3:
            king_file = black_king_sq % 8
            rook_file = sq % 8
            if (king_file < 4) == (rook_file < king_file):
                mg_mobility += TRAPPED_ROOK[0]
                eg_mobility += TRAPPED_ROOK[1]
        
        # Tropism
        distance = MANHATTAN_DISTANCE[sq, white_king_sq]
        black_piece_tropism += KING_TROPISM_WEIGHTS[3] * (KING_TROPISM_MAX_DISTANCE - distance)

        temp_bb &= temp_bb - np.uint64(1)

    temp_bb = bq_bb
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        att = get_queen_attacks(sq, all_occupancy)
        black_attacks2 |= (black_attacks & att)
        black_attacks |= att
        black_queen_attacks |= att
        moves = count_bits(att & ~black_occupancy & black_mobility_area)
        clamped = min(moves, 27)
        mg_mobility -= QUEEN_MOBILITY_BONUS[clamped, 0]
        eg_mobility -= QUEEN_MOBILITY_BONUS[clamped, 1]
        
        # Tropism
        distance = MANHATTAN_DISTANCE[sq, white_king_sq]
        black_piece_tropism += KING_TROPISM_WEIGHTS[4] * (KING_TROPISM_MAX_DISTANCE - distance)

        temp_bb &= temp_bb - np.uint64(1)

    # --- Threats Evaluation (SF11-inspired, per-type) ---
    mg_threats = np.int32(0)
    eg_threats = np.int32(0)

    # Pre-compute piece sets
    white_non_pawns = wn_bb | wb_bb | wr_bb | wq_bb
    black_non_pawns = bn_bb | bb_bb | br_bb | bq_bb
    all_white_pieces = wp_bb | white_non_pawns
    all_black_pieces = bp_bb | black_non_pawns

    # stronglyProtected: defended by pawn OR doubly attacked (and not doubly attacked by us)
    white_strongly_protected = white_pawn_attacks | (white_attacks2 & ~black_attacks2)
    black_strongly_protected = black_pawn_attacks | (black_attacks2 & ~white_attacks2)

    # weakly attacked enemy pieces (attacked by us, not strongly defended)
    white_weak = all_white_pieces & black_attacks & ~white_strongly_protected
    black_weak = all_black_pieces & white_attacks & ~black_strongly_protected

    # --- 1. ThreatBySafePawn: safe pawn attacks enemy non-pawn ---
    # White safe pawns: pawn attacks squares not strongly defended by enemy
    white_safe_pawn_sq = white_pawn_attacks & black_non_pawns & ~black_strongly_protected
    if white_safe_pawn_sq:
        count = count_bits(white_safe_pawn_sq)
        mg_threats += THREAT_SAFE_PAWN[0] * count
        eg_threats += THREAT_SAFE_PAWN[1] * count

    black_safe_pawn_sq = black_pawn_attacks & white_non_pawns & ~white_strongly_protected
    if black_safe_pawn_sq:
        count = count_bits(black_safe_pawn_sq)
        mg_threats -= THREAT_SAFE_PAWN[0] * count
        eg_threats -= THREAT_SAFE_PAWN[1] * count

    # --- 2. ThreatByMinor: minor (N/B) attacks weak enemy pieces, per type ---
    w_minor_threatens = (white_knight_attacks | white_bishop_attacks) & all_black_pieces & ~black_strongly_protected
    for pt in range(6):
        intersection = w_minor_threatens & piece_bbs[6 + pt]
        if intersection:
            count = count_bits(intersection)
            mg_threats += THREAT_BY_MINOR[pt, 0] * count
            eg_threats += THREAT_BY_MINOR[pt, 1] * count

    b_minor_threatens = (black_knight_attacks | black_bishop_attacks) & all_white_pieces & ~white_strongly_protected
    for pt in range(6):
        intersection = b_minor_threatens & piece_bbs[pt]
        if intersection:
            count = count_bits(intersection)
            mg_threats -= THREAT_BY_MINOR[pt, 0] * count
            eg_threats -= THREAT_BY_MINOR[pt, 1] * count

    # --- 3. ThreatByRook: rook attacks weak enemy pieces, per type ---
    w_rook_threatens = white_rook_attacks & black_weak
    for pt in range(6):
        intersection = w_rook_threatens & piece_bbs[6 + pt]
        if intersection:
            count = count_bits(intersection)
            mg_threats += THREAT_BY_ROOK[pt, 0] * count
            eg_threats += THREAT_BY_ROOK[pt, 1] * count

    b_rook_threatens = black_rook_attacks & white_weak
    for pt in range(6):
        intersection = b_rook_threatens & piece_bbs[pt]
        if intersection:
            count = count_bits(intersection)
            mg_threats -= THREAT_BY_ROOK[pt, 0] * count
            eg_threats -= THREAT_BY_ROOK[pt, 1] * count

    # --- 4. ThreatByKing: king attacks weak enemy pieces ---
    w_king_threatens = KING_ATTACKS[white_king_sq] & black_weak
    if w_king_threatens:
        count = count_bits(w_king_threatens)
        mg_threats += THREAT_BY_KING[0] * count
        eg_threats += THREAT_BY_KING[1] * count

    b_king_threatens = KING_ATTACKS[black_king_sq] & white_weak
    if b_king_threatens:
        count = count_bits(b_king_threatens)
        mg_threats -= THREAT_BY_KING[0] * count
        eg_threats -= THREAT_BY_KING[1] * count

    # --- 5. Hanging: weak pieces with no defenders at all OR doubly attacked ---
    black_hanging = black_weak & (~black_attacks | (black_non_pawns & white_attacks2))
    if black_hanging:
        count = count_bits(black_hanging)
        mg_threats += THREAT_HANGING[0] * count
        eg_threats += THREAT_HANGING[1] * count

    white_hanging = white_weak & (~white_attacks | (white_non_pawns & black_attacks2))
    if white_hanging:
        count = count_bits(white_hanging)
        mg_threats -= THREAT_HANGING[0] * count
        eg_threats -= THREAT_HANGING[1] * count

    # Restricted piece threat (RestrictedPiece * popcount(b))
    w_restricted = black_attacks & ~black_strongly_protected & white_attacks
    if w_restricted:
        count = count_bits(w_restricted)
        mg_threats += THREAT_RESTRICTED_PIECE[0] * count
        eg_threats += THREAT_RESTRICTED_PIECE[1] * count

    b_restricted = white_attacks & ~white_strongly_protected & black_attacks
    if b_restricted:
        count = count_bits(b_restricted)
        mg_threats -= THREAT_RESTRICTED_PIECE[0] * count
        eg_threats -= THREAT_RESTRICTED_PIECE[1] * count

    # --- 7. ThreatByPawnPush ---
    # White pawns push threat: non-rank8 squares that would attack enemy non-pawns
    w_push1 = (wp_bb << np.uint64(8)) & ~all_occupancy & ~np.uint64(0xFF00000000000000)
    w_push2 = ((w_push1 & np.uint64(0xFF0000)) << np.uint64(8)) & ~all_occupancy  # double push from rank 2
    w_push_attacks = (((w_push1 | w_push2) & NOT_A_FILE) << np.uint64(7)) | (((w_push1 | w_push2) & NOT_H_FILE) << np.uint64(9))
    w_push_threat = w_push_attacks & black_non_pawns & ~black_attacks2
    if w_push_threat:
        count = count_bits(w_push_threat)
        mg_threats += THREAT_PAWN_PUSH[0] * count
        eg_threats += THREAT_PAWN_PUSH[1] * count

    b_push1 = (bp_bb >> np.uint64(8)) & ~all_occupancy & ~np.uint64(0xFF)
    b_push2 = ((b_push1 & np.uint64(0xFF0000000000)) >> np.uint64(8)) & ~all_occupancy  # double push from rank 7
    b_push_attacks = (((b_push1 | b_push2) & NOT_H_FILE) >> np.uint64(7)) | (((b_push1 | b_push2) & NOT_A_FILE) >> np.uint64(9))
    b_push_threat = b_push_attacks & white_non_pawns & ~white_attacks2
    if b_push_threat:
        count = count_bits(b_push_threat)
        mg_threats -= THREAT_PAWN_PUSH[0] * count
        eg_threats -= THREAT_PAWN_PUSH[1] * count

    return (white_attacks, black_attacks, white_pawn_attacks, black_pawn_attacks,
            white_attacks2, black_attacks2, pinned_white, pinned_black,
            mg_mobility, eg_mobility, mg_threats, eg_threats, white_piece_tropism, black_piece_tropism, mg_outpost, eg_outpost)

@numba.njit(numba.int32(piece_bbs_signature, numba.uint64), cache=True, boundscheck=False, fastmath=True)
def _evaluate_king_pawn_endgame(piece_bbs, side_to_move):
    """
    專門為王兵殘局設計的評估函數。包含不可阻擋通路兵的檢測（方形法則）。
    """
    score = np.int32(0)

    white_pawns = piece_bbs[0]
    white_king_sq = get_lsb_index(piece_bbs[5])
    black_pawns = piece_bbs[6]
    black_king_sq = get_lsb_index(piece_bbs[11])

    # 1. 基礎兵價和位置
    # White pawns
    temp_wp = white_pawns
    while temp_wp:
        sq = get_lsb_index(temp_wp)
        score += EG_MATERIAL_VALUES[0] + PST_EG[0, sq]
        
        # Unstoppable Pawn Logic (White)
        # Check if passed
        if not (WHITE_PASSED_PAWN_MASKS[sq] & black_pawns):
            rank = sq // 8
            steps_to_promote = 7 - rank
            if rank == 1: steps_to_promote = 5 # Double push correction (approx)
            
            # Enemy King Distance
            promotion_sq = (sq % 8) + 56
            king_dist = CHEBYSHEV_DISTANCE[black_king_sq, promotion_sq]
            
            adjusted_pawn_steps = steps_to_promote
            if side_to_move == 1: # Black to move (enemy's turn)
                adjusted_pawn_steps += 1
            
            # If King is too far -> Unstoppable
            if king_dist > adjusted_pawn_steps:
                 score += 800 # Queen value approx
            
        temp_wp &= temp_wp - np.uint64(1)

    # Black pawns
    temp_bp = black_pawns
    while temp_bp:
        sq = get_lsb_index(temp_bp)
        score -= (EG_MATERIAL_VALUES[0] + PST_EG[0, sq ^ 56])
        
        # Unstoppable Pawn Logic (Black)
        if not (BLACK_PASSED_PAWN_MASKS[sq] & white_pawns):
            rank = sq // 8
            steps_to_promote = rank
            if rank == 6: steps_to_promote = 5
            
            promotion_sq = (sq % 8)
            king_dist = CHEBYSHEV_DISTANCE[white_king_sq, promotion_sq]
            
            adjusted_pawn_steps = steps_to_promote
            if side_to_move == 0: # White to move (enemy's turn)
                adjusted_pawn_steps += 1
                
            if king_dist > adjusted_pawn_steps:
                score -= 800

        temp_bp &= temp_bp - np.uint64(1)

    # King position
    score += PST_EG[5, white_king_sq]
    score -= PST_EG[5, black_king_sq ^ 56]

    # 2. 通路兵與後兵獎勵
    _, eg_pawn_score, _, _, _, _ = evaluate_pawn_structure(piece_bbs)
    score += eg_pawn_score

    # 計算王與兵的動態攻擊 (王兵殘局)
    white_pawn_attacks = ((white_pawns & NOT_A_FILE) << np.uint64(7)) | ((white_pawns & NOT_H_FILE) << np.uint64(9))
    black_pawn_attacks = ((black_pawns & NOT_H_FILE) >> np.uint64(7)) | ((black_pawns & NOT_A_FILE) >> np.uint64(9))
    white_attacks = white_pawn_attacks | KING_ATTACKS[white_king_sq]
    black_attacks = black_pawn_attacks | KING_ATTACKS[black_king_sq]

    # 建構臨時佔用陣列 (僅包含王、兵)
    occ_bbs = np.zeros(3, dtype=np.uint64)
    occ_bbs[0] = white_pawns | piece_bbs[5]
    occ_bbs[1] = black_pawns | piece_bbs[11]
    occ_bbs[2] = occ_bbs[0] | occ_bbs[1]

    # 呼叫 evaluate_passed_pawns 並疊加 endgame 得分
    _, eg_passed_score = evaluate_passed_pawns(piece_bbs, occ_bbs, white_attacks, black_attacks)
    score += eg_passed_score

    # 3. 國王活動獎勵
    # 獎勵國王靠近所有兵 (自己的和對手的)
    # White king proximity
    temp_pawns = white_pawns | black_pawns
    while temp_pawns:
        sq = get_lsb_index(temp_pawns)
        # 距離越近，獎勵/懲罰越小，所以用最大距離減去實際距離
        distance = MANHATTAN_DISTANCE[white_king_sq, sq]
        score += (KING_TROPISM_MAX_DISTANCE - distance) * 5 # 給予一個較小的權重
        temp_pawns &= temp_pawns - np.uint64(1)

    # Black king proximity
    temp_pawns = white_pawns | black_pawns
    while temp_pawns:
        sq = get_lsb_index(temp_pawns)
        distance = MANHATTAN_DISTANCE[black_king_sq, sq]
        score -= (KING_TROPISM_MAX_DISTANCE - distance) * 5
        temp_pawns &= temp_pawns - np.uint64(1)

    return score


@numba.njit(numba.types.Tuple((numba.int32, numba.int32, numba.int32))(numba.int32, numba.uint64, numba.boolean), cache=True, boundscheck=False, fastmath=True, inline='always')
def _process_piece_score_and_count(piece_type, bb, is_white):
    mg = 0
    eg = 0
    count = 0
    while bb:
        sq = get_lsb_index(bb)
        if is_white:
            mg += PST_MG[piece_type, sq]
            eg += PST_EG[piece_type, sq]
        else:
            mg -= PST_MG[piece_type, sq ^ 56]
            eg -= PST_EG[piece_type, sq ^ 56]
        count += 1
        bb &= bb - np.uint64(1)
        
    if is_white:
        mg += count * MG_MATERIAL_VALUES[piece_type]
        eg += count * EG_MATERIAL_VALUES[piece_type]
    else:
        mg -= count * MG_MATERIAL_VALUES[piece_type]
        eg -= count * EG_MATERIAL_VALUES[piece_type]
        
    return mg, eg, count


@numba.njit(cache=True, boundscheck=False, fastmath=True)
def evaluate_space(piece_bbs, occupancy_bbs, white_attacks, black_attacks, white_pawn_attacks, black_pawn_attacks, piece_counts):
    """
    評估雙方的空間控制優勢（僅影響中局）。
    """
    # 1. 計算雙方 non-pawn 材質 (MG)
    w_non_pawn = (piece_counts[1] * MG_MATERIAL_VALUES[1] +
                  piece_counts[2] * MG_MATERIAL_VALUES[2] +
                  piece_counts[3] * MG_MATERIAL_VALUES[3] +
                  piece_counts[4] * MG_MATERIAL_VALUES[4])
                  
    b_non_pawn = (piece_counts[7] * MG_MATERIAL_VALUES[1] +
                  piece_counts[8] * MG_MATERIAL_VALUES[2] +
                  piece_counts[9] * MG_MATERIAL_VALUES[3] +
                  piece_counts[10] * MG_MATERIAL_VALUES[4])

    if (w_non_pawn + b_non_pawn) < SPACE_THRESHOLD:
        return np.int32(0)

    white_pawns = piece_bbs[0]
    black_pawns = piece_bbs[6]

    # 中央 4 直行: C, D, E, F 線
    center_files = FILE_MASKS[2] | FILE_MASKS[3] | FILE_MASKS[4] | FILE_MASKS[5]

    # White space mask (Rank 2, 3, 4)
    white_space_mask = center_files & (RANK_MASKS[1] | RANK_MASKS[2] | RANK_MASKS[3])
    # Black space mask (Rank 7, 6, 5)
    black_space_mask = center_files & (RANK_MASKS[6] | RANK_MASKS[5] | RANK_MASKS[4])

    # White safe squares (not occupied by our pawns, not attacked by enemy pawns)
    white_safe = white_space_mask & ~white_pawns & ~black_pawn_attacks
    # Black safe squares
    black_safe = black_space_mask & ~black_pawns & ~white_pawn_attacks

    # White behind pawns: up to 3 squares behind (southward)
    w_behind = white_pawns
    w_behind |= (w_behind >> np.uint64(8))
    w_behind |= (w_behind >> np.uint64(16))

    # Black behind pawns: up to 3 squares behind (northward)
    b_behind = black_pawns
    b_behind |= (b_behind << np.uint64(8))
    b_behind |= (b_behind << np.uint64(16))

    # Calculate bonuses
    w_bonus = count_bits(white_safe) + count_bits(w_behind & white_safe & ~black_attacks)
    b_bonus = count_bits(black_safe) + count_bits(b_behind & black_safe & ~white_attacks)

    # Calculate total piece counts per side (all pieces, including pawns and king)
    w_pieces_count = piece_counts[0] + piece_counts[1] + piece_counts[2] + piece_counts[3] + piece_counts[4] + piece_counts[5]
    b_pieces_count = piece_counts[6] + piece_counts[7] + piece_counts[8] + piece_counts[9] + piece_counts[10] + piece_counts[11]

    w_weight = w_pieces_count - 1
    b_weight = b_pieces_count - 1

    white_space_score = (w_bonus * w_weight * w_weight) // 16
    black_space_score = (b_bonus * b_weight * b_weight) // 16

    # Scale to engine's evaluation units (using optimized // 4 division for positional safety)
    white_space_scaled = white_space_score // 4
    black_space_scaled = black_space_score // 4

    # Difference in space score
    return np.int32(white_space_scaled - black_space_scaled)


@numba.njit(cache=True, boundscheck=False, fastmath=True)
def _evaluate_position_jit(piece_bbs, occupancy_bbs, game_state, lazy: bool, search_context=None):
    """
    使用 Tapered Evaluation (加權評估) 模型評估目前局面，並從當前執棋方的角度返回分數。
    評估包括：材質、PST、國王安全、兵形結構、棋子協同性和機動性。
    
    Args:
        piece_bbs (np.ndarray): 12 個棋子的位元棋盤。
        occupancy_bbs (np.ndarray): 佔用位元棋盤。
        game_state (np.ndarray): 遊戲狀態。
        lazy (bool): 是否使用懶惰評估（僅材質和 PST，用於 Razoring 等）。
        
    Returns:
        int: 從行棋方視角的評估分數。
    """
    # --- King and Pawn Endgame Check / 檢查王兵殘局 ---
    # 檢查是否只有王和兵，如果
    all_pieces_except_pawns_and_kings = (
        piece_bbs[1] | piece_bbs[2] | piece_bbs[3] | piece_bbs[4] |
        piece_bbs[7] | piece_bbs[8] | piece_bbs[9] | piece_bbs[10]
    )
    if all_pieces_except_pawns_and_kings == 0:
        score = _evaluate_king_pawn_endgame(piece_bbs, game_state[0])
        val = np.int32(score) if game_state[0] == 0 else np.int32(-score)
        return val, np.uint64(0), np.uint64(0)
    side_to_move = game_state[0]

    # --- 1. & 2. Phase, Material, and PST (Unrolled & Fused) ---
    # Unrolled to avoid array allocation for piece_counts and improve vectorization
    
    mg_0, eg_0, cnt_0 = _process_piece_score_and_count(0, piece_bbs[0], True)
    mg_1, eg_1, cnt_1 = _process_piece_score_and_count(1, piece_bbs[1], True)
    mg_2, eg_2, cnt_2 = _process_piece_score_and_count(2, piece_bbs[2], True)
    mg_3, eg_3, cnt_3 = _process_piece_score_and_count(3, piece_bbs[3], True)
    mg_4, eg_4, cnt_4 = _process_piece_score_and_count(4, piece_bbs[4], True)
    mg_5, eg_5, cnt_5 = _process_piece_score_and_count(5, piece_bbs[5], True)
    
    mg_6, eg_6, cnt_6 = _process_piece_score_and_count(0, piece_bbs[6], False)
    mg_7, eg_7, cnt_7 = _process_piece_score_and_count(1, piece_bbs[7], False)
    mg_8, eg_8, cnt_8 = _process_piece_score_and_count(2, piece_bbs[8], False)
    mg_9, eg_9, cnt_9 = _process_piece_score_and_count(3, piece_bbs[9], False)
    mg_10, eg_10, cnt_10 = _process_piece_score_and_count(4, piece_bbs[10], False)
    mg_11, eg_11, cnt_11 = _process_piece_score_and_count(5, piece_bbs[11], False)
    
    mg_score = np.int32(mg_0 + mg_1 + mg_2 + mg_3 + mg_4 + mg_5 + mg_6 + mg_7 + mg_8 + mg_9 + mg_10 + mg_11)
    eg_score = np.int32(eg_0 + eg_1 + eg_2 + eg_3 + eg_4 + eg_5 + eg_6 + eg_7 + eg_8 + eg_9 + eg_10 + eg_11)
    
    piece_counts = (cnt_0, cnt_1, cnt_2, cnt_3, cnt_4, cnt_5, cnt_6, cnt_7, cnt_8, cnt_9, cnt_10, cnt_11)
    
    # Calculate phase
    # PHASE_WEIGHTS: [0, 1, 1, 2, 4, 0] (Pawn, Knight, Bishop, Rook, Queen, King)
    # Using tuple indexing for counts
    phase = (cnt_1 * PHASE_WEIGHTS[1] + cnt_2 * PHASE_WEIGHTS[2] + cnt_3 * PHASE_WEIGHTS[3] + cnt_4 * PHASE_WEIGHTS[4] +
             cnt_7 * PHASE_WEIGHTS[1] + cnt_8 * PHASE_WEIGHTS[2] + cnt_9 * PHASE_WEIGHTS[3] + cnt_10 * PHASE_WEIGHTS[4])
    phase = min(phase, MAX_PHASE)

    # --- Lazy Evaluation Checkpoint / 懶惰評估檢查點 ---
    if lazy:
        final_score = (mg_score * phase + eg_score * (MAX_PHASE - phase)) // MAX_PHASE
        val = np.int32(final_score) if side_to_move == 0 else np.int32(-final_score)
        return val, np.uint64(0), np.uint64(0)

    # --- Compute Attacks, Mobility, Threats (Optimized Single Pass) ---
    (white_attacks, black_attacks, white_pawn_attacks, black_pawn_attacks,
     white_attacks2, black_attacks2, pinned_white, pinned_black,
     mg_mobility, eg_mobility, mg_threats, eg_threats,
     white_piece_tropism, black_piece_tropism, mg_outpost, eg_outpost) = evaluate_attacks_mobility_threats(piece_bbs, occupancy_bbs)

    # --- 4. 加入兵形結構分數 (Moved up for King Safety dependency) ---
    if search_context is not None:
        pawn_key = game_state[PAWN_KEY_INDEX]
        mg_pawn_structure, eg_pawn_structure, white_pawn_tropism, black_pawn_tropism, white_pawn_storm, black_pawn_storm = evaluate_pawn_structure_cached(piece_bbs, pawn_key, search_context)
    else:
        mg_pawn_structure, eg_pawn_structure, white_pawn_tropism, black_pawn_tropism, white_pawn_storm, black_pawn_storm = evaluate_pawn_structure(piece_bbs)
    mg_score += mg_pawn_structure
    eg_score += eg_pawn_structure

    # --- 4.5 評估並加入動態通路兵與候選兵分數 ---
    mg_passed, eg_passed = evaluate_passed_pawns(piece_bbs, occupancy_bbs, white_attacks, black_attacks)
    mg_score += mg_passed
    eg_score += eg_passed

    # --- 3. (Full Evaluation) 加入國王安全分數 ---
    white_attack_tropism = -(white_pawn_tropism + white_piece_tropism) # White attacking Black
    black_attack_tropism = -(black_pawn_tropism + black_piece_tropism) # Black attacking White
    
    # Pass 'black_attack_tropism' to White King Safety (because it represents danger TO White King)
    # Pass 'white_attack_tropism' to Black King Safety (because it represents danger TO Black King)
    mg_king_safety, eg_king_safety = evaluate_king_safety(piece_bbs, occupancy_bbs, white_attacks, black_attacks, black_attack_tropism, white_attack_tropism, white_pawn_storm, black_pawn_storm, piece_counts, white_attacks2, black_attacks2, pinned_white, pinned_black)
    mg_score += mg_king_safety
    eg_score += eg_king_safety

    # --- 5. 加入棋子協同性分數 ---
    mg_coord, eg_coord = evaluate_piece_coordination(piece_bbs, piece_counts)
    mg_score += mg_coord
    eg_score += eg_coord

    # --- 6. 加入棋子機動性分數 ---
    mg_score += mg_mobility
    eg_score += eg_mobility

    # --- 7. Join Outpost Evaluation / 加入前哨評估 ---
    # mg_outpost, eg_outpost = evaluate_outposts(piece_bbs, occupancy_bbs, white_pawn_attacks, black_pawn_attacks)
    mg_score += mg_outpost
    eg_score += eg_outpost

    # --- 8. Join Threat Evaluation / 加入威脅評估 ---
    mg_score += mg_threats
    eg_score += eg_threats

    # --- 8.5 空間評估 / Space Evaluation ---
    # mg_space = evaluate_space(piece_bbs, occupancy_bbs, white_attacks, black_attacks, white_pawn_attacks, black_pawn_attacks, piece_counts)
    # mg_score += mg_space

    # --- 9. 根據遊戲階段進行插值計算 ---
    final_score = (mg_score * phase + eg_score * (MAX_PHASE - phase)) // MAX_PHASE

    # --- 9.5 OCB Endgame Scale Factor / 異色象殘局縮放因子 ---
    # When both sides have exactly one bishop on opposite color squares and no
    # major pieces (rooks/queens), the position is likely drawn. Scale down eg.
    # 當雙方各有一個異色格象且無車后時，局面趨向和棋，縮減殘局差距。
    if phase < MAX_PHASE // 2:  # Only in near-endgame
        w_bishops = cnt_2  # white bishop count
        b_bishops = cnt_8  # black bishop count
        w_majors = cnt_3 + cnt_4  # white rooks + queens
        b_majors = cnt_9 + cnt_10  # black rooks + queens
        if w_bishops == 1 and b_bishops == 1 and w_majors == 0 and b_majors == 0:
            # Check if bishops are on opposite color squares
            wb_sq = get_lsb_index(piece_bbs[2])
            bb_sq = get_lsb_index(piece_bbs[8])
            wb_color = (wb_sq >> 3 ^ wb_sq) & 1
            bb_color = (bb_sq >> 3 ^ bb_sq) & 1
            if wb_color != bb_color:
                # Opposite color bishops: scale based on pawn count
                total_pawns = cnt_0 + cnt_6
                if total_pawns <= 1:
                    scale = SCALE_FACTOR_OCB_ONE_PAWN    # 16/64
                elif total_pawns <= 2:
                    scale = SCALE_FACTOR_OCB_TWO_PAWNS   # 32/64
                else:
                    scale = SCALE_FACTOR_OCB_MULTIPLE_PAWNS  # 48/64
                # Apply: shrink the endgame portion of the score
                eg_portion = eg_score * (MAX_PHASE - phase) // MAX_PHASE
                mg_portion = final_score - eg_portion
                final_score = mg_portion + eg_portion * scale // SCALE_FACTOR_NORMAL

    # --- 10. (Optional) Initiative Bonus / 主動權獎勵 ---
    if phase > INITIATIVE_PHASE_THRESHOLD:
        if side_to_move == 0:
            final_score += INITIATIVE_BONUS
        else:
            final_score -= INITIATIVE_BONUS

    # --- 11. 從當前執棋方的角度返回最終分數 ---
    if side_to_move == 0:  # 白方回合
        return np.int32(final_score), pinned_white, pinned_black
    else:  # 黑方回合
        return np.int32(-final_score), pinned_white, pinned_black

def evaluate_position(piece_bbs, occupancy_bbs, game_state, lazy: bool = False):
    """
    使用 Tapered Evaluation (加權評估) 模型評估目前局面，並從當前執棋方的角度返回分數。
    """
    return _evaluate_position_jit(piece_bbs, occupancy_bbs, game_state, lazy)
