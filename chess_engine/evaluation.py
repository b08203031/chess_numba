# chess_engine/evaluation.py

import numba
import numpy as np

from chess_engine.constants import *

from chess_engine.zobrist import get_lsb_index
from chess_engine.engine_types import piece_bbs_signature, occupancy_bbs_signature, game_state_signature
from chess_engine.bitboard_utils import count_bits, KING_ATTACK_ZONES, FILE_MASKS
from chess_engine.move_generator import (
    get_bishop_attacks, get_rook_attacks, get_queen_attacks, KNIGHT_ATTACKS, PAWN_ATTACKS
)

# --- Pre-computed Manhattan Distance Table / 預計算曼哈頓距離表 ---
def _create_manhattan_distance_table():
    """
    Pre-computes a 64x64 lookup table for the manhattan distance between any two squares.
    Distance = max(abs(rank1 - rank2), abs(file1 - file2)).
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


@numba.njit(numba.types.UniTuple(numba.int32, 2)(piece_bbs_signature), cache=True, boundscheck=False, fastmath=True)
def evaluate_pawn_structure(piece_bbs):
    """
    評估雙方的兵型結構（通路兵、孤兵、重疊兵、後兵、連結兵）。
    
    Args:
        piece_bbs (np.ndarray): 12 個棋子的位元棋盤。
        
    Returns:
        tuple: (mg_score, eg_score) 從白方視角。
    """
    mg_score = np.int32(0)
    eg_score = np.int32(0)
    
    white_pawns = piece_bbs[0]
    black_pawns = piece_bbs[6]
    white_king_sq = get_lsb_index(piece_bbs[5])
    black_king_sq = get_lsb_index(piece_bbs[11])

    # --- 1. Iterate White Pawns ---
    temp_wp = white_pawns
    while temp_wp:
        sq = get_lsb_index(temp_wp)
        rank = sq // 8
        file_idx = sq % 8

        # A. Passed Pawn Logic
        if not (WHITE_PASSED_PAWN_MASKS[sq] & black_pawns):
            mg_score += PASSED_PAWN_BONUS[rank][0]
            eg_score += PASSED_PAWN_BONUS[rank][1]

            # Connected Passed Pawn Bonus
            # Check if there is a friendly pawn on adjacent files (rank +/- 1 or same)
            # Simplified: just adjacent files mask & white_pawns
            if (ADJACENT_FILES_MASKS[file_idx] & white_pawns):
                 mg_score += CONNECTED_PASSED_PAWN_BONUS[0]
                 eg_score += CONNECTED_PASSED_PAWN_BONUS[1]

            # King Proximity Logic (Stockfish-like)
            if rank > 3: # Only consider advanced passed pawns for proximity logic to save time/noise
                block_sq = sq + 8 # Square in front
                if block_sq < 64:
                    dist_friendly = MANHATTAN_DISTANCE[white_king_sq, block_sq]
                    dist_enemy = MANHATTAN_DISTANCE[black_king_sq, block_sq]
                    # Bonus if friendly king is closer, penalty if enemy is closer
                    # Weight: 5 * Enemy - 2 * Friendly
                    proximity_bonus = (dist_enemy * 5 - dist_friendly * 2) * rank # Scale by rank
                    # Limit the impact
                    proximity_bonus = max(-150, min(150, proximity_bonus))
                    eg_score += proximity_bonus

        # B. Backward Pawn Logic
        # Definition: No friendly pawn on adjacent files is at the same rank or ahead (supporting).
        # And the stop square (sq + 8) is controlled by an enemy pawn.
        else: # Not passed (optimization: backward pawns usually aren't passed, though technically possible)
             # Check adjacent friendly pawns support
             # Friendly pawns on adjacent files AND (rank >= current rank)
             # Using precomputed masks would be faster but for now:
             adjacent_pawns = ADJACENT_FILES_MASKS[file_idx] & white_pawns
             # Check if any adjacent pawn is on rank >= current rank
             # Mask for ranks >= current rank
             # We can use ~BLACK_FORWARD_RANKS[sq] which gives ranks >= rank?
             # Actually, simpler:
             # WHITE_FORWARD_RANKS[sq] gives ranks > rank.
             # We need ranks >= rank. So WHITE_FORWARD_RANKS[sq] | rank_mask[rank].

             support_mask = WHITE_FORWARD_RANKS[sq] | RANK_MASKS[rank]
             has_support = (adjacent_pawns & support_mask) != 0

             if not has_support:
                 # Check if stop square is attacked by enemy pawn
                 # Stop square for white is sq + 8
                 stop_sq = sq + 8
                 if stop_sq < 64:
                     # Check if black pawns attack stop_sq
                     # PAWN_ATTACKS[1][stop_sq] gives squares occupied by Black pawns that attack stop_sq.
                     if (PAWN_ATTACKS[1][stop_sq] & black_pawns):
                         mg_score -= BACKWARD_PAWN_PENALTY[0]
                         eg_score -= BACKWARD_PAWN_PENALTY[1]

        temp_wp &= temp_wp - np.uint64(1)

    # --- 2. Iterate Black Pawns ---
    temp_bp = black_pawns
    while temp_bp:
        sq = get_lsb_index(temp_bp)
        rank = sq // 8 # 0-7
        relative_rank = 7 - rank
        file_idx = sq % 8

        # A. Passed Pawn Logic
        if not (BLACK_PASSED_PAWN_MASKS[sq] & white_pawns):
            mg_score -= PASSED_PAWN_BONUS[relative_rank][0]
            eg_score -= PASSED_PAWN_BONUS[relative_rank][1]

            # Connected Passed Pawn Bonus
            if (ADJACENT_FILES_MASKS[file_idx] & black_pawns):
                 mg_score -= CONNECTED_PASSED_PAWN_BONUS[0]
                 eg_score -= CONNECTED_PASSED_PAWN_BONUS[1]

            # King Proximity Logic
            if relative_rank > 3:
                block_sq = sq - 8
                if block_sq >= 0:
                    dist_friendly = MANHATTAN_DISTANCE[black_king_sq, block_sq]
                    dist_enemy = MANHATTAN_DISTANCE[white_king_sq, block_sq]

                    proximity_bonus = (dist_enemy * 5 - dist_friendly * 2) * relative_rank
                    proximity_bonus = max(-150, min(150, proximity_bonus))
                    eg_score -= proximity_bonus

        # B. Backward Pawn Logic
        else:
             # Support: Friendly pawns on adjacent files and rank <= current rank (since black moves down)
             # BLACK_FORWARD_RANKS[sq] gives ranks < rank.
             support_mask = BLACK_FORWARD_RANKS[sq] | RANK_MASKS[rank]
             adjacent_pawns = ADJACENT_FILES_MASKS[file_idx] & black_pawns
             has_support = (adjacent_pawns & support_mask) != 0

             if not has_support:
                 stop_sq = sq - 8
                 if stop_sq >= 0:
                     # Check if white pawns attack stop_sq
                     # PAWN_ATTACKS[0][stop_sq] gives squares occupied by White pawns that attack stop_sq.
                     if (PAWN_ATTACKS[0][stop_sq] & white_pawns):
                         mg_score += BACKWARD_PAWN_PENALTY[0]
                         eg_score += BACKWARD_PAWN_PENALTY[1]

        temp_bp &= temp_bp - np.uint64(1)

    # --- 3. Isolated and Doubled Pawns / 孤兵與重疊兵 ---
    # Iterate through each file to check for pawn formations.
    for f in range(8):
        file_mask = FILE_MASKS[f]
        adjacent_mask = ADJACENT_FILES_MASKS[f]

        white_pawns_on_file = count_bits(white_pawns & file_mask)
        black_pawns_on_file = count_bits(black_pawns & file_mask)

        # White pawns
        if white_pawns_on_file > 0:
            # Isolated
            if not (white_pawns & adjacent_mask):
                mg_score += ISOLATED_PAWN_PENALTY[0] * white_pawns_on_file
                eg_score += ISOLATED_PAWN_PENALTY[1] * white_pawns_on_file
            # Doubled
            if white_pawns_on_file > 1:
                mg_score += DOUBLED_PAWN_PENALTY[0] * (white_pawns_on_file - 1)
                eg_score += DOUBLED_PAWN_PENALTY[1] * (white_pawns_on_file - 1)

        # Black pawns
        if black_pawns_on_file > 0:
            # Isolated
            if not (black_pawns & adjacent_mask):
                mg_score -= ISOLATED_PAWN_PENALTY[0] * black_pawns_on_file
                eg_score -= ISOLATED_PAWN_PENALTY[1] * black_pawns_on_file
            # Doubled
            if black_pawns_on_file > 1:
                mg_score -= DOUBLED_PAWN_PENALTY[0] * (black_pawns_on_file - 1)
                eg_score -= DOUBLED_PAWN_PENALTY[1] * (black_pawns_on_file - 1)

    return mg_score, eg_score


@numba.njit(numba.types.UniTuple(numba.int32, 2)(piece_bbs_signature), cache=True, boundscheck=False, fastmath=True)
def evaluate_piece_coordination(piece_bbs):
    """
    評估棋子協同性特徵（雙象、車在開放線）。
    
    Args:
        piece_bbs (np.ndarray): 12 個棋子的位元棋盤。
        
    Returns:
        tuple: (mg_score, eg_score) 從白方視角。
    """
    mg_score = np.int32(0)
    eg_score = np.int32(0)

    white_pawns = piece_bbs[0]
    white_bishops = piece_bbs[2]
    white_rooks = piece_bbs[3]
    black_pawns = piece_bbs[6]
    black_bishops = piece_bbs[8]
    black_rooks = piece_bbs[9]

    # --- 1. Bishop Pair / 雙象 ---
    # A bonus is awarded if a side has two or more bishops.
    # 擁有雙象給予獎勵。
    if count_bits(white_bishops) >= 2:
        mg_score += BISHOP_PAIR_BONUS[0]
        eg_score += BISHOP_PAIR_BONUS[1]
    if count_bits(black_bishops) >= 2:
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
    (Phase 1) Evaluates the pawn shield in front of the king for a single color.
    (階段 1) 評估單一方國王前方的兵盾。
    """
    score = np.int32(0)
    king_file = king_sq % 8
    
    original_rank = 1 if color == 0 else 6
    one_step_rank = 2 if color == 0 else 5

    for f in range(max(0, king_file - 1), min(7, king_file + 1) + 1):
        file_mask = FILE_MASKS[f]

        pawns_on_file = friendly_pawns & file_mask

        if not pawns_on_file:
            score -= PAWN_SHIELD_MISSING_PENALTY # Pawn shield missing
        else:
            pawn_sq = get_lsb_index(pawns_on_file)
            pawn_rank = pawn_sq // 8
            
            if pawn_rank == original_rank:
                score += PAWN_SHIELD_INTACT_BONUS # Intact shield bonus increased
            elif pawn_rank == one_step_rank:
                score += PAWN_SHIELD_ADVANCED_BONUS # Advanced shield bonus increased
            else:
                score -= PAWN_SHIELD_PUSHED_PENALTY # Pushed shield penalty increased

        if not (friendly_pawns & file_mask):
            if not (enemy_pawns & file_mask):
                score -= KING_OPEN_FILE_PENALTY # Open file penalty increased from 20
            else:
                score -= KING_SEMI_OPEN_FILE_PENALTY # Semi-open file penalty increased from 10

    return score

@numba.njit(numba.int32(numba.int32, numba.int32, piece_bbs_signature, occupancy_bbs_signature), cache=True, boundscheck=False, fastmath=True)
def _evaluate_king_attackers(king_sq, color, piece_bbs, occupancy_bbs):
    """
    (Phase 2) Calculates the threat score based on pieces attacking the king zone using a non-linear model.
    (階段 2) 根據攻擊國王區域的棋子，使用非線性模型計算威脅分數。
    """
    king_zone = KING_ATTACK_ZONES[king_sq]
    total_attack_units = np.int32(0)
    attacker_count = np.int32(0)

    enemy_start_idx = 6 if color == 0 else 0
    (en_p, en_n, en_b, en_r, en_q, _) = piece_bbs[enemy_start_idx : enemy_start_idx + 6]

    all_pieces_occupancy = occupancy_bbs[2]

    # Pawns - Find all enemy pawns that attack any square in the king zone
    # 兵 - 尋找所有攻擊國王區域內任何方格的敵兵
    # pawn_attack_sources_bb = np.uint64(0)
    # temp_king_zone = king_zone
    # while temp_king_zone:
    #     zone_sq = get_lsb_index(temp_king_zone)
    #     # PAWN_ATTACKS[color_of_king, zone_sq] gives squares from which enemy pawns would attack zone_sq
    #     pawn_attack_sources_bb |= PAWN_ATTACKS[color, zone_sq]
    #     temp_king_zone &= temp_king_zone - np.uint64(1)

    # actual_pawn_attackers = pawn_attack_sources_bb & en_p
    # if actual_pawn_attackers:
    #     num_pawn_attackers = count_bits(actual_pawn_attackers)
    #     total_attack_units += num_pawn_attackers * KING_SAFETY_ATTACK_UNITS[0]
    #     attacker_count += num_pawn_attackers

    # Knights
    temp_bb = en_n
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        if KNIGHT_ATTACKS[sq] & king_zone:
            total_attack_units += KING_SAFETY_ATTACK_UNITS[1]
            attacker_count += 1
        temp_bb &= temp_bb - np.uint64(1)

    # Bishops
    temp_bb = en_b
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        attacks = get_bishop_attacks(sq, all_pieces_occupancy & ~BB_SQUARES[sq])
        if attacks & king_zone:
            total_attack_units += KING_SAFETY_ATTACK_UNITS[2]
            attacker_count += 1
        temp_bb &= temp_bb - np.uint64(1)

    # Rooks
    temp_bb = en_r
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        attacks = get_rook_attacks(sq, all_pieces_occupancy & ~BB_SQUARES[sq])
        if attacks & king_zone:
            total_attack_units += KING_SAFETY_ATTACK_UNITS[3]
            attacker_count += 1
        temp_bb &= temp_bb - np.uint64(1)

    # Queens
    temp_bb = en_q
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        attacks = get_queen_attacks(sq, all_pieces_occupancy & ~BB_SQUARES[sq])
        if attacks & king_zone:
            total_attack_units += KING_SAFETY_ATTACK_UNITS[4]
            attacker_count += 1
        temp_bb &= temp_bb - np.uint64(1)

    # Only apply penalty if there are multiple attackers, to avoid penalizing single-piece harassment.
    # 僅在有多個攻擊者時才施加懲罰，以避免懲罰單個棋子的騷擾。
    if attacker_count < 2:
       return np.int32(0)

    # The score from the table is a penalty, so it should be negative.
    return -KING_SAFETY_TABLE[min(total_attack_units, len(KING_SAFETY_TABLE) - 1)]

@numba.njit(numba.int32(numba.int32, numba.int32, piece_bbs_signature), cache=True, boundscheck=False, fastmath=True)
def _evaluate_king_tropism(king_sq, color, piece_bbs):
    """
    (Phase 3) Calculates a penalty based on the proximity of enemy pieces to the king.
    (階段 3) 根據敵方棋子與國王的距離計算懲罰。
    """
    penalty = np.int32(0)
    enemy_start_idx = 6 if color == 0 else 0

    for piece_type_offset in range(5):
        piece_type = enemy_start_idx + piece_type_offset
        weight = KING_TROPISM_WEIGHTS[piece_type_offset]

        temp_bb = piece_bbs[piece_type]
        while temp_bb:
            sq = get_lsb_index(temp_bb)
            distance = MANHATTAN_DISTANCE[sq, king_sq]
            penalty += weight * (KING_TROPISM_MAX_DISTANCE - distance)
            temp_bb &= temp_bb - np.uint64(1)

    return -penalty

@numba.njit(numba.int32(numba.int32, numba.int32, piece_bbs_signature), cache=True, boundscheck=False, fastmath=True)
def _evaluate_pawn_storm(king_sq, color, piece_bbs):
    """
    (Phase 4) Calculates a penalty for enemy pawns near the king (pawn storm).
    Revised to use rank-based penalty.
    """
    penalty = np.int32(0)
    king_file = king_sq % 8
    enemy_pawn_idx = 6 if color == 0 else 0
    enemy_pawns = piece_bbs[enemy_pawn_idx]

    # Iterate over files adjacent to the king
    for f in range(max(0, king_file - 1), min(7, king_file + 1) + 1):
        file_mask = FILE_MASKS[f]
        pawns_on_file = enemy_pawns & file_mask
        
        while pawns_on_file:
            sq = get_lsb_index(pawns_on_file)
            rank = sq // 8
            
            # Determine the penalty index based on the pawn's rank relative to the defending king's back rank
            # White King (color 0) is at rank 0. Black pawns attack. 
            # Black pawn at Rank 2 (index 2) -> Very close -> High penalty.
            # Black King (color 1) is at rank 7. White pawns attack.
            # White pawn at Rank 5 (index 5) -> 7-5=2 -> Very close -> High penalty.
            
            table_idx = rank if color == 0 else (7 - rank)
            penalty -= PAWN_STORM_PENALTY_BY_RANK[table_idx] # Penalty should be subtracted (score is relative to side to move) or returned as positive penalty to be subtracted later?
            # The function returns "penalty", and caller does: white_raw_safety + white_pawn_storm.
            # But the other functions return "score" (often negative for penalties).
            # Let's check _evaluate_king_tropism: returns -penalty.
            # Let's check _evaluate_king_attackers: returns -KING_SAFETY_TABLE[...].
            # So this function should return a NEGATIVE value.
            
            pawns_on_file &= pawns_on_file - np.uint64(1)

    return penalty

@numba.njit(numba.types.UniTuple(numba.int32, 2)(piece_bbs_signature, occupancy_bbs_signature), cache=True, boundscheck=False, fastmath=True)
def evaluate_king_safety(piece_bbs, occupancy_bbs):
    """
    Refactored King Safety evaluation based on Chess Programming Wiki.
    重構的國王安全評估，基於 Chess Programming Wiki。
    """
    (wp_bb, wn_bb, wb_bb, wr_bb, wq_bb, wk_bb,
    bp_bb, bn_bb, bb_bb, br_bb, bq_bb, bk_bb) = piece_bbs

    white_king_sq = get_lsb_index(wk_bb)
    black_king_sq = get_lsb_index(bk_bb)

    # --- Calculate raw scores for each component / 計算每個組件的原始分數 ---
    white_shield = _evaluate_pawn_shield_for_color(white_king_sq, wp_bb, bp_bb, 0)
    white_attackers = _evaluate_king_attackers(white_king_sq, 0, piece_bbs, occupancy_bbs)
    white_tropism = _evaluate_king_tropism(white_king_sq, 0, piece_bbs)
    white_pawn_storm = _evaluate_pawn_storm(white_king_sq, 0, piece_bbs)

    black_shield = _evaluate_pawn_shield_for_color(black_king_sq, bp_bb, wp_bb, 1)
    black_attackers = _evaluate_king_attackers(black_king_sq, 1, piece_bbs, occupancy_bbs)
    black_tropism = _evaluate_king_tropism(black_king_sq, 1, piece_bbs)
    black_pawn_storm = _evaluate_pawn_storm(black_king_sq, 1, piece_bbs)

    # --- Sum raw scores / 加總原始分數 ---
    white_raw_safety = white_shield + white_attackers + white_tropism + white_pawn_storm
    black_raw_safety = black_shield + black_attackers + black_tropism + black_pawn_storm

    # --- Phase 4: Scaling based on enemy material / 階段 4：基於敵方材質進行縮放 ---
    black_material_for_scaling = (count_bits(bn_bb) * SCALING_WEIGHTS[0] +
                                 count_bits(bb_bb) * SCALING_WEIGHTS[1] +
                                 count_bits(br_bb) * SCALING_WEIGHTS[2] +
                                 count_bits(bq_bb) * SCALING_WEIGHTS[3])
    white_scaling_factor = black_material_for_scaling / MAX_SCALING_MATERIAL

    white_final_safety = np.int32(white_raw_safety * white_scaling_factor)

    white_material_for_scaling = (count_bits(wn_bb) * SCALING_WEIGHTS[0] +
                                 count_bits(wb_bb) * SCALING_WEIGHTS[1] +
                                 count_bits(wr_bb) * SCALING_WEIGHTS[2] +
                                 count_bits(wq_bb) * SCALING_WEIGHTS[3])
    black_scaling_factor = white_material_for_scaling / MAX_SCALING_MATERIAL

    black_final_safety = np.int32(black_raw_safety * black_scaling_factor)

    mg_safety_score = white_final_safety - black_final_safety

    # In the endgame, king safety is much less of a concern, and an active king is
    # often an advantage. The King's PST already encourages centralization.
    # Therefore, we set the endgame king safety score to 0.
    # 在殘局中，國王安全通常不那麼重要，活躍的國王往往是優勢。
    # 國王的 PST 已經鼓勵中心化。因此，我們將殘局國王安全分數設為 0（或按比例縮減）。
    eg_safety_score = mg_safety_score * EG_SAFETY_SCALE

    return mg_safety_score, eg_safety_score


@numba.njit(numba.types.UniTuple(numba.int32, 2)(piece_bbs_signature, occupancy_bbs_signature), cache=True, boundscheck=False, fastmath=True)
def evaluate_mobility(piece_bbs, occupancy_bbs):
    """
    評估雙方棋子的機動性。
    
    Args:
        piece_bbs (np.ndarray): 12 個棋子的位元棋盤。
        occupancy_bbs (np.ndarray): 佔用位元棋盤。
        
    Returns:
        tuple: (mg_score, eg_score) 從白方視角。
    """
    mg_score = np.int32(0)
    eg_score = np.int32(0)

    white_occupancy = occupancy_bbs[0]
    black_occupancy = occupancy_bbs[1]
    all_pieces_occupancy = occupancy_bbs[2]

    # --- White Mobility / 白方機動性 ---
    # Knights
    temp_bb = piece_bbs[1]
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        moves = count_bits(KNIGHT_ATTACKS[sq] & ~white_occupancy)
        mg_score += (moves - KNIGHT_MOBILITY_BASE_MOVES) * KNIGHT_MOBILITY_WEIGHT[0]
        eg_score += (moves - KNIGHT_MOBILITY_BASE_MOVES) * KNIGHT_MOBILITY_WEIGHT[1]
        temp_bb &= temp_bb - np.uint64(1)

    # Bishops
    temp_bb = piece_bbs[2]
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        moves = count_bits(get_bishop_attacks(sq, all_pieces_occupancy) & ~white_occupancy)
        mg_score += (moves - BISHOP_MOBILITY_BASE_MOVES) * BISHOP_MOBILITY_WEIGHT[0]
        eg_score += (moves - BISHOP_MOBILITY_BASE_MOVES) * BISHOP_MOBILITY_WEIGHT[1]
        temp_bb &= temp_bb - np.uint64(1)

    # Rooks
    temp_bb = piece_bbs[3]
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        moves = count_bits(get_rook_attacks(sq, all_pieces_occupancy) & ~white_occupancy)
        mg_score += (moves - ROOK_MOBILITY_BASE_MOVES) * ROOK_MOBILITY_WEIGHT[0]
        eg_score += (moves - ROOK_MOBILITY_BASE_MOVES) * ROOK_MOBILITY_WEIGHT[1]
        temp_bb &= temp_bb - np.uint64(1)

    # Queens
    temp_bb = piece_bbs[4]
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        moves = count_bits(get_queen_attacks(sq, all_pieces_occupancy) & ~white_occupancy)
        mg_score += (moves - QUEEN_MOBILITY_BASE_MOVES) * QUEEN_MOBILITY_WEIGHT[0]
        eg_score += (moves - QUEEN_MOBILITY_BASE_MOVES) * QUEEN_MOBILITY_WEIGHT[1]
        temp_bb &= temp_bb - np.uint64(1)


    # --- Black Mobility / 黑方機動性 ---
    # Knights
    temp_bb = piece_bbs[7]
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        moves = count_bits(KNIGHT_ATTACKS[sq] & ~black_occupancy)
        mg_score -= (moves - KNIGHT_MOBILITY_BASE_MOVES) * KNIGHT_MOBILITY_WEIGHT[0]
        eg_score -= (moves - KNIGHT_MOBILITY_BASE_MOVES) * KNIGHT_MOBILITY_WEIGHT[1]
        temp_bb &= temp_bb - np.uint64(1)

    # Bishops
    temp_bb = piece_bbs[8]
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        moves = count_bits(get_bishop_attacks(sq, all_pieces_occupancy) & ~black_occupancy)
        mg_score -= (moves - BISHOP_MOBILITY_BASE_MOVES) * BISHOP_MOBILITY_WEIGHT[0]
        eg_score -= (moves - BISHOP_MOBILITY_BASE_MOVES) * BISHOP_MOBILITY_WEIGHT[1]
        temp_bb &= temp_bb - np.uint64(1)

    # Rooks
    temp_bb = piece_bbs[9]
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        moves = count_bits(get_rook_attacks(sq, all_pieces_occupancy) & ~black_occupancy)
        mg_score -= (moves - ROOK_MOBILITY_BASE_MOVES) * ROOK_MOBILITY_WEIGHT[0]
        eg_score -= (moves - ROOK_MOBILITY_BASE_MOVES) * ROOK_MOBILITY_WEIGHT[1]
        temp_bb &= temp_bb - np.uint64(1)

    # Queens
    temp_bb = piece_bbs[10]
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        moves = count_bits(get_queen_attacks(sq, all_pieces_occupancy) & ~black_occupancy)
        mg_score -= (moves - QUEEN_MOBILITY_BASE_MOVES) * QUEEN_MOBILITY_WEIGHT[0]
        eg_score -= (moves - QUEEN_MOBILITY_BASE_MOVES) * QUEEN_MOBILITY_WEIGHT[1]
        temp_bb &= temp_bb - np.uint64(1)

    return mg_score, eg_score


@numba.njit(numba.int32(piece_bbs_signature), cache=True, boundscheck=False, fastmath=True)
def _evaluate_king_pawn_endgame(piece_bbs):
    """
    專門為王兵殘局設計的評估函數。
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
        score += EG_MATERIAL_VALUES[0] + PST_EG[0][sq]
        temp_wp &= temp_wp - np.uint64(1)

    # Black pawns
    temp_bp = black_pawns
    while temp_bp:
        sq = get_lsb_index(temp_bp)
        score -= (EG_MATERIAL_VALUES[0] + PST_EG[0][sq ^ 56])
        temp_bp &= temp_bp - np.uint64(1)

    # King position
    score += PST_EG[5][white_king_sq]
    score -= PST_EG[5][black_king_sq ^ 56]

    # 2. 通路兵獎勵 (使用 evaluate_pawn_structure 簡化計算)
    _, eg_pawn_score = evaluate_pawn_structure(piece_bbs)
    score += eg_pawn_score

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


@numba.njit(numba.int32(piece_bbs_signature, occupancy_bbs_signature, game_state_signature, numba.boolean), cache=True, boundscheck=False, fastmath=True)
def evaluate_position(piece_bbs, occupancy_bbs, game_state, lazy: bool = False):
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
        score = _evaluate_king_pawn_endgame(piece_bbs)
        return score if game_state[0] == 0 else -score
    side_to_move = game_state[0]

    # --- 1. 計算遊戲階段 (Game Phase) ---
    phase = np.int32(0)
    phase += count_bits(piece_bbs[1]) * PHASE_WEIGHTS[1]
    phase += count_bits(piece_bbs[2]) * PHASE_WEIGHTS[2]
    phase += count_bits(piece_bbs[3]) * PHASE_WEIGHTS[3]
    phase += count_bits(piece_bbs[4]) * PHASE_WEIGHTS[4]
    phase += count_bits(piece_bbs[7]) * PHASE_WEIGHTS[1]
    phase += count_bits(piece_bbs[8]) * PHASE_WEIGHTS[2]
    phase += count_bits(piece_bbs[9]) * PHASE_WEIGHTS[3]
    phase += count_bits(piece_bbs[10]) * PHASE_WEIGHTS[4]
    phase = min(phase, MAX_PHASE)

    # --- 2. 計算中局和殘局的基礎分數（物質 + 位置） ---
    mg_score = np.int32(0)
    eg_score = np.int32(0)

    for piece_type in range(6):
        mg_score += count_bits(piece_bbs[piece_type]) * MG_MATERIAL_VALUES[piece_type]
        eg_score += count_bits(piece_bbs[piece_type]) * EG_MATERIAL_VALUES[piece_type]
        mg_score -= count_bits(piece_bbs[piece_type + 6]) * MG_MATERIAL_VALUES[piece_type]
        eg_score -= count_bits(piece_bbs[piece_type + 6]) * EG_MATERIAL_VALUES[piece_type]

    for piece_type in range(6):
        bb = piece_bbs[piece_type]
        while bb:
            sq = get_lsb_index(bb)
            mg_score += PST_MG[piece_type][sq]
            eg_score += PST_EG[piece_type][sq]
            bb &= bb - np.uint64(1)

    for piece_type in range(6):
        bb = piece_bbs[piece_type + 6]
        while bb:
            sq = get_lsb_index(bb)
            mg_score -= PST_MG[piece_type][sq ^ 56]
            eg_score -= PST_EG[piece_type][sq ^ 56]
            bb &= bb - np.uint64(1)

    # --- Lazy Evaluation Checkpoint / 懶惰評估檢查點 ---
    if lazy:
        final_score = (mg_score * phase + eg_score * (MAX_PHASE - phase)) // MAX_PHASE
        return np.int32(final_score) if side_to_move == 0 else np.int32(-final_score)

    # --- 3. (Full Evaluation) 加入國王安全分數 ---
    mg_king_safety, eg_king_safety = evaluate_king_safety(piece_bbs, occupancy_bbs)
    mg_score += mg_king_safety
    eg_score += eg_king_safety

    # --- 4. 加入兵形結構分數 ---
    mg_pawn_structure, eg_pawn_structure = evaluate_pawn_structure(piece_bbs)
    mg_score += mg_pawn_structure
    eg_score += eg_pawn_structure

    # --- 5. 加入棋子協同性分數 ---
    mg_coord, eg_coord = evaluate_piece_coordination(piece_bbs)
    mg_score += mg_coord
    eg_score += eg_coord

    # --- 6. 加入棋子機動性分數 --- (暫時移除機動性評估以加快速度)
    mg_mobility, eg_mobility = evaluate_mobility(piece_bbs, occupancy_bbs)
    mg_score += mg_mobility
    eg_score += eg_mobility

    # --- 7. 根據遊戲階段進行插值計算 ---
    final_score = (mg_score * phase + eg_score * (MAX_PHASE - phase)) // MAX_PHASE

    # --- 8. (Optional) Initiative Bonus / 主動權獎勵 ---
    if phase > INITIATIVE_PHASE_THRESHOLD:
        final_score += INITIATIVE_BONUS

    # --- 9. 從當前執棋方的角度返回最終分數 ---
    if side_to_move == 0:  # 白方回合
        return np.int32(final_score)
    else:  # 黑方回合
        return np.int32(-final_score)
    # return np.int32(0)  # Placeholder return statement
