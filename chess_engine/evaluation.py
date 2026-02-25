# chess_engine/evaluation.py

import numba
import numpy as np

from chess_engine.constants import *

from chess_engine.bitboard_utils import get_lsb_index, count_bits, WHITE_KING_ZONES, BLACK_KING_ZONES, FILE_MASKS
from chess_engine.engine_types import piece_bbs_signature, occupancy_bbs_signature, game_state_signature, piece_counts_signature
from chess_engine.move_generator import (
    get_bishop_attacks, get_rook_attacks, get_queen_attacks, KNIGHT_ATTACKS, PAWN_ATTACKS, KING_ATTACKS
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
        # Check if there is a friendly pawn ahead on the same file
        if (WHITE_FORWARD_RANKS[sq] & white_pawns & FILE_MASKS[file_idx]):
             mg_score += DOUBLED_PAWN_PENALTY[0]
             eg_score += DOUBLED_PAWN_PENALTY[1]

        # A. Passed Pawn Logic
        if not (WHITE_PASSED_PAWN_MASKS[sq] & black_pawns):
            mg_score += PASSED_PAWN_BONUS[rank, 0]
            eg_score += PASSED_PAWN_BONUS[rank, 1]

            # Connected Passed Pawn Bonus
            if adjacent_pawns:
                 mg_score += CONNECTED_PASSED_PAWN_BONUS[0]
                 eg_score += CONNECTED_PASSED_PAWN_BONUS[1]

            # King Proximity Logic & Blocked Check
            # If the passed pawn is blocked by the enemy King, reduce the bonus drastically
            block_sq = sq + 8
            if block_sq < 64:
                dist_friendly = CHEBYSHEV_DISTANCE[white_king_sq, block_sq]
                dist_enemy = CHEBYSHEV_DISTANCE[black_king_sq, block_sq]

                # Rule 1: Pawn Blocked by Enemy King
                # If enemy King is on the stop square or directly blocking
                # For white pawn, stop sq is sq+8.
                # If enemy king is ON sq+8, distance is 0.
                is_blocked_by_king = (dist_enemy <= 1) and (dist_friendly > dist_enemy)
                
                if is_blocked_by_king:
                     # Reduce the passed pawn bonus significantly if blocked by King and undefended/unsupported
                     # Remove most of the bonus we just added
                     # e.g., reduce by 90%
                     penalty = np.int32(PASSED_PAWN_BONUS[rank, 1] * 0.9)
                     eg_score -= penalty
                     mg_score -= np.int32(PASSED_PAWN_BONUS[rank, 0] * 0.9)

                # Rule 2: Unstoppable Pawn (Simple Logic)
                # If Friendly King is closer or supports, and Enemy King is far
                # (TODO: Full Rule of the Square is complex with turn logic, this is a proxy)
                
                if rank > 3: # Only consider advanced passed pawns for proximity logic to save time/noise
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
             
             # Check if any adjacent pawn is on rank >= current rank
             # Mask for ranks >= current rank
             # We can use ~BLACK_FORWARD_RANKS[sq] which gives ranks >= rank.
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
                     # PAWN_ATTACKS[1, stop_sq] gives squares occupied by Black pawns that attack stop_sq.
                     if (PAWN_ATTACKS[1, stop_sq] & black_pawns):
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
        # Check if there is a friendly pawn ahead (towards rank 0) on the same file
        if (BLACK_FORWARD_RANKS[sq] & black_pawns & FILE_MASKS[file_idx]):
             mg_score -= DOUBLED_PAWN_PENALTY[0]
             eg_score -= DOUBLED_PAWN_PENALTY[1]

        # A. Passed Pawn Logic
        if not (BLACK_PASSED_PAWN_MASKS[sq] & white_pawns):
            mg_score -= PASSED_PAWN_BONUS[relative_rank, 0]
            eg_score -= PASSED_PAWN_BONUS[relative_rank, 1]

            # Connected Passed Pawn Bonus
            if adjacent_pawns:
                 mg_score -= CONNECTED_PASSED_PAWN_BONUS[0]
                 eg_score -= CONNECTED_PASSED_PAWN_BONUS[1]

            # King Proximity Logic & Blocked Check
            block_sq = sq - 8
            if block_sq >= 0:
                dist_friendly = CHEBYSHEV_DISTANCE[black_king_sq, block_sq]
                dist_enemy = CHEBYSHEV_DISTANCE[white_king_sq, block_sq]

                # Rule 1: Pawn Blocked by Enemy King
                is_blocked_by_king = (dist_enemy <= 1) and (dist_friendly > dist_enemy)
                
                if is_blocked_by_king:
                     penalty = np.int32(PASSED_PAWN_BONUS[relative_rank, 1] * 0.9)
                     eg_score += penalty # Add penalty because we subtracted bonus earlier (for black)
                     mg_score += np.int32(PASSED_PAWN_BONUS[relative_rank, 0] * 0.9)

                if relative_rank > 3:
                    proximity_bonus = (dist_enemy * 5 - dist_friendly * 2) * relative_rank
                    proximity_bonus = max(-150, min(150, proximity_bonus))
                    eg_score -= proximity_bonus

        # B. Backward Pawn Logic
        else:
             # Support: Friendly pawns on adjacent files and rank <= current rank (since black moves down)
             # BLACK_FORWARD_RANKS[sq] gives ranks < rank.
             support_mask = BLACK_FORWARD_RANKS[sq] | RANK_MASKS[rank]
             has_support = (adjacent_pawns & support_mask) != 0

             if not has_support:
                 stop_sq = sq - 8
                 if stop_sq >= 0:
                     # Check if white pawns attack stop_sq
                     # PAWN_ATTACKS[0, stop_sq] gives squares occupied by White pawns that attack stop_sq.
                     if (PAWN_ATTACKS[0, stop_sq] & white_pawns):
                         mg_score += BACKWARD_PAWN_PENALTY[0]
                         eg_score += BACKWARD_PAWN_PENALTY[1]

        temp_bp &= temp_bp - np.uint64(1)

    return mg_score, eg_score, white_pawn_tropism, black_pawn_tropism, white_pawn_storm, black_pawn_storm


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

@numba.njit(numba.int32(numba.int32, numba.int32, piece_bbs_signature, occupancy_bbs_signature, numba.uint64, numba.uint64, numba.uint64), cache=True, boundscheck=False, fastmath=True)
def _evaluate_king_attackers(king_sq, color, piece_bbs, occupancy_bbs, enemy_attacks_bb, friendly_attacks_bb, king_zone):
    """
    (Phase 2) Calculates the threat score based on pieces attacking the king zone using a non-linear model.
    (階段 2) 根據攻擊國王區域的棋子，使用非線性模型計算威脅分數。
    """
    # king_zone is now passed as an argument

    # Optimization: If no enemy piece attacks the king zone, we can skip individual piece checks.
    if not (enemy_attacks_bb & king_zone):
        return np.int32(0)

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
        # KNIGHT_ATTACKS[sq] gives squares from which enemy knights would attack sq
        knight_attacks_in_zone = KNIGHT_ATTACKS[sq] & king_zone
        if knight_attacks_in_zone:
            total_attack_units += KING_SAFETY_ATTACK_UNITS[1]
            attacker_count += 1

            # Check for weak squares attacked by this knight
            undefended_in_zone = knight_attacks_in_zone & ~friendly_attacks_bb
            weak_count = count_bits(undefended_in_zone)
            total_attack_units += weak_count * KING_SAFETY_WEAK_UNITS[1] # Each weak square adds more units

        temp_bb &= temp_bb - np.uint64(1)

    # Bishops
    temp_bb = en_b
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        attacks = get_bishop_attacks(sq, all_pieces_occupancy)
        attacks_in_zone = attacks & king_zone
        if attacks_in_zone:
            total_attack_units += KING_SAFETY_ATTACK_UNITS[2]
            attacker_count += 1

            # Check for weak squares attacked by this bishop
            undefended_in_zone = attacks_in_zone & ~friendly_attacks_bb
            weak_count = count_bits(undefended_in_zone)
            total_attack_units += weak_count * KING_SAFETY_WEAK_UNITS[2] # Each weak square adds more units
        temp_bb &= temp_bb - np.uint64(1)

    # Rooks
    temp_bb = en_r
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        attacks = get_rook_attacks(sq, all_pieces_occupancy)
        attack_in_zone = attacks & king_zone
        if attack_in_zone:
            total_attack_units += KING_SAFETY_ATTACK_UNITS[3]
            attacker_count += 1

            # Check for weak squares attacked by this rook
            undefended_in_zone = attack_in_zone & ~friendly_attacks_bb
            weak_count = count_bits(undefended_in_zone)
            total_attack_units += weak_count * KING_SAFETY_WEAK_UNITS[3] # Each weak square adds more units

        temp_bb &= temp_bb - np.uint64(1)

    # Queens
    temp_bb = en_q
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        attacks = get_queen_attacks(sq, all_pieces_occupancy)
        attack_in_zone = attacks & king_zone
        if attack_in_zone:
            total_attack_units += KING_SAFETY_ATTACK_UNITS[4]
            attacker_count += 1

            # Check for weak squares attacked by this queen
            undefended_in_zone = attack_in_zone & ~friendly_attacks_bb
            weak_count = count_bits(undefended_in_zone)
            total_attack_units += weak_count * KING_SAFETY_WEAK_UNITS[4] # Each weak square adds more units

        temp_bb &= temp_bb - np.uint64(1)

    if attacker_count < 2:
       return np.int32(0)
    
    # --- Weak Squares Logic (Stockfish 11) ---
    # Weak Square: A square in King Zone attacked by enemy but not defended by friendly pieces.
    # Note: Stockfish uses `~attackedBy2[Us]` which implies defended by at least 2 pieces? No, `~attackedBy2` means "not defended twice".
    # And `(~attackedBy[Us] | king | queen)`.
    # Let's simplify: Weak = Attacked by Enemy AND Not Defended by Friendly (or only King).
    # `friendly_attacks_bb` includes King attacks.
    # So `weak_squares` = (king_zone & enemy_attacks_bb) & ~friendly_attacks_bb.
    # But wait, `friendly_attacks_bb` includes King's own defense. The King defends its own zone.
    # If we want "Weak", it means King is the ONLY defender, or no defender.
    # If King defends it, it's technically defended.
    # But if King is the only defender and it's attacked by e.g. Rook, it's unsafe.
    # Let's define Weak Square as: Attacked by Enemy AND Not Defended by Friendly *except King*.
    # So we need `friendly_attacks_without_king`.
    # For now, let's use the provided `friendly_attacks_bb` which includes King.
    # If a square is attacked by enemy and NOT defended by anyone (undefended hole), it's very weak.

    # SF11 adds to `kingDanger`.
    # My `KING_SAFETY_TABLE` maps units to score.
    # Adding to units makes sense because multiple weak squares amplify the danger.

    # The score from the table is a penalty, so it should be negative.
    return -KING_SAFETY_TABLE[min(total_attack_units, len(KING_SAFETY_TABLE) - 1)]

@numba.njit(numba.types.UniTuple(numba.int32, 2)(piece_bbs_signature, occupancy_bbs_signature, numba.uint64, numba.uint64, numba.int32, numba.int32, numba.int32, numba.int32, piece_counts_signature), cache=True, boundscheck=False, fastmath=True)
def evaluate_king_safety(piece_bbs, occupancy_bbs, white_attacks, black_attacks, white_tropism, black_tropism, white_pawn_storm_score, black_pawn_storm_score, piece_counts):
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
    white_attackers = _evaluate_king_attackers(white_king_sq, 0, piece_bbs, occupancy_bbs, black_attacks, white_attacks, WHITE_KING_ZONES[white_king_sq])

    black_shield = _evaluate_pawn_shield_for_color(black_king_sq, bp_bb, wp_bb, 1)
    black_attackers = _evaluate_king_attackers(black_king_sq, 1, piece_bbs, occupancy_bbs, white_attacks, black_attacks, BLACK_KING_ZONES[black_king_sq])

    # --- Sum raw scores / 加總原始分數 ---
    white_raw_safety = white_shield + white_attackers + white_tropism + white_pawn_storm_score
    black_raw_safety = black_shield + black_attackers + black_tropism + black_pawn_storm_score

    # --- Phase 4: Scaling based on enemy material / 階段 4：基於敵方材質進行縮放 ---
    black_material_for_scaling = (piece_counts[7] * SCALING_WEIGHTS[0] +
                                 piece_counts[8] * SCALING_WEIGHTS[1] +
                                 piece_counts[9] * SCALING_WEIGHTS[2] +
                                 piece_counts[10] * SCALING_WEIGHTS[3])
    white_scaling_factor = black_material_for_scaling / MAX_SCALING_MATERIAL

    white_final_safety = np.int32(white_raw_safety * white_scaling_factor)

    white_material_for_scaling = (piece_counts[1] * SCALING_WEIGHTS[0] +
                                 piece_counts[2] * SCALING_WEIGHTS[1] +
                                 piece_counts[3] * SCALING_WEIGHTS[2] +
                                 piece_counts[4] * SCALING_WEIGHTS[3])
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


@numba.njit(numba.types.Tuple((numba.uint64, numba.uint64, numba.uint64, numba.uint64, numba.int32, numba.int32, numba.int32, numba.int32, numba.int32, numba.int32, numba.int32, numba.int32))(piece_bbs_signature, occupancy_bbs_signature), cache=True, boundscheck=False, fastmath=True)
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

    white_attacks = white_pawn_attacks
    black_attacks = black_pawn_attacks

    # --- Safe Masks for Mobility ---
    white_safe_mask = ~(black_pawn_attacks | wk_bb | wq_bb)
    black_safe_mask = ~(white_pawn_attacks | bk_bb | bq_bb)

    # --- Accumulators ---
    mg_mobility = np.int32(0)
    eg_mobility = np.int32(0)
    
    white_minor_attacks = np.uint64(0)
    white_rook_attacks = np.uint64(0)
    
    black_minor_attacks = np.uint64(0)
    black_rook_attacks = np.uint64(0)

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
        white_attacks |= att
        white_minor_attacks |= att
        moves = count_bits(att & ~white_occupancy & white_safe_mask)
        mg_mobility += (moves - KNIGHT_MOBILITY_BASE_MOVES) * KNIGHT_MOBILITY_WEIGHT[0]
        eg_mobility += (moves - KNIGHT_MOBILITY_BASE_MOVES) * KNIGHT_MOBILITY_WEIGHT[1]
        
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
        att = get_bishop_attacks(sq, all_occupancy)
        white_attacks |= att
        white_minor_attacks |= att
        moves = count_bits(att & ~white_occupancy & white_safe_mask)
        mg_mobility += (moves - BISHOP_MOBILITY_BASE_MOVES) * BISHOP_MOBILITY_WEIGHT[0]
        eg_mobility += (moves - BISHOP_MOBILITY_BASE_MOVES) * BISHOP_MOBILITY_WEIGHT[1]
        
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
        att = get_rook_attacks(sq, all_occupancy)
        white_attacks |= att
        white_rook_attacks |= att
        moves = count_bits(att & ~white_occupancy & white_safe_mask)
        mg_mobility += (moves - ROOK_MOBILITY_BASE_MOVES) * ROOK_MOBILITY_WEIGHT[0]
        eg_mobility += (moves - ROOK_MOBILITY_BASE_MOVES) * ROOK_MOBILITY_WEIGHT[1]
        
        # Tropism
        distance = MANHATTAN_DISTANCE[sq, black_king_sq]
        white_piece_tropism += KING_TROPISM_WEIGHTS[3] * (KING_TROPISM_MAX_DISTANCE - distance)

        temp_bb &= temp_bb - np.uint64(1)

    temp_bb = wq_bb
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        att = get_queen_attacks(sq, all_occupancy)
        white_attacks |= att
        moves = count_bits(att & ~white_occupancy & white_safe_mask)
        mg_mobility += (moves - QUEEN_MOBILITY_BASE_MOVES) * QUEEN_MOBILITY_WEIGHT[0]
        eg_mobility += (moves - QUEEN_MOBILITY_BASE_MOVES) * QUEEN_MOBILITY_WEIGHT[1]
        
        # Tropism
        distance = MANHATTAN_DISTANCE[sq, black_king_sq]
        white_piece_tropism += KING_TROPISM_WEIGHTS[4] * (KING_TROPISM_MAX_DISTANCE - distance)

        temp_bb &= temp_bb - np.uint64(1)

    if wk_bb:
        white_attacks |= KING_ATTACKS[get_lsb_index(wk_bb)]


    # --- Black Pieces ---
    temp_bb = bn_bb
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        att = KNIGHT_ATTACKS[sq]
        black_attacks |= att
        black_minor_attacks |= att
        moves = count_bits(att & ~black_occupancy & black_safe_mask)
        mg_mobility -= (moves - KNIGHT_MOBILITY_BASE_MOVES) * KNIGHT_MOBILITY_WEIGHT[0]
        eg_mobility -= (moves - KNIGHT_MOBILITY_BASE_MOVES) * KNIGHT_MOBILITY_WEIGHT[1]
        
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
        att = get_bishop_attacks(sq, all_occupancy)
        black_attacks |= att
        black_minor_attacks |= att
        moves = count_bits(att & ~black_occupancy & black_safe_mask)
        mg_mobility -= (moves - BISHOP_MOBILITY_BASE_MOVES) * BISHOP_MOBILITY_WEIGHT[0]
        eg_mobility -= (moves - BISHOP_MOBILITY_BASE_MOVES) * BISHOP_MOBILITY_WEIGHT[1]
        
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
        att = get_rook_attacks(sq, all_occupancy)
        black_attacks |= att
        black_rook_attacks |= att
        moves = count_bits(att & ~black_occupancy & black_safe_mask)
        mg_mobility -= (moves - ROOK_MOBILITY_BASE_MOVES) * ROOK_MOBILITY_WEIGHT[0]
        eg_mobility -= (moves - ROOK_MOBILITY_BASE_MOVES) * ROOK_MOBILITY_WEIGHT[1]
        
        # Tropism
        distance = MANHATTAN_DISTANCE[sq, white_king_sq]
        black_piece_tropism += KING_TROPISM_WEIGHTS[3] * (KING_TROPISM_MAX_DISTANCE - distance)

        temp_bb &= temp_bb - np.uint64(1)

    temp_bb = bq_bb
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        att = get_queen_attacks(sq, all_occupancy)
        black_attacks |= att
        moves = count_bits(att & ~black_occupancy & black_safe_mask)
        mg_mobility -= (moves - QUEEN_MOBILITY_BASE_MOVES) * QUEEN_MOBILITY_WEIGHT[0]
        eg_mobility -= (moves - QUEEN_MOBILITY_BASE_MOVES) * QUEEN_MOBILITY_WEIGHT[1]
        
        # Tropism
        distance = MANHATTAN_DISTANCE[sq, white_king_sq]
        black_piece_tropism += KING_TROPISM_WEIGHTS[4] * (KING_TROPISM_MAX_DISTANCE - distance)

        temp_bb &= temp_bb - np.uint64(1)

    if bk_bb:
        black_attacks |= KING_ATTACKS[get_lsb_index(bk_bb)]


    # --- Threats Evaluation ---
    mg_threats = np.int32(0)
    eg_threats = np.int32(0)

    # 1. Safe Pawn Threats
    black_non_pawns = bn_bb | bb_bb | br_bb | bq_bb
    white_non_pawns = wn_bb | wb_bb | wr_bb | wq_bb

    white_safe_pawn_threats = white_pawn_attacks & black_non_pawns
    if white_safe_pawn_threats:
        count = count_bits(white_safe_pawn_threats)
        mg_threats += THREAT_SAFE_PAWN[0] * count
        eg_threats += THREAT_SAFE_PAWN[1] * count

    black_safe_pawn_threats = black_pawn_attacks & white_non_pawns
    if black_safe_pawn_threats:
        count = count_bits(black_safe_pawn_threats)
        mg_threats -= THREAT_SAFE_PAWN[0] * count
        eg_threats -= THREAT_SAFE_PAWN[1] * count

    # 2. Hanging Pieces
    all_black_pieces = black_non_pawns | bp_bb
    black_hanging_mask = (all_black_pieces & white_attacks) & ~black_attacks
    if black_hanging_mask:
        count = count_bits(black_hanging_mask)
        mg_threats += THREAT_HANGING[0] * count
        eg_threats += THREAT_HANGING[1] * count

    all_white_pieces = white_non_pawns | wp_bb
    white_hanging_mask = (all_white_pieces & black_attacks) & ~white_attacks
    if white_hanging_mask:
        count = count_bits(white_hanging_mask)
        mg_threats -= THREAT_HANGING[0] * count
        eg_threats -= THREAT_HANGING[1] * count

    # 3. Minor Attacking Major
    black_majors = br_bb | bq_bb
    threats_w = white_minor_attacks & black_majors
    if threats_w:
        count = count_bits(threats_w)
        mg_threats += THREAT_MINOR_ON_MAJOR[0] * count
        eg_threats += THREAT_MINOR_ON_MAJOR[1] * count

    white_majors = wr_bb | wq_bb
    threats_b = black_minor_attacks & white_majors
    if threats_b:
        count = count_bits(threats_b)
        mg_threats -= THREAT_MINOR_ON_MAJOR[0] * count
        eg_threats -= THREAT_MINOR_ON_MAJOR[1] * count

    # 4. Rook Attacking Queen
    if white_rook_attacks & bq_bb:
        count = count_bits(white_rook_attacks & bq_bb)
        mg_threats += THREAT_ROOK_ON_QUEEN[0] * count
        eg_threats += THREAT_ROOK_ON_QUEEN[1] * count

    if black_rook_attacks & wq_bb:
        count = count_bits(black_rook_attacks & wq_bb)
        mg_threats -= THREAT_ROOK_ON_QUEEN[0] * count
        eg_threats -= THREAT_ROOK_ON_QUEEN[1] * count

    return (white_attacks, black_attacks, white_pawn_attacks, black_pawn_attacks, 
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
            if side_to_move == 0: # White to move
                adjusted_pawn_steps -= 1
            
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
            if side_to_move == 1: # Black to move
                adjusted_pawn_steps -= 1
                
            if king_dist > adjusted_pawn_steps:
                score -= 800

        temp_bp &= temp_bp - np.uint64(1)

    # King position
    score += PST_EG[5, white_king_sq]
    score -= PST_EG[5, black_king_sq ^ 56]

    # 2. 通路兵獎勵 (使用 evaluate_pawn_structure 簡化計算)
    _, eg_pawn_score, _, _, _, _ = evaluate_pawn_structure(piece_bbs)
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
        score = _evaluate_king_pawn_endgame(piece_bbs, game_state[0])
        return score if game_state[0] == 0 else -score
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
        return np.int32(final_score) if side_to_move == 0 else np.int32(-final_score)

    # --- Compute Attacks, Mobility, Threats (Optimized Single Pass) ---
    (white_attacks, black_attacks, white_pawn_attacks, black_pawn_attacks,
     mg_mobility, eg_mobility, mg_threats, eg_threats,
     white_piece_tropism, black_piece_tropism, mg_outpost, eg_outpost) = evaluate_attacks_mobility_threats(piece_bbs, occupancy_bbs)

    # --- 4. 加入兵形結構分數 (Moved up for King Safety dependency) ---
    mg_pawn_structure, eg_pawn_structure, white_pawn_tropism, black_pawn_tropism, white_pawn_storm, black_pawn_storm = evaluate_pawn_structure(piece_bbs)
    mg_score += mg_pawn_structure
    eg_score += eg_pawn_structure

    # --- 3. (Full Evaluation) 加入國王安全分數 ---
    white_attack_tropism = -(white_pawn_tropism + white_piece_tropism) # White attacking Black
    black_attack_tropism = -(black_pawn_tropism + black_piece_tropism) # Black attacking White
    
    # Pass 'black_attack_tropism' to White King Safety (because it represents danger TO White King)
    # Pass 'white_attack_tropism' to Black King Safety (because it represents danger TO Black King)
    mg_king_safety, eg_king_safety = evaluate_king_safety(piece_bbs, occupancy_bbs, white_attacks, black_attacks, black_attack_tropism, white_attack_tropism, white_pawn_storm, black_pawn_storm, piece_counts)
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

    # --- 9. 根據遊戲階段進行插值計算 ---
    final_score = (mg_score * phase + eg_score * (MAX_PHASE - phase)) // MAX_PHASE

    # --- 10. (Optional) Initiative Bonus / 主動權獎勵 ---
    if phase > INITIATIVE_PHASE_THRESHOLD:
        final_score += INITIATIVE_BONUS

    # --- 11. 從當前執棋方的角度返回最終分數 ---
    if side_to_move == 0:  # 白方回合
        return np.int32(final_score)
    else:  # 黑方回合
        return np.int32(-final_score)
