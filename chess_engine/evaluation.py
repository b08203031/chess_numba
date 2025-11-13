# chess_engine/evaluation.py

import numba
import numpy as np

from chess_engine.constants import *

from chess_engine.zobrist import get_lsb_index
from chess_engine.engine_types import piece_bbs_signature, occupancy_bbs_signature, game_state_signature
from chess_engine.bitboard_utils import count_bits, KING_ATTACK_ZONES, FILE_MASKS
from chess_engine.move_generator import (
    get_bishop_attacks, get_rook_attacks, get_queen_attacks, KNIGHT_ATTACKS
)

# --- Pre-computed Manhattan Distance Table ---
def _create_manhattan_distance_table():
    """
    Pre-computes a 64x64 lookup table for the manhattan distance between any two squares.
    Distance = max(abs(rank1 - rank2), abs(file1 - file2)).
    """
    table = np.zeros((64, 64), dtype=np.int32)
    for sq1 in range(64):
        rank1, file1 = sq1 // 8, sq1 % 8
        for sq2 in range(64):
            rank2, file2 = sq2 // 8, sq2 % 8
            table[sq1, sq2] = abs(rank1 - rank2) + abs(file1 - file2)
    return table

MANHATTAN_DISTANCE = _create_manhattan_distance_table()


# --- Pre-computed Masks for Pawn Structure Evaluation ---

# Masks for adjacent files (e.g., for file B, it's file A and C)
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

def _create_passed_pawn_masks():
    white_masks = np.zeros(64, dtype=np.uint64)
    black_masks = np.zeros(64, dtype=np.uint64)
    for sq in range(64):
        file_idx = sq % 8
        rank_idx = sq // 8
        
        # Mask includes the pawn's own file and adjacent files
        path_mask = FILE_MASKS[file_idx] | ADJACENT_FILES_MASKS[file_idx]
        
        # White passed pawn: no black pawns in front on the path
        white_front_span = np.uint64(0)
        for r in range(rank_idx + 1, 8):
            white_front_span |= (path_mask & (np.uint64(0xFF) << np.uint64(r * 8)))
        white_masks[sq] = white_front_span

        # Black passed pawn: no white pawns in front on the path
        black_front_span = np.uint64(0)
        for r in range(rank_idx - 1, -1, -1):
            black_front_span |= (path_mask & (np.uint64(0xFF) << np.uint64(r * 8)))
        black_masks[sq] = black_front_span
        
    return white_masks, black_masks

WHITE_PASSED_PAWN_MASKS, BLACK_PASSED_PAWN_MASKS = _create_passed_pawn_masks()


@numba.njit(numba.types.UniTuple(numba.int32, 2)(piece_bbs_signature), cache=True, boundscheck=False, fastmath=True)
def evaluate_pawn_structure(piece_bbs):
    """
    Evaluates pawn structure for both sides (passed, isolated, doubled).
    Returns a tuple of (mg_score, eg_score) from White's perspective.
    """
    mg_score = np.int32(0)
    eg_score = np.int32(0)
    
    white_pawns = piece_bbs[0]
    black_pawns = piece_bbs[6]

    # --- 1. Passed Pawns ---
    # A pawn is passed if there are no opponent pawns in front of it on its
    # own file or on adjacent files.
    
    # White passed pawns
    temp_wp = white_pawns
    while temp_wp:
        sq = get_lsb_index(temp_wp)
        if not (WHITE_PASSED_PAWN_MASKS[sq] & black_pawns):
            rank = sq // 8
            mg_score += PASSED_PAWN_BONUS[rank][0]
            eg_score += PASSED_PAWN_BONUS[rank][1]
        temp_wp &= temp_wp - np.uint64(1)

    # Black passed pawns
    temp_bp = black_pawns
    while temp_bp:
        sq = get_lsb_index(temp_bp)
        if not (BLACK_PASSED_PAWN_MASKS[sq] & white_pawns):
            rank = 7 - (sq // 8) # Rank from Black's perspective
            mg_score -= PASSED_PAWN_BONUS[rank][0]
            eg_score -= PASSED_PAWN_BONUS[rank][1]
        temp_bp &= temp_bp - np.uint64(1)

    # --- 2. Isolated and Doubled Pawns ---
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
    Evaluates piece coordination features (bishop pair, rooks on open files).
    Returns a tuple of (mg_score, eg_score) from White's perspective.
    """
    mg_score = np.int32(0)
    eg_score = np.int32(0)

    white_pawns = piece_bbs[0]
    white_bishops = piece_bbs[2]
    white_rooks = piece_bbs[3]
    black_pawns = piece_bbs[6]
    black_bishops = piece_bbs[8]
    black_rooks = piece_bbs[9]

    # --- 1. Bishop Pair ---
    # A bonus is awarded if a side has two or more bishops.
    if count_bits(white_bishops) >= 2:
        mg_score += BISHOP_PAIR_BONUS[0]
        eg_score += BISHOP_PAIR_BONUS[1]
    if count_bits(black_bishops) >= 2:
        mg_score -= BISHOP_PAIR_BONUS[0]
        eg_score -= BISHOP_PAIR_BONUS[1]

    # --- 2. Rooks on Open and Semi-Open Files ---
    for f in range(8):
        file_mask = FILE_MASKS[f]
        
        white_pawns_on_file = (white_pawns & file_mask) != 0
        black_pawns_on_file = (black_pawns & file_mask) != 0

        # White rooks
        if (white_rooks & file_mask):
            if not white_pawns_on_file:
                if not black_pawns_on_file:
                    # Open file for White
                    mg_score += ROOK_ON_OPEN_FILE_BONUS[0]
                    eg_score += ROOK_ON_OPEN_FILE_BONUS[1]
                else:
                    # Semi-open file for White
                    mg_score += ROOK_ON_SEMI_OPEN_FILE_BONUS[0]
                    eg_score += ROOK_ON_SEMI_OPEN_FILE_BONUS[1]
        
        # Black rooks
        if (black_rooks & file_mask):
            if not black_pawns_on_file:
                if not white_pawns_on_file:
                    # Open file for Black
                    mg_score -= ROOK_ON_OPEN_FILE_BONUS[0]
                    eg_score -= ROOK_ON_OPEN_FILE_BONUS[1]
                else:
                    # Semi-open file for Black
                    mg_score -= ROOK_ON_SEMI_OPEN_FILE_BONUS[0]
                    eg_score -= ROOK_ON_SEMI_OPEN_FILE_BONUS[1]

    return mg_score, eg_score


@numba.njit(numba.int32(numba.int32, numba.uint64, numba.uint64, numba.int32), cache=True, boundscheck=False, fastmath=True)
def _evaluate_pawn_shield_for_color(king_sq, friendly_pawns, enemy_pawns, color):
    """
    (Phase 1) Evaluates the pawn shield in front of the king for a single color.
    """
    score = np.int32(0)
    king_file = king_sq % 8
    
    original_rank = 1 if color == 0 else 6
    one_step_rank = 2 if color == 0 else 5

    for f in range(max(0, king_file - 1), min(7, king_file + 1) + 1):
        file_mask = FILE_MASKS[f]

        pawns_on_file = friendly_pawns & file_mask

        if not pawns_on_file:
            score -= 20
        else:
            pawn_sq = get_lsb_index(pawns_on_file)
            pawn_rank = pawn_sq // 8
            
            if pawn_rank == original_rank:
                score += 10
            elif pawn_rank == one_step_rank:
                score += 5
            else:
                score -= 10

        if not (friendly_pawns & file_mask):
            if not (enemy_pawns & file_mask):
                score -= 20
            else:
                score -= 10

    return score

@numba.njit(numba.int32(numba.int32, numba.int32, piece_bbs_signature, occupancy_bbs_signature), cache=True, boundscheck=False, fastmath=True)
def _evaluate_king_attackers(king_sq, color, piece_bbs, occupancy_bbs):
    """
    (Phase 2) Calculates the threat score based on pieces attacking the king zone.
    """
    king_zone = KING_ATTACK_ZONES[king_sq]
    total_attack_units = np.int32(0)
    attacker_count = np.int32(0)

    enemy_start_idx = 6 if color == 0 else 0
    (_, en_n, en_b, en_r, en_q, _) = piece_bbs[enemy_start_idx : enemy_start_idx + 6]

    all_pieces_occupancy = occupancy_bbs[2]

    # Knights
    temp_bb = en_n
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        if KNIGHT_ATTACKS[sq] & king_zone:
            total_attack_units += KING_SAFETY_ATTACK_UNITS[0]
            attacker_count += 1
        temp_bb &= temp_bb - np.uint64(1)

    temp_occ = all_pieces_occupancy & ~king_zone

    # Bishops
    temp_bb = en_b
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        attacks = get_bishop_attacks(sq, temp_occ & ~BB_SQUARES[sq])
        if attacks & king_zone:
            total_attack_units += KING_SAFETY_ATTACK_UNITS[1]
            attacker_count += 1
        temp_bb &= temp_bb - np.uint64(1)

    # Rooks
    temp_bb = en_r
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        attacks = get_rook_attacks(sq, temp_occ & ~BB_SQUARES[sq])
        if attacks & king_zone:
            total_attack_units += KING_SAFETY_ATTACK_UNITS[2]
            attacker_count += 1
        temp_bb &= temp_bb - np.uint64(1)

    # Queens
    temp_bb = en_q
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        attacks = get_queen_attacks(sq, temp_occ & ~BB_SQUARES[sq])
        if attacks & king_zone:
            total_attack_units += KING_SAFETY_ATTACK_UNITS[3]
            attacker_count += 1
        temp_bb &= temp_bb - np.uint64(1)

    if attacker_count < 2:
        return np.int32(0)

    return -KING_SAFETY_TABLE[min(total_attack_units, len(KING_SAFETY_TABLE) - 1)]

@numba.njit(numba.int32(numba.int32, numba.int32, piece_bbs_signature), cache=True, boundscheck=False, fastmath=True)
def _evaluate_king_tropism(king_sq, color, piece_bbs):
    """
    (Phase 3) Calculates a penalty based on the proximity of enemy pieces to the king.
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
    """
    penalty = np.int32(0)
    king_zone = KING_ATTACK_ZONES[king_sq]
    enemy_pawn_idx = 6 if color == 0 else 0

    enemy_pawns_in_zone = piece_bbs[enemy_pawn_idx] & king_zone
    penalty += count_bits(enemy_pawns_in_zone) * PAWN_STORM_PENALTY

    return penalty

@numba.njit(numba.types.UniTuple(numba.int32, 2)(piece_bbs_signature, occupancy_bbs_signature), cache=True, boundscheck=False, fastmath=True)
def evaluate_king_safety(piece_bbs, occupancy_bbs):
    """
    Refactored King Safety evaluation based on Chess Programming Wiki.
    """
    (wp_bb, wn_bb, wb_bb, wr_bb, wq_bb, wk_bb,
    bp_bb, bn_bb, bb_bb, br_bb, bq_bb, bk_bb) = piece_bbs

    white_king_sq = get_lsb_index(wk_bb)
    black_king_sq = get_lsb_index(bk_bb)

    # --- Calculate raw scores for each component ---
    white_shield = _evaluate_pawn_shield_for_color(white_king_sq, wp_bb, bp_bb, 0)
    white_attackers = _evaluate_king_attackers(white_king_sq, 0, piece_bbs, occupancy_bbs)
    white_tropism = _evaluate_king_tropism(white_king_sq, 0, piece_bbs)
    white_pawn_storm = _evaluate_pawn_storm(white_king_sq, 0, piece_bbs)

    black_shield = _evaluate_pawn_shield_for_color(black_king_sq, bp_bb, wp_bb, 1)
    black_attackers = _evaluate_king_attackers(black_king_sq, 1, piece_bbs, occupancy_bbs)
    black_tropism = _evaluate_king_tropism(black_king_sq, 1, piece_bbs)
    black_pawn_storm = _evaluate_pawn_storm(black_king_sq, 1, piece_bbs)

    # --- Sum raw scores ---
    white_raw_safety = white_shield + white_attackers + white_tropism + white_pawn_storm
    black_raw_safety = black_shield + black_attackers + black_tropism + black_pawn_storm

    # --- Phase 4: Scaling based on enemy material ---
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
    eg_safety_score = mg_safety_score * EG_SAFETY_SCALE

    return mg_safety_score, eg_safety_score


@numba.njit(numba.types.UniTuple(numba.int32, 2)(piece_bbs_signature, occupancy_bbs_signature), cache=True, boundscheck=False, fastmath=True)
def evaluate_mobility(piece_bbs, occupancy_bbs):
    """
    Evaluates piece mobility for both sides.
    """
    mg_score = np.int32(0)
    eg_score = np.int32(0)

    white_occupancy = occupancy_bbs[0]
    black_occupancy = occupancy_bbs[1]
    all_pieces_occupancy = occupancy_bbs[2]

    # --- White Mobility ---
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


    # --- Black Mobility ---
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
    """
    # --- King and Pawn Endgame Check ---
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

    # --- Lazy Evaluation Checkpoint ---
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

    # --- 6. 加入棋子機動性分數 ---
    mg_mobility, eg_mobility = evaluate_mobility(piece_bbs, occupancy_bbs)
    mg_score += mg_mobility
    eg_score += eg_mobility

    # --- 7. 根據遊戲階段進行插值計算 ---
    final_score = (mg_score * phase + eg_score * (MAX_PHASE - phase)) // MAX_PHASE

    # --- 8. (Optional) Initiative Bonus ---
    if phase > INITIATIVE_PHASE_THRESHOLD:
        final_score += INITIATIVE_BONUS

    # --- 9. 從當前執棋方的角度返回最終分數 ---
    if side_to_move == 0:  # 白方回合
        return np.int32(final_score)
    else:  # 黑方回合
        return np.int32(-final_score)
