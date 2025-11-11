# chess_engine/evaluation.py

import numba
import numpy as np

from chess_engine.constants import (
    MG_MATERIAL_VALUES, EG_MATERIAL_VALUES,
    PST_MG, PST_EG,
    PHASE_WEIGHTS, MAX_PHASE,
    PAWN_SHIELD_BONUS, SEMI_OPEN_FILE_PENALTY, ATTACKER_WEIGHTS, UNCASTLED_SHIELD_DIVISOR,
    PASSED_PAWN_BONUS, ISOLATED_PAWN_PENALTY, DOUBLED_PAWN_PENALTY,
    BISHOP_PAIR_BONUS, ROOK_ON_SEMI_OPEN_FILE_BONUS, ROOK_ON_OPEN_FILE_BONUS,
    BB_SQUARES
)

from chess_engine.zobrist import get_lsb_index
from chess_engine.engine_types import piece_bbs_signature, occupancy_bbs_signature, game_state_signature
from chess_engine.bitboard_utils import count_bits, KING_ATTACK_ZONES, FILE_MASKS
from chess_engine.move_generator import (
    get_bishop_attacks, get_rook_attacks, get_queen_attacks, KNIGHT_ATTACKS
)
from chess_engine.constants import (
    MOBILITY_BASE_MOVES, MOBILITY_WEIGHT_MG, MOBILITY_WEIGHT_EG,
    MANHATTAN_DISTANCE, MAX_MANHATTAN_DISTANCE,
    QUEEN_TROPISM_WEIGHT, ROOK_TROPISM_WEIGHT, BISHOP_TROPISM_WEIGHT, KNIGHT_TROPISM_WEIGHT,
    KING_ANTI_MOBILITY_BASE_MOVES, KING_ANTI_MOBILITY_WEIGHT, INITIATIVE_BONUS
)
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
        if not (BLACK_PASSED_PAWN_MASKS[sq] & black_pawns):
            rank = sq // 8
            mg_score += PASSED_PAWN_BONUS[rank][0]
            eg_score += PASSED_PAWN_BONUS[rank][1]
        temp_wp &= temp_wp - np.uint64(1)

    # Black passed pawns
    temp_bp = black_pawns
    while temp_bp:
        sq = get_lsb_index(temp_bp)
        if not (WHITE_PASSED_PAWN_MASKS[sq] & white_pawns):
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


@numba.njit(numba.types.UniTuple(numba.int32, 2)(piece_bbs_signature, occupancy_bbs_signature), cache=True, boundscheck=False, fastmath=True)
def evaluate_mobility(piece_bbs, occupancy_bbs):
    """
    Evaluates piece mobility for both sides.
    Returns a tuple of (mg_score, eg_score) from White's perspective.
    """
    mg_score = np.int32(0)
    eg_score = np.int32(0)

    white_pieces_bb, black_pieces_bb, _ = occupancy_bbs
    all_pieces_bb = white_pieces_bb | black_pieces_bb

    # Piece types to evaluate: KNIGHT, BISHOP, ROOK, QUEEN
    white_piece_indices = (1, 2, 3, 4)
    black_piece_indices = (7, 8, 9, 10)

    # --- White Mobility ---
    for i in range(len(white_piece_indices)):
        piece_idx = white_piece_indices[i]
        bb = piece_bbs[piece_idx]
        while bb:
            sq = get_lsb_index(bb)
            if piece_idx == 1: # Knight
                attacks = KNIGHT_ATTACKS[sq]
            elif piece_idx == 2: # Bishop
                attacks = get_bishop_attacks(sq, all_pieces_bb)
            elif piece_idx == 3: # Rook
                attacks = get_rook_attacks(sq, all_pieces_bb)
            else: # Queen
                attacks = get_queen_attacks(sq, all_pieces_bb)

            moves = count_bits(attacks & ~white_pieces_bb)

            # The piece type index for constants is 0-5
            const_idx = piece_idx
            bonus = (moves - MOBILITY_BASE_MOVES[const_idx])
            mg_score += bonus * MOBILITY_WEIGHT_MG[const_idx]
            eg_score += bonus * MOBILITY_WEIGHT_EG[const_idx]

            bb &= bb - np.uint64(1)

    # --- Black Mobility ---
    for i in range(len(black_piece_indices)):
        piece_idx = black_piece_indices[i]
        bb = piece_bbs[piece_idx]
        while bb:
            sq = get_lsb_index(bb)
            if piece_idx == 7: # Knight
                attacks = KNIGHT_ATTACKS[sq]
            elif piece_idx == 8: # Bishop
                attacks = get_bishop_attacks(sq, all_pieces_bb)
            elif piece_idx == 9: # Rook
                attacks = get_rook_attacks(sq, all_pieces_bb)
            else: # Queen
                attacks = get_queen_attacks(sq, all_pieces_bb)

            moves = count_bits(attacks & ~black_pieces_bb)

            # The piece type index for constants is 0-5
            const_idx = piece_idx - 6
            bonus = (moves - MOBILITY_BASE_MOVES[const_idx])
            mg_score -= bonus * MOBILITY_WEIGHT_MG[const_idx]
            eg_score -= bonus * MOBILITY_WEIGHT_EG[const_idx]

            bb &= bb - np.uint64(1)

    return mg_score, eg_score


@numba.njit(numba.types.UniTuple(numba.int32, 2)(piece_bbs_signature, occupancy_bbs_signature), cache=True, boundscheck=False, fastmath=True)
def evaluate_king_safety(piece_bbs, occupancy_bbs):
    """
    A hybrid evaluation of king safety, combining the performant parts of the original
    implementation with the logically correct parts of the new implementation.
    """
    mg_safety_score = np.int32(0)
    eg_safety_score = np.int32(0)
    
    (wp_bb, wn_bb, wb_bb, wr_bb, wq_bb, wk_bb,
    bp_bb, bn_bb, bb_bb, br_bb, bq_bb, bk_bb) = piece_bbs

    white_pieces_bb, black_pieces_bb, _ = occupancy_bbs
    all_pieces_bb = white_pieces_bb | black_pieces_bb

    # --- 1. Pawn Shield Evaluation (Restored from original version for performance) ---
    W_KS_SHIELD_PERFECT = BB_SQUARES[13] | BB_SQUARES[14] | BB_SQUARES[15]
    W_KS_SHIELD_ADVANCED = BB_SQUARES[21] | BB_SQUARES[22] | BB_SQUARES[23]
    W_QS_SHIELD_PERFECT = BB_SQUARES[8] | BB_SQUARES[9] | BB_SQUARES[10]
    W_QS_SHIELD_ADVANCED = BB_SQUARES[16] | BB_SQUARES[17] | BB_SQUARES[18]
    
    B_KS_SHIELD_PERFECT = BB_SQUARES[53] | BB_SQUARES[54] | BB_SQUARES[55]
    B_KS_SHIELD_ADVANCED = BB_SQUARES[45] | BB_SQUARES[46] | BB_SQUARES[47]
    B_QS_SHIELD_PERFECT = BB_SQUARES[48] | BB_SQUARES[49] | BB_SQUARES[50]
    B_QS_SHIELD_ADVANCED = BB_SQUARES[40] | BB_SQUARES[41] | BB_SQUARES[42]

    white_king_sq = get_lsb_index(wk_bb)
    black_king_sq = get_lsb_index(bk_bb)

    # White Pawn Shield
    if white_king_sq == 6:  # g1
        perfect = count_bits(wp_bb & W_KS_SHIELD_PERFECT)
        advanced = count_bits(wp_bb & W_KS_SHIELD_ADVANCED)
        mg_safety_score += perfect * PAWN_SHIELD_BONUS[0][0] + advanced * PAWN_SHIELD_BONUS[1][0]
        eg_safety_score += perfect * PAWN_SHIELD_BONUS[0][1] + advanced * PAWN_SHIELD_BONUS[1][1]
    elif white_king_sq == 2:  # c1
        perfect = count_bits(wp_bb & W_QS_SHIELD_PERFECT)
        advanced = count_bits(wp_bb & W_QS_SHIELD_ADVANCED)
        mg_safety_score += perfect * PAWN_SHIELD_BONUS[0][0] + advanced * PAWN_SHIELD_BONUS[1][0]
        eg_safety_score += perfect * PAWN_SHIELD_BONUS[0][1] + advanced * PAWN_SHIELD_BONUS[1][1]
    elif white_king_sq == 4:  # e1 (uncastled)
        W_CENTER_SHIELD_PERFECT = BB_SQUARES[11] | BB_SQUARES[12] | BB_SQUARES[13]
        W_CENTER_SHIELD_ADVANCED = BB_SQUARES[19] | BB_SQUARES[20] | BB_SQUARES[21]
        perfect = count_bits(wp_bb & W_CENTER_SHIELD_PERFECT)
        advanced = count_bits(wp_bb & W_CENTER_SHIELD_ADVANCED)
        mg_safety_score += (perfect * PAWN_SHIELD_BONUS[0][0] + advanced * PAWN_SHIELD_BONUS[1][0]) // UNCASTLED_SHIELD_DIVISOR
        eg_safety_score += (perfect * PAWN_SHIELD_BONUS[0][1] + advanced * PAWN_SHIELD_BONUS[1][1]) // UNCASTLED_SHIELD_DIVISOR

    # Black Pawn Shield
    if black_king_sq == 62:  # g8
        perfect = count_bits(bp_bb & B_KS_SHIELD_PERFECT)
        advanced = count_bits(bp_bb & B_KS_SHIELD_ADVANCED)
        mg_safety_score -= perfect * PAWN_SHIELD_BONUS[0][0] + advanced * PAWN_SHIELD_BONUS[1][0]
        eg_safety_score -= perfect * PAWN_SHIELD_BONUS[0][1] + advanced * PAWN_SHIELD_BONUS[1][1]
    elif black_king_sq == 58:  # c8
        perfect = count_bits(bp_bb & B_QS_SHIELD_PERFECT)
        advanced = count_bits(bp_bb & B_QS_SHIELD_ADVANCED)
        mg_safety_score -= perfect * PAWN_SHIELD_BONUS[0][0] + advanced * PAWN_SHIELD_BONUS[1][0]
        eg_safety_score -= perfect * PAWN_SHIELD_BONUS[0][1] + advanced * PAWN_SHIELD_BONUS[1][1]
    elif black_king_sq == 60:  # e8 (uncastled)
        B_CENTER_SHIELD_PERFECT = BB_SQUARES[51] | BB_SQUARES[52] | BB_SQUARES[53]
        B_CENTER_SHIELD_ADVANCED = BB_SQUARES[43] | BB_SQUARES[44] | BB_SQUARES[45]
        perfect = count_bits(bp_bb & B_CENTER_SHIELD_PERFECT)
        advanced = count_bits(bp_bb & B_CENTER_SHIELD_ADVANCED)
        mg_safety_score -= (perfect * PAWN_SHIELD_BONUS[0][0] + advanced * PAWN_SHIELD_BONUS[1][0]) // UNCASTLED_SHIELD_DIVISOR
        eg_safety_score -= (perfect * PAWN_SHIELD_BONUS[0][1] + advanced * PAWN_SHIELD_BONUS[1][1]) // UNCASTLED_SHIELD_DIVISOR

    # --- 2. Semi-Open Files Penalty (Restored from original version for performance) ---
    white_king_file = white_king_sq % 8
    for f in range(max(0, white_king_file - 1), min(7, white_king_file + 1) + 1):
        file_mask = FILE_MASKS[f]
        if not (wp_bb & file_mask):
            attackers = count_bits(br_bb & file_mask) + count_bits(bq_bb & file_mask)
            mg_safety_score += attackers * SEMI_OPEN_FILE_PENALTY[0]
            eg_safety_score += attackers * SEMI_OPEN_FILE_PENALTY[1]
            
    black_king_file = black_king_sq % 8
    for f in range(max(0, black_king_file - 1), min(7, black_king_file + 1) + 1):
        file_mask = FILE_MASKS[f]
        if not (bp_bb & file_mask):
            attackers = count_bits(wr_bb & file_mask) + count_bits(wq_bb & file_mask)
            mg_safety_score -= attackers * SEMI_OPEN_FILE_PENALTY[0]
            eg_safety_score -= attackers * SEMI_OPEN_FILE_PENALTY[1]

    # --- 3. Attacker Proximity Penalty (Corrected and complete version) ---
    white_king_zone = KING_ATTACK_ZONES[white_king_sq]
    black_king_zone = KING_ATTACK_ZONES[black_king_sq]

    # Attackers near White King
    mg_safety_score -= count_bits(bq_bb & white_king_zone) * ATTACKER_WEIGHTS[0][0]
    eg_safety_score -= count_bits(bq_bb & white_king_zone) * ATTACKER_WEIGHTS[0][1]
    mg_safety_score -= count_bits(br_bb & white_king_zone) * ATTACKER_WEIGHTS[1][0]
    eg_safety_score -= count_bits(br_bb & white_king_zone) * ATTACKER_WEIGHTS[1][1]
    mg_safety_score -= count_bits(bb_bb & white_king_zone) * ATTACKER_WEIGHTS[2][0]
    eg_safety_score -= count_bits(bb_bb & white_king_zone) * ATTACKER_WEIGHTS[2][1]
    mg_safety_score -= count_bits(bn_bb & white_king_zone) * ATTACKER_WEIGHTS[3][0]
    eg_safety_score -= count_bits(bn_bb & white_king_zone) * ATTACKER_WEIGHTS[3][1]
    mg_safety_score -= count_bits(bp_bb & white_king_zone) * ATTACKER_WEIGHTS[4][0]
    eg_safety_score -= count_bits(bp_bb & white_king_zone) * ATTACKER_WEIGHTS[4][1]

    # Attackers near Black King
    mg_safety_score += count_bits(wq_bb & black_king_zone) * ATTACKER_WEIGHTS[0][0]
    eg_safety_score += count_bits(wq_bb & black_king_zone) * ATTACKER_WEIGHTS[0][1]
    mg_safety_score += count_bits(wr_bb & black_king_zone) * ATTACKER_WEIGHTS[1][0]
    eg_safety_score += count_bits(wr_bb & black_king_zone) * ATTACKER_WEIGHTS[1][1]
    mg_safety_score += count_bits(wb_bb & black_king_zone) * ATTACKER_WEIGHTS[2][0]
    eg_safety_score += count_bits(wb_bb & black_king_zone) * ATTACKER_WEIGHTS[2][1]
    mg_safety_score += count_bits(wn_bb & black_king_zone) * ATTACKER_WEIGHTS[3][0]
    eg_safety_score += count_bits(wn_bb & black_king_zone) * ATTACKER_WEIGHTS[3][1]
    mg_safety_score += count_bits(wp_bb & black_king_zone) * ATTACKER_WEIGHTS[4][0]
    eg_safety_score += count_bits(wp_bb & black_king_zone) * ATTACKER_WEIGHTS[4][1]

    # --- 4. NEW: King Tropism (Attacker Distance to King) ---
    # White pieces attacking Black King
    white_attackers = (wn_bb, wb_bb, wr_bb, wq_bb)
    tropism_weights = (KNIGHT_TROPISM_WEIGHT, BISHOP_TROPISM_WEIGHT, ROOK_TROPISM_WEIGHT, QUEEN_TROPISM_WEIGHT)
    for i in range(len(white_attackers)):
        bb = white_attackers[i]
        weight = tropism_weights[i]
        while bb:
            sq = get_lsb_index(bb)
            dist = MANHATTAN_DISTANCE[sq, black_king_sq]
            bonus = (MAX_MANHATTAN_DISTANCE - dist)
            mg_safety_score += bonus * weight[0]
            eg_safety_score += bonus * weight[1]
            bb &= bb - np.uint64(1)

    # Black pieces attacking White King
    black_attackers = (bn_bb, bb_bb, br_bb, bq_bb)
    for i in range(len(black_attackers)):
        bb = black_attackers[i]
        weight = tropism_weights[i]
        while bb:
            sq = get_lsb_index(bb)
            dist = MANHATTAN_DISTANCE[sq, white_king_sq]
            bonus = (MAX_MANHATTAN_DISTANCE - dist)
            mg_safety_score -= bonus * weight[0]
            eg_safety_score -= bonus * weight[1]
            bb &= bb - np.uint64(1)

    # --- 5. NEW: King Anti-Mobility Penalty ---
    # White King
    w_king_moves = count_bits(get_queen_attacks(white_king_sq, all_pieces_bb) & ~white_pieces_bb)
    penalty = (w_king_moves - KING_ANTI_MOBILITY_BASE_MOVES)
    mg_safety_score -= penalty * KING_ANTI_MOBILITY_WEIGHT[0]
    eg_safety_score -= penalty * KING_ANTI_MOBILITY_WEIGHT[1]

    # Black King
    b_king_moves = count_bits(get_queen_attacks(black_king_sq, all_pieces_bb) & ~black_pieces_bb)
    penalty = (b_king_moves - KING_ANTI_MOBILITY_BASE_MOVES)
    mg_safety_score += penalty * KING_ANTI_MOBILITY_WEIGHT[0]
    eg_safety_score += penalty * KING_ANTI_MOBILITY_WEIGHT[1]

    return mg_safety_score, eg_safety_score


@numba.njit(numba.int32(piece_bbs_signature, occupancy_bbs_signature, game_state_signature, numba.boolean), cache=True, boundscheck=False, fastmath=True)
def evaluate_position(piece_bbs, occupancy_bbs, game_state, lazy=False):
    """
    使用 Tapered Evaluation (加權評估) 模型評估目前局面，並從當前執棋方的角度返回分數。
    新增了 lazy 參數以支援懶惰評估。
    """
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
        bb = piece_bbs[piece_type + 6]
        while bb:
            sq = get_lsb_index(bb)
            mg_score -= PST_MG[piece_type][sq ^ 56]
            eg_score -= PST_EG[piece_type][sq ^ 56]
            bb &= bb - np.uint64(1)

    # --- 3. 懶惰評估檢查點 ---
    if lazy:
        final_score = (mg_score * phase + eg_score * (MAX_PHASE - phase)) // MAX_PHASE
        return np.int32(final_score) if side_to_move == 0 else np.int32(-final_score)

    # --- 完整評估（非懶惰模式） ---
    mg_king_safety, eg_king_safety = evaluate_king_safety(piece_bbs, occupancy_bbs)
    mg_score += mg_king_safety
    eg_score += eg_king_safety

    mg_mobility, eg_mobility = evaluate_mobility(piece_bbs, occupancy_bbs)
    mg_score += mg_mobility
    eg_score += eg_mobility

    mg_pawn_structure, eg_pawn_structure = evaluate_pawn_structure(piece_bbs)
    mg_score += mg_pawn_structure
    eg_score += eg_pawn_structure

    mg_coord, eg_coord = evaluate_piece_coordination(piece_bbs)
    mg_score += mg_coord
    eg_score += eg_coord

    # --- 最終計算 ---
    final_score = (mg_score * phase + eg_score * (MAX_PHASE - phase)) // MAX_PHASE

    # --- 主動權獎勵 ---
    if phase > (MAX_PHASE * 0.4): # INITIATIVE_PHASE_THRESHOLD_RATIO is a float
        if side_to_move == 0: # White
            final_score += INITIATIVE_BONUS[0]
        else: # Black
            final_score -= INITIATIVE_BONUS[0]

    if side_to_move == 0:
        return np.int32(final_score)
    else:
        return np.int32(-final_score)
