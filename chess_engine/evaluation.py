# chess_engine/evaluation.py

import numba
import numpy as np

from chess_engine.constants import (
    MG_MATERIAL_VALUES, EG_MATERIAL_VALUES,
    PST_MG, PST_EG,
    PHASE_WEIGHTS, MAX_PHASE,
    PAWN_SHIELD_SCORE, SEMI_OPEN_FILE_PENALTY, OPEN_FILE_PENALTY,
    ATTACK_UNITS, SAFETY_TABLE,
    PASSED_PAWN_BONUS, ISOLATED_PAWN_PENALTY, DOUBLED_PAWN_PENALTY,
    BISHOP_PAIR_BONUS, ROOK_ON_SEMI_OPEN_FILE_BONUS, ROOK_ON_OPEN_FILE_BONUS,
    BB_SQUARES
)

from chess_engine.zobrist import get_lsb_index
from chess_engine.engine_types import piece_bbs_signature, occupancy_bbs_signature, game_state_signature
from chess_engine.bitboard_utils import count_bits, WHITE_KING_ZONES, BLACK_KING_ZONES, FILE_MASKS
from chess_engine.move_generator import (
    get_bishop_attacks, get_rook_attacks, get_queen_attacks, KNIGHT_ATTACKS
)
from chess_engine.constants import (
    MOBILITY_BASE_MOVES, MOBILITY_WEIGHT_MG, MOBILITY_WEIGHT_EG,
    CHEBYSHEV_DISTANCE, MAX_CHEBYSHEV_DISTANCE,
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


@numba.njit(numba.int32(numba.uint8, numba.uint64, numba.uint64, numba.boolean), cache=True, boundscheck=False, fastmath=True)
def _evaluate_pawn_shield(king_sq, friendly_pawns, enemy_pawns, is_white):
    """Helper function to evaluate the pawn shield for a single king."""
    safety = np.int32(0)
    king_file = king_sq % 8
    
    start_file = max(0, king_file - 1)
    end_file = min(7, king_file + 1)

    for f in range(start_file, end_file + 1):
        file_mask = FILE_MASKS[f]

        # 1. Score pawn presence and position
        pawns_on_file = friendly_pawns & file_mask
        if not pawns_on_file:
            safety += PAWN_SHIELD_SCORE[3] # Missing pawn
        else:
            pawn_sq = get_lsb_index(pawns_on_file)
            pawn_rank = pawn_sq // 8
            start_rank = 1 if is_white else 6

            if pawn_rank == start_rank:
                safety += PAWN_SHIELD_SCORE[0]
            elif pawn_rank == start_rank + (1 if is_white else -1):
                safety += PAWN_SHIELD_SCORE[1]
            else:
                safety += PAWN_SHIELD_SCORE[2]

        enemy_pawns_on_file = (enemy_pawns & file_mask) != 0
        friendly_pawns_on_file = (friendly_pawns & file_mask) != 0

        if not friendly_pawns_on_file:
            if not enemy_pawns_on_file:
                safety += OPEN_FILE_PENALTY
            else:
                safety += SEMI_OPEN_FILE_PENALTY
        elif not enemy_pawns_on_file:
            # This is a semi-open file for the current player
            pass
    return safety

@numba.njit(numba.types.UniTuple(numba.int32, 2)(piece_bbs_signature, occupancy_bbs_signature), cache=True, boundscheck=False, fastmath=True)
def evaluate_king_safety(piece_bbs, occupancy_bbs):
    """
    Evaluates king safety for both white and black.
    The final score is white_safety - black_safety.
    """
    mg_score = np.int32(0)
    eg_score = np.int32(0)

    wp_bb, bp_bb = piece_bbs[0], piece_bbs[6]
    wk_sq, bk_sq = get_lsb_index(piece_bbs[5]), get_lsb_index(piece_bbs[11])

    white_safety = _evaluate_pawn_shield(wk_sq, wp_bb, bp_bb, True)
    black_safety = _evaluate_pawn_shield(bk_sq, bp_bb, wp_bb, False)

    # --- Attacking King Zone Evaluation ---
    all_pieces_bb = occupancy_bbs[2]

    # White King
    white_attack_units = 0
    white_attacker_count = 0
    white_king_zone = WHITE_KING_ZONES[wk_sq]
    # Black pieces attacking white king zone
    # Knight
    bb = piece_bbs[7]
    while bb:
        sq = get_lsb_index(bb)
        if KNIGHT_ATTACKS[sq] & white_king_zone:
            white_attack_units += ATTACK_UNITS[1]
            white_attacker_count += 1
        bb &= bb - np.uint64(1)
    # Bishop
    bb = piece_bbs[8]
    while bb:
        sq = get_lsb_index(bb)
        if get_bishop_attacks(sq, all_pieces_bb) & white_king_zone:
            white_attack_units += ATTACK_UNITS[2]
            white_attacker_count += 1
        bb &= bb - np.uint64(1)
    # Rook
    bb = piece_bbs[9]
    while bb:
        sq = get_lsb_index(bb)
        if get_rook_attacks(sq, all_pieces_bb) & white_king_zone:
            white_attack_units += ATTACK_UNITS[3]
            white_attacker_count += 1
        bb &= bb - np.uint64(1)
    # Queen
    bb = piece_bbs[10]
    while bb:
        sq = get_lsb_index(bb)
        if get_queen_attacks(sq, all_pieces_bb) & white_king_zone:
            white_attack_units += ATTACK_UNITS[4]
            white_attacker_count += 1
        bb &= bb - np.uint64(1)

    if white_attacker_count >= 2:
         white_safety -= SAFETY_TABLE[min(white_attack_units, 99)]

    # Black King
    black_attack_units = 0
    black_attacker_count = 0
    black_king_zone = BLACK_KING_ZONES[bk_sq]
    # White pieces attacking black king zone
    # Knight
    bb = piece_bbs[1]
    while bb:
        sq = get_lsb_index(bb)
        if KNIGHT_ATTACKS[sq] & black_king_zone:
            black_attack_units += ATTACK_UNITS[1]
            black_attacker_count += 1
        bb &= bb - np.uint64(1)
    # Bishop
    bb = piece_bbs[2]
    while bb:
        sq = get_lsb_index(bb)
        if get_bishop_attacks(sq, all_pieces_bb) & black_king_zone:
            black_attack_units += ATTACK_UNITS[2]
            black_attacker_count += 1
        bb &= bb - np.uint64(1)
    # Rook
    bb = piece_bbs[3]
    while bb:
        sq = get_lsb_index(bb)
        if get_rook_attacks(sq, all_pieces_bb) & black_king_zone:
            black_attack_units += ATTACK_UNITS[3]
            black_attacker_count += 1
        bb &= bb - np.uint64(1)
    # Queen
    bb = piece_bbs[4]
    while bb:
        sq = get_lsb_index(bb)
        if get_queen_attacks(sq, all_pieces_bb) & black_king_zone:
            black_attack_units += ATTACK_UNITS[4]
            black_attacker_count += 1
        bb &= bb - np.uint64(1)

    if black_attacker_count >= 2:
        black_safety -= SAFETY_TABLE[min(black_attack_units, 99)]

    # --- King Tropism Evaluation ---
    # White pieces attacking Black King
    white_attackers = (piece_bbs[1], piece_bbs[2], piece_bbs[3], piece_bbs[4])
    tropism_weights = (KNIGHT_TROPISM_WEIGHT, BISHOP_TROPISM_WEIGHT, ROOK_TROPISM_WEIGHT, QUEEN_TROPISM_WEIGHT)
    for i in range(len(white_attackers)):
        bb = white_attackers[i]
        weight = tropism_weights[i]
        while bb:
            sq = get_lsb_index(bb)
            dist = CHEBYSHEV_DISTANCE[sq, bk_sq]
            penalty = weight[0] * (MAX_CHEBYSHEV_DISTANCE - dist)
            black_safety -= penalty # Lower score is worse for black
            bb &= bb - np.uint64(1)

    # Black pieces attacking White King
    black_attackers = (piece_bbs[7], piece_bbs[8], piece_bbs[9], piece_bbs[10])
    for i in range(len(black_attackers)):
        bb = black_attackers[i]
        weight = tropism_weights[i]
        while bb:
            sq = get_lsb_index(bb)
            dist = CHEBYSHEV_DISTANCE[sq, wk_sq]
            penalty = weight[0] * (MAX_CHEBYSHEV_DISTANCE - dist)
            white_safety -= penalty # Lower score is worse for white
            bb &= bb - np.uint64(1)

    # The final score is a differential.
    # We only apply it to the middlegame score for now.
    mg_score = white_safety - black_safety

    return mg_score, eg_score


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
