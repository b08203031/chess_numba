# chess_engine/classical/evaluation.py

import numba
import numpy as np

from chess_engine.classical.constants import *
from chess_engine.classical.endgame import get_endgame_scale_factor, evaluate_special_endgame

from chess_engine.classical.bitboard_utils import get_lsb_index, get_msb_index, count_bits, WHITE_KING_ZONES, BLACK_KING_ZONES, FILE_MASKS, SQUARES_BETWEEN, ROOK_RAYS, BISHOP_RAYS
from chess_engine.classical.engine_types import (
    piece_bbs_signature, occupancy_bbs_signature, game_state_signature, piece_counts_signature
)
from chess_engine.classical.move_generator import (
    get_bishop_attacks, get_rook_attacks, get_queen_attacks, KNIGHT_ATTACKS, PAWN_ATTACKS, KING_ATTACKS,
    get_pinned_pieces
)
from chess_engine.classical.pawns import (
    NOT_A_FILE, NOT_H_FILE, CENTER_FILES, CHEBYSHEV_DISTANCE, MANHATTAN_DISTANCE,
    WHITE_PASSED_PAWN_MASKS, BLACK_PASSED_PAWN_MASKS,
    WHITE_FORWARD_FILE_MASKS, BLACK_FORWARD_FILE_MASKS,
    WHITE_FORWARD_RANKS, BLACK_FORWARD_RANKS,
    ADJACENT_FILES_MASKS,
    evaluate_pawn_structure, evaluate_pawn_structure_cached,
    evaluate_piece_coordination, evaluate_shelter_aligned
)
from chess_engine.classical.material import compute_imbalance


@numba.njit(cache=True, boundscheck=False, fastmath=True)
def evaluate_passed_pawns(piece_bbs, occupancy_bbs, white_attacks, black_attacks,
                          w_passed, b_passed,
                          white_king_sq, black_king_sq):
    """
    Evaluates passed pawns and candidate passed pawns dynamically.
    Incorporates blockade scaling, Tarrasch rook/queen behind passed pawn support,
    and King proximity logic.
    """
    mg_score = np.int32(0)
    eg_score = np.int32(0)
    passed_count = count_bits(w_passed) + count_bits(b_passed)

    # SF11 passed(): no non-pawn-material scale on the passer bonus.
    # (Former PASSED_SCALE_FULL/HALF/MIN was a local heuristic; removed to align.)

    all_pieces = occupancy_bbs[2]
    white_pawns = piece_bbs[0]
    black_pawns = piece_bbs[6]
    all_pawns = white_pawns | black_pawns

    # --- 1. Iterate White Passed Pawns ---
    temp_wp = w_passed
    while temp_wp:
        sq = get_lsb_index(temp_wp)
        rank = sq >> 3
        file_idx = sq & 7

        p_bonus_mg = PASSED_PAWN_BONUS[rank, 0]
        p_bonus_eg = PASSED_PAWN_BONUS[rank, 1]

        # PassedFile adjustment
        edge_dist = np.int32(min(file_idx, 7 - file_idx))
        p_bonus_mg -= PASSED_FILE_BONUS[0] * edge_dist
        p_bonus_eg -= PASSED_FILE_BONUS[1] * edge_dist

        # Blockade and Path Safety Scaling (SF11 Dynamic Path Safety Bonus)
        block_sq = sq + 8
        block_mask = BB_SQUARES[block_sq]
        is_blocked = (all_pieces & block_mask) != 0

        if not is_blocked and rank > 2:
            # Check for enemy (Black) rooks/queens on the file behind the pawn
            enemy_rq_behind = ((piece_bbs[9] | piece_bbs[10]) & BLACK_FORWARD_FILE_MASKS[sq]) != 0

            # Calculate unsafe_squares (the passed pawn span)
            unsafe_squares = WHITE_PASSED_PAWN_MASKS[sq]
            if not enemy_rq_behind:
                # Restrict to squares attacked by enemy
                unsafe_squares &= black_attacks

            # Calculate path safety weight k (pre-scaled by 3/5 to match centipawn scale)
            squares_to_queen = WHITE_FORWARD_FILE_MASKS[sq]
            if unsafe_squares == np.uint64(0):
                k = PASSED_PATH_SAFE_NONE_ATTACK
            elif (unsafe_squares & squares_to_queen) == np.uint64(0):
                k = PASSED_PATH_SAFE_EDGE_ATTACK
            elif (unsafe_squares & block_mask) == np.uint64(0):
                k = PASSED_PATH_SAFE_BLOCK_ONLY
            else:
                k = np.int32(0)

            # If friendly (White) rook/queen is behind or the block square is defended by White, k += 3
            our_rq_behind = ((piece_bbs[3] | piece_bbs[4]) & BLACK_FORWARD_FILE_MASKS[sq]) != 0
            block_defended = (white_attacks & block_mask) != 0
            if our_rq_behind or block_defended:
                k += PASSED_PATH_SUPPORT_BONUS

            # Calculate dynamic bonus (pre-scaled)
            w = np.int32(PASSED_DYNAMICS_MULT * rank + PASSED_DYNAMICS_OFFSET)
            dynamic_bonus = k * w
            p_bonus_mg += dynamic_bonus
            p_bonus_eg += dynamic_bonus

        # Scale down bonus for candidate passers which need more than one
        # pawn push to become passed, or have a pawn in front of them.
        if rank < 7:
            next_sq = sq + 8
            next_passed = (black_pawns & WHITE_PASSED_PAWN_MASKS[next_sq]) == np.uint64(0)
            any_pawn_ahead = (all_pawns & BB_SQUARES[next_sq]) != np.uint64(0)
            if not next_passed or any_pawn_ahead:
                p_bonus_mg //= CANDIDATE_PASSER_DIVISOR
                p_bonus_eg //= CANDIDATE_PASSER_DIVISOR

        mg_score += p_bonus_mg
        eg_score += p_bonus_eg

        # King Proximity Logic (SF11: EG only; no material scale)
        if block_sq < 64:
            if rank > 3:
                w = np.int32(PASSED_DYNAMICS_MULT * rank + PASSED_DYNAMICS_OFFSET)
                dist_friendly = min(CHEBYSHEV_DISTANCE[white_king_sq, block_sq], KING_PROXIMITY_MAX_DIST)
                dist_enemy = min(CHEBYSHEV_DISTANCE[black_king_sq, block_sq], KING_PROXIMITY_MAX_DIST)

                dist_friendly_2nd = np.int32(0)
                if (block_sq >> 3) != 7:
                    dist_friendly_2nd = min(CHEBYSHEV_DISTANCE[white_king_sq, block_sq + 8], KING_PROXIMITY_MAX_DIST)

                proximity_bonus = ((dist_enemy * KING_PROX_ENEMY_MULT) // KING_PROX_ENEMY_DIV - dist_friendly * KING_PROX_FRIENDLY_MULT) * w
                if (block_sq >> 3) != 7:
                    proximity_bonus -= dist_friendly_2nd * w

                proximity_bonus = max(KING_PROX_MIN_BONUS, min(KING_PROX_MAX_BONUS, proximity_bonus))
                eg_score += proximity_bonus

        temp_wp &= temp_wp - np.uint64(1)

    # --- 2. Iterate Black Passed Pawns ---
    temp_bp = b_passed
    while temp_bp:
        sq = get_lsb_index(temp_bp)
        rank = sq >> 3
        relative_rank = 7 - rank
        file_idx = sq & 7

        p_bonus_mg = PASSED_PAWN_BONUS[relative_rank, 0]
        p_bonus_eg = PASSED_PAWN_BONUS[relative_rank, 1]

        # PassedFile adjustment
        edge_dist = np.int32(min(file_idx, 7 - file_idx))
        p_bonus_mg -= PASSED_FILE_BONUS[0] * edge_dist
        p_bonus_eg -= PASSED_FILE_BONUS[1] * edge_dist

        # Blockade and Path Safety Scaling (SF11 Dynamic Path Safety Bonus)
        block_sq = sq - 8
        block_mask = BB_SQUARES[block_sq]
        is_blocked = (all_pieces & block_mask) != 0

        if not is_blocked and relative_rank > 2:
            # Check for enemy (White) rooks/queens on the file behind the pawn
            enemy_rq_behind = ((piece_bbs[3] | piece_bbs[4]) & WHITE_FORWARD_FILE_MASKS[sq]) != 0

            # Calculate unsafe_squares (the passed pawn span)
            unsafe_squares = BLACK_PASSED_PAWN_MASKS[sq]
            if not enemy_rq_behind:
                # Restrict to squares attacked by enemy
                unsafe_squares &= white_attacks

            # Calculate path safety weight k (pre-scaled by 3/5 to match centipawn scale)
            squares_to_queen = BLACK_FORWARD_FILE_MASKS[sq]
            if unsafe_squares == np.uint64(0):
                k = PASSED_PATH_SAFE_NONE_ATTACK
            elif (unsafe_squares & squares_to_queen) == np.uint64(0):
                k = PASSED_PATH_SAFE_EDGE_ATTACK
            elif (unsafe_squares & block_mask) == np.uint64(0):
                k = PASSED_PATH_SAFE_BLOCK_ONLY
            else:
                k = np.int32(0)

            # If friendly (Black) rook/queen is behind or the block square is defended by Black, k += 3
            our_rq_behind = ((piece_bbs[9] | piece_bbs[10]) & WHITE_FORWARD_FILE_MASKS[sq]) != 0
            block_defended = (black_attacks & block_mask) != 0
            if our_rq_behind or block_defended:
                k += PASSED_PATH_SUPPORT_BONUS

            # Calculate dynamic bonus (pre-scaled)
            w = np.int32(PASSED_DYNAMICS_MULT * relative_rank + PASSED_DYNAMICS_OFFSET)
            dynamic_bonus = k * w
            p_bonus_mg += dynamic_bonus
            p_bonus_eg += dynamic_bonus

        # Scale down bonus for candidate passers which need more than one
        # pawn push to become passed, or have a pawn in front of them.
        if rank > 0:
            next_sq = sq - 8
            next_passed = (white_pawns & BLACK_PASSED_PAWN_MASKS[next_sq]) == np.uint64(0)
            any_pawn_ahead = (all_pawns & BB_SQUARES[next_sq]) != np.uint64(0)
            if not next_passed or any_pawn_ahead:
                p_bonus_mg //= CANDIDATE_PASSER_DIVISOR
                p_bonus_eg //= CANDIDATE_PASSER_DIVISOR

        mg_score -= p_bonus_mg
        eg_score -= p_bonus_eg

        # King Proximity Logic (SF11: EG only; no material scale)
        if block_sq >= 0:
            if relative_rank > 3:
                w = np.int32(PASSED_DYNAMICS_MULT * relative_rank + PASSED_DYNAMICS_OFFSET)
                dist_friendly = min(CHEBYSHEV_DISTANCE[black_king_sq, block_sq], KING_PROXIMITY_MAX_DIST)
                dist_enemy = min(CHEBYSHEV_DISTANCE[white_king_sq, block_sq], KING_PROXIMITY_MAX_DIST)

                dist_friendly_2nd = np.int32(0)
                if (block_sq >> 3) != 0:
                    dist_friendly_2nd = min(CHEBYSHEV_DISTANCE[black_king_sq, block_sq - 8], KING_PROXIMITY_MAX_DIST)

                proximity_bonus = ((dist_enemy * KING_PROX_ENEMY_MULT) // KING_PROX_ENEMY_DIV - dist_friendly * KING_PROX_FRIENDLY_MULT) * w
                if (block_sq >> 3) != 0:
                    proximity_bonus -= dist_friendly_2nd * w

                proximity_bonus = max(KING_PROX_MIN_BONUS, min(KING_PROX_MAX_BONUS, proximity_bonus))
                eg_score -= proximity_bonus

        temp_bp &= temp_bp - np.uint64(1)

    return mg_score, eg_score, passed_count


@numba.njit(cache=True, boundscheck=False, fastmath=True)
def _evaluate_king_attackers(king_sq, color, piece_bbs, occupancy_bbs, enemy_attacks_bb, friendly_attacks_bb, king_zone,
                             attacker_weight, attacker_count, king_attacks_count,
                             en_n_attacks_all, en_b_attacks_all, en_r_attacks_all, en_q_attacks_all,
                             friendly_attacks2, enemy_attacks2, friendly_queen_attacks,
                             pinned_count,
                             flank_attack, flank_defense, mg_mobility_diff, mg_shield,
                             knight_defends):
    """
    SF11 kingDanger structure in OUR ~100cp-consistent danger space.
    Pure attack coeffs are pre-scaled by 100/128; mobility_diff and mg_shield
    must already be pure mobility / shelter scores in our MG units.
    """
    friendly_start_idx = 0 if color == 0 else 6
    all_pieces_occupancy = occupancy_bbs[2]
    friendly_queen = piece_bbs[friendly_start_idx + 4]
    occ_xray = all_pieces_occupancy ^ friendly_queen

    rook_check_rays = get_rook_attacks(king_sq, occ_xray)
    bishop_check_rays = get_bishop_attacks(king_sq, occ_xray)

    # weak: Attacked by enemy, not defended twice by us, and either undefended or defended by King/Queen
    weak = enemy_attacks_bb & ~friendly_attacks2 & (~friendly_attacks_bb | KING_ATTACKS[king_sq] | friendly_queen_attacks)

    # safe: Not occupied by enemy, and either undefended by us or defended twice by enemy on a weak square
    enemy_occupancy = occupancy_bbs[1] if color == 0 else occupancy_bbs[0]
    safe = ~enemy_occupancy & (~friendly_attacks_bb | (weak & enemy_attacks2))

    safe_check_units = np.int32(0)
    unsafe_checks = np.uint64(0)

    rook_safe_checks = rook_check_rays & safe & en_r_attacks_all
    if rook_safe_checks != np.uint64(0):
        safe_check_units += SAFE_CHECK_ROOK
    else:
        unsafe_checks |= rook_check_rays & en_r_attacks_all

    queen_safe_checks = (rook_check_rays | bishop_check_rays) & safe & en_q_attacks_all & ~friendly_queen_attacks & ~rook_safe_checks
    if queen_safe_checks != np.uint64(0):
        safe_check_units += SAFE_CHECK_QUEEN

    bishop_safe_checks = bishop_check_rays & safe & en_b_attacks_all & ~queen_safe_checks
    if bishop_safe_checks != np.uint64(0):
        safe_check_units += SAFE_CHECK_BISHOP
    else:
        unsafe_checks |= bishop_check_rays & en_b_attacks_all

    knight_check_squares = KNIGHT_ATTACKS[king_sq] & en_n_attacks_all
    if (knight_check_squares & safe) != np.uint64(0):
        safe_check_units += SAFE_CHECK_KNIGHT
    else:
        unsafe_checks |= knight_check_squares

    weak_sq_count = count_bits(king_zone & weak)
    unsafe_check_count = count_bits(unsafe_checks)

    # Popcount-only flank terms: scale into same space as other coeffs (100/128)
    flank_term = (KING_FLANK_ATTACK_NUM * flank_attack * flank_attack) // KING_FLANK_ATTACK_DEN
    flank_term = (flank_term * np.int32(100)) // np.int32(128)
    flank_def_term = (KING_FLANK_DEFENSE_MULT * flank_defense * np.int32(100)) // np.int32(128)

    # Product model + linear terms (all in our-scaled danger / score space)
    kingDanger = (
          attacker_count * attacker_weight
        + safe_check_units
        + KING_DANGER_WEAK_SQ * weak_sq_count
        + KING_DANGER_UNSAFE_CHECK * unsafe_check_count
        + KING_DANGER_ATTACK_ON_KING_SQ * king_attacks_count
        + KING_DANGER_BLOCKERS * pinned_count
        + flank_term
        + mg_mobility_diff
        - (KING_SHELTER_FEEDBACK_NUM * mg_shield) // KING_SHELTER_FEEDBACK_DEN
        - flank_def_term
        + KING_DANGER_OFFSET
    )

    if knight_defends:
        kingDanger -= KING_DANGER_KNIGHT_DEF

    enemy_start_idx = 6 if color == 0 else 0
    if not piece_bbs[enemy_start_idx + 4]:
        kingDanger -= KING_DANGER_NO_QUEEN

    return kingDanger


@numba.njit(cache=True, boundscheck=False, fastmath=True)
def evaluate_king_safety(piece_bbs, occupancy_bbs, game_state, white_attacks, black_attacks, 
                         piece_counts, white_attacks2, black_attacks2, pinned_white, pinned_black, 
                         white_mg_shield, white_eg_shield, black_mg_shield, black_eg_shield,
                         white_king_sq, black_king_sq,
                         w_attacker_weight, w_attacker_count, w_king_attacks_count,
                         b_attacker_weight, b_attacker_count, b_king_attacks_count,
                         white_knight_attacks, white_bishop_attacks, white_rook_attacks, white_queen_attacks,
                         black_knight_attacks, black_bishop_attacks, black_rook_attacks, black_queen_attacks,
                         mg_mobility_pure,
                         white_king_zone, black_king_zone):
    """
    King Safety (SF11 structure, our score space):
    shelter/storm + scale-consistent kingDanger + flank extras.
    mg_mobility_pure = only MobilityBonus tables (W-B), not protector/extra piece features.
    """
    (wp_bb, wn_bb, wb_bb, wr_bb, wq_bb, wk_bb,
    bp_bb, bn_bb, bb_bb, br_bb, bq_bb, bk_bb) = piece_bbs

    # Flank Attack & Flank Defense
    CAMP_WHITE = np.uint64(0x000000FFFFFFFFFF)
    CAMP_BLACK = np.uint64(0xFFFFFFFFFF000000)

    white_king_file = white_king_sq % 8
    white_flank_bb = KING_FLANKS[white_king_file]
    w_b1 = black_attacks & white_flank_bb & CAMP_WHITE
    w_b2 = w_b1 & black_attacks2
    w_flank_attack = count_bits(w_b1) + count_bits(w_b2)
    w_flank_defense = count_bits(white_attacks & white_flank_bb & CAMP_WHITE)

    black_king_file = black_king_sq % 8
    black_flank_bb = KING_FLANKS[black_king_file]
    b_b1 = white_attacks & black_flank_bb & CAMP_BLACK
    b_b2 = b_b1 & white_attacks2
    b_flank_attack = count_bits(b_b1) + count_bits(b_b2)
    b_flank_defense = count_bits(black_attacks & black_flank_bb & CAMP_BLACK)

    w_knight_def = (white_knight_attacks & KING_ATTACKS[white_king_sq]) != np.uint64(0)
    b_knight_def = (black_knight_attacks & KING_ATTACKS[black_king_sq]) != np.uint64(0)

    # Pure mobility them-us (W-B): white king sees black-white = -mg_mobility_pure
    w_danger = _evaluate_king_attackers(
        white_king_sq, 0, piece_bbs, occupancy_bbs, black_attacks, white_attacks, white_king_zone,
        w_attacker_weight, w_attacker_count, w_king_attacks_count,
        black_knight_attacks, black_bishop_attacks, black_rook_attacks, black_queen_attacks,
        white_attacks2, black_attacks2, white_queen_attacks,
        np.int32(count_bits(pinned_white)),
        np.int32(w_flank_attack), np.int32(w_flank_defense),
        np.int32(-mg_mobility_pure), np.int32(white_mg_shield),
        w_knight_def,
    )
    b_danger = _evaluate_king_attackers(
        black_king_sq, 1, piece_bbs, occupancy_bbs, white_attacks, black_attacks, black_king_zone,
        b_attacker_weight, b_attacker_count, b_king_attacks_count,
        white_knight_attacks, white_bishop_attacks, white_rook_attacks, white_queen_attacks,
        black_attacks2, white_attacks2, black_queen_attacks,
        np.int32(count_bits(pinned_black)),
        np.int32(b_flank_attack), np.int32(b_flank_defense),
        np.int32(mg_mobility_pure), np.int32(black_mg_shield),
        b_knight_def,
    )

    # kd in our-scaled space: pen_mg = kd²*128/(4096*100), pen_eg = kd*128*120/(100*16*213)
    w_penalty_mg = np.int32(0)
    w_penalty_eg = np.int32(0)
    if w_danger > KING_DANGER_THRESHOLD:
        w_penalty_mg = (w_danger * w_danger * KING_DANGER_OUT_MG_NUM) // KING_DANGER_OUT_MG_DEN
        w_penalty_eg = (w_danger * KING_DANGER_OUT_EG_NUM) // KING_DANGER_OUT_EG_DEN

    b_penalty_mg = np.int32(0)
    b_penalty_eg = np.int32(0)
    if b_danger > KING_DANGER_THRESHOLD:
        b_penalty_mg = (b_danger * b_danger * KING_DANGER_OUT_MG_NUM) // KING_DANGER_OUT_MG_DEN
        b_penalty_eg = (b_danger * KING_DANGER_OUT_EG_NUM) // KING_DANGER_OUT_EG_DEN

    # PawnlessFlank + FlankAttacks (score terms, already in our scale)
    w_extra_mg = np.int32(0)
    w_extra_eg = np.int32(0)
    if ((wp_bb | bp_bb) & white_flank_bb) == np.uint64(0):
        w_extra_mg += PAWNLESS_FLANK[0]
        w_extra_eg += PAWNLESS_FLANK[1]
    w_extra_mg += FLANK_ATTACKS[0] * w_flank_attack
    w_extra_eg += FLANK_ATTACKS[1] * w_flank_attack

    b_extra_mg = np.int32(0)
    b_extra_eg = np.int32(0)
    if ((wp_bb | bp_bb) & black_flank_bb) == np.uint64(0):
        b_extra_mg += PAWNLESS_FLANK[0]
        b_extra_eg += PAWNLESS_FLANK[1]
    b_extra_mg += FLANK_ATTACKS[0] * b_flank_attack
    b_extra_eg += FLANK_ATTACKS[1] * b_flank_attack

    white_mg_score = white_mg_shield + w_extra_mg - w_penalty_mg
    white_eg_score = white_eg_shield + w_extra_eg - w_penalty_eg
    black_mg_score = black_mg_shield + b_extra_mg - b_penalty_mg
    black_eg_score = black_eg_shield + b_extra_eg - b_penalty_eg

    mg_safety_score = white_mg_score - black_mg_score
    eg_safety_score = np.int32((white_eg_score - black_eg_score) * EG_SAFETY_SCALE)

    # --- 6. Endgame King Proximity to Own Pawns ---
    # White King Proximity
    white_pawn_dist_penalty = np.int32(0)
    if wp_bb != 0:
        min_dist = 8
        if (wp_bb & KING_ATTACKS[white_king_sq]) != 0:
            min_dist = 1
        else:
            temp_pawns = wp_bb
            while temp_pawns:
                p_sq = get_lsb_index(temp_pawns)
                dist = CHEBYSHEV_DISTANCE[white_king_sq, p_sq]
                if dist < min_dist:
                    min_dist = dist
                temp_pawns &= temp_pawns - np.uint64(1)
        white_pawn_dist_penalty = KING_PAWN_DIST_PENALTY_EG * min_dist

    # Black King Proximity
    black_pawn_dist_penalty = np.int32(0)
    if bp_bb != 0:
        min_dist = 8
        if (bp_bb & KING_ATTACKS[black_king_sq]) != 0:
            min_dist = 1
        else:
            temp_pawns = bp_bb
            while temp_pawns:
                p_sq = get_lsb_index(temp_pawns)
                dist = CHEBYSHEV_DISTANCE[black_king_sq, p_sq]
                if dist < min_dist:
                    min_dist = dist
                temp_pawns &= temp_pawns - np.uint64(1)
        black_pawn_dist_penalty = KING_PAWN_DIST_PENALTY_EG * min_dist

    eg_safety_score += white_pawn_dist_penalty - black_pawn_dist_penalty

    return mg_safety_score, eg_safety_score


@numba.njit(cache=True, boundscheck=False, fastmath=True)
def evaluate_attacks_mobility_threats(piece_bbs, occupancy_bbs, white_king_sq, black_king_sq, white_pawn_attacks_span, black_pawn_attacks_span, castling_rights):
    (wp_bb, wn_bb, wb_bb, wr_bb, wq_bb, wk_bb,
     bp_bb, bn_bb, bb_bb, br_bb, bq_bb, bk_bb) = piece_bbs

    white_occupancy = occupancy_bbs[0]
    black_occupancy = occupancy_bbs[1]
    all_occupancy = occupancy_bbs[2]

    # --- Pawn Attacks ---
    white_pawn_attacks = ((wp_bb & NOT_A_FILE) << np.uint64(7)) | ((wp_bb & NOT_H_FILE) << np.uint64(9))
    black_pawn_attacks = ((bp_bb & NOT_H_FILE) >> np.uint64(7)) | ((bp_bb & NOT_A_FILE) >> np.uint64(9))

    # --- Outpost ranks (SF11: relative ranks 4-6 only) ---
    # White: ranks 4-6 (indices 3-5); Black: absolute ranks 3-5 (indices 2-4)
    WHITE_OUTPOST_RANKS = np.uint64(0x0000FFFFFF000000)
    BLACK_OUTPOST_RANKS = np.uint64(0x000000FFFFFF0000)
    white_outposts = WHITE_OUTPOST_RANKS & white_pawn_attacks & ~black_pawn_attacks_span
    black_outposts = BLACK_OUTPOST_RANKS & black_pawn_attacks & ~white_pawn_attacks_span

    # X-Ray occupancy mask precalculations (Loop invariant optimizations)
    bishop_xray_occ = all_occupancy ^ wq_bb ^ bq_bb
    white_rook_xray_occ = bishop_xray_occ ^ wr_bb
    black_rook_xray_occ = bishop_xray_occ ^ br_bb

    white_attacks = white_pawn_attacks | KING_ATTACKS[white_king_sq]
    black_attacks = black_pawn_attacks | KING_ATTACKS[black_king_sq]

    # --- Mobility Area (SF11-inspired) ---
    # Exclude: enemy pawn attacks, blocked low pawns, friendly queen, friendly king
    LOW_RANKS_WHITE = np.uint64(0x0000000000FFFF00)  # rank2 | rank3
    LOW_RANKS_BLACK = np.uint64(0x00FFFF0000000000)  # rank6 | rank7
    # Blocked low rank pawns: friendly pawns that are blocked or on the first two ranks (SF11-aligned)
    white_blocked_low = wp_bb & ((all_occupancy >> np.uint64(8)) | LOW_RANKS_WHITE)
    black_blocked_low = bp_bb & ((all_occupancy << np.uint64(8)) | LOW_RANKS_BLACK)

    # --- Pin Calculation (must precede mobility area) ---
    pinned_white = get_pinned_pieces(piece_bbs, occupancy_bbs, np.uint64(0))  # WHITE
    pinned_black = get_pinned_pieces(piece_bbs, occupancy_bbs, np.uint64(1))  # BLACK

    white_mobility_area = ~(black_pawn_attacks | white_blocked_low | wq_bb | wk_bb | pinned_white)
    black_mobility_area = ~(white_pawn_attacks | black_blocked_low | bq_bb | bk_bb | pinned_black)

    # --- Accumulators ---
    # mg_mobility / eg_mobility: full piece-feature bag (used in total eval)
    # mg_mobility_pure: ONLY MobilityBonus tables (W-B) for kingDanger (SF-compatible)
    mg_mobility = np.int32(0)
    eg_mobility = np.int32(0)
    mg_mobility_pure = np.int32(0)
    
    white_knight_attacks = np.uint64(0)
    white_bishop_attacks = np.uint64(0)
    white_rook_attacks = np.uint64(0)
    white_queen_attacks = np.uint64(0)
    white_minor_attacks = np.uint64(0)   # N+B combined (kept for king safety)
    
    # Initialize attackedBy2 with double pawn attacks and king-pawn overlaps
    white_double_pawn = ((wp_bb & NOT_A_FILE) << np.uint64(7)) & ((wp_bb & NOT_H_FILE) << np.uint64(9))
    white_attacks2 = white_double_pawn | (KING_ATTACKS[white_king_sq] & white_pawn_attacks)
    
    black_knight_attacks = np.uint64(0)
    black_bishop_attacks = np.uint64(0)
    black_rook_attacks = np.uint64(0)
    black_queen_attacks = np.uint64(0)
    black_minor_attacks = np.uint64(0)   # N+B combined (kept for king safety)
    
    black_double_pawn = ((bp_bb & NOT_H_FILE) >> np.uint64(7)) & ((bp_bb & NOT_A_FILE) >> np.uint64(9))
    black_attacks2 = black_double_pawn | (KING_ATTACKS[black_king_sq] & black_pawn_attacks)

    # King safety: SF11 kingRing strips squares doubly defended by our pawns
    white_king_adjacent = KING_ATTACKS[white_king_sq]
    black_king_adjacent = KING_ATTACKS[black_king_sq]
    white_king_zone = WHITE_KING_ZONES[white_king_sq] & ~white_double_pawn
    black_king_zone = BLACK_KING_ZONES[black_king_sq] & ~black_double_pawn

    # Pawn hits seed attackersCount (weight remains 0 until non-pawn attackers)
    w_attacker_count = np.int32(count_bits(white_king_zone & black_pawn_attacks))
    b_attacker_count = np.int32(count_bits(black_king_zone & white_pawn_attacks))
    w_attacker_weight = np.int32(0)
    b_attacker_weight = np.int32(0)
    w_king_attacks_count = np.int32(0)
    b_king_attacks_count = np.int32(0)

    # --- Outpost Accumulators ---
    mg_outpost = np.int32(0)
    eg_outpost = np.int32(0)

    temp_bb = wn_bb
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        att = KNIGHT_ATTACKS[sq]
        if pinned_white & (np.uint64(1) << np.uint64(sq)):
            att = np.uint64(0)
        white_attacks2 |= (white_attacks & att)
        white_attacks |= att
        white_knight_attacks |= att
        white_minor_attacks |= att
        moves = count_bits(att & white_mobility_area)
        clamped = min(moves, 8)
        mg_mobility += KNIGHT_MOBILITY_BONUS[clamped, 0]
        eg_mobility += KNIGHT_MOBILITY_BONUS[clamped, 1]
        mg_mobility_pure += KNIGHT_MOBILITY_BONUS[clamped, 0]
        
        # KingProtector: penalty per Chebyshev distance from own king
        mg_mobility -= KING_PROTECTOR[0] * CHEBYSHEV_DISTANCE[sq, white_king_sq]
        eg_mobility -= KING_PROTECTOR[1] * CHEBYSHEV_DISTANCE[sq, white_king_sq]

        # MinorBehindPawn: bonus for knight behind friendly pawn
        if sq < 56 and (wp_bb & BB_SQUARES[sq + 8]) != np.uint64(0):
            mg_mobility += MINOR_BEHIND_PAWN[0]
            eg_mobility += MINOR_BEHIND_PAWN[1]

        
        # King safety: White Knight attacking Black King ring
        if att & black_king_zone:
            b_attacker_count += 1
            b_attacker_weight += KING_ATTACK_WEIGHTS[2]  # KNIGHT
        b_king_attacks_count += count_bits(att & black_king_adjacent)

        # Outpost Logic (White Knight) — SF11: OutpostRanks & pawn-supported & ~enemy span
        is_outpost = (BB_SQUARES[sq] & white_outposts) != np.uint64(0)
        if is_outpost:
            rank = sq // 8
            mg_outpost += OUTPOST_BONUS_KNIGHT[rank, 0]
            eg_outpost += OUTPOST_BONUS_KNIGHT[rank, 1]
        elif (att & white_outposts & ~white_occupancy) != np.uint64(0):
            mg_outpost += REACHABLE_OUTPOST_BONUS[0]
            eg_outpost += REACHABLE_OUTPOST_BONUS[1]

        temp_bb &= temp_bb - np.uint64(1)

    temp_bb = wb_bb
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        # X-Ray: bishop attacks through all queens
        att = get_bishop_attacks(sq, bishop_xray_occ)
        if pinned_white & (np.uint64(1) << np.uint64(sq)):
            sq_bb = np.uint64(1) << np.uint64(sq)
            if sq_bb & BISHOP_RAYS[white_king_sq]:
                att &= BISHOP_RAYS[white_king_sq]
            else:
                att = np.uint64(0)
        white_attacks2 |= (white_attacks & att)
        white_attacks |= att
        white_bishop_attacks |= att
        white_minor_attacks |= att
        moves = count_bits(att & white_mobility_area)
        clamped = min(moves, 13)
        mg_mobility += BISHOP_MOBILITY_BONUS[clamped, 0]
        eg_mobility += BISHOP_MOBILITY_BONUS[clamped, 1]
        mg_mobility_pure += BISHOP_MOBILITY_BONUS[clamped, 0]
        
        # KingProtector: penalty per Chebyshev distance from own king
        mg_mobility -= KING_PROTECTOR[0] * CHEBYSHEV_DISTANCE[sq, white_king_sq]
        eg_mobility -= KING_PROTECTOR[1] * CHEBYSHEV_DISTANCE[sq, white_king_sq]

        # MinorBehindPawn: bonus for bishop behind friendly pawn
        if sq < 56 and (wp_bb & BB_SQUARES[sq + 8]) != np.uint64(0):
            mg_mobility += MINOR_BEHIND_PAWN[0]
            eg_mobility += MINOR_BEHIND_PAWN[1]

        # BishopPawns: penalty per own pawn on same color square as bishop
        # Square color: (rank + file) % 2, where rank = sq//8, file = sq%8
        bishop_color = (sq // 8 + sq % 8) % 2
        bishop_color_mask = np.uint64(0xAA55AA55AA55AA55) if bishop_color == 1 else np.uint64(0x55AA55AA55AA55AA)
        same_color_pawns = count_bits(wp_bb & bishop_color_mask)
        blocked_w = wp_bb & (all_occupancy >> np.uint64(8))
        center_blocked = count_bits(blocked_w & CENTER_FILES)
        multiplier = np.int32(1) + BISHOP_PAWNS_CENTER_BLOCKED_FACTOR * center_blocked
        mg_mobility -= BISHOP_PAWNS_PENALTY[0] * same_color_pawns * multiplier
        eg_mobility -= BISHOP_PAWNS_PENALTY[1] * same_color_pawns * multiplier
        

        # King safety calculation (White Bishop attacking Black King)
        if att & black_king_zone:
            b_attacker_count += 1
            b_attacker_weight += KING_ATTACK_WEIGHTS[3]  # BISHOP
        b_king_attacks_count += count_bits(att & black_king_adjacent)

        # Outpost Logic (White Bishop) — SF11: same mask as knight, no separate hole bonus
        is_outpost = (BB_SQUARES[sq] & white_outposts) != np.uint64(0)
        if is_outpost:
            rank = sq // 8
            mg_outpost += OUTPOST_BONUS_BISHOP[rank, 0]
            eg_outpost += OUTPOST_BONUS_BISHOP[rank, 1]

        # LongDiagonalBishop: bonus if bishop sees both center squares through pawns only
        CENTER_SQUARES = np.uint64(0x0000001818000000)  # d4, e4, d5, e5
        pawn_only_occ = wp_bb | bp_bb
        bishop_center_attacks = get_bishop_attacks(sq, pawn_only_occ) & CENTER_SQUARES
        if count_bits(bishop_center_attacks) >= 2:
            mg_mobility += LONG_DIAGONAL_BISHOP[0]
            eg_mobility += LONG_DIAGONAL_BISHOP[1]

        temp_bb &= temp_bb - np.uint64(1)

    temp_bb = wr_bb
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        # X-Ray: rook attacks through all queens + own rooks
        att = get_rook_attacks(sq, white_rook_xray_occ)
        if pinned_white & (np.uint64(1) << np.uint64(sq)):
            sq_bb = np.uint64(1) << np.uint64(sq)
            if sq_bb & ROOK_RAYS[white_king_sq]:
                att &= ROOK_RAYS[white_king_sq]
            else:
                att = np.uint64(0)
        white_attacks2 |= (white_attacks & att)
        white_attacks |= att
        white_rook_attacks |= att
        moves = count_bits(att & white_mobility_area)
        clamped = min(moves, 14)
        mg_mobility += ROOK_MOBILITY_BONUS[clamped, 0]
        eg_mobility += ROOK_MOBILITY_BONUS[clamped, 1]
        mg_mobility_pure += ROOK_MOBILITY_BONUS[clamped, 0]
        
        # TrappedRook: SF11 only when NOT on semi-open file (friendly pawn on file)
        rook_file = sq % 8
        if moves <= 3 and (wp_bb & FILE_MASKS[rook_file]) != np.uint64(0):
            king_file = white_king_sq % 8
            if (king_file < 4) == (rook_file < king_file):
                trap_factor = np.int32(1) if (castling_rights & np.uint8(3)) != np.uint8(0) else np.int32(2)
                mg_mobility -= TRAPPED_ROOK[0] * trap_factor
                eg_mobility -= TRAPPED_ROOK[1] * trap_factor

        # RookOnQueenFile: bonus for rook on same file as any queen
        if (wq_bb | bq_bb) & FILE_MASKS[rook_file]:
            mg_mobility += ROOK_ON_QUEEN_FILE[0]
            eg_mobility += ROOK_ON_QUEEN_FILE[1]
        

        # King safety calculation (White Rook attacking Black King)
        if att & black_king_zone:
            b_attacker_count += 1
            b_attacker_weight += KING_ATTACK_WEIGHTS[4]  # ROOK
        b_king_attacks_count += count_bits(att & black_king_adjacent)

        temp_bb &= temp_bb - np.uint64(1)

    temp_bb = wq_bb
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        att = get_queen_attacks(sq, all_occupancy)
        if pinned_white & (np.uint64(1) << np.uint64(sq)):
            sq_bb = np.uint64(1) << np.uint64(sq)
            if sq_bb & ROOK_RAYS[white_king_sq]:
                att &= ROOK_RAYS[white_king_sq]
            elif sq_bb & BISHOP_RAYS[white_king_sq]:
                att &= BISHOP_RAYS[white_king_sq]
            else:
                att = np.uint64(0)
        white_attacks2 |= (white_attacks & att)
        white_attacks |= att
        white_queen_attacks |= att
        moves = count_bits(att & white_mobility_area)
        clamped = min(moves, 27)
        mg_mobility += QUEEN_MOBILITY_BONUS[clamped, 0]
        eg_mobility += QUEEN_MOBILITY_BONUS[clamped, 1]
        mg_mobility_pure += QUEEN_MOBILITY_BONUS[clamped, 0]
        

        # King safety calculation (White Queen attacking Black King)
        if att & black_king_zone:
            b_attacker_count += 1
            b_attacker_weight += KING_ATTACK_WEIGHTS[5]  # QUEEN
        b_king_attacks_count += count_bits(att & black_king_adjacent)

        # WeakQueen: penalty if queen is in a relative pin / discovered attack line
        rook_pinners = ROOK_RAYS[sq] & br_bb
        temp_pinners = rook_pinners
        is_weak = False
        while temp_pinners and not is_weak:
            pinner_sq = get_lsb_index(temp_pinners)
            between = SQUARES_BETWEEN[sq, pinner_sq]
            blockers_between = between & all_occupancy
            if count_bits(blockers_between) == 1:
                is_weak = True
            temp_pinners &= temp_pinners - np.uint64(1)
        if not is_weak:
            bishop_pinners = BISHOP_RAYS[sq] & bb_bb
            temp_pinners = bishop_pinners
            while temp_pinners and not is_weak:
                pinner_sq = get_lsb_index(temp_pinners)
                between = SQUARES_BETWEEN[sq, pinner_sq]
                blockers_between = between & all_occupancy
                if count_bits(blockers_between) == 1:
                    is_weak = True
                temp_pinners &= temp_pinners - np.uint64(1)
        if is_weak:
            mg_mobility -= WEAK_QUEEN[0]
            eg_mobility -= WEAK_QUEEN[1]

        temp_bb &= temp_bb - np.uint64(1)

    # --- Black Pieces ---
    temp_bb = bn_bb
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        att = KNIGHT_ATTACKS[sq]
        if pinned_black & (np.uint64(1) << np.uint64(sq)):
            att = np.uint64(0)
        black_attacks2 |= (black_attacks & att)
        black_attacks |= att
        black_knight_attacks |= att
        black_minor_attacks |= att
        moves = count_bits(att & black_mobility_area)
        clamped = min(moves, 8)
        mg_mobility -= KNIGHT_MOBILITY_BONUS[clamped, 0]
        eg_mobility -= KNIGHT_MOBILITY_BONUS[clamped, 1]
        mg_mobility_pure -= KNIGHT_MOBILITY_BONUS[clamped, 0]
        
        # KingProtector: penalty per Chebyshev distance from own king
        mg_mobility += KING_PROTECTOR[0] * CHEBYSHEV_DISTANCE[sq, black_king_sq]
        eg_mobility += KING_PROTECTOR[1] * CHEBYSHEV_DISTANCE[sq, black_king_sq]

        # MinorBehindPawn: bonus for knight behind friendly pawn
        if sq >= 8 and (bp_bb & BB_SQUARES[sq - 8]) != np.uint64(0):
            mg_mobility -= MINOR_BEHIND_PAWN[0]
            eg_mobility -= MINOR_BEHIND_PAWN[1]


        # King safety calculation (Black Knight attacking White King)
        if att & white_king_zone:
            w_attacker_count += 1
            w_attacker_weight += KING_ATTACK_WEIGHTS[2]  # KNIGHT
        w_king_attacks_count += count_bits(att & white_king_adjacent)

        # Outpost Logic (Black Knight)
        is_outpost = (BB_SQUARES[sq] & black_outposts) != np.uint64(0)
        if is_outpost:
            rank = sq // 8
            rel_rank = 7 - rank
            mg_outpost -= OUTPOST_BONUS_KNIGHT[rel_rank, 0]
            eg_outpost -= OUTPOST_BONUS_KNIGHT[rel_rank, 1]
        elif (att & black_outposts & ~black_occupancy) != np.uint64(0):
            mg_outpost -= REACHABLE_OUTPOST_BONUS[0]
            eg_outpost -= REACHABLE_OUTPOST_BONUS[1]

        temp_bb &= temp_bb - np.uint64(1)

    temp_bb = bb_bb
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        # X-Ray: bishop attacks through all queens
        att = get_bishop_attacks(sq, bishop_xray_occ)
        if pinned_black & (np.uint64(1) << np.uint64(sq)):
            sq_bb = np.uint64(1) << np.uint64(sq)
            if sq_bb & BISHOP_RAYS[black_king_sq]:
                att &= BISHOP_RAYS[black_king_sq]
            else:
                att = np.uint64(0)
        black_attacks2 |= (black_attacks & att)
        black_attacks |= att
        black_bishop_attacks |= att
        black_minor_attacks |= att
        moves = count_bits(att & black_mobility_area)
        clamped = min(moves, 13)
        mg_mobility -= BISHOP_MOBILITY_BONUS[clamped, 0]
        eg_mobility -= BISHOP_MOBILITY_BONUS[clamped, 1]
        mg_mobility_pure -= BISHOP_MOBILITY_BONUS[clamped, 0]
        
        # KingProtector: penalty per Chebyshev distance from own king
        mg_mobility += KING_PROTECTOR[0] * CHEBYSHEV_DISTANCE[sq, black_king_sq]
        eg_mobility += KING_PROTECTOR[1] * CHEBYSHEV_DISTANCE[sq, black_king_sq]

        # MinorBehindPawn: bonus for bishop behind friendly pawn
        if sq >= 8 and (bp_bb & BB_SQUARES[sq - 8]) != np.uint64(0):
            mg_mobility -= MINOR_BEHIND_PAWN[0]
            eg_mobility -= MINOR_BEHIND_PAWN[1]

        # BishopPawns: penalty per own pawn on same color square as bishop
        # Square color: (rank + file) % 2, where rank = sq//8, file = sq%8
        bishop_color = (sq // 8 + sq % 8) % 2
        bishop_color_mask = np.uint64(0xAA55AA55AA55AA55) if bishop_color == 1 else np.uint64(0x55AA55AA55AA55AA)
        same_color_pawns = count_bits(bp_bb & bishop_color_mask)
        blocked_b = bp_bb & (all_occupancy << np.uint64(8))
        center_blocked = count_bits(blocked_b & CENTER_FILES)
        multiplier = np.int32(1) + BISHOP_PAWNS_CENTER_BLOCKED_FACTOR * center_blocked
        mg_mobility += BISHOP_PAWNS_PENALTY[0] * same_color_pawns * multiplier
        eg_mobility += BISHOP_PAWNS_PENALTY[1] * same_color_pawns * multiplier
        

        # King safety calculation (Black Bishop attacking White King)
        if att & white_king_zone:
            w_attacker_count += 1
            w_attacker_weight += KING_ATTACK_WEIGHTS[3]  # BISHOP
        w_king_attacks_count += count_bits(att & white_king_adjacent)

        # Outpost Logic (Black Bishop) — SF11: no separate hole bonus
        is_outpost = (BB_SQUARES[sq] & black_outposts) != np.uint64(0)
        if is_outpost:
            rank = sq // 8
            rel_rank = 7 - rank
            mg_outpost -= OUTPOST_BONUS_BISHOP[rel_rank, 0]
            eg_outpost -= OUTPOST_BONUS_BISHOP[rel_rank, 1]

        # LongDiagonalBishop: bonus if bishop sees both center squares through pawns only
        CENTER_SQUARES = np.uint64(0x0000001818000000)  # d4, e4, d5, e5
        pawn_only_occ = wp_bb | bp_bb
        bishop_center_attacks = get_bishop_attacks(sq, pawn_only_occ) & CENTER_SQUARES
        if count_bits(bishop_center_attacks) >= 2:
            mg_mobility -= LONG_DIAGONAL_BISHOP[0]
            eg_mobility -= LONG_DIAGONAL_BISHOP[1]

        temp_bb &= temp_bb - np.uint64(1)

    temp_bb = br_bb
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        # X-Ray: rook attacks through all queens + own rooks
        att = get_rook_attacks(sq, black_rook_xray_occ)
        if pinned_black & (np.uint64(1) << np.uint64(sq)):
            sq_bb = np.uint64(1) << np.uint64(sq)
            if sq_bb & ROOK_RAYS[black_king_sq]:
                att &= ROOK_RAYS[black_king_sq]
            else:
                att = np.uint64(0)
        black_attacks2 |= (black_attacks & att)
        black_attacks |= att
        black_rook_attacks |= att
        moves = count_bits(att & black_mobility_area)
        clamped = min(moves, 14)
        mg_mobility -= ROOK_MOBILITY_BONUS[clamped, 0]
        eg_mobility -= ROOK_MOBILITY_BONUS[clamped, 1]
        mg_mobility_pure -= ROOK_MOBILITY_BONUS[clamped, 0]
        
        # TrappedRook: SF11 only when NOT on semi-open file (friendly pawn on file)
        rook_file = sq % 8
        if moves <= 3 and (bp_bb & FILE_MASKS[rook_file]) != np.uint64(0):
            king_file = black_king_sq % 8
            if (king_file < 4) == (rook_file < king_file):
                trap_factor = np.int32(1) if (castling_rights & np.uint8(12)) != np.uint8(0) else np.int32(2)
                mg_mobility += TRAPPED_ROOK[0] * trap_factor
                eg_mobility += TRAPPED_ROOK[1] * trap_factor

        # RookOnQueenFile: bonus for rook on same file as any queen
        if (wq_bb | bq_bb) & FILE_MASKS[rook_file]:
            mg_mobility -= ROOK_ON_QUEEN_FILE[0]
            eg_mobility -= ROOK_ON_QUEEN_FILE[1]
        

        # King safety calculation (Black Rook attacking White King)
        if att & white_king_zone:
            w_attacker_count += 1
            w_attacker_weight += KING_ATTACK_WEIGHTS[4]  # ROOK
        w_king_attacks_count += count_bits(att & white_king_adjacent)

        temp_bb &= temp_bb - np.uint64(1)

    temp_bb = bq_bb
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        att = get_queen_attacks(sq, all_occupancy)
        if pinned_black & (np.uint64(1) << np.uint64(sq)):
            sq_bb = np.uint64(1) << np.uint64(sq)
            if sq_bb & ROOK_RAYS[black_king_sq]:
                att &= ROOK_RAYS[black_king_sq]
            elif sq_bb & BISHOP_RAYS[black_king_sq]:
                att &= BISHOP_RAYS[black_king_sq]
            else:
                att = np.uint64(0)
        black_attacks2 |= (black_attacks & att)
        black_attacks |= att
        black_queen_attacks |= att
        moves = count_bits(att & black_mobility_area)
        clamped = min(moves, 27)
        mg_mobility -= QUEEN_MOBILITY_BONUS[clamped, 0]
        eg_mobility -= QUEEN_MOBILITY_BONUS[clamped, 1]
        mg_mobility_pure -= QUEEN_MOBILITY_BONUS[clamped, 0]
        

        # King safety calculation (Black Queen attacking White King)
        if att & white_king_zone:
            w_attacker_count += 1
            w_attacker_weight += KING_ATTACK_WEIGHTS[5]  # QUEEN
        w_king_attacks_count += count_bits(att & white_king_adjacent)

        # WeakQueen: penalty if queen is in a relative pin / discovered attack line
        rook_pinners = ROOK_RAYS[sq] & wr_bb
        temp_pinners = rook_pinners
        is_weak = False
        while temp_pinners and not is_weak:
            pinner_sq = get_lsb_index(temp_pinners)
            between = SQUARES_BETWEEN[sq, pinner_sq]
            blockers_between = between & all_occupancy
            if count_bits(blockers_between) == 1:
                is_weak = True
            temp_pinners &= temp_pinners - np.uint64(1)
        if not is_weak:
            bishop_pinners = BISHOP_RAYS[sq] & wb_bb
            temp_pinners = bishop_pinners
            while temp_pinners and not is_weak:
                pinner_sq = get_lsb_index(temp_pinners)
                between = SQUARES_BETWEEN[sq, pinner_sq]
                blockers_between = between & all_occupancy
                if count_bits(blockers_between) == 1:
                    is_weak = True
                temp_pinners &= temp_pinners - np.uint64(1)
        if is_weak:
            mg_mobility += WEAK_QUEEN[0]
            eg_mobility += WEAK_QUEEN[1]

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
    # In SF11, only pawns on safe squares (not attacked by Them, or defended by Us) can threaten enemy pieces.
    # safe = ~attackedBy[Them][ALL_PIECES] | attackedBy[Us][ALL_PIECES]
    white_safe = ~black_attacks | white_attacks
    white_safe_pawns = wp_bb & white_safe
    white_safe_pawn_attacks = ((white_safe_pawns & NOT_A_FILE) << np.uint64(7)) | ((white_safe_pawns & NOT_H_FILE) << np.uint64(9))
    white_safe_pawn_sq = white_safe_pawn_attacks & black_non_pawns
    if white_safe_pawn_sq:
        count = count_bits(white_safe_pawn_sq)
        mg_threats += THREAT_SAFE_PAWN[0] * count
        eg_threats += THREAT_SAFE_PAWN[1] * count

    black_safe = ~white_attacks | black_attacks
    black_safe_pawns = bp_bb & black_safe
    black_safe_pawn_attacks = ((black_safe_pawns & NOT_H_FILE) >> np.uint64(7)) | ((black_safe_pawns & NOT_A_FILE) >> np.uint64(9))
    black_safe_pawn_sq = black_safe_pawn_attacks & white_non_pawns
    if black_safe_pawn_sq:
        count = count_bits(black_safe_pawn_sq)
        mg_threats -= THREAT_SAFE_PAWN[0] * count
        eg_threats -= THREAT_SAFE_PAWN[1] * count

    # --- 2. ThreatByMinor: SF11 uses (defended | weak) for minors ---
    # defended = strongly-protected non-pawn enemies; weak = not strongly protected & under attack
    black_defended = black_non_pawns & black_strongly_protected
    white_defended = white_non_pawns & white_strongly_protected
    w_minor_threatens = (white_knight_attacks | white_bishop_attacks) & (black_defended | black_weak)
    for pt in range(6):
        intersection = w_minor_threatens & piece_bbs[6 + pt]
        if intersection:
            count = count_bits(intersection)
            mg_threats += THREAT_BY_MINOR[pt, 0] * count
            eg_threats += THREAT_BY_MINOR[pt, 1] * count

    b_minor_threatens = (black_knight_attacks | black_bishop_attacks) & (white_defended | white_weak)
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
    # In SF11, pawn pushes are only evaluated if the landing square is unoccupied and safe.
    # b &= ~attackedBy[Them][PAWN] & safe (where safe = ~attackedBy[Them][ALL_PIECES] | attackedBy[Us][ALL_PIECES])
    w_push1 = (wp_bb << np.uint64(8)) & ~all_occupancy & ~np.uint64(0xFF00000000000000)
    w_push2 = ((w_push1 & np.uint64(0xFF0000)) << np.uint64(8)) & ~all_occupancy  # double push from rank 2
    w_push_landings = (w_push1 | w_push2) & ~black_pawn_attacks & (~black_attacks | white_attacks)
    w_push_attacks = ((w_push_landings & NOT_A_FILE) << np.uint64(7)) | ((w_push_landings & NOT_H_FILE) << np.uint64(9))
    w_push_threat = w_push_attacks & black_non_pawns
    if w_push_threat:
        count = count_bits(w_push_threat)
        mg_threats += THREAT_PAWN_PUSH[0] * count
        eg_threats += THREAT_PAWN_PUSH[1] * count

    b_push1 = (bp_bb >> np.uint64(8)) & ~all_occupancy & ~np.uint64(0xFF)
    b_push2 = ((b_push1 & np.uint64(0xFF0000000000)) >> np.uint64(8)) & ~all_occupancy  # double push from rank 7
    b_push_landings = (b_push1 | b_push2) & ~white_pawn_attacks & (~white_attacks | black_attacks)
    b_push_attacks = ((b_push_landings & NOT_H_FILE) >> np.uint64(7)) | ((b_push_landings & NOT_A_FILE) >> np.uint64(9))
    b_push_threat = b_push_attacks & white_non_pawns
    if b_push_threat:
        count = count_bits(b_push_threat)
        mg_threats -= THREAT_PAWN_PUSH[0] * count
        eg_threats -= THREAT_PAWN_PUSH[1] * count

    # --- KnightOnQueen & SliderOnQueen threats ---
    # White threats against Black Queen
    if count_bits(bq_bb) == 1:
        bq_sq = get_lsb_index(bq_bb)
        w_safe_for_queen_threat = white_mobility_area & ~black_strongly_protected

        # KnightOnQueen: white knight attacks squares that also attack black queen
        bq_knight_attacks = KNIGHT_ATTACKS[bq_sq]
        w_koq = white_knight_attacks & bq_knight_attacks & w_safe_for_queen_threat
        if w_koq != np.uint64(0):
            count = count_bits(w_koq)
            mg_threats += THREAT_KNIGHT_ON_QUEEN[0] * count
            eg_threats += THREAT_KNIGHT_ON_QUEEN[1] * count

        # SliderOnQueen: white bishop/rook attacks squares that also attack black queen
        bq_rook_attacks = get_rook_attacks(bq_sq, all_occupancy)
        bq_bishop_attacks = get_bishop_attacks(bq_sq, all_occupancy)
        w_soq = ((white_bishop_attacks & bq_bishop_attacks) | (white_rook_attacks & bq_rook_attacks)) & w_safe_for_queen_threat & white_attacks2
        if w_soq != np.uint64(0):
            count = count_bits(w_soq)
            mg_threats += THREAT_SLIDER_ON_QUEEN[0] * count
            eg_threats += THREAT_SLIDER_ON_QUEEN[1] * count

    # Black threats against White Queen
    if count_bits(wq_bb) == 1:
        wq_sq = get_lsb_index(wq_bb)
        b_safe_for_queen_threat = black_mobility_area & ~white_strongly_protected

        # KnightOnQueen: black knight attacks squares that also attack white queen
        wq_knight_attacks = KNIGHT_ATTACKS[wq_sq]
        b_koq = black_knight_attacks & wq_knight_attacks & b_safe_for_queen_threat
        if b_koq != np.uint64(0):
            count = count_bits(b_koq)
            mg_threats -= THREAT_KNIGHT_ON_QUEEN[0] * count
            eg_threats -= THREAT_KNIGHT_ON_QUEEN[1] * count

        # SliderOnQueen: black bishop/rook attacks squares that also attack white queen
        wq_rook_attacks = get_rook_attacks(wq_sq, all_occupancy)
        wq_bishop_attacks = get_bishop_attacks(wq_sq, all_occupancy)
        b_soq = ((black_bishop_attacks & wq_bishop_attacks) | (black_rook_attacks & wq_rook_attacks)) & b_safe_for_queen_threat & black_attacks2
        if b_soq != np.uint64(0):
            count = count_bits(b_soq)
            mg_threats -= THREAT_SLIDER_ON_QUEEN[0] * count
            eg_threats -= THREAT_SLIDER_ON_QUEEN[1] * count

    return (white_attacks, black_attacks, white_pawn_attacks, black_pawn_attacks,
            white_attacks2, black_attacks2, pinned_white, pinned_black,
            mg_mobility, eg_mobility, mg_threats, eg_threats, mg_outpost, eg_outpost,
            white_knight_attacks, white_bishop_attacks, white_rook_attacks, white_queen_attacks,
            black_knight_attacks, black_bishop_attacks, black_rook_attacks, black_queen_attacks,
            w_attacker_weight, w_attacker_count, w_king_attacks_count,
            b_attacker_weight, b_attacker_count, b_king_attacks_count,
            white_king_zone, black_king_zone, mg_mobility_pure)


@numba.njit(numba.int32(piece_bbs_signature, numba.uint64, numba.uint8), cache=True, boundscheck=False, fastmath=True)
def _evaluate_king_pawn_endgame(piece_bbs, side_to_move, castling_rights):
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
                 score += UNSTOPPABLE_PAWN_BONUS # Queen value approx
            
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
                score -= UNSTOPPABLE_PAWN_BONUS

        temp_bp &= temp_bp - np.uint64(1)

    # King position
    score += PST_EG[5, white_king_sq]
    score -= PST_EG[5, black_king_sq ^ 56]

    res = evaluate_pawn_structure(piece_bbs, castling_rights)
    eg_pawn_score = res[1]
    w_passed = res[6]
    b_passed = res[7]
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
    _, eg_passed_score, _ = evaluate_passed_pawns(piece_bbs, occ_bbs, white_attacks, black_attacks,
                                                 w_passed, b_passed,
                                                 white_king_sq, black_king_sq)
    score += eg_passed_score

    # 3. 國王活動獎勵
    # 獎勵國王靠近所有兵 (自己的和對手的)
    temp_pawns = white_pawns | black_pawns
    while temp_pawns:
        sq = get_lsb_index(temp_pawns)
        w_dist = MANHATTAN_DISTANCE[white_king_sq, sq]
        b_dist = MANHATTAN_DISTANCE[black_king_sq, sq]
        score += (b_dist - w_dist) * KING_PAWN_PROXIMITY_FACTOR
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

    white_space_score = (w_bonus * w_weight * w_weight) // SPACE_BONUS_DIVISOR
    black_space_score = (b_bonus * b_weight * b_weight) // SPACE_BONUS_DIVISOR

    # Scale to engine's evaluation units (using optimized // 4 division for positional safety)
    white_space_scaled = white_space_score // SPACE_SCALE_DIVISOR
    black_space_scaled = black_space_score // SPACE_SCALE_DIVISOR

    # Difference in space score
    return np.int32(white_space_scaled - black_space_scaled)


@numba.njit(cache=True, boundscheck=False, fastmath=True, inline='always')
def _compute_initiative(mg, eg, piece_bbs, passed_count, white_king_sq, black_king_sq):
    """
    Computes initiative correction based on complexity (SF11-style).
    """
    # 1. outflanking
    w_king_file = white_king_sq % 8
    b_king_file = black_king_sq % 8
    w_king_rank = white_king_sq // 8
    b_king_rank = black_king_sq // 8
    outflanking = abs(w_king_file - b_king_file) - abs(w_king_rank - b_king_rank)

    # 2. infiltration
    infiltration = (w_king_rank > 3) or (b_king_rank < 4)

    # 3. pawnsOnBothFlanks
    pawns = piece_bbs[0] | piece_bbs[6]
    pawns_on_both_flanks = ((pawns & QUEEN_SIDE_BB) != np.uint64(0)) and ((pawns & KING_SIDE_BB) != np.uint64(0))

    # 4. almostUnwinnable
    almost_unwinnable = (passed_count == 0) and (outflanking < 0) and not pawns_on_both_flanks

    # 5. pawn count
    pawn_count = count_bits(pawns)

    # 6. pure pawn endgame
    non_pawns = (
        piece_bbs[1] | piece_bbs[2] | piece_bbs[3] | piece_bbs[4] |
        piece_bbs[7] | piece_bbs[8] | piece_bbs[9] | piece_bbs[10]
    )
    pure_pawn_endgame = (non_pawns == np.uint64(0))

    # Convert booleans to int32 to ensure static JIT typing
    infiltration_val = np.int32(1) if infiltration else np.int32(0)
    pawns_on_both_flanks_val = np.int32(1) if pawns_on_both_flanks else np.int32(0)
    pure_pawn_endgame_val = np.int32(1) if pure_pawn_endgame else np.int32(0)
    almost_unwinnable_val = np.int32(1) if almost_unwinnable else np.int32(0)

    # Compute MG complexity (weights scaled by 0.78)
    mg_complexity = (
        INITIATIVE_PASSED_WEIGHT_MG * passed_count
        + INITIATIVE_PAWN_WEIGHT_MG * pawn_count
        + INITIATIVE_OUTFLANKING_WEIGHT_MG * outflanking
        + INITIATIVE_INFILTRATION_WEIGHT_MG * infiltration_val
        + INITIATIVE_BOTH_FLANKS_WEIGHT_MG * pawns_on_both_flanks_val
        + INITIATIVE_PAWN_ENDGAME_WEIGHT_MG * pure_pawn_endgame_val
        - INITIATIVE_ALMOST_UNWIN_WEIGHT_MG * almost_unwinnable_val
        + INITIATIVE_OFFSET_MG
    )

    # Compute EG complexity (weights scaled by 0.563)
    eg_complexity = (
        INITIATIVE_PASSED_WEIGHT_EG * passed_count
        + INITIATIVE_PAWN_WEIGHT_EG * pawn_count
        + INITIATIVE_OUTFLANKING_WEIGHT_EG * outflanking
        + INITIATIVE_INFILTRATION_WEIGHT_EG * infiltration_val
        + INITIATIVE_BOTH_FLANKS_WEIGHT_EG * pawns_on_both_flanks_val
        + INITIATIVE_PAWN_ENDGAME_WEIGHT_EG * pure_pawn_endgame_val
        - INITIATIVE_ALMOST_UNWIN_WEIGHT_EG * almost_unwinnable_val
        + INITIATIVE_OFFSET_EG
    )

    # Apply sign and caps (dampen only for MG, can amplify or dampen for EG)
    mg_sign = np.int32(1) if mg > 0 else (np.int32(-1) if mg < 0 else np.int32(0))
    eg_sign = np.int32(1) if eg > 0 else (np.int32(-1) if eg < 0 else np.int32(0))

    u = mg_sign * max(min(mg_complexity + INITIATIVE_MG_OFFSET, np.int32(0)), -abs(mg))
    v = eg_sign * max(eg_complexity, -abs(eg))

    return u, v


@numba.njit(cache=True, boundscheck=False, fastmath=True)
def _evaluate_position_jit(piece_bbs, occupancy_bbs, game_state, lazy: bool, search_context=None):
    """
    使用 Tapered Evaluation (加權評估) 模型評估目前局面，並從當前執棋方的角度返回分數。
    評估包括：材質、PST、國王安全、兵形結構、棋子協同性和機動性。
    
    Args:
        piece_bbs (np.ndarray): 12 個棋子的位元棋盤。
        occupancy_bbs (np.ndarray): 佔用位元棋盤。
        game_state (np.ndarray): 遊戲狀態。
        lazy (bool): 是否使用懶惰評估（僅材質 and PST，用於 Razoring 等）。
        
    Returns:
        int: 從行棋方視角的評估分數。
    """
    side_to_move = game_state[0]

    # --- Specialized endgames (SF11): KBNK, KPK, KRKP, KQKP, KXK,
    #     KNNK, KRKB, KRKN, KQKR, KNNKP ---
    # Score from evaluate_special_endgame is White's perspective.
    hit_special, special_score = evaluate_special_endgame(piece_bbs, occupancy_bbs, game_state, np.int32(0))
    if hit_special:
        val = np.int32(special_score) if side_to_move == 0 else np.int32(-special_score)
        return val, np.uint64(0), np.uint64(0)

    # --- Multi-pawn pure king-and-pawn endgame (not covered by KPK bitbase) ---
    all_pieces_except_pawns_and_kings = (
        piece_bbs[1] | piece_bbs[2] | piece_bbs[3] | piece_bbs[4] |
        piece_bbs[7] | piece_bbs[8] | piece_bbs[9] | piece_bbs[10]
    )
    if all_pieces_except_pawns_and_kings == 0:
        castling_rights = np.uint8(game_state[1])
        score = _evaluate_king_pawn_endgame(piece_bbs, game_state[0], castling_rights)
        val = np.int32(score) if side_to_move == 0 else np.int32(-score)
        return val, np.uint64(0), np.uint64(0)

    # Precompute King Squares for evaluation components
    white_king_sq = get_lsb_index(piece_bbs[5])
    black_king_sq = get_lsb_index(piece_bbs[11])

    # --- 1. & 2. Phase, Material, and PST (Unrolled & Fused) ---
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
    
    # Game phase from non-pawn material (SF11 material.cpp npm taper)
    w_npm = (cnt_1 * MG_MATERIAL_VALUES[1] + cnt_2 * MG_MATERIAL_VALUES[2]
             + cnt_3 * MG_MATERIAL_VALUES[3] + cnt_4 * MG_MATERIAL_VALUES[4])
    b_npm = (cnt_7 * MG_MATERIAL_VALUES[1] + cnt_8 * MG_MATERIAL_VALUES[2]
             + cnt_9 * MG_MATERIAL_VALUES[3] + cnt_10 * MG_MATERIAL_VALUES[4])
    npm = w_npm + b_npm
    npm_clamped = npm
    if npm_clamped < ENDGAME_LIMIT:
        npm_clamped = ENDGAME_LIMIT
    elif npm_clamped > MIDGAME_LIMIT:
        npm_clamped = MIDGAME_LIMIT
    phase = ((npm_clamped - ENDGAME_LIMIT) * PHASE_MIDGAME) // (MIDGAME_LIMIT - ENDGAME_LIMIT)

    # --- Material Imbalance ---
    mg_imb, eg_imb = compute_imbalance(cnt_0, cnt_1, cnt_2, cnt_3, cnt_4,
                                        cnt_6, cnt_7, cnt_8, cnt_9, cnt_10)
    mg_score += mg_imb
    eg_score += eg_imb

    # --- Lazy Evaluation Checkpoint ---
    if lazy:
        final_score = (mg_score * phase + eg_score * (MAX_PHASE - phase)) // MAX_PHASE
        val = np.int32(final_score) if side_to_move == 0 else np.int32(-final_score)
        return val, np.uint64(18446744073709551615), np.uint64(18446744073709551615)

    # --- 4. Pawn Structure (Cached or Uncached) ---
    if search_context is not None:
        pawn_key = game_state[PAWN_KEY_INDEX]
        castling_rights = np.uint8(game_state[1])
        (mg_pawn_structure, eg_pawn_structure, 
         white_mg_shield, white_eg_shield, black_mg_shield, black_eg_shield,
         w_passed, b_passed,
         white_pawn_attacks_span, black_pawn_attacks_span) = evaluate_pawn_structure_cached(piece_bbs, pawn_key, castling_rights, search_context)
    else:
        castling_rights = np.uint8(game_state[1])
        (mg_pawn_structure, eg_pawn_structure, 
         white_mg_shield, white_eg_shield, black_mg_shield, black_eg_shield,
         w_passed, b_passed,
         white_pawn_attacks_span, black_pawn_attacks_span) = evaluate_pawn_structure(piece_bbs, castling_rights)
    mg_score += mg_pawn_structure
    eg_score += eg_pawn_structure

    # --- Dynamic Lazy Evaluation Threshold Early Exit ---
    if not lazy:
        v = (mg_score + eg_score) // 2
        if abs(v) > LAZY_EVAL_THRESHOLD + npm // 64:
            final_score = (mg_score * phase + eg_score * (MAX_PHASE - phase)) // MAX_PHASE
            val = np.int32(final_score) if side_to_move == 0 else np.int32(-final_score)
            return val, np.uint64(18446744073709551615), np.uint64(18446744073709551615)

    # --- Compute Attacks, Mobility, Threats (Optimized Single Pass) ---
    (white_attacks, black_attacks, white_pawn_attacks, black_pawn_attacks,
     white_attacks2, black_attacks2, pinned_white, pinned_black,
     mg_mobility, eg_mobility, mg_threats, eg_threats,
     mg_outpost, eg_outpost,
     white_knight_attacks, white_bishop_attacks, white_rook_attacks, white_queen_attacks,
     black_knight_attacks, black_bishop_attacks, black_rook_attacks, black_queen_attacks,
     w_attacker_weight, w_attacker_count, w_king_attacks_count,
     b_attacker_weight, b_attacker_count, b_king_attacks_count,
     white_king_zone, black_king_zone, mg_mobility_pure) = evaluate_attacks_mobility_threats(
         piece_bbs, occupancy_bbs, white_king_sq, black_king_sq, white_pawn_attacks_span, black_pawn_attacks_span, castling_rights)

    # --- 4.5 Passed Pawns (Dynamic Evaluation) ---
    mg_passed, eg_passed, passed_count = evaluate_passed_pawns(piece_bbs, occupancy_bbs, white_attacks, black_attacks,
                                                               w_passed, b_passed,
                                                               white_king_sq, black_king_sq)
    mg_score += mg_passed
    eg_score += eg_passed

    # --- 3. King Safety ---
    mg_king_safety, eg_king_safety = evaluate_king_safety(
        piece_bbs, occupancy_bbs, game_state, white_attacks, black_attacks,
        piece_counts, white_attacks2, black_attacks2, pinned_white, pinned_black,
        white_mg_shield, white_eg_shield, black_mg_shield, black_eg_shield,
        white_king_sq, black_king_sq,
        w_attacker_weight, w_attacker_count, w_king_attacks_count,
        b_attacker_weight, b_attacker_count, b_king_attacks_count,
        white_knight_attacks, white_bishop_attacks, white_rook_attacks, white_queen_attacks,
        black_knight_attacks, black_bishop_attacks, black_rook_attacks, black_queen_attacks,
        mg_mobility_pure,
        white_king_zone, black_king_zone,
    )
    mg_score += mg_king_safety
    eg_score += eg_king_safety

    # --- 5. Piece Coordination ---
    mg_coord, eg_coord = evaluate_piece_coordination(piece_bbs, piece_counts)
    mg_score += mg_coord
    eg_score += eg_coord

    # --- 6. Piece Mobility ---
    mg_score += mg_mobility
    eg_score += eg_mobility

    # --- 7. Outpost Evaluation ---
    mg_score += mg_outpost
    eg_score += eg_outpost

    # --- 8. Threat Evaluation ---
    mg_score += mg_threats
    eg_score += eg_threats

    # --- 8.5 Space Evaluation ---
    if not lazy:
        mg_space = evaluate_space(piece_bbs, occupancy_bbs, white_attacks, black_attacks, 
                                  white_pawn_attacks, black_pawn_attacks, piece_counts)
        mg_score += mg_space

    # --- 8.75 Initiative (space evaluation is enabled above; stale "disabled" notes removed) ---
    mg_init, eg_init = _compute_initiative(mg_score, eg_score, piece_bbs, passed_count, white_king_sq, black_king_sq)
    mg_score += mg_init
    eg_score += eg_init

    # --- 9. Endgame Scale Factor + Tapered Evaluation ---
    sf = get_endgame_scale_factor(piece_bbs, occupancy_bbs, game_state, phase, eg_score)
    if sf != SCALE_FACTOR_NORMAL:
        eg_scaled = eg_score * sf // SCALE_FACTOR_NORMAL
        final_score = (mg_score * phase + eg_scaled * (MAX_PHASE - phase)) // MAX_PHASE
    else:
        final_score = (mg_score * phase + eg_score * (MAX_PHASE - phase)) // MAX_PHASE

    # --- 10. Tempo Bonus ---
    # The side to move receives the Tempo bonus.

    # --- 11. Return side-relative score ---
    if side_to_move == 0:  # White's turn
        return np.int32(final_score + TEMPO_BONUS), pinned_white, pinned_black
    else:  # Black's turn
        return np.int32(-final_score + TEMPO_BONUS), pinned_white, pinned_black


def evaluate_position(piece_bbs, occupancy_bbs, game_state, lazy: bool = False):
    """
    使用 Tapered Evaluation (加權評估) 模型評估目前局面，並從當前執棋方的角度返回分數。
    """
    return _evaluate_position_jit(piece_bbs, occupancy_bbs, game_state, lazy)
