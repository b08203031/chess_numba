# tuner/tunable_eval.py

import numba
import numpy as np

# Import constants that are NOT tunable (like masks, phase weights, bitboard utils)
from chess_engine.constants import (
    PHASE_WEIGHTS, MAX_PHASE, INITIATIVE_PHASE_THRESHOLD,
    KNIGHT_MOBILITY_BASE_MOVES, BISHOP_MOBILITY_BASE_MOVES, ROOK_MOBILITY_BASE_MOVES, QUEEN_MOBILITY_BASE_MOVES,
    MAX_SCALING_MATERIAL, KING_TROPISM_MAX_DISTANCE, EG_SAFETY_SCALE, BB_SQUARES
)
from chess_engine.zobrist import get_lsb_index
from chess_engine.engine_types import piece_bbs_signature, occupancy_bbs_signature, game_state_signature
from chess_engine.bitboard_utils import count_bits, KING_ATTACK_ZONES, FILE_MASKS
from chess_engine.move_generator import (
    get_bishop_attacks, get_rook_attacks, get_queen_attacks, KNIGHT_ATTACKS, PAWN_ATTACKS
)
# Re-import or copy helper data structures
from chess_engine.evaluation import (
    MANHATTAN_DISTANCE, ADJACENT_FILES_MASKS, WHITE_PASSED_PAWN_MASKS, BLACK_PASSED_PAWN_MASKS
)

# --- Define Offsets (Must match parameters.py) ---
# Use fixed offsets to avoid dictionary lookups inside JIT
IDX_MG_MATERIAL = 0
IDX_EG_MATERIAL = 6
IDX_PST_MG = 12
IDX_PST_EG = 396
IDX_KNIGHT_MOBILITY = 780
IDX_BISHOP_MOBILITY = 782
IDX_ROOK_MOBILITY = 784
IDX_QUEEN_MOBILITY = 786
IDX_BISHOP_PAIR = 788
IDX_ROOK_SEMI_OPEN = 790
IDX_ROOK_OPEN = 792
IDX_PASSED_PAWN = 794
IDX_ISOLATED_PAWN = 810
IDX_DOUBLED_PAWN = 812
IDX_CONNECTED_PAWN = 814
IDX_KING_SAFETY_UNITS = 816
IDX_KING_SAFETY_TABLE = 821
IDX_KING_TROPISM = 921
IDX_PAWN_STORM = 926
IDX_SCALING = 934
IDX_INITIATIVE = 939

# --- Helper Functions to retrieve params from theta ---

@numba.njit(numba.int32(numba.float64[:], numba.int32), cache=True, inline='always')
def get_int(theta, idx):
    return numba.int32(theta[idx])

@numba.njit(numba.int32(numba.float64[:], numba.int32, numba.int32), cache=True, inline='always')
def get_array_val(theta, base_idx, offset):
    return numba.int32(theta[base_idx + offset])

@numba.njit(numba.int32(numba.float64[:], numba.int32, numba.int32, numba.int32, numba.int32), cache=True, inline='always')
def get_2d_val(theta, base_idx, row, col_size, col):
    # Flattened access
    return numba.int32(theta[base_idx + row * col_size + col])

# --- Evaluation Functions ---

@numba.njit(numba.types.UniTuple(numba.int32, 2)(piece_bbs_signature, numba.float64[:]), cache=True, boundscheck=False, fastmath=True)
def evaluate_pawn_structure(piece_bbs, theta):
    mg_score = np.int32(0)
    eg_score = np.int32(0)

    white_pawns = piece_bbs[0]
    black_pawns = piece_bbs[6]

    # Constants extraction
    # passed_pawn: (8, 2) but flattened
    # isolated: (2,)
    # doubled: (2,)

    # White passed pawns
    temp_wp = white_pawns
    while temp_wp:
        sq = get_lsb_index(temp_wp)
        if not (WHITE_PASSED_PAWN_MASKS[sq] & black_pawns):
            rank = sq // 8
            # PASSED_PAWN_BONUS[rank][0] -> index = IDX_PASSED_PAWN + rank*2 + 0
            mg_score += get_2d_val(theta, IDX_PASSED_PAWN, rank, 2, 0)
            eg_score += get_2d_val(theta, IDX_PASSED_PAWN, rank, 2, 1)
        temp_wp &= temp_wp - np.uint64(1)

    # Black passed pawns
    temp_bp = black_pawns
    while temp_bp:
        sq = get_lsb_index(temp_bp)
        if not (BLACK_PASSED_PAWN_MASKS[sq] & white_pawns):
            rank = 7 - (sq // 8)
            mg_score -= get_2d_val(theta, IDX_PASSED_PAWN, rank, 2, 0)
            eg_score -= get_2d_val(theta, IDX_PASSED_PAWN, rank, 2, 1)
        temp_bp &= temp_bp - np.uint64(1)

    # Isolated / Doubled
    iso_mg = get_array_val(theta, IDX_ISOLATED_PAWN, 0)
    iso_eg = get_array_val(theta, IDX_ISOLATED_PAWN, 1)
    dbl_mg = get_array_val(theta, IDX_DOUBLED_PAWN, 0)
    dbl_eg = get_array_val(theta, IDX_DOUBLED_PAWN, 1)

    for f in range(8):
        file_mask = FILE_MASKS[f]
        adjacent_mask = ADJACENT_FILES_MASKS[f]

        white_pawns_on_file = count_bits(white_pawns & file_mask)
        black_pawns_on_file = count_bits(black_pawns & file_mask)

        # White
        if white_pawns_on_file > 0:
            if not (white_pawns & adjacent_mask):
                mg_score += iso_mg * white_pawns_on_file
                eg_score += iso_eg * white_pawns_on_file
            if white_pawns_on_file > 1:
                mg_score += dbl_mg * (white_pawns_on_file - 1)
                eg_score += dbl_eg * (white_pawns_on_file - 1)

        # Black
        if black_pawns_on_file > 0:
            if not (black_pawns & adjacent_mask):
                mg_score -= iso_mg * black_pawns_on_file
                eg_score -= iso_eg * black_pawns_on_file
            if black_pawns_on_file > 1:
                mg_score -= dbl_mg * (black_pawns_on_file - 1)
                eg_score -= dbl_eg * (black_pawns_on_file - 1)

    return mg_score, eg_score

@numba.njit(numba.types.UniTuple(numba.int32, 2)(piece_bbs_signature, numba.float64[:]), cache=True, boundscheck=False, fastmath=True)
def evaluate_piece_coordination(piece_bbs, theta):
    mg_score = np.int32(0)
    eg_score = np.int32(0)

    white_pawns = piece_bbs[0]
    white_bishops = piece_bbs[2]
    white_rooks = piece_bbs[3]
    black_pawns = piece_bbs[6]
    black_bishops = piece_bbs[8]
    black_rooks = piece_bbs[9]

    bp_mg = get_array_val(theta, IDX_BISHOP_PAIR, 0)
    bp_eg = get_array_val(theta, IDX_BISHOP_PAIR, 1)

    if count_bits(white_bishops) >= 2:
        mg_score += bp_mg
        eg_score += bp_eg
    if count_bits(black_bishops) >= 2:
        mg_score -= bp_mg
        eg_score -= bp_eg

    open_mg = get_array_val(theta, IDX_ROOK_OPEN, 0)
    open_eg = get_array_val(theta, IDX_ROOK_OPEN, 1)
    semi_mg = get_array_val(theta, IDX_ROOK_SEMI_OPEN, 0)
    semi_eg = get_array_val(theta, IDX_ROOK_SEMI_OPEN, 1)

    for f in range(8):
        file_mask = FILE_MASKS[f]
        white_pawns_on_file = (white_pawns & file_mask) != 0
        black_pawns_on_file = (black_pawns & file_mask) != 0

        if (white_rooks & file_mask):
            if not white_pawns_on_file:
                if not black_pawns_on_file:
                    mg_score += open_mg
                    eg_score += open_eg
                else:
                    mg_score += semi_mg
                    eg_score += semi_eg

        if (black_rooks & file_mask):
            if not black_pawns_on_file:
                if not white_pawns_on_file:
                    mg_score -= open_mg
                    eg_score -= open_eg
                else:
                    mg_score -= semi_mg
                    eg_score -= semi_eg

    return mg_score, eg_score

@numba.njit(numba.int32(numba.int32, numba.int32, piece_bbs_signature, occupancy_bbs_signature, numba.float64[:]), cache=True, boundscheck=False, fastmath=True)
def _evaluate_king_attackers(king_sq, color, piece_bbs, occupancy_bbs, theta):
    king_zone = KING_ATTACK_ZONES[king_sq]
    total_attack_units = np.int32(0)
    attacker_count = np.int32(0)

    enemy_start_idx = 6 if color == 0 else 0
    (en_p, en_n, en_b, en_r, en_q, _) = piece_bbs[enemy_start_idx : enemy_start_idx + 6]
    all_pieces_occupancy = occupancy_bbs[2]

    # Pawns
    pawn_attack_sources_bb = np.uint64(0)
    temp_king_zone = king_zone
    while temp_king_zone:
        zone_sq = get_lsb_index(temp_king_zone)
        pawn_attack_sources_bb |= PAWN_ATTACKS[color, zone_sq]
        temp_king_zone &= temp_king_zone - np.uint64(1)

    actual_pawn_attackers = pawn_attack_sources_bb & en_p
    if actual_pawn_attackers:
        num = count_bits(actual_pawn_attackers)
        total_attack_units += num * get_array_val(theta, IDX_KING_SAFETY_UNITS, 0)
        attacker_count += num

    # Knights (Unit Index 1)
    temp_bb = en_n
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        attacks = KNIGHT_ATTACKS[sq]
        if attacks & king_zone:
            total_attack_units += get_array_val(theta, IDX_KING_SAFETY_UNITS, 1)
            attacker_count += 1
        temp_bb &= temp_bb - np.uint64(1)

    # Bishops (Unit Index 2)
    temp_bb = en_b
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        attacks = get_bishop_attacks(sq, all_pieces_occupancy & ~BB_SQUARES[sq])
        if attacks & king_zone:
            total_attack_units += get_array_val(theta, IDX_KING_SAFETY_UNITS, 2)
            attacker_count += 1
        temp_bb &= temp_bb - np.uint64(1)

    # Rooks (Unit Index 3)
    temp_bb = en_r
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        attacks = get_rook_attacks(sq, all_pieces_occupancy & ~BB_SQUARES[sq])
        if attacks & king_zone:
            total_attack_units += get_array_val(theta, IDX_KING_SAFETY_UNITS, 3)
            attacker_count += 1
        temp_bb &= temp_bb - np.uint64(1)

    # Queens (Unit Index 4)
    temp_bb = en_q
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        attacks = get_queen_attacks(sq, all_pieces_occupancy & ~BB_SQUARES[sq])
        if attacks & king_zone:
            total_attack_units += get_array_val(theta, IDX_KING_SAFETY_UNITS, 4)
            attacker_count += 1
        temp_bb &= temp_bb - np.uint64(1)

    if attacker_count < 2:
        return np.int32(0)

    # Lookup table logic
    # Ensure index is within bounds of the table (size 100)
    # The table was initialized with 100 elements.
    idx = min(total_attack_units, 99)
    return -get_array_val(theta, IDX_KING_SAFETY_TABLE, idx)

@numba.njit(numba.int32(numba.int32, numba.int32, piece_bbs_signature, numba.float64[:]), cache=True, boundscheck=False, fastmath=True)
def _evaluate_king_tropism(king_sq, color, piece_bbs, theta):
    penalty = np.int32(0)
    enemy_start_idx = 6 if color == 0 else 0

    for piece_type_offset in range(5): # P, N, B, R, Q
        piece_type = enemy_start_idx + piece_type_offset
        weight = get_array_val(theta, IDX_KING_TROPISM, piece_type_offset)

        temp_bb = piece_bbs[piece_type]
        while temp_bb:
            sq = get_lsb_index(temp_bb)
            distance = MANHATTAN_DISTANCE[sq, king_sq]
            penalty += weight * (KING_TROPISM_MAX_DISTANCE - distance)
            temp_bb &= temp_bb - np.uint64(1)

    return -penalty

@numba.njit(numba.int32(numba.int32, numba.int32, piece_bbs_signature, numba.float64[:]), cache=True, boundscheck=False, fastmath=True)
def _evaluate_pawn_storm(king_sq, color, piece_bbs, theta):
    penalty = np.int32(0)
    king_file = king_sq % 8
    enemy_pawn_idx = 6 if color == 0 else 0
    enemy_pawns = piece_bbs[enemy_pawn_idx]

    for f in range(max(0, king_file - 1), min(7, king_file + 1) + 1):
        file_mask = FILE_MASKS[f]
        pawns_on_file = enemy_pawns & file_mask
        while pawns_on_file:
            sq = get_lsb_index(pawns_on_file)
            rank = sq // 8
            table_idx = rank if color == 0 else (7 - rank)
            penalty -= get_array_val(theta, IDX_PAWN_STORM, table_idx)
            pawns_on_file &= pawns_on_file - np.uint64(1)
    return penalty

# Copied mostly as is, but without parameters since they are constants 10/5/-10 etc.
# Wait, user might want to tune Pawn Shield too. But it was not in my parameters list explicitly
# except maybe hardcoded. The current implementation has hardcoded 10, 5, -10, -20.
# I'll leave it hardcoded for now unless I add more params.
@numba.njit(numba.int32(numba.int32, numba.uint64, numba.uint64, numba.int32), cache=True, boundscheck=False, fastmath=True)
def _evaluate_pawn_shield_for_color(king_sq, friendly_pawns, enemy_pawns, color):
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

@numba.njit(numba.types.UniTuple(numba.int32, 2)(piece_bbs_signature, occupancy_bbs_signature, numba.float64[:]), cache=True, boundscheck=False, fastmath=True)
def evaluate_king_safety(piece_bbs, occupancy_bbs, theta):
    (wp_bb, wn_bb, wb_bb, wr_bb, wq_bb, wk_bb,
    bp_bb, bn_bb, bb_bb, br_bb, bq_bb, bk_bb) = piece_bbs
    white_king_sq = get_lsb_index(wk_bb)
    black_king_sq = get_lsb_index(bk_bb)

    # Raw
    white_shield = _evaluate_pawn_shield_for_color(white_king_sq, wp_bb, bp_bb, 0)
    white_attackers = _evaluate_king_attackers(white_king_sq, 0, piece_bbs, occupancy_bbs, theta)
    white_tropism = _evaluate_king_tropism(white_king_sq, 0, piece_bbs, theta)
    white_pawn_storm = _evaluate_pawn_storm(white_king_sq, 0, piece_bbs, theta)

    black_shield = _evaluate_pawn_shield_for_color(black_king_sq, bp_bb, wp_bb, 1)
    black_attackers = _evaluate_king_attackers(black_king_sq, 1, piece_bbs, occupancy_bbs, theta)
    black_tropism = _evaluate_king_tropism(black_king_sq, 1, piece_bbs, theta)
    black_pawn_storm = _evaluate_pawn_storm(black_king_sq, 1, piece_bbs, theta)

    white_raw = white_shield + white_attackers + white_tropism + white_pawn_storm
    black_raw = black_shield + black_attackers + black_tropism + black_pawn_storm

    # Scaling
    # Weights for N, B, R, Q are at IDX_SCALING + 1..4 (since array has 0=dummy)
    w_n = get_array_val(theta, IDX_SCALING, 1)
    w_b = get_array_val(theta, IDX_SCALING, 2)
    w_r = get_array_val(theta, IDX_SCALING, 3)
    w_q = get_array_val(theta, IDX_SCALING, 4)

    black_mat = (count_bits(bn_bb)*w_n + count_bits(bb_bb)*w_b + count_bits(br_bb)*w_r + count_bits(bq_bb)*w_q)
    white_scale = black_mat / MAX_SCALING_MATERIAL
    white_final = np.int32(white_raw * white_scale)

    white_mat = (count_bits(wn_bb)*w_n + count_bits(wb_bb)*w_b + count_bits(wr_bb)*w_r + count_bits(wq_bb)*w_q)
    black_scale = white_mat / MAX_SCALING_MATERIAL
    black_final = np.int32(black_raw * black_scale)

    mg_score = white_final - black_final
    eg_score = mg_score * EG_SAFETY_SCALE
    return mg_score, eg_score

@numba.njit(numba.types.UniTuple(numba.int32, 2)(piece_bbs_signature, occupancy_bbs_signature, numba.float64[:]), cache=True, boundscheck=False, fastmath=True)
def evaluate_mobility(piece_bbs, occupancy_bbs, theta):
    mg_score = np.int32(0)
    eg_score = np.int32(0)
    white_occupancy = occupancy_bbs[0]
    black_occupancy = occupancy_bbs[1]
    all_pieces_occupancy = occupancy_bbs[2]

    # Param helpers
    def get_mob_params(idx_weight):
        return get_array_val(theta, idx_weight, 0), get_array_val(theta, idx_weight, 1)

    n_w_mg, n_w_eg = get_mob_params(IDX_KNIGHT_MOBILITY)
    b_w_mg, b_w_eg = get_mob_params(IDX_BISHOP_MOBILITY)
    r_w_mg, r_w_eg = get_mob_params(IDX_ROOK_MOBILITY)
    q_w_mg, q_w_eg = get_mob_params(IDX_QUEEN_MOBILITY)

    # White
    temp_bb = piece_bbs[1]
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        moves = count_bits(KNIGHT_ATTACKS[sq] & ~white_occupancy)
        mg_score += (moves - KNIGHT_MOBILITY_BASE_MOVES) * n_w_mg
        eg_score += (moves - KNIGHT_MOBILITY_BASE_MOVES) * n_w_eg
        temp_bb &= temp_bb - np.uint64(1)

    temp_bb = piece_bbs[2]
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        moves = count_bits(get_bishop_attacks(sq, all_pieces_occupancy) & ~white_occupancy)
        mg_score += (moves - BISHOP_MOBILITY_BASE_MOVES) * b_w_mg
        eg_score += (moves - BISHOP_MOBILITY_BASE_MOVES) * b_w_eg
        temp_bb &= temp_bb - np.uint64(1)

    temp_bb = piece_bbs[3]
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        moves = count_bits(get_rook_attacks(sq, all_pieces_occupancy) & ~white_occupancy)
        mg_score += (moves - ROOK_MOBILITY_BASE_MOVES) * r_w_mg
        eg_score += (moves - ROOK_MOBILITY_BASE_MOVES) * r_w_eg
        temp_bb &= temp_bb - np.uint64(1)

    temp_bb = piece_bbs[4]
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        moves = count_bits(get_queen_attacks(sq, all_pieces_occupancy) & ~white_occupancy)
        mg_score += (moves - QUEEN_MOBILITY_BASE_MOVES) * q_w_mg
        eg_score += (moves - QUEEN_MOBILITY_BASE_MOVES) * q_w_eg
        temp_bb &= temp_bb - np.uint64(1)

    # Black (Subtract)
    temp_bb = piece_bbs[7]
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        moves = count_bits(KNIGHT_ATTACKS[sq] & ~black_occupancy)
        mg_score -= (moves - KNIGHT_MOBILITY_BASE_MOVES) * n_w_mg
        eg_score -= (moves - KNIGHT_MOBILITY_BASE_MOVES) * n_w_eg
        temp_bb &= temp_bb - np.uint64(1)

    temp_bb = piece_bbs[8]
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        moves = count_bits(get_bishop_attacks(sq, all_pieces_occupancy) & ~black_occupancy)
        mg_score -= (moves - BISHOP_MOBILITY_BASE_MOVES) * b_w_mg
        eg_score -= (moves - BISHOP_MOBILITY_BASE_MOVES) * b_w_eg
        temp_bb &= temp_bb - np.uint64(1)

    temp_bb = piece_bbs[9]
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        moves = count_bits(get_rook_attacks(sq, all_pieces_occupancy) & ~black_occupancy)
        mg_score -= (moves - ROOK_MOBILITY_BASE_MOVES) * r_w_mg
        eg_score -= (moves - ROOK_MOBILITY_BASE_MOVES) * r_w_eg
        temp_bb &= temp_bb - np.uint64(1)

    temp_bb = piece_bbs[10]
    while temp_bb:
        sq = get_lsb_index(temp_bb)
        moves = count_bits(get_queen_attacks(sq, all_pieces_occupancy) & ~black_occupancy)
        mg_score -= (moves - QUEEN_MOBILITY_BASE_MOVES) * q_w_mg
        eg_score -= (moves - QUEEN_MOBILITY_BASE_MOVES) * q_w_eg
        temp_bb &= temp_bb - np.uint64(1)

    return mg_score, eg_score

@numba.njit(numba.int32(piece_bbs_signature, numba.float64[:]), cache=True, boundscheck=False, fastmath=True)
def _evaluate_king_pawn_endgame(piece_bbs, theta):
    score = np.int32(0)
    white_pawns = piece_bbs[0]
    white_king_sq = get_lsb_index(piece_bbs[5])
    black_pawns = piece_bbs[6]
    black_king_sq = get_lsb_index(piece_bbs[11])

    # Values
    eg_pawn_val = get_array_val(theta, IDX_EG_MATERIAL, 0)

    # White
    temp_wp = white_pawns
    while temp_wp:
        sq = get_lsb_index(temp_wp)
        score += eg_pawn_val + get_2d_val(theta, IDX_PST_EG, 0, 64, sq)
        temp_wp &= temp_wp - np.uint64(1)

    # Black
    temp_bp = black_pawns
    while temp_bp:
        sq = get_lsb_index(temp_bp)
        score -= (eg_pawn_val + get_2d_val(theta, IDX_PST_EG, 0, 64, sq^56))
        temp_bp &= temp_bp - np.uint64(1)

    score += get_2d_val(theta, IDX_PST_EG, 5, 64, white_king_sq)
    score -= get_2d_val(theta, IDX_PST_EG, 5, 64, black_king_sq^56)

    _, eg_struct = evaluate_pawn_structure(piece_bbs, theta)
    score += eg_struct

    # Proximity
    temp_pawns = white_pawns | black_pawns
    while temp_pawns:
        sq = get_lsb_index(temp_pawns)
        score += (KING_TROPISM_MAX_DISTANCE - MANHATTAN_DISTANCE[white_king_sq, sq]) * 5
        temp_pawns &= temp_pawns - np.uint64(1)

    temp_pawns = white_pawns | black_pawns
    while temp_pawns:
        sq = get_lsb_index(temp_pawns)
        score -= (KING_TROPISM_MAX_DISTANCE - MANHATTAN_DISTANCE[black_king_sq, sq]) * 5
        temp_pawns &= temp_pawns - np.uint64(1)

    return score

@numba.njit(numba.int32(piece_bbs_signature, occupancy_bbs_signature, game_state_signature, numba.float64[:]), cache=True, boundscheck=False, fastmath=True)
def evaluate_position_tunable(piece_bbs, occupancy_bbs, game_state, theta):
    # King Pawn Endgame
    all_pieces_except_pawns_and_kings = (
        piece_bbs[1] | piece_bbs[2] | piece_bbs[3] | piece_bbs[4] |
        piece_bbs[7] | piece_bbs[8] | piece_bbs[9] | piece_bbs[10]
    )
    if all_pieces_except_pawns_and_kings == 0:
        score = _evaluate_king_pawn_endgame(piece_bbs, theta)
        return score if game_state[0] == 0 else -score

    side_to_move = game_state[0]

    # Phase
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

    mg_score = np.int32(0)
    eg_score = np.int32(0)

    # Material
    for pt in range(6):
        cnt_w = count_bits(piece_bbs[pt])
        cnt_b = count_bits(piece_bbs[pt+6])
        val_mg = get_array_val(theta, IDX_MG_MATERIAL, pt)
        val_eg = get_array_val(theta, IDX_EG_MATERIAL, pt)

        mg_score += (cnt_w - cnt_b) * val_mg
        eg_score += (cnt_w - cnt_b) * val_eg

    # PST
    for pt in range(6):
        # White
        bb = piece_bbs[pt]
        while bb:
            sq = get_lsb_index(bb)
            mg_score += get_2d_val(theta, IDX_PST_MG, pt, 64, sq)
            eg_score += get_2d_val(theta, IDX_PST_EG, pt, 64, sq)
            bb &= bb - np.uint64(1)
        # Black
        bb = piece_bbs[pt+6]
        while bb:
            sq = get_lsb_index(bb)
            mg_score -= get_2d_val(theta, IDX_PST_MG, pt, 64, sq^56)
            eg_score -= get_2d_val(theta, IDX_PST_EG, pt, 64, sq^56)
            bb &= bb - np.uint64(1)

    # Sub modules
    mks, eks = evaluate_king_safety(piece_bbs, occupancy_bbs, theta)
    mg_score += mks; eg_score += eks

    mps, eps = evaluate_pawn_structure(piece_bbs, theta)
    mg_score += mps; eg_score += eps

    mpc, epc = evaluate_piece_coordination(piece_bbs, theta)
    mg_score += mpc; eg_score += epc

    mmb, emb = evaluate_mobility(piece_bbs, occupancy_bbs, theta)
    mg_score += mmb; eg_score += emb

    final_score = (mg_score * phase + eg_score * (MAX_PHASE - phase)) // MAX_PHASE

    if phase > INITIATIVE_PHASE_THRESHOLD:
        final_score += get_int(theta, IDX_INITIATIVE)

    if side_to_move == 0:
        return np.int32(final_score)
    else:
        return np.int32(-final_score)
