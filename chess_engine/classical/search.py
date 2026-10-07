# chess_engine/classical/search.py
import time
import numba
import numpy as np
import threading

from chess_engine.classical.evaluation import _evaluate_position_jit

from chess_engine.classical.move_generator import (
    _is_in_check_jit, has_sufficient_material,
    generate_pseudo_legal_moves_buffer, generate_pseudo_legal_captures_buffer,
    generate_pseudo_legal_quiets_buffer,
    is_square_attacked, is_square_attacked_with_occ, is_move_pseudo_legal,
    get_blockers_for_king,
    PAWN_ATTACKS, KNIGHT_ATTACKS, KING_ATTACKS, get_bishop_attacks, get_rook_attacks
)
from chess_engine.classical.bitboard_utils import (
    get_lsb_index, count_bits, LINE_BB, SQUARES_BETWEEN,
)
from chess_engine.classical.board_operations import make_move, unmake_move, make_null_move
from chess_engine.classical.move import (
    get_to_square, get_from_square, get_special_move_flag, get_promotion_piece,
    SPECIAL_MOVE_FLAG_PROMOTION, SPECIAL_MOVE_FLAG_EN_PASSANT, SPECIAL_MOVE_FLAG_CASTLING,
    PROMO_KNIGHT
)
from chess_engine.classical.constants import (
    BB_SQUARES, MG_MATERIAL_VALUES, INFINITY, MAX_QUIESCENCE_DEPTH,
    ASPIRATION_WINDOW_MIN, ASPIRATION_WINDOW_BASE, ASPIRATION_WINDOW_SCALE_DIV,
    MAX_PLY, LMR_MIN_DEPTH, LMR_MIN_QUIET_MOVE_INDEX,
    ENABLE_SEE_IN_QUIESCENCE, MATE_SCORE, MATE_IN_MAX_PLY, VALUE_KNOWN_WIN, NO_MOVE,
    RFP_MAX_DEPTH, RFP_NO_TT_PENALTY,
    ENABLE_DELTA_PRUNING, LMP_MOVE_COUNT,
    PROBCUT_R, PROBCUT_R_IMPROVING, PROBCUT_R_NOT_IMPROVING,
    PROBCUT_SF_BASE, PROBCUT_SF_IMPROVING, PROBCUT_TT_DEPTH_OFFSET, PROBCUT_MIN_DEPTH,
    ENABLE_PROBCUT_TT_SHORTCUT, PROBCUT_TT_SHORTCUT_MARGIN,
    ENABLE_IIR,
    ENABLE_ALPHA_RAISE_DEPTH_REDUCTION, ENABLE_FOLLOW_PV, ENABLE_HIST_LMR_DEPTH_PRUNING,
    ENABLE_CAPTURE_FUTILITY,
    CAP_FP_MAX_LMR_DEPTH, CAP_FP_BASE, CAP_FP_LMR_MULT, CAP_FP_CAPTHIST_NUM, CAP_FP_CAPTHIST_DEN,
    CAP_SEE_CAPTHIST_NUM, CAP_SEE_CAPTHIST_DEN,
    ALPHA_RAISE_DEPTH_LO, ALPHA_RAISE_DEPTH_HI, ALPHA_RAISE_DEPTH_DELTA,
    ALPHA_RAISE_MIN_IMPROVEMENT,
    STAGE_TT_MOVE, STAGE_GEN_CAPTURES, STAGE_GOOD_CAPTURES,
    STAGE_GEN_QUIETS, STAGE_GOOD_QUIETS, STAGE_BAD_CAPTURES,
    STAGE_BAD_QUIETS, STAGE_DONE,
    ENABLE_SINGULAR_EXTENSIONS, MIN_SINGULAR_DEPTH, SINGULAR_DOUBLE_EXT_MULTIPLIER,
    NMP_MIN_SIDE_NON_PAWNS, LOW_MATERIAL_PRUNING_PIECE_COUNT,
    ENABLE_HISTORY_PRUNING, PRUNING_SHALLOW_DEPTH, PRUNING_HISTORY_THRESHOLD,
    ENABLE_CONTINUATION_HISTORY_PRUNING, PRUNING_CONTINUATION_THRESHOLD,
    WHITE, BLACK, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, PAWN_KEY_INDEX, MINOR_KEY_INDEX, NON_PAWN_KEY_WHITE_INDEX, NON_PAWN_KEY_BLACK_INDEX, CORRECTION_HISTORY_MASK, CORRECTION_HISTORY_LIMIT, CORRECTION_HISTORY_DIVISOR,
    CORRECTION_HISTORY_PAWN_WEIGHT, CORRECTION_HISTORY_MINOR_WEIGHT, CORRECTION_HISTORY_NON_PAWN_WEIGHT, CORRECTION_HISTORY_UPDATE_DEPTH, SCORE_MIN, SCORE_MAX,
    CORRECTION_HISTORY_CNTCV_FALLBACK, CORRECTION_HISTORY_OUTER_SCALE,
    HISTORY_MAX_CONTINUATION, HISTORY_MAX_PAWN, PAWN_HISTORY_MASK, LOW_PLY_HISTORY_MAX,
    GOOD_QUIET_THRESHOLD,
    FIFTY_MOVE_RULE_LIMIT, FIFTY_MOVE_SCALE_THRESHOLD, FIFTY_MOVE_MAX_SCALE,
    HISTORY_QUIET_MALUS_SCALE_NUM, HISTORY_QUIET_MALUS_SCALE_DEN,
    HISTORY_QUIET_MALUS_DECAY_NUM, HISTORY_QUIET_MALUS_DECAY_DEN,
    HISTORY_LPH_SCALE_NUM, HISTORY_LPH_SCALE_DEN,
    ENABLE_MATE_DISTANCE_PRUNING,
    LMR_TTPV_INCREASE, LMR_TTPV_DECREASE_BASE, LMR_TTPV_PV_BONUS,
    LMR_LATE_CAPTURE_BONUS,
    LMR_CUTOFF_CNT_BASE, LMR_CUTOFF_CNT_EXTRA, LMR_ALLNODE_EXTRA,
    LMR_CORRECTION_DIVISOR, LMR_ALLNODE_SCALE_NUM, LMR_ALLNODE_SCALE_DENOM_BASE,
    LMR_ALLNODE_SCALE_DENOM_OFFSET,
    # Search internals (margins / formula coeffs)
    HCE_SCALE_NUM, HCE_SCALE_DEN,
    HINDSIGHT_REDUCE_MIN_PRIOR, HINDSIGHT_INCREASE_MIN_PRIOR, HINDSIGHT_EVAL_SUM_MARGIN,
    DISSONANCE_TT_DEPTH_SLACK, RAZOR_DISSONANCE_MAX, FP_DISSONANCE_THRESHOLD,
    RAZORING_MAX_DEPTH, RAZORING_COEFF,
    FULL_EVAL_KNOWN_WIN_MARGIN,
    TT_CUTOFF_HALFMOVE_MAX, TT_CONSISTENCY_MIN_DEPTH, TT_DEEP_VERIFY_DEPTH, TT_PENALIZE_MIN_DEPTH,
    HISTORY_BONUS_SCALE, HISTORY_BONUS_CAP, HISTORY_MALUS_CAP, HISTORY_NODE_WIDTH_DIV,
    CAPTURE_MALUS_SCALE_NUM, CAPTURE_MALUS_SCALE_DEN, FAIL_LOW_MIN_LEGAL_MOVES,
    NMP_MIN_DEPTH, NMP_LEGACY_DEPTH_COEF, NMP_LEGACY_IMPROVING_COEF, NMP_LEGACY_BASE,
    NMP_SF_MARGIN_DEPTH_COEF, NMP_SF_MARGIN_BASE, NMP_SF_MARGIN_IMPROVING,
    NMP_EVAL_MARGIN_DIV, NMP_EVAL_MARGIN_MAX, NMP_SF_R_BASE, NMP_SF_R_DEPTH, NMP_SF_R_DIV,
    NMP_LEGACY_R_BASE, NMP_LEGACY_R_DEPTH_DIV,
    IIR_MIN_DEPTH, FP_MAX_DEPTH, LMP_MIN_LIMIT, ENABLE_CHECK_EXTENSION, CHECK_EXT_NON_PV_MAX_DEPTH, STOP_CHECK_MASK,
    SINGULAR_TT_DEPTH_SLACK, SINGULAR_MARGIN_BASE, SINGULAR_TTPV_BONUS, SINGULAR_MARGIN_DIV,
    SINGULAR_DOUBLE_EXT_MIN_DEPTH,
    LMR_TT_SCORE_GT_ALPHA, LMR_TT_DEPTH_GE, LMR_TT_DEPTH_GE_CUT,
    LMR_ADVANCED_PAWN_RELIEF, LMR_ADVANCED_PAWN_WHITE_RANK, LMR_ADVANCED_PAWN_BLACK_RANK,
    LMR_LATE_CAPTURE_MAX_DEPTH, LMR_LATE_CAPTURE_MIN_MC, LMR_TTMOVE_CUTNODE_EXTRA,
    LMR_R_FLOOR, LMR_D_MAX_EXTRA, LMR_CAPTURE_VICTIM_SCALE, LMR_CAPTURE_VICTIM_DIV,
    LMR_STAT_SCORE_DIV, LMR_RESEARCH_DEEPER_MARGIN, LMR_RESEARCH_SHALLOWER_MARGIN,
    QS_STANDPAT_SMOOTH_A, QS_STANDPAT_SMOOTH_B, QS_FAILHIGH_SMOOTH_A, QS_FAILHIGH_SMOOTH_B, QS_SMOOTH_DIV,
    CORRECTION_CNTCV_WEIGHT, CORRECTION_BONUS_HAS_MOVE_DIV, CORRECTION_BONUS_NO_MOVE_DIV,
    CORRECTION_MINOR_SCALE_NUM, CORRECTION_MINOR_SCALE_DEN, CORRECTION_NP_SCALE_NUM, CORRECTION_NP_SCALE_DEN,
    CORRECTION_CONT_2PLY_NUM, CORRECTION_CONT_4PLY_NUM, CORRECTION_CONT_SCALE_DEN,
    DIAG_CUT_NODES, DIAG_CUT_FIRST, DIAG_CUT_MOVE_SUM, DIAG_LMR_TRY, DIAG_LMR_RESEARCH,
    DIAG_TT_CUT, DIAG_NMP_CUT, DIAG_RFP_CUT, DIAG_RAZOR_CUT, DIAG_LMP_SKIP, DIAG_FP_SKIP, DIAG_PROBCUT,
    DIAG_NMP_ELIGIBLE, DIAG_NMP_GATE_PASS, DIAG_NMP_NULL_TRY, DIAG_NMP_NULL_FH,
    DIAG_NMP_VERIFY_TRY, DIAG_NMP_VERIFY_FAIL,
    DIAG_HIST_QUIET_BONUS, DIAG_HIST_QUIET_MALUS, DIAG_HIST_CAP_BONUS, DIAG_HIST_CAP_MALUS,
    DIAG_HIST_CONT_UPD, DIAG_HIST_FAIL_LOW, DIAG_HIST_TT_HIT,
    DIAG_HIST_PRUNE, DIAG_HIST_AGE, DIAG_HIST_PIECE_TO_UPD,
    DIAG_HIST_PAWN_UPD, DIAG_HIST_LPH_UPD,
    DIAG_Q_TT_CUT, DIAG_Q_STANDPAT_CUT, DIAG_Q_IN_CHECK, DIAG_Q_MOVES,
    DIAG_Q_DELTA_SKIP, DIAG_Q_SEE_SKIP, DIAG_Q_ILLEGAL_SKIP, DIAG_Q_BETA_CUT,
    DIAG_SEE_CAP_SKIP, DIAG_SEE_QUIET_SKIP, DIAG_CHECK_EXT,
    DIAG_SINGULAR_TRY, DIAG_SINGULAR_EXT, DIAG_DOUBLE_EXT,
    DIAG_Q_SEE_TRY, DIAG_PICKER_SEE_TRY, DIAG_MAIN_CAP_SEE_TRY,
    DIAG_MAIN_QUIET_SEE_TRY, DIAG_LMR_CAP_SEE_TRY,
    DIAG_PROBCUT_SEE_TRY, DIAG_CHECK_GATE_SEE_TRY, DIAG_Q_CHECK_PROTECT,
    DIAG_Q_EVASION_PREFILTER, DIAG_MAIN_CHECK_SEE_ROUTE,
    DIAG_MAIN_QUIET_SEE_PRECHECK, DIAG_MAIN_QUIET_PRE_PIN_ILLEGAL,
    DIAG_MAIN_QUIET_PRE_KING_ILLEGAL, DIAG_MAIN_QUIET_SEE_PASS,
    DIAG_MAIN_QUIET_POST_PIN_ILLEGAL, DIAG_MAIN_QUIET_POST_KING_ILLEGAL,
    DIAG_MAIN_QUIET_FALLBACK_TRY, DIAG_MAIN_QUIET_FALLBACK_REJECT,
    DIAG_MAIN_GEOMETRY_NODE, DIAG_MAIN_SCORE_GEOMETRY_REBUILD,
    DIAG_ASPIRATION_DEPTH_SLOTS, DIAG_ASPIRATION_ITERATIONS_BASE,
    DIAG_ASPIRATION_FAIL_LOW_BASE, DIAG_ASPIRATION_FAIL_HIGH_BASE,
    DIAG_ASPIRATION_RESEARCH_BASE, DIAG_ASPIRATION_MAX_DELTA_BASE,
    DIAG_ASPIRATION_WASTED_NODES_BASE,
    DIAG_Q_CAP_NONCHECK, DIAG_Q_CAP_CHECK, DIAG_Q_CAP_FORCING,
    DIAG_Q_CAP_FORCING_MOVES, DIAG_MAIN_IN_CHECK_NODE,
    DIAG_MAIN_IN_CHECK_PICK_TRY, DIAG_MAIN_IN_CHECK_PRE_ILLEGAL,
    DIAG_MAIN_IN_CHECK_PRE_PASS, DIAG_MAIN_IN_CHECK_FALLBACK,
    DIAG_TT_VERIFY_TRY, DIAG_TT_VERIFY_PASS, DIAG_TT_VERIFY_REJECT,
    DIAG_TT_VERIFY_SAVED_CUT, DIAG_TT_VERIFY_SKIP,
    MOVE_ORDER_SOURCE_TT, MOVE_ORDER_SOURCE_GOOD_CAPTURE,
    MOVE_ORDER_SOURCE_GOOD_QUIET, MOVE_ORDER_SOURCE_BAD_CAPTURE,
    MOVE_ORDER_SOURCE_BAD_QUIET, MOVE_ORDER_SOURCE_OTHER,
    DIAG_ORDER_SOURCE_COUNT, DIAG_ORDER_TRY_BASE, DIAG_ORDER_CUT_BASE,
    DIAG_ORDER_KILLER1_TRY, DIAG_ORDER_KILLER1_CUT,
    DIAG_ORDER_KILLER2_TRY, DIAG_ORDER_KILLER2_CUT,
    DIAG_ORDER_COUNTER_TRY, DIAG_ORDER_COUNTER_CUT,
    DIAG_ORDER_QUIET_CUT, DIAG_ORDER_QUIET_CUT_RANK_SUM,
    DIAG_ORDER_NODES_BEFORE_CUT_SUM,
    DIAG_LMR_R1_TRY, DIAG_LMR_R2_TRY, DIAG_LMR_R3P_TRY,
    DIAG_LMR_R1_FAIL_HIGH, DIAG_LMR_R2_FAIL_HIGH,
    DIAG_LMR_R3P_FAIL_HIGH, DIAG_LMR_QUIET_TRY,
    DIAG_LMR_QUIET_FAIL_HIGH, DIAG_LMR_CAPTURE_TRY,
    DIAG_LMR_CAPTURE_FAIL_HIGH, DIAG_LMR_RESEARCH_KEEP,
    DIAG_LMR_RESEARCH_REJECT, DIAG_LMR_RESEARCH_DEEPER,
    DIAG_LMR_RESEARCH_SHALLOWER, DIAG_LMR_FAIL_HIGH_UNVERIFIED,
    DIAG_POST_LMR_BONUS_SAMPLE, DIAG_POST_LMR_MALUS_SAMPLE,
    DIAG_POST_LMR_UNVERIFIED_SAMPLE,
    DIAG_POST_LMR_BONUS_BUTTERFLY_NONNEG,
    DIAG_POST_LMR_MALUS_BUTTERFLY_NONNEG,
    DIAG_POST_LMR_BONUS_STAT_NONNEG,
    DIAG_POST_LMR_MALUS_STAT_NONNEG,
    HIST_PRUNE_LAYER_COUNT,
    DIAG_HP_NEG_COUNT_BASE, DIAG_HP_NEG_ABS_SUM_BASE,
    DIAG_HP_DECISIVE_BASE, DIAG_HP_DOMINANT_BASE,
    DIAG_HP_CONT_COALITION_DECISIVE, DIAG_HP_NONCONT_ALREADY_PRUNES,
    DIAG_HP_LMR_ELIGIBLE, DIAG_HP_MAIN_GATE_NEAR_ZERO,
    DIAG_HP_MARGIN_NEAR, DIAG_HP_MARGIN_MID, DIAG_HP_MARGIN_FAR,
    DIAG_HP_MARGIN_SUM, DIAG_HP_DEPTH_SUM, DIAG_HP_MOVE_INDEX_SUM,
    DIAG_HP_CONT1_CAP_HIT, DIAG_HP_CONT1_CAP_RELIEF_SUM,
    DIAG_HP_CONT1_CAP_RESCUE, HISTORY_PRUNE_CONT1_FLOOR,
    HISTORY_MAX_MAIN, HISTORY_MAX_BUTTERFLY, HISTORY_MAX_CAPTURE, HISTORY_MAX_CONTINUATION,
    HISTORY_MAX_PAWN,
    TUNE_RFP_MULT, TUNE_RAZOR_COEFF, TUNE_FP_BASE, TUNE_FP_MULT,
    TUNE_SEE_CAP_MARGIN, TUNE_SEE_QUIET_MARGIN, TUNE_LMR_BASE_OFFSET, TUNE_LMR_HIST_SCALE,
    TUNE_DELTA_MARGIN, TUNE_PROBCUT_MARGIN, TUNE_BAD_CAP_BONUS, TUNE_GOOD_CAP_RELIEF,
    TUNE_LMP_SCALE, TUNE_KILLER_RELIEF, TUNE_QS_SEE,
    TUNE_LMR_CUTNODE, TUNE_LMR_NO_TTMOVE, TUNE_LMR_TTCAP, TUNE_LMR_MC_FACTOR, TUNE_LMR_TTMOVE_RED,
    TUNE_NMP_SCOPE, TUNE_NMP_GATE, TUNE_NMP_R, TUNE_PROBCUT_STYLE,
    TUNE_LMR_TABLE_SCALE, TUNE_LMR_NOT_IMP,
    TUNE_NMP_G_BASE, TUNE_NMP_G_DEPTH, TUNE_NMP_G_IMP, TUNE_NMP_NEED_BETA,
    TUNE_NMP_R_BASE, TUNE_NMP_R_DIV, TUNE_NMP_VERIFY_D, TUNE_NMP_SCOPE_MIND,
    PINNED_UNCOMPUTED_SENTINEL,
)
from chess_engine.classical.bitboard_utils import find_piece_type_on_square_side
from chess_engine.classical.debug_utils import log_info
from chess_engine.classical.see import _see_ge_jit, get_pinned_pieces
from chess_engine.classical.transposition_table import (
    probe_tt, store_tt,
    TT_FLAG_NONE, TT_FLAG_EXACT, TT_FLAG_ALPHA, TT_FLAG_BETA,
    hashfull, penalize_tt
)
from chess_engine.classical.zobrist import get_tt_key  # GHI protection: halfmove-aware TT key
from chess_engine.classical.engine_types import (
    piece_bbs_signature, occupancy_bbs_signature, game_state_signature,
    SearchContext, search_context_type
)
from chess_engine.classical.move import move_to_uci

from chess_engine.classical.search_heuristics import (
    update_history, update_butterfly_history, update_capture_history,
    update_continuation_history, update_pawn_history,
    score_captures_lazy, score_captures_with_tt_lazy,
    score_quiets, update_quiet_stats_on_tt_hit, partial_insertion_sort_moves,
    get_quiet_stat_score, get_quiet_stat_components,
    get_continuation_pruning_score,
    compute_lmr_reduction_1024, get_lmr_stat_score,
)


quiescence_search_return_type = numba.types.Tuple([
    numba.int32, numba.uint64
])

@numba.njit(inline='always')
def trunc_div(n, d):
    return n // d if n >= 0 else -(-n // d)


# ---------------------------------------------------------------------------
# Score-band helpers (SEARCH_EVAL_INTERFACE PR-A: P0)
# SF11: do not treat VALUE_KNOWN_WIN as proven; SF18: is_win / is_loss spirit.
# ---------------------------------------------------------------------------
@numba.njit(cache=True, inline='always')
def is_win(v):
    """Static / search score is a known-win band or stronger (incl. mate)."""
    return v >= VALUE_KNOWN_WIN


@numba.njit(cache=True, inline='always')
def is_loss(v):
    return v <= -VALUE_KNOWN_WIN


@numba.njit(cache=True, inline='always')
def is_decisive(v):
    """True mate-distance scores (search-proven line), not merely known-win eval."""
    return abs(v) >= MATE_IN_MAX_PLY


@numba.njit(cache=True, inline='always')
def compute_quiet_pruning_lmr_depth(
    depth, move_count, improving, search_context, ply, from_sq, to_sq, piece_type
):
    """
    History-influenced reduced depth for quiet FP / quiet SEE (SF Step 14 spirit).

    lmrDepth ≈ (depth - 1) - base_LMR_plies, then history reduces reduction
    (good history → higher lmrDepth → harder to prune). Falls back to depth if
    LMR would not apply. piece_type is 0..11 side-absolute (-1 → no hist term).
    """
    new_depth = depth - 1
    if new_depth < 0:
        new_depth = 0
    if (not ENABLE_HIST_LMR_DEPTH_PRUNING) or depth < LMR_MIN_DEPTH or piece_type < 0:
        return depth if depth > 0 else 0

    mc = move_count if move_count >= 1 else 1
    r = compute_lmr_reduction_1024(
        depth,
        mc,
        improving,
        search_context.tune[TUNE_LMR_TABLE_SCALE],
        search_context.tune[TUNE_LMR_NOT_IMP],
    )
    r += search_context.tune[TUNE_LMR_BASE_OFFSET]
    hist = get_lmr_stat_score(search_context, ply, from_sq, to_sq, piece_type)
    r -= trunc_div(hist * search_context.tune[TUNE_LMR_HIST_SCALE], LMR_STAT_SCORE_DIV)
    if r < 0:
        r = 0
    lmr_d = new_depth - trunc_div(r, 1024)
    if lmr_d < 0:
        lmr_d = 0
    return lmr_d


@numba.njit(cache=True, nogil=True, boundscheck=False, fastmath=True)
def apply_correction_history_score(game_state, search_context, raw_static_eval, ply):
    pawn_key = game_state[PAWN_KEY_INDEX]
    minor_key = game_state[MINOR_KEY_INDEX]
    np_white_key = game_state[NON_PAWN_KEY_WHITE_INDEX]
    np_black_key = game_state[NON_PAWN_KEY_BLACK_INDEX]

    global_pawn = search_context.pawn_correction_history[pawn_key & CORRECTION_HISTORY_MASK]
    global_minor = search_context.minor_correction_history[minor_key & CORRECTION_HISTORY_MASK]
    side = game_state[0]

    # Pawn and Minor are stored in white-absolute perspective.
    # Non-Pawn is stored in side-exclusive tables (white table for WHITE, black table for BLACK) with side-relative values.
    if side == WHITE:
        correction_sum = (global_pawn * CORRECTION_HISTORY_PAWN_WEIGHT +
                         global_minor * CORRECTION_HISTORY_MINOR_WEIGHT +
                         search_context.non_pawn_correction_history_white[np_white_key & CORRECTION_HISTORY_MASK] * CORRECTION_HISTORY_NON_PAWN_WEIGHT)
    else:
        correction_sum = (-global_pawn * CORRECTION_HISTORY_PAWN_WEIGHT -
                         global_minor * CORRECTION_HISTORY_MINOR_WEIGHT +
                         search_context.non_pawn_correction_history_black[np_black_key & CORRECTION_HISTORY_MASK] * CORRECTION_HISTORY_NON_PAWN_WEIGHT)

    # Continuation Correction Heuristic (cntcv)
    # Fallback to 0 if previous move is invalid (matching old baseline)
    cntcv = CORRECTION_HISTORY_CNTCV_FALLBACK
    if ply > 0:
        m_prev = search_context.move_stack[ply - 1]
        pc_prev = search_context.piece_stack[ply - 1]
        if m_prev != NO_MOVE and pc_prev != -1:
            to_prev = get_to_square(m_prev)
            val_2 = 0
            if ply >= 2:
                m_2 = search_context.move_stack[ply - 2]
                pc_2 = search_context.piece_stack[ply - 2]
                if m_2 != NO_MOVE and pc_2 != -1:
                    val_2 = search_context.continuation_correction_history[pc_2, get_to_square(m_2), pc_prev, to_prev]

            val_4 = 0
            if ply >= 4:
                m_4 = search_context.move_stack[ply - 4]
                pc_4 = search_context.piece_stack[ply - 4]
                if m_4 != NO_MOVE and pc_4 != -1:
                    val_4 = search_context.continuation_correction_history[pc_4, get_to_square(m_4), pc_prev, to_prev]

            # HCE-specific tuning (larger than SF18's 8363)
            cntcv = CORRECTION_CNTCV_WEIGHT * (val_2 + val_4)


    correction_sum += cntcv

    # PR-C: do not apply correction onto already known-win / decisive static
    # (specialized EG, mate); mirrors SF18 clamp spirit for HCE scale.
    if is_win(raw_static_eval) or is_loss(raw_static_eval) or is_decisive(raw_static_eval):
        return np.int32(raw_static_eval)

    corrected = np.int32(raw_static_eval) + np.int32(trunc_div(correction_sum, CORRECTION_HISTORY_DIVISOR))
    # Clamp so correction cannot push ordinary scores into known-win band
    if corrected >= VALUE_KNOWN_WIN:
        corrected = VALUE_KNOWN_WIN - 1
    elif corrected <= -VALUE_KNOWN_WIN:
        corrected = -VALUE_KNOWN_WIN + 1
    return np.int32(min(max(corrected, SCORE_MIN), SCORE_MAX))

full_static_eval_return_type = numba.types.Tuple((numba.int32, numba.int32, numba.boolean, numba.uint64, numba.uint64))

@numba.njit(cache=True, nogil=True, boundscheck=False, fastmath=True)
def compute_full_corrected_static_eval(piece_bbs, occupancy_bbs, game_state, search_context, ply):
    raw_eval, pinned_white, pinned_black = _evaluate_position_jit(piece_bbs, occupancy_bbs, game_state, False, search_context)
    if pinned_white == PINNED_UNCOMPUTED_SENTINEL:
        pinned_white = get_pinned_pieces(piece_bbs, occupancy_bbs, WHITE)
        pinned_black = get_pinned_pieces(piece_bbs, occupancy_bbs, BLACK)
    corrected_eval = apply_correction_history_score(game_state, search_context, raw_eval, ply)
    search_context.static_eval_stack[ply] = corrected_eval

    improving = False
    if ply >= 2:
        ref_eval = search_context.static_eval_stack[ply - 2]
        # V3 3.5: Fallback to ply-4 when ply-2 was null move (static_eval = -INFINITY)
        if ref_eval == -INFINITY and ply >= 4:
            ref_eval = search_context.static_eval_stack[ply - 4]
        if ref_eval != -INFINITY and corrected_eval > ref_eval:
            improving = True

    return raw_eval, corrected_eval, improving, pinned_white, pinned_black

# Runtime typed constants for recursive search calls (defined before QS/_search).
# Literal[bool]/int64 specializations each clone ~5.5MB of IR.
# MUST NOT inline: inlining re-materializes literals / wider ints at the call site.
@numba.njit(numba.boolean(), cache=True, inline='never')
def _b_false():
    return False


@numba.njit(numba.boolean(), cache=True, inline='never')
def _b_true():
    return True


@numba.njit(numba.int32(numba.int64), cache=True, inline='never')
def _i32(x):
    return np.int32(x)


@numba.njit(numba.uint16(numba.int64), cache=True, inline='never')
def _u16(x):
    return np.uint16(x)


@numba.njit(cache=True, inline='always')
def _diag_add(search_context, index, value=1):
    """Increment an opt-in search diagnostic without taxing production writes."""
    if search_context.diag_enabled:
        search_context.diag_stats[index] += np.uint64(value)


@numba.njit(cache=True, inline='always')
def _diag_move_source(move, tt_move, picker_stage):
    """Map the returned picker state to a stable P4 diagnostic source."""
    if move == tt_move:
        return np.int32(MOVE_ORDER_SOURCE_TT)
    if picker_stage == STAGE_GOOD_CAPTURES:
        return np.int32(MOVE_ORDER_SOURCE_GOOD_CAPTURE)
    if picker_stage == STAGE_GOOD_QUIETS:
        return np.int32(MOVE_ORDER_SOURCE_GOOD_QUIET)
    if picker_stage == STAGE_BAD_CAPTURES:
        return np.int32(MOVE_ORDER_SOURCE_BAD_CAPTURE)
    if picker_stage == STAGE_BAD_QUIETS:
        return np.int32(MOVE_ORDER_SOURCE_BAD_QUIET)
    return np.int32(MOVE_ORDER_SOURCE_OTHER)


@numba.njit(cache=True, boundscheck=False, fastmath=True, inline='always')
def _qsearch_ordinary_move_gives_check(
    piece_bbs, move, side_to_move, their_king_bb,
    check_sq_pawn, check_sq_knight, check_sq_bishop,
    check_sq_rook, check_sq_queen, blockers_for_their_king,
):
    """Pre-make gives-check test for non-EP, non-promotion qsearch moves."""
    from_sq = get_from_square(move)
    to_sq = get_to_square(move)
    moved_piece = find_piece_type_on_square_side(
        piece_bbs, from_sq, side_to_move,
    )
    if moved_piece < 0:
        return False

    to_bb = BB_SQUARES[to_sq]
    base_piece = moved_piece % 6
    if base_piece == PAWN:
        if check_sq_pawn & to_bb:
            return True
    elif base_piece == KNIGHT:
        if check_sq_knight & to_bb:
            return True
    elif base_piece == BISHOP:
        if check_sq_bishop & to_bb:
            return True
    elif base_piece == ROOK:
        if check_sq_rook & to_bb:
            return True
    elif base_piece == QUEEN:
        if check_sq_queen & to_bb:
            return True

    if blockers_for_their_king & BB_SQUARES[from_sq]:
        if not (LINE_BB[from_sq, to_sq] & their_king_bb):
            return True
    return False


@numba.njit(cache=True, boundscheck=False, fastmath=True, inline='always')
def _classify_main_in_check_move(
    piece_bbs, occupancy_bbs, move, our_king_sq, our_king_bb,
    opponent_side, pinned_us, checker_count, evasion_targets,
):
    """Classify a main-search in-check picker candidate without make/unmake.

    Return values are intentionally small and stable for opt-in diagnostics:
      0 = conservative fallback (EP or unavailable checker geometry),
      1 = provably illegal from king/check-ray geometry,
      2 = passes the same pre-make geometry used by P0.

    P3 now uses this classifier only for opt-in shadow diagnostics.  The
    classification is deliberately conservative: EP/unknown geometry returns
    fallback, while production keeps the baseline picker and legality path.
    """
    from_sq = get_from_square(move)
    to_sq = get_to_square(move)
    special_flag = get_special_move_flag(move)

    if from_sq == our_king_sq:
        # Castling cannot evade an already active check.
        if special_flag == SPECIAL_MOVE_FLAG_CASTLING:
            return 1
        occ_without_king = occupancy_bbs[2] ^ our_king_bb
        if is_square_attacked_with_occ(
                piece_bbs, to_sq, occ_without_king, opponent_side):
            return 1
        return 2

    # EP changes the captured square and can reveal a discovered check.
    if special_flag == SPECIAL_MOVE_FLAG_EN_PASSANT:
        return 0

    # A stale/ambiguous checker set is conservatively left for post-make.
    if checker_count <= 0:
        return 0
    # Double check permits king moves only.
    if checker_count > 1:
        return 1

    # A single-check evasion must capture the checker or interpose on its ray.
    if not (evasion_targets & BB_SQUARES[to_sq]):
        return 1
    from_bb = BB_SQUARES[from_sq]
    if (pinned_us & from_bb) and not (LINE_BB[from_sq, to_sq] & our_king_bb):
        return 1
    return 2


@numba.njit(cache=True, boundscheck=False, fastmath=True, inline='always')
def _classify_and_diag_main_in_check_move(
    piece_bbs, occupancy_bbs, move, our_king_sq, our_king_bb,
    opponent_side, pinned_us, checker_count, evasion_targets, search_context,
):
    """Classify one picker candidate and publish the opt-in P3 funnel."""
    classification = _classify_main_in_check_move(
        piece_bbs, occupancy_bbs, move, our_king_sq, our_king_bb,
        opponent_side, pinned_us, checker_count, evasion_targets,
    )
    _diag_add(search_context, DIAG_MAIN_IN_CHECK_PICK_TRY)
    if classification == 1:
        _diag_add(search_context, DIAG_MAIN_IN_CHECK_PRE_ILLEGAL)
    elif classification == 2:
        _diag_add(search_context, DIAG_MAIN_IN_CHECK_PRE_PASS)
    else:
        _diag_add(search_context, DIAG_MAIN_IN_CHECK_FALLBACK)
    return classification


# cache=False: recursive + StructRef disk reload aborts (LLVM unresolved).
# Explicit signature forces one specialization (faster than multi-overload lazy).
# See JIT_COMPILE_CACHE_REPORT.md.
@numba.njit(quiescence_search_return_type(piece_bbs_signature, occupancy_bbs_signature, game_state_signature, numba.int32, numba.int32, numba.int32, search_context_type, numba.int32), cache=False, nogil=True, boundscheck=False, fastmath=True)
def quiescence_search(piece_bbs, occupancy_bbs, game_state, alpha, beta, ply, search_context, q_ply):
    q_nodes = np.uint64(1)
    search_context.nodes_searched += 1
    original_alpha = alpha
    best_move = NO_MOVE
    stand_pat = np.int32(32767)
    best_eval = np.int32(-32767)
    pinned_white = np.uint64(0)
    pinned_black = np.uint64(0)
    pinned_computed = False
    qs_check_info_computed = False
    their_king_bb_qs = np.uint64(0)
    check_sq_pawn_qs = np.uint64(0)
    check_sq_knight_qs = np.uint64(0)
    check_sq_bishop_qs = np.uint64(0)
    check_sq_rook_qs = np.uint64(0)
    check_sq_queen_qs = np.uint64(0)
    blockers_for_their_king_qs = np.uint64(0)

    # Publish progress and observe timer/node-limit stops at a bounded cadence.
    if (search_context.nodes_searched & STOP_CHECK_MASK) == 0:
        search_context.nodes_searched_array[0] = search_context.nodes_searched
        node_budget = search_context.nodes_searched_array[1]
        if node_budget > 0 and search_context.nodes_searched >= node_budget:
            search_context.stop_flag[0] = True
        if search_context.stop_flag[0]:
            return np.int32(0), q_nodes

    if ply >= MAX_PLY:
        eval_score, _, _ = _evaluate_position_jit(piece_bbs, occupancy_bbs, game_state, False, search_context)
        return np.int32(eval_score), q_nodes

    if not has_sufficient_material(piece_bbs):
        if not _is_in_check_jit(piece_bbs, occupancy_bbs, game_state):
            return np.int32(0), q_nodes

    # M4: TT Probe in QSearch — avoids re-evaluating positions already in TT
    zobrist_key = game_state[4]
    tt_key = get_tt_key(zobrist_key, int(game_state[3]))  # GHI: halfmove-aware key for TT
    tt_entry = probe_tt(search_context.transposition_table, tt_key)
    if tt_entry['flag'] != TT_FLAG_NONE:
        qs_tt_score = np.int32(tt_entry['score'])
        if qs_tt_score > MATE_IN_MAX_PLY: qs_tt_score -= ply
        elif qs_tt_score < -MATE_IN_MAX_PLY: qs_tt_score += ply
        
        should_cutoff = False
        if tt_entry['flag'] == TT_FLAG_EXACT:
            should_cutoff = True
        elif tt_entry['flag'] == TT_FLAG_ALPHA and qs_tt_score <= alpha:
            should_cutoff = True
        elif tt_entry['flag'] == TT_FLAG_BETA and qs_tt_score >= beta:
            should_cutoff = True
        if should_cutoff:
            _diag_add(search_context, DIAG_Q_TT_CUT)
            return np.int32(qs_tt_score), q_nodes

    is_currently_in_check = _is_in_check_jit(piece_bbs, occupancy_bbs, game_state)

    move_count = 0
    if is_currently_in_check:
        _diag_add(search_context, DIAG_Q_IN_CHECK)
        # If in check, we must evade. No stand_pat (can't stand pat in check).
        # H9 Fix: Add depth limit for check evasion in QSearch to prevent infinite loops.
        # Check evasions are very forcing, so we allow them to go deeper than normal QSearch.
        # (e.g., 2x MAX_QUIESCENCE_DEPTH)
        if q_ply >= MAX_QUIESCENCE_DEPTH * 2:
            if search_context.diag_enabled:
                _diag_add(search_context, DIAG_Q_CAP_CHECK)
                # The cap currently returns before normal move generation.
                # Probe only in diagnostic mode so we can tell whether the
                # bound is truncating any pseudo evasion candidates.
                cap_probe_count = generate_pseudo_legal_moves_buffer(
                    piece_bbs, occupancy_bbs, game_state,
                    search_context.moves_buffer, ply,
                )
                if cap_probe_count > 0:
                    _diag_add(search_context, DIAG_Q_CAP_FORCING)
                    _diag_add(search_context, DIAG_Q_CAP_FORCING_MOVES, cap_probe_count)
            eval_score, _, _ = _evaluate_position_jit(piece_bbs, occupancy_bbs, game_state, False, search_context)
            return np.int32(eval_score), q_nodes

        # We must generate ALL legal moves (evasions).
        move_count = generate_pseudo_legal_moves_buffer(piece_bbs, occupancy_bbs, game_state, search_context.moves_buffer, ply)
        if move_count == 0:
            # Checkmate
            return np.int32(-MATE_SCORE + ply), q_nodes
    else:
        # If not in check, we can stand pat (static evaluation)
        if q_ply >= MAX_QUIESCENCE_DEPTH:
            if search_context.diag_enabled:
                _diag_add(search_context, DIAG_Q_CAP_NONCHECK)
                # The cap currently returns before normal move generation.
                # This is a pseudo-capture upper bound, not a legality claim.
                cap_probe_count = generate_pseudo_legal_captures_buffer(
                    piece_bbs, occupancy_bbs, game_state,
                    search_context.moves_buffer, ply,
                )
                if cap_probe_count > 0:
                    _diag_add(search_context, DIAG_Q_CAP_FORCING)
                    _diag_add(search_context, DIAG_Q_CAP_FORCING_MOVES, cap_probe_count)
            eval_score, _, _ = _evaluate_position_jit(piece_bbs, occupancy_bbs, game_state, False, search_context)
            return np.int32(eval_score), q_nodes

        # --- QSearch TT static_eval probe ---
        if tt_entry['flag'] != TT_FLAG_NONE and tt_entry['static_eval'] != 32767:
            stand_pat = np.int32(tt_entry['static_eval'])
        else:
            stand_pat_val, p_w, p_b = _evaluate_position_jit(piece_bbs, occupancy_bbs, game_state, False, search_context)
            stand_pat = np.int32(stand_pat_val)
            if p_w != PINNED_UNCOMPUTED_SENTINEL:
                pinned_white = p_w
                pinned_black = p_b
                pinned_computed = True
        
        best_eval = stand_pat

        if stand_pat >= beta:
            _diag_add(search_context, DIAG_Q_STANDPAT_CUT)
            # B1: QSearch Stand-pat smoothing
            val = stand_pat
            if abs(val) < MATE_IN_MAX_PLY:
                val = np.int32(trunc_div(QS_STANDPAT_SMOOTH_A * val + QS_STANDPAT_SMOOTH_B * beta, QS_SMOOTH_DIV))
            
            tt_score = val
            if tt_score > MATE_IN_MAX_PLY: tt_score += ply
            elif tt_score < -MATE_IN_MAX_PLY: tt_score -= ply
            store_tt(search_context.transposition_table, tt_key, 0, tt_score, np.int16(stand_pat), TT_FLAG_BETA, NO_MOVE, search_context.tt_generation, False)
            return np.int32(val), q_nodes
        alpha = max(alpha, stand_pat)

        move_count = generate_pseudo_legal_captures_buffer(piece_bbs, occupancy_bbs, game_state, search_context.moves_buffer, ply)
        if move_count == 0:
            tt_score = np.int32(stand_pat)
            if tt_score > MATE_IN_MAX_PLY: tt_score += ply
            elif tt_score < -MATE_IN_MAX_PLY: tt_score -= ply
            store_tt(search_context.transposition_table, tt_key, 0, tt_score, np.int16(stand_pat), TT_FLAG_EXACT, NO_MOVE, search_context.tt_generation, False)
            return np.int32(stand_pat), q_nodes

    _diag_add(search_context, DIAG_Q_MOVES, move_count)

    # --- Sort moves in QSearch ---
    moves = search_context.moves_buffer[ply]
    scores = search_context.move_scores[ply]

    # B1: Use TT best move for QSearch ordering (since we already probe TT above)
    qs_tt_move = tt_entry['best_move'] if tt_entry['flag'] != TT_FLAG_NONE else NO_MOVE

    # P0 shadow diagnostics: classify the full pseudo-evasion set before
    # scoring, but never compact or reorder the production buffer.
    if is_currently_in_check and search_context.diag_enabled:
        shadow_side_qs = game_state[0]
        shadow_opponent_qs = np.uint8(1 - shadow_side_qs)
        shadow_king_bb_qs = (
            piece_bbs[5] if shadow_side_qs == WHITE else piece_bbs[11]
        )
        shadow_king_sq_qs = (
            get_lsb_index(shadow_king_bb_qs)
            if shadow_king_bb_qs else np.int8(0)
        )
        if not pinned_computed:
            pinned_white = get_pinned_pieces(
                piece_bbs, occupancy_bbs, WHITE
            )
            pinned_black = get_pinned_pieces(
                piece_bbs, occupancy_bbs, BLACK
            )
            pinned_computed = True
        shadow_pinned_qs = (
            pinned_white if shadow_side_qs == WHITE else pinned_black
        )
        shadow_offset_qs = 6 if shadow_opponent_qs == BLACK else 0
        shadow_occ_qs = occupancy_bbs[2]
        shadow_checkers_qs = np.uint64(0)
        shadow_checkers_qs |= (
            piece_bbs[shadow_offset_qs + PAWN]
            & PAWN_ATTACKS[shadow_opponent_qs, shadow_king_sq_qs]
        )
        shadow_checkers_qs |= (
            piece_bbs[shadow_offset_qs + KNIGHT]
            & KNIGHT_ATTACKS[shadow_king_sq_qs]
        )
        shadow_checkers_qs |= (
            (piece_bbs[shadow_offset_qs + BISHOP]
             | piece_bbs[shadow_offset_qs + QUEEN])
            & get_bishop_attacks(shadow_king_sq_qs, shadow_occ_qs)
        )
        shadow_checkers_qs |= (
            (piece_bbs[shadow_offset_qs + ROOK]
             | piece_bbs[shadow_offset_qs + QUEEN])
            & get_rook_attacks(shadow_king_sq_qs, shadow_occ_qs)
        )
        shadow_checkers_qs |= (
            piece_bbs[shadow_offset_qs + KING]
            & KING_ATTACKS[shadow_king_sq_qs]
        )
        shadow_checker_count_qs = count_bits(shadow_checkers_qs)
        shadow_targets_qs = np.uint64(0)
        if shadow_checker_count_qs == 1:
            shadow_checker_sq_qs = get_lsb_index(shadow_checkers_qs)
            shadow_targets_qs = (
                shadow_checkers_qs
                | SQUARES_BETWEEN[
                    shadow_king_sq_qs, shadow_checker_sq_qs
                ]
            )
        for shadow_idx_qs in range(move_count):
            shadow_class_qs = _classify_main_in_check_move(
                piece_bbs, occupancy_bbs, moves[shadow_idx_qs],
                shadow_king_sq_qs, shadow_king_bb_qs,
                shadow_opponent_qs, shadow_pinned_qs,
                shadow_checker_count_qs, shadow_targets_qs,
            )
            if shadow_class_qs == 1:
                _diag_add(search_context, DIAG_Q_EVASION_PREFILTER)

    # Delay and conditionally compute pinned pieces for non-evasion qsearch
    # nodes (only if we have moves that need SEE/history scoring).
    if move_count > 1 or (move_count == 1 and moves[0] != qs_tt_move):
        if not pinned_computed:
            pinned_white = get_pinned_pieces(piece_bbs, occupancy_bbs, WHITE)
            pinned_black = get_pinned_pieces(piece_bbs, occupancy_bbs, BLACK)
            pinned_computed = True

    score_captures_with_tt_lazy(
        piece_bbs, occupancy_bbs, game_state,
        moves,
        scores,
        move_count,
        qs_tt_move,
        pinned_white,
        pinned_black,
        search_context,
        ply
    )

    # Baseline qsearch legality geometry is constructed after scoring.  This
    # restores the pre-P0 production order while retaining the shadow counter.
    side_to_move_qs = game_state[0]
    opponent_side_qs = np.uint8(1 - side_to_move_qs)
    our_king_bb_qs = piece_bbs[5] if side_to_move_qs == WHITE else piece_bbs[11]
    our_king_sq_qs = (
        get_lsb_index(our_king_bb_qs) if our_king_bb_qs else np.int8(0)
    )
    pinned_us_qs = pinned_white if side_to_move_qs == WHITE else pinned_black
    qs_checkers = np.uint64(0)
    qs_evasion_targets = np.uint64(0)
    qs_checker_count = np.int32(0)
    if is_currently_in_check:
        if not pinned_computed:
            pinned_white = get_pinned_pieces(
                piece_bbs, occupancy_bbs, WHITE
            )
            pinned_black = get_pinned_pieces(
                piece_bbs, occupancy_bbs, BLACK
            )
            pinned_us_qs = (
                pinned_white if side_to_move_qs == WHITE else pinned_black
            )
            pinned_computed = True
        opponent_offset_qs = 6 if opponent_side_qs == BLACK else 0
        all_occ_qs_evasion = occupancy_bbs[2]
        qs_checkers |= (
            piece_bbs[opponent_offset_qs + PAWN]
            & PAWN_ATTACKS[opponent_side_qs, our_king_sq_qs]
        )
        qs_checkers |= (
            piece_bbs[opponent_offset_qs + KNIGHT]
            & KNIGHT_ATTACKS[our_king_sq_qs]
        )
        qs_checkers |= (
            (piece_bbs[opponent_offset_qs + BISHOP]
             | piece_bbs[opponent_offset_qs + QUEEN])
            & get_bishop_attacks(our_king_sq_qs, all_occ_qs_evasion)
        )
        qs_checkers |= (
            (piece_bbs[opponent_offset_qs + ROOK]
             | piece_bbs[opponent_offset_qs + QUEEN])
            & get_rook_attacks(our_king_sq_qs, all_occ_qs_evasion)
        )
        qs_checkers |= (
            piece_bbs[opponent_offset_qs + KING]
            & KING_ATTACKS[our_king_sq_qs]
        )
        qs_checker_count = count_bits(qs_checkers)
        if qs_checker_count == 1:
            qs_checker_sq = get_lsb_index(qs_checkers)
            qs_evasion_targets = (
                qs_checkers
                | SQUARES_BETWEEN[our_king_sq_qs, qs_checker_sq]
            )

    legal_moves_tried = 0
    for i in range(move_count):
        # Lazy Selection Sort: find the best remaining move
        best_idx = i
        for j in range(i + 1, move_count):
            if scores[j] > scores[best_idx]:
                best_idx = j
        
        # Swap moves and scores in the pre-allocated buffers
        moves[i], moves[best_idx] = moves[best_idx], moves[i]
        scores[i], scores[best_idx] = scores[best_idx], scores[i]

        move = moves[i]
        score_val = scores[i]
        qs_from_sq = get_from_square(move)
        qs_to_sq = get_to_square(move)
        qs_special_flag = get_special_move_flag(move)

        delta_prune_candidate = False
        if not is_currently_in_check:
            if ENABLE_DELTA_PRUNING:
                move_flag = qs_special_flag
                is_promotion = move_flag == SPECIAL_MOVE_FLAG_PROMOTION
                promotion_gain = MG_MATERIAL_VALUES[4] - MG_MATERIAL_VALUES[0] if is_promotion else 0
                side_to_move = game_state[0]
                if move_flag == SPECIAL_MOVE_FLAG_EN_PASSANT:
                    victim_value = MG_MATERIAL_VALUES[PAWN]
                else:
                    victim_type = find_piece_type_on_square_side(piece_bbs, get_to_square(move), 1 - side_to_move)
                    victim_value = MG_MATERIAL_VALUES[victim_type % 6] if victim_type != -1 else 0
                potential_gain = victim_value + promotion_gain
                
                if stand_pat + potential_gain + search_context.tune[TUNE_DELTA_MARGIN] < alpha:
                    delta_prune_candidate = True

        if delta_prune_candidate:
            move_flag = get_special_move_flag(move)
            gives_check = False

            if (move_flag == SPECIAL_MOVE_FLAG_EN_PASSANT
                    or move_flag == SPECIAL_MOVE_FLAG_PROMOTION):
                # EP and promotions alter check geometry in ways the ordinary
                # pre-make test cannot model exactly.  They are rare, so use an
                # exact temporary make/unmake before deciding the delta prune.
                original_side_qs = game_state[0]
                check_unmake_info = make_move(
                    piece_bbs, occupancy_bbs, game_state, move,
                )
                their_king_bb_after_qs = (
                    piece_bbs[5] if game_state[0] == WHITE else piece_bbs[11]
                )
                their_king_sq_after_qs = (
                    get_lsb_index(their_king_bb_after_qs)
                    if their_king_bb_after_qs else np.int8(0)
                )
                gives_check = is_square_attacked(
                    piece_bbs, occupancy_bbs,
                    their_king_sq_after_qs, original_side_qs,
                )
                unmake_move(
                    piece_bbs, occupancy_bbs, game_state,
                    move, check_unmake_info,
                )
            else:
                # SF11/SF18 protect checking qsearch moves from their
                # futility/delta gate.  Build geometry lazily only when delta
                # would otherwise discard a move.
                if not qs_check_info_computed:
                    their_king_bb_qs = (
                        piece_bbs[11] if side_to_move_qs == WHITE else piece_bbs[5]
                    )
                    their_king_sq_qs = (
                        get_lsb_index(their_king_bb_qs)
                        if their_king_bb_qs else np.int8(0)
                    )
                    all_occ_qs = occupancy_bbs[2]
                    check_sq_pawn_qs = PAWN_ATTACKS[side_to_move_qs, their_king_sq_qs]
                    check_sq_knight_qs = KNIGHT_ATTACKS[their_king_sq_qs]
                    check_sq_bishop_qs = get_bishop_attacks(their_king_sq_qs, all_occ_qs)
                    check_sq_rook_qs = get_rook_attacks(their_king_sq_qs, all_occ_qs)
                    check_sq_queen_qs = check_sq_bishop_qs | check_sq_rook_qs
                    blockers_for_their_king_qs, pinners_for_their_king_qs = get_blockers_for_king(
                        piece_bbs, occupancy_bbs, opponent_side_qs,
                    )
                    qs_check_info_computed = True

                gives_check = _qsearch_ordinary_move_gives_check(
                    piece_bbs, move, side_to_move_qs, their_king_bb_qs,
                    check_sq_pawn_qs, check_sq_knight_qs, check_sq_bishop_qs,
                    check_sq_rook_qs, check_sq_queen_qs,
                    blockers_for_their_king_qs,
                )

            if not gives_check:
                _diag_add(search_context, DIAG_Q_DELTA_SKIP)
                continue
            _diag_add(search_context, DIAG_Q_CHECK_PROTECT)

        if not is_currently_in_check:
            if ENABLE_SEE_IN_QUIESCENCE and move != qs_tt_move:
                c_from = get_from_square(move)
                c_to = get_to_square(move)
                _diag_add(search_context, DIAG_Q_SEE_TRY)
                if not _see_ge_jit(piece_bbs, occupancy_bbs, game_state[0], c_from, c_to, search_context.tune[TUNE_QS_SEE], pinned_white, pinned_black):
                    _diag_add(search_context, DIAG_Q_SEE_SKIP)
                    continue

        # --- Pre-make Legality Fast Path (SF18-style) ---
        skip_post_legality = False

        if is_currently_in_check:
            if qs_from_sq == our_king_sq_qs:
                # Castling can never evade an existing check.
                if qs_special_flag == SPECIAL_MOVE_FLAG_CASTLING:
                    _diag_add(search_context, DIAG_Q_ILLEGAL_SKIP)
                    continue
                occ_without_king = occupancy_bbs[2] ^ our_king_bb_qs
                if is_square_attacked_with_occ(
                    piece_bbs, qs_to_sq, occ_without_king, opponent_side_qs,
                ):
                    _diag_add(search_context, DIAG_Q_ILLEGAL_SKIP)
                    continue
                skip_post_legality = True
            elif qs_special_flag != SPECIAL_MOVE_FLAG_EN_PASSANT and qs_checker_count > 0:
                to_bb = BB_SQUARES[qs_to_sq]
                if qs_checker_count > 1 or not (qs_evasion_targets & to_bb):
                    _diag_add(search_context, DIAG_Q_ILLEGAL_SKIP)
                    continue
                from_bb = BB_SQUARES[qs_from_sq]
                if ((pinned_us_qs & from_bb)
                        and not (LINE_BB[qs_from_sq, qs_to_sq] & our_king_bb_qs)):
                    _diag_add(search_context, DIAG_Q_ILLEGAL_SKIP)
                    continue
                # Capturing the sole checker or interposing on its ray, while
                # respecting absolute pins, is sufficient for ordinary moves.
                skip_post_legality = True
        else:
            if qs_from_sq != our_king_sq_qs and qs_special_flag != SPECIAL_MOVE_FLAG_EN_PASSANT:
                # Non-king, non-EP move: use pin-based O(1) legality if pins are computed
                if pinned_computed:
                    from_bb = BB_SQUARES[qs_from_sq]
                    if not (pinned_us_qs & from_bb):
                        # Not pinned → guaranteed legal
                        skip_post_legality = True
                    elif LINE_BB[qs_from_sq, qs_to_sq] & our_king_bb_qs:
                        # Pinned but moving along pin ray → still legal
                        skip_post_legality = True
                    else:
                        # Pinned and moving off pin ray → illegal, skip make/unmake entirely
                        _diag_add(search_context, DIAG_Q_ILLEGAL_SKIP)
                        continue
            elif qs_from_sq == our_king_sq_qs:
                # King move: check destination with king removed from occupancy
                occ_without_king = occupancy_bbs[2] ^ our_king_bb_qs
                if is_square_attacked_with_occ(piece_bbs, qs_to_sq, occ_without_king, opponent_side_qs):
                    _diag_add(search_context, DIAG_Q_ILLEGAL_SKIP)
                    continue  # King walks into attack
                skip_post_legality = True
            # else: EP — fall through to post-make check

        unmake_info = make_move(piece_bbs, occupancy_bbs, game_state, move)
        
        # --- Post-make Legality Fallback (in-check evasions and EP only) ---
        if not skip_post_legality:
            king_bb_after_move = piece_bbs[5] if (1 - game_state[0]) == 0 else piece_bbs[11]
            king_sq = get_lsb_index(king_bb_after_move) if king_bb_after_move else 0
            if is_square_attacked(piece_bbs, occupancy_bbs, king_sq, game_state[0]):
                unmake_move(piece_bbs, occupancy_bbs, game_state, move, unmake_info)
                _diag_add(search_context, DIAG_Q_ILLEGAL_SKIP)
                continue
            
        legal_moves_tried += 1

        score, child_q_nodes = quiescence_search(
            piece_bbs, occupancy_bbs, game_state,
            _i32(-beta), _i32(-alpha), _i32(ply + 1), search_context, _i32(q_ply + 1),
        )
        unmake_move(piece_bbs, occupancy_bbs, game_state, move, unmake_info)

        # Preserve completed child telemetry even when the timer fired inside
        # that child.  Search result handling still stops immediately below.
        q_nodes += child_q_nodes

        if search_context.stop_flag[0]:
            return np.int32(0), q_nodes
        score = np.int32(-score)

        if score > best_eval:
            best_eval = score
            if score > alpha:
                best_move = move
                if score >= beta:
                    _diag_add(search_context, DIAG_Q_BETA_CUT)
                    break
                alpha = score

    if is_currently_in_check and legal_moves_tried == 0:
        return np.int32(-MATE_SCORE + ply), q_nodes

    # C1: QSearch Move Loop fail-high smoothing
    if not is_currently_in_check and best_eval > beta and abs(best_eval) < MATE_IN_MAX_PLY:
        best_eval = np.int32(trunc_div(QS_FAILHIGH_SMOOTH_A * best_eval + QS_FAILHIGH_SMOOTH_B * beta, QS_SMOOTH_DIV))

    tt_score = np.int32(best_eval)
    if tt_score > MATE_IN_MAX_PLY: tt_score += ply
    elif tt_score < -MATE_IN_MAX_PLY: tt_score -= ply

    tt_flag = TT_FLAG_BETA if best_eval >= beta else (TT_FLAG_ALPHA if alpha <= original_alpha else TT_FLAG_EXACT)

    tt_static_eval_to_store = np.int16(32767)
    if not is_currently_in_check and stand_pat != 32767:
        tt_static_eval_to_store = np.int16(stand_pat)
    store_tt(search_context.transposition_table, tt_key, 0, tt_score, tt_static_eval_to_store, tt_flag, best_move, search_context.tt_generation, False)

    return np.int32(best_eval), q_nodes


get_next_move_return_type = numba.uint16

@numba.njit(cache=True, nogil=True, boundscheck=False, fastmath=True)
def get_next_move(
    piece_bbs, occupancy_bbs, game_state, search_context, ply, tt_move,
    excluded_move, killer_1, killer_2, counter_move, pinned_white,
    pinned_black, pawn_key_idx,
    is_in_check, in_check_our_king_sq, in_check_our_king_bb,
    in_check_opponent_side, in_check_pinned_us, in_check_checker_count,
    in_check_evasion_targets,
):
    moves = search_context.moves_buffer[ply]
    scores = search_context.move_scores[ply]
    bad_captures = search_context.bad_captures[ply]

    while search_context.mp_stage[ply] < STAGE_DONE:
        move = NO_MOVE
        mp_stage = search_context.mp_stage[ply]
        
        if mp_stage == STAGE_TT_MOVE:
            if tt_move != NO_MOVE and tt_move != excluded_move:
                if is_move_pseudo_legal(piece_bbs, occupancy_bbs, game_state, tt_move):
                    move = tt_move
            search_context.mp_stage[ply] = STAGE_GEN_CAPTURES
            if move != NO_MOVE:
                if is_in_check and search_context.diag_enabled:
                    _classify_and_diag_main_in_check_move(
                        piece_bbs, occupancy_bbs, move,
                        in_check_our_king_sq, in_check_our_king_bb,
                        in_check_opponent_side, in_check_pinned_us,
                        in_check_checker_count, in_check_evasion_targets,
                        search_context,
                    )
                return move
            
        elif mp_stage == STAGE_GEN_CAPTURES:
            search_context.mp_captures_end[ply] = generate_pseudo_legal_captures_buffer(piece_bbs, occupancy_bbs, game_state, search_context.moves_buffer, ply)
            if is_in_check and search_context.diag_enabled:
                generated_captures = search_context.mp_captures_end[ply]
                for capture_idx in range(generated_captures):
                    candidate_capture = moves[capture_idx]
                    _classify_and_diag_main_in_check_move(
                        piece_bbs, occupancy_bbs, candidate_capture,
                        in_check_our_king_sq, in_check_our_king_bb,
                        in_check_opponent_side, in_check_pinned_us,
                        in_check_checker_count, in_check_evasion_targets,
                        search_context,
                    )
            score_captures_lazy(piece_bbs, occupancy_bbs, game_state, moves, scores, 0, search_context.mp_captures_end[ply], search_context, pinned_white, pinned_black, ply)
            partial_insertion_sort_moves(moves, scores, 0, search_context.mp_captures_end[ply], -1000000)
            search_context.mp_current_idx[ply] = 0
            search_context.mp_stage[ply] = STAGE_GOOD_CAPTURES
            continue
            
        elif mp_stage == STAGE_GOOD_CAPTURES:
            captures_end = search_context.mp_captures_end[ply]
            current_idx = search_context.mp_current_idx[ply]
            if current_idx < captures_end:
                candidate_move = moves[current_idx]
                search_context.mp_current_idx[ply] += 1
                
                if candidate_move == tt_move or candidate_move == excluded_move:
                    continue
                    
                # Lazy SEE Evaluation with threshold 0
                c_from = get_from_square(candidate_move)
                c_to = get_to_square(candidate_move)
                _diag_add(search_context, DIAG_PICKER_SEE_TRY)
                is_good = _see_ge_jit(piece_bbs, occupancy_bbs, game_state[0], c_from, c_to, 0, pinned_white, pinned_black)
                
                if is_good:
                    move = candidate_move
                    return move
                else:
                    bad_captures[search_context.mp_bad_captures_count[ply]] = candidate_move
                    search_context.mp_bad_captures_count[ply] += 1
                    continue
            else:
                search_context.mp_stage[ply] = STAGE_GEN_QUIETS
                continue
                
        elif mp_stage == STAGE_GEN_QUIETS:
            captures_end = search_context.mp_captures_end[ply]
            search_context.mp_quiets_end[ply] = generate_pseudo_legal_quiets_buffer(piece_bbs, occupancy_bbs, game_state, search_context.moves_buffer, ply, captures_end)
            if is_in_check and search_context.diag_enabled:
                generated_quiets = search_context.mp_quiets_end[ply]
                for quiet_idx in range(captures_end, generated_quiets):
                    candidate_quiet = moves[quiet_idx]
                    _classify_and_diag_main_in_check_move(
                        piece_bbs, occupancy_bbs, candidate_quiet,
                        in_check_our_king_sq, in_check_our_king_bb,
                        in_check_opponent_side, in_check_pinned_us,
                        in_check_checker_count, in_check_evasion_targets,
                        search_context,
                    )
            score_quiets(
                piece_bbs, occupancy_bbs, game_state, moves, scores,
                captures_end, search_context.mp_quiets_end[ply], search_context,
                ply, killer_1, killer_2, counter_move, pawn_key_idx,
                pinned_white, pinned_black,
            )
            
            # Sort good quiets (score >= GOOD_QUIET_THRESHOLD) to the front. Bad quiets (score < GOOD_QUIET_THRESHOLD) remain at the back.
            num_sorted = partial_insertion_sort_moves(moves, scores, captures_end, search_context.mp_quiets_end[ply], GOOD_QUIET_THRESHOLD)
            
            search_context.mp_current_idx[ply] = captures_end
            search_context.mp_stage[ply] = STAGE_GOOD_QUIETS
            continue
            
        elif mp_stage == STAGE_GOOD_QUIETS:
            quiets_end = search_context.mp_quiets_end[ply]
            current_idx = search_context.mp_current_idx[ply]
            
            # Since Good Quiets are strictly sorted and have score >= GOOD_QUIET_THRESHOLD at the front, we just yield them.
            if current_idx < quiets_end and scores[current_idx] >= GOOD_QUIET_THRESHOLD:
                candidate_move = moves[current_idx]
                search_context.mp_current_idx[ply] += 1
                
                if candidate_move == tt_move or candidate_move == excluded_move:
                    continue
                move = candidate_move
                return move
            else:
                search_context.mp_stage[ply] = STAGE_BAD_CAPTURES
                continue
                
        elif mp_stage == STAGE_BAD_CAPTURES:
            if search_context.mp_bad_captures_idx[ply] < search_context.mp_bad_captures_count[ply]:
                move = bad_captures[search_context.mp_bad_captures_idx[ply]]
                search_context.mp_bad_captures_idx[ply] += 1
                return move
            else:
                search_context.mp_stage[ply] = STAGE_BAD_QUIETS
                continue
                
        elif mp_stage == STAGE_BAD_QUIETS:
            quiets_end = search_context.mp_quiets_end[ply]
            current_idx = search_context.mp_current_idx[ply]
            
            # These are the quiets with score < GOOD_QUIET_THRESHOLD that were left at the end of the array.
            if current_idx < quiets_end:
                candidate_move = moves[current_idx]
                search_context.mp_current_idx[ply] += 1
                
                if candidate_move == tt_move or candidate_move == excluded_move:
                    continue
                move = candidate_move
                return move
            else:
                search_context.mp_stage[ply] = STAGE_DONE
                continue


    return NO_MOVE

search_return_type = numba.types.Tuple([
    numba.int32, numba.uint16, numba.uint64, numba.uint64, numba.uint64
])

@numba.njit(search_return_type(piece_bbs_signature, occupancy_bbs_signature, game_state_signature, numba.int32, numba.int32, numba.int32, search_context_type, numba.int32, numba.uint16, numba.boolean, numba.boolean), cache=False, nogil=True, boundscheck=False, fastmath=True)
def _search(piece_bbs, occupancy_bbs, game_state, depth, alpha, beta, search_context, ply, excluded_move, is_pv, cut_node):
    # ==========================================
    # PHASE 1: Initialize and Base Cases
    # ==========================================
    nodes_searched = np.uint64(1)
    search_context.nodes_searched += 1

    quiescence_nodes = np.uint64(0)
    tt_hits = np.uint64(0)
    is_exclusion_search = excluded_move != NO_MOVE

    if ply == 0:
        search_context.cutoff_cnt[0] = 0
        search_context.cutoff_cnt[1] = 0

    if ply + 2 < MAX_PLY:
        search_context.cutoff_cnt[ply + 2] = 0

    # SF followPV: root always follows; deeper nodes follow if parent followed and
    # the move that led here matches last iteration's PV at this ply.
    if ENABLE_FOLLOW_PV:
        if ply == 0:
            follow_pv = True
        else:
            parent_follow = search_context.follow_pv_stack[ply - 1]
            prev_move = search_context.move_stack[ply - 1]
            lpv_len = search_context.last_iteration_pv_len
            follow_pv = (
                parent_follow
                and (ply - 1) < lpv_len
                and prev_move == search_context.last_iteration_pv[ply - 1]
            )
    else:
        follow_pv = False
    search_context.follow_pv_stack[ply] = follow_pv

    # Publish progress and observe timer/node-limit stops at a bounded cadence.
    if (search_context.nodes_searched & STOP_CHECK_MASK) == 0:
        search_context.nodes_searched_array[0] = search_context.nodes_searched
        node_budget = search_context.nodes_searched_array[1]
        if node_budget > 0 and search_context.nodes_searched >= node_budget:
            search_context.stop_flag[0] = True
        if search_context.stop_flag[0]:
            return (np.int32(0), NO_MOVE, nodes_searched, quiescence_nodes, tt_hits)

    if ply >= MAX_PLY:
        eval_score, _, _ = _evaluate_position_jit(piece_bbs, occupancy_bbs, game_state, False, search_context)
        return (np.int32(eval_score), NO_MOVE, nodes_searched, quiescence_nodes, tt_hits)

    search_context.pv_table[ply, ply] = NO_MOVE

    # --- Mate Distance Pruning ---
    if ENABLE_MATE_DISTANCE_PRUNING:
        # If we find a mate at 'ply', the score would be MATE_SCORE - ply.
        # We can't do better than mating at the current ply (MATE_SCORE - ply).
        # We can't do worse than being mated at the current ply (-MATE_SCORE + ply).
        
        # Lower bound: if we are mated in 'ply', score is -MATE_SCORE + ply.
        # Any score below this is impossible.
        alpha = max(alpha, -MATE_SCORE + ply)
        
        # Upper bound: if we mate in 'ply', score is MATE_SCORE - ply.
        # But we need to make a move to mate, so technically MATE_SCORE - (ply + 1)?
        # Actually standard logic is:
        # We are at ply. The best we can hope for is mate in 1 (at ply+1).
        # Score: MATE_SCORE - (ply + 1).
        beta = min(beta, MATE_SCORE - ply - 1)
        
        if alpha >= beta:
            return (np.int32(alpha), NO_MOVE, nodes_searched, quiescence_nodes, tt_hits)

    # --- Repetition Detection ---
    # Performance Optimization (Bolt): Use halfmove_clock to limit checks and skip odd plies.
    # Impact: ~1.8% NPS improvement on Kiwipete benchmark.
    zobrist_key = game_state[4]
    
    # Store current key in path stack
    search_context.ply_path_stack[ply] = zobrist_key
    
    repetition_count = 0
    
    # Check against Game History and Path, but only up to the last irreversible move
    # game_state[3] is the halfmove clock.
    halfmove_clock = int(game_state[3])
    
    if halfmove_clock >= 2:
        # Check Path (even plies only, as odd plies have different side-to-move)
        # Current position is at 'ply', so we check ply-2, ply-4, etc.
        for i in range(ply - 2, max(-1, ply - halfmove_clock - 1), -2):
            if search_context.ply_path_stack[i] == zobrist_key:
                repetition_count += 1
                if repetition_count >= 1: break
        
        if repetition_count < 1:
            # Check Game History
            # Number of moves in history that are within the halfmove clock
            # halfmove_clock includes moves from both path and history.
            hist_to_check = halfmove_clock - ply
            if hist_to_check >= 1:
                # Same side to move:
                # If ply is even, we match game_history[count-3, count-5, ...]
                # If ply is odd, we match game_history[count-2, count-4, ...]
                start_offset = 2 if (ply % 2) != 0 else 3
                start_idx = search_context.game_history_count - start_offset
                end_idx = max(-1, search_context.game_history_count - hist_to_check - 1)
                for i in range(start_idx, end_idx - 1, -2):
                    if i < 0: break
                    if search_context.game_history[i] == zobrist_key:
                        repetition_count += 1
                        if repetition_count >= 1: break

    # Avoid 1st repetition (2nd occurrence total) or 50-move rule
    # Crucial fix: Do not prune at the root (ply 0).
    if ply > 0:
        if repetition_count >= 1:
            search_context.pv_table[ply, ply] = NO_MOVE
            return (np.int32(0), NO_MOVE, nodes_searched, quiescence_nodes, tt_hits)

        # 50-move rule and insufficient material cannot override checkmate
        if halfmove_clock >= FIFTY_MOVE_RULE_LIMIT or not has_sufficient_material(piece_bbs):
            if not _is_in_check_jit(piece_bbs, occupancy_bbs, game_state):
                search_context.pv_table[ply, ply] = NO_MOVE
                return (np.int32(0), NO_MOVE, nodes_searched, quiescence_nodes, tt_hits)
    
    # M3: 50-move rule scale-down — gradually reduce eval towards draw as halfmove_clock approaches limit
    # This prevents the "cliff effect" where deep search sees halfmove_clock=limit at leaf nodes
    fifty_move_scale = FIFTY_MOVE_MAX_SCALE  # Full scale (no reduction)
    if halfmove_clock >= FIFTY_MOVE_SCALE_THRESHOLD and ply > 0:
        # Linear scale-down
        fifty_move_scale = max(0, (FIFTY_MOVE_RULE_LIMIT - halfmove_clock) * FIFTY_MOVE_MAX_SCALE // (FIFTY_MOVE_RULE_LIMIT - FIFTY_MOVE_SCALE_THRESHOLD))
    
    # Initialize PV for this ply to avoid ghost moves from previous searches
    search_context.pv_table[ply, ply] = NO_MOVE

    original_alpha = alpha
    
    # --- Initialization ---
    move_count = 0
    pawn_key_idx = game_state[PAWN_KEY_INDEX] & PAWN_HISTORY_MASK
    # ==========================================
    # PHASE 2: Transposition Table Probe
    # ==========================================
    moves_generated = False

    tt_move = NO_MOVE
    tt_key = get_tt_key(zobrist_key, halfmove_clock)  # GHI: halfmove-aware key for TT
    tt_entry = probe_tt(search_context.transposition_table, tt_key)
    tt_hit = tt_entry['flag'] != TT_FLAG_NONE
    search_context.tt_hit_stack[ply] = tt_hit

    # Initialize tt_pv flag (TT-PV Memory Heuristic)
    if is_exclusion_search:
        tt_pv = search_context.tt_pv_stack[ply]
    else:
        tt_pv = is_pv or (tt_entry['flag'] != TT_FLAG_NONE and tt_entry['is_pv'])
    search_context.tt_pv_stack[ply] = tt_pv

    # 1. ALWAYS retrieve the best move if available (Critical for Move Ordering)
    tt_capture = False
    if tt_entry['flag'] != TT_FLAG_NONE:
        tt_move = tt_entry['best_move']
        if tt_move != NO_MOVE:
            tt_to = get_to_square(tt_move)
            tt_flag_special = get_special_move_flag(tt_move)
            enemy_side = 1 - game_state[0]
            tt_capture = ((occupancy_bbs[enemy_side] & BB_SQUARES[tt_to]) != 0) or (tt_flag_special == SPECIAL_MOVE_FLAG_EN_PASSANT)

        # 2. ONLY perform a Score Cutoff (Return) if:
        #    a. We are NOT at the Root Node (ply > 0)
        #    b. NOT a PV node (H9: PV nodes must not be cut off by TT bounds)
        #    c. The stored depth is sufficient
        #    d. The score bounds (Alpha/Beta) are valid for a cutoff
        #    e. We are NOT in a singular extension search (excluded_move == NO_MOVE)
        #       OR the TT move is NOT the excluded move.
        #    f. (P1) halfmove < TT_CUTOFF_HALFMOVE_MAX to avoid GHI pollution near 50-move rule.
        if ply > 0 and not is_pv and not is_exclusion_search and halfmove_clock < TT_CUTOFF_HALFMOVE_MAX:
            tt_score = np.int32(tt_entry['score'])

            # Adjust mate scores relative to the current ply
            if tt_score > MATE_IN_MAX_PLY: tt_score -= ply
            elif tt_score < -MATE_IN_MAX_PLY: tt_score += ply

            # Apply 50-move scale down immediately to TT scores to avoid overlooking impending draws
            if fifty_move_scale < FIFTY_MOVE_MAX_SCALE and abs(tt_score) < MATE_IN_MAX_PLY:
                tt_score = tt_score * fifty_move_scale // FIFTY_MOVE_MAX_SCALE

            # Align with SF18 depth logic: ttData.depth > depth - (ttData.value <= beta)
            required_depth = depth + (0 if tt_score <= beta else 1)
            
            if tt_entry['depth'] >= required_depth:
                # Align with SF18 cutNode consistency guard
                tt_score_ge_beta = tt_score >= beta
                consistency_guard = (cut_node == tt_score_ge_beta) or depth > TT_CONSISTENCY_MIN_DEPTH
                
                if consistency_guard:
                    should_cutoff = False
                    if tt_entry['flag'] == TT_FLAG_EXACT:
                        should_cutoff = True
                    elif tt_entry['flag'] == TT_FLAG_ALPHA and tt_score <= alpha:
                        should_cutoff = True
                    elif tt_entry['flag'] == TT_FLAG_BETA and tt_score >= beta:
                        should_cutoff = True

                    if should_cutoff:
                        # Align with SF18 Step 4 (Deep Move Verification)
                        run_cutoff = True
                        tt_verify_attempted = False
                        tt_verify_rejected = False
                        tt_verify_skipped = False
                        if depth >= TT_DEEP_VERIFY_DEPTH and tt_move != NO_MOVE and abs(tt_score) < MATE_IN_MAX_PLY:
                            tt_verify_attempted = True
                            _diag_add(search_context, DIAG_TT_VERIFY_TRY)
                            if is_move_pseudo_legal(piece_bbs, occupancy_bbs, game_state, tt_move):
                                unmake_info = make_move(piece_bbs, occupancy_bbs, game_state, tt_move)
                                king_bb = piece_bbs[5] if (1 - game_state[0]) == 0 else piece_bbs[11]
                                king_sq = get_lsb_index(king_bb) if king_bb else 0
                                if not is_square_attacked(piece_bbs, occupancy_bbs, king_sq, game_state[0]):
                                    child_zobrist_key = game_state[4]
                                    child_halfmove_clock = int(game_state[3])
                                    child_tt_key = get_tt_key(child_zobrist_key, child_halfmove_clock)
                                    child_tt_entry = probe_tt(search_context.transposition_table, child_tt_key)
                                    unmake_move(piece_bbs, occupancy_bbs, game_state, tt_move, unmake_info)
                                    
                                    if child_tt_entry['flag'] != TT_FLAG_NONE:
                                        child_score = np.int32(child_tt_entry['score'])
                                        if child_score > MATE_IN_MAX_PLY: child_score -= (ply + 1)
                                        elif child_score < -MATE_IN_MAX_PLY: child_score += (ply + 1)
                                        
                                        if tt_score_ge_beta != (-child_score >= beta):
                                            run_cutoff = False
                                            tt_verify_rejected = True
                                    # A missing child entry is a valid
                                    # non-contradictory verification result.
                                else:
                                    unmake_move(piece_bbs, occupancy_bbs, game_state, tt_move, unmake_info)
                                    tt_verify_skipped = True
                            else:
                                # Preserve the legacy cutoff behavior, but
                                # record that deep verification could not
                                # validate its TT move.
                                tt_verify_skipped = True

                            if search_context.diag_enabled:
                                if tt_verify_rejected:
                                    _diag_add(search_context, DIAG_TT_VERIFY_REJECT)
                                elif tt_verify_skipped:
                                    _diag_add(search_context, DIAG_TT_VERIFY_SKIP)
                                else:
                                    _diag_add(search_context, DIAG_TT_VERIFY_PASS)

                        if run_cutoff:
                            if tt_verify_attempted:
                                _diag_add(search_context, DIAG_TT_VERIFY_SAVED_CUT)
                            tt_hits += np.uint64(1)
                            # --- P3-C: History Update on TT Fail-High ---
                            if tt_entry['flag'] == TT_FLAG_BETA and tt_score >= beta and tt_move != NO_MOVE:
                                tt_from = get_from_square(tt_move)
                                tt_to = get_to_square(tt_move)
                                tt_flag_special = get_special_move_flag(tt_move)
                                
                                our_side = game_state[0]
                                enemy_side = 1 - our_side
                                
                                tt_is_capture = ((occupancy_bbs[enemy_side] & BB_SQUARES[tt_to]) != 0) or (tt_flag_special == SPECIAL_MOVE_FLAG_EN_PASSANT)
                                tt_is_promotion = tt_flag_special == SPECIAL_MOVE_FLAG_PROMOTION

                                if not tt_is_capture and not tt_is_promotion:
                                    tt_aggressor = find_piece_type_on_square_side(piece_bbs, tt_from, our_side)
                                    if tt_aggressor != -1 and ((occupancy_bbs[our_side] & BB_SQUARES[tt_to]) == 0):
                                        tt_bonus = min(HISTORY_BONUS_SCALE * depth, HISTORY_BONUS_CAP)
                                        update_quiet_stats_on_tt_hit(search_context, tt_move, tt_aggressor, tt_to, pawn_key_idx, tt_bonus, ply)
                                        _diag_add(search_context, DIAG_HIST_TT_HIT)
                                        _diag_add(search_context, DIAG_HIST_QUIET_BONUS)
                                        _diag_add(search_context, DIAG_HIST_PIECE_TO_UPD)
                                        _diag_add(search_context, DIAG_HIST_PAWN_UPD)
                                        _diag_add(search_context, DIAG_HIST_CONT_UPD)

                            search_context.pv_table[ply, ply] = NO_MOVE
                            _diag_add(search_context, DIAG_TT_CUT)
                            return (np.int32(tt_score), tt_entry['best_move'], nodes_searched, quiescence_nodes, tt_hits)
                
                # Penalize TT depth if a window-bound mismatch is the only reason cutoff failed
                if depth > TT_PENALIZE_MIN_DEPTH:
                    if tt_entry['flag'] == (TT_FLAG_ALPHA if tt_score >= beta else TT_FLAG_BETA):
                        penalize_tt(search_context.transposition_table, tt_key, 1)

    # ==========================================
    # PHASE 3: Static Evaluation and Pre-Search Pruning
    # ==========================================
    # --- H1: is_in_check MUST be computed before any node-level pruning/sub-searches ---
    is_currently_in_check = _is_in_check_jit(piece_bbs, occupancy_bbs, game_state)

    # --- Depth 0: drop into Quiescence Search ---
    if depth <= 0:
        search_context.pv_table[ply, ply] = NO_MOVE
        eval_score, q_nodes = quiescence_search(
            piece_bbs, occupancy_bbs, game_state, alpha, beta, ply, search_context, _i32(0)
        )
        return (np.int32(eval_score), NO_MOVE, nodes_searched, q_nodes, tt_hits)

    # --- H2: Static Evaluation & Improving Flag (computed BEFORE any pruning) ---
    static_score = -INFINITY
    improving = False
    
    # Correction History Adjustment
    raw_static_eval = -INFINITY
    static_eval_is_full = False
    cached_pinned_white = np.uint64(0)
    cached_pinned_black = np.uint64(0)

    if not is_currently_in_check:
        if tt_entry['flag'] != TT_FLAG_NONE and tt_entry['static_eval'] != 32767:
            raw_static_eval = np.int32(tt_entry['static_eval'])
        else:
            raw_static_eval, _, _ = _evaluate_position_jit(piece_bbs, occupancy_bbs, game_state, True, search_context)

        static_score = apply_correction_history_score(game_state, search_context, raw_static_eval, ply)
        
        # --- H2 Enhancement: Refine static_score using trusted TT Score Bounds ---
        if tt_entry['flag'] != TT_FLAG_NONE:
            tt_score = np.int32(tt_entry['score'])
            
            # Adjust mate scores relative to current ply
            if tt_score > MATE_IN_MAX_PLY: tt_score -= ply
            elif tt_score < -MATE_IN_MAX_PLY: tt_score += ply

            trusted_bound_depth = tt_entry['depth'] >= max(1, depth - 2)
            trusted_exact_depth = tt_entry['depth'] >= depth
            trusted_tt_score = trusted_bound_depth and abs(tt_score) < MATE_IN_MAX_PLY

            if trusted_tt_score:
                # If TT says score is at least tt_score (LOWER), and tt_score > static_score
                if tt_entry['flag'] == TT_FLAG_BETA and tt_score > static_score:
                    static_score = tt_score
                # If TT says score is at most tt_score (UPPER), and tt_score < static_score
                elif tt_entry['flag'] == TT_FLAG_ALPHA and tt_score < static_score:
                    static_score = tt_score
                # Exact scores may replace static eval only when they are deep enough for this node.
                elif tt_entry['flag'] == TT_FLAG_EXACT and trusted_exact_depth:
                    static_score = tt_score
        
        search_context.static_eval_stack[ply] = static_score

        # V3 3.5: Improved improving calculation — fallback to ply-4 when ply-2 was null move
        if ply >= 2:
            ref_eval = search_context.static_eval_stack[ply - 2]
            if ref_eval == -INFINITY and ply >= 4:
                ref_eval = search_context.static_eval_stack[ply - 4]
            if ref_eval != -INFINITY and static_score > ref_eval:
                improving = True
    else:
        search_context.static_eval_stack[ply] = -INFINITY

    # --- opponentWorsening & Hindsight Depth Adjustment ---
    opponent_worsening = False
    if not is_currently_in_check and ply > 0:
        parent_eval = search_context.static_eval_stack[ply - 1]
        if static_score != -INFINITY and parent_eval != -INFINITY:
            opponent_worsening = static_score > -parent_eval
            
            prior_reduction = search_context.reduction_stack[ply]
            if prior_reduction >= HINDSIGHT_REDUCE_MIN_PRIOR and not opponent_worsening:
                depth += 1
            if prior_reduction >= HINDSIGHT_INCREASE_MIN_PRIOR and depth >= 2:
                if static_score + parent_eval > HINDSIGHT_EVAL_SUM_MARGIN:
                    depth -= 1

    if depth <= 0:
        search_context.pv_table[ply, ply] = NO_MOVE
        eval_score, q_nodes = quiescence_search(
            piece_bbs, occupancy_bbs, game_state, alpha, beta, ply, search_context, _i32(0)
        )
        return (np.int32(eval_score), NO_MOVE, nodes_searched, q_nodes, tt_hits)

    # --- Dissonance Calculation (DGP Algorithm) ---
    dissonance = 0
    if tt_entry['flag'] != TT_FLAG_NONE and not is_currently_in_check and static_score != -INFINITY:
        # Check if depth is reasonably close to current depth to trust the TT score
        if tt_entry['depth'] >= depth - DISSONANCE_TT_DEPTH_SLACK:
            tt_score = np.int32(tt_entry['score'])
            # Avoid using mate scores for dissonance calculation
            if abs(tt_score) < MATE_IN_MAX_PLY and abs(static_score) < MATE_IN_MAX_PLY:
                dissonance = abs(tt_score - static_score)

    total_piece_count = np.int32(0)
    side_non_pawn_count = np.int32(0)
    if not is_currently_in_check:
        total_piece_count = count_bits(occupancy_bbs[WHITE] | occupancy_bbs[BLACK])

        stm_piece_offset = 0 if game_state[0] == WHITE else 6
        side_non_pawn_count = count_bits(
            piece_bbs[stm_piece_offset + KNIGHT] |
            piece_bbs[stm_piece_offset + BISHOP] |
            piece_bbs[stm_piece_offset + ROOK] |
            piece_bbs[stm_piece_offset + QUEEN]
        )

    low_material_pruning_guard = total_piece_count <= LOW_MATERIAL_PRUNING_PIECE_COUNT

    # PR-C: if lazy/TT static is already near known-win band, force full HCE before pruning
    if (not is_currently_in_check and not static_eval_is_full
            and (is_win(static_score) or is_loss(static_score)
                 or abs(static_score) >= (VALUE_KNOWN_WIN - FULL_EVAL_KNOWN_WIN_MARGIN))):
        raw_static_eval, static_score, improving, cached_pinned_white, cached_pinned_black = compute_full_corrected_static_eval(
            piece_bbs, occupancy_bbs, game_state, search_context, ply
        )
        static_eval_is_full = True

    # =====================================================================
    # --- Node-Level Pruning (H3: ALL guarded by not is_currently_in_check) ---
    # =====================================================================
    if not is_currently_in_check and not is_pv and not is_exclusion_search:
        # --- Step 8: Razoring (SF19 Step 8: eval < alpha - 482 * depth * depth) ---
        if (search_context.enable_razoring and depth <= RAZORING_MAX_DEPTH and not low_material_pruning_guard
                and not is_win(static_score) and not is_loss(alpha)
                and abs(alpha) < VALUE_KNOWN_WIN):
            razor_margin = search_context.tune[TUNE_RAZOR_COEFF] * depth * depth
            if static_score < alpha - razor_margin:
                if not static_eval_is_full:
                    raw_static_eval, static_score, improving, cached_pinned_white, cached_pinned_black = compute_full_corrected_static_eval(
                        piece_bbs, occupancy_bbs, game_state, search_context, ply
                    )
                    static_eval_is_full = True
                if (static_score < alpha - razor_margin
                        and not is_win(static_score) and not is_loss(alpha)):
                    search_context.pv_table[ply, ply] = NO_MOVE
                    razor_score, child_q_nodes = quiescence_search(
                        piece_bbs, occupancy_bbs, game_state, alpha, beta, ply, search_context, _i32(0)
                    )
                    quiescence_nodes += child_q_nodes
                    _diag_add(search_context, DIAG_RAZOR_CUT)
                    return (np.int32(razor_score), NO_MOVE, nodes_searched, quiescence_nodes, tt_hits)

        # --- M5: Reverse Futility Pruning (SF19 Step 9: !tt_pv, !tt_move or tt_capture) ---
        # P2: never return known-win / loss-band static as a proven cutoff (SF11/18/19).
        if (search_context.enable_rfp and depth <= RFP_MAX_DEPTH and not low_material_pruning_guard
                and not tt_pv and (tt_move == NO_MOVE or tt_capture)
                and not is_win(static_score) and not is_loss(beta)):
            # SF11: futility_margin = 217 * (depth - improving)
            # Larger margin when NOT improving → harder to RFP (safer). Do not invert this.
            rfp_multiplier = search_context.tune[TUNE_RFP_MULT]
            if tt_entry['flag'] == TT_FLAG_NONE:
                rfp_multiplier += RFP_NO_TT_PENALTY  # small safety when no TT
            rfp_depth_factor = depth - (1 if improving else 0)
            if rfp_depth_factor < 1:
                rfp_depth_factor = 1
            rfp_margin = rfp_multiplier * rfp_depth_factor
                
            if static_score - rfp_margin >= beta:
                if not static_eval_is_full:
                    raw_static_eval, static_score, improving, cached_pinned_white, cached_pinned_black = compute_full_corrected_static_eval(piece_bbs, occupancy_bbs, game_state, search_context, ply)
                    static_eval_is_full = True
                    rfp_depth_factor = depth - (1 if improving else 0)
                    if rfp_depth_factor < 1:
                        rfp_depth_factor = 1
                    rfp_margin = rfp_multiplier * rfp_depth_factor
                # Re-check known-win after full eval (specialized EG may surface)
                if (static_score - rfp_margin >= beta
                        and not is_win(static_score) and not is_loss(beta)):
                    _diag_add(search_context, DIAG_RFP_CUT)
                    return (np.int32(static_score), NO_MOVE, nodes_searched, quiescence_nodes, tt_hits)

    # --- ProbCut (must be before NMP, guarded by H3) ---
    # PR-B: SF18 !is_decisive(beta); also skip known-win band (VALUE_KNOWN_WIN).
    # TUNE_PROBCUT_STYLE: 0=flat margin, 1=SF11-ish, 2=force off
    _pc_style = search_context.tune[TUNE_PROBCUT_STYLE]
    probcut_r = PROBCUT_R_IMPROVING if improving else PROBCUT_R_NOT_IMPROVING
    probcut_min_depth = probcut_r + 1
    if (search_context.enable_probcut and _pc_style != 2 and not is_exclusion_search and depth >= probcut_min_depth
            and abs(beta) < MATE_IN_MAX_PLY and abs(beta) < VALUE_KNOWN_WIN
            and not is_currently_in_check and not is_win(static_score) and not is_loss(beta)):
        # --- TT-based ProbCut Shortcut (SF 18 Step 12) ---
        # If a previous search at depth >= depth - 4 proved a lower bound >= probcut_beta,
        # we can prune immediately without capture generation, SEE, or recursive sub-search.
        if ENABLE_PROBCUT_TT_SHORTCUT and ply > 0:
            probcut_beta_s12 = beta + PROBCUT_TT_SHORTCUT_MARGIN
            if tt_entry['flag'] == TT_FLAG_BETA or tt_entry['flag'] == TT_FLAG_EXACT:
                if tt_entry['depth'] >= depth - probcut_r:
                    tt_score_s12 = np.int32(tt_entry['score'])
                    if tt_score_s12 > MATE_IN_MAX_PLY:
                        tt_score_s12 -= ply
                    elif tt_score_s12 < -MATE_IN_MAX_PLY:
                        tt_score_s12 += ply

                    if fifty_move_scale < FIFTY_MOVE_MAX_SCALE and abs(tt_score_s12) < MATE_IN_MAX_PLY:
                        tt_score_s12 = tt_score_s12 * fifty_move_scale // FIFTY_MOVE_MAX_SCALE

                    if (tt_score_s12 >= probcut_beta_s12
                            and not is_decisive(tt_score_s12) and not is_win(tt_score_s12)):
                        search_context.pv_table[ply, ply] = NO_MOVE
                        _diag_add(search_context, DIAG_PROBCUT)
                        tt_hits += np.uint64(1)
                        return (np.int32(probcut_beta_s12), tt_entry['best_move'], nodes_searched, quiescence_nodes, tt_hits)

        if _pc_style == 1:
            # SF11-ish raisedBeta; margins scaled by HCE_SCALE for our eval
            _pc_add = (PROBCUT_SF_BASE - PROBCUT_SF_IMPROVING * (1 if improving else 0)) * HCE_SCALE_NUM // HCE_SCALE_DEN
            if _pc_add < 1:
                _pc_add = 1
            probcut_beta = beta + _pc_add
        else:
            probcut_beta = beta + search_context.tune[TUNE_PROBCUT_MARGIN]
        
        # --- NEW: ProbCut TT Defense ---
        # If TT bound is UPPER (Alpha) and TT score is less than probcut_beta,
        # it is highly unlikely a reduced depth search will exceed probcut_beta. Skip ProbCut.
        skip_probcut = False
        if tt_entry['flag'] != TT_FLAG_NONE:
            tt_score_pc = np.int32(tt_entry['score'])
            if tt_score_pc > MATE_IN_MAX_PLY: tt_score_pc -= ply
            elif tt_score_pc < -MATE_IN_MAX_PLY: tt_score_pc += ply
            
            if tt_score_pc < probcut_beta:
                skip_probcut = True
                
        if not skip_probcut:
            # We must only try captures.
            pc_move_count = generate_pseudo_legal_captures_buffer(piece_bbs, occupancy_bbs, game_state, search_context.moves_buffer, ply)
            
            if static_eval_is_full and cached_pinned_white != PINNED_UNCOMPUTED_SENTINEL:
                pc_pinned_w = cached_pinned_white
                pc_pinned_b = cached_pinned_black
            else:
                pc_pinned_w = get_pinned_pieces(piece_bbs, occupancy_bbs, WHITE)
                pc_pinned_b = get_pinned_pieces(piece_bbs, occupancy_bbs, BLACK)

            score_captures_lazy(
                piece_bbs, occupancy_bbs, game_state,
                search_context.moves_buffer[ply],
                search_context.move_scores[ply],
                0, pc_move_count, search_context,
                pc_pinned_w, pc_pinned_b,
                ply
            )

            pc_moves = search_context.moves_buffer[ply]
            pc_scores = search_context.move_scores[ply]
    
            for i in range(pc_move_count):
                # Selection sort
                best_idx = i
                for j in range(i + 1, pc_move_count):
                    if pc_scores[j] > pc_scores[best_idx]:
                        best_idx = j
                pc_moves[i], pc_moves[best_idx] = pc_moves[best_idx], pc_moves[i]
                pc_scores[i], pc_scores[best_idx] = pc_scores[best_idx], pc_scores[i]
    
                pc_move = pc_moves[i]
                if pc_move == excluded_move:
                    continue
                pc_from = get_from_square(pc_move)
                pc_to = get_to_square(pc_move)
    
                # SEE filter: only try captures whose SEE >= probcut_beta - static_eval
                see_threshold_pc = probcut_beta - static_score if static_score != -INFINITY else 0
                _diag_add(search_context, DIAG_PROBCUT_SEE_TRY)
                if not _see_ge_jit(piece_bbs, occupancy_bbs, game_state[0], pc_from, pc_to, see_threshold_pc, pc_pinned_w, pc_pinned_b):
                    continue
    
                pc_original_side = game_state[0]
                pc_unmake = make_move(piece_bbs, occupancy_bbs, game_state, pc_move)
    
                # Legality check
                pc_king_bb = piece_bbs[5] if (1 - game_state[0]) == 0 else piece_bbs[11]
                pc_king_sq = get_lsb_index(pc_king_bb) if pc_king_bb else 0
                if is_square_attacked(piece_bbs, occupancy_bbs, pc_king_sq, game_state[0]):
                    unmake_move(piece_bbs, occupancy_bbs, game_state, pc_move, pc_unmake)
                    continue
    
                pc_q_score, pc_q_nodes = quiescence_search(
                    piece_bbs, occupancy_bbs, game_state,
                    _i32(-probcut_beta), _i32(-probcut_beta + 1), _i32(ply + 1), search_context, _i32(0),
                )
                quiescence_nodes += pc_q_nodes
                pc_q_score = -pc_q_score

                if search_context.stop_flag[0]:
                    unmake_move(piece_bbs, occupancy_bbs, game_state, pc_move, pc_unmake)
                    return (np.int32(0), NO_MOVE, nodes_searched, quiescence_nodes, tt_hits)

                if pc_q_score < probcut_beta:
                    unmake_move(piece_bbs, occupancy_bbs, game_state, pc_move, pc_unmake)
                    continue

                saved_pc_stack_move = search_context.move_stack[ply]
                saved_pc_stack_piece = search_context.piece_stack[ply]
                pc_moved_piece_type = pc_unmake[0]
                if pc_moved_piece_type != -1 and pc_original_side == BLACK:
                    pc_moved_piece_type += 6
                search_context.move_stack[ply] = pc_move
                search_context.piece_stack[ply] = pc_moved_piece_type
                search_context.reduction_stack[ply + 1] = 0
                res_pc = _search(
                    piece_bbs, occupancy_bbs, game_state, _i32(depth - probcut_r),
                    _i32(-probcut_beta), _i32(-probcut_beta + 1), search_context, _i32(ply + 1),
                    _u16(NO_MOVE), _b_false(), not cut_node,
                )
                search_context.move_stack[ply] = saved_pc_stack_move
                search_context.piece_stack[ply] = saved_pc_stack_piece
                pc_score = -res_pc[0]
                
                # IMPORTANT: Must aggregate the search stats from ProbCut back into the parent
                child_pc_nodes = res_pc[2]; child_pc_q_nodes = res_pc[3]
                child_pc_tth = res_pc[4]
                
                nodes_searched += child_pc_nodes; quiescence_nodes += child_pc_q_nodes
                tt_hits += child_pc_tth

                unmake_move(piece_bbs, occupancy_bbs, game_state, pc_move, pc_unmake)

                if search_context.stop_flag[0]:
                    return (np.int32(0), NO_MOVE, nodes_searched, quiescence_nodes, tt_hits)
    
                if pc_score >= probcut_beta:
                    probcut_return_score = pc_score - (probcut_beta - beta)
                    # PR-B: do not propagate unproven known-win / mate from reduced ProbCut
                    if is_win(probcut_return_score) or is_decisive(probcut_return_score):
                        probcut_return_score = beta
                    # Store in TT for future use
                    tt_store_score_pc = probcut_return_score
                    if tt_store_score_pc > MATE_IN_MAX_PLY: tt_store_score_pc += ply
                    elif tt_store_score_pc < -MATE_IN_MAX_PLY: tt_store_score_pc -= ply
                    store_tt(search_context.transposition_table, tt_key, depth - PROBCUT_TT_DEPTH_OFFSET, tt_store_score_pc, np.int16(32767), TT_FLAG_BETA, pc_move, search_context.tt_generation, tt_pv)
                    _diag_add(search_context, DIAG_PROBCUT)
                    return (np.int32(probcut_return_score), pc_move, nodes_searched, quiescence_nodes, tt_hits)

    # --- Null Move Pruning (guarded by H3) ---
    # P1: skip when static is already known-win; do not treat unproven big scores as cuts.
    # Runtime scope (TUNE_NMP_SCOPE):
    #   0 = cut_node only (legacy production before 2026-07-19)
    #   1 = all non-PV (S1a: nodes slightly down, pass −7 — not default)
    #   2 = non-PV only when depth >= mind (does NOT keep shallow cut NMP)
    #   3 = cut_node OR (non-PV and depth >= mind)  ← production default
    # gate 0=legacy / 1=SF raw / 2=SF*78 / 3=custom G_*;
    # R 0=legacy / 1=SF11 / 2=custom R_BASE+d//R_DIV.
    _nmp_scope = search_context.tune[TUNE_NMP_SCOPE]
    _nmp_mind = search_context.tune[TUNE_NMP_SCOPE_MIND]
    if _nmp_scope == 0:
        _nmp_ok_node = cut_node
    elif _nmp_scope == 1:
        _nmp_ok_node = not is_pv
    elif _nmp_scope == 2:
        _nmp_ok_node = (not is_pv) and (depth >= _nmp_mind)
    else:
        # scope >= 3: keep cut NMP everywhere + add deep non-PV NMP
        _nmp_ok_node = cut_node or ((not is_pv) and (depth >= _nmp_mind))
    if (search_context.enable_nmp and _nmp_ok_node and not is_exclusion_search and depth >= NMP_MIN_DEPTH and not is_currently_in_check
            and not (ply > 0 and search_context.move_stack[ply - 1] == NO_MOVE)
            and side_non_pawn_count >= NMP_MIN_SIDE_NON_PAWNS
            and abs(beta) < VALUE_KNOWN_WIN
            and not is_win(static_score)):

        _diag_add(search_context, DIAG_NMP_ELIGIBLE)

        _nmp_gate = search_context.tune[TUNE_NMP_GATE]
        if _nmp_gate == 0:
            # Legacy: harder mid-depth gate (see SEARCH_ANALYSIS §9.3)
            nmp_threshold = (beta - NMP_LEGACY_DEPTH_COEF * depth
                             - NMP_LEGACY_IMPROVING_COEF * (1 if improving else 0)
                             + NMP_LEGACY_BASE)
            nmp_need_ge_beta = False
        elif _nmp_gate == 3:
            # Custom continuous gate (defaults == legacy when G_* left at init)
            nmp_threshold = (beta - search_context.tune[TUNE_NMP_G_DEPTH] * depth
                             - search_context.tune[TUNE_NMP_G_IMP] * (1 if improving else 0)
                             + search_context.tune[TUNE_NMP_G_BASE])
            nmp_need_ge_beta = search_context.tune[TUNE_NMP_NEED_BETA] != 0
        else:
            # SF11: eval >= beta and staticEval >= beta + margin
            # gate=1 raw SF coeffs; gate=2 scale margin by HCE_SCALE
            _sf_margin = (-NMP_SF_MARGIN_DEPTH_COEF * depth + NMP_SF_MARGIN_BASE
                          - NMP_SF_MARGIN_IMPROVING * (1 if improving else 0))
            if _nmp_gate == 2:
                _sf_margin = _sf_margin * HCE_SCALE_NUM // HCE_SCALE_DEN
            nmp_threshold = beta + _sf_margin
            nmp_need_ge_beta = True

        nmp_gate_ok = static_score >= nmp_threshold
        if nmp_need_ge_beta:
            nmp_gate_ok = nmp_gate_ok and (static_score >= beta)

        if nmp_gate_ok:
            if not static_eval_is_full:
                raw_static_eval, static_score, improving, cached_pinned_white, cached_pinned_black = compute_full_corrected_static_eval(piece_bbs, occupancy_bbs, game_state, search_context, ply)
                static_eval_is_full = True
                if _nmp_gate == 0:
                    nmp_threshold = (beta - NMP_LEGACY_DEPTH_COEF * depth
                                     - NMP_LEGACY_IMPROVING_COEF * (1 if improving else 0)
                                     + NMP_LEGACY_BASE)
                elif _nmp_gate == 3:
                    nmp_threshold = (beta - search_context.tune[TUNE_NMP_G_DEPTH] * depth
                                     - search_context.tune[TUNE_NMP_G_IMP] * (1 if improving else 0)
                                     + search_context.tune[TUNE_NMP_G_BASE])
                else:
                    _sf_margin = (-NMP_SF_MARGIN_DEPTH_COEF * depth + NMP_SF_MARGIN_BASE
                                  - NMP_SF_MARGIN_IMPROVING * (1 if improving else 0))
                    if _nmp_gate == 2:
                        _sf_margin = _sf_margin * HCE_SCALE_NUM // HCE_SCALE_DEN
                    nmp_threshold = beta + _sf_margin
                nmp_gate_ok = static_score >= nmp_threshold
                if nmp_need_ge_beta:
                    nmp_gate_ok = nmp_gate_ok and (static_score >= beta)

        # Re-check after full eval: specialized EG may raise static into known-win band
        if nmp_gate_ok and not is_win(static_score):
            _diag_add(search_context, DIAG_NMP_GATE_PASS)
            nmp_saved_side = game_state[0]
            nmp_saved_ep   = game_state[2]
            nmp_saved_hmc  = game_state[3]
            nmp_saved_key  = game_state[4]

            search_context.move_stack[ply] = NO_MOVE
            search_context.piece_stack[ply] = -1

            make_null_move(game_state)
            _diag_add(search_context, DIAG_NMP_NULL_TRY)

            # R formula: legacy / SF11 / custom
            eval_margin = (static_score - beta) // NMP_EVAL_MARGIN_DIV
            if eval_margin < 0:
                eval_margin = 0
            elif eval_margin > NMP_EVAL_MARGIN_MAX:
                eval_margin = NMP_EVAL_MARGIN_MAX
            _nmp_r_mode = search_context.tune[TUNE_NMP_R]
            if _nmp_r_mode == 1:
                nmp_reduction = (NMP_SF_R_BASE + NMP_SF_R_DEPTH * depth) // NMP_SF_R_DIV + eval_margin
            elif _nmp_r_mode == 2:
                _r_div = search_context.tune[TUNE_NMP_R_DIV]
                if _r_div < 1:
                    _r_div = 1
                nmp_reduction = search_context.tune[TUNE_NMP_R_BASE] + depth // _r_div + eval_margin
            else:
                nmp_reduction = NMP_LEGACY_R_BASE + depth // NMP_LEGACY_R_DEPTH_DIV + eval_margin
            search_depth = max(0, depth - nmp_reduction)
            search_context.reduction_stack[ply + 1] = nmp_reduction
            res_nm = _search(
                piece_bbs, occupancy_bbs, game_state, _i32(search_depth),
                _i32(-beta), _i32(-beta + 1), search_context, _i32(ply + 1),
                _u16(NO_MOVE), _b_false(), not cut_node,
            )
            null_move_score = res_nm[0]
            child_nodes = res_nm[2]
            child_q_nodes = res_nm[3]
            child_tt_hits = res_nm[4]

            game_state[0] = nmp_saved_side
            game_state[2] = nmp_saved_ep
            game_state[3] = nmp_saved_hmc
            game_state[4] = nmp_saved_key
            null_move_score = -null_move_score

            nodes_searched += child_nodes; quiescence_nodes += child_q_nodes; tt_hits += child_tt_hits

            if search_context.stop_flag[0]:
                return (np.int32(0), NO_MOVE, nodes_searched, quiescence_nodes, tt_hits)

            if null_move_score >= beta:
                _diag_add(search_context, DIAG_NMP_NULL_FH)
                # Cap unproven wins (KNOWN_WIN or mate-distance) to beta — SF18 !is_win(null)
                if is_win(null_move_score) or is_decisive(null_move_score):
                    null_move_score = beta
                null_cutoff_verified = True

                _nmp_verify_d = search_context.tune[TUNE_NMP_VERIFY_D]
                if depth >= _nmp_verify_d and abs(null_move_score) < MATE_IN_MAX_PLY:
                    _diag_add(search_context, DIAG_NMP_VERIFY_TRY)
                    nmp_was_enabled = search_context.enable_nmp
                    search_context.enable_nmp = False
                    verification_depth = max(1, depth - nmp_reduction)
                    search_context.reduction_stack[ply] = 0
                    res_verify = _search(
                        piece_bbs, occupancy_bbs, game_state, _i32(verification_depth),
                        _i32(beta - 1), _i32(beta), search_context, _i32(ply),
                        _u16(NO_MOVE), _b_false(), cut_node,
                    )
                    search_context.enable_nmp = nmp_was_enabled

                    verify_score = res_verify[0]
                    nodes_searched += res_verify[2]
                    quiescence_nodes += res_verify[3]
                    tt_hits += res_verify[4]

                    if search_context.stop_flag[0]:
                        return (np.int32(0), NO_MOVE, nodes_searched, quiescence_nodes, tt_hits)

                    if verify_score < beta:
                        null_cutoff_verified = False
                        _diag_add(search_context, DIAG_NMP_VERIFY_FAIL)
                    else:
                        # Cap verification fail-high the same way
                        if is_win(verify_score) or is_decisive(verify_score):
                            null_move_score = beta
                        else:
                            null_move_score = verify_score

                if null_cutoff_verified:
                    search_context.pv_table[ply, ply] = NO_MOVE
                    _diag_add(search_context, DIAG_NMP_CUT)
                    return (np.int32(null_move_score), NO_MOVE, nodes_searched, quiescence_nodes, tt_hits)
    # --- M1: IIR (Internal Iterative Reduction) replaces IID ---
    # Mild IIR only (1 ply). Skip on last-iteration PV path (SF !followPV gate) —
    # reducing depth on the critical line hides tactics for little gain.
    if (tt_move == NO_MOVE and ENABLE_IIR and not is_exclusion_search
            and not is_currently_in_check and not follow_pv):
        if depth >= IIR_MIN_DEPTH:
            depth -= 1

    # --- Move Ordering ---
    # Retrieve Counter Move if available (SF: [prev_piece][prev_to])
    counter_move = NO_MOVE
    if ply > 0:
        prev_move = search_context.move_stack[ply - 1]
        prev_piece = search_context.piece_stack[ply - 1]
        if prev_move != NO_MOVE and prev_piece != -1:
            prev_to = get_to_square(prev_move)
            counter_move = search_context.counter_moves[prev_piece, prev_to]

    moves = search_context.moves_buffer[ply]
    scores = search_context.move_scores[ply]

    # Safe access to killers:
    safe_ply = min(ply, MAX_PLY - 1)
    killer_1 = search_context.killer_moves[safe_ply*2]
    killer_2 = search_context.killer_moves[safe_ply*2+1]

    # Optimization: Reuse pinned pieces from full evaluation if available
    if static_eval_is_full and cached_pinned_white != PINNED_UNCOMPUTED_SENTINEL:
        pinned_white = cached_pinned_white
        pinned_black = cached_pinned_black
    else:
        pinned_white = get_pinned_pieces(piece_bbs, occupancy_bbs, WHITE)
        pinned_black = get_pinned_pieces(piece_bbs, occupancy_bbs, BLACK)

    # --- Phase 2: Pre-compute check info for Pre-make Legality + O(1) Gives-Check ---
    _diag_add(search_context, DIAG_MAIN_GEOMETRY_NODE)
    side_to_move_s = game_state[0]
    opponent_side_s = np.uint8(1 - side_to_move_s)
    
    # Our king info (for pre-make legality)
    our_king_bb_s = piece_bbs[5] if side_to_move_s == WHITE else piece_bbs[11]
    our_king_sq_s = get_lsb_index(our_king_bb_s) if our_king_bb_s else np.int8(0)
    pinned_us_s = pinned_white if side_to_move_s == WHITE else pinned_black
    all_occ_s = occupancy_bbs[2]

    # P3 shadow diagnostics: build checker/evasion geometry only while
    # diagnostics are enabled.  Production ordering remains the baseline.
    main_checkers_s = np.uint64(0)
    main_evasion_targets_s = np.uint64(0)
    main_checker_count_s = np.int32(0)
    if is_currently_in_check and search_context.diag_enabled:
        _diag_add(search_context, DIAG_MAIN_IN_CHECK_NODE)
        opponent_offset_s = 6 if opponent_side_s == BLACK else 0
        main_checkers_s |= (
            piece_bbs[opponent_offset_s + PAWN]
            & PAWN_ATTACKS[opponent_side_s, our_king_sq_s]
        )
        main_checkers_s |= (
            piece_bbs[opponent_offset_s + KNIGHT]
            & KNIGHT_ATTACKS[our_king_sq_s]
        )
        main_checkers_s |= (
            (piece_bbs[opponent_offset_s + BISHOP]
             | piece_bbs[opponent_offset_s + QUEEN])
            & get_bishop_attacks(our_king_sq_s, all_occ_s)
        )
        main_checkers_s |= (
            (piece_bbs[opponent_offset_s + ROOK]
             | piece_bbs[opponent_offset_s + QUEEN])
            & get_rook_attacks(our_king_sq_s, all_occ_s)
        )
        main_checkers_s |= (
            piece_bbs[opponent_offset_s + KING]
            & KING_ATTACKS[our_king_sq_s]
        )
        main_checker_count_s = count_bits(main_checkers_s)
        if main_checker_count_s == 1:
            main_checker_sq_s = get_lsb_index(main_checkers_s)
            main_evasion_targets_s = (
                main_checkers_s | SQUARES_BETWEEN[our_king_sq_s, main_checker_sq_s]
            )
    
    # Their king info (for O(1) gives-check detection)
    their_king_bb_s = piece_bbs[11] if side_to_move_s == WHITE else piece_bbs[5]
    their_king_sq_s = get_lsb_index(their_king_bb_s) if their_king_bb_s else np.int8(0)
    
    # Pre-compute check_squares: squares from which each piece type gives check
    # Equivalent to SF18's set_check_info(): checkSquares[Pt] = attacks_bb<Pt>(ksq)
    check_sq_pawn_s = PAWN_ATTACKS[side_to_move_s, their_king_sq_s]
    check_sq_knight_s = KNIGHT_ATTACKS[their_king_sq_s]
    check_sq_bishop_s = get_bishop_attacks(their_king_sq_s, all_occ_s)
    check_sq_rook_s = get_rook_attacks(their_king_sq_s, all_occ_s)
    check_sq_queen_s = check_sq_bishop_s | check_sq_rook_s
    
    # Pre-compute blockers_for_their_king (for discovered check detection)
    blockers_for_their_king_s, _ = get_blockers_for_king(piece_bbs, occupancy_bbs, opponent_side_s)

    best_move, max_eval = NO_MOVE, np.int32(-INFINITY)
    best_moved_piece = np.int8(-1)
    best_is_capture = False
    best_victim_type = np.int8(-1)
    quiet_move_counter = 0
    legal_moves_tried = 0
    searched_legal_moves = 0
    pruned_moves = 0
    
    # Track tried quiet moves for history malus
    quiet_moves_tried = search_context.quiet_moves_tried[ply]
    quiet_pieces_tried = search_context.quiet_pieces_tried[ply]
    quiet_moves_tried_count = 0

    # Track tried capture moves for history malus
    capture_moves_tried = search_context.capture_moves_tried[ply]
    capture_aggressor_tried = search_context.capture_aggressor_tried[ply]
    capture_victim_tried = search_context.capture_victim_tried[ply]
    capture_tosq_tried = search_context.capture_tosq_tried[ply]
    capture_moves_tried_count = 0

    # Staged Move Generation State Init
    search_context.mp_stage[ply] = STAGE_TT_MOVE
    search_context.mp_current_idx[ply] = 0
    search_context.mp_captures_end[ply] = 0
    search_context.mp_quiets_end[ply] = 0
    search_context.mp_bad_captures_count[ply] = 0
    search_context.mp_bad_captures_idx[ply] = 0

    # opponent_pieces_bb is constant throughout this node (make/unmake restores occupancy)
    opponent_pieces_bb = occupancy_bbs[1] if game_state[0] == 0 else occupancy_bbs[0]

    while True:
        move = get_next_move(
            piece_bbs, occupancy_bbs, game_state, search_context, ply,
            tt_move, excluded_move, killer_1, killer_2, counter_move,
            pinned_white, pinned_black, pawn_key_idx,
            is_currently_in_check, our_king_sq_s, our_king_bb_s,
            opponent_side_s, pinned_us_s, main_checker_count_s,
            main_evasion_targets_s,
        )
        if move == NO_MOVE:
            break
        diag_picker_source_s = np.int32(MOVE_ORDER_SOURCE_OTHER)
        diag_nodes_before_move_s = np.uint64(0)
        if search_context.diag_enabled:
            diag_picker_source_s = _diag_move_source(
                move, tt_move, search_context.mp_stage[ply],
            )
            # Node-local work already spent before this candidate's child.
            # It is only an ordering-cost proxy; it never enters search logic.
            diag_nodes_before_move_s = nodes_searched + quiescence_nodes
        # --- Pre-move checks for extensions and move type ---
        original_side = game_state[0]
        from_sq = get_from_square(move)
        to_sq = get_to_square(move)
        flag = get_special_move_flag(move)
        is_capture = ((opponent_pieces_bb & BB_SQUARES[to_sq]) != 0) or (flag == SPECIAL_MOVE_FLAG_EN_PASSANT)
        is_promotion = flag == SPECIAL_MOVE_FLAG_PROMOTION
        is_pseudo_quiet = not is_capture and not is_promotion
        is_bad_capture = False
        # P1 diagnostics: remember whether this quiet candidate passed the
        # shallow SEE gate before the legality fast path below.
        quiet_see_passed_s = False
        skip_post_legality_s = False
        quiet_counted = False
        singular_extension = 0
        check_gate_see_ok = False

        # P1 shadow diagnostics: classify before SEE, but never reject or
        # reorder a production candidate here.  Baseline legality remains
        # after the shallow SEE gate below.
        if (ply > 0 and move != tt_move and search_context.enable_see_pruning and not is_exclusion_search
                and depth <= PRUNING_SHALLOW_DEPTH and not is_currently_in_check
                and not is_pv and is_pseudo_quiet
                and not low_material_pruning_guard and not follow_pv):
            if search_context.diag_enabled:
                _diag_add(search_context, DIAG_MAIN_QUIET_SEE_PRECHECK)
                if from_sq != our_king_sq_s and flag != SPECIAL_MOVE_FLAG_EN_PASSANT:
                    if ((pinned_us_s & BB_SQUARES[from_sq])
                            and not (LINE_BB[from_sq, to_sq] & our_king_bb_s)):
                        _diag_add(search_context, DIAG_MAIN_QUIET_PRE_PIN_ILLEGAL)
                elif from_sq == our_king_sq_s and flag != SPECIAL_MOVE_FLAG_CASTLING:
                    occ_without_king_diag = occupancy_bbs[2] ^ our_king_bb_s
                    if is_square_attacked_with_occ(
                            piece_bbs, to_sq, occ_without_king_diag,
                            opponent_side_s):
                        _diag_add(search_context, DIAG_MAIN_QUIET_PRE_KING_ILLEGAL)

        # --- NEW: Shallow Depth Pruning (Stockfish Step 14) ---
        if (ply > 0 and move != tt_move and search_context.enable_see_pruning and not is_exclusion_search
                and depth <= PRUNING_SHALLOW_DEPTH and not is_currently_in_check and not is_pv):
            side_to_move = original_side

            if is_capture:
                cap_piece = int(find_piece_type_on_square_side(piece_bbs, from_sq, side_to_move))
                if flag == SPECIAL_MOVE_FLAG_EN_PASSANT:
                    cap_victim = PAWN
                else:
                    opp = 1 - side_to_move
                    v = int(find_piece_type_on_square_side(piece_bbs, to_sq, opp))
                    cap_victim = (v % 6) if v >= 0 else -1

                cap_hist = 0
                if cap_piece >= 0 and cap_victim >= 0:
                    cap_hist = int(search_context.capture_history[cap_piece, to_sq, cap_victim])

                # --- Stockfish 18 Step 14: Capture Futility Pruning (Pre-make) ---
                if ENABLE_CAPTURE_FUTILITY and static_score != -INFINITY and cap_piece >= 0 and cap_victim >= 0 and not low_material_pruning_guard:
                    pre_move_gives_check = flag == SPECIAL_MOVE_FLAG_CASTLING or flag == SPECIAL_MOVE_FLAG_EN_PASSANT or is_promotion
                    if not pre_move_gives_check:
                        pre_move_gives_check = _qsearch_ordinary_move_gives_check(
                            piece_bbs, move, side_to_move_s, their_king_bb_s,
                            check_sq_pawn_s, check_sq_knight_s, check_sq_bishop_s,
                            check_sq_rook_s, check_sq_queen_s,
                            blockers_for_their_king_s,
                        )
                    if not pre_move_gives_check:
                        lmr_r = compute_lmr_reduction_1024(depth, searched_legal_moves + 1, improving)
                        cap_lmr_d = (depth - 1) - (lmr_r // 1024)
                        if cap_lmr_d < 0:
                            cap_lmr_d = 0
                        if cap_lmr_d < CAP_FP_MAX_LMR_DEPTH:
                            victim_val = int(MG_MATERIAL_VALUES[cap_victim])
                            futility_value = (
                                static_score
                                + CAP_FP_BASE
                                + CAP_FP_LMR_MULT * cap_lmr_d
                                + victim_val
                                + (cap_hist * CAP_FP_CAPTHIST_NUM // CAP_FP_CAPTHIST_DEN)
                            )
                            if futility_value <= alpha:
                                pruned_moves += 1
                                _diag_add(search_context, DIAG_SEE_CAP_SKIP)
                                continue

                # --- Capture SEE Pruning ---
                threshold = search_context.tune[TUNE_SEE_CAP_MARGIN] * depth
                if ENABLE_CAPTURE_FUTILITY and cap_piece >= 0 and cap_victim >= 0:
                    threshold = threshold - cap_hist * CAP_SEE_CAPTHIST_NUM // CAP_SEE_CAPTHIST_DEN
                _diag_add(search_context, DIAG_MAIN_CAP_SEE_TRY)
                if not _see_ge_jit(piece_bbs, occupancy_bbs, side_to_move, from_sq, to_sq, threshold, pinned_white, pinned_black):
                    pruned_moves += 1
                    _diag_add(search_context, DIAG_SEE_CAP_SKIP)
                    continue
            # Quiet SEE skip: also respect followPV (do not prune last-iter PV path)
            elif is_pseudo_quiet and not low_material_pruning_guard and not follow_pv:
                # SF11 and SF18 classify checking quiets with captures/checks,
                # not with the more selective quiet-history pruning branch.
                # Determine ordinary direct/discovered checks before make_move;
                # castling is rare and protected here because its rook move is
                # not represented by the ordinary geometry helper.
                pre_move_gives_check = flag == SPECIAL_MOVE_FLAG_CASTLING
                if not pre_move_gives_check:
                    pre_move_gives_check = _qsearch_ordinary_move_gives_check(
                        piece_bbs, move, side_to_move_s, their_king_bb_s,
                        check_sq_pawn_s, check_sq_knight_s, check_sq_bishop_s,
                        check_sq_rook_s, check_sq_queen_s,
                        blockers_for_their_king_s,
                    )

                can_prune_quiet = not pre_move_gives_check
                if pre_move_gives_check:
                    # Common SF11/SF18 safety shape: checking moves may still
                    # fail a permissive SEE gate, but never the quiet lmrDepth
                    # gate.  This preserves speculative checks without making
                    # all unsound checks free.
                    threshold = search_context.tune[TUNE_SEE_CAP_MARGIN] * depth
                    _diag_add(search_context, DIAG_MAIN_CHECK_SEE_ROUTE)
                    if not _see_ge_jit(piece_bbs, occupancy_bbs, side_to_move, from_sq, to_sq, threshold, pinned_white, pinned_black):
                        pruned_moves += 1
                        continue
                else:
                    see_piece = int(find_piece_type_on_square_side(
                        piece_bbs, from_sq, side_to_move,
                    ))
                    quiet_move_counter += 1
                    quiet_counted = True

                    # --- Step 14a: Late Move Pruning (LMP) ---
                    if (search_context.enable_lmp and not is_exclusion_search):
                        limit = LMP_MOVE_COUNT[min(depth, MAX_PLY - 1)] // (2 - (1 if improving else 0))
                        limit = (limit * search_context.tune[TUNE_LMP_SCALE]) // 100
                        limit = max(limit, LMP_MIN_LIMIT)

                        if quiet_move_counter >= limit:
                            _diag_add(search_context, DIAG_LMP_SKIP)
                            pruned_moves += 1
                            continue

                    # --- Step 14b: History Pruning ---
                    prune_quiet_move = False
                    if ENABLE_HISTORY_PRUNING and see_piece != -1:
                        # Continuation History Pruning (Stockfish 17 Step 14)
                        if ENABLE_CONTINUATION_HISTORY_PRUNING:
                            cont_prune_score = get_continuation_pruning_score(
                                search_context, ply, see_piece, to_sq, pawn_key_idx
                            )
                            if cont_prune_score < PRUNING_CONTINUATION_THRESHOLD * depth:
                                prune_quiet_move = True

                        if not prune_quiet_move:
                            history_score = get_quiet_stat_score(
                                search_context, ply, from_sq, to_sq,
                                see_piece, pawn_key_idx,
                                HISTORY_PRUNE_CONT1_FLOOR,
                            )
                            main_history_score = search_context.history_table[see_piece, to_sq]
                        if search_context.diag_enabled:
                            hp_raw_terms = get_quiet_stat_components(
                                search_context, ply, from_sq, to_sq,
                                see_piece, pawn_key_idx,
                            )
                            hp_raw_cont1 = hp_raw_terms[3]
                            if hp_raw_cont1 < HISTORY_PRUNE_CONT1_FLOOR:
                                hp_raw_score = np.int64(0)
                                for hp_raw_i in range(HIST_PRUNE_LAYER_COUNT):
                                    hp_raw_score += hp_raw_terms[hp_raw_i]
                                _diag_add(search_context, DIAG_HP_CONT1_CAP_HIT)
                                _diag_add(
                                    search_context,
                                    DIAG_HP_CONT1_CAP_RELIEF_SUM,
                                    HISTORY_PRUNE_CONT1_FLOOR - hp_raw_cont1,
                                )
                                hp_threshold = PRUNING_HISTORY_THRESHOLD * depth
                                if (main_history_score < 0
                                        and hp_raw_score < hp_threshold
                                        and history_score >= hp_threshold):
                                    _diag_add(search_context, DIAG_HP_CONT1_CAP_RESCUE)
                        if history_score < PRUNING_HISTORY_THRESHOLD * depth and main_history_score < 0:
                            prune_quiet_move = True

                    if prune_quiet_move:
                        pruned_moves += 1
                        _diag_add(search_context, DIAG_HIST_PRUNE)
                        if search_context.diag_enabled:
                            # P5/P6 attribution: reconstruct the production pruning
                            hp_terms = get_quiet_stat_components(
                                search_context, ply, from_sq, to_sq,
                                see_piece, pawn_key_idx,
                            )
                            hp_threshold = PRUNING_HISTORY_THRESHOLD * depth
                            hp_dominant_idx = 0
                            hp_dominant_value = hp_terms[0]
                            hp_cont_sum = np.int64(0)
                            for hp_i in range(HIST_PRUNE_LAYER_COUNT):
                                hp_value = hp_terms[hp_i]
                                if hp_i == 3 and hp_value < HISTORY_PRUNE_CONT1_FLOOR:
                                    hp_value = HISTORY_PRUNE_CONT1_FLOOR
                                if hp_i >= 3:
                                    hp_cont_sum += hp_value
                                if hp_value < 0:
                                    _diag_add(search_context, DIAG_HP_NEG_COUNT_BASE + hp_i)
                                    _diag_add(
                                        search_context,
                                        DIAG_HP_NEG_ABS_SUM_BASE + hp_i,
                                        -hp_value,
                                    )
                                    if history_score - hp_value >= hp_threshold:
                                        _diag_add(search_context, DIAG_HP_DECISIVE_BASE + hp_i)
                                if hp_value < hp_dominant_value:
                                    hp_dominant_value = hp_value
                                    hp_dominant_idx = hp_i
                            _diag_add(search_context, DIAG_HP_DOMINANT_BASE + hp_dominant_idx)

                            if history_score - hp_cont_sum >= hp_threshold:
                                _diag_add(search_context, DIAG_HP_CONT_COALITION_DECISIVE)
                            if history_score - hp_cont_sum < hp_threshold:
                                _diag_add(search_context, DIAG_HP_NONCONT_ALREADY_PRUNES)
                            if (depth >= LMR_MIN_DEPTH
                                    and quiet_move_counter >= LMR_MIN_QUIET_MOVE_INDEX):
                                _diag_add(search_context, DIAG_HP_LMR_ELIGIBLE)
                            if main_history_score >= -512:
                                _diag_add(search_context, DIAG_HP_MAIN_GATE_NEAR_ZERO)

                            hp_margin = hp_threshold - history_score
                            if hp_margin <= 1024:
                                _diag_add(search_context, DIAG_HP_MARGIN_NEAR)
                            elif hp_margin <= 4096:
                                _diag_add(search_context, DIAG_HP_MARGIN_MID)
                            else:
                                _diag_add(search_context, DIAG_HP_MARGIN_FAR)
                            _diag_add(search_context, DIAG_HP_MARGIN_SUM, hp_margin)
                            _diag_add(search_context, DIAG_HP_DEPTH_SUM, depth)
                            _diag_add(
                                search_context,
                                DIAG_HP_MOVE_INDEX_SUM,
                                quiet_move_counter,
                            )
                        continue

                    # --- Step 14c: Futility Pruning (FP) ---
                    if (search_context.enable_fp and not is_exclusion_search and depth <= FP_MAX_DEPTH
                            and static_score != -INFINITY):
                        fp_lmr_d = compute_quiet_pruning_lmr_depth(
                            depth, quiet_move_counter, improving, search_context, ply,
                            from_sq, to_sq, see_piece,
                        )
                        margin = search_context.tune[TUNE_FP_BASE] + search_context.tune[TUNE_FP_MULT] * fp_lmr_d
                        if dissonance > FP_DISSONANCE_THRESHOLD:
                            margin += dissonance // 2

                        if margin > 0 and static_score + margin < alpha:
                            if not static_eval_is_full:
                                raw_static_eval, static_score, improving, cached_pinned_white, cached_pinned_black = compute_full_corrected_static_eval(
                                    piece_bbs, occupancy_bbs, game_state, search_context, ply
                                )
                                static_eval_is_full = True
                                fp_lmr_d = compute_quiet_pruning_lmr_depth(
                                    depth, quiet_move_counter, improving, search_context, ply,
                                    from_sq, to_sq, see_piece,
                                )
                                margin = search_context.tune[TUNE_FP_BASE] + search_context.tune[TUNE_FP_MULT] * fp_lmr_d
                                if dissonance > FP_DISSONANCE_THRESHOLD:
                                    margin += dissonance // 2
                            if margin > 0 and static_score + margin < alpha:
                                _diag_add(search_context, DIAG_FP_SKIP)
                                pruned_moves += 1
                                continue

                    # --- Step 14d: Quiet SEE Pruning ---
                    see_lmr_d = compute_quiet_pruning_lmr_depth(
                        depth, quiet_move_counter, improving, search_context, ply,
                        from_sq, to_sq, see_piece,
                    )
                    threshold = (
                        search_context.tune[TUNE_SEE_QUIET_MARGIN]
                        * see_lmr_d * see_lmr_d
                    )
                    _diag_add(search_context, DIAG_MAIN_QUIET_SEE_TRY)
                    if not _see_ge_jit(piece_bbs, occupancy_bbs, side_to_move, from_sq, to_sq, threshold, pinned_white, pinned_black):
                        pruned_moves += 1
                        _diag_add(search_context, DIAG_SEE_QUIET_SKIP)
                        continue
                quiet_see_passed_s = True
                _diag_add(search_context, DIAG_MAIN_QUIET_SEE_PASS)
                  
        # --- Pre-make Legality Fast Path (SF18-style Phase 2) ---
        if not skip_post_legality_s and not is_currently_in_check:
            if from_sq != our_king_sq_s and flag != SPECIAL_MOVE_FLAG_EN_PASSANT:
                # Non-king, non-EP move: use pin-based O(1) legality
                from_bb_s = BB_SQUARES[from_sq]
                if not (pinned_us_s & from_bb_s):
                    # Not pinned → guaranteed legal
                    skip_post_legality_s = True
                elif LINE_BB[from_sq, to_sq] & our_king_bb_s:
                    # Pinned but moving along pin ray → still legal
                    skip_post_legality_s = True
                else:
                    # Pinned and moving off pin ray → illegal, skip make/unmake entirely
                    if quiet_see_passed_s:
                        _diag_add(search_context, DIAG_MAIN_QUIET_POST_PIN_ILLEGAL)
                    continue
            elif from_sq == our_king_sq_s and flag != SPECIAL_MOVE_FLAG_CASTLING:
                # King move (non-castling): check destination with king removed from occupancy
                occ_without_king_s = occupancy_bbs[2] ^ our_king_bb_s
                if is_square_attacked_with_occ(piece_bbs, to_sq, occ_without_king_s, opponent_side_s):
                    if quiet_see_passed_s:
                        _diag_add(search_context, DIAG_MAIN_QUIET_POST_KING_ILLEGAL)
                    continue  # King walks into attack
                skip_post_legality_s = True
            # else: EP or Castling — fall through to post-make check

        if quiet_see_passed_s and not skip_post_legality_s:
            _diag_add(search_context, DIAG_MAIN_QUIET_FALLBACK_TRY)

        # Capture LMR consumes SEE(0), but SEE must inspect the parent board.
        # Avoid the old unconditional call for shallow/first captures that can
        # never enter LMR, while keeping the result available after make_move.
        if (is_capture and search_context.enable_lmr and depth >= LMR_MIN_DEPTH
                and searched_legal_moves >= 1):
            _diag_add(search_context, DIAG_LMR_CAP_SEE_TRY)
            is_bad_capture = not _see_ge_jit(
                piece_bbs, occupancy_bbs, original_side,
                from_sq, to_sq, 0, pinned_white, pinned_black,
            )

        # Pre-compute SEE on parent board for check extension to avoid post-make unmake/remake round-trip
        if ENABLE_CHECK_EXTENSION and not is_pv and move != tt_move and depth > CHECK_EXT_NON_PV_MAX_DEPTH:
            gives_check_candidate = (
                flag == SPECIAL_MOVE_FLAG_CASTLING or flag == SPECIAL_MOVE_FLAG_EN_PASSANT or is_promotion
                or _qsearch_ordinary_move_gives_check(
                    piece_bbs, move, original_side, their_king_bb_s,
                    check_sq_pawn_s, check_sq_knight_s, check_sq_bishop_s,
                    check_sq_rook_s, check_sq_queen_s,
                    blockers_for_their_king_s,
                )
            )
            if gives_check_candidate:
                _diag_add(search_context, DIAG_CHECK_GATE_SEE_TRY)
                if (is_capture and search_context.enable_lmr and depth >= LMR_MIN_DEPTH
                        and searched_legal_moves >= 1):
                    check_gate_see_ok = not is_bad_capture
                else:
                    check_gate_see_ok = _see_ge_jit(
                        piece_bbs, occupancy_bbs, original_side,
                        from_sq, to_sq, 0, pinned_white, pinned_black,
                    )

        # --- Step 15: Singular Extensions & Multi-Cut (Pre-make, move == tt_move) ---
        if (ENABLE_SINGULAR_EXTENSIONS and ply > 0 and excluded_move == NO_MOVE
                and move == tt_move and depth >= MIN_SINGULAR_DEPTH + (1 if tt_pv else 0)
                and not is_currently_in_check and skip_post_legality_s):
            if ((tt_entry['flag'] == TT_FLAG_BETA or tt_entry['flag'] == TT_FLAG_EXACT)
                    and tt_entry['depth'] >= depth - SINGULAR_TT_DEPTH_SLACK):
                se_tt_score = np.int32(tt_entry['score'])
                if se_tt_score > MATE_IN_MAX_PLY:
                    se_tt_score -= ply
                elif se_tt_score < -MATE_IN_MAX_PLY:
                    se_tt_score += ply

                if abs(se_tt_score) < MATE_IN_MAX_PLY:
                    se_tt_pv_bonus = SINGULAR_TTPV_BONUS if (tt_pv and not is_pv) else 0
                    singular_margin = (SINGULAR_MARGIN_BASE + se_tt_pv_bonus) * depth // SINGULAR_MARGIN_DIV
                    exclusion_beta = se_tt_score - singular_margin

                    # Save MovePicker & Search Context State to prevent Singular Extension from clobbering the parent
                    saved_mp_stage = search_context.mp_stage[ply]
                    saved_mp_idx = search_context.mp_current_idx[ply]
                    saved_mp_captures = search_context.mp_captures_end[ply]
                    saved_mp_quiets = search_context.mp_quiets_end[ply]
                    saved_mp_bad_cap = search_context.mp_bad_captures_count[ply]
                    saved_mp_bad_idx = search_context.mp_bad_captures_idx[ply]
                    saved_follow_pv = search_context.follow_pv_stack[ply]
                    saved_static_eval = search_context.static_eval_stack[ply]
                    saved_tt_hit = search_context.tt_hit_stack[ply]
                    saved_tt_pv = search_context.tt_pv_stack[ply]
                    saved_reduction = search_context.reduction_stack[ply]
                    saved_cutoff_cnt = search_context.cutoff_cnt[ply]

                    search_context.reduction_stack[ply] = 0
                    _diag_add(search_context, DIAG_SINGULAR_TRY)
                    res_ex = _search(
                        piece_bbs, occupancy_bbs, game_state, _i32((depth - 1) // 2),
                        _i32(exclusion_beta - 1), _i32(exclusion_beta), search_context, _i32(ply),
                        _u16(tt_move), _b_false(), cut_node,
                    )
                    exclusion_score = res_ex[0]
                    nodes_searched += res_ex[2]
                    quiescence_nodes += res_ex[3]
                    tt_hits += res_ex[4]

                    # Restore MovePicker & Search Context State
                    search_context.mp_stage[ply] = saved_mp_stage
                    search_context.mp_current_idx[ply] = saved_mp_idx
                    search_context.mp_captures_end[ply] = saved_mp_captures
                    search_context.mp_quiets_end[ply] = saved_mp_quiets
                    search_context.mp_bad_captures_count[ply] = saved_mp_bad_cap
                    search_context.mp_bad_captures_idx[ply] = saved_mp_bad_idx
                    search_context.follow_pv_stack[ply] = saved_follow_pv
                    search_context.static_eval_stack[ply] = saved_static_eval
                    search_context.tt_hit_stack[ply] = saved_tt_hit
                    search_context.tt_pv_stack[ply] = saved_tt_pv
                    search_context.reduction_stack[ply] = saved_reduction
                    search_context.cutoff_cnt[ply] = saved_cutoff_cnt
                    search_context.pv_table[ply, ply] = NO_MOVE

                    # 15a. Multi-cut pruning (SF18 Step 15)
                    # If without ttMove another move still achieves score >= beta, prune immediately (zero make/unmake!)
                    if search_context.enable_multicut and exclusion_score >= beta and not is_decisive(exclusion_score):
                        return (np.int32(exclusion_score), NO_MOVE, nodes_searched, quiescence_nodes, tt_hits)

                    # 15b. Search stop check
                    if search_context.stop_flag[0]:
                        return (np.int32(0), NO_MOVE, nodes_searched, quiescence_nodes, tt_hits)

                    # 15c. Extension calculation
                    if exclusion_score < exclusion_beta:
                        singular_extension = 1
                        _diag_add(search_context, DIAG_SINGULAR_EXT)
                        double_margin = SINGULAR_DOUBLE_EXT_MULTIPLIER * depth
                        if depth >= SINGULAR_DOUBLE_EXT_MIN_DEPTH and exclusion_score < exclusion_beta - double_margin:
                            singular_extension = 2
                            _diag_add(search_context, DIAG_DOUBLE_EXT)
                    elif exclusion_score >= exclusion_beta:
                        # Negative extensions when not singular (SF18 Step 15)
                        if not is_pv:
                            if exclusion_score >= beta:
                                singular_extension = -2
                            elif cut_node:
                                singular_extension = -1

        # --- Make the move ---
        unmake_info = make_move(piece_bbs, occupancy_bbs, game_state, move)
        moved_piece_type = unmake_info[0]
        if moved_piece_type != -1 and original_side == BLACK:
            moved_piece_type += 6
        
        # --- Post-make Legality Fallback (in-check evasions, EP, Castling) ---
        if not skip_post_legality_s:
            post_king_bb = piece_bbs[5] if (1 - game_state[0]) == 0 else piece_bbs[11]
            post_king_sq = get_lsb_index(post_king_bb) if post_king_bb else 0
            if is_square_attacked(piece_bbs, occupancy_bbs, post_king_sq, game_state[0]):
                if quiet_see_passed_s:
                    _diag_add(search_context, DIAG_MAIN_QUIET_FALLBACK_REJECT)
                unmake_move(piece_bbs, occupancy_bbs, game_state, move, unmake_info)
                continue
        
        # --- O(1) Gives-Check Detection (SF18-style) ---
        is_giving_check_after_move = False
        
        # 1. Direct check: does the moved piece land on a check square?
        to_bb_s = BB_SQUARES[to_sq]
        base_piece_type = moved_piece_type % 6 if moved_piece_type != -1 else -1
        
        if base_piece_type == PAWN:
            if is_promotion:
                # Promotion changes occupancy (pawn removed, new piece appears),
                # so pre-computed slider check_squares may be inaccurate.
                # Knight promotion is a leaper — always correct with O(1).
                # For slider promotions, use post-make is_square_attacked.
                promo_piece = get_promotion_piece(move)
                if promo_piece == PROMO_KNIGHT:
                    if check_sq_knight_s & to_bb_s:
                        is_giving_check_after_move = True
                else:
                    # Queen/Rook/Bishop promotion: post-make check for correctness
                    their_king_bb_promo = piece_bbs[11] if original_side == WHITE else piece_bbs[5]
                    their_king_sq_promo = get_lsb_index(their_king_bb_promo) if their_king_bb_promo else 0
                    if is_square_attacked(piece_bbs, occupancy_bbs, their_king_sq_promo, original_side):
                        is_giving_check_after_move = True
            else:
                if check_sq_pawn_s & to_bb_s:
                    is_giving_check_after_move = True
        elif base_piece_type == KNIGHT:
            if check_sq_knight_s & to_bb_s:
                is_giving_check_after_move = True
        elif base_piece_type == BISHOP:
            if check_sq_bishop_s & to_bb_s:
                is_giving_check_after_move = True
        elif base_piece_type == ROOK:
            if check_sq_rook_s & to_bb_s:
                is_giving_check_after_move = True
        elif base_piece_type == QUEEN:
            if check_sq_queen_s & to_bb_s:
                is_giving_check_after_move = True
        
        # 2. Discovered check: if the moved piece was blocking a slider ray to their king
        if not is_giving_check_after_move:
            from_bb_dc = BB_SQUARES[from_sq]
            if blockers_for_their_king_s & from_bb_dc:
                # The piece was blocking a ray to their king.
                # If it moved off that ray, it's a discovered check.
                if not (LINE_BB[from_sq, to_sq] & their_king_bb_s):
                    is_giving_check_after_move = True
        
        # 3. Special case: EP discovered check through the captured pawn
        if not is_giving_check_after_move and flag == SPECIAL_MOVE_FLAG_EN_PASSANT:
            # The captured pawn is removed — this could open a rank for slider attacks
            # Use a post-make check for this rare case
            # Note: after make_move, game_state[0] has flipped to opponent.
            # We want to check if THEIR king is attacked by OUR side (original_side).
            their_king_bb_post = piece_bbs[11] if original_side == WHITE else piece_bbs[5]
            their_king_sq_post = get_lsb_index(their_king_bb_post) if their_king_bb_post else 0
            if is_square_attacked(piece_bbs, occupancy_bbs, their_king_sq_post, original_side):
                is_giving_check_after_move = True
        
        # 4. Special case: Castling gives check via the rook's final position
        #    Use post-make is_square_attacked since occupancy changes after castling
        #    (both king and rook move). Castling is rare so this is negligible overhead.
        if not is_giving_check_after_move and flag == SPECIAL_MOVE_FLAG_CASTLING:
            their_king_bb_castle = piece_bbs[11] if original_side == WHITE else piece_bbs[5]
            their_king_sq_castle = get_lsb_index(their_king_bb_castle) if their_king_bb_castle else 0
            if is_square_attacked(piece_bbs, occupancy_bbs, their_king_sq_castle, original_side):
                is_giving_check_after_move = True
            
        legal_moves_tried += 1

        search_context.move_stack[ply] = move # Record move in stack
        search_context.piece_stack[ply] = moved_piece_type # Record piece in stack
        
        # Final determination of quiet move
        is_quiet_move = is_pseudo_quiet
        if is_quiet_move and not quiet_counted:
            quiet_move_counter += 1

        # --- Determine total extension ---
        check_extension = 0
        if ENABLE_CHECK_EXTENSION and is_giving_check_after_move:
            if is_pv or move == tt_move or depth <= CHECK_EXT_NON_PV_MAX_DEPTH:
                check_extension = 1
            elif check_gate_see_ok:
                check_extension = 1
        
        current_extension = check_extension
        if singular_extension != 0:
            if singular_extension > 0:
                current_extension = max(check_extension, singular_extension)
            else:
                current_extension = singular_extension
        if check_extension > 0:
            _diag_add(search_context, DIAG_CHECK_EXT)

        search_depth = max(0, depth - 1 + current_extension)

        searched_legal_moves += 1
        search_context.move_count_stack[ply] = searched_legal_moves
        search_context.move_is_capture_stack[ply] = is_capture
        if search_context.diag_enabled:
            _diag_add(
                search_context,
                DIAG_ORDER_TRY_BASE + diag_picker_source_s,
            )
            if is_quiet_move:
                if move == killer_1:
                    _diag_add(search_context, DIAG_ORDER_KILLER1_TRY)
                elif move == killer_2:
                    _diag_add(search_context, DIAG_ORDER_KILLER2_TRY)
                elif move == counter_move:
                    _diag_add(search_context, DIAG_ORDER_COUNTER_TRY)

        if is_quiet_move:
            quiet_moves_tried[quiet_moves_tried_count] = move
            quiet_pieces_tried[quiet_moves_tried_count] = moved_piece_type
            quiet_moves_tried_count += 1
        elif is_capture and moved_piece_type != -1:
            victim_type = unmake_info[1]  # relative 0..5 or -1
            if victim_type != -1:
                if capture_moves_tried_count < 64:
                    capture_moves_tried[capture_moves_tried_count] = move
                    capture_aggressor_tried[capture_moves_tried_count] = moved_piece_type
                    capture_victim_tried[capture_moves_tried_count] = victim_type % 6
                    capture_tosq_tried[capture_moves_tried_count] = to_sq
                    capture_moves_tried_count += 1

        evaluation = 0
        if searched_legal_moves == 1:
            search_context.reduction_stack[ply + 1] = 0
            res = _search(
                piece_bbs, occupancy_bbs, game_state, _i32(search_depth),
                _i32(-beta), _i32(-alpha), search_context, _i32(ply + 1),
                _u16(NO_MOVE), is_pv, _b_false(),
            )
            evaluation = -res[0]
            child_nodes = res[2]; child_q_nodes = res[3]; child_tt_hits = res[4]
        else:
            # --- 1024-scale LMR (Phase B) ---
            lmr = 0
            lmr_stat_score = 0
            if search_context.enable_lmr and depth >= LMR_MIN_DEPTH:
                is_eligible = False
                if is_quiet_move and quiet_move_counter >= LMR_MIN_QUIET_MOVE_INDEX and moved_piece_type != -1:
                    is_eligible = True
                elif is_capture:
                    is_eligible = True

                if is_eligible:
                    # 1. Base LMR 1024-scale calculation:
                    r = compute_lmr_reduction_1024(
                        depth,
                        searched_legal_moves,
                        improving,
                        search_context.tune[TUNE_LMR_TABLE_SCALE],
                        search_context.tune[TUNE_LMR_NOT_IMP],
                    )

                    if tt_pv:
                        r += LMR_TTPV_INCREASE  # TT PV node increase (SF: 1006)
                        r -= LMR_TTPV_DECREASE_BASE  # TT PV decrease base (SF: 2766)
                        if is_pv:
                            r -= LMR_TTPV_PV_BONUS  # PV node bonus (SF: 1017)
                        if tt_entry['flag'] != TT_FLAG_NONE:
                            tt_score = np.int32(tt_entry['score'])
                            if tt_score > MATE_IN_MAX_PLY: tt_score -= ply
                            elif tt_score < -MATE_IN_MAX_PLY: tt_score += ply
                            if tt_score > alpha:
                                r -= LMR_TT_SCORE_GT_ALPHA
                            if tt_entry['depth'] >= depth:
                                r -= (LMR_TT_DEPTH_GE + (LMR_TT_DEPTH_GE_CUT if cut_node else 0))

                    r += search_context.tune[TUNE_LMR_BASE_OFFSET]  # Base offset (SF: 714)
                    # Linear damping to log-scale LMR growth (SF: -mc*62); runtime-tunable
                    r -= searched_legal_moves * search_context.tune[TUNE_LMR_MC_FACTOR]

                    # Scale centipawn difference back to raw correction scale before dividing by LMR_CORRECTION_DIVISOR
                    correction_mag = abs(static_score - raw_static_eval)
                    r -= (correction_mag * CORRECTION_HISTORY_DIVISOR) // LMR_CORRECTION_DIVISOR

                    if cut_node:
                        r += search_context.tune[TUNE_LMR_CUTNODE]  # Cut node bonus (SF ~3995; default 2048)
                        if tt_move == NO_MOVE:
                            r += search_context.tune[TUNE_LMR_NO_TTMOVE]  # No TT move (SF: 1059)

                    if tt_capture:
                        r += search_context.tune[TUNE_LMR_TTCAP]  # TT capture bonus (SF: 1039)

                    if is_capture:
                        if not is_bad_capture:
                            r -= search_context.tune[TUNE_GOOD_CAP_RELIEF]
                        else:
                            r += search_context.tune[TUNE_BAD_CAP_BONUS]
                        if (LMR_LATE_CAPTURE_BONUS > 0 and depth < LMR_LATE_CAPTURE_MAX_DEPTH
                                and searched_legal_moves > LMR_LATE_CAPTURE_MIN_MC):
                            r += LMR_LATE_CAPTURE_BONUS

                    is_pawn = (moved_piece_type % 6) == PAWN
                    if is_pawn:
                        to_rank = to_sq // 8
                        if ((original_side == WHITE and to_rank >= LMR_ADVANCED_PAWN_WHITE_RANK)
                                or (original_side == BLACK and to_rank <= LMR_ADVANCED_PAWN_BLACK_RANK)):
                            r -= LMR_ADVANCED_PAWN_RELIEF

                    if move == killer_1 or move == killer_2 or move == counter_move:
                        r -= search_context.tune[TUNE_KILLER_RELIEF]

                    child_cutoff_cnt = search_context.cutoff_cnt[ply + 1]
                    if child_cutoff_cnt > 1:
                        r += LMR_CUTOFF_CNT_BASE  # Cutoff count base (SF: 236)
                        if child_cutoff_cnt > 2:
                            r += LMR_CUTOFF_CNT_EXTRA  # Cutoff count extra (SF: 1079)
                        all_node = not is_pv and not cut_node
                        if all_node:
                            r += LMR_ALLNODE_EXTRA  # All-node extra (SF: 1143)
                    elif move == tt_move:
                        r = max(
                            LMR_R_FLOOR,
                            r - search_context.tune[TUNE_LMR_TTMOVE_RED]
                            + LMR_TTMOVE_CUTNODE_EXTRA * (1 if cut_node else 0),
                        )

                    # History adjustment
                    if is_quiet_move:
                        lmr_stat_score = get_lmr_stat_score(search_context, ply, from_sq, to_sq, moved_piece_type)
                    else:
                        victim_type_lmr = unmake_info[1]
                        if victim_type_lmr >= 0:
                            victim_idx = victim_type_lmr % 6
                            cap_hist = search_context.capture_history[moved_piece_type, to_sq, victim_idx]
                        else:
                            cap_hist = 0
                            victim_idx = 0
                        # SF18: 809 * PieceValue[captured] / 128 + captHist
                        victim_piece_val = int(MG_MATERIAL_VALUES[victim_idx])
                        lmr_stat_score = (
                            LMR_CAPTURE_VICTIM_SCALE * victim_piece_val // LMR_CAPTURE_VICTIM_DIV + cap_hist
                        )

                    r -= trunc_div(
                        lmr_stat_score * search_context.tune[TUNE_LMR_HIST_SCALE], LMR_STAT_SCORE_DIV
                    )

                    # Scale up reductions for expected ALL nodes
                    all_node = not is_pv and not cut_node
                    if all_node:
                        r += trunc_div(r * LMR_ALLNODE_SCALE_NUM, LMR_ALLNODE_SCALE_DENOM_BASE * depth + LMR_ALLNODE_SCALE_DENOM_OFFSET)

                    # Convert 1024-scale to integer ply
                    d = max(1, min(search_depth - trunc_div(r, 1024), search_depth + LMR_D_MAX_EXTRA))
                    lmr = search_depth - d

                    # Clamping & Safeguards
                    if is_giving_check_after_move:
                        lmr = max(0, lmr - 1)

                    lmr = max(0, min(lmr, search_depth - 1))

            search_context.reduction_stack[ply + 1] = lmr
            if lmr > 0:
                _diag_add(search_context, DIAG_LMR_TRY)
                if search_context.diag_enabled:
                    if lmr == 1:
                        _diag_add(search_context, DIAG_LMR_R1_TRY)
                    elif lmr == 2:
                        _diag_add(search_context, DIAG_LMR_R2_TRY)
                    else:
                        _diag_add(search_context, DIAG_LMR_R3P_TRY)
                    if is_quiet_move:
                        _diag_add(search_context, DIAG_LMR_QUIET_TRY)
                    else:
                        _diag_add(search_context, DIAG_LMR_CAPTURE_TRY)
            res = _search(
                piece_bbs, occupancy_bbs, game_state, _i32(search_depth - lmr),
                _i32(-alpha - 1), _i32(-alpha), search_context, _i32(ply + 1),
                _u16(NO_MOVE), _b_false(), _b_true(),
            )
            evaluation = -res[0]
            child_nodes = res[2]; child_q_nodes = res[3]; child_tt_hits = res[4]

            if lmr > 0 and evaluation > alpha and search_context.diag_enabled:
                if lmr == 1:
                    _diag_add(search_context, DIAG_LMR_R1_FAIL_HIGH)
                elif lmr == 2:
                    _diag_add(search_context, DIAG_LMR_R2_FAIL_HIGH)
                else:
                    _diag_add(search_context, DIAG_LMR_R3P_FAIL_HIGH)
                if is_quiet_move:
                    _diag_add(search_context, DIAG_LMR_QUIET_FAIL_HIGH)
                else:
                    _diag_add(search_context, DIAG_LMR_CAPTURE_FAIL_HIGH)

            if evaluation > alpha:
                do_full_pv_search = is_pv and evaluation < beta
                adjusted_depth = search_depth
                
                if lmr > 0:
                    # LMR failed high on zero window. Verify with adjusted depth and zero window.
                    lmr_depth = search_depth - lmr
                    do_deeper = (lmr_depth < search_depth) and (evaluation > max_eval + LMR_RESEARCH_DEEPER_MARGIN)
                    do_shallower = (evaluation < max_eval + LMR_RESEARCH_SHALLOWER_MARGIN)
                    if search_context.diag_enabled:
                        if do_deeper:
                            _diag_add(search_context, DIAG_LMR_RESEARCH_DEEPER)
                        if do_shallower:
                            _diag_add(search_context, DIAG_LMR_RESEARCH_SHALLOWER)
                    adjusted_depth = search_depth + (1 if do_deeper else 0) - (1 if do_shallower else 0)
                    adjusted_depth = max(1, min(adjusted_depth, search_depth + 1))
                    
                    if adjusted_depth > lmr_depth:
                        # 只有在調整後深度大於先前已搜尋的 LMR 深度時，才進行重新搜尋
                        _diag_add(search_context, DIAG_LMR_RESEARCH)
                        search_context.reduction_stack[ply + 1] = 0
                        res = _search(
                            piece_bbs, occupancy_bbs, game_state, _i32(adjusted_depth),
                            _i32(-alpha - 1), _i32(-alpha), search_context, _i32(ply + 1),
                            _u16(NO_MOVE), _b_false(), not cut_node,
                        )
                        evaluation = -res[0]
                        child_nodes += res[2]; child_q_nodes += res[3]; child_tt_hits += res[4]
                        if evaluation > alpha:
                            if search_context.diag_enabled:
                                _diag_add(search_context, DIAG_LMR_RESEARCH_KEEP)
                                if is_quiet_move:
                                    _diag_add(search_context, DIAG_POST_LMR_BONUS_SAMPLE)
                                    if search_context.butterfly_history[from_sq, to_sq] >= 0:
                                        _diag_add(
                                            search_context,
                                            DIAG_POST_LMR_BONUS_BUTTERFLY_NONNEG,
                                        )
                                    if lmr_stat_score >= 0:
                                        _diag_add(
                                            search_context,
                                            DIAG_POST_LMR_BONUS_STAT_NONNEG,
                                        )
                        else:
                            if search_context.diag_enabled:
                                _diag_add(search_context, DIAG_LMR_RESEARCH_REJECT)
                                if is_quiet_move:
                                    _diag_add(search_context, DIAG_POST_LMR_MALUS_SAMPLE)
                                    if search_context.butterfly_history[from_sq, to_sq] >= 0:
                                        _diag_add(
                                            search_context,
                                            DIAG_POST_LMR_MALUS_BUTTERFLY_NONNEG,
                                        )
                                    if lmr_stat_score >= 0:
                                        _diag_add(
                                            search_context,
                                            DIAG_POST_LMR_MALUS_STAT_NONNEG,
                                        )
                    elif search_context.diag_enabled:
                        _diag_add(search_context, DIAG_LMR_FAIL_HIGH_UNVERIFIED)
                        if is_quiet_move:
                            _diag_add(search_context, DIAG_POST_LMR_UNVERIFIED_SAMPLE)
                    
                    if is_pv and evaluation > alpha and evaluation < beta:
                        do_full_pv_search = True
                    else:
                        do_full_pv_search = False
                
                if do_full_pv_search:
                    # Re-search with full window
                    search_context.reduction_stack[ply + 1] = 0
                    res = _search(
                        piece_bbs, occupancy_bbs, game_state, _i32(adjusted_depth),
                        _i32(-beta), _i32(-alpha), search_context, _i32(ply + 1),
                        _u16(NO_MOVE), is_pv, _b_false(),
                    )
                    evaluation = -res[0]
                    child_nodes += res[2]; child_q_nodes += res[3]; child_tt_hits += res[4]

        unmake_move(piece_bbs, occupancy_bbs, game_state, move, unmake_info)

        # Count the child before propagating a stop.  Otherwise fixed-node and
        # timed searches under-report the interrupted iteration while the
        # shared all-node timer correctly observed those nodes.
        nodes_searched += child_nodes; quiescence_nodes += child_q_nodes; tt_hits += child_tt_hits

        if search_context.stop_flag[0]:
            return (np.int32(0), NO_MOVE, nodes_searched, quiescence_nodes, tt_hits)

        if evaluation > max_eval:
            max_eval, best_move = evaluation, move
            best_moved_piece = np.int8(moved_piece_type)
            best_is_capture = is_capture
            best_victim_type = np.int8(unmake_info[1] % 6) if (is_capture and unmake_info[1] != -1) else np.int8(-1)
            search_context.pv_table[ply, ply] = move
            
            i = ply + 1
            if ply + 1 < MAX_PLY:
                j = ply + 1
                while j < MAX_PLY and search_context.pv_table[ply + 1, j] != NO_MOVE:
                    search_context.pv_table[ply, i] = search_context.pv_table[ply + 1, j]
                    i += 1; j += 1
            
            if i < MAX_PLY: search_context.pv_table[ply, i] = NO_MOVE

        # Alpha-raise depth reduction (SF18 spirit): after a non-fail-high improvement,
        # remaining siblings at this node use shallower depth. Applies on PV too
        # (null-window non-PV almost never has alpha < score < beta). HCE: skip root
        # (ply > 0) so bestmove ranking stays full-width; band/delta from constants.
        if evaluation > alpha + ALPHA_RAISE_MIN_IMPROVEMENT:
            if evaluation < beta:
                if (ENABLE_ALPHA_RAISE_DEPTH_REDUCTION and ply > 0
                        and depth > ALPHA_RAISE_DEPTH_LO and depth < ALPHA_RAISE_DEPTH_HI
                        and not is_decisive(evaluation)):
                    depth -= ALPHA_RAISE_DEPTH_DELTA
                    if depth < 1:
                        depth = 1
            alpha = evaluation
        elif evaluation > alpha:
            # Raised alpha but below min-improvement threshold: update window only
            alpha = evaluation
        if alpha >= beta:
            search_context.cutoff_cnt[ply] += 1
            _diag_add(search_context, DIAG_CUT_NODES)
            _diag_add(search_context, DIAG_CUT_MOVE_SUM, searched_legal_moves)
            if searched_legal_moves == 1:
                _diag_add(search_context, DIAG_CUT_FIRST)
            if search_context.diag_enabled:
                _diag_add(
                    search_context,
                    DIAG_ORDER_CUT_BASE + diag_picker_source_s,
                )
                _diag_add(
                    search_context,
                    DIAG_ORDER_NODES_BEFORE_CUT_SUM,
                    diag_nodes_before_move_s,
                )
                if is_quiet_move:
                    _diag_add(search_context, DIAG_ORDER_QUIET_CUT)
                    _diag_add(
                        search_context,
                        DIAG_ORDER_QUIET_CUT_RANK_SUM,
                        quiet_moves_tried_count,
                    )
                    if move == killer_1:
                        _diag_add(search_context, DIAG_ORDER_KILLER1_CUT)
                    elif move == killer_2:
                        _diag_add(search_context, DIAG_ORDER_KILLER2_CUT)
                    elif move == counter_move:
                        _diag_add(search_context, DIAG_ORDER_COUNTER_CUT)

            # Update killer moves on beta cutoff for quiet moves (SF18 Step 20)
            if is_quiet_move:
                if move != search_context.killer_moves[ply * 2]:
                    search_context.killer_moves[ply * 2 + 1] = search_context.killer_moves[ply * 2]
                    search_context.killer_moves[ply * 2] = move
            break

    if legal_moves_tried == 0 and pruned_moves == 0:
        # Singular verification excludes the TT move deliberately. If no
        # alternative legal move remains, this is a fail-low, not a real
        # mate/stalemate terminal (Stockfish 11/18 Step 20).
        if is_exclusion_search:
            return (np.int32(original_alpha), NO_MOVE, nodes_searched, quiescence_nodes, tt_hits)
        if is_currently_in_check:
            return (np.int32(-MATE_SCORE + ply), NO_MOVE, nodes_searched, quiescence_nodes, tt_hits)
        else:
            return (np.int32(0), NO_MOVE, nodes_searched, quiescence_nodes, tt_hits)

    if max_eval == -INFINITY:
        max_eval = original_alpha

    # M3: Apply 50-move scale-down to non-mate evaluations
    if fifty_move_scale < FIFTY_MOVE_MAX_SCALE and abs(max_eval) < MATE_IN_MAX_PLY:
        max_eval = max_eval * fifty_move_scale // FIFTY_MOVE_MAX_SCALE

    # A1: SF18 Fail-High Smoothing (主搜尋深度加權平滑)
    # 只有在 fail-high 且非 decisive (非將死分數) 時才進行平滑
    if max_eval >= beta and abs(max_eval) < MATE_IN_MAX_PLY and abs(alpha) < MATE_IN_MAX_PLY:
        max_eval = np.int32(trunc_div(max_eval * depth + beta, depth + 1))

    final_flag = TT_FLAG_ALPHA if max_eval <= original_alpha else (TT_FLAG_BETA if max_eval >= beta else TT_FLAG_EXACT)

    # ==========================================
    # STEP 21: Unified Move Stats Update (Stockfish 17/18 Step 21)
    # ==========================================
    # If a move improved alpha (PV node or beta cutoff), update all history tables
    # for best_move and penalize inferior searched moves with malus.
    if final_flag != TT_FLAG_ALPHA and best_move != NO_MOVE:
        # Phase 1a: linear bonus/malus
        bonus = min(HISTORY_BONUS_SCALE * depth, HISTORY_BONUS_CAP)
        malus = min(HISTORY_BONUS_SCALE * depth, HISTORY_MALUS_CAP)

        # Node Width Scaling
        if not is_pv:
            bonus += bonus * (quiet_moves_tried_count + capture_moves_tried_count) // HISTORY_NODE_WIDTH_DIV

        # --- Refutation Penalty ---
        if ply > 0:
            parent_was_capture = search_context.move_is_capture_stack[ply - 1]
            parent_move = search_context.move_stack[ply - 1]
            parent_aggressor = search_context.piece_stack[ply - 1]

            if parent_move != NO_MOVE and parent_aggressor != -1 and not parent_was_capture:
                parent_move_count = search_context.move_count_stack[ply - 1]
                parent_tt_hit = search_context.tt_hit_stack[ply - 1]

                if parent_move_count == 1 + (1 if parent_tt_hit else 0):
                    refute_malus = malus

                    if ply > 1:
                        prev_move_1 = search_context.move_stack[ply - 2]
                        prev_piece_1 = search_context.piece_stack[ply - 2]
                        if prev_move_1 != NO_MOVE and prev_piece_1 != -1:
                            update_continuation_history(search_context, 0, prev_move_1, prev_piece_1, parent_move, parent_aggressor, -refute_malus)
                    if ply > 2:
                        prev_move_2 = search_context.move_stack[ply - 3]
                        prev_piece_2 = search_context.piece_stack[ply - 3]
                        if prev_move_2 != NO_MOVE and prev_piece_2 != -1:
                            update_continuation_history(search_context, 1, prev_move_2, prev_piece_2, parent_move, parent_aggressor, -refute_malus)
                    if ply > 3:
                        prev_move_3 = search_context.move_stack[ply - 4]
                        prev_piece_3 = search_context.piece_stack[ply - 4]
                        if prev_move_3 != NO_MOVE and prev_piece_3 != -1:
                            update_continuation_history(search_context, 2, prev_move_3, prev_piece_3, parent_move, parent_aggressor, -refute_malus)
                    if ply > 4:
                        prev_move_4 = search_context.move_stack[ply - 5]
                        prev_piece_4 = search_context.piece_stack[ply - 5]
                        if prev_move_4 != NO_MOVE and prev_piece_4 != -1:
                            update_continuation_history(search_context, 3, prev_move_4, prev_piece_4, parent_move, parent_aggressor, -refute_malus)
                    if ply > 6:
                        prev_move_6 = search_context.move_stack[ply - 7]
                        prev_piece_6 = search_context.piece_stack[ply - 7]
                        if prev_move_6 != NO_MOVE and prev_piece_6 != -1:
                            update_continuation_history(search_context, 4, prev_move_6, prev_piece_6, parent_move, parent_aggressor, -refute_malus)

        # --- Best Move Capture History Bonus ---
        if best_is_capture and best_moved_piece != -1:
            if best_victim_type != -1:
                best_to_sq = get_to_square(best_move)
                update_capture_history(search_context.capture_history, best_moved_piece, best_to_sq, best_victim_type, bonus)
                _diag_add(search_context, DIAG_HIST_CAP_BONUS)

        # --- Capture Malus to all previous capture moves that failed low ---
        cap_malus = malus * CAPTURE_MALUS_SCALE_NUM // CAPTURE_MALUS_SCALE_DEN
        for c_idx in range(capture_moves_tried_count):
            bad_cap_move = capture_moves_tried[c_idx]
            if bad_cap_move == best_move:
                continue
            bad_cap_agg = capture_aggressor_tried[c_idx]
            bad_cap_vic = capture_victim_tried[c_idx]
            bad_cap_to = capture_tosq_tried[c_idx]
            update_capture_history(search_context.capture_history, bad_cap_agg, bad_cap_to, bad_cap_vic, -cap_malus)
            _diag_add(search_context, DIAG_HIST_CAP_MALUS)

        # --- Best Move Quiet History Bonus & Malus ---
        if not best_is_capture and best_moved_piece != -1:
            aggressor_type = best_moved_piece
            best_to_sq = get_to_square(best_move)
            best_from_sq = get_from_square(best_move)

            # Apply Gravity Bonus to the best quiet move
            update_history(search_context.history_table, aggressor_type, best_to_sq, bonus)
            update_butterfly_history(search_context.butterfly_history, best_from_sq, best_to_sq, bonus)
            update_pawn_history(search_context.pawn_history, pawn_key_idx, aggressor_type, best_to_sq, bonus)
            _diag_add(search_context, DIAG_HIST_QUIET_BONUS)
            _diag_add(search_context, DIAG_HIST_PIECE_TO_UPD)
            _diag_add(search_context, DIAG_HIST_PAWN_UPD)
            _diag_add(search_context, DIAG_HIST_CONT_UPD)

            # Continuation History Update (Bonus)
            if ply > 0:
                prev_move_played = search_context.move_stack[ply - 1]
                prev_piece_played = search_context.piece_stack[ply - 1]
                if prev_move_played != NO_MOVE and prev_piece_played != -1:
                    update_continuation_history(search_context, 0, prev_move_played, prev_piece_played, best_move, aggressor_type, bonus)

            if ply > 1:
                prev_move_played = search_context.move_stack[ply - 2]
                prev_piece_played = search_context.piece_stack[ply - 2]
                if prev_move_played != NO_MOVE and prev_piece_played != -1:
                    update_continuation_history(search_context, 1, prev_move_played, prev_piece_played, best_move, aggressor_type, bonus)

            if ply > 2:
                prev_move_played = search_context.move_stack[ply - 3]
                prev_piece_played = search_context.piece_stack[ply - 3]
                if prev_move_played != NO_MOVE and prev_piece_played != -1:
                    update_continuation_history(search_context, 2, prev_move_played, prev_piece_played, best_move, aggressor_type, bonus)

            if ply > 3:
                prev_move_played = search_context.move_stack[ply - 4]
                prev_piece_played = search_context.piece_stack[ply - 4]
                if prev_move_played != NO_MOVE and prev_piece_played != -1:
                    update_continuation_history(search_context, 3, prev_move_played, prev_piece_played, best_move, aggressor_type, bonus)

            if ply > 5:
                prev_move_played = search_context.move_stack[ply - 6]
                prev_piece_played = search_context.piece_stack[ply - 6]
                if prev_move_played != NO_MOVE and prev_piece_played != -1:
                    update_continuation_history(search_context, 4, prev_move_played, prev_piece_played, best_move, aggressor_type, bonus)

            if ply < 5:
                lph_bonus = bonus * HISTORY_LPH_SCALE_NUM // HISTORY_LPH_SCALE_DEN
                curr = search_context.low_ply_history[ply, best_move]
                clamped = min(max(lph_bonus, -LOW_PLY_HISTORY_MAX), LOW_PLY_HISTORY_MAX)
                search_context.low_ply_history[ply, best_move] = curr + clamped - (curr * abs(clamped)) // LOW_PLY_HISTORY_MAX
                _diag_add(search_context, DIAG_HIST_LPH_UPD)

            # Update Counter Move (Phase 1a: prev_piece x prev_to)
            if ply > 0:
                prev_move_played = search_context.move_stack[ply - 1]
                prev_piece_cm = search_context.piece_stack[ply - 1]
                if prev_move_played != NO_MOVE and prev_piece_cm != -1:
                    search_context.counter_moves[prev_piece_cm, get_to_square(prev_move_played)] = best_move

            # Apply History Malus to all previous quiet moves that failed low
            actual_malus = malus * HISTORY_QUIET_MALUS_SCALE_NUM // HISTORY_QUIET_MALUS_SCALE_DEN
            for q_idx in range(quiet_moves_tried_count):
                bad_move = quiet_moves_tried[q_idx]
                if bad_move == best_move:
                    continue
                actual_malus = actual_malus * HISTORY_QUIET_MALUS_DECAY_NUM // HISTORY_QUIET_MALUS_DECAY_DEN
                bad_from = get_from_square(bad_move)
                bad_to = get_to_square(bad_move)
                bad_aggressor = quiet_pieces_tried[q_idx]
                if bad_aggressor == -1:
                    continue
                update_history(search_context.history_table, bad_aggressor, bad_to, -actual_malus)
                update_butterfly_history(search_context.butterfly_history, bad_from, bad_to, -actual_malus)
                update_pawn_history(search_context.pawn_history, pawn_key_idx, bad_aggressor, bad_to, -actual_malus)
                _diag_add(search_context, DIAG_HIST_QUIET_MALUS)
                _diag_add(search_context, DIAG_HIST_PIECE_TO_UPD)
                _diag_add(search_context, DIAG_HIST_PAWN_UPD)
                _diag_add(search_context, DIAG_HIST_CONT_UPD)

                # Continuation History Update (Malus)
                if ply > 0:
                    prev_move_played = search_context.move_stack[ply - 1]
                    prev_piece_played = search_context.piece_stack[ply - 1]
                    if prev_move_played != NO_MOVE and prev_piece_played != -1:
                        update_continuation_history(search_context, 0, prev_move_played, prev_piece_played, bad_move, bad_aggressor, -actual_malus)

                if ply > 1:
                    prev_move_played = search_context.move_stack[ply - 2]
                    prev_piece_played = search_context.piece_stack[ply - 2]
                    if prev_move_played != NO_MOVE and prev_piece_played != -1:
                        update_continuation_history(search_context, 1, prev_move_played, prev_piece_played, bad_move, bad_aggressor, -actual_malus)

                if ply > 2:
                    prev_move_played = search_context.move_stack[ply - 3]
                    prev_piece_played = search_context.piece_stack[ply - 3]
                    if prev_move_played != NO_MOVE and prev_piece_played != -1:
                        update_continuation_history(search_context, 2, prev_move_played, prev_piece_played, bad_move, bad_aggressor, -actual_malus)

                if ply > 3:
                    prev_move_played = search_context.move_stack[ply - 4]
                    prev_piece_played = search_context.piece_stack[ply - 4]
                    if prev_move_played != NO_MOVE and prev_piece_played != -1:
                        update_continuation_history(search_context, 3, prev_move_played, prev_piece_played, bad_move, bad_aggressor, -actual_malus)

                if ply > 5:
                    prev_move_played = search_context.move_stack[ply - 6]
                    prev_piece_played = search_context.piece_stack[ply - 6]
                    if prev_move_played != NO_MOVE and prev_piece_played != -1:
                        update_continuation_history(search_context, 4, prev_move_played, prev_piece_played, bad_move, bad_aggressor, -actual_malus)

                if ply < 5:
                    lph_malus = actual_malus * HISTORY_LPH_SCALE_NUM // HISTORY_LPH_SCALE_DEN
                    curr = search_context.low_ply_history[ply, bad_move]
                    clamped = min(max(-lph_malus, -LOW_PLY_HISTORY_MAX), LOW_PLY_HISTORY_MAX)
                    search_context.low_ply_history[ply, bad_move] = curr + clamped - (curr * abs(clamped)) // LOW_PLY_HISTORY_MAX

    # --- Opponent Move Bonus (Fail-Low / Alpha Flag Bonus) ---
    elif final_flag == TT_FLAG_ALPHA and ply > 0 and legal_moves_tried > FAIL_LOW_MIN_LEGAL_MOVES:
        prev_move = search_context.move_stack[ply - 1]
        prev_piece = search_context.piece_stack[ply - 1]
        
        # Reward the opponent's previous move if it successfully caused us to fail low
        if prev_move != NO_MOVE and prev_piece != -1:
            if (prev_piece % 6) != 0:
                prev_to = get_to_square(prev_move)
                prev_from = get_from_square(prev_move)
                pawn_key_idx = game_state[PAWN_KEY_INDEX] & PAWN_HISTORY_MASK
                
                base_bonus = min(HISTORY_BONUS_SCALE * depth, HISTORY_BONUS_CAP)
                sub_bonus = base_bonus // 4
                
                update_pawn_history(search_context.pawn_history, pawn_key_idx, prev_piece, prev_to, sub_bonus)
                update_history(search_context.history_table, prev_piece, prev_to, sub_bonus)
                update_butterfly_history(search_context.butterfly_history, prev_from, prev_to, sub_bonus)
                _diag_add(search_context, DIAG_HIST_FAIL_LOW)
                _diag_add(search_context, DIAG_HIST_PIECE_TO_UPD)
                _diag_add(search_context, DIAG_HIST_PAWN_UPD)
                _diag_add(search_context, DIAG_HIST_CONT_UPD)
                
                if ply > 1:
                    our_prev_move = search_context.move_stack[ply - 2]
                    our_prev_piece = search_context.piece_stack[ply - 2]
                    if our_prev_move != NO_MOVE and our_prev_piece != -1:
                        update_continuation_history(search_context, 0, our_prev_move, our_prev_piece, prev_move, prev_piece, sub_bonus)

                if ply > 2:
                    opp_prev_move = search_context.move_stack[ply - 3]
                    opp_prev_piece = search_context.piece_stack[ply - 3]
                    if opp_prev_move != NO_MOVE and opp_prev_piece != -1:
                        update_continuation_history(search_context, 1, opp_prev_move, opp_prev_piece, prev_move, prev_piece, sub_bonus)

                if ply > 3:
                    our_prev_move_2 = search_context.move_stack[ply - 4]
                    our_prev_piece_2 = search_context.piece_stack[ply - 4]
                    if our_prev_move_2 != NO_MOVE and our_prev_piece_2 != -1:
                        update_continuation_history(search_context, 2, our_prev_move_2, our_prev_piece_2, prev_move, prev_piece, sub_bonus)

                if ply > 4:
                    opp_prev_move_2 = search_context.move_stack[ply - 5]
                    opp_prev_piece_2 = search_context.piece_stack[ply - 5]
                    if opp_prev_move_2 != NO_MOVE and opp_prev_piece_2 != -1:
                        update_continuation_history(search_context, 3, opp_prev_move_2, opp_prev_piece_2, prev_move, prev_piece, sub_bonus)
 
                if ply > 6:
                    opp_prev_move_3 = search_context.move_stack[ply - 7]
                    opp_prev_piece_3 = search_context.piece_stack[ply - 7]
                    if opp_prev_move_3 != NO_MOVE and opp_prev_piece_3 != -1:
                        update_continuation_history(search_context, 4, opp_prev_move_3, opp_prev_piece_3, prev_move, prev_piece, sub_bonus)

    original_best_move = best_move

    if best_move == NO_MOVE and search_context.mp_captures_end[ply] + search_context.mp_quiets_end[ply] > 0:
        # Fallback to first move that is not excluded
        for i in range(search_context.mp_captures_end[ply] + search_context.mp_quiets_end[ply]):
            if moves[i] != excluded_move:
                best_move = moves[i]
                break
    
    # --- Update Correction History ---
    if depth >= CORRECTION_HISTORY_UPDATE_DEPTH and not is_currently_in_check and abs(max_eval) < MATE_SCORE - MAX_PLY:
        # Only update if we have a valid static eval (raw_static_eval != -INFINITY)
        if raw_static_eval != -INFINITY:
            pawn_key = game_state[PAWN_KEY_INDEX]
            minor_key = game_state[MINOR_KEY_INDEX]
            np_white_key = game_state[NON_PAWN_KEY_WHITE_INDEX]
            np_black_key = game_state[NON_PAWN_KEY_BLACK_INDEX]
            side = game_state[0]
            
            is_capture = False
            if original_best_move != NO_MOVE:
                to_sq = get_to_square(original_best_move)
                move_flag = get_special_move_flag(original_best_move)
                enemy_side = 1 - side
                is_capture = ((occupancy_bbs[enemy_side] & BB_SQUARES[to_sq]) != 0) or (move_flag == SPECIAL_MOVE_FLAG_EN_PASSANT)
            
            has_best_move = (original_best_move != NO_MOVE)
            if not is_capture and (max_eval > static_score) == has_best_move:
                # Based on static_score calculation of bonus
                diff = max_eval - static_score
                bonus = trunc_div(
                    diff * depth,
                    (CORRECTION_BONUS_HAS_MOVE_DIV if has_best_move else CORRECTION_BONUS_NO_MOVE_DIV),
                )
                
                # Limit single bonus to 1/4 of the total limit
                bonus = min(max(bonus, -CORRECTION_HISTORY_LIMIT // 4), CORRECTION_HISTORY_LIMIT // 4)

                # Apply outer scaling (SF18: 1114 * bonus / 1024; we use OUTER_SCALE)
                bonus = trunc_div(bonus * CORRECTION_HISTORY_OUTER_SCALE, 1024)

                # Absolute White perspective for Pawn, Minor, and Non-Pawn
                white_bonus = bonus if side == WHITE else -bonus

                # Update Pawn Correction (white-absolute)
                idx_pawn = pawn_key & CORRECTION_HISTORY_MASK
                curr_pawn = search_context.pawn_correction_history[idx_pawn]
                search_context.pawn_correction_history[idx_pawn] = curr_pawn + white_bonus - trunc_div(curr_pawn * abs(white_bonus), CORRECTION_HISTORY_LIMIT)

                # Update Minor Correction (white-absolute)
                idx_minor = minor_key & CORRECTION_HISTORY_MASK
                curr_minor = search_context.minor_correction_history[idx_minor]
                scaled_minor_bonus = trunc_div(
                    white_bonus * CORRECTION_MINOR_SCALE_NUM, CORRECTION_MINOR_SCALE_DEN
                )
                search_context.minor_correction_history[idx_minor] = curr_minor + scaled_minor_bonus - trunc_div(curr_minor * abs(scaled_minor_bonus), CORRECTION_HISTORY_LIMIT)

                # Update Non-Pawn Correction (side-exclusive, relative to Side to Move)
                scaled_np_bonus = trunc_div(
                    bonus * CORRECTION_NP_SCALE_NUM, CORRECTION_NP_SCALE_DEN
                )
                if side == WHITE:
                    idx_np = np_white_key & CORRECTION_HISTORY_MASK
                    curr_np = search_context.non_pawn_correction_history_white[idx_np]
                    search_context.non_pawn_correction_history_white[idx_np] = curr_np + scaled_np_bonus - trunc_div(curr_np * abs(scaled_np_bonus), CORRECTION_HISTORY_LIMIT)
                else:
                    idx_np = np_black_key & CORRECTION_HISTORY_MASK
                    curr_np = search_context.non_pawn_correction_history_black[idx_np]
                    search_context.non_pawn_correction_history_black[idx_np] = curr_np + scaled_np_bonus - trunc_div(curr_np * abs(scaled_np_bonus), CORRECTION_HISTORY_LIMIT)

                # Update Continuation Correction History
                if ply > 0:
                    m_prev = search_context.move_stack[ply - 1]
                    pc_prev = search_context.piece_stack[ply - 1]
                    if m_prev != NO_MOVE and pc_prev != -1:
                        to_prev = get_to_square(m_prev)
                        
                        # Update 2-ply offset
                        if ply >= 2:
                            m_2 = search_context.move_stack[ply - 2]
                            pc_2 = search_context.piece_stack[ply - 2]
                            if m_2 != NO_MOVE and pc_2 != -1:
                                bonus_2 = trunc_div(
                                    bonus * CORRECTION_CONT_2PLY_NUM, CORRECTION_CONT_SCALE_DEN
                                )
                                curr_val = search_context.continuation_correction_history[pc_2, get_to_square(m_2), pc_prev, to_prev]
                                search_context.continuation_correction_history[pc_2, get_to_square(m_2), pc_prev, to_prev] = \
                                    curr_val + bonus_2 - trunc_div(curr_val * abs(bonus_2), CORRECTION_HISTORY_LIMIT)
                                    
                        # Update 4-ply offset
                        if ply >= 4:
                            m_4 = search_context.move_stack[ply - 4]
                            pc_4 = search_context.piece_stack[ply - 4]
                            if m_4 != NO_MOVE and pc_4 != -1:
                                bonus_4 = trunc_div(
                                    bonus * CORRECTION_CONT_4PLY_NUM, CORRECTION_CONT_SCALE_DEN
                                )
                                curr_val = search_context.continuation_correction_history[pc_4, get_to_square(m_4), pc_prev, to_prev]
                                search_context.continuation_correction_history[pc_4, get_to_square(m_4), pc_prev, to_prev] = \
                                    curr_val + bonus_4 - trunc_div(curr_val * abs(bonus_4), CORRECTION_HISTORY_LIMIT)

                # Update Continuation Correction History (duplicate block kept for parity)
                if ply > 0:
                    m_prev = search_context.move_stack[ply - 1]
                    pc_prev = search_context.piece_stack[ply - 1]
                    if m_prev != NO_MOVE and pc_prev != -1:
                        to_prev = get_to_square(m_prev)
                        
                        if ply >= 2:
                            m_2 = search_context.move_stack[ply - 2]
                            pc_2 = search_context.piece_stack[ply - 2]
                            if m_2 != NO_MOVE and pc_2 != -1:
                                bonus_2 = trunc_div(
                                    bonus * CORRECTION_CONT_2PLY_NUM, CORRECTION_CONT_SCALE_DEN
                                )
                                curr_val = search_context.continuation_correction_history[pc_2, get_to_square(m_2), pc_prev, to_prev]
                                search_context.continuation_correction_history[pc_2, get_to_square(m_2), pc_prev, to_prev] = \
                                    curr_val + bonus_2 - trunc_div(curr_val * abs(bonus_2), CORRECTION_HISTORY_LIMIT)
                                    
                        if ply >= 4:
                            m_4 = search_context.move_stack[ply - 4]
                            pc_4 = search_context.piece_stack[ply - 4]
                            if m_4 != NO_MOVE and pc_4 != -1:
                                bonus_4 = trunc_div(
                                    bonus * CORRECTION_CONT_4PLY_NUM, CORRECTION_CONT_SCALE_DEN
                                )
                                curr_val = search_context.continuation_correction_history[pc_4, get_to_square(m_4), pc_prev, to_prev]
                                search_context.continuation_correction_history[pc_4, get_to_square(m_4), pc_prev, to_prev] = \
                                    curr_val + bonus_4 - trunc_div(curr_val * abs(bonus_4), CORRECTION_HISTORY_LIMIT)

    tt_score = max_eval
    if tt_score > MATE_IN_MAX_PLY: tt_score += ply
    elif tt_score < -MATE_IN_MAX_PLY: tt_score -= ply
    # Determine static eval to store logic
    tt_static_eval_to_store = np.int16(32767)
    if raw_static_eval != -INFINITY and abs(raw_static_eval) < MATE_SCORE - MAX_PLY:
        tt_static_eval_to_store = np.int16(raw_static_eval)

    # Propagate ttPv on fail-low (Alpha Flag)
    if final_flag == TT_FLAG_ALPHA and ply > 0:
        tt_pv = tt_pv or search_context.tt_pv_stack[ply - 1]
        search_context.tt_pv_stack[ply] = tt_pv

    if not is_exclusion_search:
        store_tt(search_context.transposition_table, tt_key, depth, tt_score, tt_static_eval_to_store, final_flag, best_move, search_context.tt_generation, tt_pv)

    return (np.int32(max_eval), best_move, nodes_searched, quiescence_nodes, tt_hits)

def reset_search_diagnostics(search_context):
    """Zero tree-efficiency counters (safe from pure Python)."""
    search_context.diag_stats[:] = 0


def _table_energy_stats(arr, ceiling):
    """Return mean|x|, p50|x|, % near ±ceiling for a history ndarray."""
    flat = np.asarray(arr, dtype=np.int64).ravel()
    if flat.size == 0:
        return {"mean_abs": 0.0, "p50_abs": 0.0, "sat_pct": 0.0, "n": 0}
    absv = np.abs(flat)
    thr = max(1, int(0.90 * abs(ceiling)))
    sat = float(np.mean(absv >= thr) * 100.0)
    return {
        "mean_abs": float(np.mean(absv)),
        "p50_abs": float(np.median(absv)),
        "sat_pct": sat,
        "n": int(flat.size),
    }


def format_history_table_stats(search_context) -> str:
    """Energy / saturation snapshot of history tables (Python-side)."""
    rows = [
        ("piece_to/history", search_context.history_table, HISTORY_MAX_MAIN),
        ("butterfly", search_context.butterfly_history, HISTORY_MAX_BUTTERFLY),
        ("capture", search_context.capture_history, HISTORY_MAX_CAPTURE),
        ("continuation", search_context.continuation_history, HISTORY_MAX_CONTINUATION),
        ("pawn", search_context.pawn_history, HISTORY_MAX_PAWN),
        ("low_ply", search_context.low_ply_history, LOW_PLY_HISTORY_MAX),
    ]
    lines = ["history_tables (mean|x|, p50|x|, sat%>=0.9*ceil):"]
    for name, arr, ceil in rows:
        st = _table_energy_stats(arr, ceil)
        lines.append(
            f"  {name}: mean_abs={st['mean_abs']:.1f} p50_abs={st['p50_abs']:.1f} "
            f"sat%={st['sat_pct']:.2f} (ceil={ceil})"
        )
    return "\n".join(lines)


def _collect_aspiration_diag(search_context, total_nodes):
    """Collect P2 aspiration counters, including a compact per-depth view.

    Aspiration counters are written by the Python iterative-deepening loop, so
    this reporting helper intentionally stays outside the Numba hot path.
    ``wasted_nodes`` means nodes spent by windows that failed low/high and had
    to be searched again; those searches may still warm the TT, so the label
    describes re-search overhead rather than provably useless work.
    """
    d = search_context.diag_stats
    iterations = 0
    fail_low = 0
    fail_high = 0
    research = 0
    wasted_nodes = 0
    max_delta = 0
    by_depth = []
    for depth in range(2, DIAG_ASPIRATION_DEPTH_SLOTS):
        i = int(d[DIAG_ASPIRATION_ITERATIONS_BASE + depth])
        low = int(d[DIAG_ASPIRATION_FAIL_LOW_BASE + depth])
        high = int(d[DIAG_ASPIRATION_FAIL_HIGH_BASE + depth])
        re = int(d[DIAG_ASPIRATION_RESEARCH_BASE + depth])
        delta = int(d[DIAG_ASPIRATION_MAX_DELTA_BASE + depth])
        waste = int(d[DIAG_ASPIRATION_WASTED_NODES_BASE + depth])
        if i or low or high or re or delta or waste:
            by_depth.append({
                "depth": depth,
                "iterations": i,
                "fail_low": low,
                "fail_high": high,
                "research": re,
                "max_delta": delta,
                "wasted_nodes": waste,
                "research_per_iteration": (re / i) if i else 0.0,
            })
        iterations += i
        fail_low += low
        fail_high += high
        research += re
        wasted_nodes += waste
        if delta > max_delta:
            max_delta = delta

    n = max(1, int(total_nodes))
    return {
        "aspiration_iterations": iterations,
        "aspiration_fail_low": fail_low,
        "aspiration_fail_high": fail_high,
        "aspiration_research": research,
        "aspiration_max_delta": max_delta,
        "aspiration_wasted_nodes": wasted_nodes,
        "aspiration_fail_low_per_iteration": (
            fail_low / iterations if iterations else 0.0
        ),
        "aspiration_fail_high_per_iteration": (
            fail_high / iterations if iterations else 0.0
        ),
        "aspiration_research_per_iteration": (
            research / iterations if iterations else 0.0
        ),
        "aspiration_research_search_pct": (
            100.0 * research / (iterations + research)
            if iterations + research else 0.0
        ),
        "aspiration_wasted_nodes_pct": 100.0 * wasted_nodes / n,
        "aspiration_by_depth": by_depth,
    }


_HIST_PRUNE_LAYER_NAMES = (
    "main", "butterfly", "pawn", "cont1", "cont2", "cont3", "cont4", "cont6",
)


def _collect_history_prune_attribution(diag_stats, prune_count: int) -> dict:
    """Decode P5 attribution counters into stable JSON-friendly fields."""
    prune_n = max(1, int(prune_count))
    out = {}
    for i, name in enumerate(_HIST_PRUNE_LAYER_NAMES):
        neg = int(diag_stats[DIAG_HP_NEG_COUNT_BASE + i])
        neg_abs = int(diag_stats[DIAG_HP_NEG_ABS_SUM_BASE + i])
        decisive = int(diag_stats[DIAG_HP_DECISIVE_BASE + i])
        dominant = int(diag_stats[DIAG_HP_DOMINANT_BASE + i])
        out[f"hist_prune_{name}_negative"] = neg
        out[f"hist_prune_{name}_negative_pct"] = 100.0 * neg / prune_n
        out[f"hist_prune_{name}_negative_abs_sum"] = neg_abs
        out[f"hist_prune_{name}_negative_mean_abs"] = neg_abs / neg if neg else 0.0
        out[f"hist_prune_{name}_decisive"] = decisive
        out[f"hist_prune_{name}_decisive_pct"] = 100.0 * decisive / prune_n
        out[f"hist_prune_{name}_dominant"] = dominant
        out[f"hist_prune_{name}_dominant_pct"] = 100.0 * dominant / prune_n

    cont_decisive = int(diag_stats[DIAG_HP_CONT_COALITION_DECISIVE])
    noncont_prunes = int(diag_stats[DIAG_HP_NONCONT_ALREADY_PRUNES])
    lmr_eligible = int(diag_stats[DIAG_HP_LMR_ELIGIBLE])
    main_near_zero = int(diag_stats[DIAG_HP_MAIN_GATE_NEAR_ZERO])
    near = int(diag_stats[DIAG_HP_MARGIN_NEAR])
    mid = int(diag_stats[DIAG_HP_MARGIN_MID])
    far = int(diag_stats[DIAG_HP_MARGIN_FAR])
    cap_hit = int(diag_stats[DIAG_HP_CONT1_CAP_HIT])
    cap_relief = int(diag_stats[DIAG_HP_CONT1_CAP_RELIEF_SUM])
    cap_rescue = int(diag_stats[DIAG_HP_CONT1_CAP_RESCUE])
    out.update({
        "hist_prune_cont_coalition_decisive": cont_decisive,
        "hist_prune_cont_coalition_decisive_pct": 100.0 * cont_decisive / prune_n,
        "hist_prune_noncont_already_prunes": noncont_prunes,
        "hist_prune_noncont_already_prunes_pct": 100.0 * noncont_prunes / prune_n,
        "hist_prune_lmr_eligible": lmr_eligible,
        "hist_prune_lmr_eligible_pct": 100.0 * lmr_eligible / prune_n,
        "hist_prune_main_gate_near_zero": main_near_zero,
        "hist_prune_main_gate_near_zero_pct": 100.0 * main_near_zero / prune_n,
        "hist_prune_margin_near": near,
        "hist_prune_margin_mid": mid,
        "hist_prune_margin_far": far,
        "hist_prune_margin_sum": int(diag_stats[DIAG_HP_MARGIN_SUM]),
        "hist_prune_depth_sum": int(diag_stats[DIAG_HP_DEPTH_SUM]),
        "hist_prune_move_index_sum": int(diag_stats[DIAG_HP_MOVE_INDEX_SUM]),
        "hist_prune_cont1_cap_hit": cap_hit,
        "hist_prune_cont1_cap_relief_sum": cap_relief,
        "hist_prune_cont1_cap_relief_mean": cap_relief / cap_hit if cap_hit else 0.0,
        "hist_prune_cont1_cap_rescue": cap_rescue,
        "hist_prune_cont1_cap_rescue_pct": 100.0 * cap_rescue / cap_hit if cap_hit else 0.0,
        "hist_prune_margin_mean": (
            int(diag_stats[DIAG_HP_MARGIN_SUM]) / prune_count if prune_count else 0.0
        ),
        "hist_prune_depth_mean": (
            int(diag_stats[DIAG_HP_DEPTH_SUM]) / prune_count if prune_count else 0.0
        ),
        "hist_prune_move_index_mean": (
            int(diag_stats[DIAG_HP_MOVE_INDEX_SUM]) / prune_count if prune_count else 0.0
        ),
    })
    return out


def collect_hist_diag_dict(search_context, total_nodes=0) -> dict:
    """Structured hist + mid/shallow prune counters for bench / experiment logs."""
    d = search_context.diag_stats
    n = max(1, int(total_nodes))
    q_bonus = int(d[DIAG_HIST_QUIET_BONUS])
    q_malus = int(d[DIAG_HIST_QUIET_MALUS])
    prune = int(d[DIAG_HIST_PRUNE])
    lmp_skip = int(d[DIAG_LMP_SKIP])
    fp_skip = int(d[DIAG_FP_SKIP])
    rfp_cut = int(d[DIAG_RFP_CUT])
    razor_cut = int(d[DIAG_RAZOR_CUT])
    nmp_cut = int(d[DIAG_NMP_CUT])
    nmp_el = int(d[DIAG_NMP_ELIGIBLE])
    nmp_gp = int(d[DIAG_NMP_GATE_PASS])
    nmp_try = int(d[DIAG_NMP_NULL_TRY])
    nmp_fh = int(d[DIAG_NMP_NULL_FH])
    nmp_vt = int(d[DIAG_NMP_VERIFY_TRY])
    nmp_vf = int(d[DIAG_NMP_VERIFY_FAIL])
    tt_cut = int(d[DIAG_TT_CUT])
    probcut = int(d[DIAG_PROBCUT])
    lmr_try = int(d[DIAG_LMR_TRY])
    lmr_re = int(d[DIAG_LMR_RESEARCH])
    cut_nodes = int(d[DIAG_CUT_NODES])
    cut_first = int(d[DIAG_CUT_FIRST])
    quiet_precheck = int(d[DIAG_MAIN_QUIET_SEE_PRECHECK])
    quiet_pre_pin = int(d[DIAG_MAIN_QUIET_PRE_PIN_ILLEGAL])
    quiet_pre_king = int(d[DIAG_MAIN_QUIET_PRE_KING_ILLEGAL])
    quiet_see_pass = int(d[DIAG_MAIN_QUIET_SEE_PASS])
    quiet_post_pin = int(d[DIAG_MAIN_QUIET_POST_PIN_ILLEGAL])
    quiet_post_king = int(d[DIAG_MAIN_QUIET_POST_KING_ILLEGAL])
    quiet_fallback_try = int(d[DIAG_MAIN_QUIET_FALLBACK_TRY])
    quiet_fallback_reject = int(d[DIAG_MAIN_QUIET_FALLBACK_REJECT])
    geometry_nodes = int(d[DIAG_MAIN_GEOMETRY_NODE])
    score_geometry_rebuild = int(d[DIAG_MAIN_SCORE_GEOMETRY_REBUILD])
    q_cap_noncheck = int(d[DIAG_Q_CAP_NONCHECK])
    q_cap_check = int(d[DIAG_Q_CAP_CHECK])
    q_cap_forcing = int(d[DIAG_Q_CAP_FORCING])
    q_cap_forcing_moves = int(d[DIAG_Q_CAP_FORCING_MOVES])
    main_in_check_nodes = int(d[DIAG_MAIN_IN_CHECK_NODE])
    main_in_check_pick_try = int(d[DIAG_MAIN_IN_CHECK_PICK_TRY])
    main_in_check_pre_illegal = int(d[DIAG_MAIN_IN_CHECK_PRE_ILLEGAL])
    main_in_check_pre_pass = int(d[DIAG_MAIN_IN_CHECK_PRE_PASS])
    main_in_check_fallback = int(d[DIAG_MAIN_IN_CHECK_FALLBACK])
    tt_verify_try = int(d[DIAG_TT_VERIFY_TRY])
    tt_verify_pass = int(d[DIAG_TT_VERIFY_PASS])
    tt_verify_reject = int(d[DIAG_TT_VERIFY_REJECT])
    tt_verify_saved_cut = int(d[DIAG_TT_VERIFY_SAVED_CUT])
    tt_verify_skip = int(d[DIAG_TT_VERIFY_SKIP])
    order_tries = [
        int(d[DIAG_ORDER_TRY_BASE + i])
        for i in range(DIAG_ORDER_SOURCE_COUNT)
    ]
    order_cuts = [
        int(d[DIAG_ORDER_CUT_BASE + i])
        for i in range(DIAG_ORDER_SOURCE_COUNT)
    ]
    lmr_r_tries = [
        int(d[DIAG_LMR_R1_TRY]),
        int(d[DIAG_LMR_R2_TRY]),
        int(d[DIAG_LMR_R3P_TRY]),
    ]
    lmr_r_fail_high = [
        int(d[DIAG_LMR_R1_FAIL_HIGH]),
        int(d[DIAG_LMR_R2_FAIL_HIGH]),
        int(d[DIAG_LMR_R3P_FAIL_HIGH]),
    ]
    lmr_quiet_try = int(d[DIAG_LMR_QUIET_TRY])
    lmr_quiet_fh = int(d[DIAG_LMR_QUIET_FAIL_HIGH])
    lmr_capture_try = int(d[DIAG_LMR_CAPTURE_TRY])
    lmr_capture_fh = int(d[DIAG_LMR_CAPTURE_FAIL_HIGH])
    lmr_keep = int(d[DIAG_LMR_RESEARCH_KEEP])
    lmr_reject = int(d[DIAG_LMR_RESEARCH_REJECT])
    post_bonus = int(d[DIAG_POST_LMR_BONUS_SAMPLE])
    post_malus = int(d[DIAG_POST_LMR_MALUS_SAMPLE])
    result = {
        "diagnostics_enabled": bool(search_context.diag_enabled),
        "nodes": int(total_nodes),
        "cut_nodes": cut_nodes,
        "cut_first": cut_first,
        "lmr_try": lmr_try,
        "lmr_research": lmr_re,
        # Mid / shallow prune family (n10k regime)
        "tt_cut": tt_cut,
        "lmp_skip": lmp_skip,
        "fp_skip": fp_skip,
        "rfp_cut": rfp_cut,
        "razor_cut": razor_cut,
        "nmp_cut": nmp_cut,
        "nmp_eligible": nmp_el,
        "nmp_gate_pass": nmp_gp,
        "nmp_null_try": nmp_try,
        "nmp_null_fh": nmp_fh,
        "nmp_verify_try": nmp_vt,
        "nmp_verify_fail": nmp_vf,
        "probcut": probcut,
        "lmp_skip_per_1k_nodes": 1000.0 * lmp_skip / n,
        "fp_skip_per_1k_nodes": 1000.0 * fp_skip / n,
        "rfp_cut_per_1k_nodes": 1000.0 * rfp_cut / n,
        "razor_cut_per_1k_nodes": 1000.0 * razor_cut / n,
        "nmp_cut_per_1k_nodes": 1000.0 * nmp_cut / n,
        "tt_cut_per_1k_nodes": 1000.0 * tt_cut / n,
        "probcut_per_1k_nodes": 1000.0 * probcut / n,
        "nmp_gate_pct": (100.0 * nmp_gp / nmp_el) if nmp_el else 0.0,
        "nmp_fh_pct": (100.0 * nmp_fh / nmp_try) if nmp_try else 0.0,
        "nmp_vfail_pct": (100.0 * nmp_vf / nmp_vt) if nmp_vt else 0.0,
        "hist_quiet_bonus": q_bonus,
        "hist_quiet_malus": q_malus,
        "hist_cap_bonus": int(d[DIAG_HIST_CAP_BONUS]),
        "hist_cap_malus": int(d[DIAG_HIST_CAP_MALUS]),
        "hist_cont_upd": int(d[DIAG_HIST_CONT_UPD]),
        "hist_tt_hit": int(d[DIAG_HIST_TT_HIT]),
        "hist_fail_low": int(d[DIAG_HIST_FAIL_LOW]),
        "hist_prune": prune,
        "hist_piece_to_upd": int(d[DIAG_HIST_PIECE_TO_UPD]),
        "hist_pawn_upd": int(d[DIAG_HIST_PAWN_UPD]),
        "hist_lph_upd": int(d[DIAG_HIST_LPH_UPD]),
        "hist_age": int(d[DIAG_HIST_AGE]),
        "hist_prune_per_1k_nodes": 1000.0 * prune / n,
        "hist_bonus_malus_ratio": (q_bonus / q_malus) if q_malus else float("inf") if q_bonus else 0.0,
        "lmr_research_pct": (100.0 * lmr_re / lmr_try) if lmr_try else 0.0,
        "cut_first_pct": (100.0 * cut_first / cut_nodes) if cut_nodes else 0.0,
        "order_quiet_cut": int(d[DIAG_ORDER_QUIET_CUT]),
        "order_quiet_cut_rank_sum": int(d[DIAG_ORDER_QUIET_CUT_RANK_SUM]),
        "order_nodes_before_cut_sum": int(d[DIAG_ORDER_NODES_BEFORE_CUT_SUM]),
        "order_source_try_total": sum(order_tries),
        "order_source_cut_total": sum(order_cuts),
        "order_source_cut_coverage_pct": (
            100.0 * sum(order_cuts) / cut_nodes if cut_nodes else 0.0
        ),
        "order_quiet_cut_avg_rank": (
            int(d[DIAG_ORDER_QUIET_CUT_RANK_SUM])
            / int(d[DIAG_ORDER_QUIET_CUT])
            if d[DIAG_ORDER_QUIET_CUT] else 0.0
        ),
        "order_avg_nodes_before_cut": (
            int(d[DIAG_ORDER_NODES_BEFORE_CUT_SUM]) / cut_nodes
            if cut_nodes else 0.0
        ),
        "order_killer1_try": int(d[DIAG_ORDER_KILLER1_TRY]),
        "order_killer1_cut": int(d[DIAG_ORDER_KILLER1_CUT]),
        "order_killer1_cut_pct": (
            100.0 * int(d[DIAG_ORDER_KILLER1_CUT])
            / int(d[DIAG_ORDER_KILLER1_TRY])
            if d[DIAG_ORDER_KILLER1_TRY] else 0.0
        ),
        "order_killer2_try": int(d[DIAG_ORDER_KILLER2_TRY]),
        "order_killer2_cut": int(d[DIAG_ORDER_KILLER2_CUT]),
        "order_killer2_cut_pct": (
            100.0 * int(d[DIAG_ORDER_KILLER2_CUT])
            / int(d[DIAG_ORDER_KILLER2_TRY])
            if d[DIAG_ORDER_KILLER2_TRY] else 0.0
        ),
        "order_counter_try": int(d[DIAG_ORDER_COUNTER_TRY]),
        "order_counter_cut": int(d[DIAG_ORDER_COUNTER_CUT]),
        "order_counter_cut_pct": (
            100.0 * int(d[DIAG_ORDER_COUNTER_CUT])
            / int(d[DIAG_ORDER_COUNTER_TRY])
            if d[DIAG_ORDER_COUNTER_TRY] else 0.0
        ),
        "lmr_r1_try": lmr_r_tries[0],
        "lmr_r2_try": lmr_r_tries[1],
        "lmr_r3p_try": lmr_r_tries[2],
        "lmr_r1_fail_high": lmr_r_fail_high[0],
        "lmr_r2_fail_high": lmr_r_fail_high[1],
        "lmr_r3p_fail_high": lmr_r_fail_high[2],
        "lmr_r1_fail_high_pct": (
            100.0 * lmr_r_fail_high[0] / lmr_r_tries[0]
            if lmr_r_tries[0] else 0.0
        ),
        "lmr_r2_fail_high_pct": (
            100.0 * lmr_r_fail_high[1] / lmr_r_tries[1]
            if lmr_r_tries[1] else 0.0
        ),
        "lmr_r3p_fail_high_pct": (
            100.0 * lmr_r_fail_high[2] / lmr_r_tries[2]
            if lmr_r_tries[2] else 0.0
        ),
        "lmr_quiet_try": lmr_quiet_try,
        "lmr_quiet_fail_high": lmr_quiet_fh,
        "lmr_quiet_fail_high_pct": (
            100.0 * lmr_quiet_fh / lmr_quiet_try if lmr_quiet_try else 0.0
        ),
        "lmr_capture_try": lmr_capture_try,
        "lmr_capture_fail_high": lmr_capture_fh,
        "lmr_capture_fail_high_pct": (
            100.0 * lmr_capture_fh / lmr_capture_try
            if lmr_capture_try else 0.0
        ),
        "lmr_research_keep": lmr_keep,
        "lmr_research_reject": lmr_reject,
        "lmr_research_reject_pct": (
            100.0 * lmr_reject / (lmr_keep + lmr_reject)
            if lmr_keep + lmr_reject else 0.0
        ),
        "lmr_research_deeper": int(d[DIAG_LMR_RESEARCH_DEEPER]),
        "lmr_research_shallower": int(d[DIAG_LMR_RESEARCH_SHALLOWER]),
        "lmr_fail_high_unverified": int(d[DIAG_LMR_FAIL_HIGH_UNVERIFIED]),
        "post_lmr_bonus_sample": post_bonus,
        "post_lmr_malus_sample": post_malus,
        "post_lmr_unverified_sample": int(d[DIAG_POST_LMR_UNVERIFIED_SAMPLE]),
        "post_lmr_bonus_butterfly_nonneg": int(
            d[DIAG_POST_LMR_BONUS_BUTTERFLY_NONNEG]
        ),
        "post_lmr_malus_butterfly_nonneg": int(
            d[DIAG_POST_LMR_MALUS_BUTTERFLY_NONNEG]
        ),
        "post_lmr_bonus_stat_nonneg": int(d[DIAG_POST_LMR_BONUS_STAT_NONNEG]),
        "post_lmr_malus_stat_nonneg": int(d[DIAG_POST_LMR_MALUS_STAT_NONNEG]),
        "post_lmr_bonus_butterfly_nonneg_pct": (
            100.0 * int(d[DIAG_POST_LMR_BONUS_BUTTERFLY_NONNEG]) / post_bonus
            if post_bonus else 0.0
        ),
        "post_lmr_malus_butterfly_nonneg_pct": (
            100.0 * int(d[DIAG_POST_LMR_MALUS_BUTTERFLY_NONNEG]) / post_malus
            if post_malus else 0.0
        ),
        "post_lmr_bonus_stat_nonneg_pct": (
            100.0 * int(d[DIAG_POST_LMR_BONUS_STAT_NONNEG]) / post_bonus
            if post_bonus else 0.0
        ),
        "post_lmr_malus_stat_nonneg_pct": (
            100.0 * int(d[DIAG_POST_LMR_MALUS_STAT_NONNEG]) / post_malus
            if post_malus else 0.0
        ),
        "q_tt_cut": int(d[DIAG_Q_TT_CUT]),
        "q_standpat_cut": int(d[DIAG_Q_STANDPAT_CUT]),
        "q_in_check": int(d[DIAG_Q_IN_CHECK]),
        "q_moves": int(d[DIAG_Q_MOVES]),
        "q_delta_skip": int(d[DIAG_Q_DELTA_SKIP]),
        "q_see_skip": int(d[DIAG_Q_SEE_SKIP]),
        "q_illegal_skip": int(d[DIAG_Q_ILLEGAL_SKIP]),
        "q_evasion_prefilter": int(d[DIAG_Q_EVASION_PREFILTER]),
        "q_beta_cut": int(d[DIAG_Q_BETA_CUT]),
        "see_capture_skip": int(d[DIAG_SEE_CAP_SKIP]),
        "see_quiet_skip": int(d[DIAG_SEE_QUIET_SKIP]),
        "check_extension": int(d[DIAG_CHECK_EXT]),
        "singular_try": int(d[DIAG_SINGULAR_TRY]),
        "singular_extension": int(d[DIAG_SINGULAR_EXT]),
        "double_extension": int(d[DIAG_DOUBLE_EXT]),
        "q_see_try": int(d[DIAG_Q_SEE_TRY]),
        "picker_see_try": int(d[DIAG_PICKER_SEE_TRY]),
        "main_capture_see_try": int(d[DIAG_MAIN_CAP_SEE_TRY]),
        "main_quiet_see_try": int(d[DIAG_MAIN_QUIET_SEE_TRY]),
        "lmr_capture_see_try": int(d[DIAG_LMR_CAP_SEE_TRY]),
        "probcut_see_try": int(d[DIAG_PROBCUT_SEE_TRY]),
        "check_gate_see_try": int(d[DIAG_CHECK_GATE_SEE_TRY]),
        "main_check_see_route": int(d[DIAG_MAIN_CHECK_SEE_ROUTE]),
        # P1 main quiet SEE / legality / geometry classification.
        "main_quiet_see_precheck": quiet_precheck,
        "main_quiet_pre_pin_illegal": quiet_pre_pin,
        "main_quiet_pre_king_illegal": quiet_pre_king,
        "main_quiet_see_pass": quiet_see_pass,
        "main_quiet_post_pin_illegal": quiet_post_pin,
        "main_quiet_post_king_illegal": quiet_post_king,
        "main_quiet_fallback_try": quiet_fallback_try,
        "main_quiet_fallback_reject": quiet_fallback_reject,
        "main_geometry_node": geometry_nodes,
        "main_score_geometry_rebuild": score_geometry_rebuild,
        "main_quiet_pre_illegal_per_see": (
            100.0 * (quiet_pre_pin + quiet_pre_king) / quiet_precheck
            if quiet_precheck else 0.0
        ),
        "main_quiet_post_illegal_per_see_pass": (
            100.0 * (quiet_post_pin + quiet_post_king + quiet_fallback_reject)
            / quiet_see_pass if quiet_see_pass else 0.0
        ),
        "main_quiet_fallback_reject_pct": (
            100.0 * quiet_fallback_reject / quiet_fallback_try
            if quiet_fallback_try else 0.0
        ),
        "main_score_geometry_per_node": (
            1000.0 * score_geometry_rebuild / geometry_nodes
            if geometry_nodes else 0.0
        ),
        # P3 qsearch cap / main in-check / TT deep-verify funnel.
        "q_cap_noncheck": q_cap_noncheck,
        "q_cap_check": q_cap_check,
        "q_cap_total": q_cap_noncheck + q_cap_check,
        "q_cap_forcing": q_cap_forcing,
        "q_cap_forcing_moves": q_cap_forcing_moves,
        "q_cap_forcing_pct": (
            100.0 * q_cap_forcing / (q_cap_noncheck + q_cap_check)
            if q_cap_noncheck + q_cap_check else 0.0
        ),
        "q_cap_forcing_moves_per_hit": (
            q_cap_forcing_moves / (q_cap_noncheck + q_cap_check)
            if q_cap_noncheck + q_cap_check else 0.0
        ),
        "main_in_check_nodes": main_in_check_nodes,
        "main_in_check_pick_try": main_in_check_pick_try,
        "main_in_check_pre_illegal": main_in_check_pre_illegal,
        "main_in_check_pre_pass": main_in_check_pre_pass,
        "main_in_check_fallback": main_in_check_fallback,
        "main_in_check_pre_illegal_pct": (
            100.0 * main_in_check_pre_illegal / main_in_check_pick_try
            if main_in_check_pick_try else 0.0
        ),
        "main_in_check_fallback_pct": (
            100.0 * main_in_check_fallback / main_in_check_pick_try
            if main_in_check_pick_try else 0.0
        ),
        "tt_verify_try": tt_verify_try,
        "tt_verify_pass": tt_verify_pass,
        "tt_verify_reject": tt_verify_reject,
        "tt_verify_saved_cut": tt_verify_saved_cut,
        "tt_verify_skip": tt_verify_skip,
        "tt_verify_pass_pct": (
            100.0 * tt_verify_pass / tt_verify_try if tt_verify_try else 0.0
        ),
        "tt_verify_reject_pct": (
            100.0 * tt_verify_reject / tt_verify_try if tt_verify_try else 0.0
        ),
        "tt_verify_skip_pct": (
            100.0 * tt_verify_skip / tt_verify_try if tt_verify_try else 0.0
        ),
        "tt_verify_saved_cut_pct": (
            100.0 * tt_verify_saved_cut / tt_verify_try if tt_verify_try else 0.0
        ),
        "tables": {
            "piece_to": _table_energy_stats(search_context.history_table, HISTORY_MAX_MAIN),
            "butterfly": _table_energy_stats(search_context.butterfly_history, HISTORY_MAX_BUTTERFLY),
            "capture": _table_energy_stats(search_context.capture_history, HISTORY_MAX_CAPTURE),
            "continuation": _table_energy_stats(
                search_context.continuation_history, HISTORY_MAX_CONTINUATION
            ),
            "pawn": _table_energy_stats(search_context.pawn_history, HISTORY_MAX_PAWN),
        },
    }
    source_names = (
        "tt", "good_capture", "good_quiet",
        "bad_capture", "bad_quiet", "other",
    )
    for i, name in enumerate(source_names):
        tries = order_tries[i]
        cuts = order_cuts[i]
        result[f"order_{name}_try"] = tries
        result[f"order_{name}_cut"] = cuts
        result[f"order_{name}_cut_pct"] = (
            100.0 * cuts / tries if tries else 0.0
        )
    result.update(_collect_aspiration_diag(search_context, total_nodes))
    result.update(_collect_history_prune_attribution(d, prune))
    return result


def format_search_diagnostics(search_context, total_nodes=0, q_nodes=0) -> str:
    """Human-readable summary of diag_stats after a search."""
    d = search_context.diag_stats
    cuts = int(d[DIAG_CUT_NODES])
    first = int(d[DIAG_CUT_FIRST])
    move_sum = int(d[DIAG_CUT_MOVE_SUM])
    lmr_try = int(d[DIAG_LMR_TRY])
    lmr_re = int(d[DIAG_LMR_RESEARCH])
    first_pct = (100.0 * first / cuts) if cuts else 0.0
    avg_cut_move = (move_sum / cuts) if cuts else 0.0
    lmr_re_pct = (100.0 * lmr_re / lmr_try) if lmr_try else 0.0
    q_pct = (100.0 * int(q_nodes) / int(total_nodes)) if total_nodes else 0.0
    nmp_el = int(d[DIAG_NMP_ELIGIBLE])
    nmp_gp = int(d[DIAG_NMP_GATE_PASS])
    nmp_try = int(d[DIAG_NMP_NULL_TRY])
    nmp_fh = int(d[DIAG_NMP_NULL_FH])
    nmp_vt = int(d[DIAG_NMP_VERIFY_TRY])
    nmp_vf = int(d[DIAG_NMP_VERIFY_FAIL])
    nmp_cut = int(d[DIAG_NMP_CUT])
    gate_pct = (100.0 * nmp_gp / nmp_el) if nmp_el else 0.0
    fh_pct = (100.0 * nmp_fh / nmp_try) if nmp_try else 0.0
    vf_pct = (100.0 * nmp_vf / nmp_vt) if nmp_vt else 0.0
    n = max(1, int(total_nodes))
    h_bonus = int(d[DIAG_HIST_QUIET_BONUS])
    h_malus = int(d[DIAG_HIST_QUIET_MALUS])
    h_prune = int(d[DIAG_HIST_PRUNE])
    bm_ratio = (h_bonus / h_malus) if h_malus else (float("inf") if h_bonus else 0.0)
    hp_attr = _collect_history_prune_attribution(d, h_prune)
    q_cap_nc = int(d[DIAG_Q_CAP_NONCHECK])
    q_cap_check = int(d[DIAG_Q_CAP_CHECK])
    q_cap_force = int(d[DIAG_Q_CAP_FORCING])
    q_cap_force_moves = int(d[DIAG_Q_CAP_FORCING_MOVES])
    q_cap_total = q_cap_nc + q_cap_check
    in_check_nodes = int(d[DIAG_MAIN_IN_CHECK_NODE])
    in_check_try = int(d[DIAG_MAIN_IN_CHECK_PICK_TRY])
    in_check_illegal = int(d[DIAG_MAIN_IN_CHECK_PRE_ILLEGAL])
    in_check_pass = int(d[DIAG_MAIN_IN_CHECK_PRE_PASS])
    in_check_fallback = int(d[DIAG_MAIN_IN_CHECK_FALLBACK])
    tt_verify_try = int(d[DIAG_TT_VERIFY_TRY])
    tt_verify_pass = int(d[DIAG_TT_VERIFY_PASS])
    tt_verify_reject = int(d[DIAG_TT_VERIFY_REJECT])
    tt_verify_saved = int(d[DIAG_TT_VERIFY_SAVED_CUT])
    tt_verify_skip = int(d[DIAG_TT_VERIFY_SKIP])
    aspiration = _collect_aspiration_diag(search_context, total_nodes)
    aspiration_fail_depths = " ".join(
        f"d{row['depth']}={row['fail_low']}/{row['fail_high']}/{row['research']}"
        for row in aspiration["aspiration_by_depth"]
        if row["research"]
    )
    order_try_fmt = [
        int(d[DIAG_ORDER_TRY_BASE + i])
        for i in range(DIAG_ORDER_SOURCE_COUNT)
    ]
    order_cut_fmt = [
        int(d[DIAG_ORDER_CUT_BASE + i])
        for i in range(DIAG_ORDER_SOURCE_COUNT)
    ]
    lmr_keep_fmt = int(d[DIAG_LMR_RESEARCH_KEEP])
    lmr_reject_fmt = int(d[DIAG_LMR_RESEARCH_REJECT])
    post_bonus_fmt = int(d[DIAG_POST_LMR_BONUS_SAMPLE])
    post_malus_fmt = int(d[DIAG_POST_LMR_MALUS_SAMPLE])
    lines = [
        f"cut_nodes={cuts}  cut_first={first} ({first_pct:.1f}%)  avg_move_at_cut={avg_cut_move:.2f}",
        f"lmr_try={lmr_try}  lmr_research={lmr_re} ({lmr_re_pct:.1f}%)",
        f"P4 order cut/try: tt={order_cut_fmt[0]}/{order_try_fmt[0]} "
        f"goodcap={order_cut_fmt[1]}/{order_try_fmt[1]} "
        f"goodquiet={order_cut_fmt[2]}/{order_try_fmt[2]} "
        f"badcap={order_cut_fmt[3]}/{order_try_fmt[3]} "
        f"badquiet={order_cut_fmt[4]}/{order_try_fmt[4]} "
        f"other={order_cut_fmt[5]}/{order_try_fmt[5]}",
        f"P4 quiet order: k1={int(d[DIAG_ORDER_KILLER1_CUT])}/"
        f"{int(d[DIAG_ORDER_KILLER1_TRY])} "
        f"k2={int(d[DIAG_ORDER_KILLER2_CUT])}/"
        f"{int(d[DIAG_ORDER_KILLER2_TRY])} "
        f"counter={int(d[DIAG_ORDER_COUNTER_CUT])}/"
        f"{int(d[DIAG_ORDER_COUNTER_TRY])} "
        f"avg_quiet_rank="
        f"{int(d[DIAG_ORDER_QUIET_CUT_RANK_SUM]) / int(d[DIAG_ORDER_QUIET_CUT]) if d[DIAG_ORDER_QUIET_CUT] else 0.0:.2f}",
        f"P4 LMR r1/r2/r3+ fh/try="
        f"{int(d[DIAG_LMR_R1_FAIL_HIGH])}/{int(d[DIAG_LMR_R1_TRY])} "
        f"{int(d[DIAG_LMR_R2_FAIL_HIGH])}/{int(d[DIAG_LMR_R2_TRY])} "
        f"{int(d[DIAG_LMR_R3P_FAIL_HIGH])}/{int(d[DIAG_LMR_R3P_TRY])} "
        f"verify keep/reject={lmr_keep_fmt}/{lmr_reject_fmt} "
        f"unverified={int(d[DIAG_LMR_FAIL_HIGH_UNVERIFIED])}",
        f"P4 post-LMR shadow: bonus={post_bonus_fmt} malus={post_malus_fmt} "
        f"unverified={int(d[DIAG_POST_LMR_UNVERIFIED_SAMPLE])} "
        f"butterfly_nonneg bonus/malus="
        f"{100.0 * int(d[DIAG_POST_LMR_BONUS_BUTTERFLY_NONNEG]) / post_bonus_fmt if post_bonus_fmt else 0.0:.1f}%/"
        f"{100.0 * int(d[DIAG_POST_LMR_MALUS_BUTTERFLY_NONNEG]) / post_malus_fmt if post_malus_fmt else 0.0:.1f}%",
        f"aspiration: iter={aspiration['aspiration_iterations']} "
        f"low={aspiration['aspiration_fail_low']} "
        f"high={aspiration['aspiration_fail_high']} "
        f"research={aspiration['aspiration_research']} "
        f"research/iter={aspiration['aspiration_research_per_iteration']:.3f} "
        f"wasted_nodes={aspiration['aspiration_wasted_nodes']} "
        f"({aspiration['aspiration_wasted_nodes_pct']:.2f}%) "
        f"max_delta={aspiration['aspiration_max_delta']}",
        f"tt_cut={int(d[DIAG_TT_CUT])}  nmp_cut={nmp_cut}  rfp_cut={int(d[DIAG_RFP_CUT])}  "
        f"razor_cut={int(d[DIAG_RAZOR_CUT])}  probcut={int(d[DIAG_PROBCUT])}",
        f"nmp_funnel: elig={nmp_el} gate={nmp_gp} ({gate_pct:.1f}%) try={nmp_try} "
        f"fh={nmp_fh} ({fh_pct:.1f}%) verify={nmp_vt} vfail={nmp_vf} ({vf_pct:.1f}%) cut={nmp_cut}",
        f"lmp_skip={int(d[DIAG_LMP_SKIP])}  fp_skip={int(d[DIAG_FP_SKIP])}",
        f"total_nodes={int(total_nodes)}  q_nodes={int(q_nodes)} ({q_pct:.1f}%)",
        f"qsearch: tt_cut={int(d[DIAG_Q_TT_CUT])} standpat_cut={int(d[DIAG_Q_STANDPAT_CUT])} "
        f"in_check={int(d[DIAG_Q_IN_CHECK])} moves={int(d[DIAG_Q_MOVES])} "
        f"delta_skip={int(d[DIAG_Q_DELTA_SKIP])} see_skip={int(d[DIAG_Q_SEE_SKIP])} "
        f"illegal={int(d[DIAG_Q_ILLEGAL_SKIP])} beta_cut={int(d[DIAG_Q_BETA_CUT])} "
        f"check_protect={int(d[DIAG_Q_CHECK_PROTECT])} "
        f"evasion_prefilter={int(d[DIAG_Q_EVASION_PREFILTER])}",
        f"P3 q_cap: noncheck={q_cap_nc} check={q_cap_check} forcing={q_cap_force} "
        f"({100.0 * q_cap_force / q_cap_total if q_cap_total else 0.0:.1f}%) "
        f"pseudo_moves={q_cap_force_moves} "
        f"avg_all_cap={q_cap_force_moves / q_cap_total if q_cap_total else 0.0:.2f}",
        f"P3 main_in_check: nodes={in_check_nodes} picker_try={in_check_try} "
        f"pre_illegal={in_check_illegal} "
        f"({100.0 * in_check_illegal / in_check_try if in_check_try else 0.0:.1f}%) "
        f"pre_pass={in_check_pass} fallback={in_check_fallback}",
        f"P3 tt_verify: try={tt_verify_try} pass={tt_verify_pass} "
        f"reject={tt_verify_reject} skip={tt_verify_skip} saved_cut={tt_verify_saved} "
        f"saved/try={100.0 * tt_verify_saved / tt_verify_try if tt_verify_try else 0.0:.1f}%",
        f"see/ext: cap_skip={int(d[DIAG_SEE_CAP_SKIP])} quiet_skip={int(d[DIAG_SEE_QUIET_SKIP])} "
        f"check_ext={int(d[DIAG_CHECK_EXT])} singular={int(d[DIAG_SINGULAR_EXT])}/"
        f"{int(d[DIAG_SINGULAR_TRY])} double={int(d[DIAG_DOUBLE_EXT])}",
        f"see_calls: q={int(d[DIAG_Q_SEE_TRY])} picker={int(d[DIAG_PICKER_SEE_TRY])} "
        f"main_cap={int(d[DIAG_MAIN_CAP_SEE_TRY])} main_quiet={int(d[DIAG_MAIN_QUIET_SEE_TRY])} "
        f"lmr_cap={int(d[DIAG_LMR_CAP_SEE_TRY])} probcut={int(d[DIAG_PROBCUT_SEE_TRY])} "
        f"check_gate={int(d[DIAG_CHECK_GATE_SEE_TRY])} "
        f"main_check_route={int(d[DIAG_MAIN_CHECK_SEE_ROUTE])}",
        f"P1 quiet: precheck={int(d[DIAG_MAIN_QUIET_SEE_PRECHECK])} "
        f"pre_pin={int(d[DIAG_MAIN_QUIET_PRE_PIN_ILLEGAL])} "
        f"pre_king={int(d[DIAG_MAIN_QUIET_PRE_KING_ILLEGAL])} "
        f"see_pass={int(d[DIAG_MAIN_QUIET_SEE_PASS])} "
        f"post_pin={int(d[DIAG_MAIN_QUIET_POST_PIN_ILLEGAL])} "
        f"post_king={int(d[DIAG_MAIN_QUIET_POST_KING_ILLEGAL])} "
        f"fallback={int(d[DIAG_MAIN_QUIET_FALLBACK_TRY])}/"
        f"{int(d[DIAG_MAIN_QUIET_FALLBACK_REJECT])} "
        f"geometry={int(d[DIAG_MAIN_GEOMETRY_NODE])}/"
        f"score_rebuild={int(d[DIAG_MAIN_SCORE_GEOMETRY_REBUILD])}",
        # History infrastructure (wired live path 2026-07-17)
        f"hist: q_bonus={h_bonus} q_malus={h_malus} ratio={bm_ratio:.3f}  "
        f"cap_b={int(d[DIAG_HIST_CAP_BONUS])} cap_m={int(d[DIAG_HIST_CAP_MALUS])}  "
        f"prune={h_prune} ({1000.0 * h_prune / n:.2f}/1k nodes)  "
        f"tt_hit={int(d[DIAG_HIST_TT_HIT])} fail_low={int(d[DIAG_HIST_FAIL_LOW])}  "
        f"cont_upd={int(d[DIAG_HIST_CONT_UPD])} piece_to={int(d[DIAG_HIST_PIECE_TO_UPD])}  "
        f"pawn={int(d[DIAG_HIST_PAWN_UPD])} lph={int(d[DIAG_HIST_LPH_UPD])}",
        f"P5 hist-prune attribution: cont_required="
        f"{hp_attr['hist_prune_cont_coalition_decisive_pct']:.1f}% "
        f"noncont_already={hp_attr['hist_prune_noncont_already_prunes_pct']:.1f}% "
        f"LMR_eligible={hp_attr['hist_prune_lmr_eligible_pct']:.1f}% "
        f"margin_mean={hp_attr['hist_prune_margin_mean']:.1f} "
        f"cont1_cap(hit/rescue/relief_mean)="
        f"{hp_attr['hist_prune_cont1_cap_hit']}/"
        f"{hp_attr['hist_prune_cont1_cap_rescue']}/"
        f"{hp_attr['hist_prune_cont1_cap_relief_mean']:.1f} "
        f"dominant(main/bf/pawn/c1/c2/c3/c4/c6)="
        f"{hp_attr['hist_prune_main_dominant']}/"
        f"{hp_attr['hist_prune_butterfly_dominant']}/"
        f"{hp_attr['hist_prune_pawn_dominant']}/"
        f"{hp_attr['hist_prune_cont1_dominant']}/"
        f"{hp_attr['hist_prune_cont2_dominant']}/"
        f"{hp_attr['hist_prune_cont3_dominant']}/"
        f"{hp_attr['hist_prune_cont4_dominant']}/"
        f"{hp_attr['hist_prune_cont6_dominant']}",
        format_history_table_stats(search_context),
    ]
    if aspiration_fail_depths:
        lines.insert(3, "aspiration_fail_depths (low/high/research): " + aspiration_fail_depths)
    return "\n".join(lines)


def warmup_search_jit(search_context=None):
    """
    Force-compile recursive search/QS (lazy njit) on a tiny position.
    Call from UCI `isready` or tournament harness so the first timed go is hot.
    """
    from chess_engine.classical.fen_parser import parse_fen
    from chess_engine.classical.transposition_table import create_transposition_table
    from chess_engine.classical.engine_types import SearchContext

    p_bbs, o_bbs, g_state = parse_fen(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    )
    if search_context is None:
        MAX_PLY_L = 128
        tt = create_transposition_table(16)
        search_context = SearchContext(
            tt,
            np.zeros(MAX_PLY_L * 2, dtype=np.uint16),
            np.zeros((MAX_PLY_L, MAX_PLY_L), dtype=np.uint16),
            np.zeros((12, 64), dtype=np.int32),
            np.zeros((64, 64), dtype=np.int32),
            np.zeros((5, 12, 64, 12, 64), dtype=np.int16),
            np.zeros((12, 64, 6), dtype=np.int32),
            np.full((8192, 12, 64), -1238, dtype=np.int16),
            np.zeros(16384, dtype=np.int16),
            np.zeros(16384, dtype=np.int16),
            np.zeros(16384, dtype=np.int16),
            np.zeros(16384, dtype=np.int16),
            np.zeros((12, 64, 12, 64), dtype=np.int16),
        )
    iterative_deepening_search(
        p_bbs, o_bbs, g_state, 1,
        {"optimum_time": 0, "maximum_time": 0},
        search_context,
        verbose=False,
    )
    return True


def iterative_deepening_search(piece_bbs, occupancy_bbs, game_state, max_depth, time_config, search_context, game_history_list=None, tt_generation=0, verbose=True):
    start_time = time.time()
    
    # --- Age history tables (Phase 1a: ÷2; Phase 2 soft-age rolled back) ---
    # Python's // 2 is floor division: -1 // 2 = -1 (sticks forever).
    # np.divide(..., out=..., casting='unsafe') truncates towards zero.
    np.divide(search_context.history_table, 2, out=search_context.history_table, casting='unsafe')
    np.divide(search_context.butterfly_history, 2, out=search_context.butterfly_history, casting='unsafe')
    np.divide(search_context.capture_history, 2, out=search_context.capture_history, casting='unsafe')
    np.divide(search_context.pawn_history, 2, out=search_context.pawn_history, casting='unsafe')
    np.divide(search_context.continuation_history, 2, out=search_context.continuation_history, casting='unsafe')
    np.divide(search_context.low_ply_history, 2, out=search_context.low_ply_history, casting='unsafe')
    # Fresh diagnostics each root search (not aged)
    if search_context.diag_enabled:
        search_context.diag_stats[:] = 0
        search_context.diag_stats[DIAG_HIST_AGE] += np.uint64(1)

    # --- Setup Game History ---
    if game_history_list is not None:
        count = len(game_history_list)
        limit = min(count, 1024)
        for i in range(limit):
            search_context.game_history[i] = game_history_list[i]
        search_context.game_history_count = limit
    else:
        search_context.game_history_count = 0
        
    # --- Setup TT Generation ---
    search_context.tt_generation = tt_generation

    maximum_time_ms = time_config.get('maximum_time', 0)
    optimum_time_ms = time_config.get('optimum_time', 0)
    
    transposition_table = search_context.transposition_table
    
    if maximum_time_ms > 0:
        search_context.end_time = start_time + (maximum_time_ms / 1000.0)
    else:
        search_context.end_time = 0.0
    
    search_context.stop_flag[0] = False

    nodes_limit = time_config.get('nodes_limit', 0)

    last_score, best_move_total = 0, NO_MOVE
    # Init before timer thread so nodes_limit closure never races NameError
    # ``total_nodes`` is the all-node counter used by UCI and fixed-node
    # stopping.  The public return tuple historically exposes main-search
    # nodes and qnodes separately, so keep a distinct accumulator for res[2].
    # Callers may then compute total nodes exactly once as search + qnodes.
    total_nodes, total_search_nodes, total_q_nodes, total_tt_hits = (
        np.uint64(v) for v in [0] * 4
    )
    last_completed_depth = 0
    best_move_from_last_depth = NO_MOVE
    # Fresh root search: no previous-iteration PV until depth 1 completes
    search_context.last_iteration_pv_len = np.int32(0)

    # Spawn a Python timer only for wall-clock limits. Fixed-node stopping is
    # checked directly in JIT through nodes_searched_array[1].
    timer_thread = None
    nodes_searched_array = search_context.nodes_searched_array
    # Slot 0 publishes per-iteration progress; slot 1 carries the remaining
    # fixed-node budget for direct checks in the JIT hot path.
    # Leaving the previous move's value here can make a fresh search stop
    # before depth 1 and return NO_MOVE (reported by tournament as failure).
    nodes_searched_array[0] = np.uint64(0)
    nodes_searched_array[1] = np.uint64(0)
    if search_context.end_time > 0.0:
        def timer_worker():
            end_time = search_context.end_time
            
            while True:
                if search_context.stop_flag[0]:
                    return
                # Check time limit
                if time.time() >= end_time:
                    search_context.stop_flag[0] = True
                    return
                time.sleep(0.002) # Polling interval of 2ms for high responsiveness
        
        timer_thread = threading.Thread(target=timer_worker, daemon=True)
        timer_thread.start()

    # Clear Counter Moves at start of search? Stockfish doesn't seem to reset them per search, 
    # but usually they are part of thread data. We can keep them or clear them.
    # Clearing them ensures no pollution from previous moves in different game contexts if not handled by generations.
    # However, for the same game, it might be useful. 
    # Let's clear them to be safe and consistent with "new search".
    # PRESERVED: search_context.counter_moves is NOT cleared here to prevent state pollution loss
    search_context.move_stack.fill(NO_MOVE)
    search_context.piece_stack.fill(-1)

    for current_depth in range(1, max_depth + 1):
        # Aspiration Window Logic
        alpha = -INFINITY
        beta = INFINITY
        delta = ASPIRATION_WINDOW_MIN
        aspiration_depth_slot = min(current_depth, DIAG_ASPIRATION_DEPTH_SLOTS - 1)
        
        if current_depth > 1:
            # HCE-tuned SF11 Dynamic Delta (等比縮放到 Centipawn)
            delta = max(
                ASPIRATION_WINDOW_MIN,
                ASPIRATION_WINDOW_BASE
                + abs(int(last_score)) // ASPIRATION_WINDOW_SCALE_DIV,
            )
            alpha = max(-INFINITY, last_score - delta)
            beta = min(INFINITY, last_score + delta)
            if search_context.diag_enabled:
                search_context.diag_stats[
                    DIAG_ASPIRATION_ITERATIONS_BASE + aspiration_depth_slot
                ] += np.uint64(1)
                max_delta_idx = (
                    DIAG_ASPIRATION_MAX_DELTA_BASE + aspiration_depth_slot
                )
                if delta > int(search_context.diag_stats[max_delta_idx]):
                    search_context.diag_stats[max_delta_idx] = np.uint64(delta)

        search_context.nodes_searched = np.uint64(0)
        # ``total_nodes`` already contains completed iterations; the shared
        # timer counter must start at zero for this new iteration.
        nodes_searched_array[0] = np.uint64(0)
        if nodes_limit > 0:
            remaining_nodes = int(nodes_limit) - int(total_nodes)
            nodes_searched_array[1] = np.uint64(max(1, remaining_nodes))
        else:
            nodes_searched_array[1] = np.uint64(0)
        failed_high_cnt = 0
        
        while True:
            search_context.reduction_stack[0] = 0
            # Fail-High 降深重搜策略
            adjusted_depth = max(1, current_depth - failed_high_cnt)
            # Root entry: force signature types so only one _search specialization is built.
            res = _search(
                piece_bbs, occupancy_bbs, game_state,
                _i32(adjusted_depth), _i32(alpha), _i32(beta),
                search_context, _i32(0), _u16(NO_MOVE), _b_true(), _b_false(),
            )
            score = res[0]
            
            # Always accumulate stats from the search (even if stopped or failed window)
            total_search_nodes += res[2]
            total_q_nodes += res[3]; total_tt_hits += res[4]

            if search_context.stop_flag[0]:
                break
            
            if score <= alpha:
                # Fail-Low: Beta 鎖定至 alpha (已證偽不可能大於 alpha)，Alpha 基於回傳分數擴張 (對齊 SF17/18)
                if search_context.diag_enabled:
                    search_context.diag_stats[
                        DIAG_ASPIRATION_FAIL_LOW_BASE + aspiration_depth_slot
                    ] += np.uint64(1)
                    search_context.diag_stats[
                        DIAG_ASPIRATION_RESEARCH_BASE + aspiration_depth_slot
                    ] += np.uint64(1)
                    search_context.diag_stats[
                        DIAG_ASPIRATION_WASTED_NODES_BASE + aspiration_depth_slot
                    ] += np.uint64(int(res[2]) + int(res[3]))
                beta = alpha
                alpha = max(-INFINITY, score - delta)
                failed_high_cnt = 0
                delta += max(1, 44 * delta // 128)
                if search_context.diag_enabled:
                    max_delta_idx = (
                        DIAG_ASPIRATION_MAX_DELTA_BASE + aspiration_depth_slot
                    )
                    if delta > int(search_context.diag_stats[max_delta_idx]):
                        search_context.diag_stats[max_delta_idx] = np.uint64(delta)
                # log_info(f"depth {current_depth} fail low ({score} <= {alpha}), widening to [{alpha}, {beta}]")
            elif score >= beta:
                # Fail-High: Beta 基於回傳分數擴張，Alpha 向上拉升收窄視窗 (對齊 SF17/18)
                if search_context.diag_enabled:
                    search_context.diag_stats[
                        DIAG_ASPIRATION_FAIL_HIGH_BASE + aspiration_depth_slot
                    ] += np.uint64(1)
                    search_context.diag_stats[
                        DIAG_ASPIRATION_RESEARCH_BASE + aspiration_depth_slot
                    ] += np.uint64(1)
                    search_context.diag_stats[
                        DIAG_ASPIRATION_WASTED_NODES_BASE + aspiration_depth_slot
                    ] += np.uint64(int(res[2]) + int(res[3]))
                alpha = max(beta - delta, alpha)
                beta = min(INFINITY, score + delta)
                failed_high_cnt += 1
                delta += max(1, 44 * delta // 128)
                if search_context.diag_enabled:
                    max_delta_idx = (
                        DIAG_ASPIRATION_MAX_DELTA_BASE + aspiration_depth_slot
                    )
                    if delta > int(search_context.diag_stats[max_delta_idx]):
                        search_context.diag_stats[max_delta_idx] = np.uint64(delta)
                # log_info(f"depth {current_depth} fail high ({score} >= {beta}), widening to [{alpha}, {beta}]")
            else:
                # Score is within window, we are done with this depth
                break
        
        total_nodes += search_context.nodes_searched

        if nodes_limit > 0 and total_nodes >= nodes_limit:
            search_context.stop_flag[0] = True

        if search_context.stop_flag[0]:
            if verbose:
                log_info(f"Search stopped at depth {current_depth} due to limit.")
            break

        # --- This block only runs if the search for the current depth was fully completed ---
        last_completed_depth = current_depth
        last_score = score

        if res[1] != NO_MOVE:
            best_move_from_last_depth = res[1]
        else:
            # GHI: use halfmove-aware key to match the key used inside the search
            _id_tt_key = np.uint64(get_tt_key(game_state[4], int(game_state[3])))
            tt_entry = probe_tt(transposition_table, _id_tt_key)
            if tt_entry['flag'] != TT_FLAG_NONE and tt_entry['best_move'] != NO_MOVE:
                best_move_from_last_depth = tt_entry['best_move']
            elif search_context.pv_table[0, 0] != NO_MOVE:
                best_move_from_last_depth = search_context.pv_table[0, 0]

        # Snapshot PV for next iteration's followPV tracking (SF lastIterationPV)
        lpv_len = 0
        for i in range(MAX_PLY):
            m = search_context.pv_table[0, i]
            if m == NO_MOVE:
                break
            search_context.last_iteration_pv[i] = m
            lpv_len += 1
        search_context.last_iteration_pv_len = np.int32(lpv_len)

        elapsed_time_ms = (time.time() - start_time) * 1000

        pv_moves = []
        for i in range(lpv_len):
            pv_moves.append(move_to_uci(search_context.last_iteration_pv[i]))

        pv_string = " ".join(pv_moves)
        uci_score_string = format_score_for_uci(score)

        if verbose:
            nps = int(total_nodes / (elapsed_time_ms / 1000)) if elapsed_time_ms > 0 else 0
            hashfull_val = hashfull(transposition_table, search_context.tt_generation)
            print(f"info depth {current_depth} score {uci_score_string} nodes {total_nodes} nps {nps} hashfull {hashfull_val / 10:.1f}% time {int(elapsed_time_ms)} pv {pv_string}")

        # Predictive soft time limit
        if maximum_time_ms > 0:
            if optimum_time_ms > 0 and elapsed_time_ms > optimum_time_ms:
                if verbose:
                    log_info(f"Optimum time reached at depth {current_depth}. Stopping.")
                break
            if elapsed_time_ms * 2.0 > maximum_time_ms:
                if verbose:
                    log_info(f"Predictive termination at depth {current_depth} to avoid timeout.")
                break

    # Signal timer thread to stop and clean up
    search_context.stop_flag[0] = True
    nodes_searched_array[1] = np.uint64(0)
    if timer_thread is not None:
        timer_thread.join()

    final_best_move = best_move_from_last_depth
    return (final_best_move, last_score, total_search_nodes, total_q_nodes, total_tt_hits, last_completed_depth)

def format_score_for_uci(score):
    if abs(score) > MATE_IN_MAX_PLY:
        if score > 0:
            moves_to_mate = (MATE_SCORE - score + 1) // 2
            return f"mate {moves_to_mate}"
        else:
            moves_to_mate = (MATE_SCORE + score + 1) // 2
            return f"mate {-moves_to_mate}"
    return f"cp {score}"
