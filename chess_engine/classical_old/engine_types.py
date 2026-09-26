# chess_engine/engine_types.py
import numba
import numpy as np
from numba import types
from numba.experimental import structref
from chess_engine.classical_old.transposition_table import numba_tt_entry_type
from chess_engine.classical_old.constants import (
    MAX_PLY, CORRECTION_HISTORY_SIZE, DIAG_SIZE, TUNE_SIZE,
    ENABLE_NMP, ENABLE_RFP, ENABLE_RAZORING,
    ENABLE_LMR, ENABLE_SHALLOW_SEE_PRUNING, ENABLE_LMP, ENABLE_FP,
    ENABLE_PROBCUT, ENABLE_MULTICUT,
    RFP_BASE_MULT, RAZORING_MARGIN, FP_BASE, FP_MULTIPLIER,
    PRUNING_CAPTURE_SEE_MARGIN, PRUNING_QUIET_SEE_MARGIN,
    LMR_BASE_OFFSET, LMR_HISTORY_SCALE, DELTA_PRUNING_MARGIN, PROBCUT_MARGIN,
    LMR_BAD_CAPTURE_BONUS, LMR_GOOD_CAPTURE_RELIEF, LMR_KILLER_COUNTER_RELIEF, LMP_SCALE_PERCENT,
    QS_SEE_THRESHOLD,
    LMR_CUTNODE_BONUS, LMR_NO_TTMOVE_BONUS, LMR_TTCAPTURE_BONUS,
    LMR_MOVECOUNT_FACTOR, LMR_TTMOVE_REDUCTION,
    TUNE_RFP_MULT, TUNE_RAZOR_MARGIN, TUNE_FP_BASE, TUNE_FP_MULT,
    TUNE_SEE_CAP_MARGIN, TUNE_SEE_QUIET_MARGIN, TUNE_LMR_BASE_OFFSET,
    TUNE_LMR_HIST_SCALE, TUNE_DELTA_MARGIN, TUNE_PROBCUT_MARGIN,
    TUNE_BAD_CAP_BONUS, TUNE_GOOD_CAP_RELIEF, TUNE_LMP_SCALE, TUNE_KILLER_RELIEF, TUNE_QS_SEE,
    TUNE_LMR_CUTNODE, TUNE_LMR_NO_TTMOVE, TUNE_LMR_TTCAP, TUNE_LMR_MC_FACTOR, TUNE_LMR_TTMOVE_RED,
    TUNE_NMP_SCOPE, TUNE_NMP_GATE, TUNE_NMP_R, TUNE_PROBCUT_STYLE,
    TUNE_LMR_TABLE_SCALE, TUNE_LMR_NOT_IMP,
    TUNE_NMP_G_BASE, TUNE_NMP_G_DEPTH, TUNE_NMP_G_IMP, TUNE_NMP_NEED_BETA,
    TUNE_NMP_R_BASE, TUNE_NMP_R_DIV, TUNE_NMP_VERIFY_D, TUNE_NMP_SCOPE_MIND,
    LMR_TABLE_SCALE_PERCENT, LMR_NOT_IMP_NUM,
    NMP_LEGACY_BASE, NMP_LEGACY_DEPTH_COEF, NMP_LEGACY_IMPROVING_COEF,
    NMP_LEGACY_R_BASE, NMP_LEGACY_R_DEPTH_DIV, NMP_VERIFICATION_DEPTH,
    NMP_SCOPE_MODE, NMP_SCOPE_MIN_DEPTH, NMP_GATE_MODE, NMP_R_MODE,
    PROBCUT_STYLE_MODE, NMP_NEED_BETA,
)

"""
此模組定義了西洋棋引擎中使用的 Numba 類型和類別。

SearchContext uses StructRef (not jitclass) so Numba disk-cache keys are
stable across processes. jitclass embeds id(class) in the type name, which
forced a full recompile of search (~minutes) on every engine restart.
"""

# --- Color Constants / 顏色常量 ---
WHITE, BLACK = 0, 1

# --- Piece Type Constants / 棋子類型常量 ---
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 0, 1, 2, 3, 4, 5


# --- Numba Type Signatures for the Refactored Board State / 重構後棋盤狀態的 Numba 類型簽名 ---
piece_bbs_signature = numba.uint64[::1]
occupancy_bbs_signature = numba.uint64[::1]
game_state_signature = numba.uint64[::1]
piece_counts_signature = numba.types.UniTuple(numba.int32, 12)

unmake_info_signature = numba.types.Tuple([
    numba.int8, numba.int8, numba.uint8, numba.uint8, numba.uint8, numba.uint64, numba.uint64,
    numba.uint64, numba.uint64, numba.uint64
])

# --- Search Context / 搜尋上下文 (StructRef) ---
# Field order is part of the type; keep stable for disk cache.
# Prefer C-contiguous array types so typeof(constructed ctx) matches the
# signature type used in @njit (layout mismatches create extra overloads).
search_context_spec = [
    ('transposition_table', numba.types.Array(numba_tt_entry_type, 1, 'C')),
    ('killer_moves', numba.uint16[::1]),
    ('pv_table', numba.uint16[:, ::1]),
    ('history_table', numba.int32[:, ::1]),
    ('nodes_searched', numba.uint64),
    ('nodes_searched_array', numba.uint64[::1]),
    ('end_time', numba.float64),
    ('stop_flag', numba.boolean[::1]),
    ('game_history', numba.uint64[::1]),
    ('game_history_count', numba.int32),
    ('ply_path_stack', numba.uint64[::1]),
    ('tt_generation', numba.uint8),
    ('counter_moves', numba.uint16[:, ::1]),
    ('move_stack', numba.uint16[::1]),
    ('piece_stack', numba.int8[::1]),
    ('static_eval_stack', numba.int32[::1]),

    ('move_scores', numba.int32[:, ::1]),
    ('moves_buffer', numba.uint16[:, ::1]),
    ('quiet_moves_tried', numba.uint16[:, ::1]),
    ('quiet_pieces_tried', numba.int8[:, ::1]),
    ('aggressor_cache', numba.int8[:, ::1]),
    ('victim_cache', numba.int8[:, ::1]),
    ('bad_captures', numba.uint16[:, ::1]),
    ('mp_stage', numba.int32[::1]),
    ('mp_current_idx', numba.int32[::1]),
    ('mp_captures_end', numba.int32[::1]),
    ('mp_quiets_end', numba.int32[::1]),
    ('mp_bad_captures_count', numba.int32[::1]),
    ('mp_bad_captures_idx', numba.int32[::1]),
    ('continuation_history', numba.int16[:, :, :, :, ::1]),
    ('low_ply_history', numba.int16[:, ::1]),
    ('pawn_history', numba.int16[:, :, ::1]),
    ('pawn_correction_history', numba.int16[::1]),
    ('minor_correction_history', numba.int16[::1]),
    ('non_pawn_correction_history_white', numba.int16[::1]),
    ('non_pawn_correction_history_black', numba.int16[::1]),
    ('continuation_correction_history', numba.int16[:, :, :, ::1]),
    ('butterfly_history', numba.int32[:, ::1]),
    ('capture_history', numba.int32[:, :, ::1]),
    ('capture_moves_tried', numba.uint16[:, ::1]),
    ('capture_aggressor_tried', numba.int8[:, ::1]),
    ('capture_victim_tried', numba.int8[:, ::1]),
    ('capture_tosq_tried', numba.int8[:, ::1]),
    ('enable_nmp', numba.boolean),
    ('enable_rfp', numba.boolean),
    ('enable_razoring', numba.boolean),
    ('enable_lmr', numba.boolean),
    ('enable_see_pruning', numba.boolean),
    ('enable_lmp', numba.boolean),
    ('enable_fp', numba.boolean),
    ('enable_probcut', numba.boolean),
    ('enable_multicut', numba.boolean),
    ('cutoff_cnt', numba.int32[::1]),
    ('move_count_stack', numba.int32[::1]),
    ('tt_hit_stack', numba.boolean[::1]),
    ('move_is_capture_stack', numba.boolean[::1]),
    ('reduction_stack', numba.int32[::1]),
    ('tt_pv_stack', numba.boolean[::1]),
    # SF followPV: previous iterative-deepening PV path for IIR / quiet-prune exemption
    ('follow_pv_stack', numba.boolean[::1]),
    ('last_iteration_pv', numba.uint16[::1]),
    ('last_iteration_pv_len', numba.int32),
    ('pawn_table_keys', numba.uint64[::1]),
    ('pawn_table_w_king_sq', numba.int8[::1]),
    ('pawn_table_b_king_sq', numba.int8[::1]),
    ('pawn_table_castling_rights', numba.uint8[::1]),
    ('pawn_table_mg', numba.int32[::1]),
    ('pawn_table_eg', numba.int32[::1]),
    ('pawn_table_w_shield_mg', numba.int32[::1]),
    ('pawn_table_w_shield_eg', numba.int32[::1]),
    ('pawn_table_b_shield_mg', numba.int32[::1]),
    ('pawn_table_b_shield_eg', numba.int32[::1]),
    ('pawn_table_w_passed', numba.uint64[::1]),
    ('pawn_table_b_passed', numba.uint64[::1]),
    ('pawn_table_w_attacks_span', numba.uint64[::1]),
    ('pawn_table_b_attacks_span', numba.uint64[::1]),
    ('diag_stats', numba.uint64[::1]),
    ('tune', numba.int32[::1]),
    # Runtime HCE weights for match-SPSA (live under Numba when mutated in-place)
    ('eval_weights', numba.int32[::1]),
    # Opt-in: production searches avoid diagnostic hot-path memory writes.
    ('diag_enabled', numba.boolean),
]

_SEARCH_CONTEXT_FIELD_NAMES = [name for name, _ in search_context_spec]


class SearchContextProxy(structref.StructRefProxy):
    """
    Python-side proxy for the search context StructRef.
    njit sees fields via structref.register; pure-Python attribute access is
    wired below (StructRef does not auto-mirror fields to Python).
    """

    def __new__(cls, *args):
        return structref.StructRefProxy.__new__(cls, *args)


@structref.register
class SearchContextType(types.StructRef):
    def preprocess_fields(self, fields):
        # Drop literal specializations so cache keys stay stable.
        return tuple((name, types.unliteral(typ)) for name, typ in fields)


structref.define_proxy(SearchContextProxy, SearchContextType, _SEARCH_CONTEXT_FIELD_NAMES)

# Fixed type used in @njit signatures (stable across processes).
search_context_type = SearchContextType(tuple(search_context_spec))

# --- Pure-Python field accessors (required by UCI / iterative_deepening aging) ---
# Generated once: each getter/setter is a tiny njit that reads/writes the field.
_SCALAR_FIELDS = {
    "nodes_searched", "end_time", "game_history_count", "tt_generation", "diag_enabled",
    "enable_nmp", "enable_rfp", "enable_razoring", "enable_lmr",
    "enable_see_pruning", "enable_lmp", "enable_fp", "enable_probcut",
    "enable_multicut",
    "last_iteration_pv_len",
}


def _install_python_field_accessors():
    g = globals()
    for name in _SEARCH_CONTEXT_FIELD_NAMES:
        # Getter
        # cache=False: these are exec()'d helpers (no source locator for disk cache).
        # They only run from pure-Python UCI / setup paths, not the hot search loop.
        getter_src = (
            f"@numba.njit(cache=False)\n"
            f"def _sc_get_{name}(self):\n"
            f"    return self.{name}\n"
        )
        exec(getter_src, g)
        getter = g[f"_sc_get_{name}"]

        def _make_prop(get_fn, set_fn=None):
            if set_fn is None:
                return property(lambda self, _g=get_fn: _g(self))
            return property(
                lambda self, _g=get_fn: _g(self),
                lambda self, value, _s=set_fn: _s(self, value),
            )

        if name in _SCALAR_FIELDS:
            setter_src = (
                f"@numba.njit(cache=False)\n"
                f"def _sc_set_{name}(self, value):\n"
                f"    self.{name} = value\n"
            )
            exec(setter_src, g)
            setter = g[f"_sc_set_{name}"]
            setattr(SearchContextProxy, name, _make_prop(getter, setter))
        else:
            # Arrays: return reference so in-place mutation works from Python.
            setattr(SearchContextProxy, name, _make_prop(getter))


_install_python_field_accessors()


def SearchContext(
    transposition_table, killer_moves, pv_table, history_table, butterfly_history,
    continuation_history, capture_history, pawn_history, pawn_correction_history,
    minor_correction_history, non_pawn_correction_history_white, non_pawn_correction_history_black,
    continuation_correction_history=None, enable_diagnostics=False
):
    """
    Build a SearchContext StructRef. Allocation stays in pure Python so construction
    does not require a process-specific jitclass compile.
    """
    if continuation_history.shape[0] < 5:
        new_shape = (5,) + continuation_history.shape[1:]
        new_continuation_history = np.zeros(new_shape, dtype=np.int16)
        new_continuation_history[0:continuation_history.shape[0]] = continuation_history
        continuation_history = new_continuation_history

    if continuation_correction_history is None:
        continuation_correction_history = np.zeros((12, 64, 12, 64), dtype=np.int16)

    # Ensure C-contiguous so typeof matches search_context_type layouts.
    transposition_table = np.ascontiguousarray(transposition_table)
    killer_moves = np.ascontiguousarray(killer_moves)
    pv_table = np.ascontiguousarray(pv_table)
    history_table = np.ascontiguousarray(history_table)
    butterfly_history = np.ascontiguousarray(butterfly_history)
    continuation_history = np.ascontiguousarray(continuation_history)
    capture_history = np.ascontiguousarray(capture_history)
    pawn_history = np.ascontiguousarray(pawn_history)
    pawn_correction_history = np.ascontiguousarray(pawn_correction_history)
    minor_correction_history = np.ascontiguousarray(minor_correction_history)
    non_pawn_correction_history_white = np.ascontiguousarray(non_pawn_correction_history_white)
    non_pawn_correction_history_black = np.ascontiguousarray(non_pawn_correction_history_black)
    continuation_correction_history = np.ascontiguousarray(continuation_correction_history)

    low_ply_history = np.zeros((5, 65536), dtype=np.int16)
    nodes_searched = np.uint64(0)
    # [0] publishes current iteration progress to Python; [1] is the optional
    # in-JIT node budget for this iteration (0 disables fixed-node stopping).
    nodes_searched_array = np.zeros(2, dtype=np.uint64)
    end_time = 0.0
    stop_flag = np.array([False], dtype=np.bool_)

    game_history = np.zeros(1024, dtype=np.uint64)
    game_history_count = np.int32(0)
    ply_path_stack = np.zeros(MAX_PLY, dtype=np.uint64)
    tt_generation = np.uint8(0)

    counter_moves = np.zeros((12, 64), dtype=np.uint16)
    move_stack = np.zeros(MAX_PLY, dtype=np.uint16)
    piece_stack = np.full(MAX_PLY, -1, dtype=np.int8)
    static_eval_stack = np.zeros(MAX_PLY, dtype=np.int32)

    move_scores = np.zeros((MAX_PLY, 256), dtype=np.int32)
    moves_buffer = np.zeros((MAX_PLY, 256), dtype=np.uint16)
    quiet_moves_tried = np.zeros((MAX_PLY, 256), dtype=np.uint16)
    quiet_pieces_tried = np.zeros((MAX_PLY, 256), dtype=np.int8)
    capture_moves_tried = np.zeros((MAX_PLY, 64), dtype=np.uint16)
    capture_aggressor_tried = np.zeros((MAX_PLY, 64), dtype=np.int8)
    capture_victim_tried = np.zeros((MAX_PLY, 64), dtype=np.int8)
    capture_tosq_tried = np.zeros((MAX_PLY, 64), dtype=np.int8)
    aggressor_cache = np.zeros((MAX_PLY, 256), dtype=np.int8)
    victim_cache = np.zeros((MAX_PLY, 256), dtype=np.int8)
    bad_captures = np.zeros((MAX_PLY, 256), dtype=np.uint16)

    mp_stage = np.zeros(MAX_PLY, dtype=np.int32)
    mp_current_idx = np.zeros(MAX_PLY, dtype=np.int32)
    mp_captures_end = np.zeros(MAX_PLY, dtype=np.int32)
    mp_quiets_end = np.zeros(MAX_PLY, dtype=np.int32)
    mp_bad_captures_count = np.zeros(MAX_PLY, dtype=np.int32)
    mp_bad_captures_idx = np.zeros(MAX_PLY, dtype=np.int32)

    cutoff_cnt = np.zeros(MAX_PLY, dtype=np.int32)
    move_count_stack = np.zeros(MAX_PLY, dtype=np.int32)
    tt_hit_stack = np.zeros(MAX_PLY, dtype=np.bool_)
    move_is_capture_stack = np.zeros(MAX_PLY, dtype=np.bool_)
    reduction_stack = np.zeros(MAX_PLY, dtype=np.int32)
    tt_pv_stack = np.zeros(MAX_PLY, dtype=np.bool_)
    follow_pv_stack = np.zeros(MAX_PLY, dtype=np.bool_)
    last_iteration_pv = np.zeros(MAX_PLY, dtype=np.uint16)
    last_iteration_pv_len = np.int32(0)

    pawn_table_keys = np.full(262144, np.uint64(0xFFFFFFFFFFFFFFFF), dtype=np.uint64)
    pawn_table_w_king_sq = np.full(262144, -1, dtype=np.int8)
    pawn_table_b_king_sq = np.full(262144, -1, dtype=np.int8)
    pawn_table_castling_rights = np.zeros(262144, dtype=np.uint8)
    pawn_table_mg = np.zeros(262144, dtype=np.int32)
    pawn_table_eg = np.zeros(262144, dtype=np.int32)
    pawn_table_w_shield_mg = np.zeros(262144, dtype=np.int32)
    pawn_table_w_shield_eg = np.zeros(262144, dtype=np.int32)
    pawn_table_b_shield_mg = np.zeros(262144, dtype=np.int32)
    pawn_table_b_shield_eg = np.zeros(262144, dtype=np.int32)
    pawn_table_w_passed = np.zeros(262144, dtype=np.uint64)
    pawn_table_b_passed = np.zeros(262144, dtype=np.uint64)
    pawn_table_w_attacks_span = np.zeros(262144, dtype=np.uint64)
    pawn_table_b_attacks_span = np.zeros(262144, dtype=np.uint64)

    diag_stats = np.zeros(DIAG_SIZE, dtype=np.uint64)
    diag_enabled = bool(enable_diagnostics)
    tune = np.zeros(TUNE_SIZE, dtype=np.int32)
    try:
        from tune_eval_match.eval_weights import default_eval_weights
        eval_weights = np.ascontiguousarray(default_eval_weights())
    except Exception:
        eval_weights = np.zeros(64, dtype=np.int32)
    tune[TUNE_RFP_MULT] = RFP_BASE_MULT
    tune[TUNE_RAZOR_MARGIN] = RAZORING_MARGIN
    tune[TUNE_FP_BASE] = FP_BASE
    tune[TUNE_FP_MULT] = FP_MULTIPLIER
    tune[TUNE_SEE_CAP_MARGIN] = PRUNING_CAPTURE_SEE_MARGIN
    tune[TUNE_SEE_QUIET_MARGIN] = PRUNING_QUIET_SEE_MARGIN
    tune[TUNE_LMR_BASE_OFFSET] = LMR_BASE_OFFSET
    tune[TUNE_LMR_HIST_SCALE] = LMR_HISTORY_SCALE
    tune[TUNE_DELTA_MARGIN] = DELTA_PRUNING_MARGIN
    tune[TUNE_PROBCUT_MARGIN] = PROBCUT_MARGIN
    tune[TUNE_BAD_CAP_BONUS] = LMR_BAD_CAPTURE_BONUS
    tune[TUNE_GOOD_CAP_RELIEF] = LMR_GOOD_CAPTURE_RELIEF
    tune[TUNE_LMP_SCALE] = LMP_SCALE_PERCENT
    tune[TUNE_KILLER_RELIEF] = LMR_KILLER_COUNTER_RELIEF
    tune[TUNE_QS_SEE] = QS_SEE_THRESHOLD
    tune[TUNE_LMR_CUTNODE] = LMR_CUTNODE_BONUS
    tune[TUNE_LMR_NO_TTMOVE] = LMR_NO_TTMOVE_BONUS
    tune[TUNE_LMR_TTCAP] = LMR_TTCAPTURE_BONUS
    tune[TUNE_LMR_MC_FACTOR] = LMR_MOVECOUNT_FACTOR
    tune[TUNE_LMR_TTMOVE_RED] = LMR_TTMOVE_REDUCTION
    tune[TUNE_NMP_SCOPE] = NMP_SCOPE_MODE
    tune[TUNE_NMP_GATE] = NMP_GATE_MODE
    tune[TUNE_NMP_R] = NMP_R_MODE
    tune[TUNE_PROBCUT_STYLE] = PROBCUT_STYLE_MODE
    tune[TUNE_LMR_TABLE_SCALE] = LMR_TABLE_SCALE_PERCENT
    tune[TUNE_LMR_NOT_IMP] = LMR_NOT_IMP_NUM
    tune[TUNE_NMP_G_BASE] = NMP_LEGACY_BASE
    tune[TUNE_NMP_G_DEPTH] = NMP_LEGACY_DEPTH_COEF
    tune[TUNE_NMP_G_IMP] = NMP_LEGACY_IMPROVING_COEF
    tune[TUNE_NMP_NEED_BETA] = NMP_NEED_BETA
    tune[TUNE_NMP_R_BASE] = NMP_LEGACY_R_BASE
    tune[TUNE_NMP_R_DIV] = NMP_LEGACY_R_DEPTH_DIV
    tune[TUNE_NMP_VERIFY_D] = NMP_VERIFICATION_DEPTH
    tune[TUNE_NMP_SCOPE_MIND] = NMP_SCOPE_MIN_DEPTH

    # Optional runtime override for match A/B (does not re-JIT search).
    import os
    env_scale = os.environ.get("CHESS_LMR_TABLE_SCALE")
    if env_scale is not None and str(env_scale).strip() != "":
        try:
            tune[TUNE_LMR_TABLE_SCALE] = int(env_scale)
        except ValueError:
            pass

    # Positional args MUST match search_context_spec order.
    ctx = SearchContextProxy(
        transposition_table,
        killer_moves,
        pv_table,
        history_table,
        nodes_searched,
        nodes_searched_array,
        end_time,
        stop_flag,
        game_history,
        game_history_count,
        ply_path_stack,
        tt_generation,
        counter_moves,
        move_stack,
        piece_stack,
        static_eval_stack,
        move_scores,
        moves_buffer,
        quiet_moves_tried,
        quiet_pieces_tried,
        aggressor_cache,
        victim_cache,
        bad_captures,
        mp_stage,
        mp_current_idx,
        mp_captures_end,
        mp_quiets_end,
        mp_bad_captures_count,
        mp_bad_captures_idx,
        continuation_history,
        low_ply_history,
        pawn_history,
        pawn_correction_history,
        minor_correction_history,
        non_pawn_correction_history_white,
        non_pawn_correction_history_black,
        continuation_correction_history,
        butterfly_history,
        capture_history,
        capture_moves_tried,
        capture_aggressor_tried,
        capture_victim_tried,
        capture_tosq_tried,
        bool(ENABLE_NMP),
        bool(ENABLE_RFP),
        bool(ENABLE_RAZORING),
        bool(ENABLE_LMR),
        bool(ENABLE_SHALLOW_SEE_PRUNING),
        bool(ENABLE_LMP),
        bool(ENABLE_FP),
        bool(ENABLE_PROBCUT),
        bool(ENABLE_MULTICUT),
        cutoff_cnt,
        move_count_stack,
        tt_hit_stack,
        move_is_capture_stack,
        reduction_stack,
        tt_pv_stack,
        follow_pv_stack,
        last_iteration_pv,
        last_iteration_pv_len,
        pawn_table_keys,
        pawn_table_w_king_sq,
        pawn_table_b_king_sq,
        pawn_table_castling_rights,
        pawn_table_mg,
        pawn_table_eg,
        pawn_table_w_shield_mg,
        pawn_table_w_shield_eg,
        pawn_table_b_shield_mg,
        pawn_table_b_shield_eg,
        pawn_table_w_passed,
        pawn_table_b_passed,
        pawn_table_w_attacks_span,
        pawn_table_b_attacks_span,
        diag_stats,
        tune,
        eval_weights,
        diag_enabled,
    )
    return ctx
