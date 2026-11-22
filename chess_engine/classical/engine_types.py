# chess_engine/engine_types.py
import numba
import numpy as np
from numba import types
from numba.experimental import structref
from chess_engine.classical.transposition_table import tt_entry_dtype, numba_tt_entry_type
from chess_engine.classical.constants import (
    MAX_PLY, CORRECTION_HISTORY_SIZE, DIAG_SIZE, TUNE_SIZE,
    ENABLE_NMP, ENABLE_RFP, ENABLE_RAZORING,
    ENABLE_LMR, ENABLE_SHALLOW_SEE_PRUNING, ENABLE_LMP, ENABLE_FP,
    ENABLE_PROBCUT, ENABLE_MULTICUT,
    RFP_BASE_MULT, RAZORING_COEFF, FP_BASE, FP_MULTIPLIER,
    PRUNING_CAPTURE_SEE_MARGIN, PRUNING_QUIET_SEE_MARGIN,
    LMR_BASE_OFFSET, LMR_HISTORY_SCALE, DELTA_PRUNING_MARGIN, PROBCUT_MARGIN,
    LMR_BAD_CAPTURE_BONUS, LMR_GOOD_CAPTURE_RELIEF, LMR_KILLER_COUNTER_RELIEF, LMP_SCALE_PERCENT,
    QS_SEE_THRESHOLD,
    LMR_CUTNODE_BONUS, LMR_NO_TTMOVE_BONUS, LMR_TTCAPTURE_BONUS,
    LMR_MOVECOUNT_FACTOR, LMR_TTMOVE_REDUCTION,
    TUNE_RFP_MULT, TUNE_RAZOR_COEFF, TUNE_FP_BASE, TUNE_FP_MULT,
    TUNE_SEE_CAP_MARGIN, TUNE_SEE_QUIET_MARGIN, TUNE_LMR_BASE_OFFSET,
    TUNE_LMR_HIST_SCALE, TUNE_DELTA_MARGIN, TUNE_PROBCUT_MARGIN,
    TUNE_BAD_CAP_BONUS, TUNE_GOOD_CAP_RELIEF, TUNE_LMP_SCALE, TUNE_KILLER_RELIEF, TUNE_QS_SEE,
    TUNE_LMR_CUTNODE, TUNE_LMR_NO_TTMOVE, TUNE_LMR_TTCAP, TUNE_LMR_MC_FACTOR, TUNE_LMR_TTMOVE_RED,
    TUNE_PROBCUT_STYLE,
    TUNE_LMR_TABLE_SCALE, TUNE_LMR_NOT_IMP,
    TUNE_NMP_EVAL_DIV, TUNE_NMP_BASE, TUNE_NMP_DEPTH_COEF, TUNE_NMP_IMP_COEF,
    TUNE_NMP_R_BASE, TUNE_NMP_R_DIV, TUNE_NMP_VERIFY_DEPTH,
    TUNE_SINGULAR_MARGIN_BASE, TUNE_SINGULAR_TTPV_BONUS, SINGULAR_MARGIN_BASE, SINGULAR_TTPV_BONUS,
    TUNE_RFP_BLEND_BETA, TUNE_CONT_PRUNE_THR, RFP_SCORE_BETA_WEIGHT, PRUNING_CONTINUATION_THRESHOLD,
    TUNE_HIST_BONUS_SCALE, TUNE_HIST_BONUS_CAP, TUNE_HIST_MALUS_CAP,
    TUNE_ORDER_KILLER_1, TUNE_ORDER_COUNTER,
    TUNE_HIST_MALUS_DECAY, TUNE_LMR_DIV_STEP, TUNE_HIST_MALUS_SCALE,
    TUNE_CHECK_SEE_THR, TUNE_CAP_SEE_HIST_NUM,
    CHECK_SEE_THRESHOLD, CAP_SEE_CAPTHIST_NUM,
    HISTORY_QUIET_MALUS_DECAY_NUM, LMR_DIVISOR_DEPTH_STEP, HISTORY_QUIET_MALUS_SCALE_NUM,
    HISTORY_BONUS_SCALE, HISTORY_BONUS_CAP, HISTORY_MALUS_CAP,
    QUIET_ORDER_KILLER_1, QUIET_ORDER_COUNTER,
    LMR_TABLE_SCALE_PERCENT, LMR_NOT_IMP_NUM,
    NMP_BASE, NMP_DEPTH_COEF, NMP_IMP_COEF, NMP_EVAL_MARGIN_DIV,
    NMP_R_BASE, NMP_R_DIV, NMP_VERIFICATION_DEPTH,
    PROBCUT_STYLE_MODE,
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
pawn_table_entry_dtype = np.dtype([
    ('keys', np.uint64),            # 8B (offset 0)
    ('w_passed', np.uint64),        # 8B (offset 8)
    ('b_passed', np.uint64),        # 8B (offset 16)
    ('w_attacks_span', np.uint64),  # 8B (offset 24)
    ('b_attacks_span', np.uint64),  # 8B (offset 32)
    ('mg', np.int32),               # 4B (offset 40)
    ('eg', np.int32),               # 4B (offset 44)
    ('w_shield_mg', np.int16),      # 2B (offset 48)
    ('w_shield_eg', np.int16),      # 2B (offset 50)
    ('b_shield_mg', np.int16),      # 2B (offset 52)
    ('b_shield_eg', np.int16),      # 2B (offset 54)
    ('w_king_sq', np.int8),         # 1B (offset 56)
    ('b_king_sq', np.int8),         # 1B (offset 57)
    ('castling_rights', np.uint8),  # 1B (offset 58)
    ('padding', np.uint8, 5),       # 5B (offset 59..63) -> 64 bytes
])
numba_pawn_table_entry_type = numba.from_dtype(pawn_table_entry_dtype)

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
    ('nmp_min_ply', numba.int32),
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
    ('pawn_table', numba.types.Array(numba_pawn_table_entry_type, 1, 'C')),
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
        if cls is SearchContextProxy and len(args) == len(_SEARCH_CONTEXT_FIELD_NAMES):
            return _construct_search_context_proxy(*args)
        return structref.StructRefProxy.__new__(cls, *args)


@structref.register
class SearchContextType(types.StructRef):
    def preprocess_fields(self, fields):
        # Drop literal specializations so cache keys stay stable.
        return tuple((name, types.unliteral(typ)) for name, typ in fields)


structref.define_proxy(SearchContextProxy, SearchContextType, _SEARCH_CONTEXT_FIELD_NAMES)

# Fixed type used in @njit signatures (stable across processes).
search_context_type = SearchContextType(tuple(search_context_spec))


@numba.njit(cache=True)
def _construct_search_context_proxy(
    transposition_table, killer_moves, pv_table, history_table, nodes_searched,
    nodes_searched_array, end_time, stop_flag, game_history, game_history_count,
    ply_path_stack, tt_generation, counter_moves, move_stack, piece_stack,
    static_eval_stack, move_scores, moves_buffer, quiet_moves_tried,
    quiet_pieces_tried, bad_captures, mp_stage,
    mp_current_idx, mp_captures_end, mp_quiets_end, mp_bad_captures_count,
    mp_bad_captures_idx, continuation_history, low_ply_history, pawn_history,
    pawn_correction_history, minor_correction_history,
    non_pawn_correction_history_white, non_pawn_correction_history_black,
    continuation_correction_history, butterfly_history, capture_history,
    capture_moves_tried, capture_aggressor_tried, capture_victim_tried,
    capture_tosq_tried, enable_nmp, nmp_min_ply, enable_rfp, enable_razoring,
    enable_lmr, enable_see_pruning, enable_lmp, enable_fp, enable_probcut,
    enable_multicut, cutoff_cnt, move_count_stack, tt_hit_stack,
    move_is_capture_stack, reduction_stack, tt_pv_stack, follow_pv_stack,
    last_iteration_pv, last_iteration_pv_len, pawn_table,
    diag_stats, tune, eval_weights, diag_enabled,
):
    """Disk-cached StructRef constructor matching search_context_spec field order."""
    return SearchContextProxy(
        transposition_table, killer_moves, pv_table, history_table, nodes_searched,
        nodes_searched_array, end_time, stop_flag, game_history, game_history_count,
        ply_path_stack, tt_generation, counter_moves, move_stack, piece_stack,
        static_eval_stack, move_scores, moves_buffer, quiet_moves_tried,
        quiet_pieces_tried, bad_captures, mp_stage,
        mp_current_idx, mp_captures_end, mp_quiets_end, mp_bad_captures_count,
        mp_bad_captures_idx, continuation_history, low_ply_history, pawn_history,
        pawn_correction_history, minor_correction_history,
        non_pawn_correction_history_white, non_pawn_correction_history_black,
        continuation_correction_history, butterfly_history, capture_history,
        capture_moves_tried, capture_aggressor_tried, capture_victim_tried,
        capture_tosq_tried, enable_nmp, nmp_min_ply, enable_rfp, enable_razoring,
        enable_lmr, enable_see_pruning, enable_lmp, enable_fp, enable_probcut,
        enable_multicut, cutoff_cnt, move_count_stack, tt_hit_stack,
        move_is_capture_stack, reduction_stack, tt_pv_stack, follow_pv_stack,
        last_iteration_pv, last_iteration_pv_len, pawn_table,
        diag_stats, tune, eval_weights, diag_enabled,
    )


# --- Pure-Python field accessors (required by UCI / iterative_deepening aging) ---
# Static typed getters/setters grouped by Numba field type keep real source
# locators for disk caching (cache=True) while avoiding 96 separate .nbc blobs.
_SCALAR_FIELDS = {
    "nodes_searched", "end_time", "game_history_count", "tt_generation", "diag_enabled",
    "enable_nmp", "nmp_min_ply", "enable_rfp", "enable_razoring", "enable_lmr",
    "enable_see_pruning", "enable_lmp", "enable_fp", "enable_probcut",
    "enable_multicut",
    "last_iteration_pv_len",
}


@numba.njit(cache=True, nogil=True)
def _sc_get_u64(self):
    return self.nodes_searched


@numba.njit(cache=True, nogil=True)
def _sc_set_u64(self, value):
    self.nodes_searched = value


@numba.njit(cache=True, nogil=True)
def _sc_get_f64(self):
    return self.end_time


@numba.njit(cache=True, nogil=True)
def _sc_set_f64(self, value):
    self.end_time = value


@numba.njit(cache=True, nogil=True)
def _sc_get_u8(self):
    return self.tt_generation


@numba.njit(cache=True, nogil=True)
def _sc_set_u8(self, value):
    self.tt_generation = value


@numba.njit(cache=True, nogil=True)
def _sc_get_i32(self, idx):
    if idx == 0:
        return self.game_history_count
    elif idx == 1:
        return self.nmp_min_ply
    else:
        return self.last_iteration_pv_len


@numba.njit(cache=True, nogil=True)
def _sc_set_i32(self, idx, value):
    if idx == 0:
        self.game_history_count = value
    elif idx == 1:
        self.nmp_min_ply = value
    else:
        self.last_iteration_pv_len = value


@numba.njit(cache=True, nogil=True)
def _sc_get_bool(self, idx):
    if idx == 0:
        return self.enable_nmp
    elif idx == 1:
        return self.enable_rfp
    elif idx == 2:
        return self.enable_razoring
    elif idx == 3:
        return self.enable_lmr
    elif idx == 4:
        return self.enable_see_pruning
    elif idx == 5:
        return self.enable_lmp
    elif idx == 6:
        return self.enable_fp
    elif idx == 7:
        return self.enable_probcut
    elif idx == 8:
        return self.enable_multicut
    else:
        return self.diag_enabled


@numba.njit(cache=True, nogil=True)
def _sc_set_bool(self, idx, value):
    if idx == 0:
        self.enable_nmp = value
    elif idx == 1:
        self.enable_rfp = value
    elif idx == 2:
        self.enable_razoring = value
    elif idx == 3:
        self.enable_lmr = value
    elif idx == 4:
        self.enable_see_pruning = value
    elif idx == 5:
        self.enable_lmp = value
    elif idx == 6:
        self.enable_fp = value
    elif idx == 7:
        self.enable_probcut = value
    elif idx == 8:
        self.enable_multicut = value
    else:
        self.diag_enabled = value


@numba.njit(cache=True, nogil=True)
def _sc_get_tt(self):
    return self.transposition_table


@numba.njit(cache=True, nogil=True)
def _sc_get_pawn_table(self):
    return self.pawn_table


@numba.njit(cache=True, nogil=True)
def _sc_get_i8_1d(self):
    return self.piece_stack


@numba.njit(cache=True, nogil=True)
def _sc_get_i8_2d(self, idx):
    if idx == 0:
        return self.quiet_pieces_tried
    elif idx == 1:
        return self.capture_aggressor_tried
    elif idx == 2:
        return self.capture_victim_tried
    else:
        return self.capture_tosq_tried


@numba.njit(cache=True, nogil=True)
def _sc_get_u16_1d(self, idx):
    if idx == 0:
        return self.killer_moves
    elif idx == 1:
        return self.move_stack
    else:
        return self.last_iteration_pv


@numba.njit(cache=True, nogil=True)
def _sc_get_u16_2d(self, idx):
    if idx == 0:
        return self.pv_table
    elif idx == 1:
        return self.counter_moves
    elif idx == 2:
        return self.moves_buffer
    elif idx == 3:
        return self.quiet_moves_tried
    elif idx == 4:
        return self.bad_captures
    else:
        return self.capture_moves_tried


@numba.njit(cache=True, nogil=True)
def _sc_get_i16_1d(self, idx):
    if idx == 0:
        return self.pawn_correction_history
    elif idx == 1:
        return self.minor_correction_history
    elif idx == 2:
        return self.non_pawn_correction_history_white
    else:
        return self.non_pawn_correction_history_black


@numba.njit(cache=True, nogil=True)
def _sc_get_i16_2d(self):
    return self.low_ply_history


@numba.njit(cache=True, nogil=True)
def _sc_get_i16_3d(self):
    return self.pawn_history


@numba.njit(cache=True, nogil=True)
def _sc_get_i16_4d(self):
    return self.continuation_correction_history


@numba.njit(cache=True, nogil=True)
def _sc_get_i16_5d(self):
    return self.continuation_history


@numba.njit(cache=True, nogil=True)
def _sc_get_i32_1d(self, idx):
    if idx == 0:
        return self.static_eval_stack
    elif idx == 1:
        return self.mp_stage
    elif idx == 2:
        return self.mp_current_idx
    elif idx == 3:
        return self.mp_captures_end
    elif idx == 4:
        return self.mp_quiets_end
    elif idx == 5:
        return self.mp_bad_captures_count
    elif idx == 6:
        return self.mp_bad_captures_idx
    elif idx == 7:
        return self.cutoff_cnt
    elif idx == 8:
        return self.move_count_stack
    elif idx == 9:
        return self.reduction_stack
    elif idx == 10:
        return self.tune
    else:
        return self.eval_weights


@numba.njit(cache=True, nogil=True)
def _sc_get_i32_2d(self, idx):
    if idx == 0:
        return self.history_table
    elif idx == 1:
        return self.move_scores
    else:
        return self.butterfly_history


@numba.njit(cache=True, nogil=True)
def _sc_get_i32_3d(self):
    return self.capture_history


@numba.njit(cache=True, nogil=True)
def _sc_get_u64_1d(self, idx):
    if idx == 0:
        return self.nodes_searched_array
    elif idx == 1:
        return self.game_history
    elif idx == 2:
        return self.ply_path_stack
    else:
        return self.diag_stats


@numba.njit(cache=True, nogil=True)
def _sc_get_bool_1d(self, idx):
    if idx == 0:
        return self.stop_flag
    elif idx == 1:
        return self.tt_hit_stack
    elif idx == 2:
        return self.move_is_capture_stack
    elif idx == 3:
        return self.tt_pv_stack
    else:
        return self.follow_pv_stack


def _coerce_u64(value):
    return value if type(value) is np.uint64 else np.uint64(int(value) & 0xFFFFFFFFFFFFFFFF)


def _coerce_f64(value):
    return value if type(value) is float else float(value)


def _coerce_i32(value):
    if type(value) is np.int32:
        return value
    return np.int32(((int(value) + 0x80000000) & 0xFFFFFFFF) - 0x80000000)


def _coerce_u8(value):
    return value if type(value) is np.uint8 else np.uint8(int(value) & 0xFF)


def _coerce_bool(value):
    return value if type(value) is bool else bool(value)


_SCALAR_ACCESSOR_GROUPS = (
    (_sc_get_u64, _sc_set_u64, _coerce_u64, ("nodes_searched",)),
    (_sc_get_f64, _sc_set_f64, _coerce_f64, ("end_time",)),
    (_sc_get_u8, _sc_set_u8, _coerce_u8, ("tt_generation",)),
    (_sc_get_i32, _sc_set_i32, _coerce_i32, ("game_history_count", "nmp_min_ply", "last_iteration_pv_len")),
    (
        _sc_get_bool,
        _sc_set_bool,
        _coerce_bool,
        (
            "enable_nmp", "enable_rfp", "enable_razoring", "enable_lmr",
            "enable_see_pruning", "enable_lmp", "enable_fp", "enable_probcut",
            "enable_multicut", "diag_enabled",
        ),
    ),
)

_ARRAY_ACCESSOR_GROUPS = (
    (_sc_get_tt, ("transposition_table",)),
    (_sc_get_pawn_table, ("pawn_table",)),
    (_sc_get_i8_1d, ("piece_stack",)),
    (
        _sc_get_i8_2d,
        (
            "quiet_pieces_tried", "capture_aggressor_tried",
            "capture_victim_tried", "capture_tosq_tried",
        ),
    ),
    (_sc_get_u16_1d, ("killer_moves", "move_stack", "last_iteration_pv")),
    (
        _sc_get_u16_2d,
        (
            "pv_table", "counter_moves", "moves_buffer",
            "quiet_moves_tried", "bad_captures", "capture_moves_tried",
        ),
    ),
    (
        _sc_get_i16_1d,
        (
            "pawn_correction_history", "minor_correction_history",
            "non_pawn_correction_history_white", "non_pawn_correction_history_black",
        ),
    ),
    (_sc_get_i16_2d, ("low_ply_history",)),
    (_sc_get_i16_3d, ("pawn_history",)),
    (_sc_get_i16_4d, ("continuation_correction_history",)),
    (_sc_get_i16_5d, ("continuation_history",)),
    (
        _sc_get_i32_1d,
        (
            "static_eval_stack", "mp_stage", "mp_current_idx", "mp_captures_end",
            "mp_quiets_end", "mp_bad_captures_count", "mp_bad_captures_idx",
            "cutoff_cnt", "move_count_stack", "reduction_stack",
            "tune", "eval_weights",
        ),
    ),
    (_sc_get_i32_2d, ("history_table", "move_scores", "butterfly_history")),
    (_sc_get_i32_3d, ("capture_history",)),
    (
        _sc_get_u64_1d,
        ("nodes_searched_array", "game_history", "ply_path_stack", "diag_stats"),
    ),
    (
        _sc_get_bool_1d,
        ("stop_flag", "tt_hit_stack", "move_is_capture_stack", "tt_pv_stack", "follow_pv_stack"),
    ),
)


def _make_prop(getter, idx=None, setter=None, coercer=None):
    if idx is None:
        if setter is None:
            return property(lambda self, _g=getter: _g(self))
        return property(
            lambda self, _g=getter: _g(self),
            lambda self, value, _s=setter, _c=coercer: _s(self, _c(value)),
        )
    idx_i32 = np.int32(idx)
    if setter is None:
        return property(lambda self, _g=getter, _i=idx_i32: _g(self, _i))
    return property(
        lambda self, _g=getter, _i=idx_i32: _g(self, _i),
        lambda self, value, _s=setter, _i=idx_i32, _c=coercer: _s(self, _i, _c(value)),
    )


def _install_python_field_accessors():
    """Bind cached StructRef property accessors onto SearchContextProxy."""
    registered_scalars = set()
    registered_arrays = set()

    for getter, setter, coercer, names in _SCALAR_ACCESSOR_GROUPS:
        use_index = len(names) > 1
        for idx, name in enumerate(names):
            registered_scalars.add(name)
            setattr(
                SearchContextProxy,
                name,
                _make_prop(getter, idx if use_index else None, setter, coercer),
            )

    for getter, names in _ARRAY_ACCESSOR_GROUPS:
        use_index = len(names) > 1
        for idx, name in enumerate(names):
            registered_arrays.add(name)
            # Arrays: return live reference so in-place mutation works from Python.
            setattr(
                SearchContextProxy,
                name,
                _make_prop(getter, idx if use_index else None),
            )

    if registered_scalars != _SCALAR_FIELDS:
        raise RuntimeError(f"Scalar accessor mismatch: {registered_scalars ^ _SCALAR_FIELDS}")
    all_fields = set(_SEARCH_CONTEXT_FIELD_NAMES)
    if (registered_scalars | registered_arrays) != all_fields:
        raise RuntimeError(
            f"SearchContext accessor mismatch: {(registered_scalars | registered_arrays) ^ all_fields}"
        )

    # Backward-compatible Python array views into the AoS pawn_table.
    for _pt_field in (
        "keys", "w_king_sq", "b_king_sq", "castling_rights",
        "mg", "eg", "w_shield_mg", "w_shield_eg", "b_shield_mg", "b_shield_eg",
        "w_passed", "b_passed", "w_attacks_span", "b_attacks_span",
    ):
        setattr(
            SearchContextProxy,
            f"pawn_table_{_pt_field}",
            property(lambda self, _f=_pt_field: _sc_get_pawn_table(self)[_f]),
        )


_install_python_field_accessors()


def SearchContext(
    transposition_table, killer_moves, pv_table, history_table, butterfly_history,
    continuation_history, capture_history, pawn_history, pawn_correction_history,
    minor_correction_history, non_pawn_correction_history_white, non_pawn_correction_history_black,
    continuation_correction_history=None, enable_diagnostics=False, stop_flag=None
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
    if stop_flag is None:
        stop_flag = np.array([False], dtype=np.bool_)
    else:
        stop_flag = np.ascontiguousarray(stop_flag)

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
    quiet_moves_tried = np.zeros((MAX_PLY, 64), dtype=np.uint16)
    quiet_pieces_tried = np.zeros((MAX_PLY, 64), dtype=np.int8)
    capture_moves_tried = np.zeros((MAX_PLY, 64), dtype=np.uint16)
    capture_aggressor_tried = np.zeros((MAX_PLY, 64), dtype=np.int8)
    capture_victim_tried = np.zeros((MAX_PLY, 64), dtype=np.int8)
    capture_tosq_tried = np.zeros((MAX_PLY, 64), dtype=np.int8)
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

    pawn_table = np.zeros(262144, dtype=pawn_table_entry_dtype)
    pawn_table['keys'].fill(np.uint64(0xFFFFFFFFFFFFFFFF))
    pawn_table['w_king_sq'].fill(-1)
    pawn_table['b_king_sq'].fill(-1)

    diag_stats = np.zeros(DIAG_SIZE, dtype=np.uint64)
    diag_enabled = bool(enable_diagnostics)
    tune = np.zeros(TUNE_SIZE, dtype=np.int32)
    try:
        from chess_engine.classical import constants as _classical_constants
        from tune_eval_match.eval_weights import default_eval_weights
        eval_weights = np.ascontiguousarray(default_eval_weights(_classical_constants))
    except Exception:
        eval_weights = np.zeros(64, dtype=np.int32)
    tune[TUNE_RFP_MULT] = RFP_BASE_MULT
    tune[TUNE_RAZOR_COEFF] = RAZORING_COEFF
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
    tune[TUNE_NMP_EVAL_DIV] = NMP_EVAL_MARGIN_DIV
    tune[TUNE_SINGULAR_MARGIN_BASE] = SINGULAR_MARGIN_BASE
    tune[TUNE_SINGULAR_TTPV_BONUS] = SINGULAR_TTPV_BONUS
    tune[TUNE_PROBCUT_STYLE] = PROBCUT_STYLE_MODE
    tune[TUNE_LMR_TABLE_SCALE] = LMR_TABLE_SCALE_PERCENT
    tune[TUNE_LMR_NOT_IMP] = LMR_NOT_IMP_NUM
    tune[TUNE_NMP_BASE] = NMP_BASE
    tune[TUNE_NMP_DEPTH_COEF] = NMP_DEPTH_COEF
    tune[TUNE_NMP_IMP_COEF] = NMP_IMP_COEF
    tune[TUNE_RFP_BLEND_BETA] = RFP_SCORE_BETA_WEIGHT
    tune[TUNE_NMP_R_BASE] = NMP_R_BASE
    tune[TUNE_NMP_R_DIV] = NMP_R_DIV
    tune[TUNE_NMP_VERIFY_DEPTH] = NMP_VERIFICATION_DEPTH
    tune[TUNE_CONT_PRUNE_THR] = PRUNING_CONTINUATION_THRESHOLD
    tune[TUNE_HIST_BONUS_SCALE] = HISTORY_BONUS_SCALE
    tune[TUNE_HIST_BONUS_CAP] = HISTORY_BONUS_CAP
    tune[TUNE_HIST_MALUS_CAP] = HISTORY_MALUS_CAP
    tune[TUNE_ORDER_KILLER_1] = QUIET_ORDER_KILLER_1
    tune[TUNE_ORDER_COUNTER] = QUIET_ORDER_COUNTER
    tune[TUNE_HIST_MALUS_DECAY] = HISTORY_QUIET_MALUS_DECAY_NUM
    tune[TUNE_LMR_DIV_STEP] = LMR_DIVISOR_DEPTH_STEP
    tune[TUNE_HIST_MALUS_SCALE] = HISTORY_QUIET_MALUS_SCALE_NUM
    tune[TUNE_CHECK_SEE_THR] = CHECK_SEE_THRESHOLD
    tune[TUNE_CAP_SEE_HIST_NUM] = CAP_SEE_CAPTHIST_NUM

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
        np.int32(0),
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
        pawn_table,
        diag_stats,
        tune,
        eval_weights,
        diag_enabled,
    )
    return ctx


def create_smp_helper_context(main_context):
    """
    Create a thread-private SearchContext for a Lazy SMP helper worker.
    Wraps main_context.transposition_table and main_context.stop_flag via np.frombuffer
    so the helper thread owns an independent Numba NRT_MemInfo header (preventing
    atomic refcount contention across CPU cores) while sharing physical memory.
    """
    main_tt = main_context.transposition_table
    main_stop = main_context.stop_flag
    shared_tt = np.frombuffer(main_tt.data, dtype=tt_entry_dtype)
    shared_stop_flag = np.frombuffer(main_stop.data, dtype=np.bool_)

    killer_moves = np.zeros(MAX_PLY * 2, dtype=np.uint16)
    pv_table = np.zeros((MAX_PLY, MAX_PLY), dtype=np.uint16)
    history_table = np.zeros((12, 64), dtype=np.int32)
    butterfly_history = np.zeros((64, 64), dtype=np.int32)
    continuation_history = np.zeros((5, 12, 64, 12, 64), dtype=np.int16)
    capture_history = np.zeros((12, 64, 6), dtype=np.int32)
    pawn_history = np.full((8192, 12, 64), -1238, dtype=np.int16)
    pawn_correction_history = np.zeros(CORRECTION_HISTORY_SIZE, dtype=np.int16)
    minor_correction_history = np.zeros(CORRECTION_HISTORY_SIZE, dtype=np.int16)
    non_pawn_correction_history_white = np.zeros(CORRECTION_HISTORY_SIZE, dtype=np.int16)
    non_pawn_correction_history_black = np.zeros(CORRECTION_HISTORY_SIZE, dtype=np.int16)
    continuation_correction_history = np.zeros((12, 64, 12, 64), dtype=np.int16)

    helper_ctx = SearchContext(
        shared_tt,
        killer_moves,
        pv_table,
        history_table,
        butterfly_history,
        continuation_history,
        capture_history,
        pawn_history,
        pawn_correction_history,
        minor_correction_history,
        non_pawn_correction_history_white,
        non_pawn_correction_history_black,
        continuation_correction_history,
        enable_diagnostics=False,
        stop_flag=shared_stop_flag,
    )
    helper_ctx.tune[:] = main_context.tune[:]
    helper_ctx.eval_weights[:] = main_context.eval_weights[:]
    helper_ctx.enable_nmp = bool(main_context.enable_nmp)
    helper_ctx.enable_rfp = bool(main_context.enable_rfp)
    helper_ctx.enable_razoring = bool(main_context.enable_razoring)
    helper_ctx.enable_lmr = bool(main_context.enable_lmr)
    helper_ctx.enable_see_pruning = bool(main_context.enable_see_pruning)
    helper_ctx.enable_lmp = bool(main_context.enable_lmp)
    helper_ctx.enable_fp = bool(main_context.enable_fp)
    helper_ctx.enable_probcut = bool(main_context.enable_probcut)
    helper_ctx.enable_multicut = bool(main_context.enable_multicut)
    helper_ctx.diag_enabled = False
    return helper_ctx


def reset_search_context(ctx):
    """
    Reset all thread-local history tables, stacks, pawn hash tables, and counters on ctx
    (used on ucinewgame for both main and Lazy SMP helper contexts).
    """
    ctx.killer_moves.fill(0)
    ctx.pv_table.fill(0)
    ctx.history_table.fill(0)
    ctx.butterfly_history.fill(0)
    ctx.continuation_history.fill(0)
    ctx.capture_history.fill(0)
    ctx.pawn_history.fill(-1238)
    ctx.pawn_correction_history.fill(0)
    ctx.minor_correction_history.fill(0)
    ctx.non_pawn_correction_history_white.fill(0)
    ctx.non_pawn_correction_history_black.fill(0)
    ctx.continuation_correction_history.fill(0)
    ctx.counter_moves.fill(0)
    ctx.low_ply_history.fill(0)
    ctx.move_stack.fill(0)
    ctx.piece_stack.fill(-1)
    ctx.static_eval_stack.fill(0)
    ctx.reduction_stack.fill(0)
    ctx.tt_pv_stack.fill(False)
    ctx.follow_pv_stack.fill(False)
    ctx.last_iteration_pv.fill(0)
    ctx.last_iteration_pv_len = np.int32(0)
    ctx.nodes_searched = np.uint64(0)
    ctx.nodes_searched_array.fill(0)
    ctx.pawn_table_keys.fill(np.uint64(0xFFFFFFFFFFFFFFFF))
    ctx.pawn_table_w_king_sq.fill(-1)
    ctx.pawn_table_b_king_sq.fill(-1)
    ctx.stop_flag[0] = False
