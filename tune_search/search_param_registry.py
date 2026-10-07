"""Single source of truth for classical-search tuning parameters.

The old :mod:`tune_search.param_config` listed a mixture of stale names and
only described the handful of values that had once been swept.  This module
keeps the *engine* defaults in ``classical.constants`` and adds tuning
metadata around them:

* ``runtime_attr`` points at a ``SearchContext.tune`` slot.  These parameters
  can be changed in-place by a fixed-node/in-process tuner after one JIT.
* ``source_name`` is the scalar constant patched before importing ``main``.
  Compile-time candidates are safe to test in subprocesses, but require a
  fresh process/JIT after each parameter set.
* ``kind`` distinguishes continuous SPSA axes from discrete mode/depth gates.

The registry intentionally contains more candidates than the 34 runtime
slots.  ``runtime`` is the conservative live-slot profile; ``lmr_runtime``
and ``lmr_compile`` split the LMR group by execution cost, while the named
groups and ``all`` profile are for staged campaigns rather than one giant SPSA
vector.
No registry code is imported by the Numba hot path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

from chess_engine.classical import constants as C


# The registry predates the canonical Fishtest SPSA interface.  Its
# ``spsa_r_end`` values were recorded as relative aggressiveness numbers
# (0.4--0.8), not as the dimensionless ``a/c²`` ratio consumed by Fishtest.
# Keep the source metadata stable for old reports, but expose the converted
# ratio explicitly to the new tuner.  A separate property makes the unit
# conversion auditable and prevents an accidental silent reinterpretation of
# old checkpoints.
LEGACY_SPSA_R_END_SCALE = 0.01


@dataclass(frozen=True)
class SearchParamSpec:
    """Metadata for one scalar search parameter."""

    name: str
    source_name: str
    minimum: int
    maximum: int
    step: int
    group: str
    kind: str = "continuous"  # continuous | discrete
    runtime_attr: Optional[str] = None
    spsa_r_end: float = 1.0
    description: str = ""

    @property
    def default(self) -> int:
        return int(getattr(C, self.source_name))

    @property
    def runtime_index(self) -> Optional[int]:
        if self.runtime_attr is None:
            return None
        return int(getattr(C, self.runtime_attr))

    @property
    def is_runtime(self) -> bool:
        return self.runtime_attr is not None

    @property
    def canonical_r_end(self) -> float:
        """Return the canonical Fishtest ``r_end = a_end / c_end²``.

        Existing registry files use the historical relative-gain notation in
        ``spsa_r_end``.  New campaigns use this converted value; old state
        files are intentionally rejected by the tuner update-mode guard.
        """
        return float(self.spsa_r_end) * LEGACY_SPSA_R_END_SCALE

    def legacy_tuple(self) -> Tuple[int, int, int, int]:
        """Return the tuple expected by the legacy command-line tuner."""
        return (self.default, self.minimum, self.maximum, self.step)


def _spec(
    name: str,
    source_name: str,
    minimum: int,
    maximum: int,
    step: int,
    group: str,
    *,
    kind: str = "continuous",
    runtime_attr: Optional[str] = None,
    spsa_r_end: float = 1.0,
    description: str = "",
) -> SearchParamSpec:
    return SearchParamSpec(
        name=name,
        source_name=source_name,
        minimum=int(minimum),
        maximum=int(maximum),
        step=int(step),
        group=group,
        kind=kind,
        runtime_attr=runtime_attr,
        spsa_r_end=float(spsa_r_end),
        description=description,
    )


# ---------------------------------------------------------------------------
# Runtime-backed axes (one compile; 39 slots in SearchContext.tune[])
# ---------------------------------------------------------------------------
_RUNTIME_SPECS = [
    _spec("RFP_BASE_MULT", "RFP_BASE_MULT", 120, 230, 5, "pruning", runtime_attr="TUNE_RFP_MULT", spsa_r_end=0.8),
    _spec("RAZORING_COEFF", "RAZORING_COEFF", 200, 800, 25, "pruning", runtime_attr="TUNE_RAZOR_COEFF", spsa_r_end=0.8),
    _spec("FP_BASE", "FP_BASE", 100, 300, 10, "pruning", runtime_attr="TUNE_FP_BASE", spsa_r_end=0.8),
    _spec("FP_MULTIPLIER", "FP_MULTIPLIER", 60, 220, 10, "pruning", runtime_attr="TUNE_FP_MULT", spsa_r_end=0.8),
    _spec("PRUNING_CAPTURE_SEE_MARGIN", "PRUNING_CAPTURE_SEE_MARGIN", -250, 0, 25, "pruning", runtime_attr="TUNE_SEE_CAP_MARGIN", spsa_r_end=0.6),
    _spec("PRUNING_QUIET_SEE_MARGIN", "PRUNING_QUIET_SEE_MARGIN", -200, -5, 15, "pruning", runtime_attr="TUNE_SEE_QUIET_MARGIN", spsa_r_end=0.6),
    _spec("LMR_BASE_OFFSET", "LMR_BASE_OFFSET", 250, 700, 25, "lmr", runtime_attr="TUNE_LMR_BASE_OFFSET", spsa_r_end=0.8),
    _spec("LMR_HISTORY_SCALE", "LMR_HISTORY_SCALE", 50, 300, 25, "lmr", runtime_attr="TUNE_LMR_HIST_SCALE", spsa_r_end=0.8),
    _spec("DELTA_PRUNING_MARGIN", "DELTA_PRUNING_MARGIN", 150, 700, 25, "qsearch", runtime_attr="TUNE_DELTA_MARGIN", spsa_r_end=0.8),
    _spec("PROBCUT_MARGIN", "PROBCUT_MARGIN", 80, 300, 10, "pruning", runtime_attr="TUNE_PROBCUT_MARGIN", spsa_r_end=0.6),
    _spec("LMR_BAD_CAPTURE_BONUS", "LMR_BAD_CAPTURE_BONUS", 0, 1800, 64, "lmr", runtime_attr="TUNE_BAD_CAP_BONUS", spsa_r_end=0.8),
    _spec("LMR_GOOD_CAPTURE_RELIEF", "LMR_GOOD_CAPTURE_RELIEF", 0, 1800, 64, "lmr", runtime_attr="TUNE_GOOD_CAP_RELIEF", spsa_r_end=0.8),
    _spec("LMP_SCALE_PERCENT", "LMP_SCALE_PERCENT", 70, 140, 5, "pruning", runtime_attr="TUNE_LMP_SCALE", spsa_r_end=0.6),
    _spec("LMR_KILLER_COUNTER_RELIEF", "LMR_KILLER_COUNTER_RELIEF", 0, 1800, 64, "lmr", runtime_attr="TUNE_KILLER_RELIEF", spsa_r_end=0.8),
    _spec("QS_SEE_THRESHOLD", "QS_SEE_THRESHOLD", -200, 0, 10, "qsearch", runtime_attr="TUNE_QS_SEE", spsa_r_end=0.6),
    _spec("LMR_CUTNODE_BONUS", "LMR_CUTNODE_BONUS", 0, 3000, 128, "lmr", runtime_attr="TUNE_LMR_CUTNODE", spsa_r_end=0.8),
    _spec("LMR_NO_TTMOVE_BONUS", "LMR_NO_TTMOVE_BONUS", 0, 2500, 128, "lmr", runtime_attr="TUNE_LMR_NO_TTMOVE", spsa_r_end=0.8),
    _spec("LMR_TTCAPTURE_BONUS", "LMR_TTCAPTURE_BONUS", 0, 2500, 128, "lmr", runtime_attr="TUNE_LMR_TTCAP", spsa_r_end=0.8),
    _spec("LMR_MOVECOUNT_FACTOR", "LMR_MOVECOUNT_FACTOR", 0, 100, 4, "lmr", runtime_attr="TUNE_LMR_MC_FACTOR", spsa_r_end=0.8),
    _spec("LMR_TTMOVE_REDUCTION", "LMR_TTMOVE_REDUCTION", 0, 2500, 128, "lmr", runtime_attr="TUNE_LMR_TTMOVE_RED", spsa_r_end=0.8),
    _spec("NMP_SCOPE_MODE", "NMP_SCOPE_MODE", 0, 3, 1, "nmp", kind="discrete", runtime_attr="TUNE_NMP_SCOPE", spsa_r_end=0.4),
    _spec("NMP_GATE_MODE", "NMP_GATE_MODE", 0, 3, 1, "nmp", kind="discrete", runtime_attr="TUNE_NMP_GATE", spsa_r_end=0.4),
    _spec("NMP_R_MODE", "NMP_R_MODE", 0, 2, 1, "nmp", kind="discrete", runtime_attr="TUNE_NMP_R", spsa_r_end=0.4),
    _spec("PROBCUT_STYLE_MODE", "PROBCUT_STYLE_MODE", 0, 2, 1, "pruning", kind="discrete", runtime_attr="TUNE_PROBCUT_STYLE", spsa_r_end=0.4),
    _spec("LMR_TABLE_SCALE_PERCENT", "LMR_TABLE_SCALE_PERCENT", 70, 130, 5, "lmr", runtime_attr="TUNE_LMR_TABLE_SCALE", spsa_r_end=0.8),
    _spec("LMR_NOT_IMP_NUM", "LMR_NOT_IMP_NUM", 0, 384, 16, "lmr", runtime_attr="TUNE_LMR_NOT_IMP", spsa_r_end=0.8),
    _spec("NMP_LEGACY_BASE", "NMP_LEGACY_BASE", 100, 350, 10, "nmp", runtime_attr="TUNE_NMP_G_BASE", spsa_r_end=0.8),
    _spec("NMP_LEGACY_DEPTH_COEF", "NMP_LEGACY_DEPTH_COEF", 0, 30, 2, "nmp", runtime_attr="TUNE_NMP_G_DEPTH", spsa_r_end=0.8),
    _spec("NMP_LEGACY_IMPROVING_COEF", "NMP_LEGACY_IMPROVING_COEF", 0, 90, 5, "nmp", runtime_attr="TUNE_NMP_G_IMP", spsa_r_end=0.8),
    _spec("NMP_NEED_BETA", "NMP_NEED_BETA", 0, 1, 1, "nmp", kind="discrete", runtime_attr="TUNE_NMP_NEED_BETA", spsa_r_end=0.4),
    _spec("NMP_LEGACY_R_BASE", "NMP_LEGACY_R_BASE", 4, 12, 1, "nmp", runtime_attr="TUNE_NMP_R_BASE", spsa_r_end=0.6),
    _spec("NMP_LEGACY_R_DEPTH_DIV", "NMP_LEGACY_R_DEPTH_DIV", 1, 8, 1, "nmp", kind="discrete", runtime_attr="TUNE_NMP_R_DIV", spsa_r_end=0.6),
    _spec("NMP_VERIFICATION_DEPTH", "NMP_VERIFICATION_DEPTH", 5, 12, 1, "nmp", kind="discrete", runtime_attr="TUNE_NMP_VERIFY_D", spsa_r_end=0.4),
    _spec("NMP_SCOPE_MIN_DEPTH", "NMP_SCOPE_MIN_DEPTH", 3, 10, 1, "nmp", kind="discrete", runtime_attr="TUNE_NMP_SCOPE_MIND", spsa_r_end=0.4),
    _spec("HISTORY_BONUS_SCALE", "HISTORY_BONUS_SCALE", 40, 240, 10, "history", runtime_attr="TUNE_HIST_BONUS_SCALE", spsa_r_end=0.8),
    _spec("HISTORY_BONUS_CAP", "HISTORY_BONUS_CAP", 800, 3000, 100, "history", runtime_attr="TUNE_HIST_BONUS_CAP", spsa_r_end=0.8),
    _spec("HISTORY_MALUS_CAP", "HISTORY_MALUS_CAP", 800, 3000, 100, "history", runtime_attr="TUNE_HIST_MALUS_CAP", spsa_r_end=0.8),
    _spec("QUIET_ORDER_KILLER_1", "QUIET_ORDER_KILLER_1", 100000, 600000, 10000, "ordering", runtime_attr="TUNE_ORDER_KILLER_1", spsa_r_end=0.5),
    _spec("QUIET_ORDER_COUNTER", "QUIET_ORDER_COUNTER", 100000, 500000, 10000, "ordering", runtime_attr="TUNE_ORDER_COUNTER", spsa_r_end=0.5),
]


# ---------------------------------------------------------------------------
# Compile-time candidates.  They are deliberately separate from the runtime
# profile: changing one requires a fresh engine process/JIT, but they are
# still valid scalars used by search/qsearch/ordering and should not be lost
# in a stale 32-entry config file.
# ---------------------------------------------------------------------------
_COMPILE_SPECS = [
    # Aspiration and root control
    _spec("ASPIRATION_WINDOW_MIN", "ASPIRATION_WINDOW_MIN", 8, 40, 2, "aspiration", spsa_r_end=0.5),
    _spec("ASPIRATION_WINDOW_BASE", "ASPIRATION_WINDOW_BASE", 10, 60, 2, "aspiration", spsa_r_end=0.5),
    _spec("ASPIRATION_WINDOW_SCALE_DIV", "ASPIRATION_WINDOW_SCALE_DIV", 60, 240, 10, "aspiration", spsa_r_end=0.5),
    # Pruning gates and margins
    _spec("PRUNING_HISTORY_THRESHOLD", "PRUNING_HISTORY_THRESHOLD", -10000, -3000, 250, "pruning", spsa_r_end=0.5),
    _spec("PRUNING_SHALLOW_DEPTH", "PRUNING_SHALLOW_DEPTH", 6, 16, 1, "pruning", kind="discrete", spsa_r_end=0.4),
    _spec("RFP_MAX_DEPTH", "RFP_MAX_DEPTH", 3, 8, 1, "pruning", kind="discrete", spsa_r_end=0.4),
    _spec("RFP_NO_TT_PENALTY", "RFP_NO_TT_PENALTY", 0, 80, 5, "pruning", spsa_r_end=0.5),
    _spec("FP_MAX_DEPTH", "FP_MAX_DEPTH", 3, 10, 1, "pruning", kind="discrete", spsa_r_end=0.4),
    _spec("LMP_MIN_LIMIT", "LMP_MIN_LIMIT", 0, 8, 1, "pruning", kind="discrete", spsa_r_end=0.4),
    _spec("PROBCUT_R", "PROBCUT_R", 2, 6, 1, "pruning", kind="discrete", spsa_r_end=0.4),
    _spec("PROBCUT_MIN_DEPTH", "PROBCUT_MIN_DEPTH", 3, 8, 1, "pruning", kind="discrete", spsa_r_end=0.4),
    _spec("PROBCUT_SF_BASE", "PROBCUT_SF_BASE", 100, 260, 10, "pruning", spsa_r_end=0.5),
    _spec("PROBCUT_SF_IMPROVING", "PROBCUT_SF_IMPROVING", 0, 80, 5, "pruning", spsa_r_end=0.5),
    _spec("PROBCUT_TT_DEPTH_OFFSET", "PROBCUT_TT_DEPTH_OFFSET", 1, 5, 1, "pruning", kind="discrete", spsa_r_end=0.4),
    # NMP compile-time gates / SF-style branch
    _spec("NMP_MIN_DEPTH", "NMP_MIN_DEPTH", 2, 6, 1, "nmp", kind="discrete", spsa_r_end=0.4),
    _spec("NMP_MIN_SIDE_NON_PAWNS", "NMP_MIN_SIDE_NON_PAWNS", 0, 5, 1, "nmp", kind="discrete", spsa_r_end=0.4),
    _spec("NMP_SF_MARGIN_DEPTH_COEF", "NMP_SF_MARGIN_DEPTH_COEF", 0, 64, 4, "nmp", spsa_r_end=0.5),
    _spec("NMP_SF_MARGIN_BASE", "NMP_SF_MARGIN_BASE", 150, 450, 10, "nmp", spsa_r_end=0.5),
    _spec("NMP_SF_MARGIN_IMPROVING", "NMP_SF_MARGIN_IMPROVING", 0, 80, 5, "nmp", spsa_r_end=0.5),
    _spec("NMP_EVAL_MARGIN_DIV", "NMP_EVAL_MARGIN_DIV", 50, 300, 10, "nmp", spsa_r_end=0.5),
    _spec("NMP_EVAL_MARGIN_MAX", "NMP_EVAL_MARGIN_MAX", 0, 6, 1, "nmp", kind="discrete", spsa_r_end=0.4),
    _spec("NMP_SF_R_BASE", "NMP_SF_R_BASE", 500, 1100, 25, "nmp", spsa_r_end=0.5),
    _spec("NMP_SF_R_DEPTH", "NMP_SF_R_DEPTH", 0, 128, 8, "nmp", spsa_r_end=0.5),
    _spec("NMP_SF_R_DIV", "NMP_SF_R_DIV", 128, 512, 16, "nmp", spsa_r_end=0.5),
    # LMR and extension geometry
    _spec("LMR_MIN_DEPTH", "LMR_MIN_DEPTH", 2, 6, 1, "lmr", kind="discrete", spsa_r_end=0.4),
    _spec("LMR_MIN_QUIET_MOVE_INDEX", "LMR_MIN_QUIET_MOVE_INDEX", 1, 8, 1, "lmr", kind="discrete", spsa_r_end=0.4),
    _spec("LMR_TT_SCORE_GT_ALPHA", "LMR_TT_SCORE_GT_ALPHA", 400, 1200, 32, "lmr", spsa_r_end=0.5),
    _spec("LMR_TT_DEPTH_GE", "LMR_TT_DEPTH_GE", 500, 1300, 32, "lmr", spsa_r_end=0.5),
    _spec("LMR_TT_DEPTH_GE_CUT", "LMR_TT_DEPTH_GE_CUT", 500, 1400, 32, "lmr", spsa_r_end=0.5),
    _spec("LMR_ADVANCED_PAWN_RELIEF", "LMR_ADVANCED_PAWN_RELIEF", 0, 2500, 128, "lmr", spsa_r_end=0.5),
    _spec("LMR_CAPTURE_VICTIM_SCALE", "LMR_CAPTURE_VICTIM_SCALE", 400, 1200, 32, "lmr", spsa_r_end=0.5),
    _spec("LMR_STAT_SCORE_DIV", "LMR_STAT_SCORE_DIV", 2048, 8192, 256, "lmr", spsa_r_end=0.5),
    _spec("LMR_RESEARCH_DEEPER_MARGIN", "LMR_RESEARCH_DEEPER_MARGIN", 0, 100, 5, "lmr", spsa_r_end=0.5),
    _spec("LMR_RESEARCH_SHALLOWER_MARGIN", "LMR_RESEARCH_SHALLOWER_MARGIN", 0, 100, 5, "lmr", spsa_r_end=0.5),
    _spec("CHECK_EXT_NON_PV_MAX_DEPTH", "CHECK_EXT_NON_PV_MAX_DEPTH", 2, 8, 1, "extensions", kind="discrete", spsa_r_end=0.4),
    _spec("MIN_SINGULAR_DEPTH", "MIN_SINGULAR_DEPTH", 5, 10, 1, "extensions", kind="discrete", spsa_r_end=0.4),
    _spec("SINGULAR_MARGIN_BASE", "SINGULAR_MARGIN_BASE", 20, 120, 5, "extensions", spsa_r_end=0.5),
    _spec("SINGULAR_TTPV_BONUS", "SINGULAR_TTPV_BONUS", 0, 140, 5, "extensions", spsa_r_end=0.5),
    _spec("SINGULAR_MARGIN_DIV", "SINGULAR_MARGIN_DIV", 30, 100, 5, "extensions", spsa_r_end=0.5),
    _spec("SINGULAR_TT_DEPTH_SLACK", "SINGULAR_TT_DEPTH_SLACK", 1, 6, 1, "extensions", kind="discrete", spsa_r_end=0.4),
    _spec("SINGULAR_DOUBLE_EXT_MIN_DEPTH", "SINGULAR_DOUBLE_EXT_MIN_DEPTH", 7, 14, 1, "extensions", kind="discrete", spsa_r_end=0.4),
    # QSearch and ordering scores
    _spec("MAX_QUIESCENCE_DEPTH", "MAX_QUIESCENCE_DEPTH", 4, 8, 1, "qsearch", kind="discrete", spsa_r_end=0.4),
    _spec("NULL_MOVE_REDUCTION", "NULL_MOVE_REDUCTION", 1, 4, 1, "nmp", kind="discrete", spsa_r_end=0.4),
    _spec("SCORE_GOOD_CAPTURE_BONUS", "SCORE_GOOD_CAPTURE_BONUS", 10000, 30000, 1000, "ordering", spsa_r_end=0.5),
    _spec("SCORE_BAD_CAPTURE_PENALTY", "SCORE_BAD_CAPTURE_PENALTY", -10000, -1000, 500, "ordering", spsa_r_end=0.5),
    _spec("GOOD_QUIET_THRESHOLD", "GOOD_QUIET_THRESHOLD", -3000, 0, 100, "ordering", spsa_r_end=0.5),
    _spec("CHECK_BONUS", "CHECK_BONUS", 10000, 30000, 1000, "ordering", spsa_r_end=0.5),
    _spec("CHECK_SEE_THRESHOLD", "CHECK_SEE_THRESHOLD", -200, 0, 10, "ordering", spsa_r_end=0.5),
    _spec("THREAT_MULTIPLIER", "THREAT_MULTIPLIER", 0, 40, 2, "ordering", spsa_r_end=0.5),
    _spec("QUIET_ORDER_KILLER_2", "QUIET_ORDER_KILLER_2", 100000, 600000, 10000, "ordering", spsa_r_end=0.5),
    _spec("LPH_ORDER_SCALE", "LPH_ORDER_SCALE", 0, 16, 1, "ordering", spsa_r_end=0.5),
    # History / correction feedback
    _spec("HISTORY_WEIGHT_MAIN", "HISTORY_WEIGHT_MAIN", 1, 6, 1, "history", spsa_r_end=0.5),
    _spec("HISTORY_WEIGHT_CONT_1", "HISTORY_WEIGHT_CONT_1", 0, 8, 1, "history", spsa_r_end=0.5),
    _spec("HISTORY_WEIGHT_CONT_2", "HISTORY_WEIGHT_CONT_2", 0, 8, 1, "history", spsa_r_end=0.5),
    _spec("HISTORY_WEIGHT_CONT_3", "HISTORY_WEIGHT_CONT_3", 0, 8, 1, "history", spsa_r_end=0.5),
    _spec("HISTORY_WEIGHT_CONT_4", "HISTORY_WEIGHT_CONT_4", 0, 8, 1, "history", spsa_r_end=0.5),
    _spec("HISTORY_WEIGHT_CONT_5", "HISTORY_WEIGHT_CONT_5", 0, 8, 1, "history", spsa_r_end=0.5),
    _spec("HISTORY_PRUNE_CONT1_FLOOR", "HISTORY_PRUNE_CONT1_FLOOR", -10000, 0, 256, "history", spsa_r_end=0.5),
    _spec("HISTORY_NODE_WIDTH_DIV", "HISTORY_NODE_WIDTH_DIV", 64, 512, 16, "history", spsa_r_end=0.5),
    _spec("FAIL_LOW_MIN_LEGAL_MOVES", "FAIL_LOW_MIN_LEGAL_MOVES", 1, 10, 1, "history", kind="discrete", spsa_r_end=0.4),
    _spec("FAIL_LOW_STATIC_MARGIN", "FAIL_LOW_STATIC_MARGIN", 0, 180, 10, "history", spsa_r_end=0.5),
    _spec("FAIL_LOW_PARENT_STATIC_MARGIN", "FAIL_LOW_PARENT_STATIC_MARGIN", 0, 180, 10, "history", spsa_r_end=0.5),
    _spec("FAIL_LOW_BASE_DEPTH_MULT", "FAIL_LOW_BASE_DEPTH_MULT", 0, 200, 10, "history", spsa_r_end=0.5),
    _spec("FAIL_LOW_SCALE_DIV", "FAIL_LOW_SCALE_DIV", 256, 1024, 32, "history", spsa_r_end=0.5),
    # TT consistency / hindsight safeguards
    _spec("HINDSIGHT_REDUCE_MIN_PRIOR", "HINDSIGHT_REDUCE_MIN_PRIOR", 1, 6, 1, "safeguards", kind="discrete", spsa_r_end=0.4),
    _spec("HINDSIGHT_INCREASE_MIN_PRIOR", "HINDSIGHT_INCREASE_MIN_PRIOR", 1, 5, 1, "safeguards", kind="discrete", spsa_r_end=0.4),
    _spec("HINDSIGHT_EVAL_SUM_MARGIN", "HINDSIGHT_EVAL_SUM_MARGIN", 50, 400, 25, "safeguards", spsa_r_end=0.5),
    _spec("RAZOR_DISSONANCE_MAX", "RAZOR_DISSONANCE_MAX", 100, 500, 25, "safeguards", spsa_r_end=0.5),
    _spec("FP_DISSONANCE_THRESHOLD", "FP_DISSONANCE_THRESHOLD", 50, 300, 25, "safeguards", spsa_r_end=0.5),
    _spec("TT_CUTOFF_HALFMOVE_MAX", "TT_CUTOFF_HALFMOVE_MAX", 50, 100, 5, "safeguards", kind="discrete", spsa_r_end=0.4),
    _spec("TT_CONSISTENCY_MIN_DEPTH", "TT_CONSISTENCY_MIN_DEPTH", 2, 8, 1, "safeguards", kind="discrete", spsa_r_end=0.4),
    _spec("TT_DEEP_VERIFY_DEPTH", "TT_DEEP_VERIFY_DEPTH", 4, 10, 1, "safeguards", kind="discrete", spsa_r_end=0.4),
    _spec("TT_PENALIZE_MIN_DEPTH", "TT_PENALIZE_MIN_DEPTH", 3, 8, 1, "safeguards", kind="discrete", spsa_r_end=0.4),
    _spec("CORRECTION_HISTORY_DIVISOR", "CORRECTION_HISTORY_DIVISOR", 65536, 262144, 8192, "history", spsa_r_end=0.5),
    _spec("CORRECTION_HISTORY_UPDATE_DEPTH", "CORRECTION_HISTORY_UPDATE_DEPTH", 2, 8, 1, "history", kind="discrete", spsa_r_end=0.4),
]


SEARCH_PARAM_SPECS: Tuple[SearchParamSpec, ...] = tuple(_RUNTIME_SPECS + _COMPILE_SPECS)
SEARCH_PARAM_BY_NAME: Dict[str, SearchParamSpec] = {
    spec.name: spec for spec in SEARCH_PARAM_SPECS
}

# Stable profiles are used by the tuner.  A large SPSA vector is available for
# experiments, but is never selected implicitly.
PROFILE_NAMES = (
    "runtime",
    "runtime_continuous",
    "pruning",
    "nmp",
    "lmr",
    "lmr_runtime",
    "lmr_compile",
    "qsearch",
    "history",
    "ordering",
    "safeguards",
    "aspiration",
    "all",
)
DEFAULT_PROFILE = "runtime"

# Unordered switches: value 2 is not "between" 1 and 3, so SPSA's finite
# difference over them carries no gradient information.  Evaluate these with
# dedicated A/B matches (tools/run_search_param_match.py), never inside SPSA.
CATEGORICAL_PARAMS = frozenset(
    {
        "NMP_SCOPE_MODE",
        "NMP_GATE_MODE",
        "NMP_R_MODE",
        "PROBCUT_STYLE_MODE",
        "NMP_NEED_BETA",
    }
)


def validate_registry() -> Tuple[str, ...]:
    """Return human-readable invariant errors (empty tuple means valid)."""
    errors = []
    seen_names = set()
    seen_runtime = set()
    for spec in SEARCH_PARAM_SPECS:
        if spec.name in seen_names:
            errors.append(f"duplicate parameter name: {spec.name}")
        seen_names.add(spec.name)
        if not hasattr(C, spec.source_name):
            errors.append(f"missing source constant: {spec.name} -> {spec.source_name}")
            continue
        if spec.minimum > spec.maximum:
            errors.append(f"invalid bounds: {spec.name}")
        if spec.step <= 0:
            errors.append(f"non-positive step: {spec.name}")
        if not (spec.minimum <= spec.default <= spec.maximum):
            errors.append(
                f"default outside bounds: {spec.name}={spec.default} "
                f"not in [{spec.minimum}, {spec.maximum}]"
            )
        if spec.kind not in ("continuous", "discrete"):
            errors.append(f"invalid kind: {spec.name}={spec.kind!r}")
        if spec.runtime_attr is not None:
            if not hasattr(C, spec.runtime_attr):
                errors.append(f"missing runtime index: {spec.name} -> {spec.runtime_attr}")
            else:
                idx = spec.runtime_index
                if idx in seen_runtime:
                    errors.append(f"duplicate runtime slot: {spec.name} -> {idx}")
                seen_runtime.add(idx)
                if idx < 0 or idx >= int(C.TUNE_SIZE):
                    errors.append(f"runtime slot out of range: {spec.name} -> {idx}")
    if len(SEARCH_PARAM_SPECS) <= int(C.TUNE_SIZE):
        errors.append(
            f"registry must exceed runtime slots: {len(SEARCH_PARAM_SPECS)} <= {int(C.TUNE_SIZE)}"
        )
    return tuple(errors)


def get_spec(name: str) -> SearchParamSpec:
    try:
        return SEARCH_PARAM_BY_NAME[name]
    except KeyError as exc:
        raise KeyError(f"unknown search parameter {name!r}") from exc


def profile_names(profile: str = DEFAULT_PROFILE) -> Tuple[str, ...]:
    key = str(profile).strip().lower()
    if key == "runtime":
        return tuple(spec.name for spec in SEARCH_PARAM_SPECS if spec.is_runtime)
    if key == "runtime_continuous":
        return tuple(
            spec.name
            for spec in SEARCH_PARAM_SPECS
            if spec.is_runtime and spec.name not in CATEGORICAL_PARAMS
        )
    if key == "all":
        return tuple(spec.name for spec in SEARCH_PARAM_SPECS)
    if key == "lmr_runtime":
        return tuple(
            spec.name
            for spec in SEARCH_PARAM_SPECS
            if spec.group == "lmr" and spec.is_runtime
        )
    if key == "lmr_compile":
        return tuple(
            spec.name
            for spec in SEARCH_PARAM_SPECS
            if spec.group == "lmr" and not spec.is_runtime
        )
    if key not in PROFILE_NAMES:
        raise ValueError(f"unknown search profile {profile!r}; choose from {PROFILE_NAMES}")
    return tuple(spec.name for spec in SEARCH_PARAM_SPECS if spec.group == key)


def resolve_names(
    profile: str = DEFAULT_PROFILE,
    names: Optional[Iterable[str]] = None,
) -> Tuple[str, ...]:
    """Resolve a profile or an explicit comma-separated/name iterable."""
    if names is None:
        selected = list(profile_names(profile))
    else:
        selected = []
        for raw in names:
            for name in str(raw).split(","):
                name = name.strip()
                if name:
                    selected.append(name)
    unknown = [name for name in selected if name not in SEARCH_PARAM_BY_NAME]
    if unknown:
        raise ValueError(f"unknown search parameter(s): {', '.join(unknown)}")
    # Preserve user order while rejecting accidental duplicate SPSA axes.
    if len(set(selected)) != len(selected):
        raise ValueError("duplicate search parameter in selection")
    return tuple(selected)


def get_default_params(names: Optional[Sequence[str]] = None) -> Dict[str, int]:
    selected = resolve_names(names=names) if names is not None else profile_names("all")
    return {name: get_spec(name).default for name in selected}


def legacy_search_params() -> Dict[str, Tuple[int, int, int, int]]:
    """Return all registry entries in the old ``SEARCH_PARAMS`` shape."""
    return {spec.name: spec.legacy_tuple() for spec in SEARCH_PARAM_SPECS}


def get_bounds(name: str) -> Tuple[int, int]:
    spec = get_spec(name)
    return spec.minimum, spec.maximum


def get_step(name: str) -> int:
    return get_spec(name).step


def quantize_value(name: str, value: float) -> int:
    """Clip and snap a value to the registry step (including discrete gates)."""
    spec = get_spec(name)
    # Preserve explicit bounds even when (max - min) is not a multiple of the
    # step (e.g. 0..1800 with a 64-unit LMR step).  The source default is the
    # quantization origin: several real engine defaults (460 with step 25,
    # 125 with step 10, ...) are intentionally not aligned to ``minimum`` and
    # must not be silently changed before the first match.
    if value <= spec.minimum:
        return int(spec.minimum)
    if value >= spec.maximum:
        return int(spec.maximum)
    value = max(float(spec.minimum), min(float(spec.maximum), float(value)))
    steps = round((value - spec.default) / spec.step)
    snapped = spec.default + steps * spec.step
    return int(max(spec.minimum, min(spec.maximum, snapped)))


def quantize_integer_value(name: str, value: float) -> int:
    """Clip a candidate to the engine bounds on a one-unit integer grid.

    Registry ``step`` values are deliberately coarse probe radii for the
    original SPSA campaign (25/128 units for several LMR axes).  The fine
    integer tuner still needs to pass values that the live ``tune[]`` slots can
    represent between those coarse probes.  This helper keeps the same safety
    bounds while removing the registry-step snapping from the applied value.
    """
    spec = get_spec(name)
    if value <= spec.minimum:
        return int(spec.minimum)
    if value >= spec.maximum:
        return int(spec.maximum)
    return int(max(spec.minimum, min(spec.maximum, round(float(value)))))


_REGISTRY_ERRORS = validate_registry()
if _REGISTRY_ERRORS:
    raise RuntimeError("invalid search parameter registry:\n- " + "\n- ".join(_REGISTRY_ERRORS))
