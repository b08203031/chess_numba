"""
Match-SPSA parameter blocks and specifications for classical HCE.

Defaults are read dynamically from ``chess_engine.classical.constants`` via
``tune_eval_match.eval_weights.default_eval_weights()``, never stale hardcoded
snapshots.  Each parameter carries tuning metadata: ``(fallback, min, max,
step_c, spsa_r_end)``.
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

from tune_eval_match.eval_weights import (
    EVAL_PARAM_INDEX,
    default_eval_weights,
    params_from_weights,
)

ParamSpec = Tuple[float, float, float, float, float]  # fallback, lo, hi, c, r_end

# ---------------------------------------------------------------------------
# Master parameter specification registry (covers all EW indices 0..98)
# ---------------------------------------------------------------------------
ALL_PARAM_SPECS: Dict[str, ParamSpec] = {
    # --- G4: Tempo & King Danger linear ---
    "TEMPO_BONUS": (19, 0, 50, 2, 2.0),
    "SAFE_CHECK_QUEEN": (356, 100, 900, 25, 1.5),
    "SAFE_CHECK_ROOK": (451, 100, 950, 25, 1.5),
    "SAFE_CHECK_BISHOP": (358, 80, 800, 22, 1.5),
    "SAFE_CHECK_KNIGHT": (463, 100, 950, 25, 1.5),
    "KING_DANGER_WEAK_SQ": (136, 20, 350, 12, 1.5),
    "KING_DANGER_UNSAFE_CHECK": (80, 5, 250, 10, 1.5),
    "KING_DANGER_BLOCKERS": (43, 0, 160, 6, 1.5),
    "KING_DANGER_ATTACK_ON_KING_SQ": (51, 0, 160, 6, 1.5),
    "KING_DANGER_NO_QUEEN": (539, 100, 1200, 30, 1.5),
    "KING_DANGER_NO_QUEEN_ROOKLESS": (867, 150, 1600, 35, 1.5),
    "KING_DANGER_KNIGHT_DEF": (79, 0, 250, 10, 1.5),
    "KING_DANGER_OFFSET": (0, -180, 180, 12, 1.5),
    "KING_DANGER_THRESHOLD": (66, 0, 220, 8, 1.5),
    # --- Block A: King Flank / Shelter Feedback / Danger Output ---
    "KING_FLANK_ATTACK_NUM": (3, 1, 12, 1, 2.0),
    "KING_FLANK_DEFENSE_MULT": (4, 0, 16, 1, 2.0),
    "KING_SHELTER_FEEDBACK_NUM": (6, 1, 16, 1, 2.0),
    "PAWNLESS_FLANK_MG": (-12, -80, 20, 3, 1.5),
    "PAWNLESS_FLANK_EG": (-56, -150, 20, 5, 1.5),
    "FLANK_ATTACKS_MG": (-2, -30, 15, 1, 2.0),
    "FLANK_ATTACKS_EG": (0, -30, 20, 2, 1.5),
    "KING_DANGER_OUT_MG_NUM": (120, 40, 400, 8, 1.5),
    "KING_DANGER_OUT_EG_NUM": (17076, 4000, 45000, 800, 1.5),
    "EG_SAFETY_SCALE": (110, 20, 220, 8, 1.5),
    "SHELTER_BASE_MG": (20, -40, 80, 4, 1.5),
    "SHELTER_BASE_EG": (9, -40, 60, 3, 1.5),
    "BLOCKED_STORM": (-42, -150, 10, 6, 1.5),
    "BLOCKED_STORM_EG": (-31, -120, 20, 5, 1.5),
    # --- King Attack Weights ---
    "KING_ATTACK_WEIGHT_N": (56, 5, 150, 5, 1.5),
    "KING_ATTACK_WEIGHT_B": (19, 0, 120, 5, 1.5),
    "KING_ATTACK_WEIGHT_R": (29, 0, 120, 5, 1.5),
    "KING_ATTACK_WEIGHT_Q": (20, 0, 120, 5, 1.5),
    # --- Space ---
    "SPACE_THRESHOLD": (6000, 1000, 14000, 400, 1.5),
    "SPACE_BONUS_DIVISOR": (15, 4, 40, 2, 2.0),
    "SPACE_SCALE_DIVISOR": (6, 2, 20, 1, 2.0),
    # --- Threats (MG) ---
    "THREAT_SAFE_PAWN_MG": (58, 5, 140, 5, 1.5),
    "THREAT_KNIGHT_ON_QUEEN_MG": (8, 0, 60, 3, 1.5),
    "THREAT_SLIDER_ON_QUEEN_MG": (15, 0, 80, 3, 1.5),
    "THREAT_BY_MINOR_P_MG": (6, 0, 50, 3, 1.5),
    "THREAT_BY_MINOR_N_MG": (24, 0, 90, 4, 1.5),
    "THREAT_BY_MINOR_B_MG": (42, 0, 110, 4, 1.5),
    "THREAT_BY_MINOR_R_MG": (51, 0, 130, 5, 1.5),
    "THREAT_BY_MINOR_Q_MG": (43, 0, 140, 5, 1.5),
    "THREAT_BY_ROOK_P_MG": (4, 0, 50, 3, 1.5),
    "THREAT_BY_ROOK_N_MG": (22, 0, 90, 4, 1.5),
    "THREAT_BY_ROOK_B_MG": (25, 0, 90, 4, 1.5),
    "THREAT_BY_ROOK_R_MG": (1, 0, 80, 3, 1.5),
    "THREAT_BY_ROOK_Q_MG": (38, 0, 120, 4, 1.5),
    "THREAT_BY_KING_MG": (18, 0, 70, 3, 1.5),
    "THREAT_HANGING_MG": (4, 0, 80, 4, 1.5),
    "THREAT_RESTRICTED_PIECE_MG": (0, -20, 40, 2, 1.5),
    "THREAT_PAWN_PUSH_MG": (11, 0, 60, 3, 1.5),
    # --- Threats (EG) ---
    "THREAT_SAFE_PAWN_EG": (41, 0, 110, 4, 1.5),
    "THREAT_KNIGHT_ON_QUEEN_EG": (0, 0, 50, 3, 1.5),
    "THREAT_SLIDER_ON_QUEEN_EG": (0, 0, 50, 3, 1.5),
    "THREAT_BY_MINOR_P_EG": (15, 0, 60, 3, 1.5),
    "THREAT_BY_MINOR_N_EG": (27, 0, 90, 4, 1.5),
    "THREAT_BY_MINOR_B_EG": (29, 0, 90, 4, 1.5),
    "THREAT_BY_MINOR_R_EG": (41, 0, 110, 4, 1.5),
    "THREAT_BY_MINOR_Q_EG": (73, 0, 160, 6, 1.5),
    "THREAT_BY_ROOK_P_EG": (22, 0, 70, 3, 1.5),
    "THREAT_BY_ROOK_N_EG": (28, 0, 90, 4, 1.5),
    "THREAT_BY_ROOK_B_EG": (25, 0, 90, 4, 1.5),
    "THREAT_BY_ROOK_R_EG": (20, 0, 80, 4, 1.5),
    "THREAT_BY_ROOK_Q_EG": (22, 0, 100, 4, 1.5),
    "THREAT_BY_KING_EG": (33, 0, 90, 4, 1.5),
    "THREAT_HANGING_EG": (17, 0, 80, 4, 1.5),
    "THREAT_RESTRICTED_PIECE_EG": (0, -20, 40, 2, 1.5),
    "THREAT_PAWN_PUSH_EG": (14, 0, 60, 3, 1.5),
    # --- Passed scalars ---
    "PASSED_PATH_SAFE_NONE_ATTACK": (10, 2, 40, 2, 2.0),
    "PASSED_PATH_SAFE_EDGE_ATTACK": (6, 1, 24, 1, 2.0),
    "PASSED_PATH_SAFE_BLOCK_ONLY": (2, 0, 16, 1, 2.0),
    "PASSED_PATH_SUPPORT_BONUS": (3, 0, 14, 1, 2.0),
    "PASSED_DYNAMICS_MULT": (4, 1, 16, 1, 2.0),
    "PASSED_DYNAMICS_OFFSET": (-10, -40, 10, 2, 2.0),
    "KING_PROX_ENEMY_MULT": (8, 1, 24, 1, 2.0),
    "KING_PROX_FRIENDLY_MULT": (1, 0, 8, 1, 2.5),
    "PASSED_FILE_BONUS_MG": (11, 0, 50, 2, 2.0),
    "PASSED_FILE_BONUS_EG": (6, 0, 24, 1, 2.0),
    "UNSTOPPABLE_PAWN_BONUS": (800, 400, 1200, 40, 1.5),
    # --- Material & Imbalance ---
    "MAT_N_MG": (353, 280, 420, 8, 1.5),
    "MAT_N_EG": (419, 330, 500, 10, 1.5),
    "MAT_B_MG": (361, 300, 450, 8, 1.5),
    "MAT_B_EG": (439, 350, 520, 10, 1.5),
    "MAT_R_MG": (567, 480, 680, 12, 1.5),
    "MAT_R_EG": (702, 560, 820, 14, 1.5),
    "MAT_Q_MG": (1180, 1000, 1380, 20, 1.5),
    "MAT_Q_EG": (1317, 1100, 1520, 22, 1.5),
    "IMBALANCE_DIVISOR": (20, 8, 45, 2, 2.0),
    "IMBALANCE_SCALE_MG": (53, 15, 120, 4, 1.5),
    "IMBALANCE_SCALE_EG": (79, 25, 150, 5, 1.5),
    # --- Passed rank table (ranks 3..6) ---
    "PASSED_RANK3_MG": (0, -10, 40, 2, 2.0),
    "PASSED_RANK3_EG": (36, 0, 75, 3, 2.0),
    "PASSED_RANK4_MG": (30, 5, 80, 3, 2.0),
    "PASSED_RANK4_EG": (52, 5, 110, 4, 2.0),
    "PASSED_RANK5_MG": (66, 20, 160, 6, 2.0),
    "PASSED_RANK5_EG": (102, 20, 180, 6, 2.0),
    "PASSED_RANK6_MG": (126, 40, 250, 10, 2.0),
    "PASSED_RANK6_EG": (95, 30, 220, 8, 2.0),
}


def _block(names: Sequence[str]) -> Dict[str, ParamSpec]:
    return {name: ALL_PARAM_SPECS[name] for name in names}


BLOCK_THREATS_MG = _block([
    "THREAT_SAFE_PAWN_MG",
    "THREAT_KNIGHT_ON_QUEEN_MG",
    "THREAT_SLIDER_ON_QUEEN_MG",
    "THREAT_BY_MINOR_P_MG",
    "THREAT_BY_MINOR_N_MG",
    "THREAT_BY_MINOR_B_MG",
    "THREAT_BY_MINOR_R_MG",
    "THREAT_BY_MINOR_Q_MG",
    "THREAT_BY_ROOK_P_MG",
    "THREAT_BY_ROOK_N_MG",
    "THREAT_BY_ROOK_B_MG",
    "THREAT_BY_ROOK_R_MG",
    "THREAT_BY_ROOK_Q_MG",
    "THREAT_BY_KING_MG",
    "THREAT_HANGING_MG",
    "THREAT_RESTRICTED_PIECE_MG",
    "THREAT_PAWN_PUSH_MG",
])

BLOCK_THREATS_EG = _block([
    "THREAT_SAFE_PAWN_EG",
    "THREAT_KNIGHT_ON_QUEEN_EG",
    "THREAT_SLIDER_ON_QUEEN_EG",
    "THREAT_BY_MINOR_P_EG",
    "THREAT_BY_MINOR_N_EG",
    "THREAT_BY_MINOR_B_EG",
    "THREAT_BY_MINOR_R_EG",
    "THREAT_BY_MINOR_Q_EG",
    "THREAT_BY_ROOK_P_EG",
    "THREAT_BY_ROOK_N_EG",
    "THREAT_BY_ROOK_B_EG",
    "THREAT_BY_ROOK_R_EG",
    "THREAT_BY_ROOK_Q_EG",
    "THREAT_BY_KING_EG",
    "THREAT_HANGING_EG",
    "THREAT_RESTRICTED_PIECE_EG",
    "THREAT_PAWN_PUSH_EG",
])

BLOCK_THREATS = {**BLOCK_THREATS_MG, **BLOCK_THREATS_EG}

BLOCK_G4 = _block([
    "TEMPO_BONUS",
    "SAFE_CHECK_QUEEN",
    "SAFE_CHECK_ROOK",
    "SAFE_CHECK_BISHOP",
    "SAFE_CHECK_KNIGHT",
    "KING_DANGER_WEAK_SQ",
    "KING_DANGER_UNSAFE_CHECK",
    "KING_DANGER_BLOCKERS",
    "KING_DANGER_ATTACK_ON_KING_SQ",
    "KING_DANGER_NO_QUEEN",
    "KING_DANGER_NO_QUEEN_ROOKLESS",
    "KING_DANGER_KNIGHT_DEF",
    "KING_DANGER_OFFSET",
    "KING_DANGER_THRESHOLD",
])

BLOCK_A = _block([
    "KING_FLANK_ATTACK_NUM",
    "KING_FLANK_DEFENSE_MULT",
    "KING_SHELTER_FEEDBACK_NUM",
    "PAWNLESS_FLANK_MG",
    "PAWNLESS_FLANK_EG",
    "FLANK_ATTACKS_MG",
    "FLANK_ATTACKS_EG",
    "KING_DANGER_OUT_MG_NUM",
    "KING_DANGER_OUT_EG_NUM",
    "EG_SAFETY_SCALE",
    "SHELTER_BASE_MG",
    "SHELTER_BASE_EG",
    "BLOCKED_STORM",
    "BLOCKED_STORM_EG",
])

BLOCK_KING_WEIGHTS = _block([
    "KING_ATTACK_WEIGHT_N",
    "KING_ATTACK_WEIGHT_B",
    "KING_ATTACK_WEIGHT_R",
    "KING_ATTACK_WEIGHT_Q",
    "KING_DANGER_NO_QUEEN",
    "KING_DANGER_NO_QUEEN_ROOKLESS",
    "KING_DANGER_OUT_MG_NUM",
    "KING_DANGER_OUT_EG_NUM",
    "KING_DANGER_THRESHOLD",
    "EG_SAFETY_SCALE",
    "KING_FLANK_ATTACK_NUM",
])

BLOCK_SPACE = _block([
    "SPACE_THRESHOLD",
    "SPACE_BONUS_DIVISOR",
    "SPACE_SCALE_DIVISOR",
])

BLOCK_PASSED = _block([
    "PASSED_PATH_SAFE_NONE_ATTACK",
    "PASSED_PATH_SAFE_EDGE_ATTACK",
    "PASSED_PATH_SAFE_BLOCK_ONLY",
    "PASSED_PATH_SUPPORT_BONUS",
    "PASSED_DYNAMICS_MULT",
    "PASSED_DYNAMICS_OFFSET",
    "KING_PROX_ENEMY_MULT",
    "KING_PROX_FRIENDLY_MULT",
    "PASSED_FILE_BONUS_MG",
    "PASSED_FILE_BONUS_EG",
])

BLOCK_MATERIAL = _block([
    "MAT_N_MG",
    "MAT_N_EG",
    "MAT_B_MG",
    "MAT_B_EG",
    "MAT_R_MG",
    "MAT_R_EG",
    "MAT_Q_MG",
    "MAT_Q_EG",
    "IMBALANCE_SCALE_MG",
    "IMBALANCE_SCALE_EG",
])

BLOCK_IMBALANCE = _block([
    "IMBALANCE_DIVISOR",
    "IMBALANCE_SCALE_MG",
    "IMBALANCE_SCALE_EG",
])

BLOCK_PASSED_RANK = _block([
    "PASSED_RANK3_MG",
    "PASSED_RANK3_EG",
    "PASSED_RANK4_MG",
    "PASSED_RANK4_EG",
    "PASSED_RANK5_MG",
    "PASSED_RANK5_EG",
    "PASSED_RANK6_MG",
    "PASSED_RANK6_EG",
])

BLOCKS: Dict[str, Dict[str, ParamSpec]] = {
    "threats_mg": BLOCK_THREATS_MG,
    "threats_eg": BLOCK_THREATS_EG,
    "threats": BLOCK_THREATS,
    "king_weights": BLOCK_KING_WEIGHTS,
    "g4": BLOCK_G4,
    "a": BLOCK_A,
    "space": BLOCK_SPACE,
    "passed": BLOCK_PASSED,
    "passed_rank": BLOCK_PASSED_RANK,
    "material": BLOCK_MATERIAL,
    "imbalance": BLOCK_IMBALANCE,
}

BLOCK_ORDER = (
    "threats_mg",
    "threats_eg",
    "king_weights",
    "g4",
    "a",
    "space",
    "passed",
    "imbalance",
    "passed_rank",
    "material",
)

# Active block for SPSA (mutated by set_active_block / select_custom_params)
_ACTIVE_BLOCK = "threats_mg"
EVAL_MATCH_PARAMS: Dict[str, ParamSpec] = BLOCK_THREATS_MG


def set_active_block(name: str) -> str:
    """Select SPSA parameter dict. Returns canonical block name."""
    global _ACTIVE_BLOCK, EVAL_MATCH_PARAMS
    key = name.strip().lower()
    if key not in BLOCKS:
        raise ValueError(f"Unknown block {name!r}; choose from {sorted(BLOCKS)}")
    _ACTIVE_BLOCK = key
    EVAL_MATCH_PARAMS = BLOCKS[key]
    return key


def select_custom_params(names: Sequence[str]) -> str:
    """Select an explicit list of parameters from EVAL_PARAM_INDEX."""
    global _ACTIVE_BLOCK, EVAL_MATCH_PARAMS
    cleaned = [n.strip() for n in names if n.strip()]
    if not cleaned:
        raise ValueError("Custom --params list cannot be empty")
    unknown = [n for n in cleaned if n not in ALL_PARAM_SPECS or n not in EVAL_PARAM_INDEX]
    if unknown:
        raise ValueError(f"Unknown eval params: {unknown}")
    deduped = list(dict.fromkeys(cleaned))
    _ACTIVE_BLOCK = "custom"
    EVAL_MATCH_PARAMS = {n: ALL_PARAM_SPECS[n] for n in deduped}
    return _ACTIVE_BLOCK


def active_block() -> str:
    return _ACTIVE_BLOCK


def param_names():
    return list(EVAL_MATCH_PARAMS.keys())


def default_params(constants_module=None) -> dict:
    """Load live defaults for the active block from constants_module (current by default)."""
    names = param_names()
    live = params_from_weights(default_eval_weights(constants_module), names)
    return {k: float(live[k]) for k in names}


def bounds(name: str):
    _d, lo, hi, _c, _r = EVAL_MATCH_PARAMS[name]
    return float(lo), float(hi)


def step_c(name: str) -> float:
    return float(EVAL_MATCH_PARAMS[name][3])


def r_end(name: str) -> float:
    return float(EVAL_MATCH_PARAMS[name][4])


def clip_params_float(params: dict) -> dict:
    live = default_params()
    out = {}
    for k in EVAL_MATCH_PARAMS:
        if k not in params:
            out[k] = float(live[k])
            continue
        lo, hi = bounds(k)
        x = float(params[k])
        if x < lo:
            x = lo
        elif x > hi:
            x = hi
        out[k] = x
    return out


def quantize_params(params: dict) -> dict:
    live = default_params()
    out = {}
    for k in EVAL_MATCH_PARAMS:
        lo, hi = bounds(k)
        v = params[k] if k in params else live[k]
        iv = int(round(float(v)))
        out[k] = int(max(lo, min(hi, iv)))
    return out


def clip_params(params: dict) -> dict:
    return quantize_params(params)
