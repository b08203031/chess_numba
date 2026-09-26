"""
Match-SPSA parameter blocks for classical HCE.

Blocks (run in order after G4 done):
  a         — king-safety neighbours (current)
  passed    — 10 passed-pawn scalars (after A sync)
  material  — 8 N/B/R/Q MG+EG + 2 imbalance scales (after passed)

Format: name -> (default, min, max, step_c, spsa_r_end)
  a_end = r_end * c^2 ; keep θ float; quantize on apply / L2.
"""
from __future__ import annotations

from typing import Dict, Tuple

ParamSpec = Tuple[float, float, float, float, float]  # default, lo, hi, c, r_end

# ---------------------------------------------------------------------------
# Block A (active while king-safety match runs)
# ---------------------------------------------------------------------------
BLOCK_A: Dict[str, ParamSpec] = {
    "KING_FLANK_ATTACK_NUM": (3, 1, 12, 1, 2.0),
    "KING_FLANK_DEFENSE_MULT": (4, 0, 16, 1, 2.0),
    "KING_SHELTER_FEEDBACK_NUM": (6, 1, 16, 1, 2.0),
    "PAWNLESS_FLANK_MG": (-14, -80, 20, 3, 1.5),
    "PAWNLESS_FLANK_EG": (-51, -150, 20, 5, 1.5),
    "FLANK_ATTACKS_MG": (-2, -30, 15, 1, 2.0),
    "KING_DANGER_OUT_MG_NUM": (128, 40, 400, 10, 1.5),
    "EG_SAFETY_SCALE": (100, 20, 200, 8, 1.5),
    "SHELTER_BASE_MG": (18, -40, 80, 4, 1.5),
    "SHELTER_BASE_EG": (6, -40, 60, 3, 1.5),
    "BLOCKED_STORM": (-63, -150, 0, 6, 1.5),
    "BLOCKED_STORM_EG": (-42, -120, 20, 5, 1.5),
}

# ---------------------------------------------------------------------------
# Block Passed — 10 scalars (CANDIDATE_DIV frozen=2; KING_PROX_ENEMY_DIV frozen=4)
# ---------------------------------------------------------------------------
BLOCK_PASSED: Dict[str, ParamSpec] = {
    "PASSED_PATH_SAFE_NONE_ATTACK": (16, 4, 40, 2, 2.0),
    "PASSED_PATH_SAFE_EDGE_ATTACK": (8, 1, 24, 1, 2.0),
    "PASSED_PATH_SAFE_BLOCK_ONLY": (4, 1, 16, 1, 2.0),
    "PASSED_PATH_SUPPORT_BONUS": (3, 0, 12, 1, 2.0),
    "PASSED_DYNAMICS_MULT": (5, 1, 16, 1, 2.0),
    "PASSED_DYNAMICS_OFFSET": (-13, -40, 10, 2, 2.0),
    "KING_PROX_ENEMY_MULT": (9, 2, 24, 1, 2.0),
    "KING_PROX_FRIENDLY_MULT": (1, 0, 6, 1, 2.5),
    "PASSED_FILE_BONUS_MG": (19, 0, 50, 2, 2.0),
    "PASSED_FILE_BONUS_EG": (5, 0, 20, 1, 2.0),
}

# ---------------------------------------------------------------------------
# Block Material — pawn frozen; N/B/R/Q MG+EG + imbalance scales
# ---------------------------------------------------------------------------
BLOCK_MATERIAL: Dict[str, ParamSpec] = {
    "MAT_N_MG": (347, 280, 420, 8, 1.5),
    "MAT_N_EG": (410, 330, 500, 10, 1.5),
    "MAT_B_MG": (373, 300, 450, 8, 1.5),
    "MAT_B_EG": (430, 350, 520, 10, 1.5),
    "MAT_R_MG": (578, 480, 680, 12, 1.5),
    "MAT_R_EG": (678, 560, 800, 14, 1.5),
    "MAT_Q_MG": (1177, 1000, 1350, 20, 1.5),
    "MAT_Q_EG": (1266, 1100, 1450, 22, 1.5),
    "IMBALANCE_SCALE_MG": (74, 40, 120, 4, 1.5),
    "IMBALANCE_SCALE_EG": (53, 25, 90, 3, 1.5),
}

# ---------------------------------------------------------------------------
# Block Passed rank table — ranks 3..6 MG+EG (8 dims). Ranks 0,1,2,7 frozen.
# Defaults from constants PASSED_PAWN_BONUS[rank].
# ---------------------------------------------------------------------------
BLOCK_PASSED_RANK: Dict[str, ParamSpec] = {
    "PASSED_RANK3_MG": (2, 0, 40, 1, 2.0),
    "PASSED_RANK3_EG": (21, 0, 60, 2, 2.0),
    "PASSED_RANK4_MG": (31, 5, 80, 3, 2.0),
    "PASSED_RANK4_EG": (36, 5, 90, 3, 2.0),
    "PASSED_RANK5_MG": (78, 20, 160, 6, 2.0),
    "PASSED_RANK5_EG": (76, 20, 160, 6, 2.0),
    "PASSED_RANK6_MG": (133, 40, 250, 10, 2.0),
    "PASSED_RANK6_EG": (111, 30, 220, 8, 2.0),
}

BLOCKS: Dict[str, Dict[str, ParamSpec]] = {
    "a": BLOCK_A,
    "passed": BLOCK_PASSED,
    "material": BLOCK_MATERIAL,
    "passed_rank": BLOCK_PASSED_RANK,
}

BLOCK_ORDER = ("a", "passed", "passed_rank", "material")

# Active block for SPSA (mutated by set_active_block / --block)
_ACTIVE_BLOCK = "a"
EVAL_MATCH_PARAMS: Dict[str, ParamSpec] = BLOCK_A


def set_active_block(name: str) -> str:
    """Select SPSA parameter dict. Returns canonical block name."""
    global _ACTIVE_BLOCK, EVAL_MATCH_PARAMS
    key = name.strip().lower()
    if key not in BLOCKS:
        raise ValueError(f"Unknown block {name!r}; choose from {list(BLOCKS)}")
    _ACTIVE_BLOCK = key
    EVAL_MATCH_PARAMS = BLOCKS[key]
    return key


def active_block() -> str:
    return _ACTIVE_BLOCK


def param_names():
    return list(EVAL_MATCH_PARAMS.keys())


def default_params() -> dict:
    return {k: float(v[0]) for k, v in EVAL_MATCH_PARAMS.items()}


def bounds(name: str):
    _d, lo, hi, _c, _r = EVAL_MATCH_PARAMS[name]
    return float(lo), float(hi)


def step_c(name: str) -> float:
    return float(EVAL_MATCH_PARAMS[name][3])


def r_end(name: str) -> float:
    return float(EVAL_MATCH_PARAMS[name][4])


def clip_params_float(params: dict) -> dict:
    out = {}
    for k in EVAL_MATCH_PARAMS:
        if k not in params:
            out[k] = float(EVAL_MATCH_PARAMS[k][0])
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
    out = {}
    for k in EVAL_MATCH_PARAMS:
        lo, hi = bounds(k)
        v = params[k] if k in params else EVAL_MATCH_PARAMS[k][0]
        iv = int(round(float(v)))
        out[k] = int(max(lo, min(hi, iv)))
    return out


def clip_params(params: dict) -> dict:
    return quantize_params(params)
