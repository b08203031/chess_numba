# chess_engine/classical/material.py

import numba
import numpy as np

from chess_engine.classical_old.constants import (
    IMBALANCE_QUADRATIC_OURS, IMBALANCE_QUADRATIC_THEIRS, IMBALANCE_DIVISOR,
    IMBALANCE_SCALE_MG, IMBALANCE_SCALE_EG
)

# Must match tune_eval_match.eval_weights
_EW_IMB_SCALE_MG = 44
_EW_IMB_SCALE_EG = 45


@numba.njit(cache=True, inline="always")
def _ew_get_mat(ew, use_ew, idx, default):
    if use_ew:
        return ew[idx]
    return np.int32(default)


@numba.njit(cache=True, boundscheck=False, fastmath=True)
def compute_imbalance(cnt_0, cnt_1, cnt_2, cnt_3, cnt_4, cnt_6, cnt_7, cnt_8, cnt_9, cnt_10,
                      ew, use_ew):
    """
    Compute polynomial material imbalance (SF11 material.cpp style).
    Scale MG/EG are runtime-tunable via ew (match-SPSA). Quadratic tables fixed.
    """
    scale_mg = _ew_get_mat(ew, use_ew, _EW_IMB_SCALE_MG, IMBALANCE_SCALE_MG)
    scale_eg = _ew_get_mat(ew, use_ew, _EW_IMB_SCALE_EG, IMBALANCE_SCALE_EG)

    w_bp = np.int32(1) if cnt_2 > 1 else np.int32(0)
    b_bp = np.int32(1) if cnt_8 > 1 else np.int32(0)

    w = np.array([w_bp, np.int32(cnt_0), np.int32(cnt_1), np.int32(cnt_2),
                  np.int32(cnt_3), np.int32(cnt_4)], dtype=np.int32)
    b = np.array([b_bp, np.int32(cnt_6), np.int32(cnt_7), np.int32(cnt_8),
                  np.int32(cnt_9), np.int32(cnt_10)], dtype=np.int32)

    w_bonus = np.int32(0)
    for pt1 in range(6):
        if w[pt1] == 0:
            continue
        v = np.int32(0)
        for pt2 in range(pt1 + 1):
            v += IMBALANCE_QUADRATIC_OURS[pt1, pt2] * w[pt2]
            v += IMBALANCE_QUADRATIC_THEIRS[pt1, pt2] * b[pt2]
        w_bonus += w[pt1] * v

    b_bonus = np.int32(0)
    for pt1 in range(6):
        if b[pt1] == 0:
            continue
        v = np.int32(0)
        for pt2 in range(pt1 + 1):
            v += IMBALANCE_QUADRATIC_OURS[pt1, pt2] * b[pt2]
            v += IMBALANCE_QUADRATIC_THEIRS[pt1, pt2] * w[pt2]
        b_bonus += b[pt1] * v

    raw = (w_bonus - b_bonus) // IMBALANCE_DIVISOR
    mg_imbalance = raw * scale_mg // np.int32(100)
    eg_imbalance = raw * scale_eg // np.int32(100)
    return mg_imbalance, eg_imbalance
