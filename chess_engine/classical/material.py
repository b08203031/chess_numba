# chess_engine/classical/material.py

import numba
import numpy as np

from chess_engine.classical.constants import (
    IMBALANCE_QUADRATIC_OURS, IMBALANCE_QUADRATIC_THEIRS, IMBALANCE_DIVISOR,
    IMBALANCE_SCALE_MG, IMBALANCE_SCALE_EG
)

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def compute_imbalance(cnt_0, cnt_1, cnt_2, cnt_3, cnt_4, cnt_6, cnt_7, cnt_8, cnt_9, cnt_10):
    """
    Compute polynomial material imbalance (SF11 material.cpp style).
    Uses QuadraticOurs/QuadraticTheirs matrices to model cross-piece-type interactions.
    Index order: [BishopPair, Pawn, Knight, Bishop, Rook, Queen]
    
    Returns (mg_imbalance, eg_imbalance) from White's perspective.
    """
    # Pack into pieceCount format: [bishop_pair, pawn, knight, bishop, rook, queen]
    w_bp = np.int32(1) if cnt_2 > 1 else np.int32(0)
    b_bp = np.int32(1) if cnt_8 > 1 else np.int32(0)

    w = np.array([w_bp, np.int32(cnt_0), np.int32(cnt_1), np.int32(cnt_2),
                  np.int32(cnt_3), np.int32(cnt_4)], dtype=np.int32)
    b = np.array([b_bp, np.int32(cnt_6), np.int32(cnt_7), np.int32(cnt_8),
                  np.int32(cnt_9), np.int32(cnt_10)], dtype=np.int32)

    # White imbalance (Us=White, Them=Black)
    w_bonus = np.int32(0)
    for pt1 in range(6):
        if w[pt1] == 0:
            continue
        v = np.int32(0)
        for pt2 in range(pt1 + 1):
            v += IMBALANCE_QUADRATIC_OURS[pt1, pt2] * w[pt2]
            v += IMBALANCE_QUADRATIC_THEIRS[pt1, pt2] * b[pt2]
        w_bonus += w[pt1] * v

    # Black imbalance (Us=Black, Them=White)
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
    mg_imbalance = raw * IMBALANCE_SCALE_MG // np.int32(100)
    eg_imbalance = raw * IMBALANCE_SCALE_EG // np.int32(100)
    return mg_imbalance, eg_imbalance
