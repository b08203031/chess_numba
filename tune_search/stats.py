"""Pure-Python match statistics for search-parameter tuning (no Numba / no engine).

The in-process matcher plays opening *pairs* (colour swapped), so the correct
independent sample is the pair, not the game.  ``penta`` below is the
pentanomial histogram ``[0, 0.5, 1, 1.5, 2]`` of pair points for the tested
side, as returned by ``tune_eval_match.inprocess_match.run_match``.

Provides Elo / confidence interval, a normal-approximation SPSA/SPRT LLR
(Fishtest-style pentanomial), and helpers used by ``summarize_run.py``.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Sequence, Tuple

# Score of one pair, normalised to a per-game score in [0, 1].
_PAIR_SCORE = (0.0, 0.25, 0.5, 0.75, 1.0)


def score_to_elo(score: float) -> float:
    score = min(max(float(score), 1e-9), 1.0 - 1e-9)
    return -400.0 * math.log10(1.0 / score - 1.0)


def elo_to_score(elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-float(elo) / 400.0))


def penta_moments(penta: Sequence[int]) -> Tuple[int, float, float]:
    """Return ``(pairs, mean_score, variance_of_pair_score)``."""
    if len(penta) != 5:
        raise ValueError("penta must have 5 bins")
    n = int(sum(penta))
    if n <= 0:
        return 0, 0.5, 0.0
    mean = sum(c * s for c, s in zip(penta, _PAIR_SCORE)) / n
    var = sum(c * (s - mean) ** 2 for c, s in zip(penta, _PAIR_SCORE)) / n
    return n, mean, var


def elo_estimate(penta: Sequence[int], z: float = 1.96) -> Dict[str, float]:
    """Elo with a delta-method confidence interval from pentanomial variance."""
    n, mean, var = penta_moments(penta)
    if n == 0:
        return {"pairs": 0, "games": 0, "score": 0.5, "elo": 0.0, "elo_err": float("inf")}
    mean_c = min(max(mean, 1e-9), 1.0 - 1e-9)
    stderr = math.sqrt(var / n) if n > 0 else float("inf")
    d_elo = 400.0 / math.log(10.0) / (mean_c * (1.0 - mean_c))
    return {
        "pairs": n,
        "games": 2 * n,
        "score": mean,
        "elo": score_to_elo(mean),
        "elo_err": z * stderr * d_elo,
    }


def sprt_bounds(alpha: float = 0.05, beta: float = 0.05) -> Tuple[float, float]:
    """Return ``(lower, upper)`` LLR acceptance bounds."""
    return math.log(beta / (1.0 - alpha)), math.log((1.0 - beta) / alpha)


def sprt_llr(penta: Sequence[int], elo0: float, elo1: float) -> float:
    """Normal-approximation LLR of H1 (elo1) vs H0 (elo0) from pairs.

    ``LLR = N * (s1 - s0) * (2*mean - s0 - s1) / (2 * var)`` where ``N`` is the
    number of pairs and ``var`` the variance of a pair's per-game score; this
    is the standard pentanomial approximation (no draw-ratio assumption).
    """
    n, mean, var = penta_moments(penta)
    if n == 0 or var <= 0.0:
        return 0.0
    s0 = elo_to_score(elo0)
    s1 = elo_to_score(elo1)
    return n * (s1 - s0) * (2.0 * mean - s0 - s1) / (2.0 * var)


def sprt_decision(llr: float, alpha: float = 0.05, beta: float = 0.05) -> str:
    lower, upper = sprt_bounds(alpha, beta)
    if llr >= upper:
        return "accept_h1"
    if llr <= lower:
        return "accept_h0"
    return "continue"


def merge_penta(parts: Iterable[Sequence[int]]) -> list:
    out = [0, 0, 0, 0, 0]
    for part in parts:
        for i in range(5):
            out[i] += int(part[i])
    return out


def drift_z(raw_updates: Sequence[float]) -> float:
    """z-score of SPSA net drift against a zero-mean random walk.

    ``z = sum(u) / sqrt(sum(u^2))``.  ``|z| < 2`` means the parameter's total
    movement is indistinguishable from noise (no evidence of a gradient).
    """
    denom = math.sqrt(sum(u * u for u in raw_updates))
    if denom <= 0.0:
        return 0.0
    return sum(raw_updates) / denom


# Cutechess-like defaults: a result is declared only when *consecutive* plies
# (i.e. both engines' independent searches) agree, which makes a wrong call
# very unlikely at the cp margins below.
DEFAULT_ADJUDICATION = {
    "win_cp": 800,        # |score| >= this ...
    "win_plies": 6,       # ... for this many consecutive plies -> decisive
    "draw_cp": 10,        # |score| <= this ...
    "draw_plies": 10,     # ... for this many consecutive plies ...
    "draw_min_ply": 60,   # ... once the game is at least this long -> draw
}


class _Adjudicator:
    """Tracks consecutive-ply score streaks (white POV) for one game."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = {**DEFAULT_ADJUDICATION, **(cfg or {})}
        self._sign = 0
        self._win_run = 0
        self._draw_run = 0

    def update(self, ply: int, white_score: int) -> float | None:
        cfg = self.cfg
        sign = 1 if white_score >= cfg["win_cp"] else (-1 if white_score <= -cfg["win_cp"] else 0)
        if sign != 0 and sign == self._sign:
            self._win_run += 1
        else:
            self._win_run = 1 if sign != 0 else 0
        self._sign = sign
        if self._win_run >= cfg["win_plies"]:
            return 1.0 if sign > 0 else 0.0

        if ply >= cfg["draw_min_ply"] and abs(white_score) <= cfg["draw_cp"]:
            self._draw_run += 1
        else:
            self._draw_run = 0
        if self._draw_run >= cfg["draw_plies"]:
            return 0.5
        return None

