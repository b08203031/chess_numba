#!/usr/bin/env python3
"""Multi-model SPRT analyzer for tournament PGNs or WDL/pentanomial counts.

Computes multiple SPRT variants on ``tournament_results.pgn`` (or custom PGNs /
counts), including:
  - Pentanomial Normal Approximation (matches ``tools/match_core.py`` & ``tune_search/stats.py``)
  - Pentanomial Exact Constrained MLE (Fishtest / Fastchess style)
  - Pentanomial Normalized Elo (nElo) Normal Approximation
  - Trinomial (per-game WDL) Normal Approximation & Exact Constrained MLE
  - Chronological LLR trajectory, early-stop detection, and Brownian sample-size estimates

Usage:
  python tournament_analysis/calc_sprt.py
  python tournament_analysis/calc_sprt.py tournament_analysis/tournament_results.pgn --target New
  python tournament_analysis/calc_sprt.py --elo0 0 --elo1 5
  python tournament_analysis/calc_sprt.py --bounds "0:10,0:5,0:3,0:2,-3:1,-1.75:0.25"
  python tournament_analysis/calc_sprt.py --wdl 309 1186 281
  python tournament_analysis/calc_sprt.py --penta 29 168 471 186 34
  python tournament_analysis/calc_sprt.py --plot tournament_analysis/sprt_trajectory.png
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
DEFAULT_PGN = HERE / "tournament_results.pgn"

# Default SPRT hypothesis presets: (label, elo0, elo1)
DEFAULT_BOUNDS: List[Tuple[str, float, float]] = [
    ("新舊對戰", -3.0, 3.0),
    ("Tournament預設", 0.0, 10.0),
    ("中等增益", 0.0, 5.0),
    ("細微增益", 0.0, 3.0),
    ("Fishtest經典", 0.0, 2.0),
    ("Fishtest STC", 0.5, 2.5),
    ("非退步(寬鬆)", -3.0, 1.0),
    ("簡化測試(SF)", -1.75, 0.25),
    ("零退步底線", -5.0, 0.0),
    ("對稱顯著性", -2.0, 2.0),
]

PENTA_SCORES_NORM = (0.0, 0.25, 0.5, 0.75, 1.0)
TRINOMIAL_SCORES = (0.0, 0.5, 1.0)
PAIR_9_KEYS = ("WW", "WD", "WL", "DW", "DD", "DL", "LW", "LD", "LL")
PAIR_SCORE_TO_BIN = {0.0: 0, 0.5: 1, 1.0: 2, 1.5: 3, 2.0: 4}
NELO_SCALE = 800.0 / math.log(10.0)


@dataclass
class ParsedMatchData:
    source_paths: List[Path]
    target_engine: str
    opponents: List[str]
    sequence_order: str
    total_games_raw: int
    excluded_failure_games: int
    excluded_failure_pairs: int
    unpaired_games: int
    pair_scores: List[float]  # Per-pair points in {0.0, 0.5, 1.0, 1.5, 2.0}
    pair_keys: List[str]      # e.g. "WD", "DD", ... (target White then target Black)
    same_fen_pairs: int
    wins: int
    draws: int
    losses: int
    w_wins: int
    w_draws: int
    w_losses: int
    b_wins: int
    b_draws: int
    b_losses: int
    penta: List[int]          # [0, 0.5, 1.0, 1.5, 2.0] counts
    pairs9: Dict[str, int]


def elo_to_score(elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-float(elo) / 400.0))


def score_to_elo(score: float) -> float:
    if score <= 0.0:
        return -math.inf
    if score >= 1.0:
        return math.inf
    return -400.0 * math.log10(1.0 / score - 1.0)


def sprt_bounds(alpha: float = 0.05, beta: float = 0.05) -> Tuple[float, float]:
    """Return ``(lower_bound_B, upper_bound_A)`` for Wald's SPRT."""
    if not (0.0 < alpha < 1.0 and 0.0 < beta < 1.0):
        raise ValueError("alpha and beta must be strictly between 0 and 1")
    if alpha + beta >= 1.0:
        raise ValueError("alpha + beta must be < 1")
    return math.log(beta / (1.0 - alpha)), math.log((1.0 - beta) / alpha)


def sprt_status(llr: float, lower: float, upper: float) -> str:
    if llr >= upper:
        return "H1 Accepted"
    if llr <= lower:
        return "H0 Accepted"
    return "Continue"


def detect_engines(paths: Sequence[Path]) -> List[str]:
    counts: Counter[str] = Counter()
    for path in paths:
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace")
            counts.update(re.findall(r'\[(?:White|Black)\s+"([^"]+)"\]', text))
    return [name for name, _ in counts.most_common()]


def _extract_header(block: str, tag: str) -> str:
    m = re.search(rf'\[{tag}\s+"([^"]*)"\]', block)
    return m.group(1) if m else ""


def parse_pgn_matches(
    file_paths: str | Path | Sequence[str | Path],
    target_engine: Optional[str] = None,
    *,
    include_failures: bool = False,
) -> ParsedMatchData:
    """Parse one or more tournament PGNs into ordered pairs and WDL totals."""
    if isinstance(file_paths, (str, Path)):
        raw_paths = [Path(file_paths)]
    else:
        raw_paths = [Path(p) for p in file_paths]

    seen_paths = set()
    paths: List[Path] = []
    for p in raw_paths:
        res = p.resolve()
        if res not in seen_paths and p.exists():
            seen_paths.add(res)
            paths.append(p)
    if not paths:
        raise FileNotFoundError(f"No valid PGN files found among: {raw_paths}")

    if target_engine is None:
        engines = detect_engines(paths)
        if "New" in engines:
            target_engine = "New"
        elif engines:
            target_engine = engines[0]
        else:
            target_engine = "New"

    games: List[Dict[str, Any]] = []
    opponents_counter: Counter[str] = Counter()
    current_max_round = 0
    offset_applied = False

    for path in paths:
        content = path.read_text(encoding="utf-8", errors="replace")
        raw_games = re.split(r"\[Event ", content)[1:]
        file_games: List[Dict[str, Any]] = []

        for game_idx, block in enumerate(raw_games):
            white = _extract_header(block, "White")
            black = _extract_header(block, "Black")
            result = _extract_header(block, "Result")
            if not white or not black or result not in {"1-0", "0-1", "1/2-1/2"}:
                continue
            if target_engine not in (white, black):
                continue

            opponent = black if white == target_engine else white
            opponents_counter[opponent] += 1

            rnd_str = _extract_header(block, "Round")
            rnd_match = re.match(r"\s*(\d+)", rnd_str)
            round_number = int(rnd_match.group(1)) if rnd_match else 0
            fen = _extract_header(block, "FEN").strip()
            has_failure = bool(_extract_header(block, "Failure"))

            is_target_white = white == target_engine
            if result == "1-0":
                score = 1.0 if is_target_white else 0.0
                rt = "W" if is_target_white else "L"
            elif result == "0-1":
                score = 0.0 if is_target_white else 1.0
                rt = "L" if is_target_white else "W"
            else:
                score = 0.5
                rt = "D"

            file_games.append({
                "round": round_number,
                "orig_round": round_number,
                "file_index": game_idx,
                "is_target_white": is_target_white,
                "score": score,
                "rt": rt,
                "fen": fen,
                "has_failure": has_failure,
            })

        file_rounds = [g["orig_round"] for g in file_games if g["orig_round"] > 0]
        if file_rounds:
            min_r = min(file_rounds)
            if current_max_round > 0 and min_r <= current_max_round:
                round_offset = ((current_max_round + 1) // 2) * 2
                offset_applied = True
            else:
                round_offset = 0
            for g in file_games:
                if g["orig_round"] > 0:
                    g["round"] = g["orig_round"] + round_offset
            current_max_round = max(current_max_round, max(g["round"] for g in file_games))

        games.extend(file_games)

    rounds = [g["round"] for g in games]
    if rounds and all(r > 0 for r in rounds) and len(set(rounds)) == len(rounds):
        orig_order = list(rounds)
        games.sort(key=lambda g: g["round"])
        reordered = orig_order != [g["round"] for g in games]
        multi = f" ({len(paths)} files" + (", offset applied" if offset_applied else "") + ")" if len(paths) > 1 else ""
        seq_order = "numeric Round order" + multi + (" (reordered)" if reordered else "")
    else:
        seq_order = "PGN file order (Round missing or duplicated)"

    # Group into color-swapped pairs
    by_pair: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    use_round_pairs = bool(rounds and all(r > 0 for r in rounds))
    raw_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    paired_ids: set[int] = set()

    if use_round_pairs:
        for g in games:
            by_pair[(g["round"] + 1) // 2].append(g)
        for pair_num in sorted(by_pair):
            bucket = by_pair[pair_num]
            expected = {2 * pair_num - 1, 2 * pair_num}
            if len(bucket) != 2 or {g["round"] for g in bucket} != expected:
                continue
            tw = next((g for g in bucket if g["is_target_white"]), None)
            tb = next((g for g in bucket if not g["is_target_white"]), None)
            if tw is not None and tb is not None:
                raw_pairs.append((tw, tb))
                paired_ids.add(id(tw))
                paired_ids.add(id(tb))

    if not raw_pairs:
        for idx in range(0, len(games) - 1, 2):
            tw, tb = games[idx], games[idx + 1]
            if tw["is_target_white"] == tb["is_target_white"]:
                continue
            if not tw["is_target_white"]:
                tw, tb = tb, tw
            raw_pairs.append((tw, tb))
            paired_ids.add(id(tw))
            paired_ids.add(id(tb))

    excluded_failure_pairs = 0
    excluded_failure_games = 0
    kept_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for tw, tb in raw_pairs:
        if not include_failures and (tw["has_failure"] or tb["has_failure"]):
            excluded_failure_pairs += 1
            excluded_failure_games += int(tw["has_failure"]) + int(tb["has_failure"])
            continue
        kept_pairs.append((tw, tb))

    unpaired = [
        g for g in games
        if id(g) not in paired_ids and (include_failures or not g["has_failure"])
    ]
    if not include_failures:
        excluded_failure_games += sum(1 for g in games if id(g) not in paired_ids and g["has_failure"])

    pair_scores: List[float] = []
    pair_keys: List[str] = []
    penta = [0, 0, 0, 0, 0]
    pairs9: Dict[str, int] = {k: 0 for k in PAIR_9_KEYS}
    same_fen_pairs = 0

    wins = draws = losses = 0
    w_wins = w_draws = w_losses = 0
    b_wins = b_draws = b_losses = 0

    def _accum_game(g: Dict[str, Any]) -> None:
        nonlocal wins, draws, losses, w_wins, w_draws, w_losses, b_wins, b_draws, b_losses
        rt = g["rt"]
        if rt == "W":
            wins += 1
            if g["is_target_white"]:
                w_wins += 1
            else:
                b_wins += 1
        elif rt == "D":
            draws += 1
            if g["is_target_white"]:
                w_draws += 1
            else:
                b_draws += 1
        else:
            losses += 1
            if g["is_target_white"]:
                w_losses += 1
            else:
                b_losses += 1

    for tw, tb in kept_pairs:
        ps = tw["score"] + tb["score"]
        pk = tw["rt"] + tb["rt"]
        pair_scores.append(ps)
        pair_keys.append(pk)
        penta[PAIR_SCORE_TO_BIN[ps]] += 1
        if pk in pairs9:
            pairs9[pk] += 1
        if tw["fen"] and tw["fen"] == tb["fen"]:
            same_fen_pairs += 1
        _accum_game(tw)
        _accum_game(tb)

    for g in unpaired:
        _accum_game(g)

    return ParsedMatchData(
        source_paths=[p.resolve() for p in paths],
        target_engine=target_engine,
        opponents=[name for name, _ in opponents_counter.most_common()],
        sequence_order=seq_order,
        total_games_raw=len(games),
        excluded_failure_games=excluded_failure_games,
        excluded_failure_pairs=excluded_failure_pairs,
        unpaired_games=len(unpaired),
        pair_scores=pair_scores,
        pair_keys=pair_keys,
        same_fen_pairs=same_fen_pairs,
        wins=wins,
        draws=draws,
        losses=losses,
        w_wins=w_wins,
        w_draws=w_draws,
        w_losses=w_losses,
        b_wins=b_wins,
        b_draws=b_draws,
        b_losses=b_losses,
        penta=penta,
        pairs9=pairs9,
    )


def _constrained_mle_loglik(
    counts: Sequence[int],
    scores: Sequence[float],
    target_mean: float,
) -> Optional[float]:
    """Exact multinomial log-likelihood maximized under E[score] = target_mean.

    Solves ``sum_j p_hat_j * (a_j - mu) / (1 + lambda * (a_j - mu)) = 0`` via
    safe bracketing + safeguarded Newton-bisection on the open feasibility domain.
    Returns None when ``target_mean`` lies outside the convex hull of active bins.
    """
    active = [(int(c), float(a)) for c, a in zip(counts, scores) if c > 0]
    if not active:
        return None
    total = sum(c for c, _ in active)
    a_min = min(a for _, a in active)
    a_max = max(a for _, a in active)
    if target_mean <= a_min or target_mean >= a_max:
        return None

    emp_mean = sum(c * a for c, a in active) / total
    if abs(emp_mean - target_mean) <= 1e-14:
        return sum(c * math.log(c / total) for c, _ in active)

    terms = [(c / total, a - target_mean, c) for c, a in active]

    def _eval(lam: float) -> Tuple[float, float]:
        f = 0.0
        df = 0.0
        for p_hat, d, _ in terms:
            denom = 1.0 + lam * d
            f += p_hat * d / denom
            df -= p_hat * (d * d) / (denom * denom)
        return f, df

    lam_min = max((-1.0 / d for _, d, _ in terms if d > 0.0))
    lam_max = min((-1.0 / d for _, d, _ in terms if d < 0.0))

    if emp_mean > target_mean:
        lo = 0.0
        step = 0.5
        hi = lam_max * step
        while _eval(hi)[0] > 0.0 and step < 1.0 - 1e-15:
            step = 1.0 - (1.0 - step) * 0.5
            hi = lam_max * step
    else:
        hi = 0.0
        step = 0.5
        lo = lam_min * step
        while _eval(lo)[0] < 0.0 and step < 1.0 - 1e-15:
            step = 1.0 - (1.0 - step) * 0.5
            lo = lam_min * step

    if _eval(lo)[0] < 0.0 or _eval(hi)[0] > 0.0:
        return None

    lam = 0.5 * (lo + hi)
    for _ in range(80):
        f_val, df_val = _eval(lam)
        if abs(f_val) <= 1e-14:
            break
        if f_val > 0.0:
            lo = lam
        else:
            hi = lam
        if df_val < 0.0:
            cand = lam - f_val / df_val
            if lo < cand < hi:
                lam = cand
            else:
                lam = 0.5 * (lo + hi)
        else:
            lam = 0.5 * (lo + hi)
        if abs(hi - lo) <= 1e-14 * (1.0 + abs(lam)):
            break

    ll = 0.0
    for p_hat, d, c in terms:
        denom = 1.0 + lam * d
        if denom <= 0.0:
            return None
        ll += c * math.log(p_hat / denom)
    return ll


def multinomial_exact_llr(
    counts: Sequence[int],
    scores: Sequence[float],
    elo0: float,
    elo1: float,
) -> Optional[float]:
    mu0 = elo_to_score(elo0)
    mu1 = elo_to_score(elo1)
    ll0 = _constrained_mle_loglik(counts, scores, mu0)
    ll1 = _constrained_mle_loglik(counts, scores, mu1)
    if ll0 is None or ll1 is None:
        return None
    return ll1 - ll0


def pentanomial_moments(penta: Sequence[int]) -> Tuple[int, float, float]:
    n = sum(int(c) for c in penta)
    if n <= 0:
        return 0, 0.5, 0.0
    mean = sum(c * s for c, s in zip(penta, PENTA_SCORES_NORM)) / n
    var = max(0.0, sum(c * (s - mean) ** 2 for c, s in zip(penta, PENTA_SCORES_NORM)) / n)
    return n, mean, var


def trinomial_moments(wins: int, draws: int, losses: int) -> Tuple[int, float, float]:
    g = wins + draws + losses
    if g <= 0:
        return 0, 0.5, 0.0
    mean = (wins + 0.5 * draws) / g
    var = max(0.0, (wins + 0.25 * draws) / g - mean * mean)
    return g, mean, var


def normal_sprt_llr(n: int, mean: float, var: float, elo0: float, elo1: float) -> float:
    """Normal-approximation LLR (matches ``SPRTTest`` in ``tools/match_core.py``)."""
    if n <= 0 or var <= 1e-12:
        return 0.0
    mu0 = elo_to_score(elo0)
    mu1 = elo_to_score(elo1)
    return n * (mu1 - mu0) * (2.0 * mean - mu0 - mu1) / (2.0 * var)


def nelo_normal_llr(n_pairs: int, mean: float, var_pair: float, nelo0: float, nelo1: float) -> float:
    """Fishtest Normalized Elo (nElo) normal-approximation LLR."""
    if n_pairs <= 0 or var_pair <= 1e-12:
        return 0.0
    sigma_eff = math.sqrt(2.0 * var_pair)
    nelo_hat = NELO_SCALE * (mean - 0.5) / sigma_eff
    return (2.0 * n_pairs) / (NELO_SCALE * NELO_SCALE) * (nelo1 - nelo0) * (
        nelo_hat - 0.5 * (nelo0 + nelo1)
    )


def estimate_elo_ci(n: int, mean: float, var: float, z: float = 1.96) -> Tuple[float, float, float, float]:
    """Return ``(elo, err_95, ci_lo, ci_hi)`` using delta-method SE."""
    if n <= 0:
        return 0.0, math.inf, -math.inf, math.inf
    elo = score_to_elo(mean)
    if not math.isfinite(elo):
        return elo, math.inf, elo, elo
    stderr = math.sqrt(var / n)
    lo_s = max(1e-12, min(1.0 - 1e-12, mean - z * stderr))
    hi_s = max(1e-12, min(1.0 - 1e-12, mean + z * stderr))
    ci_lo = score_to_elo(lo_s)
    ci_hi = score_to_elo(hi_s)
    grad = 400.0 / (math.log(10.0) * mean * (1.0 - mean))
    return elo, z * stderr * grad, ci_lo, ci_hi


def estimate_nelo_ci(n_pairs: int, mean: float, var_pair: float, z: float = 1.96) -> Tuple[float, float]:
    if n_pairs <= 0 or var_pair <= 1e-12:
        return 0.0, math.inf
    sigma_eff = math.sqrt(2.0 * var_pair)
    nelo = NELO_SCALE * (mean - 0.5) / sigma_eff
    err = z * NELO_SCALE / math.sqrt(2.0 * n_pairs)
    return nelo, err


def wald_projection(
    llr_curr: float,
    mean: float,
    var_pair: float,
    elo0: float,
    elo1: float,
    lower: float,
    upper: float,
) -> Dict[str, Any]:
    """Brownian hitting-time & expected sample-size estimates under observed drift."""
    if var_pair <= 1e-12:
        return {"pass_prob": None, "exp_rem_pairs": None, "exp_h1_pairs": None, "exp_h0_pairs": None}

    mu0 = elo_to_score(elo0)
    mu1 = elo_to_score(elo1)
    delta = mu1 - mu0
    drift_obs = delta * (2.0 * mean - mu0 - mu1) / (2.0 * var_pair)
    step_var = (delta * delta) / var_pair

    # Expected total pairs from 0 under H1 (mean=mu1) and H0 (mean=mu0)
    drift_h1 = (delta * delta) / (2.0 * var_pair)
    alpha = math.exp(-upper) / (1.0 + math.exp(-upper))  # approx if symmetric, exact only for reporting
    # Compute exact alpha/beta from bounds: A=(1-b)/a, B=b/(1-a)
    exp_a = math.exp(upper)
    exp_b = math.exp(lower)
    beta_val = exp_b * (exp_a - 1.0) / (exp_a - exp_b) if exp_a > exp_b else 0.05
    alpha_val = (1.0 - beta_val) / exp_a
    exp_h1_pairs = (upper * (1.0 - beta_val) + lower * beta_val) / drift_h1 if drift_h1 > 0 else None
    exp_h0_pairs = (upper * alpha_val + lower * (1.0 - alpha_val)) / (-drift_h1) if drift_h1 > 0 else None

    if llr_curr >= upper:
        return {"pass_prob": 1.0, "exp_rem_pairs": 0.0, "exp_h1_pairs": exp_h1_pairs, "exp_h0_pairs": exp_h0_pairs}
    if llr_curr <= lower:
        return {"pass_prob": 0.0, "exp_rem_pairs": 0.0, "exp_h1_pairs": exp_h1_pairs, "exp_h0_pairs": exp_h0_pairs}

    h = 2.0 * drift_obs / step_var if step_var > 0 else 0.0
    if abs(h * (upper - lower)) < 1e-4:
        pass_prob = (llr_curr - lower) / (upper - lower)
    else:
        # Numerically stable evaluation of (exp(-h*x) - exp(-h*B)) / (exp(-h*A) - exp(-h*B))
        shift = min(-h * upper, -h * lower, -h * llr_curr)
        num = math.exp(-h * llr_curr - shift) - math.exp(-h * lower - shift)
        den = math.exp(-h * upper - shift) - math.exp(-h * lower - shift)
        pass_prob = min(1.0, max(0.0, num / den)) if den != 0 else 0.5

    if abs(drift_obs) < 1e-7:
        exp_rem = (upper - llr_curr) * (llr_curr - lower) / step_var if step_var > 0 else None
    else:
        exp_rem = max(0.0, (upper * pass_prob + lower * (1.0 - pass_prob) - llr_curr) / drift_obs)

    return {
        "pass_prob": pass_prob,
        "exp_rem_pairs": exp_rem,
        "exp_h1_pairs": exp_h1_pairs,
        "exp_h0_pairs": exp_h0_pairs,
    }


def compute_trajectory(
    pair_scores: Sequence[float],
    elo0: float,
    elo1: float,
    lower: float,
    upper: float,
    *,
    min_pairs: int = 5,
) -> Dict[str, Any]:
    """Compute pair-by-pair Pentanomial Normal LLR trajectory (matches ``SPRTTest``)."""
    llrs: List[float] = []
    running = [0, 0, 0, 0, 0]
    first_hit: Optional[Tuple[int, str, float]] = None
    min_llr = (0, 0.0)
    max_llr = (0, 0.0)

    for idx, ps in enumerate(pair_scores, start=1):
        running[PAIR_SCORE_TO_BIN[ps]] += 1
        if idx < min_pairs:
            llr = 0.0
        else:
            n, mean, var = pentanomial_moments(running)
            llr = normal_sprt_llr(n, mean, var, elo0, elo1)
        llrs.append(llr)

        if idx >= min_pairs:
            if llr < min_llr[1] or min_llr[0] == 0:
                min_llr = (idx, llr)
            if llr > max_llr[1] or max_llr[0] == 0:
                max_llr = (idx, llr)
            if first_hit is None:
                st = sprt_status(llr, lower, upper)
                if st != "Continue":
                    first_hit = (idx, st, llr)

    return {
        "llrs": llrs,
        "first_hit": first_hit,
        "min_llr": min_llr,
        "max_llr": max_llr,
    }


def parse_bounds_arg(bounds_str: str) -> List[Tuple[str, float, float]]:
    parsed: List[Tuple[str, float, float]] = []
    for raw in bounds_str.split(","):
        token = raw.strip()
        if not token:
            continue
        sep = ":" if ":" in token else "/"
        if sep not in token:
            raise ValueError(f"Invalid bound {token!r}; expected ELO0:ELO1")
        e0_s, e1_s = token.split(sep, 1)
        e0, e1 = float(e0_s), float(e1_s)
        if e1 <= e0:
            raise ValueError(f"Bound {token!r} must satisfy elo1 > elo0")
        parsed.append((f"[{e0:g}, {e1:g}]", e0, e1))
    if not parsed:
        raise ValueError("No valid bounds parsed")
    return parsed


def _display_width(text: str) -> int:
    width = 0
    for ch in text:
        if unicodedata.east_asian_width(ch) in ("W", "F"):
            width += 2
        else:
            width += 1
    return width


def _pad_display(text: str, width: int, align: str = "left") -> str:
    pad = max(0, width - _display_width(text))
    if align == "right":
        return (" " * pad) + text
    return text + (" " * pad)


def _truncate_display(text: str, width: int) -> str:
    if _display_width(text) <= width:
        return text
    out: List[str] = []
    cur = 0
    for ch in text:
        cw = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        if cur + cw + 1 > width:
            out.append("…")
            break
        out.append(ch)
        cur += cw
    return "".join(out)


def _fmt_llr(val: Optional[float]) -> str:
    return "N/A" if val is None else f"{val:+6.2f}"


def _fmt_status_short(llr: Optional[float], lower: float, upper: float) -> str:
    if llr is None:
        return "N/A"
    if llr >= upper:
        return "H1 ✅"
    if llr <= lower:
        return "H0 ❌"
    return "Cont ⏳"


def _fmt_pairs_games(pairs: Optional[float]) -> str:
    if pairs is None or not math.isfinite(pairs):
        return "-"
    games = pairs * 2.0
    if games >= 1_000_000:
        return ">1M"
    return f"{int(round(games)):,}g"


def analyze_sprt(
    *,
    penta: Sequence[int],
    wins: int,
    draws: int,
    losses: int,
    bounds: Sequence[Tuple[str, float, float]],
    alpha: float,
    beta: float,
    pair_scores: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Compute summary statistics and all SPRT models across requested bounds."""
    lower, upper = sprt_bounds(alpha, beta)
    n_pairs, mean_p, var_p = pentanomial_moments(penta)
    n_games, mean_g, var_g = trinomial_moments(wins, draws, losses)

    elo_p, err_p, ci_p_lo, ci_p_hi = estimate_elo_ci(n_pairs, mean_p, var_p)
    elo_g, err_g, ci_g_lo, ci_g_hi = estimate_elo_ci(n_games, mean_g, var_g)
    nelo, nelo_err = estimate_nelo_ci(n_pairs, mean_p, var_p)

    var_ratio = (var_p / (var_g / 2.0)) if (n_pairs > 0 and var_g > 1e-12) else None

    rows: List[Dict[str, Any]] = []
    trajectories: Dict[str, Dict[str, Any]] = {}
    for label, elo0, elo1 in bounds:
        llr_p_norm = normal_sprt_llr(n_pairs, mean_p, var_p, elo0, elo1) if n_pairs >= 5 else 0.0
        llr_p_exact = multinomial_exact_llr(penta, PENTA_SCORES_NORM, elo0, elo1) if n_pairs > 0 else None
        llr_nelo = nelo_normal_llr(n_pairs, mean_p, var_p, elo0, elo1) if n_pairs >= 5 else 0.0
        llr_t_norm = normal_sprt_llr(n_games, mean_g, var_g, elo0, elo1) if n_games >= 10 else 0.0
        llr_t_exact = multinomial_exact_llr((losses, draws, wins), TRINOMIAL_SCORES, elo0, elo1) if n_games > 0 else None

        proj = wald_projection(llr_p_norm, mean_p, var_p, elo0, elo1, lower, upper) if n_pairs >= 5 else {
            "pass_prob": None, "exp_rem_pairs": None, "exp_h1_pairs": None, "exp_h0_pairs": None
        }

        traj = None
        if pair_scores:
            traj = compute_trajectory(pair_scores, elo0, elo1, lower, upper)
            trajectories[f"{elo0:g},{elo1:g}"] = traj

        rows.append({
            "label": label,
            "elo0": elo0,
            "elo1": elo1,
            "penta_normal": llr_p_norm,
            "penta_exact": llr_p_exact,
            "penta_nelo": llr_nelo,
            "trinomial_normal": llr_t_norm,
            "trinomial_exact": llr_t_exact,
            "status_penta_normal": sprt_status(llr_p_norm, lower, upper),
            "status_penta_exact": sprt_status(llr_p_exact, lower, upper) if llr_p_exact is not None else "N/A",
            "first_hit": traj["first_hit"] if traj else None,
            "min_llr": traj["min_llr"] if traj else None,
            "max_llr": traj["max_llr"] if traj else None,
            "projection": proj,
        })

    return {
        "alpha": alpha,
        "beta": beta,
        "bounds_llr": [lower, upper],
        "pairs": n_pairs,
        "games": n_games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": mean_g if n_games > 0 else mean_p,
        "draw_rate": (draws / n_games) if n_games > 0 else 0.0,
        "penta": list(penta),
        "var_penta": var_p,
        "var_trinomial_half": var_g / 2.0,
        "var_ratio": var_ratio,
        "elo_penta": {"elo": elo_p, "err95": err_p, "ci95": [ci_p_lo, ci_p_hi]},
        "elo_trinomial": {"elo": elo_g, "err95": err_g, "ci95": [ci_g_lo, ci_g_hi]},
        "nelo": {"nelo": nelo, "err95": nelo_err},
        "sprt_rows": rows,
        "trajectories": trajectories,
    }


def format_report(
    analysis: Dict[str, Any],
    *,
    match_data: Optional[ParsedMatchData] = None,
) -> str:
    lines: List[str] = []
    lower, upper = analysis["bounds_llr"]
    alpha, beta = analysis["alpha"], analysis["beta"]

    lines.append("==================== SPRT 多模型分析報告 ====================")
    if match_data is not None:
        if len(match_data.source_paths) == 1:
            lines.append(f"來源 PGN : {match_data.source_paths[0]}")
        else:
            lines.append(f"來源 PGN : {len(match_data.source_paths)} 檔合併")
            for sp in match_data.source_paths:
                lines.append(f"  - {sp}")
        opp_str = ", ".join(match_data.opponents) if match_data.opponents else "?"
        lines.append(f"測試對象 : {match_data.target_engine} vs {opp_str}")
        lines.append(f"序列順序 : {match_data.sequence_order}")
        if match_data.excluded_failure_pairs > 0 or match_data.excluded_failure_games > 0:
            lines.append(
                f"失效剔除 : 已排除 {match_data.excluded_failure_pairs} 組含 Failure 的配對 "
                f"(含 {match_data.excluded_failure_games} 局 Failure)"
            )
        if match_data.unpaired_games > 0:
            lines.append(f"未配對局 : {match_data.unpaired_games} 局（僅計入 Trinomial WDL，不計入 Pentanomial）")
        lines.append(
            f"配對品質 : {len(match_data.pair_scores)} 對 | 同 FEN 換先 {match_data.same_fen_pairs} 對"
        )
    else:
        lines.append("輸入模式 : 手動輸入統計次數")

    lines.append("")
    lines.append("[1. 基本戰績與變異數結構]")
    lines.append(
        f"對局統計 : {analysis['games']} 局 ({analysis['pairs']} 對) | "
        f"W {analysis['wins']} - D {analysis['draws']} - L {analysis['losses']} | "
        f"得分率 {analysis['score']:.2%} | 和棋率 {analysis['draw_rate']:.2%}"
    )
    if match_data is not None:
        lines.append(
            f"執色明細 : 持白 W{match_data.w_wins}-D{match_data.w_draws}-L{match_data.w_losses} "
            f"({(match_data.w_wins + 0.5 * match_data.w_draws) / max(1, match_data.w_wins + match_data.w_draws + match_data.w_losses):.2%}) | "
            f"持黑 W{match_data.b_wins}-D{match_data.b_draws}-L{match_data.b_losses} "
            f"({(match_data.b_wins + 0.5 * match_data.b_draws) / max(1, match_data.b_wins + match_data.b_draws + match_data.b_losses):.2%})"
        )

    ep = analysis["elo_penta"]
    eg = analysis["elo_trinomial"]
    ne = analysis["nelo"]
    if analysis["pairs"] > 0:
        lines.append(
            f"Pentanomial [0, 0.5, 1, 1.5, 2] : {analysis['penta']}"
        )
        if match_data is not None:
            p9 = " | ".join(f"{k}:{match_data.pairs9[k]}" for k in PAIR_9_KEYS if match_data.pairs9[k] > 0)
            lines.append(f"九宮格 (W後B) : {p9}")
        lines.append(
            f"Elo (Pentanomial) : {ep['elo']:+.2f} ± {ep['err95']:.2f} "
            f"(95% CI [{ep['ci95'][0]:+.2f}, {ep['ci95'][1]:+.2f}])"
        )
        lines.append(
            f"nElo (Normalized) : {ne['nelo']:+.2f} ± {ne['err95']:.2f}"
        )
        vr = analysis["var_ratio"]
        vr_note = f" (配對/單局變異數比 = {vr:.3f})" if vr is not None else ""
        lines.append(
            f"Elo (Trinomial)   : {eg['elo']:+.2f} ± {eg['err95']:.2f} "
            f"(95% CI [{eg['ci95'][0]:+.2f}, {eg['ci95'][1]:+.2f}]){vr_note}"
        )
    else:
        lines.append(
            f"Elo (Trinomial)   : {eg['elo']:+.2f} ± {eg['err95']:.2f} "
            f"(95% CI [{eg['ci95'][0]:+.2f}, {eg['ci95'][1]:+.2f}])"
        )

    lines.append("")
    lines.append(
        f"[2. 多模型 SPRT 總覽]  (α={alpha:g}, β={beta:g} → 邊界 [{lower:+.2f}, {upper:+.2f}])"
    )
    has_penta = analysis["pairs"] > 0
    if has_penta:
        lines.append(
            f"{_pad_display('假設區間 [elo0, elo1]', 28)} | "
            f"{_pad_display('Penta常態', 9, 'right')} | "
            f"{_pad_display('Penta精確', 9, 'right')} | "
            f"{_pad_display('nElo常態', 9, 'right')} | "
            f"{_pad_display('WDL常態', 8, 'right')} | "
            f"{_pad_display('WDL精確', 8, 'right')} | 判定(Penta)"
        )
        lines.append("-" * 102)
        for row in analysis["sprt_rows"]:
            bound_str = _truncate_display(f"{row['label']} [{row['elo0']:g}, {row['elo1']:g}]", 28)
            st = _fmt_status_short(row["penta_normal"], lower, upper)
            lines.append(
                f"{_pad_display(bound_str, 28)} | "
                f"{_pad_display(_fmt_llr(row['penta_normal']), 9, 'right')} | "
                f"{_pad_display(_fmt_llr(row['penta_exact']), 9, 'right')} | "
                f"{_pad_display(_fmt_llr(row['penta_nelo']), 9, 'right')} | "
                f"{_pad_display(_fmt_llr(row['trinomial_normal']), 8, 'right')} | "
                f"{_pad_display(_fmt_llr(row['trinomial_exact']), 8, 'right')} | {st}"
            )
    else:
        lines.append(
            f"{_pad_display('假設區間 [elo0, elo1]', 28)} | "
            f"{_pad_display('WDL常態', 8, 'right')} | "
            f"{_pad_display('WDL精確', 8, 'right')} | 判定(WDL)"
        )
        lines.append("-" * 64)
        for row in analysis["sprt_rows"]:
            bound_str = _truncate_display(f"{row['label']} [{row['elo0']:g}, {row['elo1']:g}]", 28)
            st = _fmt_status_short(row["trinomial_normal"], lower, upper)
            lines.append(
                f"{_pad_display(bound_str, 28)} | "
                f"{_pad_display(_fmt_llr(row['trinomial_normal']), 8, 'right')} | "
                f"{_pad_display(_fmt_llr(row['trinomial_exact']), 8, 'right')} | {st}"
            )

    if has_penta:
        lines.append("")
        lines.append("[3. 序貫軌跡與隨機遊走推估]  (基於 Pentanomial 常態模型)")
        lines.append(
            f"{_pad_display('假設區間', 24)} | "
            f"{_pad_display('首次觸線', 17)} | "
            f"{_pad_display('Min / Max LLR', 17)} | "
            f"{_pad_display('P(過H1|現率)', 12, 'right')} | "
            f"{_pad_display('預估剩餘', 9, 'right')} | "
            f"{_pad_display('H1理論均長', 10, 'right')}"
        )
        lines.append("-" * 104)
        for row in analysis["sprt_rows"]:
            bound_str = _truncate_display(f"[{row['elo0']:g}, {row['elo1']:g}] {row['label']}", 24)
            fh = row["first_hit"]
            if fh is not None:
                hit_pair, hit_st, _ = fh
                short_st = "H1" if "H1" in hit_st else "H0"
                fh_str = f"{short_st} @{hit_pair}對({hit_pair * 2}局)"
            else:
                fh_str = "尚未觸線"

            mn = row["min_llr"]
            mx = row["max_llr"]
            mm_str = f"{mn[1]:+.2f} / {mx[1]:+.2f}" if (mn and mx) else "-"

            proj = row["projection"]
            p_h1 = f"{proj['pass_prob']:.1%}" if proj["pass_prob"] is not None else "-"
            rem_str = _fmt_pairs_games(proj["exp_rem_pairs"])
            h1_len = _fmt_pairs_games(proj["exp_h1_pairs"])

            lines.append(
                f"{_pad_display(bound_str, 24)} | "
                f"{_pad_display(fh_str, 17)} | "
                f"{_pad_display(mm_str, 17)} | "
                f"{_pad_display(p_h1, 12, 'right')} | "
                f"{_pad_display(rem_str, 9, 'right')} | "
                f"{_pad_display(h1_len, 10, 'right')}"
            )

    lines.append("")
    lines.append("註：")
    lines.append("  1. Penta常態 = 現行 tools/match_core.py (SPRTTest) 與 tune_search/stats.py 完全同式。")
    lines.append("  2. Penta精確 = Fishtest/Fastchess 限制最大概似多項式檢定 (無常態近似誤差)。")
    lines.append("  3. nElo常態  = Fishtest Normalized Elo 檢定 (依當前和棋率校正有效標準差，單位為 nElo)。")
    lines.append("  4. WDL常態/精確 = 單局三項式模型 (未配對聚合)；對比 Penta 可看出換先配對變異數效果。")
    lines.append("  5. P(過H1|現率) 與 預估剩餘 為假設「真實勝率 = 目前觀測均值」下的 Wald/布朗運動觸線推估。")
    return "\n".join(lines)


def plot_trajectories(
    analysis: Dict[str, Any],
    output_path: Path,
    *,
    title_prefix: str = "SPRT LLR Trajectories",
) -> Path:
    import matplotlib.pyplot as plt

    lower, upper = analysis["bounds_llr"]
    rows = analysis["sprt_rows"]
    trajs = analysis["trajectories"]
    if not trajs:
        raise ValueError("No pair trajectory data available to plot")

    fig, ax = plt.subplots(figsize=(11, 6))
    for row in rows:
        key = f"{row['elo0']:g},{row['elo1']:g}"
        traj = trajs.get(key)
        if not traj:
            continue
        ys = traj["llrs"]
        xs = list(range(1, len(ys) + 1))
        ax.plot(xs, ys, linewidth=1.8, label=f"[{row['elo0']:g}, {row['elo1']:g}] ({ys[-1]:+.2f})")

    ax.axhline(upper, color="#2ca02c", linestyle="--", linewidth=1.4, label=f"H1 bound (+{upper:.2f})")
    ax.axhline(lower, color="#d62728", linestyle="--", linewidth=1.4, label=f"H0 bound ({lower:.2f})")
    ax.axhline(0.0, color="#888888", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Color-swapped Pairs (N)")
    ax.set_ylabel("LLR (Pentanomial Normal)")
    ax.set_title(f"{title_prefix} (games={analysis['games']}, Elo={analysis['elo_penta']['elo']:+.2f})")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute multiple SPRT models and bounds on tournament PGNs or WDL/pentanomial counts."
    )
    parser.add_argument("pgn_files", nargs="*", type=Path, help="PGN file(s) to analyze (default: tournament_analysis/tournament_results.pgn)")
    parser.add_argument("--pgn", nargs="+", action="extend", type=Path, default=None, help="Explicit PGN file(s)")
    parser.add_argument("--target", default=None, help="Target engine name (default: auto / New)")
    parser.add_argument("--bounds", default=None, help="Comma-separated elo0:elo1 pairs, e.g. '0:10,0:5,-3:1'")
    parser.add_argument("--elo0", type=float, default=None, help="Single custom H0 Elo")
    parser.add_argument("--elo1", type=float, default=None, help="Single custom H1 Elo")
    parser.add_argument("--alpha", type=float, default=0.05, help="Type I error (default: 0.05)")
    parser.add_argument("--beta", type=float, default=0.05, help="Type II error (default: 0.05)")
    parser.add_argument("--wdl", nargs=3, type=int, metavar=("W", "D", "L"), default=None, help="Manual W D L counts")
    parser.add_argument("--penta", nargs=5, type=int, metavar=("C0", "C1", "C2", "C3", "C4"), default=None, help="Manual pentanomial counts [0, 0.5, 1, 1.5, 2]")
    parser.add_argument("--include-failures", action="store_true", help="Include pairs with [Failure ...] headers")
    parser.add_argument("--plot", nargs="?", const=str(HERE / "sprt_trajectory.png"), default=None, help="Save LLR trajectory chart")
    parser.add_argument("--json", nargs="?", const="-", default=None, help="Output JSON (to stdout if '-', or file path)")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")

    args = _parse_args(argv)

    if (args.elo0 is None) != (args.elo1 is None):
        print("Error: --elo0 and --elo1 must be specified together.", file=sys.stderr)
        return 2

    try:
        if args.bounds is not None:
            bounds = parse_bounds_arg(args.bounds)
        elif args.elo0 is not None and args.elo1 is not None:
            if args.elo1 <= args.elo0:
                print("Error: --elo1 must be greater than --elo0.", file=sys.stderr)
                return 2
            bounds = [("自訂區間", args.elo0, args.elo1)]
        else:
            bounds = list(DEFAULT_BOUNDS)
        sprt_bounds(args.alpha, args.beta)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    match_data: Optional[ParsedMatchData] = None
    if args.penta is not None:
        c0, c1, c2, c3, c4 = (int(x) for x in args.penta)
        if min(c0, c1, c2, c3, c4) < 0:
            print("Error: pentanomial counts must be non-negative.", file=sys.stderr)
            return 2
        penta = [c0, c1, c2, c3, c4]
        # Note: without 9-cell breakdown, C2 (1.0 pair score) could be DD or WL/LW;
        # use --wdl alongside --penta if exact WDL counts are known.
        if args.wdl is not None:
            wins, draws, losses = (int(x) for x in args.wdl)
        else:
            # Approximate W/D/L assuming C2 is DD (minimum variance/draw representation)
            wins = 2 * c4 + c3
            draws = c3 + 2 * c2 + c1
            losses = 2 * c0 + c1
        pair_scores = None
    elif args.wdl is not None:
        wins, draws, losses = (int(x) for x in args.wdl)
        if min(wins, draws, losses) < 0:
            print("Error: WDL counts must be non-negative.", file=sys.stderr)
            return 2
        penta = [0, 0, 0, 0, 0]
        pair_scores = None
    else:
        raw_paths: List[Path] = []
        if args.pgn:
            raw_paths.extend(args.pgn)
        if args.pgn_files:
            raw_paths.extend(args.pgn_files)
        if not raw_paths:
            raw_paths = [DEFAULT_PGN]

        missing = [p for p in raw_paths if not p.exists()]
        if missing and len(missing) == len(raw_paths):
            for p in missing:
                print(f"Error: PGN not found: {p}", file=sys.stderr)
            return 1

        match_data = parse_pgn_matches(
            raw_paths,
            target_engine=args.target,
            include_failures=args.include_failures,
        )
        penta = match_data.penta
        wins = match_data.wins
        draws = match_data.draws
        losses = match_data.losses
        pair_scores = match_data.pair_scores

    analysis = analyze_sprt(
        penta=penta,
        wins=wins,
        draws=draws,
        losses=losses,
        bounds=bounds,
        alpha=args.alpha,
        beta=args.beta,
        pair_scores=pair_scores,
    )

    if args.plot and analysis["trajectories"]:
        out_img = plot_trajectories(
            analysis,
            Path(args.plot),
            title_prefix=f"SPRT LLR — {match_data.target_engine if match_data else 'Custom'}",
        )
        print(f"Saved trajectory plot: {out_img}", file=sys.stderr)

    if args.json is not None:
        payload = dict(analysis)
        payload.pop("trajectories", None)
        if match_data is not None:
            payload["sources"] = [str(p) for p in match_data.source_paths]
            payload["target"] = match_data.target_engine
            payload["opponents"] = match_data.opponents
            payload["pairs9"] = match_data.pairs9
        text_json = json.dumps(payload, ensure_ascii=False, indent=2)
        if args.json == "-":
            print(text_json)
        else:
            Path(args.json).write_text(text_json + "\n", encoding="utf-8", newline="\n")
            print(f"Saved JSON report: {args.json}", file=sys.stderr)
        return 0

    print(format_report(analysis, match_data=match_data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
