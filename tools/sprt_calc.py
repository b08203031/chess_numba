#!/usr/bin/env python3
"""Lightweight CLI SPRT / Elo calculator for tournament notebooks, PGNs, and WDL/Pentanomial counts.

Starts instantly (imports only stdlib + `tools.match_core`, never Numba/search).

Examples:
    python tools/sprt_calc.py
    python tools/sprt_calc.py "tournament_colab_runner_se_multicut copy.ipynb"
    python tools/sprt_calc.py tournament_analysis/tournament_results.pgn --bounds -1.75,0.25 -2,0
    python tools/sprt_calc.py --penta 28,173,365,173,30 --bounds -1.75,0.25 -2,0
    python tools/sprt_calc.py --wdl 316,910,312
    python tools/sprt_calc.py -w 316 -d 910 -l 312 --only-custom --bounds -3,0 -5,0
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.match_core import SPRTTest, calculate_elo_statistics

DEFAULT_BOUNDS: tuple[tuple[float, float], ...] = (
    (-1.75, 0.25),
    (-2.0, 0.0),
    (-3.0, 1.0),
    (-3.0, 0.0),
    (-5.0, 0.0),
    (-10.0, 0.0),
    (-12.0, 0.0),
    (-3.0, 3.0),
    (0.0, 5.0),
    (0.0, 10.0),
)

GAME_HEADER_RE = re.compile(r"^Game\s+(\d+):\s+(.+?)\s+\(White\)\s+vs\s+(.+?)\s+\(Black\)\s*$")
RESULT_LINE_RE = re.compile(r"^Result:\s+(1-0|0-1|1/2-1/2)\b")
FORFEIT_LINE_RE = re.compile(r"^(.+?)\s+forfeits\b")
SUMMARY_RE = re.compile(
    r"After\s+(\d+)\s+pairs\s+\((\d+)\s+games\):\s+(.+?)\s+([\d.]+)\s+\((\d+)W\s+(\d+)D\s+(\d+)L\)"
)
PGN_TAG_RE = re.compile(r'^\[(\w+)\s+"(.*)"\]\s*$')


@dataclass
class MatchData:
    source_desc: str
    target_engine: str
    wins: int
    draws: int
    losses: int
    penta: Optional[tuple[int, int, int, int, int]] = None  # LL, LD/DL, DD/WL, WD/DW, WW
    pair_scores: Optional[list[float]] = None
    unpaired_games: int = 0

    @property
    def total_games(self) -> int:
        return self.wins + self.draws + self.losses


def _score_from_result(result: str, target_is_white: bool) -> Optional[float]:
    if result == "1-0":
        return 1.0 if target_is_white else 0.0
    if result == "0-1":
        return 0.0 if target_is_white else 1.0
    if result == "1/2-1/2":
        return 0.5
    return None


def _pair_to_wdl_and_penta(
    games: list[tuple[int, str, str, str]], target_engine: str
) -> tuple[int, int, int, tuple[int, int, int, int, int], list[float], int]:
    """Group individual games into color-swapped pairs and compute WDL + Pentanomial."""
    wins = draws = losses = 0
    penta_counts = [0, 0, 0, 0, 0]  # index = int(round(pair_score * 2))
    pair_scores: list[float] = []

    by_pair: dict[int, list[tuple[bool, float]]] = {}
    no_round: list[tuple[bool, float]] = []

    for round_num, white, black, result in games:
        if target_engine not in (white, black):
            continue
        target_is_white = white == target_engine
        score = _score_from_result(result, target_is_white)
        if score is None:
            continue
        if score == 1.0:
            wins += 1
        elif score == 0.0:
            losses += 1
        else:
            draws += 1

        if round_num > 0:
            pair_id = (round_num - 1) // 2
            by_pair.setdefault(pair_id, []).append((target_is_white, score))
        else:
            no_round.append((target_is_white, score))

    unpaired = 0
    for pair_id in sorted(by_pair):
        entries = by_pair[pair_id]
        if len(entries) == 2 and entries[0][0] != entries[1][0]:
            ps = entries[0][1] + entries[1][1]
            pair_scores.append(ps)
            penta_counts[int(round(ps * 2))] += 1
        else:
            unpaired += len(entries)

    i = 0
    while i + 1 < len(no_round):
        g1, g2 = no_round[i], no_round[i + 1]
        if g1[0] != g2[0]:
            ps = g1[1] + g2[1]
            pair_scores.append(ps)
            penta_counts[int(round(ps * 2))] += 1
            i += 2
        else:
            unpaired += 1
            i += 1
    if i < len(no_round):
        unpaired += 1

    penta = (
        penta_counts[0],
        penta_counts[1],
        penta_counts[2],
        penta_counts[3],
        penta_counts[4],
    )
    return wins, draws, losses, penta, pair_scores, unpaired


def _extract_ipynb_text(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    chunks: list[str] = []
    for cell in data.get("cells", []):
        for out in cell.get("outputs", []):
            text = out.get("text")
            if isinstance(text, list):
                chunks.append("".join(text))
            elif isinstance(text, str):
                chunks.append(text)
            plain = out.get("data", {}).get("text/plain")
            if isinstance(plain, list):
                chunks.append("".join(plain))
            elif isinstance(plain, str):
                chunks.append(plain)
    return "\n".join(chunks)


def _infer_text_target(games: list[tuple[int, str, str, str]], preferred: str) -> str:
    if any(preferred in (w, b) for _, w, b, _ in games):
        return preferred
    for _, w, _, _ in games:
        return w
    return preferred


def _parse_text_log(text: str, target_engine: str, source_desc: str) -> Optional[MatchData]:
    games: list[tuple[int, str, str, str]] = []
    cur_round: Optional[int] = None
    cur_white = cur_black = ""

    for raw_line in text.splitlines():
        line = raw_line.strip()
        hm = GAME_HEADER_RE.match(line)
        if hm:
            cur_round = int(hm.group(1))
            cur_white = hm.group(2)
            cur_black = hm.group(3)
            continue
        if cur_round is None:
            continue
        rm = RESULT_LINE_RE.match(line)
        if rm:
            games.append((cur_round, cur_white, cur_black, rm.group(1)))
            cur_round = None
            continue
        fm = FORFEIT_LINE_RE.match(line)
        if fm:
            loser = fm.group(1)
            if loser == cur_white:
                games.append((cur_round, cur_white, cur_black, "0-1"))
                cur_round = None
            elif loser == cur_black:
                games.append((cur_round, cur_white, cur_black, "1-0"))
                cur_round = None

    if games:
        target = _infer_text_target(games, target_engine)
        w, d, l, penta, pair_scores, unpaired = _pair_to_wdl_and_penta(games, target)
        if w + d + l > 0:
            if pair_scores:
                return MatchData(
                    source_desc=source_desc,
                    target_engine=target,
                    wins=w,
                    draws=d,
                    losses=l,
                    penta=penta,
                    pair_scores=pair_scores,
                    unpaired_games=unpaired,
                )
            return MatchData(source_desc=source_desc, target_engine=target, wins=w, draws=d, losses=l)

    summaries = SUMMARY_RE.findall(text)
    if summaries:
        matching = [s for s in summaries if s[2] == target_engine]
        chosen = matching[-1] if matching else summaries[-1]
        return MatchData(
            source_desc=f"{source_desc} (summary only)",
            target_engine=chosen[2],
            wins=int(chosen[4]),
            draws=int(chosen[5]),
            losses=int(chosen[6]),
        )
    return None


def _parse_pgn_file(path: Path, target_engine: str) -> Optional[MatchData]:
    games: list[tuple[int, str, str, str]] = []
    tags: dict[str, str] = {}

    def _flush() -> None:
        if "White" in tags and "Black" in tags and "Result" in tags:
            res = tags["Result"].strip()
            if res in ("1-0", "0-1", "1/2-1/2"):
                rnd = 0
                rm = re.match(r"\s*(\d+)", tags.get("Round", ""))
                if rm:
                    rnd = int(rm.group(1))
                games.append((rnd, tags["White"], tags["Black"], res))
        tags.clear()

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("["):
                m = PGN_TAG_RE.match(line.strip())
                if m:
                    key, val = m.group(1), m.group(2)
                    if key == "Event" and tags:
                        _flush()
                    tags[key] = val
    _flush()

    if not games:
        return None
    target = _infer_text_target(games, target_engine)
    w, d, l, penta, pair_scores, unpaired = _pair_to_wdl_and_penta(games, target)
    if w + d + l == 0:
        return None
    if pair_scores:
        return MatchData(
            source_desc=str(path),
            target_engine=target,
            wins=w,
            draws=d,
            losses=l,
            penta=penta,
            pair_scores=pair_scores,
            unpaired_games=unpaired,
        )
    return MatchData(source_desc=str(path), target_engine=target, wins=w, draws=d, losses=l)


def _load_file(path: Path, target_engine: str) -> Optional[MatchData]:
    suffix = path.suffix.lower()
    if suffix == ".ipynb":
        return _parse_text_log(_extract_ipynb_text(path), target_engine, str(path))
    if suffix == ".pgn":
        return _parse_pgn_file(path, target_engine)
    return _parse_text_log(path.read_text(encoding="utf-8", errors="replace"), target_engine, str(path))


def _discover_latest_source(target_engine: str) -> MatchData:
    candidates: list[Path] = list(REPO_ROOT.glob("tournament_colab_runner*.ipynb"))
    default_pgn = REPO_ROOT / "tournament_analysis" / "tournament_results.pgn"
    if default_pgn.exists():
        candidates.append(default_pgn)
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    for cand in candidates:
        try:
            data = _load_file(cand, target_engine)
        except Exception:
            continue
        if data is not None and data.total_games > 0:
            return data

    raise SystemExit(
        "No tournament results found automatically. Pass a file path (.ipynb/.pgn/.txt), "
        "--penta LL,LD,DD,WD,WW, or --wdl W,D,L."
    )


def _parse_int_list(raw: list[str], expected_len: int, flag_name: str) -> list[int]:
    tokens = [tok for item in raw for tok in item.replace(",", " ").split() if tok]
    if len(tokens) != expected_len:
        raise ValueError(f"{flag_name} expects {expected_len} integers, got {len(tokens)}: {tokens}")
    nums = [int(t) for t in tokens]
    if any(v < 0 for v in nums):
        raise ValueError(f"{flag_name} counts must be non-negative: {nums}")
    return nums


_BOUND_ARG_RE = re.compile(r"^[\[\(]?\s*[-+]?\d+(?:\.\d+)?\s*[,:]\s*[-+]?\d+(?:\.\d+)?\s*[\]\)]?$")


def _extract_bounds_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    """Pre-extract --bounds/-b values before argparse so negative values like `-1.75,0.25` are not rejected as flags."""
    cleaned: list[str] = []
    bounds_tokens: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in ("--bounds", "-b"):
            i += 1
            if i >= len(argv):
                raise ValueError(f"{tok} requires at least one bound such as -1.75,0.25.")
            # Always consume at least one token; keep consuming subsequent bound-shaped tokens.
            bounds_tokens.append(argv[i])
            i += 1
            while i < len(argv) and _BOUND_ARG_RE.match(argv[i]):
                bounds_tokens.append(argv[i])
                i += 1
            continue
        if tok.startswith("--bounds="):
            val = tok.split("=", 1)[1]
            if not val:
                raise ValueError("--bounds= requires a value such as --bounds=-1.75,0.25.")
            bounds_tokens.extend(p for p in re.split(r"\s+", val.strip()) if p)
            i += 1
            continue
        cleaned.append(tok)
        i += 1
    return cleaned, bounds_tokens


def _parse_bound_token(token: str) -> tuple[float, float]:
    cleaned = token.strip().lstrip("[(").rstrip("])")
    parts = [p.strip() for p in re.split(r"[,:]", cleaned) if p.strip() != ""]
    if len(parts) != 2:
        raise ValueError(f"Invalid SPRT bound '{token}'. Expected 'elo0,elo1' (e.g. -1.75,0.25).")
    elo0, elo1 = float(parts[0]), float(parts[1])
    if elo1 <= elo0:
        raise ValueError(f"SPRT bound must have elo1 > elo0, got [{elo0}, {elo1}].")
    return elo0, elo1


def _build_bounds_list(raw_bounds: Optional[list[str]], only_custom: bool) -> list[tuple[float, float]]:
    custom = [_parse_bound_token(tok) for tok in (raw_bounds or [])]
    if only_custom:
        if not custom:
            raise ValueError("--only-custom requires --bounds.")
        return custom
    combined = list(DEFAULT_BOUNDS)
    for b in custom:
        if b not in combined:
            combined.append(b)
    return combined


def _los_pct(mean_diff: float, se_diff: float) -> float:
    if se_diff <= 0.0:
        if mean_diff > 0.0:
            return 100.0
        if mean_diff < 0.0:
            return 0.0
        return 50.0
    return 50.0 * (1.0 + math.erf(mean_diff / (math.sqrt(2.0) * se_diff)))


def _fmt_games_est(upper_bound: float, drift_per_unit: float, games_per_unit: int) -> str:
    if drift_per_unit <= 1e-12:
        return "inf"
    est = upper_bound / drift_per_unit * games_per_unit
    return f"~{est:,.0f}" if est < 1_000_000 else ">1M"


def _evaluate_bounds(
    data: MatchData, bounds: Iterable[tuple[float, float]], alpha: float, beta: float
) -> tuple[str, list[dict[str, str]]]:
    rows: list[dict[str, str]] = []
    if data.pair_scores:
        mode = "Pentanomial (pair-based, exact match_core.SPRTTest)"
        n_pairs = len(data.pair_scores)
        mean_pair = sum(data.pair_scores) / n_pairs
        var_pair = sum(v * v for v in data.pair_scores) / n_pairs - mean_pair * mean_pair
        a_bound = math.log((1.0 - beta) / alpha)
        b_bound = math.log(beta / (1.0 - alpha))

        for elo0, elo1 in bounds:
            sprt = SPRTTest(elo0=elo0, elo1=elo1, alpha=alpha, beta=beta)
            for ps in data.pair_scores:
                sprt.add_pair_result(ps)
            status, llr = sprt.check_status()
            mu0 = 2.0 / (1.0 + 10.0 ** (-elo0 / 400.0))
            mu1 = 2.0 / (1.0 + 10.0 ** (-elo1 / 400.0))
            if var_pair > 1e-10 and n_pairs >= 5:
                drift0 = (mu1 - mu0) / var_pair * (1.0 - (mu0 + mu1) / 2.0)
                drift_cur = llr / n_pairs
            else:
                drift0 = drift_cur = 0.0
            rows.append(
                {
                    "interval": f"[{elo0:+.2f}, {elo1:+.2f}]",
                    "llr": f"{llr:+.4f}",
                    "bounds": f"[{b_bound:+.2f}, {a_bound:+.2f}]",
                    "status": status,
                    "est0": _fmt_games_est(a_bound, drift0, 2),
                    "est_cur": _fmt_games_est(a_bound, drift_cur, 2),
                }
            )
        return mode, rows

    mode = "Trinomial WDL approximation (no pair data)"
    n_games = data.total_games
    mean_g = (data.wins + 0.5 * data.draws) / n_games
    score_sum = data.wins + 0.5 * data.draws
    var_g = max(0.0, (data.wins + 0.25 * data.draws) / n_games - mean_g * mean_g)
    a_bound = math.log((1.0 - beta) / alpha)
    b_bound = math.log(beta / (1.0 - alpha))

    for elo0, elo1 in bounds:
        mu0 = 1.0 / (1.0 + 10.0 ** (-elo0 / 400.0))
        mu1 = 1.0 / (1.0 + 10.0 ** (-elo1 / 400.0))
        if var_g <= 1e-10 or n_games < 10:
            llr = 0.0
            status = "Continue"
            drift0 = drift_cur = 0.0
        else:
            llr = (mu1 - mu0) / var_g * (score_sum - n_games * (mu0 + mu1) / 2.0)
            if llr >= a_bound:
                status = "H1 Accepted (Engine improved)"
            elif llr <= b_bound:
                status = "H0 Accepted (No improvement)"
            else:
                status = "Continue"
            drift0 = (mu1 - mu0) / var_g * (0.5 - (mu0 + mu1) / 2.0)
            drift_cur = llr / n_games
        rows.append(
            {
                "interval": f"[{elo0:+.2f}, {elo1:+.2f}]",
                "llr": f"{llr:+.4f}",
                "bounds": f"[{b_bound:+.2f}, {a_bound:+.2f}]",
                "status": status,
                "est0": _fmt_games_est(a_bound, drift0, 1),
                "est_cur": _fmt_games_est(a_bound, drift_cur, 1),
            }
        )
    return mode, rows


def _print_report(data: MatchData, bounds: list[tuple[float, float]], alpha: float, beta: float) -> None:
    total = data.total_games
    points = data.wins + 0.5 * data.draws
    score_pct = 100.0 * points / total
    penta_only = data.source_desc.startswith("CLI --penta")

    print(f"Source   : {data.source_desc}")
    print(f"Target   : {data.target_engine} | Total: {total} games | Points: {points:.1f}/{total} ({score_pct:.2f}%)")
    if penta_only:
        print(f"WDL(est) : {data.wins}W {data.draws}D {data.losses}L (approx; DD/WL bucket treated as draws)")
    else:
        print(f"WDL      : {data.wins}W {data.draws}D {data.losses}L (Net: {data.wins - data.losses:+d})")
    if data.unpaired_games:
        print(f"Note     : {data.unpaired_games} unpaired game(s) excluded from Pentanomial LLR")

    if data.penta and data.pair_scores:
        ll, ld, dd, wd, ww = data.penta
        n_pairs = len(data.pair_scores)
        mean_p = sum(data.pair_scores) / n_pairs
        var_p = max(0.0, sum(v * v for v in data.pair_scores) / n_pairs - mean_p * mean_p)
        score_g = mean_p / 2.0
        se_g = math.sqrt(var_p / n_pairs) / 2.0
        if 0.0 < score_g < 1.0:
            grad = 400.0 / (math.log(10.0) * score_g * (1.0 - score_g))
            p_elo = -400.0 * math.log10(1.0 / score_g - 1.0)
            p_ci = 1.96 * se_g * grad
        else:
            p_elo, p_ci = calculate_elo_statistics(data.wins, data.draws, data.losses)
        p_los = _los_pct(mean_p - 1.0, math.sqrt(var_p / n_pairs))
        print(f"Penta    : [LL={ll}, LD={ld}, DD/WL={dd}, WD={wd}, WW={ww}] ({n_pairs} pairs; WW-LL={ww-ll:+d}, WD-LD={wd-ld:+d})")
        print(f"Elo(Penta): {p_elo:+.2f} +/- {p_ci:.2f} (95% CI [{p_elo - p_ci:+.2f}, {p_elo + p_ci:+.2f}]), LOS={p_los:.1f}%")
    else:
        print("Penta    : unavailable (pass a log/PGN/.ipynb or --penta for exact Pentanomial SPRT)")

    if not penta_only:
        tri_elo, tri_ci = calculate_elo_statistics(data.wins, data.draws, data.losses)
        tri_var = max(0.0, (data.wins + 0.25 * data.draws) / total - (points / total) ** 2)
        tri_los = _los_pct(points / total - 0.5, math.sqrt(tri_var / total))
        print(f"Elo(WDL) : {tri_elo:+.2f} +/- {tri_ci:.2f} (95% CI [{tri_elo - tri_ci:+.2f}, {tri_elo + tri_ci:+.2f}]), LOS={tri_los:.1f}%")

    mode, rows = _evaluate_bounds(data, bounds, alpha, beta)
    print(f"\nSPRT Mode: {mode} | alpha={alpha}, beta={beta}")
    print("-" * 92)
    print(f"{'Interval [H0, H1]':<20} {'LLR':>9}   {'Bounds':<14} {'Est@0Elo':>10} {'Est@Cur':>10}  {'Status'}")
    print("-" * 92)
    for r in rows:
        print(
            f"{r['interval']:<20} {r['llr']:>9}   {r['bounds']:<14} "
            f"{r['est0']:>10} {r['est_cur']:>10}  {r['status']}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute Pentanomial/WDL Elo, LOS, and multi-interval SPRT LLR."
    )
    parser.add_argument("input_path", nargs="?", help="Path to .ipynb, .pgn, or stdout .txt/.log file.")
    parser.add_argument("-f", "--file", dest="file_opt", help="Explicit file path (.ipynb/.pgn/.txt/.log).")
    parser.add_argument("--target", default="New", help="Target engine name in PGN/logs (default: New).")
    parser.add_argument("--penta", nargs="+", help="Pentanomial counts: LL LD DD/WL WD WW (comma or space separated).")
    parser.add_argument("--wdl", nargs="+", help="WDL counts: W D L (comma or space separated).")
    parser.add_argument("-w", "--wins", type=int, help="Win count (use with -d and -l).")
    parser.add_argument("-d", "--draws", type=int, default=0, help="Draw count (use with -w and -l).")
    parser.add_argument("-l", "--losses", type=int, help="Loss count (use with -w and -d).")
    parser.add_argument(
        "-b", "--bounds", nargs="+", help="Custom SPRT bounds elo0,elo1 (e.g. --bounds -1.75,0.25 -2,0)."
    )
    parser.add_argument(
        "--only-custom", action="store_true", help="Show only --bounds instead of appending to presets."
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="False-positive rate alpha (default: 0.05).")
    parser.add_argument("--beta", type=float, default=0.05, help="False-negative rate beta (default: 0.05).")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        cleaned_argv, raw_bounds = _extract_bounds_argv(raw_argv)
    except ValueError as exc:
        parser.error(str(exc))
    args = parser.parse_args(cleaned_argv)
    if args.bounds:
        raw_bounds.extend(args.bounds)

    path_str = args.file_opt or args.input_path
    has_wdl_flags = args.wins is not None or args.losses is not None
    modes_used = int(bool(path_str)) + int(bool(args.penta)) + int(bool(args.wdl) or has_wdl_flags)
    if modes_used > 1:
        parser.error("Choose only one input mode: file path, --penta, or --wdl / -w -d -l.")

    try:
        bounds = _build_bounds_list(raw_bounds, args.only_custom)
        if not (0.0 < args.alpha < 0.5 and 0.0 < args.beta < 0.5):
            raise ValueError("alpha and beta must be in (0, 0.5).")
    except ValueError as exc:
        parser.error(str(exc))

    if args.penta:
        try:
            ll, ld, dd, wd, ww = _parse_int_list(args.penta, 5, "--penta")
        except ValueError as exc:
            parser.error(str(exc))
        if ll + ld + dd + wd + ww <= 0:
            parser.error("--penta counts sum to 0.")
        # Approximate W/D/L assuming DD/WL split is mostly draws (reported alongside exact pair counts).
        # Exact pair_scores are fully determined by the 5 pentanomial buckets anyway!
        pair_scores = [0.0] * ll + [0.5] * ld + [1.0] * dd + [1.5] * wd + [2.0] * ww
        approx_w = 2 * ww + wd
        approx_l = 2 * ll + ld
        approx_d = 2 * dd + wd + ld
        data = MatchData(
            source_desc="CLI --penta (note: DD/WL bucket counted under draws for WDL display)",
            target_engine=args.target,
            wins=approx_w,
            draws=approx_d,
            losses=approx_l,
            penta=(ll, ld, dd, wd, ww),
            pair_scores=pair_scores,
        )
    elif args.wdl or has_wdl_flags:
        if args.wdl:
            try:
                w, d, l = _parse_int_list(args.wdl, 3, "--wdl")
            except ValueError as exc:
                parser.error(str(exc))
        else:
            if args.wins is None or args.losses is None:
                parser.error("-w/--wins and -l/--losses are required together.")
            w, d, l = int(args.wins), int(args.draws), int(args.losses)
            if w < 0 or d < 0 or l < 0:
                parser.error("WDL counts must be non-negative.")
        if w + d + l <= 0:
            parser.error("Total games must be > 0.")
        data = MatchData(source_desc="CLI --wdl", target_engine=args.target, wins=w, draws=d, losses=l)
    elif path_str:
        p = Path(path_str)
        if not p.exists():
            parser.error(f"File not found: {p}")
        data = _load_file(p, args.target)
        if data is None or data.total_games == 0:
            raise SystemExit(f"No tournament games/results parsed from {p}.")
    else:
        data = _discover_latest_source(args.target)

    _print_report(data, bounds, args.alpha, args.beta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
