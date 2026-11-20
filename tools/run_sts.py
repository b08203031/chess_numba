#!/usr/bin/env python3
"""Strategic Test Suite (STS) runner for Classical and UCI chess engines.

Evaluates positional/strategic understanding across the 15 STS themes (1,500
positions in v3 or 1,188 Stockfish-15 filtered positions in v6).

Default backend uses ``chess_engine.classical`` in-process via
``tools.match_core.InProcessEngineFactory`` so Numba JIT compiles/loads once
and a ThreadPool (``-c N``) runs positions in parallel with independent
``SearchContext`` instances (Numba releases the GIL).

Examples:
  # Quick smoke test (5 positions per theme = 75 positions, 4 threads)
  python tools/run_sts.py --per-theme 5 --limit n50000 -c 4

  # Full STS v6 (default: 1,188 SF15 positions, max 100 pts) at 50k nodes/pos
  python tools/run_sts.py --limit n50000 -c 4 --out-json tournament_analysis/sts_classical_v6_n50k.json

  # Full STS v3 (1,500 original positions) + CCRL 40/4 rating estimate mode
  python tools/run_sts.py --suite v3 --getrating -c 1

  # Compare classical vs frozen classical_old baseline
  python tools/run_sts.py --engine classical_old --limit n50000 -c 4 --out-json tournament_analysis/sts_old_n50k.json
  python tools/run_sts.py --engine classical --limit n50000 -c 4 --compare-json tournament_analysis/sts_old_n50k.json
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import sys
import threading
import time
import timeit
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Optional, Sequence

import chess

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.match_core import (
    InProcessEngineFactory,
    SearchLimit,
    UciEngineFactory,
    parse_limit_token,
)


ROOT = Path(__file__).resolve().parents[1]
STS_DIR = ROOT / "data" / "sts"

SUITE_FILES = {
    "v3": STS_DIR / "STS1-STS15_LAN_v3.epd",
    "v6": STS_DIR / "STS1-STS15_LAN_v6.epd",
}

STS_THEMES: dict[int, str] = {
    1: "Undermining",
    2: "Open Files and Diagonals",
    3: "Knight Outposts",
    4: "Square Vacancy",
    5: "Bishop vs Knight",
    6: "Re-Capturing",
    7: "Offer of Simplification",
    8: "Advancement of f/g/h Pawns",
    9: "Advancement of a/b/c Pawns",
    10: "Simplification",
    11: "Activity of the King",
    12: "Center Control",
    13: "Pawn Play in the Center",
    14: "Queens and Rooks to the 7th Rank",
    15: "Avoid Pointless Exchange",
}

# Calibration constants from fsmosca/STS-Rating (sts_rating.py)
_STS_REF_BENCH_SEC = 2.5534  # Intel i7-2600K reference
_STS_REF_MOVETIME_MS = 200.0
_STS_RATING_SLOPE = 44.523
_STS_RATING_INTERCEPT = -242.85

_THEME_ID_RE = re.compile(r"STS\(v(\d+)\.\d+\)")


@dataclass(frozen=True)
class StsPosition:
    """Parsed single STS EPD test position."""

    index: int
    fen: str
    epd_id: str
    theme_id: int
    move_scores: dict[str, int]
    best_move: str
    max_points: int
    raw_line: str


@dataclass(frozen=True)
class StsEvalResult:
    """Search outcome for a single STS position."""

    pos: StsPosition
    engine_move: str
    points: int
    is_best1: bool
    depth: Optional[int]
    nodes: Optional[int]
    time_ms: Optional[int]
    info: str


def _parse_env(items: Sequence[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--env expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("--env contains an empty key")
        values[key] = value
    return values


def _parse_uci_options(items: Sequence[str]) -> dict[str, object]:
    options: dict[str, object] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--uci-option expects KEY=VALUE, got {item!r}")
        key, raw_val = item.split("=", 1)
        key = key.strip()
        val_str = raw_val.strip()
        if not key:
            raise ValueError("--uci-option contains an empty key")
        if val_str.lower() in {"true", "false"}:
            options[key] = val_str.lower() == "true"
        else:
            try:
                options[key] = int(val_str)
            except ValueError:
                options[key] = val_str
    return options


def _parse_theme_filter(raw: Optional[str]) -> Optional[set[int]]:
    if not raw:
        return None
    selected: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if lo > hi:
                lo, hi = hi, lo
            for t in range(lo, hi + 1):
                selected.add(t)
        else:
            selected.add(int(part))
    return selected


def load_sts_positions(
    epd_path: Path,
    *,
    themes: Optional[set[int]] = None,
    per_theme: Optional[int] = None,
    max_positions: Optional[int] = None,
) -> list[StsPosition]:
    """Load and validate STS EPD positions using python-chess."""
    if not epd_path.is_file():
        raise FileNotFoundError(f"STS EPD file not found: {epd_path}")

    positions: list[StsPosition] = []
    theme_counts: dict[int, int] = defaultdict(int)

    with epd_path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            board = chess.Board()
            ops = board.set_epd(line)
            epd_id = str(ops.get("id", f"pos_{line_no}"))
            match = _THEME_ID_RE.search(epd_id)
            theme_id = int(match.group(1)) if match else 0

            if themes is not None and theme_id not in themes:
                continue
            if per_theme is not None and theme_counts[theme_id] >= per_theme:
                continue

            c8_raw = str(ops.get("c8", "")).strip()
            c9_raw = str(ops.get("c9", "")).strip()
            if not c8_raw or not c9_raw:
                raise ValueError(f"Line {line_no} in {epd_path} missing c8/c9 opcodes: {line!r}")

            scores = [int(tok) for tok in c8_raw.split()]
            moves = c9_raw.split()
            if len(scores) != len(moves):
                raise ValueError(
                    f"Line {line_no} in {epd_path} has mismatched c8 ({len(scores)}) "
                    f"and c9 ({len(moves)}) lengths"
                )

            move_scores: dict[str, int] = {}
            for mv, pts in zip(moves, scores):
                # Keep highest score if a duplicate move string ever appears
                if mv not in move_scores or pts > move_scores[mv]:
                    move_scores[mv] = pts

            best_move = moves[0]
            max_points = max(scores)
            theme_counts[theme_id] += 1

            positions.append(
                StsPosition(
                    index=len(positions),
                    fen=board.fen(),
                    epd_id=epd_id,
                    theme_id=theme_id,
                    move_scores=move_scores,
                    best_move=best_move,
                    max_points=max_points,
                    raw_line=line,
                )
            )
            if max_positions is not None and len(positions) >= max_positions:
                break

    if not positions:
        raise ValueError(f"No STS positions matched filters in {epd_path}")
    return positions


def parse_info_metrics(info: str) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """Extract (depth, nodes, time_ms) from a match_core info summary string."""
    depth: Optional[int] = None
    nodes: Optional[int] = None
    time_ms: Optional[int] = None
    if not info:
        return depth, nodes, time_ms

    for part in info.split(","):
        token = part.strip()
        if token.startswith("d="):
            try:
                depth = int(token[2:])
            except ValueError:
                pass
        elif token.startswith("n="):
            try:
                nodes = int(token[2:])
            except ValueError:
                pass
        elif token.startswith("t=") and token.endswith("ms"):
            try:
                time_ms = int(token[2:-2])
            except ValueError:
                pass
    return depth, nodes, time_ms


def calibrate_sts_movetime_ms() -> tuple[float, int]:
    """Calibrate movetime (ms) using fsmosca/STS-Rating's CPU benchmark."""
    bench_sec = timeit.timeit('"-".join(str(n) for n in range(100))', number=100000)
    movetime_ms = max(50, int(round(_STS_REF_MOVETIME_MS * bench_sec / _STS_REF_BENCH_SEC)))
    return bench_sec, movetime_ms


def estimate_ccrl_elo(score_percent: float) -> float:
    """STS-Rating linear mapping to CCRL 40/4 Elo (valid on v3 with --getrating)."""
    return _STS_RATING_SLOPE * score_percent + _STS_RATING_INTERCEPT


def _evaluate_position(
    engine: object,
    pos: StsPosition,
    limit: SearchLimit,
) -> StsEvalResult:
    engine.new_game(pos.fen)
    move, info = engine.get_move(limit)
    uci = move.uci()
    points = pos.move_scores.get(uci, 0)
    depth, nodes, time_ms = parse_info_metrics(info)
    return StsEvalResult(
        pos=pos,
        engine_move=uci,
        points=points,
        is_best1=(uci == pos.best_move),
        depth=depth,
        nodes=nodes,
        time_ms=time_ms,
        info=info,
    )


def run_sts_benchmark(
    positions: Sequence[StsPosition],
    factory: object,
    limit: SearchLimit,
    *,
    concurrency: int = 1,
    progress_every: int = 50,
) -> tuple[list[StsEvalResult], float]:
    """Run all STS positions using a pool of independent engine instances."""
    num_workers = max(1, min(int(concurrency), len(positions)))
    engines: list[object] = []
    engine_pool: queue.Queue[object] = queue.Queue()

    try:
        for worker_id in range(num_workers):
            engine = factory.create(worker_id)
            engine.start()
            factory.prepare(engine, worker_id)
            engines.append(engine)
            engine_pool.put(engine)

        results: list[Optional[StsEvalResult]] = [None] * len(positions)
        completed = 0
        running_score = 0
        running_max = 0
        running_best1 = 0
        lock = threading.Lock()
        started = time.perf_counter()

        def worker_task(pos: StsPosition) -> StsEvalResult:
            eng = engine_pool.get()
            try:
                return _evaluate_position(eng, pos, limit)
            finally:
                engine_pool.put(eng)

        if num_workers == 1:
            for pos in positions:
                res = worker_task(pos)
                results[pos.index] = res
                completed += 1
                running_score += res.points
                running_max += pos.max_points
                running_best1 += int(res.is_best1)
                if progress_every > 0 and (completed % progress_every == 0 or completed == len(positions)):
                    pct = 100.0 * running_score / running_max if running_max else 0.0
                    elapsed = time.perf_counter() - started
                    print(
                        f"[{completed:4d}/{len(positions)}] "
                        f"score={pct:5.2f}% ({running_score}/{running_max}) "
                        f"best1={running_best1}/{completed} "
                        f"elapsed={elapsed:5.1f}s",
                        flush=True,
                    )
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(worker_task, pos) for pos in positions]
                for fut in as_completed(futures):
                    res = fut.result()
                    results[res.pos.index] = res
                    with lock:
                        completed += 1
                        running_score += res.points
                        running_max += res.pos.max_points
                        running_best1 += int(res.is_best1)
                        if progress_every > 0 and (completed % progress_every == 0 or completed == len(positions)):
                            pct = 100.0 * running_score / running_max if running_max else 0.0
                            elapsed = time.perf_counter() - started
                            print(
                                f"[{completed:4d}/{len(positions)}] "
                                f"score={pct:5.2f}% ({running_score}/{running_max}) "
                                f"best1={running_best1}/{completed} "
                                f"elapsed={elapsed:5.1f}s",
                                flush=True,
                            )

        duration_s = time.perf_counter() - started
        final_results = [r for r in results if r is not None]
        return final_results, duration_s
    finally:
        for eng in engines:
            try:
                eng.terminate()
            except Exception:
                pass


def summarize_results(
    results: Sequence[StsEvalResult],
    *,
    engine_label: str,
    suite_label: str,
    limit: SearchLimit,
    duration_s: float,
    include_elo: bool = False,
) -> dict[str, object]:
    """Aggregate per-theme and overall STS statistics."""
    by_theme: dict[int, list[StsEvalResult]] = defaultdict(list)
    for res in results:
        by_theme[res.pos.theme_id].append(res)

    theme_summaries: list[dict[str, object]] = []
    for theme_id in sorted(by_theme):
        group = by_theme[theme_id]
        pos_cnt = len(group)
        score_sum = sum(r.points for r in group)
        max_sum = sum(r.pos.max_points for r in group)
        best1_cnt = sum(1 for r in group if r.is_best1)
        pct = 100.0 * score_sum / max_sum if max_sum else 0.0
        depths = [r.depth for r in group if r.depth is not None]
        nodes_list = [r.nodes for r in group if r.nodes is not None]
        entry: dict[str, object] = {
            "theme_id": theme_id,
            "theme_name": STS_THEMES.get(theme_id, f"Theme {theme_id}"),
            "positions": pos_cnt,
            "best1": best1_cnt,
            "best1_pct": 100.0 * best1_cnt / pos_cnt if pos_cnt else 0.0,
            "score": score_sum,
            "max_score": max_sum,
            "score_pct": pct,
            "avg_depth": sum(depths) / len(depths) if depths else None,
            "avg_nodes": sum(nodes_list) / len(nodes_list) if nodes_list else None,
        }
        if include_elo:
            entry["est_elo"] = estimate_ccrl_elo(pct)
        theme_summaries.append(entry)

    total_pos = len(results)
    total_score = sum(r.points for r in results)
    total_max = sum(r.pos.max_points for r in results)
    total_best1 = sum(1 for r in results if r.is_best1)
    overall_pct = 100.0 * total_score / total_max if total_max else 0.0

    all_depths = [r.depth for r in results if r.depth is not None]
    all_nodes = [r.nodes for r in results if r.nodes is not None]
    all_times = [r.time_ms for r in results if r.time_ms is not None]
    total_nodes = sum(all_nodes) if all_nodes else 0
    total_search_ms = sum(all_times) if all_times else 0
    nps = int(total_nodes * 1000 / total_search_ms) if total_search_ms > 0 else None

    overall: dict[str, object] = {
        "positions": total_pos,
        "best1": total_best1,
        "best1_pct": 100.0 * total_best1 / total_pos if total_pos else 0.0,
        "score": total_score,
        "max_score": total_max,
        "score_pct": overall_pct,
        "avg_depth": sum(all_depths) / len(all_depths) if all_depths else None,
        "total_nodes": total_nodes if all_nodes else None,
        "avg_nodes": total_nodes / len(all_nodes) if all_nodes else None,
        "nps": nps,
        "duration_s": duration_s,
    }
    if include_elo:
        overall["est_elo"] = estimate_ccrl_elo(overall_pct)

    return {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "engine": engine_label,
            "suite": suite_label,
            "limit": limit.describe(),
            "limit_token": limit.token(),
            "include_elo": include_elo,
        },
        "overall": overall,
        "themes": theme_summaries,
        "positions": [
            {
                "id": r.pos.epd_id,
                "theme_id": r.pos.theme_id,
                "engine_move": r.engine_move,
                "best_move": r.pos.best_move,
                "points": r.points,
                "max_points": r.pos.max_points,
                "is_best1": r.is_best1,
                "depth": r.depth,
                "nodes": r.nodes,
                "time_ms": r.time_ms,
            }
            for r in results
        ],
    }


def print_summary(summary: Mapping[str, object], baseline: Optional[Mapping[str, object]] = None) -> None:
    """Print formatted STS summary and optional baseline comparison."""
    meta = summary["meta"]
    overall = summary["overall"]
    themes = list(summary["themes"])
    include_elo = bool(meta.get("include_elo", False))

    print("\n" + "=" * 86)
    print(
        f"STS Report | Engine: {meta['engine']} | Suite: {meta['suite']} | "
        f"Limit: {meta['limit']}"
    )
    print("=" * 86)

    header = (
        f"{'STS':>5}  {'Theme':<33} {'Pos':>4} {'Best1':>6} "
        f"{'Score':>12} {'Score%':>7} {'AvgD':>5}"
    )
    if include_elo:
        header += f" {'Elo':>6}"
    print(header)
    print("-" * len(header))

    for t in themes:
        avg_d = f"{t['avg_depth']:5.1f}" if t["avg_depth"] is not None else "    -"
        score_col = f"{t['score']}/{t['max_score']}"
        line = (
            f"STS{t['theme_id']:02d}  {t['theme_name']:<33} {t['positions']:>4d} "
            f"{t['best1']:>6d} {score_col:>12} {t['score_pct']:6.2f}% {avg_d}"
        )
        if include_elo:
            line += f" {t['est_elo']:6.0f}"
        print(line)

    print("-" * len(header))
    avg_d_all = f"{overall['avg_depth']:5.1f}" if overall["avg_depth"] is not None else "    -"
    score_col_all = f"{overall['score']}/{overall['max_score']}"
    all_line = (
        f"{'ALL':>5}  {'All Themes':<33} {overall['positions']:>4d} "
        f"{overall['best1']:>6d} {score_col_all:>12} {overall['score_pct']:6.2f}% {avg_d_all}"
    )
    if include_elo:
        all_line += f" {overall['est_elo']:6.0f}"
    print(all_line)

    extras = [
        f"Best1 Rate: {overall['best1_pct']:.2f}%",
        f"Duration: {overall['duration_s']:.1f}s",
    ]
    if overall["avg_nodes"] is not None:
        extras.append(f"Avg Nodes/Pos: {overall['avg_nodes']:.0f}")
    if overall["nps"] is not None:
        extras.append(f"Search NPS: {overall['nps']:,}")
    print("\n" + " | ".join(extras))

    if include_elo:
        print(
            f"Estimated CCRL 40/4 Rating (STS-Rating linear formula): "
            f"{overall['est_elo']:.0f} Elo"
        )

    if len(themes) >= 5:
        ranked = sorted(themes, key=lambda x: x["score_pct"], reverse=True)
        print("\nTop 5 Strongest Themes:")
        for i, t in enumerate(ranked[:5], 1):
            print(f"  {i}. STS{t['theme_id']:02d} {t['theme_name']}: {t['score_pct']:.2f}%")
        print("Top 5 Weakest Themes:")
        for i, t in enumerate(reversed(ranked[-5:]), 1):
            print(f"  {i}. STS{t['theme_id']:02d} {t['theme_name']}: {t['score_pct']:.2f}%")

    if baseline is not None:
        base_themes = {t["theme_id"]: t for t in baseline.get("themes", [])}
        base_overall = baseline.get("overall", {})
        base_meta = baseline.get("meta", {})
        print("\n" + "-" * 86)
        print(
            f"Comparison vs Baseline ({base_meta.get('engine', 'baseline')}, "
            f"{base_meta.get('limit', '?')}):"
        )
        print(
            f"{'STS':>5}  {'Theme':<33} {'Base%':>7} {'Curr%':>7} "
            f"{'Delta%':>8} {'dBest1':>7}"
        )
        print("-" * 74)
        for t in themes:
            bt = base_themes.get(t["theme_id"])
            if bt is None:
                continue
            dpct = t["score_pct"] - bt["score_pct"]
            dbest = t["best1"] - bt["best1"]
            print(
                f"STS{t['theme_id']:02d}  {t['theme_name']:<33} "
                f"{bt['score_pct']:6.2f}% {t['score_pct']:6.2f}% "
                f"{dpct:+7.2f}% {dbest:+7d}"
            )
        if "score_pct" in base_overall:
            dpct_all = overall["score_pct"] - base_overall["score_pct"]
            dbest_all = overall["best1"] - base_overall["best1"]
            print("-" * 74)
            print(
                f"{'ALL':>5}  {'All Themes':<33} "
                f"{base_overall['score_pct']:6.2f}% {overall['score_pct']:6.2f}% "
                f"{dpct_all:+7.2f}% {dbest_all:+7d}"
            )


def _build_factory(args: argparse.Namespace) -> tuple[object, str]:
    env_extra = _parse_env(args.env)
    if args.uci:
        uci_path = Path(args.uci)
        if not uci_path.is_file():
            raise FileNotFoundError(f"UCI engine not found: {uci_path}")
        options = _parse_uci_options(args.uci_option)
        if str(uci_path).lower().endswith(".py"):
            options.setdefault("OwnBook", False)
        label = args.name or uci_path.stem
        factory = UciEngineFactory(
            label,
            str(uci_path),
            options=options,
            env_extra=env_extra,
        )
        return factory, label

    namespace = (
        "chess_engine.classical"
        if args.engine == "classical"
        else "chess_engine.classical_old"
    )
    label = args.name or args.engine
    factory = InProcessEngineFactory(
        label,
        namespace,
        tt_mb=args.tt_mb,
        env_extra=env_extra,
    )
    return factory, label


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Strategic Test Suite (STS) on chess_engine.classical or UCI engines.",
    )
    parser.add_argument(
        "--suite",
        choices=sorted(SUITE_FILES.keys()),
        default="v6",
        help="STS dataset version: v6 (1188 SF15 pos, max 100 pts; default) or v3 (1500 original pos, max 10 pts, CCRL formula).",
    )
    parser.add_argument(
        "--epd",
        type=Path,
        default=None,
        help="Custom EPD file path (overrides --suite).",
    )
    parser.add_argument(
        "--engine",
        choices=("classical", "classical_old"),
        default="classical",
        help="In-process engine namespace (default: classical).",
    )
    parser.add_argument(
        "--uci",
        type=str,
        default=None,
        help="Path to an external UCI executable or Python UCI script (overrides --engine).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Display name for the engine in reports.",
    )
    parser.add_argument(
        "--limit",
        type=str,
        default=None,
        help="Search limit token: n<nodes> (e.g. n50000), d<depth> (e.g. d10), or <ms> (e.g. 200). Default: n50000.",
    )
    parser.add_argument(
        "--getrating",
        action="store_true",
        help="Use STS-Rating calibrated movetime and compute estimated CCRL 40/4 Elo (calibrated for v3).",
    )
    parser.add_argument(
        "-c",
        "--concurrency",
        type=int,
        default=1,
        help="Number of parallel worker contexts/processes (default: 1).",
    )
    parser.add_argument(
        "--tt-mb",
        type=int,
        default=64,
        help="Transposition table size in MB per worker for in-process engines (default: 64).",
    )
    parser.add_argument(
        "--themes",
        type=str,
        default=None,
        help="Comma/range list of theme IDs to run, e.g. '1,2,11' or '1-5'.",
    )
    parser.add_argument(
        "--per-theme",
        type=int,
        default=None,
        help="Limit to first N positions per theme (useful for quick smoke tests).",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=None,
        help="Limit total number of positions evaluated.",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="KEY=VALUE environment override passed to engine contexts.",
    )
    parser.add_argument(
        "--uci-option",
        action="append",
        default=[],
        help="KEY=VALUE UCI setoption for --uci engines.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress every N positions (0 disables; default: 50).",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional path to save full JSON report.",
    )
    parser.add_argument(
        "--compare-json",
        type=Path,
        default=None,
        help="Optional baseline JSON report to compare against.",
    )
    parser.add_argument(
        "--miss-epd",
        type=Path,
        default=None,
        help="Optional path to write EPD positions where engine scored 0 (or < max with --miss-mode suboptimal).",
    )
    parser.add_argument(
        "--miss-mode",
        choices=("zero", "suboptimal"),
        default="zero",
        help="Which positions to export with --miss-epd (default: zero).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    epd_path = args.epd if args.epd is not None else SUITE_FILES[args.suite]
    suite_label = epd_path.name if args.epd is not None else f"STS-{args.suite} ({epd_path.name})"
    theme_filter = _parse_theme_filter(args.themes)

    positions = load_sts_positions(
        epd_path,
        themes=theme_filter,
        per_theme=args.per_theme,
        max_positions=args.max_positions,
    )

    if args.getrating and args.limit is None:
        bench_sec, cal_ms = calibrate_sts_movetime_ms()
        limit = SearchLimit(time_ms=cal_ms)
        print(
            f"[calibrate] python bench={bench_sec:.4f}s (ref={_STS_REF_BENCH_SEC:.4f}s) "
            f"-> movetime={cal_ms}ms",
            flush=True,
        )
        if args.suite != "v3" and args.epd is None:
            print("[warn] --getrating Elo formula is calibrated for --suite v3; pass --suite v3 for Elo estimation.", flush=True)
        if args.concurrency > 1:
            print("[warn] --getrating is normally run with -c 1 to avoid CPU contention.", flush=True)
    else:
        limit = parse_limit_token(args.limit or "n50000")

    factory, engine_label = _build_factory(args)
    print(
        f"[sts] engine={engine_label} | positions={len(positions)} | "
        f"limit={limit.describe()} | workers={max(1, min(args.concurrency, len(positions)))}",
        flush=True,
    )

    results, duration_s = run_sts_benchmark(
        positions,
        factory,
        limit,
        concurrency=args.concurrency,
        progress_every=args.progress_every,
    )

    include_elo = bool(args.getrating and (args.epd is not None or args.suite == "v3"))
    summary = summarize_results(
        results,
        engine_label=engine_label,
        suite_label=suite_label,
        limit=limit,
        duration_s=duration_s,
        include_elo=include_elo,
    )

    baseline = None
    if args.compare_json is not None:
        baseline = json.loads(args.compare_json.read_text(encoding="utf-8"))

    print_summary(summary, baseline=baseline)

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with args.out_json.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"\n[saved] JSON report -> {args.out_json}")

    if args.miss_epd is not None:
        args.miss_epd.parent.mkdir(parents=True, exist_ok=True)
        missed = [
            r.pos.raw_line
            for r in results
            if (r.points == 0 if args.miss_mode == "zero" else r.points < r.pos.max_points)
        ]
        with args.miss_epd.open("w", encoding="utf-8", newline="\n") as f:
            for line in missed:
                f.write(line + "\n")
        print(f"[saved] {len(missed)} missed positions -> {args.miss_epd}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
