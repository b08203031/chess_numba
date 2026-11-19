"""Tournament runner for the current and baseline Classical engines.

Default mode imports ``classical`` and ``classical_old`` in one Python process.
Each worker owns two independent SearchContexts, while the compiled Numba code
is shared by every worker in the same backend namespace.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.match_core import (
    InProcessEngineFactory,
    SearchLimit,
    StageResult,
    UciEngine,
    UciEngineFactory,
    SPRTTest,
    calculate_elo_statistics,
    clear_numba_cache,
    close_engine_pairs,
    create_engine_pairs,
    load_openings,
    parse_info_line,
    parse_limit_token,
    run_stage,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENGINE1 = ROOT / "main.py"
DEFAULT_ENGINE2 = ROOT / "main_old.py"
DEFAULT_NODES = 50_000
DEFAULT_TT_MB = 128

# Backwards-compatible name for scripts that imported the old UCI wrapper.
Engine = UciEngine


def _parse_env(items: Sequence[str], option: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"{option} expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"{option} contains an empty key")
        values[key] = value
    return values


def parse_limit(limit_str: str) -> dict[str, Optional[int]]:
    """Compatibility wrapper around the validated shared limit parser."""
    limit = parse_limit_token(limit_str)
    return {"time": limit.time_ms, "depth": limit.depth, "nodes": limit.nodes}


def parse_stages(stages_str: str) -> list[dict[str, object]]:
    """Parse ``games:limit`` or ``games:limit1:limit2`` stages; every stage must contain full color pairs."""
    stages: list[dict[str, object]] = []
    for raw in stages_str.split(","):
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(
                f"Invalid stage {raw!r}; expected games:limit or games:limit1:limit2"
            )
        games = int(parts[0])
        if games < 2 or games % 2:
            raise ValueError(f"Stage games must be an even integer >= 2, got {games}")
        limit1 = parse_limit_token(parts[1])
        limit2 = parse_limit_token(parts[2]) if len(parts) == 3 else limit1
        limit_str = (
            limit1.token()
            if limit1 == limit2
            else f"{limit1.token()}_vs_{limit2.token()}"
        )
        stages.append({
            "games": games,
            "limit": limit1,
            "limit1": limit1,
            "limit2": limit2,
            "limit_str": limit_str,
        })
    if not stages:
        raise ValueError("No stages were parsed")
    return stages


def _safe_token(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def _stage_pgn_path(
    base: Optional[str],
    stage_number: int,
    limit: SearchLimit,
    multi: bool,
    *,
    second_limit: Optional[SearchLimit] = None,
) -> str:
    output_dir = ROOT / "tournament_analysis"
    token = (
        limit.token()
        if second_limit is None or second_limit == limit
        else f"{limit.token()}_vs_{second_limit.token()}"
    )
    if base:
        path = Path(base).resolve()
        if not multi:
            return str(path)
        suffix = path.suffix or ".pgn"
        return str(path.with_name(f"{path.stem}_stage{stage_number}_{_safe_token(token)}{suffix}"))
    if multi:
        return str(output_dir / f"tournament_results_stage{stage_number}_{_safe_token(token)}.pgn")
    return str(output_dir / "tournament_results.pgn")


def _is_default_pair(engine1_path: str, engine2_path: str) -> bool:
    return (
        Path(engine1_path).resolve() == DEFAULT_ENGINE1.resolve()
        and Path(engine2_path).resolve() == DEFAULT_ENGINE2.resolve()
    )


def _select_backend(
    requested: str,
    engine1_path: str,
    engine2_path: str,
    own_book1: bool,
    own_book2: bool,
) -> str:
    if requested != "auto":
        selected = requested
    else:
        selected = "inprocess" if _is_default_pair(engine1_path, engine2_path) else "uci"
    if selected == "inprocess" and (own_book1 or own_book2):
        raise ValueError("In-process matches use external opening positions; OwnBook requires --backend uci")
    return selected


def _print_result(result: StageResult, first_name: str, second_name: str) -> None:
    elo, error = calculate_elo_statistics(result.wins, result.draws, result.losses)
    print("\n=== Stage Finished ===")
    print(f"Games: {result.games}")
    print(
        f"{first_name}: {result.points:.1f} "
        f"({result.wins}W {result.draws}D {result.losses}L)"
    )
    print(f"{second_name}: {result.games - result.points:.1f}")
    if math.isinf(elo):
        print("Elo difference: beyond finite estimate (perfect or zero score)")
    else:
        print(f"Elo difference: {elo:+.2f} ±{error:.2f} (95% CI)")
    print(f"Status: {result.status}; LLR={result.llr:.2f}")
    print(f"PGN: {result.pgn_path}")


def _validate_positive_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer >= 1, got {value!r}")
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer >= 1, got {value!r}") from exc
    if parsed < 1:
        raise ValueError(f"{label} must be >= 1")
    return parsed


def run_tournament(
    engine1_path: str,
    engine2_path: str,
    games_count: int,
    time_ms: Optional[int],
    engine1_name: str = "Engine_A",
    engine2_name: str = "Engine_B",
    depth: Optional[int] = None,
    nodes: Optional[int] = None,
    concurrency: int = 1,
    pgn_file: Optional[str] = None,
    engine1_env: Optional[Mapping[str, object]] = None,
    engine2_env: Optional[Mapping[str, object]] = None,
    external_barriers: object = None,
    early_trash: bool = True,
    early_trash_min_games: int = 80,
    early_trash_max_score_pct: float = 40.0,
    early_trash_max_elo: object = None,
    own_book1: bool = False,
    own_book2: bool = False,
    use_openings: bool = True,
    use_sprt: bool = True,
    openings_pgn: Optional[str] = None,
    opening_plies: int = 16,
    stages: Optional[Sequence[Mapping[str, object]]] = None,
    *,
    backend: str = "auto",
    engine1_module: str = "chess_engine.classical",
    engine2_module: str = "chess_engine.classical_old",
    tt_mb: int = DEFAULT_TT_MB,
    threads1: int = 1,
    threads2: int = 1,
    nodes1: Optional[int] = None,
    depth1: Optional[int] = None,
    time1_ms: Optional[int] = None,
    nodes2: Optional[int] = None,
    depth2: Optional[int] = None,
    time2_ms: Optional[int] = None,
    max_plies: int = 400,
    search_timeout_s: float = 300.0,
    sprt_elo0: float = 0.0,
    sprt_elo1: float = 10.0,
    seed: int = 0,
    adjudicate: bool = True,
    adjudicate_win_score: int = 500,
    adjudicate_win_plies: int = 10,
    adjudicate_draw_score: int = 8,
    adjudicate_draw_min_move: int = 40,
    adjudicate_draw_plies: int = 12,
) -> list[StageResult]:
    """Run one or more stages while keeping the same worker pool alive."""
    del early_trash_max_elo  # Kept in the public signature for compatibility.
    if external_barriers is not None:
        raise ValueError("external_barriers belonged to the removed subprocess-JIT design")
    concurrency = _validate_positive_int(concurrency, "concurrency")
    tt_mb = _validate_positive_int(tt_mb, "tt_mb")
    threads1 = _validate_positive_int(threads1, "threads1")
    threads2 = _validate_positive_int(threads2, "threads2")
    max_plies = _validate_positive_int(max_plies, "max_plies")
    if float(search_timeout_s) <= 0.0:
        raise ValueError("search_timeout_s must be > 0")
    if adjudicate_win_score < 0 or adjudicate_draw_score < 0:
        raise ValueError("adjudicate scores must be >= 0")
    _validate_positive_int(adjudicate_win_plies, "adjudicate_win_plies")
    _validate_positive_int(adjudicate_draw_min_move, "adjudicate_draw_min_move")
    _validate_positive_int(adjudicate_draw_plies, "adjudicate_draw_plies")

    if stages:
        normalized: list[dict[str, object]] = []
        for stage in stages:
            if (
                "limit1" in stage
                or "limit2" in stage
                or "first_limit" in stage
                or "second_limit" in stage
            ):
                raw_l1 = stage.get("limit1", stage.get("first_limit", stage.get("limit")))
                raw_l2 = stage.get("limit2", stage.get("second_limit", raw_l1))
                limit1 = (
                    raw_l1
                    if isinstance(raw_l1, SearchLimit)
                    else SearchLimit.from_mapping(raw_l1)  # type: ignore[arg-type]
                )
                limit2 = (
                    raw_l2
                    if isinstance(raw_l2, SearchLimit)
                    else SearchLimit.from_mapping(raw_l2)  # type: ignore[arg-type]
                )
            elif isinstance(stage.get("limit"), SearchLimit):
                limit1 = limit2 = stage["limit"]
            else:
                limit1 = limit2 = SearchLimit.from_mapping(stage)
            normalized.append({
                "games": int(stage["games"]),
                "limit1": limit1,
                "limit2": limit2,
                "pgn_file": stage.get("pgn_file"),
            })
    else:
        if nodes is not None:
            shared_limit: Optional[SearchLimit] = SearchLimit(nodes=nodes)
        elif depth is not None:
            shared_limit = SearchLimit(depth=depth)
        elif time_ms is not None:
            shared_limit = SearchLimit(time_ms=time_ms)
        else:
            shared_limit = None

        def _resolve_side_limit(
            side_nodes: Optional[int],
            side_depth: Optional[int],
            side_time_ms: Optional[int],
        ) -> SearchLimit:
            if sum(value is not None for value in (side_nodes, side_depth, side_time_ms)) > 1:
                raise ValueError("Per-side limit must specify at most one of nodes/depth/time")
            if side_nodes is not None:
                return SearchLimit(nodes=side_nodes)
            if side_depth is not None:
                return SearchLimit(depth=side_depth)
            if side_time_ms is not None:
                return SearchLimit(time_ms=side_time_ms)
            if shared_limit is not None:
                return shared_limit
            return SearchLimit(nodes=DEFAULT_NODES)

        limit1 = _resolve_side_limit(nodes1, depth1, time1_ms)
        limit2 = _resolve_side_limit(nodes2, depth2, time2_ms)
        normalized = [{
            "games": int(games_count),
            "limit1": limit1,
            "limit2": limit2,
            "pgn_file": pgn_file,
        }]

    for stage in normalized:
        games = int(stage["games"])
        if games < 2 or games % 2:
            raise ValueError("Every stage must use an even game count >= 2")

    selected = _select_backend(backend, engine1_path, engine2_path, own_book1, own_book2)
    if selected == "inprocess":
        factory1 = InProcessEngineFactory(
            engine1_name,
            engine1_module,
            tt_mb=tt_mb,
            env_extra=engine1_env,
            threads=threads1,
        )
        factory2 = InProcessEngineFactory(
            engine2_name,
            engine2_module,
            tt_mb=tt_mb,
            env_extra=engine2_env,
            threads=threads2,
        )
    else:
        for path in (engine1_path, engine2_path):
            if not os.path.exists(path):
                raise FileNotFoundError(path)
        factory1 = UciEngineFactory(
            engine1_name,
            engine1_path,
            options={"OwnBook": own_book1, "Hash": tt_mb, "Threads": threads1},
            env_extra=engine1_env,
            search_timeout_s=search_timeout_s,
        )
        factory2 = UciEngineFactory(
            engine2_name,
            engine2_path,
            options={"OwnBook": own_book2, "Hash": tt_mb, "Threads": threads2},
            env_extra=engine2_env,
            search_timeout_s=search_timeout_s,
        )

    epd = str(ROOT / "data" / "openings.epd")
    openings = load_openings(
        epd_path=epd,
        pgn_path=openings_pgn,
        pgn_plies=opening_plies,
        enabled=use_openings,
    )
    random.Random(seed).shuffle(openings)
    max_pairs = max(int(stage["games"]) // 2 for stage in normalized)
    actual_concurrency = min(int(concurrency), max_pairs)
    print(
        f"Backend: {selected}; workers={actual_concurrency}; "
        f"threads={engine1_name}:{threads1} vs {engine2_name}:{threads2}; "
        f"TT={tt_mb} MB/context; openings={len(openings)}; seed={seed}",
        flush=True,
    )
    if engine1_name == engine2_name:
        print(
            f"[warning] engine1_name and engine2_name are both {engine1_name!r}; "
            "PGN White/Black headers will be indistinguishable.",
            flush=True,
        )
    cpu_count = os.cpu_count()
    peak_threads = actual_concurrency * max(threads1, threads2)
    if cpu_count is not None and peak_threads > cpu_count:
        print(
            f"[warning] Requested concurrency * max_threads ({actual_concurrency} * {max(threads1, threads2)} = "
            f"{peak_threads}) exceeds CPU count ({cpu_count}); CPU contention may affect timing and NPS.",
            flush=True,
        )
    if adjudicate:
        print(
            f"Adjudication: win >= {adjudicate_win_score} cp ({adjudicate_win_plies} plies), "
            f"draw <= {adjudicate_draw_score} cp ({adjudicate_draw_plies} plies after move {adjudicate_draw_min_move})",
            flush=True,
        )
    else:
        print("Adjudication: disabled (games play until game-over or max plies)", flush=True)
    if selected == "inprocess":
        if engine1_module == engine2_module:
            print(
                f"JIT plan: {engine1_module} once; worker contexts for both sides share compiled code.",
                flush=True,
            )
        else:
            print(
                f"JIT plan: {engine1_module} once + {engine2_module} once; worker contexts share compiled code.",
                flush=True,
            )
    print(
        "Warm-up: Kiwipete depth 14 for each Python/Numba backend, then one "
        "complete startpos game at depth 1 (all results discarded).",
        flush=True,
    )

    pool = create_engine_pairs(factory1, factory2, actual_concurrency)
    results: list[StageResult] = []
    try:
        multi = len(normalized) > 1
        for number, stage in enumerate(normalized, 1):
            limit1 = stage["limit1"]
            limit2 = stage["limit2"]
            output = stage.get("pgn_file") or _stage_pgn_path(
                pgn_file, number, limit1, multi, second_limit=limit2
            )
            limit_summary = (
                f"limit={limit1.describe()}"
                if limit1 == limit2
                else f"{engine1_name}={limit1.describe()}; {engine2_name}={limit2.describe()}"
            )
            print(
                f"\n>>> Stage {number}/{len(normalized)}: games={stage['games']}, "
                f"{limit_summary}",
                flush=True,
            )
            result = run_stage(
                pool,
                openings,
                games=int(stage["games"]),
                first_limit=limit1,
                second_limit=limit2,
                first_name=engine1_name,
                second_name=engine2_name,
                pgn_path=str(output),
                event="Engine Tournament",
                use_sprt=use_sprt,
                sprt_elo0=sprt_elo0,
                sprt_elo1=sprt_elo1,
                early_trash=early_trash,
                early_trash_min_games=early_trash_min_games,
                early_trash_max_score_pct=early_trash_max_score_pct,
                max_plies=max_plies,
                adjudicate=adjudicate,
                adjudicate_win_score=adjudicate_win_score,
                adjudicate_win_plies=adjudicate_win_plies,
                adjudicate_draw_score=adjudicate_draw_score,
                adjudicate_draw_min_move=adjudicate_draw_min_move,
                adjudicate_draw_plies=adjudicate_draw_plies,
            )
            results.append(result)
            _print_result(result, engine1_name, engine2_name)
    finally:
        close_engine_pairs(pool)
    return results


class _SharedThreadsAction(argparse.Action):
    """Apply --threads N to both sides unless overridden by --threads1/--threads2."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        if not getattr(namespace, "_threads1_explicit", False):
            namespace.threads1 = values
        if not getattr(namespace, "_threads2_explicit", False):
            namespace.threads2 = values


class _SideThreadsAction(argparse.Action):
    """Record explicit per-side thread count so --threads cannot overwrite it."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, f"_{self.dest}_explicit", True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a paired tournament between current and baseline engines.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Limits: nN=fixed nodes, dN=fixed depth, plain N=movetime ms.\n"
            "Default without a limit option: n50000.\n"
            "Examples:\n"
            "  python tools/tournament.py -c 8 --nodes 100000\n"
            "  python tools/tournament.py -c 8 --stages 200:n20000,400:n100000\n"
            "  python tools/tournament.py --engine2 main.py --name1 SMP4 --name2 SMP1 --threads1 4 --threads2 1 --nodes 50000\n"
            "  python tools/tournament.py --depth 12\n"
            "  python tools/tournament.py --time 1000"
        ),
    )
    parser.add_argument("--engine1", default=str(DEFAULT_ENGINE1))
    parser.add_argument("--engine2", default=str(DEFAULT_ENGINE2))
    parser.add_argument("--name1", default="New")
    parser.add_argument("--name2", default="Old")
    parser.add_argument("--games", type=int, default=100)
    limits = parser.add_mutually_exclusive_group()
    limits.add_argument("--nodes", type=int, help=f"fixed nodes/move (default: {DEFAULT_NODES})")
    limits.add_argument("--depth", type=int, help="fixed depth/move")
    limits.add_argument("--time", type=int, help="fixed movetime in ms/move")
    limits1 = parser.add_mutually_exclusive_group()
    limits1.add_argument("--nodes1", type=int, help="override fixed nodes/move for engine 1")
    limits1.add_argument("--depth1", type=int, help="override fixed depth/move for engine 1")
    limits1.add_argument("--time1", type=int, help="override fixed movetime in ms/move for engine 1")
    limits2 = parser.add_mutually_exclusive_group()
    limits2.add_argument("--nodes2", type=int, help="override fixed nodes/move for engine 2")
    limits2.add_argument("--depth2", type=int, help="override fixed depth/move for engine 2")
    limits2.add_argument("--time2", type=int, help="override fixed movetime in ms/move for engine 2")
    parser.add_argument("--stages", metavar="SPEC", help="games:limit or games:limit1:limit2 list, e.g. 200:n20000,400:n100000")
    parser.add_argument("--concurrency", "-c", type=int, default=1, help="parallel games/independent contexts")
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        action=_SharedThreadsAction,
        help="threads for both engines unless overridden by --threads1/--threads2",
    )
    parser.add_argument(
        "--threads1",
        type=int,
        default=1,
        action=_SideThreadsAction,
        help="search threads for engine 1 (default: 1)",
    )
    parser.add_argument(
        "--threads2",
        type=int,
        default=1,
        action=_SideThreadsAction,
        help="search threads for engine 2 (default: 1)",
    )
    parser.add_argument("--backend", choices=("auto", "inprocess", "uci"), default="auto")
    parser.add_argument("--engine1-module", default="chess_engine.classical")
    parser.add_argument("--engine2-module", default="chess_engine.classical_old")
    parser.add_argument("--tt-mb", type=int, default=DEFAULT_TT_MB, help="TT MB per engine context (default: 128)")
    parser.add_argument("--pgn", help="output PGN; multi-stage adds a stage suffix")
    parser.add_argument("--engine1-env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--engine2-env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--own-book1", action="store_true", help="UCI backend only")
    parser.add_argument("--own-book2", action="store_true", help="UCI backend only")
    parser.add_argument("--no-openings", action="store_true")
    parser.add_argument("--openings-pgn", metavar="PATH")
    parser.add_argument("--opening-moves", type=int, default=8)
    parser.add_argument("--seed", type=int, default=time.time_ns(), help="deterministic opening shuffle seed")
    parser.add_argument("--no-sprt", action="store_true")
    parser.add_argument("--sprt-elo0", type=float, default=0.0)
    parser.add_argument("--sprt-elo1", type=float, default=10.0)
    parser.add_argument("--no-early-trash", action="store_true")
    parser.add_argument("--early-trash-min-games", type=int, default=80)
    parser.add_argument("--early-trash-max-score-pct", type=float, default=40.0)
    parser.add_argument("--max-plies", type=int, default=400)
    parser.add_argument("--search-timeout", type=float, default=300.0, metavar="SECONDS")
    parser.add_argument("--clear-cache", action="store_true", help="explicitly clear Numba caches before startup")
    parser.add_argument("--no-clear-cache", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--adjudicate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="toggle early adjudication (default: enabled; --no-adjudicate disables)",
    )
    parser.add_argument(
        "--adjudicate-win-score",
        type=int,
        default=500,
        help="centipawn threshold for win adjudication (default: 500)",
    )
    parser.add_argument(
        "--adjudicate-win-moves",
        type=int,
        default=5,
        help="consecutive full moves for win adjudication (default: 5, i.e. 10 plies)",
    )
    parser.add_argument(
        "--adjudicate-draw-score",
        type=int,
        default=8,
        help="centipawn threshold for draw adjudication (default: 8)",
    )
    parser.add_argument(
        "--adjudicate-draw-min-move",
        type=int,
        default=40,
        help="minimum move number to enable draw adjudication (default: 40)",
    )
    parser.add_argument(
        "--adjudicate-draw-moves",
        type=int,
        default=6,
        help="consecutive full moves for draw adjudication (default: 6, i.e. 12 plies)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.clear_cache and args.no_clear_cache:
            parser.error("--clear-cache conflicts with --no-clear-cache")
        if args.clear_cache:
            clear_numba_cache(str(ROOT))
        if args.no_openings and args.openings_pgn:
            parser.error("--no-openings conflicts with --openings-pgn")
        if args.opening_moves < 0:
            parser.error("--opening-moves must be >= 0")
        if args.stages and any(
            value is not None
            for value in (
                args.nodes, args.depth, args.time,
                args.nodes1, args.depth1, args.time1,
                args.nodes2, args.depth2, args.time2,
            )
        ):
            parser.error("--stages conflicts with --nodes/--depth/--time")
        if args.threads is not None and args.threads < 1:
            raise ValueError("--threads must be >= 1")
        if args.threads1 < 1:
            raise ValueError("--threads1 must be >= 1")
        if args.threads2 < 1:
            raise ValueError("--threads2 must be >= 1")
        engine1_env = _parse_env(args.engine1_env, "--engine1-env")
        engine2_env = _parse_env(args.engine2_env, "--engine2-env")
        stages = parse_stages(args.stages) if args.stages else None
        if args.nodes is not None:
            nodes, depth, time_ms = args.nodes, None, None
        elif args.depth is not None:
            nodes, depth, time_ms = None, args.depth, None
        elif args.time is not None:
            nodes, depth, time_ms = None, None, args.time
        elif any(
            value is not None
            for value in (args.nodes1, args.depth1, args.time1, args.nodes2, args.depth2, args.time2)
        ):
            nodes, depth, time_ms = None, None, None
        else:
            nodes, depth, time_ms = DEFAULT_NODES, None, None
        run_tournament(
            args.engine1,
            args.engine2,
            args.games,
            time_ms,
            args.name1,
            args.name2,
            depth=depth,
            nodes=nodes,
            concurrency=args.concurrency,
            pgn_file=args.pgn,
            engine1_env=engine1_env,
            engine2_env=engine2_env,
            early_trash=not args.no_early_trash,
            early_trash_min_games=args.early_trash_min_games,
            early_trash_max_score_pct=args.early_trash_max_score_pct,
            own_book1=args.own_book1,
            own_book2=args.own_book2,
            use_openings=not args.no_openings,
            use_sprt=not args.no_sprt,
            openings_pgn=args.openings_pgn,
            opening_plies=args.opening_moves * 2,
            stages=stages,
            backend=args.backend,
            engine1_module=args.engine1_module,
            engine2_module=args.engine2_module,
            tt_mb=args.tt_mb,
            threads1=args.threads1,
            threads2=args.threads2,
            nodes1=args.nodes1,
            depth1=args.depth1,
            time1_ms=args.time1,
            nodes2=args.nodes2,
            depth2=args.depth2,
            time2_ms=args.time2,
            max_plies=args.max_plies,
            search_timeout_s=args.search_timeout,
            sprt_elo0=args.sprt_elo0,
            sprt_elo1=args.sprt_elo1,
            seed=args.seed,
            adjudicate=args.adjudicate,
            adjudicate_win_score=args.adjudicate_win_score,
            adjudicate_win_plies=args.adjudicate_win_moves * 2,
            adjudicate_draw_score=args.adjudicate_draw_score,
            adjudicate_draw_min_move=args.adjudicate_draw_min_move,
            adjudicate_draw_plies=args.adjudicate_draw_moves * 2,
        )
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
