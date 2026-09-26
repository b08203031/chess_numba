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
    """Parse ``games:limit`` stages; every stage must contain full color pairs."""
    stages: list[dict[str, object]] = []
    for raw in stages_str.split(","):
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid stage {raw!r}; expected games:limit")
        games = int(parts[0])
        if games < 2 or games % 2:
            raise ValueError(f"Stage games must be an even integer >= 2, got {games}")
        limit = parse_limit_token(parts[1])
        stages.append({
            "games": games,
            "limit": limit,
            "limit_str": limit.token(),
        })
    if not stages:
        raise ValueError("No stages were parsed")
    return stages


def _safe_token(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def _stage_pgn_path(base: Optional[str], stage_number: int, limit: SearchLimit, multi: bool) -> str:
    output_dir = ROOT / "tournament_analysis"
    if base:
        path = Path(base).resolve()
        if not multi:
            return str(path)
        suffix = path.suffix or ".pgn"
        return str(path.with_name(f"{path.stem}_stage{stage_number}_{_safe_token(limit.token())}{suffix}"))
    if multi:
        return str(output_dir / f"tournament_results_stage{stage_number}_{_safe_token(limit.token())}.pgn")
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
    max_plies: int = 400,
    search_timeout_s: float = 300.0,
    sprt_elo0: float = 0.0,
    sprt_elo1: float = 10.0,
    seed: int = 0,
) -> list[StageResult]:
    """Run one or more stages while keeping the same worker pool alive."""
    del early_trash_max_elo  # Kept in the public signature for compatibility.
    if external_barriers is not None:
        raise ValueError("external_barriers belonged to the removed subprocess-JIT design")
    if concurrency <= 0:
        raise ValueError("concurrency must be >= 1")
    if tt_mb <= 0:
        raise ValueError("tt_mb must be >= 1")

    if stages:
        normalized: list[dict[str, object]] = []
        for stage in stages:
            if isinstance(stage.get("limit"), SearchLimit):
                limit = stage["limit"]
            else:
                limit = SearchLimit.from_mapping(stage)
            normalized.append({
                "games": int(stage["games"]),
                "limit": limit,
                "pgn_file": stage.get("pgn_file"),
            })
    else:
        if nodes is not None:
            limit = SearchLimit(nodes=nodes)
        elif depth is not None:
            limit = SearchLimit(depth=depth)
        elif time_ms is not None:
            limit = SearchLimit(time_ms=time_ms)
        else:
            limit = SearchLimit(nodes=DEFAULT_NODES)
        normalized = [{"games": int(games_count), "limit": limit, "pgn_file": pgn_file}]

    for stage in normalized:
        games = int(stage["games"])
        if games < 2 or games % 2:
            raise ValueError("Every stage must use an even game count >= 2")

    selected = _select_backend(backend, engine1_path, engine2_path, own_book1, own_book2)
    if selected == "inprocess":
        factory1 = InProcessEngineFactory(
            engine1_name, engine1_module, tt_mb=tt_mb, env_extra=engine1_env
        )
        factory2 = InProcessEngineFactory(
            engine2_name, engine2_module, tt_mb=tt_mb, env_extra=engine2_env
        )
    else:
        for path in (engine1_path, engine2_path):
            if not os.path.exists(path):
                raise FileNotFoundError(path)
        factory1 = UciEngineFactory(
            engine1_name,
            engine1_path,
            options={"OwnBook": own_book1},
            env_extra=engine1_env,
            search_timeout_s=search_timeout_s,
        )
        factory2 = UciEngineFactory(
            engine2_name,
            engine2_path,
            options={"OwnBook": own_book2},
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
        f"Backend: {selected}; workers={actual_concurrency}; TT={tt_mb} MB/context; "
        f"openings={len(openings)}; seed={seed}",
        flush=True,
    )
    if selected == "inprocess":
        print("JIT plan: classical once + classical_old once; worker contexts share compiled code.", flush=True)
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
            limit = stage["limit"]
            output = stage.get("pgn_file") or _stage_pgn_path(pgn_file, number, limit, multi)
            print(
                f"\n>>> Stage {number}/{len(normalized)}: games={stage['games']}, "
                f"limit={limit.describe()}",
                flush=True,
            )
            result = run_stage(
                pool,
                openings,
                games=int(stage["games"]),
                first_limit=limit,
                second_limit=limit,
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
            )
            results.append(result)
            _print_result(result, engine1_name, engine2_name)
    finally:
        close_engine_pairs(pool)
    return results


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
    parser.add_argument("--stages", metavar="SPEC", help="games:limit list, e.g. 200:n20000,400:n100000")
    parser.add_argument("--concurrency", "-c", type=int, default=1, help="parallel games/independent contexts")
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
    parser.add_argument("--seed", type=int, default=0, help="deterministic opening shuffle seed")
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
        if args.stages and any(value is not None for value in (args.nodes, args.depth, args.time)):
            parser.error("--stages conflicts with --nodes/--depth/--time")
        engine1_env = _parse_env(args.engine1_env, "--engine1-env")
        engine2_env = _parse_env(args.engine2_env, "--engine2-env")
        stages = parse_stages(args.stages) if args.stages else None
        if args.nodes is not None:
            nodes, depth, time_ms = args.nodes, None, None
        elif args.depth is not None:
            nodes, depth, time_ms = None, args.depth, None
        elif args.time is not None:
            nodes, depth, time_ms = None, None, args.time
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
            max_plies=args.max_plies,
            search_timeout_s=args.search_timeout,
            sprt_elo0=args.sprt_elo0,
            sprt_elo1=args.sprt_elo1,
            seed=args.seed,
        )
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
