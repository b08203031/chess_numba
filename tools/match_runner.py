"""Run the local Classical engine against an external Stockfish executable.

The local engine defaults to the shared in-process SearchContext pool.  Each
worker still owns one long-lived Stockfish subprocess because Stockfish is an
external native UCI engine.
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
DEFAULT_ENGINE = ROOT / "main.py"
DEFAULT_NODES = 50_000
DEFAULT_TT_MB = 128

Engine = UciEngine


def _default_stockfish() -> Path:
    if os.name == "nt":
        return ROOT / "external" / "stockfish" / "stockfish-windows-x86-64-avx2.exe"
    return ROOT / "external" / "stockfish" / "src" / "stockfish"


def _parse_env(items: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--engine-env expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        if not key.strip():
            raise ValueError("--engine-env contains an empty key")
        result[key.strip()] = value
    return result


def parse_limit(limit_str: str) -> dict[str, Optional[int]]:
    limit = parse_limit_token(limit_str)
    return {"time": limit.time_ms, "depth": limit.depth, "nodes": limit.nodes}


def parse_stages(stages_str: str) -> list[dict[str, object]]:
    """Parse ``games:engine_limit:sf_limit`` stage specifications."""
    stages: list[dict[str, object]] = []
    for raw in stages_str.split(","):
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid stage {raw!r}; expected games:engine_limit:sf_limit"
            )
        games = int(parts[0])
        if games < 2 or games % 2:
            raise ValueError(f"Stage games must be an even integer >= 2, got {games}")
        engine_limit = parse_limit_token(parts[1])
        sf_limit = parse_limit_token(parts[2])
        stages.append({
            "games": games,
            "engine_limit": engine_limit,
            "sf_limit": sf_limit,
            "engine_limit_str": engine_limit.token(),
            "sf_limit_str": sf_limit.token(),
        })
    if not stages:
        raise ValueError("No stages were parsed")
    return stages


def _pgn_path(base: Optional[str], stage_number: int, multi: bool) -> str:
    output_dir = ROOT / "tournament_analysis"
    if base:
        path = Path(base)
        if not path.is_absolute():
            path = output_dir / path
        path = path.resolve()
        if not multi:
            return str(path)
        suffix = path.suffix or ".pgn"
        return str(path.with_name(f"{path.stem}_stage{stage_number}{suffix}"))
    if multi:
        return str(output_dir / f"sf_match_stage_{stage_number}.pgn")
    return str(output_dir / "sf_match_results.pgn")


def _print_result(result: StageResult, engine_name: str, sf_name: str) -> None:
    elo, error = calculate_elo_statistics(result.wins, result.draws, result.losses)
    print("\n=== Stockfish Match Stage Finished ===")
    print(
        f"{engine_name}: {result.points:.1f}/{result.games} "
        f"({result.wins}W {result.draws}D {result.losses}L)"
    )
    print(f"{sf_name}: {result.games - result.points:.1f}/{result.games}")
    if math.isinf(elo):
        print("Elo difference: beyond finite estimate (perfect or zero score)")
    else:
        print(f"Elo difference ({engine_name} - {sf_name}): {elo:+.2f} ±{error:.2f}")
    print(f"Status: {result.status}; LLR={result.llr:.2f}")
    print(f"PGN: {result.pgn_path}")


def _engine_backend(requested: str, engine_path: str) -> str:
    if requested != "auto":
        return requested
    return "inprocess" if Path(engine_path).resolve() == DEFAULT_ENGINE.resolve() else "uci"


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


def run_match_stages(
    engine_path: str,
    sf_path: str,
    stages: Sequence[Mapping[str, object]],
    *,
    engine_name: str = "MyEngine",
    sf_name: str = "Stockfish",
    concurrency: int = 1,
    pgn_base: Optional[str] = None,
    no_sprt: bool = False,
    engine_backend: str = "auto",
    engine_module: str = "chess_engine.classical",
    tt_mb: int = DEFAULT_TT_MB,
    sf_hash_mb: Optional[int] = None,
    engine_threads: int = 1,
    sf_threads: int = 1,
    engine_env: Optional[Mapping[str, object]] = None,
    use_openings: bool = True,
    openings_pgn: Optional[str] = None,
    opening_plies: int = 16,
    seed: int = 0,
    max_plies: int = 400,
    search_timeout_s: float = 300.0,
    sprt_elo0: float = 0.0,
    sprt_elo1: float = 10.0,
    early_trash: bool = False,
    early_trash_min_games: int = 80,
    early_trash_max_score_pct: float = 40.0,
    adjudicate: bool = True,
    adjudicate_win_score: int = 500,
    adjudicate_win_plies: int = 10,
    adjudicate_draw_score: int = 8,
    adjudicate_draw_min_move: int = 40,
    adjudicate_draw_plies: int = 12,
) -> list[StageResult]:
    concurrency = _validate_positive_int(concurrency, "concurrency")
    tt_mb = _validate_positive_int(tt_mb, "tt_mb")
    if sf_hash_mb is not None:
        sf_hash_mb = _validate_positive_int(sf_hash_mb, "sf_hash_mb")
    engine_threads = _validate_positive_int(engine_threads, "engine_threads")
    sf_threads = _validate_positive_int(sf_threads, "sf_threads")
    max_plies = _validate_positive_int(max_plies, "max_plies")
    if float(search_timeout_s) <= 0.0:
        raise ValueError("search_timeout_s must be > 0")
    if adjudicate_win_score < 0 or adjudicate_draw_score < 0:
        raise ValueError("adjudicate scores must be >= 0")
    _validate_positive_int(adjudicate_win_plies, "adjudicate_win_plies")
    _validate_positive_int(adjudicate_draw_min_move, "adjudicate_draw_min_move")
    _validate_positive_int(adjudicate_draw_plies, "adjudicate_draw_plies")
    if not os.path.exists(sf_path):
        raise FileNotFoundError(sf_path)
    normalized: list[dict[str, object]] = []
    for stage in stages:
        engine_limit = stage.get("engine_limit")
        sf_limit = stage.get("sf_limit")
        if not isinstance(engine_limit, SearchLimit):
            engine_limit = SearchLimit.from_mapping(stage.get("engine_limits", stage))
        if not isinstance(sf_limit, SearchLimit):
            sf_limit = SearchLimit.from_mapping(stage.get("sf_limits", stage))
        games = int(stage["games"])
        if games < 2 or games % 2:
            raise ValueError("Every stage must use an even game count >= 2")
        normalized.append({
            "games": games,
            "engine_limit": engine_limit,
            "sf_limit": sf_limit,
        })
    if not normalized:
        raise ValueError("No match stages supplied")

    selected = _engine_backend(engine_backend, engine_path)
    if selected == "inprocess":
        engine_factory = InProcessEngineFactory(
            engine_name,
            engine_module,
            tt_mb=tt_mb,
            env_extra=engine_env,
            threads=engine_threads,
        )
    else:
        if not os.path.exists(engine_path):
            raise FileNotFoundError(engine_path)
        engine_factory = UciEngineFactory(
            engine_name,
            engine_path,
            options={"OwnBook": False, "Hash": tt_mb, "Threads": engine_threads},
            env_extra=engine_env,
            search_timeout_s=search_timeout_s,
        )
    sf_options: dict[str, object] = {"Threads": sf_threads, "Ponder": False}
    if sf_hash_mb is not None:
        sf_options["Hash"] = sf_hash_mb
    sf_factory = UciEngineFactory(
        sf_name,
        sf_path,
        options=sf_options,
        search_timeout_s=search_timeout_s,
    )

    openings = load_openings(
        epd_path=str(ROOT / "data" / "openings.epd"),
        pgn_path=openings_pgn,
        pgn_plies=opening_plies,
        enabled=use_openings,
    )
    random.Random(seed).shuffle(openings)
    max_pairs = max(int(stage["games"]) // 2 for stage in normalized)
    actual_concurrency = min(concurrency, max_pairs)
    print(
        f"Local backend: {selected}; workers={actual_concurrency}; "
        f"{engine_name} Threads={engine_threads}, TT={tt_mb} MB/context; "
        f"{sf_name} Threads={sf_threads}/process, "
        f"Hash={sf_hash_mb if sf_hash_mb is not None else 'engine default'}",
        flush=True,
    )
    cpu_count = os.cpu_count()
    peak_active_threads = actual_concurrency * max(engine_threads, sf_threads)
    total_allocated_threads = actual_concurrency * (engine_threads + sf_threads)
    if cpu_count is not None and peak_active_threads > cpu_count:
        print(
            f"[warning] Peak active search threads ({actual_concurrency} * max({engine_threads}, {sf_threads}) = "
            f"{peak_active_threads}; total allocated {total_allocated_threads}) exceeds available CPU count "
            f"({cpu_count}); CPU contention may affect timing and NPS.",
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
        print(
            f"Process plan: one local Python/Numba process + {actual_concurrency} long-lived Stockfish processes.",
            flush=True,
        )
    print(
        "Warm-up: local Python/Numba backend searches Kiwipete at depth 14, "
        "then one complete startpos game at depth 1 (all results discarded).",
        flush=True,
    )

    pool = create_engine_pairs(engine_factory, sf_factory, actual_concurrency)
    results: list[StageResult] = []
    try:
        multi = len(normalized) > 1
        for number, stage in enumerate(normalized, 1):
            engine_limit = stage["engine_limit"]
            sf_limit = stage["sf_limit"]
            print(
                f"\n>>> Stage {number}/{len(normalized)}: games={stage['games']}; "
                f"{engine_name}={engine_limit.describe()}; {sf_name}={sf_limit.describe()}",
                flush=True,
            )
            result = run_stage(
                pool,
                openings,
                games=int(stage["games"]),
                first_limit=engine_limit,
                second_limit=sf_limit,
                first_name=engine_name,
                second_name=sf_name,
                pgn_path=_pgn_path(pgn_base, number, multi),
                event="Engine vs Stockfish Match",
                use_sprt=not no_sprt,
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
            _print_result(result, engine_name, sf_name)
    finally:
        close_engine_pairs(pool)
    return results


def run_match_tournament(
    engine_path: str,
    sf_path: str,
    games_count: int,
    engine_limits: Mapping[str, Optional[int]],
    sf_limits: Mapping[str, Optional[int]],
    engine_name: str = "MyEngine",
    sf_name: str = "Stockfish",
    concurrency: int = 1,
    pgn_filename: str = "sf_match_results.pgn",
    no_sprt: bool = False,
    **kwargs: object,
) -> list[StageResult]:
    """Compatibility entry point for one-stage callers."""
    stage = {
        "games": games_count,
        "engine_limit": SearchLimit.from_mapping(engine_limits),
        "sf_limit": SearchLimit.from_mapping(sf_limits),
    }
    return run_match_stages(
        engine_path,
        sf_path,
        [stage],
        engine_name=engine_name,
        sf_name=sf_name,
        concurrency=concurrency,
        pgn_base=pgn_filename,
        no_sprt=no_sprt,
        **kwargs,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the local engine against Stockfish.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Limit tokens: nN=fixed nodes, dN=fixed depth, plain N=movetime ms.\n"
            "Default without limit options: both sides n50000.\n"
            "Examples:\n"
            "  python tools/match_runner.py -c 8 --engine-nodes 50000 --sf-nodes 50000\n"
            "  python tools/match_runner.py --stages 20:n20000:n20000,40:n100000:n100000"
        ),
    )
    parser.add_argument("--engine", default=str(DEFAULT_ENGINE))
    parser.add_argument("--sf", default=str(_default_stockfish()))
    parser.add_argument("--engine-name", "--engine_name", dest="engine_name", default="MyEngine")
    parser.add_argument("--sf-name", "--sf_name", dest="sf_name", default="Stockfish")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--concurrency", "-c", type=int, default=1)

    engine_limits = parser.add_mutually_exclusive_group()
    engine_limits.add_argument("--engine-nodes", "--engine_nodes", dest="engine_nodes", type=int)
    engine_limits.add_argument("--engine-depth", "--engine_depth", dest="engine_depth", type=int)
    engine_limits.add_argument("--engine-time", "--engine_time", dest="engine_time", type=int)
    sf_limits = parser.add_mutually_exclusive_group()
    sf_limits.add_argument("--sf-nodes", "--sf_nodes", dest="sf_nodes", type=int)
    sf_limits.add_argument("--sf-depth", "--sf_depth", dest="sf_depth", type=int)
    sf_limits.add_argument("--sf-time", "--sf_time", dest="sf_time", type=int)
    parser.add_argument(
        "--stages",
        metavar="SPEC",
        help="games:engine_limit:sf_limit list, e.g. 20:n20000:n20000,40:n100000:n100000",
    )

    parser.add_argument("--engine-backend", choices=("auto", "inprocess", "uci"), default="auto")
    parser.add_argument("--engine-module", default="chess_engine.classical")
    parser.add_argument("--tt-mb", type=int, default=DEFAULT_TT_MB, help="local TT MB/context (default: 128)")
    parser.add_argument(
        "--sf-hash-mb",
        type=int,
        default=None,
        help="set Stockfish Hash MB/process (default: keep Stockfish's own default)",
    )
    parser.add_argument(
        "--engine-threads",
        type=int,
        default=1,
        help="local engine search threads (default: 1)",
    )
    parser.add_argument("--sf-threads", type=int, default=1, help="Stockfish threads/process; keep 1 with -c > 1")
    parser.add_argument("--engine-env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--pgn", help="PGN filename/path; multi-stage adds a stage suffix")
    parser.add_argument("--no-openings", action="store_true")
    parser.add_argument("--openings-pgn")
    parser.add_argument("--opening-moves", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-sprt", action="store_true", help="play every scheduled game")
    parser.add_argument("--sprt-elo0", type=float, default=0.0)
    parser.add_argument("--sprt-elo1", type=float, default=10.0)
    parser.add_argument("--early-trash", action="store_true", help="optional; disabled by default for SF calibration")
    parser.add_argument("--early-trash-min-games", type=int, default=80)
    parser.add_argument("--early-trash-max-score-pct", type=float, default=40.0)
    parser.add_argument("--max-plies", type=int, default=400)
    parser.add_argument("--search-timeout", type=float, default=300.0, metavar="SECONDS")
    parser.add_argument("--clear-cache", action="store_true")
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
        if args.clear_cache:
            clear_numba_cache(str(ROOT))
        if args.no_openings and args.openings_pgn:
            parser.error("--no-openings conflicts with --openings-pgn")
        if args.opening_moves < 0:
            parser.error("--opening-moves must be >= 0")
        if args.stages and any(
            value is not None
            for value in (
                args.engine_nodes, args.engine_depth, args.engine_time,
                args.sf_nodes, args.sf_depth, args.sf_time,
            )
        ):
            parser.error("--stages conflicts with per-side limit options")
        if args.stages:
            stages = parse_stages(args.stages)
        else:
            if args.engine_nodes is not None:
                engine_limit = SearchLimit(nodes=args.engine_nodes)
            elif args.engine_depth is not None:
                engine_limit = SearchLimit(depth=args.engine_depth)
            elif args.engine_time is not None:
                engine_limit = SearchLimit(time_ms=args.engine_time)
            else:
                engine_limit = SearchLimit(nodes=DEFAULT_NODES)
            if args.sf_nodes is not None:
                sf_limit = SearchLimit(nodes=args.sf_nodes)
            elif args.sf_depth is not None:
                sf_limit = SearchLimit(depth=args.sf_depth)
            elif args.sf_time is not None:
                sf_limit = SearchLimit(time_ms=args.sf_time)
            else:
                sf_limit = SearchLimit(nodes=DEFAULT_NODES)
            stages = [{"games": args.games, "engine_limit": engine_limit, "sf_limit": sf_limit}]

        run_match_stages(
            args.engine,
            args.sf,
            stages,
            engine_name=args.engine_name,
            sf_name=args.sf_name,
            concurrency=args.concurrency,
            pgn_base=args.pgn,
            no_sprt=args.no_sprt,
            engine_backend=args.engine_backend,
            engine_module=args.engine_module,
            tt_mb=args.tt_mb,
            sf_hash_mb=args.sf_hash_mb,
            engine_threads=args.engine_threads,
            sf_threads=args.sf_threads,
            engine_env=_parse_env(args.engine_env),
            use_openings=not args.no_openings,
            openings_pgn=args.openings_pgn,
            opening_plies=args.opening_moves * 2,
            seed=args.seed,
            max_plies=args.max_plies,
            search_timeout_s=args.search_timeout,
            sprt_elo0=args.sprt_elo0,
            sprt_elo1=args.sprt_elo1,
            early_trash=args.early_trash,
            early_trash_min_games=args.early_trash_min_games,
            early_trash_max_score_pct=args.early_trash_max_score_pct,
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
