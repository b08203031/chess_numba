"""Fixed-depth puzzle diagnostics for search-regression analysis.

This script intentionally lives outside the engine search code. It parses the
embedded puzzle list in ``test_puzzle.py`` without importing that file, then runs
selected positions at one or more fixed depths and writes per-position JSON.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import time
import types
from pathlib import Path


CUSTOM_BASELINE_NAME = "對抗chess.com 2900時遇到的，引擎搜尋不到，但一搜到就知道大優"

BASELINE_FAILED_IDS = [
    CUSTOM_BASELINE_NAME,
    "DX0K6",
    "2iWRI",
    "645Ft",
    "NT5SC",
    "GG2Zh",
    "OrHZ5",
    "3yR9L",
    "7zZlh",
    "KAk6d",
    "Fg8W1",
    "51ULj",
    "4acOZ",
    "JMp3d",
    "3uwMh",
    "JOOYp",
    "6eIxF",
    "GNuB5",
    "GEmqW",
]

EXPERIMENT_NEW_FAIL_IDS = ["66EDQ", "Ljb97", "1cMH9", "Lhz8A", "1EuxB", "2lBRO", "IV4Rm"]
EXPERIMENT_FIXED_IDS = ["2iWRI", "Fg8W1", "GEmqW", "3yR9L", "JMp3d", "NT5SC"]
HANDOFF_IDS = BASELINE_FAILED_IDS + EXPERIMENT_NEW_FAIL_IDS + EXPERIMENT_FIXED_IDS


def load_puzzles(path: Path) -> list[dict]:
    source = path.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(path))

    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "puzzles":
                return ast.literal_eval(node.value)

    raise ValueError(f"Could not find a top-level puzzles assignment in {path}")


def resolve_identifiers(puzzles: list[dict], identifiers: list[str]) -> tuple[list[tuple[int, str, dict]], list[str]]:
    selected: list[tuple[int, str, dict]] = []
    missing: list[str] = []
    seen_indexes: set[int] = set()

    for identifier in identifiers:
        match = None
        for index, puzzle in enumerate(puzzles):
            name = str(puzzle.get("name", ""))
            if identifier == name or identifier in name:
                match = (index, identifier, puzzle)
                break

        if match is None:
            missing.append(identifier)
            continue

        if match[0] not in seen_indexes:
            selected.append(match)
            seen_indexes.add(match[0])

    return selected, missing


def choose_identifiers(args: argparse.Namespace) -> list[str]:
    if args.ids:
        return args.ids
    if args.set == "baseline":
        return BASELINE_FAILED_IDS
    if args.set == "experiment-new":
        return EXPERIMENT_NEW_FAIL_IDS
    if args.set == "experiment-fixed":
        return EXPERIMENT_FIXED_IDS
    if args.set == "handoff":
        return HANDOFF_IDS
    return []


def ensure_chess_engine_package_alias() -> None:
    """Allow local checkouts whose directory is not literally named chess_engine."""
    project_root = Path(__file__).parent.parent
    root_text = str(project_root) if str(project_root) else "."
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    if not (project_root / "chess_engine").is_dir():
        if "chess_engine" not in sys.modules:
            package = types.ModuleType("chess_engine")
            package.__path__ = [root_text]
            sys.modules["chess_engine"] = package


def import_engine_modules():
    ensure_chess_engine_package_alias()

    import numpy as np
    from chess_engine.classical.constants import CORRECTION_HISTORY_SIZE, MAX_PLY, PAWN_HISTORY_SIZE, TT_SIZE_MB
    from chess_engine.classical.engine_types import SearchContext
    from chess_engine.classical.fen_parser import parse_fen
    from chess_engine.classical.move import move_to_uci
    from chess_engine.classical.search import iterative_deepening_search
    from chess_engine.classical.transposition_table import clear_transposition_table, create_transposition_table

    return {
        "np": np,
        "CORRECTION_HISTORY_SIZE": CORRECTION_HISTORY_SIZE,
        "MAX_PLY": MAX_PLY,
        "PAWN_HISTORY_SIZE": PAWN_HISTORY_SIZE,
        "TT_SIZE_MB": TT_SIZE_MB,
        "SearchContext": SearchContext,
        "parse_fen": parse_fen,
        "move_to_uci": move_to_uci,
        "iterative_deepening_search": iterative_deepening_search,
        "clear_transposition_table": clear_transposition_table,
        "create_transposition_table": create_transposition_table,
    }


def make_context(engine: dict, transposition_table, pawn_history=None):
    np = engine["np"]
    max_ply = engine["MAX_PLY"]
    correction_size = engine["CORRECTION_HISTORY_SIZE"]
    pawn_history_size = engine["PAWN_HISTORY_SIZE"]
    SearchContext = engine["SearchContext"]

    if pawn_history is None:
        pawn_history = np.full((pawn_history_size, 12, 64), -1238, dtype=np.int16)

    return SearchContext(
        transposition_table,
        np.zeros(max_ply * 2, dtype=np.uint16),
        np.zeros((max_ply, max_ply), dtype=np.uint16),
        np.zeros((12, 64), dtype=np.int32),
        np.zeros((64, 64), dtype=np.int32),
        np.zeros((3, 12, 64, 12, 64), dtype=np.int16),
        np.zeros((12, 64, 12), dtype=np.int32),
        pawn_history,
        np.zeros(correction_size, dtype=np.int16),
        np.zeros(correction_size, dtype=np.int16),
        np.zeros(correction_size, dtype=np.int16),
        np.zeros(correction_size, dtype=np.int16),
    )


def is_solution(engine_move: str, solution) -> bool:
    if isinstance(solution, list):
        return engine_move in solution
    return engine_move == solution


def run_one(engine: dict, transposition_table, puzzle: dict, depth: int, time_ms: int, pawn_history=None) -> dict:
    parse_fen = engine["parse_fen"]
    move_to_uci = engine["move_to_uci"]
    iterative_deepening_search = engine["iterative_deepening_search"]
    clear_transposition_table = engine["clear_transposition_table"]

    piece_bbs, occupancy_bbs, game_state = parse_fen(puzzle["fen"])
    clear_transposition_table(transposition_table)
    search_context = make_context(engine, transposition_table, pawn_history=pawn_history)

    start_time = time.time()
    best_move, score, nodes, qnodes, tt_hits, completed_depth = iterative_deepening_search(
        piece_bbs,
        occupancy_bbs,
        game_state,
        depth,
        {"optimum_time": time_ms, "maximum_time": time_ms},
        search_context,
        verbose=False,
    )
    elapsed = time.time() - start_time

    engine_move = move_to_uci(best_move)
    total_nodes = int(nodes) + int(qnodes)
    qnode_pct = (int(qnodes) / total_nodes * 100.0) if total_nodes else 0.0
    tt_hit_rate = (int(tt_hits) / total_nodes * 100.0) if total_nodes else 0.0

    return {
        "name": puzzle["name"],
        "fen": puzzle["fen"],
        "theme": puzzle.get("theme"),
        "rating": puzzle.get("rating"),
        "solution": puzzle["solution"],
        "depth_requested": depth,
        "completed_depth": int(completed_depth),
        "engine_move": engine_move,
        "passed": is_solution(engine_move, puzzle["solution"]),
        "score": int(score),
        "total_time_seconds": elapsed,
        "total_nodes": total_nodes,
        "engine_reported_nodes": int(nodes),
        "search_nodes": int(nodes),
        "quiescence_nodes": int(qnodes),
        "quiescence_percentage": qnode_pct,
        "tt_hits": int(tt_hits),
        "tt_hit_rate": tt_hit_rate,
        "nps": int(total_nodes / elapsed) if elapsed > 0 else 0,
    }


def warm_up(engine: dict) -> None:
    create_transposition_table = engine["create_transposition_table"]
    transposition_table = create_transposition_table(16)
    context = make_context(engine, transposition_table)
    piece_bbs, occupancy_bbs, game_state = engine["parse_fen"](
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    )
    engine["iterative_deepening_search"](
        piece_bbs,
        occupancy_bbs,
        game_state,
        2,
        {"optimum_time": 0, "maximum_time": 0},
        context,
        verbose=False,
    )


def print_selection(selected: list[tuple[int, str, dict]], missing: list[str]) -> None:
    print(f"Selected puzzles: {len(selected)}")
    for index, identifier, puzzle in selected:
        print(f"  {index + 1:3d}  {identifier}  {puzzle['name']}  [{puzzle.get('theme', '')}]")
    if missing:
        print(f"Missing identifiers: {len(missing)}")
        for identifier in missing:
            print(f"  {identifier}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fixed-depth search diagnostics on embedded test puzzles.")
    parser.add_argument("--puzzle-source", default=str(Path(__file__).parent / "test_puzzle.py"), help="Python file containing the puzzles list.")
    parser.add_argument("--set", choices=["handoff", "baseline", "experiment-new", "experiment-fixed", "all"], default="handoff")
    parser.add_argument("--ids", nargs="*", help="Puzzle names or substrings to run; overrides --set.")
    parser.add_argument("--depths", nargs="+", type=int, default=[13, 15], help="Fixed depths to run.")
    parser.add_argument("--time-ms", type=int, default=0, help="0 means true fixed-depth search with no time cutoff.")
    parser.add_argument("--tt-mb", type=int, default=None, help="Override TT size in MB for diagnostics.")
    parser.add_argument("--reuse-pawn-history", action="store_true", help="Reuse pawn history across runs like test_puzzle.py currently does.")
    parser.add_argument("--no-warmup", action="store_true", help="Skip JIT warm-up search.")
    parser.add_argument("--list-only", action="store_true", help="Only print selected/missing puzzle IDs; do not import or run the engine.")
    parser.add_argument("--output", default="search_diagnostics_results.json", help="JSON output path.")
    args = parser.parse_args()

    puzzles = load_puzzles(Path(args.puzzle_source))
    identifiers = choose_identifiers(args)
    if identifiers:
        selected, missing = resolve_identifiers(puzzles, identifiers)
    else:
        selected = [(index, puzzle.get("name", str(index + 1)), puzzle) for index, puzzle in enumerate(puzzles)]
        missing = []

    print_selection(selected, missing)
    if args.list_only:
        return 0

    try:
        engine = import_engine_modules()
    except ModuleNotFoundError as exc:
        missing_module = exc.name or str(exc)
        print(
            f"Cannot run engine diagnostics: missing Python module {missing_module!r}.",
            file=sys.stderr,
        )
        print("Use an environment with the project dependencies installed, especially numpy and numba.", file=sys.stderr)
        return 2

    if not args.no_warmup:
        print("Warming up JIT...")
        warm_up(engine)

    tt_size_mb = args.tt_mb if args.tt_mb is not None else engine["TT_SIZE_MB"]
    transposition_table = engine["create_transposition_table"](tt_size_mb)
    shared_pawn_history = None
    if args.reuse_pawn_history:
        np = engine["np"]
        shared_pawn_history = np.full((engine["PAWN_HISTORY_SIZE"], 12, 64), -1238, dtype=np.int16)

    results = []
    for depth in args.depths:
        passed = 0
        for index, identifier, puzzle in selected:
            result = run_one(
                engine,
                transposition_table,
                puzzle,
                depth,
                args.time_ms,
                pawn_history=shared_pawn_history if args.reuse_pawn_history else None,
            )
            result["puzzle_index"] = index + 1
            result["identifier"] = identifier
            results.append(result)
            passed += int(result["passed"])
            status = "PASS" if result["passed"] else "FAIL"
            print(
                f"depth {depth:2d} {status} {identifier}: {result['engine_move']} "
                f"score={result['score']} nodes={result['total_nodes']} "
                f"q={result['quiescence_percentage']:.1f}% tt={result['tt_hit_rate']:.1f}%"
            )
        print(f"Depth {depth}: {passed}/{len(selected)} passed")

    payload = {
        "puzzle_source": args.puzzle_source,
        "set": args.set,
        "ids": identifiers,
        "missing_ids": missing,
        "depths": args.depths,
        "time_ms": args.time_ms,
        "reuse_pawn_history": args.reuse_pawn_history,
        "results": results,
    }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())