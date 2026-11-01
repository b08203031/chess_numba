"""Build an independent HCE validation dataset (val_pgn.npz) from engine self-play PGNs.

Extracts quiet (non-capture, non-promotion, non-check, non-mate) positions from
high-node tournament PGNs across the repository, preserving full ``game_state``
fields used by Classical HCE:
- ``game_state[0]``: side_to_move (0=White, 1=Black)
- ``game_state[1]``: castling_rights (K=1 | Q=2 | k=4 | q=8)
- ``game_state[2]``: legal en-passant square (0..63, or 64 if none)
- ``game_state[3]``: halfmove_clock (50-move rule decay in endgame.py)

Duplicate positions across paired openings / repeated games are merged exactly
and their game outcomes (WDL) and White-POV search scores are averaged to remove
paired-game ordering bias and reduce label variance.

Usage (from repo root)::

    python -m tuner.data_pipeline.build_pgn_validation
    python -m tuner.data_pipeline.build_pgn_validation --min-median-nodes 150000 --workers 10
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import os
from pathlib import Path
import re
import time

import chess
import chess.pgn
import numpy as np

from tuner.data_pipeline.preprocess_data import board_to_arrays

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "tuner" / "data" / "hce" / "val_pgn.npz"

_COMMENT_RE = re.compile(r"\{([^}]*)\}")
_EVAL_RE = re.compile(r"eval=([^,\s}]+)")
_DEPTH_RE = re.compile(r"\bd=(\d+)")
_NODES_RE = re.compile(r"\bn=(\d+)")
_RESULT_MAP = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}

_SKIP_DIR_PARTS = (
    "stockfish_repo",
    "stockfish_11",
    ".git",
    "external",
    "node_modules",
    "__pycache__",
)
_SKIP_SUFFIXES = (
    ".png",
    ".json",
    ".jsonl",
    ".ipynb",
    ".py",
    ".md",
    ".csv",
    ".txt",
    ".npz",
    ".bin",
    ".pth",
)


def discover_pgn_files(
    root: Path = ROOT,
    min_median_nodes: int = 150_000,
    include_external_opponents: bool = False,
) -> list[dict]:
    """Scan repo for PGN / tournament_results_* files meeting the node-budget floor."""
    candidates: list[Path] = []
    tuner_data_dir = (root / "tuner" / "data").resolve()
    for dirpath, _, filenames in os.walk(root):
        dp = Path(dirpath).resolve()
        if dp == tuner_data_dir or tuner_data_dir in dp.parents:
            continue
        if any(part in dirpath for part in _SKIP_DIR_PARTS):
            continue
        for fname in filenames:
            lower = fname.lower()
            if lower.endswith(_SKIP_SUFFIXES):
                continue
            if not (lower.endswith(".pgn") or fname.startswith("tournament_results_")):
                continue
            if not include_external_opponents and "sf11" in lower:
                continue
            candidates.append(Path(dirpath) / fname)

    selected: list[dict] = []
    for path in sorted(candidates):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        games = text.count("[Event ")
        if games <= 0:
            continue
        nodes = [
            int(n_str)
            for c_body in _COMMENT_RE.findall(text)
            for n_str in _NODES_RE.findall(c_body)
        ]
        median_nodes = sorted(nodes)[len(nodes) // 2] if nodes else 0
        if min_median_nodes > 0 and median_nodes < min_median_nodes:
            continue
        selected.append(
            {
                "path": str(path),
                "rel_path": str(path.relative_to(root)),
                "games": games,
                "median_nodes": median_nodes,
                "size_mb": round(len(text) / 1e6, 2),
            }
        )
    return selected


def _parse_single_pgn_file(
    task: tuple[str, int, int],
) -> dict:
    """Worker: parse one PGN file and return compact arrays."""
    logging.getLogger("chess.pgn").setLevel(logging.CRITICAL)
    file_path, min_setup_ply, min_startpos_ply = task
    piece_list: list[np.ndarray] = []
    occ_list: list[np.ndarray] = []
    gs_list: list[np.ndarray] = []
    res_list: list[float] = []
    cp_list: list[float] = []
    depth_list: list[int] = []

    games_read = 0
    games_used = 0

    with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
        while True:
            try:
                game = chess.pgn.read_game(fh)
            except Exception:
                continue
            if game is None:
                break
            games_read += 1
            if game.errors:
                continue
            result = _RESULT_MAP.get(game.headers.get("Result", "*"))
            if result is None:
                continue

            is_setup = game.headers.get("SetUp", "0") == "1" or ("FEN" in game.headers)
            min_ply = min_setup_ply if is_setup else min_startpos_ply

            try:
                board = game.board()
            except Exception:
                continue

            games_used += 1
            ply = 0
            for node in game.mainline():
                move = node.move
                comment = node.comment or ""
                if ply >= min_ply:
                    eval_match = _EVAL_RE.search(comment)
                    eval_token = eval_match.group(1) if eval_match else None
                    is_mate = eval_token is not None and "#" in eval_token

                    if (
                        not is_mate
                        and not board.is_check()
                        and not board.is_capture(move)
                        and move.promotion is None
                        and board.pieces_mask(chess.KING, chess.WHITE)
                        and board.pieces_mask(chess.KING, chess.BLACK)
                    ):
                        p_bb, o_bb, g_st = board_to_arrays(board)
                        cp_val = float("nan")
                        if eval_token is not None:
                            try:
                                raw_eval = float(eval_token)
                                if not (raw_eval == 0.0 and board.halfmove_clock >= 90):
                                    cp_val = raw_eval * 100.0
                                    if board.turn == chess.BLACK:
                                        cp_val = -cp_val
                            except ValueError:
                                cp_val = float("nan")

                        depth_match = _DEPTH_RE.search(comment)
                        depth_val = int(depth_match.group(1)) if depth_match else 0

                        piece_list.append(p_bb)
                        occ_list.append(o_bb)
                        gs_list.append(g_st)
                        res_list.append(result)
                        cp_list.append(cp_val)
                        depth_list.append(min(depth_val, 32000))

                board.push(move)
                ply += 1

    if not res_list:
        return {
            "games_read": games_read,
            "games_used": games_used,
            "n_raw": 0,
        }

    return {
        "games_read": games_read,
        "games_used": games_used,
        "n_raw": len(res_list),
        "piece_bbs": np.asarray(piece_list, dtype=np.uint64),
        "occupancy_bbs": np.asarray(occ_list, dtype=np.uint64),
        "game_states": np.asarray(gs_list, dtype=np.uint64),
        "results": np.asarray(res_list, dtype=np.float32),
        "search_cp": np.asarray(cp_list, dtype=np.float32),
        "depth": np.asarray(depth_list, dtype=np.int16),
    }


def deduplicate_and_average(
    piece_bbs: np.ndarray,
    occupancy_bbs: np.ndarray,
    game_states: np.ndarray,
    results: np.ndarray,
    search_cp: np.ndarray,
    depth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact 128-byte key deduplication with WDL & search-score averaging."""
    n = len(results)
    keys = np.empty((n, 16), dtype=np.uint64)
    keys[:, 0:12] = piece_bbs
    keys[:, 12] = game_states[:, 0]  # side_to_move
    keys[:, 13] = game_states[:, 1]  # castling_rights
    keys[:, 14] = game_states[:, 2]  # legal ep_square
    rule50_bucket = np.maximum(0, game_states[:, 3].astype(np.int64) - 12) // 4
    keys[:, 15] = rule50_bucket.astype(np.uint64)

    void_keys = np.ascontiguousarray(keys).view(np.dtype((np.void, 16 * 8))).ravel()
    _, first_idx, inverse = np.unique(
        void_keys, return_index=True, return_inverse=True
    )
    n_unique = len(first_idx)

    counts = np.bincount(inverse, minlength=n_unique).astype(np.float64)
    res_sum = np.bincount(inverse, weights=results.astype(np.float64), minlength=n_unique)
    avg_results = (res_sum / counts).astype(np.float32)

    valid_cp = np.isfinite(search_cp)
    cp_cnt = np.bincount(inverse[valid_cp], minlength=n_unique).astype(np.float64)
    cp_sum = np.bincount(
        inverse[valid_cp],
        weights=search_cp[valid_cp].astype(np.float64),
        minlength=n_unique,
    )
    avg_cp = np.full(n_unique, np.nan, dtype=np.float32)
    has_cp = cp_cnt > 0
    avg_cp[has_cp] = (cp_sum[has_cp] / cp_cnt[has_cp]).astype(np.float32)

    max_depth = np.zeros(n_unique, dtype=np.int16)
    np.maximum.at(max_depth, inverse, depth)

    # Restore original chronological/file appearance order rather than bitboard sort order
    order = np.argsort(first_idx)
    rep = first_idx[order]
    return (
        piece_bbs[rep],
        occupancy_bbs[rep],
        game_states[rep],
        avg_results[order],
        avg_cp[order],
        max_depth[order],
    )


def build_pgn_validation(
    output_path: Path = DEFAULT_OUTPUT,
    min_median_nodes: int = 150_000,
    min_setup_ply: int = 6,
    min_startpos_ply: int = 16,
    max_positions: int = 0,
    workers: int | None = None,
    seed: int = 42,
) -> dict:
    t0 = time.time()
    discovered = discover_pgn_files(ROOT, min_median_nodes=min_median_nodes)
    if not discovered:
        raise RuntimeError(
            f"No PGN files found with median_nodes >= {min_median_nodes}."
        )

    total_est_games = sum(item["games"] for item in discovered)
    print(
        f"Discovered {len(discovered)} PGN file(s) with median_nodes >= {min_median_nodes:,} "
        f"(~{total_est_games:,} games)."
    )

    if workers is None or workers <= 0:
        workers = min(12, max(1, (os.cpu_count() or 4) - 2))

    tasks = [
        (item["path"], min_setup_ply, min_startpos_ply)
        for item in discovered
    ]

    chunks_pb: list[np.ndarray] = []
    chunks_ob: list[np.ndarray] = []
    chunks_gs: list[np.ndarray] = []
    chunks_res: list[np.ndarray] = []
    chunks_cp: list[np.ndarray] = []
    chunks_d: list[np.ndarray] = []
    total_games_used = 0
    total_raw = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(_parse_single_pgn_file, task): item["rel_path"]
            for task, item in zip(tasks, discovered)
        }
        done_files = 0
        for fut in as_completed(future_map):
            rel = future_map[fut]
            out = fut.result()
            done_files += 1
            total_games_used += out["games_used"]
            n_raw = out["n_raw"]
            total_raw += n_raw
            if n_raw > 0:
                chunks_pb.append(out["piece_bbs"])
                chunks_ob.append(out["occupancy_bbs"])
                chunks_gs.append(out["game_states"])
                chunks_res.append(out["results"])
                chunks_cp.append(out["search_cp"])
                chunks_d.append(out["depth"])
            if done_files % 10 == 0 or done_files == len(discovered):
                print(
                    f"  parsed {done_files}/{len(discovered)} files "
                    f"(games={total_games_used:,}, raw quiet plies={total_raw:,}, "
                    f"elapsed={time.time() - t0:.1f}s)"
                )

    piece_bbs = np.concatenate(chunks_pb, axis=0)
    occupancy_bbs = np.concatenate(chunks_ob, axis=0)
    game_states = np.concatenate(chunks_gs, axis=0)
    results = np.concatenate(chunks_res, axis=0)
    search_cp = np.concatenate(chunks_cp, axis=0)
    depth = np.concatenate(chunks_d, axis=0)
    del chunks_pb, chunks_ob, chunks_gs, chunks_res, chunks_cp, chunks_d

    print(f"Deduplicating {len(results):,} raw quiet positions...")
    (
        piece_bbs,
        occupancy_bbs,
        game_states,
        results,
        search_cp,
        depth,
    ) = deduplicate_and_average(
        piece_bbs, occupancy_bbs, game_states, results, search_cp, depth
    )
    n_unique = len(results)
    print(
        f"Unique positions: {n_unique:,} "
        f"(dedup ratio {n_unique / max(total_raw, 1):.1%})"
    )

    if max_positions and 0 < max_positions < n_unique:
        idx = np.sort(
            np.random.default_rng(seed).choice(
                n_unique, size=int(max_positions), replace=False
            )
        )
        piece_bbs = piece_bbs[idx]
        occupancy_bbs = occupancy_bbs[idx]
        game_states = game_states[idx]
        results = results[idx]
        search_cp = search_cp[idx]
        depth = depth[idx]
        print(f"Subsampled to {len(results):,} unique positions (seed={seed}).")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        piece_bbs=piece_bbs,
        occupancy_bbs=occupancy_bbs,
        game_states=game_states,
        results=results,
        search_cp=search_cp,
        depth=depth,
    )

    castling_frac = float(np.mean(game_states[:, 1] != 0))
    ep_frac = float(np.mean(game_states[:, 2] < 64))
    cp_valid_frac = float(np.mean(np.isfinite(search_cp)))
    elapsed = time.time() - t0
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(
        f"Saved {output_path} ({size_mb:.1f} MB) in {elapsed:.1f}s:\n"
        f"  positions      : {len(results):,}\n"
        f"  games used     : {total_games_used:,} across {len(discovered)} files\n"
        f"  mean WDL       : {float(results.mean()):.4f} (std={float(results.std()):.4f})\n"
        f"  castling != 0  : {castling_frac:.2%} ({int(np.count_nonzero(game_states[:, 1] != 0)):,})\n"
        f"  legal EP       : {ep_frac:.2%}\n"
        f"  valid search_cp: {cp_valid_frac:.2%}"
    )
    return {
        "output_path": str(output_path),
        "positions": int(len(results)),
        "games_used": total_games_used,
        "files_used": len(discovered),
        "castling_frac": castling_frac,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build independent PGN validation dataset (val_pgn.npz) from engine self-play PGNs."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-median-nodes", type=int, default=150_000)
    parser.add_argument("--min-setup-ply", type=int, default=6)
    parser.add_argument("--min-startpos-ply", type=int, default=16)
    parser.add_argument(
        "--max-positions",
        type=int,
        default=0,
        help="Optional cap on unique positions (0 = keep all unique positions)",
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_pgn_validation(
        output_path=args.output,
        min_median_nodes=args.min_median_nodes,
        min_setup_ply=args.min_setup_ply,
        min_startpos_ply=args.min_startpos_ply,
        max_positions=args.max_positions,
        workers=args.workers,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
