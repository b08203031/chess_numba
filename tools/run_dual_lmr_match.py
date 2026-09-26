#!/usr/bin/env python3
"""
Dual New-vs-Old match with unified JIT barriers (like tournament -c 4):

  Match A: New CHESS_LMR_TABLE_SCALE=100 vs Old   (-c 2 workers)
  Match B: New CHESS_LMR_TABLE_SCALE=85  vs Old   (-c 2 workers)

Compile schedule:
  1) Start + warm 4 New engines in parallel (2× scale100 + 2× scale85)
  2) Barrier — all New ready
  3) Start + warm 4 Old engines in parallel
  4) Barrier — all Old ready
  5) Both matches play (--nodes 200000 by default)

PGNs:
  tournament_analysis/tournament_results_lmr100.pgn
  tournament_analysis/tournament_results_lmr85.pgn

Usage (repo root):
  python tools/run_dual_lmr_match.py --nodes 200000 --games 100
  python tools/run_dual_lmr_match.py --nodes 200000 --games 100 --no-clear-cache
"""
from __future__ import annotations

import argparse
import math
import os
import queue
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.tournament import (  # noqa: E402
    Engine,
    SPRTTest,
    calculate_elo_statistics,
    clear_numba_cache,
    count_result,
    play_game,
)


def _load_openings():
    openings = [None]
    path = ROOT / "data" / "openings.epd"
    if path.exists():
        lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if lines:
            openings = lines
    import random

    random.shuffle(openings)
    return openings


def _run_match_side(
    *,
    worker_ids: list[int],
    global_barrier_new: threading.Barrier,
    global_barrier_old: threading.Barrier,
    engine1_path: str,
    engine2_path: str,
    engine1_name: str,
    engine2_name: str,
    engine1_env: dict,
    games: int,
    time_ms: int,
    nodes: int | None,
    depth: int | None,
    pgn_file: Path,
    openings: list,
    print_lock: threading.Lock,
    label: str,
):
    """Run one New-vs-Old match using a subset of workers; share global JIT barriers."""
    pairs_count = games // 2
    pair_queue: queue.Queue = queue.Queue()
    for pair_i in range(1, pairs_count + 1):
        pair_queue.put(pair_i)

    scores = {engine1_name: 0.0, engine2_name: 0.0}
    e1_stats = {"wins": 0, "draws": 0, "losses": 0}
    sprt = SPRTTest(elo0=0.0, elo1=10.0, alpha=0.05, beta=0.05)
    status, llr = "Continue", 0.0
    total_matches = 0
    state_lock = threading.Lock()
    sprt_stopped = threading.Event()

    pgn_file.parent.mkdir(parents=True, exist_ok=True)
    pgn_file.write_text("", encoding="utf-8")

    def worker(local_id: int, global_id: int):
        nonlocal total_matches, status, llr
        verbose = global_id == 0  # only first global worker prints init/warmup detail
        # Unique process names for NUMBA_CACHE_DIR; rename for PGN after start.
        e1 = Engine(
            f"{engine1_name}_w{global_id}",
            engine1_path,
            verbose=verbose,
            env_extra=engine1_env,
        )
        e2 = Engine(
            f"{engine2_name}_{label}_w{global_id}",
            engine2_path,
            verbose=verbose and local_id == 0,
            env_extra=None,
        )

        try:
            with print_lock:
                print(f"[{label}] worker {global_id}: start NEW scale={engine1_env.get('CHESS_LMR_TABLE_SCALE')} ...", flush=True)
            e1.start()
            e1.warm_up()
            e1.name = engine1_name
            with print_lock:
                print(f"[{label}] worker {global_id}: NEW ready — waiting global New barrier", flush=True)
            try:
                global_barrier_new.wait()
            except threading.BrokenBarrierError:
                return

            with print_lock:
                print(f"[{label}] worker {global_id}: start OLD ...", flush=True)
            e2.start()
            e2.warm_up()
            e2.name = engine2_name
            with print_lock:
                print(f"[{label}] worker {global_id}: OLD ready — waiting global Old barrier", flush=True)
            try:
                global_barrier_old.wait()
            except threading.BrokenBarrierError:
                return

            with print_lock:
                if global_id == worker_ids[0]:
                    print(f"[{label}] all engines ready — playing (nodes={nodes})", flush=True)

            while not sprt_stopped.is_set():
                try:
                    pair_i = pair_queue.get_nowait()
                except queue.Empty:
                    break

                opening_fen = openings[(pair_i - 1) % len(openings)]
                pair_verbose = verbose and pair_i == 1

                game_num_a = 2 * pair_i - 1
                res_a, pgn_a, log_a = play_game(
                    e1, e2, time_ms, game_num_a, opening_fen, depth=depth, nodes=nodes, verbose=pair_verbose
                )
                game_num_b = 2 * pair_i
                res_b, pgn_b, log_b = play_game(
                    e2, e1, time_ms, game_num_b, opening_fen, depth=depth, nodes=nodes, verbose=pair_verbose
                )

                with state_lock:
                    if sprt_stopped.is_set():
                        pair_queue.task_done()
                        break

                    if not pair_verbose:
                        with print_lock:
                            print(log_a + log_b, end="", flush=True)

                    score_a = count_result(res_a, is_white=True)
                    score_b = count_result(res_b, is_white=False)
                    pair_score = score_a + score_b
                    sprt.add_pair_result(pair_score)
                    scores[engine1_name] += pair_score
                    scores[engine2_name] += 2.0 - pair_score
                    for s in (score_a, score_b):
                        if s == 1.0:
                            e1_stats["wins"] += 1
                        elif s == 0.0:
                            e1_stats["losses"] += 1
                        else:
                            e1_stats["draws"] += 1

                    with open(pgn_file, "a", encoding="utf-8") as f:
                        f.write(str(pgn_a) + "\n\n")
                        f.write(str(pgn_b) + "\n\n")

                    status, llr = sprt.check_status()
                    total_matches += 2
                    with print_lock:
                        print(
                            f"\n[{label}] After {len(sprt.pair_scores)} pairs "
                            f"({len(sprt.pair_scores) * 2} games):",
                            flush=True,
                        )
                        print(f"  {engine1_name}: {scores[engine1_name]}", flush=True)
                        print(f"  {engine2_name}: {scores[engine2_name]}", flush=True)
                        print(
                            f"  SPRT LLR: {llr:.2f} bounds: [{sprt.b:.2f}, {sprt.a:.2f}]  status={status}",
                            flush=True,
                        )
                        print("-" * 40, flush=True)

                    if status != "Continue":
                        with print_lock:
                            print(f"[{label}] SPRT stop.", flush=True)
                        sprt_stopped.set()
                        pair_queue.task_done()
                        break
                pair_queue.task_done()
        except Exception as e:
            import traceback

            with print_lock:
                print(f"[{label}] worker {global_id} error: {e}\n{traceback.format_exc()}", flush=True)
            try:
                global_barrier_new.abort()
            except Exception:
                pass
            try:
                global_barrier_old.abort()
            except Exception:
                pass
        finally:
            e1.terminate()
            e2.terminate()

    threads = []
    for local_id, gid in enumerate(worker_ids):
        t = threading.Thread(target=worker, args=(local_id, gid), daemon=False)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    elo_diff, err = calculate_elo_statistics(e1_stats["wins"], e1_stats["draws"], e1_stats["losses"])
    with print_lock:
        print(f"\n=== [{label}] Finished ===", flush=True)
        print(f"Games: {total_matches}  {engine1_name}: {scores[engine1_name]}  "
              f"({e1_stats['wins']}W-{e1_stats['draws']}D-{e1_stats['losses']}L)", flush=True)
        print(f"  {engine2_name}: {scores[engine2_name]}", flush=True)
        print(f"  SPRT: {status} LLR={llr:.2f}", flush=True)
        if math.isinf(elo_diff):
            print("  Elo: extreme score", flush=True)
        else:
            print(f"  Elo ({engine1_name}-{engine2_name}): {elo_diff:+.2f} ± {err:.2f}", flush=True)
        print(f"  PGN: {pgn_file}", flush=True)

    return {
        "label": label,
        "scores": scores,
        "stats": e1_stats,
        "status": status,
        "llr": llr,
        "games": total_matches,
        "pgn": str(pgn_file),
    }


def main():
    ap = argparse.ArgumentParser(description="Dual LMR-scale matches with unified New/Old JIT barriers")
    ap.add_argument("--games", type=int, default=100, help="Games per match (even; default 100)")
    ap.add_argument("--nodes", type=int, default=200000)
    ap.add_argument("--time", type=int, default=1000, help="Unused if --nodes set")
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--c-per-match", type=int, default=2, help="Workers per match (default 2 → 4 total)")
    ap.add_argument("--no-clear-cache", action="store_true")
    args = ap.parse_args()

    if args.games % 2 != 0:
        print("--games must be even", file=sys.stderr)
        sys.exit(1)

    c = args.c_per_match
    total_workers = c * 2
    if total_workers < 2:
        print("need at least 1 worker per match", file=sys.stderr)
        sys.exit(1)

    if not args.no_clear_cache:
        clear_numba_cache(str(ROOT))

    engine1 = str(ROOT / "main.py")
    engine2 = str(ROOT / "main_old.py")
    openings = _load_openings()
    print_lock = threading.Lock()

    # Global barriers: all 4 New first, then all 4 Old
    barrier_new = threading.Barrier(total_workers)
    barrier_old = threading.Barrier(total_workers)

    out_dir = ROOT / "tournament_analysis"
    pgn100 = out_dir / "tournament_results_lmr100.pgn"
    pgn85 = out_dir / "tournament_results_lmr85.pgn"

    print("=" * 60, flush=True)
    print("Dual LMR match", flush=True)
    print(f"  Match A: New scale=100 vs Old  workers={c}  nodes={args.nodes}  games={args.games}", flush=True)
    print(f"  Match B: New scale=85  vs Old  workers={c}  nodes={args.nodes}  games={args.games}", flush=True)
    print(f"  JIT: {total_workers} New engines in parallel → barrier → {total_workers} Old engines → play", flush=True)
    print(f"  PGN A: {pgn100}", flush=True)
    print(f"  PGN B: {pgn85}", flush=True)
    print("=" * 60, flush=True)

    results = [None, None]

    def run_a():
        results[0] = _run_match_side(
            worker_ids=list(range(0, c)),
            global_barrier_new=barrier_new,
            global_barrier_old=barrier_old,
            engine1_path=engine1,
            engine2_path=engine2,
            engine1_name="New100",
            engine2_name="Old",
            engine1_env={"CHESS_LMR_TABLE_SCALE": "100"},
            games=args.games,
            time_ms=args.time,
            nodes=args.nodes,
            depth=args.depth,
            pgn_file=pgn100,
            openings=openings,
            print_lock=print_lock,
            label="LMR100",
        )

    def run_b():
        results[1] = _run_match_side(
            worker_ids=list(range(c, c * 2)),
            global_barrier_new=barrier_new,
            global_barrier_old=barrier_old,
            engine1_path=engine1,
            engine2_path=engine2,
            engine1_name="New85",
            engine2_name="Old",
            engine1_env={"CHESS_LMR_TABLE_SCALE": "85"},
            games=args.games,
            time_ms=args.time,
            nodes=args.nodes,
            depth=args.depth,
            pgn_file=pgn85,
            openings=openings,
            print_lock=print_lock,
            label="LMR85",
        )

    t_a = threading.Thread(target=run_a, name="match_lmr100")
    t_b = threading.Thread(target=run_b, name="match_lmr85")
    t_a.start()
    t_b.start()
    t_a.join()
    t_b.join()

    print("\n======== DUAL MATCH SUMMARY ========", flush=True)
    for r in results:
        if r:
            print(r, flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
