#!/usr/bin/env python3
"""
Match-based SPSA for classical HCE eval weights.

ONE process, ONE Numba JIT compile:
  - Two long-lived SearchContexts (θ+, θ−)
  - In-place eval_weights mutation (StructRef field — live under nopython)
  - Fixed-node games; no subprocess spawn per iteration

Protocol (default L1):
  - each iteration: simultaneous perturbation θ±c·Δ
  - ~games games @ nodes (paired color swap)
  - update θ with SPSA gradient estimate
  - checkpoint every N iters; optional L2 accept @ more games/nodes

Usage (repo root):
  python -u tune_eval_match/spsa_match.py
  python -u tune_eval_match/spsa_match.py --games 100 --nodes 2000 --iters 200
  python -u tune_eval_match/spsa_match.py --resume tournament_analysis/spsa_eval/checkpoint.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tune_eval_match.eval_weights import (  # noqa: E402
    default_eval_weights,
    params_from_weights,
)
from tune_eval_match.inprocess_match import (  # noqa: E402
    expand_worker_pool,
    load_openings,
    make_worker_pool,
    run_match,
    set_eval_params,
    verify_live_weights,
    warmup_jit,
)
from tune_eval_match.param_config import (  # noqa: E402
    BLOCKS,
    active_block,
    clip_params_float,
    default_params,
    param_names,
    quantize_params,
    r_end,
    set_active_block,
    step_c,
)


def elo_from_score(score: float) -> float:
    score = min(max(score, 1e-6), 1.0 - 1e-6)
    return -400.0 * math.log10(1.0 / score - 1.0)


def best_from_l2_ints(theta_int: dict) -> dict:
    """L2-proven weights: store as integer-valued floats (what the match actually used)."""
    return {k: float(int(v)) for k, v in theta_int.items()}


def spsa_a_end(c: float, r: float) -> float:
    """Stockfish-style: a_end = r * c^2."""
    return r * c * c


def load_checkpoint(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="In-process match SPSA for HCE eval weights")
    # Defaults (i5-12500H / 16GB class): permanent production protocol.
    #   L1 250 @ n50k, L2 250 @ n200k every 10 iters, -c 10, TT 4MB, float θ.
    p.add_argument("--games", type=int, default=250, help="Games per SPSA iteration (L1)")
    p.add_argument("--nodes", type=int, default=20000, help="Fixed nodes per move (L1)")
    p.add_argument("--iters", type=int, default=500, help="SPSA iterations (0 = infinite)")
    p.add_argument("--A", type=float, default=50.0, help="SPSA A stability constant")
    p.add_argument("--alpha", type=float, default=0.602, help="SPSA alpha")
    p.add_argument("--gamma", type=float, default=0.101, help="SPSA gamma")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--tt-mb", type=int, default=4,
        help="TT size per SearchContext in MB (4 is enough for n50k–n200k; "
             "history tables dominate memory, not TT)",
    )
    p.add_argument(
        "-c", "--concurrency", type=int, default=10,
        help="Parallel opening-pairs via ThreadPool of independent SearchContext pairs "
             "(still ONE process / ONE JIT). Default 10 for 12C/16T + 16GB.",
    )
    p.add_argument("--openings", type=str, default="", help="EPD/FEN openings file")
    p.add_argument("--out-dir", type=str, default="tournament_analysis/spsa_eval")
    p.add_argument("--resume", type=str, default="", help="checkpoint.json path")
    p.add_argument("--ckpt-every", type=int, default=5)
    p.add_argument("--log-every-games", type=int, default=20)
    p.add_argument("--l2-every", type=int, default=25, help="L2 accept every N iters (0=off)")
    p.add_argument("--l2-games", type=int, default=250, help="Games for L2 accept match")
    p.add_argument(
        "--l2-nodes", type=int, default=200000,
        help="Nodes per move for L2 (default 200k; stronger accept than L1 n20k)",
    )
    p.add_argument(
        "--l2-early-trash-games", type=int, default=150,
        help="L2 early-trash: after this many games, if score_A < threshold → reject",
    )
    p.add_argument(
        "--l2-early-trash-score", type=float, default=0.45,
        help="L2 early-trash score_A upper bound (θ hopeless vs best)",
    )
    p.add_argument(
        "--l2-stop-after-fails", type=int, default=2,
        help="Stop SPSA after this many consecutive L2 rejects (0=never stop on fails)",
    )
    # Optional one-shot promote (used by the in-flight run that started lighter).
    # When phase2 targets == defaults, upgrade is a no-op after first L2.
    p.add_argument(
        "--phase2-after-l2", action="store_true", default=True,
        help="After first L2, switch to phase2 L1/L2/-c if different from current",
    )
    p.add_argument("--no-phase2-after-l2", action="store_true", help="Disable phase2 upgrade")
    p.add_argument("--phase2-c", type=int, default=10, help="Concurrency after first L2")
    p.add_argument("--phase2-games", type=int, default=250, help="L1 games after first L2")
    p.add_argument("--phase2-nodes", type=int, default=20000, help="L1 nodes after first L2")
    p.add_argument("--phase2-l2-games", type=int, default=250, help="L2 games after first L2")
    p.add_argument("--phase2-l2-nodes", type=int, default=200000, help="L2 nodes after first L2")
    p.add_argument("--smoke", action="store_true", help="Tiny run for wiring check")
    p.add_argument("--no-verify", action="store_true")
    p.add_argument(
        "--block",
        type=str,
        default="a",
        choices=sorted(BLOCKS.keys()),
        help="SPSA parameter block: a | passed | material (see ROADMAP_BLOCKS.md)",
    )
    args = p.parse_args(argv)
    if args.no_phase2_after_l2:
        args.phase2_after_l2 = False

    block_name = set_active_block(args.block)
    print(f"[block] active={block_name}  params={param_names()}", flush=True)

    if args.smoke:
        args.games = 4
        args.nodes = 800
        args.iters = 3
        args.l2_every = 0
        args.ckpt_every = 1
        args.concurrency = max(1, min(args.concurrency, 2))
        args.tt_mb = min(args.tt_mb, 4)

    args.concurrency = max(1, int(args.concurrency))
    args.tt_mb = max(1, int(args.tt_mb))

    # Default out-dir per block if still the legacy path
    if args.out_dir == "tournament_analysis/spsa_eval" and block_name != "a":
        args.out_dir = f"tournament_analysis/spsa_eval_{block_name}"

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "spsa_log.jsonl"
    ckpt_path = out_dir / "checkpoint.json"
    best_path = out_dir / "best_params.json"

    rng = random.Random(args.seed)
    names = param_names()

    # --- State ---
    start_iter = 0
    theta = default_params()
    best_theta = dict(theta)
    best_l2_score = 0.5
    history_scores = []
    l2_count = 0
    l2_fail_streak = 0
    l2_skip_no_int = 0
    last_l2_gate_iter = -1  # last time we considered L2 (ran or skipped)
    phase2_active = False

    if args.resume:
        ck = load_checkpoint(Path(args.resume))
        # Float state if present; old int checkpoints still load fine
        theta = clip_params_float(ck["theta"])
        # Prefer explicit best int snapshot; else quantize stored best (snap float→L2 ints)
        if ck.get("best_theta_int"):
            best_theta = best_from_l2_ints(ck["best_theta_int"])
        else:
            best_theta = best_from_l2_ints(quantize_params(clip_params_float(ck.get("best_theta", theta))))
        best_l2_score = float(ck.get("best_l2_score", 0.5))
        start_iter = int(ck.get("iter", 0))
        l2_count = int(ck.get("l2_count", 0))
        l2_fail_streak = int(ck.get("l2_fail_streak", 0))
        l2_skip_no_int = int(ck.get("l2_skip_no_int", 0))
        last_l2_gate_iter = int(ck.get("last_l2_gate_iter", -1))
        phase2_active = bool(ck.get("phase2_active", False))
        print(f"[resume] iter={start_iter} theta from {args.resume}", flush=True)
        if phase2_active or l2_count >= 1:
            # Restore heavier protocol if already past first L2
            args.concurrency = max(args.concurrency, int(ck.get("concurrency", args.phase2_c) or args.phase2_c))
            if phase2_active or l2_count >= 1:
                args.games = int(ck.get("games", args.phase2_games) or args.phase2_games)
                args.nodes = int(ck.get("nodes", args.phase2_nodes) or args.phase2_nodes)
                args.l2_games = int(ck.get("l2_games", args.phase2_l2_games) or args.phase2_l2_games)
                args.l2_nodes = int(ck.get("l2_nodes", args.phase2_l2_nodes) or args.phase2_l2_nodes)
                args.concurrency = int(ck.get("concurrency", args.phase2_c) or args.phase2_c)
                phase2_active = True
                print(
                    f"[resume] phase2: L1 {args.games}g@n{args.nodes}  "
                    f"L2 {args.l2_games}g@n{args.l2_nodes}  -c {args.concurrency}",
                    flush=True,
                )

    openings_path = Path(args.openings) if args.openings else None
    openings = load_openings(openings_path, limit=max(500, args.games * 2))
    print(f"[openings] {len(openings)} positions", flush=True)

    # --- One-time engine construction + single JIT ---
    print(
        f"[init] building worker pool concurrency={args.concurrency} "
        f"(one process, one JIT)...",
        flush=True,
    )
    worker_pool = make_worker_pool(args.concurrency, tt_mb=args.tt_mb)
    warmup_jit(worker_pool[0][0], nodes=min(5000, max(2000, args.nodes)))
    warmup_jit(worker_pool[0][1], nodes=min(1500, args.nodes))
    for i, (wa, wb) in enumerate(worker_pool[1:], start=1):
        print(f"[warmup] worker {i} touch...", flush=True)
        warmup_jit(wa, nodes=min(800, args.nodes))
        warmup_jit(wb, nodes=min(800, args.nodes))

    if not args.no_verify:
        if not verify_live_weights(worker_pool[0][0]):
            print("ERROR: eval_weights mutation not visible to Numba — abort.", flush=True)
            return 2
        set_eval_params(worker_pool[0][0], quantize_params(theta))

    meta = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "block": active_block(),
        "games": args.games,
        "nodes": args.nodes,
        "iters": args.iters,
        "concurrency": args.concurrency,
        "tt_mb": args.tt_mb,
        "A": args.A,
        "alpha": args.alpha,
        "gamma": args.gamma,
        "seed": args.seed,
        "params": names,
        "r_end": {n: r_end(n) for n in names},
        "c": {n: step_c(n) for n in names},
        "theta_mode": "float_accumulate_quantize_on_apply",
        "l2_every": args.l2_every,
        "phase2_after_l2": bool(args.phase2_after_l2),
        "phase2": {
            "c": args.phase2_c,
            "games": args.phase2_games,
            "nodes": args.phase2_nodes,
            "l2_games": args.phase2_l2_games,
            "l2_nodes": args.phase2_l2_nodes,
        },
        "pid": os.getpid(),
        "note": "in-process match SPSA; single process / single JIT; ThreadPool -c; float θ",
        "roadmap": "a -> passed -> material (see tune_eval_match/ROADMAP_BLOCKS.md)",
    }
    save_json(out_dir / "run_meta.json", meta)
    print(json.dumps(meta, indent=2), flush=True)
    print(f"[run] log={log_path}", flush=True)

    iter_i = start_iter
    max_iters = args.iters if args.iters > 0 else 10**9
    t_run0 = time.time()

    while iter_i < max_iters:
        # SPSA simultaneous perturbation (float)
        delta = {n: (1 if rng.random() < 0.5 else -1) for n in names}
        c_k = {}
        a_k = {}
        theta_p = {}
        theta_m = {}
        for n in names:
            c = step_c(n) / ((iter_i + 1) ** args.gamma)
            c_k[n] = c
            a0 = spsa_a_end(step_c(n), r_end(n)) * ((args.A + 1) ** args.alpha)
            a = a0 / ((args.A + iter_i + 1) ** args.alpha)
            a_k[n] = a
            theta_p[n] = theta[n] + c * delta[n]
            theta_m[n] = theta[n] - c * delta[n]
        theta_p = clip_params_float(theta_p)
        theta_m = clip_params_float(theta_m)
        # Engine only sees integers
        theta_p_i = quantize_params(theta_p)
        theta_m_i = quantize_params(theta_m)

        print(
            f"\n=== SPSA iter {iter_i}  games={args.games} nodes={args.nodes} "
            f"c={args.concurrency} ===",
            flush=True,
        )
        match = run_match(
            theta_p_i,
            theta_m_i,
            openings=openings,
            nodes=args.nodes,
            games=args.games,
            log_every=args.log_every_games,
            worker_pool=worker_pool,
            concurrency=args.concurrency,
        )
        score_plus = match["score_a"]
        y_diff = 2.0 * score_plus - 1.0
        elo = elo_from_score(score_plus)

        new_theta = {}
        for n in names:
            g = y_diff / (2.0 * c_k[n] * delta[n])
            new_theta[n] = theta[n] + a_k[n] * g
        theta = clip_params_float(new_theta)
        theta_i = quantize_params(theta)
        history_scores.append(score_plus)

        # Health: how far float θ moved / whether int weights changed
        step_l1 = sum(abs(a_k[n] * y_diff / (2.0 * c_k[n] * delta[n])) for n in names)

        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "iter": iter_i,
            "score_plus": score_plus,
            "elo_plus_vs_minus": elo,
            "pts_plus": match["pts_a"],
            "pts_minus": match["pts_b"],
            "games": match["games"],
            "nodes": args.nodes,
            "match_s": match["elapsed_s"],
            "theta": dict(theta),
            "theta_int": dict(theta_i),
            "theta_plus_int": theta_p_i,
            "theta_minus_int": theta_m_i,
            "delta": delta,
            "y_diff": y_diff,
            "step_l1_float": step_l1,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        # Compact progress line (A-block highlights)
        fa = theta_i.get("KING_FLANK_ATTACK_NUM", "?")
        out_mg = theta_i.get("KING_DANGER_OUT_MG_NUM", "?")
        sh_mg = theta_i.get("SHELTER_BASE_MG", "?")
        print(
            f"[iter {iter_i}] score+={score_plus:.3f} ({elo:+.1f} Elo)  "
            f"games={match['games']}  {match['elapsed_s']:.1f}s  "
            f"flankAtt={fa} outMg={out_mg} shBase={sh_mg}  "
            f"stepL1={step_l1:.3f}",
            flush=True,
        )

        # L2 accept: quantized θ vs quantized best
        # scheduled every l2_every; catch-up if we never handled a gate past the first window.
        l2_scheduled = args.l2_every > 0 and (iter_i + 1) % args.l2_every == 0
        l2_catchup = (
            args.phase2_after_l2
            and l2_count == 0
            and last_l2_gate_iter < 0
            and args.l2_every > 0
            and (iter_i + 1) >= args.l2_every
        )
        if l2_scheduled or l2_catchup:
            why = "scheduled" if l2_scheduled else "catch-up-first"
            last_l2_gate_iter = iter_i
            theta_i_gate = quantize_params(theta)
            best_i_gate = quantize_params(best_theta)
            # If all SPSA ints equal best, L2 is θ≡best — skip and wait another l2_every L1s.
            if theta_i_gate == best_i_gate:
                l2_skip_no_int += 1
                next_gate = iter_i + args.l2_every
                print(
                    f"\n--- L2 SKIP @ iter {iter_i} ({why}): "
                    f"theta_int == best_int (no integer change) ---",
                    flush=True,
                )
                print(
                    f"[L2] skip match; continue L1 for another {args.l2_every} iters "
                    f"(next L2 gate ~iter {next_gate}; skips={l2_skip_no_int})",
                    flush=True,
                )
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "ts": datetime.now(timezone.utc).isoformat(),
                                "type": "l2_skip_no_int_change",
                                "iter": iter_i,
                                "why": why,
                                "l2_every": args.l2_every,
                                "next_l2_gate_iter": next_gate,
                                "l2_skip_no_int": l2_skip_no_int,
                                "theta_int": theta_i_gate,
                                "best_theta_int": best_i_gate,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            else:
                print(
                    f"\n--- L2 accept @ iter {iter_i} ({why}): "
                    f"{args.l2_games}g @ n{args.l2_nodes} vs best ---",
                    flush=True,
                )
                l2 = run_match(
                    theta_i_gate,
                    best_i_gate,
                    openings=openings,
                    nodes=args.l2_nodes,
                    games=args.l2_games,
                    log_every=max(20, args.log_every_games),
                    worker_pool=worker_pool,
                    concurrency=args.concurrency,
                    early_trash_min_games=int(args.l2_early_trash_games),
                    early_trash_max_score=float(args.l2_early_trash_score),
                )
                l2_score = l2["score_a"]
                l2_elo = elo_from_score(l2_score)
                l2_trash = bool(l2.get("early_trash"))
                print(
                    f"[L2] theta vs best: score={l2_score:.3f} ({l2_elo:+.1f} Elo)  "
                    f"games={l2['games']}  {l2['elapsed_s']:.1f}s"
                    f"{'  EARLY_TRASH' if l2_trash else ''}",
                    flush=True,
                )
                l2_row = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "type": "l2_accept",
                    "iter": iter_i,
                    "score_theta_vs_best": l2_score,
                    "elo": l2_elo,
                    "theta": dict(theta),
                    "theta_int": dict(theta_i_gate),
                    "best_theta": dict(best_theta),
                    "best_theta_int": dict(best_i_gate),
                    "games": l2["games"],
                    "nodes": args.l2_nodes,
                    "early_trash": l2_trash,
                }
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(l2_row, ensure_ascii=False) + "\n")
                # Accept if θ beats the current best_theta by the gate (default ≥0.52).
                # Early trash ⇒ automatic reject (already score_A < 0.45).
                l2_accepted = False
                if (not l2_trash) and l2_score >= 0.52:
                    # best = exact ints used in L2; working θ keeps float (preserves sub-int progress).
                    best_theta = best_from_l2_ints(theta_i_gate)
                    best_l2_score = l2_score  # bookkeeping only
                    l2_accepted = True
                    l2_fail_streak = 0
                    save_json(
                        best_path,
                        {
                            "theta": best_theta,
                            "theta_int": dict(theta_i_gate),
                            "l2_score": best_l2_score,
                            "iter": iter_i,
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "note": "best = L2 match ints; working θ NOT re-centered (stays float)",
                        },
                    )
                    print(
                        f"[L2] NEW BEST accepted score={best_l2_score:.3f} vs prior best "
                        f"(best=L2 ints; θ keeps float)",
                        flush=True,
                    )
                elif l2_trash:
                    l2_fail_streak += 1
                    print(
                        f"[L2] reject EARLY_TRASH "
                        f"(score={l2_score:.3f} < {args.l2_early_trash_score} "
                        f"after {l2['games']} games)  fail_streak={l2_fail_streak}",
                        flush=True,
                    )
                else:
                    l2_fail_streak += 1
                    print(
                        f"[L2] reject (need score>=0.52 vs best_theta, got {l2_score:.3f})  "
                        f"fail_streak={l2_fail_streak}",
                        flush=True,
                    )

                l2_count += 1
                l2_row["accepted"] = l2_accepted
                l2_row["l2_fail_streak"] = l2_fail_streak
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "ts": datetime.now(timezone.utc).isoformat(),
                                "type": "l2_result",
                                "iter": iter_i,
                                "accepted": l2_accepted,
                                "early_trash": l2_trash,
                                "score": l2_score,
                                "l2_fail_streak": l2_fail_streak,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                if (
                    args.l2_stop_after_fails > 0
                    and l2_fail_streak >= args.l2_stop_after_fails
                ):
                    print(
                        f"[stop] {l2_fail_streak} consecutive L2 rejects "
                        f"(limit={args.l2_stop_after_fails}). Stopping SPSA.",
                        flush=True,
                    )
                    save_json(
                        ckpt_path,
                        {
                            "iter": iter_i + 1,
                            "theta": theta,
                            "theta_int": quantize_params(theta),
                            "best_theta": best_theta,
                            "best_theta_int": quantize_params(best_theta),
                            "best_l2_score": best_l2_score,
                            "recent_scores": history_scores[-50:],
                            "elapsed_run_s": time.time() - t_run0,
                            "l2_count": l2_count,
                            "l2_fail_streak": l2_fail_streak,
                            "l2_skip_no_int": l2_skip_no_int,
                            "last_l2_gate_iter": last_l2_gate_iter,
                            "phase2_active": phase2_active,
                            "games": args.games,
                            "nodes": args.nodes,
                            "l2_games": args.l2_games,
                            "l2_nodes": args.l2_nodes,
                            "concurrency": args.concurrency,
                            "stopped_reason": "consecutive_l2_rejects",
                            "meta": meta,
                        },
                    )
                    save_json(
                        out_dir / "final_params.json",
                        {
                            "theta": theta,
                            "best_theta": best_theta,
                            "best_theta_int": quantize_params(best_theta),
                            "stopped_reason": "consecutive_l2_rejects",
                            "l2_fail_streak": l2_fail_streak,
                            "iter": iter_i,
                        },
                    )
                    return 0

                # Promote to heavier protocol once (same process / no re-JIT)
                if args.phase2_after_l2 and not phase2_active and l2_count >= 1:
                    phase2_active = True
                    old = (
                        f"L1 {args.games}g@n{args.nodes} L2 {args.l2_games}g@n{args.l2_nodes} "
                        f"c={args.concurrency}"
                    )
                    args.games = int(args.phase2_games)
                    args.nodes = int(args.phase2_nodes)
                    args.l2_games = int(args.phase2_l2_games)
                    args.l2_nodes = int(args.phase2_l2_nodes)
                    target_c = max(1, int(args.phase2_c))
                    if target_c > args.concurrency:
                        worker_pool = expand_worker_pool(
                            worker_pool, target_c, tt_mb=args.tt_mb, warmup_nodes=800
                        )
                    args.concurrency = target_c
                    print(
                        f"[phase2] after L2#{l2_count}: {old}  ->  "
                        f"L1 {args.games}g@n{args.nodes} L2 {args.l2_games}g@n{args.l2_nodes} "
                        f"c={args.concurrency}",
                        flush=True,
                    )
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(
                            json.dumps(
                                {
                                    "ts": datetime.now(timezone.utc).isoformat(),
                                    "type": "phase2_upgrade",
                                    "iter": iter_i,
                                    "l2_count": l2_count,
                                    "games": args.games,
                                    "nodes": args.nodes,
                                    "l2_games": args.l2_games,
                                    "l2_nodes": args.l2_nodes,
                                    "concurrency": args.concurrency,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

        if (iter_i + 1) % args.ckpt_every == 0 or iter_i == 0 or phase2_active:
            save_json(
                ckpt_path,
                {
                    "iter": iter_i + 1,
                    "theta": theta,
                    "theta_int": quantize_params(theta),
                    "best_theta": best_theta,
                    "best_theta_int": quantize_params(best_theta),
                    "best_l2_score": best_l2_score,
                    "recent_scores": history_scores[-50:],
                    "elapsed_run_s": time.time() - t_run0,
                    "l2_count": l2_count,
                    "l2_fail_streak": l2_fail_streak,
                    "l2_skip_no_int": l2_skip_no_int,
                    "last_l2_gate_iter": last_l2_gate_iter,
                    "phase2_active": phase2_active,
                    "games": args.games,
                    "nodes": args.nodes,
                    "l2_games": args.l2_games,
                    "l2_nodes": args.l2_nodes,
                    "concurrency": args.concurrency,
                    "meta": meta,
                },
            )
            save_json(
                out_dir / "latest_params.json",
                {
                    "theta": theta,
                    "theta_int": quantize_params(theta),
                    "iter": iter_i + 1,
                    "phase2_active": phase2_active,
                    "games": args.games,
                    "nodes": args.nodes,
                    "concurrency": args.concurrency,
                },
            )

        iter_i += 1

    print(f"\n[done] {iter_i} iterations, elapsed {time.time() - t_run0:.1f}s", flush=True)
    save_json(out_dir / "final_params.json", {"theta": theta, "best_theta": best_theta})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
