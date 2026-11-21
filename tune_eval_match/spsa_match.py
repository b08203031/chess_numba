#!/usr/bin/env python3
"""
Match-based SPSA for classical HCE eval weights.

ONE process, ONE Numba JIT compile:
  - ThreadPool of long-lived SearchContext pairs (θ+, θ− / candidate vs best)
  - In-place eval_weights mutation (StructRef field — live under nopython)
  - Fixed-node games with disjoint L1/L2 opening streams and score adjudication

Typical workflow:
  python -u tune_eval_match/spsa_match.py --block threats_mg \
    --nodes 80000 --games 500 --l2-every 10 --l2-nodes 200000 --l2-games 500
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tune_eval_match.eval_weights import (  # noqa: E402
    EVAL_PARAM_INDEX,
    format_constants_diff,
)
from tune_eval_match.param_config import (  # noqa: E402
    BLOCKS,
    active_block,
    bounds,
    clip_params_float,
    default_params,
    param_names,
    quantize_params,
    r_end,
    select_custom_params,
    set_active_block,
    step_c,
)
from tune_search.stats import (  # noqa: E402
    drift_z,
    elo_estimate,
    merge_penta,
    sprt_bounds,
    sprt_decision,
    sprt_llr,
)


def default_concurrency() -> int:
    cpus = os.cpu_count() or 4
    return max(1, min(8, cpus - 2))


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
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _average_tail_thetas(window_thetas: list[dict], names: list[str]) -> dict:
    """Polyak-Ruppert average over the second half of the current L1 window."""
    tail = window_thetas[len(window_thetas) // 2 :] if len(window_thetas) > 1 else window_thetas
    n = max(1, len(tail))
    return {k: sum(float(row[k]) for row in tail) / n for k in names}


def _rng_state_to_json(rng: random.Random) -> list:
    """JSON-serialisable ``random.Random`` state (version, 625 ints, gauss_next)."""
    version, internal, gauss = rng.getstate()
    return [version, list(internal), gauss]


def _restore_rng(rng: random.Random, ck: dict, seed: int, start_iter: int) -> bool:
    """Restore the SPSA perturbation RNG from a checkpoint.

    Returns True if the exact state was restored.  Legacy checkpoints (written
    before ``rng_state`` was saved) would otherwise restart from ``seed`` and
    replay the first iterations' +/-1 deltas, so they get a seed derived from
    the resume iteration instead.
    """
    state = ck.get("rng_state")
    if isinstance(state, list) and len(state) == 3:
        version, internal, gauss = state
        rng.setstate((version, tuple(internal), gauss))
        return True
    rng.seed(int(seed) * 1_000_003 + int(start_iter))
    return False


def _parse_sprt(raw: str):
    if not raw or not raw.strip():
        return None
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 2:
        raise ValueError(f"--l2-sprt must look like ELO0,ELO1, got {raw!r}")
    elo0, elo1 = float(parts[0]), float(parts[1])
    if elo1 <= elo0:
        raise ValueError(f"--l2-sprt requires ELO1 > ELO0, got {raw!r}")
    return (elo0, elo1)


def _print_blocks_and_params() -> None:
    print("Available Match-SPSA blocks (with live constants defaults):")
    for bname, spec_map in BLOCKS.items():
        set_active_block(bname)
        live = default_params()
        print(f"\n[{bname}] ({len(spec_map)} params)")
        for pname, (_, lo, hi, c, r) in spec_map.items():
            print(f"  {pname:32s} cur={int(live[pname]):6d}  bounds=[{lo:g}, {hi:g}]  c={c:g}  r_end={r:g}")
    print(f"\nTotal runtime-wired eval params in EVAL_PARAM_INDEX: {len(EVAL_PARAM_INDEX)}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="In-process match SPSA for HCE eval weights")
    p.add_argument("--list-blocks", action="store_true", help="List all blocks with live defaults and exit")
    p.add_argument("--block", type=str, default="threats_mg", choices=sorted(BLOCKS.keys()),
                   help="SPSA parameter block (ignored if --params is passed)")
    p.add_argument("--params", type=str, default="",
                   help="Optional comma-separated custom parameter list from EVAL_PARAM_INDEX")
    # L1 / L2 Match Protocol (defaults match 80k L1 -> 200k L2 -> 400k final verify)
    p.add_argument("--games", type=int, default=500, help="Games per SPSA iteration (L1)")
    p.add_argument("--nodes", type=int, default=80000, help="Fixed nodes per move (L1)")
    p.add_argument("--iters", type=int, default=200, help="SPSA iterations (0 = infinite)")
    p.add_argument("--A", type=float, default=25.0, help="SPSA A stability constant")
    p.add_argument("--alpha", type=float, default=0.602, help="SPSA alpha")
    p.add_argument("--gamma", type=float, default=0.101, help="SPSA gamma")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for SPSA ±1 perturbations")
    p.add_argument("--tt-mb", type=int, default=8,
                   help="TT size per SearchContext in MB (8MB recommended for 80k-200k nodes)")
    p.add_argument("-c", "--concurrency", type=int, default=default_concurrency(),
                   help="Parallel opening-pairs via ThreadPool of independent SearchContext pairs")
    p.add_argument("--openings", type=str, default="", help="EPD/FEN openings file (defaults to data/openings.epd)")
    p.add_argument("--openings-limit", type=int, default=0,
                   help="Max openings to load (0 = load all positions in file)")
    p.add_argument("--opening-seed", type=int, default=20261119,
                   help="Private RNG seed for shuffling openings (decoupled from SPSA delta seed)")
    p.add_argument("--l2-holdout-frac", type=float, default=0.25,
                   help="Fraction of shuffled openings reserved exclusively for L2 validation")
    p.add_argument("--out-dir", type=str, default="",
                   help="Output directory (defaults to tournament_analysis/spsa_eval_<block>)")
    p.add_argument("--resume", type=str, default="", help="checkpoint.json path to resume from")
    p.add_argument("--ckpt-every", type=int, default=2)
    p.add_argument("--log-every-games", type=int, default=40)
    # L2 Gate Settings
    p.add_argument("--l2-every", type=int, default=10, help="L2 accept every N L1 iters (0=off)")
    p.add_argument("--l2-games", type=int, default=500, help="Max games for L2 accept match")
    p.add_argument("--l2-nodes", type=int, default=200000, help="Nodes per move for L2 validation")
    p.add_argument("--l2-batch-games", type=int, default=100,
                   help="Run L2 in batches of N games for Elo/CI/SPRT tracking")
    p.add_argument("--l2-candidate", type=str, default="tail", choices=("tail", "latest"),
                   help="Evaluate window tail-averaged θ (recommended) or latest θ at L2")
    p.add_argument("--l2-min-score", type=float, default=0.510,
                   help="Minimum L2 score vs best to accept when SPRT is off or reaches cap (default 0.510 ≈ +7 Elo)")
    p.add_argument("--l2-sprt", type=str, default="",
                   help="Optional pentanomial SPRT bounds ELO0,ELO1 for L2 (e.g. '0,10' or '-3,5')")
    p.add_argument("--sprt-alpha", type=float, default=0.05)
    p.add_argument("--sprt-beta", type=float, default=0.05)
    p.add_argument("--l2-early-trash-games", type=int, default=150,
                   help="L2 early-trash: after this many games, if score < threshold -> reject")
    p.add_argument("--l2-early-trash-score", type=float, default=0.45,
                   help="L2 early-trash score upper bound")
    p.add_argument("--l2-stop-after-fails", type=int, default=3,
                   help="Stop SPSA after this many consecutive L2 rejects (0=never stop)")
    # Adjudication
    p.add_argument("--adjudicate", action="store_true", default=True,
                   help="Use score-based win/draw adjudication (default: on)")
    p.add_argument("--no-adjudicate", action="store_true", help="Disable score adjudication")
    # Legacy phase2 flags kept opt-in (default OFF so user settings are never silently clobbered)
    p.add_argument("--phase2-after-l2", action="store_true", default=False,
                   help="Opt-in: switch protocol after first L2")
    p.add_argument("--no-phase2-after-l2", action="store_true", help="Legacy no-op (phase2 is off by default)")
    p.add_argument("--phase2-c", type=int, default=default_concurrency())
    p.add_argument("--phase2-games", type=int, default=500)
    p.add_argument("--phase2-nodes", type=int, default=80000)
    p.add_argument("--phase2-l2-games", type=int, default=500)
    p.add_argument("--phase2-l2-nodes", type=int, default=200000)
    # Smoke / verification
    p.add_argument("--smoke", action="store_true", help="Tiny run for end-to-end wiring check")
    p.add_argument("--no-verify", action="store_true")
    args = p.parse_args(argv)

    if args.list_blocks:
        _print_blocks_and_params()
        return 0

    try:
        l2_sprt = _parse_sprt(args.l2_sprt)
    except ValueError as exc:
        p.error(str(exc))

    if args.no_adjudicate:
        args.adjudicate = False
    if args.no_phase2_after_l2:
        args.phase2_after_l2 = False

    if args.params.strip():
        block_name = select_custom_params(args.params.split(","))
    else:
        block_name = set_active_block(args.block)

    names = param_names()
    print(f"[block] active={block_name} ({len(names)} params): {names}", flush=True)

    if args.smoke:
        args.games = 4
        args.nodes = 800
        args.iters = 3
        args.l2_every = 2
        args.l2_games = 4
        args.l2_nodes = 800
        args.l2_batch_games = 4
        args.l2_early_trash_games = 0
        args.ckpt_every = 1
        args.log_every_games = 4
        args.concurrency = max(1, min(args.concurrency, 2))
        args.tt_mb = min(args.tt_mb, 2)

    args.concurrency = max(1, int(args.concurrency))
    args.tt_mb = max(1, int(args.tt_mb))
    args.games = max(2, int(args.games) - (int(args.games) % 2))
    args.l2_games = max(2, int(args.l2_games) - (int(args.l2_games) % 2))
    args.l2_batch_games = max(2, int(args.l2_batch_games) - (int(args.l2_batch_games) % 2))

    if not args.out_dir:
        args.out_dir = f"tournament_analysis/spsa_eval_{block_name}"
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "spsa_log.jsonl"
    ckpt_path = out_dir / "checkpoint.json"
    best_path = out_dir / "best_params.json"
    patch_path = out_dir / "best_constants_diff.txt"

    from tune_eval_match.inprocess_match import (  # noqa: E402
        expand_worker_pool,
        load_openings,
        make_worker_pool,
        run_match,
        set_eval_params,
        verify_live_weights,
        warmup_jit,
    )

    rng = random.Random(args.seed)

    # --- State ---
    start_iter = 0
    theta = default_params()
    best_theta = dict(theta)
    best_l2_score = 0.5
    history_scores: list[float] = []
    window_thetas: list[dict] = []
    raw_updates: dict[str, list[float]] = {n: [] for n in names}
    l2_count = 0
    l2_fail_streak = 0
    l2_skip_no_int = 0
    last_l2_gate_iter = -1
    l1_opening_offset = 0
    l2_opening_offset = 0
    phase2_active = False

    if args.resume:
        ck = load_checkpoint(Path(args.resume))
        theta = clip_params_float(ck["theta"])
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
        l1_opening_offset = int(ck.get("l1_opening_offset", start_iter * max(1, args.games // 2)))
        l2_opening_offset = int(ck.get("l2_opening_offset", l2_count * max(1, args.l2_games // 2)))
        window_thetas = [clip_params_float(t) for t in ck.get("window_thetas", [])]
        if isinstance(ck.get("raw_updates"), dict):
            for n in names:
                raw_updates[n] = [float(x) for x in ck["raw_updates"].get(n, [])]
        if not _restore_rng(rng, ck, args.seed, start_iter):
            print(
                "[resume] WARNING: checkpoint has no rng_state (legacy); "
                f"reseeded perturbation RNG from seed+iter={start_iter}",
                flush=True,
            )
        phase2_active = bool(ck.get("phase2_active", False))
        print(
            f"[resume] iter={start_iter} l1_off={l1_opening_offset} "
            f"l2_off={l2_opening_offset} from {args.resume}",
            flush=True,
        )

    openings_path = Path(args.openings) if args.openings else None
    openings = list(load_openings(openings_path, limit=int(args.openings_limit)))
    random.Random(int(args.opening_seed)).shuffle(openings)
    if len(openings) >= 200 and 0.0 < args.l2_holdout_frac < 0.8:
        split = max(1, int(round(len(openings) * (1.0 - float(args.l2_holdout_frac)))))
        l1_openings = openings[:split]
        l2_openings = openings[split:]
    else:
        l1_openings = openings
        l2_openings = openings
    print(
        f"[openings] total={len(openings)} l1={len(l1_openings)} l2_holdout={len(l2_openings)} "
        f"adjudicate={args.adjudicate}",
        flush=True,
    )

    # --- One-time engine construction + single JIT ---
    print(
        f"[init] building worker pool concurrency={args.concurrency} tt={args.tt_mb}MB "
        f"(one process, one JIT)...",
        flush=True,
    )
    worker_pool = make_worker_pool(args.concurrency, tt_mb=args.tt_mb)
    warmup_jit(worker_pool[0][0], nodes=min(5000, max(1500, args.nodes)))
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
        "opening_seed": args.opening_seed,
        "adjudicate": bool(args.adjudicate),
        "params": names,
        "initial_theta": dict(theta),
        "bounds": {n: list(bounds(n)) for n in names},
        "r_end": {n: r_end(n) for n in names},
        "c": {n: step_c(n) for n in names},
        "theta_mode": "float_accumulate_quantize_on_apply",
        "l2_every": args.l2_every,
        "l2_games": args.l2_games,
        "l2_nodes": args.l2_nodes,
        "l2_candidate": args.l2_candidate,
        "l2_min_score": args.l2_min_score,
        "l2_sprt": l2_sprt,
        "pid": os.getpid(),
    }
    save_json(out_dir / "run_meta.json", meta)
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)
    print(f"[run] log={log_path}", flush=True)

    adjudicate_cfg = {} if args.adjudicate else None
    iter_i = start_iter
    max_iters = args.iters if args.iters > 0 else 10**9
    t_run0 = time.time()

    def _save_checkpoint(next_iter: int, stopped_reason: str | None = None) -> None:
        """``next_iter`` = first iteration to run on resume (completed iters count)."""
        payload = {
            "iter": int(next_iter),
            "rng_state": _rng_state_to_json(rng),
            "block": active_block(),
            "params": names,
            "theta": theta,
            "theta_int": quantize_params(theta),
            "best_theta": best_theta,
            "best_theta_int": quantize_params(best_theta),
            "best_l2_score": best_l2_score,
            "recent_scores": history_scores[-50:],
            "window_thetas": window_thetas[-50:],
            "raw_updates": {n: raw_updates[n][-200:] for n in names},
            "elapsed_run_s": time.time() - t_run0,
            "l2_count": l2_count,
            "l2_fail_streak": l2_fail_streak,
            "l2_skip_no_int": l2_skip_no_int,
            "last_l2_gate_iter": last_l2_gate_iter,
            "l1_opening_offset": l1_opening_offset,
            "l2_opening_offset": l2_opening_offset,
            "phase2_active": phase2_active,
            "games": args.games,
            "nodes": args.nodes,
            "l2_games": args.l2_games,
            "l2_nodes": args.l2_nodes,
            "concurrency": args.concurrency,
            "meta": meta,
        }
        if stopped_reason:
            payload["stopped_reason"] = stopped_reason
        save_json(ckpt_path, payload)

    while iter_i < max_iters:
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
        theta_p_i = quantize_params(theta_p)
        theta_m_i = quantize_params(theta_m)

        print(
            f"\n=== SPSA iter {iter_i}  games={args.games} nodes={args.nodes} "
            f"c={args.concurrency} l1_off={l1_opening_offset} ===",
            flush=True,
        )
        match = run_match(
            theta_p_i,
            theta_m_i,
            openings=l1_openings,
            nodes=args.nodes,
            games=args.games,
            log_every=args.log_every_games,
            worker_pool=worker_pool,
            concurrency=args.concurrency,
            opening_offset=l1_opening_offset,
            adjudicate=adjudicate_cfg,
        )
        l1_opening_offset += max(1, int(match["games"]) // 2)

        score_plus = float(match["score_a"])
        y_diff = 2.0 * score_plus - 1.0
        elo = elo_from_score(score_plus)

        new_theta = {}
        step_l1 = 0.0
        for n in names:
            g = y_diff / (2.0 * c_k[n] * delta[n])
            upd = a_k[n] * g
            raw_updates[n].append(upd)
            step_l1 += abs(upd)
            new_theta[n] = theta[n] + upd
        theta = clip_params_float(new_theta)
        theta_i = quantize_params(theta)
        history_scores.append(score_plus)
        window_thetas.append(dict(theta))

        z_map = {n: round(drift_z(raw_updates[n]), 2) for n in names}
        top_z = sorted(z_map.items(), key=lambda kv: abs(kv[1]), reverse=True)[:4]
        top_z_str = " ".join(f"{k}:{v:+.2f}" for k, v in top_z)

        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "iter": iter_i,
            "score_plus": score_plus,
            "elo_plus_vs_minus": elo,
            "pts_plus": match["pts_a"],
            "pts_minus": match["pts_b"],
            "penta_plus": match.get("penta"),
            "games": match["games"],
            "nodes": args.nodes,
            "opening_offset": match.get("opening_offset"),
            "match_s": match["elapsed_s"],
            "theta": dict(theta),
            "theta_int": dict(theta_i),
            "theta_plus_int": theta_p_i,
            "theta_minus_int": theta_m_i,
            "delta": delta,
            "y_diff": y_diff,
            "step_l1_float": step_l1,
            "drift_z": z_map,
        }
        with open(log_path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(
            f"[iter {iter_i}] score+={score_plus:.3f} ({elo:+.1f} Elo)  "
            f"games={match['games']} {match['elapsed_s']:.1f}s  "
            f"stepL1={step_l1:.3f}  topZ=[{top_z_str}]",
            flush=True,
        )

        # --- L2 Validation Gate ---
        l2_scheduled = args.l2_every > 0 and (iter_i + 1) % args.l2_every == 0
        if l2_scheduled:
            last_l2_gate_iter = iter_i
            best_i_gate = quantize_params(best_theta)
            tail_theta = _average_tail_thetas(window_thetas, names) if window_thetas else dict(theta)
            tail_i_gate = quantize_params(tail_theta)
            latest_i_gate = quantize_params(theta)
            if args.l2_candidate == "tail":
                if tail_i_gate != best_i_gate:
                    cand_source = "tail"
                    cand_i_gate = tail_i_gate
                else:
                    cand_source = "tail==best->latest"
                    cand_i_gate = latest_i_gate
            else:
                cand_source = "latest"
                cand_i_gate = latest_i_gate

            if cand_i_gate == best_i_gate:
                l2_skip_no_int += 1
                next_gate = iter_i + args.l2_every
                print(
                    f"\n--- L2 SKIP @ iter {iter_i}: cand_int == best_int (no integer change, skips={l2_skip_no_int}) ---",
                    flush=True,
                )
                with open(log_path, "a", encoding="utf-8", newline="\n") as f:
                    f.write(
                        json.dumps(
                            {
                                "ts": datetime.now(timezone.utc).isoformat(),
                                "type": "l2_skip_no_int_change",
                                "iter": iter_i,
                                "next_l2_gate_iter": next_gate,
                                "l2_skip_no_int": l2_skip_no_int,
                                "theta_int": cand_i_gate,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            else:
                print(
                    f"\n--- L2 validate @ iter {iter_i} ({cand_source}): "
                    f"<= {args.l2_games}g @ n{args.l2_nodes} vs best (l2_off={l2_opening_offset}) ---",
                    flush=True,
                )
                print(f"[L2] diffs vs best:\n{format_constants_diff(cand_i_gate)}", flush=True)
                penta_parts: list[list[int]] = []
                pts_cand = 0.0
                pts_best = 0.0
                games_done = 0
                l2_trash = False
                l2_decision = "fixed_cap"
                l2_llr = 0.0
                t_l2_0 = time.time()

                while games_done < args.l2_games:
                    this_batch = min(args.l2_batch_games, args.l2_games - games_done)
                    if this_batch % 2:
                        this_batch += 1
                    res = run_match(
                        cand_i_gate,
                        best_i_gate,
                        openings=l2_openings,
                        nodes=args.l2_nodes,
                        games=this_batch,
                        log_every=max(20, args.log_every_games),
                        worker_pool=worker_pool,
                        concurrency=args.concurrency,
                        opening_offset=l2_opening_offset,
                        adjudicate=adjudicate_cfg,
                    )
                    l2_opening_offset += max(1, int(res["games"]) // 2)
                    penta_parts.append(list(res["penta"]))
                    games_done += int(res["games"])
                    pts_cand += float(res["pts_a"])
                    pts_best += float(res["pts_b"])

                    penta_cand = merge_penta(penta_parts)
                    est = elo_estimate(penta_cand)
                    batch_msg = (
                        f"  [L2 batch] games={games_done}/{args.l2_games} "
                        f"score={est['score']:.3f} elo={est['elo']:+.1f}±{est['elo_err']:.1f} "
                        f"penta={penta_cand}"
                    )
                    if l2_sprt is not None:
                        l2_llr = sprt_llr(penta_cand, l2_sprt[0], l2_sprt[1])
                        lo_b, hi_b = sprt_bounds(args.sprt_alpha, args.sprt_beta)
                        l2_decision = sprt_decision(l2_llr, args.sprt_alpha, args.sprt_beta)
                        batch_msg += f" LLR={l2_llr:+.2f} [{lo_b:.2f},{hi_b:.2f}] -> {l2_decision}"
                    print(batch_msg, flush=True)

                    if (
                        args.l2_early_trash_games > 0
                        and games_done >= args.l2_early_trash_games
                        and est["score"] < args.l2_early_trash_score
                    ):
                        l2_trash = True
                        l2_decision = "early_trash"
                        break
                    if l2_sprt is not None and l2_decision != "continue":
                        break

                l2_elapsed = time.time() - t_l2_0
                penta_cand = merge_penta(penta_parts)
                est = elo_estimate(penta_cand)
                l2_score = float(est["score"])
                l2_elo = float(est["elo"])

                if l2_trash:
                    l2_accepted = False
                elif l2_sprt is not None:
                    if l2_decision == "accept_h1":
                        l2_accepted = True
                    elif l2_decision == "accept_h0":
                        l2_accepted = False
                    else:
                        l2_accepted = l2_score >= float(args.l2_min_score)
                        l2_decision = f"cap_score_{'pass' if l2_accepted else 'fail'}"
                else:
                    l2_accepted = l2_score >= float(args.l2_min_score)
                    l2_decision = f"score_{'pass' if l2_accepted else 'fail'}"

                if l2_accepted:
                    best_theta = best_from_l2_ints(cand_i_gate)
                    best_l2_score = l2_score
                    l2_fail_streak = 0
                    window_thetas.clear()
                    diff_txt = format_constants_diff(cand_i_gate)
                    save_json(
                        best_path,
                        {
                            "block": active_block(),
                            "theta": best_theta,
                            "theta_int": dict(cand_i_gate),
                            "cand_source": cand_source,
                            "l2_score": best_l2_score,
                            "l2_elo": l2_elo,
                            "l2_elo_err95": est["elo_err"],
                            "l2_penta": penta_cand,
                            "l2_decision": l2_decision,
                            "iter": iter_i,
                            "ts": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    save_text(patch_path, diff_txt + "\n")
                    print(
                        f"[L2] NEW BEST ACCEPTED ({l2_decision}): score={l2_score:.3f} "
                        f"({l2_elo:+.1f}±{est['elo_err']:.1f} Elo) in {l2_elapsed:.1f}s",
                        flush=True,
                    )
                else:
                    l2_fail_streak += 1
                    print(
                        f"[L2] REJECT ({l2_decision}): score={l2_score:.3f} "
                        f"({l2_elo:+.1f}±{est['elo_err']:.1f} Elo)  fail_streak={l2_fail_streak}",
                        flush=True,
                    )

                l2_count += 1
                l2_row = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "type": "l2_accept",
                    "iter": iter_i,
                    "cand_source": cand_source,
                    "accepted": l2_accepted,
                    "decision": l2_decision,
                    "score_theta_vs_best": l2_score,
                    "elo": l2_elo,
                    "elo_err95": est["elo_err"],
                    "penta": penta_cand,
                    "llr": l2_llr if l2_sprt is not None else None,
                    "games": games_done,
                    "nodes": args.l2_nodes,
                    "elapsed_s": l2_elapsed,
                    "early_trash": l2_trash,
                    "l2_fail_streak": l2_fail_streak,
                    "theta_int": dict(cand_i_gate),
                    "best_theta_int": dict(best_i_gate),
                }
                with open(log_path, "a", encoding="utf-8", newline="\n") as f:
                    f.write(json.dumps(l2_row, ensure_ascii=False) + "\n")

                if args.l2_stop_after_fails > 0 and l2_fail_streak >= args.l2_stop_after_fails:
                    print(
                        f"[stop] {l2_fail_streak} consecutive L2 rejects "
                        f"(limit={args.l2_stop_after_fails}). Stopping SPSA.",
                        flush=True,
                    )
                    _save_checkpoint(iter_i + 1, stopped_reason="consecutive_l2_rejects")
                    save_json(
                        out_dir / "final_params.json",
                        {
                            "block": active_block(),
                            "theta": theta,
                            "best_theta": best_theta,
                            "best_theta_int": quantize_params(best_theta),
                            "stopped_reason": "consecutive_l2_rejects",
                            "l2_fail_streak": l2_fail_streak,
                            "iter": iter_i,
                        },
                    )
                    return 0

                if args.phase2_after_l2 and not phase2_active and l2_count >= 1:
                    phase2_active = True
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

        if (iter_i + 1) % args.ckpt_every == 0 or iter_i == 0 or l2_scheduled:
            # l2_scheduled: persist best_theta / fail streak immediately so a
            # disconnect cannot lose an accepted L2 result or replay the gate.
            _save_checkpoint(iter_i + 1)
            save_json(
                out_dir / "latest_params.json",
                {
                    "block": active_block(),
                    "theta": theta,
                    "theta_int": quantize_params(theta),
                    "best_theta_int": quantize_params(best_theta),
                    "iter": iter_i + 1,
                    "drift_z": z_map,
                },
            )

        iter_i += 1

    print(f"\n[done] {iter_i} iterations, elapsed {time.time() - t_run0:.1f}s", flush=True)
    _save_checkpoint(iter_i, stopped_reason="max_iters")
    save_json(
        out_dir / "final_params.json",
        {
            "block": active_block(),
            "theta": theta,
            "best_theta": best_theta,
            "best_theta_int": quantize_params(best_theta),
            "stopped_reason": "max_iters",
            "iter": iter_i,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
