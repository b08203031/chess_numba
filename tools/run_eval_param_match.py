#!/usr/bin/env python3
"""Fixed-node A/B SPRT verification match for classical HCE eval_weights candidates.

Intended as Stage 3 (Final Confirmation, e.g. @ 400k nodes) after Match-SPSA
L1/L2 produces a candidate in ``best_params.json``:

  python tools/run_eval_param_match.py \
    --params-json tournament_analysis/spsa_eval_threats_mg/best_params.json \
    --nodes 400000 --games 2000 --sprt 0,10 --batch-games 50 \
    --out tournament_analysis/verify_threats_mg_400k.json

Single process, single Numba JIT compile via ``tune_eval_match.inprocess_match``.
Reports pentanomial Elo ± 95% CI plus primary ``--sprt`` and simultaneous
``[0, 10]``, ``[-3, 3]``, ``[-5, 0]`` LLRs.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tune_eval_match.eval_weights import (  # noqa: E402
    EVAL_PARAM_INDEX,
    default_eval_weights,
    format_constants_diff,
    params_from_weights,
)
from tune_search.stats import (  # noqa: E402
    elo_estimate,
    merge_penta,
    sprt_bounds,
    sprt_decision,
    sprt_llr,
)


def _default_concurrency() -> int:
    cpus = os.cpu_count() or 4
    return max(1, min(8, cpus - 2))


def _load_params_json(path: Path) -> dict[str, int]:
    """Accept best_params.json, final_params.json, latest_params.json, or flat dict."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        keys = (
            ("theta_int", "theta", "best_theta_int", "best_theta")
            if path.name.startswith("latest")
            else ("best_theta_int", "theta_int", "theta", "best_theta")
        )
        for key in keys:
            if isinstance(data.get(key), dict):
                data = data[key]
                break
    out: dict[str, int] = {}
    for k, v in data.items():
        name = str(k)
        if name not in EVAL_PARAM_INDEX:
            raise ValueError(f"Unknown eval parameter {name!r} in {path}")
        out[name] = int(round(float(v)))
    return out


def _parse_overrides(raw_values: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for raw in raw_values:
        if "=" not in raw:
            raise ValueError(f"--param must use NAME=VALUE, got {raw!r}")
        name, val = raw.split("=", 1)
        name = name.strip()
        if name not in EVAL_PARAM_INDEX:
            raise ValueError(f"Unknown eval parameter {name!r}")
        out[name] = int(round(float(val.strip())))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nodes", type=int, default=400000)
    ap.add_argument("--games", type=int, default=2000, help="Even game cap (or exact count if --sprt off)")
    ap.add_argument("-c", "--concurrency", type=int, default=_default_concurrency())
    ap.add_argument("--tt-mb", type=int, default=16)
    ap.add_argument("--openings", type=Path, default=Path("data/openings.epd"))
    ap.add_argument("--openings-limit", type=int, default=0, help="0 = load all openings")
    ap.add_argument("--opening-seed", type=int, default=20261120)
    ap.add_argument("--params-json", type=Path, default=None, help="Candidate JSON (e.g. best_params.json)")
    ap.add_argument("--param", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--baseline-json", type=Path, default=None, help="Optional baseline JSON")
    ap.add_argument("--baseline-old", action="store_true",
                    help="Compare candidate against chess_engine.classical_old.constants")
    ap.add_argument("--sprt", default="0,10", metavar="ELO0,ELO1",
                    help="Primary SPRT window ELO0,ELO1 (set 'off' to disable early stopping)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--batch-games", type=int, default=50, help="Even games per SPRT batch")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--adjudicate", action="store_true", default=True)
    ap.add_argument("--no-adjudicate", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("tournament_analysis/eval_verify_400k.json"))
    args = ap.parse_args(argv)

    if args.no_adjudicate:
        args.adjudicate = False
    if args.games < 2 or args.games % 2:
        ap.error("--games must be an even integer >= 2")
    if args.batch_games < 2 or args.batch_games % 2:
        ap.error("--batch-games must be an even integer >= 2")

    sprt = None
    if args.sprt and args.sprt.strip().lower() != "off":
        try:
            elo0, elo1 = (float(x) for x in args.sprt.split(","))
        except ValueError:
            ap.error("--sprt must look like ELO0,ELO1 or 'off'")
        if elo1 <= elo0:
            ap.error("--sprt requires ELO1 > ELO0")
        sprt = (elo0, elo1)

    overrides: dict[str, int] = {}
    if args.params_json:
        overrides.update(_load_params_json(args.params_json))
    if args.param:
        overrides.update(_parse_overrides(args.param))
    if not overrides:
        ap.error("Provide --params-json and/or --param NAME=VALUE")

    const_mod = None
    if args.baseline_old:
        from chess_engine.classical_old import constants as old_constants
        const_mod = old_constants

    base_full = params_from_weights(default_eval_weights(const_mod))
    if args.baseline_json:
        base_full.update(_load_params_json(args.baseline_json))
    cand_full = dict(base_full)
    cand_full.update(overrides)

    diffs = {k: (base_full[k], cand_full[k]) for k in cand_full if cand_full[k] != base_full[k]}
    if not diffs:
        print("WARNING: candidate matches baseline on every parameter (0 diffs).", flush=True)

    from tune_eval_match.inprocess_match import (  # noqa: E402
        load_openings,
        make_worker_pool,
        run_match,
        warmup_jit,
    )

    openings_path = args.openings if args.openings.is_absolute() else (ROOT / args.openings)
    openings = list(load_openings(openings_path, limit=int(args.openings_limit)))
    random.Random(int(args.opening_seed)).shuffle(openings)

    print(
        f"[eval_verify] nodes={args.nodes} games<={args.games} c={args.concurrency} "
        f"tt={args.tt_mb}MB openings={len(openings)} sprt={sprt} adjudicate={args.adjudicate}",
        flush=True,
    )
    print(f"[eval_verify] changed params ({len(diffs)}):\n{format_constants_diff(cand_full, const_mod)}", flush=True)

    pool = make_worker_pool(args.concurrency, tt_mb=args.tt_mb)
    warmup_nodes = min(5000, max(1500, int(args.nodes)))
    for i, (ctx_a, ctx_b) in enumerate(pool):
        warmup_jit(ctx_a, nodes=warmup_nodes if i == 0 else min(800, warmup_nodes))
        warmup_jit(ctx_b, nodes=min(1500, warmup_nodes) if i == 0 else min(800, warmup_nodes))

    t0 = time.time()
    penta_cand_parts: list[list[int]] = []
    pairs_done = 0
    games_done = 0
    pts_a = 0.0
    pts_b = 0.0
    decision = "fixed_budget"
    llr = 0.0
    lo_b, hi_b = sprt_bounds(args.alpha, args.beta)
    batch_games = args.batch_games if sprt else args.games
    adjudicate_cfg = {} if args.adjudicate else None

    while games_done < args.games:
        this_batch = min(batch_games, args.games - games_done)
        if this_batch % 2:
            this_batch += 1
        result = run_match(
            base_full,
            cand_full,
            openings=openings,
            nodes=int(args.nodes),
            games=int(this_batch),
            log_every=max(2, int(args.log_every)),
            worker_pool=pool,
            concurrency=int(args.concurrency),
            opening_offset=pairs_done,
            adjudicate=adjudicate_cfg,
        )
        penta_cand_parts.append(list(result["penta_b"]))
        pairs_done += sum(result["penta_b"])
        games_done += int(result["games"])
        pts_a += float(result["pts_a"])
        pts_b += float(result["pts_b"])

        penta_cand = merge_penta(penta_cand_parts)
        est = elo_estimate(penta_cand)
        llr_0_10 = sprt_llr(penta_cand, 0.0, 10.0)
        llr_m3_3 = sprt_llr(penta_cand, -3.0, 3.0)
        llr_m5_0 = sprt_llr(penta_cand, -5.0, 0.0)
        line = (
            f"[verify] g={games_done} score={est['score']:.4f} "
            f"elo={est['elo']:+.1f}±{est['elo_err']:.1f} "
            f"LLR[0,10]={llr_0_10:+.2f} [-3,3]={llr_m3_3:+.2f} [-5,0]={llr_m5_0:+.2f}"
        )
        if sprt is not None:
            llr = sprt_llr(penta_cand, sprt[0], sprt[1])
            decision = sprt_decision(llr, args.alpha, args.beta)
            line += f" | SPRT{sprt}={llr:+.2f} [{lo_b:.2f},{hi_b:.2f}] -> {decision}"
        print(line, flush=True)
        if sprt is not None and decision != "continue":
            break

    if sprt is not None and decision == "continue":
        decision = "inconclusive_at_cap"

    penta_cand = merge_penta(penta_cand_parts)
    penta_a = list(reversed(penta_cand))
    est = elo_estimate(penta_cand)
    out = {
        "nodes": int(args.nodes),
        "games": int(games_done),
        "pairs": int(pairs_done),
        "concurrency": int(args.concurrency),
        "elapsed_s": time.time() - t0,
        "changed_params": diffs,
        "candidate_score": est["score"],
        "candidate_elo": est["elo"],
        "candidate_elo_err95": est["elo_err"],
        "penta_baseline": penta_a,
        "penta_candidate": penta_cand,
        "llr_0_10": sprt_llr(penta_cand, 0.0, 10.0),
        "llr_m3_3": sprt_llr(penta_cand, -3.0, 3.0),
        "llr_m5_0": sprt_llr(penta_cand, -5.0, 0.0),
        "sprt": None if sprt is None else {
            "elo0": sprt[0], "elo1": sprt[1],
            "alpha": args.alpha, "beta": args.beta,
            "llr": llr, "decision": decision,
        },
        "candidate": cand_full,
        "baseline": base_full,
    }
    out_path = args.out if args.out.is_absolute() else (ROOT / args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[eval_verify] saved {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
