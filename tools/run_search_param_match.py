#!/usr/bin/env python3
"""Fixed-node A/B match for a runtime search-parameter candidate.

The baseline remains the registry defaults; candidate values are applied only
to the in-process ``SearchContext`` objects.  This is intentionally separate
from SPSA so an L2 promotion test compares one fixed candidate against one
fixed baseline instead of another random perturbation.

Statistics: the report is pentanomial (opening pairs are the independent
samples) with Elo +/- 95% CI.  Optionally run as an SPRT (``--sprt``) in
batches so a clearly good/bad candidate stops early instead of burning a fixed
game budget; ``--games`` is then the hard cap.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

# Running ``python tools/...`` places ``tools`` (rather than the repository
# root) on sys.path; make the project packages explicit for both CLI and UCI
# launches.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tune_search.inprocess_match import (
    load_openings,
    make_worker_pool,
    run_match,
    warmup_jit,
)
from tune_search.search_param_registry import get_default_params, get_spec, resolve_names
from tune_search.stats import (
    elo_estimate,
    merge_penta,
    sprt_bounds,
    sprt_decision,
    sprt_llr,
)


DEFAULT_NAMES = (
    "LMR_BASE_OFFSET",
    "LMR_HISTORY_SCALE",
    "LMR_CUTNODE_BONUS",
    "LMR_TTCAPTURE_BONUS",
    "LMR_MOVECOUNT_FACTOR",
    "LMR_TABLE_SCALE_PERCENT",
)


def _parse_overrides(raw_values: list[str], names: tuple[str, ...]) -> dict[str, int]:
    out: dict[str, int] = {}
    allowed = set(names)
    for raw in raw_values:
        if "=" not in raw:
            raise ValueError(f"--param must use NAME=VALUE, got {raw!r}")
        name, value = raw.split("=", 1)
        name = name.strip()
        if name not in allowed:
            raise ValueError(f"candidate parameter {name!r} is not in --params")
        if not get_spec(name).is_runtime:
            raise ValueError(f"candidate parameter {name!r} is compile-time")
        if name in out:
            raise ValueError(f"duplicate candidate parameter: {name}")
        try:
            out[name] = int(value.strip())
        except ValueError as exc:
            raise ValueError(f"candidate value must be an integer: {raw!r}") from exc
    return out


def _load_params_json(path: Path | str) -> dict[str, int]:
    """Accept ``summarize_run.py --out`` JSON or a flat ``{name: value}`` map."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "averaged_params" in data:
        data = data["averaged_params"]
    return {str(k): int(v) for k, v in data.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nodes", type=int, default=200000)
    ap.add_argument(
        "--games", type=int, default=100,
        help="even number of games (hard cap when --sprt is used)",
    )
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--tt-mb", type=int, default=16)
    ap.add_argument("--openings", type=Path, default=Path("data/hist_diag_openings_500.epd"))
    ap.add_argument("--params", default=",".join(DEFAULT_NAMES))
    ap.add_argument("--param", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument(
        "--params-json", type=Path, default=None,
        help="candidate from summarize_run.py --out (averaged_params) or a flat map; "
             "its keys define --params",
    )
    ap.add_argument(
        "--baseline-json", type=Path, default=None,
        help="baseline parameters from JSON (defaults to engine registry defaults)",
    )
    ap.add_argument(
        "--sprt", default=None, metavar="ELO0,ELO1",
        help="run in batches and stop on SPRT decision, e.g. 0,3 (H0: <=0 Elo, H1: >=3 Elo)",
    )
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--batch-games", type=int, default=100, help="games per SPRT batch")
    ap.add_argument("--opening-seed", type=int, default=777, help="opening shuffle seed")
    ap.add_argument(
        "--adjudicate", action="store_true",
        help="score-based win/draw adjudication (use the same setting as the tuning run)",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.games < 2 or args.games % 2:
        ap.error("--games must be an even integer >= 2")
    if args.batch_games < 2 or args.batch_games % 2:
        ap.error("--batch-games must be an even integer >= 2")
    sprt = None
    if args.sprt:
        try:
            elo0, elo1 = (float(x) for x in args.sprt.split(","))
        except ValueError:
            ap.error("--sprt must look like ELO0,ELO1")
        if elo1 <= elo0:
            ap.error("--sprt requires ELO1 > ELO0")
        sprt = (elo0, elo1)

    try:
        json_params = _load_params_json(args.params_json) if args.params_json else {}
        names = resolve_names(
            names=[",".join(json_params)] if json_params else [args.params]
        )
        overrides = dict(json_params)
        overrides.update(_parse_overrides(args.param, names))
        for name in overrides:
            if not get_spec(name).is_runtime:
                raise ValueError(f"candidate parameter {name!r} is compile-time")
    except ValueError as exc:
        ap.error(str(exc))
    if not overrides:
        ap.error("at least one --param candidate override (or --params-json) is required")

    base = get_default_params(names=names)
    if args.baseline_json:
        base.update(_load_params_json(args.baseline_json))
    candidate = dict(base)
    candidate.update(overrides)
    openings = list(load_openings(args.openings, limit=max(500, args.games * 2)))
    random.Random(int(args.opening_seed)).shuffle(openings)
    print(
        f"search_param_match nodes={args.nodes} games<={args.games} "
        f"concurrency={args.concurrency} openings={len(openings)} "
        f"sprt={'off' if sprt is None else sprt}",
        flush=True,
    )
    print(f"baseline={base}", flush=True)
    print(f"candidate={candidate}", flush=True)

    pool = make_worker_pool(args.concurrency, tt_mb=args.tt_mb)
    warmup_nodes = min(5000, max(1000, int(args.nodes)))
    for pair_index, (ctx_a, ctx_b) in enumerate(pool):
        warmup_jit(ctx_a, nodes=warmup_nodes if pair_index == 0 else min(800, warmup_nodes))
        warmup_jit(ctx_b, nodes=min(1500, warmup_nodes) if pair_index == 0 else min(800, warmup_nodes))

    t0 = time.time()
    # A = baseline, B = candidate.  penta_a counts pair points for the baseline;
    # the candidate's histogram is its mirror image.
    penta_a_parts: list[list[int]] = []
    pairs_done = 0
    games_done = 0
    pts_a = 0.0
    pts_b = 0.0
    decision = "fixed_budget"
    llr = 0.0
    batch_games = args.batch_games if sprt else args.games
    while games_done < args.games:
        this_batch = min(batch_games, args.games - games_done)
        result = run_match(
            base,
            candidate,
            openings=openings,
            nodes=int(args.nodes),
            games=int(this_batch),
            log_every=10,
            worker_pool=pool,
            concurrency=int(args.concurrency),
            early_trash_min_games=0,
            scale_mode="integer",
            opening_offset=pairs_done,
            adjudicate={} if args.adjudicate else None,
        )
        penta_a_parts.append(list(result["penta"]))
        pairs_done += sum(result["penta"])
        games_done += int(result["games"])
        pts_a += float(result["pts_a"])
        pts_b += float(result["pts_b"])
        penta_cand = merge_penta([p[::-1] for p in penta_a_parts])
        est = elo_estimate(penta_cand)
        line = (
            f"[verify] games={games_done} candidate_score={est['score']:.4f} "
            f"elo={est['elo']:+.1f} +/- {est['elo_err']:.1f}"
        )
        if sprt:
            llr = sprt_llr(penta_cand, sprt[0], sprt[1])
            lower, upper = sprt_bounds(args.alpha, args.beta)
            decision = sprt_decision(llr, args.alpha, args.beta)
            line += f" LLR={llr:+.2f} [{lower:.2f},{upper:.2f}] -> {decision}"
        print(line, flush=True)
        if sprt and decision != "continue":
            break
    if sprt and decision == "continue":
        decision = "inconclusive_at_cap"

    penta_a = merge_penta(penta_a_parts)
    penta_cand = merge_penta([p[::-1] for p in penta_a_parts])
    est = elo_estimate(penta_cand)
    total_pts = pts_a + pts_b
    out = {
        "nodes": int(args.nodes),
        "games": int(games_done),
        "concurrency": int(args.concurrency),
        "params": list(names),
        "baseline": base,
        "candidate": candidate,
        "score_a": pts_a / total_pts if total_pts else 0.5,
        "pts_a": pts_a,
        "pts_b": pts_b,
        "candidate_score": est["score"],
        "candidate_elo": est["elo"],
        "candidate_elo_err95": est["elo_err"],
        "penta_baseline": penta_a,
        "penta_candidate": penta_cand,
        "sprt": None if sprt is None else {
            "elo0": sprt[0], "elo1": sprt[1],
            "alpha": args.alpha, "beta": args.beta, "llr": llr, "decision": decision,
        },
        "opening_seed": int(args.opening_seed),
        "wall_s": time.time() - t0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"candidate elo={est['elo']:+.1f} +/- {est['elo_err']:.1f} over {games_done} games"
        f"{'' if sprt is None else f' ({decision})'}",
        flush=True,
    )
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
