#!/usr/bin/env python3
"""Fixed-node A/B match for a runtime search-parameter candidate.

The baseline remains the registry defaults; candidate values are applied only
to the in-process ``SearchContext`` objects.  This is intentionally separate
from SPSA so an L2 promotion test compares one fixed candidate against one
fixed baseline instead of another random perturbation.
"""

from __future__ import annotations

import argparse
import json
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nodes", type=int, default=200000)
    ap.add_argument("--games", type=int, default=100, help="even number of games")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--tt-mb", type=int, default=16)
    ap.add_argument("--openings", type=Path, default=Path("data/hist_diag_openings_500.epd"))
    ap.add_argument("--params", default=",".join(DEFAULT_NAMES))
    ap.add_argument("--param", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.games < 2 or args.games % 2:
        ap.error("--games must be an even integer >= 2")
    try:
        names = resolve_names(names=[args.params])
        overrides = _parse_overrides(args.param, names)
    except ValueError as exc:
        ap.error(str(exc))
    if not overrides:
        ap.error("at least one --param candidate override is required")

    base = get_default_params(names=names)
    candidate = dict(base)
    candidate.update(overrides)
    openings = load_openings(args.openings, limit=max(500, args.games * 2))
    print(
        f"search_param_match nodes={args.nodes} games={args.games} "
        f"concurrency={args.concurrency} openings={len(openings)}",
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
    result = run_match(
        base,
        candidate,
        openings=openings,
        nodes=int(args.nodes),
        games=int(args.games),
        log_every=10,
        worker_pool=pool,
        concurrency=int(args.concurrency),
        early_trash_min_games=0,
        scale_mode="integer",
    )
    result = dict(result)
    result.update({
        "nodes": int(args.nodes),
        "games": int(args.games),
        "concurrency": int(args.concurrency),
        "params": list(names),
        "baseline": base,
        "candidate": candidate,
        "candidate_score": 1.0 - float(result["score_a"]),
        "wall_s": time.time() - t0,
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
