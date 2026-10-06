#!/usr/bin/env python3
"""Summarise a search-SPSA JSONL log: is there signal, and what to promote?

SPSA's last iterate is a noisy random walk; the statistically sound candidate
is the *average* of the late iterates (Polyak-Ruppert averaging).  This tool

* checks whether the per-batch match scores are wider than pure sampling noise
  (if not, the run learned nothing and the parameters merely random-walked),
* reports per-parameter net drift as a z-score versus a zero-mean walk, and
* writes the averaged parameter set for ``tools/run_search_param_match.py``.

Pure Python: no engine import, no JIT.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tune_search.search_param_registry import get_spec  # noqa: E402
from tune_search.stats import drift_z, merge_penta, penta_moments  # noqa: E402


def load_iterations(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return [r for r in rows if r.get("event") == "iteration"]


def _theta_of(row, names):
    theta = row.get("theta")
    if theta:
        return {n: float(theta[n]) for n in names}
    return {n: float(row["updated_params"][n]) for n in names}


def summarize(iterations, tail_fraction: float = 0.5, weight_mode: str = "uniform"):
    if not iterations:
        raise ValueError("log contains no iteration records")
    names = list(iterations[0]["params"])
    scores = [float(r["score_plus"]) for r in iterations]
    pairs_per_batch = int(iterations[0].get("match_pairs", max(1, int(iterations[0]["games"]) // 2)))

    penta_parts = [r["match"]["penta"] for r in iterations if r.get("match", {}).get("penta")]
    if penta_parts:
        total_penta = merge_penta(penta_parts)
        _, _, var_pair = penta_moments(total_penta)
    else:
        total_penta = None
        var_pair = 0.08  # conservative fallback for drawish fixed-node games
    expected_sd = math.sqrt(var_pair / pairs_per_batch)
    observed_sd = statistics.pstdev(scores) if len(scores) > 1 else float("nan")

    tail_n = max(1, int(round(len(iterations) * float(tail_fraction))))
    tail = iterations[-tail_n:]
    averaged = {}
    report = []
    for name in names:
        spec = get_spec(name)
        tail_values = [_theta_of(r, names)[name] for r in tail]
        if weight_mode == "linear":
            weights = list(range(1, len(tail_values) + 1))
            avg = sum(v * w for v, w in zip(tail_values, weights)) / sum(weights)
        else:
            avg = sum(tail_values) / len(tail_values)
        averaged[name] = int(round(min(spec.maximum, max(spec.minimum, avg))))
        updates = [float(r["raw_updates"][name]) for r in iterations if "raw_updates" in r]
        first = _theta_of(iterations[0], names)[name] - float(iterations[0]["raw_updates"][name]) \
            if "raw_updates" in iterations[0] else float(spec.default)
        last = _theta_of(iterations[-1], names)[name]
        span = float(spec.maximum - spec.minimum) or 1.0
        report.append(
            {
                "name": name,
                "default": spec.default,
                "start": first,
                "last": last,
                "tail_avg": avg,
                "moved_pct_of_range": 100.0 * (last - first) / span,
                "drift_z": drift_z(updates) if updates else 0.0,
                "c_end": spec.step,
            }
        )
    games = sum(int(r["match"].get("games", r.get("games", 0))) for r in iterations)
    return {
        "iterations": len(iterations),
        "games": games,
        "mean_score": sum(scores) / len(scores),
        "observed_score_sd": observed_sd,
        "expected_noise_sd": expected_sd,
        "sd_ratio": observed_sd / expected_sd if expected_sd > 0 else float("nan"),
        "tail_iterations": tail_n,
        "weight_mode": weight_mode,
        "averaged_params": averaged,
        "params": report,
        "penta_total": total_penta,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", type=Path, help="JSONL written by search_tuner --log")
    ap.add_argument("--tail", type=float, default=0.5, help="fraction of final iterations to average")
    ap.add_argument(
        "--weight-mode",
        choices=["uniform", "linear"],
        default="uniform",
        help="weighting method for tail averaging: uniform (default) or linear",
    )
    ap.add_argument("--out", type=Path, default=None, help="write averaged params JSON here")
    args = ap.parse_args()

    result = summarize(load_iterations(args.log), tail_fraction=args.tail, weight_mode=args.weight_mode)
    print(
        f"iterations={result['iterations']} games={result['games']} "
        f"mean_score={result['mean_score']:.4f}"
    )
    print(
        f"score sd observed={result['observed_score_sd']:.4f} "
        f"noise-only expected={result['expected_noise_sd']:.4f} "
        f"ratio={result['sd_ratio']:.2f}"
    )
    if result["sd_ratio"] < 1.3:
        print(
            "  -> ratio ~1: batch scores are indistinguishable from sampling noise; "
            "this run carries (almost) no gradient signal. Use more games per batch / "
            "more batches before trusting any parameter movement."
        )
    print(f"\n{'param':32s} {'default':>8s} {'last':>9s} {'tail_avg':>9s} {'moved%rng':>9s} {'drift_z':>8s}")
    for row in result["params"]:
        flag = "" if abs(row["drift_z"]) >= 2.0 else "  (noise)"
        print(
            f"{row['name']:32s} {row['default']:8d} {row['last']:9.1f} {row['tail_avg']:9.1f} "
            f"{row['moved_pct_of_range']:9.2f} {row['drift_z']:8.2f}{flag}"
        )
    print(
        f"\naveraged candidate (last "
        f"{result['tail_iterations']} iterations, weight_mode={result['weight_mode']}):"
    )
    print(" ".join(f"--param {k}={v}" for k, v in result["averaged_params"].items()))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
