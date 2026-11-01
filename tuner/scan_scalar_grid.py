"""Exhaustive fixed-validation scan for a small grid of scalar HCE knobs."""
from __future__ import annotations

import argparse
import itertools

import numpy as np

from tuner.parameters import ParameterManager
from tuner.tuner import compute_squared_errors, split_dataset_indices
from tuner.validate_tuned import _bucket_masks


def _parse_grid(spec: str) -> tuple[str, list[int]]:
    if "=" not in spec:
        raise ValueError(f"Expected NAME=VALUES, got {spec!r}")
    name, raw = spec.split("=", 1)
    if ":" in raw:
        parts = [int(v) for v in raw.split(":")]
        if len(parts) == 2:
            lo, hi = parts
            step = 1 if hi >= lo else -1
        elif len(parts) == 3:
            lo, hi, step = parts
            if step == 0 or (hi - lo) * step < 0:
                raise ValueError(f"Invalid grid step in {spec!r}")
        else:
            raise ValueError(f"Expected LO:HI or LO:HI:STEP, got {raw!r}")
        values = list(range(lo, hi + (1 if step > 0 else -1), step))
    else:
        values = [int(v) for v in raw.split(",")]
    if not values:
        raise ValueError(f"Empty grid for {name}")
    return name, values


def run_scalar_grid_scan(
    dataset,
    grid_specs: list[str],
    *,
    pm: ParameterManager | None = None,
    baseline: np.ndarray | None = None,
    split: str = "dev",
    eval_sample: int = 1_000_000,
    holdout_sample: int = 2_000_000,
    exclude_fixed_val: int = 0,
    k_factor: float = 1.0,
    seed: int = 2,
    top: int = 0,
    verbose: bool = True,
) -> dict:
    grids = [_parse_grid(spec) for spec in grid_specs]
    if pm is None:
        pm = ParameterManager()
    if baseline is None:
        baseline = pm.get_initial_theta()

    scalar_indices: list[int] = []
    for name, _ in grids:
        start, count = pm.get_param_indices(name)
        if count != 1:
            raise ValueError(f"Grid scan supports scalars only: {name}")
        scalar_indices.append(start)

    if isinstance(dataset, dict):
        data = dataset
    else:
        with np.load(dataset) as npz:
            data = {
                "piece_bbs": npz["piece_bbs"],
                "occupancy_bbs": npz["occupancy_bbs"],
                "game_states": npz["game_states"],
                "results": npz["results"],
            }

    n_total = len(data["results"])
    train_idx, eval_idx, hold_idx = split_dataset_indices(
        n_total, eval_sample=eval_sample, holdout_sample=holdout_sample
    )
    excluded_n = min(max(exclude_fixed_val, 0), n_total)
    if excluded_n:
        pool = train_idx
        sample_n = min(max(eval_sample, 1), len(pool))
        indices = np.sort(
            np.random.default_rng(seed).choice(pool, size=sample_n, replace=False)
        )
    elif split == "holdout" and hold_idx is not None:
        sample_n = min(max(eval_sample, 1), len(hold_idx))
        indices = hold_idx[:sample_n] if sample_n == len(hold_idx) else np.sort(
            np.random.default_rng(seed).choice(hold_idx, size=sample_n, replace=False)
        )
    elif split == "train" and len(train_idx) > 0:
        sample_n = min(max(eval_sample, 1), len(train_idx))
        indices = np.sort(
            np.random.default_rng(seed).choice(train_idx, size=sample_n, replace=False)
        )
    elif eval_idx is not None:
        sample_n = min(max(eval_sample, 1), len(eval_idx))
        indices = eval_idx[:sample_n]
    else:
        sample_n = min(max(eval_sample, 1), n_total)
        indices = np.sort(
            np.random.default_rng(0).choice(n_total, size=sample_n, replace=False)
        )

    bbs = data["piece_bbs"][indices]
    occ = data["occupancy_bbs"][indices]
    states = data["game_states"][indices]
    results = data["results"][indices]
    buckets = dict(_bucket_masks(bbs, baseline, pm))
    report_names = (
        "all",
        "broad-EG phase<=64",
        "phase=0",
        "queenless",
        "minor-only EG",
        "rook EG queenless",
    )

    if verbose:
        print(
            f"Scan sample ({split}): {len(indices):,} (K={k_factor:.3f}); "
            f"excluded fixed validation={excluded_n:,}"
        )
    base_errors = compute_squared_errors(
        bbs, occ, states, results, baseline, k_factor
    )
    base_mse = {
        name: float(np.mean(base_errors[buckets[name]])) for name in report_names
    }

    rows = []
    combinations = list(itertools.product(*(values for _, values in grids)))
    for number, values in enumerate(combinations, 1):
        theta = baseline.copy()
        for index, value in zip(scalar_indices, values):
            theta[index] = value
        errors = compute_squared_errors(bbs, occ, states, results, theta, k_factor)
        deltas = {
            name: float(np.mean(errors[buckets[name]])) - base_mse[name]
            for name in report_names
        }
        rows.append((values, deltas))
        if verbose:
            print(f"  evaluated {number:>3}/{len(combinations)}: {values}")

    rows.sort(key=lambda row: (row[1]["all"], row[1]["broad-EG phase<=64"]))
    if verbose:
        headings = " ".join(f"{name:>12s}" for name, _ in grids)
        print("Candidates sorted by overall delta (negative is better):")
        print(
            f"  {headings} {'all':>12s} {'broad-EG':>12s} {'phase=0':>12s} "
            f"{'queenless':>12s} {'minor-EG':>12s} {'rook-EG':>12s}"
        )
        shown_rows = rows[:top] if top > 0 else rows
        for values, delta in shown_rows:
            value_text = " ".join(f"{value:>12d}" for value in values)
            print(
                f"  {value_text} {delta['all']:+.5e} "
                f"{delta['broad-EG phase<=64']:+.5e} {delta['phase=0']:+.5e} "
                f"{delta['queenless']:+.5e} {delta['minor-only EG']:+.5e} "
                f"{delta['rook EG queenless']:+.5e}"
            )

    best_values, best_delta = rows[0]
    best_theta = baseline.copy()
    for index, value in zip(scalar_indices, best_values):
        best_theta[index] = value
    return {
        "grids": grids,
        "rows": rows,
        "best_values": dict(zip((n for n, _ in grids), best_values)),
        "best_delta": best_delta,
        "best_theta": best_theta,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--grid", action="append", required=True, metavar="NAME=LO:HI"
    )
    parser.add_argument(
        "--split",
        choices=("dev", "train", "holdout"),
        default="dev",
        help="Pool to scan before independent holdout/PGN verification (default: dev)",
    )
    parser.add_argument("--eval-sample", type=int, default=1_000_000)
    parser.add_argument("--holdout-sample", type=int, default=2_000_000)
    parser.add_argument("--exclude-fixed-val", type=int, default=0)
    parser.add_argument("--k-factor", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="Print only the best N rows after sorting (0 prints every row)",
    )
    args = parser.parse_args()

    run_scalar_grid_scan(
        dataset=args.dataset,
        grid_specs=args.grid,
        split=args.split,
        eval_sample=args.eval_sample,
        holdout_sample=args.holdout_sample,
        exclude_fixed_val=args.exclude_fixed_val,
        k_factor=args.k_factor,
        seed=args.seed,
        top=args.top,
        verbose=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
