"""Paired MSE validation for a generated HCE tuning candidate.

Supports leak-free validation on:
1. The strictly unseen 2M ``holdout`` split of ``dataset_mix03_20m_s2.npz``
   (via ``split_dataset_indices``; never seen by SPSA gradients or early stopping).
2. An independent external PGN validation set (``--pgn-val tuner/data/hce/val_pgn.npz``)
   with real castling rights, en-passant squares, and 50-move clocks, evaluated
   at both ``K=1.0`` and baseline-calibrated ``K*``.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np

from tuner.apply_tuned import _eval_const_module
from tuner.constants_snapshot import (
    PHASE_ENDGAME_RATIO_DEN,
    PHASE_ENDGAME_RATIO_NUM,
    PHASE_MIDGAME,
)
from tuner.parameters import ParameterManager
from tuner.tuner import (
    compute_white_scores,
    fit_best_k,
    split_dataset_indices,
    squared_errors_from_scores,
)


def candidate_theta_from_file(pm: ParameterManager, tuned_path: Path) -> np.ndarray:
    theta = pm.get_initial_theta()
    namespace = _eval_const_module(tuned_path)
    for param in pm.param_map:
        name = param["name"]
        if name not in namespace:
            continue
        values = np.asarray(namespace[name], dtype=np.float64).reshape(-1)
        if len(values) != param["count"]:
            raise ValueError(f"Shape/count mismatch for {name} in {tuned_path}")
        start = param["start"]
        theta[start : start + param["count"]] = values
    return theta


# Backward-compatible private alias
_candidate_theta = candidate_theta_from_file


def _bucket_masks(
    piece_bbs: np.ndarray,
    theta: np.ndarray,
    pm: ParameterManager,
    game_states: np.ndarray | None = None,
):
    n = len(piece_bbs)
    lut = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)
    byte_view = np.ascontiguousarray(piece_bbs).view(np.uint8).reshape(n, 12, 8)
    counts = np.sum(lut[byte_view], axis=2, dtype=np.int16)
    total = np.sum(counts, axis=1, dtype=np.int16)

    mg_start, _ = pm.get_param_indices("MG_MATERIAL_VALUES")
    mat_n, mat_b, mat_r, mat_q = (
        int(theta[mg_start + i]) for i in range(1, 5)
    )
    npm = (
        (counts[:, 1] + counts[:, 7]) * mat_n
        + (counts[:, 2] + counts[:, 8]) * mat_b
        + (counts[:, 3] + counts[:, 9]) * mat_r
        + (counts[:, 4] + counts[:, 10]) * mat_q
    ).astype(np.int64)
    mid = 2 * (2 * mat_n + 2 * mat_b + 2 * mat_r + mat_q)
    end = (
        mid * int(PHASE_ENDGAME_RATIO_NUM)
        + int(PHASE_ENDGAME_RATIO_DEN) // 2
    ) // int(PHASE_ENDGAME_RATIO_DEN)
    phase = (
        (np.clip(npm, end, mid) - end) * int(PHASE_MIDGAME) // (mid - end)
    ).astype(np.int16)

    minor = counts[:, 1] + counts[:, 2] + counts[:, 7] + counts[:, 8]
    rooks = counts[:, 3] + counts[:, 9]
    queens = counts[:, 4] + counts[:, 10]
    non_pawns = minor + rooks + queens
    buckets = [
        ("all", np.ones(n, dtype=bool)),
        ("phase=0", phase == 0),
        ("phase=1..31", (phase >= 1) & (phase <= 31)),
        ("phase=32..63", (phase >= 32) & (phase <= 63)),
        ("phase=64..95", (phase >= 64) & (phase <= 95)),
        ("phase=96..128", (phase >= 96) & (phase <= 128)),
        ("broad-EG phase<=64", phase <= 64),
        ("pieces=8..12", (total >= 8) & (total <= 12)),
        ("pieces=13..20", (total >= 13) & (total <= 20)),
        ("queenless", queens == 0),
        ("pure king+pawns", non_pawns == 0),
        (
            "minor-only EG",
            (phase <= 64) & (minor > 0) & (rooks == 0) & (queens == 0),
        ),
        ("rook EG queenless", (phase <= 64) & (rooks > 0) & (queens == 0)),
    ]
    if game_states is not None:
        buckets.append(("castling!=0", game_states[:, 1] != 0))
    return tuple(buckets)


def _select_indices(
    n_total: int,
    split: str,
    sample: int,
    eval_sample: int,
    holdout_sample: int,
    exclude_fixed_val: int,
    seed: int,
) -> tuple[np.ndarray, str]:
    train_idx, eval_idx, hold_idx = split_dataset_indices(
        n_total, eval_sample=eval_sample, holdout_sample=holdout_sample
    )
    if split == "holdout":
        if hold_idx is not None and len(hold_idx) > 0:
            pool = hold_idx
            desc = (
                f"Unseen 3-way holdout split (excluded dev={len(eval_idx) if eval_idx is not None else 0:,}, "
                f"train={len(train_idx):,})"
            )
        else:
            pool = np.arange(n_total, dtype=np.int64)
            desc = "Full dataset (3-way holdout auto-disabled on small dataset)"
    elif split == "eval":
        pool = eval_idx if eval_idx is not None else np.arange(n_total, dtype=np.int64)
        desc = "Fixed SPSA dev validation split"
    elif split == "all":
        pool = np.arange(n_total, dtype=np.int64)
        desc = "Full external validation dataset"
    elif split == "train_pool":
        split_rng = np.random.default_rng(0)
        excluded = np.zeros(n_total, dtype=bool)
        n_excluded = min(max(exclude_fixed_val, 0), n_total)
        if n_excluded:
            excluded[split_rng.choice(n_total, size=n_excluded, replace=False)] = True
        pool = np.flatnonzero(~excluded).astype(np.int64)
        desc = f"Legacy train-pool sample (excluded fixed val={n_excluded:,})"
    else:
        raise ValueError(f"Unknown split {split!r}")

    if sample and 0 < sample < len(pool):
        chosen = np.sort(
            np.random.default_rng(seed).choice(pool, size=int(sample), replace=False)
        ).astype(np.int64)
    else:
        chosen = np.asarray(pool, dtype=np.int64)
    return chosen, desc


def _paired_summary_from_errors(base_err: np.ndarray, cand_err: np.ndarray) -> dict:
    count = int(len(base_err))
    if count == 0:
        return {
            "n": 0,
            "base_mse": float("nan"),
            "cand_mse": float("nan"),
            "delta": float("nan"),
            "se": float("nan"),
            "z": float("nan"),
        }
    diff = cand_err - base_err
    mean = float(np.mean(diff))
    variance = float(np.var(diff, ddof=1)) if count > 1 else 0.0
    se = float(np.sqrt(max(variance, 0.0) / count))
    z = mean / se if se > 0.0 else (0.0 if mean == 0.0 else float("nan"))
    return {
        "n": count,
        "base_mse": float(np.mean(base_err)),
        "cand_mse": float(np.mean(cand_err)),
        "delta": mean,
        "se": se,
        "z": z,
    }


def run_paired_validation(
    dataset,
    baseline: np.ndarray,
    candidate: np.ndarray,
    pm: ParameterManager,
    *,
    label: str = "validation",
    split: str = "holdout",
    sample: int = 2_000_000,
    eval_sample: int = 1_000_000,
    holdout_sample: int = 2_000_000,
    exclude_fixed_val: int = 1_000_000,
    chunk_size: int = 500_000,
    k_factor: float = 1.0,
    fit_k: bool = False,
    seed: int = 2,
    cached_base_scores: np.ndarray | None = None,
    indices_override: np.ndarray | None = None,
    verbose: bool = True,
) -> dict:
    """Evaluate paired candidate vs baseline MSE, SE, and z-scores across buckets."""
    if isinstance(dataset, (str, Path)):
        with np.load(dataset) as npz:
            data = {
                "piece_bbs": npz["piece_bbs"],
                "occupancy_bbs": npz["occupancy_bbs"],
                "game_states": npz["game_states"],
                "results": npz["results"],
            }
    else:
        data = dataset

    n_total = len(data["results"])
    if indices_override is not None:
        indices = np.asarray(indices_override, dtype=np.int64)
        split_desc = f"Pre-selected indices ({len(indices):,})"
    else:
        indices, split_desc = _select_indices(
            n_total=n_total,
            split=split,
            sample=sample,
            eval_sample=eval_sample,
            holdout_sample=holdout_sample,
            exclude_fixed_val=exclude_fixed_val,
            seed=seed,
        )

    n_sel = len(indices)
    if verbose:
        print(f"[{label}] {split_desc}: n={n_sel:,} (K={k_factor:.3f})")

    has_castling = bool(np.any(data["game_states"][indices, 1] != 0)) if n_sel else False
    dummy_gs = data["game_states"][indices[:1]] if has_castling else None
    bucket_names = [
        b_name
        for b_name, _ in _bucket_masks(
            data["piece_bbs"][indices[:1]], baseline, pm, game_states=dummy_gs
        )
    ]
    stats = {
        name: {"n": 0, "base": 0.0, "cand": 0.0, "d": 0.0, "d2": 0.0}
        for name in bucket_names
    }

    all_base_scores = (
        cached_base_scores
        if (cached_base_scores is not None and len(cached_base_scores) == n_sel)
        else np.empty(n_sel, dtype=np.int32)
    )
    need_base_eval = cached_base_scores is None or len(cached_base_scores) != n_sel
    all_cand_scores = np.empty(n_sel, dtype=np.int32) if fit_k else None

    step = max(int(chunk_size), 1)
    for offset in range(0, n_sel, step):
        end_off = min(offset + step, n_sel)
        idx = indices[offset:end_off]
        bbs = data["piece_bbs"][idx]
        occ = data["occupancy_bbs"][idx]
        states = data["game_states"][idx]
        results = data["results"][idx]

        if need_base_eval:
            b_scores = compute_white_scores(bbs, occ, states, baseline)
            all_base_scores[offset:end_off] = b_scores
        else:
            b_scores = all_base_scores[offset:end_off]

        c_scores = compute_white_scores(bbs, occ, states, candidate)
        if all_cand_scores is not None:
            all_cand_scores[offset:end_off] = c_scores

        base_err = squared_errors_from_scores(b_scores, results, k_factor)
        cand_err = squared_errors_from_scores(c_scores, results, k_factor)
        diff = cand_err - base_err

        chunk_gs = states if has_castling else None
        for b_name, mask in _bucket_masks(bbs, baseline, pm, game_states=chunk_gs):
            count = int(np.count_nonzero(mask))
            if not count:
                continue
            selected_diff = diff[mask]
            stat = stats[b_name]
            stat["n"] += count
            stat["base"] += float(np.sum(base_err[mask]))
            stat["cand"] += float(np.sum(cand_err[mask]))
            stat["d"] += float(np.sum(selected_diff))
            stat["d2"] += float(np.dot(selected_diff, selected_diff))

        if verbose and n_sel > step:
            print(f"  processed {end_off:,}/{n_sel:,}")

    bucket_reports: dict[str, dict] = {}
    if verbose:
        print(f"Paired candidate - baseline @ K={k_factor:.3f} (negative is better):")
    for b_name in bucket_names:
        stat = stats[b_name]
        count = stat["n"]
        if count == 0:
            bucket_reports[b_name] = {
                "n": 0,
                "base_mse": float("nan"),
                "cand_mse": float("nan"),
                "delta": float("nan"),
                "se": float("nan"),
                "z": float("nan"),
            }
            continue
        mean = stat["d"] / count
        variance = max((stat["d2"] - stat["d"] ** 2 / count) / max(count - 1, 1), 0.0)
        se = float(np.sqrt(variance / count))
        z = mean / se if se > 0.0 else (0.0 if mean == 0.0 else float("nan"))
        base_mse = stat["base"] / count
        cand_mse = stat["cand"] / count
        bucket_reports[b_name] = {
            "n": count,
            "base_mse": base_mse,
            "cand_mse": cand_mse,
            "delta": mean,
            "se": se,
            "z": z,
        }
        if verbose:
            print(
                f"  {b_name:20s} n={count:>10,}  "
                f"base={base_mse:.10f}  cand={cand_mse:.10f}  "
                f"delta={mean:+.10e}  SE={se:.3e}  z={z:+.3f}"
            )

    fitted_k_report = None
    if fit_k and n_sel > 0 and all_cand_scores is not None:
        sel_results = data["results"][indices]
        best_k, _ = fit_best_k(all_base_scores, sel_results)
        b_err_k = squared_errors_from_scores(all_base_scores, sel_results, best_k)
        c_err_k = squared_errors_from_scores(all_cand_scores, sel_results, best_k)
        fitted_k_report = _paired_summary_from_errors(b_err_k, c_err_k)
        fitted_k_report["k_star"] = best_k
        if verbose:
            print(
                f"  [Calibrated K*={best_k:.3f} (fit on baseline only)]  "
                f"base={fitted_k_report['base_mse']:.10f}  "
                f"cand={fitted_k_report['cand_mse']:.10f}  "
                f"delta={fitted_k_report['delta']:+.10e}  "
                f"SE={fitted_k_report['se']:.3e}  z={fitted_k_report['z']:+.3f}"
            )

    return {
        "label": label,
        "split": split,
        "n": n_sel,
        "k_factor": k_factor,
        "buckets": bucket_reports,
        "fitted_k": fitted_k_report,
        "base_scores": all_base_scores,
        "indices": indices,
    }


def print_effective_integer_changes(
    pm: ParameterManager, baseline: np.ndarray, candidate: np.ndarray
) -> list[dict]:
    rounded_delta = np.rint(candidate).astype(np.int64) - np.rint(baseline).astype(np.int64)
    changes: list[dict] = []
    print("Effective integer changes:")
    for param in pm.param_map:
        start, count = param["start"], param["count"]
        local = rounded_delta[start : start + count]
        if not np.any(local):
            continue
        if count == 1:
            old_v = int(round(baseline[start]))
            new_v = int(round(candidate[start]))
            print(f"  {param['name']}: {old_v} -> {new_v}")
            changes.append({"name": param["name"], "count": 1, "old": old_v, "new": new_v})
        else:
            n_ch = int(np.count_nonzero(local))
            l1 = int(np.sum(np.abs(local)))
            print(f"  {param['name']}: changed={n_ch}/{count}, L1={l1}")
            changes.append({"name": param["name"], "count": count, "changed": n_ch, "l1": l1})
    if not changes:
        print("  (none)")
    return changes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tuned", required=True)
    parser.add_argument(
        "--pgn-val",
        default=None,
        help="Optional independent PGN validation .npz (e.g. tuner/data/hce/val_pgn.npz)",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Override a scalar or indexed candidate value after loading --tuned",
    )
    parser.add_argument(
        "--split",
        choices=("holdout", "train_pool", "eval", "all"),
        default="holdout",
        help="Validation split on --dataset (default: unseen 2M 'holdout')",
    )
    parser.add_argument("--sample", type=int, default=2_000_000)
    parser.add_argument("--eval-sample", type=int, default=1_000_000)
    parser.add_argument("--holdout-sample", type=int, default=2_000_000)
    parser.add_argument("--exclude-fixed-val", type=int, default=1_000_000)
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--k-factor", type=float, default=1.0)
    parser.add_argument(
        "--fit-k",
        action="store_true",
        help="Also report paired delta at baseline-fitted optimal K*",
    )
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()

    pm = ParameterManager()
    baseline = pm.get_initial_theta()
    candidate = candidate_theta_from_file(pm, Path(args.tuned))
    indexed_name = re.compile(r"^(.+)\[(\d+)\]$")
    for assignment in args.override:
        if "=" not in assignment:
            raise ValueError(f"Expected NAME=VALUE, got {assignment!r}")
        name, raw_value = assignment.split("=", 1)
        match = indexed_name.match(name)
        if match:
            param_name, raw_index = match.groups()
            start, count = pm.get_param_indices(param_name)
            relative_index = int(raw_index)
            if not 0 <= relative_index < count:
                raise ValueError(f"Override index out of range: {name}")
            candidate[start + relative_index] = float(raw_value)
        else:
            start, count = pm.get_param_indices(name)
            if count != 1:
                raise ValueError(
                    f"Array override requires NAME[flat_index]=VALUE: {name}"
                )
            candidate[start] = float(raw_value)

    print_effective_integer_changes(pm, baseline, candidate)

    run_paired_validation(
        dataset=args.dataset,
        baseline=baseline,
        candidate=candidate,
        pm=pm,
        label=Path(args.dataset).name,
        split=args.split,
        sample=args.sample,
        eval_sample=args.eval_sample,
        holdout_sample=args.holdout_sample,
        exclude_fixed_val=args.exclude_fixed_val,
        chunk_size=args.chunk_size,
        k_factor=args.k_factor,
        fit_k=args.fit_k,
        seed=args.seed,
        verbose=True,
    )

    if args.pgn_val:
        print()
        run_paired_validation(
            dataset=args.pgn_val,
            baseline=baseline,
            candidate=candidate,
            pm=pm,
            label=Path(args.pgn_val).name,
            split="all",
            sample=0,
            chunk_size=args.chunk_size,
            k_factor=args.k_factor,
            fit_k=True,
            seed=args.seed,
            verbose=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
