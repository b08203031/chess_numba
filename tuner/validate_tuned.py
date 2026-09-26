"""Paired MSE validation for a generated HCE tuning candidate.

The default sample excludes tuner.py's fixed one-million-position validation
split.  It is therefore a useful secondary robustness check, although it is
not a genuinely unseen test set because SPSA mini-batches come from that pool.
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
from tuner.tuner import compute_squared_errors


def _candidate_theta(pm: ParameterManager, tuned_path: Path) -> np.ndarray:
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
        theta[start:start + param["count"]] = values
    return theta


def _bucket_masks(piece_bbs: np.ndarray, theta: np.ndarray, pm: ParameterManager):
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
    return (
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
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tuned", required=True)
    parser.add_argument(
        "--override", action="append", default=[], metavar="NAME=VALUE",
        help="Override a scalar candidate value after loading --tuned",
    )
    parser.add_argument("--sample", type=int, default=10_000_000)
    parser.add_argument("--exclude-fixed-val", type=int, default=1_000_000)
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()

    data = np.load(args.dataset)
    n_total = len(data["results"])
    pm = ParameterManager()
    baseline = pm.get_initial_theta()
    candidate = _candidate_theta(pm, Path(args.tuned))
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

    rounded_delta = np.rint(candidate).astype(np.int64) - np.rint(baseline).astype(np.int64)
    print("Effective integer changes:")
    for param in pm.param_map:
        start, count = param["start"], param["count"]
        local = rounded_delta[start:start + count]
        if not np.any(local):
            continue
        if count == 1:
            print(
                f"  {param['name']}: {int(round(baseline[start]))} -> "
                f"{int(round(candidate[start]))}"
            )
        else:
            print(
                f"  {param['name']}: changed={np.count_nonzero(local)}/{count}, "
                f"L1={int(np.sum(np.abs(local)))}"
            )

    split_rng = np.random.default_rng(0)
    excluded = np.zeros(n_total, dtype=bool)
    n_excluded = min(max(args.exclude_fixed_val, 0), n_total)
    if n_excluded:
        excluded[split_rng.choice(n_total, size=n_excluded, replace=False)] = True
    pool = np.flatnonzero(~excluded)
    sample_n = min(max(args.sample, 1), len(pool))
    if sample_n < len(pool):
        indices = np.random.default_rng(args.seed).choice(
            pool, size=sample_n, replace=False
        )
    else:
        indices = pool
    print(
        f"Secondary sample: {len(indices):,} from the SPSA train pool; "
        f"excluded fixed validation={n_excluded:,}"
    )

    names = [name for name, _ in _bucket_masks(data["piece_bbs"][indices[:1]], baseline, pm)]
    stats = {
        name: {"n": 0, "base": 0.0, "cand": 0.0, "d": 0.0, "d2": 0.0}
        for name in names
    }
    for offset in range(0, len(indices), args.chunk_size):
        idx = indices[offset:offset + args.chunk_size]
        bbs = data["piece_bbs"][idx]
        occ = data["occupancy_bbs"][idx]
        states = data["game_states"][idx]
        results = data["results"][idx]
        base_err = compute_squared_errors(bbs, occ, states, results, baseline)
        cand_err = compute_squared_errors(bbs, occ, states, results, candidate)
        diff = cand_err - base_err
        for name, mask in _bucket_masks(bbs, baseline, pm):
            count = int(np.count_nonzero(mask))
            if not count:
                continue
            selected_diff = diff[mask]
            stat = stats[name]
            stat["n"] += count
            stat["base"] += float(np.sum(base_err[mask]))
            stat["cand"] += float(np.sum(cand_err[mask]))
            stat["d"] += float(np.sum(selected_diff))
            stat["d2"] += float(np.dot(selected_diff, selected_diff))
        print(f"  processed {min(offset + len(idx), len(indices)):,}/{len(indices):,}")

    print("Paired candidate - baseline (negative is better):")
    for name in names:
        stat = stats[name]
        count = stat["n"]
        mean = stat["d"] / count
        variance = max((stat["d2"] - stat["d"] ** 2 / count) / max(count - 1, 1), 0.0)
        se = np.sqrt(variance / count)
        z = mean / se if se > 0.0 else float("nan")
        print(
            f"  {name:20s} n={count:>10,}  "
            f"base={stat['base']/count:.10f}  cand={stat['cand']/count:.10f}  "
            f"delta={mean:+.10e}  SE={se:.3e}  z={z:+.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
