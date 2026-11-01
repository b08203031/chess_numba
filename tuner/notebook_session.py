"""In-process session controller for interactive HCE Texel/SPSA notebook tuning.

Why in-process:
- ``dataset_mix03_20m_s2.npz`` (20M rows) occupies ~3.44 GB uncompressed in RAM
  and ``val_pgn.npz`` (~1.59M rows) occupies ~0.28 GB (~3.72 GB total).
- Keeping a single preloaded copy inside the Jupyter kernel avoids spawning
  subprocesses that would allocate a second 3.44 GB copy (risking OOM/swap on
  16 GB machines) and avoids repeating Numba JIT warmups across blocks.
- ``tuner/tunable_eval.py`` reads all 109 registered parameters dynamically from
  ``theta: float64[:]``, so accepting a block into ``constants.py`` and reloading
  ``constants_snapshot`` updates ``theta0`` immediately with **zero** Numba
  recompilation of ``evaluate_position_tunable``.
"""
from __future__ import annotations

from datetime import datetime
import gc
import importlib
from pathlib import Path
import re
import shutil
import time

import numpy as np

from tuner import apply_tuned as apply_tuned_mod
from tuner import constants_snapshot
from tuner import parameters as parameters_mod
from tuner import sync_constants_snapshot
from tuner.scan_scalar_grid import run_scalar_grid_scan
from tuner.tunable_eval import evaluate_position_tunable
from tuner.tuner import (
    SPSAOptimizer,
    compute_white_scores,
    fit_best_k,
    split_dataset_indices,
    squared_errors_from_scores,
)
from tuner.validate_tuned import (
    candidate_theta_from_file,
    print_effective_integer_changes,
    run_paired_validation,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_NPZ = ROOT / "tuner" / "data" / "hce" / "dataset_mix03_20m_s2.npz"
DEFAULT_PGN_VAL_NPZ = ROOT / "tuner" / "data" / "hce" / "val_pgn.npz"
DEFAULT_RUNS_DIR = ROOT / "tuner" / "runs"
CONSTANTS_PATH = ROOT / "chess_engine" / "classical" / "constants.py"


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def _avail_ram_gb() -> tuple[float | None, float | None]:
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return stat.ullAvailPhys / (1024 ** 3), stat.ullTotalPhys / (1024 ** 3)
    except Exception:
        return None, None


def _ram_summary_gb() -> str:
    avail, total = _avail_ram_gb()
    if avail is None or total is None:
        return "n/a"
    return f"{avail:.2f} GB free / {total:.2f} GB total"


class TexelTuningSession:
    """Single-copy RAM manager for sequential block-coordinate HCE tuning."""

    def __init__(
        self,
        train_npz: str | Path = DEFAULT_TRAIN_NPZ,
        pgn_val_npz: str | Path | None = DEFAULT_PGN_VAL_NPZ,
        runs_dir: str | Path = DEFAULT_RUNS_DIR,
        eval_sample: int = 1_000_000,
        holdout_sample: int = 2_000_000,
        max_train_rows: int | str | None = "auto",
        chunk_size: int = 500_000,
    ):
        self.train_path = Path(train_npz)
        self.pgn_val_path = Path(pgn_val_npz) if pgn_val_npz else None
        self.runs_dir = Path(runs_dir)
        self.backups_dir = self.runs_dir / "backups"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.backups_dir.mkdir(parents=True, exist_ok=True)

        self.eval_sample = int(eval_sample)
        self.holdout_sample = int(holdout_sample)
        self.chunk_size = int(chunk_size)

        if not self.train_path.is_file():
            raise FileNotFoundError(f"Missing training dataset: {self.train_path}")

        t0 = time.time()
        avail_gb, _ = _avail_ram_gb()
        print(f"[1/4] Loading primary dataset: {self.train_path.name} (RAM before: {_ram_summary_gb()})")
        with np.load(self.train_path) as npz:
            results_full = npz["results"]
            n_full = len(results_full)
            train_orig, dev_orig, hold_orig = split_dataset_indices(
                n_full, eval_sample=self.eval_sample, holdout_sample=self.holdout_sample
            )
            resolved_max_train: int | None
            if max_train_rows == "auto":
                if len(train_orig) > 10_000_000 and avail_gb is not None and avail_gb < 4.8:
                    resolved_max_train = 10_000_000
                    print(
                        f"      [auto-RAM] Free physical RAM is {avail_gb:.2f} GB (< 4.8 GB); "
                        "capping train pool to 10,000,000 rows while keeping full 1M dev + 2M holdout "
                        "(pass max_train_rows=None to force all 17M)."
                    )
                else:
                    resolved_max_train = None
            else:
                resolved_max_train = int(max_train_rows) if max_train_rows is not None else None

            if resolved_max_train is not None and 0 < resolved_max_train < len(train_orig):
                sub_train = np.sort(
                    np.random.default_rng(123).choice(
                        train_orig, size=resolved_max_train, replace=False
                    )
                )
                parts = [sub_train]
                if dev_orig is not None:
                    parts.append(dev_orig)
                if hold_orig is not None:
                    parts.append(hold_orig)
                keep_idx = np.sort(np.concatenate(parts))
                # Map original indices into compact [0..len(keep_idx)-1] positions
                self.train_idx = np.searchsorted(keep_idx, sub_train).astype(np.int64)
                self.dev_idx = (
                    np.searchsorted(keep_idx, dev_orig).astype(np.int64)
                    if dev_orig is not None
                    else None
                )
                self.holdout_idx = (
                    np.searchsorted(keep_idx, hold_orig).astype(np.int64)
                    if hold_orig is not None
                    else None
                )
                results_kept = results_full[keep_idx]
                del results_full
                gc.collect()
                pb_kept = npz["piece_bbs"][keep_idx]
                gc.collect()
                ob_kept = npz["occupancy_bbs"][keep_idx]
                gc.collect()
                gs_kept = npz["game_states"][keep_idx]
                gc.collect()
                self.train_data = {
                    "label": f"{self.train_path.name} (subsampled train={len(self.train_idx):,})",
                    "piece_bbs": pb_kept,
                    "occupancy_bbs": ob_kept,
                    "game_states": gs_kept,
                    "results": results_kept,
                }
            else:
                self.train_idx = train_orig
                self.dev_idx = dev_orig
                self.holdout_idx = hold_orig
                self.train_data = {
                    "label": self.train_path.name,
                    "piece_bbs": npz["piece_bbs"],
                    "occupancy_bbs": npz["occupancy_bbs"],
                    "game_states": npz["game_states"],
                    "results": results_full,
                }
        gc.collect()

        n_total = len(self.train_data["results"])
        print(
            f"      Loaded {n_total:,} positions -> "
            f"train={len(self.train_idx):,} | "
            f"dev={len(self.dev_idx) if self.dev_idx is not None else 0:,} | "
            f"unseen_holdout={len(self.holdout_idx) if self.holdout_idx is not None else 0:,} "
            f"(RAM now: {_ram_summary_gb()})"
        )

        self.pgn_data = None
        if self.pgn_val_path and self.pgn_val_path.is_file():
            print(f"[2/4] Loading independent PGN validation set: {self.pgn_val_path.name}")
            with np.load(self.pgn_val_path) as npz:
                self.pgn_data = {
                    "label": self.pgn_val_path.name,
                    "piece_bbs": npz["piece_bbs"],
                    "occupancy_bbs": npz["occupancy_bbs"],
                    "game_states": npz["game_states"],
                    "results": npz["results"],
                    "search_cp": npz["search_cp"] if "search_cp" in npz.files else None,
                    "depth": npz["depth"] if "depth" in npz.files else None,
                }
            gc.collect()
            n_pgn = len(self.pgn_data["results"])
            n_castle = int(np.count_nonzero(self.pgn_data["game_states"][:, 1] != 0))
            print(
                f"      Loaded {n_pgn:,} unique PGN positions "
                f"(castling!=0: {n_castle:,} [{n_castle / max(n_pgn, 1):.2%}], "
                f"RAM now: {_ram_summary_gb()})"
            )
        else:
            print(f"[2/4] PGN validation file not found ({self.pgn_val_path}); running holdout-only.")

        print("[3/4] Initializing ParameterManager & warming up Numba JIT...")
        self._reload_parameter_manager()
        self.initial_theta0 = self.step_baseline_theta.copy()

        # Warm up JIT kernels on a 16-row slice
        sl = slice(0, min(16, n_total))
        _ = evaluate_position_tunable(
            self.train_data["piece_bbs"][0],
            self.train_data["occupancy_bbs"][0],
            self.train_data["game_states"][0],
            self.step_baseline_theta,
            False,
        )
        _ = compute_white_scores(
            self.train_data["piece_bbs"][sl],
            self.train_data["occupancy_bbs"][sl],
            self.train_data["game_states"][sl],
            self.step_baseline_theta,
        )

        print("[4/4] Caching baseline scores on unseen validation splits...")
        self._step_holdout_scores: np.ndarray | None = None
        self._step_pgn_scores: np.ndarray | None = None
        self._refresh_baseline_score_caches()
        self._initial_holdout_scores = (
            self._step_holdout_scores.copy() if self._step_holdout_scores is not None else None
        )
        self._initial_pgn_scores = (
            self._step_pgn_scores.copy() if self._step_pgn_scores is not None else None
        )

        self.history: dict[str, dict] = {}
        self.backup_stack: list[tuple[str, Path]] = []
        print(f"Session ready in {time.time() - t0:.1f}s! (RAM: {_ram_summary_gb()})")

    def _reload_parameter_manager(self) -> None:
        sync_constants_snapshot.main()
        importlib.reload(constants_snapshot)
        importlib.reload(parameters_mod)
        self.pm = parameters_mod.ParameterManager()
        self.step_baseline_theta = self.pm.get_initial_theta()

    def _eval_scores_chunked(
        self, data: dict, theta: np.ndarray, indices: np.ndarray | None = None
    ) -> np.ndarray:
        if indices is None:
            n = len(data["results"])
            out = np.empty(n, dtype=np.int32)
            for off in range(0, n, self.chunk_size):
                end = min(off + self.chunk_size, n)
                out[off:end] = compute_white_scores(
                    data["piece_bbs"][off:end],
                    data["occupancy_bbs"][off:end],
                    data["game_states"][off:end],
                    theta,
                )
            return out
        n = len(indices)
        out = np.empty(n, dtype=np.int32)
        for off in range(0, n, self.chunk_size):
            end = min(off + self.chunk_size, n)
            idx = indices[off:end]
            out[off:end] = compute_white_scores(
                data["piece_bbs"][idx],
                data["occupancy_bbs"][idx],
                data["game_states"][idx],
                theta,
            )
        return out

    def _refresh_baseline_score_caches(self) -> None:
        if self.holdout_idx is not None and len(self.holdout_idx) > 0:
            self._step_holdout_scores = self._eval_scores_chunked(
                self.train_data, self.step_baseline_theta, self.holdout_idx
            )
        else:
            self._step_holdout_scores = self._eval_scores_chunked(
                self.train_data, self.step_baseline_theta, None
            )
        if self.pgn_data is not None:
            self._step_pgn_scores = self._eval_scores_chunked(
                self.pgn_data, self.step_baseline_theta, None
            )

    def baseline_report(self) -> dict:
        """Print baseline MSE & optimal K* on unseen mix03 holdout and PGN validation."""
        report: dict = {}
        print("=== Current Step Baseline Health Report ===")
        h_idx = (
            self.holdout_idx
            if (self.holdout_idx is not None and len(self.holdout_idx) > 0)
            else np.arange(len(self.train_data["results"]), dtype=np.int64)
        )
        h_res = self.train_data["results"][h_idx]
        h_mse_k1 = float(np.mean(squared_errors_from_scores(self._step_holdout_scores, h_res, 1.0)))
        h_k_star, h_mse_kstar = fit_best_k(self._step_holdout_scores, h_res)
        print(
            f"  [1] {self.train_path.name} unseen holdout (n={len(h_idx):,}):\n"
            f"      MSE @ K=1.000 : {h_mse_k1:.7f}\n"
            f"      MSE @ K*={h_k_star:.3f}: {h_mse_kstar:.7f}"
        )
        report["holdout"] = {
            "n": int(len(h_idx)),
            "mse_k1": h_mse_k1,
            "k_star": h_k_star,
            "mse_k_star": h_mse_kstar,
        }

        if self.pgn_data is not None and self._step_pgn_scores is not None:
            p_res = self.pgn_data["results"]
            p_mse_k1 = float(np.mean(squared_errors_from_scores(self._step_pgn_scores, p_res, 1.0)))
            p_k_star, p_mse_kstar = fit_best_k(self._step_pgn_scores, p_res)
            castle_mask = self.pgn_data["game_states"][:, 1] != 0
            p_castle_mse = float(
                np.mean(squared_errors_from_scores(self._step_pgn_scores[castle_mask], p_res[castle_mask], p_k_star))
            ) if np.any(castle_mask) else float("nan")
            print(
                f"  [2] {self.pgn_val_path.name} independent PGN val (n={len(p_res):,}):\n"
                f"      Static Eval MSE @ K=1.000 : {p_mse_k1:.7f}\n"
                f"      Static Eval MSE @ K*={p_k_star:.3f}: {p_mse_kstar:.7f}\n"
                f"      Castling!=0 MSE @ K*      : {p_castle_mse:.7f} (n={int(np.count_nonzero(castle_mask)):,})"
            )
            scp = self.pgn_data.get("search_cp")
            if scp is not None:
                valid = np.isfinite(scp)
                if np.any(valid):
                    sk, smse = fit_best_k(scp[valid], p_res[valid])
                    print(
                        f"      300k Search Ceiling @ K*={sk:.3f}: {smse:.7f} "
                        f"(gap={p_mse_kstar - smse:+.7f})"
                    )
            report["pgn_val"] = {
                "n": int(len(p_res)),
                "mse_k1": p_mse_k1,
                "k_star": p_k_star,
                "mse_k_star": p_mse_kstar,
            }
        return report

    def _gate_summary(self, changes: list[dict], hold_res: dict, pgn_res: dict | None) -> str:
        if not changes:
            verdict = "NO-OP (0 integer changes)"
            print(f"\n>>> GATE VERDICT: {verdict}")
            return verdict

        h_all = hold_res["buckets"]["all"]
        h_eg = hold_res["buckets"]["broad-EG phase<=64"]
        reasons: list[str] = []
        severe = False

        if h_all["delta"] >= 0.0:
            severe = True
            reasons.append(f"mix03 holdout 'all' worsened ({h_all['delta']:+.2e}, z={h_all['z']:+.2f})")
        if h_eg["n"] > 0 and h_eg["z"] > 1.8:
            severe = True
            reasons.append(f"mix03 holdout 'broad-EG' regressed (z={h_eg['z']:+.2f})")
        elif h_eg["n"] > 0 and h_eg["delta"] > 0.0:
            reasons.append(f"mix03 holdout 'broad-EG' slightly positive (z={h_eg['z']:+.2f})")

        castling_sensitive = any(
            ch["name"]
            in {
                "PST_MG",
                "PST_EG",
                "SHELTER_STRENGTH",
                "UNBLOCKED_STORM",
                "BLOCKED_STORM",
                "BLOCKED_STORM_EG",
                "SHELTER_BASE_MG",
                "SHELTER_BASE_EG",
                "PAWNLESS_FLANK",
                "FLANK_ATTACKS",
                "TRAPPED_ROOK",
                "KING_ATTACK_WEIGHTS",
                "SAFE_CHECK_QUEEN",
                "SAFE_CHECK_ROOK",
                "SAFE_CHECK_BISHOP",
                "SAFE_CHECK_KNIGHT",
                "KING_DANGER_WEAK_SQ",
                "KING_DANGER_UNSAFE_CHECK",
                "KING_DANGER_BLOCKERS",
                "KING_DANGER_ATTACK_ON_KING_SQ",
                "KING_DANGER_NO_QUEEN",
                "KING_DANGER_NO_QUEEN_ROOKLESS",
                "KING_DANGER_KNIGHT_DEF",
                "KING_DANGER_OFFSET",
                "KING_DANGER_THRESHOLD",
            }
            for ch in changes
        )

        if pgn_res is not None:
            p_kstar = pgn_res["fitted_k"] or pgn_res["buckets"]["all"]
            p_all = pgn_res["buckets"]["all"]
            if p_kstar["z"] > 2.0 and p_all["z"] > 2.0:
                severe = True
                reasons.append(
                    f"val_pgn worsened significantly (z@K=1={p_all['z']:+.2f}, z@K*={p_kstar['z']:+.2f})"
                )
            elif p_kstar["delta"] > 0.0:
                reasons.append(
                    f"val_pgn @ K* slightly positive ({p_kstar['delta']:+.2e}, z={p_kstar['z']:+.2f})"
                )
            if "castling!=0" in pgn_res["buckets"]:
                c_b = pgn_res["buckets"]["castling!=0"]
                # PGN positions within a game share opening castling state (~10 plies/game),
                # inflating naive i.i.d. z-scores ~3x on tiny sub-1.5e-5 noise blips.
                z_warn = 2.5 if castling_sensitive else 3.5
                d_warn = 1.5e-5 if castling_sensitive else 3.0e-5
                if c_b["n"] > 0 and c_b["z"] > z_warn and c_b["delta"] > d_warn:
                    reasons.append(
                        f"val_pgn 'castling!=0' regressed (delta={c_b['delta']:+.2e}, z={c_b['z']:+.2f})"
                    )

        if severe:
            verdict = "REJECT"
        elif reasons:
            verdict = "CAUTION"
        else:
            verdict = "PASS"

        print("\n" + "=" * 72)
        print(f">>> GATE VERDICT: {verdict}  ({len(changes)} parameter group(s) changed)")
        print(
            f"    mix03 2M holdout : delta={h_all['delta']:+.4e} (z={h_all['z']:+.2f}) | "
            f"broad-EG delta={h_eg['delta']:+.4e} (z={h_eg['z']:+.2f})"
        )
        if pgn_res is not None:
            pk = pgn_res["fitted_k"]
            pa = pgn_res["buckets"]["all"]
            cb = pgn_res["buckets"].get("castling!=0")
            castle_txt = (
                f" | castling!=0 z={cb['z']:+.2f}" if (cb and cb["n"] > 0) else ""
            )
            print(
                f"    PGN independent  : delta@K=1={pa['delta']:+.4e} (z={pa['z']:+.2f}) | "
                f"delta@K*={pk['delta']:+.4e} (z={pk['z']:+.2f}){castle_txt}"
            )
        for r in reasons:
            print(f"    ! Note: {r}")
        print("=" * 72)
        return verdict

    def validate_candidate_theta(
        self, block_id: str, candidate: np.ndarray, out_path: Path | None = None
    ) -> dict:
        changes = print_effective_integer_changes(
            self.pm, self.step_baseline_theta, candidate
        )
        print()
        hold_indices = (
            self.holdout_idx
            if (self.holdout_idx is not None and len(self.holdout_idx) > 0)
            else np.arange(len(self.train_data["results"]), dtype=np.int64)
        )
        hold_res = run_paired_validation(
            dataset=self.train_data,
            baseline=self.step_baseline_theta,
            candidate=candidate,
            pm=self.pm,
            label=f"{self.train_path.name} [unseen holdout]",
            chunk_size=self.chunk_size,
            k_factor=1.0,
            fit_k=True,
            cached_base_scores=self._step_holdout_scores,
            indices_override=hold_indices,
            verbose=True,
        )

        pgn_res = None
        if self.pgn_data is not None:
            print()
            pgn_res = run_paired_validation(
                dataset=self.pgn_data,
                baseline=self.step_baseline_theta,
                candidate=candidate,
                pm=self.pm,
                label=f"{self.pgn_val_path.name} [independent PGN]",
                split="all",
                sample=0,
                chunk_size=self.chunk_size,
                k_factor=1.0,
                fit_k=True,
                cached_base_scores=self._step_pgn_scores,
                indices_override=np.arange(len(self.pgn_data["results"]), dtype=np.int64),
                verbose=True,
            )

        verdict = self._gate_summary(changes, hold_res, pgn_res)
        record = {
            "block_id": block_id,
            "path": str(out_path) if out_path else None,
            "verdict": verdict,
            "changes": changes,
            "holdout": hold_res,
            "pgn_val": pgn_res,
        }
        self.history[block_id] = record
        return record

    def run_block(
        self,
        block_id: str,
        *,
        tune: list[str] | None = None,
        exclude: list[str] | None = None,
        pst_pieces: str | None = None,
        king_pst_only: bool = False,
        non_king_pst_only: bool = False,
        iters: int = 3000,
        c: float = 2.5,
        c_floor: float = 1.25,
        alpha: float = 50000.0,
        batch_size: int = 65536,
        report_every: int = 50,
        patience: int = 25,
        metrics_sample: int = 250_000,
        seed: int = 0,
        k_factor: float = 1.0,
        auto_scale: bool = True,
        verbose_buckets: str = "endpoints",
    ) -> dict:
        """Run SPSA on one block in-memory and validate on both unseen sets."""
        out_path = self.runs_dir / f"{block_id}.py"
        print(f"=== Running Block [{block_id}] -> {_display_path(out_path)} ===")
        optimizer = SPSAOptimizer(self.train_data, self.pm)
        optimizer.optimize(
            iterations=iters,
            alpha=alpha,
            c=c,
            include_list=tune,
            exclude_list=exclude,
            batch_size=batch_size,
            eval_sample=self.eval_sample,
            holdout_sample=self.holdout_sample,
            metrics_sample=metrics_sample,
            seed=seed,
            report_every=report_every,
            patience=patience,
            king_pst_only=king_pst_only,
            non_king_pst_only=non_king_pst_only,
            pst_pieces=pst_pieces,
            save_path=str(out_path),
            k_factor=k_factor,
            train_indices_override=self.train_idx,
            eval_indices_override=self.dev_idx,
            c_floor=c_floor,
            auto_scale=auto_scale,
            verbose_buckets=verbose_buckets,
        )
        print(f"\n--- Dual Unseen Validation for [{block_id}] ---")
        return self.validate_candidate_theta(
            block_id, optimizer.best_theta.copy(), out_path=out_path
        )

    def scan_grid(
        self,
        block_id: str,
        grids: list[str],
        *,
        split: str = "dev",
        eval_sample: int = 1_000_000,
        k_factor: float = 1.0,
        top: int = 10,
    ) -> dict:
        """Exhaustively scan discrete integer scalars on dev, then verify on unseen sets."""
        out_path = self.runs_dir / f"{block_id}.py"
        print(f"=== Discrete Grid Scan [{block_id}] -> {_display_path(out_path)} ===")
        scan_pool = (
            {
                "piece_bbs": self.train_data["piece_bbs"][self.dev_idx],
                "occupancy_bbs": self.train_data["occupancy_bbs"][self.dev_idx],
                "game_states": self.train_data["game_states"][self.dev_idx],
                "results": self.train_data["results"][self.dev_idx],
            }
            if (split == "dev" and self.dev_idx is not None)
            else self.train_data
        )
        scan = run_scalar_grid_scan(
            dataset=scan_pool,
            grid_specs=grids,
            pm=self.pm,
            baseline=self.step_baseline_theta,
            split=split,
            eval_sample=eval_sample,
            holdout_sample=self.holdout_sample,
            k_factor=k_factor,
            top=top,
            verbose=True,
        )
        best_theta = scan["best_theta"]
        content = self.pm.update_constants_string(best_theta)
        with out_path.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(content)
        print(f"\nBest discrete grid point on {split}: {scan['best_values']}")
        print(f"--- Dual Unseen Validation for [{block_id}] ---")
        res = self.validate_candidate_theta(block_id, best_theta, out_path=out_path)
        res["grid_best_values"] = scan["best_values"]
        return res

    def validate_file(
        self,
        block_id_or_path: str | Path,
        overrides: list[str] | None = None,
    ) -> dict:
        """Validate a saved candidate file (with optional ``NAME=VAL`` overrides)."""
        p = Path(block_id_or_path)
        if not p.is_file():
            p = self.runs_dir / f"{block_id_or_path}.py"
        if not p.is_file():
            raise FileNotFoundError(f"Candidate file not found: {p}")

        cand = candidate_theta_from_file(self.pm, p)
        indexed_name = re.compile(r"^(.+)\[(\d+)\]$")
        for assignment in overrides or []:
            name, raw_val = assignment.split("=", 1)
            m = indexed_name.match(name)
            if m:
                pname, ridx = m.groups()
                start, count = self.pm.get_param_indices(pname)
                idx = int(ridx)
                if not 0 <= idx < count:
                    raise ValueError(f"Override index out of range: {assignment}")
                cand[start + idx] = float(raw_val)
            else:
                start, count = self.pm.get_param_indices(name)
                if count != 1:
                    raise ValueError(f"Array override needs NAME[idx]=VAL: {assignment}")
                cand[start] = float(raw_val)
        return self.validate_candidate_theta(p.stem, cand, out_path=p)

    def accept(self, block_id: str, only: list[str] | None = None) -> list[str]:
        """Backup constants.py, apply candidate via AST-safe merge, sync snapshot & caches."""
        cand_path = self.runs_dir / f"{block_id}.py"
        if not cand_path.is_file():
            raise FileNotFoundError(f"Missing candidate dump {cand_path}. Run block first.")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backups_dir / f"{ts}_before_{block_id}_constants.py"
        shutil.copyfile(CONSTANTS_PATH, backup_path)
        self.backup_stack.append((block_id, backup_path))

        changed = apply_tuned_mod.apply_tuned(
            tuned_path=cand_path,
            target_path=CONSTANTS_PATH,
            dry_run=False,
            only=only,
        )
        if not changed:
            print(f"[{block_id}] Nothing changed; constants.py unchanged.")
            return []

        self._reload_parameter_manager()
        self._refresh_baseline_score_caches()
        print(
            f"[{block_id}] Accepted {len(changed)} parameter group(s). "
            f"Backup saved to {_display_path(backup_path)}."
        )
        return changed

    def rollback(self, block_id: str | None = None) -> Path:
        """Restore constants.py from backup and resync snapshot & in-memory baseline."""
        if not self.backup_stack:
            backups = sorted(self.backups_dir.glob("*_constants.py"))
            if not backups:
                raise RuntimeError("No backups available in tuner/runs/backups/.")
            chosen = backups[-1]
            label = chosen.name
        elif block_id is None:
            label, chosen = self.backup_stack.pop()
        else:
            matches = [
                (i, b_id, b_path)
                for i, (b_id, b_path) in enumerate(self.backup_stack)
                if b_id == block_id
            ]
            if not matches:
                raise ValueError(f"No backup recorded for block_id={block_id!r}")
            idx, label, chosen = matches[-1]
            self.backup_stack = self.backup_stack[:idx]

        text = chosen.read_text(encoding="utf-8").replace("\r\n", "\n")
        with CONSTANTS_PATH.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        self._reload_parameter_manager()
        self._refresh_baseline_score_caches()
        print(f"Rolled back constants.py using {_display_path(chosen)} (tag={label}).")
        return chosen

    def cumulative_report(self) -> dict:
        """Compare current accepted constants.py against initial session-start theta0."""
        print("=== Cumulative Progress vs Initial Session Baseline (theta0) ===")
        changes = print_effective_integer_changes(
            self.pm, self.initial_theta0, self.step_baseline_theta
        )
        hold_indices = (
            self.holdout_idx
            if (self.holdout_idx is not None and len(self.holdout_idx) > 0)
            else np.arange(len(self.train_data["results"]), dtype=np.int64)
        )
        print()
        hold_res = run_paired_validation(
            dataset=self.train_data,
            baseline=self.initial_theta0,
            candidate=self.step_baseline_theta,
            pm=self.pm,
            label=f"{self.train_path.name} [cumulative vs start]",
            chunk_size=self.chunk_size,
            k_factor=1.0,
            fit_k=True,
            cached_base_scores=self._initial_holdout_scores,
            indices_override=hold_indices,
            verbose=True,
        )
        pgn_res = None
        if self.pgn_data is not None:
            print()
            pgn_res = run_paired_validation(
                dataset=self.pgn_data,
                baseline=self.initial_theta0,
                candidate=self.step_baseline_theta,
                pm=self.pm,
                label=f"{self.pgn_val_path.name} [cumulative vs start]",
                split="all",
                sample=0,
                chunk_size=self.chunk_size,
                k_factor=1.0,
                fit_k=True,
                cached_base_scores=self._initial_pgn_scores,
                indices_override=np.arange(len(self.pgn_data["results"]), dtype=np.int64),
                verbose=True,
            )
        verdict = self._gate_summary(changes, hold_res, pgn_res)
        return {
            "verdict": verdict,
            "changes": changes,
            "holdout": hold_res,
            "pgn_val": pgn_res,
        }
