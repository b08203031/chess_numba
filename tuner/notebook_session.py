"""In-process session controller for interactive HCE Texel/SPSA notebook tuning.

Why in-process:
- ``dataset_epd_wdl_20m.npz`` (20M rows) occupies ~3.44 GB uncompressed in RAM
  and ``val_pgn.npz`` (~1.59M rows) occupies ~0.28 GB (~3.72 GB total).
- Keeping a single preloaded copy inside the Jupyter kernel avoids spawning
  subprocesses that would allocate a second 3.44 GB copy (risking OOM/swap on
  16 GB machines) and avoids repeating Numba JIT warmups across blocks.
- ``tuner/tunable_eval.py`` reads all 115 registered parameters dynamically from
  ``theta: float64[:]``, so accepting a block into ``constants.py`` and reloading
  ``constants_snapshot`` updates ``theta0`` immediately with **zero** Numba
  recompilation of ``evaluate_position_tunable``.
"""
from __future__ import annotations

from datetime import datetime
import gc
import importlib
import os
from pathlib import Path
import re
import shutil
import time

import numpy as np

from tuner import apply_tuned as apply_tuned_mod
from tuner import constants_snapshot
from tuner import parameters as parameters_mod
from tuner import sync_constants_snapshot
from tuner.data_pipeline.annotate_sf_labels import (
    MATE_CP_BASE,
    UNANNOTATED_CP,
    default_sidecar_path,
)
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
DEFAULT_TRAIN_NPZ = ROOT / "tuner" / "data" / "hce" / "dataset_epd_wdl_20m.npz"
DEFAULT_PGN_VAL_NPZ = ROOT / "tuner" / "data" / "hce" / "val_pgn.npz"
DEFAULT_ZURICHESS_VAL_NPZ = ROOT / "tuner" / "data" / "hce" / "val_zurichess.npz"
DEFAULT_RUNS_DIR = ROOT / "tuner" / "runs"
ENGINE_CONSTANTS_PATHS: dict[str, Path] = {
    "classical": ROOT / "chess_engine" / "classical" / "constants.py",
    "classical_old": ROOT / "chess_engine" / "classical_old" / "constants.py",
}
ENGINE_CACHE_DIRS: dict[str, Path] = {
    "classical": ROOT / "chess_engine" / "classical" / "__pycache__",
    "classical_old": ROOT / "chess_engine" / "classical_old" / "__pycache__",
}
CONSTANTS_PATH = ENGINE_CONSTANTS_PATHS["classical"]


def _resolve_target_engine(
    target_engine: str,
    target_constants_path: str | Path | None = None,
) -> tuple[str, Path, Path | None, bool]:
    engine_key = str(target_engine).strip()
    if engine_key not in ENGINE_CONSTANTS_PATHS:
        valid = ", ".join(sorted(ENGINE_CONSTANTS_PATHS))
        raise ValueError(
            f"Invalid target_engine={target_engine!r}; expected one of: {valid}"
        )
    if target_constants_path is not None:
        c_path = Path(target_constants_path)
        return engine_key, c_path, None, True
    return (
        engine_key,
        ENGINE_CONSTANTS_PATHS[engine_key],
        ENGINE_CACHE_DIRS[engine_key],
        False,
    )


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def _parse_proc_meminfo_gb(meminfo_text: str) -> tuple[float | None, float | None]:
    vals_kib: dict[str, int] = {}
    for line in meminfo_text.splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        parts = rest.strip().split()
        if parts and parts[0].isdigit():
            vals_kib[key.strip()] = int(parts[0])
    total_kib = vals_kib.get("MemTotal")
    avail_kib = vals_kib.get("MemAvailable", vals_kib.get("MemFree"))
    if total_kib is None or avail_kib is None:
        return None, None
    return avail_kib / (1024 ** 2), total_kib / (1024 ** 2)


def _avail_ram_gb() -> tuple[float | None, float | None]:
    try:
        if os.name == "nt":
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

        meminfo_path = Path("/proc/meminfo")
        if meminfo_path.is_file():
            return _parse_proc_meminfo_gb(meminfo_path.read_text(encoding="utf-8"))
        return None, None
    except Exception:
        return None, None


def _ram_summary_gb() -> str:
    avail, total = _avail_ram_gb()
    if avail is None or total is None:
        return "n/a"
    return f"{avail:.2f} GB free / {total:.2f} GB total"


def build_blended_labels(
    results_wdl: np.ndarray,
    sf_cp: np.ndarray,
    sf_lambda: float,
    *,
    sf_d1_cp: np.ndarray | None = None,
    max_tactical_gap: int = 180,
    fit_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Calibrate ``sf_cp`` to empirical ``results_wdl`` and compute ``(1-lambda)*wdl + lambda*p_sf``.

    - Calibrates ``K_sf`` via ``fit_best_k`` strictly on ``fit_indices`` (e.g. ``train_idx``)
      so Stockfish 19's self-play WDL-normalized centipawns map unbiasedly onto the
      dataset's real game WDL distribution.
    - Detects non-quiet tactical swings where ``|sf_cp - sf_d1_cp| > max_tactical_gap``
      (when at least one side is within ``±800 cp``): sets ``lambda_eff = 0`` on those rows
      and marks ``quiet_annotated_mask = False`` so SPSA gradient batches skip tactical traps.
    """
    lam = float(sf_lambda)
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"sf_lambda must be in [0.0, 1.0], got {sf_lambda!r}")
    if len(results_wdl) != len(sf_cp):
        raise ValueError(
            f"Length mismatch: results_wdl={len(results_wdl)} vs sf_cp={len(sf_cp)}"
        )

    annotated_mask = np.asarray(sf_cp != UNANNOTATED_CP, dtype=bool)
    n_annotated = int(np.count_nonzero(annotated_mask))
    if n_annotated == 0:
        raise ValueError(
            "SF sidecar contains 0 annotated positions (all rows == UNANNOTATED_CP)."
        )

    if sf_d1_cp is not None and max_tactical_gap and max_tactical_gap > 0:
        has_d1 = np.asarray(sf_d1_cp != UNANNOTATED_CP, dtype=bool)
        cp_i32 = sf_cp.astype(np.int32)
        d1_i32 = sf_d1_cp.astype(np.int32)
        gap = np.abs(cp_i32 - d1_i32)
        not_already_decided = (np.abs(cp_i32) < 800) | (np.abs(d1_i32) < 800)
        tactical_mask = annotated_mask & has_d1 & (gap > int(max_tactical_gap)) & not_already_decided
    else:
        tactical_mask = np.zeros(len(results_wdl), dtype=bool)

    quiet_annotated_mask = annotated_mask & (~tactical_mask)

    # Calibrate K_sf on competitive quiet scores (|sf_cp| <= 1000) so tablebase /
    # decided outliers (±2000..±27999 cp) with drawn/blundered WDL outcomes do not
    # distort K_sf downward.
    cp_abs = np.abs(sf_cp.astype(np.int32))
    competitive = cp_abs <= 1000
    non_mate = cp_abs < (MATE_CP_BASE - 1000)
    if fit_indices is not None and len(fit_indices) > 0:
        fit_mask = np.zeros(len(results_wdl), dtype=bool)
        fit_mask[np.asarray(fit_indices, dtype=np.int64)] = True
        cal_mask = fit_mask & quiet_annotated_mask & competitive
        if not np.any(cal_mask):
            cal_mask = fit_mask & annotated_mask & non_mate
    else:
        cal_mask = quiet_annotated_mask & competitive
        if not np.any(cal_mask):
            cal_mask = annotated_mask & non_mate
    if not np.any(cal_mask):
        cal_mask = annotated_mask

    k_sf, sf_cal_mse = fit_best_k(sf_cp[cal_mask], results_wdl[cal_mask])

    blended = np.array(results_wdl, dtype=np.float32, copy=True)
    if lam > 0.0 and np.any(quiet_annotated_mask):
        scale = 0.00575646273 * k_sf
        scores_q = sf_cp[quiet_annotated_mask].astype(np.float64) * scale
        p_sf = 1.0 / (1.0 + np.exp(-np.clip(scores_q, -40.0, 40.0)))
        wdl_q = results_wdl[quiet_annotated_mask].astype(np.float64)
        blended[quiet_annotated_mask] = ((1.0 - lam) * wdl_q + lam * p_sf).astype(
            np.float32
        )

    stats = {
        "sf_lambda": lam,
        "k_sf": float(k_sf),
        "sf_vs_wdl_mse": float(sf_cal_mse),
        "n_total": int(len(results_wdl)),
        "n_annotated": n_annotated,
        "n_tactical_filtered": int(np.count_nonzero(tactical_mask)),
        "n_quiet_annotated": int(np.count_nonzero(quiet_annotated_mask)),
        "annotated_mask": annotated_mask,
        "tactical_mask": tactical_mask,
        "quiet_annotated_mask": quiet_annotated_mask,
    }
    return blended, stats


class TexelTuningSession:
    """Single-copy RAM manager for sequential block-coordinate HCE tuning."""

    def __init__(
        self,
        train_npz: str | Path = DEFAULT_TRAIN_NPZ,
        pgn_val_npz: str | Path | None = DEFAULT_PGN_VAL_NPZ,
        runs_dir: str | Path | None = None,
        eval_sample: int = 1_000_000,
        holdout_sample: int = 2_000_000,
        max_train_rows: int | str | None = "auto",
        chunk_size: int = 500_000,
        *,
        use_pgn_val: bool = True,
        use_zurichess_val: bool = False,
        zurichess_val_npz: str | Path | None = DEFAULT_ZURICHESS_VAL_NPZ,
        target_engine: str = "classical",
        target_constants_path: str | Path | None = None,
        use_sf_labels: bool = False,
        sf_lambda: float = 0.25,
        sf_labels_path: str | Path | None = None,
        sf_max_tactical_gap: int = 180,
        drive_backup_dir: str | Path | None = None,
        random_seed: bool = False,
    ):
        self.train_path = Path(train_npz)
        self.pgn_val_path = Path(pgn_val_npz) if pgn_val_npz else None
        self.zurichess_val_path = (
            Path(zurichess_val_npz) if zurichess_val_npz else None
        )
        self.use_pgn_val = bool(use_pgn_val)
        self.use_zurichess_val = bool(use_zurichess_val)
        self.use_sf_labels = bool(use_sf_labels)
        self.sf_lambda = float(sf_lambda)
        self.sf_labels_path = (
            Path(sf_labels_path)
            if sf_labels_path is not None
            else default_sidecar_path(self.train_path)
        )
        self.sf_max_tactical_gap = int(sf_max_tactical_gap)
        self.drive_backup_dir = Path(drive_backup_dir) if drive_backup_dir else None
        self.random_seed = bool(random_seed)
        (
            self.target_engine,
            self.target_constants_path,
            self.target_cache_dir,
            self._custom_constants_override,
        ) = _resolve_target_engine(target_engine, target_constants_path)
        self._custom_runs_dir = Path(runs_dir) if runs_dir is not None else None
        self._update_run_directories()

        self.eval_sample = int(eval_sample)
        self.holdout_sample = int(holdout_sample)
        self.chunk_size = int(chunk_size)
        self.history: dict[str, dict] = {}
        self.backup_stack: list[tuple[str, Path]] = []
        self.require_pass_to_accept: bool = False

        if not self.train_path.is_file():
            raise FileNotFoundError(f"Missing training dataset: {self.train_path}")
        if not self.target_constants_path.is_file():
            raise FileNotFoundError(
                f"Missing target constants file: {self.target_constants_path}"
            )

        t0 = time.time()
        avail_gb, _ = _avail_ram_gb()
        print(
            f"[1/4] Loading primary dataset: {self.train_path.name} "
            f"(RAM before: {_ram_summary_gb()})"
        )
        self._sf_cp: np.ndarray | None = None
        self._sf_d1_cp: np.ndarray | None = None
        with np.load(self.train_path) as npz:
            results_full = npz["results"]
            n_full = len(results_full)
            self._n_full = n_full
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
                resolved_max_train = (
                    int(max_train_rows) if max_train_rows is not None else None
                )

            has_inline_sf = "sf_cp" in npz.files
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
                self._keep_idx: np.ndarray | None = keep_idx
                # Map original indices into compact [0..len(keep_idx)-1] positions
                self._base_train_idx = np.searchsorted(keep_idx, sub_train).astype(np.int64)
                self._base_dev_idx = (
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
                if has_inline_sf:
                    self._sf_cp = np.asarray(npz["sf_cp"][keep_idx], dtype=np.int16)
                    if "sf_d1_cp" in npz.files:
                        self._sf_d1_cp = np.asarray(npz["sf_d1_cp"][keep_idx], dtype=np.int16)
                self.train_data = {
                    "label": f"{self.train_path.name} (subsampled train={len(self._base_train_idx):,})",
                    "piece_bbs": pb_kept,
                    "occupancy_bbs": ob_kept,
                    "game_states": gs_kept,
                    "results": results_kept.copy(),
                }
                self._raw_results_wdl = results_kept
            else:
                self._keep_idx = None
                self._base_train_idx = train_orig
                self._base_dev_idx = dev_orig
                self.holdout_idx = hold_orig
                if has_inline_sf:
                    self._sf_cp = np.asarray(npz["sf_cp"], dtype=np.int16)
                    if "sf_d1_cp" in npz.files:
                        self._sf_d1_cp = np.asarray(npz["sf_d1_cp"], dtype=np.int16)
                self._raw_results_wdl = np.array(results_full, dtype=np.float32, copy=True)
                self.train_data = {
                    "label": self.train_path.name,
                    "piece_bbs": npz["piece_bbs"],
                    "occupancy_bbs": npz["occupancy_bbs"],
                    "game_states": npz["game_states"],
                    "results": results_full,
                }
        gc.collect()

        self.train_idx = self._base_train_idx.copy()
        self.dev_idx = (
            self._base_dev_idx.copy() if self._base_dev_idx is not None else None
        )
        self.sf_blend_stats: dict | None = None
        self._apply_label_mode()

        n_total = len(self.train_data["results"])
        print(
            f"      Loaded {n_total:,} positions -> "
            f"active_train={len(self.train_idx):,} | "
            f"active_dev={len(self.dev_idx) if self.dev_idx is not None else 0:,} | "
            f"unseen_holdout(WDL)={len(self.holdout_idx) if self.holdout_idx is not None else 0:,} "
            f"(RAM now: {_ram_summary_gb()})"
        )

        self.pgn_data = None
        if not self.use_pgn_val:
            print(
                "[2/4] Independent PGN validation disabled (use_pgn_val=False); "
                "running unseen holdout-only mode."
            )
        else:
            self._load_pgn_data_if_needed()

        self.zurichess_data = None
        if self.use_zurichess_val:
            self._load_zurichess_data_if_needed()

        print(
            f"[3/4] Initializing ParameterManager for target={self.target_engine!r} "
            f"({_display_path(self.target_constants_path)}) & warming up Numba JIT..."
        )
        self._reload_parameter_manager(clear_target_cache=False)
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

        print("[4/4] Caching baseline scores & calibrating K* on splits...")
        self._step_holdout_scores: np.ndarray | None = None
        self._step_pgn_scores: np.ndarray | None = None
        self._step_zurichess_scores: np.ndarray | None = None
        self._step_dev_k_star: float = 1.0
        self._step_holdout_k_star: float = 1.0
        self._step_pgn_k_star: float = 1.0
        self._step_zurichess_k_star: float = 1.0
        self._refresh_baseline_score_caches()
        self._initial_dev_k_star = self._step_dev_k_star
        self._initial_holdout_k_star = self._step_holdout_k_star
        self._initial_pgn_k_star = self._step_pgn_k_star
        self._initial_zurichess_k_star = self._step_zurichess_k_star
        self._initial_holdout_scores = (
            self._step_holdout_scores.copy()
            if self._step_holdout_scores is not None
            else None
        )
        self._initial_pgn_scores = (
            self._step_pgn_scores.copy() if self._step_pgn_scores is not None else None
        )
        self._initial_zurichess_scores = (
            self._step_zurichess_scores.copy()
            if self._step_zurichess_scores is not None
            else None
        )

        self.history = {}
        self.backup_stack = []
        lbl_tag = (
            f"sf_mix(lambda={self.sf_lambda:.2f}, K_sf={self.sf_blend_stats['k_sf']:.3f})"
            if (self.use_sf_labels and self.sf_lambda > 0.0 and self.sf_blend_stats)
            else "pure_WDL"
        )
        print(
            f"Session ready in {time.time() - t0:.1f}s! "
            f"[target={self.target_engine} -> {_display_path(self.target_constants_path)} | "
            f"use_pgn_val={self.use_pgn_val} | labels={lbl_tag} | "
            f"auto_K*={self._step_dev_k_star:.3f} | runs={_display_path(self.runs_dir)}] "
            f"(RAM: {_ram_summary_gb()})"
        )

    def _update_run_directories(self) -> None:
        if self._custom_runs_dir is not None:
            self.runs_dir = self._custom_runs_dir
        else:
            val_tag = "with_pgn" if self.use_pgn_val else "holdout_only"
            sf_tag = (
                f"_sf{int(round(self.sf_lambda * 100)):02d}"
                if (getattr(self, "use_sf_labels", False) and getattr(self, "sf_lambda", 0.0) > 0.0)
                else ""
            )
            self.runs_dir = DEFAULT_RUNS_DIR / f"{self.target_engine}_{val_tag}{sf_tag}"
        self.backups_dir = self.runs_dir / "backups"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.backups_dir.mkdir(parents=True, exist_ok=True)

    def sync_to_drive(self, tag: str | None = None) -> Path | None:
        """Mirror active target constants.py, runs/, and backups/ to drive_backup_dir if set."""
        backup_root = getattr(self, "drive_backup_dir", None)
        if not backup_root:
            return None
        backup_root = Path(backup_root)
        target_path = getattr(self, "target_constants_path", CONSTANTS_PATH)
        target_engine = getattr(self, "target_engine", "classical")
        dst_run_dir = backup_root / self.runs_dir.name
        dst_backups_dir = dst_run_dir / "backups"
        try:
            backup_root.mkdir(parents=True, exist_ok=True)
            dst_run_dir.mkdir(parents=True, exist_ok=True)
            dst_backups_dir.mkdir(parents=True, exist_ok=True)

            if target_path.is_file():
                const_text = target_path.read_text(encoding="utf-8").replace("\r\n", "\n")
                for const_dst in (
                    backup_root / f"{target_engine}_constants.py",
                    dst_run_dir / f"{target_engine}_constants.py",
                ):
                    with const_dst.open("w", encoding="utf-8", newline="\n") as fh:
                        fh.write(const_text)

            for src_dir, dst_dir in (
                (self.runs_dir, dst_run_dir),
                (self.backups_dir, dst_backups_dir),
            ):
                if not src_dir.is_dir():
                    continue
                for src_file in src_dir.glob("*.py"):
                    dst_file = dst_dir / src_file.name
                    if (
                        not dst_file.is_file()
                        or dst_file.stat().st_size != src_file.stat().st_size
                        or src_file.stat().st_mtime > dst_file.stat().st_mtime + 0.5
                    ):
                        shutil.copy2(src_file, dst_file)

            tag_msg = f" ({tag})" if tag else ""
            print(f"[Drive Backup]{tag_msg} Synced -> {dst_run_dir}")
            return dst_run_dir
        except Exception as exc:
            print(f"[WARN] Drive backup skipped ({exc})")
            return None

    def _load_sf_sidecar_if_needed(self) -> None:
        if self._sf_cp is not None:
            return
        if not self.sf_labels_path or not self.sf_labels_path.is_file():
            cmd_hint = (
                f"python -m tuner.data_pipeline.annotate_sf_labels "
                f"--dataset {_display_path(self.train_path)} --nodes 4000 --skip-holdout"
            )
            raise FileNotFoundError(
                f"SF19 sidecar labels not found: {self.sf_labels_path}\n"
                f"Generate it first from repo root:\n  {cmd_hint}"
            )
        print(f"      [SF19] Loading sidecar search scores: {self.sf_labels_path.name}")
        with np.load(self.sf_labels_path) as z:
            raw_cp = np.asarray(z["sf_cp"], dtype=np.int16)
            raw_d1 = (
                np.asarray(z["sf_d1_cp"], dtype=np.int16)
                if "sf_d1_cp" in z.files
                else None
            )
        if self._keep_idx is not None and len(raw_cp) == self._n_full:
            self._sf_cp = raw_cp[self._keep_idx]
            self._sf_d1_cp = raw_d1[self._keep_idx] if raw_d1 is not None else None
        elif len(raw_cp) == len(self._raw_results_wdl):
            self._sf_cp = raw_cp
            self._sf_d1_cp = raw_d1
        else:
            raise ValueError(
                f"SF sidecar row count ({len(raw_cp):,}) does not match dataset "
                f"({self._n_full:,} full / {len(self._raw_results_wdl):,} loaded)."
            )
        gc.collect()

    def _apply_label_mode(self) -> None:
        """Configure active train/dev labels (pure WDL vs SF19+WDL blend) while keeping holdout pure WDL."""
        if not self.use_sf_labels or self.sf_lambda <= 0.0:
            self.train_data["results"] = self._raw_results_wdl.copy()
            self.train_idx = self._base_train_idx.copy()
            self.dev_idx = (
                self._base_dev_idx.copy() if self._base_dev_idx is not None else None
            )
            self.sf_blend_stats = None
            return

        self._load_sf_sidecar_if_needed()
        assert self._sf_cp is not None
        blended, stats = build_blended_labels(
            self._raw_results_wdl,
            self._sf_cp,
            self.sf_lambda,
            sf_d1_cp=self._sf_d1_cp,
            max_tactical_gap=self.sf_max_tactical_gap,
            fit_indices=self._base_train_idx,
        )
        ann = stats["annotated_mask"]

        # Retain all annotated rows (including tactical swings, which fall back to pure WDL
        # in `blended`) so train_idx and dev_idx share the exact same distribution and avoid
        # survivorship bias against breakthrough features.
        act_train = self._base_train_idx[ann[self._base_train_idx]]
        if len(act_train) == 0:
            raise RuntimeError(
                "0 annotated training rows available for SF mixed-label mode. "
                "Run annotate_sf_labels on more rows before enabling use_sf_labels."
            )
        self.train_idx = act_train

        if self._base_dev_idx is not None:
            act_dev = self._base_dev_idx[ann[self._base_dev_idx]]
            self.dev_idx = act_dev if len(act_dev) > 0 else None
        else:
            self.dev_idx = None

        # Holdout split strictly retains pure WDL labels as an invariant evaluation ruler
        if self.holdout_idx is not None and len(self.holdout_idx) > 0:
            blended[self.holdout_idx] = self._raw_results_wdl[self.holdout_idx]

        self.train_data["results"] = blended
        self.sf_blend_stats = {
            k: v
            for k, v in stats.items()
            if not isinstance(v, np.ndarray)
        }
        print(
            f"      [SF19 Blend] lambda={self.sf_lambda:.2f} | calibrated K_sf={stats['k_sf']:.3f} "
            f"(SF-vs-WDL MSE={stats['sf_vs_wdl_mse']:.6f}) | "
            f"active_train={len(self.train_idx):,}/{len(self._base_train_idx):,} "
            f"(tactical_skipped={stats['n_tactical_filtered']:,}) | "
            f"active_dev={len(self.dev_idx) if self.dev_idx is not None else 0:,} | "
            f"holdout=pure_WDL"
        )

    def _holdout_eval_view(self) -> dict:
        """Shallow view of ``train_data`` guaranteed to expose pure WDL ``results``."""
        return {
            "label": self.train_data["label"],
            "piece_bbs": self.train_data["piece_bbs"],
            "occupancy_bbs": self.train_data["occupancy_bbs"],
            "game_states": self.train_data["game_states"],
            "results": self._raw_results_wdl,
        }

    def _load_pgn_data_if_needed(self) -> None:
        if self.pgn_data is not None:
            return
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

    def _load_zurichess_data_if_needed(self) -> None:
        if getattr(self, "zurichess_data", None) is not None:
            return
        if self.zurichess_val_path and self.zurichess_val_path.is_file():
            print(
                f"[2b/4] Loading Zurichess quiet EPD tie-breaker set: {self.zurichess_val_path.name}"
            )
            with np.load(self.zurichess_val_path) as npz:
                self.zurichess_data = {
                    "label": self.zurichess_val_path.name,
                    "piece_bbs": npz["piece_bbs"],
                    "occupancy_bbs": npz["occupancy_bbs"],
                    "game_states": npz["game_states"],
                    "results": npz["results"],
                }
            gc.collect()
            n_z = len(self.zurichess_data["results"])
            n_castle = int(np.count_nonzero(self.zurichess_data["game_states"][:, 1] != 0))
            print(
                f"       Loaded {n_z:,} Zurichess positions "
                f"(castling!=0: {n_castle:,} [{n_castle / max(n_z, 1):.2%}], "
                f"RAM now: {_ram_summary_gb()})"
            )
        else:
            print(f"[2b/4] Zurichess validation file not found ({self.zurichess_val_path}); skipping.")

    def reconfigure(
        self,
        *,
        use_pgn_val: bool | None = None,
        use_zurichess_val: bool | None = None,
        zurichess_val_npz: str | Path | None = None,
        target_engine: str | None = None,
        pgn_val_npz: str | Path | None = None,
        target_constants_path: str | Path | None = None,
        use_sf_labels: bool | None = None,
        sf_lambda: float | None = None,
        sf_labels_path: str | Path | None = None,
        sf_max_tactical_gap: int | None = None,
        drive_backup_dir: str | Path | None = None,
        random_seed: bool | None = None,
    ) -> None:
        """Switch validation mode, SF19 label blend, or target engine in-place without reloading 20M rows."""
        t0 = time.time()
        if random_seed is not None:
            self.random_seed = bool(random_seed)
        if drive_backup_dir is not None:
            self.drive_backup_dir = Path(drive_backup_dir) if drive_backup_dir else None
        if pgn_val_npz is not None:
            new_pgn_path = Path(pgn_val_npz) if pgn_val_npz else None
            if new_pgn_path != self.pgn_val_path:
                self.pgn_val_path = new_pgn_path
                self.pgn_data = None
        if zurichess_val_npz is not None:
            new_z_path = Path(zurichess_val_npz) if zurichess_val_npz else None
            if new_z_path != getattr(self, "zurichess_val_path", None):
                self.zurichess_val_path = new_z_path
                self.zurichess_data = None
        if use_pgn_val is not None:
            self.use_pgn_val = bool(use_pgn_val)
        if use_zurichess_val is not None:
            self.use_zurichess_val = bool(use_zurichess_val)

        need_label_refresh = False
        if sf_labels_path is not None:
            new_sf_path = Path(sf_labels_path)
            if new_sf_path != getattr(self, "sf_labels_path", None):
                self.sf_labels_path = new_sf_path
                self._sf_cp = None
                self._sf_d1_cp = None
                need_label_refresh = True
        if use_sf_labels is not None and bool(use_sf_labels) != getattr(self, "use_sf_labels", False):
            self.use_sf_labels = bool(use_sf_labels)
            need_label_refresh = True
        if sf_lambda is not None and float(sf_lambda) != getattr(self, "sf_lambda", 0.25):
            self.sf_lambda = float(sf_lambda)
            need_label_refresh = True
        if sf_max_tactical_gap is not None and int(sf_max_tactical_gap) != getattr(
            self, "sf_max_tactical_gap", 180
        ):
            self.sf_max_tactical_gap = int(sf_max_tactical_gap)
            need_label_refresh = True

        if target_engine is not None or target_constants_path is not None:
            (
                self.target_engine,
                self.target_constants_path,
                self.target_cache_dir,
                self._custom_constants_override,
            ) = _resolve_target_engine(
                target_engine if target_engine is not None else self.target_engine,
                target_constants_path,
            )
            if not self.target_constants_path.is_file():
                raise FileNotFoundError(
                    f"Missing target constants file: {self.target_constants_path}"
                )

        if need_label_refresh:
            self._apply_label_mode()

        self._update_run_directories()
        if self.use_pgn_val:
            self._load_pgn_data_if_needed()
        if self.use_zurichess_val:
            self._load_zurichess_data_if_needed()

        self._reload_parameter_manager(clear_target_cache=False)
        self.initial_theta0 = self.step_baseline_theta.copy()
        self._refresh_baseline_score_caches()
        self._initial_dev_k_star = self._step_dev_k_star
        self._initial_holdout_k_star = self._step_holdout_k_star
        self._initial_pgn_k_star = self._step_pgn_k_star
        self._initial_zurichess_k_star = self._step_zurichess_k_star
        self._initial_holdout_scores = (
            self._step_holdout_scores.copy()
            if self._step_holdout_scores is not None
            else None
        )
        self._initial_pgn_scores = (
            self._step_pgn_scores.copy() if self._step_pgn_scores is not None else None
        )
        self._initial_zurichess_scores = (
            self._step_zurichess_scores.copy()
            if self._step_zurichess_scores is not None
            else None
        )
        self.history = {}
        self.backup_stack = []
        lbl_desc = (
            f"SF19+WDL mix (lambda={self.sf_lambda:.2f}, K_sf={self.sf_blend_stats['k_sf']:.3f})"
            if (self.use_sf_labels and self.sf_lambda > 0.0 and self.sf_blend_stats)
            else "pure WDL"
        )
        print(
            f"Reconfigured session in {time.time() - t0:.1f}s (reused preloaded {self.train_path.name}):\n"
            f"  target_engine = {self.target_engine!r} -> {_display_path(self.target_constants_path)}\n"
            f"  use_pgn_val   = {self.use_pgn_val} | use_zurichess_val = {self.use_zurichess_val}\n"
            f"  train_labels  = {lbl_desc} | dev_K*={self._step_dev_k_star:.3f} | "
            f"holdout_K*={self._step_holdout_k_star:.3f}\n"
            f"  runs_dir      = {_display_path(self.runs_dir)}"
        )

    def _reload_parameter_manager(self, *, clear_target_cache: bool = True) -> None:
        target_engine = getattr(self, "target_engine", "classical")
        target_constants_path = getattr(self, "target_constants_path", CONSTANTS_PATH)
        custom_override = getattr(self, "_custom_constants_override", False)
        target_cache_dir = getattr(self, "target_cache_dir", None)

        if target_engine == "classical" and not custom_override:
            sync_constants_snapshot.main()
            importlib.reload(constants_snapshot)
        elif clear_target_cache and target_cache_dir is not None:
            removed = sync_constants_snapshot.clear_classical_numba_cache(target_cache_dir)
            print(
                f"Updated {_display_path(target_constants_path)}; "
                f"removed {removed} stale {target_engine} Numba cache file(s)"
            )

        importlib.reload(parameters_mod)
        self.pm = parameters_mod.ParameterManager(constants_source=target_constants_path)
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
        raw_wdl = getattr(self, "_raw_results_wdl", self.train_data["results"])
        if self.holdout_idx is not None and len(self.holdout_idx) > 0:
            self._step_holdout_scores = self._eval_scores_chunked(
                self.train_data, self.step_baseline_theta, self.holdout_idx
            )
            h_wdl = raw_wdl[self.holdout_idx]
        else:
            self._step_holdout_scores = self._eval_scores_chunked(
                self.train_data, self.step_baseline_theta, None
            )
            h_wdl = raw_wdl
        self._step_holdout_k_star, _ = fit_best_k(self._step_holdout_scores, h_wdl)

        # Fit SPSA/grid k_factor strictly on the 1M dev split (zero holdout leakage)
        if self.dev_idx is not None and len(self.dev_idx) > 0:
            dev_scores = self._eval_scores_chunked(
                self.train_data, self.step_baseline_theta, self.dev_idx
            )
            dev_labels = self.train_data["results"][self.dev_idx]
            self._step_dev_k_star, _ = fit_best_k(dev_scores, dev_labels)
        else:
            self._step_dev_k_star = self._step_holdout_k_star

        if getattr(self, "use_pgn_val", True) and self.pgn_data is not None:
            self._step_pgn_scores = self._eval_scores_chunked(
                self.pgn_data, self.step_baseline_theta, None
            )
            self._step_pgn_k_star, _ = (
                fit_best_k(self._step_pgn_scores, self.pgn_data["results"])
                if len(self._step_pgn_scores)
                else (1.0, 0.0)
            )
        else:
            self._step_pgn_scores = None
            self._step_pgn_k_star = 1.0

        if getattr(self, "use_zurichess_val", False) and getattr(self, "zurichess_data", None) is not None:
            self._step_zurichess_scores = self._eval_scores_chunked(
                self.zurichess_data, self.step_baseline_theta, None
            )
            self._step_zurichess_k_star, _ = (
                fit_best_k(self._step_zurichess_scores, self.zurichess_data["results"])
                if len(self._step_zurichess_scores)
                else (1.0, 0.0)
            )
        else:
            self._step_zurichess_scores = None
            self._step_zurichess_k_star = 1.0

    def _resolve_k_factor(self, k_factor: float | str | None) -> float:
        if k_factor is None or (isinstance(k_factor, str) and k_factor.strip().lower() == "auto"):
            return float(getattr(self, "_step_dev_k_star", 1.0))
        return float(k_factor)

    def baseline_report(self) -> dict:
        """Print baseline MSE & optimal K* on unseen holdout and (optional) PGN / Zurichess validation."""
        use_pgn = getattr(self, "use_pgn_val", True)
        use_zuri = getattr(self, "use_zurichess_val", False)
        target_engine = getattr(self, "target_engine", "classical")
        target_path = getattr(self, "target_constants_path", CONSTANTS_PATH)
        use_sf = getattr(self, "use_sf_labels", False) and getattr(self, "sf_lambda", 0.0) > 0.0
        report: dict = {
            "target_engine": target_engine,
            "target_constants_path": str(target_path),
            "use_pgn_val": use_pgn,
            "use_zurichess_val": use_zuri,
            "use_sf_labels": use_sf,
            "sf_lambda": getattr(self, "sf_lambda", 0.0) if use_sf else 0.0,
            "auto_k_factor": float(getattr(self, "_step_dev_k_star", 1.0)),
            "sf_blend_stats": getattr(self, "sf_blend_stats", None),
        }
        lbl_str = (
            f"SF19+WDL(lambda={self.sf_lambda:.2f})"
            if use_sf
            else "pure_WDL"
        )
        raw_wdl = getattr(self, "_raw_results_wdl", self.train_data["results"])
        h_idx = (
            self.holdout_idx
            if (self.holdout_idx is not None and len(self.holdout_idx) > 0)
            else np.arange(len(raw_wdl), dtype=np.int64)
        )
        h_res = raw_wdl[h_idx]
        h_k_star, h_mse_kstar = fit_best_k(self._step_holdout_scores, h_res)
        self._step_holdout_k_star = h_k_star
        report["holdout_k_factor"] = h_k_star
        print(
            f"=== Current Step Baseline Health Report "
            f"[target={target_engine} -> {_display_path(target_path)} | "
            f"use_pgn_val={use_pgn} | train={lbl_str} | "
            f"dev_K*={report['auto_k_factor']:.3f} | hold_K*={h_k_star:.3f}] ==="
        )
        print(
            f"  [1] {self.train_path.name} unseen holdout [pure WDL] (n={len(h_idx):,}):\n"
            f"      MSE @ K*={h_k_star:.3f}: {h_mse_kstar:.7f}"
        )
        report["holdout"] = {
            "n": int(len(h_idx)),
            "k_star": h_k_star,
            "mse_k_star": h_mse_kstar,
        }

        if use_pgn and self.pgn_data is not None and self._step_pgn_scores is not None:
            p_res = self.pgn_data["results"]
            p_k_star, p_mse_kstar = fit_best_k(self._step_pgn_scores, p_res)
            self._step_pgn_k_star = p_k_star
            castle_mask = self.pgn_data["game_states"][:, 1] != 0
            p_castle_mse = (
                float(
                    np.mean(
                        squared_errors_from_scores(
                            self._step_pgn_scores[castle_mask], p_res[castle_mask], p_k_star
                        )
                    )
                )
                if np.any(castle_mask)
                else float("nan")
            )
            print(
                f"  [2] {self.pgn_val_path.name} independent PGN val (n={len(p_res):,}):\n"
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
                "k_star": p_k_star,
                "mse_k_star": p_mse_kstar,
            }
        elif not use_pgn:
            print("  [2] Independent PGN val: disabled (use_pgn_val=False)")

        if (
            use_zuri
            and getattr(self, "zurichess_data", None) is not None
            and self._step_zurichess_scores is not None
        ):
            z_res = self.zurichess_data["results"]
            z_k_star, z_mse_kstar = fit_best_k(self._step_zurichess_scores, z_res)
            self._step_zurichess_k_star = z_k_star
            print(
                f"  [3] {self.zurichess_val_path.name} Zurichess quiet EPD (n={len(z_res):,}):\n"
                f"      Static Eval MSE @ K*={z_k_star:.3f}: {z_mse_kstar:.7f}"
            )
            report["zurichess_val"] = {
                "n": int(len(z_res)),
                "k_star": z_k_star,
                "mse_k_star": z_mse_kstar,
            }
        return report

    @staticmethod
    def _check_subbucket_guard(
        buckets: dict[str, dict],
        bucket_name: str,
        *,
        dataset_label: str,
        severe_z: float,
        severe_delta: float,
        warn_z: float,
        warn_delta: float,
        reasons: list[str],
    ) -> bool:
        """Check one validation sub-bucket for severe collapse or mild regression."""
        b = buckets.get(bucket_name)
        if not b or b["n"] <= 0:
            return False
        if b["z"] > severe_z and b["delta"] > severe_delta:
            reasons.append(
                f"{dataset_label} '{bucket_name}' collapsed "
                f"(delta={b['delta']:+.2e}, z={b['z']:+.2f})"
            )
            return True
        if b["z"] > warn_z and b["delta"] > warn_delta:
            reasons.append(
                f"{dataset_label} '{bucket_name}' regressed "
                f"(delta={b['delta']:+.2e}, z={b['z']:+.2f})"
            )
        return False

    @classmethod
    def _eval_independent_gate(
        cls,
        dataset_label: str,
        res: dict,
        *,
        castling_sensitive: bool,
        reasons: list[str],
    ) -> bool:
        """Evaluate an independent PGN/Zurichess validation report at calibrated K*."""
        severe = False
        kstar = res.get("fitted_k") or res["buckets"]["all"]
        if kstar["z"] > 2.5:
            severe = True
            reasons.append(
                f"{dataset_label} @ K* worsened significantly (z={kstar['z']:+.2f})"
            )
        elif kstar["z"] > 1.2:
            reasons.append(
                f"{dataset_label} @ K* mildly positive ({kstar['delta']:+.2e}, z={kstar['z']:+.2f})"
            )

        # PGN positions within a game share opening castling state (~10 plies/game),
        # inflating naive i.i.d. z-scores on small noise blips.
        z_warn = 2.5 if castling_sensitive else 3.5
        d_warn = 1.5e-5 if castling_sensitive else 3.0e-5
        if cls._check_subbucket_guard(
            res["buckets"],
            "castling!=0",
            dataset_label=dataset_label,
            severe_z=4.0,
            severe_delta=2.0e-5,
            warn_z=z_warn,
            warn_delta=d_warn,
            reasons=reasons,
        ):
            severe = True
        if cls._check_subbucket_guard(
            res["buckets"],
            "broad-EG phase<=64",
            dataset_label=dataset_label,
            severe_z=3.0,
            severe_delta=1.5e-5,
            warn_z=2.0,
            warn_delta=1.0e-5,
            reasons=reasons,
        ):
            severe = True
        if cls._check_subbucket_guard(
            res["buckets"],
            "pure king+pawns",
            dataset_label=dataset_label,
            severe_z=4.0,
            severe_delta=2.0e-5,
            warn_z=2.2,
            warn_delta=1.0e-5,
            reasons=reasons,
        ):
            severe = True
        return severe

    def _gate_summary(
        self,
        changes: list[dict],
        hold_res: dict,
        pgn_res: dict | None,
        zurichess_res: dict | None = None,
    ) -> str:
        if not changes:
            verdict = "NO-OP (0 integer changes)"
            print(f"\n>>> GATE VERDICT: {verdict}")
            return verdict

        use_pgn = getattr(self, "use_pgn_val", True)
        target_engine = getattr(self, "target_engine", "classical")
        hold_name = self.train_path.name
        h_all = hold_res["buckets"]["all"]
        h_kstar = hold_res.get("fitted_k") or h_all
        h_eg = hold_res["buckets"]["broad-EG phase<=64"]
        reasons: list[str] = []
        severe = False

        # Judge holdout strictly at its own pure-WDL baseline-calibrated K*.
        if h_kstar["delta"] >= 0.0:
            severe = True
            reasons.append(
                f"{hold_name} holdout @ K* worsened (delta={h_kstar['delta']:+.2e}, z={h_kstar['z']:+.2f})"
            )
        elif h_kstar["z"] > 2.0:
            severe = True
            reasons.append(
                f"{hold_name} holdout @ K* regressed (z={h_kstar['z']:+.2f})"
            )
        elif h_all["z"] > 2.0:
            severe = True
            reasons.append(
                f"{hold_name} holdout regressed vs baseline scale "
                f"(z={h_all['z']:+.2f}, z@K*={h_kstar['z']:+.2f})"
            )
        elif h_kstar["z"] > 1.0 or h_all["z"] > 1.0:
            reasons.append(
                f"{hold_name} holdout slightly positive at calibrated scale "
                f"(z={h_all['z']:+.2f}, z@K*={h_kstar['z']:+.2f})"
            )

        if h_eg["n"] > 0 and h_eg["z"] > 1.8:
            severe = True
            reasons.append(f"{hold_name} holdout 'broad-EG' regressed (z={h_eg['z']:+.2f})")
        elif h_eg["n"] > 0 and h_eg["z"] > 1.0:
            reasons.append(f"{hold_name} holdout 'broad-EG' mildly positive (z={h_eg['z']:+.2f})")

        # Catch localized late-endgame collapse masked by aggregate broad-EG phases.
        if self._check_subbucket_guard(
            hold_res["buckets"],
            "pure king+pawns",
            dataset_label=f"{hold_name} holdout",
            severe_z=3.5,
            severe_delta=2.0e-5,
            warn_z=2.0,
            warn_delta=1.0e-5,
            reasons=reasons,
        ):
            severe = True
        if self._check_subbucket_guard(
            hold_res["buckets"],
            "phase=0",
            dataset_label=f"{hold_name} holdout",
            severe_z=3.5,
            severe_delta=1.0e-5,
            warn_z=2.0,
            warn_delta=5.0e-6,
            reasons=reasons,
        ):
            severe = True

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
                "KING_DANGER_OUT_MG_NUM",
                "KING_DANGER_OUT_EG_NUM",
                "EG_SAFETY_SCALE",
                "KING_FLANK_ATTACK_NUM",
                "KING_FLANK_DEFENSE_MULT",
                "KING_SHELTER_FEEDBACK_NUM",
            }
            for ch in changes
        )

        if pgn_res is not None:
            if self._eval_independent_gate(
                "val_pgn", pgn_res, castling_sensitive=castling_sensitive, reasons=reasons
            ):
                severe = True

        if zurichess_res is not None:
            if self._eval_independent_gate(
                "val_zurichess",
                zurichess_res,
                castling_sensitive=castling_sensitive,
                reasons=reasons,
            ):
                severe = True

        if severe:
            verdict = "REJECT"
        elif reasons:
            verdict = "CAUTION"
        else:
            verdict = "PASS"

        hold_kp = hold_res["buckets"].get("pure king+pawns")
        kp_txt = f" | K+P z={hold_kp['z']:+.2f}" if (hold_kp and hold_kp["n"] > 0) else ""
        hold_k_tag = (
            f"K={hold_res['k_factor']:.3f}" if hold_res.get("k_factor") is not None else "K*"
        )
        print("\n" + "=" * 72)
        print(
            f">>> GATE VERDICT: {verdict}  "
            f"({len(changes)} parameter group(s) changed | target={target_engine}, use_pgn_val={use_pgn})"
        )
        print(
            f"    unseen holdout   : delta@{hold_k_tag}={h_kstar['delta']:+.4e} "
            f"(z={h_kstar['z']:+.2f}) | broad-EG z={h_eg['z']:+.2f}{kp_txt}"
        )
        if pgn_res is not None:
            pk = pgn_res.get("fitted_k") or pgn_res["buckets"]["all"]
            cb = pgn_res["buckets"].get("castling!=0")
            castle_txt = (
                f" | castling!=0 z={cb['z']:+.2f}" if (cb and cb["n"] > 0) else ""
            )
            pgn_k_tag = (
                f"K={pgn_res['k_factor']:.3f}" if pgn_res.get("k_factor") is not None else "K*"
            )
            print(
                f"    PGN independent  : delta@{pgn_k_tag}={pk['delta']:+.4e} "
                f"(z={pk['z']:+.2f}){castle_txt}"
            )
        elif not use_pgn:
            print("    PGN independent  : skipped (use_pgn_val=False)")
        if zurichess_res is not None:
            zk = zurichess_res.get("fitted_k") or zurichess_res["buckets"]["all"]
            z_k_tag = (
                f"K={zurichess_res['k_factor']:.3f}"
                if zurichess_res.get("k_factor") is not None
                else "K*"
            )
            print(
                f"    Zurichess quiet  : delta@{z_k_tag}={zk['delta']:+.4e} (z={zk['z']:+.2f})"
            )
        for r in reasons:
            print(f"    ! Note: {r}")
        print("=" * 72)
        return verdict

    def validate_candidate_theta(
        self,
        block_id: str,
        candidate: np.ndarray,
        out_path: Path | None = None,
        seed: int | None = None,
    ) -> dict:
        changes = print_effective_integer_changes(
            self.pm, self.step_baseline_theta, candidate
        )
        print()
        hold_view = self._holdout_eval_view()
        hold_indices = (
            self.holdout_idx
            if (self.holdout_idx is not None and len(self.holdout_idx) > 0)
            else np.arange(len(hold_view["results"]), dtype=np.int64)
        )
        hold_k = float(getattr(self, "_step_holdout_k_star", 1.0))
        hold_res = run_paired_validation(
            dataset=hold_view,
            baseline=self.step_baseline_theta,
            candidate=candidate,
            pm=self.pm,
            label=f"{self.train_path.name} [unseen holdout, pure WDL]",
            chunk_size=self.chunk_size,
            k_factor=hold_k,
            fit_k=False,
            cached_base_scores=self._step_holdout_scores,
            indices_override=hold_indices,
            verbose=True,
        )

        pgn_res = None
        if getattr(self, "use_pgn_val", True) and self.pgn_data is not None:
            pgn_k = float(getattr(self, "_step_pgn_k_star", 1.0))
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
                k_factor=pgn_k,
                fit_k=False,
                cached_base_scores=self._step_pgn_scores,
                indices_override=np.arange(len(self.pgn_data["results"]), dtype=np.int64),
                verbose=True,
            )

        zurichess_res = None
        if getattr(self, "use_zurichess_val", False) and getattr(self, "zurichess_data", None) is not None:
            z_k = float(getattr(self, "_step_zurichess_k_star", 1.0))
            print()
            zurichess_res = run_paired_validation(
                dataset=self.zurichess_data,
                baseline=self.step_baseline_theta,
                candidate=candidate,
                pm=self.pm,
                label=f"{self.zurichess_val_path.name} [Zurichess quiet EPD]",
                split="all",
                sample=0,
                chunk_size=self.chunk_size,
                k_factor=z_k,
                fit_k=False,
                cached_base_scores=self._step_zurichess_scores,
                indices_override=np.arange(len(self.zurichess_data["results"]), dtype=np.int64),
                verbose=True,
            )

        verdict = self._gate_summary(changes, hold_res, pgn_res, zurichess_res)
        record = {
            "block_id": block_id,
            "path": str(out_path) if out_path else None,
            "seed": seed,
            "verdict": verdict,
            "target_engine": getattr(self, "target_engine", "classical"),
            "use_pgn_val": getattr(self, "use_pgn_val", True),
            "use_sf_labels": getattr(self, "use_sf_labels", False),
            "sf_lambda": getattr(self, "sf_lambda", 0.0),
            "k_factor": hold_k,
            "changes": changes,
            "holdout": hold_res,
            "pgn_val": pgn_res,
            "zurichess_val": zurichess_res,
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
        A: float = 100.0,
        batch_size: int = 65536,
        report_every: int = 50,
        patience: int = 25,
        metrics_sample: int = 250_000,
        seed: int | str | None = 0,
        k_factor: float | str = "auto",
        auto_scale: bool = True,
        verbose_buckets: str = "endpoints",
    ) -> dict:
        """Run SPSA on one block in-memory and validate on unseen sets."""
        if (
            (getattr(self, "random_seed", False) and (seed is None or seed == 0 or seed == "random" or seed == "time"))
            or seed is None
            or seed == "random"
            or seed == "time"
        ):
            eff_seed = int((time.time() * 1000) % 2_147_483_647)
            if eff_seed == 0:
                eff_seed = 1
            print(f"  [SPSA Seed] Using dynamic time seed: {eff_seed}")
        else:
            eff_seed = int(seed)

        eff_k = self._resolve_k_factor(k_factor)
        out_path = self.runs_dir / f"{block_id}.py"
        lbl_tag = (
            f"SF_mix(lambda={self.sf_lambda:.2f})"
            if (getattr(self, "use_sf_labels", False) and getattr(self, "sf_lambda", 0.0) > 0.0)
            else "WDL"
        )
        print(
            f"=== Running Block [{block_id}] -> {_display_path(out_path)} "
            f"(K={eff_k:.3f}, labels={lbl_tag}) ==="
        )
        optimizer = SPSAOptimizer(self.train_data, self.pm)
        optimizer.optimize(
            iterations=iters,
            alpha=alpha,
            c=c,
            A=A,
            include_list=tune,
            exclude_list=exclude,
            batch_size=batch_size,
            eval_sample=self.eval_sample,
            holdout_sample=self.holdout_sample,
            metrics_sample=metrics_sample,
            seed=eff_seed,
            report_every=report_every,
            patience=patience,
            king_pst_only=king_pst_only,
            non_king_pst_only=non_king_pst_only,
            pst_pieces=pst_pieces,
            save_path=str(out_path),
            k_factor=eff_k,
            train_indices_override=self.train_idx,
            eval_indices_override=self.dev_idx,
            c_floor=c_floor,
            auto_scale=auto_scale,
            verbose_buckets=verbose_buckets,
        )
        val_header = (
            "Dual Unseen Validation"
            if (getattr(self, "use_pgn_val", True) and self.pgn_data is not None)
            else "Unseen Holdout Validation"
        )
        print(f"\n--- {val_header} for [{block_id}] ---")
        res = self.validate_candidate_theta(
            block_id, optimizer.best_theta.copy(), out_path=out_path, seed=eff_seed
        )
        self.sync_to_drive(tag=f"{block_id}_candidate")
        return res

    def scan_grid(
        self,
        block_id: str,
        grids: list[str],
        *,
        split: str = "dev",
        eval_sample: int = 1_000_000,
        k_factor: float | str = "auto",
        top: int = 10,
    ) -> dict:
        """Exhaustively scan discrete integer scalars on dev, then verify on unseen sets."""
        eff_k = self._resolve_k_factor(k_factor)
        out_path = self.runs_dir / f"{block_id}.py"
        print(
            f"=== Discrete Grid Scan [{block_id}] -> {_display_path(out_path)} "
            f"(K={eff_k:.3f}) ==="
        )
        scan = run_scalar_grid_scan(
            dataset=self.train_data,
            grid_specs=grids,
            pm=self.pm,
            baseline=self.step_baseline_theta,
            split=split,
            eval_sample=eval_sample,
            holdout_sample=self.holdout_sample,
            k_factor=eff_k,
            top=top,
            verbose=True,
            indices_override=(self.dev_idx if split == "dev" else None),
        )
        best_theta = scan["best_theta"]
        content = self.pm.update_constants_string(best_theta)
        with out_path.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(content)
        print(f"\nBest discrete grid point on {split}: {scan['best_values']}")
        val_header = (
            "Dual Unseen Validation"
            if (getattr(self, "use_pgn_val", True) and self.pgn_data is not None)
            else "Unseen Holdout Validation"
        )
        print(f"--- {val_header} for [{block_id}] ---")
        res = self.validate_candidate_theta(block_id, best_theta, out_path=out_path)
        res["grid_best_values"] = scan["best_values"]
        self.sync_to_drive(tag=f"{block_id}_candidate")
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

    def accept(
        self,
        block_id: str,
        only: list[str] | None = None,
        *,
        require_pass: bool | None = None,
    ) -> list[str]:
        """Backup target constants.py, apply candidate via AST-safe merge, sync snapshot & caches.

        If require_pass is True (or self.require_pass_to_accept is True),
        only applies if the gate verdict for block_id in self.history was 'PASS'.
        """
        target_path = getattr(self, "target_constants_path", CONSTANTS_PATH)
        target_engine = getattr(self, "target_engine", "classical")
        target_rel = _display_path(target_path)
        must_pass = (
            require_pass
            if require_pass is not None
            else getattr(self, "require_pass_to_accept", False)
        )
        if must_pass:
            rec = self.history.get(block_id)
            if rec is None:
                print(f"\n[WARN] [{block_id}] 查無驗證紀錄 (可能尚未執行) -> 略過套用。")
                return []
            verdict = rec.get("verdict")
            if verdict != "PASS":
                print(
                    f"\n[SKIP] [{block_id}] Gate 判定為 '{verdict}' (非 PASS) -> "
                    f"自動跳過，保持既有 {target_rel} 參數不變。"
                )
                return []
            print(
                f"\n[PASS] [{block_id}] Gate 判定為 PASS -> "
                f"正在套用變更至 {target_rel} 並建立備份..."
            )

        cand_path = self.runs_dir / f"{block_id}.py"
        if not cand_path.is_file():
            raise FileNotFoundError(f"Missing candidate dump {cand_path}. Run block first.")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backups_dir / f"{ts}_{target_engine}_before_{block_id}_constants.py"
        shutil.copyfile(target_path, backup_path)
        self.backup_stack.append((block_id, backup_path))

        changed = apply_tuned_mod.apply_tuned(
            tuned_path=cand_path,
            target_path=target_path,
            dry_run=False,
            only=only,
        )
        if not changed:
            print(f"[{block_id}] Nothing changed; {target_rel} unchanged.")
            return []

        self._reload_parameter_manager(clear_target_cache=True)
        self._refresh_baseline_score_caches()
        print(
            f"[{block_id}] Accepted {len(changed)} parameter group(s) into {target_rel}. "
            f"Backup saved to {_display_path(backup_path)}."
        )
        self.sync_to_drive(tag=f"{block_id}_accept")
        return changed

    def accept_if_pass(self, block_id: str, only: list[str] | None = None) -> list[str]:
        """Accept candidate only if the gate verdict for block_id was PASS."""
        return self.accept(block_id, only=only, require_pass=True)

    def rollback(self, block_id: str | None = None) -> Path:
        """Restore target constants.py from backup and resync snapshot & in-memory baseline."""
        target_path = getattr(self, "target_constants_path", CONSTANTS_PATH)
        target_rel = _display_path(target_path)
        if not self.backup_stack:
            backups = sorted(self.backups_dir.glob("*_constants.py"))
            if not backups:
                raise RuntimeError(f"No backups available in {_display_path(self.backups_dir)}.")
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
        with target_path.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        self._reload_parameter_manager(clear_target_cache=True)
        self._refresh_baseline_score_caches()
        print(f"Rolled back {target_rel} using {_display_path(chosen)} (tag={label}).")
        self.sync_to_drive(tag=f"{label}_rollback")
        return chosen

    def cumulative_report(self) -> dict:
        """Compare current accepted target constants.py against initial session-start theta0."""
        use_pgn = getattr(self, "use_pgn_val", True)
        use_zuri = getattr(self, "use_zurichess_val", False)
        target_engine = getattr(self, "target_engine", "classical")
        target_path = getattr(self, "target_constants_path", CONSTANTS_PATH)
        print(
            f"=== Cumulative Progress vs Initial Session Baseline (theta0) "
            f"[target={target_engine} -> {_display_path(target_path)} | use_pgn_val={use_pgn}] ==="
        )
        changes = print_effective_integer_changes(
            self.pm, self.initial_theta0, self.step_baseline_theta
        )
        hold_view = self._holdout_eval_view()
        hold_indices = (
            self.holdout_idx
            if (self.holdout_idx is not None and len(self.holdout_idx) > 0)
            else np.arange(len(hold_view["results"]), dtype=np.int64)
        )
        init_k = float(
            getattr(
                self,
                "_initial_holdout_k_star",
                getattr(self, "_step_holdout_k_star", 1.0),
            )
        )
        print()
        hold_res = run_paired_validation(
            dataset=hold_view,
            baseline=self.initial_theta0,
            candidate=self.step_baseline_theta,
            pm=self.pm,
            label=f"{self.train_path.name} [cumulative vs start, pure WDL]",
            chunk_size=self.chunk_size,
            k_factor=init_k,
            fit_k=False,
            cached_base_scores=self._initial_holdout_scores,
            indices_override=hold_indices,
            verbose=True,
        )
        pgn_res = None
        if use_pgn and self.pgn_data is not None:
            init_pgn_k = float(
                getattr(self, "_initial_pgn_k_star", getattr(self, "_step_pgn_k_star", 1.0))
            )
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
                k_factor=init_pgn_k,
                fit_k=False,
                cached_base_scores=self._initial_pgn_scores,
                indices_override=np.arange(len(self.pgn_data["results"]), dtype=np.int64),
                verbose=True,
            )
        zurichess_res = None
        if use_zuri and getattr(self, "zurichess_data", None) is not None:
            init_z_k = float(
                getattr(
                    self,
                    "_initial_zurichess_k_star",
                    getattr(self, "_step_zurichess_k_star", 1.0),
                )
            )
            print()
            zurichess_res = run_paired_validation(
                dataset=self.zurichess_data,
                baseline=self.initial_theta0,
                candidate=self.step_baseline_theta,
                pm=self.pm,
                label=f"{self.zurichess_val_path.name} [cumulative vs start]",
                split="all",
                sample=0,
                chunk_size=self.chunk_size,
                k_factor=init_z_k,
                fit_k=False,
                cached_base_scores=self._initial_zurichess_scores,
                indices_override=np.arange(len(self.zurichess_data["results"]), dtype=np.int64),
                verbose=True,
            )
        verdict = self._gate_summary(changes, hold_res, pgn_res, zurichess_res)
        return {
            "verdict": verdict,
            "target_engine": target_engine,
            "use_pgn_val": use_pgn,
            "changes": changes,
            "holdout": hold_res,
            "pgn_val": pgn_res,
            "zurichess_val": zurichess_res,
        }
