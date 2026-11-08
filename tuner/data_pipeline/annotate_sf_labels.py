"""Annotate an HCE ``.npz`` dataset with Stockfish 19 search scores for mixed-label tuning.

Overview
--------
Given an existing HCE dataset (default: ``tuner/data/hce/dataset_epd_wdl_20m.npz``),
this script runs Stockfish 19 (UCI) at a fixed node budget (default ``--nodes 4000``)
across a multi-process worker pool and writes a lightweight sidecar archive
``<dataset_stem>.sf19.npz`` (~35 MB for 20M positions) containing:

- ``sf_cp`` (``int16[N]``): White-POV centipawn score from the **last completed exact**
  search iteration (ignoring partial aspiration ``upperbound`` / ``lowerbound`` lines).
  Unannotated rows hold ``UNANNOTATED_CP = -32768``.
- ``sf_d1_cp`` (``int16[N]``): White-POV exact ``depth 1`` score extracted from the same
  search at zero extra UCI round-trip cost, used to detect non-quiet tactical spikes
  (``|sf_cp - sf_d1_cp| > max_tactical_gap``) during mixed-label blending.

Why store raw ``sf_cp`` in a sidecar instead of overwriting ``results``:
1. Stockfish 19 UCI ``cp`` is already internally WDL-normalized (``+100 cp`` = 50% win
   rate in SF self-play). Storing raw ``sf_cp`` lets ``TexelTuningSession`` calibrate
   ``K_sf`` directly against the dataset's empirical ``results`` (WDL) via ``fit_best_k``
   and blend ``(1 - lambda) * wdl + lambda * sigmoid(K_sf * sf_cp)`` dynamically in RAM
   for any ``lambda`` without rewriting 674 MB of bitboards.
2. Crash-safe ``--resume`` (enabled by default): progress is checkpointed via memory-mapped
   ``.npy`` files every chunk. Pressing ``Ctrl+C`` saves a usable partial ``.sf19.npz``
   immediately; re-running continues from where it stopped.
3. Smart priority order: annotates the ``1M dev`` split first, then ``train`` (starting
   with the deterministic ``10M`` auto-RAM subset), and ``2M unseen holdout`` last (or
   skips holdout with ``--skip-holdout``, since holdout Gate evaluation uses pure WDL).

Usage (from repo root)::

    # Quick pilot: annotate 1M dev + 2M train (3M total) to test mixed labels quickly
    python -m tuner.data_pipeline.annotate_sf_labels --limit 3000000 --nodes 4000 --skip-holdout

    # Full train + dev annotation (resumes automatically from any prior partial run)
    python -m tuner.data_pipeline.annotate_sf_labels --nodes 4000 --skip-holdout
"""
from __future__ import annotations

import argparse
import atexit
from concurrent.futures import ThreadPoolExecutor
import gc
import os
from pathlib import Path
import subprocess
import threading
import time

import numpy as np

from tuner.tuner import split_dataset_indices

UNANNOTATED_CP = np.int16(-32768)
MATE_CP_BASE = 29000
MAX_NORMAL_CP = 27999

DEFAULT_DATASET = "tuner/data/hce/dataset_epd_wdl_20m.npz"
SF_CANDIDATES = (
    "external/stockfish/stockfish-windows-x86-64-avx2.exe",
    "external/stockfish/stockfish-windows-x86-64-universal.exe",
    "stockfish_repo/dist/stockfish/stockfish-windows-x86-64-avx2.exe",
    "external/stockfish_dir/stockfish-windows-x86-64-avx2.exe",
)

_PIECE_CHARS = "PNBRQKpnbrqk"
_FILES = "abcdefgh"
_RANKS = "12345678"


def _close_memmap(mm) -> None:
    """Explicitly release Windows file mapping handle so temp/ckpt files can be unlinked."""
    if mm is None:
        return
    raw_mmap = getattr(mm, "_mmap", None)
    if raw_mmap is not None:
        try:
            raw_mmap.close()
        except Exception:
            pass


def find_sf_binary(explicit_path: str | None = None) -> str:
    if explicit_path:
        if os.path.isfile(explicit_path):
            return explicit_path
        raise FileNotFoundError(f"Stockfish binary not found: {explicit_path}")
    for cand in SF_CANDIDATES:
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(
        "Stockfish executable not found in standard paths; pass --stockfish PATH"
    )


def default_sidecar_path(dataset_path: str | Path) -> Path:
    p = Path(dataset_path)
    return p.with_name(f"{p.stem}.sf19.npz")


def arrays_to_fen(piece_bb: np.ndarray, game_state: np.ndarray) -> str:
    """Convert ``(piece_bb[12], game_state[6])`` back into a standard 6-field FEN."""
    board_chars = [""] * 64
    for p_idx in range(12):
        bb = int(piece_bb[p_idx])
        ch = _PIECE_CHARS[p_idx]
        while bb:
            lsb = bb & -bb
            sq = lsb.bit_length() - 1
            board_chars[sq] = ch
            bb ^= lsb

    rank_strs: list[str] = []
    for rank in range(7, -1, -1):
        base = rank * 8
        empty = 0
        out_r: list[str] = []
        for file_idx in range(8):
            c = board_chars[base + file_idx]
            if c:
                if empty:
                    out_r.append(str(empty))
                    empty = 0
                out_r.append(c)
            else:
                empty += 1
        if empty:
            out_r.append(str(empty))
        rank_strs.append("".join(out_r))

    stm = "w" if int(game_state[0]) == 0 else "b"
    castle_bits = int(game_state[1])
    if castle_bits == 0:
        castle_str = "-"
    else:
        c_parts = []
        if castle_bits & 1:
            c_parts.append("K")
        if castle_bits & 2:
            c_parts.append("Q")
        if castle_bits & 4:
            c_parts.append("k")
        if castle_bits & 8:
            c_parts.append("q")
        castle_str = "".join(c_parts) or "-"

    ep_sq = int(game_state[2])
    if 0 <= ep_sq < 64:
        ep_str = f"{_FILES[ep_sq & 7]}{_RANKS[ep_sq >> 3]}"
    else:
        ep_str = "-"

    halfmove = int(game_state[3])
    return f"{'/'.join(rank_strs)} {stm} {castle_str} {ep_str} {halfmove} 1"


def _parse_uci_score_token(tokens: list[str], score_idx: int) -> tuple[int | None, bool]:
    """Parse ``score cp <v>`` or ``score mate <m>`` and whether it is an exact bound."""
    if score_idx + 2 >= len(tokens):
        return None, False
    kind = tokens[score_idx + 1]
    try:
        raw_val = int(tokens[score_idx + 2])
    except ValueError:
        return None, False

    tail = tokens[score_idx + 3 : score_idx + 5]
    is_bound = ("upperbound" in tail) or ("lowerbound" in tail)

    if kind == "cp":
        val = max(-MAX_NORMAL_CP, min(MAX_NORMAL_CP, raw_val))
    elif kind == "mate":
        mag = MATE_CP_BASE - min(abs(raw_val), 999)
        val = mag if raw_val > 0 else (-mag if raw_val < 0 else 0)
    else:
        return None, False
    return val, not is_bound


def parse_sf_search_output(lines: list[str], stm_is_black: bool) -> tuple[int, int]:
    """Extract ``(white_sf_cp, white_sf_d1_cp)`` from UCI lines of a single ``go`` search.

    Prefers the last **exact** ``info`` score (ignoring aspiration ``upperbound`` /
    ``lowerbound`` fail lines when a node cap stops mid-window) and captures exact
    ``depth 1`` score at the same time.
    """
    d1_stm_cp: int | None = None
    last_exact_stm_cp: int | None = None
    last_any_stm_cp: int | None = None

    for line in lines:
        if not line.startswith("info ") or " score " not in line:
            continue
        tokens = line.split()
        if "multipv" in tokens:
            mpv_i = tokens.index("multipv")
            if mpv_i + 1 < len(tokens) and tokens[mpv_i + 1] != "1":
                continue
        try:
            sc_i = tokens.index("score")
        except ValueError:
            continue
        val, is_exact = _parse_uci_score_token(tokens, sc_i)
        if val is None:
            continue

        last_any_stm_cp = val
        if is_exact:
            last_exact_stm_cp = val
            if d1_stm_cp is None and "depth" in tokens:
                d_i = tokens.index("depth")
                if d_i + 1 < len(tokens) and tokens[d_i + 1] == "1":
                    d1_stm_cp = val

    final_stm = last_exact_stm_cp if last_exact_stm_cp is not None else last_any_stm_cp
    if final_stm is None:
        return int(UNANNOTATED_CP), int(UNANNOTATED_CP)
    if d1_stm_cp is None:
        d1_stm_cp = final_stm

    sign = -1 if stm_is_black else 1
    return sign * final_stm, sign * d1_stm_cp


class _SfWorkerPoolState:
    """Thread-local Stockfish UCI subprocess manager (works in CLI, Jupyter, and Windows)."""

    def __init__(self, sf_path: str, hash_mb: int):
        self.sf_path = sf_path
        self.hash_mb = int(hash_mb)
        self._local = threading.local()
        self._lock = threading.Lock()
        self._all_procs: list[subprocess.Popen] = []

    def _spawn(self) -> subprocess.Popen:
        proc = subprocess.Popen(
            [self.sf_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(
            f"uci\nsetoption name Threads value 1\n"
            f"setoption name Hash value {self.hash_mb}\nisready\n"
        )
        proc.stdin.flush()
        while True:
            ln = proc.stdout.readline()
            if not ln:
                raise RuntimeError("Stockfish terminated during UCI handshake")
            if ln.strip() == "readyok":
                break
        with self._lock:
            self._all_procs.append(proc)
        return proc

    def get_proc(self) -> subprocess.Popen:
        proc: subprocess.Popen | None = getattr(self._local, "proc", None)
        if proc is None or proc.poll() is not None:
            proc = self._spawn()
            self._local.proc = proc
            self._local.pos_count = 0
        return proc

    def reset_thread_proc(self) -> subprocess.Popen:
        proc: subprocess.Popen | None = getattr(self._local, "proc", None)
        self._local.proc = None
        if proc is not None:
            _terminate_proc(proc)
        return self.get_proc()

    def annotate_batch(
        self, pbs: np.ndarray, gss: np.ndarray, go_cmd: str
    ) -> tuple[np.ndarray, np.ndarray]:
        m = len(pbs)
        out_cp = np.full(m, UNANNOTATED_CP, dtype=np.int16)
        out_d1 = np.full(m, UNANNOTATED_CP, dtype=np.int16)
        proc = self.get_proc()

        for i in range(m):
            pos_count = int(getattr(self._local, "pos_count", 0))
            if pos_count > 0 and (pos_count % 512 == 0):
                assert proc.stdin is not None and proc.stdout is not None
                proc.stdin.write("ucinewgame\nisready\n")
                proc.stdin.flush()
                while True:
                    r = proc.stdout.readline()
                    if not r or r.strip() == "readyok":
                        break

            fen = arrays_to_fen(pbs[i], gss[i])
            stm_black = int(gss[i, 0]) == 1
            try:
                assert proc.stdin is not None and proc.stdout is not None
                proc.stdin.write(f"position fen {fen}\n{go_cmd}\n")
                proc.stdin.flush()
                lines: list[str] = []
                while True:
                    ln = proc.stdout.readline()
                    if not ln:
                        raise RuntimeError("Unexpected EOF from Stockfish stdout")
                    if ln.startswith("bestmove"):
                        break
                    if ln.startswith("info "):
                        lines.append(ln)
                cp_val, d1_val = parse_sf_search_output(lines, stm_is_black=stm_black)
                out_cp[i] = np.int16(cp_val)
                out_d1[i] = np.int16(d1_val)
                self._local.pos_count = pos_count + 1
            except Exception:
                proc = self.reset_thread_proc()

        return out_cp, out_d1

    def close_all(self) -> None:
        with self._lock:
            procs = list(self._all_procs)
            self._all_procs.clear()
        for proc in procs:
            _terminate_proc(proc)


def _terminate_proc(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if proc.stdin:
            proc.stdin.write("quit\n")
            proc.stdin.flush()
        proc.wait(timeout=2)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def build_priority_order(
    n_total: int,
    *,
    eval_sample: int = 1_000_000,
    holdout_sample: int = 2_000_000,
    skip_holdout: bool = False,
) -> np.ndarray:
    """Order indices so partial runs cover ``1M dev`` + ``10M auto-RAM train`` first."""
    train_idx, dev_idx, hold_idx = split_dataset_indices(
        n_total, eval_sample=eval_sample, holdout_sample=holdout_sample
    )
    parts: list[np.ndarray] = []
    if dev_idx is not None and len(dev_idx) > 0:
        parts.append(dev_idx)

    if len(train_idx) > 10_000_000:
        # Match TexelTuningSession's deterministic rng(123) 10M auto-RAM subset first!
        auto_sub = np.sort(
            np.random.default_rng(123).choice(train_idx, size=10_000_000, replace=False)
        )
        rest_mask = ~np.isin(train_idx, auto_sub, assume_unique=True)
        parts.append(auto_sub)
        parts.append(train_idx[rest_mask])
    elif len(train_idx) > 0:
        parts.append(train_idx)

    if not skip_holdout and hold_idx is not None and len(hold_idx) > 0:
        parts.append(hold_idx)

    return np.concatenate(parts).astype(np.int64)


def _save_sidecar_npz(
    output_path: Path,
    sf_cp: np.ndarray,
    sf_d1_cp: np.ndarray,
    *,
    nodes: int,
    depth: int,
    sf_banner: str,
    source_dataset: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp_path,
        sf_cp=np.asarray(sf_cp, dtype=np.int16),
        sf_d1_cp=np.asarray(sf_d1_cp, dtype=np.int16),
        nodes=np.int64(nodes),
        depth=np.int64(depth),
        sf_banner=np.asarray(sf_banner),
        source_dataset=np.asarray(source_dataset),
    )
    os.replace(tmp_path, output_path)


def _query_sf_banner(sf_path: str) -> str:
    try:
        res = subprocess.run(
            [sf_path],
            input="uci\nquit\n",
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
        first = res.stdout.splitlines()[0].strip() if res.stdout else ""
        return first or os.path.basename(sf_path)
    except Exception:
        return os.path.basename(sf_path)


def annotate_dataset(
    dataset_path: str | Path = DEFAULT_DATASET,
    output_path: str | Path | None = None,
    *,
    stockfish_path: str | None = None,
    nodes: int = 4000,
    depth: int = 0,
    workers: int | None = None,
    batch_size: int = 512,
    save_every: int = 50_000,
    hash_mb: int = 16,
    limit: int = 0,
    skip_holdout: bool = False,
    eval_sample: int = 1_000_000,
    holdout_sample: int = 2_000_000,
    resume: bool = True,
) -> Path:
    ds_path = Path(dataset_path)
    if not ds_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {ds_path}")
    out_path = Path(output_path) if output_path else default_sidecar_path(ds_path)
    sf_bin = find_sf_binary(stockfish_path)
    sf_banner = _query_sf_banner(sf_bin)
    go_cmd = f"go depth {int(depth)}" if depth and depth > 0 else f"go nodes {int(nodes)}"
    n_workers = workers or max(1, (os.cpu_count() or 4) - 2)

    print(f"=== Stockfish Mixed-Label Sidecar Annotator ===")
    print(f"  Dataset   : {ds_path}")
    print(f"  Output    : {out_path}")
    print(f"  Engine    : {sf_banner} ({sf_bin})")
    print(f"  Search    : '{go_cmd}' | workers={n_workers} | hash={hash_mb} MB/worker")

    t_load = time.time()
    with np.load(ds_path) as npz:
        piece_bbs = npz["piece_bbs"]
        game_states = npz["game_states"]
    n_total = len(piece_bbs)
    print(f"  Loaded {n_total:,} bitboard rows in {time.time() - t_load:.1f}s")

    ckpt_cp_path = out_path.with_suffix(".ckpt_cp.npy")
    ckpt_d1_path = out_path.with_suffix(".ckpt_d1.npy")

    if resume and ckpt_cp_path.is_file() and ckpt_d1_path.is_file():
        cp_mmap = np.lib.format.open_memmap(ckpt_cp_path, mode="r+")
        d1_mmap = np.lib.format.open_memmap(ckpt_d1_path, mode="r+")
        if len(cp_mmap) != n_total or len(d1_mmap) != n_total:
            raise ValueError(
                f"Checkpoint length mismatch ({len(cp_mmap)} vs dataset {n_total}); "
                "pass --no-resume or delete .ckpt_*.npy files."
            )
        prev_done = int(np.count_nonzero(cp_mmap != UNANNOTATED_CP))
        print(f"  [Resume] Loaded memmap checkpoints ({prev_done:,}/{n_total:,} already annotated)")
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cp_mmap = np.lib.format.open_memmap(
            ckpt_cp_path, mode="w+", dtype=np.int16, shape=(n_total,)
        )
        d1_mmap = np.lib.format.open_memmap(
            ckpt_d1_path, mode="w+", dtype=np.int16, shape=(n_total,)
        )
        cp_mmap[:] = UNANNOTATED_CP
        d1_mmap[:] = UNANNOTATED_CP
        if resume and out_path.is_file():
            with np.load(out_path) as prev:
                if len(prev["sf_cp"]) == n_total:
                    cp_mmap[:] = prev["sf_cp"]
                    if "sf_d1_cp" in prev.files:
                        d1_mmap[:] = prev["sf_d1_cp"]
                    prev_done = int(np.count_nonzero(cp_mmap != UNANNOTATED_CP))
                    print(f"  [Resume] Seeded checkpoint from existing {out_path.name} ({prev_done:,} rows)")
        cp_mmap.flush()
        d1_mmap.flush()

    priority_idx = build_priority_order(
        n_total,
        eval_sample=eval_sample,
        holdout_sample=holdout_sample,
        skip_holdout=skip_holdout,
    )
    if limit and 0 < limit < len(priority_idx):
        target_idx = priority_idx[: int(limit)]
    else:
        target_idx = priority_idx

    todo_mask = cp_mmap[target_idx] == UNANNOTATED_CP
    todo_idx = target_idx[todo_mask]
    already_in_target = len(target_idx) - len(todo_idx)
    print(
        f"  Target pool: {len(target_idx):,} rows "
        f"(already done: {already_in_target:,}, remaining todo: {len(todo_idx):,})"
    )

    if len(todo_idx) == 0:
        _save_sidecar_npz(
            out_path,
            cp_mmap,
            d1_mmap,
            nodes=nodes,
            depth=depth,
            sf_banner=sf_banner,
            source_dataset=ds_path.name,
        )
        _close_memmap(cp_mmap)
        _close_memmap(d1_mmap)
        del cp_mmap, d1_mmap
        gc.collect()
        print(f"Nothing left to annotate! Saved {out_path}.")
        return out_path

    # Group into worker batches
    batches: list[np.ndarray] = [
        todo_idx[s : s + batch_size] for s in range(0, len(todo_idx), batch_size)
    ]

    t_start = time.time()
    done_now = 0
    since_flush = 0
    interrupted = False
    worker_state = _SfWorkerPoolState(sf_bin, hash_mb)
    atexit.register(worker_state.close_all)

    def _run_batch(b_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return worker_state.annotate_batch(piece_bbs[b_idx], game_states[b_idx], go_cmd)

    try:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for b_idx, (b_cp, b_d1) in zip(batches, pool.map(_run_batch, batches)):
                cp_mmap[b_idx] = b_cp
                d1_mmap[b_idx] = b_d1
                step_n = len(b_idx)
                done_now += step_n
                since_flush += step_n

                if since_flush >= save_every:
                    cp_mmap.flush()
                    d1_mmap.flush()
                    since_flush = 0

                elapsed = max(time.time() - t_start, 1e-6)
                rate = done_now / elapsed
                rem_sec = (len(todo_idx) - done_now) / max(rate, 1e-6)
                pct = 100.0 * (already_in_target + done_now) / len(target_idx)
                print(
                    f"  [{pct:6.2f}%] +{done_now:,}/{len(todo_idx):,} "
                    f"| {rate:,.0f} pos/s | ETA {rem_sec / 60.0:5.1f} min",
                    end="\r",
                )
    except KeyboardInterrupt:
        interrupted = True
        print("\n  [Interrupted] Ctrl+C received! Flushing progress and writing partial .sf19.npz...")
    finally:
        worker_state.close_all()
        cp_mmap.flush()
        d1_mmap.flush()
        _save_sidecar_npz(
            out_path,
            cp_mmap,
            d1_mmap,
            nodes=nodes,
            depth=depth,
            sf_banner=sf_banner,
            source_dataset=ds_path.name,
        )

    print()
    total_annotated = int(np.count_nonzero(cp_mmap != UNANNOTATED_CP))
    valid_cp = np.array(cp_mmap[cp_mmap != UNANNOTATED_CP], dtype=np.int16)
    valid_d1 = np.array(
        d1_mmap[(cp_mmap != UNANNOTATED_CP) & (d1_mmap != UNANNOTATED_CP)],
        dtype=np.int16,
    )
    _close_memmap(cp_mmap)
    _close_memmap(d1_mmap)
    del cp_mmap, d1_mmap
    gc.collect()

    print(
        f"Saved sidecar: {out_path} ({out_path.stat().st_size / (1024 ** 2):.2f} MB) "
        f"| annotated={total_annotated:,}/{n_total:,} ({100.0 * total_annotated / n_total:.2f}%)"
    )
    if len(valid_cp) > 0:
        print(
            f"  sf_cp stats : mean={float(np.mean(valid_cp)):+.1f} cp, "
            f"std={float(np.std(valid_cp)):.1f} cp, "
            f"p10={float(np.percentile(valid_cp, 10)):+.0f}, "
            f"p90={float(np.percentile(valid_cp, 90)):+.0f}"
        )
    if len(valid_d1) == len(valid_cp) and len(valid_cp) > 0:
        gaps = np.abs(valid_cp.astype(np.int32) - valid_d1.astype(np.int32))
        print(
            f"  |sf - d1|   : median={float(np.median(gaps)):.0f} cp, "
            f"p90={float(np.percentile(gaps, 90)):.0f} cp, "
            f">180cp={100.0 * float(np.mean(gaps > 180)):.2f}%"
        )

    if not interrupted and total_annotated >= len(target_idx) and not (limit and limit < len(priority_idx)):
        # Clean up temporary raw checkpoints once full target run finishes cleanly
        for ckpt in (ckpt_cp_path, ckpt_d1_path):
            try:
                ckpt.unlink()
            except OSError:
                pass
    elif interrupted:
        print("  Tip: Re-run the exact same command anytime to resume from checkpoint!")

    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Annotate HCE .npz positions with Stockfish 19 search scores (.sf19.npz sidecar)."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Input HCE .npz dataset (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output sidecar .npz path (default: <dataset_stem>.sf19.npz)",
    )
    parser.add_argument(
        "--stockfish",
        default=None,
        help="Path to Stockfish 19 binary (auto-detected under external/stockfish if omitted)",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=4000,
        help="Fixed node budget per position (default: 4000; ignored if --depth > 0)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=0,
        help="Optional fixed search depth instead of --nodes (default: 0 = use --nodes)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel Stockfish UCI worker processes (default: cpu_count - 2)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Positions per worker task batch (default: 512)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50_000,
        help="Flush memmap checkpoint every N newly annotated rows (default: 50000)",
    )
    parser.add_argument(
        "--hash-mb",
        type=int,
        default=16,
        help="UCI Hash MB per worker process (default: 16)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max positions to annotate in priority order: 1M dev -> 10M train -> rest (0 = all)",
    )
    parser.add_argument(
        "--skip-holdout",
        action="store_true",
        help="Skip the 2M unseen holdout split (holdout Gate always evaluates pure WDL)",
    )
    parser.add_argument(
        "--eval-sample",
        type=int,
        default=1_000_000,
        help="Dev split size matching TexelTuningSession (default: 1000000)",
    )
    parser.add_argument(
        "--holdout-sample",
        type=int,
        default=2_000_000,
        help="Unseen holdout split size matching TexelTuningSession (default: 2000000)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh instead of resuming from existing checkpoint / sidecar",
    )
    args = parser.parse_args()

    annotate_dataset(
        dataset_path=args.dataset,
        output_path=args.output,
        stockfish_path=args.stockfish,
        nodes=args.nodes,
        depth=args.depth,
        workers=args.workers,
        batch_size=args.batch_size,
        save_every=args.save_every,
        hash_mb=args.hash_mb,
        limit=args.limit,
        skip_holdout=args.skip_holdout,
        eval_sample=args.eval_sample,
        holdout_sample=args.holdout_sample,
        resume=not args.no_resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
