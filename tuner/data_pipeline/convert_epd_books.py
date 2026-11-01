"""
Convert resolved/quiet EPD "books" (Ethereal E12.xx, lichess-big3-resolved,
zurichess quiet-labeled) into an HCE Texel dataset ``.npz`` with **WDL labels**.

Supported line formats (labels are always White-relative):

* Ethereal / lichess-big3::

      <fen 4-6 fields> [1.0|0.5|0.0] [optional score]

  The optional trailing score (E12.52) is **ignored**: it was verified to be
  uncorrelated with both game result and material (~50 % sign agreement).

* zurichess quiet-labeled::

      <fen 4 fields> c9 "1-0"; | "0-1"; | "1/2-1/2";

Pipeline (two passes, bounded RAM):

1. Pass 1 (single process): read every line, hash the first four FEN fields
   (board / stm / castling / ep) to a 64-bit key, record label + location.
   Duplicate positions across/within files are merged and their labels averaged.
2. Random subsample of unique positions (``--max-rows``), then pass 2 parses only
   the selected lines with python-chess in a process pool, preserving castling
   rights, legal en-passant square and halfmove clock (same encoding as
   ``preprocess_data.board_to_arrays`` / ``fen_parser.parse_fen``).
   Positions that are invalid or in check are dropped.

Output keys: ``piece_bbs``, ``occupancy_bbs``, ``game_states``, ``results``
(compatible with ``tuner.tuner`` / ``TexelTuningSession``) plus ``source_ids``
(uint8) and ``source_names``.

Usage (repo root)::

    python -m tuner.data_pipeline.convert_epd_books \\
        --inputs tuner/data/hce/E12.33-1M-D12-Resolved tuner/data/hce/E12.41-1M-D12-Resolved \\
                 tuner/data/hce/E12.52-1M-D12-Resolved tuner/data/hce/lichess-big3-resolved \\
        --output tuner/data/hce/dataset_epd_wdl_20m.npz --max-rows 20000000

    python -m tuner.data_pipeline.convert_epd_books \\
        --inputs tuner/data/hce/quiet-labeled.v7.epd \\
        --output tuner/data/hce/val_zurichess.npz --max-rows 0
"""
from __future__ import annotations

import argparse
import hashlib
import os
import time
from array import array
from concurrent.futures import ProcessPoolExecutor

import chess
import numpy as np

_RESULT_TOKENS = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}
_KEY_MASK = (1 << 64) - 1


def _resolve_files(paths: list[str]) -> list[str]:
    """Expand directories into their ``.book`` / ``.epd`` files (sorted)."""
    files: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, names in os.walk(p):
                for n in sorted(names):
                    if n.endswith((".book", ".epd")):
                        files.append(os.path.join(root, n))
        elif os.path.isfile(p):
            files.append(p)
        else:
            raise FileNotFoundError(p)
    return files


def parse_label(line: str) -> float | None:
    """Return White-relative WDL label in {0, 0.5, 1} or ``None`` if unparsable."""
    lb = line.find("[")
    if lb >= 0:
        rb = line.find("]", lb)
        if rb < 0:
            return None
        try:
            v = float(line[lb + 1:rb])
        except ValueError:
            return None
        return v if v in (0.0, 0.5, 1.0) else None
    q1 = line.find('"')
    if q1 >= 0:
        q2 = line.find('"', q1 + 1)
        return _RESULT_TOKENS.get(line[q1 + 1:q2]) if q2 > q1 else None
    return None


def fen_of(line: str) -> str:
    """Extract a python-chess compatible FEN from a book/EPD line."""
    parts = line.split()
    head = parts[:4]
    # Ethereal FENs carry halfmove/fullmove; EPD (zurichess) does not.
    if len(parts) >= 6 and parts[4].isdigit() and parts[5].isdigit():
        head += parts[4:6]
    else:
        head += ["0", "1"]
    return " ".join(head)


def position_key(line: str) -> int:
    """64-bit hash of board / side / castling / ep (ignores move counters)."""
    key_str = " ".join(line.split()[:4]).encode("ascii", "replace")
    return int.from_bytes(hashlib.blake2b(key_str, digest_size=8).digest(), "little") & _KEY_MASK


def pass1_scan(files: list[str]):
    """Hash every line. Returns (keys, labels, file_idx, line_no)."""
    # Compact typed buffers: ~21 bytes/line instead of ~120 for Python lists.
    keys, labels, fidx, lno = array("Q"), array("f"), array("B"), array("q")
    bad = 0
    for fi, path in enumerate(files):
        t0 = time.time()
        n0 = len(keys)
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for li, line in enumerate(fh):
                lab = parse_label(line)
                if lab is None:
                    bad += 1
                    continue
                keys.append(position_key(line))
                labels.append(lab)
                fidx.append(fi)
                lno.append(li)
        print(f"  [pass1] {os.path.basename(path)}: {len(keys) - n0:,} lines in {time.time() - t0:.0f}s")
    print(f"  [pass1] total={len(keys):,} unparsable={bad:,}")
    return (
        np.frombuffer(keys, dtype=np.uint64).copy(),
        np.frombuffer(labels, dtype=np.float32).copy(),
        np.frombuffer(fidx, dtype=np.uint8).copy(),
        np.frombuffer(lno, dtype=np.int64).copy(),
    )


def _encode_chunk(args):
    """Worker: parse FENs → (piece_bbs, occupancy_bbs, game_states, ok mask)."""
    fens = args
    n = len(fens)
    pb = np.zeros((n, 12), dtype=np.uint64)
    ob = np.zeros((n, 3), dtype=np.uint64)
    gs = np.zeros((n, 6), dtype=np.uint64)
    ok = np.zeros(n, dtype=np.bool_)
    for i, fen in enumerate(fens):
        try:
            board = chess.Board(fen)
        except ValueError:
            continue
        if not board.is_valid() or board.is_check():
            continue
        for pt in range(1, 7):
            pb[i, pt - 1] = board.pieces_mask(pt, chess.WHITE)
            pb[i, pt + 5] = board.pieces_mask(pt, chess.BLACK)
        ob[i, 0] = board.occupied_co[chess.WHITE]
        ob[i, 1] = board.occupied_co[chess.BLACK]
        ob[i, 2] = board.occupied
        gs[i, 0] = 0 if board.turn == chess.WHITE else 1
        c = 0
        if board.has_kingside_castling_rights(chess.WHITE):
            c |= 1
        if board.has_queenside_castling_rights(chess.WHITE):
            c |= 2
        if board.has_kingside_castling_rights(chess.BLACK):
            c |= 4
        if board.has_queenside_castling_rights(chess.BLACK):
            c |= 8
        gs[i, 1] = c
        ep = board.ep_square if board.has_legal_en_passant() else None
        gs[i, 2] = ep if ep is not None else 64
        gs[i, 3] = board.halfmove_clock
        ok[i] = True
    return pb, ob, gs, ok


def convert(
    inputs: list[str],
    output: str,
    max_rows: int = 20_000_000,
    seed: int = 2026,
    workers: int | None = None,
    chunk: int = 100_000,
) -> int:
    files = _resolve_files(inputs)
    if not files:
        raise SystemExit("No input files found.")
    if len(files) > 255:
        raise SystemExit("Too many input files (source_ids is uint8).")
    print("Inputs:")
    for i, f in enumerate(files):
        print(f"  [{i}] {f}")

    t_start = time.time()
    keys, labels, fidx, lno = pass1_scan(files)

    # --- Deduplicate (average labels of identical positions) ---
    uniq, first_idx, inverse = np.unique(keys, return_index=True, return_inverse=True)
    sums = np.bincount(inverse, weights=labels.astype(np.float64), minlength=len(uniq))
    cnts = np.bincount(inverse, minlength=len(uniq))
    avg = (sums / cnts).astype(np.float32)
    n_uniq = len(uniq)
    print(f"  [dedup] {len(keys):,} -> {n_uniq:,} unique ({len(keys) - n_uniq:,} duplicates merged)")
    del keys, inverse, sums

    rng = np.random.default_rng(seed)
    if max_rows and 0 < max_rows < n_uniq:
        sel = np.sort(rng.choice(n_uniq, size=max_rows, replace=False))
    else:
        sel = np.arange(n_uniq)
    rep = first_idx[sel]                 # representative line (global index)
    sel_labels = avg[sel]
    sel_dups = cnts[sel]
    del first_idx, avg, cnts

    # --- Pass 2: re-read only selected lines, per file in line order ---
    order = np.lexsort((lno[rep], fidx[rep]))
    rep = rep[order]
    sel_labels = sel_labels[order]
    sel_dups = sel_dups[order]
    rep_f = fidx[rep]
    rep_l = lno[rep]
    del fidx, lno

    fens: list[str] = []
    for fi, path in enumerate(files):
        wanted = rep_l[rep_f == fi]
        if len(wanted) == 0:
            continue
        wi = 0
        nw = len(wanted)
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for li, line in enumerate(fh):
                if wi >= nw:
                    break
                if li == wanted[wi]:
                    fens.append(fen_of(line))
                    wi += 1
        print(f"  [pass2] collected {wi:,} FENs from {os.path.basename(path)}")
    assert len(fens) == len(rep), (len(fens), len(rep))

    n = len(fens)
    pb = np.zeros((n, 12), dtype=np.uint64)
    ob = np.zeros((n, 3), dtype=np.uint64)
    gs = np.zeros((n, 6), dtype=np.uint64)
    ok = np.zeros(n, dtype=np.bool_)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    t0 = time.time()
    bounds = [(s, min(s + chunk, n)) for s in range(0, n, chunk)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for (s, e), (cpb, cob, cgs, cok) in zip(
            bounds, ex.map(_encode_chunk, (fens[s:e] for s, e in bounds))
        ):
            pb[s:e], ob[s:e], gs[s:e], ok[s:e] = cpb, cob, cgs, cok
            done = e
            if (s // chunk) % 10 == 0:
                rate = done / max(time.time() - t0, 1e-6)
                print(f"  [encode] {done:,}/{n:,}  {rate:,.0f}/s", end="\r")
    print(f"\n  [encode] done in {time.time() - t0:.0f}s; dropped {int((~ok).sum()):,} invalid/in-check")
    del fens

    # Shuffle so contiguous slices mix sources (split_dataset_indices is random anyway).
    keep = np.flatnonzero(ok)
    keep = keep[rng.permutation(len(keep))]
    pb, ob, gs = pb[keep], ob[keep], gs[keep]
    results = sel_labels[keep]
    source_ids = rep_f[keep].astype(np.uint8)
    dup_counts = np.minimum(sel_dups[keep], 255).astype(np.uint8)

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    print(f"Saving {output} ...")
    np.savez_compressed(
        output,
        piece_bbs=pb,
        occupancy_bbs=ob,
        game_states=gs,
        results=results,
        source_ids=source_ids,
        dup_counts=dup_counts,
        source_names=np.asarray([os.path.basename(f) for f in files]),
    )

    m = len(results)
    print(f"Done: {m:,} positions in {time.time() - t_start:.0f}s")
    print(
        f"  label mean={results.mean():.4f}  win={np.mean(results == 1.0):.3f} "
        f"draw={np.mean(results == 0.5):.3f} loss={np.mean(results == 0.0):.3f} "
        f"fractional(avg dup)={np.mean((results != 0) & (results != 0.5) & (results != 1)):.4f}"
    )
    print(f"  castling!=0={np.mean(gs[:, 1] != 0):.2%}  ep={np.mean(gs[:, 2] < 64):.2%}  "
          f"stm_black={np.mean(gs[:, 0] == 1):.3f}")
    for i, f in enumerate(files):
        print(f"  src[{i}] {os.path.basename(f)}: {int(np.sum(source_ids == i)):,}")
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description="Resolved EPD books -> HCE Texel WDL dataset.npz")
    ap.add_argument("--inputs", nargs="+", required=True, help="Files or directories (.book/.epd)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-rows", type=int, default=20_000_000,
                    help="Random subsample of unique positions (0 = keep all)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=100_000)
    args = ap.parse_args()
    convert(args.inputs, args.output, max_rows=args.max_rows, seed=args.seed,
            workers=args.workers, chunk=args.chunk)


if __name__ == "__main__":
    main()
