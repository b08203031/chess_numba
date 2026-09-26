"""
Convert Primer Bullet .bin (NNUE training dump) → HCE SPSA dataset.npz.

Bullet record (32 bytes), board already oriented so **side-to-move = White**:
  occupancy, packed pieces, SF score (stm), result (0/1/2), kings, pad

HCE output keys (same as preprocess_data.py):
  piece_bbs, occupancy_bbs, game_states, results

Label modes:
  game — map result 0/1/2 → 0.0 / 0.5 / 1.0  (recommended for Elo-oriented HCE)
  sf   — sigmoid of SF score in centipawns (distillation; same idea as NNUE targets)
  mix  — (1 - λ)*game + λ*sf  (λ = --sf-weight, default 0.3)

Usage (repo root)::

    python -m tuner.data_pipeline.convert_bullet_to_hce
    python -m tuner.data_pipeline.convert_bullet_to_hce --limit 2000000 --label-mode game
    python -m tuner.data_pipeline.convert_bullet_to_hce --label-mode mix --sf-weight 0.3 \\
        --limit 10000000 --output tuner/data/hce/dataset_mix03.npz
"""
from __future__ import annotations

import argparse
import math
import os
import struct
import time

import numpy as np

DEFAULT_BIN = "tuner/data/nnue/official_data_farseerT75.bin"
DEFAULT_OUT = "tuner/data/hce/dataset.npz"
RECORD_SIZE = 32
STRUCT_FMT = "<Q16shBBB3s"
# Stockfish Value → cp (same as convert_bullet_bin.py)
PAWN_VALUE_EG = 208.0
K_SIGMOID = math.log(10) / 400.0  # match tuner.compute_mse scale ≈ 0.005756

# Bullet nibble → piece index 0..11 (STM=white 0-5, NSTM=black 6-11)
_BULLET_TO_PT = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5,
    8: 6, 9: 7, 10: 8, 11: 9, 12: 10, 13: 11,
}


def _decode_record_fast(occupancy: int, pieces_arr: np.ndarray):
    """Decode Bullet board (STM already oriented as White) → HCE arrays."""
    piece_bbs = np.zeros(12, dtype=np.uint64)
    temp = int(occupancy) & 0xFFFFFFFFFFFFFFFF
    p_idx = 0
    while temp:
        bit = temp & -temp
        sq = bit.bit_length() - 1
        byte_val = int(pieces_arr[p_idx // 2])
        bullet_piece = ((byte_val >> 4) & 0x0F) if (p_idx % 2) else (byte_val & 0x0F)
        p_idx += 1
        pt = _BULLET_TO_PT.get(bullet_piece)
        if pt is not None:
            piece_bbs[pt] |= np.uint64(1) << np.uint64(sq)
        temp &= temp - 1

    white = np.uint64(0)
    black = np.uint64(0)
    for i in range(6):
        white |= piece_bbs[i]
        black |= piece_bbs[i + 6]
    occupancy_bbs = np.array([white, black, white | black], dtype=np.uint64)
    # STM-oriented board → evaluate as White to move
    game_state = np.zeros(6, dtype=np.uint64)
    game_state[0] = np.uint64(0)
    game_state[2] = np.uint64(64)  # no EP
    return piece_bbs, occupancy_bbs, game_state


def score_to_label(score: int) -> float:
    cp = score * 100.0 / PAWN_VALUE_EG
    # clamp extreme
    if cp > 10000:
        cp = 10000.0
    elif cp < -10000:
        cp = -10000.0
    return 1.0 / (1.0 + math.exp(-cp * K_SIGMOID))


def result_to_label(result: int) -> float:
    # 0 black/stm-loss, 1 draw, 2 white/stm-win  (aligned with score polarity on this dump)
    if result == 0:
        return 0.0
    if result == 2:
        return 1.0
    return 0.5


def mix_label(result: int, score: int, sf_weight: float) -> float:
    """Blend game WDL and SF-sigmoid: (1-λ)*game + λ*sf."""
    lam = float(sf_weight)
    if lam < 0.0 or lam > 1.0:
        raise ValueError(f"sf_weight must be in [0, 1], got {sf_weight}")
    g = result_to_label(result)
    s = score_to_label(score)
    return (1.0 - lam) * g + lam * s


def convert(
    input_bin: str,
    output_npz: str,
    limit: int = 2_000_000,
    label_mode: str = "game",
    max_abs_score: int = 10000,
    stride: int = 1,
    sf_weight: float = 0.3,
) -> int:
    if not os.path.isfile(input_bin):
        raise FileNotFoundError(input_bin)
    if label_mode not in ("game", "sf", "mix"):
        raise ValueError(f"Unknown label_mode: {label_mode}")
    if label_mode == "mix" and not (0.0 <= float(sf_weight) <= 1.0):
        raise ValueError(f"sf_weight must be in [0, 1], got {sf_weight}")

    file_size = os.path.getsize(input_bin)
    total = file_size // RECORD_SIZE
    target = min(limit, (total + stride - 1) // stride)
    print(f"Input: {input_bin}")
    extra = f"  sf_weight={sf_weight}" if label_mode == "mix" else ""
    print(f"  records={total:,}  convert≈{target:,}  stride={stride}  label={label_mode}{extra}")

    piece_bbs = np.zeros((target, 12), dtype=np.uint64)
    occupancy_bbs = np.zeros((target, 3), dtype=np.uint64)
    game_states = np.zeros((target, 6), dtype=np.uint64)
    results = np.zeros(target, dtype=np.float32)

    count = 0
    skipped = 0
    t0 = time.time()
    struct_len = struct.calcsize(STRUCT_FMT)
    assert struct_len == RECORD_SIZE

    with open(input_bin, "rb") as f:
        idx_in = 0
        while count < target:
            raw = f.read(RECORD_SIZE)
            if len(raw) < RECORD_SIZE:
                break
            if (idx_in % stride) != 0:
                idx_in += 1
                continue
            idx_in += 1

            s = struct.unpack(STRUCT_FMT, raw)
            occupancy = s[0]
            pieces_bytes = s[1]
            score = int(s[2])
            result = int(s[3])

            if abs(score) > max_abs_score:
                skipped += 1
                continue

            pieces_arr = np.frombuffer(pieces_bytes, dtype=np.uint8)
            p_bbs, o_bbs, g_st = _decode_record_fast(occupancy, pieces_arr)

            # Sanity: both kings present
            if p_bbs[5] == 0 or p_bbs[11] == 0:
                skipped += 1
                continue

            if label_mode == "game":
                label = result_to_label(result)
            elif label_mode == "sf":
                label = score_to_label(score)
            else:  # mix
                label = mix_label(result, score, sf_weight)

            piece_bbs[count] = p_bbs
            occupancy_bbs[count] = o_bbs
            game_states[count] = g_st
            results[count] = label
            count += 1

            if count % 200_000 == 0:
                elapsed = time.time() - t0
                print(f"  {count:,}/{target:,}  {count/max(elapsed,1e-6):,.0f}/s  skip={skipped}", end="\r")

    elapsed = time.time() - t0
    print(f"\nConverted {count:,} positions in {elapsed:.1f}s (skipped {skipped})")

    piece_bbs = piece_bbs[:count]
    occupancy_bbs = occupancy_bbs[:count]
    game_states = game_states[:count]
    results = results[:count]

    os.makedirs(os.path.dirname(output_npz) or ".", exist_ok=True)
    print(f"Saving {output_npz} ...")
    np.savez_compressed(
        output_npz,
        piece_bbs=piece_bbs,
        occupancy_bbs=occupancy_bbs,
        game_states=game_states,
        results=results,
    )
    n_unique = len(np.unique(np.round(results, 4)))
    print(
        f"Done. Label stats: min={results.min():.3f} max={results.max():.3f} "
        f"mean={results.mean():.3f} unique≈{n_unique}"
        + (f"  (mix λ={sf_weight})" if label_mode == "mix" else "")
    )
    return count


def main():
    parser = argparse.ArgumentParser(description="Bullet NNUE bin → HCE dataset.npz")
    parser.add_argument("--input", default=DEFAULT_BIN)
    parser.add_argument("--output", default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=2_000_000, help="Max positions to keep")
    parser.add_argument("--stride", type=int, default=1, help="Take every N-th record (subsample)")
    parser.add_argument(
        "--label-mode",
        choices=("game", "sf", "mix"),
        default="game",
        help="game=WDL only; sf=SF sigmoid; mix=(1-λ)*game+λ*sf",
    )
    parser.add_argument(
        "--sf-weight",
        type=float,
        default=0.3,
        help="λ for --label-mode mix: label=(1-λ)*game + λ*sf (default 0.3)",
    )
    parser.add_argument("--max-abs-score", type=int, default=10000)
    args = parser.parse_args()
    convert(
        args.input,
        args.output,
        limit=args.limit,
        label_mode=args.label_mode,
        max_abs_score=args.max_abs_score,
        stride=args.stride,
        sf_weight=args.sf_weight,
    )


if __name__ == "__main__":
    main()
