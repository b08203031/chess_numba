"""
Validate Primer Bullet dump suitability for HCE SPSA using Stockfish 18.

Checks (on a random sample):
  1) Decode → legal python-chess boards (kings, piece counts, not check if claimed quiet)
  2) Stored SF score (bullet) vs SF18 eval correlation
  3) Game result vs SF18 score polarity / accuracy
  4) Quiescence: depth-1 vs deeper gap
  5) Our classical HCE vs SF18 correlation (baseline before tuning)

Usage (repo root)::

    python -m tuner.data_pipeline.validate_bullet_with_sf --sample 300 --depth 12
"""
from __future__ import annotations

import argparse
import math
import os
import random
import struct
import sys
import time
from collections import Counter

import chess
import chess.engine
import numpy as np

from tuner.data_pipeline.convert_bullet_to_hce import (
    DEFAULT_BIN,
    PAWN_VALUE_EG,
    RECORD_SIZE,
    STRUCT_FMT,
    _BULLET_TO_PT,
    _decode_record_fast,
    result_to_label,
    score_to_label,
)

DEFAULT_SF = "external/stockfish/stockfish-windows-x86-64-avx2.exe"
# Fallback paths
SF_CANDIDATES = [
    DEFAULT_SF,
    "stockfish_repo/dist/stockfish/stockfish-windows-x86-64-avx2.exe",
    "external/stockfish_dir/stockfish-windows-x86-64-avx2.exe",
]


def find_sf(path: str | None) -> str:
    if path and os.path.isfile(path):
        return path
    for p in SF_CANDIDATES:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError("Stockfish executable not found; pass --stockfish")


def piece_bbs_to_board(piece_bbs: np.ndarray) -> chess.Board:
    board = chess.Board()
    board.clear_board()
    types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    for i, pt in enumerate(types):
        for color_off, color in ((0, chess.WHITE), (6, chess.BLACK)):
            bb = int(piece_bbs[i + color_off])
            while bb:
                bit = bb & -bb
                sq = bit.bit_length() - 1
                board.set_piece_at(sq, chess.Piece(pt, color))
                bb &= bb - 1
    board.turn = chess.WHITE  # STM-oriented
    board.castling_rights = 0
    board.ep_square = None
    return board


def bullet_score_to_cp(score: int) -> float:
    return score * 100.0 / PAWN_VALUE_EG


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    if denom < 1e-12:
        return float("nan")
    return float((x * y).sum() / denom)


def sample_records(bin_path: str, n: int, seed: int):
    file_size = os.path.getsize(bin_path)
    total = file_size // RECORD_SIZE
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(total), min(n, total)))
    out = []
    with open(bin_path, "rb") as f:
        for idx in indices:
            f.seek(idx * RECORD_SIZE)
            raw = f.read(RECORD_SIZE)
            if len(raw) < RECORD_SIZE:
                continue
            s = struct.unpack(STRUCT_FMT, raw)
            out.append(
                {
                    "idx": idx,
                    "occupancy": s[0],
                    "pieces": np.frombuffer(s[1], dtype=np.uint8),
                    "score": int(s[2]),
                    "result": int(s[3]),
                    "wk": int(s[4]),
                    "bk": int(s[5]),
                }
            )
    return out, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin", default=DEFAULT_BIN)
    parser.add_argument("--stockfish", default=None)
    parser.add_argument("--sample", type=int, default=300)
    parser.add_argument("--depth", type=int, default=12, help="Deep SF18 search depth")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-hce", action="store_true", help="Skip classical eval (faster)")
    args = parser.parse_args()

    sf_path = find_sf(args.stockfish)
    print(f"Stockfish: {sf_path}")
    print(f"Bullet:    {args.bin}")
    print(f"Sample:    {args.sample}  deep depth={args.depth}  seed={args.seed}")

    records, total = sample_records(args.bin, args.sample, args.seed)
    print(f"File records: {total:,}  loaded sample: {len(records)}")

    # Optional HCE
    hce_fn = None
    if not args.skip_hce:
        try:
            from chess_engine.classical.evaluation import evaluate_position
            from chess_engine.classical.fen_parser import parse_fen

            def hce_score_from_board(board: chess.Board) -> int:
                fen = board.fen()
                pb, ob, gs = parse_fen(fen)
                ref = evaluate_position(pb, ob, gs, lazy=False)
                return int(ref[0] if isinstance(ref, tuple) else ref)

            hce_fn = hce_score_from_board
            print("Classical HCE: enabled")
        except Exception as e:
            print(f"Classical HCE disabled ({e})")

    engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    # Align threads for throughput
    try:
        engine.configure({"Threads": 1, "Hash": 64})
    except Exception:
        pass

    stats = {
        "legal_ok": 0,
        "illegal": 0,
        "in_check": 0,
        "decode_fail": 0,
        "mate_d1": 0,
        "mate_deep": 0,
        "engine_err": 0,
    }
    bullet_cp = []
    sf_d1 = []
    sf_deep = []
    game_lab = []
    hce_scores = []
    quiet_gaps = []  # |deep - d1|
    result_agree = Counter()  # SF sign vs game result

    t0 = time.time()
    try:
        for i, rec in enumerate(records):
            try:
                p_bbs, o_bbs, g_st = _decode_record_fast(rec["occupancy"], rec["pieces"])
                board = piece_bbs_to_board(p_bbs)
            except Exception:
                stats["decode_fail"] += 1
                continue

            # Basic legality
            if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
                stats["illegal"] += 1
                continue
            if not board.is_valid():
                stats["illegal"] += 1
                continue
            stats["legal_ok"] += 1
            if board.is_check():
                stats["in_check"] += 1

            bcp = bullet_score_to_cp(rec["score"])
            gl = result_to_label(rec["result"])

            try:
                info1 = engine.analyse(board, chess.engine.Limit(depth=1))
                s1 = info1["score"].white()
                if s1.is_mate():
                    stats["mate_d1"] += 1
                    cp1 = 10000 if s1.mate() > 0 else -10000
                else:
                    cp1 = float(s1.score())

                info_d = engine.analyse(board, chess.engine.Limit(depth=args.depth))
                sd = info_d["score"].white()
                if sd.is_mate():
                    stats["mate_deep"] += 1
                    cpd = 10000 if sd.mate() > 0 else -10000
                else:
                    cpd = float(sd.score())
            except (chess.engine.EngineError, chess.engine.EngineTerminatedError):
                stats["engine_err"] += 1
                continue

            bullet_cp.append(bcp)
            sf_d1.append(cp1)
            sf_deep.append(cpd)
            game_lab.append(gl)
            quiet_gaps.append(abs(cpd - cp1))

            # Agreement: game result polarity vs deep SF
            if gl == 0.5:
                result_agree["draw_label"] += 1
            elif (gl > 0.5 and cpd > 0) or (gl < 0.5 and cpd < 0):
                result_agree["sign_match"] += 1
            elif abs(cpd) < 50:
                result_agree["sf_near_draw"] += 1
            else:
                result_agree["sign_mismatch"] += 1

            if hce_fn is not None:
                try:
                    hce_scores.append(float(hce_fn(board)))
                except Exception:
                    hce_scores.append(float("nan"))

            if (i + 1) % 25 == 0:
                print(f"  analysed {i+1}/{len(records)} ...", end="\r")
    finally:
        engine.quit()

    elapsed = time.time() - t0
    n = len(sf_deep)
    print(f"\nDone in {elapsed:.1f}s  usable={n}")

    bullet_cp = np.array(bullet_cp, dtype=np.float64)
    sf_d1 = np.array(sf_d1, dtype=np.float64)
    sf_deep = np.array(sf_deep, dtype=np.float64)
    game_lab = np.array(game_lab, dtype=np.float64)
    quiet_gaps = np.array(quiet_gaps, dtype=np.float64)

    print("\n========== STRUCTURE / FILTER QUALITY ==========")
    print(f"  decode_fail={stats['decode_fail']}  illegal={stats['illegal']}  legal_ok={stats['legal_ok']}")
    print(f"  in_check (should be ~0 if Primer filtered): {stats['in_check']} / {stats['legal_ok']}")
    print(f"  mate@d1={stats['mate_d1']}  mate@d{args.depth}={stats['mate_deep']}  engine_err={stats['engine_err']}")
    if stats["legal_ok"]:
        print(f"  check_rate={100*stats['in_check']/stats['legal_ok']:.2f}%")

    print("\n========== SCORE DISTRIBUTIONS (cp) ==========")
    for name, arr in (("bullet_stored", bullet_cp), ("sf18_d1", sf_d1), (f"sf18_d{args.depth}", sf_deep)):
        if len(arr):
            print(
                f"  {name:14s}  mean={arr.mean():+7.1f}  std={arr.std():6.1f}  "
                f"p10={np.percentile(arr,10):+7.1f}  p90={np.percentile(arr,90):+7.1f}"
            )

    print("\n========== CORRELATIONS ==========")
    r_b_d1 = pearson(bullet_cp, sf_d1)
    r_b_deep = pearson(bullet_cp, sf_deep)
    r_d1_deep = pearson(sf_d1, sf_deep)
    print(f"  bullet vs SF18 d1:     r = {r_b_d1:.4f}")
    print(f"  bullet vs SF18 d{args.depth}:   r = {r_b_deep:.4f}")
    print(f"  SF18 d1 vs d{args.depth}:       r = {r_d1_deep:.4f}")
    mae_b_deep = float(np.mean(np.abs(bullet_cp - sf_deep)))
    mae_b_d1 = float(np.mean(np.abs(bullet_cp - sf_d1)))
    print(f"  MAE bullet vs SF18 d1:   {mae_b_d1:.1f} cp")
    print(f"  MAE bullet vs SF18 d{args.depth}: {mae_b_deep:.1f} cp")

    # Game label vs SF deep (point-biserial style via pearson with 0/0.5/1)
    r_game_deep = pearson(game_lab, sf_deep)
    print(f"  game_result vs SF18 d{args.depth}: r = {r_game_deep:.4f}")

    # Soft accuracy: among decisive labels, SF agrees on sign
    decisive = (game_lab != 0.5) & (np.abs(sf_deep) >= 50)
    if decisive.sum():
        agree = ((game_lab[decisive] > 0.5) == (sf_deep[decisive] > 0)).mean()
        print(f"  decisive label sign agreement with SF18 (|cp|≥50): {100*agree:.1f}%  (n={decisive.sum()})")
    print(f"  result buckets: {dict(result_agree)}")

    print("\n========== QUIESCENCE (tactical noise) ==========")
    print(f"  |SF d{args.depth} - d1|  mean={quiet_gaps.mean():.1f}  median={np.median(quiet_gaps):.1f}  "
          f"p90={np.percentile(quiet_gaps,90):.1f}  >100cp={100*(quiet_gaps>100).mean():.1f}%  "
          f">200cp={100*(quiet_gaps>200).mean():.1f}%")

    if hce_fn is not None and hce_scores:
        hce = np.array(hce_scores, dtype=np.float64)
        mask = np.isfinite(hce)
        if mask.sum() > 10:
            print("\n========== OUR HCE vs SF18 (pre-tune baseline) ==========")
            print(f"  HCE vs SF18 d{args.depth}: r = {pearson(hce[mask], sf_deep[mask]):.4f}")
            print(f"  HCE vs bullet:     r = {pearson(hce[mask], bullet_cp[mask]):.4f}")
            print(f"  MAE HCE vs SF18 d{args.depth}: {np.mean(np.abs(hce[mask]-sf_deep[mask])):.1f} cp")

    # Verdict
    print("\n========== VERDICT (for HCE SPSA) ==========")
    issues = []
    ok = []
    if stats["legal_ok"] / max(len(records), 1) < 0.95:
        issues.append("Too many illegal/decode failures after conversion.")
    else:
        ok.append("Boards decode to legal positions at high rate.")
    if stats["legal_ok"] and stats["in_check"] / stats["legal_ok"] > 0.05:
        issues.append(f"Check rate high ({100*stats['in_check']/stats['legal_ok']:.1f}%); static eval labels noisy.")
    else:
        ok.append("Low check rate (Primer quiet filter looks intact).")
    if r_b_deep < 0.7:
        issues.append(f"Bullet stored score poorly tracks SF18 (r={r_b_deep:.2f}); scale/side bug or stale labels.")
    else:
        ok.append(f"Bullet scores track SF18 well (r={r_b_deep:.2f}).")
    if r_game_deep < 0.25:
        issues.append(f"Game results weakly related to SF18 (r={r_game_deep:.2f}); weak for Texel.")
    else:
        ok.append(f"Game results correlate with SF18 (r={r_game_deep:.2f}) — usable Texel labels.")
    if len(quiet_gaps) and (quiet_gaps > 200).mean() > 0.25:
        issues.append("Many tactically unstable positions (|Δ|>200cp); prefer stricter filter or SF soft labels carefully.")
    else:
        ok.append("Quiescence gap distribution acceptable for static tuning.")

    for s in ok:
        print(f"  [OK]  {s}")
    for s in issues:
        print(f"  [!!]  {s}")

    if not issues:
        print("\n  CONCLUSION: Data looks SUITABLE for HCE parameter tuning.")
        print("  Prefer --label-mode game for Elo; optional sf soft labels as secondary.")
        return 0
    if len(issues) <= 1 and r_b_deep >= 0.6:
        print("\n  CONCLUSION: CONDITIONALLY suitable — fix/filter issues above if possible.")
        return 0
    print("\n  CONCLUSION: NOT recommended as-is until issues are resolved.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
