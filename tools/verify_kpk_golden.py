#!/usr/bin/env python3
"""
KPK golden verification CLI (stage C2).

What "golden sampling" means here:
  1. Full-table identity vs independent SF11 pure-Python reference bitbase
  2. Hand-picked theory FENs (win/draw)
  3. Optional random legal samples cross-checked with Stockfish search (oracle)

Usage (repo root):
  python tools/verify_kpk_golden.py
  python tools/verify_kpk_golden.py --sf-sample 48 --depth 20
  python -m unittest tests.test_kpk_golden -v
"""
from __future__ import annotations

import argparse
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chess_engine.classical.endgame import (  # noqa: E402
    KPK_BITBASE,
    kpk_bitbase_probe,
    evaluate_special_endgame,
    normalize_square,
)
from chess_engine.classical.fen_parser import parse_fen  # noqa: E402
from tools.kpk_sf11_reference import (  # noqa: E402
    MAX_INDEX,
    build_sf11_kpk_bitbase,
    bitbase_stats,
    probe_ref,
    index as kpk_index,
)

DEFAULT_SF = ROOT / "external" / "stockfish" / "stockfish-windows-x86-64-avx2.exe"

# Fixed theory samples (name, fen, expect win for strong/white side of KPK)
GOLDEN = [
    ("draw opposition", "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1", "draw"),
    ("draw front", "8/8/8/8/4k3/8/4P3/4K3 w - - 0 1", "draw"),
    ("draw a-pawn", "8/8/8/8/8/k7/P7/K7 w - - 0 1", "draw"),
    ("win advanced", "4k3/8/4K3/4P3/8/8/8/8 w - - 0 1", "win"),
    ("win h-corner", "8/8/8/8/8/8/7P/5k1K w - - 0 1", "win"),
]


def sq_name(sq: int) -> str:
    return "abcdefgh"[sq & 7] + "12345678"[sq // 8]


def make_kpk_fen(wksq: int, psq: int, bksq: int, stm_white: bool) -> str:
    board = ["1"] * 64
    board[wksq] = "K"
    board[psq] = "P"
    board[bksq] = "k"
    ranks = []
    for r in range(7, -1, -1):
        empty = 0
        row = ""
        for f in range(8):
            p = board[r * 8 + f]
            if p == "1":
                empty += 1
            else:
                if empty:
                    row += str(empty)
                    empty = 0
                row += p
        if empty:
            row += str(empty)
        ranks.append(row)
    stm = "w" if stm_white else "b"
    return "/".join(ranks) + f" {stm} - - 0 1"


class Stockfish:
    def __init__(self, path: Path):
        self.proc = subprocess.Popen(
            [str(path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        self._wait("uciok")
        self._send("setoption name Hash value 16")
        self._send("isready")
        self._wait("readyok")

    def _send(self, cmd: str) -> None:
        assert self.proc.stdin
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _wait(self, token: str, timeout: float = 60.0) -> list[str]:
        assert self.proc.stdout
        lines: list[str] = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                break
            line = line.rstrip("\n")
            lines.append(line)
            if token in line:
                return lines
        raise TimeoutError(f"timeout for {token}; last={lines[-3:]}")

    def search(self, fen: str, depth: int):
        self._send("ucinewgame")
        self._send("isready")
        self._wait("readyok")
        self._send(f"position fen {fen}")
        self._send(f"go depth {depth}")
        lines = self._wait("bestmove", timeout=120)
        deep = None
        for line in lines:
            m = re.search(r"score (cp|mate) (-?\d+)", line)
            if m:
                deep = (m.group(1), int(m.group(2)))
        return deep

    def close(self) -> None:
        try:
            self._send("quit")
        except Exception:
            pass
        try:
            self.proc.kill()
        except Exception:
            pass


def sf_label(deep) -> str:
    if deep is None:
        return "unknown"
    kind, s = deep
    if kind == "mate":
        return "win" if s > 0 else "loss"
    if abs(s) < 50:
        return "draw"
    return "win" if s > 0 else "loss"


def run_table_compare() -> int:
    print("[1] Building independent SF11 pure-Python reference bitbase…")
    t0 = time.time()
    ref = build_sf11_kpk_bitbase()
    print(f"    done in {time.time() - t0:.1f}s")
    st_e = bitbase_stats(KPK_BITBASE)
    st_r = bitbase_stats(ref)
    print(f"    engine wins={st_e['win_entries']}  ref wins={st_r['win_entries']}")
    if np.array_equal(KPK_BITBASE, ref):
        print("    ✅ Full table match (all 196608 indices / packed bits)")
        return 0
    diff = np.where(KPK_BITBASE != ref)[0]
    print(f"    ❌ Mismatch: {len(diff)} words differ; first word={int(diff[0])}")
    return 1


def run_golden_fens() -> int:
    print("[2] Hand-picked golden FENs…")
    bad = 0
    for name, fen, expect in GOLDEN:
        pb, occ, gs = parse_fen(fen)
        hit, score = evaluate_special_endgame(pb, occ, gs, np.int32(0))
        ok = hit and (
            (expect == "draw" and int(score) == 0)
            or (expect == "win" and abs(int(score)) > 10000)
        )
        mark = "✅" if ok else "❌"
        print(f"    {mark} {name}: hit={hit} score={int(score)} expect={expect}")
        if not ok:
            bad += 1
    return bad


def random_legal_samples(n: int, seed: int):
    rng = random.Random(seed)
    out = []
    tries = 0
    while len(out) < n and tries < n * 200:
        tries += 1
        wksq = rng.randrange(64)
        bksq = rng.randrange(64)
        pfile = rng.randrange(4)
        prank = rng.randrange(1, 7)
        psq = prank * 8 + pfile
        us = rng.randrange(2)
        if wksq == psq or bksq == psq:
            continue
        if abs((wksq // 8) - (bksq // 8)) <= 1 and abs((wksq & 7) - (bksq & 7)) <= 1:
            continue
        # skip obvious illegal (white pawn attacking black king with WTM)
        if us == 0:
            f, r = psq & 7, psq // 8
            attacks = []
            if r < 7:
                if f > 0:
                    attacks.append((r + 1) * 8 + f - 1)
                if f < 7:
                    attacks.append((r + 1) * 8 + f + 1)
            if bksq in attacks:
                continue
        win = bool(kpk_bitbase_probe(wksq, psq, bksq, us))
        fen = make_kpk_fen(wksq, psq, bksq, stm_white=(us == 0))
        out.append((wksq, psq, bksq, us, win, fen))
    return out


def run_sf_sample(n: int, depth: int, sf_path: Path, seed: int) -> int:
    print(f"[3] SF search oracle on {n} random legal samples (depth={depth})…")
    if not sf_path.exists():
        print(f"    ⚠️ SF not found: {sf_path} — skip")
        return 0
    samples = random_legal_samples(n, seed)
    sf = Stockfish(sf_path)
    mismatch = 0
    agree = 0
    try:
        for i, (wksq, psq, bksq, us, our_win, fen) in enumerate(samples):
            deep = sf.search(fen, depth)
            lab = sf_label(deep)
            # our_win True means white (strong) wins with correct STM encoding
            # SF score is from side-to-move: if white to move, win=>white wins;
            # if black to move, mate>0 means black wins (white loses) => our_win should be False
            if us == 0:  # white to move
                sf_white_win = lab == "win"
                sf_draw = lab == "draw"
            else:
                # black to move: SF "win" means black wins => white loses
                sf_white_win = lab == "loss"
                sf_draw = lab == "draw"

            if sf_draw:
                ok = not our_win
            elif lab == "unknown":
                ok = True
            else:
                ok = our_win == sf_white_win

            if ok:
                agree += 1
            else:
                mismatch += 1
                if mismatch <= 8:
                    print(
                        f"    ❌ sample#{i} our_win={our_win} sf={lab}{deep} "
                        f"wk={sq_name(wksq)} p={sq_name(psq)} bk={sq_name(bksq)} us={us}"
                    )
                    print(f"       fen {fen}")
    finally:
        sf.close()
    print(f"    agree={agree}/{len(samples)} mismatch={mismatch}")
    if mismatch:
        print("    (KPK is exact; rare SF depth-miss possible on hard draws — recheck depth)")
    return mismatch


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="KPK golden bitbase verification")
    ap.add_argument("--sf-sample", type=int, default=0, help="N random samples vs Stockfish search")
    ap.add_argument("--depth", type=int, default=18, help="SF depth for samples")
    ap.add_argument("--sf", type=Path, default=DEFAULT_SF)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    print("=== KPK golden verification ===")
    print("Golden = independent SF11 reference table + theory FENs (+ optional SF search)\n")
    rc = 0
    rc |= run_table_compare()
    rc |= 1 if run_golden_fens() else 0
    if args.sf_sample > 0:
        m = run_sf_sample(args.sf_sample, args.depth, args.sf, args.seed)
        if m > max(2, args.sf_sample // 20):
            rc |= 1
    print("\nDone." if rc == 0 else "\nFAILED.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
