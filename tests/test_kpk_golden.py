"""
KPK golden verification (C2).

1. Full-table bit identity: engine KPK_BITBASE vs independent SF11 pure-Python reference
2. Hand-picked theory FEN sample (win/draw labels)

No external Stockfish required (CI-friendly). Optional SF search sampling:
  python tools/verify_kpk_golden.py --sf-sample 64
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chess_engine.classical.endgame import (
    KPK_BITBASE,
    kpk_bitbase_probe,
    evaluate_special_endgame,
    normalize_square,
)
from chess_engine.classical.fen_parser import parse_fen
from tools.kpk_sf11_reference import (
    MAX_INDEX,
    build_sf11_kpk_bitbase,
    bitbase_stats,
    probe_ref,
)


# (name, fen, expected: "win" | "draw") — white is always the strong side with one pawn
GOLDEN_FENS = [
    ("draw opposition", "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1", "draw"),
    ("draw front of pawn", "8/8/8/8/4k3/8/4P3/4K3 w - - 0 1", "draw"),
    ("draw a-pawn classic", "8/8/8/8/8/k7/P7/K7 w - - 0 1", "draw"),
    ("win advanced", "4k3/8/4K3/4P3/8/8/8/8 w - - 0 1", "win"),
    ("win h-pawn corner", "8/8/8/8/8/8/7P/5k1K w - - 0 1", "win"),
]


class TestKpkGolden(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reference build is pure Python retrograde (~few seconds)
        cls.ref = build_sf11_kpk_bitbase()

    def test_full_table_matches_sf11_reference(self):
        """Every packed bit identical to independent SF11 port."""
        self.assertEqual(KPK_BITBASE.shape, self.ref.shape)
        if not np.array_equal(KPK_BITBASE, self.ref):
            # Locate first mismatch for debugging
            diff = np.where(KPK_BITBASE != self.ref)[0]
            i = int(diff[0])
            self.fail(
                f"KPK bitbase mismatch at word {i}: "
                f"engine=0x{int(KPK_BITBASE[i]):08x} ref=0x{int(self.ref[i]):08x} "
                f"({len(diff)} words differ)"
            )

    def test_win_count_sane(self):
        st = bitbase_stats(KPK_BITBASE)
        # Sanity: substantial wins, not empty/full table
        self.assertGreater(st["win_entries"], 10_000)
        self.assertLess(st["win_entries"], MAX_INDEX)
        # Reference must agree
        self.assertEqual(st["win_entries"], bitbase_stats(self.ref)["win_entries"])

    def test_probe_api_matches_ref_on_grid(self):
        """Sparse grid of legal-ish squares: engine probe == reference probe."""
        samples = 0
        for wksq in range(0, 64, 7):
            for bksq in range(0, 64, 5):
                for pfile in range(4):
                    for prank in range(1, 7):
                        psq = prank * 8 + pfile
                        if wksq == psq or bksq == psq:
                            continue
                        for us in (0, 1):
                            a = bool(kpk_bitbase_probe(wksq, psq, bksq, us))
                            b = probe_ref(self.ref, wksq, psq, bksq, us)
                            self.assertEqual(a, b, f"wksq={wksq} psq={psq} bksq={bksq} us={us}")
                            samples += 1
        self.assertGreater(samples, 500)

    def test_golden_fens(self):
        for name, fen, expect in GOLDEN_FENS:
            with self.subTest(name=name):
                piece_bbs, occ, gs = parse_fen(fen)
                hit, score = evaluate_special_endgame(piece_bbs, occ, gs, np.int32(0))
                self.assertTrue(hit, f"{name}: expected special KPK hit")
                if expect == "draw":
                    self.assertEqual(int(score), 0, f"{name}: expected draw score 0")
                else:
                    # White strong → positive known win; black to move may still be white-win score
                    stm = int(gs[0])
                    if stm == 0:
                        self.assertGreater(int(score), 10000, f"{name}: expected white win")
                    else:
                        # score is white POV: win for white is still > 10000
                        self.assertNotEqual(int(score), 0, f"{name}: expected decisive")
                        self.assertGreater(abs(int(score)), 10000, f"{name}: known-win scale")


if __name__ == "__main__":
    unittest.main()
