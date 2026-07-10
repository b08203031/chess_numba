"""
Specialized endgame unit tests (daily / CI-friendly).

Win/draw/loss labels were cross-checked with Stockfish search via
tools/verify_endgame_vs_sf.py (optional oracle). Scores use SF11 HCE scale
(VALUE_KNOWN_WIN ≈ 10000), not NNUE centipawns.

See also: tests/test_phase_a_eval.py (A1–A4 + extra KPK edges).
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
import numpy as np
from chess_engine.classical.fen_parser import parse_fen
from chess_engine.classical.endgame import evaluate_special_endgame
from chess_engine.classical.evaluation import evaluate_position


class TestEndgameConformance(unittest.TestCase):
    def setUp(self):
        self.phase = np.int32(0)

    def _special(self, fen):
        piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
        hit, score = evaluate_special_endgame(piece_bbs, occupancy_bbs, game_state, self.phase)
        val, _, _ = evaluate_position(piece_bbs, occupancy_bbs, game_state)
        return hit, int(score), int(val), piece_bbs, occupancy_bbs, game_state

    # ----- KBNK (SF d40: mate) -----
    def test_kbnk_white(self):
        # SF: mate in ~34 from this setup
        fen = "4k3/8/8/8/3BN3/8/8/7K w - - 0 1"
        hit, score, val, *_ = self._special(fen)
        self.assertTrue(hit)
        self.assertGreater(score, 10000)
        self.assertEqual(val, score)  # white to move

    def test_kbnk_black(self):
        fen = "7k/8/8/3bn3/8/8/8/4K3 w - - 0 1"
        hit, score, val, *_ = self._special(fen)
        self.assertTrue(hit)
        self.assertLess(score, -10000)
        # White to move: STM score == white-perspective score (negative = black winning)
        self.assertEqual(val, score)

    # ----- KXK (SF: mate) -----
    def test_kxk_krk(self):
        fen = "4k3/8/8/8/3R4/8/8/7K w - - 0 1"
        hit, score, val, *_ = self._special(fen)
        self.assertTrue(hit)
        self.assertGreater(score, 10000)
        self.assertGreater(val, 10000)

    def test_kxk_kqk(self):
        fen = "4k3/8/8/8/3Q4/8/8/7K w - - 0 1"
        hit, score, val, *_ = self._special(fen)
        self.assertTrue(hit)
        self.assertGreater(score, 10000)

    # ----- KPK (SF-verified) -----
    def test_kpk_draw_opposition(self):
        # SF d40: cp 0 — black king blocks pawn path with opposition
        fen = "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1"
        hit, score, val, *_ = self._special(fen)
        self.assertTrue(hit)
        self.assertEqual(score, 0)
        self.assertEqual(val, 0)

    def test_kpk_draw_front_of_pawn(self):
        # SF d40: cp 0
        fen = "8/8/8/8/4k3/8/4P3/4K3 w - - 0 1"
        hit, score, val, *_ = self._special(fen)
        self.assertTrue(hit)
        self.assertEqual(score, 0)

    def test_kpk_win_h_pawn_corner(self):
        # SF d40: mate in 16 — previously mis-labeled as draw in our tests
        fen = "8/8/8/8/8/8/7P/5k1K w - - 0 1"
        hit, score, val, *_ = self._special(fen)
        self.assertTrue(hit)
        self.assertGreater(score, 10000)
        self.assertGreater(val, 10000)

    def test_kpk_draw_rook_pawn_a_file(self):
        # Classic a-pawn draw: defending king in front of rook pawn
        # SF should be ~0; bitbase draw
        fen = "8/8/8/8/8/k7/P7/K7 w - - 0 1"
        hit, score, val, *_ = self._special(fen)
        self.assertTrue(hit)
        self.assertEqual(score, 0)

    def test_kpk_win_advanced(self):
        # White king in front of pawn, black king displaced — should be win
        fen = "4k3/8/4K3/4P3/8/8/8/8 w - - 0 1"
        hit, score, val, *_ = self._special(fen)
        self.assertTrue(hit)
        self.assertGreater(score, 10000)

    # ----- KRKP / KQKP (SF: white better / mate) -----
    def test_krkp(self):
        # SF d22: ~+500 cp for white
        fen = "4k3/8/8/8/3R4/8/4p3/4K3 w - - 0 1"
        hit, score, val, *_ = self._special(fen)
        self.assertTrue(hit)
        self.assertGreater(score, 0)
        self.assertGreater(val, 0)

    def test_kqkp(self):
        # SF d22: mate
        fen = "4k3/4p3/8/8/8/8/8/3QK3 w - - 0 1"
        hit, score, val, *_ = self._special(fen)
        self.assertTrue(hit)
        self.assertGreater(score, 0)

    # ----- Non-special -----
    def test_not_special_kbk(self):
        # SF d22: cp 0 theoretical draw; npm < rook so not KXK
        fen = "4k3/8/8/8/3B4/8/8/4K3 w - - 0 1"
        hit, score, *_ = self._special(fen)
        self.assertFalse(hit)

    def test_not_special_opening(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        hit, score, *_ = self._special(fen)
        self.assertFalse(hit)


if __name__ == "__main__":
    unittest.main()
