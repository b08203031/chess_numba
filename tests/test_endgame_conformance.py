import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
import numpy as np
from chess_engine.classical.fen_parser import parse_fen
from chess_engine.classical.endgame import evaluate_special_endgame

class TestEndgameConformance(unittest.TestCase):
    def setUp(self):
        # game_state layout: [castling_rights, en_passant_sq, active_color, halfmove_clock, fullmove_number]
        self.game_state = np.array([15, -1, 0, 0, 1], dtype=np.int32)
        self.phase = np.int32(0) # Endgame phase

    def test_evaluate_special_endgame_basic_mates_and_draws(self):
        # White KB vs Black K (draw)
        fen = "2b1k3/8/8/8/8/8/8/4K3 w - - 0 1"
        piece_bbs, occupancy_bbs, active_color = parse_fen(fen)
        is_special, score = evaluate_special_endgame(piece_bbs, occupancy_bbs, self.game_state, self.phase)
        self.assertFalse(is_special)

        # White KN vs Black K (draw)
        fen = "4k3/8/8/8/8/2N5/8/4K3 w - - 0 1"
        piece_bbs, occupancy_bbs, active_color = parse_fen(fen)
        is_special, score = evaluate_special_endgame(piece_bbs, occupancy_bbs, self.game_state, self.phase)
        self.assertFalse(is_special)

        # White KNN vs Black K (draw)
        fen = "4k3/8/8/8/8/2N1N3/8/4K3 w - - 0 1"
        piece_bbs, occupancy_bbs, active_color = parse_fen(fen)
        is_special, score = evaluate_special_endgame(piece_bbs, occupancy_bbs, self.game_state, self.phase)
        self.assertFalse(is_special)

        # White KQ vs Black K (win)
        fen = "4k3/8/8/8/3Q4/8/8/7K w - - 0 1"
        piece_bbs, occupancy_bbs, active_color = parse_fen(fen)
        is_special, score = evaluate_special_endgame(piece_bbs, occupancy_bbs, self.game_state, self.phase)
        self.assertFalse(is_special)

        # Black KQ vs White K (win)
        fen = "4k3/8/8/8/3q4/8/8/7K w - - 0 1"
        piece_bbs, occupancy_bbs, active_color = parse_fen(fen)
        is_special, score = evaluate_special_endgame(piece_bbs, occupancy_bbs, self.game_state, self.phase)
        self.assertFalse(is_special)

        # White KR vs Black K (win)
        fen = "4k3/8/8/8/3R4/8/8/7K w - - 0 1"
        piece_bbs, occupancy_bbs, active_color = parse_fen(fen)
        is_special, score = evaluate_special_endgame(piece_bbs, occupancy_bbs, self.game_state, self.phase)
        self.assertFalse(is_special)

        # White KBB vs Black K (win)
        fen = "4k3/8/8/8/3BB3/8/8/7K w - - 0 1"
        piece_bbs, occupancy_bbs, active_color = parse_fen(fen)
        is_special, score = evaluate_special_endgame(piece_bbs, occupancy_bbs, self.game_state, self.phase)
        self.assertFalse(is_special)

        # White KBN vs Black K (win)
        # bishop = 27 (d4), knight = 28 (e4), wk_sq = 7 (h1), bk_sq = 60 (e8)
        # PUSH_TO_CORNERS[60] = 5440. 5440 // 32 = 170.
        # PUSH_CLOSE[7] // 2 = 10 // 2 = 5.
        # mate_kbnk = 170 + 5 = 175.
        # score = 650 + 175 = 825
        fen = "4k3/8/8/8/3BN3/8/8/7K w - - 0 1"
        piece_bbs, occupancy_bbs, active_color = parse_fen(fen)
        is_special, score = evaluate_special_endgame(piece_bbs, occupancy_bbs, self.game_state, self.phase)
        self.assertTrue(is_special)
        self.assertEqual(score, 825)

    def test_evaluate_special_endgame_krkb_krkn_kqkr(self):
        # White KR vs Black KB (KRKB)
        fen = "4k3/8/8/5b2/3R4/8/8/7K w - - 0 1"
        piece_bbs, occupancy_bbs, active_color = parse_fen(fen)
        is_special, score = evaluate_special_endgame(piece_bbs, occupancy_bbs, self.game_state, self.phase)
        self.assertFalse(is_special)

        # Black KR vs White KB (KRKB)
        fen = "4k3/8/8/5B2/3r4/8/8/7K w - - 0 1"
        piece_bbs, occupancy_bbs, active_color = parse_fen(fen)
        is_special, score = evaluate_special_endgame(piece_bbs, occupancy_bbs, self.game_state, self.phase)
        self.assertFalse(is_special)

        # White KR vs Black KN (KRKN)
        fen = "4k3/8/6n1/8/3R4/8/8/7K w - - 0 1"
        piece_bbs, occupancy_bbs, active_color = parse_fen(fen)
        is_special, score = evaluate_special_endgame(piece_bbs, occupancy_bbs, self.game_state, self.phase)
        self.assertFalse(is_special)

        # Black KR vs White KN (KRKN)
        fen = "4k3/8/6N1/8/3r4/8/8/7K w - - 0 1"
        piece_bbs, occupancy_bbs, active_color = parse_fen(fen)
        is_special, score = evaluate_special_endgame(piece_bbs, occupancy_bbs, self.game_state, self.phase)
        self.assertFalse(is_special)

        # White KQ vs Black KR (KQKR)
        fen = "4k3/8/r7/8/3Q4/8/8/7K w - - 0 1"
        piece_bbs, occupancy_bbs, active_color = parse_fen(fen)
        is_special, score = evaluate_special_endgame(piece_bbs, occupancy_bbs, self.game_state, self.phase)
        self.assertFalse(is_special)

        # Black KQ vs White KR (KQKR)
        fen = "4k3/8/8/4q3/3R4/8/8/7K w - - 0 1"
        piece_bbs, occupancy_bbs, active_color = parse_fen(fen)
        is_special, score = evaluate_special_endgame(piece_bbs, occupancy_bbs, self.game_state, self.phase)
        self.assertFalse(is_special)

    def test_evaluate_special_endgame_krkp_kqkp_knnkp(self):
        # White KR vs Black KP (KRKP, strong side in front of pawn win)
        fen = "8/8/8/3k4/8/3p4/3K4/3R4 w - - 0 1"
        piece_bbs, occupancy_bbs, active_color = parse_fen(fen)
        is_special, score = evaluate_special_endgame(piece_bbs, occupancy_bbs, self.game_state, self.phase)
        self.assertFalse(is_special)

        # Black KR vs White KP (KRKP)
        fen = "3r4/3k4/3P4/8/3K4/8/8/8 b - - 0 1"
        piece_bbs, occupancy_bbs, active_color = parse_fen(fen)
        is_special, score = evaluate_special_endgame(piece_bbs, occupancy_bbs, self.game_state, self.phase)
        self.assertFalse(is_special)

        # White KQ vs Black KP (KQKP, pawn on 7th rank / file C bishop pawn draw)
        fen = "3Q3K/8/8/8/8/8/2p5/1k6 w - - 0 1"
        piece_bbs, occupancy_bbs, active_color = parse_fen(fen)
        is_special, score = evaluate_special_endgame(piece_bbs, occupancy_bbs, self.game_state, self.phase)
        self.assertFalse(is_special)

        # Black KQ vs White KP (KQKP)
        fen = "1K6/2P5/8/8/8/8/8/3q3k w - - 0 1"
        piece_bbs, occupancy_bbs, active_color = parse_fen(fen)
        is_special, score = evaluate_special_endgame(piece_bbs, occupancy_bbs, self.game_state, self.phase)
        self.assertFalse(is_special)

        # White KNN vs Black KP (KNNKP)
        fen = "8/8/8/8/8/2N5/1p6/1K2N2k w - - 0 1"
        piece_bbs, occupancy_bbs, active_color = parse_fen(fen)
        is_special, score = evaluate_special_endgame(piece_bbs, occupancy_bbs, self.game_state, self.phase)
        self.assertFalse(is_special)

        # Black KNN vs White KP (KNNKP)
        fen = "1k2n2K/1P6/2n5/8/8/8/8/8 b - - 0 1"
        piece_bbs, occupancy_bbs, active_color = parse_fen(fen)
        is_special, score = evaluate_special_endgame(piece_bbs, occupancy_bbs, self.game_state, self.phase)
        self.assertFalse(is_special)

if __name__ == "__main__":
    unittest.main()
