
import unittest
import numpy as np
import numba
from chess_engine.see import see, see_ge
from chess_engine.constants import MG_MATERIAL_VALUES, BB_SQUARES
from chess_engine.engine_types import WHITE, BLACK
from chess_engine.fen_parser import parse_fen

class TestSEE(unittest.TestCase):
    def test_scandinavian_exchange(self):
        # 1. e4 d5. White P e4 captures d5. Black Q d8 can recapture.
        # FEN: rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1
        fen = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"
        piece_bbs, occupancy_bbs, game_state = parse_fen(fen)

        from_sq = 28 # e4
        to_sq = 35   # d5

        # P(100) x P(100). Q(900) x P(100).
        # White gain: 100.
        # Black gain: 100 (recapture).
        # Net for White: 0.

        val = see(piece_bbs, occupancy_bbs, WHITE, from_sq, to_sq)
        self.assertEqual(val, 0)

        # Check see_ge
        self.assertTrue(see_ge(piece_bbs, occupancy_bbs, WHITE, from_sq, to_sq, 0))
        self.assertTrue(see_ge(piece_bbs, occupancy_bbs, WHITE, from_sq, to_sq, -50))
        self.assertFalse(see_ge(piece_bbs, occupancy_bbs, WHITE, from_sq, to_sq, 1))

    def test_hanging_piece(self):
        # Remove Black Queen so d5 is hanging.
        fen = "rnb1kbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"
        piece_bbs, occupancy_bbs, game_state = parse_fen(fen)

        from_sq = 28 # e4
        to_sq = 35   # d5

        # P x P. No recapture.
        # Gain 100.
        val = see(piece_bbs, occupancy_bbs, WHITE, from_sq, to_sq)
        self.assertEqual(val, 100)

        self.assertTrue(see_ge(piece_bbs, occupancy_bbs, WHITE, from_sq, to_sq, 100))
        self.assertFalse(see_ge(piece_bbs, occupancy_bbs, WHITE, from_sq, to_sq, 101))

    def test_bad_capture(self):
        # Queen captures protected Pawn.
        # White Q d1. Black P d5, protected by P c6.
        fen = "rnbqkbnr/pp2pppp/2p5/3p4/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        piece_bbs, occupancy_bbs, game_state = parse_fen(fen)

        from_sq = 3 # d1 (Queen)
        to_sq = 35  # d5 (Pawn)

        # Q(900) x P(100). P(100) x Q(900).
        # White gain: 100.
        # Black gain: 900.
        # Net: -800.
        val = see(piece_bbs, occupancy_bbs, WHITE, from_sq, to_sq)
        self.assertEqual(val, -800)

        self.assertFalse(see_ge(piece_bbs, occupancy_bbs, WHITE, from_sq, to_sq, 0))
        self.assertTrue(see_ge(piece_bbs, occupancy_bbs, WHITE, from_sq, to_sq, -800))
        self.assertFalse(see_ge(piece_bbs, occupancy_bbs, WHITE, from_sq, to_sq, -799))

if __name__ == '__main__':
    unittest.main()
