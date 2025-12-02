
import unittest
import numpy as np
import numba
from chess_engine.see import see, see_ge
from chess_engine.constants import MG_MATERIAL_VALUES, BB_SQUARES
from chess_engine.engine_types import WHITE, BLACK
from chess_engine.fen_parser import parse_fen

class TestSEE(unittest.TestCase):
    def test_scandinavian_exchange(self):
        fen = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"
        piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
        from_sq = 28 # e4
        to_sq = 35   # d5
        val = see(piece_bbs, occupancy_bbs, WHITE, from_sq, to_sq)
        self.assertEqual(val, 0)
        self.assertTrue(see_ge(piece_bbs, occupancy_bbs, WHITE, from_sq, to_sq, 0))

    def test_hanging_piece(self):
        fen = "rnb1kbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"
        piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
        from_sq = 28 # e4
        to_sq = 35   # d5
        val = see(piece_bbs, occupancy_bbs, WHITE, from_sq, to_sq)
        self.assertEqual(val, 100)
        self.assertTrue(see_ge(piece_bbs, occupancy_bbs, WHITE, from_sq, to_sq, 100))

    def test_bad_capture(self):
        fen = "rnbqkbnr/pp2pppp/2p5/3p4/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
        from_sq = 3 # d1 (Queen)
        to_sq = 35  # d5 (Pawn)
        val = see(piece_bbs, occupancy_bbs, WHITE, from_sq, to_sq)
        self.assertEqual(val, -800)
        self.assertFalse(see_ge(piece_bbs, occupancy_bbs, WHITE, from_sq, to_sq, 0))

    def test_quiet_move_to_safe_square(self):
        # White Knight on b1. Move to c3 (safe).
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
        from_sq = 1 # b1
        to_sq = 18  # c3

        # Logic:
        # Victim value = 0.
        # Moved piece = N (320).
        # Swap = 0 - threshold.
        # Swap = 320 - Swap = 320 + threshold.
        # If threshold is e.g. -50. Swap = 270.
        # Attackers to c3? None.
        # Loop ends. Result is 1 (True).
        # see() should return 0 (no gain, no loss).

        val = see(piece_bbs, occupancy_bbs, WHITE, from_sq, to_sq)
        print(f"Quiet Safe SEE: {val}")
        self.assertEqual(val, 0)

        # Check against negative threshold (History guard often uses negative)
        # see_ge(..., -50) -> 0 >= -50 -> True.
        self.assertTrue(see_ge(piece_bbs, occupancy_bbs, WHITE, from_sq, to_sq, -50))

    def test_quiet_move_to_hung_square(self):
        # White Knight on b1. Move to d5 (attacked by Queen on d8).
        fen = "rnbqkbnr/pppppppp/8/3n4/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
        # Wait, fen has N on d5? No, I want to move TO d5.
        # Setup: White N on b1. Black Q on d8. d5 is empty.
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        # N on b1. Q on d8.
        # d5 (35) is attacked by Q on d8 (59)? No, pawn on d7 blocks.
        # Let's open the d file.
        fen = "rnbqkbnr/ppp1pppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        # d7 pawn moved to d6/d5? No, just missing.
        # Now d8 Queen attacks d5? Yes.

        piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
        from_sq = 1 # b1
        to_sq = 35  # d5 (empty)

        # Logic:
        # Victim: 0.
        # Moved: N (320).
        # 1. Capture (Move): Gain 0.
        # 2. Recapture (QxN): Gain 320.
        # Net for White: 0 - 320 = -320.

        val = see(piece_bbs, occupancy_bbs, WHITE, from_sq, to_sq)
        print(f"Quiet Hung SEE: {val}")
        self.assertEqual(val, -320)

        self.assertFalse(see_ge(piece_bbs, occupancy_bbs, WHITE, from_sq, to_sq, 0))
        self.assertTrue(see_ge(piece_bbs, occupancy_bbs, WHITE, from_sq, to_sq, -320))
        self.assertFalse(see_ge(piece_bbs, occupancy_bbs, WHITE, from_sq, to_sq, -319))

if __name__ == '__main__':
    unittest.main()
