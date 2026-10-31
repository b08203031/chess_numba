import unittest
import numpy as np

from chess_engine.classical.fen_parser import parse_fen
from chess_engine.classical.bitboard_utils import BB_SQUARES
from chess_engine.classical.evaluation import evaluate_attacks_mobility_threats


class TestPinnedEvalGeometry(unittest.TestCase):
    def test_white_queen_orthogonal_pin_geometry(self):
        # White Ke1 (sq 4), Qe3 (sq 20), Black Re8 (sq 60), Kh8 (sq 63)
        # Queen is strictly pinned on the e-file.
        # Off-ray diagonal attacks on Rank 1 (e.g. c1=sq 2, g1=sq 6) must NOT be present in queen attacks.
        fen = "4r2k/8/8/8/8/4Q3/8/4K3 w - - 0 1"
        piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
        res = evaluate_attacks_mobility_threats(
            piece_bbs, occupancy_bbs, 4, 63, np.uint64(0), np.uint64(0), np.uint8(0)
        )
        white_queen_attacks = res[17]
        pinned_white = res[6]

        self.assertTrue(bool(pinned_white & BB_SQUARES[20]), "White Queen on e3 must be detected as pinned")
        self.assertFalse(bool(white_queen_attacks & BB_SQUARES[2]), "Off-ray attack on c1 must be masked out")
        self.assertFalse(bool(white_queen_attacks & BB_SQUARES[6]), "Off-ray attack on g1 must be masked out")
        self.assertTrue(bool(white_queen_attacks & BB_SQUARES[28]), "Pin-ray attack on e4 must be retained")
        self.assertTrue(bool(white_queen_attacks & BB_SQUARES[12]), "Pin-ray attack on e2 must be retained")

    def test_white_queen_diagonal_pin_geometry(self):
        # White Ke1 (sq 4), Qg3 (sq 22), Black Bh4 (sq 31), Kh8 (sq 63)
        # Queen is strictly pinned on the e1-h4 diagonal (sq 4, 13, 22, 31).
        # Off-ray horizontal attack c3 (sq 18) lies on the other diagonal through e1 and must NOT be present.
        fen = "7k/8/8/8/7b/6Q1/8/4K3 w - - 0 1"
        piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
        res = evaluate_attacks_mobility_threats(
            piece_bbs, occupancy_bbs, 4, 63, np.uint64(0), np.uint64(0), np.uint8(0)
        )
        white_queen_attacks = res[17]
        pinned_white = res[6]

        self.assertTrue(bool(pinned_white & BB_SQUARES[22]), "White Queen on g3 must be detected as pinned")
        self.assertFalse(bool(white_queen_attacks & BB_SQUARES[18]), "Off-ray attack on c3 must be masked out")
        self.assertTrue(bool(white_queen_attacks & BB_SQUARES[31]), "Pin-ray attack on h4 must be retained")
        self.assertTrue(bool(white_queen_attacks & BB_SQUARES[13]), "Pin-ray attack on f2 must be retained")

    def test_black_queen_orthogonal_pin_geometry(self):
        # Black Ke8 (sq 60), Qe6 (sq 44), White Re1 (sq 4), Kh1 (sq 7)
        # Black Queen is strictly pinned on the e-file.
        # Off-ray diagonal attacks on Rank 8 (c8=sq 58, g8=sq 62) must NOT be present in queen attacks.
        fen = "4k3/8/4q3/8/8/8/8/4R2K b - - 0 1"
        piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
        res = evaluate_attacks_mobility_threats(
            piece_bbs, occupancy_bbs, 7, 60, np.uint64(0), np.uint64(0), np.uint8(0)
        )
        black_queen_attacks = res[21]
        pinned_black = res[7]

        self.assertTrue(bool(pinned_black & BB_SQUARES[44]), "Black Queen on e6 must be detected as pinned")
        self.assertFalse(bool(black_queen_attacks & BB_SQUARES[58]), "Off-ray attack on c8 must be masked out")
        self.assertFalse(bool(black_queen_attacks & BB_SQUARES[62]), "Off-ray attack on g8 must be masked out")
        self.assertTrue(bool(black_queen_attacks & BB_SQUARES[52]), "Pin-ray attack on e7 must be retained")
        self.assertTrue(bool(black_queen_attacks & BB_SQUARES[36]), "Pin-ray attack on e5 must be retained")

    def test_black_queen_diagonal_pin_geometry(self):
        # Black Ke8 (sq 60), Qg6 (sq 46), White Bh5 (sq 39), Kh1 (sq 7)
        # Queen is strictly pinned on the e8-h5 diagonal (sq 60, 53, 46, 39).
        # Off-ray horizontal attack c6 (sq 42) lies on the other diagonal through e8 and must NOT be present.
        fen = "4k3/8/6q1/7B/8/8/8/7K b - - 0 1"
        piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
        res = evaluate_attacks_mobility_threats(
            piece_bbs, occupancy_bbs, 7, 60, np.uint64(0), np.uint64(0), np.uint8(0)
        )
        black_queen_attacks = res[21]
        pinned_black = res[7]

        self.assertTrue(bool(pinned_black & BB_SQUARES[46]), "Black Queen on g6 must be detected as pinned")
        self.assertFalse(bool(black_queen_attacks & BB_SQUARES[42]), "Off-ray attack on c6 must be masked out")
        self.assertTrue(bool(black_queen_attacks & BB_SQUARES[39]), "Pin-ray attack on h5 must be retained")
        self.assertTrue(bool(black_queen_attacks & BB_SQUARES[53]), "Pin-ray attack on f7 must be retained")


if __name__ == "__main__":
    unittest.main()
