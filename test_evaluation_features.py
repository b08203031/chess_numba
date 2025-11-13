# test_evaluation_features.py
import unittest
import numpy as np

from chess_engine.fen_parser import parse_fen
from chess_engine.evaluation import _evaluate_pawn_shield_for_color, _evaluate_king_attackers, _evaluate_king_tropism, _evaluate_pawn_storm, evaluate_king_safety
from chess_engine.zobrist import get_lsb_index
from chess_engine.constants import KING_SAFETY_TABLE, KING_TROPISM_MAX_DISTANCE, KING_TROPISM_WEIGHTS, PAWN_STORM_PENALTY
from chess_engine.evaluation import MANHATTAN_DISTANCE

class TestPawnShieldEvaluation(unittest.TestCase):
    # ... (tests remain the same) ...
    def test_perfect_pawn_shield(self):
        fen = "rnbqk2r/pppppppp/8/8/8/8/PPPPPPPP/RNBQ1RK1 w kq - 0 1"
        piece_bbs, _, _ = parse_fen(fen)
        wk_sq = get_lsb_index(piece_bbs[5])
        score = _evaluate_pawn_shield_for_color(wk_sq, piece_bbs[0], piece_bbs[6], 0)
        self.assertEqual(score, 30)
    def test_pawn_advanced_one_step(self):
        fen = "rnbqk2r/pppppppp/8/8/8/5P2/PPPPP1PP/RNBQ1RK1 w kq - 0 1"
        piece_bbs, _, _ = parse_fen(fen)
        wk_sq = get_lsb_index(piece_bbs[5])
        score = _evaluate_pawn_shield_for_color(wk_sq, piece_bbs[0], piece_bbs[6], 0)
        self.assertEqual(score, 25)
    def test_pawn_advanced_two_steps(self):
        fen = "rnbqk2r/pppppppp/8/8/5P2/8/PPPPP1PP/RNBQ1RK1 w kq - 0 1"
        piece_bbs, _, _ = parse_fen(fen)
        wk_sq = get_lsb_index(piece_bbs[5])
        score = _evaluate_pawn_shield_for_color(wk_sq, piece_bbs[0], piece_bbs[6], 0)
        self.assertEqual(score, 10)
    def test_missing_pawn(self):
        fen = "rnbqk2r/pppppppp/8/8/8/8/PPPPP1PP/RNBQ1RK1 w kq - 0 1"
        piece_bbs, _, _ = parse_fen(fen)
        piece_bbs[0] &= ~np.uint64(1 << 13)
        wk_sq = get_lsb_index(piece_bbs[5])
        score = _evaluate_pawn_shield_for_color(wk_sq, piece_bbs[0], piece_bbs[6], 0)
        self.assertEqual(score, -10)
    def test_open_file(self):
        fen = "rnbqk2r/ppppp1pp/8/8/8/8/PPPPP1PP/RNBQ1RK1 w kq - 0 1"
        piece_bbs, _, _ = parse_fen(fen)
        piece_bbs[0] &= ~np.uint64(1 << 13)
        wk_sq = get_lsb_index(piece_bbs[5])
        score = _evaluate_pawn_shield_for_color(wk_sq, piece_bbs[0], piece_bbs[6], 0)
        self.assertEqual(score, -20)
    def test_uncastled_king_in_center(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        piece_bbs, _, _ = parse_fen(fen)
        wk_sq = get_lsb_index(piece_bbs[5])
        score = _evaluate_pawn_shield_for_color(wk_sq, piece_bbs[0], piece_bbs[6], 0)
        self.assertEqual(score, 30)

class TestKingAttackerEvaluation(unittest.TestCase):
    # ... (tests remain the same) ...
    def test_no_attackers(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        piece_bbs, occupancy_bbs, _ = parse_fen(fen)
        wk_sq = get_lsb_index(piece_bbs[5])
        bk_sq = get_lsb_index(piece_bbs[11])
        white_penalty = _evaluate_king_attackers(wk_sq, 0, piece_bbs, occupancy_bbs)
        black_penalty = _evaluate_king_attackers(bk_sq, 1, piece_bbs, occupancy_bbs)
        self.assertEqual(white_penalty, 0)
        self.assertEqual(black_penalty, 0)
    def test_one_attacker(self):
        fen = "rnbq1rk1/pppppppp/8/8/8/5n2/PPPPPPPP/RNBQ1RK1 w - - 0 1"
        piece_bbs, occupancy_bbs, _ = parse_fen(fen)
        wk_sq = get_lsb_index(piece_bbs[5])
        white_penalty = _evaluate_king_attackers(wk_sq, 0, piece_bbs, occupancy_bbs)
        self.assertEqual(white_penalty, 0)
    def test_two_knights_attacking(self):
        fen = "rnbq1rk1/pppppppp/8/8/8/5n1n/PPPPPPPP/RNBQ1RK1 w - - 0 1"
        piece_bbs, occupancy_bbs, _ = parse_fen(fen)
        wk_sq = get_lsb_index(piece_bbs[5])
        expected_penalty = -KING_SAFETY_TABLE[4]
        white_penalty = _evaluate_king_attackers(wk_sq, 0, piece_bbs, occupancy_bbs)
        self.assertEqual(white_penalty, expected_penalty)
    def test_queen_and_rook_attacking_corrected(self):
        fen = "rnb1k1r1/pppppppp/8/8/6qr/8/PPPPPPPP/RNBQ1RK1 w q - 0 1"
        piece_bbs, occupancy_bbs, _ = parse_fen(fen)
        wk_sq = get_lsb_index(piece_bbs[5])
        expected_penalty = -KING_SAFETY_TABLE[8]
        white_penalty = _evaluate_king_attackers(wk_sq, 0, piece_bbs, occupancy_bbs)
        self.assertEqual(white_penalty, expected_penalty)
    def test_slider_attack_blocked_outside_zone(self):
        fen = "rnbq1rk1/pppppp1p/8/8/8/8/PPPPPPPP/RNBQ1RK1 w - - 0 1"
        piece_bbs, occupancy_bbs, _ = parse_fen(fen)
        wk_sq = get_lsb_index(piece_bbs[5])
        white_penalty = _evaluate_king_attackers(wk_sq, 0, piece_bbs, occupancy_bbs)
        self.assertEqual(white_penalty, 0)

class TestKingTropismEvaluation(unittest.TestCase):
    # ... (tests remain the same) ...
    def test_single_piece_tropism(self):
        fen = "4k3/8/8/8/4q3/8/8/4K3 w - - 0 1"
        piece_bbs, _, _ = parse_fen(fen)
        wk_sq = get_lsb_index(piece_bbs[5])
        expected_penalty = -55
        tropism_penalty = _evaluate_king_tropism(wk_sq, 0, piece_bbs)
        self.assertEqual(tropism_penalty, expected_penalty)
    def test_multiple_pieces_tropism(self):
        fen = "4k3/8/8/8/4q3/8/8/r3K3 w - - 0 1"
        piece_bbs, _, _ = parse_fen(fen)
        wk_sq = get_lsb_index(piece_bbs[5])
        expected_penalty = -(55 + 30)
        tropism_penalty = _evaluate_king_tropism(wk_sq, 0, piece_bbs)
        self.assertEqual(tropism_penalty, expected_penalty)

class TestAdvancedKingSafety(unittest.TestCase):

    def test_pawn_storm(self):
        # White King on g1, Black pawn on f3, which is in the king zone
        fen = "rnbq1rk1/pppp1ppp/8/8/8/5p2/PPPPPPPP/RNBQ1RK1 w - - 0 1"
        piece_bbs, _, _ = parse_fen(fen)
        wk_sq = get_lsb_index(piece_bbs[5])

        # Expected: 1 pawn in zone * penalty
        expected_penalty = 1 * PAWN_STORM_PENALTY
        pawn_storm_penalty = _evaluate_pawn_storm(wk_sq, 0, piece_bbs)
        self.assertEqual(pawn_storm_penalty, expected_penalty)

    def test_scaling_factor_logic(self):
        # This test now focuses purely on the scaling logic.
        # It creates a situation with a clear attack and verifies that
        # removing an attacking piece reduces the penalty, due to scaling.

        # Black king on g8 is attacked by a White Knight on f6
        fen_attack = "rnbq1rk1/pppp1ppp/5n2/8/8/8/PPPPPPPP/RNBQ1RK1 b - - 0 1"
        p_bbs, o_bbs, _ = parse_fen(fen_attack)
        # Score is from the current player's (Black's) perspective, so a negative score is expected
        score_with_knight, _ = evaluate_king_safety(p_bbs, o_bbs)

        # Remove the attacking White knight
        fen_no_knight = "rnbq1rk1/pppp1ppp/8/8/8/8/PPPPPPPP/RNBQ1RK1 b - - 0 1"
        p_bbs_no_k, o_bbs_no_k, _ = parse_fen(fen_no_knight)
        score_no_knight, _ = evaluate_king_safety(p_bbs_no_k, o_bbs_no_k)

        # The penalty for Black should be less severe (closer to zero)
        # when the attacking knight is removed.
        # Since scores are from the current player's perspective,
        # score_no_knight (less penalty) should be GREATER than score_with_knight (more penalty).
        self.assertTrue(score_no_knight > score_with_knight)


if __name__ == '__main__':
    unittest.main()
