"""
Tests for Passed Pawn Order of Operations (ORD-1).
Verifies that unobstructed true passed pawns strictly outscore candidate
or pawn-blocked passers, and that edge distance penalty (PassedFile) is
deducted after candidate halving, preventing the inverted score bonus bug.
"""

import unittest
import numpy as np
import chess_engine.classical.evaluation as ce
from chess_engine.classical.fen_parser import parse_fen


class TestPassedPawnOrder(unittest.TestCase):
    def test_white_d4_true_passer_outscores_candidate(self):
        """
        White passed pawn on d4 (rank=3, sq=27).
        True passer: no enemy pawns controlling path.
        Candidate passer: black pawn on e6 (sq=44) controlling d5 (next_sq + 8).
        With ORD-1 fix, true passer must score higher than candidate passer in both MG and EG.
        """
        w_passed = np.uint64(1) << np.uint64(27)
        b_passed = np.uint64(0)

        # True passer
        fen_true = '4k3/8/8/8/3P4/8/8/4K3 w - - 0 1'
        pb_t, occ_t, _ = parse_fen(fen_true)
        res_t = ce.evaluate_passed_pawns(
            pb_t, occ_t, np.uint64(0), np.uint64(0),
            w_passed, b_passed, 4, 60, ce._DUMMY_EW, False
        )

        # Candidate passer (black pawn on e6)
        fen_cand = '4k3/8/4p3/8/3P4/8/8/4K3 w - - 0 1'
        pb_c, occ_c, _ = parse_fen(fen_cand)
        res_c = ce.evaluate_passed_pawns(
            pb_c, occ_c, np.uint64(0), np.uint64(0),
            w_passed, b_passed, 4, 60, ce._DUMMY_EW, False
        )

        # res = (mg, eg, count)
        self.assertGreater(
            res_t[0], res_c[0],
            f"True passer MG ({res_t[0]}) must strictly exceed candidate passer MG ({res_c[0]})"
        )
        self.assertGreater(
            res_t[1], res_c[1],
            f"True passer EG ({res_t[1]}) must strictly exceed candidate passer EG ({res_c[1]})"
        )

    def test_white_d4_true_passer_outscores_blocked_passer(self):
        """
        White passed pawn on d4 (sq=27).
        Blocked passer: black pawn directly on d5 (sq=35, any_pawn_ahead=True).
        """
        w_passed = np.uint64(1) << np.uint64(27)
        b_passed = np.uint64(0)

        fen_true = '4k3/8/8/8/3P4/8/8/4K3 w - - 0 1'
        pb_t, occ_t, _ = parse_fen(fen_true)
        res_t = ce.evaluate_passed_pawns(
            pb_t, occ_t, np.uint64(0), np.uint64(0),
            w_passed, b_passed, 4, 60, ce._DUMMY_EW, False
        )

        fen_blk = '4k3/8/8/3p4/3P4/8/8/4K3 w - - 0 1'
        pb_b, occ_b, _ = parse_fen(fen_blk)
        res_b = ce.evaluate_passed_pawns(
            pb_b, occ_b, np.uint64(0), np.uint64(0),
            w_passed, b_passed, 4, 60, ce._DUMMY_EW, False
        )

        self.assertGreater(
            res_t[0], res_b[0],
            f"True passer MG ({res_t[0]}) must exceed blocked passer MG ({res_b[0]})"
        )
        self.assertGreater(
            res_t[1], res_b[1],
            f"True passer EG ({res_t[1]}) must exceed blocked passer EG ({res_b[1]})"
        )

    def test_black_d5_true_passer_outscores_candidate(self):
        """
        Black passed pawn on d5 (sq=35, relative_rank=3).
        True passer: no white pawns controlling path.
        Candidate passer: white pawn on e3 (sq=20) controlling d4.
        Note evaluate_passed_pawns subtracts black passer bonuses from mg_score/eg_score.
        So for Black, more favorable bonus means lower (more negative) raw score.
        """
        w_passed = np.uint64(0)
        b_passed = np.uint64(1) << np.uint64(35)

        # True passer
        fen_true = '4k3/8/8/3p4/8/8/8/4K3 w - - 0 1'
        pb_t, occ_t, _ = parse_fen(fen_true)
        res_t = ce.evaluate_passed_pawns(
            pb_t, occ_t, np.uint64(0), np.uint64(0),
            w_passed, b_passed, 4, 60, ce._DUMMY_EW, False
        )

        # Candidate passer (white pawn on e3 controlling d4)
        fen_cand = '4k3/8/8/3p4/8/4P3/8/4K3 w - - 0 1'
        pb_c, occ_c, _ = parse_fen(fen_cand)
        res_c = ce.evaluate_passed_pawns(
            pb_c, occ_c, np.uint64(0), np.uint64(0),
            w_passed, b_passed, 4, 60, ce._DUMMY_EW, False
        )

        # Because Black bonus is subtracted from mg_score/eg_score:
        # True passer bonus > Candidate passer bonus <=> res_t < res_c
        self.assertLess(
            res_t[0], res_c[0],
            f"True black passer MG ({res_t[0]}) must be more negative than candidate passer MG ({res_c[0]})"
        )
        self.assertLess(
            res_t[1], res_c[1],
            f"True black passer EG ({res_t[1]}) must be more negative than candidate passer EG ({res_c[1]})"
        )

    def test_edge_penalty_not_halved_for_candidate(self):
        """
        On d3 (rank=2), base bonus is (0, 0).
        Edge distance = 3, file_mg = 16, file_eg = 4.
        True passer: bonus 0 - 48 = -48 MG, 0 - 12 = -12 EG.
        Candidate passer: bonus (0 // 2) - 48 = -48 MG, (0 // 2) - 12 = -12 EG.
        Under the old bug, candidate passer was (-48 // 2) = -24 MG, -6 EG (which gave it +24 cp!).
        Verify candidate passer on d3 receives -48 MG, NOT -24 MG.
        """
        w_passed = np.uint64(1) << np.uint64(19)
        b_passed = np.uint64(0)

        fen_cand = '4k3/8/8/4p3/8/3P4/8/4K3 w - - 0 1'
        pb_c, occ_c, _ = parse_fen(fen_cand)
        res_c = ce.evaluate_passed_pawns(
            pb_c, occ_c, np.uint64(0), np.uint64(0),
            w_passed, b_passed, 4, 60, ce._DUMMY_EW, False
        )

        expected_mg = ce._trunc_div(int(ce.PASSED_PAWN_BONUS[2, 0]), 2) - int(ce.PASSED_FILE_BONUS[0]) * 3
        expected_eg = ce._trunc_div(int(ce.PASSED_PAWN_BONUS[2, 1]), 2) - int(ce.PASSED_FILE_BONUS[1]) * 3
        self.assertEqual(res_c[0], expected_mg, f"Candidate passer MG must be {expected_mg}, got {res_c[0]}")
        self.assertEqual(res_c[1], expected_eg, f"Candidate passer EG must be {expected_eg}, got {res_c[1]}")

    def test_rank_4_pawn_receives_king_proximity(self):
        """
        ORD-2: Verify rank 3 (Rank 4 pawn, e.g. d4) receives King Proximity support.
        In the buggy code, threshold was `rank > 3`, which skipped rank 3 pawns entirely.
        With ORD-2 (`rank > 2`), moving friendly King close to d4 must increase EG score.
        """
        w_passed = np.uint64(1) << np.uint64(27)
        b_passed = np.uint64(0)

        # White King close at e4 (sq=28), Black King far at h8 (sq=63)
        fen_close = '7k/8/8/8/3PK3/8/8/8 w - - 0 1'
        pb_c, occ_c, _ = parse_fen(fen_close)
        res_close = ce.evaluate_passed_pawns(
            pb_c, occ_c, np.uint64(0), np.uint64(0),
            w_passed, b_passed, 28, 63, ce._DUMMY_EW, False
        )

        # White King far at a1 (sq=0), Black King far at h8 (sq=63)
        fen_far = '7k/8/8/8/3P4/8/8/K7 w - - 0 1'
        pb_f, occ_f, _ = parse_fen(fen_far)
        res_far = ce.evaluate_passed_pawns(
            pb_f, occ_f, np.uint64(0), np.uint64(0),
            w_passed, b_passed, 0, 63, ce._DUMMY_EW, False
        )

        self.assertGreater(
            res_close[1], res_far[1],
            f"Friendly King close EG ({res_close[1]}) must exceed King far EG ({res_far[1]}) for Rank 4 pawn"
        )
        self.assertGreater(res_close[1] - res_far[1], 0)

    def test_candidate_passer_halves_king_proximity(self):
        """
        ORD-2: Verify King Proximity bonus is accumulated into p_bonus_eg BEFORE candidate halving.
        For a true passer on d5 (rank 4), friendly King support is 40 EG.
        For a candidate passer on d5 (defended by black pawn on e7), King support must be halved to 20 EG.
        """
        w_passed = np.uint64(1) << np.uint64(35)
        b_passed = np.uint64(0)

        # True passer: King close (e5, sq=36) vs King far (a1, sq=0)
        fen_tc = '7k/8/8/3PK3/8/8/8/8 w - - 0 1'
        fen_tf = '7k/8/8/3P4/8/8/8/K7 w - - 0 1'
        pb_tc, occ_tc, _ = parse_fen(fen_tc)
        pb_tf, occ_tf, _ = parse_fen(fen_tf)
        res_tc = ce.evaluate_passed_pawns(pb_tc, occ_tc, np.uint64(0), np.uint64(0), w_passed, b_passed, 36, 63, ce._DUMMY_EW, False)
        res_tf = ce.evaluate_passed_pawns(pb_tf, occ_tf, np.uint64(0), np.uint64(0), w_passed, b_passed, 0, 63, ce._DUMMY_EW, False)
        delta_true = res_tc[1] - res_tf[1]

        # Candidate passer: black pawn on e7 (sq=52) controlling d6
        fen_cc = '7k/4p3/8/3PK3/8/8/8/8 w - - 0 1'
        fen_cf = '7k/4p3/8/3P4/8/8/8/K7 w - - 0 1'
        pb_cc, occ_cc, _ = parse_fen(fen_cc)
        pb_cf, occ_cf, _ = parse_fen(fen_cf)
        res_cc = ce.evaluate_passed_pawns(pb_cc, occ_cc, np.uint64(0), np.uint64(0), w_passed, b_passed, 36, 63, ce._DUMMY_EW, False)
        res_cf = ce.evaluate_passed_pawns(pb_cf, occ_cf, np.uint64(0), np.uint64(0), w_passed, b_passed, 0, 63, ce._DUMMY_EW, False)
        delta_cand = res_cc[1] - res_cf[1]

        self.assertGreater(delta_true, 0)
        self.assertGreater(delta_cand, 0)
        self.assertEqual(delta_cand, delta_true // 2)

    def test_trunc_div_semantics(self):
        """
        ORD-3: Verify _trunc_div truncates towards zero (C++ integer division semantics).
        In Python default floor division:
            -1 // 2 == -1  (floor towards -inf)
        In C++ / Stockfish 11:
            -1 / 2 == 0    (truncate towards zero)
            -3 / 2 == -1
            -4 / 2 == -2
        """
        self.assertEqual(ce._trunc_div(np.int32(5), np.int32(2)), 2)
        self.assertEqual(ce._trunc_div(np.int32(-5), np.int32(2)), -2)
        self.assertEqual(ce._trunc_div(np.int32(-1), np.int32(2)), 0)
        self.assertEqual(ce._trunc_div(np.int32(1), np.int32(2)), 0)
        self.assertEqual(ce._trunc_div(np.int32(-3), np.int32(2)), -1)
        self.assertEqual(ce._trunc_div(np.int32(-4), np.int32(2)), -2)


if __name__ == '__main__':
    unittest.main()

