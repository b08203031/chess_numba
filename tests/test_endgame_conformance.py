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
from chess_engine.classical.endgame import (
    evaluate_special_endgame,
    get_endgame_scale_factor,
)
from chess_engine.classical.constants import SCALE_FACTOR_DRAW, TEMPO_BONUS
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

    # ----- P2: KNNK / KRKB / KRKN / KQKR / KNNKP -----
    def test_knnk_draw(self):
        # SF: KNNK is always VALUE_DRAW (must not fall into KXK)
        fen = "4k3/8/8/8/8/2N1N3/8/4K3 w - - 0 1"
        hit, score, val, *_ = self._special(fen)
        self.assertTrue(hit)
        self.assertEqual(score, 0)
        self.assertEqual(val, 0)

    def test_knnk_black_draw(self):
        fen = "4k3/2n1n3/8/8/8/8/8/4K3 w - - 0 1"
        hit, score, val, *_ = self._special(fen)
        self.assertTrue(hit)
        self.assertEqual(score, 0)

    def test_krkb_drawish(self):
        # SF: only PushToEdges; edge king scores higher than center for attacker
        fen_edge = "k7/8/8/8/3R4/8/8/4K3 w - - 0 1"   # black king a8
        fen_center = "4k3/8/8/8/3R4/8/8/4K3 w - - 0 1"  # black king e8-ish edge still
        # Use true center-ish: black king on e4
        fen_mid = "8/8/8/4k3/3R4/8/8/4K3 w - - 0 1"
        hit_e, score_e, val_e, *_ = self._special(fen_edge)
        hit_m, score_m, val_m, *_ = self._special(fen_mid)
        self.assertTrue(hit_e and hit_m)
        self.assertGreater(score_e, 0)
        self.assertGreater(score_m, 0)
        self.assertGreater(score_e, score_m)  # corner/edge > more central
        self.assertEqual(val_e, score_e)

    def test_krkb_black_strong(self):
        fen = "4k3/8/8/3r4/8/8/8/B3K3 w - - 0 1"
        hit, score, val, *_ = self._special(fen)
        self.assertTrue(hit)
        self.assertLess(score, 0)

    def test_krkn_better_when_king_far_from_knight(self):
        # PushAway: larger dist(king, knight) → higher score
        fen_far = "k7/8/8/8/3R4/8/8/4K1n1 w - - 0 1"  # Ka8, ng1 far
        fen_near = "k7/8/8/8/3R4/8/n7/4K3 w - - 0 1"  # Ka8, na2 closer
        hit_f, score_f, _, *_ = self._special(fen_far)
        hit_n, score_n, _, *_ = self._special(fen_near)
        self.assertTrue(hit_f and hit_n)
        self.assertGreater(score_f, score_n)

    def test_kqkr_white_advantage(self):
        # Queen vs rook: material + geometry, clearly positive for Q side
        fen = "4k3/8/8/8/3Q4/8/8/3rK3 w - - 0 1"
        hit, score, val, *_ = self._special(fen)
        self.assertTrue(hit)
        # EG queen-rook ≈ 950-530 = 420 plus edges
        self.assertGreater(score, 400)
        self.assertEqual(val, score)

    def test_kqkr_black_strong(self):
        fen = "3Rk3/8/8/8/3q4/8/8/4K3 w - - 0 1"
        hit, score, val, *_ = self._special(fen)
        self.assertTrue(hit)
        self.assertLess(score, -400)

    def test_knnkp(self):
        # Two knights vs king + pawn: ~2*N - P + edges > 0 for white
        fen = "4k3/4p3/8/8/8/2N1N3/8/4K3 w - - 0 1"
        hit, score, val, *_ = self._special(fen)
        self.assertTrue(hit)
        self.assertGreater(score, 0)
        self.assertEqual(val, score)

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

    def test_scale_function_only_applies_to_evaluated_strong_side(self):
        # White has the wrong-bishop rook-pawn fortress pattern, but Black's
        # extra pawns can make Black the evaluated strong side. SF11 only probes
        # the scale function registered for that strong side.
        fen = "k7/P7/8/8/8/3pp3/8/2B4K w - - 0 1"
        piece_bbs, occupancy_bbs, game_state = parse_fen(fen)

        white_sf = int(get_endgame_scale_factor(
            piece_bbs, occupancy_bbs, game_state, np.int32(0), np.int32(100)
        ))
        black_sf = int(get_endgame_scale_factor(
            piece_bbs, occupancy_bbs, game_state, np.int32(0), np.int32(-100)
        ))

        self.assertEqual(white_sf, int(SCALE_FACTOR_DRAW))
        self.assertNotEqual(black_sf, int(SCALE_FACTOR_DRAW))


    # ----- King-Pawn Endgame Hardening Tests -----
    def test_kpsk_scale_factor_draw(self):
        # Two rook pawns blocked by defending king in front -> theoretical draw (SF11 KPsK)
        fen = "8/k7/8/P7/P7/8/8/7K w - - 0 1"
        piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
        val, _, _ = evaluate_position(piece_bbs, occupancy_bbs, game_state)
        self.assertEqual(val, 0)

    def test_kpkp_scale_factor_draw(self):
        # Symmetrical blocked pawns (e4 vs e5, kings e2 vs e7) -> theoretical draw (SF11 KPKP)
        # Endgame<KPKP> returns SCALE_FACTOR_DRAW (0), which short-circuits pure K+P eval to 0.
        fen = "8/4k3/8/4p3/4P3/8/4K3/8 w - - 0 1"
        piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
        val, _, _ = evaluate_position(piece_bbs, occupancy_bbs, game_state)
        self.assertEqual(val, 0)

    def test_kpkp_asymmetric_bitbase_draw(self):
        # Asymmetrical KPKP: White has centralized Ke3 vs Black Kd8 (raw eval > 0),
        # but probing KPK bitbase without Black's e5 pawn (Ke3+Pe4 vs Kd8, Black to move) is a draw.
        fen = "3k4/8/8/4p3/4P3/4K3/8/8 b - - 0 1"
        piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
        sf = get_endgame_scale_factor(piece_bbs, occupancy_bbs, game_state, np.int32(0), np.int32(100))
        val, _, _ = evaluate_position(piece_bbs, occupancy_bbs, game_state)
        self.assertEqual(int(sf), int(SCALE_FACTOR_DRAW))
        self.assertEqual(val, 0)

    def test_krpkb_corner_trapped_king_returns_48(self):
        # SF11 Endgame<KRPKB>: rook pawn on 5th rank, same-color bishop, weak king in corner (d == 0).
        # If strong king is on a6 trapping weak king on a8 (ksq == wksq + 16), scale factor is 48 (not 24).
        fen_trapped = "k7/8/K7/P7/8/2b5/4R3/8 w - - 0 1"
        pb_t, occ_t, gs_t = parse_fen(fen_trapped)
        sf_trapped = get_endgame_scale_factor(pb_t, occ_t, gs_t, np.int32(0), np.int32(500))
        self.assertEqual(int(sf_trapped), 48)

        # Moving strong king to b6 avoids trapping weak king directly in front -> scale factor 24.
        fen_free = "k7/8/1K6/P7/8/2b5/4R3/8 w - - 0 1"
        pb_f, occ_f, gs_f = parse_fen(fen_free)
        sf_free = get_endgame_scale_factor(pb_f, occ_f, gs_f, np.int32(0), np.int32(500))
        self.assertEqual(int(sf_free), 24)

    def test_kqkrps_multiple_pawns_fortress_draw(self):
        # KQ vs KR + 2 pawns fortress: LSB pawn is h5 (not defending Rf6), second pawn g7
        # is in King zone of Kg8 and defends Rf6 -> bitwise mask must detect fortress draw.
        fen = "6k1/6p1/5r2/7p/4K3/8/8/3Q4 w - - 0 1"
        piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
        sf = get_endgame_scale_factor(piece_bbs, occupancy_bbs, game_state, np.int32(0), np.int32(500))
        self.assertEqual(int(sf), int(SCALE_FACTOR_DRAW))

    def test_pawn_hash_partial_hit_shelter_recompute(self):
        from chess_engine.classical.constants import MAX_PLY
        from chess_engine.classical.engine_types import SearchContext
        from chess_engine.classical.evaluation import _evaluate_position_jit
        from chess_engine.classical.transposition_table import create_transposition_table

        tt = create_transposition_table(1)
        ctx = SearchContext(
            tt,
            np.zeros(MAX_PLY * 2, dtype=np.uint16),
            np.zeros((MAX_PLY, MAX_PLY), dtype=np.uint16),
            np.zeros((12, 64), dtype=np.int32),
            np.zeros((64, 64), dtype=np.int32),
            np.zeros((5, 12, 64, 12, 64), dtype=np.int16),
            np.zeros((12, 64, 6), dtype=np.int32),
            np.full((8192, 12, 64), -1238, dtype=np.int16),
            np.zeros(16384, dtype=np.int16),
            np.zeros(16384, dtype=np.int16),
            np.zeros(16384, dtype=np.int16),
            np.zeros(16384, dtype=np.int16),
            np.zeros((12, 64, 12, 64), dtype=np.int16),
        )
        # Same pawn structure across all 4 FENs, varying White/Black king squares and castling rights
        fens = [
            "r3k2r/ppp2ppp/8/8/8/8/PPP2PPP/R3K2R w KQkq - 0 1",
            "r3k2r/ppp2ppp/8/8/8/8/PPP2PPP/R4K1R b kq - 1 1",   # White king moved + lost KQ rights
            "r4k1r/ppp2ppp/8/8/8/8/PPP2PPP/R4K1R w - - 2 2",    # Black king moved + lost kq rights
            "r3k2r/ppp2ppp/8/8/8/8/PPP2PPP/R3K2R w K - 0 1",    # White castling rights changed (K vs KQkq)
        ]
        for fen in fens:
            pb, occ, gs = parse_fen(fen)
            cached_val, _, _ = _evaluate_position_jit(pb, occ, gs, False, ctx)
            uncached_val, _, _ = _evaluate_position_jit(pb, occ, gs, False, None)
            self.assertEqual(int(cached_val), int(uncached_val), msg=f"Pawn hash shelter mismatch for {fen}")

    def test_doubled_pawns_front_blocks_rear_unstoppable(self):
        # White has doubled pawns on a2 and a3, black king far away on h8 with irrelevant h7 pawn.
        # Rear pawn on a2 cannot be unstoppable because a3 blocks its path.
        fen_doubled = "7k/7p/8/8/8/P7/P7/7K w - - 0 1"
        fen_single = "7k/7p/8/8/8/P7/8/7K w - - 0 1"
        p1, o1, g1 = parse_fen(fen_doubled)
        p2, o2, g2 = parse_fen(fen_single)
        v1, _, _ = evaluate_position(p1, o1, g1)
        v2, _, _ = evaluate_position(p2, o2, g2)
        diff = v1 - v2
        self.assertLess(diff, 500, f"Doubled pawn scored extra {diff} cp, likely double-counting unstoppable bonus!")
        self.assertGreater(diff, 0, f"Doubled pawn should still be slightly better due to extra material ({diff} cp)")

    def test_black_doubled_pawns_front_blocks_rear_unstoppable(self):
        # Mirror test for Black: rear pawn on a7 blocked by front pawn on a6
        fen_doubled = "7k/p7/p7/8/8/8/7P/7K b - - 0 1"
        fen_single = "7k/8/p7/8/8/8/7P/7K b - - 0 1"
        p1, o1, g1 = parse_fen(fen_doubled)
        p2, o2, g2 = parse_fen(fen_single)
        v1, _, _ = evaluate_position(p1, o1, g1)
        v2, _, _ = evaluate_position(p2, o2, g2)
        # evaluate_position is side-to-move relative (Black's turn, so positive is better for Black)
        diff_black = v1 - v2
        self.assertLess(diff_black, 500, f"Black doubled pawn scored extra {diff_black} cp, likely double-counting unstoppable bonus!")
        self.assertGreater(diff_black, 0, f"Black doubled pawn should still be slightly better due to extra material ({diff_black} cp)")

    def test_king_on_path_blocks_unstoppable_pawn(self):
        # 1. Advanced pawn: White Ka5 blocks Pa4. Black Kf4 is at distance 5 from a8.
        # With Ka5 blocking, Pa4 needs (4 + 1) = 5 steps, so Black king (dist 5) can catch it -> no unstoppable bonus.
        # If White king is moved away to b1, Pa4 needs 4 steps -> unstoppable bonus granted!
        fen_blocked = "8/7p/8/K7/P4k2/8/8/8 w - - 0 1"
        fen_free = "8/7p/8/8/P4k2/8/8/1K6 w - - 0 1"
        v_blocked, _, _ = evaluate_position(*parse_fen(fen_blocked))
        v_free, _, _ = evaluate_position(*parse_fen(fen_free))
        self.assertGreater(v_free - v_blocked, 700, "Unblocked pawn must score unstoppable bonus (+800 cp) over blocked pawn")

        # 2. Starting pawn: White Ka3 blocks Pa2.
        # With Ka3 stepping aside, Pa2 can double push: 1 (king) + 5 (pawn) = 6 steps to queen.
        # If Black king is at distance 6 (g4), Black can catch it -> no unstoppable bonus.
        # If Black king is at distance 7 (h4), Black cannot catch it -> unstoppable bonus granted!
        fen_start_catchable = "8/7p/8/8/6k1/K7/P7/8 w - - 0 1"
        fen_start_unstoppable = "8/7p/8/8/7k/K7/P7/8 w - - 0 1"
        v_catchable, _, _ = evaluate_position(*parse_fen(fen_start_catchable))
        v_unstoppable, _, _ = evaluate_position(*parse_fen(fen_start_unstoppable))
        self.assertGreater(v_unstoppable - v_catchable, 700, "Pa2 with Ka3 on path must queen in 6 steps and be unstoppable against dist 7 king")

        # 3. Mirror test for Black: Black pawn a7 with Ka6 on path vs White king at distance 7 (h5).
        fen_black_unstoppable = "8/p7/k7/7K/8/8/7P/8 b - - 0 1"
        v_b, _, _ = evaluate_position(*parse_fen(fen_black_unstoppable))
        self.assertGreater(v_b, 500, "Black a7 pawn with Ka6 on path must queen in 6 steps and be unstoppable against dist 7 king (scaled by sf)")

    def test_kbpsk_wrong_bishop_distance_2_draw(self):
        # White dark-squared bishop + a3 pawn vs Black king on c7 (distance 2 from a8 light square)
        fen = "8/2k3B1/8/8/8/P7/8/5K2 w - - 0 1"
        piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
        sf = get_endgame_scale_factor(piece_bbs, occupancy_bbs, game_state, np.int32(0), np.int32(500))
        self.assertEqual(int(sf), int(SCALE_FACTOR_DRAW))

    def test_kbpsk_lazy_eval_draw(self):
        # Wrong bishop + a-pawn fortress under lazy=True must not leak unscaled material (+648 cp)
        from chess_engine.classical.evaluation import _evaluate_position_jit
        fen = "8/Pk6/1B6/1K6/8/8/8/8 w - - 1 72"
        piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
        eval_lazy, _, _ = _evaluate_position_jit(piece_bbs, occupancy_bbs, game_state, True, None)
        self.assertEqual(int(eval_lazy), 0)

    def test_kbpsk_barrier_winning_position_not_draw(self):
        # White Kb6, Pa6, Bd5 (dark bishop) vs Black Kc8
        # Black king on c8 is locked out by a6 pawn (controlling b7) and dark bishop (controlling b8)
        # Must not be falsely judged as SCALE_FACTOR_DRAW
        fen = "2k5/8/KP6/3B4/8/8/8/8 b - - 0 1"
        piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
        sf = get_endgame_scale_factor(piece_bbs, occupancy_bbs, game_state, np.int32(0), np.int32(500))
        self.assertNotEqual(int(sf), int(SCALE_FACTOR_DRAW))

    def test_pure_pawn_winning_eval_not_discounted(self):
        # Winning pure king-and-pawn endgame with unstoppable pawn must retain full score (~10000 cp)
        # and must not be discounted by generic pawn scale factor (36 + 7 * wp)
        fen = "7k/8/8/8/8/8/P7/7K w - - 0 1"
        piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
        val, _, _ = evaluate_position(piece_bbs, occupancy_bbs, game_state)
        self.assertGreater(val, 9000)


if __name__ == "__main__":
    unittest.main()

