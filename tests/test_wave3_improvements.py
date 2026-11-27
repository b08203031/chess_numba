"""Wave 3 high-ROI improvement tests.

Currently covers:
- Reverted Wave 3-1 (S-1) side-exclusive Non-Pawn Correction History contract.
- Wave 3-2 (S-10): SF19 effort-weighted averageScore aspiration window center,
  tighter initial bounds (MIN=10, BASE=14), and GROW_NUM=47 in both main and
  Lazy SMP helper iterative deepening loops.
"""
import unittest
from pathlib import Path

import numpy as np

from chess_engine.classical.constants import (
    ASPIRATION_EMA_CHI_DEN,
    ASPIRATION_EMA_CHI_NUM,
    ASPIRATION_EMA_MAX_WEIGHT,
    ASPIRATION_EMA_MIN_WEIGHT,
    ASPIRATION_EMA_SCALE,
    ASPIRATION_WINDOW_BASE,
    ASPIRATION_WINDOW_GROW_NUM,
    ASPIRATION_WINDOW_MIN,
    CORRECTION_HISTORY_SIZE,
    MATE_IN_MAX_PLY,
    MAX_PLY,
    NO_MOVE,
    PAWN_HISTORY_SIZE,
    VALUE_KNOWN_WIN,
)
from chess_engine.classical.core import generate_legal_moves
from chess_engine.classical.engine_types import SearchContext, reset_search_context
from chess_engine.classical.fen_parser import parse_fen
from chess_engine.classical.search import (
    _update_aspiration_average_score,
    collect_hist_diag_dict,
    iterative_deepening_search,
)
from chess_engine.classical.transposition_table import (
    clear_transposition_table,
    create_transposition_table,
)

ROOT = Path(__file__).resolve().parent.parent


def _make_context(tt_mb: int = 4):
    tt = create_transposition_table(tt_mb)
    ctx = SearchContext(
        tt,
        np.zeros(MAX_PLY * 2, dtype=np.uint16),
        np.zeros((MAX_PLY, MAX_PLY), dtype=np.uint16),
        np.zeros((12, 64), dtype=np.int32),
        np.zeros((64, 64), dtype=np.int32),
        np.zeros((5, 12, 64, 12, 64), dtype=np.int16),
        np.zeros((12, 64, 6), dtype=np.int32),
        np.full((PAWN_HISTORY_SIZE, 12, 64), -1238, dtype=np.int16),
        np.zeros(CORRECTION_HISTORY_SIZE, dtype=np.int16),
        np.zeros(CORRECTION_HISTORY_SIZE, dtype=np.int16),
        np.zeros(CORRECTION_HISTORY_SIZE, dtype=np.int16),
        np.zeros(CORRECTION_HISTORY_SIZE, dtype=np.int16),
        np.zeros((12, 64, 12, 64), dtype=np.int16),
    )
    return tt, ctx


class TestWave3S10AspirationAverageScore(unittest.TestCase):
    def test_s1_reverted_and_s10_source_contracts(self):
        src = (ROOT / "chess_engine" / "classical" / "search.py").read_text(encoding="utf-8")

        apply_start = src.index("def apply_correction_history_score(")
        apply_end = src.index("# Continuation Correction Heuristic", apply_start)
        apply_block = src[apply_start:apply_end]
        self.assertNotIn("np_sum = global_np_w + global_np_b", apply_block)
        self.assertIn("if side == WHITE:", apply_block)

        upd_start = src.index("# Update Non-Pawn Correction (side-exclusive")
        upd_end = src.index("# Update Continuation Correction History", upd_start)
        upd_block = src[upd_start:upd_end]
        self.assertIn("if side == WHITE:", upd_block)
        self.assertIn("bonus * CORRECTION_NP_SCALE_NUM", upd_block)

        helper_start = src.index("def _lazy_smp_helper_loop_jit(")
        helper_end = src.index("def select_best_thread(", helper_start)
        helper_block = src[helper_start:helper_end]
        self.assertIn("has_average_score", helper_block)
        self.assertIn("ASPIRATION_WINDOW_GROW_NUM", helper_block)
        self.assertIn("ASPIRATION_EMA_MIN_WEIGHT", helper_block)

        main_start = src.index("def iterative_deepening_search(")
        main_end = src.index("def format_score_for_uci(", main_start)
        main_block = src[main_start:main_end]
        self.assertIn("_update_aspiration_average_score(", main_block)
        self.assertIn("ASPIRATION_WINDOW_GROW_NUM", main_block)
        self.assertEqual(ASPIRATION_WINDOW_MIN, 10)
        self.assertEqual(ASPIRATION_WINDOW_BASE, 14)
        self.assertEqual(ASPIRATION_WINDOW_GROW_NUM, 47)

    def test_aspiration_ema_math_bounds_and_decisive_reset(self):
        # 1. First iteration initializes averageScore verbatim
        avg, eff, has_avg = _update_aspiration_average_score(35, 0, 0, False, 120)
        self.assertEqual((avg, eff, has_avg), (35, 120, True))

        # 2. Extreme low step_nodes clamps weight to MIN_WEIGHT (12 / 32)
        # w = 12 => (100 * 12 + 0 * 20) // 32 = 1200 // 32 = 37
        avg_min_w, eff2, _ = _update_aspiration_average_score(100, 0, 10**9, True, 1)
        self.assertEqual(ASPIRATION_EMA_MIN_WEIGHT, 12)
        self.assertEqual(avg_min_w, (100 * ASPIRATION_EMA_MIN_WEIGHT) // ASPIRATION_EMA_SCALE)
        self.assertEqual(eff2, 10**9 + 1)

        # 3. Extreme high step_nodes clamps weight to MAX_WEIGHT (24 / 32)
        avg_max_w, _, _ = _update_aspiration_average_score(100, 0, 1, True, 10**6)
        self.assertEqual(ASPIRATION_EMA_MAX_WEIGHT, 24)
        self.assertEqual(avg_max_w, (100 * ASPIRATION_EMA_MAX_WEIGHT) // ASPIRATION_EMA_SCALE)

        # 4. Typical geometric growth factor: n_eff == prev_e -> w = 32*2 / (2+3) = 12
        w_eq = (ASPIRATION_EMA_SCALE * ASPIRATION_EMA_CHI_DEN) // (
            ASPIRATION_EMA_CHI_DEN + ASPIRATION_EMA_CHI_NUM
        )
        self.assertEqual(w_eq, 12)

        # 5. Symmetric truncation toward zero prevents downward drift on negative scores
        # -100 * 12 + (-101) * 20 = -3220 -> -100 (whereas Python // would yield -101,
        # and -1 * 12 + 0 * 20 = -12 -> trunc toward zero yields 0 instead of -1)
        avg_neg, _, _ = _update_aspiration_average_score(-1, 0, 10**9, True, 1)
        self.assertEqual(avg_neg, 0)

        # 6. Crossing into/out of decisive or mate range resets cleanly without blending
        mate_sc = MATE_IN_MAX_PLY + 5
        avg_dec1, eff_dec1, _ = _update_aspiration_average_score(mate_sc, 40, 5000, True, 800)
        self.assertEqual((avg_dec1, eff_dec1), (mate_sc, 800))

        avg_dec2, eff_dec2, _ = _update_aspiration_average_score(50, VALUE_KNOWN_WIN, 5000, True, 900)
        self.assertEqual((avg_dec2, eff_dec2), (50, 900))

    def test_aspiration_search_single_and_smp_smoke(self):
        tt, ctx = _make_context(tt_mb=4)

        # Startpos 1T with diagnostics enabled
        clear_transposition_table(tt)
        reset_search_context(ctx)
        ctx.diag_enabled = True
        pb, ob, gs = parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        move1, score1, ns1, qs1, _th1, d1 = iterative_deepening_search(
            pb, ob, gs, 7, {"optimum_time": 0, "maximum_time": 0},
            ctx, verbose=False, num_threads=1,
        )
        self.assertNotEqual(int(move1), int(NO_MOVE))
        self.assertEqual(int(d1), 7)
        legal1 = [int(m) for m in generate_legal_moves(pb, ob, gs)]
        self.assertIn(int(move1), legal1)
        diag = collect_hist_diag_dict(ctx, int(ns1) + int(qs1))
        self.assertGreaterEqual(diag["aspiration_iterations"], 6)
        self.assertGreaterEqual(diag["aspiration_max_delta"], ASPIRATION_WINDOW_MIN)

        # Startpos 2T Lazy SMP helper loop compiles and chooses a legal move
        clear_transposition_table(tt)
        reset_search_context(ctx)
        move2, _score2, _ns2, _qs2, _th2, d2 = iterative_deepening_search(
            pb, ob, gs, 7, {"optimum_time": 0, "maximum_time": 0},
            ctx, verbose=False, num_threads=2,
        )
        self.assertNotEqual(int(move2), int(NO_MOVE))
        self.assertEqual(int(d2), 7)
        self.assertIn(int(move2), legal1)

        # Kiwipete 1T
        clear_transposition_table(tt)
        reset_search_context(ctx)
        pb_k, ob_k, gs_k = parse_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
        move_k, _sc_k, _nsk, _qsk, _thk, dk = iterative_deepening_search(
            pb_k, ob_k, gs_k, 7, {"optimum_time": 0, "maximum_time": 0},
            ctx, verbose=False, num_threads=1,
        )
        self.assertEqual(int(dk), 7)
        self.assertIn(int(move_k), [int(m) for m in generate_legal_moves(pb_k, ob_k, gs_k)])

        # Mate-in-2 FEN (1. Qf7+ Kh8 2. Qf8# / Qg8#; White Queen + Knight mate)
        clear_transposition_table(tt)
        reset_search_context(ctx)
        pb_m, ob_m, gs_m = parse_fen("6k1/5ppp/8/8/8/8/4Q3/6K1 w - - 0 1")
        move_m, sc_m, _nsm, _qsm, _thm, _dm = iterative_deepening_search(
            pb_m, ob_m, gs_m, 6, {"optimum_time": 0, "maximum_time": 0},
            ctx, verbose=False, num_threads=1,
        )
        self.assertIn(int(move_m), [int(m) for m in generate_legal_moves(pb_m, ob_m, gs_m)])
        self.assertGreaterEqual(abs(int(sc_m)), VALUE_KNOWN_WIN)
        # Immediate back-rank mate: 1. Qe8#
        self.assertGreaterEqual(int(sc_m), MATE_IN_MAX_PLY)


if __name__ == "__main__":
    unittest.main()
