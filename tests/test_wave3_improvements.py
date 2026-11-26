"""Wave 3 high-ROI improvement tests.

Currently covers:
- Wave 3-1 (S-1): Dual-side Non-Pawn Correction History (white-absolute symmetry
  reading and updating both np_white_key and np_black_key regardless of side to move).
"""
import unittest
from pathlib import Path

import numpy as np

from chess_engine.classical.constants import (
    CORRECTION_HISTORY_DIVISOR,
    CORRECTION_HISTORY_LIMIT,
    CORRECTION_HISTORY_MASK,
    CORRECTION_HISTORY_NON_PAWN_WEIGHT,
    CORRECTION_HISTORY_SIZE,
    MAX_PLY,
    NON_PAWN_KEY_BLACK_INDEX,
    NON_PAWN_KEY_WHITE_INDEX,
    PAWN_HISTORY_SIZE,
)
from chess_engine.classical.engine_types import SearchContext, reset_search_context
from chess_engine.classical.fen_parser import parse_fen
from chess_engine.classical.search import (
    apply_correction_history_score,
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


class TestWave3S1DualSideNonPawnCorrection(unittest.TestCase):
    def test_source_contract_reads_and_updates_both_non_pawn_tables_unconditionally(self):
        src = (ROOT / "chess_engine" / "classical" / "search.py").read_text(encoding="utf-8")

        apply_start = src.index("def apply_correction_history_score(")
        apply_end = src.index("# Continuation Correction Heuristic", apply_start)
        apply_block = src[apply_start:apply_end]
        self.assertIn("global_np_w = search_context.non_pawn_correction_history_white[", apply_block)
        self.assertIn("global_np_b = search_context.non_pawn_correction_history_black[", apply_block)
        self.assertIn("np_sum = global_np_w + global_np_b", apply_block)

        upd_start = src.index("# Update both White and Black Non-Pawn Correction")
        upd_end = src.index("# Update Continuation Correction History", upd_start)
        upd_block = src[upd_start:upd_end]
        self.assertNotIn("if side == WHITE:", upd_block)
        self.assertIn("search_context.non_pawn_correction_history_white[idx_np_w]", upd_block)
        self.assertIn("search_context.non_pawn_correction_history_black[idx_np_b]", upd_block)

    def test_apply_correction_history_score_sums_both_tables_white_absolute(self):
        _tt, ctx = _make_context(tt_mb=1)
        reset_search_context(ctx)

        _pb_w, _ob_w, gs_w = parse_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
        _pb_b, _ob_b, gs_b = parse_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1")

        idx_w = int(gs_w[NON_PAWN_KEY_WHITE_INDEX]) & CORRECTION_HISTORY_MASK
        idx_b = int(gs_w[NON_PAWN_KEY_BLACK_INDEX]) & CORRECTION_HISTORY_MASK
        self.assertEqual(idx_w, int(gs_b[NON_PAWN_KEY_WHITE_INDEX]) & CORRECTION_HISTORY_MASK)
        self.assertEqual(idx_b, int(gs_b[NON_PAWN_KEY_BLACK_INDEX]) & CORRECTION_HISTORY_MASK)

        # Populate ONLY the black non-pawn table with +300 and white table with +200
        # so any side-exclusive lookup would drop one term or flip sign incorrectly.
        ctx.non_pawn_correction_history_white[idx_w] = np.int16(200)
        ctx.non_pawn_correction_history_black[idx_b] = np.int16(300)

        expected_delta = (500 * CORRECTION_HISTORY_NON_PAWN_WEIGHT) // CORRECTION_HISTORY_DIVISOR
        self.assertGreater(expected_delta, 0)

        score_w = int(apply_correction_history_score(gs_w, ctx, np.int32(100), np.int32(0)))
        score_b = int(apply_correction_history_score(gs_b, ctx, np.int32(100), np.int32(0)))
        self.assertEqual(score_w, 100 + expected_delta)
        self.assertEqual(score_b, 100 - expected_delta)

    def test_depth9_search_populates_both_non_pawn_tables_pos_and_neg(self):
        tt, ctx = _make_context(tt_mb=4)
        clear_transposition_table(tt)
        reset_search_context(ctx)

        # Even a single shallow depth-5 search must write to BOTH white and black non-pawn tables.
        pb0, ob0, gs0 = parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        iterative_deepening_search(
            pb0, ob0, gs0, 5, {"optimum_time": 0, "maximum_time": 0},
            ctx, verbose=False, num_threads=1,
        )
        self.assertGreater(int(np.count_nonzero(ctx.non_pawn_correction_history_white)), 0)
        self.assertGreater(int(np.count_nonzero(ctx.non_pawn_correction_history_black)), 0)

        # Across diverse opening + middlegame positions (both White and Black to move) at depth 9,
        # both tables accumulate positive and negative white-absolute entries bounded by limit.
        clear_transposition_table(tt)
        reset_search_context(ctx)
        for fen in (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "r2q1rk1/ppp2ppp/2n1bn2/2bpp3/4P3/2NP1NP1/PPP1QPBP/R1B2RK1 b - - 0 8",
        ):
            pb, ob, gs = parse_fen(fen)
            iterative_deepening_search(
                pb, ob, gs, 9, {"optimum_time": 0, "maximum_time": 0},
                ctx, verbose=False, num_threads=1,
            )
        for name, arr in (
            ("non_pawn_correction_history_white", ctx.non_pawn_correction_history_white),
            ("non_pawn_correction_history_black", ctx.non_pawn_correction_history_black),
        ):
            self.assertGreater(int(np.count_nonzero(arr > 0)), 0, f"{name} missing >0")
            self.assertGreater(int(np.count_nonzero(arr < 0)), 0, f"{name} missing <0")
            self.assertLessEqual(int(np.max(np.abs(arr))), CORRECTION_HISTORY_LIMIT)


if __name__ == "__main__":
    unittest.main()
