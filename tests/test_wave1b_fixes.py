"""Wave 1B regression tests: high-impact search / TT / evaluation-interface bugfixes.

Currently covers:
- Bug #1 Part A: Negative Correction History enabled on fail-low (TT_FLAG_ALPHA) nodes.
- Bug #2: Prevent lazy=True crude eval from polluting TT.static_eval and correction history.
"""
import unittest
from pathlib import Path

import numpy as np

from chess_engine.classical.constants import (
    CORRECTION_HISTORY_LIMIT,
    CORRECTION_HISTORY_SIZE,
    MAX_PLY,
    PAWN_HISTORY_SIZE,
)
from chess_engine.classical.engine_types import SearchContext, reset_search_context
from chess_engine.classical.evaluation import _evaluate_position_jit
from chess_engine.classical.fen_parser import parse_fen
from chess_engine.classical.search import iterative_deepening_search
from chess_engine.classical.transposition_table import (
    clear_transposition_table,
    create_transposition_table,
    probe_tt_snapshot,
)
from chess_engine.classical.zobrist import get_tt_key

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


class TestBug1PartANegativeCorrectionHistory(unittest.TestCase):
    def test_source_contract_gates_has_best_move_on_non_alpha_flag(self):
        src = (ROOT / "chess_engine" / "classical" / "search.py").read_text(encoding="utf-8")
        corr_idx = src.index("# --- Update Correction History ---")
        corr_end = src.index("tt_score = max_eval", corr_idx)
        block = src[corr_idx:corr_end]

        self.assertIn("has_best_move = (final_flag != TT_FLAG_ALPHA) and (best_move != NO_MOVE)", block)
        self.assertIn("if has_best_move:", block)
        self.assertLess(block.index("has_best_move ="), block.index("if has_best_move:"))
        self.assertIn("if not is_capture_or_promo and (max_eval > static_score) == has_best_move:", block)

    def test_live_search_populates_both_positive_and_negative_correction_entries(self):
        tt, ctx = _make_context(tt_mb=4)
        clear_transposition_table(tt)
        reset_search_context(ctx)

        # Searching opening + middlegame positions at depth 9 exercises both fail-high and
        # fail-low nodes across all five correction tables with full HCE static baselines.
        for fen in (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        ):
            p_bbs, o_bbs, g_state = parse_fen(fen)
            iterative_deepening_search(
                p_bbs, o_bbs, g_state, 9, {"optimum_time": 0, "maximum_time": 0}, ctx, verbose=False, num_threads=1
            )

        # Side-relative tables previously had 0 negative entries across the entire search!
        for name, arr in (
            ("pawn_correction_history", ctx.pawn_correction_history),
            ("minor_correction_history", ctx.minor_correction_history),
            ("non_pawn_correction_history_white", ctx.non_pawn_correction_history_white),
            ("non_pawn_correction_history_black", ctx.non_pawn_correction_history_black),
            ("continuation_correction_history", ctx.continuation_correction_history),
        ):
            pos_cnt = int(np.count_nonzero(arr > 0))
            neg_cnt = int(np.count_nonzero(arr < 0))
            self.assertGreater(pos_cnt, 0, f"{name} should have positive entries")
            self.assertGreater(neg_cnt, 0, f"{name} should now have negative fail-low entries")
            self.assertLessEqual(int(np.max(np.abs(arr))), CORRECTION_HISTORY_LIMIT)


class TestBug2LazyEvalTTIsolation(unittest.TestCase):
    def test_source_contract_guards_tt_static_eval_and_correction_history(self):
        src = (ROOT / "chess_engine" / "classical" / "search.py").read_text(encoding="utf-8")

        probe_idx = src.index("if tt_hit and tt_static_eval_raw != 32767:")
        probe_block = src[probe_idx:probe_idx + 220]
        self.assertIn("static_eval_is_full = True", probe_block)

        corr_idx = src.index("# --- Update Correction History ---")
        store_idx = src.index("store_tt(", corr_idx)
        corr_block = src[corr_idx:store_idx]
        self.assertIn("if not is_capture_or_promo and not static_eval_is_full:", corr_block)
        self.assertLess(
            corr_block.index("if not is_capture_or_promo and not static_eval_is_full:"),
            corr_block.index("if not is_capture_or_promo and (max_eval > static_score) == has_best_move:"),
        )
        self.assertIn(
            "if static_eval_is_full and raw_static_eval != -INFINITY and abs(raw_static_eval) < MATE_SCORE - MAX_PLY:",
            corr_block,
        )

    def test_tt_static_eval_never_stores_lazy_crude_score(self):
        tt, ctx = _make_context(tt_mb=4)
        fens = (
            "r4rk1/ppp3p1/2n4p/4p1Nq/2B2Q2/8/PPP3PP/R4R1K w - - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        )
        for fen in fens:
            p_bbs, o_bbs, g_state = parse_fen(fen)
            lazy_eval = int(_evaluate_position_jit(p_bbs, o_bbs, g_state, True, ctx)[0])
            full_eval = int(_evaluate_position_jit(p_bbs, o_bbs, g_state, False, ctx)[0])
            self.assertGreater(abs(full_eval - lazy_eval), 20, f"Expected meaningful lazy/full gap on {fen}")

            for depth in (2, 6):
                clear_transposition_table(tt)
                reset_search_context(ctx)
                p_b, o_b, g_s = parse_fen(fen)
                iterative_deepening_search(
                    p_b, o_b, g_s, depth, {"optimum_time": 0, "maximum_time": 0}, ctx, verbose=False, num_threads=1
                )
                tt_key = get_tt_key(g_s[4], int(g_s[3]))
                _flag, _d, _sc, tt_se, _pv, _bm = probe_tt_snapshot(tt, tt_key)
                tt_se_i = int(tt_se)
                self.assertIn(tt_se_i, (32767, full_eval), f"depth={depth} fen={fen}")
                self.assertNotEqual(tt_se_i, lazy_eval, f"lazy score leaked into TT at depth={depth}")


if __name__ == "__main__":
    unittest.main()
