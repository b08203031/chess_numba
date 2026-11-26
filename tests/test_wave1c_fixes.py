"""Wave 1C regression tests: zero-search-tree-change P2 safety, contract & SMP fixes.

Covers:
- Bug #20: get_blockers_for_king populates pinners_bb and removes snipers from occupancy.
- Bug #21: is_move_pseudo_legal rejects non-promotion moves with promotion bits and invalid EP moves.
- Bug #24: select_best_thread resolves HCE (-MATE_IN_MAX_PLY, -VALUE_KNOWN_WIN] voting dead zone.
- Bug #26: DIAG_HP_MARGIN_SUM avoids uint64 underflow on continuation-pruned moves.
- Bug #27: LMR_TTMOVE_REDUCTION frozen in registry and unused import removed from search.py.
- Bug #28: Bounded _SMP_HELPER_CONTEXT_CACHE LRU, helper reset via reset_search_context, and 2T+ nodes_limit stop.
- Bug #31: Boundary guards (format_score_for_uci >=, uint16 old_halfmove_clock, HCE clamp,
  compute_full_corrected_static_eval pinned sentinel backfill, enable_see_pruning scope,
  QS smoothing abs(beta) < MATE_IN_MAX_PLY guard, debug_utils uint64 unary-minus fix).
"""
import contextlib
import io
import unittest
import warnings
from pathlib import Path

import numpy as np

from chess_engine.classical.board_operations import make_move, unmake_move
from chess_engine.classical.constants import (
    BB_SQUARES,
    BLACK,
    CORRECTION_HISTORY_SIZE,
    DIAG_FP_SKIP,
    DIAG_HP_MARGIN_SUM,
    DIAG_LMP_SKIP,
    DIAG_MAIN_CAP_SEE_TRY,
    MATE_IN_MAX_PLY,
    MAX_PLY,
    NO_MOVE,
    PAWN_HISTORY_SIZE,
    PINNED_UNCOMPUTED_SENTINEL,
    VALUE_KNOWN_WIN,
    WHITE,
)
from chess_engine.classical.debug_utils import print_board
from chess_engine.classical.engine_types import SearchContext, reset_search_context
from chess_engine.classical.evaluation import _evaluate_position_jit
from chess_engine.classical.fen_parser import parse_fen
from chess_engine.classical.move import (
    PROMO_QUEEN,
    SPECIAL_MOVE_FLAG_EN_PASSANT,
    SPECIAL_MOVE_FLAG_NORMAL,
    encode_move,
)
from chess_engine.classical.move_generator import (
    get_blockers_for_king,
    is_move_pseudo_legal,
)
from chess_engine.classical.search import (
    _SMP_HELPER_CACHE_MAX_OWNERS,
    _SMP_HELPER_CONTEXT_CACHE,
    _get_or_create_helper_contexts,
    clear_smp_helper_cache,
    compute_full_corrected_static_eval,
    format_score_for_uci,
    iterative_deepening_search,
    RootThreadResult,
    select_best_thread,
)
from chess_engine.classical.transposition_table import (
    clear_transposition_table,
    create_transposition_table,
)
from tune_search.search_param_registry import FROZEN_RUNTIME_PARAMS

ROOT = Path(__file__).resolve().parent.parent


def _make_context(tt_mb: int = 2, enable_diagnostics: bool = False):
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
        enable_diagnostics=enable_diagnostics,
    )
    return tt, ctx


class TestBug20BlockersAndPinners(unittest.TestCase):
    def test_populates_pinners_and_excludes_snipers(self):
        # White Ke1(4), Ne2(12); Black Re4(28), Ke8(60) -> e2 is blocker, e4 is pinner
        p_bbs, o_bbs, _ = parse_fen("4k3/8/8/8/4r3/8/4N3/4K3 w - - 0 1")
        blockers, pinners = get_blockers_for_king(p_bbs, o_bbs, WHITE)
        self.assertEqual(int(blockers), int(BB_SQUARES[12]))
        self.assertEqual(int(pinners), int(BB_SQUARES[28]))

        # Stacked enemy sliders: Black Re8(60->Re8 is a8=56+4=60, move king to c8):
        # Black Re8(60), Black Re4(28), White Ne2(12), White Ke1(4), Black Kc8(58)
        p_bbs2, o_bbs2, _ = parse_fen("2k1r3/8/8/8/4r3/8/4N3/4K3 w - - 0 1")
        blockers2, pinners2 = get_blockers_for_king(p_bbs2, o_bbs2, WHITE)
        self.assertEqual(int(blockers2), int(BB_SQUARES[12]))
        self.assertEqual(int(pinners2), int(BB_SQUARES[28] | BB_SQUARES[60]))

        # Enemy piece acting as blocker (discovered-check blocker, NOT a friendly pin)
        p_bbs3, o_bbs3, _ = parse_fen("2k1r3/8/8/8/8/8/4p3/4K3 w - - 0 1")
        blockers3, pinners3 = get_blockers_for_king(p_bbs3, o_bbs3, WHITE)
        self.assertEqual(int(blockers3), int(BB_SQUARES[12]))
        self.assertEqual(int(pinners3), 0)


class TestBug21IsMovePseudoLegal(unittest.TestCase):
    def test_rejects_corrupt_promo_bits_and_invalid_ep(self):
        p_bbs, o_bbs, g_state = parse_fen("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3")
        # e2-e4 (12 -> 28) is legal normally, but corrupt non-zero promo bits must fail
        corrupt_normal = encode_move(12, 28, PROMO_QUEEN, SPECIAL_MOVE_FLAG_NORMAL)
        self.assertFalse(bool(is_move_pseudo_legal(p_bbs, o_bbs, g_state, corrupt_normal)))

        # Valid EP e5xd6 (36 -> 43)
        valid_ep = encode_move(36, 43, 0, SPECIAL_MOVE_FLAG_EN_PASSANT)
        self.assertTrue(bool(is_move_pseudo_legal(p_bbs, o_bbs, g_state, valid_ep)))

        # Invalid EP onto an occupied target square d6(43)
        p_occ = p_bbs.copy()
        o_occ = o_bbs.copy()
        p_occ[7] |= BB_SQUARES[43]
        o_occ[1] |= BB_SQUARES[43]
        o_occ[2] |= BB_SQUARES[43]
        self.assertFalse(bool(is_move_pseudo_legal(p_occ, o_occ, g_state, valid_ep)))

        # Invalid EP when no enemy pawn stands on d5(35)
        p_no_victim = p_bbs.copy()
        o_no_victim = o_bbs.copy()
        p_no_victim[6] &= ~BB_SQUARES[35]
        o_no_victim[1] &= ~BB_SQUARES[35]
        o_no_victim[2] &= ~BB_SQUARES[35]
        self.assertFalse(bool(is_move_pseudo_legal(p_no_victim, o_no_victim, g_state, valid_ep)))


class TestBug24SelectBestThreadDeadZone(unittest.TestCase):
    def test_non_decisive_loss_band_allows_helper_voting(self):
        m1, m2 = 101, 202
        # Both Thread 0 and Helpers in (-MATE_IN_MAX_PLY, -VALUE_KNOWN_WIN]
        threads = [
            RootThreadResult(0, m1, -(VALUE_KNOWN_WIN + 4500), depth=8, pv=[m1, 10]),
            RootThreadResult(1, m2, -(VALUE_KNOWN_WIN + 1200), depth=8, pv=[m2, 20]),
            RootThreadResult(2, m2, -(VALUE_KNOWN_WIN + 1150), depth=8, pv=[m2, 20]),
        ]
        chosen = select_best_thread(threads)
        self.assertEqual(chosen.move, m2)

        # Non-losing helper still beats losing Thread 0 even with fewer votes
        threads_rescue = [
            RootThreadResult(0, m1, -(VALUE_KNOWN_WIN + 500), depth=8, pv=[m1]),
            RootThreadResult(1, m1, -(VALUE_KNOWN_WIN + 450), depth=8, pv=[m1]),
            RootThreadResult(2, m2, -120, depth=7, pv=[m2, 20]),
        ]
        self.assertEqual(select_best_thread(threads_rescue).move, m2)


class TestBug26DiagHpMarginUnderflow(unittest.TestCase):
    def test_diag_hp_margin_sum_no_uint64_underflow(self):
        src = (ROOT / "chess_engine" / "classical" / "search.py").read_text(encoding="utf-8")
        hp_idx = src.index("hp_margin = hp_threshold - history_score")
        hp_block = src[hp_idx:hp_idx + 450]
        self.assertIn("if hp_margin > 0:", hp_block)

        tt, ctx = _make_context(tt_mb=2, enable_diagnostics=True)
        clear_transposition_table(tt)
        reset_search_context(ctx)
        pb, ob, gs = parse_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
        iterative_deepening_search(pb, ob, gs, 6, {"optimum_time": 0, "maximum_time": 0}, ctx, verbose=False, num_threads=1)
        self.assertLess(int(ctx.diag_stats[DIAG_HP_MARGIN_SUM]), 1 << 40)


class TestBug27DeadTuneAxis(unittest.TestCase):
    def test_lmr_ttmove_red_frozen_and_unimported_in_search(self):
        self.assertIn("LMR_TTMOVE_REDUCTION", FROZEN_RUNTIME_PARAMS)
        src = (ROOT / "chess_engine" / "classical" / "search.py").read_text(encoding="utf-8")
        self.assertNotIn("TUNE_LMR_TTMOVE_RED", src)


class TestBug28HelperCacheAndMultiThreadNodesLimit(unittest.TestCase):
    def test_lru_cache_bounded_and_resets_helpers(self):
        clear_smp_helper_cache()
        try:
            ctxs = [_make_context(tt_mb=1) for _ in range(_SMP_HELPER_CACHE_MAX_OWNERS + 3)]
            for _tt, ctx in ctxs:
                _get_or_create_helper_contexts(ctx, 1)
            self.assertLessEqual(len(_SMP_HELPER_CONTEXT_CACHE), _SMP_HELPER_CACHE_MAX_OWNERS)

            # Most recent owner still cached: verify reset_search_context clears its helper tables
            _tt_last, ctx_last = ctxs[-1]
            helpers = _get_or_create_helper_contexts(ctx_last, 1)
            helpers[0].history_table[0, 0] = 777
            reset_search_context(ctx_last)
            self.assertEqual(int(helpers[0].history_table[0, 0]), 0)
        finally:
            clear_smp_helper_cache()

    def test_multithread_nodes_limit_enforced(self):
        clear_smp_helper_cache()
        tt, ctx = _make_context(tt_mb=2)
        try:
            clear_transposition_table(tt)
            reset_search_context(ctx)
            pb, ob, gs = parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            res = iterative_deepening_search(
                pb, ob, gs, 20, {"nodes_limit": 50000, "optimum_time": 0, "maximum_time": 0},
                ctx, verbose=False, num_threads=2,
            )
            combined_nodes = int(res[2]) + int(res[3])
            self.assertGreater(combined_nodes, 0)
            self.assertLess(combined_nodes, 110000)
        finally:
            clear_smp_helper_cache()


class TestBug31BoundaryGuards(unittest.TestCase):
    def test_format_score_for_uci_mate_boundary(self):
        self.assertTrue(format_score_for_uci(MATE_IN_MAX_PLY).startswith("mate "))
        self.assertTrue(format_score_for_uci(-MATE_IN_MAX_PLY).startswith("mate -"))
        self.assertTrue(format_score_for_uci(MATE_IN_MAX_PLY - 1).startswith("cp "))

    def test_unmake_preserves_high_halfmove_clock(self):
        pb, ob, gs = parse_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
        gs[3] = np.uint64(300)
        mv = encode_move(0, 8, 0, SPECIAL_MOVE_FLAG_NORMAL)
        info = make_move(pb, ob, gs, mv)
        unmake_move(pb, ob, gs, mv, info)
        self.assertEqual(int(gs[3]), 300)

    def test_non_specialized_hce_clamped_below_known_win(self):
        _tt, ctx = _make_context(tt_mb=1)
        # 8 White Queens vs Black Bishop avoids KXK specialized endgame while far exceeding VALUE_KNOWN_WIN
        pb, ob, gs = parse_fen("4k3/QQQQQQQQ/8/8/8/8/8/b3K3 w - - 0 1")
        full_score = int(_evaluate_position_jit(pb, ob, gs, False, ctx)[0])
        lazy_score = int(_evaluate_position_jit(pb, ob, gs, True, ctx)[0])
        self.assertLess(abs(full_score), VALUE_KNOWN_WIN)
        self.assertLess(abs(lazy_score), VALUE_KNOWN_WIN)

    def test_compute_full_corrected_static_eval_backfills_pinned_sentinel(self):
        _tt, ctx = _make_context(tt_mb=1)
        # KPK specialized endgame triggers early return with PINNED_UNCOMPUTED_SENTINEL in _evaluate_position_jit
        pb, ob, gs = parse_fen("8/8/8/8/8/3k4/4P3/3K4 w - - 0 1")
        _raw, _corr, _imp, pw, pb_pin = compute_full_corrected_static_eval(pb, ob, gs, ctx, 0)
        self.assertNotEqual(int(pw), int(PINNED_UNCOMPUTED_SENTINEL))
        self.assertNotEqual(int(pb_pin), int(PINNED_UNCOMPUTED_SENTINEL))

    def test_enable_see_pruning_false_only_disables_see_sites(self):
        tt, ctx = _make_context(tt_mb=2, enable_diagnostics=True)
        clear_transposition_table(tt)
        reset_search_context(ctx)
        ctx.enable_see_pruning = False
        pb, ob, gs = parse_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
        iterative_deepening_search(pb, ob, gs, 5, {"optimum_time": 0, "maximum_time": 0}, ctx, verbose=False, num_threads=1)
        self.assertEqual(int(ctx.diag_stats[DIAG_MAIN_CAP_SEE_TRY]), 0)
        self.assertGreater(int(ctx.diag_stats[DIAG_LMP_SKIP]) + int(ctx.diag_stats[DIAG_FP_SKIP]), 0)

    def test_debug_print_board_emits_no_uint64_warnings(self):
        pb, _ob, _gs = parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with contextlib.redirect_stdout(io.StringIO()):
                print_board(pb)
        self.assertEqual(len(caught), 0, f"Unexpected warnings: {[str(w.message) for w in caught]}")

    def test_qsearch_smoothing_guards_mate_beta(self):
        src = (ROOT / "chess_engine" / "classical" / "search.py").read_text(encoding="utf-8")
        b1_idx = src.index("# B1: QSearch Stand-pat smoothing")
        c1_idx = src.index("# C1: QSearch Move Loop fail-high smoothing")
        self.assertIn("abs(beta) < MATE_IN_MAX_PLY", src[b1_idx:b1_idx + 200])
        self.assertIn("abs(beta) < MATE_IN_MAX_PLY", src[c1_idx:c1_idx + 200])


if __name__ == "__main__":
    unittest.main()
