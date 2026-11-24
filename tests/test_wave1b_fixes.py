"""Wave 1B regression tests: high-impact search / TT / evaluation-interface bugfixes.

Currently covers:
- Bug #1 Part A: Negative Correction History enabled on fail-low (TT_FLAG_ALPHA) nodes.
- Bug #2: Prevent lazy=True crude eval from polluting TT.static_eval and correction history.
- Bug #3: Prevent quiescence_search from storing TT_FLAG_EXACT and overwriting deep TT entries.
- Bug #8: Lockless TT snapshot/checksum safety in store_tt, penalize_tt, and probe_tt_best_move.
"""
import unittest
from pathlib import Path

import numpy as np

from chess_engine.classical.constants import (
    CORRECTION_HISTORY_LIMIT,
    CORRECTION_HISTORY_SIZE,
    MAX_PLY,
    NO_MOVE,
    PAWN_HISTORY_SIZE,
)
from chess_engine.classical.engine_types import SearchContext, reset_search_context
from chess_engine.classical.evaluation import _evaluate_position_jit
from chess_engine.classical.fen_parser import parse_fen
from chess_engine.classical.search import iterative_deepening_search, quiescence_search
from chess_engine.classical.transposition_table import (
    TT_FLAG_ALPHA,
    TT_FLAG_BETA,
    TT_FLAG_EXACT,
    TT_FLAG_NONE,
    clear_transposition_table,
    create_transposition_table,
    penalize_tt,
    probe_tt_best_move,
    probe_tt_snapshot,
    store_tt,
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

        # Searching opening + middlegame positions (both White and Black to move) at depth 9
        # exercises both fail-high and fail-low nodes across all five correction tables.
        for fen in (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "r2q1rk1/ppp2ppp/2n1bn2/2bpp3/4P3/2NP1NP1/PPP1QPBP/R1B2RK1 b - - 0 8",
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


class TestBug3QSearchTTNoExactOverwrite(unittest.TestCase):
    def test_source_contract_qsearch_never_stores_exact_and_guards_standpat(self):
        src = (ROOT / "chess_engine" / "classical" / "search.py").read_text(encoding="utf-8")
        qs_start = src.index("def quiescence_search(")
        qs_end = src.index("def get_next_move(", qs_start)
        qs_src = src[qs_start:qs_end]

        self.assertNotIn("original_alpha", qs_src)
        standpat_idx = qs_src.index("if stand_pat >= beta:")
        self.assertNotIn("TT_FLAG_EXACT", qs_src[standpat_idx:])
        self.assertIn("if qs_tt_flag == TT_FLAG_NONE:", qs_src[standpat_idx:qs_src.index("alpha = max(alpha, stand_pat)")])
        self.assertEqual(qs_src.count("store_tt("), 3)
        self.assertEqual(qs_src.count("bool(qs_tt_is_pv)"), 3)

    def test_store_tt_depth0_exact_guard(self):
        tt = create_transposition_table(1)
        zkey = np.uint64(0x1122334455667788)
        store_tt(tt, zkey, 10, 100, 90, TT_FLAG_BETA, 222, 3, True)
        store_tt(tt, zkey, 0, 50, 60, TT_FLAG_EXACT, 0, 3, False)
        snap = probe_tt_snapshot(tt, zkey)
        self.assertEqual(
            (int(snap[0]), int(snap[1]), int(snap[2]), int(snap[3]), bool(snap[4]), int(snap[5])),
            (int(TT_FLAG_BETA), 10, 100, 90, True, 222),
        )

        # depth=0 EXACT can overwrite an existing depth=0 entry
        zkey_d0 = np.uint64(0x2233445566778899)
        store_tt(tt, zkey_d0, 0, 40, 35, TT_FLAG_BETA, 111, 3, False)
        store_tt(tt, zkey_d0, 0, 55, 35, TT_FLAG_EXACT, 111, 3, True)
        snap_d0 = probe_tt_snapshot(tt, zkey_d0)
        self.assertEqual((int(snap_d0[0]), int(snap_d0[1]), int(snap_d0[2]), bool(snap_d0[4])), (int(TT_FLAG_EXACT), 0, 55, True))

        # depth=1 EXACT can overwrite depth=10
        store_tt(tt, zkey, 1, 125, 95, TT_FLAG_EXACT, 333, 3, True)
        snap_d1 = probe_tt_snapshot(tt, zkey)
        self.assertEqual((int(snap_d1[0]), int(snap_d1[1]), int(snap_d1[2]), int(snap_d1[5])), (int(TT_FLAG_EXACT), 1, 125, 333))

    def test_live_qsearch_preserves_deep_tt_and_ids_leaves_zero_d0_exact(self):
        tt, ctx = _make_context(tt_mb=4)

        # 1. No-capture leaf (move_count == 0) must not overwrite deep TT entry with EXACT or drop is_pv
        quiet_fen = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"
        pb, ob, gs = parse_fen(quiet_fen)
        clear_transposition_table(tt)
        reset_search_context(ctx)
        ctx.tt_generation = np.uint8(1)
        tt_key = get_tt_key(gs[4], int(gs[3]))
        store_tt(tt, tt_key, 12, 80, 70, TT_FLAG_BETA, 555, 1, True)
        quiescence_search(pb, ob, gs, np.int32(50), np.int32(150), np.int32(1), ctx, np.int32(0))
        snap = probe_tt_snapshot(tt, tt_key)
        self.assertEqual(
            (int(snap[0]), int(snap[1]), int(snap[2]), int(snap[3]), bool(snap[4]), int(snap[5])),
            (int(TT_FLAG_BETA), 12, 80, 70, True, 555),
        )

        # 2. Stand-pat >= beta with existing TT entry does not overwrite deep TT entry
        clear_transposition_table(tt)
        reset_search_context(ctx)
        ctx.tt_generation = np.uint8(1)
        store_tt(tt, tt_key, 12, 80, 70, TT_FLAG_ALPHA, 555, 1, True)
        quiescence_search(pb, ob, gs, np.int32(50), np.int32(60), np.int32(1), ctx, np.int32(0))
        snap_sp = probe_tt_snapshot(tt, tt_key)
        self.assertEqual(
            (int(snap_sp[0]), int(snap_sp[1]), int(snap_sp[2]), int(snap_sp[3]), bool(snap_sp[4]), int(snap_sp[5])),
            (int(TT_FLAG_ALPHA), 12, 80, 70, True, 555),
        )

        # 3. Tactical QS improving alpha without beta cutoff must not store TT_FLAG_EXACT over deep TT
        kiwipete = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
        pb_k, ob_k, gs_k = parse_fen(kiwipete)
        clear_transposition_table(tt)
        reset_search_context(ctx)
        ctx.tt_generation = np.uint8(1)
        tt_key_k = get_tt_key(gs_k[4], int(gs_k[3]))
        store_tt(tt, tt_key_k, 12, 50, 40, TT_FLAG_ALPHA, 555, 1, True)
        quiescence_search(pb_k, ob_k, gs_k, np.int32(-500), np.int32(5000), np.int32(1), ctx, np.int32(0))
        snap_k = probe_tt_snapshot(tt, tt_key_k)
        self.assertEqual(
            (int(snap_k[0]), int(snap_k[1]), int(snap_k[2]), int(snap_k[3]), bool(snap_k[4]), int(snap_k[5])),
            (int(TT_FLAG_ALPHA), 12, 50, 40, True, 555),
        )

        # 4. Full search produces zero depth=0 TT_FLAG_EXACT entries
        for fen in ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", kiwipete):
            clear_transposition_table(tt)
            reset_search_context(ctx)
            p_b, o_b, g_s = parse_fen(fen)
            iterative_deepening_search(
                p_b, o_b, g_s, 6, {"optimum_time": 0, "maximum_time": 0}, ctx, verbose=False, num_threads=1
            )
            occupied = tt["flag"] != TT_FLAG_NONE
            d0_exact = int(np.count_nonzero(occupied & (tt["depth"] == 0) & (tt["flag"] == TT_FLAG_EXACT)))
            self.assertEqual(d0_exact, 0, f"Unexpected depth=0 TT_FLAG_EXACT entries for {fen}")


class TestBug8LocklessTTSafety(unittest.TestCase):
    @staticmethod
    def _slot_index(tt, zkey):
        num_buckets_minus_1 = np.uint64((len(tt) >> 2) - 1)
        base_idx = int(((np.uint64(zkey) >> np.uint64(32)) & num_buckets_minus_1) << np.uint64(2))
        key32 = np.uint32(zkey)
        for i in range(4):
            if np.uint32(tt[base_idx + i]["key"]) == key32 and np.uint8(tt[base_idx + i]["flag"]) != TT_FLAG_NONE:
                return base_idx + i
        raise AssertionError("Expected TT entry not found")

    def test_penalize_tt_refuses_corrupt_entry_and_skips_noop(self):
        tt = create_transposition_table(1)
        zkey = np.uint64(0xAABBCCDDEEFF0011)
        store_tt(tt, zkey, 8, 100, 90, TT_FLAG_BETA, 333, 1, False)
        slot = self._slot_index(tt, zkey)
        tt[slot]["score"] = np.int16(9999)  # corrupt without updating checksum

        penalize_tt(tt, zkey, 1)
        self.assertEqual(int(tt[slot]["depth"]), 8)
        self.assertEqual(int(probe_tt_snapshot(tt, zkey)[0]), int(TT_FLAG_NONE))

        # No-op cases (depth=0 or penalty=0) keep valid entries untouched
        zkey_d0 = np.uint64(0xBBCCDDEEFF001122)
        store_tt(tt, zkey_d0, 0, 20, 15, TT_FLAG_ALPHA, 111, 1, False)
        penalize_tt(tt, zkey_d0, 1)
        self.assertEqual(int(probe_tt_snapshot(tt, zkey_d0)[0]), int(TT_FLAG_ALPHA))
        self.assertEqual(int(probe_tt_snapshot(tt, zkey_d0)[1]), 0)

        zkey_p0 = np.uint64(0xCCDDEEFF00112233)
        store_tt(tt, zkey_p0, 5, 30, 25, TT_FLAG_BETA, 222, 1, True)
        penalize_tt(tt, zkey_p0, 0)
        snap_p0 = probe_tt_snapshot(tt, zkey_p0)
        self.assertEqual((int(snap_p0[0]), int(snap_p0[1]), int(snap_p0[5])), (int(TT_FLAG_BETA), 5, 222))

    def test_store_tt_corrupt_match_replaces_cleanly_without_inheriting_torn_fields(self):
        tt = create_transposition_table(1)
        zkey = np.uint64(0x1234567890ABCDEF)
        store_tt(tt, zkey, 10, 100, 90, TT_FLAG_BETA, 333, 1, True)
        slot = self._slot_index(tt, zkey)
        tt[slot]["score"] = np.int16(9999)

        # Shallower same-generation store would hit else if entry were valid;
        # because existing entry is torn, store_tt must cleanly overwrite with new fields only.
        store_tt(tt, zkey, 2, 42, 32767, TT_FLAG_ALPHA, 0, 1, False)
        snap = probe_tt_snapshot(tt, zkey)
        self.assertEqual(
            (int(snap[0]), int(snap[1]), int(snap[2]), int(snap[3]), bool(snap[4]), int(snap[5])),
            (int(TT_FLAG_ALPHA), 2, 42, 32767, False, int(NO_MOVE)),
        )

    def test_store_tt_replace_preserves_static_eval_with_valid_checksum(self):
        tt = create_transposition_table(1)
        zkey = np.uint64(0x0F1E2D3C4B5A6978)
        store_tt(tt, zkey, 4, 50, 123, TT_FLAG_BETA, 111, 1, False)
        store_tt(tt, zkey, 6, 75, 32767, TT_FLAG_BETA, 222, 1, False)
        snap = probe_tt_snapshot(tt, zkey)
        self.assertEqual(
            (int(snap[0]), int(snap[1]), int(snap[2]), int(snap[3]), int(snap[5])),
            (int(TT_FLAG_BETA), 6, 75, 123, 222),
        )
        self.assertEqual(tuple(int(x) for x in probe_tt_best_move(tt, zkey)), (int(TT_FLAG_BETA), 222))

    def test_store_tt_non_replace_best_move_and_int16_min_mate_aging(self):
        tt = create_transposition_table(1)

        # 1. int16 min (-32768) must not overflow abs(); TT_FLAG_ALPHA must not overwrite e_best_move == 0
        zkey1 = np.uint64(0xA1B2C3D4E5F60718)
        store_tt(tt, zkey1, 10, -32768, 10, TT_FLAG_BETA, 0, 2, False)
        store_tt(tt, zkey1, 3, 20, 32767, TT_FLAG_ALPHA, 444, 2, False)
        snap1a = probe_tt_snapshot(tt, zkey1)
        self.assertEqual((int(snap1a[0]), int(snap1a[1]), int(snap1a[5])), (int(TT_FLAG_BETA), 9, 0))

        # Now a BETA store with non-zero best_move fills e_best_move == 0
        store_tt(tt, zkey1, 3, 20, 32767, TT_FLAG_BETA, 555, 2, False)
        snap1b = probe_tt_snapshot(tt, zkey1)
        self.assertEqual((int(snap1b[0]), int(snap1b[1]), int(snap1b[5])), (int(TT_FLAG_BETA), 8, 555))

        # 2. depth > 0 BETA store updates best_move on existing ALPHA entry
        zkey2 = np.uint64(0xB2C3D4E5F6071829)
        store_tt(tt, zkey2, 10, 20, 10, TT_FLAG_ALPHA, 111, 2, False)
        store_tt(tt, zkey2, 2, 30, 32767, TT_FLAG_BETA, 777, 2, False)
        snap2 = probe_tt_snapshot(tt, zkey2)
        self.assertEqual((int(snap2[0]), int(snap2[1]), int(snap2[5])), (int(TT_FLAG_ALPHA), 10, 777))

        # 3. Existing BETA best_move is NOT overwritten in non-replace branch
        zkey3 = np.uint64(0xC3D4E5F60718293A)
        store_tt(tt, zkey3, 10, 20, 10, TT_FLAG_BETA, 111, 2, False)
        store_tt(tt, zkey3, 2, 30, 32767, TT_FLAG_BETA, 888, 2, False)
        snap3 = probe_tt_snapshot(tt, zkey3)
        self.assertEqual((int(snap3[0]), int(snap3[1]), int(snap3[5])), (int(TT_FLAG_BETA), 10, 111))

    def test_probe_tt_best_move_uses_scalar_snapshot_and_rejects_torn_entries(self):
        tt_src = (ROOT / "chess_engine" / "classical" / "transposition_table.py").read_text(encoding="utf-8")
        fn_start = tt_src.index("def probe_tt_best_move(")
        fn_end = tt_src.index("def store_tt(", fn_start)
        self.assertNotIn("probe_tt(", tt_src[fn_start:fn_end])

        tt = create_transposition_table(1)
        zkey = np.uint64(0xDEADBEEF13572468)
        store_tt(tt, zkey, 6, 65, 60, TT_FLAG_BETA, 666, 1, False)
        slot = self._slot_index(tt, zkey)
        tt[slot]["best_move"] = np.uint16(999)
        self.assertEqual(tuple(int(x) for x in probe_tt_best_move(tt, zkey)), (int(TT_FLAG_NONE), int(NO_MOVE)))


if __name__ == "__main__":
    unittest.main()
