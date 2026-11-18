import unittest
from pathlib import Path
import numpy as np

from chess_engine.classical.constants import NO_MOVE
from chess_engine.classical.core import generate_legal_moves
from chess_engine.classical.engine_types import (
    SearchContext,
    create_smp_helper_context,
    reset_search_context,
)
from chess_engine.classical.fen_parser import parse_fen
from chess_engine.classical.move import move_to_uci
from chess_engine.classical.transposition_table import (
    TT_FLAG_ALPHA,
    TT_FLAG_BETA,
    TT_FLAG_EXACT,
    TT_FLAG_NONE,
    _compute_tt_checksum,
    clear_transposition_table,
    create_transposition_table,
    penalize_tt,
    probe_tt,
    probe_tt_best_move,
    probe_tt_snapshot,
    store_tt,
)


ROOT = Path(__file__).resolve().parents[1]


def _make_main_context(tt_mb=4):
    max_ply = 128
    tt = create_transposition_table(tt_mb)
    return SearchContext(
        tt,
        np.zeros(max_ply * 2, dtype=np.uint16),
        np.zeros((max_ply, max_ply), dtype=np.uint16),
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


class TestLazySMP(unittest.TestCase):
    def test_lockless_tt_xor_checksum_verification(self):
        tt = create_transposition_table(1)
        zkey = np.uint64(0xCAFEBABE12345678)

        # 1. Valid store_tt should pass checksum in probe_tt, probe_tt_snapshot, and probe_tt_best_move
        store_tt(tt, zkey, 8, 145, 120, TT_FLAG_EXACT, 1234, 3, True)
        flag_bm, bm = probe_tt_best_move(tt, zkey)
        self.assertEqual((int(flag_bm), int(bm)), (int(TT_FLAG_EXACT), 1234))

        snap = probe_tt_snapshot(tt, zkey)
        self.assertEqual(
            (int(snap[0]), int(snap[1]), int(snap[2]), int(snap[3]), bool(snap[4]), int(snap[5])),
            (int(TT_FLAG_EXACT), 8, 145, 120, True, 1234),
        )

        # 2. penalize_tt updates depth and recomputes a valid checksum
        penalize_tt(tt, zkey, 3)
        snap_pen = probe_tt_snapshot(tt, zkey)
        self.assertEqual(int(snap_pen[0]), int(TT_FLAG_EXACT))
        self.assertEqual(int(snap_pen[1]), 5)
        self.assertEqual(int(snap_pen[5]), 1234)

        # 3. Non-replace exact-key store_tt branch (shallower depth, same generation, non-EXACT)
        #    updates static_eval when previously 32767 and keeps checksum valid
        zkey2 = np.uint64(0xDEADBEEF87654321)
        store_tt(tt, zkey2, 10, 31500, 32767, TT_FLAG_BETA, 2222, 5, False)
        store_tt(tt, zkey2, 2, 50, 88, TT_FLAG_ALPHA, 0, 5, False)
        snap2 = probe_tt_snapshot(tt, zkey2)
        self.assertEqual(int(snap2[0]), int(TT_FLAG_BETA))
        self.assertEqual(int(snap2[1]), 9)  # mate-score secondary aging (-1 depth)
        self.assertEqual(int(snap2[3]), 88)  # static_eval filled in
        self.assertEqual(int(snap2[5]), 2222)  # previous best_move preserved

        # 4. Simulated torn read / corrupted payload with flag != TT_FLAG_NONE must be rejected as miss
        num_buckets_minus_1 = np.uint64((len(tt) >> 2) - 1)
        base_idx = int(((zkey >> np.uint64(32)) & num_buckets_minus_1) << np.uint64(2))
        tt[base_idx]['score'] = np.int16(9999)  # corrupt score without updating padding checksum

        flag_corrupt, bm_corrupt = probe_tt_best_move(tt, zkey)
        self.assertEqual((int(flag_corrupt), int(bm_corrupt)), (int(TT_FLAG_NONE), int(NO_MOVE)))
        snap_corrupt = probe_tt_snapshot(tt, zkey)
        self.assertEqual(
            (int(snap_corrupt[0]), int(snap_corrupt[1]), int(snap_corrupt[2]), int(snap_corrupt[3]), bool(snap_corrupt[4]), int(snap_corrupt[5])),
            (int(TT_FLAG_NONE), 0, 0, 32767, False, int(NO_MOVE)),
        )

        # Restore score and corrupt best_move instead
        tt[base_idx]['score'] = np.int16(145)
        tt[base_idx]['best_move'] = np.uint16(4321)
        self.assertEqual(int(probe_tt_snapshot(tt, zkey)[0]), int(TT_FLAG_NONE))
        self.assertEqual(int(probe_tt_best_move(tt, zkey)[0]), int(TT_FLAG_NONE))

        # Restore best_move and corrupt flag to out-of-range value (> TT_FLAG_BETA)
        tt[base_idx]['best_move'] = np.uint16(1234)
        chk_invalid_flag = _compute_tt_checksum(
            np.uint32(zkey), np.int16(145), np.int16(120), np.uint8(5), np.uint8(7), np.uint16(1234), True
        )
        tt[base_idx]['flag'] = np.uint8(7)
        tt[base_idx]['padding'][0] = np.uint8(chk_invalid_flag & np.uint16(0xFF))
        tt[base_idx]['padding'][1] = np.uint8(chk_invalid_flag >> np.uint16(8))
        self.assertEqual(int(probe_tt_snapshot(tt, zkey)[0]), int(TT_FLAG_NONE))
        self.assertEqual(int(probe_tt_best_move(tt, zkey)[0]), int(TT_FLAG_NONE))

        # 5. Contract preservation: flag == TT_FLAG_NONE with matching key32 returns raw entry in probe_tt_best_move
        clear_transposition_table(tt)
        zero_low_key = np.uint64(0x1234567800000000)
        store_tt(tt, zero_low_key, 4, 42, 40, TT_FLAG_EXACT, 777, 1, True)
        tt['key'] = np.uint32(0)
        tt['flag'] = np.uint8(TT_FLAG_NONE)
        flag_none, bm_none = probe_tt_best_move(tt, zero_low_key)
        self.assertEqual((int(flag_none), int(bm_none)), (int(TT_FLAG_NONE), 777))

    def test_create_smp_helper_context_shares_tt_and_stop_flag_via_frombuffer(self):
        main_ctx = _make_main_context(tt_mb=1)
        main_ctx.tune[0] = 777
        main_ctx.eval_weights[0] = 888
        main_ctx.enable_multicut = True

        helper_ctx = create_smp_helper_context(main_ctx)

        # Distinct Python/Numba ndarray wrapper objects (separate NRT_MemInfo headers)
        self.assertIsNot(main_ctx.transposition_table, helper_ctx.transposition_table)
        self.assertIsNot(main_ctx.stop_flag, helper_ctx.stop_flag)

        # Exact same physical memory buffer addresses
        self.assertEqual(
            main_ctx.transposition_table.ctypes.data,
            helper_ctx.transposition_table.ctypes.data,
        )
        self.assertEqual(
            main_ctx.stop_flag.ctypes.data,
            helper_ctx.stop_flag.ctypes.data,
        )

        # Writing to main_ctx.stop_flag or TT is immediately visible in helper_ctx
        main_ctx.stop_flag[0] = True
        self.assertTrue(bool(helper_ctx.stop_flag[0]))
        helper_ctx.stop_flag[0] = False
        self.assertFalse(bool(main_ctx.stop_flag[0]))

        zkey = np.uint64(0xABCDEF0123456789)
        store_tt(main_ctx.transposition_table, zkey, 6, 80, 75, TT_FLAG_BETA, 555, 2, False)
        snap = probe_tt_snapshot(helper_ctx.transposition_table, zkey)
        self.assertEqual((int(snap[0]), int(snap[1]), int(snap[2]), int(snap[5])), (int(TT_FLAG_BETA), 6, 80, 555))

        # Thread-private history & stack arrays must NOT share memory
        for field_name in (
            "killer_moves",
            "pv_table",
            "history_table",
            "butterfly_history",
            "continuation_history",
            "capture_history",
            "pawn_history",
            "pawn_correction_history",
            "minor_correction_history",
            "non_pawn_correction_history_white",
            "non_pawn_correction_history_black",
            "continuation_correction_history",
            "move_stack",
            "piece_stack",
            "ply_path_stack",
            "nodes_searched_array",
            "pawn_table_keys",
        ):
            main_arr = getattr(main_ctx, field_name)
            helper_arr = getattr(helper_ctx, field_name)
            self.assertNotEqual(
                main_arr.ctypes.data,
                helper_arr.ctypes.data,
                f"Field {field_name} unexpectedly shares memory between main and helper contexts",
            )

        # Mutating helper history does not affect main history
        helper_ctx.history_table[0, 0] = 12345
        self.assertEqual(int(main_ctx.history_table[0, 0]), 0)

        # Synced configuration & reset_search_context check
        self.assertEqual(int(helper_ctx.tune[0]), 777)
        self.assertEqual(int(helper_ctx.eval_weights[0]), 888)
        self.assertTrue(bool(helper_ctx.enable_multicut))
        self.assertFalse(bool(helper_ctx.diag_enabled))

        reset_search_context(helper_ctx)
        self.assertEqual(int(helper_ctx.history_table[0, 0]), 0)
        self.assertEqual(int(helper_ctx.pawn_history[0, 0, 0]), -1238)
        self.assertEqual(int(helper_ctx.piece_stack[0]), -1)
        self.assertEqual(np.uint64(helper_ctx.pawn_table_keys[0]), np.uint64(0xFFFFFFFFFFFFFFFF))

    def test_uci_and_gui_threads_wiring_contracts(self):
        main_src = (ROOT / "main.py").read_text(encoding="utf-8")
        assistant_src = (ROOT / "assistant.py").read_text(encoding="utf-8")

        self.assertIn('print("option name Threads type spin default 1 min 1 max 64")', main_src)
        self.assertIn('opt_name.lower() == "threads"', main_src)
        self.assertIn("create_smp_helper_context(global_search_context)", main_src)
        self.assertIn("reset_search_context(hc)", main_src)
        self.assertIn("num_threads=num_threads, helper_contexts=helper_search_contexts", main_src)

        self.assertIn('self.threads_var = tk.StringVar(value="1")', assistant_src)
        self.assertIn("def on_threads_changed(", assistant_src)
        self.assertIn("setoption name Threads value", assistant_src)

    def test_multithreaded_iterative_deepening_search_correctness_and_scaling(self):
        from chess_engine.classical.search import iterative_deepening_search, warmup_search_jit

        main_ctx = _make_main_context(tt_mb=16)
        warmup_search_jit(main_ctx)
        helpers_4 = [create_smp_helper_context(main_ctx) for _ in range(3)]

        positions = [
            ("startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 9, None, True),
            ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 9, None, True),
            (
                "mate_in_2",
                "r2qkb1r/pp2nppp/3p4/2pNN1B1/2BnP3/3P4/PPP2PPP/R2bK2R w KQkq - 1 1",
                6,
                {"d5f6"},
                False,
            ),
        ]

        for name, fen, depth, expected_uci_set, check_node_scaling in positions:
            nodes_by_threads = {}
            for n_threads in (1, 2, 4):
                clear_transposition_table(main_ctx.transposition_table)
                reset_search_context(main_ctx)
                for hc in helpers_4:
                    reset_search_context(hc)

                p_bbs, o_bbs, g_state = parse_fen(fen)
                legal_moves = [int(m) for m in generate_legal_moves(p_bbs, o_bbs, g_state)]

                best_move, score, s_nodes, q_nodes, tt_hits, comp_depth = iterative_deepening_search(
                    p_bbs,
                    o_bbs,
                    g_state,
                    depth,
                    {"optimum_time": 0, "maximum_time": 0},
                    main_ctx,
                    game_history_list=[int(g_state[4])],
                    tt_generation=1,
                    verbose=False,
                    num_threads=n_threads,
                    helper_contexts=helpers_4[: max(0, n_threads - 1)],
                )

                self.assertNotEqual(int(best_move), int(NO_MOVE), f"{name} (threads={n_threads}) returned NO_MOVE")
                self.assertIn(int(best_move), legal_moves, f"{name} (threads={n_threads}) returned illegal move")
                self.assertGreaterEqual(int(comp_depth), depth)

                total_nodes = int(s_nodes) + int(q_nodes)
                self.assertGreater(total_nodes, 0)
                nodes_by_threads[n_threads] = total_nodes

                if expected_uci_set is not None:
                    self.assertIn(
                        move_to_uci(best_move),
                        expected_uci_set,
                        f"{name} (threads={n_threads}) failed tactical solution: got {move_to_uci(best_move)} (score={score})",
                    )

            if check_node_scaling:
                self.assertGreater(
                    nodes_by_threads[4],
                    nodes_by_threads[1],
                    f"{name}: expected 4-thread total nodes ({nodes_by_threads[4]}) > 1-thread nodes ({nodes_by_threads[1]})",
                )


if __name__ == "__main__":
    unittest.main()
