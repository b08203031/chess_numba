import contextlib
import io
import unittest
from pathlib import Path
from unittest import mock
import numpy as np

from chess_engine.classical.constants import (
    INFINITY,
    MATE_IN_MAX_PLY,
    MATE_SCORE,
    MAX_PLY,
    NO_MOVE,
    VALUE_KNOWN_WIN,
)
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
            ("startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 9, None),
            ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 9, None),
            (
                "mate_in_2",
                "r2qkb1r/pp2nppp/3p4/2pNN1B1/2BnP3/3P4/PPP2PPP/R2bK2R w KQkq - 1 1",
                6,
                {"d5f6"},
            ),
        ]

        for name, fen, depth, expected_uci_set in positions:
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
                self.assertEqual(int(comp_depth), depth)

                total_nodes = int(s_nodes) + int(q_nodes)
                self.assertGreater(total_nodes, 0)

                if expected_uci_set is not None:
                    self.assertIn(
                        move_to_uci(best_move),
                        expected_uci_set,
                        f"{name} (threads={n_threads}) failed tactical solution: got {move_to_uci(best_move)} (score={score})",
                    )

        # Timed throughput scaling check (4 threads vs 1 thread under a fixed movetime budget)
        timed_nodes = {}
        kiwipete_fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
        for n_threads in (1, 4):
            clear_transposition_table(main_ctx.transposition_table)
            reset_search_context(main_ctx)
            for hc in helpers_4:
                reset_search_context(hc)
            p_bbs, o_bbs, g_state = parse_fen(kiwipete_fen)
            legal_moves = [int(m) for m in generate_legal_moves(p_bbs, o_bbs, g_state)]
            best_move, _, s_nodes, q_nodes, _, comp_depth = iterative_deepening_search(
                p_bbs,
                o_bbs,
                g_state,
                64,
                {"optimum_time": 120, "maximum_time": 120},
                main_ctx,
                game_history_list=[int(g_state[4])],
                tt_generation=1,
                verbose=False,
                num_threads=n_threads,
                helper_contexts=helpers_4[: max(0, n_threads - 1)],
            )
            self.assertIn(int(best_move), legal_moves)
            self.assertGreaterEqual(int(comp_depth), 1)
            timed_nodes[n_threads] = int(s_nodes) + int(q_nodes)
        self.assertGreater(
            timed_nodes[4],
            timed_nodes[1],
            f"kiwipete timed scaling: expected 4-thread nodes ({timed_nodes[4]}) > 1-thread nodes ({timed_nodes[1]})",
        )

    def test_sf19_get_best_thread_voting_and_consensus(self):
        from chess_engine.classical.search import RootThreadResult, select_best_thread

        m1, m2, m3 = 101, 202, 303

        # 1. Score-weighted consensus beats single slightly higher thread:
        #    min_score = 40 -> M1 gets (50 - 40 + 14) = 24;
        #    M2 gets 2 * (40 - 40 + 14) = 28 -> M2 wins.
        threads = [
            RootThreadResult(0, m1, 50, depth=8, pv=[m1, 11]),
            RootThreadResult(1, m2, 40, depth=8, pv=[m2, 12]),
            RootThreadResult(2, m2, 40, depth=8, pv=[m2, 12]),
        ]
        best = select_best_thread(threads)
        self.assertEqual(best.move, m2)
        self.assertEqual(best.thread_id, 1)

        # 2. Deeper-lower-score / single much higher score wins when vote weight dominates:
        #    min_score = 15 -> M1 = 14, M2 = 80 - 15 + 14 = 79 -> M2 wins.
        threads_high = [
            RootThreadResult(0, m1, 15, depth=6, pv=[m1]),
            RootThreadResult(1, m2, 80, depth=7, pv=[m2, 21]),
        ]
        self.assertEqual(select_best_thread(threads_high).move, m2)

        # 3. Same move consensus where helper has deeper/lower score:
        #    votes[m1] is equal across threads voting m1 -> longer PV wins, else (score, depth).
        threads_same_move_pv = [
            RootThreadResult(0, m1, 50, depth=6, pv=[m1, 11]),
            RootThreadResult(1, m1, 35, depth=7, pv=[m1, 11, 12, 13]),
        ]
        best_same = select_best_thread(threads_same_move_pv)
        self.assertEqual(best_same.thread_id, 1)
        self.assertEqual(best_same.pv_len, 4)

        # 4. Invalid threads (NO_MOVE or -INFINITY) are ignored and do not skew min_score.
        threads_with_invalid = [
            RootThreadResult(0, int(NO_MOVE), -INFINITY, depth=0, pv=[]),
            RootThreadResult(1, m3, 12, depth=3, pv=[m3]),
            RootThreadResult(2, m1, 30, depth=4, pv=[m1, 11]),
        ]
        best_valid = select_best_thread(threads_with_invalid)
        self.assertEqual(best_valid.move, m1)
        self.assertEqual(best_valid.thread_id, 2)

    def test_sf19_get_best_thread_decisive_mate_and_loss_guards(self):
        from chess_engine.classical.search import RootThreadResult, select_best_thread

        m1, m2, m3 = 111, 222, 333
        mate_in_2 = MATE_SCORE - 3   # 29997
        mate_in_3 = MATE_SCORE - 5   # 29995
        mate_in_4 = MATE_SCORE - 7   # 29993

        # 1. Exact decisive beats non-decisive even if non-decisive has multi-thread vote lead.
        threads = [
            RootThreadResult(0, m1, 450, depth=9, pv=[m1, 10]),
            RootThreadResult(1, m1, 440, depth=9, pv=[m1, 10]),
            RootThreadResult(2, m1, 460, depth=9, pv=[m1, 10]),
            RootThreadResult(3, m2, mate_in_4, depth=8, pv=[m2, 20, 21]),
        ]
        self.assertEqual(select_best_thread(threads).move, m2)

        # 2. Among exact decisive mates, shorter mate (larger abs(score)) wins;
        #    a longer mate does NOT override a shorter mate.
        threads_mate = [
            RootThreadResult(0, m1, mate_in_3, depth=8, pv=[m1, 10, 11]),
            RootThreadResult(1, m2, mate_in_2, depth=8, pv=[m2, 20]),
            RootThreadResult(2, m3, mate_in_4, depth=9, pv=[m3, 30, 31, 32]),
        ]
        self.assertEqual(select_best_thread(threads_mate).move, m2)

        # Reverse order: Thread 0 already has shorter mate-in-2.
        threads_mate_rev = [
            RootThreadResult(0, m2, mate_in_2, depth=8, pv=[m2, 20]),
            RootThreadResult(1, m1, mate_in_3, depth=9, pv=[m1, 10, 11]),
        ]
        self.assertEqual(select_best_thread(threads_mate_rev).thread_id, 0)

        # 3. Inexact decisive scores (inexact_lower or inexact_upper) are rejected
        #    against an exact decisive mate and never gain exact-decisive priority.
        threads_inexact = [
            RootThreadResult(0, m1, mate_in_2, depth=8, pv=[m1], inexact_lower=True),
            RootThreadResult(1, m2, mate_in_4, depth=7, pv=[m2, 20, 21]),
            RootThreadResult(2, m3, mate_in_2, depth=8, pv=[m3], inexact_upper=True),
        ]
        self.assertEqual(select_best_thread(threads_inexact).move, m2)

        threads_inexact_vs_exact = [
            RootThreadResult(0, m2, mate_in_3, depth=7, pv=[m2, 20]),
            RootThreadResult(1, m1, mate_in_2, depth=9, pv=[m1], inexact_lower=True),
        ]
        self.assertEqual(select_best_thread(threads_inexact_vs_exact).move, m2)

        # 4. Unconfirmed is_loss cannot beat non-losing moves in voting, even if
        #    other threads voted for that same move.
        unconfirmed_loss = -(VALUE_KNOWN_WIN + 500)
        inexact_mated = -(MATE_SCORE - 5)
        threads_loss_guard = [
            RootThreadResult(0, m1, 20, depth=8, pv=[m1, 10]),
            RootThreadResult(1, m2, unconfirmed_loss, depth=8, pv=[m2, 20]),
            RootThreadResult(2, m3, inexact_mated, depth=8, pv=[m3], inexact_lower=True),
            RootThreadResult(3, m2, 25, depth=8, pv=[m2, 20, 21]),
        ]
        chosen = select_best_thread(threads_loss_guard)
        self.assertEqual(chosen.thread_id, 3)
        self.assertEqual(chosen.move, m2)
        self.assertEqual(chosen.score, 25)

        # Non-losing helper overrides Thread 0 when Thread 0 has an unconfirmed loss.
        threads_t0_loss = [
            RootThreadResult(0, m1, unconfirmed_loss, depth=6, pv=[m1]),
            RootThreadResult(1, m2, -150, depth=6, pv=[m2, 20]),
        ]
        self.assertEqual(select_best_thread(threads_t0_loss).move, m2)

        # 5. Tiebreaks when votes are equal: longer PV wins, then higher score, then higher depth.
        threads_pv_tie = [
            RootThreadResult(0, m1, 30, depth=8, pv=[m1, 10]),
            RootThreadResult(1, m2, 30, depth=7, pv=[m2, 20, 21, 22]),
        ]
        self.assertEqual(select_best_thread(threads_pv_tie).move, m2)

        threads_score_tie = [
            RootThreadResult(0, m1, 30, depth=8, pv=[m1, 10]),
            RootThreadResult(1, m1, 40, depth=7, pv=[m1, 10]),
        ]
        self.assertEqual(select_best_thread(threads_score_tie).thread_id, 1)

        threads_depth_tie = [
            RootThreadResult(0, m1, 30, depth=7, pv=[m1, 10]),
            RootThreadResult(1, m2, 30, depth=9, pv=[m2, 20]),
        ]
        self.assertEqual(select_best_thread(threads_depth_tie).move, m2)

        # Exact tie preserves Thread 0.
        threads_exact_tie = [
            RootThreadResult(0, m1, 30, depth=8, pv=[m1, 10]),
            RootThreadResult(1, m2, 30, depth=8, pv=[m2, 20]),
        ]
        self.assertEqual(select_best_thread(threads_exact_tie).thread_id, 0)

    def test_root_iteration_guards_aborted_loss_and_forgotten_mate(self):
        from chess_engine.classical.search import apply_root_iteration_guards

        m1, m2 = 101, 202
        mate_in_3 = MATE_SCORE - 5
        mate_in_4 = MATE_SCORE - 7
        mate_in_2 = MATE_SCORE - 3

        # 1. abortedLossSearch with prior completed depth rolls back to prior depth
        res = apply_root_iteration_guards(
            cand_valid=True,
            cand_move=m2,
            cand_score=-(MATE_SCORE - 3),
            cand_uci_score=-(MATE_SCORE - 3),
            cand_depth=6,
            cand_inexact_lower=False,
            cand_inexact_upper=False,
            cand_pv=[m2, 22],
            last_best_valid=True,
            last_best_move=m1,
            last_best_score=40,
            last_best_depth=5,
            last_best_pv=[m1, 11],
            current_depth=6,
            stopped_in_iter=True,
        )
        self.assertEqual(res[0], m1)
        self.assertEqual(res[1], 40)
        self.assertEqual(res[3], 5)
        self.assertFalse(res[4])
        self.assertFalse(res[5])
        self.assertEqual(res[6], [m1, 11])
        self.assertTrue(res[12])  # rolled_back

        # 2. abortedLossSearch at depth 1 (no prior completed depth) marks inexact_lower=True
        res_d1 = apply_root_iteration_guards(
            cand_valid=True,
            cand_move=m1,
            cand_score=-(MATE_SCORE - 3),
            cand_uci_score=-(MATE_SCORE - 3),
            cand_depth=1,
            cand_inexact_lower=False,
            cand_inexact_upper=False,
            cand_pv=[m1],
            last_best_valid=False,
            last_best_move=int(NO_MOVE),
            last_best_score=-INFINITY,
            last_best_depth=0,
            last_best_pv=[],
            current_depth=1,
            stopped_in_iter=True,
        )
        self.assertEqual(res_d1[0], m1)
        self.assertTrue(res_d1[4])  # inexact_lower
        self.assertFalse(res_d1[12])

        # 3. Aborted fail-low loss (inexact_upper=True) is NOT rolled back
        res_fail_low = apply_root_iteration_guards(
            cand_valid=True,
            cand_move=m1,
            cand_score=-(VALUE_KNOWN_WIN + 500),
            cand_uci_score=10,
            cand_depth=6,
            cand_inexact_lower=False,
            cand_inexact_upper=True,
            cand_pv=[m1],
            last_best_valid=True,
            last_best_move=m1,
            last_best_score=40,
            last_best_depth=5,
            last_best_pv=[m1, 11],
            current_depth=6,
            stopped_in_iter=True,
        )
        self.assertEqual(res_fail_low[1], -(VALUE_KNOWN_WIN + 500))
        self.assertTrue(res_fail_low[5])  # inexact_upper preserved
        self.assertFalse(res_fail_low[12])

        # 4. forgottenMate rolls back downgraded or inexact mate on completed/aborted iteration
        res_forgot = apply_root_iteration_guards(
            cand_valid=True,
            cand_move=m2,
            cand_score=mate_in_4,
            cand_uci_score=mate_in_4,
            cand_depth=6,
            cand_inexact_lower=False,
            cand_inexact_upper=False,
            cand_pv=[m2, 22],
            last_best_valid=True,
            last_best_move=m1,
            last_best_score=mate_in_3,
            last_best_depth=5,
            last_best_pv=[m1, 11, 12],
            current_depth=6,
            stopped_in_iter=False,
        )
        self.assertEqual(res_forgot[0], m1)
        self.assertEqual(res_forgot[1], mate_in_3)
        self.assertEqual(res_forgot[3], 6)  # completed depth retained
        self.assertEqual(res_forgot[6], [m1, 11, 12])
        self.assertEqual(res_forgot[9], mate_in_3)  # last_best_score unchanged
        self.assertTrue(res_forgot[12])

        # Inexact mate on aborted iteration also triggers forgottenMate rollback
        res_forgot_inexact = apply_root_iteration_guards(
            cand_valid=True,
            cand_move=m2,
            cand_score=mate_in_2,
            cand_uci_score=mate_in_3,
            cand_depth=6,
            cand_inexact_lower=True,
            cand_inexact_upper=False,
            cand_pv=[m2],
            last_best_valid=True,
            last_best_move=m1,
            last_best_score=mate_in_3,
            last_best_depth=5,
            last_best_pv=[m1, 11, 12],
            current_depth=6,
            stopped_in_iter=True,
        )
        self.assertEqual(res_forgot_inexact[0], m1)
        self.assertEqual(res_forgot_inexact[1], mate_in_3)
        self.assertEqual(res_forgot_inexact[3], 5)
        self.assertFalse(res_forgot_inexact[4])
        self.assertTrue(res_forgot_inexact[12])

        # 5. Strictly shorter exact mate supersedes previous mate
        res_shorter = apply_root_iteration_guards(
            cand_valid=True,
            cand_move=m2,
            cand_score=mate_in_2,
            cand_uci_score=mate_in_2,
            cand_depth=6,
            cand_inexact_lower=False,
            cand_inexact_upper=False,
            cand_pv=[m2, 22],
            last_best_valid=True,
            last_best_move=m1,
            last_best_score=mate_in_3,
            last_best_depth=5,
            last_best_pv=[m1, 11, 12],
            current_depth=6,
            stopped_in_iter=False,
        )
        self.assertEqual(res_shorter[0], m2)
        self.assertEqual(res_shorter[1], mate_in_2)
        self.assertEqual(res_shorter[9], mate_in_2)
        self.assertFalse(res_shorter[12])

    def test_root_partial_iteration_preservation_live(self):
        from chess_engine.classical.search import (
            _b_false,
            _b_true,
            _i32,
            _lazy_smp_helper_loop_jit,
            _search,
            _u16,
            warmup_search_jit,
        )

        ctx = _make_main_context(tt_mb=4)
        warmup_search_jit(ctx)
        clear_transposition_table(ctx.transposition_table)
        reset_search_context(ctx)

        # Kiwipete depth 6 with node_budget=4096 completes at least one root move before
        # crossing 4096 nodes, so _search at ply==0 stops mid-iteration and preserves
        # the finished root candidate, its score, and pv_table[0].
        p_bbs, o_bbs, g_state = parse_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
        )
        legal_moves = [int(m) for m in generate_legal_moves(p_bbs, o_bbs, g_state)]

        ctx.nodes_searched = np.uint64(0)
        ctx.nodes_searched_array[0] = np.uint64(0)
        ctx.nodes_searched_array[1] = np.uint64(4096)
        ctx.stop_flag[0] = False
        res = _search(
            p_bbs,
            o_bbs,
            g_state,
            _i32(6),
            _i32(-INFINITY),
            _i32(INFINITY),
            ctx,
            _i32(0),
            _u16(NO_MOVE),
            _b_true(),
            _b_false(),
        )
        self.assertTrue(bool(ctx.stop_flag[0]))
        self.assertNotEqual(int(res[1]), int(NO_MOVE))
        self.assertIn(int(res[1]), legal_moves)
        self.assertEqual(int(ctx.pv_table[0, 0]), int(res[1]))

        # When stopped during move 1 (before move 1 completes), _search returns (0, NO_MOVE).
        clear_transposition_table(ctx.transposition_table)
        reset_search_context(ctx)
        ctx.nodes_searched = np.uint64(4000)
        ctx.nodes_searched_array[0] = np.uint64(4000)
        ctx.nodes_searched_array[1] = np.uint64(4096)
        ctx.stop_flag[0] = False
        res_move1_stopped = _search(
            p_bbs,
            o_bbs,
            g_state,
            _i32(6),
            _i32(-INFINITY),
            _i32(INFINITY),
            ctx,
            _i32(0),
            _u16(NO_MOVE),
            _b_true(),
            _b_false(),
        )
        self.assertTrue(bool(ctx.stop_flag[0]))
        self.assertEqual(int(res_move1_stopped[1]), int(NO_MOVE))

        # When stop_flag[0] is already True on entry at ply==0, _search returns NO_MOVE
        # without corrupting the caller's preserved candidate.
        res_prestopped = _search(
            p_bbs,
            o_bbs,
            g_state,
            _i32(6),
            _i32(-INFINITY),
            _i32(INFINITY),
            ctx,
            _i32(0),
            _u16(NO_MOVE),
            _b_true(),
            _b_false(),
        )
        self.assertEqual(int(res_prestopped[1]), int(NO_MOVE))

        # Helper loop preserves valid root candidate and PV when stopped mid-search
        import threading
        import time

        clear_transposition_table(ctx.transposition_table)
        reset_search_context(ctx)
        helper_ctx = create_smp_helper_context(ctx)
        helper_out = [None]
        helper_ctx.stop_flag[0] = False

        def _run_helper():
            helper_out[0] = _lazy_smp_helper_loop_jit(
                p_bbs.copy(),
                o_bbs.copy(),
                g_state.copy(),
                np.int32(16),
                helper_ctx,
                np.int32(1),
                ctx.game_history,
                np.int32(0),
                np.uint8(0),
            )

        t = threading.Thread(target=_run_helper, daemon=True)
        t.start()
        deadline = time.monotonic() + 5.0
        while int(helper_ctx.nodes_searched_array[0]) < 4096 and time.monotonic() < deadline:
            time.sleep(0.002)
        helper_ctx.stop_flag[0] = True
        t.join(timeout=5.0)
        self.assertFalse(t.is_alive())
        h_res = helper_out[0]
        self.assertIsNotNone(h_res)
        h_move, h_score, h_sn, h_qn, _, h_depth, h_uci, h_il, h_iu, h_pv_len = h_res
        self.assertIn(int(h_move), legal_moves)
        self.assertGreater(int(h_score), -INFINITY)
        self.assertGreaterEqual(int(h_depth), 1)
        self.assertFalse(bool(h_il) and bool(h_iu))
        self.assertGreater(int(h_pv_len), 0)
        self.assertEqual(int(helper_ctx.last_iteration_pv_len), int(h_pv_len))
        self.assertEqual(int(helper_ctx.last_iteration_pv[0]), int(h_move))

    def test_fixed_depth_vs_voting_and_helper_uci_output(self):
        from chess_engine.classical import search as search_mod

        main_ctx = _make_main_context(tt_mb=4)
        search_mod.warmup_search_jit(main_ctx)
        helper_ctx = create_smp_helper_context(main_ctx)
        p_bbs, o_bbs, g_state = parse_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        )
        legal_moves = [int(m) for m in generate_legal_moves(p_bbs, o_bbs, g_state)]
        alt_move = next(m for m in legal_moves if move_to_uci(m) == "d2d4")
        alt_reply = next(
            m for m in generate_legal_moves(
                *parse_fen("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1")
            )
            if move_to_uci(m) == "d7d5"
        )

        def _fake_helper_loop(pb, ob, gs, max_d, h_ctx, tid, gh, gh_cnt, tt_gen):
            h_ctx.last_iteration_pv[0] = np.uint16(alt_move)
            h_ctx.last_iteration_pv[1] = np.uint16(alt_reply)
            h_ctx.last_iteration_pv_len = np.int32(2)
            return (
                np.uint16(alt_move),
                np.int32(550),
                np.uint64(1234),
                np.uint64(567),
                np.uint64(89),
                np.int32(5),
                np.int32(550),
                False,
                False,
                np.int32(2),
            )

        with mock.patch.object(search_mod, "_lazy_smp_helper_loop_jit", side_effect=_fake_helper_loop):
            # 1. Pure fixed-depth (max_depth < MAX_PLY, no time/node limits) keeps Thread 0
            clear_transposition_table(main_ctx.transposition_table)
            reset_search_context(main_ctx)
            reset_search_context(helper_ctx)
            fixed_res = search_mod.iterative_deepening_search(
                p_bbs,
                o_bbs,
                g_state,
                3,
                {"optimum_time": 0, "maximum_time": 0},
                main_ctx,
                verbose=False,
                num_threads=2,
                helper_contexts=[helper_ctx],
            )
            self.assertNotEqual(int(fixed_res[1]), 550)
            self.assertEqual(int(fixed_res[5]), 3)

            # 2. Node-limited (non-fixed-depth) runs SF19 voting -> helper wins and emits UCI info
            clear_transposition_table(main_ctx.transposition_table)
            reset_search_context(main_ctx)
            reset_search_context(helper_ctx)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                voted_res = search_mod.iterative_deepening_search(
                    p_bbs,
                    o_bbs,
                    g_state,
                    MAX_PLY,
                    {"optimum_time": 0, "maximum_time": 0, "nodes_limit": 4000},
                    main_ctx,
                    verbose=True,
                    num_threads=2,
                    helper_contexts=[helper_ctx],
                )
            self.assertEqual(int(voted_res[0]), int(alt_move))
            self.assertEqual(int(voted_res[1]), 550)
            self.assertEqual(int(voted_res[5]), 5)
            out_lines = [ln for ln in buf.getvalue().splitlines() if ln.startswith("info depth ")]
            self.assertTrue(out_lines)
            self.assertIn("info depth 5 score cp 550 ", out_lines[-1])
            self.assertIn(" pv d2d4 d7d5", out_lines[-1])

    def test_multithreaded_timed_and_nodes_smoke(self):
        from chess_engine.classical.search import iterative_deepening_search, warmup_search_jit

        main_ctx = _make_main_context(tt_mb=8)
        warmup_search_jit(main_ctx)
        helpers = [create_smp_helper_context(main_ctx) for _ in range(3)]
        fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"

        for time_cfg in (
            {"optimum_time": 60, "maximum_time": 80},
            {"optimum_time": 0, "maximum_time": 0, "nodes_limit": 25000},
        ):
            clear_transposition_table(main_ctx.transposition_table)
            reset_search_context(main_ctx)
            for hc in helpers:
                reset_search_context(hc)
            p_bbs, o_bbs, g_state = parse_fen(fen)
            legal_moves = [int(m) for m in generate_legal_moves(p_bbs, o_bbs, g_state)]

            best_move, score, s_nodes, q_nodes, tt_hits, comp_depth = iterative_deepening_search(
                p_bbs,
                o_bbs,
                g_state,
                MAX_PLY,
                time_cfg,
                main_ctx,
                game_history_list=[int(g_state[4])],
                tt_generation=1,
                verbose=False,
                num_threads=4,
                helper_contexts=helpers,
            )
            self.assertNotEqual(int(best_move), int(NO_MOVE))
            self.assertIn(int(best_move), legal_moves)
            self.assertGreaterEqual(int(comp_depth), 1)
            self.assertGreater(int(s_nodes) + int(q_nodes), 0)
            self.assertGreater(int(score), -INFINITY)
            self.assertLess(int(score), INFINITY)


if __name__ == "__main__":
    unittest.main()
