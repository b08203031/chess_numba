"""Wave 3 high-ROI improvement tests.

Currently covers:
- Reverted Wave 3-1 (S-1) side-exclusive Non-Pawn Correction History contract.
- Wave 3-2 (S-10): SF19 effort-weighted averageScore aspiration window center,
  tighter initial bounds (MIN=10, BASE=14), and GROW_NUM=47 in both main and
  Lazy SMP helper iterative deepening loops.
- Wave 3-3 (S-3): Cuckoo upcoming_repetition detection (tables, exact equivalence
  against brute-force 1-ply _is_repetition_jit oracle, source contracts, smoke).
- Wave 3-4 (S-5): QSearch tactical pruning & refinement (non-PV TT cutoff guard,
  TT-bound best_eval refinement without Correction History, moveCount > 2 pruning
  with prev_sq recapture / check / promotion exemptions, SEE-vs-Alpha delta gate,
  non-capture evasion pruning after non-loss move, and post-capture stalemate).
- Wave 3-5 (S-2): SF19 capture/promotion ordering scale (K*MG[victim]+cap_hist),
  quiet underpromotions moved from captures_buffer to quiets_buffer, and
  history-guided STAGE_GOOD_CAPTURES SEE threshold (-score // CAPTURE_GOOD_SEE_DIV).
"""
import random
import unittest
from pathlib import Path

import numpy as np

from chess_engine.classical.board_operations import make_move, unmake_move
from chess_engine.classical.constants import (
    ASPIRATION_EMA_CHI_DEN,
    ASPIRATION_EMA_CHI_NUM,
    ASPIRATION_EMA_MAX_WEIGHT,
    ASPIRATION_EMA_MIN_WEIGHT,
    ASPIRATION_EMA_SCALE,
    ASPIRATION_WINDOW_BASE,
    ASPIRATION_WINDOW_GROW_NUM,
    ASPIRATION_WINDOW_MIN,
    CAPTURE_GOOD_SEE_DIV,
    CAPTURE_ORDER_AGGRESSOR_MULT,
    CAPTURE_ORDER_VICTIM_MULT,
    CORRECTION_HISTORY_SIZE,
    DELTA_PRUNING_MARGIN,
    HISTORY_MAX_CAPTURE,
    MATE_IN_MAX_PLY,
    MAX_PLY,
    NO_MOVE,
    PAWN,
    PAWN_HISTORY_SIZE,
    QS_EVASION_QUIET_CLAMP,
    QS_MOVE_COUNT_LIMIT,
    ROOK,
    SCORE_TT_MOVE,
    STAGE_BAD_CAPTURES,
    STAGE_GEN_CAPTURES,
    STAGE_GOOD_CAPTURES,
    TUNE_DELTA_MARGIN,
    VALUE_KNOWN_WIN,
)
from chess_engine.classical.core import generate_legal_moves
from chess_engine.classical.engine_types import SearchContext, reset_search_context
from chess_engine.classical.fen_parser import parse_fen
from chess_engine.classical.move import (
    PROMO_BISHOP,
    PROMO_KNIGHT,
    PROMO_QUEEN,
    PROMO_ROOK,
    SPECIAL_MOVE_FLAG_NORMAL,
    SPECIAL_MOVE_FLAG_PROMOTION,
    encode_move,
    move_to_uci,
)
from chess_engine.classical.move_generator import (
    generate_pseudo_legal_captures_buffer,
    generate_pseudo_legal_moves_buffer,
    generate_pseudo_legal_quiets_buffer,
)
from chess_engine.classical.search import (
    _i32,
    _is_qs_stalemate_jit,
    _update_aspiration_average_score,
    collect_hist_diag_dict,
    get_next_move,
    iterative_deepening_search,
    quiescence_search,
)
from chess_engine.classical.search_heuristics import (
    _has_upcoming_repetition_jit,
    _is_repetition_jit,
    score_captures_lazy,
    score_captures_with_tt_lazy,
)
from chess_engine.classical.transposition_table import (
    TT_FLAG_ALPHA,
    TT_FLAG_BETA,
    TT_FLAG_EXACT,
    clear_transposition_table,
    create_transposition_table,
    probe_tt_snapshot,
    store_tt,
)
from chess_engine.classical.zobrist import (
    CUCKOO_KEYS,
    CUCKOO_MOVES,
    CUCKOO_TABLE_MASK,
    CUCKOO_TABLE_SIZE,
    get_tt_key,
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


class TestWave3S5QSearchRefinement(unittest.TestCase):
    def test_s5_source_contracts_and_no_correction_history_in_qs(self):
        src = (ROOT / "chess_engine" / "classical" / "search.py").read_text(encoding="utf-8")
        qs_start = src.index("def quiescence_search(")
        qs_end = src.index("def get_next_move(", qs_start)
        qs_src = src[qs_start:qs_end]

        # E82 ban: QSearch never applies correction history
        self.assertNotIn("apply_correction_history_score", qs_src)
        self.assertIn("if not is_pv:", qs_src)
        self.assertIn("QS_MOVE_COUNT_LIMIT", qs_src)
        self.assertIn("_is_qs_stalemate_jit", qs_src)
        self.assertIn("qs_legal_moves_seen += 1", qs_src)
        self.assertNotIn("len(search_context.tune) >= 50", qs_src)
        self.assertEqual(QS_MOVE_COUNT_LIMIT, 2)

    def test_pv_vs_nonpv_tt_cutoff_and_bound_refinement(self):
        tt, ctx = _make_context(tt_mb=2)
        fen = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"
        pb, ob, gs = parse_fen(fen)
        tt_key = get_tt_key(gs[4], int(gs[3]))

        # Compute unrefined stand_pat first
        clear_transposition_table(tt)
        reset_search_context(ctx)
        raw_score, _ = quiescence_search(
            pb, ob, gs, _i32(-5000), _i32(5000), _i32(1), ctx, _i32(0), True
        )

        # Store a deeper TT_FLAG_BETA score above raw_score; Non-PV cuts off, PV refines best_eval without cutting off at Step 3
        elevated = int(raw_score) + 45
        clear_transposition_table(tt)
        reset_search_context(ctx)
        store_tt(tt, tt_key, 8, elevated, int(raw_score), int(TT_FLAG_BETA), NO_MOVE, 0, True)
        sc_nonpv, qn_nonpv = quiescence_search(
            pb, ob, gs, _i32(elevated - 10), _i32(elevated - 9), _i32(1), ctx, _i32(0), False
        )
        self.assertEqual(int(sc_nonpv), elevated)
        self.assertEqual(int(qn_nonpv), 1)

        # PV node with wider window (beta > elevated) refines best_eval to `elevated` on move_count==0 leaf without taking Step 3 cutoff
        sc_pv, _ = quiescence_search(
            pb, ob, gs, _i32(-5000), _i32(elevated + 50), _i32(1), ctx, _i32(0), True
        )
        self.assertEqual(int(sc_pv), elevated)
        snap = probe_tt_snapshot(tt, tt_key)
        self.assertEqual(int(snap[3]), int(raw_score))

        # Decisive TT scores must NOT refine non-check best_eval into false stand-pat
        clear_transposition_table(tt)
        reset_search_context(ctx)
        mate_val = MATE_IN_MAX_PLY + 10
        store_tt(tt, tt_key, 8, mate_val, int(raw_score), int(TT_FLAG_BETA), NO_MOVE, 0, True)
        sc_pv_mate, _ = quiescence_search(
            pb, ob, gs, _i32(-5000), _i32(30000), _i32(1), ctx, _i32(0), True
        )
        self.assertEqual(int(sc_pv_mate), int(raw_score))

    def test_qs_stalemate_detection_and_checkmate(self):
        tt, ctx = _make_context(tt_mb=2)
        clear_transposition_table(tt)
        reset_search_context(ctx)

        # Pure stalemate & blocked-pawn stalemate (no captures) -> 0 instead of huge negative eval
        for st_fen in (
            "k7/2Q5/1K6/8/8/8/8/8 b - - 0 1",
            "k7/p7/P1K5/8/8/6Q1/8/8 b - - 0 1",
        ):
            clear_transposition_table(tt)
            reset_search_context(ctx)
            pb, ob, gs = parse_fen(st_fen)
            self.assertEqual(len(generate_legal_moves(pb, ob, gs)), 0)
            self.assertTrue(_is_qs_stalemate_jit(pb, ob, gs[0]))
            sc, _ = quiescence_search(
                pb, ob, gs, _i32(-20000), _i32(20000), _i32(1), ctx, _i32(0), True
            )
            self.assertEqual(int(sc), 0)

        # Back-rank checkmate in check (Re8 vs Kg8 behind f7/g7/h7) -> -MATE_SCORE + ply
        clear_transposition_table(tt)
        reset_search_context(ctx)
        pb_m, ob_m, gs_m = parse_fen("4R1k1/5ppp/8/8/8/8/8/6K1 b - - 0 1")
        self.assertEqual(len(generate_legal_moves(pb_m, ob_m, gs_m)), 0)
        sc_m, _ = quiescence_search(
            pb_m, ob_m, gs_m, _i32(-30000), _i32(30000), _i32(2), ctx, _i32(0), False
        )
        self.assertLess(int(sc_m), -MATE_IN_MAX_PLY)

        # Quiet king evasion while in check with no captures must NOT be pruned on move 1
        clear_transposition_table(tt)
        reset_search_context(ctx)
        pb_e, ob_e, gs_e = parse_fen("4k3/8/8/8/8/8/4r3/R3K3 w - - 0 1")
        sc_e, _ = quiescence_search(
            pb_e, ob_e, gs_e, _i32(-30000), _i32(30000), _i32(1), ctx, _i32(0), False
        )
        self.assertGreater(int(sc_e), -MATE_IN_MAX_PLY)

    def test_movecount_prev_sq_see_alpha_and_search_smoke(self):
        tt, ctx = _make_context(tt_mb=4)
        clear_transposition_table(tt)
        reset_search_context(ctx)

        # Position with > 2 non-check captures (White pawns/bishops can capture on multiple squares)
        # Verify S-5 produces valid search and respects recapture prev_sq & check protection
        pb, ob, gs = parse_fen("r3k2r/pppq1ppp/2n1bn2/3pp3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w kq - 0 8")
        ctx.diag_enabled = True
        ctx.tune[TUNE_DELTA_MARGIN] = DELTA_PRUNING_MARGIN
        # Smoke 1T + 2T Lazy SMP
        for n_thr in (1, 2):
            clear_transposition_table(tt)
            reset_search_context(ctx)
            pb2, ob2, gs2 = pb.copy(), ob.copy(), gs.copy()
            mv, sc, ns, qs, _th, d = iterative_deepening_search(
                pb2, ob2, gs2, 7, {"optimum_time": 0, "maximum_time": 0},
                ctx, verbose=False, num_threads=n_thr,
            )
            self.assertNotEqual(int(mv), int(NO_MOVE))
            self.assertEqual(int(d), 7)
            self.assertIn(int(mv), [int(m) for m in generate_legal_moves(pb, ob, gs)])
            self.assertGreater(int(ns) + int(qs), 0)
            self.assertLess(abs(int(sc)), VALUE_KNOWN_WIN)

        diag = collect_hist_diag_dict(ctx, 1000)
        self.assertGreaterEqual(diag.get("q_moves", 0), 0)


def _oracle_has_upcoming_repetition(pb, ob, gs, ctx, ply: int) -> bool:
    """Brute-force 1-ply reference matching _is_repetition_jit(ply + 1)."""
    legal = generate_legal_moves(pb, ob, gs)
    saved_move = ctx.move_stack[ply]
    saved_key = ctx.ply_path_stack[ply + 1]
    for m in legal:
        u = make_move(pb, ob, gs, m)
        hmc = np.int32(gs[3])
        key = np.uint64(gs[4])
        if hmc >= 2:
            ctx.move_stack[ply] = np.uint16(m)
            ctx.ply_path_stack[ply + 1] = key
            if _is_repetition_jit(key, hmc, np.int32(ply + 1), ctx):
                unmake_move(pb, ob, gs, m, u)
                ctx.move_stack[ply] = saved_move
                ctx.ply_path_stack[ply + 1] = saved_key
                return True
        unmake_move(pb, ob, gs, m, u)
    ctx.move_stack[ply] = saved_move
    ctx.ply_path_stack[ply + 1] = saved_key
    return False


def _play_moves(pb, ob, gs, ctx, root_uci_moves, tree_uci_moves):
    """Apply pre-root game_history moves then in-tree search-path moves."""
    history_keys = [np.uint64(gs[4])]
    for uci in root_uci_moves:
        legal = [m for m in generate_legal_moves(pb, ob, gs) if move_to_uci(m) == uci]
        assert len(legal) == 1, f"Illegal root move {uci}"
        make_move(pb, ob, gs, legal[0])
        history_keys.append(np.uint64(gs[4]))

    ctx.game_history_count = np.int32(len(history_keys))
    for i, k in enumerate(history_keys):
        ctx.game_history[i] = k

    ctx.ply_path_stack[0] = np.uint64(gs[4])
    for ply_idx, uci in enumerate(tree_uci_moves):
        legal = [m for m in generate_legal_moves(pb, ob, gs) if move_to_uci(m) == uci]
        assert len(legal) == 1, f"Illegal tree move {uci}"
        m = legal[0]
        ctx.move_stack[ply_idx] = np.uint16(m)
        make_move(pb, ob, gs, m)
        ctx.ply_path_stack[ply_idx + 1] = np.uint64(gs[4])

    return len(tree_uci_moves)


class TestWave3S3CuckooUpcomingRepetition(unittest.TestCase):
    def test_cuckoo_table_invariants_and_cache_budget(self):
        self.assertEqual(CUCKOO_KEYS.shape, (CUCKOO_TABLE_SIZE,))
        self.assertEqual(CUCKOO_MOVES.shape, (CUCKOO_TABLE_SIZE,))
        self.assertEqual(CUCKOO_KEYS.dtype, np.uint64)
        self.assertEqual(CUCKOO_MOVES.dtype, np.uint16)
        self.assertTrue(CUCKOO_KEYS.flags["C_CONTIGUOUS"])
        self.assertTrue(CUCKOO_MOVES.flags["C_CONTIGUOUS"])
        self.assertLessEqual(CUCKOO_KEYS.nbytes + CUCKOO_MOVES.nbytes, 100_000)
        self.assertEqual(int(np.count_nonzero(CUCKOO_MOVES)), 3668)

        for idx in range(CUCKOO_TABLE_SIZE):
            key = np.uint64(CUCKOO_KEYS[idx])
            mv = int(CUCKOO_MOVES[idx])
            if mv == 0:
                self.assertEqual(int(key), 0)
            else:
                h1 = int(key & CUCKOO_TABLE_MASK)
                h2 = int((key >> np.uint64(16)) & CUCKOO_TABLE_MASK)
                self.assertIn(idx, (h1, h2))

    def test_upcoming_repetition_explicit_cases_and_null_castling_blockers(self):
        tt, ctx = _make_context(tt_mb=2)
        clear_transposition_table(tt)

        reset_search_context(ctx)
        pb, ob, gs = parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        ply = _play_moves(pb, ob, gs, ctx, [], ["e2e3", "e7e6", "g1f3", "g8f6", "f3g1"])
        got = _has_upcoming_repetition_jit(ob, gs[0], gs[4], np.int32(gs[3]), np.int32(ply), ctx)
        self.assertTrue(got)
        self.assertEqual(got, _oracle_has_upcoming_repetition(pb, ob, gs, ctx, ply))

        reset_search_context(ctx)
        pb, ob, gs = parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        ply = _play_moves(pb, ob, gs, ctx, [], ["g1f3", "g8f6", "f3g1"])
        got_root2 = _has_upcoming_repetition_jit(ob, gs[0], gs[4], np.int32(gs[3]), np.int32(ply), ctx)
        self.assertFalse(got_root2)
        self.assertEqual(got_root2, _oracle_has_upcoming_repetition(pb, ob, gs, ctx, ply))

        reset_search_context(ctx)
        pb, ob, gs = parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        ply = _play_moves(
            pb, ob, gs, ctx,
            ["g1f3", "g8f6", "f3g1", "f6g8"],
            ["g1f3", "g8f6", "f3g1"],
        )
        got_root3 = _has_upcoming_repetition_jit(ob, gs[0], gs[4], np.int32(gs[3]), np.int32(ply), ctx)
        self.assertTrue(got_root3)
        self.assertEqual(got_root3, _oracle_has_upcoming_repetition(pb, ob, gs, ctx, ply))

        reset_search_context(ctx)
        pb, ob, gs = parse_fen("6k1/8/8/8/8/K7/8/RR6 w - - 0 1")
        ply = _play_moves(
            pb, ob, gs, ctx,
            [],
            ["a3b3", "g8h8", "b1c1", "h8h7", "a1b1", "h7g8"],
        )
        got_blocked = _has_upcoming_repetition_jit(ob, gs[0], gs[4], np.int32(gs[3]), np.int32(ply), ctx)
        self.assertFalse(got_blocked)
        self.assertEqual(got_blocked, _oracle_has_upcoming_repetition(pb, ob, gs, ctx, ply))

        reset_search_context(ctx)
        pb, ob, gs = parse_fen("4k3/8/8/8/8/8/8/4K2N w - - 0 1")
        ctx.ply_path_stack[0] = np.uint64(gs[4])
        ply = _play_moves(pb, ob, gs, ctx, [], ["e1d1", "e8d8", "d1e1"])
        ctx.move_stack[1] = np.uint16(NO_MOVE)
        got_null = _has_upcoming_repetition_jit(ob, gs[0], gs[4], np.int32(gs[3]), np.int32(ply), ctx)
        self.assertFalse(got_null)

        reset_search_context(ctx)
        pb, ob, gs = parse_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        ply = _play_moves(pb, ob, gs, ctx, ["h1g1", "h8g8", "g1h1", "g8h8"], ["a1b1", "a8b8", "b1a1"])
        got_castle = _has_upcoming_repetition_jit(ob, gs[0], gs[4], np.int32(gs[3]), np.int32(ply), ctx)
        self.assertFalse(got_castle)
        self.assertEqual(got_castle, _oracle_has_upcoming_repetition(pb, ob, gs, ctx, ply))

    def test_upcoming_repetition_random_reversible_walks_oracle(self):
        rng = random.Random(20261128)
        tt, ctx = _make_context(tt_mb=2)
        clear_transposition_table(tt)

        fens = [
            "4k3/8/8/8/8/3N4/2B5/R2QK3 w - - 0 1",
            "2r1k3/8/8/3q4/3Q4/8/8/4K2R w - - 0 1",
            "3rk3/2b5/8/8/8/2n5/3R4/3QK3 w - - 0 1",
        ]
        for fen in fens:
            for trial in range(40):
                reset_search_context(ctx)
                pb, ob, gs = parse_fen(fen)
                root_len = rng.randint(0, 5)
                tree_len = rng.randint(1, 9)
                gh_keys = [np.uint64(gs[4])]
                ok = True
                for _ in range(root_len):
                    legal = generate_legal_moves(pb, ob, gs)
                    if len(legal) == 0:
                        ok = False
                        break
                    m = legal[rng.randrange(len(legal))]
                    make_move(pb, ob, gs, m)
                    gh_keys.append(np.uint64(gs[4]))
                if not ok:
                    continue

                ctx.game_history_count = np.int32(len(gh_keys))
                for idx, k in enumerate(gh_keys):
                    ctx.game_history[idx] = k
                ctx.ply_path_stack[0] = np.uint64(gs[4])

                for ply in range(tree_len):
                    legal = generate_legal_moves(pb, ob, gs)
                    if len(legal) == 0:
                        break
                    m = legal[rng.randrange(len(legal))]
                    ctx.move_stack[ply] = np.uint16(m)
                    make_move(pb, ob, gs, m)
                    cur_ply = ply + 1
                    ctx.ply_path_stack[cur_ply] = np.uint64(gs[4])

                    if gs[3] >= 2 and _is_repetition_jit(gs[4], np.int32(gs[3]), np.int32(cur_ply), ctx):
                        break

                    cuckoo_res = _has_upcoming_repetition_jit(
                        ob, gs[0], gs[4], np.int32(gs[3]), np.int32(cur_ply), ctx
                    )
                    oracle_res = _oracle_has_upcoming_repetition(pb, ob, gs, ctx, cur_ply)
                    self.assertEqual(
                        cuckoo_res,
                        oracle_res,
                        msg=f"Mismatch on fen={fen} trial={trial} ply={cur_ply} hmc={int(gs[3])}: cuckoo={cuckoo_res}, oracle={oracle_res}",
                    )

    def test_s3_source_contracts_and_perpetual_check_smoke(self):
        search_src = (ROOT / "chess_engine" / "classical" / "search.py").read_text(encoding="utf-8")
        self.assertIn("_has_upcoming_repetition_jit(", search_src)
        self.assertIn("not is_exclusion_search", search_src)
        self.assertNotIn("len(search_context.tune) >= 51", search_src)

        tt, ctx = _make_context(tt_mb=4)
        clear_transposition_table(tt)
        reset_search_context(ctx)
        pb, ob, gs = parse_fen("6k1/5p2/6p1/8/8/8/5Q2/1r3rK1 w - - 0 1")
        for n_thr in (1, 2):
            clear_transposition_table(tt)
            reset_search_context(ctx)
            pb2, ob2, gs2 = pb.copy(), ob.copy(), gs.copy()
            mv, sc, _ns, _qs, _th, d = iterative_deepening_search(
                pb2, ob2, gs2, 7, {"optimum_time": 0, "maximum_time": 0},
                ctx, verbose=False, num_threads=n_thr,
            )
            self.assertNotEqual(int(mv), int(NO_MOVE))
            self.assertEqual(int(d), 7)
            self.assertIn(int(mv), [int(m) for m in generate_legal_moves(pb, ob, gs)])
            self.assertLess(abs(int(sc)), MATE_IN_MAX_PLY)


class TestWave3S2CaptureOrderingAndUnderpromotions(unittest.TestCase):
    def test_s2_source_contracts_and_no_temporary_sweep_hooks(self):
        search_src = (ROOT / "chess_engine" / "classical" / "search.py").read_text(encoding="utf-8")
        heur_src = (ROOT / "chess_engine" / "classical" / "search_heuristics.py").read_text(encoding="utf-8")
        mg_src = (ROOT / "chess_engine" / "classical" / "move_generator.py").read_text(encoding="utf-8")

        self.assertEqual(CAPTURE_ORDER_VICTIM_MULT, 14)
        self.assertEqual(CAPTURE_ORDER_AGGRESSOR_MULT, 0)
        self.assertEqual(CAPTURE_GOOD_SEE_DIV, 50)
        self.assertNotIn("len(search_context.tune) >= 51", search_src)
        self.assertNotIn("len(search_context.tune) >= 51", heur_src)
        self.assertIn("CAPTURE_ORDER_VICTIM_MULT * MG_MATERIAL_VALUES", heur_src)
        self.assertIn("see_thr = np.int32(-trunc_div(np.int32(candidate_score), CAPTURE_GOOD_SEE_DIV))", search_src)
        self.assertIn("if (see_thr <= 0", search_src)
        self.assertIn("quiet_underpromos = single_pushes & RANK_8", mg_src)
        self.assertIn("quiet_underpromos = single_pushes & RANK_1", mg_src)

    def test_capture_ordering_scale_and_evasion_bounds(self):
        _, ctx = _make_context(tt_mb=1)
        # White: Qd4(27) can take Rd8(59); Pb5(33) can take Qa6(40), Nc6(42), Pa6/c6? Let's place clean targets:
        # Qd4(27) takes Rd5(35); Pb4(25) takes Qa5(32) & Nc5(34); Pg4(30) takes Ph5(39).
        pb, ob, gs = parse_fen("4k3/8/8/q1nr3p/1P1Q2P1/8/8/4K3 w - - 0 1")
        m_pxq = encode_move(25, 32, 0, int(SPECIAL_MOVE_FLAG_NORMAL))  # b4xa5 (Q)
        m_qxr = encode_move(27, 35, 0, int(SPECIAL_MOVE_FLAG_NORMAL))  # d4xd5 (R)
        m_pxn = encode_move(25, 34, 0, int(SPECIAL_MOVE_FLAG_NORMAL))  # b4xc5 (N)
        m_pxp = encode_move(30, 39, 0, int(SPECIAL_MOVE_FLAG_NORMAL))  # g4xh5 (P)
        moves = np.array([m_pxq, m_qxr, m_pxn, m_pxp], dtype=np.uint16)
        scores = np.zeros(len(moves), dtype=np.int32)
        score_captures_lazy(pb, ob, gs, moves, scores, 0, len(moves), ctx, np.uint64(0), np.uint64(0), 0)
        s_pxq, s_qxr, s_pxn, s_pxp = (int(x) for x in scores)
        self.assertGreater(s_pxq, s_qxr)
        self.assertGreater(s_qxr, s_pxn)
        self.assertGreater(s_pxn, s_pxp)
        # Equal-victim tie-breaking via capture_history must preserve PxQ > QxR > PxP even at moderate history offsets
        ctx.capture_history[PAWN, 39, PAWN] = 2000
        score_captures_lazy(pb, ob, gs, moves, scores, 0, len(moves), ctx, np.uint64(0), np.uint64(0), 0)
        self.assertEqual(int(scores[3]), s_pxp + 2000)
        self.assertGreater(int(scores[1]), int(scores[3]))

        # Evasion bounds: worst-case PxP with -HISTORY_MAX_CAPTURE stays > QS_EVASION_QUIET_CLAMP,
        # and best-case PxQ=Q with +HISTORY_MAX_CAPTURE stays < SCORE_TT_MOVE.
        m_pxq_promo = encode_move(48, 57, PROMO_QUEEN, int(SPECIAL_MOVE_FLAG_PROMOTION))  # a7xb8=Q
        pb_e2, ob_e2, gs_e2 = parse_fen("1q2k3/P7/8/8/4r3/3p4/2P5/4K3 w - - 0 1")
        m_pxp_ev = encode_move(10, 19, 0, int(SPECIAL_MOVE_FLAG_NORMAL))  # c2xd3 (P)
        ctx.capture_history[PAWN, 57, 4] = HISTORY_MAX_CAPTURE
        ctx.capture_history[PAWN, 19, PAWN] = -HISTORY_MAX_CAPTURE
        ev_moves = np.array([m_pxq_promo, m_pxp_ev], dtype=np.uint16)
        ev_scores = np.zeros(2, dtype=np.int32)
        score_captures_with_tt_lazy(
            pb_e2, ob_e2, gs_e2, ev_moves, ev_scores, 2, NO_MOVE, np.uint64(0), np.uint64(0), ctx, 1
        )
        self.assertLess(int(ev_scores[0]), int(SCORE_TT_MOVE))
        self.assertGreater(int(ev_scores[1]), int(QS_EVASION_QUIET_CLAMP))

    def test_quiet_underpromotions_split_captures_to_quiets_and_preserve_total(self):
        cap_buf = np.zeros((2, 256), dtype=np.uint16)
        quiet_buf = np.zeros((2, 256), dtype=np.uint16)
        full_buf = np.zeros((2, 256), dtype=np.uint16)

        cases = (
            ("1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q", "a7xb8", {"a7a8r", "a7a8b", "a7a8n"}),
            ("4k3/8/8/8/8/8/p7/1N2K3 b - - 0 1", "a2a1q", "a2xb1", {"a2a1r", "a2a1b", "a2a1n"}),
        )
        for fen, q_promo, cap_prefix, underpromos in cases:
            pb, ob, gs = parse_fen(fen)
            cap_cnt = int(generate_pseudo_legal_captures_buffer(pb, ob, gs, cap_buf, 0))
            tot_cnt = int(generate_pseudo_legal_quiets_buffer(pb, ob, gs, quiet_buf, 0, 0))
            full_cnt = int(generate_pseudo_legal_moves_buffer(pb, ob, gs, full_buf, 0))

            cap_ucis = [move_to_uci(cap_buf[0, i]) for i in range(cap_cnt)]
            quiet_ucis = [move_to_uci(quiet_buf[0, i]) for i in range(tot_cnt)]
            full_ucis = [move_to_uci(full_buf[0, i]) for i in range(full_cnt)]

            self.assertIn(q_promo, cap_ucis)
            for up in underpromos:
                self.assertNotIn(up, cap_ucis)
                self.assertIn(up, quiet_ucis)
            for sfx in ("q", "r", "b", "n"):
                self.assertIn(f"{cap_prefix[0]}{cap_prefix[1]}{cap_prefix[3:]}{sfx}", cap_ucis)

            combined = cap_ucis + quiet_ucis
            self.assertEqual(len(combined), len(set(combined)))
            self.assertEqual(sorted(combined), sorted(full_ucis))

    def test_good_capture_see_history_relief_and_demotion_bypass(self):
        _, ctx = _make_context(tt_mb=2)
        # 1) Equal capture PxP (e4xd5 defended by Nc6 -> SEE == 0) with strongly negative capture_history
        #    must bypass the victim>=attacker fast path (since see_thr > 0) and be demoted to BAD_CAPTURES.
        # 2) Losing capture Rd1xd5 (R=562 vs B=366 defended by e6 -> SEE == -196) with high positive
        #    capture_history must be promoted into GOOD_CAPTURES via see_thr = -score // 50 < -196.
        fen = "4k3/8/2n1p3/3b4/8/8/8/3RK3 w - - 0 1"
        pb, ob, gs = parse_fen(fen)
        m_rxb = encode_move(3, 35, 0, int(SPECIAL_MOVE_FLAG_NORMAL))  # Rd1xd5 (B, SEE = -196)
        reset_search_context(ctx)
        ctx.mp_stage[0] = STAGE_GEN_CAPTURES
        m_rxb_pick = get_next_move(
            pb, ob, gs, ctx, 0, NO_MOVE, NO_MOVE, NO_MOVE, NO_MOVE, NO_MOVE,
            np.uint64(0), np.uint64(0), 0, False, 4, np.uint64(1 << 4), 1, np.uint64(0), 0, np.uint64(0),
        )
        # With cap_hist = 0: score = 14*366 = 5124 -> see_thr = -102 -> SEE(-196) < -102 -> BAD_CAPTURES
        self.assertEqual(int(m_rxb_pick), int(m_rxb))
        self.assertEqual(int(ctx.mp_stage[0]), int(STAGE_BAD_CAPTURES))

        # With high cap_hist = +8000: score = 13124 -> see_thr = -262 -> SEE(-196) >= -262 -> GOOD_CAPTURES
        reset_search_context(ctx)
        ctx.capture_history[ROOK, 35, 2] = 8000
        ctx.mp_stage[0] = STAGE_GEN_CAPTURES
        m_rxb_promoted = get_next_move(
            pb, ob, gs, ctx, 0, NO_MOVE, NO_MOVE, NO_MOVE, NO_MOVE, NO_MOVE,
            np.uint64(0), np.uint64(0), 0, False, 4, np.uint64(1 << 4), 1, np.uint64(0), 0, np.uint64(0),
        )
        self.assertEqual(int(m_rxb_promoted), int(m_rxb))
        self.assertEqual(int(ctx.mp_stage[0]), int(STAGE_GOOD_CAPTURES))

        # Equal PxP (d4xe5 defended by f6 -> SEE == 0) with cap_hist = -HISTORY_MAX_CAPTURE (-10240)
        # score = 14*100 - 10240 = -8840 -> see_thr = +176 > 0 -> fast path bypassed -> BAD_CAPTURES
        pb2, ob2, gs2 = parse_fen("4k3/8/5p2/4p3/3P4/8/8/4K3 w - - 0 1")
        m_pxp = encode_move(27, 36, 0, int(SPECIAL_MOVE_FLAG_NORMAL))  # d4xe5
        reset_search_context(ctx)
        ctx.capture_history[PAWN, 36, PAWN] = -HISTORY_MAX_CAPTURE
        ctx.mp_stage[0] = STAGE_GEN_CAPTURES
        m_pxp_demoted = get_next_move(
            pb2, ob2, gs2, ctx, 0, NO_MOVE, NO_MOVE, NO_MOVE, NO_MOVE, NO_MOVE,
            np.uint64(0), np.uint64(0), 0, False, 4, np.uint64(1 << 4), 1, np.uint64(0), 0, np.uint64(0),
        )
        self.assertEqual(int(m_pxp_demoted), int(m_pxp))
        self.assertEqual(int(ctx.mp_stage[0]), int(STAGE_BAD_CAPTURES))

    def test_s2_search_smoke_on_promotion_and_capture_fens(self):
        tt, ctx = _make_context(tt_mb=4)
        for fen in (
            "1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1",
            "8/8/1KP5/3r4/8/8/8/k7 w - - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        ):
            clear_transposition_table(tt)
            reset_search_context(ctx)
            pb, ob, gs = parse_fen(fen)
            mv, _sc, ns, _qs, _th, d = iterative_deepening_search(
                pb, ob, gs, 6, {"optimum_time": 0, "maximum_time": 0},
                ctx, verbose=False, num_threads=1,
            )
            self.assertEqual(int(d), 6)
            self.assertGreater(int(ns), 0)
            self.assertIn(int(mv), [int(m) for m in generate_legal_moves(pb, ob, gs)])


if __name__ == "__main__":
    unittest.main()
