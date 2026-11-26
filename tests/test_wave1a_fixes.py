"""Wave 1A regression tests: safety / interface / endgame correctness fixes.

Covers audit bugs #4 (MAX_PLY guard), #5 (Polyglot castling/EP), #6 (mate_kbnk), #14 (KBPsK b/g),
#15 (KRPKN removal), #16 (KXK stalemate), #17 (eval_weights fallback), #18 (FEN sanitation),
#25 (KRPPKRP off-by-one), #29 (reset_search_context), #30 (move_to_uci NO_MOVE).
"""
import inspect
import os
import unittest

import numpy as np

from chess_engine.classical import engine_types
from chess_engine.classical.constants import MATE_IN_MAX_PLY, MAX_PLY, SCALE_FACTOR_DRAW
from chess_engine.classical.endgame import evaluate_special_endgame, get_endgame_scale_factor
from chess_engine.classical.engine_types import SearchContext, reset_search_context
from chess_engine.classical.fen_parser import parse_fen
from chess_engine.classical.move import SPECIAL_MOVE_FLAG_CASTLING, SPECIAL_MOVE_FLAG_EN_PASSANT, encode_move
from chess_engine.classical.move import get_special_move_flag, move_to_uci
from chess_engine.classical.move_generator import generate_legal_moves
from chess_engine.classical.opening_book import OpeningBook, _convert_polyglot_move_to_engine_move
from chess_engine.classical.transposition_table import create_transposition_table

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _special(fen):
    b = parse_fen(fen)
    hit, score = evaluate_special_endgame(b[0], b[1], b[2], np.int32(0))
    return bool(hit), int(score)


def _scale(fen, eg_score=100):
    b = parse_fen(fen)
    return int(get_endgame_scale_factor(b[0], b[1], b[2], np.int32(0), np.int32(eg_score)))


def _poly(from_file, from_rank, to_file, to_rank, promo=0):
    return np.uint16(to_file | (to_rank << 3) | (from_file << 6) | (from_rank << 9) | (promo << 12))


class TestPolyglotBookInterface(unittest.TestCase):
    """Bug #5 + #30."""

    def test_white_castle_mapped_to_king_destination(self):
        b = parse_fen("r1bqkb1r/1ppp1ppp/p1n2n2/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 5")
        m = _convert_polyglot_move_to_engine_move(_poly(4, 0, 7, 0), b)
        self.assertEqual(move_to_uci(m), "e1g1")
        self.assertEqual(get_special_move_flag(m), SPECIAL_MOVE_FLAG_CASTLING)

    def test_black_castles_both_wings(self):
        b = parse_fen("r3k2r/pppq1ppp/2n5/3pP3/8/8/PPP2PPP/R3K2R b KQkq - 0 1")
        ks = _convert_polyglot_move_to_engine_move(_poly(4, 7, 7, 7), b)
        qs = _convert_polyglot_move_to_engine_move(_poly(4, 7, 0, 7), b)
        self.assertEqual((move_to_uci(ks), move_to_uci(qs)), ("e8g8", "e8c8"))
        self.assertEqual(get_special_move_flag(ks), SPECIAL_MOVE_FLAG_CASTLING)
        self.assertEqual(get_special_move_flag(qs), SPECIAL_MOVE_FLAG_CASTLING)

    def test_en_passant_flag_inferred(self):
        b = parse_fen("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3")
        m = _convert_polyglot_move_to_engine_move(_poly(4, 4, 3, 5), b)
        self.assertEqual(move_to_uci(m), "e5d6")
        self.assertEqual(get_special_move_flag(m), SPECIAL_MOVE_FLAG_EN_PASSANT)

    def test_ordinary_capture_not_flagged_en_passant(self):
        b = parse_fen("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3")
        m = _convert_polyglot_move_to_engine_move(_poly(4, 3, 3, 4), b)
        self.assertEqual(get_special_move_flag(m), 0)

    def test_real_book_moves_are_legal(self):
        path = os.path.join(_ROOT, "chess_engine", "polyglot.bin")
        if not os.path.exists(path):
            self.skipTest("polyglot.bin not available")
        book = OpeningBook(path)
        fens = [
            "r1bqkb1r/1ppp1ppp/p1n2n2/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 5",
            "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        ]
        for fen in fens:
            b = parse_fen(fen)
            entries = book.lookup(b[2][4], b)
            self.assertTrue(entries, fen)
            legal = {int(x) for x in generate_legal_moves(b[0], b[1], b[2])}
            for move, _weight in entries:
                self.assertIn(int(move), legal, f"{move_to_uci(move)} illegal in {fen}")

    def test_no_move_uci_is_null_move(self):
        self.assertEqual(move_to_uci(np.uint16(0)), "0000")
        self.assertEqual(move_to_uci(0), "0000")
        self.assertEqual(move_to_uci(encode_move(12, 28, 0, 0)), "e2e4")


class TestFenSanitation(unittest.TestCase):
    """Bug #18."""

    def test_stale_castling_rights_removed(self):
        # White rook on a1 only: K/Q rights need king e1 + rook h1/a1; black has no king/rook home -> 0.
        b = parse_fen("8/8/8/8/8/8/8/R3K2k w KQkq - 0 1")
        self.assertEqual(int(b[2][1]), 2)  # only Q survives

    def test_negative_and_huge_halfmove(self):
        self.assertEqual(int(parse_fen("4k3/8/8/8/8/8/8/4K3 w - - -5 1")[2][3]), 0)
        self.assertEqual(int(parse_fen("4k3/8/8/8/8/8/8/4K3 w - - 999 1")[2][3]), 255)

    def test_phantom_en_passant_dropped(self):
        b = parse_fen("4k3/8/8/8/8/8/8/4K3 w - e3 0 1")
        self.assertEqual(int(b[2][2]), 64)
        # wrong rank for side to move
        b = parse_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e3 0 1")
        self.assertEqual(int(b[2][2]), 64)

    def test_valid_en_passant_kept(self):
        # Bug #13: capturable EP (adjacent black pawn on d4) keeps EP square,
        # while uncapturable double-push EP without adjacent enemy pawn drops to 64.
        b_cap = parse_fen("rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 2")
        self.assertEqual(int(b_cap[2][2]), 20)
        b_uncap = parse_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
        self.assertEqual(int(b_uncap[2][2]), 64)

    def test_startpos_unchanged(self):
        b = parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        self.assertEqual((int(b[2][1]), int(b[2][2]), int(b[2][3])), (15, 64, 0))


class TestKbnkMate(unittest.TestCase):
    """Bug #6."""

    def test_undefended_bishop_capture_is_not_mate(self):
        _, score = _special("k7/B1K5/1N6/8/8/8/8/8 b - - 0 1")
        self.assertLess(score, MATE_IN_MAX_PLY - 1)

    def test_strong_side_to_move_is_not_stalemate_draw(self):
        hit, score = _special("k7/2K5/2N5/8/8/8/8/6B1 w - - 0 1")
        self.assertTrue(hit)
        self.assertGreater(score, 1000)

    def test_weak_side_to_move_stalemate_is_draw(self):
        hit, score = _special("k7/2K5/2N5/8/8/8/8/6B1 b - - 0 1")
        self.assertTrue(hit)
        self.assertEqual(score, 0)

    def test_real_checkmate_still_recognised(self):
        # Black Ka8 mated by Bh1 (check), Kb6 (a7/b7) and Nd7 (b8).
        hit, score = _special("k7/3N4/1K6/8/8/8/8/7B b - - 0 1")
        self.assertTrue(hit)
        self.assertEqual(score, MATE_IN_MAX_PLY - 1)


class TestKxkStalemate(unittest.TestCase):
    """Bug #16."""

    def test_white_strong_stalemate_is_draw(self):
        hit, score = _special("k7/2Q5/1K6/8/8/8/8/8 b - - 0 1")
        self.assertTrue(hit)
        self.assertEqual(score, 0)

    def test_black_strong_stalemate_is_draw(self):
        hit, score = _special("8/8/8/8/8/1k6/2q5/K7 w - - 0 1")
        self.assertTrue(hit)
        self.assertEqual(score, 0)

    def test_strong_side_to_move_still_wins(self):
        hit, score = _special("k7/2Q5/1K6/8/8/8/8/8 w - - 0 1")
        self.assertTrue(hit)
        self.assertGreater(score, 1000)

    def test_weak_side_with_a_flight_square_still_wins(self):
        hit, score = _special("4k3/8/8/8/8/8/8/KQ6 b - - 0 1")
        self.assertTrue(hit)
        self.assertGreater(score, 1000)


class TestScaleFactorFixes(unittest.TestCase):
    """Bugs #14, #15, #25."""

    def test_kbpsk_b_file_requires_all_pawns_on_file(self):
        # Extra black e-pawn: SF11 does not apply the b-file fortress.
        self.assertNotEqual(_scale("1k6/1p6/1P6/4p3/8/3B4/8/4K3 w - - 0 1"), SCALE_FACTOR_DRAW)

    def test_kbpsk_b_file_fortress_kept_when_all_pawns_on_b(self):
        self.assertEqual(_scale("1k6/1p6/1P6/8/8/3B4/8/4K3 w - - 0 1"), SCALE_FACTOR_DRAW)

    def test_krpkn_not_zeroed(self):
        self.assertNotEqual(_scale("8/4k3/4P3/3nK3/8/8/8/4R3 w - - 0 1"), SCALE_FACTOR_DRAW)
        self.assertNotEqual(_scale("8/4k3/R3n3/4P3/4K3/8/8/8 w - - 0 1"), SCALE_FACTOR_DRAW)

    def test_krppkrp_second_rank_reachable(self):
        self.assertEqual(_scale("r4k2/5p2/8/8/8/8/4PP2/R3K3 w - - 0 1"), 9)
        self.assertEqual(_scale("r4k2/5p2/8/8/8/4PP2/8/R3K3 w - - 0 1"), 10)


class TestSearchContextSafety(unittest.TestCase):
    """Bugs #17 + #29 + #4 (source contract)."""

    def test_eval_weights_default_is_full_table(self):
        ctx = SearchContext(*_ctx_args())
        self.assertGreaterEqual(len(ctx.eval_weights), 128)
        self.assertTrue(np.any(ctx.eval_weights != 0))

    def test_eval_weights_fallback_is_not_silent(self):
        src = inspect.getsource(engine_types.SearchContext) if inspect.isfunction(engine_types.SearchContext) \
            else inspect.getsource(engine_types)
        self.assertNotIn("eval_weights = np.zeros(64", src)

    def test_reset_clears_per_game_state(self):
        ctx = SearchContext(*_ctx_args())
        ctx.game_history[:5] = 7
        ctx.game_history_count = np.int32(5)
        ctx.tt_generation = np.uint8(9)
        ctx.nmp_min_ply = np.int32(11)
        ctx.diag_stats[:3] = 4
        reset_search_context(ctx)
        self.assertEqual(int(ctx.game_history.sum()), 0)
        self.assertEqual(int(ctx.game_history_count), 0)
        self.assertEqual(int(ctx.tt_generation), 0)
        self.assertEqual(int(ctx.nmp_min_ply), 0)
        self.assertEqual(int(ctx.diag_stats.sum()), 0)

    def test_search_guards_max_ply_before_stack_access(self):
        with open(os.path.join(_ROOT, "chess_engine", "classical", "search.py"), encoding="utf-8") as fh:
            src = fh.read()
        start = src.index("def _search(")
        body = src[start:start + 6000]
        guard = body.index("if ply >= MAX_PLY - 1:")
        self.assertLess(guard, body.index("search_context.cutoff_cnt[0] = 0"))
        self.assertLess(guard, body.index("search_context.follow_pv_stack[ply] = follow_pv"))


def _ctx_args():
    from chess_engine.classical.constants import CORRECTION_HISTORY_SIZE, PAWN_HISTORY_SIZE
    tt = create_transposition_table(1)
    return (
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


if __name__ == "__main__":
    unittest.main()
