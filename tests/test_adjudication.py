import unittest
import chess

from tools.match_core import SearchLimit, extract_score_from_info, play_game
from tools.tournament import build_parser

DEFAULT_MOVES = [
    'e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1c4', 'f8c5', 'c2c3', 'g8f6',
    'd2d3', 'd7d6', 'e1g1', 'e8g8', 'h2h3', 'a7a6', 'f1e1', 'c5a7',
    'b1d2', 'c6e7', 'd2f1', 'e7g6', 'f1g3', 'c7c6', 'c4b3', 'd6d5',
    'e4d5', 'f6d5', 'd3d4', 'e5d4', 'f3d4', 'f8e8', 'e1e8', 'd8e8',
    'd4f3', 'g6e5', 'f3e5', 'e8e5', 'd1f3', 'c8e6', 'c1d2', 'a8e8'
]


class _MockEngine:
    """Mock engine that plays legal moves and reports fixed evaluation."""

    def __init__(self, name: str, scores: list[int], moves: list[str] = None) -> None:
        self.name = name
        self.scores = list(scores)
        self.score_idx = 0
        self.last_score_cp = None
        self.moves = list(moves) if moves is not None else list(DEFAULT_MOVES)
        self.move_idx = 0
        self.board = chess.Board()

    def new_game(self, start_fen: str = None) -> None:
        self.board = chess.Board(start_fen) if start_fen else chess.Board()
        self.last_score_cp = None
        self.score_idx = 0
        self.move_idx = 0

    def push(self, move_uci: str) -> None:
        self.board.push_uci(move_uci)

    def get_move(self, _limit: SearchLimit) -> tuple[chess.Move, str]:
        if self.scores:
            score = self.scores[min(self.score_idx, len(self.scores) - 1)]
            self.score_idx += 1
        else:
            score = 0
        self.last_score_cp = score

        # Pick move corresponding to current ply
        ply = len(self.board.move_stack)
        if ply < len(self.moves):
            move = chess.Move.from_uci(self.moves[ply])
        else:
            move = next(iter(self.board.legal_moves))

        info = f"d=10, eval={score / 100:+.2f}, n=50000"
        return move, info


class TestEarlyAdjudication(unittest.TestCase):
    def test_extract_score_from_info(self) -> None:
        self.assertEqual(extract_score_from_info("d=10, eval=+5.23, n=50000"), 523)
        self.assertEqual(extract_score_from_info("d=10, eval=-0.08, n=50000"), -8)
        self.assertEqual(extract_score_from_info("d=10, eval=#3, n=50000"), 30000)
        self.assertEqual(extract_score_from_info("d=10, eval=#-2, n=50000"), -30000)
        self.assertEqual(extract_score_from_info("d=10, eval=cp 150, n=50000"), 150)
        self.assertIsNone(extract_score_from_info(""))
        self.assertIsNone(extract_score_from_info("d=10, n=50000"))

    def test_white_win_adjudication(self) -> None:
        # White reports +550 (White winning), Black reports -520 (Black losing, so White winning)
        white_engine = _MockEngine("White", [550])
        black_engine = _MockEngine("Black", [-520])

        result = play_game(
            white_engine,
            black_engine,
            SearchLimit(depth=1),
            SearchLimit(depth=1),
            game_number=1,
            max_plies=100,
            adjudicate=True,
            adjudicate_win_score=500,
            adjudicate_win_plies=10,
        )

        self.assertEqual(result.result, "1-0")
        self.assertIn("adjudicated: White decisive advantage", result.pgn.headers["Termination"])
        self.assertEqual(len(list(result.pgn.mainline_moves())), 10)

    def test_black_win_adjudication(self) -> None:
        # White reports -600 (White losing), Black reports +650 (Black winning)
        white_engine = _MockEngine("White", [-600])
        black_engine = _MockEngine("Black", [650])

        result = play_game(
            white_engine,
            black_engine,
            SearchLimit(depth=1),
            SearchLimit(depth=1),
            game_number=2,
            max_plies=100,
            adjudicate=True,
            adjudicate_win_score=500,
            adjudicate_win_plies=10,
        )

        self.assertEqual(result.result, "0-1")
        self.assertIn("adjudicated: Black decisive advantage", result.pgn.headers["Termination"])
        self.assertEqual(len(list(result.pgn.mainline_moves())), 10)

    def test_draw_adjudication_with_min_move(self) -> None:
        # Both report score = 0 cp
        white_engine = _MockEngine("White", [0])
        black_engine = _MockEngine("Black", [0])

        # Test A: min_move = 1, should adjudicate after 12 plies
        result = play_game(
            white_engine,
            black_engine,
            SearchLimit(depth=1),
            SearchLimit(depth=1),
            game_number=3,
            max_plies=100,
            adjudicate=True,
            adjudicate_draw_score=8,
            adjudicate_draw_min_move=1,
            adjudicate_draw_plies=12,
        )
        self.assertEqual(result.result, "1/2-1/2")
        self.assertIn("adjudicated draw", result.pgn.headers["Termination"])
        self.assertEqual(len(list(result.pgn.mainline_moves())), 12)

        # Test B: min_move = 40 (ply 80), should NOT adjudicate before ply 80
        white_engine.new_game()
        black_engine.new_game()
        result_b = play_game(
            white_engine,
            black_engine,
            SearchLimit(depth=1),
            SearchLimit(depth=1),
            game_number=4,
            max_plies=20,
            adjudicate=True,
            adjudicate_draw_score=8,
            adjudicate_draw_min_move=40,
            adjudicate_draw_plies=12,
        )
        self.assertEqual(result_b.result, "1/2-1/2")
        # Ended because max_plies reached, not early draw adjudication
        self.assertIn("adjudicated draw at 20 plies", result_b.pgn.headers["Termination"])
        self.assertEqual(len(list(result_b.pgn.mainline_moves())), 20)

    def test_adjudicate_toggle_disabled(self) -> None:
        # With adjudicate=False, games do NOT end at 10 plies even with +600 cp
        white_engine = _MockEngine("White", [600])
        black_engine = _MockEngine("Black", [-600])

        result = play_game(
            white_engine,
            black_engine,
            SearchLimit(depth=1),
            SearchLimit(depth=1),
            game_number=5,
            max_plies=20,
            adjudicate=False,
            adjudicate_win_score=500,
            adjudicate_win_plies=10,
        )
        # Should play all 20 plies
        self.assertEqual(len(list(result.pgn.mainline_moves())), 20)
        self.assertIn("adjudicated draw at 20 plies", result.pgn.headers["Termination"])

    def test_streak_reset_on_evaluation_drop(self) -> None:
        # White starts with +600 for 4 turns, but turn 5 drops to +300
        # Ply 1 (W: +600), Ply 2 (B: -600), Ply 3 (W: +600), Ply 4 (B: -600)
        # Ply 5 (W: +600), Ply 6 (B: -600), Ply 7 (W: +600), Ply 8 (B: -600)
        # Ply 9 (W: +300) -> resets streak!
        white_scores = [600, 600, 600, 600, 300, 600, 600]
        black_scores = [-600]

        white_engine = _MockEngine("White", white_scores)
        black_engine = _MockEngine("Black", black_scores)

        result = play_game(
            white_engine,
            black_engine,
            SearchLimit(depth=1),
            SearchLimit(depth=1),
            game_number=6,
            max_plies=12,
            adjudicate=True,
            adjudicate_win_score=500,
            adjudicate_win_plies=10,
        )
        # Game should NOT have adjudicated at ply 10 because streak was reset at ply 9
        self.assertEqual(len(list(result.pgn.mainline_moves())), 12)

    def test_cli_parser_defaults_and_toggles(self) -> None:
        parser = build_parser()

        # Default: adjudicate is True
        args_default = parser.parse_args([])
        self.assertTrue(args_default.adjudicate)
        self.assertEqual(args_default.adjudicate_win_score, 500)
        self.assertEqual(args_default.adjudicate_win_moves, 5)
        self.assertEqual(args_default.adjudicate_draw_score, 8)
        self.assertEqual(args_default.adjudicate_draw_min_move, 40)
        self.assertEqual(args_default.adjudicate_draw_moves, 6)

        # One-click disable
        args_disabled = parser.parse_args(["--no-adjudicate"])
        self.assertFalse(args_disabled.adjudicate)

        # Explicit enable
        args_enabled = parser.parse_args(["--adjudicate"])
        self.assertTrue(args_enabled.adjudicate)

        # Custom thresholds
        args_custom = parser.parse_args([
            "--adjudicate-win-score", "600",
            "--adjudicate-win-moves", "4",
            "--adjudicate-draw-score", "10",
            "--adjudicate-draw-min-move", "35",
            "--adjudicate-draw-moves", "8",
        ])
        self.assertEqual(args_custom.adjudicate_win_score, 600)
        self.assertEqual(args_custom.adjudicate_win_moves, 4)
        self.assertEqual(args_custom.adjudicate_draw_score, 10)
        self.assertEqual(args_custom.adjudicate_draw_min_move, 35)
        self.assertEqual(args_custom.adjudicate_draw_moves, 8)


if __name__ == "__main__":
    unittest.main()
