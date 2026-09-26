from __future__ import annotations

import importlib.util
import io
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tournament_analysis" / "統計數據.py"
SPEC = importlib.util.spec_from_file_location("tournament_game_stats", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
stats_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stats_module)


class TournamentGameStatsTests(unittest.TestCase):
    def test_main_reconfigures_cp950_like_text_stream(self) -> None:
        class ReconfigurableBuffer(io.StringIO):
            def __init__(self) -> None:
                super().__init__()
                self.config = None

            def reconfigure(self, **kwargs) -> None:
                self.config = kwargs

        stream = ReconfigurableBuffer()
        original_stdout = stats_module.sys.stdout
        original_stderr = stats_module.sys.stderr
        try:
            stats_module.sys.stdout = stream
            stats_module.sys.stderr = stream
            result = stats_module.main(["--pgn", "missing-test-file.pgn"])
        finally:
            stats_module.sys.stdout = original_stdout
            stats_module.sys.stderr = original_stderr

        self.assertEqual(result, 1)
        self.assertEqual(stream.config, {"encoding": "utf-8", "errors": "replace"})

    def test_metadata_ignores_eval_decimals_comments_and_variations(self) -> None:
        movetext = (
            "10. e4 { d=12, eval=+299.93, fake=Q } 10... e5 "
            "(10... c5 { eval=-123.45 }) 11. e8=Q+ { eval=#3 } 1-0"
        )
        fen = "8/8/8/8/8/8/8/8 w - - 0 10"

        moves, first_move, promotions, mate_flag = stats_module.extract_movetext_metadata(
            movetext, fen
        )

        self.assertEqual(moves, 2)
        self.assertEqual(first_move, "e4")
        self.assertEqual(promotions, 1)
        self.assertTrue(mate_flag)

    def test_games_are_round_sorted_and_paired_by_color(self) -> None:
        pgn = """[Event "Test"]
[Date "2026.07.30"]
[Round "2"]
[White "Old"]
[Black "New"]
[Result "1-0"]
[FEN "8/8/8/8/8/8/8/8 w - - 0 1"]

1. e4 { eval=+299.93 } 1... e5 2. Nf3 1-0

[Event "Test"]
[Date "2026.07.30"]
[Round "1"]
[White "New"]
[Black "Old"]
[Result "1-0"]
[FEN "8/8/8/8/8/8/8/8 w - - 0 1"]

1. d4 { eval=-299.12 } 1... d5 2. c4 1-0
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "match.pgn"
            path.write_text(pgn, encoding="utf-8")
            analyzer = stats_module.ChessEngineAnalyzer("New")
            analyzer.parse_pgn(path)

        self.assertEqual([game["round"] for game in analyzer.games], [1, 2])
        self.assertEqual(analyzer.score_sequence, [1.0, 0.0])
        self.assertEqual(analyzer.lengths, [2, 2])
        self.assertEqual(analyzer.total_pairs, 1)
        self.assertEqual(analyzer.consecutive_pair_counts["WL"], 1)
        self.assertEqual(analyzer.fen_pair_counts["WL"], 1)

    def test_elo_ci_uses_complete_pairs_as_independent_units(self) -> None:
        analyzer = stats_module.ChessEngineAnalyzer("New")
        analyzer.games = [
            {"score": 1.0, "rt": "W", "is_target_white": True},
            {"score": 0.5, "rt": "D", "is_target_white": False},
            {"score": 0.0, "rt": "L", "is_target_white": True},
            {"score": 0.5, "rt": "D", "is_target_white": False},
        ]
        analyzer.score_sequence = [1.0, 0.5, 0.0, 0.5]
        analyzer.match_pairs = [
            (1, analyzer.games[0], analyzer.games[1]),
            (2, analyzer.games[2], analyzer.games[3]),
        ]
        analyzer.w_wins = 1
        analyzer.w_losses = 1
        analyzer.b_draws = 2

        result = analyzer.calculate_elo_and_ci()

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["CI_Method"], "color-swapped pair clustered")
        self.assertEqual(result["Independent_Units"], 2)
        self.assertEqual(result["Expected_Score"], 0.5)


if __name__ == "__main__":
    unittest.main()
