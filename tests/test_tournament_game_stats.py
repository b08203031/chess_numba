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

SEARCH_MODULE_PATH = ROOT / "tournament_analysis" / "parse_search_stats.py"
SEARCH_SPEC = importlib.util.spec_from_file_location("parse_search_stats", SEARCH_MODULE_PATH)
assert SEARCH_SPEC is not None and SEARCH_SPEC.loader is not None
search_stats_module = importlib.util.module_from_spec(SEARCH_SPEC)
SEARCH_SPEC.loader.exec_module(search_stats_module)


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

    def test_multi_pgn_merge_with_round_offset_and_duplicate_fen_detection(self) -> None:
        fen_a = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
        fen_b = "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 4"

        pgn1 = f"""[Event "Batch 1"]
[Round "1"]
[White "New"]
[Black "Old"]
[Result "1-0"]
[FEN "{fen_a}"]

1. d4 {{ d=10, eval=+0.30, n=1000, t=100ms, nps=10000 }} 1... d5 {{ d=10, eval=-0.30, n=1000, t=100ms, nps=10000 }} 1-0

[Event "Batch 1"]
[Round "2"]
[White "Old"]
[Black "New"]
[Result "0-1"]
[FEN "{fen_a}"]

1. e4 {{ d=10, eval=+0.20, n=1000, t=100ms, nps=10000 }} 1... e5 {{ d=10, eval=-0.20, n=1000, t=100ms, nps=10000 }} 0-1
"""

        # Batch 2 with disjoint FEN
        pgn2_disjoint = f"""[Event "Batch 2"]
[Round "1"]
[White "New"]
[Black "Old"]
[Result "1/2-1/2"]
[FEN "{fen_b}"]

1. c4 {{ d=10, eval=+0.10, n=1000, t=100ms, nps=10000 }} 1... c5 {{ d=10, eval=-0.10, n=1000, t=100ms, nps=10000 }} 1/2-1/2

[Event "Batch 2"]
[Round "2"]
[White "Old"]
[Black "New"]
[Result "1/2-1/2"]
[FEN "{fen_b}"]

1. Nf3 {{ d=10, eval=+0.05, n=1000, t=100ms, nps=10000 }} 1... Nf6 {{ d=10, eval=-0.05, n=1000, t=100ms, nps=10000 }} 1/2-1/2
"""

        # Batch 3 with duplicate FEN (re-using fen_a)
        pgn3_duplicate = f"""[Event "Batch 3"]
[Round "1"]
[White "New"]
[Black "Old"]
[Result "1-0"]
[FEN "{fen_a}"]

1. d4 {{ d=10, eval=+0.30, n=1000, t=100ms, nps=10000 }} 1... d5 {{ d=10, eval=-0.30, n=1000, t=100ms, nps=10000 }} 1-0

[Event "Batch 3"]
[Round "2"]
[White "Old"]
[Black "New"]
[Result "0-1"]
[FEN "{fen_a}"]

1. e4 {{ d=10, eval=+0.20, n=1000, t=100ms, nps=10000 }} 1... e5 {{ d=10, eval=-0.20, n=1000, t=100ms, nps=10000 }} 0-1
"""

        with tempfile.TemporaryDirectory() as temp_dir:
            f1 = Path(temp_dir) / "run1.pgn"
            f2 = Path(temp_dir) / "run2.pgn"
            f3 = Path(temp_dir) / "run3.pgn"
            f1.write_text(pgn1, encoding="utf-8")
            f2.write_text(pgn2_disjoint, encoding="utf-8")
            f3.write_text(pgn3_duplicate, encoding="utf-8")

            # 1. Test merging disjoint PGN files
            analyzer_disjoint = stats_module.ChessEngineAnalyzer("New")
            analyzer_disjoint.parse_pgn([f1, f2])

            self.assertEqual(len(analyzer_disjoint.games), 4)
            # Check rounds were offset (f1 had rounds 1,2; f2 had rounds 1,2 -> offset to 3,4)
            self.assertEqual([g["round"] for g in analyzer_disjoint.games], [1, 2, 3, 4])
            self.assertEqual(analyzer_disjoint.total_pairs, 2)
            fa_disjoint = analyzer_disjoint.fen_analysis
            self.assertEqual(fa_disjoint["unique_fens"], 2)
            self.assertEqual(len(fa_disjoint["cross_file_duplicates"]), 0)
            self.assertEqual(len(fa_disjoint["excess_fens"]), 0)

            # 2. Test merging overlapping PGN files (run1 + run3 both use fen_a)
            analyzer_dup = stats_module.ChessEngineAnalyzer("New")
            analyzer_dup.parse_pgn([f1, f3])
            self.assertEqual(len(analyzer_dup.games), 4)
            fa_dup = analyzer_dup.fen_analysis
            self.assertEqual(fa_dup["unique_fens"], 1)
            self.assertEqual(len(fa_dup["cross_file_duplicates"]), 1)
            self.assertEqual(len(fa_dup["excess_fens"]), 1)
            self.assertEqual(fa_dup["excess_fens"][0]["count"], 4)

            # 3. Test generate_report creates text with duplicate warnings
            report_text = analyzer_dup.generate_report(Path(temp_dir), write_file=False)
            self.assertIn("跨檔案重複盤面", report_text)
            self.assertIn("超額重複開局", report_text)

    def test_parse_search_stats_multi_pgn_and_fen_check(self) -> None:
        fen_a = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
        fen_b = "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 4"

        pgn1 = f"""[Event "Batch 1"]
[White "New"]
[Black "Old"]
[Result "1-0"]
[FEN "{fen_a}"]

1. e4 {{ d=12, eval=+0.50, n=2000, t=10ms, nps=200000 }} 1... e5 {{ d=11, eval=-0.40, n=1800, t=10ms, nps=180000 }} 1-0
"""
        pgn2 = f"""[Event "Batch 2"]
[White "Old"]
[Black "New"]
[Result "0-1"]
[FEN "{fen_a}"]

1. d4 {{ d=13, eval=+0.30, n=3000, t=15ms, nps=200000 }} 1... d5 {{ d=14, eval=-0.30, n=3500, t=15ms, nps=233333 }} 0-1
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            f1 = Path(temp_dir) / "part1.pgn"
            f2 = Path(temp_dir) / "part2.pgn"
            f1.write_text(pgn1, encoding="utf-8")
            f2.write_text(pgn2, encoding="utf-8")

            stats = search_stats_module.parse_pgn([f1, f2], "New", "Old")
            self.assertIsNotNone(stats)
            assert stats is not None
            new_stats, old_stats, by_color, new_mate, old_mate, meta = stats
            self.assertEqual(meta["games_parsed"], 2)
            self.assertEqual(len(new_stats["depth"]), 2)
            self.assertEqual(len(old_stats["depth"]), 2)

            fen_check = meta["fen_check"]
            self.assertEqual(fen_check["unique_fens"], 1)
            # fen_a appears in part1.pgn and part2.pgn
            self.assertEqual(len(fen_check["cross_file_duplicates"]), 1)
            self.assertEqual(fen_check["cross_file_duplicates"][0]["total_count"], 2)


if __name__ == "__main__":
    unittest.main()
