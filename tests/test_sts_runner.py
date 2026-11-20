import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import chess

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.match_core import SearchLimit
from tools.run_sts import (
    STS_THEMES,
    SUITE_FILES,
    _parse_theme_filter,
    build_arg_parser,
    estimate_ccrl_elo,
    load_sts_positions,
    parse_info_metrics,
    run_sts_benchmark,
    summarize_results,
)


class _DummyEngine:
    def __init__(self, move_selector):
        self.move_selector = move_selector
        self.fen = None
        self.terminated = False

    def start(self):
        return

    def new_game(self, fen):
        self.fen = fen

    def get_move(self, limit):
        uci = self.move_selector(self.fen)
        return chess.Move.from_uci(uci), "d=8, eval=+0.25, n=12000, t=20ms, nps=600000"

    def terminate(self):
        self.terminated = True


class _DummyFactory:
    def __init__(self, move_selector):
        self.move_selector = move_selector
        self.engines = []

    def create(self, worker_id):
        eng = _DummyEngine(self.move_selector)
        self.engines.append(eng)
        return eng

    def prepare(self, engine, worker_id):
        return


class TestStsRunner(unittest.TestCase):
    def test_default_suite_is_v6(self):
        args = build_arg_parser().parse_args([])
        self.assertEqual(args.suite, "v6")

    def test_v3_and_v6_datasets_present_and_valid(self):
        self.assertTrue(SUITE_FILES["v3"].is_file())
        self.assertTrue(SUITE_FILES["v6"].is_file())

        v3_pos = load_sts_positions(SUITE_FILES["v3"])
        self.assertEqual(len(v3_pos), 1500)
        self.assertEqual({p.theme_id for p in v3_pos}, set(STS_THEMES.keys()))
        self.assertTrue(all(p.max_points == 10 for p in v3_pos))

        v6_pos = load_sts_positions(SUITE_FILES["v6"])
        self.assertEqual(len(v6_pos), 1188)
        self.assertEqual({p.theme_id for p in v6_pos}, set(STS_THEMES.keys()))
        self.assertTrue(all(p.max_points == 100 for p in v6_pos))

    def test_theme_and_count_filters(self):
        themes = _parse_theme_filter("1,3-4")
        self.assertEqual(themes, {1, 3, 4})
        subset = load_sts_positions(SUITE_FILES["v3"], themes=themes, per_theme=2)
        self.assertEqual(len(subset), 6)
        self.assertEqual([p.theme_id for p in subset], [1, 1, 3, 3, 4, 4])

        capped = load_sts_positions(SUITE_FILES["v3"], max_positions=5)
        self.assertEqual(len(capped), 5)

    def test_parse_info_metrics_and_elo(self):
        d, n, t = parse_info_metrics("d=11, eval=+0.42, n=50000, t=85ms, nps=588235")
        self.assertEqual((d, n, t), (11, 50000, 85))
        self.assertEqual(parse_info_metrics(""), (None, None, None))
        self.assertAlmostEqual(estimate_ccrl_elo(50.0), 44.523 * 50.0 - 242.85, places=3)

    def test_run_sts_benchmark_and_summarize(self):
        positions = load_sts_positions(SUITE_FILES["v3"], per_theme=1)
        self.assertEqual(len(positions), 15)

        fen_to_best = {p.fen: p.best_move for p in positions}
        # Answer best move on even indices, first legal (likely 0-pt or partial) on odd
        def selector(fen):
            return fen_to_best[fen]

        factory = _DummyFactory(selector)
        results, duration_s = run_sts_benchmark(
            positions,
            factory,
            SearchLimit(nodes=1000),
            concurrency=3,
            progress_every=0,
        )
        self.assertEqual(len(results), 15)
        self.assertGreaterEqual(duration_s, 0.0)
        self.assertTrue(all(eng.terminated for eng in factory.engines))

        summary = summarize_results(
            results,
            engine_label="dummy",
            suite_label="v3",
            limit=SearchLimit(nodes=1000),
            duration_s=duration_s,
            include_elo=True,
        )
        self.assertEqual(summary["overall"]["score"], 150)
        self.assertEqual(summary["overall"]["max_score"], 150)
        self.assertAlmostEqual(summary["overall"]["score_pct"], 100.0)
        self.assertEqual(summary["overall"]["best1"], 15)
        self.assertEqual(len(summary["themes"]), 15)


if __name__ == "__main__":
    unittest.main()
