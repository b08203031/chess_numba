import unittest

from tune_search.search_param_registry import (
    CATEGORICAL_PARAMS,
    get_spec,
    profile_names,
)
from tune_search.stats import (
    drift_z,
    elo_estimate,
    elo_to_score,
    merge_penta,
    penta_moments,
    score_to_elo,
    sprt_bounds,
    sprt_decision,
    sprt_llr,
    _Adjudicator,
)


class TestTuneStats(unittest.TestCase):
    def test_elo_score_round_trip(self):
        for elo in (-200.0, -3.0, 0.0, 5.0, 150.0):
            self.assertAlmostEqual(score_to_elo(elo_to_score(elo)), elo, places=6)

    def test_symmetric_penta_is_zero_elo(self):
        est = elo_estimate([5, 10, 30, 10, 5])
        self.assertAlmostEqual(est["elo"], 0.0, places=9)
        self.assertEqual(est["games"], 120)
        self.assertGreater(est["elo_err"], 0.0)

    def test_more_pairs_shrink_error(self):
        small = elo_estimate([5, 10, 30, 10, 5])["elo_err"]
        large = elo_estimate([50, 100, 300, 100, 50])["elo_err"]
        self.assertLess(large, small)

    def test_penta_moments_mean(self):
        n, mean, var = penta_moments([0, 0, 0, 0, 10])
        self.assertEqual(n, 10)
        self.assertAlmostEqual(mean, 1.0)
        self.assertAlmostEqual(var, 0.0)

    def test_llr_sign_follows_evidence(self):
        good = [10, 20, 60, 70, 40]   # clearly > 0 Elo
        bad = [40, 70, 60, 20, 10]
        self.assertGreater(sprt_llr(good, 0.0, 5.0), 0.0)
        self.assertLess(sprt_llr(bad, 0.0, 5.0), 0.0)
        self.assertEqual(sprt_llr([0, 0, 0, 0, 0], 0.0, 5.0), 0.0)

    def test_decision_uses_bounds(self):
        lower, upper = sprt_bounds(0.05, 0.05)
        self.assertLess(lower, 0.0)
        self.assertGreater(upper, 0.0)
        self.assertEqual(sprt_decision(upper + 0.1), "accept_h1")
        self.assertEqual(sprt_decision(lower - 0.1), "accept_h0")
        self.assertEqual(sprt_decision(0.0), "continue")

    def test_merge_penta(self):
        self.assertEqual(merge_penta([[1, 0, 2, 0, 3], [0, 1, 0, 1, 0]]), [1, 1, 2, 1, 3])

    def test_drift_z_detects_consistent_direction(self):
        self.assertAlmostEqual(drift_z([1.0] * 16), 4.0)
        self.assertAlmostEqual(drift_z([1.0, -1.0] * 8), 0.0)
        self.assertEqual(drift_z([]), 0.0)


class TestRuntimeContinuousProfile(unittest.TestCase):
    def test_excludes_only_categorical_axes(self):
        runtime = set(profile_names("runtime"))
        cont = set(profile_names("runtime_continuous"))
        self.assertEqual(runtime - cont, set(CATEGORICAL_PARAMS))
        self.assertTrue(all(get_spec(n).is_runtime for n in cont))


class TestAdjudicator(unittest.TestCase):
    def test_decisive_needs_consecutive_same_sign(self):
        adj = _Adjudicator({"win_cp": 800, "win_plies": 3})
        self.assertIsNone(adj.update(0, 900))
        self.assertIsNone(adj.update(1, 950))
        self.assertIsNone(adj.update(2, 100))      # streak broken
        self.assertIsNone(adj.update(3, 900))
        self.assertIsNone(adj.update(4, -900))     # sign flip resets
        self.assertIsNone(adj.update(5, -900))
        self.assertEqual(adj.update(6, -900), 0.0)  # black wins

    def test_white_win(self):
        adj = _Adjudicator({"win_plies": 2})
        self.assertIsNone(adj.update(0, 1200))
        self.assertEqual(adj.update(1, 1500), 1.0)

    def test_draw_requires_min_ply_and_streak(self):
        adj = _Adjudicator({"draw_min_ply": 10, "draw_plies": 3, "draw_cp": 10})
        for ply in range(0, 9):
            self.assertIsNone(adj.update(ply, 0))  # before min ply
        self.assertIsNone(adj.update(10, 5))
        self.assertIsNone(adj.update(11, -5))
        self.assertEqual(adj.update(12, 0), 0.5)

    def test_draw_streak_breaks_on_large_score(self):
        adj = _Adjudicator({"draw_min_ply": 0, "draw_plies": 3, "draw_cp": 10})
        self.assertIsNone(adj.update(0, 0))
        self.assertIsNone(adj.update(1, 50))
        self.assertIsNone(adj.update(2, 0))
        self.assertIsNone(adj.update(3, 0))
        self.assertEqual(adj.update(4, 0), 0.5)


if __name__ == "__main__":
    unittest.main()

