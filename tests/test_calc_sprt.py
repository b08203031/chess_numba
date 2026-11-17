from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

from tools.match_core import SPRTTest
from tune_search.stats import sprt_llr as tune_sprt_llr

ROOT = Path(__file__).resolve().parents[1]
CALC_PATH = ROOT / "tournament_analysis" / "calc_sprt.py"
SPEC = importlib.util.spec_from_file_location("calc_sprt", CALC_PATH)
assert SPEC is not None and SPEC.loader is not None
calc_sprt = importlib.util.module_from_spec(SPEC)
sys.modules["calc_sprt"] = calc_sprt
SPEC.loader.exec_module(calc_sprt)


class CalcSprtTests(unittest.TestCase):
    def test_pentanomial_normal_matches_match_core_and_tune_stats(self) -> None:
        pair_scores = [2.0, 1.5, 1.0, 1.0, 0.5, 1.5, 2.0, 1.0, 0.0, 1.5]
        sprt = SPRTTest(elo0=0.0, elo1=10.0, alpha=0.05, beta=0.05)
        for ps in pair_scores:
            sprt.add_pair_result(ps)
        _, core_llr = sprt.check_status()

        penta = [0, 0, 0, 0, 0]
        for ps in pair_scores:
            penta[calc_sprt.PAIR_SCORE_TO_BIN[ps]] += 1

        n, mean, var = calc_sprt.pentanomial_moments(penta)
        calc_llr = calc_sprt.normal_sprt_llr(n, mean, var, 0.0, 10.0)
        tune_llr = tune_sprt_llr(penta, 0.0, 10.0)

        self.assertAlmostEqual(calc_llr, core_llr, places=10)
        self.assertAlmostEqual(calc_llr, tune_llr, places=10)

    def test_exact_binomial_llr_matches_closed_form(self) -> None:
        # Pure decisive games (D=0): constrained MLE must equal exact binomial log-likelihood ratio.
        wins, draws, losses = 60, 0, 40
        elo0, elo1 = 0.0, 10.0
        mu0 = calc_sprt.elo_to_score(elo0)
        mu1 = calc_sprt.elo_to_score(elo1)
        expected = (
            wins * math.log(mu1 / mu0)
            + losses * math.log((1.0 - mu1) / (1.0 - mu0))
        )
        got = calc_sprt.multinomial_exact_llr((losses, draws, wins), calc_sprt.TRINOMIAL_SCORES, elo0, elo1)
        self.assertIsNotNone(got)
        assert got is not None
        self.assertAlmostEqual(got, expected, places=10)

    def test_exact_mle_returns_none_outside_support(self) -> None:
        # Only draws observed -> support is {0.5}, so mu1 > 0.5 has no MLE on support.
        got = calc_sprt.multinomial_exact_llr((0, 0, 20, 0, 0), calc_sprt.PENTA_SCORES_NORM, 0.0, 5.0)
        self.assertIsNone(got)

    def test_nelo_matches_logistic_at_zero_draws_independent_pairs(self) -> None:
        # With D=0 and independent games: p(W)=0.55, p(L)=0.45 -> pair bins [0.45^2, 0, 2*0.45*0.55, 0, 0.55^2].
        # Var(pair_norm) = 0.25 * 0.55 * 0.45 / 0.5? Actually Var(mean of 2 Bernoullis) = p(1-p)/2.
        # At p -> 0.5, sigma_eff = sqrt(2 * 0.25/2) = 0.5, so nElo slope == Logistic Elo slope.
        penta = [25, 0, 50, 0, 25]  # symmetric, zero Elo
        n, mean, var = calc_sprt.pentanomial_moments(penta)
        nelo, err = calc_sprt.estimate_nelo_ci(n, mean, var)
        self.assertAlmostEqual(nelo, 0.0, places=9)
        self.assertGreater(err, 0.0)

    def test_pgn_parsing_sorts_rounds_and_excludes_failure_pairs(self) -> None:
        pgn = """[Event "T"]
[Round "2"]
[White "Old"]
[Black "New"]
[Result "1-0"]
[FEN "8/8/8/8/8/8/8/8 w - - 0 1"]
[Failure "Old: timeout"]

1. e4 1-0

[Event "T"]
[Round "1"]
[White "New"]
[Black "Old"]
[Result "1-0"]
[FEN "8/8/8/8/8/8/8/8 w - - 0 1"]

1. d4 1-0

[Event "T"]
[Round "4"]
[White "Old"]
[Black "New"]
[Result "0-1"]
[FEN "8/8/8/8/8/8/8/8 w - - 0 1"]

1. c4 0-1

[Event "T"]
[Round "3"]
[White "New"]
[Black "Old"]
[Result "1/2-1/2"]
[FEN "8/8/8/8/8/8/8/8 w - - 0 1"]

1. Nf3 1/2-1/2
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.pgn"
            path.write_text(pgn, encoding="utf-8")

            excluded = calc_sprt.parse_pgn_matches(path, target_engine="New", include_failures=False)
            self.assertEqual(excluded.excluded_failure_pairs, 1)
            self.assertEqual(excluded.excluded_failure_games, 1)
            self.assertEqual(excluded.pair_scores, [1.5])
            self.assertEqual(excluded.wins, 1)
            self.assertEqual(excluded.draws, 1)
            self.assertEqual(excluded.losses, 0)

            included = calc_sprt.parse_pgn_matches(path, target_engine="New", include_failures=True)
            self.assertEqual(included.excluded_failure_pairs, 0)
            self.assertEqual(included.pair_scores, [1.0, 1.5])

    def test_cli_wdl_and_json_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_json = Path(tmp) / "sprt.json"
            rc = calc_sprt.main([
                "--wdl", "320", "1200", "280",
                "--bounds", "0:10,0:5,-3:1",
                "--json", str(out_json),
            ])
            self.assertEqual(rc, 0)
            data = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(data["games"], 1800)
            self.assertEqual(data["pairs"], 0)
            self.assertEqual(len(data["sprt_rows"]), 3)


if __name__ == "__main__":
    unittest.main()
