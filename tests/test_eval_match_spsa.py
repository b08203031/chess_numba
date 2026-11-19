"""Pure-Python unit tests for tune_eval_match and tools.run_eval_param_match."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.run_eval_param_match import _load_params_json, _parse_overrides
from tune_eval_match.export_constants import build_constants_patch, export_params_to_files
from tune_eval_match.spsa_match import _average_tail_thetas, _parse_sprt


class TestEvalMatchSpsa(unittest.TestCase):
    def test_average_tail_thetas_takes_second_half(self):
        hist = [
            {"A": 10.0, "B": 100.0},
            {"A": 12.0, "B": 102.0},
            {"A": 20.0, "B": 110.0},
            {"A": 22.0, "B": 114.0},
        ]
        avg = _average_tail_thetas(hist, ["A", "B"])
        self.assertAlmostEqual(avg["A"], 21.0)
        self.assertAlmostEqual(avg["B"], 112.0)

    def test_parse_sprt_validation(self):
        self.assertIsNone(_parse_sprt(""))
        self.assertEqual(_parse_sprt("0,10"), (0.0, 10.0))
        with self.assertRaises(ValueError):
            _parse_sprt("10,0")
        with self.assertRaises(ValueError):
            _parse_sprt("bad")

    def test_load_params_json_supports_best_and_flat_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            p_best = Path(tmp) / "best_params.json"
            p_best.write_text(
                json.dumps({"block": "space", "theta_int": {"SPACE_THRESHOLD": 6200}}),
                encoding="utf-8",
            )
            self.assertEqual(_load_params_json(p_best), {"SPACE_THRESHOLD": 6200})

            p_flat = Path(tmp) / "flat.json"
            p_flat.write_text(json.dumps({"SPACE_BONUS_DIVISOR": 14}), encoding="utf-8")
            self.assertEqual(_load_params_json(p_flat), {"SPACE_BONUS_DIVISOR": 14})

    def test_parse_overrides_validates_known_names(self):
        self.assertEqual(
            _parse_overrides(["THREAT_SAFE_PAWN_MG=62", "SPACE_THRESHOLD=5800"]),
            {"THREAT_SAFE_PAWN_MG": 62, "SPACE_THRESHOLD": 5800},
        )
        with self.assertRaises(ValueError):
            _parse_overrides(["NON_EXISTENT_PARAM=1"])

    def test_export_constants_updates_arrays_and_scalars_preserving_syntax(self):
        import numpy as np

        updated_src, snippet_src, changed = build_constants_patch(
            {
                "THREAT_SAFE_PAWN_MG": 71,
                "THREAT_BY_MINOR_Q_MG": 55,
                "SPACE_THRESHOLD": 6123,
            }
        )
        self.assertEqual(
            set(changed),
            {"THREAT_SAFE_PAWN", "THREAT_BY_MINOR", "SPACE_THRESHOLD"},
        )
        ns = {"np": np}
        exec(compile(updated_src, "<patched_constants>", "exec"), ns)
        self.assertEqual(int(ns["THREAT_SAFE_PAWN"][0]), 71)
        self.assertEqual(int(ns["THREAT_BY_MINOR"][4, 0]), 55)
        self.assertEqual(int(ns["SPACE_THRESHOLD"]), 6123)
        self.assertIn("THREAT_SAFE_PAWN = np.array([71,", snippet_src)

        with tempfile.TemporaryDirectory() as tmp:
            out_c = Path(tmp) / "constants.py"
            out_s = Path(tmp) / "snippet.py"
            export_params_to_files(
                {"THREAT_SAFE_PAWN_MG": 71},
                out_constants=out_c,
                out_snippet=out_s,
            )
            self.assertTrue(out_c.exists())
            self.assertTrue(out_s.exists())


if __name__ == "__main__":
    unittest.main()
