import unittest
import random
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from chess_engine.classical import constants as C
from tune_search import param_config
from tune_search.search_param_registry import (
    DEFAULT_PROFILE,
    SEARCH_PARAM_BY_NAME,
    SEARCH_PARAM_SPECS,
    get_spec,
    profile_names,
    quantize_integer_value,
    quantize_value,
    validate_registry,
)
from tune_search.runtime_apply import apply_runtime_params, read_runtime_params

import numpy as np
from tune_search.search_tuner import SPSAOptimizer, _append_jsonl, _load_state, _save_state


class TestSearchParamRegistry(unittest.TestCase):
    def test_registry_is_larger_than_runtime_slots(self):
        self.assertGreater(len(SEARCH_PARAM_SPECS), int(C.TUNE_SIZE))
        self.assertEqual(len(profile_names("runtime")), int(C.TUNE_SIZE))

    def test_registry_invariants_and_sources(self):
        self.assertEqual(validate_registry(), ())
        self.assertEqual(len(SEARCH_PARAM_BY_NAME), len(SEARCH_PARAM_SPECS))
        for spec in SEARCH_PARAM_SPECS:
            self.assertEqual(spec.default, int(getattr(C, spec.source_name)))
            self.assertGreater(spec.step, 0)
            self.assertLessEqual(spec.minimum, spec.default)
            self.assertLessEqual(spec.default, spec.maximum)
            if spec.is_runtime:
                self.assertIsNotNone(spec.runtime_index)
                self.assertGreaterEqual(spec.runtime_index, 0)
                self.assertLess(spec.runtime_index, int(C.TUNE_SIZE))

    def test_old_missing_names_are_not_silently_tunable(self):
        for stale in ("FP_MARGIN_D1", "FP_MARGIN_D2", "RFP_MARGIN_D1", "SINGULAR_EXTENSION_MARGIN"):
            self.assertNotIn(stale, SEARCH_PARAM_BY_NAME)
            self.assertNotIn(stale, param_config.SEARCH_PARAMS)

    def test_legacy_facade_matches_registry(self):
        self.assertEqual(set(param_config.SEARCH_PARAMS), set(SEARCH_PARAM_BY_NAME))
        for spec in SEARCH_PARAM_SPECS:
            self.assertEqual(param_config.SEARCH_PARAMS[spec.name], spec.legacy_tuple())

    def test_profiles_are_staged_and_nonempty(self):
        self.assertEqual(DEFAULT_PROFILE, "runtime")
        for profile in ("runtime", "pruning", "nmp", "lmr", "lmr_runtime", "lmr_compile", "qsearch", "history", "ordering", "safeguards", "aspiration", "all"):
            names = profile_names(profile)
            self.assertTrue(names, profile)
            self.assertEqual(len(names), len(set(names)))
        self.assertEqual(len(profile_names("all")), len(SEARCH_PARAM_SPECS))
        self.assertTrue(all(get_spec(name).is_runtime for name in profile_names("lmr_runtime")))
        self.assertTrue(all(not get_spec(name).is_runtime for name in profile_names("lmr_compile")))
        self.assertEqual(
            set(profile_names("lmr_runtime")) | set(profile_names("lmr_compile")),
            set(profile_names("lmr")),
        )

    def test_quantization_respects_step_and_bounds(self):
        for spec in SEARCH_PARAM_SPECS:
            low = quantize_value(spec.name, spec.minimum - 100000)
            high = quantize_value(spec.name, spec.maximum + 100000)
            self.assertEqual(low, spec.minimum)
            self.assertEqual(high, spec.maximum)
            value = quantize_value(spec.name, spec.default + spec.step / 2.0)
            self.assertGreaterEqual(value, spec.minimum)
            self.assertLessEqual(value, spec.maximum)
            self.assertEqual((value - spec.default) % spec.step, 0)
            self.assertEqual(quantize_value(spec.name, spec.default), spec.default)

    def test_integer_quantization_uses_one_unit_grid(self):
        self.assertEqual(quantize_integer_value("LMR_BASE_OFFSET", 459.4), 459)
        self.assertEqual(quantize_integer_value("LMR_CUTNODE_BONUS", 2047.6), 2048)
        self.assertEqual(quantize_integer_value("LMR_BASE_OFFSET", -1000), 250)
        self.assertEqual(quantize_integer_value("LMR_CUTNODE_BONUS", 99999), 3000)

    def test_runtime_mode_defaults_are_explicit_constants(self):
        self.assertEqual(get_spec("NMP_GATE_MODE").default, 0)
        self.assertEqual(get_spec("NMP_R_MODE").default, 0)
        self.assertEqual(get_spec("PROBCUT_STYLE_MODE").default, 0)
        self.assertEqual(get_spec("NMP_NEED_BETA").default, 0)

    def test_runtime_bridge_mutates_only_runtime_slots(self):
        class FakeContext:
            def __init__(self):
                self.tune = np.zeros(int(C.TUNE_SIZE), dtype=np.int32)

        ctx = FakeContext()
        apply_runtime_params(ctx, {"LMR_BASE_OFFSET": 999999, "NMP_GATE_MODE": 2})
        self.assertEqual(int(ctx.tune[get_spec("LMR_BASE_OFFSET").runtime_index]), 700)
        self.assertEqual(int(ctx.tune[get_spec("NMP_GATE_MODE").runtime_index]), 2)
        values = read_runtime_params(ctx, ("LMR_BASE_OFFSET", "NMP_GATE_MODE"))
        self.assertEqual(values, {"LMR_BASE_OFFSET": 700, "NMP_GATE_MODE": 2})
        apply_runtime_params(ctx, {"LMR_BASE_OFFSET": 459}, scale_mode="integer")
        self.assertEqual(
            int(ctx.tune[get_spec("LMR_BASE_OFFSET").runtime_index]), 459
        )
        with self.assertRaises(ValueError):
            apply_runtime_params(ctx, {"ASPIRATION_WINDOW_BASE": 30})

    def test_spsa_uses_registry_scales_and_integer_axes(self):
        names = ("LMR_BASE_OFFSET", "NMP_GATE_MODE")
        initial = {name: get_spec(name).default for name in names}
        optimizer = SPSAOptimizer(names, initial)
        captured = {}

        def fake_match(base, plus):
            captured["base"] = dict(base)
            captured["plus"] = dict(plus)
            return 0.75

        import random

        random.seed(7)
        optimizer.step(fake_match)
        for name in names:
            spec = get_spec(name)
            self.assertIsInstance(captured["plus"][name], int)
            self.assertGreaterEqual(captured["plus"][name], spec.minimum)
            self.assertLessEqual(captured["plus"][name], spec.maximum)
            self.assertEqual((captured["plus"][name] - spec.default) % spec.step, 0)
            self.assertIsInstance(optimizer.current_params[name], int)

    def test_spsa_iteration_record_is_jsonl_serializable(self):
        names = ("LMR_BASE_OFFSET", "LMR_HISTORY_SCALE")
        initial = {name: get_spec(name).default for name in names}
        optimizer = SPSAOptimizer(names, initial)
        old_random_state = random.getstate()
        try:
            random.seed(17)
            record = optimizer.step(lambda _base, _plus: 0.5)
        finally:
            random.setstate(old_random_state)
        self.assertEqual(record["iteration"], 1)
        self.assertIn("theta_plus", record)
        self.assertIn("updated_params", record)
        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "campaign.jsonl"
            _append_jsonl(log_path, {"event": "iteration", **record})
            line = log_path.read_text(encoding="utf-8").strip()
            loaded = json.loads(line)
            self.assertEqual(loaded["event"], "iteration")
            self.assertEqual(loaded["iteration"], 1)

    def test_spsa_uses_fishtest_pair_schedule_and_boundary_rounding(self):
        names = ("LMR_BASE_OFFSET", "LMR_CUTNODE_BONUS")
        initial = {name: get_spec(name).default for name in names}
        optimizer = SPSAOptimizer(
            names,
            initial,
            match_games=50,
            total_batches=24,
            scale_mode="integer",
        )
        old_random_state = random.getstate()
        try:
            random.seed(7)
            record = optimizer.step(lambda _base, _plus: 0.42)
        finally:
            random.setstate(old_random_state)
        self.assertEqual(record["scale_mode"], "integer")
        self.assertEqual(record["match_pairs"], 25)
        self.assertEqual(record["spsa_iteration"], 25)
        self.assertEqual(record["total_pairs"], 600)
        self.assertEqual(record["A"], 60.0)
        for name in names:
            spec = get_spec(name)
            self.assertIsInstance(record["theta_plus"][name], int)
            self.assertIsInstance(record["theta_minus"][name], int)
            self.assertGreater(record["c_k"][name], float(spec.step))
            self.assertAlmostEqual(record["r_end"][name], spec.canonical_r_end)
            self.assertGreater(record["r_k"][name], 0.0)
            self.assertGreaterEqual(optimizer.current_params[name], spec.minimum)
            self.assertLessEqual(optimizer.current_params[name], spec.maximum)

    def test_spsa_state_round_trip_rejects_mixed_parameter_sets(self):
        names = ("LMR_BASE_OFFSET", "LMR_HISTORY_SCALE")
        initial = {name: get_spec(name).default for name in names}
        optimizer = SPSAOptimizer(names, initial)
        optimizer.iteration = 3
        optimizer.theta["LMR_BASE_OFFSET"] = 512.5
        optimizer.current_params["LMR_BASE_OFFSET"] = 512
        args = SimpleNamespace(games=12, nodes=3000, concurrency=4)

        old_random_state = random.getstate()
        try:
            with TemporaryDirectory() as tmp:
                state_path = Path(tmp) / "state.json"
                _save_state(
                    state_path,
                    optimizer,
                    backend="inprocess",
                    profile="explicit",
                    param_names=names,
                    args=args,
                )
                restored = SPSAOptimizer(names, initial)
                state = _load_state(state_path, restored, names)
                self.assertEqual(state["iteration"], 3)
                self.assertIn("random_state", state)
                self.assertEqual(restored.current_params["LMR_BASE_OFFSET"], 512)
                with self.assertRaises(ValueError):
                    _load_state(state_path, restored, ("LMR_BASE_OFFSET",))
                with self.assertRaises(ValueError):
                    _load_state(
                        state_path,
                        restored,
                        names,
                        expected_backend="uci",
                    )
                with self.assertRaises(ValueError):
                    _load_state(
                        state_path,
                        restored,
                        names,
                        expected_scale_mode="integer",
                    )
        finally:
            random.setstate(old_random_state)


if __name__ == "__main__":
    unittest.main()
