"""Lightweight invariants for the HCE Texel/SPSA parameterization."""
from __future__ import annotations

import unittest

import numpy as np

from chess_engine.classical.constants import (
    BISHOP,
    ENDGAME_LIMIT,
    KNIGHT,
    MG_MATERIAL_VALUES,
    MIDGAME_LIMIT,
    PHASE_ENDGAME_RATIO_DEN,
    PHASE_ENDGAME_RATIO_NUM,
    QUEEN,
    ROOK,
)
from tuner.apply_tuned import _extract_assignments, apply_tuned_source
from tuner.constraint_manager import ConstraintManager
from tuner.data_pipeline.preprocess_data import parse_fen_to_arrays
from tuner.parameters import ParameterManager
from tuner.tuner import SPSAOptimizer, split_dataset_indices


class _DummyParameterManager:
    def __init__(self):
        self.param_map = [
            {"name": "TABLE", "start": 0, "count": 4, "shape": (4,)}
        ]


class TestPhaseMaterialCoupling(unittest.TestCase):
    def test_limits_follow_current_mg_material(self):
        expected_midgame = 2 * (
            2 * int(MG_MATERIAL_VALUES[KNIGHT])
            + 2 * int(MG_MATERIAL_VALUES[BISHOP])
            + 2 * int(MG_MATERIAL_VALUES[ROOK])
            + int(MG_MATERIAL_VALUES[QUEEN])
        )
        expected_endgame = (
            expected_midgame * int(PHASE_ENDGAME_RATIO_NUM)
            + int(PHASE_ENDGAME_RATIO_DEN) // 2
        ) // int(PHASE_ENDGAME_RATIO_DEN)

        self.assertEqual(int(MIDGAME_LIMIT), expected_midgame)
        self.assertEqual(int(ENDGAME_LIMIT), expected_endgame)
        self.assertGreater(int(MIDGAME_LIMIT), int(ENDGAME_LIMIT))


class TestFixedMeanConstraint(unittest.TestCase):
    def test_projection_preserves_mean_and_range(self):
        manager = ConstraintManager(_DummyParameterManager())
        manager.add_range("TABLE", min_val=0.0, max_val=30.0)
        manager.add_fixed_mean("TABLE", range(4), target_mean=15.0)
        theta = np.array([50.0, -10.0, 25.0, 30.0], dtype=np.float64)

        manager.apply(theta)

        self.assertAlmostEqual(float(np.mean(theta)), 15.0, places=9)
        self.assertTrue(np.all(theta >= 0.0))
        self.assertTrue(np.all(theta <= 30.0))

    def test_projection_does_not_move_frozen_entries(self):
        manager = ConstraintManager(_DummyParameterManager())
        manager.add_range("TABLE", min_val=0.0, max_val=30.0)
        manager.add_fixed_mean("TABLE", range(4), target_mean=15.0)
        theta = np.array([0.0, 20.0, 20.0, 30.0], dtype=np.float64)
        active_mask = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64)

        manager.apply(theta, active_mask=active_mask)

        self.assertAlmostEqual(float(np.mean(theta)), 15.0, places=9)
        self.assertEqual(theta[0], 0.0)
        self.assertEqual(theta[3], 30.0)

    def test_quantized_projection_preserves_float_and_integer_sum(self):
        manager = ConstraintManager(_DummyParameterManager())
        manager.add_fixed_mean(
            "TABLE", range(4), target_mean=0.0, quantized=True
        )
        # Continuous sum is already zero, but banker's rounding sums to -1.
        theta = np.array([0.49, 0.49, 0.49, -1.47], dtype=np.float64)

        manager.apply(theta)

        self.assertAlmostEqual(float(np.sum(theta)), 0.0, places=9)
        self.assertEqual(int(np.sum(np.rint(theta))), 0)
        self.assertTrue(np.any(np.abs(theta - np.rint(theta)) > 1e-6))


class TestIntegerAwareTuningBands(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pm = ParameterManager()
        optimizer = SPSAOptimizer.__new__(SPSAOptimizer)
        optimizer.param_manager = cls.pm
        optimizer.constraint_manager = ConstraintManager(cls.pm)
        optimizer._setup_constraints()
        optimizer.best_theta = cls.pm.get_initial_theta()
        cls.optimizer = optimizer
        cls.cm = optimizer.constraint_manager

    def test_small_initiative_weight_can_reach_adjacent_integers(self):
        name = "INITIATIVE_OUTFLANKING_WEIGHT_EG"
        start, _ = self.pm.get_param_indices(name)
        value = float(self.pm.theta[start])
        lower, upper = self.cm._absolute_bounds(name, np.array([start]))

        self.assertLessEqual(lower[0], value - 1.0)
        self.assertGreaterEqual(upper[0], value + 1.0)

    def test_imbalance_structural_zeros_stay_pinned(self):
        name = "IMBALANCE_QUADRATIC_OURS"
        start, count = self.pm.get_param_indices(name)
        values = np.asarray(self.pm.theta[start:start + count], dtype=np.float64)
        zero_rel = int(np.flatnonzero(values == 0.0)[0])
        nonzero_rel = int(np.flatnonzero((values != 0.0) & (np.abs(values) < 10.0))[0])
        indices = np.array([start + zero_rel, start + nonzero_rel])
        lower, upper = self.cm._absolute_bounds(name, indices)

        self.assertEqual((lower[0], upper[0]), (0.0, 0.0))
        self.assertLessEqual(lower[1], values[nonzero_rel] - 1.0)
        self.assertGreaterEqual(upper[1], values[nonzero_rel] + 1.0)

    def test_imbalance_zero_cells_are_excluded_from_spsa_mask(self):
        names = ["IMBALANCE_QUADRATIC_OURS", "IMBALANCE_QUADRATIC_THEIRS"]
        mask = self.optimizer._create_mask(include_list=names)

        for name in names:
            start, count = self.pm.get_param_indices(name)
            values = np.asarray(
                self.pm.theta[start:start + count], dtype=np.float64
            )
            local_mask = mask[start:start + count]
            self.assertTrue(np.all(local_mask[values == 0.0] == 0.0))
            self.assertTrue(np.all(local_mask[values != 0.0] == 1.0))

    def test_king_attack_structural_weights_are_excluded_from_mask(self):
        name = "KING_ATTACK_WEIGHTS"
        mask = self.optimizer._create_mask(include_list=[name])
        start, count = self.pm.get_param_indices(name)

        self.assertEqual(count, 6)
        np.testing.assert_array_equal(
            mask[start:start + count],
            np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
        )

    def test_dead_initiative_pawn_endgame_weights_are_pinned_and_frozen(self):
        dead = [
            "INITIATIVE_PAWN_ENDGAME_WEIGHT_MG",
            "INITIATIVE_PAWN_ENDGAME_WEIGHT_EG",
        ]
        mask = self.optimizer._create_mask(include_list=dead)
        for name in dead:
            start, _ = self.pm.get_param_indices(name)
            v0 = float(self.pm.theta[start])
            lower, upper = self.cm._absolute_bounds(name, np.array([start]))
            self.assertEqual((lower[0], upper[0]), (v0, v0))
            self.assertEqual(mask[start], 0.0)

    def test_param_scales_respect_unit_classes_and_tight_boxes(self):
        theta0 = self.pm.get_initial_theta()
        mask = self.optimizer._create_mask(
            include_list=[
                "MG_MATERIAL_VALUES",
                "PST_MG",
                "SAFE_CHECK_ROOK",
                "PASSED_PAWN_BONUS",
                "PASSED_PATH_SAFE_NONE_ATTACK",
                "IMBALANCE_QUADRATIC_OURS",
            ]
        )
        scales = self.optimizer._build_param_scales(theta0, mask, c=2.5)

        mg_s, mg_c = self.pm.get_param_indices("MG_MATERIAL_VALUES")
        pst_s, pst_c = self.pm.get_param_indices("PST_MG")
        np.testing.assert_allclose(scales[mg_s : mg_s + mg_c], 1.0)
        np.testing.assert_allclose(scales[pst_s : pst_s + pst_c], 1.0)

        scr_s, _ = self.pm.get_param_indices("SAFE_CHECK_ROOK")
        self.assertGreaterEqual(scales[scr_s], 10.0)

        pp_s, _ = self.pm.get_param_indices("PASSED_PAWN_BONUS")
        self.assertEqual(scales[pp_s + 2], 1.0)      # small low-rank entry
        self.assertGreater(scales[pp_s + 12], 3.0)   # rank-6 132cp entry

        path_s, _ = self.pm.get_param_indices("PASSED_PATH_SAFE_NONE_ATTACK")
        self.assertEqual(scales[path_s], 1.0)        # ±1 box stays 1.0


class TestSplitDatasetIndices(unittest.TestCase):
    def test_three_way_split_is_disjoint_and_exhaustive(self):
        train_idx, eval_idx, hold_idx = split_dataset_indices(
            20_000, eval_sample=1_000, holdout_sample=2_000
        )
        self.assertIsNotNone(eval_idx)
        self.assertIsNotNone(hold_idx)
        self.assertEqual(len(train_idx), 17_000)
        self.assertEqual(len(eval_idx), 1_000)
        self.assertEqual(len(hold_idx), 2_000)

        s_train = set(train_idx.tolist())
        s_eval = set(eval_idx.tolist())
        s_hold = set(hold_idx.tolist())
        self.assertEqual(len(s_train & s_eval), 0)
        self.assertEqual(len(s_train & s_hold), 0)
        self.assertEqual(len(s_eval & s_hold), 0)
        self.assertEqual(len(s_train | s_eval | s_hold), 20_000)

    def test_small_smoke_dataset_disables_holdout_gracefully(self):
        train_idx, eval_idx, hold_idx = split_dataset_indices(
            500, eval_sample=1_000, holdout_sample=2_000
        )
        self.assertIsNone(eval_idx)
        self.assertIsNone(hold_idx)
        self.assertEqual(len(train_idx), 500)


class TestApplyTunedPreservesComments(unittest.TestCase):
    def test_ast_replacement_preserves_inline_and_following_comments(self):
        target_src = (
            "import numpy as np\n\n"
            "INITIATIVE_MG_OFFSET = np.int32(39)  # keep inline comment\n\n"
            "KING_DANGER_NO_QUEEN = 560\n"
            "# Important explanation block line 1\n"
            "# Important explanation block line 2\n"
            "KING_DANGER_NO_QUEEN_ROOKLESS = 810\n"
            "PASSED_FILE_BONUS = np.array([16, 4], dtype=np.int32)\n"
        )
        tuned_src = (
            "import numpy as np\n"
            "INITIATIVE_MG_OFFSET = 42\n"
            "KING_DANGER_NO_QUEEN = 575\n"
            "KING_DANGER_NO_QUEEN_ROOKLESS = 810\n"
            "PASSED_FILE_BONUS = np.array([14, 5], dtype=np.int32)\n"
        )
        target_ns = {"np": np}
        exec(target_src, target_ns)
        tuned_ns = {"np": np}
        exec(tuned_src, tuned_ns)
        tuned_assigns = _extract_assignments(tuned_src)

        new_src, changed, skipped = apply_tuned_source(
            target_src=target_src,
            tuned_ns=tuned_ns,
            target_ns=target_ns,
            tuned_assigns=tuned_assigns,
            names=list(tuned_assigns.keys()),
            verbose=False,
        )
        self.assertEqual(
            changed,
            ["INITIATIVE_MG_OFFSET", "KING_DANGER_NO_QUEEN", "PASSED_FILE_BONUS"],
        )
        self.assertEqual(skipped, 1)
        self.assertIn(
            "INITIATIVE_MG_OFFSET = np.int32(42)  # keep inline comment\n\n",
            new_src,
        )
        self.assertIn(
            "KING_DANGER_NO_QUEEN = 575\n"
            "# Important explanation block line 1\n"
            "# Important explanation block line 2\n"
            "KING_DANGER_NO_QUEEN_ROOKLESS = 810\n",
            new_src,
        )
        self.assertIn(
            "PASSED_FILE_BONUS = np.array([14, 5], dtype=np.int32)\n",
            new_src,
        )


class TestPreprocessCastlingAndEp(unittest.TestCase):
    def test_castling_rights_and_legal_ep_encoding(self):
        # Pseudo-EP (no adjacent black pawn on d4/f4) -> ep stored as 64
        _, _, gs_pseudo = parse_fen_to_arrays(
            "r3k2r/pppppppp/8/8/4P3/8/PPPP1PPP/R3K2R b KQkq e3 0 7"
        )
        self.assertEqual(int(gs_pseudo[0]), 1)
        self.assertEqual(int(gs_pseudo[1]), 15)  # K=1 | Q=2 | k=4 | q=8
        self.assertEqual(int(gs_pseudo[2]), 64)
        self.assertEqual(int(gs_pseudo[3]), 0)

        # Legal EP (black pawn on d4 can capture on e3 = sq 20)
        _, _, gs_legal = parse_fen_to_arrays(
            "r3k2r/ppp1pppp/8/8/3pP3/8/PPPP1PPP/R3K2R b KQk e3 0 7"
        )
        self.assertEqual(int(gs_legal[1]), 7)  # K=1 | Q=2 | k=4
        self.assertEqual(int(gs_legal[2]), 20)


if __name__ == "__main__":
    unittest.main()
