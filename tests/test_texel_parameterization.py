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
from tuner.constraint_manager import ConstraintManager
from tuner.parameters import ParameterManager
from tuner.tuner import SPSAOptimizer


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


if __name__ == "__main__":
    unittest.main()
