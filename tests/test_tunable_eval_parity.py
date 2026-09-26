"""
Parity: classical evaluate_position vs tuner evaluate_position_tunable(theta0).

Run from repo root (first JIT may take ~1 min):

    python -m unittest tests.test_tunable_eval_parity -v
"""
from __future__ import annotations

import unittest

import numpy as np

from chess_engine.classical.evaluation import (
    _phase_limits_from_material as production_phase_limits,
    evaluate_position,
)
from chess_engine.classical.fen_parser import parse_fen
from chess_engine.classical.endgame import evaluate_special_endgame_with_material
from tuner.constraint_manager import ConstraintManager
from tuner.parameters import ParameterManager
from tuner.tunable_eval import (
    _phase_limits_from_material as tuner_phase_limits,
    evaluate_position_tunable,
    get_2d_val,
    get_array_val,
    get_int,
)
from tuner.tuner import SPSAOptimizer, compute_mse, compute_squared_errors

FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
]


class TestTunableEvalParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.param_manager = ParameterManager()
        cls.theta = cls.param_manager.get_initial_theta()

    def test_parity_on_sample_fens(self):
        for fen in FENS:
            with self.subTest(fen=fen[:40]):
                piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
                ref = evaluate_position(piece_bbs, occupancy_bbs, game_state, lazy=False)
                ref_score = int(ref[0] if isinstance(ref, tuple) else ref)
                tun = int(
                    evaluate_position_tunable(
                        piece_bbs, occupancy_bbs, game_state, self.theta, False
                    )
                )
                self.assertEqual(ref_score, tun, msg=f"ref={ref_score} tun={tun} fen={fen}")

    def test_phase_limit_helper_parity_for_candidate_material(self):
        candidate = (371, 409, 628, 1247)
        expected_mid = 2 * (
            2 * candidate[0] + 2 * candidate[1] + 2 * candidate[2] + candidate[3]
        )
        production = tuple(int(v) for v in production_phase_limits(*candidate))
        tunable = tuple(int(v) for v in tuner_phase_limits(*candidate))

        self.assertEqual(production, tunable)
        self.assertEqual(production[0], expected_mid)

    def test_candidate_material_reaches_specialized_endgame(self):
        piece_bbs, occupancy_bbs, game_state = parse_fen(
            "4k3/8/8/8/3Q4/8/8/3rK3 w - - 0 1"
        )
        candidate = self.theta.copy()
        mg_start, _ = self.param_manager.get_param_indices("MG_MATERIAL_VALUES")
        eg_start, _ = self.param_manager.get_param_indices("EG_MATERIAL_VALUES")
        candidate[mg_start + 3] += 9
        candidate[mg_start + 4] += 17
        candidate[eg_start + 3] -= 11
        candidate[eg_start + 4] += 37

        hit, expected = evaluate_special_endgame_with_material(
            piece_bbs, occupancy_bbs, game_state, np.int32(0),
            *(np.rint(candidate[mg_start:mg_start + 5]).astype(np.int32)),
            *(np.rint(candidate[eg_start:eg_start + 5]).astype(np.int32)),
        )
        actual = int(evaluate_position_tunable(
            piece_bbs, occupancy_bbs, game_state, candidate, False
        ))

        self.assertTrue(hit)
        self.assertEqual(actual, int(expected))

    def test_bucket_error_vector_matches_scalar_mse(self):
        parsed = [parse_fen(fen) for fen in FENS[:2]]
        piece_bbs = np.stack([position[0] for position in parsed])
        occupancy_bbs = np.stack([position[1] for position in parsed])
        game_states = np.stack([position[2] for position in parsed])
        results = np.array([1.0, 0.0], dtype=np.float64)

        scalar = float(
            compute_mse(
                piece_bbs,
                occupancy_bbs,
                game_states,
                results,
                self.theta,
            )
        )
        errors = compute_squared_errors(
            piece_bbs,
            occupancy_bbs,
            game_states,
            results,
            self.theta,
        )

        self.assertAlmostEqual(float(np.mean(errors)), scalar, places=12)

    def test_theta_quantization_matches_saved_constants_rounding(self):
        theta = np.array([2.4, 2.5, 2.6, 3.5, -2.4, -2.5, -2.6, -3.5])
        expected = np.round(theta).astype(np.int32)

        for idx in range(len(theta)):
            self.assertEqual(int(get_int(theta, idx)), int(expected[idx]))
            self.assertEqual(int(get_array_val(theta, 0, idx)), int(expected[idx]))
            self.assertEqual(int(get_2d_val(theta, 0, idx // 2, 2, idx % 2)), int(expected[idx]))

    def test_tuner_gauges_preserve_pst_means_and_mobility_anchors(self):
        optimizer = SPSAOptimizer.__new__(SPSAOptimizer)
        optimizer.param_manager = self.param_manager
        optimizer.constraint_manager = ConstraintManager(self.param_manager)
        optimizer.best_theta = self.theta.copy()
        optimizer._setup_constraints()

        theta = self.theta.copy()
        active_mask = np.zeros_like(theta)
        rng = np.random.default_rng(7)

        for table_name in ("PST_MG", "PST_EG"):
            start, count = self.param_manager.get_param_indices(table_name)
            theta[start : start + count] += rng.normal(0.0, 7.0, size=count)
            active_mask[start : start + count] = 1.0

        mobility_names = (
            "KNIGHT_MOBILITY_BONUS",
            "BISHOP_MOBILITY_BONUS",
            "ROOK_MOBILITY_BONUS",
            "QUEEN_MOBILITY_BONUS",
        )
        for name in mobility_names:
            start, count = self.param_manager.get_param_indices(name)
            theta[start : start + count] += rng.normal(0.0, 3.0, size=count)
            active_mask[start : start + count] = 1.0

        optimizer.constraint_manager.apply(theta, active_mask=active_mask)

        for table_name in ("PST_MG", "PST_EG"):
            start, _ = self.param_manager.get_param_indices(table_name)
            for piece in range(6):
                squares = range(8, 56) if piece == 0 else range(64)
                indices = [start + piece * 64 + sq for sq in squares]
                self.assertAlmostEqual(
                    float(np.mean(theta[indices])),
                    float(np.mean(self.theta[indices])),
                    places=8,
                )
                self.assertEqual(
                    int(np.sum(np.rint(theta[indices]))),
                    int(np.sum(np.rint(self.theta[indices]))),
                )

        for name in mobility_names:
            start, _ = self.param_manager.get_param_indices(name)
            self.assertEqual(theta[start], self.theta[start])
            self.assertEqual(theta[start + 1], self.theta[start + 1])

        material_mask = optimizer._create_mask(
            include_list=["MG_MATERIAL_VALUES", "EG_MATERIAL_VALUES"]
        )
        mg_start, _ = self.param_manager.get_param_indices("MG_MATERIAL_VALUES")
        eg_start, _ = self.param_manager.get_param_indices("EG_MATERIAL_VALUES")
        self.assertEqual(material_mask[mg_start], 0.0)
        self.assertEqual(material_mask[mg_start + 5], 0.0)
        self.assertEqual(material_mask[eg_start], 1.0)
        self.assertEqual(material_mask[eg_start + 5], 0.0)

        knight_mask = optimizer._create_mask(
            include_list=[
                "MG_MATERIAL_VALUES",
                "EG_MATERIAL_VALUES",
                "PST_MG",
                "PST_EG",
            ],
            pst_pieces="N",
        )
        self.assertEqual(int(np.count_nonzero(knight_mask[mg_start : mg_start + 6])), 1)
        self.assertEqual(int(np.count_nonzero(knight_mask[eg_start : eg_start + 6])), 1)
        self.assertEqual(knight_mask[mg_start + 1], 1.0)
        self.assertEqual(knight_mask[eg_start + 1], 1.0)

    def test_validation_bucket_membership_uses_reference_phase(self):
        optimizer = SPSAOptimizer.__new__(SPSAOptimizer)
        optimizer.param_manager = self.param_manager
        optimizer.piece_bbs = np.zeros((3, 12), dtype=np.uint64)

        # Kings only.
        optimizer.piece_bbs[0, 5] = np.uint64(1)
        optimizer.piece_bbs[0, 11] = np.uint64(2)
        # Kings + eight pawns: broad endgame and 8..12 pieces.
        optimizer.piece_bbs[1, 0] = np.uint64(0xF)
        optimizer.piece_bbs[1, 6] = np.uint64(0xF0)
        optimizer.piece_bbs[1, 5] = np.uint64(1 << 8)
        optimizer.piece_bbs[1, 11] = np.uint64(1 << 9)
        # Full starting material counts (square overlap is irrelevant here).
        start_counts = (8, 2, 2, 2, 1, 1, 8, 2, 2, 2, 1, 1)
        for piece, count in enumerate(start_counts):
            optimizer.piece_bbs[2, piece] = np.uint64((1 << count) - 1)

        buckets = dict(
            optimizer._build_validation_buckets(
                np.arange(3, dtype=np.int64), self.theta
            )
        )

        self.assertTrue(buckets["phase=0"][0])
        self.assertTrue(buckets["broad-EG phase<=64"][1])
        self.assertTrue(buckets["pieces=8..12"][1])
        self.assertTrue(buckets["phase=96..128"][2])
        self.assertFalse(buckets["queenless"][2])
        self.assertTrue(buckets["pure king+pawns"][0])
        self.assertTrue(buckets["pure king+pawns"][1])
        self.assertFalse(buckets["pure king+pawns"][2])


if __name__ == "__main__":
    unittest.main()
