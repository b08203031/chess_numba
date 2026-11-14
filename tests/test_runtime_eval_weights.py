from types import SimpleNamespace
import unittest
import numpy as np

from chess_engine.classical import constants as classical_constants
from chess_engine.classical.constants import (
    EG_SAFETY_SCALE,
    MAX_PLY,
)
from chess_engine.classical.fen_parser import parse_fen
from chess_engine.classical.transposition_table import create_transposition_table
from chess_engine.classical.engine_types import SearchContext
from chess_engine.classical.evaluation import _evaluate_position_jit
from tune_eval_match.eval_weights import (
    default_eval_weights,
    EW_EG_SAFETY_SCALE,
    EW_MAT_Q_EG,
    EW_PASS_R4_EG,
)


def make_test_context():
    tt = create_transposition_table(1)
    return SearchContext(
        tt,
        np.zeros(MAX_PLY * 2, dtype=np.uint16),
        np.zeros((MAX_PLY, MAX_PLY), dtype=np.uint16),
        np.zeros((12, 64), dtype=np.int32),
        np.zeros((64, 64), dtype=np.int32),
        np.zeros((5, 12, 64, 12, 64), dtype=np.int16),
        np.zeros((12, 64, 6), dtype=np.int32),
        np.full((8192, 12, 64), -1238, dtype=np.int16),
        np.zeros(16384, dtype=np.int16),
        np.zeros(16384, dtype=np.int16),
        np.zeros(16384, dtype=np.int16),
        np.zeros(16384, dtype=np.int16),
        np.zeros((12, 64, 12, 64), dtype=np.int16),
    )


class TestRuntimeEvalWeights(unittest.TestCase):
    def test_eg_safety_scale_unified_constant(self):
        """EG_SAFETY_SCALE in constants.py and eval_weights.py must agree dynamically."""
        self.assertGreater(int(EG_SAFETY_SCALE), 0)
        w = default_eval_weights()
        self.assertEqual(int(w[EW_EG_SAFETY_SCALE]), int(EG_SAFETY_SCALE))

        custom_consts = SimpleNamespace(
            **{
                k: getattr(classical_constants, k)
                for k in dir(classical_constants)
                if k.isupper()
            }
        )
        custom_consts.EG_SAFETY_SCALE = 137
        w_custom = default_eval_weights(custom_consts)
        self.assertEqual(int(w_custom[EW_EG_SAFETY_SCALE]), 137)

    def test_specialized_endgame_respects_runtime_material_weights(self):
        """Specialized endgames (e.g., KQKR) must use active runtime material weights."""
        # KQKR endgame: White Kh1, Qa8 vs Black Kh8, Ra1
        fen = "Q6k/8/8/8/8/8/8/R6K w - - 0 1"
        p, o, g = parse_fen(fen)

        # Baseline evaluation
        ctx_default = make_test_context()
        score_base, _, _ = _evaluate_position_jit(p, o, g, False, ctx_default)

        # Mutate Queen EG value by +200 cp in runtime weights (in-place array mutation)
        ctx_boosted = make_test_context()
        ctx_boosted.eval_weights[EW_MAT_Q_EG] += np.int32(200)

        score_boosted, _, _ = _evaluate_position_jit(p, o, g, False, ctx_boosted)

        # KQKR should give White a larger score when Queen EG material is boosted
        self.assertGreater(
            score_boosted,
            score_base,
            f"Boosted Queen EG material did not increase KQKR score: {score_boosted} <= {score_base}",
        )

    def test_king_pawn_endgame_respects_eval_weights_connection(self):
        """Multi-pawn KP endgame must receive runtime eval weights and apply KP scale factor."""
        # 1) Unscaled KP (strong side has 4 pawns -> sf = min(64, 36 + 7*4) = 64):
        # White Ka1, passed Pa5 (rank index 4) vs Black Kh8, Pe7, Pf7, Pg7, Ph7
        fen_unscaled = "7k/4pppp/8/P7/8/8/8/K7 w - - 0 1"
        p4, o4, g4 = parse_fen(fen_unscaled)

        ctx_base4 = make_test_context()
        score_base4, _, _ = _evaluate_position_jit(p4, o4, g4, False, ctx_base4)

        ctx_boosted4 = make_test_context()
        ctx_boosted4.eval_weights[EW_PASS_R4_EG] += np.int32(-64)
        score_boosted4, _, _ = _evaluate_position_jit(p4, o4, g4, False, ctx_boosted4)

        self.assertEqual(
            int(score_boosted4 - score_base4),
            -64,
            f"Unscaled KP (sf=64) must reflect full -64 EW_PASS_R4_EG delta: got {score_boosted4 - score_base4}",
        )

        # 2) Scaled KP (strong side has 2 pawns -> sf = min(64, 36 + 7*2) = 50; -64 * 50 // 64 = -50, 0 remainder):
        # White Ka1, Pd5 vs Black Kh8, Pg7, Ph7
        fen_scaled = "7k/6pp/8/3P4/8/8/8/K7 w - - 0 1"
        p2, o2, g2 = parse_fen(fen_scaled)

        ctx_base2 = make_test_context()
        score_base2, _, _ = _evaluate_position_jit(p2, o2, g2, False, ctx_base2)

        ctx_boosted2 = make_test_context()
        ctx_boosted2.eval_weights[EW_PASS_R4_EG] += np.int32(-64)
        score_boosted2, _, _ = _evaluate_position_jit(p2, o2, g2, False, ctx_boosted2)

        self.assertEqual(
            int(score_boosted2 - score_base2),
            -50,
            f"Scaled KP (sf=50) must scale -64 EW_PASS_R4_EG delta to -50 cp: got {score_boosted2 - score_base2}",
        )


if __name__ == "__main__":
    unittest.main()
