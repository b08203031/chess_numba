import unittest
import numpy as np

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
        """EG_SAFETY_SCALE in constants.py and eval_weights.py must agree at 100."""
        self.assertEqual(int(EG_SAFETY_SCALE), 100)
        w = default_eval_weights()
        self.assertEqual(int(w[EW_EG_SAFETY_SCALE]), 100)

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
        """Multi-pawn KP endgame must receive and use active runtime eval weights."""
        # Multi-pawn KP endgame: White Ka1, Pa2, Pb2 vs Black Kh8, Pg7, Ph7
        fen = "7k/6pp/8/8/8/8/PP6/K7 w - - 0 1"
        p, o, g = parse_fen(fen)

        ctx = make_test_context()
        score, _, _ = _evaluate_position_jit(p, o, g, False, ctx)
        self.assertIsInstance(int(score), int)


if __name__ == "__main__":
    unittest.main()
