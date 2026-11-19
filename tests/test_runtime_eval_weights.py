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
from chess_engine.classical.evaluation import _evaluate_position_jit, evaluate_position
from tune_eval_match.eval_weights import (
    EVAL_PARAM_INDEX,
    EW_EG_SAFETY_SCALE,
    EW_IMBALANCE_DIVISOR,
    EW_KING_ATTACK_WEIGHT_N,
    EW_KING_DANGER_NO_QUEEN_ROOKLESS,
    EW_MAT_Q_EG,
    EW_PASS_R4_EG,
    EW_SPACE_BONUS_DIVISOR,
    EW_SPACE_THRESHOLD,
    EW_THREAT_BY_MINOR_Q_MG,
    EW_THREAT_BY_ROOK_N_MG,
    EW_THREAT_SAFE_PAWN_MG,
    EW_UNSTOPPABLE_PAWN_BONUS,
    default_eval_weights,
    format_constants_diff,
    params_from_weights,
)
from tune_eval_match.param_config import (
    ALL_PARAM_SPECS,
    BLOCKS,
    bounds,
    default_params,
    select_custom_params,
    set_active_block,
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

    def test_no_ctx_matches_default_eval_weights_exactly(self):
        fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        ]
        ctx = make_test_context()
        for fen in fens:
            p, o, g = parse_fen(fen)
            no_ctx = int(evaluate_position(p, o, g, lazy=False)[0])
            with_ctx = int(_evaluate_position_jit(p, o, g, False, ctx)[0])
            self.assertEqual(no_ctx, with_ctx, msg=f"Mismatch on {fen}")

    def test_threats_space_king_and_imbalance_div_are_live(self):
        # Position with pawn threat, minor-on-queen/rook threats, rook threats, imbalance & space
        fen = "r2q1rk1/ppp2ppp/2n1bn2/2b1p1N1/2B1P3/2NP2Q1/PPP2PPP/R1B2RK1 w - - 0 1"
        p, o, g = parse_fen(fen)
        base_ctx = make_test_context()
        base_score = int(_evaluate_position_jit(p, o, g, False, base_ctx)[0])

        # Space threshold raised above npm -> zeroes space in asymmetric space position
        p_sp, o_sp, g_sp = parse_fen("rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 5")
        ctx_sp0 = make_test_context()
        ctx_sp1 = make_test_context()
        ctx_sp1.eval_weights[EW_SPACE_THRESHOLD] = np.int32(99999)
        self.assertNotEqual(
            int(_evaluate_position_jit(p_sp, o_sp, g_sp, False, ctx_sp0)[0]),
            int(_evaluate_position_jit(p_sp, o_sp, g_sp, False, ctx_sp1)[0]),
        )

        # Knight king attack weight boosted above threshold -> changes king safety score
        ctx_kaw = make_test_context()
        ctx_kaw.eval_weights[EW_KING_ATTACK_WEIGHT_N] += np.int32(250)
        self.assertNotEqual(int(_evaluate_position_jit(p, o, g, False, ctx_kaw)[0]), base_score)

        # Safe pawn threat on knight (e4 pawn threatens d5/f5 -> use d4-d5 vs Nc6)
        p_t, o_t, g_t = parse_fen("r1bqkbnr/pppp1ppp/8/2n1p3/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 1")
        ctx_t0 = make_test_context()
        ctx_t1 = make_test_context()
        ctx_t1.eval_weights[EW_THREAT_SAFE_PAWN_MG] += np.int32(50)
        self.assertNotEqual(
            int(_evaluate_position_jit(p_t, o_t, g_t, False, ctx_t0)[0]),
            int(_evaluate_position_jit(p_t, o_t, g_t, False, ctx_t1)[0]),
        )

        # Rookless no-queen suppression
        p_rl, o_rl, g_rl = parse_fen("4k3/8/8/4n3/8/8/4PPPP/4K2B b - - 0 1")
        ctx_rl0 = make_test_context()
        ctx_rl1 = make_test_context()
        ctx_rl1.eval_weights[EW_KING_DANGER_NO_QUEEN_ROOKLESS] += np.int32(-1200)
        self.assertNotEqual(
            int(_evaluate_position_jit(p_rl, o_rl, g_rl, False, ctx_rl0)[0]),
            int(_evaluate_position_jit(p_rl, o_rl, g_rl, False, ctx_rl1)[0]),
        )

        # Imbalance divisor (material imbalance position below lazy threshold)
        p_imb, o_imb, g_imb = parse_fen("r2qk2r/pppp1ppp/2n2n2/8/8/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 0 1")
        ctx_imb0 = make_test_context()
        ctx_imb1 = make_test_context()
        ctx_imb1.eval_weights[EW_IMBALANCE_DIVISOR] = np.int32(5)
        self.assertNotEqual(
            int(_evaluate_position_jit(p_imb, o_imb, g_imb, False, ctx_imb0)[0]),
            int(_evaluate_position_jit(p_imb, o_imb, g_imb, False, ctx_imb1)[0]),
        )

        # Unstoppable pawn bonus in multi-pawn KP (2 white pawns -> sf = 36 + 7*2 = 50; 100*50//64 = 78)
        p_up, o_up, g_up = parse_fen("7k/8/8/P7/8/8/6P1/K7 w - - 0 1")
        ctx_up0 = make_test_context()
        ctx_up1 = make_test_context()
        ctx_up1.eval_weights[EW_UNSTOPPABLE_PAWN_BONUS] += np.int32(100)
        self.assertEqual(
            int(_evaluate_position_jit(p_up, o_up, g_up, False, ctx_up1)[0])
            - int(_evaluate_position_jit(p_up, o_up, g_up, False, ctx_up0)[0]),
            78,
        )

    def test_param_config_blocks_use_live_constants_within_bounds(self):
        live_all = params_from_weights(default_eval_weights())
        self.assertEqual(set(ALL_PARAM_SPECS.keys()), set(EVAL_PARAM_INDEX.keys()))
        try:
            for bname in BLOCKS:
                set_active_block(bname)
                dp = default_params()
                for k, v in dp.items():
                    self.assertEqual(int(v), live_all[k], msg=f"{bname}.{k} not live")
                    lo, hi = bounds(k)
                    self.assertGreaterEqual(v, lo, msg=f"{bname}.{k}={v} < lo={lo}")
                    self.assertLessEqual(v, hi, msg=f"{bname}.{k}={v} > hi={hi}")
            select_custom_params(["THREAT_SAFE_PAWN_MG", "SPACE_THRESHOLD"])
            self.assertEqual(list(default_params().keys()), ["THREAT_SAFE_PAWN_MG", "SPACE_THRESHOLD"])
            diff_text = format_constants_diff({"THREAT_SAFE_PAWN_MG": live_all["THREAT_SAFE_PAWN_MG"] + 5})
            self.assertIn("THREAT_SAFE_PAWN_MG:", diff_text)
        finally:
            set_active_block("threats_mg")


if __name__ == "__main__":
    unittest.main()
