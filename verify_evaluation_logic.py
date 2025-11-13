# verify_evaluation_logic.py
"""
This script contains a collection of verification tests designed to confirm the
correctness of the individual components within the `evaluation.py` module.

It avoids the unittest framework to prevent potential conflicts with Numba's JIT
compiler, running as a simple, procedural script.

You can run this script directly from your terminal:
python verify_evaluation_logic.py
"""
import numpy as np

from chess_engine.fen_parser import parse_fen
from chess_engine.evaluation import (
    evaluate_pawn_structure,
    evaluate_piece_coordination,
    evaluate_mobility,
    _evaluate_pawn_shield_for_color,
    _evaluate_king_attackers,
    _evaluate_king_tropism,
    _evaluate_pawn_storm
)
from chess_engine.zobrist import get_lsb_index
from chess_engine.constants import *

def verify(test_name, fen, expected_mg, expected_eg, eval_function):
    """Helper to run a tapered evaluation test and assert its correctness."""
    print(f"[TESTING] {test_name}...")
    piece_bbs, occupancy_bbs, _ = parse_fen(fen)

    args = ()
    if eval_function.__name__ in ["evaluate_pawn_structure", "evaluate_piece_coordination"]:
        args = (piece_bbs,)
    elif eval_function.__name__ == "evaluate_mobility":
        args = (piece_bbs, occupancy_bbs)

    mg_score, eg_score = eval_function(*args)

    assert mg_score == expected_mg, f"[FAIL] MG score mismatch for '{test_name}'. FEN: {fen}. Expected {expected_mg}, got {mg_score}"
    assert eg_score == expected_eg, f"[FAIL] EG score mismatch for '{test_name}'. FEN: {fen}. Expected {expected_eg}, got {eg_score}"
    print(f"[ OK ] {test_name}")

def verify_single_value(test_name, fen, expected_val, eval_function):
    """Helper for non-tapered king safety sub-component tests."""
    print(f"[TESTING] {test_name}...")
    p_bbs, o_bbs, _ = parse_fen(fen)
    wk_sq = get_lsb_index(p_bbs[5]) # Assume white king exists

    args = ()
    if eval_function.__name__ == "_evaluate_pawn_shield_for_color":
        args = (wk_sq, p_bbs[0], p_bbs[6], 0)
    elif eval_function.__name__ == "_evaluate_king_attackers":
         args = (wk_sq, 0, p_bbs, o_bbs)
    elif eval_function.__name__ == "_evaluate_king_tropism":
        wk_sq_t = get_lsb_index(p_bbs[5])
        args = (wk_sq_t, 0, p_bbs)
    elif eval_function.__name__ == "_evaluate_pawn_storm":
         args = (wk_sq, 0, p_bbs)

    score = eval_function(*args)
    assert score == expected_val, f"[FAIL] Score mismatch for '{test_name}'. FEN: {fen}. Expected {expected_val}, got {score}"
    print(f"[ OK ] {test_name}")


print("--- Verifying Pawn Structure ---")
fen_doubled = "rnbqkbnr/pppppppp/8/8/3P4/3P4/PPP1PPPP/RNBQKBNR w KQkq - 0 1"
verify("Doubled Pawns", fen_doubled, DOUBLED_PAWN_PENALTY[0], DOUBLED_PAWN_PENALTY[1], evaluate_pawn_structure)
fen_isolated = "8/2ppp3/8/8/3P4/8/8/k7 w - - 0 1"
verify("Isolated Pawn", fen_isolated, ISOLATED_PAWN_PENALTY[0], ISOLATED_PAWN_PENALTY[1], evaluate_pawn_structure)
fen_passed_isolated = "8/8/8/3P4/8/8/8/k7 w - - 0 1"
mg_pi = PASSED_PAWN_BONUS[4][0] + ISOLATED_PAWN_PENALTY[0]
eg_pi = PASSED_PAWN_BONUS[4][1] + ISOLATED_PAWN_PENALTY[1]
verify("Passed & Isolated Pawn", fen_passed_isolated, mg_pi, eg_pi, evaluate_pawn_structure)

print("\\n--- Verifying Piece Coordination ---")
fen_bishop_pair = "k7/8/8/8/8/8/3BB3/K7 w - - 0 1"
verify("Bishop Pair", fen_bishop_pair, BISHOP_PAIR_BONUS[0], BISHOP_PAIR_BONUS[1], evaluate_piece_coordination)
fen_semi_open = "k7/3p4/8/8/8/8/3R4/K7 w - - 0 1"
verify("Rook on Semi-Open File", fen_semi_open, ROOK_ON_SEMI_OPEN_FILE_BONUS[0], ROOK_ON_SEMI_OPEN_FILE_BONUS[1], evaluate_piece_coordination)
fen_open = "k7/8/8/8/8/8/4R3/K7 w - - 0 1"
verify("Rook on Open File", fen_open, ROOK_ON_OPEN_FILE_BONUS[0], ROOK_ON_OPEN_FILE_BONUS[1], evaluate_piece_coordination)

print("\\n--- Verifying Mobility ---")
fen_knight = "k7/8/8/8/3n4/1P6/P1P5/N3K3 b - - 0 1"
w_knight_moves, b_knight_moves = 0, 8
mg_k = ((w_knight_moves-KNIGHT_MOBILITY_BASE_MOVES)*KNIGHT_MOBILITY_WEIGHT[0] - (b_knight_moves-KNIGHT_MOBILITY_BASE_MOVES)*KNIGHT_MOBILITY_WEIGHT[0])
eg_k = ((w_knight_moves-KNIGHT_MOBILITY_BASE_MOVES)*KNIGHT_MOBILITY_WEIGHT[1] - (b_knight_moves-KNIGHT_MOBILITY_BASE_MOVES)*KNIGHT_MOBILITY_WEIGHT[1])
verify("Knight Mobility", fen_knight, mg_k, eg_k, evaluate_mobility)
fen_queen = "k7/8/8/8/3Q4/8/8/K7 w - - 0 1"
w_queen_moves = 26
mg_q = (w_queen_moves - QUEEN_MOBILITY_BASE_MOVES) * QUEEN_MOBILITY_WEIGHT[0]
eg_q = (w_queen_moves - QUEEN_MOBILITY_BASE_MOVES) * QUEEN_MOBILITY_WEIGHT[1]
verify("Queen Mobility", fen_queen, mg_q, eg_q, evaluate_mobility)

print("\\n--- Verifying King Safety Sub-components ---")
fen_castled = "rnbq1rk1/pppppppp/8/8/8/8/PPPPPPPP/RNBQ1RK1 w - - 0 1"
verify_single_value("Pawn Shield", fen_castled, 30, _evaluate_pawn_shield_for_color)
verify_single_value("King Attackers (None)", fen_castled, 0, _evaluate_king_attackers)
fen_tropism = "4k3/8/8/8/3q4/8/8/4K3 w - - 0 1"
verify_single_value("King Tropism", fen_tropism, -50, _evaluate_king_tropism)
fen_pstorm = "rnbq1rk1/ppppppp1/8/8/8/7p/PPPPPPPP/RNBQ1RK1 w - - 0 1"
verify_single_value("Pawn Storm", fen_pstorm, -5, _evaluate_pawn_storm)

print("\\nAll verification tests completed successfully.")
