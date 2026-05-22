import time
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chess_engine.fen_parser import parse_fen
from chess_engine.evaluation import evaluate_position
from chess_engine.constants import (
    BISHOP_PAWNS_PENALTY,
    TRAPPED_ROOK,
    THREAT_RESTRICTED_PIECE,
    THREAT_BY_MINOR,
    QUEEN_MOBILITY_BONUS,
    KNIGHT_MOBILITY_BONUS,
    BISHOP_MOBILITY_BONUS,
    ROOK_MOBILITY_BONUS,
    THREAT_SAFE_PAWN,
    THREAT_HANGING,
    PAWN_SHIELD_MISSING_PENALTY,
    KING_SEMI_OPEN_FILE_PENALTY,
    KING_OPEN_FILE_PENALTY
)

def evaluate_fen(fen):
    piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
    return evaluate_position(piece_bbs, occupancy_bbs, game_state, False)

def run_test(name, expected_cond, msg_success, msg_fail, actual_diff=None):
    if expected_cond:
        print(f"[PASS] {name}: {msg_success}")
        return True
    else:
        diff_str = f" (Actual diff: {actual_diff})" if actual_diff is not None else ""
        print(f"[FAIL] {name}: {msg_fail}{diff_str}")
        return False

# ==========================================
# 1. Existing Tests (BishopPawns, TrappedRook)
# ==========================================

def test_bishop_pawns_color():
    print("\n--- Test BishopPawns ---")
    score1_mg = evaluate_fen("4k3/8/8/8/8/8/1P1P4/2B1K3 w - - 0 1") # Dark pawns (b2, d2) for c1 Dark bishop
    score2_mg = evaluate_fen("4k3/8/8/8/8/8/P1P5/2B1K3 w - - 0 1") # Light pawns (a2, c2)
    
    diff_mg = score2_mg - score1_mg # Light pawns should be better (no penalty)
    run_test("White BishopPawns", diff_mg > 0, f"Diff {diff_mg} > 0", "Penalty not applied", diff_mg)

def test_trapped_rook():
    print("\n--- Test TrappedRook ---")
    score1_mg = evaluate_fen("4k3/8/8/8/8/8/5PPP/4K2R w - - 0 1") # Trapped
    score2_mg = evaluate_fen("4k3/8/8/8/8/8/5PPP/3RK3 w - - 0 1") # Open (Moved Rook to d1)
    
    diff_mg = score2_mg - score1_mg
    run_test("White TrappedRook", diff_mg > 0, f"Diff {diff_mg} > 0", "Penalty insufficient", diff_mg)

# ==========================================
# 2. Mobility Tests
# ==========================================

def test_mobility():
    print("\n--- Test Mobility (N, B, R, Q) ---")
    
    # KNIGHT
    score_corner_n = evaluate_fen("4k3/8/8/8/8/8/8/N3K3 w - - 0 1") # Knight in corner (2 moves)
    score_center_n = evaluate_fen("4k3/8/8/8/3N4/8/8/4K3 w - - 0 1") # Knight in center (8 moves)
    diff_n = score_center_n - score_corner_n
    run_test("Knight Mobility", diff_n > 0, f"Center vs Corner diff: {diff_n}", "No mobility diff", diff_n)

    # QUEEN
    score_corner_q = evaluate_fen("4k3/8/8/8/8/8/8/Q3K3 w - - 0 1") 
    score_center_q = evaluate_fen("4k3/8/8/8/3Q4/8/8/4K3 w - - 0 1") 
    diff_q = score_center_q - score_corner_q
    run_test("Queen Mobility", diff_q > 50, f"Center vs Corner diff: {diff_q}", "Insufficient mobility diff", diff_q)

    # BISHOP
    score_corner_b = evaluate_fen("4k3/8/8/8/8/8/8/B3K3 w - - 0 1")
    score_center_b = evaluate_fen("4k3/8/8/8/3B4/8/8/4K3 w - - 0 1")
    diff_b = score_center_b - score_corner_b
    run_test("Bishop Mobility", diff_b > 20, f"Center vs Corner diff: {diff_b}", "Insufficient mobility diff", diff_b)

# ==========================================
# 3. Threats Tests
# ==========================================

def test_threats():
    print("\n--- Test Threats ---")
    
    # Safe Pawn Threat (White pawn attacking undefended Black Knight)
    score_safe_pawn = evaluate_fen("4k3/8/8/4n3/3P4/8/8/4K3 w - - 0 1") # d4 attacks e5
    score_no_threat = evaluate_fen("4k3/8/8/4n3/8/8/3P4/4K3 w - - 0 1") # d2 doesn't attack
    
    diff_safe_pawn = score_safe_pawn - score_no_threat
    run_test("Safe Pawn Threat", diff_safe_pawn > 0, f"Diff w/ threat: {diff_safe_pawn}", "No threat bonus", diff_safe_pawn)

    # Minor attacking weak piece (Knight attacks unprotected Rook)
    score_minor_t = evaluate_fen("4k3/8/4r3/3N4/8/8/8/4K3 w - - 0 1") # d5 attacks e6
    score_minor_nt = evaluate_fen("4k3/8/4r3/8/8/8/3N4/4K3 w - - 0 1") # d2 doesn't attack
    diff_minor = score_minor_t - score_minor_nt
    run_test("Minor Threat", diff_minor > 0, f"Diff w/ threat: {diff_minor}", "No threat bonus", diff_minor)

    # Hanging Piece (Black knight hanging to White Queen)
    score_hanging = evaluate_fen("4k3/8/4n3/8/3Q4/8/8/4K3 w - - 0 1") # d4 attacks e6
    score_no_attack = evaluate_fen("4k3/8/4n3/8/8/8/8/Q3K3 w - - 0 1") # a1 doesn't attack
    diff_hanging = score_hanging - score_no_attack
    run_test("Hanging Piece Threat", diff_hanging > 0, f"Bonus for finding hanging: {diff_hanging}", "No hanging bonus", diff_hanging)

# ==========================================
# 4. King Safety Tests
# ==========================================

def test_king_safety():
    print("\n--- Test King Safety ---")
    
    # Baseline: King castled kingside with full pawn shield
    score_safe_king = evaluate_fen("4k3/8/8/8/8/8/5PPP/5RK1 w - - 0 1")
    
    # Missing pawn shield (g2 missing) -> material loss!
    # Instead of missing, we PUSH the pawn forward to g4!
    score_pushed_shield = evaluate_fen("4k3/8/8/8/6P1/8/5P1P/5RK1 w - - 0 1")
    diff_shield = score_safe_king - score_pushed_shield
    run_test("Pawn Shield Penalty", diff_shield > 0, f"Penalty applied (diff: {diff_shield})", "No shield penalty", diff_shield)

    # Open file near King vs Closed
    # B1: Closed (pawns on f2, g2, h2)
    # B2: Open f-file (move f pawn to a file far away so it doesn't leave material)
    # B1 = safe_king
    score_open_file = evaluate_fen("4k3/8/8/8/8/8/P5PP/5RK1 w - - 0 1") # f2 pawn moved to a2!
    diff_open_file = score_safe_king - score_open_file
    run_test("Open File near King", diff_open_file > 0, f"Penalty applied (diff: {diff_open_file})", "No open file penalty", diff_open_file)
    
    # Attack Units: Enemy pieces near our king (Black Queen and Knight attacking kingside)
    # vs Enemy pieces far away
    score_attacked_king = evaluate_fen("4k3/8/7q/6n1/8/8/5PPP/5RK1 w - - 0 1") # g5 n, h6 q
    score_safe_from_attack = evaluate_fen("q3k3/8/8/8/8/8/5PPP/5RK1 w - - 0 1") # q at a8
    diff_attack = score_safe_from_attack - score_attacked_king
    run_test("King Danger Penalty", diff_attack > 10, f"Danger penalty applied (diff: {diff_attack})", "Danger penalty insufficient", diff_attack)


if __name__ == "__main__":
    print("====================================")
    print("   EVALUATION ENGINE TEST SUITE     ")
    print("====================================")
    
    # Warmup
    evaluate_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    
    test_bishop_pawns_color()
    test_trapped_rook()
    test_mobility()
    test_threats()
    test_king_safety()
    
    print("\nFinished running all tests.")
