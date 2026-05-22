import sys
import os
import chess
import numpy as np
import numba

# Ensure we can import the engine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from chess_engine.nnue.ml_eval.inference import init_accumulator, nnue_forward_incremental, weights_loaded

if not weights_loaded:
    print("NNUE weights not loaded. Exiting.")
    sys.exit(1)

@numba.njit(numba.int32(numba.types.Array(numba.uint64, 1, 'C'), numba.int32), fastmath=True, cache=True)
def evaluate_position_with_stm(piece_bbs, stm):
    # Correct size for 512-wide Layer 1
    accumulator_stack = np.zeros((1, 2, 512), dtype=np.int32)
    init_accumulator(piece_bbs, accumulator_stack)
    
    # Calculate piece count for bucket selection (including kings)
    piece_count = 0
    for bb in piece_bbs:
        temp_bb = bb
        while temp_bb:
            piece_count += 1
            temp_bb &= temp_bb - np.uint64(1)
            
    logit = nnue_forward_incremental(0, stm, accumulator_stack, piece_count) 
    return logit

def board_to_bbs(board):
    bbs = np.zeros(12, dtype=np.uint64)
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            p_type = piece.piece_type - 1 # PAWN=0 ... KING=5
            color_offset = 0 if piece.color == chess.WHITE else 6
            bbs[p_type + color_offset] |= np.uint64(1 << sq)
    return bbs

def run_material_test():
    base_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    pieces_to_remove = [
        ("Pawn", "e2"),
        ("Knight", "g1"),
        ("Bishop", "f1"),
        ("Rook", "h1"),
        ("Queen", "d1")
    ]
    
    print("=========================================")
    print("=== NNUE Material Value Calibration   ===")
    print("=========================================\n")
    
    # Baseline (Starting Position)
    board = chess.Board(base_fen)
    bbs = board_to_bbs(board)
    baseline_score = evaluate_position_with_stm(bbs, 0) # White to move
    print(f"Starting Position Score: {baseline_score:+.2f}")
    
    for name, sq_name in pieces_to_remove:
        board = chess.Board(base_fen)
        sq = chess.parse_square(sq_name)
        board.remove_piece_at(sq)
        
        bbs = board_to_bbs(board)
        score = evaluate_position_with_stm(bbs, 0) # White to move
        diff = score - baseline_score
        
        print(f"Missing {name} ({sq_name}): {score:+.2f} (Diff: {diff:+.2f})")

if __name__ == '__main__':
    run_material_test()
