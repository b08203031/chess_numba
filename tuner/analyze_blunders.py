import sys
import os
import json
import chess
import numpy as np
import numba

# Ensure we can import the engine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'chess_engine_v2')))
from chess_engine.nnue.ml_eval.inference import init_accumulator, nnue_forward_incremental, weights_loaded

if not weights_loaded:
    print("NNUE weights not loaded. Exiting.")
    sys.exit(1)

@numba.njit(numba.int32(numba.types.Array(numba.uint64, 1, 'C'), numba.int32), fastmath=True, cache=True)
def evaluate_position_with_stm(piece_bbs, stm):
    # Calculate piece_count for LayerStack bucket selection
    piece_count = 0
    for i in range(12):
        bb = piece_bbs[i]
        while bb:
            piece_count += 1
            bb &= bb - np.uint64(1)

    accumulator_stack = np.zeros((1, 2, 512), dtype=np.int32)
    init_accumulator(piece_bbs, accumulator_stack)
    logit = nnue_forward_incremental(0, stm, accumulator_stack, piece_count) 
    # Use the same conversion as evaluation_nn.py
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

def format_score(score):
    return f"{score/100:+.2f}"

def run_analysis():
    print("=========================================")
    print("=== NNUE Blind-Spot Static Analysis   ===")
    print("=== Analyzing failed_puzzles.json     ===")
    print("=========================================\n")
    
    json_path = os.path.join(os.path.dirname(__file__), '..', 'failed_puzzles.json')
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    try:
        with open(json_path, 'r') as f:
            puzzles = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return

    for p in puzzles:
        print(f"Name: {p['name']}")
        print(f"FEN:  {p['fen']}")
        board = chess.Board(p['fen'])
        stm_color = board.turn
        player_str = "White" if stm_color == chess.WHITE else "Black"
        
        # Get correct move(s)
        solution = p['solution']
        if isinstance(solution, list):
            correct_move_uci = solution[0]
        else:
            correct_move_uci = solution
            
        engine_move_uci = p['engine_move']
        
        # Correct move evaluation
        try:
            board_correct = board.copy()
            board_correct.push_uci(correct_move_uci)
            next_stm_int = 0 if board_correct.turn == chess.WHITE else 1
            bbs_correct = board_to_bbs(board_correct)
            correct_score = -evaluate_position_with_stm(bbs_correct, next_stm_int)
        except Exception as e:
            print(f"Error evaluating correct move {correct_move_uci}: {e}")
            continue
            
        # Engine move evaluation
        try:
            board_engine = board.copy()
            board_engine.push_uci(engine_move_uci)
            next_stm_int_eng = 0 if board_engine.turn == chess.WHITE else 1
            bbs_engine = board_to_bbs(board_engine)
            engine_score = -evaluate_position_with_stm(bbs_engine, next_stm_int_eng)
        except Exception as e:
            print(f"Error evaluating engine move {engine_move_uci}: {e}")
            continue

        print(f"Player: {player_str}")
        print(f"Correct Move ({correct_move_uci}): {format_score(correct_score)}")
        print(f"Engine Move  ({engine_move_uci}): {format_score(engine_score)}")
        
        diff = correct_score - engine_score
        print(f"Static Eval Diff: {format_score(diff)}")
        
        if engine_score > correct_score:
            print("=> 🔴 BLIND SPOT: NNUE static evaluation prefers the wrong move.")
            print("   Suggestion: The training data might lack this tactical pattern or quantization noise is too high.")
        else:
            print("=> 🟢 SEARCH ISSUE: NNUE static evaluation correctly prefers the solution.")
            print("   Suggestion: The move was pruned or the search depth was insufficient.")
            
        print("-" * 60)

if __name__ == '__main__':
    run_analysis()
