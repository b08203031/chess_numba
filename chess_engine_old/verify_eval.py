import sys
import numpy as np
# Mocking parts of chess_engine if needed or just import directly
from chess_engine.fen_parser import parse_fen
from chess_engine.evaluation import evaluate_position
from test_puzzle import puzzles

def run_static_eval_test():
    """
    Evaluates all puzzles and prints the FEN and static evaluation score.
    """
    print(f"Number of puzzles: {len(puzzles)}")
    
    for i, puzzle in enumerate(puzzles):
        fen = puzzle['fen']
        piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
        
        # We perform static evaluation
        score = evaluate_position(piece_bbs, occupancy_bbs, game_state, False)
        
        print(f"Puzzle {i}: {score}")

if __name__ == "__main__":
    # Add parent dir to path to find chess_engine if running from root
    import os
    sys.path.append(os.getcwd())
    
    try:
        run_static_eval_test()
    except Exception as e:
        print(f"Error: {e}")
