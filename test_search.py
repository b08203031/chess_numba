
import time
import numpy as np
import shutil
from pathlib import Path

def clear_numba_cache():
    """
    Finds and removes all __pycache__ directories in the project.
    """
    print("--- Clearing Numba Cache ---")
    project_root = Path(__file__).parent
    cache_dirs = list(project_root.rglob("__pycache__"))
    
    for cache_dir in cache_dirs:
        if cache_dir.is_dir():
            print(f"Removing cache directory: {cache_dir}")
            shutil.rmtree(cache_dir)
    print("--- Cache Cleared ---")

# Clear cache before importing the engine to avoid stale cache issues
clear_numba_cache()

from chess_engine.fen_parser import parse_fen
from chess_engine.search import search_position
from chess_engine.move import get_from_square, get_to_square
from chess_engine.core import SQUARE_TO_ALGEBRAIC

def run_search_test():
    """
    Runs a search test from the initial position and prints statistics.
    """
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    depth = 7

    print("--- Starting Search Performance Test ---")
    print(f"FEN: {fen}")
    print(f"Depth: {depth}")
    print("----------------------------------------")

    piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
    board_state_flat = piece_bbs + occupancy_bbs + game_state
    
    start_time = time.time()
    
    best_move, best_eval, nodes_searched, cutoffs = search_position(board_state_flat, depth)
    
    end_time = time.time()

    elapsed_time = end_time - start_time
    nps = int(nodes_searched / elapsed_time) if elapsed_time > 0 else 0
    
    from_sq_alg = SQUARE_TO_ALGEBRAIC[get_from_square(best_move)]
    to_sq_alg = SQUARE_TO_ALGEBRAIC[get_to_square(best_move)]


    print(f"Search complete.")
    print(f"Best move found: {from_sq_alg}{to_sq_alg} (raw: {best_move}), Evaluation: {best_eval}")
    print("--- Statistics ---")
    print(f"1. Nodes Searched: {nodes_searched}")
    print(f"2. Cutoffs:          {cutoffs}")
    print(f"3. Time Spent:       {elapsed_time:.4f} seconds")
    print(f"4. NPS (Nodes/Sec):  {nps}")
    print("----------------------------------------")

if __name__ == "__main__":
    run_search_test()
