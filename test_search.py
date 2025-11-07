
import time
import numpy as np
import shutil
from pathlib import Path

# --- Transposition Table Setup ---
from chess_engine.transposition_table import TT_SIZE_MB, create_transposition_table, clear_transposition_table
transposition_table = create_transposition_table(TT_SIZE_MB)


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

# Maximum search depth (Ply) for arrays like killer moves
MAX_PLY = 64

def run_search_test():
    """
    Runs a search test from the initial position and prints statistics.
    """
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    depth = 5

    print("--- Starting Search Performance Test ---")
    print(f"FEN: {fen}")
    print(f"Depth: {depth}")
    print("----------------------------------------")

    piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
    board_state_flat = piece_bbs + occupancy_bbs + game_state
    
    # Initialize Killer Moves table
    killer_moves = np.zeros((MAX_PLY, 2), dtype=np.uint16)
    
    # Clear TT before each search
    clear_transposition_table(transposition_table)

    start_time = time.time()
    
    best_move, best_eval, nodes_searched, quiescence_nodes, cutoffs = search_position(board_state_flat, depth, transposition_table, killer_moves)
    
    end_time = time.time()

    elapsed_time = end_time - start_time
    
    total_nodes = nodes_searched + quiescence_nodes
    nps = int(total_nodes / elapsed_time) if elapsed_time > 0 else 0
    
    from_sq_alg = SQUARE_TO_ALGEBRAIC[get_from_square(best_move)]
    to_sq_alg = SQUARE_TO_ALGEBRAIC[get_to_square(best_move)]

    q_node_percentage = (quiescence_nodes / total_nodes * 100) if total_nodes > 0 else 0


    print(f"Search complete.")
    print(f"Best move found: {from_sq_alg}{to_sq_alg} (raw: {best_move}), Evaluation: {best_eval}")
    print("--- Statistics ---")
    print(f"1. Nodes Searched:   {total_nodes}")
    print(f"2. Quiescence Nodes: {quiescence_nodes} ({q_node_percentage:.2f}%)")
    print(f"3. Cutoffs:          {cutoffs}")
    print(f"4. Time Spent:       {elapsed_time:.4f} seconds")
    print(f"5. NPS (Nodes/Sec):  {nps}")
    print("----------------------------------------")

if __name__ == "__main__":
    run_search_test()
