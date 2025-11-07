
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
from chess_engine.search import iterative_deepening_search
from chess_engine.move import move_to_uci
from chess_engine.core import SQUARE_TO_ALGEBRAIC

# Maximum search depth (Ply) for arrays like killer moves
MAX_PLY = 64

def run_search_test():
    """
    Runs a search test from the initial position and prints statistics.
    """
    fen = "2kr2r1/1bp4n/1pq1p2p/p1P5/1P3B2/P6P/5RP1/RB3QK1 b - - 4 26"
    depth = 20
    time_limit_ms = 10000

    print("--- Starting Iterative Deepening Search Test ---")
    print(f"FEN: {fen}")
    print(f"Max Depth: {depth}")
    print(f"Time Limit: {time_limit_ms / 1000}s")
    print("----------------------------------------")

    piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
    board_state_flat = piece_bbs + occupancy_bbs + game_state
    
    # Initialize Killer Moves table
    killer_moves = np.zeros((MAX_PLY, 2), dtype=np.uint16)
    
    # Clear TT before each search
    clear_transposition_table(transposition_table)

    start_time = time.time()
    
    best_move, best_eval, nodes_searched, quiescence_nodes, cutoffs = iterative_deepening_search(
        board_state_flat, depth, time_limit_ms, transposition_table, killer_moves
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    total_entries = len(transposition_table)
    used_entries = np.count_nonzero(transposition_table['key'])
    usage_percentage = (used_entries / total_entries * 100) if total_entries > 0 else 0

    total_nodes = nodes_searched + quiescence_nodes
    nps = int(total_nodes / elapsed_time) if elapsed_time > 0 else 0
    q_node_percentage = (quiescence_nodes / total_nodes * 100) if total_nodes > 0 else 0

    print("----------------------------------------")
    print(f"Search finished in {elapsed_time:.4f} seconds.")
    print(f"Final best move: {move_to_uci(best_move)}")
    print(f"Final evaluation: {best_eval}")
    print(f"TT usage: {used_entries} / {total_entries} ({usage_percentage:.2f}%)")
    print("--- Statistics ---")
    print(f"1. Nodes Searched:   {total_nodes}")
    print(f"2. Quiescence Nodes: {quiescence_nodes} ({q_node_percentage:.2f}%)")
    print(f"3. Cutoffs:          {cutoffs}")
    print(f"4. NPS (Nodes/Sec):  {nps}")
    print("----------------------------------------")

if __name__ == "__main__":
    run_search_test()
