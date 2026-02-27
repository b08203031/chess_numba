
import time
import numpy as np
import shutil
from pathlib import Path

# --- Transposition Table Setup ---
from chess_engine.transposition_table import TT_SIZE_MB, create_transposition_table, clear_transposition_table
from chess_engine.engine_types import SearchContext
transposition_table = create_transposition_table(TT_SIZE_MB)

from chess_engine.debug_utils import log_info

def clear_numba_cache():
    """
    Finds and removes all __pycache__ directories in the project.
    """
    log_info("--- Clearing Numba Cache ---")
    project_root = Path(__file__).parent
    cache_dirs = list(project_root.rglob("__pycache__"))
    
    for cache_dir in cache_dirs:
        if cache_dir.is_dir():
            log_info(f"Removing cache directory: {cache_dir}")
            shutil.rmtree(cache_dir)
    log_info("--- Cache Cleared ---")

# Clear cache before importing the engine to avoid stale cache issues
clear_numba_cache()

from chess_engine.fen_parser import parse_fen
from chess_engine.search import iterative_deepening_search
from chess_engine.move import move_to_uci

def run_search_test():
    """
    Runs a search test from a given position and prints statistics.
    """

    # 1. Warm-up (triggers JIT compilation)
    print("--- WARMING UP (Compiling JIT functions) ---")
    fen_start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    p_bbs, o_bbs, g_state = parse_fen(fen_start)
    tt = create_transposition_table(16)
    killer_moves = np.zeros(256, dtype=np.uint16)
    history_table = np.zeros((12, 64), dtype=np.int32)
    butterfly_history = np.zeros((64, 64), dtype=np.int32)
    continuation_history = np.zeros((12, 64, 12, 64), dtype=np.int16)
    pv_table = np.zeros((128, 128), dtype=np.uint16)
    ctx = SearchContext(tt, killer_moves, pv_table, history_table, butterfly_history, continuation_history)
    
    # Run a quick search
    iterative_deepening_search(p_bbs, o_bbs, g_state, 2, {'optimum_time': 0, 'maximum_time': 0}, ctx)
    print("--- WARM-UP COMPLETE ---\n")

    # 2. Actual Search Test
    fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - "
    depth = 10
    time_limit = 3000
    time_limit_ms = time_limit * 1000

    log_info("--- Starting Iterative Deepening Search Test ---")
    log_info(f"FEN: {fen}")
    log_info(f"Max Depth: {depth}")
    log_info(f"Time Limit: {time_limit_ms / 1000}s")
    log_info("----------------------------------------")

    piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
    
    # Clear TT before each search
    clear_transposition_table(transposition_table)
    ctx.killer_moves.fill(0)
    ctx.history_table.fill(0)

    start_time = time.time()
    
    # Pass the NumPy arrays directly to the search function
    time_config = {
        'optimum_time': time_limit_ms,
        'maximum_time': time_limit_ms,
    }

    # Initialize search context components
    MAX_PLY = 128
    killer_moves = np.zeros(MAX_PLY * 2, dtype=np.uint16)
    pv_table = np.zeros((MAX_PLY, MAX_PLY), dtype=np.uint16)
    history_table = np.zeros((12, 64), dtype=np.int32)
    butterfly_history = np.zeros((64, 64), dtype=np.int32)
    continuation_history = np.zeros((12, 64, 12, 64), dtype=np.int16)
    
    search_context = SearchContext(
        transposition_table, killer_moves, pv_table, history_table,
        butterfly_history, continuation_history
    )

    (best_move, best_eval, nodes_searched, quiescence_nodes, cutoffs, tt_hits,
     last_completed_depth, total_nmc, total_fp, total_ru, total_rfp, total_lmp, total_pcp, total_qdp, total_qsp,
     total_iid, total_se) = iterative_deepening_search(
        piece_bbs, occupancy_bbs, game_state, depth, time_config, search_context
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    total_entries = len(transposition_table)
    used_entries = np.count_nonzero(transposition_table['key'])
    usage_percentage = (used_entries / total_entries * 100) if total_entries > 0 else 0

    total_nodes = nodes_searched + quiescence_nodes
    nps = int(total_nodes / elapsed_time) if elapsed_time > 0 else 0
    q_node_percentage = (quiescence_nodes / total_nodes * 100) if total_nodes > 0 else 0

    log_info("----------------------------------------")
    log_info(f"Search finished in {elapsed_time:.4f} seconds.")
    log_info(f"Final best move: {move_to_uci(best_move)}")
    log_info(f"Final evaluation: {best_eval}")
    log_info(f"TT usage: {used_entries} / {total_entries} ({usage_percentage:.2f}%)")
    log_info("--- Statistics ---")
    log_info(f"1. Nodes Searched:      {total_nodes}")
    log_info(f"2. Quiescence Nodes:    {quiescence_nodes} ({q_node_percentage:.2f}%)")
    log_info(f"3. Cutoffs:             {cutoffs}")
    log_info(f"4. NPS (Nodes/Sec):     {nps}")
    log_info("--- Pruning Stats ---")
    log_info(f"5. Null Move Cutoffs:   {total_nmc}")
    log_info(f"6.  QS Delta Pruned:   {total_qdp}")
    log_info(f"7.  QS SEE Pruned:   {total_qsp}")  
    log_info(f"8. Futility Pruned:     {total_fp}")
    log_info(f"9. Razoring Pruned:     {total_ru}")
    log_info(f"10. RFP Pruned:          {total_rfp}")
    log_info(f"11. LMP Pruned:          {total_lmp}")
    log_info(f"12. ProbCut Pruned:     {total_pcp}")
    log_info(f"13. IID Searches:       {total_iid}")
    log_info(f"14. Singular Extensions:{total_se}")
    log_info("----------------------------------------")

if __name__ == "__main__":
    run_search_test()
