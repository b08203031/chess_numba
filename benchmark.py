
import time
import numpy as np
import shutil
from pathlib import Path

# --- Transposition Table Setup ---
from chess_engine.transposition_table import TT_SIZE_MB, create_transposition_table, clear_transposition_table
transposition_table = create_transposition_table(TT_SIZE_MB)

from chess_engine.debug_utils import log_info

def clear_numba_cache():
    """
    Finds and removes all __pycache__ directories in the project.
    """
    project_root = Path(__file__).parent
    cache_dirs = list(project_root.rglob("__pycache__"))

    for cache_dir in cache_dirs:
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir)

# Clear cache before importing the engine to avoid stale cache issues
clear_numba_cache()

from chess_engine.fen_parser import parse_fen
from chess_engine.search import iterative_deepening_search
from chess_engine.move import move_to_uci

def run_benchmark_for_fen(fen, depth, name):
    """
    Runs a search benchmark for a given FEN string to a fixed depth.
    """
    time_limit_ms = 0 # No time limit, run to completion for the given depth

    log_info(f"--- Starting Benchmark for: {name} ---")
    log_info(f"FEN: {fen}")
    log_info(f"Depth: {depth}")
    log_info("----------------------------------------")

    piece_bbs, occupancy_bbs, game_state = parse_fen(fen)

    # Clear TT before search
    clear_transposition_table(transposition_table)

    start_time = time.time()

    (best_move, best_eval, nodes_searched, quiescence_nodes, cutoffs, tt_hits,
     last_completed_depth, total_nmc, total_fp, total_ru, total_rfp, total_lmp, total_pcp, total_qdp, total_qsp,
     total_iid, total_se) = iterative_deepening_search(
        piece_bbs, occupancy_bbs, game_state, depth, time_limit_ms, transposition_table
    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    total_nodes = nodes_searched + quiescence_nodes
    nps = int(total_nodes / elapsed_time) if elapsed_time > 0 else 0

    log_info(f"--- Benchmark Results for: {name} ---")
    log_info(f"Search finished in {elapsed_time:.4f} seconds.")
    log_info(f"Total Nodes Searched: {total_nodes}")
    log_info(f"Nodes Per Second (NPS): {nps}")
    log_info(f"Best move: {move_to_uci(best_move)}")
    log_info(f"Evaluation: {best_eval}")
    log_info("----------------------------------------\n")

def run_all_benchmarks():
    """Runs all benchmark tests."""

    # Benchmark 1: Standard starting position
    startpos_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    run_benchmark_for_fen(startpos_fen, depth=14, name="Start Position")

    # Benchmark 2: Kiwipete position
    kiwipete_fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    run_benchmark_for_fen(kiwipete_fen, depth=12, name="Kiwipete")


if __name__ == "__main__":
    run_all_benchmarks()
