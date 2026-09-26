import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import time
import numpy as np
import shutil
from pathlib import Path

from chess_engine.classical.engine_types import create_search_context
from chess_engine.classical.transposition_table import TT_SIZE_MB, create_transposition_table, clear_transposition_table

transposition_table = create_transposition_table(TT_SIZE_MB)
search_context, _hist_tables = create_search_context(transposition_table)

from chess_engine.classical.debug_utils import log_info

def clear_numba_cache():
    """
    Finds and removes all __pycache__ directories in the project.
    ?¥æ¾ä¸¦åª?¤é??®ä¸­?æ???__pycache__ ?®é???
    """
    project_root = Path(__file__).parent
    cache_dirs = list(project_root.rglob("__pycache__"))

    for cache_dir in cache_dirs:
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir)

# clear_numba_cache()

from chess_engine.classical.fen_parser import parse_fen
from chess_engine.classical.search import iterative_deepening_search
from chess_engine.classical.move import move_to_uci

def run_benchmark_for_fen(fen, depth, name):
    """
    Runs a search benchmark for a given FEN string to a fixed depth.
    ?ºçµ¦å®ç? FEN å­ä¸²?è??°åºå®æ·±åº¦ç??å??ºæ?æ¸¬è©¦??
    """
    # No time limit, run to completion for the given depth / ?¡æ??é??¶ï??è??°å??çµ¦å®æ·±åº?
    time_config = {'optimum_time': 0, 'maximum_time': 0}

    log_info(f"--- Starting Benchmark for: {name} ---")
    log_info(f"FEN: {fen}")
    log_info(f"Depth: {depth}")
    log_info("----------------------------------------")

    piece_bbs, occupancy_bbs, game_state = parse_fen(fen)

    # Clear TT before search / ?å??æ??¤ç½®?è¡¨
    clear_transposition_table(transposition_table)
    
    # Reset search context stats / ?ç½®?å?ä¸ä??çµ±è¨æ¸??
    search_context.nodes_searched = np.uint64(0)
    search_context.killer_moves.fill(0)
    search_context.history_table.fill(0)
    search_context.main_history.fill(0)
    search_context.pv_table.fill(0)

    start_time = time.time()

    (best_move, best_eval, nodes_searched, quiescence_nodes, tt_hits, last_completed_depth) = iterative_deepening_search(
        piece_bbs, occupancy_bbs, game_state, depth, time_config, search_context
    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    total_nodes = nodes_searched + quiescence_nodes
    nps = int(total_nodes / elapsed_time) if elapsed_time > 0 else 0

    log_info(f"--- Benchmark Results for: {name} ---")
    log_info(f"Search finished in {elapsed_time:.4f} seconds.")
    log_info(f"Total Nodes Searched: {total_nodes}")
    log_info(f"Nodes Per Second (NPS): {nps}")
    log_info(f"Quiescence Nodes: {quiescence_nodes}")
    log_info(f"Best move: {move_to_uci(best_move)}")
    log_info(f"Evaluation: {best_eval}")
    log_info("----------------------------------------\n")

def run_all_benchmarks():
    """Runs all benchmark tests."""

    # Warm up: Standard starting position
    startpos_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    run_benchmark_for_fen(startpos_fen, depth=2, name="Start Position") # Reduced depth for faster verification

    # Benchmark 1: Standard starting position
    startpos_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    run_benchmark_for_fen(startpos_fen, depth=15, name="Start Position")

    # Benchmark 2: Kiwipete position
    kiwipete_fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    run_benchmark_for_fen(kiwipete_fen, depth=15, name="Kiwipete")


if __name__ == "__main__":
    run_all_benchmarks()
