
import time
import numpy as np
import shutil
from pathlib import Path

from chess_engine.engine_types import SearchContext
from chess_engine.transposition_table import TT_SIZE_MB, create_transposition_table, clear_transposition_table

# --- Transposition Table Setup / 置換表設置 ---
transposition_table = create_transposition_table(TT_SIZE_MB)
# Create SearchContext needed for search / 創建搜尋所需的上下文
killer_moves = np.zeros(128, dtype=np.uint16) # Adjusted for potential array size mismatch if not using constant
# Wait, killer moves is defined as MAX_PLY * 2 in main.py. MAX_PLY is 128 in search.py but 64 in main.py.
# Let's use the constant from constants.py
from chess_engine.constants import MAX_PLY
killer_moves = np.zeros(MAX_PLY * 2, dtype=np.uint16) # MAX_PLY is 128
history_table = np.zeros((12, 64), dtype=np.int32)
pv_table = np.zeros((MAX_PLY, MAX_PLY), dtype=np.uint16)

search_context = SearchContext(transposition_table, killer_moves, pv_table, history_table)

from chess_engine.debug_utils import log_info

def clear_numba_cache():
    """
    Finds and removes all __pycache__ directories in the project.
    查找並刪除項目中的所有 __pycache__ 目錄。
    """
    project_root = Path(__file__).parent
    cache_dirs = list(project_root.rglob("__pycache__"))

    for cache_dir in cache_dirs:
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir)

# Clear cache before importing the engine to avoid stale cache issues / 在導入引擎之前清除緩存以避免過時的緩存問題
clear_numba_cache()

from chess_engine.fen_parser import parse_fen
from chess_engine.search import iterative_deepening_search
from chess_engine.move import move_to_uci

def run_benchmark_for_fen(fen, depth, name):
    """
    Runs a search benchmark for a given FEN string to a fixed depth.
    為給定的 FEN 字串運行到固定深度的搜尋基準測試。
    """
    # No time limit, run to completion for the given depth / 無時間限制，運行到完成給定深度
    time_config = {'optimum_time': 0, 'maximum_time': 0}

    log_info(f"--- Starting Benchmark for: {name} ---")
    log_info(f"FEN: {fen}")
    log_info(f"Depth: {depth}")
    log_info("----------------------------------------")

    piece_bbs, occupancy_bbs, game_state = parse_fen(fen)

    # Clear TT before search / 搜尋前清除置換表
    clear_transposition_table(transposition_table)
    
    # Reset search context stats / 重置搜尋上下文統計數據
    search_context.nodes_searched = np.uint64(0)
    search_context.killer_moves.fill(0)
    search_context.history_table.fill(0)
    search_context.pv_table.fill(0)

    start_time = time.time()

    (best_move, best_eval, nodes_searched, quiescence_nodes, cutoffs, tt_hits,
     last_completed_depth, total_nmc, total_fp, total_ru, total_rfp, total_lmp, total_pcp, total_qdp, total_qsp,
     total_iid, total_se) = iterative_deepening_search(
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
    log_info(f"Best move: {move_to_uci(best_move)}")
    log_info(f"Evaluation: {best_eval}")
    log_info("----------------------------------------\n")

def run_all_benchmarks():
    """Runs all benchmark tests. / 運行所有基準測試。"""

    # Benchmark 1: Standard starting position
    startpos_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    run_benchmark_for_fen(startpos_fen, depth=12, name="Start Position") # Reduced depth for faster verification

    # Benchmark 2: Kiwipete position
    kiwipete_fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    run_benchmark_for_fen(kiwipete_fen, depth=10, name="Kiwipete")


if __name__ == "__main__":
    run_all_benchmarks()
