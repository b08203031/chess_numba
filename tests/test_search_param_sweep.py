import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import numpy as np
import shutil
import io
import chess.pgn
from pathlib import Path

from chess_engine.classical.transposition_table import (
    TT_SIZE_MB, create_transposition_table, clear_transposition_table
)
from chess_engine.classical.engine_types import SearchContext
from chess_engine.classical.constants import (
    TUNE_FP_MULT, TUNE_RFP_MULT, TUNE_LMR_BASE_OFFSET,
    FP_MULTIPLIER, RFP_BASE_MULT, LMR_BASE_OFFSET
)
from chess_engine.classical.debug_utils import log_info

transposition_table = create_transposition_table(TT_SIZE_MB)

def make_search_context(tt):
    MAX_PLY = 128
    killer_moves = np.zeros(MAX_PLY * 2, dtype=np.uint16)
    pv_table = np.zeros((MAX_PLY, MAX_PLY), dtype=np.uint16)
    history_table = np.zeros((12, 64), dtype=np.int32)
    butterfly_history = np.zeros((64, 64), dtype=np.int32)
    continuation_history = np.zeros((5, 12, 64, 12, 64), dtype=np.int16)
    capture_history = np.zeros((12, 64, 6), dtype=np.int32)
    pawn_history = np.full((8192, 12, 64), -1238, dtype=np.int16)
    pawn_correction_history = np.zeros(16384, dtype=np.int16)
    minor_correction_history = np.zeros(16384, dtype=np.int16)
    non_pawn_correction_history_white = np.zeros(16384, dtype=np.int16)
    non_pawn_correction_history_black = np.zeros(16384, dtype=np.int16)
    continuation_correction_history = np.zeros((12, 64, 12, 64), dtype=np.int16)

    return SearchContext(
        tt, killer_moves, pv_table, history_table,
        butterfly_history, continuation_history, capture_history, pawn_history,
        pawn_correction_history, minor_correction_history,
        non_pawn_correction_history_white, non_pawn_correction_history_black,
        continuation_correction_history
    )

def clear_numba_cache():
    project_root = Path(__file__).parent
    cache_dirs = list(project_root.rglob("__pycache__"))
    for cache_dir in cache_dirs:
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir)

clear_numba_cache()

from chess_engine.classical.fen_parser import parse_fen
from chess_engine.classical.search import iterative_deepening_search
from chess_engine.classical.move import move_to_uci
from tests.test_search import GAME_PGN, extract_benchmark_positions

def run_param_sweep(param_name, tune_index, candidate_values):
    """
    Runs the 162-position search benchmark for each value in candidate_values.
    """
    positions = extract_benchmark_positions()
    
    # 1. Warm-up JIT compilation
    print("--- WARMING UP JIT COMPILATION ---")
    kiwipete_fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    p_bbs, o_bbs, g_state = parse_fen(kiwipete_fen)
    clear_transposition_table(transposition_table)
    ctx = make_search_context(transposition_table)
    iterative_deepening_search(p_bbs, o_bbs, g_state, 12, {'optimum_time': 300000, 'maximum_time': 300000}, ctx, verbose=False)
    print("--- WARM-UP COMPLETE ---\n")

    print(f"==========================================================================")
    print(f"  SEARCH PARAMETER SWEEP: {param_name}")
    print(f"  Candidate values (5 terms arithmetic progression): {candidate_values}")
    print(f"  Positions: {len(positions)} (15. Be2 to 95... Kf2) | Time limit: 1.0s/pos")
    print(f"==========================================================================\n")

    summary_results = []
    time_config = {'optimum_time': 1000, 'maximum_time': 1000}

    for val in candidate_values:
        print(f"\n>>> Running Benchmark for {param_name} = {val} <<<")
        nodes_list = []
        nps_list = []
        depth_list = []

        start_val_time = time.time()
        for step_idx, (ply, move_str, fen) in enumerate(positions, 1):
            piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
            clear_transposition_table(transposition_table)
            search_context = make_search_context(transposition_table)
            
            # Apply tune parameter override
            search_context.tune[tune_index] = int(val)

            start_pos_time = time.time()
            (best_move, best_eval, nodes_searched, quiescence_nodes, tt_hits, last_completed_depth) = iterative_deepening_search(
                piece_bbs, occupancy_bbs, game_state, 64, time_config, search_context, verbose=False
            )
            elapsed_time = time.time() - start_pos_time

            total_nodes = nodes_searched + quiescence_nodes
            nps = int(total_nodes / elapsed_time) if elapsed_time > 0 else 0

            nodes_list.append(total_nodes)
            nps_list.append(nps)
            depth_list.append(last_completed_depth)

        val_elapsed = time.time() - start_val_time
        mean_nodes = float(np.mean(nodes_list))
        median_nodes = float(np.median(nodes_list))
        mean_nps = float(np.mean(nps_list))
        median_nps = float(np.median(nps_list))
        mean_depth = float(np.mean(depth_list))
        median_depth = float(np.median(depth_list))

        summary_results.append({
            "param_value": val,
            "mean_nodes": mean_nodes,
            "median_nodes": median_nodes,
            "mean_nps": mean_nps,
            "median_nps": median_nps,
            "mean_depth": mean_depth,
            "median_depth": median_depth,
            "total_time": val_elapsed
        })

        print(f"Done {param_name}={val} in {val_elapsed:.1f}s | Mean Depth: {mean_depth:.2f} | Mean Nodes: {mean_nodes:.1f} | Mean NPS: {mean_nps:.1f}")

    print("\n" + "=" * 95)
    print(f"                           FINAL SWEEP COMPARISON: {param_name}                         ")
    print("=" * 95)
    header = f"{'Value':<8} | {'Mean Nodes':<12} | {'Median Nodes':<12} | {'Mean NPS':<12} | {'Median NPS':<12} | {'Mean Depth':<10} | {'Median Depth':<12}"
    print(header)
    print("-" * len(header))
    for res in summary_results:
        is_baseline = " (Base)" if res["param_value"] == candidate_values[2] else ""
        val_str = f"{res['param_value']}{is_baseline}"
        print(
            f"{val_str:<8} | {res['mean_nodes']:12.1f} | {res['median_nodes']:12.1f} | "
            f"{res['mean_nps']:12.1f} | {res['median_nps']:12.1f} | "
            f"{res['mean_depth']:10.2f} | {res['median_depth']:12.2f}"
        )
    print("=" * 95)

if __name__ == "__main__":
    # 1. FP_MULTIPLIER sweep centered at 125 (step 10)
    fp_mult_values = [105, 115, 125, 135, 145]
    run_param_sweep("FP_MULTIPLIER", TUNE_FP_MULT, fp_mult_values)

    # 2. RFP_BASE_MULT sweep centered at 170 (step 10)
    rfp_mult_values = [150, 160, 170, 180, 190]
    run_param_sweep("RFP_BASE_MULT", TUNE_RFP_MULT, rfp_mult_values)

    # 3. LMR_BASE_OFFSET sweep centered at 460 (step 20)
    lmr_offset_values = [420, 440, 460, 480, 500]
    run_param_sweep("LMR_BASE_OFFSET", TUNE_LMR_BASE_OFFSET, lmr_offset_values)
