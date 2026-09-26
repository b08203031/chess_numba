import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import numpy as np
import shutil
import io
import chess.pgn
from pathlib import Path

# --- Transposition Table Setup ---
from chess_engine.classical_old.transposition_table import (
    TT_SIZE_MB, create_transposition_table, clear_transposition_table
)
from chess_engine.classical_old.engine_types import SearchContext
transposition_table = create_transposition_table(TT_SIZE_MB)

from chess_engine.classical_old.debug_utils import log_info

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

from chess_engine.classical_old.fen_parser import parse_fen
from chess_engine.classical_old.search import iterative_deepening_search, format_score_for_uci
from chess_engine.classical_old.move import move_to_uci

GAME_PGN = """
1. d4 Nf6 2. c4 e6 3. Nc3 Bb4 4. Qc2 c5 5. dxc5 O-O 6. a3 Bxc5 7. Nf3 b6 8. b4 $6
Be7 $1 9. e4 a5 10. b5 d6 $1 11. Bd3 Bb7 12. O-O Nbd7 13. Bf4 Qc7 14. Rfd1 $6 Rfc8
15. Be2 Nc5 $2 16. e5 Nfe4 17. Nxe4 $2 Bxe4 18. Qd2 dxe5 19. Nxe5 $2 Qb7 $9 20. Bh5 $9
f6 $1 21. Nc6 Rxc6 $3 22. bxc6 Bxc6 $2 23. Qe2 $2 Bxg2 $6 24. f3 Bh3 25. Bd6 Rc8 $6 26.
Bg4 $6 Bxg4 27. fxg4 Kf7 28. Rab1 $4 Qc6 $9 29. Bxe7 Kxe7 30. Rb5 e5 $2 31. Qf2 $9 Qe6
32. h3 Qc6 $2 33. Rdb1 Nd7 34. Rd5 h6 35. Rbd1 Nc5 $6 36. Qf5 Ne6 37. Qh7 Rc7 38.
Qg6 $2 Qxc4 39. R5d3 $2 Qa4 $9 40. Rd6 $1 Qb5 41. Qe4 $2 Qc5+ 42. Kh1 Nf4 $9 43. Qh7 $1 Kf7
44. Qf5 $1 Qxa3 $6 45. Rxf6+ $3 gxf6 46. Qh7+ $1 Ke8 47. Qg8+ Qf8 48. Rd8+ $1 Kxd8 49.
Qxf8+ Kd7 50. Qxf6 Rc5 51. Qxb6 Rc1+ 52. Kh2 Rc3 53. Qxa5 Rxh3+ 54. Kg1 Kd6 55.
Kf2 Nd3+ 56. Ke2 Nf4+ 57. Kd2 Rd3+ 58. Kc2 Rg3 59. Qd8+ Kc5 60. Qa5+ Kd4 61.
Qb6+ Ke4 62. Qc6+ Kd4 63. Qa4+ Kc5 64. Qa5+ Kd4 65. Qa1+ Kd5 66. Qe1 Rg2+ 67.
Kb3 Rb2+ 68. Ka3 Re2 69. Qb1 Kd4 70. Qb6+ Ke4 71. Qxh6 Kf3 72. g5 Rg2 73. Qh1
Kf2 74. Qh6 e4 75. Qf6 Kf3 76. Ka4 $2 e3 $9 77. Qc6+ Kf2 78. Qc5 Rg3 79. Ka5 $2 Rf3 $9
80. Qd4 Ng6 81. Qb6 Rf5+ 82. Ka4 Ne5 83. g6 Nf3 84. g7 Rg5 85. Qa7 Rg6 86. Kb4
Rg4+ 87. Kc3 Ne5 88. Kc2 Nf3 89. Kc3 Ne5 90. Kc2 Rg2 91. Qc5 Kf1+ 92. Kb3 e2 93.
Qf8+ Kg1 94. g8=Q Rxg8 95. Qxg8+ Kf2 96. Qf8+ Nf3 97. Qc5+ Kg2 98. Qe7 e1=Q 99.
Qxe1 Nxe1 1/2-1/2
"""

def extract_benchmark_positions():
    """
    Parses the PGN and returns a list of tuples:
    (ply, move_str, fen) for moves from 15. Be2 (ply 29) to 95... Kf2 (ply 190).
    """
    game = chess.pgn.read_game(io.StringIO(GAME_PGN))
    board = game.board()
    moves = list(game.mainline_moves())
    
    positions = []
    for idx, move in enumerate(moves):
        ply = idx + 1
        full_move_num = (idx // 2) + 1
        is_white = (idx % 2 == 0)
        move_san = board.san(move)
        move_str = f"{full_move_num}. {move_san}" if is_white else f"{full_move_num}... {move_san}"
        
        # Ply 29 is 15. Be2; Ply 190 is 95... Kf2
        if 29 <= ply <= 190:
            positions.append((ply, move_str, board.fen()))
            
        board.push(move)
        
    return positions

def run_search_test():
    """
    Runs 1-second search tests across all positions from 15. Be2 to 95... Kf2 in the game PGN.
    Displays only the deepest search info for each move, and outputs summary statistics at the end.
    """
    # 1. Warm-up (triggers JIT compilation with Kiwipete Depth 12)
    print("--- WARMING UP (Compiling JIT functions with Kiwipete Depth 12) ---")
    kiwipete_fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    p_bbs, o_bbs, g_state = parse_fen(kiwipete_fen)
    clear_transposition_table(transposition_table)
    ctx = make_search_context(transposition_table)
    
    warmup_start = time.time()
    iterative_deepening_search(p_bbs, o_bbs, g_state, 12, {'optimum_time': 300000, 'maximum_time': 300000}, ctx, verbose=False)
    warmup_elapsed = time.time() - warmup_start
    print(f"--- WARM-UP COMPLETE ({warmup_elapsed:.2f}s) ---\n")

    # 2. Extract benchmark positions
    positions = extract_benchmark_positions()

    print("=========================================================================================================")
    print(f"  STARTING FULL GAME SEARCH BENCHMARK ({len(positions)} POSITIONS: 15. Be2 to 95... Kf2)                 ")
    print("  Time Limit: 1.0s per position                                                                          ")
    print("=========================================================================================================")

    nodes_list = []
    nps_list = []
    depth_list = []

    time_config = {'optimum_time': 1000, 'maximum_time': 1000}

    for step_idx, (ply, move_str, fen) in enumerate(positions, 1):
        piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
        clear_transposition_table(transposition_table)
        search_context = make_search_context(transposition_table)

        start_time = time.time()
        (best_move, best_eval, nodes_searched, quiescence_nodes, tt_hits, last_completed_depth) = iterative_deepening_search(
            piece_bbs, occupancy_bbs, game_state, 64, time_config, search_context, verbose=False
        )
        elapsed_time = time.time() - start_time

        total_nodes = nodes_searched + quiescence_nodes
        nps = int(total_nodes / elapsed_time) if elapsed_time > 0 else 0
        best_move_str = move_to_uci(best_move)
        eval_str = format_score_for_uci(best_eval)

        nodes_list.append(total_nodes)
        nps_list.append(nps)
        depth_list.append(last_completed_depth)

        # Output only the deepest search info for this step
        print(
            f"Step {step_idx:3d}/{len(positions)} [{move_str:<12}] | Depth: {last_completed_depth:2d} | "
            f"Eval: {eval_str:<6} | Nodes: {total_nodes:8d} | NPS: {nps:8d} | "
            f"Time: {elapsed_time:.3f}s | Move: {best_move_str}"
        )

    # 3. Calculate and print summary statistics
    mean_nodes = float(np.mean(nodes_list))
    median_nodes = float(np.median(nodes_list))
    mean_nps = float(np.mean(nps_list))
    median_nps = float(np.median(nps_list))
    mean_depth = float(np.mean(depth_list))
    median_depth = float(np.median(depth_list))

    print("\n==========================================================================")
    print("                      BENCHMARK SUMMARY STATISTICS                        ")
    print("==========================================================================")
    print(f"Total Positions Tested:               {len(positions)}")
    print(f"平均每步節點數 (Mean Nodes/Move):     {mean_nodes:14.1f}")
    print(f"每步節點中位數 (Median Nodes/Move):   {median_nodes:14.1f}")
    print(f"平均每步 NPS     (Mean NPS):          {mean_nps:14.1f}")
    print(f"每步 NPS 中位數  (Median NPS):        {median_nps:14.1f}")
    print(f"平均每步深度     (Mean Depth):        {mean_depth:14.2f}")
    print(f"每步深度中位數   (Median Depth):      {median_depth:14.2f}")
    print("==========================================================================")

if __name__ == "__main__":
    run_search_test()
