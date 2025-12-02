
import sys
import os
import time
from chess_engine.search import iterative_deepening_search
from chess_engine.fen_parser import parse_fen
from chess_engine.transposition_table import create_transposition_table, clear_transposition_table
from chess_engine.engine_types import SearchContext
from chess_engine.move import move_to_uci
import numpy as np

def run_debug():
    # 1. Warm-up (triggers JIT compilation)
    print("--- WARMING UP (Compiling JIT functions) ---")
    fen_start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    p_bbs, o_bbs, g_state = parse_fen(fen_start)
    tt = create_transposition_table(16)
    killer_moves = np.zeros(256, dtype=np.uint16)
    history_table = np.zeros((12, 64), dtype=np.int32)
    pv_table = np.zeros((128, 128), dtype=np.uint16)
    ctx = SearchContext(tt, killer_moves, pv_table, history_table)
    
    # Run a quick search
    iterative_deepening_search(p_bbs, o_bbs, g_state, 2, {'optimum_time': 0, 'maximum_time': 0}, ctx)
    print("--- WARM-UP COMPLETE ---\n")

    # 2. Actual Benchmark
    fen = "8/6pk/5p2/p1bN3p/P4P2/1bN4P/1Pr3P1/1R5K w - - 2 39"
    print(f"Analyzing FEN: {fen}")

    # Clear TT for fair test
    clear_transposition_table(tt)
    ctx.killer_moves.fill(0)
    ctx.history_table.fill(0)
    
    piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
    
    max_depth = 15
    print(f"Running search up to depth {max_depth}...")
    
    start_time = time.time()
    best_move, score, nodes, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = iterative_deepening_search(
        piece_bbs, occupancy_bbs, game_state,
        max_depth, {'optimum_time': 0, 'maximum_time': 0}, ctx
    )
    end_time = time.time()
    
    duration = end_time - start_time
    nps = nodes / duration if duration > 0 else 0

    print(f"\nSearch Finished in {duration:.4f}s")
    print(f"Best Move: {move_to_uci(best_move)}")
    print(f"Score: {score}")
    print(f"Nodes: {nodes}")
    print(f"NPS: {nps:.2f}")

if __name__ == "__main__":
    run_debug()
