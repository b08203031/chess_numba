import sys
import os
import time
import numpy as np

# Directly append workspace path
workspace_path = r"c:\Users\ren cian\OneDrive\桌面\chess\chess_numba\chess_numba"
sys.path.insert(0, workspace_path)

from chess_engine.classical.fen_parser import parse_fen
from chess_engine.classical.search import iterative_deepening_search
from chess_engine.classical.move import move_to_uci
from chess_engine.classical.transposition_table import create_transposition_table, clear_transposition_table
from chess_engine.classical.engine_types import SearchContext

def run_test():
    tt = create_transposition_table(64)
    MAX_PLY = 128
    killer_moves = np.zeros(MAX_PLY * 2, dtype=np.uint16)
    pv_table = np.zeros((MAX_PLY, MAX_PLY), dtype=np.uint16)
    history_table = np.zeros((12, 64), dtype=np.int32)
    butterfly_history = np.zeros((64, 64), dtype=np.int32)
    continuation_history = np.zeros((3, 12, 64, 12, 64), dtype=np.int16)
    capture_history = np.zeros((12, 64, 12), dtype=np.int32)
    pawn_history = np.full((8192, 12, 64), -1238, dtype=np.int16)
    pawn_correction_history = np.zeros(16384, dtype=np.int16)
    minor_correction_history = np.zeros(16384, dtype=np.int16)
    non_pawn_correction_history_white = np.zeros(16384, dtype=np.int16)
    non_pawn_correction_history_black = np.zeros(16384, dtype=np.int16)

    ctx = SearchContext(
        tt, killer_moves, pv_table, history_table,
        butterfly_history, continuation_history, capture_history, pawn_history,
        pawn_correction_history, minor_correction_history,
        non_pawn_correction_history_white, non_pawn_correction_history_black
    )

    tests = [
        {
            "name": "KZDTb",
            "fen": "2r3k1/pp1b2pp/1q2Pb2/2np4/6B1/2N5/PPP3PP/R3QRK1 b - - 0 18",
            "solution": "c5d3"
        },
        {
            "name": "GNuB5",
            "fen": "6r1/pppk2qp/3pR3/8/4Qp2/6PP/PPP2P2/1K6 b - - 1 29",
            "solution": "g7f7"
        }
    ]

    for t in tests:
        print(f"\n--- Running {t['name']} (Time Limit: 10s) ---")
        p_bbs, o_bbs, g_state = parse_fen(t["fen"])
        clear_transposition_table(tt)
        ctx.killer_moves.fill(0)
        ctx.history_table.fill(0)
        
        # 10 seconds time limit
        time_config = {'optimum_time': 10000, 'maximum_time': 10000}
        
        best_move, best_eval, nodes, q_nodes, tt_hits, last_depth = iterative_deepening_search(
            p_bbs, o_bbs, g_state, 25, time_config, ctx
        )
        
        move_uci = move_to_uci(best_move)
        print(f"Result: {move_uci} (Solution: {t['solution']})")
        print(f"Depth: {last_depth}, Nodes: {nodes + q_nodes}, Eval: {best_eval}")
        if move_uci == t["solution"]:
            print("Status: SUCCESS")
        else:
            print("Status: FAILED")

if __name__ == "__main__":
    run_test()
