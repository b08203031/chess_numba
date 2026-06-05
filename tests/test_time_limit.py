import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import time
import numpy as np
from chess_engine.classical.search import iterative_deepening_search
from chess_engine.classical.fen_parser import parse_fen
from chess_engine.classical.transposition_table import create_transposition_table, TT_SIZE_MB
from chess_engine.classical.engine_types import SearchContext

def test_search_time_limit():
    print("Testing search time limit...")
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
    
    transposition_table = create_transposition_table(TT_SIZE_MB)
    MAX_PLY = 128
    killer_moves = np.zeros(MAX_PLY * 2, dtype=np.uint16)
    pv_table = np.zeros((MAX_PLY, MAX_PLY), dtype=np.uint16)
    history_table = np.zeros((12, 64), dtype=np.int32)
    butterfly_history = np.zeros((64, 64), dtype=np.int32)
    continuation_history = np.zeros((4, 12, 64, 12, 64), dtype=np.int16)
    capture_history = np.zeros((12, 64, 12), dtype=np.int32)
    pawn_history = np.full((8192, 12, 64), -1238, dtype=np.int16)
    pawn_correction_history = np.zeros(16384, dtype=np.int16)
    minor_correction_history = np.zeros(16384, dtype=np.int16)
    non_pawn_correction_history_white = np.zeros(16384, dtype=np.int16)
    non_pawn_correction_history_black = np.zeros(16384, dtype=np.int16)

    search_context = SearchContext(
        transposition_table, killer_moves, pv_table, history_table,
        butterfly_history, continuation_history, capture_history, pawn_history,
        pawn_correction_history, minor_correction_history,
        non_pawn_correction_history_white, non_pawn_correction_history_black
    )
    
    # Warm-up JIT
    print("Warming up JIT...")
    iterative_deepening_search(piece_bbs, occupancy_bbs, game_state, 1, {'optimum_time': 0, 'maximum_time': 0}, search_context)
    
    # Set a 100ms time limit
    time_config = {'optimum_time': 100, 'maximum_time': 100}
    
    print("Starting timed search...")
    start_time = time.time()
    iterative_deepening_search(piece_bbs, occupancy_bbs, game_state, 64, time_config, search_context)
    end_time = time.time()
    
    elapsed = (end_time - start_time) * 1000
    print(f"Elapsed time: {elapsed:.2f}ms")
    
    # Allow some overhead, but it should be close to 100ms
    # The engine checks every 2048 nodes, so it might overrun slightly.
    # But it shouldn't be 5 seconds.
    if elapsed < 500:
        print("PASS: Search stopped within reasonable time.")
    else:
        print("FAIL: Search took too long.")

if __name__ == "__main__":
    test_search_time_limit()
