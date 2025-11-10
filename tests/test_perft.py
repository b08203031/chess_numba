# tests/test_perft.py
import pytest
import numpy as np
import time

from chess_engine.board import Board
from chess_engine.core import _jit_perft_divide, PERFT_RESULTS, _jit_perft_divide_v2
from chess_engine.move import move_to_uci
from chess_engine.zobrist import compute_initial_hash

def perft_divide(board, depth):
    """Helper function to call the JIT-compiled perft divide."""
    return _jit_perft_divide(board.piece_bbs, board.occupancy_bbs, board.game_state, depth)

def perft_divide_v2(board):
    """Helper function to prepare arrays for the v2 JIT-compiled perft divide."""
    piece_bbs_array = np.array(board.piece_bbs, dtype=np.uint64)
    occupancy_bbs_array = np.array(board.occupancy_bbs, dtype=np.uint64)
    side, castling, ep, halfmove, zobrist = board.game_state
    game_state_array = np.array([side, castling, ep, halfmove], dtype=np.int32)
    zobrist_key_holder = np.array([zobrist], dtype=np.uint64)
    return piece_bbs_array, occupancy_bbs_array, game_state_array, zobrist_key_holder

@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_perft_startpos(depth):
    board = Board()
    results = perft_divide(board, depth)
    total_nodes = np.sum(results[:, 1])
    assert total_nodes == PERFT_RESULTS["startpos"][depth]

@pytest.mark.parametrize("depth", [1, 2, 3, 4, 5])
def test_perft_v2_startpos(depth):
    print(f"--- Running Perft V2 Depth {depth} ---")
    board = Board()

    # Prepare arrays
    piece_bbs_array, occupancy_bbs_array, game_state_array, zobrist_key_holder = perft_divide_v2(board)

    start_time = time.time()
    results = _jit_perft_divide_v2(
        piece_bbs_array, occupancy_bbs_array, game_state_array, zobrist_key_holder, depth
    )
    end_time = time.time()

    total_nodes = np.sum(results[:, 1])

    print(f"Total nodes: {total_nodes}, Time: {end_time - start_time:.2f}s")

    # Sort results for consistent comparison
    sorted_results = sorted([(move_to_uci(move), count) for move, count in results], key=lambda x: x[0])

    for uci, count in sorted_results:
        print(f"{uci}: {count}")

    assert total_nodes == PERFT_RESULTS["startpos"][depth]
