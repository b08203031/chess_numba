# tests/test_move_generator.py
import pytest
import numpy as np

from chess_engine.board import Board
from chess_engine.move_generator import generate_legal_moves, generate_legal_moves_v2, generate_captures, generate_captures_v2
from chess_engine.move import move_to_uci

# A diverse set of FEN strings for testing
FEN_STRINGS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Start position
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",  # Kiwipete
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",  # Position 3
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", # Position 5
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", # Tricky promotions
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", # Quiet position
]

def sort_moves(moves):
    """Sorts a list of moves based on their UCI string representation."""
    return sorted([move_to_uci(m) for m in moves])

@pytest.mark.parametrize("fen", FEN_STRINGS)
def test_generate_legal_moves_v2_consistency(fen):
    """
    Asserts that generate_legal_moves_v2 produces the same output as the original.
    """
    board = Board(fen)

    # Original version
    moves_v1 = generate_legal_moves(board.piece_bbs, board.occupancy_bbs, board.game_state)

    # V2 version
    piece_bbs_array = np.array(board.piece_bbs, dtype=np.uint64)
    occupancy_bbs_array = np.array(board.occupancy_bbs, dtype=np.uint64)
    side, castling, ep, halfmove, zobrist = board.game_state
    game_state_array = np.array([side, castling, ep, halfmove], dtype=np.int32)
    zobrist_key_holder = np.array([zobrist], dtype=np.uint64)
    moves_v2 = generate_legal_moves_v2(piece_bbs_array, occupancy_bbs_array, game_state_array, zobrist_key_holder)

    assert sort_moves(moves_v1) == sort_moves(moves_v2)

@pytest.mark.parametrize("fen", FEN_STRINGS)
def test_generate_captures_v2_consistency(fen):
    """
    Asserts that generate_captures_v2 produces the same output as the original.
    """
    board = Board(fen)

    # Original version
    moves_v1 = generate_captures(board.piece_bbs, board.occupancy_bbs, board.game_state)

    # V2 version
    piece_bbs_array = np.array(board.piece_bbs, dtype=np.uint64)
    occupancy_bbs_array = np.array(board.occupancy_bbs, dtype=np.uint64)
    side, castling, ep, halfmove, zobrist = board.game_state
    game_state_array = np.array([side, castling, ep, halfmove], dtype=np.int32)
    zobrist_key_holder = np.array([zobrist], dtype=np.uint64)
    moves_v2 = generate_captures_v2(piece_bbs_array, occupancy_bbs_array, game_state_array, zobrist_key_holder)

    assert sort_moves(moves_v1) == sort_moves(moves_v2)
