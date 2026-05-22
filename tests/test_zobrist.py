import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# test_zobrist.py
import pytest
import numpy as np

from chess_engine.classical.fen_parser import parse_fen
from chess_engine.classical.board_operations import make_move, unmake_move
from chess_engine.classical.zobrist import compute_initial_hash
from chess_engine.classical.move_generator import generate_legal_moves
from chess_engine.classical.move import move_to_uci

def uci_to_move(piece_bbs, occupancy_bbs, game_state, uci_string):
    """Convert UCI string to move object."""
    moves = generate_legal_moves(piece_bbs, occupancy_bbs, game_state)
    for move in moves:
        if move_to_uci(move) == uci_string:
            return move
    return None

def _test_zobrist_hash(fen, uci_move):
    """Test Zobrist hash updates incrementally and matches scratch calculation."""
    # 1. Load position and get initial hash
    piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
    original_key = game_state[4]

    # Find the move object corresponding to the UCI string
    move = uci_to_move(piece_bbs, occupancy_bbs, game_state, uci_move)
    assert move is not None, f"Could not find legal move for {uci_move} in FEN {fen}"

    # 2. Make the move and get the incrementally updated key
    unmake_info = make_move(piece_bbs, occupancy_bbs, game_state, move)
    updated_key = game_state[4]

    # 3. Recompute the hash from the new position and verify
    temp_game_state_for_recompute = game_state.copy()
    temp_game_state_for_recompute[4] = np.uint64(0)
    
    recomputed_key = compute_initial_hash(piece_bbs, temp_game_state_for_recompute)
    assert updated_key == recomputed_key, f"Zobrist key mismatch after make_move for {uci_move} in FEN {fen}"

    # 4. Unmake the move
    unmake_move(piece_bbs, occupancy_bbs, game_state, move, unmake_info)
    
    # 5. Verify the key is restored
    restored_key = game_state[4]
    assert restored_key == original_key, f"Zobrist key mismatch after unmake_move for {uci_move} in FEN {fen}"

# --- White Piece Test Cases ---

def test_basic_move():
    """Test basic Zobrist hash update."""
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    uci_move = "e2e4"
    _test_zobrist_hash(fen, uci_move)

def test_capture():
    """Test capture Zobrist hash update."""
    fen = "rnbqkbnr/pppp2pp/5p2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3"
    uci_move = "f3e5"
    _test_zobrist_hash(fen, uci_move)

def test_castling_rights_loss_king_move():
    """Test king move castling rights loss Zobrist update."""
    fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
    uci_move = "e1e2"
    _test_zobrist_hash(fen, uci_move)

def test_castling_rights_loss_rook_move():
    """Test rook move castling rights loss Zobrist update."""
    fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"
    uci_move = "h1g1"
    _test_zobrist_hash(fen, uci_move)

def test_en_passant():
    """Test en passant Zobrist update."""
    fen = "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"
    uci_move = "e5f6"
    _test_zobrist_hash(fen, uci_move)

def test_promotion_simple():
    """Test simple promotion Zobrist update."""
    fen = "rnbqkbr1/pp5P/2p1pp2/3p4/8/8/PPPP1PP1/RNBQKBNR w KQq - 0 1"
    uci_move = "h7h8q"
    _test_zobrist_hash(fen, uci_move)

def test_promotion_capture():
    """Test promotion capture Zobrist update."""
    fen = "rnb1kbnr/ppP4p/4pp2/3p4/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"
    uci_move = "c7b8q"
    _test_zobrist_hash(fen, uci_move)

# --- Black Piece Test Cases ---

def test_black_en_passant():
    """Test Black en passant Zobrist update."""
    fen = "rnbqkbnr/pppp1ppp/8/8/4PpP1/8/PPPP3P/RNBQKBNR b KQkq g3 0 3"
    uci_move = "f4g3"
    _test_zobrist_hash(fen, uci_move)

def test_black_promotion_capture():
    """Test Black promotion capture Zobrist update."""
    fen = "rnbqkbnr/1Ppppp1p/8/8/8/8/pP1P1P1P/RNBQKBNR b KQkq - 0 1"
    uci_move = "a2b1q"
    _test_zobrist_hash(fen, uci_move)

def test_black_castling():
    """Test Black castling Zobrist update."""
    fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1"
    uci_move = "e8g8"
    _test_zobrist_hash(fen, uci_move)

if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__]))
