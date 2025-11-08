# test_zobrist.py
import pytest
import numpy as np

from chess_engine.fen_parser import parse_fen
from chess_engine.board_operations import make_move, unmake_move
from chess_engine.zobrist import compute_initial_hash
from chess_engine.move_generator import generate_legal_moves
from chess_engine.move import move_to_uci

def uci_to_move(piece_bbs, occupancy_bbs, game_state, uci_string):
    """
    Finds the encoded move corresponding to a UCI string.
    """
    moves = generate_legal_moves(piece_bbs, occupancy_bbs, game_state)
    for move in moves:
        if move_to_uci(move) == uci_string:
            return move
    return None

def _test_zobrist_hash(fen, uci_move):
    """
    Core Zobrist hash test function.
    
    1. Loads a position from FEN.
    2. Makes a move.
    3. Verifies that the incrementally updated Zobrist key matches a fully recomputed key.
    4. Unmakes the move.
    5. Verifies that the Zobrist key is perfectly restored.
    """
    # 1. Load position and get initial hash
    piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
    original_key = game_state[4]

    # Find the move object corresponding to the UCI string
    move = uci_to_move(piece_bbs, occupancy_bbs, game_state, uci_move)
    assert move is not None, f"Could not find legal move for {uci_move} in FEN {fen}"

    # 2. Make the move and get the incrementally updated key
    new_piece_bbs, new_occupancy_bbs, new_game_state, unmake_info = make_move(
        piece_bbs, occupancy_bbs, game_state, move
    )
    updated_key = new_game_state[4]

    # 3. Recompute the hash from the new position and verify
    # We create a temporary game state with a zeroed key to ensure the re-computation is truly from scratch.
    temp_game_state_for_recompute = (
        new_game_state[0], new_game_state[1], new_game_state[2], new_game_state[3], np.uint64(0)
    )
    recomputed_key = compute_initial_hash(new_piece_bbs, temp_game_state_for_recompute)
    assert updated_key == recomputed_key, f"Zobrist key mismatch after make_move for {uci_move} in FEN {fen}"

    # 4. Unmake the move
    restored_piece_bbs, restored_occupancy_bbs, restored_game_state = unmake_move(
        new_piece_bbs, new_occupancy_bbs, new_game_state, move, unmake_info
    )
    
    # 5. Verify the key is restored
    restored_key = restored_game_state[4]
    assert restored_key == original_key, f"Zobrist key mismatch after unmake_move for {uci_move} in FEN {fen}"

# --- White Piece Test Cases ---

def test_basic_move():
    """Tests the Zobrist key update for a simple pawn push."""
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    uci_move = "e2e4"
    _test_zobrist_hash(fen, uci_move)

def test_capture():
    """Tests the Zobrist key update for a simple capture."""
    # This position is after 1. e4 e5 2. Nf3 f6
    fen = "rnbqkbnr/pppp2pp/5p2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3"
    uci_move = "f3e5"
    _test_zobrist_hash(fen, uci_move)

def test_castling_rights_loss_king_move():
    """Tests Zobrist update after king move forfeits castling rights."""
    fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
    uci_move = "e1e2"
    _test_zobrist_hash(fen, uci_move)

def test_castling_rights_loss_rook_move():
    """Tests Zobrist update after rook move forfeits a castling right."""
    fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"
    uci_move = "h1g1"
    _test_zobrist_hash(fen, uci_move)

def test_en_passant():
    """Tests the Zobrist key update for an en passant capture."""
    fen = "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"
    uci_move = "e5f6"
    _test_zobrist_hash(fen, uci_move)

def test_promotion_simple():
    """Tests the Zobrist key update for a simple promotion."""
    fen = "rnbqkbr1/pp5P/2p1pp2/3p4/8/8/PPPP1PP1/RNBQKBNR w KQq - 0 1"
    uci_move = "h7h8q"
    _test_zobrist_hash(fen, uci_move)

def test_promotion_capture():
    """Tests the Zobrist key update for a capture promotion."""
    fen = "rnb1kbnr/ppP4p/4pp2/3p4/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"
    uci_move = "c7b8q"
    _test_zobrist_hash(fen, uci_move)

# --- Black Piece Test Cases ---

def test_black_en_passant():
    """Tests the Zobrist key update for a Black en passant capture."""
    fen = "rnbqkbnr/pppp1ppp/8/8/4PpP1/8/PPPP3P/RNBQKBNR b KQkq g3 0 3"
    uci_move = "f4g3"
    _test_zobrist_hash(fen, uci_move)

def test_black_promotion_capture():
    """Tests the Zobrist key update for a Black capture promotion."""
    fen = "rnbqkbnr/1Ppppp1p/8/8/8/8/pP1P1P1P/RNBQKBNR b KQkq - 0 1"
    uci_move = "a2b1q"
    _test_zobrist_hash(fen, uci_move)

def test_black_castling():
    """Tests the Zobrist key update for Black castling."""
    fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1"
    uci_move = "e8g8"
    _test_zobrist_hash(fen, uci_move)
