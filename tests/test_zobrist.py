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
    """
    å°‹æ‰¾??UCI å­—ä¸²å°æ??„ç·¨ç¢¼ç§»?•ã€?
    """
    # Ensure game_state tuple has Numba-compatible types, as it's passed to a JIT function
    # ç¢ºä? game_state ?·æ? Numba ?¼å®¹?„é???
    # side, castling, ep, halfmove, zobrist = game_state
    # The piece and occupancy bbs are already typed correctly when this is called from _test_zobrist_hash
    # ?¶æ­¤?½æ•¸è¢«èª¿?¨æ?ï¼Œpiece_bbs ??occupancy_bbs å·²ç??¯æ­£ç¢ºç? NumPy ???
    moves = generate_legal_moves(piece_bbs, occupancy_bbs, game_state)
    for move in moves:
        if move_to_uci(move) == uci_string:
            return move
    return None

def _test_zobrist_hash(fen, uci_move):
    """
    ?¸å? Zobrist ?ˆå?æ¸¬è©¦?½æ•¸??
    
    1. å¾?FEN ? è?å±€?¢ã€?
    2. ?·è?ä¸€?‹ç§»?•ã€?
    3. é©—è?å¢é??´æ–°??Zobrist ?µå€¼è?å®Œå…¨?æ–°è¨ˆç??„éµ?¼åŒ¹?ã€?
    4. ?¤éŠ·ç§»å???
    5. é©—è? Zobrist ?µå€¼å??¨æ¢å¾©ã€?
    """
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
    # We create a temporary game state with a zeroed key to ensure the re-computation is truly from scratch.
    # ?µå»ºä¸€?‹è‡¨?‚ç? game_state ä¸¦å? key æ­¸é›¶ï¼Œä»¥ç¢ºä??æ–°è¨ˆç??¯å??­é?å§‹ç???
    temp_game_state_for_recompute = game_state.copy()
    temp_game_state_for_recompute[4] = np.uint64(0)
    
    recomputed_key = compute_initial_hash(piece_bbs, temp_game_state_for_recompute)
    assert updated_key == recomputed_key, f"Zobrist key mismatch after make_move for {uci_move} in FEN {fen}"

    # 4. Unmake the move
    unmake_move(piece_bbs, occupancy_bbs, game_state, move, unmake_info)
    
    # 5. Verify the key is restored
    restored_key = game_state[4]
    assert restored_key == original_key, f"Zobrist key mismatch after unmake_move for {uci_move} in FEN {fen}"

# --- White Piece Test Cases / ?½æ–¹æ£‹å?æ¸¬è©¦?¨ä? ---

def test_basic_move():
    """æ¸¬è©¦ç°¡å–®?µæ¨?²ç? Zobrist ?µå€¼æ›´?°ã€?""
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    uci_move = "e2e4"
    _test_zobrist_hash(fen, uci_move)

def test_capture():
    """æ¸¬è©¦ç°¡å–®?ƒå???Zobrist ?µå€¼æ›´?°ã€?""
    # This position is after 1. e4 e5 2. Nf3 f6
    fen = "rnbqkbnr/pppp2pp/5p2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3"
    uci_move = "f3e5"
    _test_zobrist_hash(fen, uci_move)

def test_castling_rights_loss_king_move():
    """æ¸¬è©¦?‹ç§»?•å¤±?»æ?ä½æ?å¾Œç? Zobrist ?´æ–°??""
    fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
    uci_move = "e1e2"
    _test_zobrist_hash(fen, uci_move)

def test_castling_rights_loss_rook_move():
    """æ¸¬è©¦è»Šç§»?•å¤±?»æ?ä½æ?å¾Œç? Zobrist ?´æ–°??""
    fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"
    uci_move = "h1g1"
    _test_zobrist_hash(fen, uci_move)

def test_en_passant():
    """æ¸¬è©¦?ƒé?è·¯å…µ??Zobrist ?µå€¼æ›´?°ã€?""
    fen = "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"
    uci_move = "e5f6"
    _test_zobrist_hash(fen, uci_move)

def test_promotion_simple():
    """æ¸¬è©¦ç°¡å–®?‡è???Zobrist ?µå€¼æ›´?°ã€?""
    fen = "rnbqkbr1/pp5P/2p1pp2/3p4/8/8/PPPP1PP1/RNBQKBNR w KQq - 0 1"
    uci_move = "h7h8q"
    _test_zobrist_hash(fen, uci_move)

def test_promotion_capture():
    """æ¸¬è©¦?ƒå??‡è???Zobrist ?µå€¼æ›´?°ã€?""
    fen = "rnb1kbnr/ppP4p/4pp2/3p4/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"
    uci_move = "c7b8q"
    _test_zobrist_hash(fen, uci_move)

# --- Black Piece Test Cases / é»‘æ–¹æ£‹å?æ¸¬è©¦?¨ä? ---

def test_black_en_passant():
    """æ¸¬è©¦é»‘æ–¹?ƒé?è·¯å…µ??Zobrist ?µå€¼æ›´?°ã€?""
    fen = "rnbqkbnr/pppp1ppp/8/8/4PpP1/8/PPPP3P/RNBQKBNR b KQkq g3 0 3"
    uci_move = "f4g3"
    _test_zobrist_hash(fen, uci_move)

def test_black_promotion_capture():
    """æ¸¬è©¦é»‘æ–¹?ƒå??‡è???Zobrist ?µå€¼æ›´?°ã€?""
    fen = "rnbqkbnr/1Ppppp1p/8/8/8/8/pP1P1P1P/RNBQKBNR b KQkq - 0 1"
    uci_move = "a2b1q"
    _test_zobrist_hash(fen, uci_move)

def test_black_castling():
    """æ¸¬è©¦é»‘æ–¹?‹è??“ä???Zobrist ?µå€¼æ›´?°ã€?""
    fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1"
    uci_move = "e8g8"
    _test_zobrist_hash(fen, uci_move)
