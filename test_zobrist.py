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
    尋找與 UCI 字串對應的編碼移動。
    """
    # Ensure game_state tuple has Numba-compatible types, as it's passed to a JIT function
    # 確保 game_state 具有 Numba 兼容的類型
    # side, castling, ep, halfmove, zobrist = game_state
    # The piece and occupancy bbs are already typed correctly when this is called from _test_zobrist_hash
    # 當此函數被調用時，piece_bbs 和 occupancy_bbs 已經是正確的 NumPy 陣列
    moves = generate_legal_moves(piece_bbs, occupancy_bbs, game_state)
    for move in moves:
        if move_to_uci(move) == uci_string:
            return move
    return None

def _test_zobrist_hash(fen, uci_move):
    """
    核心 Zobrist 哈希測試函數。
    
    1. 從 FEN 加載局面。
    2. 執行一個移動。
    3. 驗證增量更新的 Zobrist 鍵值與完全重新計算的鍵值匹配。
    4. 撤銷移動。
    5. 驗證 Zobrist 鍵值完全恢復。
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
    # 創建一個臨時的 game_state 並將 key 歸零，以確保重新計算是從頭開始的。
    temp_game_state_for_recompute = game_state.copy()
    temp_game_state_for_recompute[4] = np.uint64(0)
    
    recomputed_key = compute_initial_hash(piece_bbs, temp_game_state_for_recompute)
    assert updated_key == recomputed_key, f"Zobrist key mismatch after make_move for {uci_move} in FEN {fen}"

    # 4. Unmake the move
    unmake_move(piece_bbs, occupancy_bbs, game_state, move, unmake_info)
    
    # 5. Verify the key is restored
    restored_key = game_state[4]
    assert restored_key == original_key, f"Zobrist key mismatch after unmake_move for {uci_move} in FEN {fen}"

# --- White Piece Test Cases / 白方棋子測試用例 ---

def test_basic_move():
    """測試簡單兵推進的 Zobrist 鍵值更新。"""
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    uci_move = "e2e4"
    _test_zobrist_hash(fen, uci_move)

def test_capture():
    """測試簡單吃子的 Zobrist 鍵值更新。"""
    # This position is after 1. e4 e5 2. Nf3 f6
    fen = "rnbqkbnr/pppp2pp/5p2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3"
    uci_move = "f3e5"
    _test_zobrist_hash(fen, uci_move)

def test_castling_rights_loss_king_move():
    """測試王移動失去易位權後的 Zobrist 更新。"""
    fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
    uci_move = "e1e2"
    _test_zobrist_hash(fen, uci_move)

def test_castling_rights_loss_rook_move():
    """測試車移動失去易位權後的 Zobrist 更新。"""
    fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"
    uci_move = "h1g1"
    _test_zobrist_hash(fen, uci_move)

def test_en_passant():
    """測試吃過路兵的 Zobrist 鍵值更新。"""
    fen = "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"
    uci_move = "e5f6"
    _test_zobrist_hash(fen, uci_move)

def test_promotion_simple():
    """測試簡單升變的 Zobrist 鍵值更新。"""
    fen = "rnbqkbr1/pp5P/2p1pp2/3p4/8/8/PPPP1PP1/RNBQKBNR w KQq - 0 1"
    uci_move = "h7h8q"
    _test_zobrist_hash(fen, uci_move)

def test_promotion_capture():
    """測試吃子升變的 Zobrist 鍵值更新。"""
    fen = "rnb1kbnr/ppP4p/4pp2/3p4/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"
    uci_move = "c7b8q"
    _test_zobrist_hash(fen, uci_move)

# --- Black Piece Test Cases / 黑方棋子測試用例 ---

def test_black_en_passant():
    """測試黑方吃過路兵的 Zobrist 鍵值更新。"""
    fen = "rnbqkbnr/pppp1ppp/8/8/4PpP1/8/PPPP3P/RNBQKBNR b KQkq g3 0 3"
    uci_move = "f4g3"
    _test_zobrist_hash(fen, uci_move)

def test_black_promotion_capture():
    """測試黑方吃子升變的 Zobrist 鍵值更新。"""
    fen = "rnbqkbnr/1Ppppp1p/8/8/8/8/pP1P1P1P/RNBQKBNR b KQkq - 0 1"
    uci_move = "a2b1q"
    _test_zobrist_hash(fen, uci_move)

def test_black_castling():
    """測試黑方王車易位的 Zobrist 鍵值更新。"""
    fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1"
    uci_move = "e8g8"
    _test_zobrist_hash(fen, uci_move)
