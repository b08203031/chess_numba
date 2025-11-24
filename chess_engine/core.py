
import numba
import numpy as np

from .board_operations import make_move, unmake_move
from .move_generator import generate_legal_moves, is_square_attacked
from .engine_types import piece_bbs_signature, occupancy_bbs_signature, game_state_signature
import numba.types as nbt
from .zobrist import get_lsb_index

@numba.njit(numba.boolean(piece_bbs_signature, occupancy_bbs_signature, game_state_signature), cache=True)
def is_king_in_check(piece_bbs, occupancy_bbs, game_state):
    """
    檢查當前行棋方的王是否處於被將軍狀態。

    Args:
        piece_bbs (np.ndarray): 12 個棋子的位元棋盤。
        occupancy_bbs (np.ndarray): 佔用位元棋盤。
        game_state (np.ndarray): 遊戲狀態陣列。

    Returns:
        bool: 如果王被將軍則返回 True，否則返回 False。
    """
    side_to_move = game_state[0]
    king_bb_index = 5 if side_to_move == 0 else 11
    king_sq = get_lsb_index(piece_bbs[king_bb_index])

    # Check if the king's square is attacked by the opponent / 檢查王的方格是否被對手攻擊
    return is_square_attacked(piece_bbs, occupancy_bbs, game_state, king_sq, 1 - side_to_move)

PERFT_RESULTS = {
    "startpos": {
        1: 20, 2: 400, 3: 8902, 4: 197281, 5: 4865609, 6: 119060324
    },
    "kiwipete": {
        1: 48, 2: 2039, 3: 97862, 4: 4085603, 5: 193690690
    },
    "Position 3": {
        1: 14, 2: 191, 3: 2812, 4: 43238, 5: 674624, 6: 11030083, 7: 178633661
    },
    "Position 4": {
        1: 6, 2: 264, 3: 9467, 4: 422333, 5: 15833292
    },
    "Position 5": {
        1: 44, 2: 1486, 3: 62379, 4: 2103487, 5: 89941194
    }
}
"""
標準 Perft 結果，用於驗證。
"""

SQUARE_TO_ALGEBRAIC = {i: f"{chr(ord('a') + i % 8)}{i // 8 + 1}" for i in range(64)}
"""
將方格索引映射到代數記號的字典。
"""

@numba.jit(nbt.uint64(piece_bbs_signature, occupancy_bbs_signature, game_state_signature, nbt.intc), nopython=True)
def perft(piece_bbs, occupancy_bbs, game_state, depth: int):
    """
    核心遞歸 Perft 函數，重構為 Make-Unmake 模式。
    計算給定深度下的節點總數（走法總數）。

    Args:
        piece_bbs (np.ndarray): 棋子位元棋盤。
        occupancy_bbs (np.ndarray): 佔用位元棋盤。
        game_state (np.ndarray): 遊戲狀態陣列。
        depth (int): 剩餘深度。

    Returns:
        np.uint64: 節點總數。
    """
    if depth == 0:
        return np.uint64(1)

    nodes = np.uint64(0)
    moves = generate_legal_moves(piece_bbs, occupancy_bbs, game_state)

    if depth == 1:
        return np.uint64(len(moves))

    for i in range(len(moves)):
        move = moves[i]
        unmake_info = make_move(piece_bbs, occupancy_bbs, game_state, move)
        nodes += perft(piece_bbs, occupancy_bbs, game_state, depth - 1)
        unmake_move(piece_bbs, occupancy_bbs, game_state, move, unmake_info)
    return nodes

@numba.jit(nbt.types.Array(nbt.uint64, 2, "C")(piece_bbs_signature, occupancy_bbs_signature, game_state_signature, nbt.intc), nopython=True)
def _jit_perft_divide(piece_bbs, occupancy_bbs, game_state, depth: int):
    """
    JIT 編譯的核心 Perft Divide 邏輯，重構為 Make-Unmake 模式。
    計算第一步每個合法走法的子節點數。

    Args:
        piece_bbs (np.ndarray): 棋子位元棋盤。
        occupancy_bbs (np.ndarray): 佔用位元棋盤。
        game_state (np.ndarray): 遊戲狀態陣列。
        depth (int): 搜尋深度。

    Returns:
        np.ndarray: 形狀為 (N, 2) 的二維陣列，每行包含 [移動, 節點數]。
    """
    if depth == 0:
        return np.zeros((0, 2), dtype=np.uint64)

    moves = generate_legal_moves(piece_bbs, occupancy_bbs, game_state)

    results = np.zeros((len(moves), 2), dtype=np.uint64)
    for i in range(len(moves)):
        move = moves[i]
        unmake_info = make_move(piece_bbs, occupancy_bbs, game_state, move)
        nodes = perft(piece_bbs, occupancy_bbs, game_state, depth - 1)
        unmake_move(piece_bbs, occupancy_bbs, game_state, move, unmake_info)

        results[i, 0] = move
        results[i, 1] = nodes

    return results
