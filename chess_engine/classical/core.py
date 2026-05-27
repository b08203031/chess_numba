import numba
import numpy as np

from .board_operations import make_move, unmake_move
from .move_generator import generate_legal_moves_buffer, is_square_attacked, generate_legal_moves
from .engine_types import piece_bbs_signature, occupancy_bbs_signature, game_state_signature
import numba.types as nbt
from .zobrist import get_lsb_index

@numba.njit(cache=True)
def _is_king_in_check_jit(piece_bbs, occupancy_bbs, game_state):
    """
    檢查當前行棋方的王是否處於被將軍狀態。
    """
    side_to_move = game_state[0]
    king_bb_index = 5 if side_to_move == 0 else 11
    king_sq = get_lsb_index(piece_bbs[king_bb_index])

    # Check if the king's square is attacked by the opponent
    return is_square_attacked(piece_bbs, occupancy_bbs, king_sq, 1 - side_to_move)


def is_king_in_check(piece_bbs, occupancy_bbs, game_state):
    return _is_king_in_check_jit(piece_bbs, occupancy_bbs, game_state)


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

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def _perft_jit(piece_bbs, occupancy_bbs, game_state, depth: int):
    """
    核心遞歸 Perft 函數，使用 preallocated 緩衝區。
    """
    if depth == 0:
        return np.uint64(1)

    moves = np.zeros((1, 256), dtype=np.uint16)
    count = generate_legal_moves_buffer(piece_bbs, occupancy_bbs, game_state, moves, 0)

    if depth == 1:
        return np.uint64(count)

    nodes = np.uint64(0)
    for i in range(count):
        move = moves[0, i]
        unmake_info = make_move(piece_bbs, occupancy_bbs, game_state, move)
        nodes += _perft_jit(piece_bbs, occupancy_bbs, game_state, depth - 1)
        unmake_move(piece_bbs, occupancy_bbs, game_state, move, unmake_info)
    return nodes


def perft(piece_bbs, occupancy_bbs, game_state, depth: int):
    return _perft_jit(piece_bbs, occupancy_bbs, game_state, depth)


@numba.njit(cache=True, boundscheck=False, fastmath=True)
def _perft_divide_jit(piece_bbs, occupancy_bbs, game_state, depth: int):
    """
    JIT 編編的核心 Perft Divide 邏輯。
    """
    if depth == 0:
        return np.zeros((0, 2), dtype=np.uint64)

    moves = np.zeros((1, 256), dtype=np.uint16)
    count = generate_legal_moves_buffer(piece_bbs, occupancy_bbs, game_state, moves, 0)

    results = np.zeros((count, 2), dtype=np.uint64)
    for i in range(count):
        move = moves[0, i]
        unmake_info = make_move(piece_bbs, occupancy_bbs, game_state, move)
        nodes = _perft_jit(piece_bbs, occupancy_bbs, game_state, depth - 1)
        unmake_move(piece_bbs, occupancy_bbs, game_state, move, unmake_info)

        results[i, 0] = move
        results[i, 1] = nodes

    return results


def _jit_perft_divide(piece_bbs, occupancy_bbs, game_state, depth: int):
    return _perft_divide_jit(piece_bbs, occupancy_bbs, game_state, depth)
