
import numba as nb
import numpy as np

from .board_operations import make_move
from .move_generator import generate_legal_moves
from .engine_types import piece_bbs_signature, occupancy_bbs_signature, game_state_signature
import numba.types as nbt

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
Standard Perft results for verification.
"""

SQUARE_TO_ALGEBRAIC = {i: f"{chr(ord('a') + i % 8)}{i // 8 + 1}" for i in range(64)}
"""
A dictionary that maps square indices to algebraic notation.
"""

@nb.jit(nbt.uint64(piece_bbs_signature, occupancy_bbs_signature, game_state_signature, nbt.intc), nopython=True)
def perft(piece_bbs, occupancy_bbs, game_state, depth: int):
    """
    Core recursive Perft function.

    Args:
        piece_bbs: A tuple of 12 bitboards representing the pieces.
        occupancy_bbs: A tuple of 3 bitboards representing the occupancy of the board.
        game_state: A tuple representing the current game state.
        depth: The depth to run the test to.

    Returns:
        The number of legal moves to a given depth.
    """
    if depth == 0:
        return np.uint64(1)

    nodes = np.uint64(0)
    side_to_move = game_state[0]
    castling_rights = game_state[1]
    en_passant_square = np.int8(game_state[2])

    board_state_flat = piece_bbs + occupancy_bbs + (castling_rights, en_passant_square, side_to_move)
    moves = generate_legal_moves(board_state_flat)

    if depth == 1:
        return np.uint64(len(moves))

    for i in range(len(moves)):
        move = moves[i]
        new_piece_bbs, new_occupancy_bbs, new_game_state, _ = make_move(
            piece_bbs, occupancy_bbs, game_state, move
        )
        nodes += perft(new_piece_bbs, new_occupancy_bbs, new_game_state, depth - 1)
    return nodes

@nb.jit(nbt.types.Array(nbt.uint64, 2, "C")(piece_bbs_signature, occupancy_bbs_signature, game_state_signature, nbt.intc), nopython=True)
def _jit_perft_divide(piece_bbs, occupancy_bbs, game_state, depth: int):
    """
    JIT-compiled core logic for perft_divide.

    Args:
        piece_bbs: A tuple of 12 bitboards representing the pieces.
        occupancy_bbs: A tuple of 3 bitboards representing the occupancy of the board.
        game_state: A tuple representing the current game state.
        depth: The depth to run the test to.

    Returns:
        An array of moves and their corresponding node counts.
    """
    if depth == 0:
        return np.zeros((0, 2), dtype=np.uint64)

    board_state_flat = piece_bbs + occupancy_bbs + (game_state[1], np.int8(game_state[2]), game_state[0])
    moves = generate_legal_moves(board_state_flat)

    results = np.zeros((len(moves), 2), dtype=np.uint64)
    for i in range(len(moves)):
        move = moves[i]
        new_piece_bbs, new_occupancy_bbs, new_game_state, _ = make_move(
            piece_bbs, occupancy_bbs, game_state, move
        )
        nodes = perft(new_piece_bbs, new_occupancy_bbs, new_game_state, depth - 1)
        results[i, 0] = move
        results[i, 1] = nodes

    return results
