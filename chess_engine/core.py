
import numba as nb
import numpy as np

from .board_operations import make_move
from .move_generator import generate_legal_moves, generate_pseudo_legal_moves, is_square_attacked
from .engine_types import piece_bbs_signature, occupancy_bbs_signature, game_state_signature
from .bitboard_utils import get_ls1b_index
import numba.types as nbt

# --- Constants ---
WHITE, BLACK = 0, 1

# --- Standard Perft Results for Verification ---
PERFT_RESULTS = {
    "startpos": {
        1: 20, 2: 400, 3: 8902, 4: 197281, 5: 4865609, 6: 119060324
    },
    "kiwipete": {
        1: 48, 2: 2039, 3: 97862, 4: 4085603, 5: 193690690
    },
}

SQUARE_TO_ALGEBRAIC = {i: f"{chr(ord('a') + i % 8)}{i // 8 + 1}" for i in range(64)}

@nb.jit(nbt.uint64(piece_bbs_signature, occupancy_bbs_signature, game_state_signature, nbt.intc), nopython=True)
def perft(piece_bbs, occupancy_bbs, game_state, depth: int):
    """
    Core recursive Perft function.
    Counts the total number of legal moves to a given depth.
    """
    side_to_move = game_state[0]
    castling_rights = game_state[1]
    en_passant_square = np.int8(game_state[2])

    board_state_flat = piece_bbs + occupancy_bbs + (castling_rights, en_passant_square, side_to_move)
    moves = generate_legal_moves(board_state_flat)

    if depth == 1:
        return np.uint64(len(moves))

    nodes = np.uint64(0)
    for i in range(len(moves)):
        move = moves[i]
        new_piece_bbs, new_occupancy_bbs, new_game_state, _ = make_move(piece_bbs, occupancy_bbs, game_state, move)
        nodes += perft(new_piece_bbs, new_occupancy_bbs, new_game_state, depth - 1)
    return nodes

@nb.jit(nbt.types.Array(nbt.uint64, 2, "C")(piece_bbs_signature, occupancy_bbs_signature, game_state_signature, nbt.intc), nopython=True)
def _jit_perft_divide(piece_bbs, occupancy_bbs, game_state, depth: int):
    """
    JIT-compiled core logic for perft_divide.
    Returns an array of moves and their corresponding node counts.
    """
    if depth == 0:
        return np.zeros((0, 2), dtype=np.uint64)

    side_to_move = game_state[0]
    castling_rights = game_state[1]
    en_passant_square = np.int8(game_state[2])

    board_state_flat = piece_bbs + occupancy_bbs + (castling_rights, en_passant_square, side_to_move)
    moves = generate_pseudo_legal_moves(board_state_flat)

    results = np.zeros((len(moves), 2), dtype=np.uint64)
    count = 0
    for i in range(len(moves)):
        move = moves[i]
        new_piece_bbs, new_occupancy_bbs, new_game_state, _ = make_move(piece_bbs, occupancy_bbs, game_state, move)

        king_bb = new_piece_bbs[5] if side_to_move == WHITE else new_piece_bbs[11]
        if king_bb == 0:
            continue

        new_side_to_move = new_game_state[0]
        new_castling_rights = new_game_state[1]
        new_en_passant_square = np.int8(new_game_state[2])

        king_sq = get_ls1b_index(king_bb)
        new_board_state_flat = new_piece_bbs + new_occupancy_bbs + (new_castling_rights, new_en_passant_square, new_side_to_move)
        opponent_color = 1 - side_to_move

        if not is_square_attacked(king_sq, opponent_color, new_board_state_flat):
            nodes = perft(new_piece_bbs, new_occupancy_bbs, new_game_state, depth - 1)
            results[count, 0] = move
            results[count, 1] = nodes
            count += 1
    return results[:count]
