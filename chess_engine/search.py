# chess_engine/search.py

import numba
import numpy as np

from chess_engine.evaluation import evaluate_position
from chess_engine.move_generator import generate_legal_moves, get_ls1b_index, is_square_attacked
from chess_engine.board_operations import make_move
from chess_engine.move import get_to_square
import numba as nb
from chess_engine.constants import NEG_INFINITY, POS_INFINITY


NO_MOVE = 0 # Represents an invalid or null move

from chess_engine.engine_types import piece_bbs_signature, occupancy_bbs_signature, game_state_signature


# Define the return type for the _search function
search_return_type = nb.types.Tuple([
    nb.int32,   # evaluation
    nb.uint16,  # best_move
    nb.uint64,  # nodes_searched
    nb.uint64   # cutoffs
])

@numba.njit(search_return_type(piece_bbs_signature, occupancy_bbs_signature, game_state_signature, nb.int32, nb.int32, nb.int32), cache=True)
def _search(piece_bbs, occupancy_bbs, game_state, depth, alpha, beta):
    """
    The core negamax search function with alpha-beta pruning.
    This function is fully Numba-compatible and uses the refactored board state.
    """
    nodes_searched = np.uint64(1)
    cutoffs = np.uint64(0)

    # 1. Base Case: If we've reached the desired depth, evaluate the position.
    if depth == 0:
        eval_score = evaluate_position(piece_bbs, occupancy_bbs, game_state)
        return eval_score, NO_MOVE, nodes_searched, cutoffs

    # 2. Generate all legal moves for the current position.
    moves = generate_legal_moves(piece_bbs, occupancy_bbs, game_state)

    # If there are no legal moves, it's either checkmate or stalemate.
    if len(moves) == 0:
        side_to_move = game_state[0]
        king_bb = piece_bbs[5] if side_to_move == 0 else piece_bbs[11]
        king_sq = get_ls1b_index(king_bb)

        if is_square_attacked(piece_bbs, occupancy_bbs, game_state, king_sq, 1 - side_to_move):
            # CHECKMATE: Loss for the current player.
            return NEG_INFINITY, NO_MOVE, nodes_searched, cutoffs
        else:
            # STALEMATE: Draw.
            return np.int32(0), NO_MOVE, nodes_searched, cutoffs

    # 3. Move Ordering (Simple version: captures first)
    move_scores = np.zeros(len(moves), dtype=np.int32)
    all_pieces_bb = occupancy_bbs[0] | occupancy_bbs[1]
    for i, move in enumerate(moves):
        to_sq = get_to_square(move)
        if (all_pieces_bb >> to_sq) & 1:
            move_scores[i] = 100

    sorted_indices = np.argsort(move_scores)[::-1]
    
    # 4. Iterate through sorted legal moves and perform the search.
    best_move = moves[sorted_indices[0]]
    max_eval = NEG_INFINITY

    for i in range(len(sorted_indices)):
        move = moves[sorted_indices[i]]
        
        # Make the move on new board state representations
        new_piece_bbs, new_occupancy_bbs, new_game_state, _ = make_move(
            piece_bbs, occupancy_bbs, game_state, move
        )

        # Recursive call with negated alpha/beta
        evaluation, _, child_nodes, child_cutoffs = _search(
            new_piece_bbs, new_occupancy_bbs, new_game_state, depth - 1, -beta, -alpha
        )
        nodes_searched += child_nodes
        cutoffs += child_cutoffs
        evaluation = -evaluation

        if evaluation > max_eval:
            max_eval = evaluation
            best_move = move

        alpha = max(alpha, evaluation)
        if alpha >= beta:
            cutoffs += 1
            break  # Beta cutoff

    return max_eval, best_move, nodes_searched, cutoffs

# Define the return type for the numba entry point
numba_entry_return_type = nb.types.Tuple([
    nb.uint16,  # best_move
    nb.int32,   # best_eval
    nb.uint64,  # nodes_searched
    nb.uint64   # cutoffs
])

@numba.njit(numba_entry_return_type(piece_bbs_signature, occupancy_bbs_signature, game_state_signature, nb.int32), cache=True)
def _find_best_move_numba(piece_bbs, occupancy_bbs, game_state, max_depth):
    """
    A Numba-compatible entry point for the search.
    """
    best_eval, best_move, nodes_searched, cutoffs = _search(
        piece_bbs, occupancy_bbs, game_state, max_depth, NEG_INFINITY, POS_INFINITY
    )
    return best_move, best_eval, nodes_searched, cutoffs

def search_position(board_state, max_depth):
    """
    The top-level function to find the best move. This is the main interface.
    It unpacks the board_state tuple and calls the Numba-compiled function.
    """
    # Unpack the board state into the structures Numba expects
    piece_bbs = board_state[:12]
    occupancy_bbs = board_state[12:15]
    side_to_move, castling_rights, en_passant_square, halfmove_clock, zobrist_key = board_state[15:]
    game_state = (side_to_move, castling_rights, en_passant_square, halfmove_clock, zobrist_key)

    # Call the JIT'd function
    best_move, best_eval, nodes_searched, cutoffs = _find_best_move_numba(
        piece_bbs, occupancy_bbs, game_state, max_depth
    )
    
    return best_move, best_eval, nodes_searched, cutoffs
