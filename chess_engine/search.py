# chess_engine/search.py

import numba
import numpy as np

from chess_engine.evaluation import evaluate_position
from chess_engine.move_generator import generate_legal_moves, get_ls1b_index, is_square_attacked
from chess_engine.board_operations import make_move
from chess_engine.move import get_to_square

# A large number representing infinity for alpha-beta pruning
NEG_INFINITY = -999999
POS_INFINITY = 999999
NO_MOVE = 0 # Represents an invalid or null move

@numba.jit(nopython=True, cache=True)
def search(board_state, depth, alpha, beta):
    """
    The core negamax search function with alpha-beta pruning.
    This function is fully Numba-compatible.
    """
    # 1. Base Case: If we've reached the desired depth, evaluate the position.
    if depth == 0:
        return evaluate_position(board_state), NO_MOVE

    # 2. Generate all legal moves for the current position.
    moves = generate_legal_moves(board_state)

    # If there are no legal moves, it's either checkmate or stalemate.
    if len(moves) == 0:
        # Check if the king is currently in check
        side_to_move = board_state[17]
        king_bb = board_state[5] if side_to_move == 0 else board_state[11]
        king_sq = get_ls1b_index(king_bb)

        if is_square_attacked(king_sq, 1 - side_to_move, board_state):
            # CHECKMATE: The current player is in check and has no legal moves.
            # This is a loss for the current player. Return a very low score.
            return NEG_INFINITY, NO_MOVE
        else:
            # STALEMATE: The current player is not in check but has no legal moves.
            # This is a draw. Return a score of 0.
            return 0, NO_MOVE

    # 3. Move Ordering (Simple version: captures first)
    # A simple scoring system to sort moves. Higher score = better move to search first.
    move_scores = np.zeros(len(moves), dtype=np.int32)
    (_, _, _, _, _, _, _, _, _, _, _, _, _, _, all_pieces_bb, _, _, _, _) = board_state
    for i, move in enumerate(moves):
        to_sq = get_to_square(move)
        to_bb = np.uint64(1) << to_sq
        # If the destination square has a piece, it's a capture. Give it a higher score.
        if (all_pieces_bb & to_bb):
            move_scores[i] = 100 # Arbitrary high score for captures

    # Sort moves in descending order of their scores
    sorted_indices = np.argsort(move_scores)[::-1]
    sorted_moves = moves[sorted_indices]

    # 4. Iterate through sorted legal moves and perform the search.
    best_move = sorted_moves[0]
    max_eval = NEG_INFINITY

    for move in sorted_moves:
        # Create a new board state by making the move.
        new_board_state = make_move(board_state, move)

        # Recursively call search for the other player (hence the negative sign).
        # The result of the recursive call is from the opponent's perspective.
        evaluation, _ = search(new_board_state, depth - 1, -beta, -alpha)

        # Negate the evaluation to get it back to our perspective.
        evaluation = -evaluation

        # Update the best evaluation and best move.
        if evaluation > max_eval:
            max_eval = evaluation
            best_move = move

        # Alpha-beta pruning.
        alpha = max(alpha, evaluation)
        if alpha >= beta:
            break # Beta cutoff

    return max_eval, best_move

@numba.jit(nopython=True, cache=True)
def find_best_move_numba(board_state, max_depth):
    """
    A Numba-compatible entry point for the search.
    """
    # Start the search from the root node.
    best_eval, best_move = search(board_state, max_depth, NEG_INFINITY, POS_INFINITY)
    return best_move, best_eval

def find_best_move(board_state, max_depth):
    """
    The top-level function to find the best move. This is the main interface.
    It calls the Numba-compiled function.
    """
    # Numba's first call has a compilation overhead. Subsequent calls are fast.
    # We call the JIT'd function from this non-JIT'd wrapper.
    best_move, best_eval = find_best_move_numba(board_state, max_depth)

    # Here you could add logic to print search statistics, etc.
    print(f"Search complete. Best move found: {best_move}, Evaluation: {best_eval}")

    return best_move
