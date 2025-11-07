# chess_engine/search.py

import numba
import numpy as np

from chess_engine.evaluation import evaluate_position
from chess_engine.move_generator import generate_legal_moves, get_ls1b_index, is_square_attacked, generate_tactical_moves
from chess_engine.board_operations import make_move, find_piece_type_for_square
from chess_engine.move import get_to_square, get_from_square
import numba as nb
from chess_engine.constants import NEG_INFINITY, POS_INFINITY


NO_MOVE = 0 # Represents an invalid or null move

# --- Basic Engine Constants ---
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 0, 1, 2, 3, 4, 5
WHITE, BLACK = 0, 1


from chess_engine.engine_types import piece_bbs_signature, occupancy_bbs_signature, game_state_signature


# Define the return type for the _search function
search_return_type = nb.types.Tuple([
    nb.int32,   # evaluation
    nb.uint16,  # best_move
    nb.uint64,  # nodes_searched
    nb.uint64   # cutoffs
])

quiescence_search_return_type = nb.types.Tuple([
    nb.int32,   # evaluation
    nb.uint64,  # nodes_searched
    nb.uint64   # cutoffs
])

PIECE_VALUES = np.array([100, 320, 330, 500, 975, 20000], dtype=np.int32)

@nb.njit(quiescence_search_return_type(piece_bbs_signature, occupancy_bbs_signature, game_state_signature, nb.int32, nb.int32), cache=True)
def quiescence_search(piece_bbs, occupancy_bbs, game_state, alpha, beta):
    nodes_searched = np.uint64(1)
    cutoffs = np.uint64(0)
    
    stand_pat_score = evaluate_position(piece_bbs, occupancy_bbs, game_state)

    if stand_pat_score >= beta:
        return beta, nodes_searched, cutoffs

    alpha = max(alpha, stand_pat_score)

    tactical_moves = generate_tactical_moves(piece_bbs, occupancy_bbs, game_state)
    
    # --- Iterate through tactical moves (unsorted, as per original request) ---
    for move in tactical_moves:
        new_piece_bbs, new_occupancy_bbs, new_game_state, _ = make_move(
            piece_bbs, occupancy_bbs, game_state, move
        )
        
        side_to_move = game_state[0]
        king_bb = new_piece_bbs[5] if side_to_move == WHITE else new_piece_bbs[11]
        
        if king_bb == 0: continue
            
        king_sq = get_ls1b_index(king_bb)
        if is_square_attacked(new_piece_bbs, new_occupancy_bbs, new_game_state, king_sq, new_game_state[0]):
            continue

        score, child_nodes, child_cutoffs = quiescence_search(new_piece_bbs, new_occupancy_bbs, new_game_state, -beta, -alpha)
        nodes_searched += child_nodes
        cutoffs += child_cutoffs
        score = -score

        if score >= beta:
            cutoffs += 1
            return beta, nodes_searched, cutoffs

        alpha = max(alpha, score)

    return alpha, nodes_searched, cutoffs


@numba.njit(search_return_type(piece_bbs_signature, occupancy_bbs_signature, game_state_signature, nb.int32, nb.int32, nb.int32), cache=True)
def _search(piece_bbs, occupancy_bbs, game_state, depth, alpha, beta):
    nodes_searched = np.uint64(1)
    cutoffs = np.uint64(0)

    if depth == 0:
        eval_score, q_nodes, q_cutoffs = quiescence_search(piece_bbs, occupancy_bbs, game_state, alpha, beta)
        nodes_searched += q_nodes
        cutoffs += q_cutoffs
        return eval_score, NO_MOVE, nodes_searched, cutoffs

    moves = generate_legal_moves(piece_bbs, occupancy_bbs, game_state)

    if len(moves) == 0:
        side_to_move = game_state[0]
        king_bb = piece_bbs[5] if side_to_move == WHITE else piece_bbs[11]
        king_sq = get_ls1b_index(king_bb)

        if is_square_attacked(piece_bbs, occupancy_bbs, game_state, king_sq, 1 - side_to_move):
            return NEG_INFINITY, NO_MOVE, nodes_searched, cutoffs
        else:
            return np.int32(0), NO_MOVE, nodes_searched, cutoffs

    # Simple move ordering for the main search (as it was before)
    move_scores = np.zeros(len(moves), dtype=np.int32)
    opponent_color = 1 - game_state[0]
    for i, move in enumerate(moves):
        to_sq = get_to_square(move)
        victim_piece_type = find_piece_type_for_square(piece_bbs, to_sq, opponent_color)
        if victim_piece_type != -1:
             move_scores[i] = PIECE_VALUES[victim_piece_type]

    sorted_indices = np.argsort(move_scores)[::-1]
    
    best_move = NO_MOVE
    max_eval = NEG_INFINITY

    for i in sorted_indices:
        move = moves[i]
        
        new_piece_bbs, new_occupancy_bbs, new_game_state, _ = make_move(
            piece_bbs, occupancy_bbs, game_state, move
        )

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
            break
            
    # If no move improved alpha (e.g., all moves led to immediate mate), return the first legal move.
    if best_move == NO_MOVE and len(moves) > 0:
        best_move = moves[sorted_indices[0]]

    return max_eval, best_move, nodes_searched, cutoffs

numba_entry_return_type = nb.types.Tuple([ nb.uint16, nb.int32, nb.uint64, nb.uint64 ])

@numba.njit(numba_entry_return_type(piece_bbs_signature, occupancy_bbs_signature, game_state_signature, nb.int32), cache=True)
def _find_best_move_numba(piece_bbs, occupancy_bbs, game_state, max_depth):
    best_eval, best_move, nodes_searched, cutoffs = _search(
        piece_bbs, occupancy_bbs, game_state, max_depth, NEG_INFINITY, POS_INFINITY
    )
    return best_move, best_eval, nodes_searched, cutoffs

def search_position(board_state, max_depth):
    piece_bbs = board_state[:12]
    occupancy_bbs = board_state[12:15]
    side_to_move, castling_rights, en_passant_square, halfmove_clock, zobrist_key = board_state[15:]
    game_state = (side_to_move, castling_rights, en_passant_square, halfmove_clock, zobrist_key)

    best_move, best_eval, nodes_searched, cutoffs = _find_best_move_numba(
        piece_bbs, occupancy_bbs, game_state, max_depth
    )
    
    return best_move, best_eval, nodes_searched, cutoffs
