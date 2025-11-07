# chess_engine/search.py

import numba
import numpy as np

from chess_engine.evaluation import evaluate_position
from chess_engine.move_generator import generate_legal_moves, get_ls1b_index, is_square_attacked
from chess_engine.board_operations import make_move
from chess_engine.move import get_to_square, get_special_move_flag, SPECIAL_MOVE_FLAG_EN_PASSANT
import numba as nb
from chess_engine.constants import NEG_INFINITY, POS_INFINITY, MAX_QUIESCENCE_DEPTH

# Import TT components
from chess_engine.transposition_table import (
    probe_tt, store_tt, numba_tt_entry_type,
    TT_FLAG_NONE, TT_FLAG_EXACT, TT_FLAG_ALPHA, TT_FLAG_BETA
)

NO_MOVE = 0 # Represents an invalid or null move

from chess_engine.engine_types import piece_bbs_signature, occupancy_bbs_signature, game_state_signature

# Define the Numba type for the transposition table
tt_signature = nb.types.Array(numba_tt_entry_type, 1, 'C')

quiescence_search_return_type = nb.types.Tuple([
    nb.int32,  # evaluation
    nb.uint64  # q_nodes
])

@numba.njit(quiescence_search_return_type(piece_bbs_signature, occupancy_bbs_signature, game_state_signature, nb.int32, nb.int32, nb.int32), cache=True)
def quiescence_search(piece_bbs, occupancy_bbs, game_state, alpha, beta, ply):
    q_nodes = np.uint64(1)

    if ply >= MAX_QUIESCENCE_DEPTH:
        return evaluate_position(piece_bbs, occupancy_bbs, game_state), q_nodes

    stand_pat = evaluate_position(piece_bbs, occupancy_bbs, game_state)
    if stand_pat >= beta:
        return beta, q_nodes
    alpha = max(alpha, stand_pat)

    moves = generate_legal_moves(piece_bbs, occupancy_bbs, game_state)
    
    # Move Ordering (captures only)
    capture_moves = []
    opponent_pieces = occupancy_bbs[1] if game_state[0] == 0 else occupancy_bbs[0]
    for move in moves:
        to_sq = get_to_square(move)
        # Check if the destination square has an opponent's piece or if it's an en passant capture
        is_capture = ((opponent_pieces >> to_sq) & 1) != 0
        is_en_passant = get_special_move_flag(move) == SPECIAL_MOVE_FLAG_EN_PASSANT

        if is_capture or is_en_passant:
            capture_moves.append(move)

    if not capture_moves:
        return stand_pat, q_nodes

    for move in capture_moves:
        new_piece_bbs, new_occupancy_bbs, new_game_state, _ = make_move(
            piece_bbs, occupancy_bbs, game_state, move
        )
        score, child_q_nodes = quiescence_search(
            new_piece_bbs, new_occupancy_bbs, new_game_state, -beta, -alpha, ply + 1
        )
        q_nodes += child_q_nodes
        score = -score

        if score >= beta:
            return beta, q_nodes
        alpha = max(alpha, score)

    return alpha, q_nodes


# Define the return type for the _search function
search_return_type = nb.types.Tuple([
    nb.int32,   # evaluation
    nb.uint16,  # best_move
    nb.uint64,  # nodes_searched
    nb.uint64,   # quiescence_nodes
    nb.uint64   # cutoffs
])

@numba.njit(search_return_type(
    piece_bbs_signature, occupancy_bbs_signature, game_state_signature, 
    nb.int32, nb.int32, nb.int32, tt_signature
), cache=True)
def _search(piece_bbs, occupancy_bbs, game_state, depth, alpha, beta, transposition_table):
    """
    The core negamax search function with alpha-beta pruning and transposition table integration.
    """
    nodes_searched = np.uint64(1)
    quiescence_nodes = np.uint64(0)
    cutoffs = np.uint64(0)

    # --- 1. Transposition Table Probe ---
    original_alpha = alpha
    zobrist_key = game_state[4]  # Zobrist key is the 5th element (index 4)

    tt_entry = probe_tt(transposition_table, zobrist_key)
    if tt_entry['flag'] != TT_FLAG_NONE and tt_entry['depth'] >= depth:
        if tt_entry['flag'] == TT_FLAG_EXACT:
            return tt_entry['score'], tt_entry['best_move'], nodes_searched, quiescence_nodes, cutoffs
        elif tt_entry['flag'] == TT_FLAG_ALPHA:
            beta = min(beta, tt_entry['score'])
        elif tt_entry['flag'] == TT_FLAG_BETA:
            alpha = max(alpha, tt_entry['score'])

        if alpha >= beta:
            return tt_entry['score'], tt_entry['best_move'], nodes_searched, quiescence_nodes, cutoffs

    # --- 2. Base Case & Quiescence Search ---
    if depth == 0:
        eval_score, q_nodes = quiescence_search(piece_bbs, occupancy_bbs, game_state, alpha, beta, 0)
        return eval_score, NO_MOVE, nodes_searched, q_nodes, cutoffs

    # --- 3. Move Generation & Mate/Stalemate Check ---
    moves = generate_legal_moves(piece_bbs, occupancy_bbs, game_state)
    if len(moves) == 0:
        side_to_move = game_state[0]
        king_bb = piece_bbs[5] if side_to_move == 0 else piece_bbs[11]
        king_sq = get_ls1b_index(king_bb)
        if is_square_attacked(piece_bbs, occupancy_bbs, game_state, king_sq, 1 - side_to_move):
            return NEG_INFINITY, NO_MOVE, nodes_searched, quiescence_nodes, cutoffs  # Checkmate
        else:
            return np.int32(0), NO_MOVE, nodes_searched, quiescence_nodes, cutoffs  # Stalemate

    # --- 4. Move Ordering (Simple version: captures first) ---
    move_scores = np.zeros(len(moves), dtype=np.int32)
    all_pieces_bb = occupancy_bbs[0] | occupancy_bbs[1]
    for i, move in enumerate(moves):
        to_sq = get_to_square(move)
        if (all_pieces_bb >> to_sq) & 1:
            move_scores[i] = 100
    sorted_indices = np.argsort(move_scores)[::-1]
    
    # --- 5. Iterate through moves and search ---
    best_move = NO_MOVE
    max_eval = NEG_INFINITY

    for i in range(len(sorted_indices)):
        move = moves[sorted_indices[i]]
        new_piece_bbs, new_occupancy_bbs, new_game_state, _ = make_move(
            piece_bbs, occupancy_bbs, game_state, move
        )
        evaluation, _, child_nodes, child_q_nodes, child_cutoffs = _search(
            new_piece_bbs, new_occupancy_bbs, new_game_state, depth - 1, -beta, -alpha, transposition_table
        )
        nodes_searched += child_nodes
        quiescence_nodes += child_q_nodes
        cutoffs += child_cutoffs
        evaluation = -evaluation

        if evaluation > max_eval:
            max_eval = evaluation
            best_move = move

        alpha = max(alpha, evaluation)
        if alpha >= beta:
            cutoffs += 1
            break  # Beta cutoff

    # --- 6. Transposition Table Store ---
    if max_eval <= original_alpha:
        final_flag = TT_FLAG_ALPHA
    elif max_eval >= beta:
        final_flag = TT_FLAG_BETA
    else:
        final_flag = TT_FLAG_EXACT
    
    if best_move == NO_MOVE and len(moves) > 0:
        best_move = moves[sorted_indices[0]]

    store_tt(transposition_table, zobrist_key, depth, max_eval, final_flag, best_move)

    return max_eval, best_move, nodes_searched, quiescence_nodes, cutoffs

# Define the return type for the numba entry point
numba_entry_return_type = nb.types.Tuple([
    nb.uint16,  # best_move
    nb.int32,   # best_eval
    nb.uint64,  # nodes_searched
    nb.uint64,  # quiescence_nodes
    nb.uint64   # cutoffs
])

@numba.njit(numba_entry_return_type(
    piece_bbs_signature, occupancy_bbs_signature, game_state_signature, 
    nb.int32, tt_signature
), cache=True)
def _find_best_move_numba(piece_bbs, occupancy_bbs, game_state, max_depth, transposition_table):
    """
    A Numba-compatible entry point for the search.
    """
    best_eval, best_move, nodes_searched, quiescence_nodes, cutoffs = _search(
        piece_bbs, occupancy_bbs, game_state, max_depth, NEG_INFINITY, POS_INFINITY, transposition_table
    )
    return best_move, best_eval, nodes_searched, quiescence_nodes, cutoffs

def search_position(board_state, max_depth, transposition_table):
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
    best_move, best_eval, nodes_searched, quiescence_nodes, cutoffs = _find_best_move_numba(
        piece_bbs, occupancy_bbs, game_state, max_depth, transposition_table
    )
    
    return best_move, best_eval, nodes_searched, quiescence_nodes, cutoffs
