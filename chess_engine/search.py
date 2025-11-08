# chess_engine/search.py

import numba
import numpy as np

from chess_engine.evaluation import evaluate_position
from chess_engine.move_generator import (
    generate_legal_moves, is_square_attacked,
    is_in_check, has_sufficient_material
)
from chess_engine.zobrist import get_lsb_index
from chess_engine.board_operations import make_move, make_null_move
from chess_engine.move import get_to_square, get_from_square, get_special_move_flag, SPECIAL_MOVE_FLAG_EN_PASSANT
import numba as nb
from chess_engine.constants import (
    BB_SQUARES, MG_MATERIAL_VALUES, INFINITY, MAX_QUIESCENCE_DEPTH, ASPIRATION_WINDOW_SIZE,
    NULL_MOVE_REDUCTION
)
from chess_engine.bitboard_utils import find_piece_type_on_square
from chess_engine.debug_utils import log_info

# Import TT components
from chess_engine.transposition_table import (
    probe_tt, store_tt, numba_tt_entry_type,
    TT_FLAG_NONE, TT_FLAG_EXACT, TT_FLAG_ALPHA, TT_FLAG_BETA
)

NO_MOVE = 0 # Represents an invalid or null move

from chess_engine.engine_types import (
    piece_bbs_signature, occupancy_bbs_signature, game_state_signature,
    SearchContext, search_context_type
)


@nb.njit(cache=True)
def sort_moves(piece_bbs, occupancy_bbs, game_state, moves, tt_move, killer_moves_at_ply):
    """
    對棋步列表進行排序。
    返回一個根據啟發式評分排序後的新列表。
    """
    move_scores = np.zeros(len(moves), dtype=np.int32)
    opponent_pieces_bb = occupancy_bbs[1] if game_state[0] == 0 else occupancy_bbs[0]

    for i in range(len(moves)):
        move = moves[i]
        score = 0

        if move == tt_move:
            score += 100_000
            move_scores[i] = score
            continue

        to_square = get_to_square(move)
        is_capture = (opponent_pieces_bb & BB_SQUARES[to_square]) != 0
        is_en_passant = get_special_move_flag(move) == SPECIAL_MOVE_FLAG_EN_PASSANT

        if is_capture:
            aggressor_type = find_piece_type_on_square(piece_bbs, get_from_square(move))
            victim_type = find_piece_type_on_square(piece_bbs, to_square)
            if victim_type != -1: # Ensure a piece is actually on the square
                score += 10_000 + (MG_MATERIAL_VALUES[victim_type % 6] - MG_MATERIAL_VALUES[aggressor_type % 6])
        elif is_en_passant:
            # En passant is always Pawn captures Pawn
            aggressor_type = find_piece_type_on_square(piece_bbs, get_from_square(move))
            score += 10_000 + (MG_MATERIAL_VALUES[0] - MG_MATERIAL_VALUES[aggressor_type % 6])
        else:
            if move == killer_moves_at_ply[0]:
                score += 5_000
            elif move == killer_moves_at_ply[1]:
                score += 4_000
        
        move_scores[i] = score

    for i in range(1, len(moves)):
        key_move = moves[i]
        key_score = move_scores[i]
        j = i - 1
        while j >= 0 and move_scores[j] < key_score:
            moves[j + 1] = moves[j]
            move_scores[j + 1] = move_scores[j]
            j -= 1
        moves[j + 1] = key_move
        move_scores[j + 1] = key_score

    return moves

quiescence_search_return_type = nb.types.Tuple([
    nb.int32, nb.uint64
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
    
    capture_moves = []
    opponent_pieces_bb = occupancy_bbs[1] if game_state[0] == 0 else occupancy_bbs[0]
    for move in moves:
        to_sq = get_to_square(move)
        is_capture = (opponent_pieces_bb & BB_SQUARES[to_sq]) != 0
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

search_return_type = nb.types.Tuple([
    nb.int32, nb.uint16, nb.uint64, nb.uint64, nb.uint64
])

@numba.njit(search_return_type(
    piece_bbs_signature, occupancy_bbs_signature, game_state_signature, 
    nb.int32, nb.int32, nb.int32, search_context_type, nb.int32
), cache=True)
def _search(piece_bbs, occupancy_bbs, game_state, depth, alpha, beta, search_context, ply):
    nodes_searched = np.uint64(1)
    quiescence_nodes = np.uint64(0)
    cutoffs = np.uint64(0)

    original_alpha = alpha
    zobrist_key = game_state[4]

    tt_move = NO_MOVE
    tt_entry = probe_tt(search_context.transposition_table, zobrist_key)
    if tt_entry['flag'] != TT_FLAG_NONE and tt_entry['depth'] >= depth:
        if tt_entry['flag'] == TT_FLAG_EXACT:
            return tt_entry['score'], tt_entry['best_move'], nodes_searched, quiescence_nodes, cutoffs
        elif tt_entry['flag'] == TT_FLAG_ALPHA:
            beta = min(beta, tt_entry['score'])
        elif tt_entry['flag'] == TT_FLAG_BETA:
            alpha = max(alpha, tt_entry['score'])
        if alpha >= beta:
            return tt_entry['score'], tt_entry['best_move'], nodes_searched, quiescence_nodes, cutoffs
        tt_move = tt_entry['best_move']

    # --- Null Move Pruning (NMP) ---
    if depth >= 3 and not is_in_check(piece_bbs, occupancy_bbs, game_state) and has_sufficient_material(piece_bbs, game_state[0]):
        null_move_piece_bbs, null_move_occupancy_bbs, null_move_game_state = make_null_move(piece_bbs, occupancy_bbs, game_state)
        
        null_move_score, _, _, _, _ = _search(
            null_move_piece_bbs, null_move_occupancy_bbs, null_move_game_state,
            depth - 1 - NULL_MOVE_REDUCTION,
            -beta,
            -beta + 1,
            search_context,
            ply + 1
        )
        null_move_score = -null_move_score

        if null_move_score >= beta:
            return beta, NO_MOVE, nodes_searched, quiescence_nodes, cutoffs

    if depth == 0:
        eval_score, q_nodes = quiescence_search(piece_bbs, occupancy_bbs, game_state, alpha, beta, 0)
        return eval_score, NO_MOVE, nodes_searched, q_nodes, cutoffs

    moves = generate_legal_moves(piece_bbs, occupancy_bbs, game_state)
    if len(moves) == 0:
        side_to_move = game_state[0]
        king_bb = piece_bbs[5] if side_to_move == 0 else piece_bbs[11]
        king_sq = get_lsb_index(king_bb)
        if is_square_attacked(piece_bbs, occupancy_bbs, game_state, king_sq, 1 - side_to_move):
            return -INFINITY, NO_MOVE, nodes_searched, quiescence_nodes, cutoffs
        else:
            return np.int32(0), NO_MOVE, nodes_searched, quiescence_nodes, cutoffs

    sorted_moves = sort_moves(piece_bbs, occupancy_bbs, game_state, moves, tt_move, search_context.killer_moves[ply])
    
    best_move = NO_MOVE
    max_eval = -INFINITY

    for move in sorted_moves:
        new_piece_bbs, new_occupancy_bbs, new_game_state, _ = make_move(
            piece_bbs, occupancy_bbs, game_state, move
        )
        evaluation, _, child_nodes, child_q_nodes, child_cutoffs = _search(
            new_piece_bbs, new_occupancy_bbs, new_game_state, depth - 1, -beta, -alpha, search_context, ply + 1
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
            opponent_pieces_bb = occupancy_bbs[1] if game_state[0] == 0 else occupancy_bbs[0]
            is_capture = (opponent_pieces_bb & BB_SQUARES[get_to_square(move)]) != 0
            is_en_passant = get_special_move_flag(move) == SPECIAL_MOVE_FLAG_EN_PASSANT
            if not is_capture and not is_en_passant:
                if move != search_context.killer_moves[ply, 0]:
                    search_context.killer_moves[ply, 1] = search_context.killer_moves[ply, 0]
                    search_context.killer_moves[ply, 0] = move
            break

    if max_eval <= original_alpha:
        final_flag = TT_FLAG_ALPHA
    elif max_eval >= beta:
        final_flag = TT_FLAG_BETA
    else:
        final_flag = TT_FLAG_EXACT
    
    if best_move == NO_MOVE and len(moves) > 0:
        best_move = sorted_moves[0]

    store_tt(search_context.transposition_table, zobrist_key, depth, max_eval, final_flag, best_move)

    return max_eval, best_move, nodes_searched, quiescence_nodes, cutoffs

@numba.njit(search_return_type(
    piece_bbs_signature, occupancy_bbs_signature, game_state_signature, 
    nb.int32, nb.int32, nb.int32, numba.types.Array(numba_tt_entry_type, 1, 'C'), nb.types.Array(nb.uint16, 2, 'C')
), cache=True)
def _search_wrapper(piece_bbs, occupancy_bbs, game_state, depth, alpha, beta, transposition_table, killer_moves):
    search_context = SearchContext(transposition_table, killer_moves)
    return _search(
        piece_bbs, occupancy_bbs, game_state, depth, alpha, beta, search_context, 0
    )

def iterative_deepening_search(board_state, max_depth, max_time_ms, transposition_table, killer_moves):
    import time
    from .move import move_to_uci

    start_time = time.time()
    
    piece_bbs = board_state[:12]
    occupancy_bbs = board_state[12:15]
    side_to_move, castling_rights, en_passant_square, halfmove_clock, zobrist_key = board_state[15:]
    game_state = (side_to_move, castling_rights, en_passant_square, halfmove_clock, zobrist_key)
    
    last_score = 0
    best_move_total = NO_MOVE
    total_nodes_searched = 0
    total_quiescence_nodes = 0
    total_cutoffs = 0

    for current_depth in range(1, max_depth + 1):
        if current_depth > 1:
            alpha = last_score - ASPIRATION_WINDOW_SIZE
            beta = last_score + ASPIRATION_WINDOW_SIZE
        else:
            alpha, beta = -INFINITY, INFINITY

        nodes, q_nodes, cutoffs = 0, 0, 0
        score, _, nodes, q_nodes, cutoffs = _search_wrapper(piece_bbs, occupancy_bbs, game_state, current_depth, alpha, beta, transposition_table, killer_moves)

        if score <= alpha or score >= beta:
            log_info(f"depth {current_depth} aspiration window failed, re-searching...")
            alpha, beta = -INFINITY, INFINITY
            score, _, nodes, q_nodes, cutoffs = _search_wrapper(piece_bbs, occupancy_bbs, game_state, current_depth, alpha, beta, transposition_table, killer_moves)

        last_score = score
        total_nodes_searched += nodes
        total_quiescence_nodes += q_nodes
        total_cutoffs += cutoffs
        
        tt_entry = probe_tt(transposition_table, game_state[4])
        if tt_entry['flag'] != TT_FLAG_NONE:
            best_move_total = tt_entry['best_move']
            
        elapsed_time = (time.time() - start_time) * 1000
        log_info(f"depth {current_depth} score cp {score} time {int(elapsed_time)} pv {move_to_uci(best_move_total)}")

        if elapsed_time > max_time_ms:
            break
            
    return best_move_total, last_score, total_nodes_searched, total_quiescence_nodes, total_cutoffs
