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
from chess_engine.move import (
    get_to_square, get_from_square, get_special_move_flag, 
    SPECIAL_MOVE_FLAG_EN_PASSANT, SPECIAL_MOVE_FLAG_PROMOTION
)
import numba as nb
from chess_engine.constants import (
    BB_SQUARES, MG_MATERIAL_VALUES, INFINITY, MAX_QUIESCENCE_DEPTH, ASPIRATION_WINDOW_SIZE,
    NULL_MOVE_REDUCTION, MAX_PLY, LMR_MIN_DEPTH, LMR_MIN_QUIET_MOVE_INDEX, LMR_REDUCTION, SEE_THRESHOLD,
    MATE_SCORE, MATE_IN_MAX_PLY
)
from chess_engine.bitboard_utils import find_piece_type_on_square
from chess_engine.debug_utils import log_info
from chess_engine.see import see

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


def format_score_for_uci(score):
    """
    Converts the internal score to a UCI-compliant string.
    Handles centipawn scores and mate scores.
    """
    if abs(score) > MATE_IN_MAX_PLY:
        # It's a mate score
        if score > 0:
            # Mate in X for the engine
            moves_to_mate = MATE_SCORE - score
            mate_in_x = (moves_to_mate + 1) // 2
            return f"mate {mate_in_x}"
        else:
            # Mate in X for the opponent
            moves_to_mate = MATE_SCORE + score
            mate_in_x = (moves_to_mate + 1) // 2
            return f"mate -{mate_in_x}"
    else:
        # It's a centipawn score
        return f"cp {score}"


# chess_engine/search.py
# ... (imports)

@nb.njit(cache=True)
def partition(moves, scores, low, high):
    pivot_score = scores[high]
    i = low - 1
    for j in range(low, high):
        if scores[j] >= pivot_score: # Sort descending
            i += 1
            moves[i], moves[j] = moves[j], moves[i]
            scores[i], scores[j] = scores[j], scores[i]

    moves[i + 1], moves[high] = moves[high], moves[i + 1]
    scores[i + 1], scores[high] = scores[high], scores[i + 1]
    return i + 1

@nb.njit(cache=True)
def quicksort_recursive(moves, scores, low, high):
    if low < high:
        pi = partition(moves, scores, low, high)
        quicksort_recursive(moves, scores, low, pi - 1)
        quicksort_recursive(moves, scores, pi + 1, high)

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

    # <<< FIX: Replace insertion sort with Quicksort >>>
    if len(moves) > 1:
        quicksort_recursive(moves, move_scores, 0, len(moves) - 1)

    return moves

quiescence_search_return_type = nb.types.Tuple([
    nb.int32, nb.uint64
])

@numba.njit(quiescence_search_return_type(
    piece_bbs_signature, occupancy_bbs_signature, game_state_signature,
    nb.int32, nb.int32, nb.int32, nb.types.List(nb.uint16)
), cache=True)
def quiescence_search(piece_bbs, occupancy_bbs, game_state, alpha, beta, ply, legal_moves):
    q_nodes = np.uint64(1)

    if ply >= MAX_QUIESCENCE_DEPTH:
        return evaluate_position(piece_bbs, occupancy_bbs, game_state), q_nodes

    stand_pat = evaluate_position(piece_bbs, occupancy_bbs, game_state)
    if stand_pat >= beta:
        return beta, q_nodes
    alpha = max(alpha, stand_pat)
    
    capture_moves = []
    opponent_pieces_bb = occupancy_bbs[1] if game_state[0] == 0 else occupancy_bbs[0]
    for move in legal_moves:
        to_sq = get_to_square(move)
        is_capture = (opponent_pieces_bb & BB_SQUARES[to_sq]) != 0
        is_en_passant = get_special_move_flag(move) == SPECIAL_MOVE_FLAG_EN_PASSANT

        if is_capture or is_en_passant:
            capture_moves.append(move)

    if not capture_moves:
        return stand_pat, q_nodes
    
    for move in capture_moves:
        # SEE Pruning
        if see(piece_bbs, occupancy_bbs, game_state, get_from_square(move), get_to_square(move)) < SEE_THRESHOLD:
            continue

        new_piece_bbs, new_occupancy_bbs, new_game_state, _ = make_move(
            piece_bbs, occupancy_bbs, game_state, move
        )
        # Quiescence search in child nodes still needs to generate its own moves
        child_moves = generate_legal_moves(new_piece_bbs, new_occupancy_bbs, new_game_state)
        score, child_q_nodes = quiescence_search(
            new_piece_bbs, new_occupancy_bbs, new_game_state, -beta, -alpha, ply + 1, child_moves
        )
        q_nodes += child_q_nodes
        score = -score

        if score >= beta:
            return beta, q_nodes
        alpha = max(alpha, score)

    return alpha, q_nodes

search_return_type = nb.types.Tuple([
    nb.int32, nb.uint16, nb.uint64, nb.uint64, nb.uint64, nb.uint64,
    nb.uint64, nb.uint64, nb.uint64, nb.uint64, nb.uint64
])

@numba.njit(search_return_type(
    piece_bbs_signature, occupancy_bbs_signature, game_state_signature, 
    nb.int32, nb.int32, nb.int32, search_context_type, nb.int32
), cache=True)
def _search(piece_bbs, occupancy_bbs, game_state, depth, alpha, beta, search_context, ply):
    nodes_searched = np.uint64(1)
    quiescence_nodes = np.uint64(0)
    cutoffs = np.uint64(0)
    tt_hits = np.uint64(0)
    null_move_cutoffs = np.uint64(0)
    futility_pruned = np.uint64(0)
    razoring_used = np.uint64(0)
    qs_delta_pruned = np.uint64(0)
    qs_see_pruned = np.uint64(0)


    if ply >= MAX_PLY:
        return (evaluate_position(piece_bbs, occupancy_bbs, game_state), NO_MOVE, nodes_searched, quiescence_nodes, cutoffs, tt_hits,
                null_move_cutoffs, futility_pruned, razoring_used, qs_delta_pruned, qs_see_pruned)

    original_alpha = alpha
    zobrist_key = game_state[4]

    tt_move = NO_MOVE
    tt_entry = probe_tt(search_context.transposition_table, zobrist_key)
    if tt_entry['flag'] != TT_FLAG_NONE and tt_entry['depth'] >= depth:
        tt_hits += 1
        tt_score = np.int32(tt_entry['score'])

        # Adjust mate score from TT
        if tt_score > MATE_IN_MAX_PLY:
            tt_score += ply
        elif tt_score < -MATE_IN_MAX_PLY:
            tt_score -= ply

        if tt_entry['flag'] == TT_FLAG_EXACT:
            return (tt_score, tt_entry['best_move'], nodes_searched, quiescence_nodes, cutoffs, tt_hits,
                    null_move_cutoffs, futility_pruned, razoring_used, qs_delta_pruned, qs_see_pruned)
        elif tt_entry['flag'] == TT_FLAG_ALPHA:
            beta = min(beta, tt_score)
        elif tt_entry['flag'] == TT_FLAG_BETA:
            alpha = max(alpha, tt_score)

        if alpha >= beta:
            return (tt_score, tt_entry['best_move'], nodes_searched, quiescence_nodes, cutoffs, tt_hits,
                    null_move_cutoffs, futility_pruned, razoring_used, qs_delta_pruned, qs_see_pruned)

        tt_move = tt_entry['best_move']

    is_currently_in_check = is_in_check(piece_bbs, occupancy_bbs, game_state)

    # --- Null Move Pruning (NMP) ---
    if depth >= 3 and not is_currently_in_check and has_sufficient_material(piece_bbs, game_state[0]):
        null_move_piece_bbs, null_move_occupancy_bbs, null_move_game_state = make_null_move(piece_bbs, occupancy_bbs, game_state)
        
        # Note: The unpacking below needs to be updated to match the new return type
        null_move_score, _, _, _, _, child_tt_hits, _, _, _, _, _ = _search(
            null_move_piece_bbs, null_move_occupancy_bbs, null_move_game_state,
            depth - 1 - NULL_MOVE_REDUCTION,
            -beta, -beta + 1, search_context, ply + 1
        )
        null_move_score = -null_move_score
        tt_hits += child_tt_hits

        if null_move_score >= beta:
            null_move_cutoffs += 1
            return (np.int32(beta), NO_MOVE, nodes_searched, quiescence_nodes, cutoffs, tt_hits,
                    null_move_cutoffs, futility_pruned, razoring_used, qs_delta_pruned, qs_see_pruned)

    moves = generate_legal_moves(piece_bbs, occupancy_bbs, game_state)

    has_legal_moves = False
    for m in moves:
        if m != 0:
            has_legal_moves = True
            break

    if not has_legal_moves:
        if is_currently_in_check:
            # Checkmate
            return (np.int32(-MATE_SCORE + ply), NO_MOVE, nodes_searched, quiescence_nodes, cutoffs, tt_hits,
                    null_move_cutoffs, futility_pruned, razoring_used, qs_delta_pruned, qs_see_pruned)
        else:
            # Stalemate
            return (np.int32(0), NO_MOVE, nodes_searched, quiescence_nodes, cutoffs, tt_hits,
                    null_move_cutoffs, futility_pruned, razoring_used, qs_delta_pruned, qs_see_pruned)

    if depth == 0:
        eval_score, q_nodes = quiescence_search(piece_bbs, occupancy_bbs, game_state, alpha, beta, 0, moves)
        return (eval_score, NO_MOVE, nodes_searched, q_nodes, cutoffs, tt_hits,
                null_move_cutoffs, futility_pruned, razoring_used, qs_delta_pruned, qs_see_pruned)

    sorted_moves = sort_moves(piece_bbs, occupancy_bbs, game_state, moves, tt_move, search_context.killer_moves[ply])
    
    best_move = NO_MOVE
    max_eval = -INFINITY
    quiet_move_counter = 0

    for i, move in enumerate(sorted_moves):
        # --- Static Exchange Evaluation (SEE) Pruning ---
        opponent_pieces_bb = occupancy_bbs[1] if game_state[0] == 0 else occupancy_bbs[0]
        is_capture = (opponent_pieces_bb & BB_SQUARES[get_to_square(move)]) != 0
        
        if is_capture:
            if see(piece_bbs, occupancy_bbs, game_state, get_from_square(move), get_to_square(move)) < SEE_THRESHOLD:
                continue

        new_piece_bbs, new_occupancy_bbs, new_game_state, _ = make_move(
            piece_bbs, occupancy_bbs, game_state, move
        )
        
        is_giving_check = is_in_check(new_piece_bbs, new_occupancy_bbs, new_game_state)

        # --- Check Extensions ---
        search_depth = depth - 1
        if is_giving_check:
             search_depth += 1

        opponent_pieces_bb = occupancy_bbs[1] if game_state[0] == 0 else occupancy_bbs[0]
        is_capture = (opponent_pieces_bb & BB_SQUARES[get_to_square(move)]) != 0
        is_promotion = get_special_move_flag(move) == SPECIAL_MOVE_FLAG_PROMOTION
        is_quiet_move = not is_capture and not is_promotion and not is_giving_check

        evaluation = 0

        # --- Principal Variation Search (PVS) & Late Move Reductions (LMR) ---
        if i == 0:  # First move (PV node): Full window search
            evaluation, _, child_nodes, child_q_nodes, child_cutoffs, child_tt_hits, child_nmc, child_fp, child_ru, child_qdp, child_qsp = _search(
                new_piece_bbs, new_occupancy_bbs, new_game_state, search_depth, -beta, -alpha, search_context, ply + 1
            )
            nodes_searched += child_nodes; quiescence_nodes += child_q_nodes; cutoffs += child_cutoffs; tt_hits += child_tt_hits
            null_move_cutoffs += child_nmc; futility_pruned += child_fp; razoring_used += child_ru; qs_delta_pruned += child_qdp; qs_see_pruned += child_qsp
            evaluation = -evaluation
        else:  # Subsequent moves (Non-PV nodes): Zero-window search
            lmr_reduction = 0
            if depth >= LMR_MIN_DEPTH and is_quiet_move and quiet_move_counter >= LMR_MIN_QUIET_MOVE_INDEX:
                lmr_reduction = LMR_REDUCTION
            
            reduced_search_depth = search_depth - lmr_reduction

            # 1. Zero-window search with potential LMR
            evaluation, _, child_nodes, child_q_nodes, child_cutoffs, child_tt_hits, child_nmc, child_fp, child_ru, child_qdp, child_qsp = _search(
                new_piece_bbs, new_occupancy_bbs, new_game_state, reduced_search_depth, -alpha - 1, -alpha, search_context, ply + 1
            )
            nodes_searched += child_nodes; quiescence_nodes += child_q_nodes; cutoffs += child_cutoffs; tt_hits += child_tt_hits
            null_move_cutoffs += child_nmc; futility_pruned += child_fp; razoring_used += child_ru; qs_delta_pruned += child_qdp; qs_see_pruned += child_qsp
            evaluation = -evaluation

            # 2. If it fails high, re-search with a full window
            if evaluation > alpha: # Corrected PVS re-search condition
                evaluation, _, child_nodes, child_q_nodes, child_cutoffs, child_tt_hits, child_nmc, child_fp, child_ru, child_qdp, child_qsp = _search(
                    new_piece_bbs, new_occupancy_bbs, new_game_state, search_depth, -beta, -alpha, search_context, ply + 1
                )
                nodes_searched += child_nodes; quiescence_nodes += child_q_nodes; cutoffs += child_cutoffs; tt_hits += child_tt_hits
                null_move_cutoffs += child_nmc; futility_pruned += child_fp; razoring_used += child_ru; qs_delta_pruned += child_qdp; qs_see_pruned += child_qsp
                evaluation = -evaluation

        if evaluation > max_eval:
            max_eval = evaluation
            best_move = move

            # --- PV Tracking ---
            search_context.pv_table[ply, ply] = move
            # Copy PV from child node
            i = ply + 1
            while i < MAX_PLY and search_context.pv_table[ply + 1, i] != NO_MOVE:
                search_context.pv_table[ply, i] = search_context.pv_table[ply + 1, i]
                i += 1
            # Clear the rest of the line
            while i < MAX_PLY and search_context.pv_table[ply, i] != NO_MOVE:
                search_context.pv_table[ply, i] = NO_MOVE
                i += 1


        alpha = max(alpha, evaluation)
        if alpha >= beta:
            cutoffs += 1
            if is_quiet_move:
                if move != search_context.killer_moves[ply, 0]:
                    search_context.killer_moves[ply, 1] = search_context.killer_moves[ply, 0]
                    search_context.killer_moves[ply, 0] = move
            break
        
        if is_quiet_move:
            quiet_move_counter += 1

    if max_eval <= original_alpha:
        final_flag = TT_FLAG_ALPHA
    elif max_eval >= beta:
        final_flag = TT_FLAG_BETA
    else:
        final_flag = TT_FLAG_EXACT
    
    if best_move == NO_MOVE and len(moves) > 0:
        best_move = sorted_moves[0]

    # Adjust mate score before storing
    tt_score = max_eval
    if tt_score > MATE_IN_MAX_PLY:
        tt_score -= ply
    elif tt_score < -MATE_IN_MAX_PLY:
        tt_score += ply

    store_tt(search_context.transposition_table, zobrist_key, depth, tt_score, final_flag, best_move)

    return (max_eval, best_move, nodes_searched, quiescence_nodes, cutoffs, tt_hits,
            null_move_cutoffs, futility_pruned, razoring_used, qs_delta_pruned, qs_see_pruned)

@numba.njit(search_return_type(
    piece_bbs_signature, occupancy_bbs_signature, game_state_signature, 
    nb.int32, nb.int32, nb.int32, numba.types.Array(numba_tt_entry_type, 1, 'C'),
    nb.types.Array(nb.uint16, 2, 'C'), nb.types.Array(nb.uint16, 2, 'C')
), cache=True)
def _search_wrapper(piece_bbs, occupancy_bbs, game_state, depth, alpha, beta, transposition_table, killer_moves, pv_table):
    search_context = SearchContext(transposition_table, killer_moves, pv_table)
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
    game_state = (side_to_move, castling_rights, en_passant_square, halfmove_clock, np.uint64(zobrist_key))
    
    pv_table = np.zeros((MAX_PLY, MAX_PLY), dtype=np.uint16)
    last_score = 0
    best_move_total = NO_MOVE
    total_nodes_searched, total_quiescence_nodes, total_cutoffs, total_tt_hits = np.uint64(0), np.uint64(0), np.uint64(0), np.uint64(0)
    total_nmc, total_fp, total_ru, total_qdp, total_qsp = np.uint64(0), np.uint64(0), np.uint64(0), np.uint64(0), np.uint64(0)
    last_completed_depth = 0

    for current_depth in range(1, max_depth + 1):
        if current_depth > 1:
            alpha = last_score - ASPIRATION_WINDOW_SIZE
            beta = last_score + ASPIRATION_WINDOW_SIZE
        else:
            alpha, beta = -INFINITY, INFINITY

        pv_table.fill(0)
        score, _, nodes, q_nodes, cutoffs, tt_hits, nmc, fp, ru, qdp, qsp = _search_wrapper(
            piece_bbs, occupancy_bbs, game_state, current_depth, alpha, beta, transposition_table, killer_moves, pv_table
        )

        if score <= alpha or score >= beta:
            log_info(f"depth {current_depth} aspiration window failed, re-searching...")
            alpha, beta = -INFINITY, INFINITY
            pv_table.fill(0)
            score, _, nodes, q_nodes, cutoffs, tt_hits, nmc, fp, ru, qdp, qsp = _search_wrapper(
                piece_bbs, occupancy_bbs, game_state, current_depth, alpha, beta, transposition_table, killer_moves, pv_table
            )

        last_score = score
        total_nodes_searched += nodes; total_quiescence_nodes += q_nodes; total_cutoffs += cutoffs; total_tt_hits += tt_hits
        total_nmc += nmc; total_fp += fp; total_ru += ru; total_qdp += qdp; total_qsp += qsp
        
        tt_entry = probe_tt(transposition_table, game_state[4])
        if tt_entry['flag'] != TT_FLAG_NONE:
            best_move_total = tt_entry['best_move']

        elapsed_time = (time.time() - start_time) * 1000

        # --- Format PV for UCI ---
        pv_moves = []
        for i in range(MAX_PLY):
            move = pv_table[0, i]
            if move == NO_MOVE:
                break
            pv_moves.append(move_to_uci(move))
        pv_string = " ".join(pv_moves)

        # --- Format Score for UCI ---
        uci_score_string = format_score_for_uci(score)

        # --- Print UCI Info String ---
        nps = int(total_nodes_searched / (elapsed_time / 1000)) if elapsed_time > 0 else 0
        print(f"info depth {current_depth} score {uci_score_string} nodes {total_nodes_searched} nps {nps} time {int(elapsed_time)} pv {pv_string}")

        last_completed_depth = current_depth

        # --- Early exit if mate is found ---
        if "mate" in uci_score_string:
            # Check if the mate is positive (beneficial for us)
            mate_value = int(uci_score_string.split()[1])
            if mate_value > 0:
                log_info(f"Mate in {mate_value} found, stopping search.")
                break

        if elapsed_time > max_time_ms:
            break
            
    return (best_move_total, last_score, total_nodes_searched, total_quiescence_nodes, total_cutoffs, total_tt_hits,
            last_completed_depth, total_nmc, total_fp, total_ru, total_qdp, total_qsp)
