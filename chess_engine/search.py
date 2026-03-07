# chess_engine/search.py
import time
import numba
import math
import numpy as np
import threading

from chess_engine.evaluation import evaluate_position
from chess_engine.move_generator import (
    generate_legal_moves, is_in_check, has_sufficient_material, generate_captures,
    generate_legal_moves_buffer, generate_captures_buffer,
    generate_pseudo_legal_moves_buffer, generate_pseudo_legal_captures_buffer,
    generate_pseudo_legal_quiets_buffer,
    is_square_attacked, is_move_pseudo_legal
)
from chess_engine.bitboard_utils import (
    get_lsb_index, WHITE_KING_ZONES, BLACK_KING_ZONES,
    ROOK_RAYS, BISHOP_RAYS, SQUARES_BETWEEN
)
from chess_engine.board_operations import make_move, unmake_move, make_null_move
from chess_engine.move import (
    get_to_square, get_from_square, get_special_move_flag,
    SPECIAL_MOVE_FLAG_PROMOTION
)
from chess_engine.constants import (
    BB_SQUARES, MG_MATERIAL_VALUES, INFINITY, MAX_QUIESCENCE_DEPTH, ASPIRATION_WINDOW_SIZE,
    NULL_MOVE_REDUCTION, MAX_PLY, LMR_MIN_DEPTH, LMR_MIN_QUIET_MOVE_INDEX, LMR_REDUCTION, SEE_THRESHOLD,
    ENABLE_SEE_IN_QUIESCENCE, MATE_SCORE, MATE_IN_MAX_PLY, NO_MOVE,
    RAZORING_MARGIN, FP_MARGIN_D1, FP_MARGIN_D2, FP_BASE, FP_MULTIPLIER, RFP_MARGIN_D1,
    ENABLE_DELTA_PRUNING, DELTA_PRUNING_MARGIN, LMP_MOVE_COUNT, ENABLE_LMP,
    ENABLE_PROBCUT, PROBCUT_R, PROBCUT_R_PRIME, PROBCUT_MARGIN,
    ENABLE_NMP, ENABLE_RAZORING, ENABLE_FP, ENABLE_RFP, ENABLE_LMR, ENABLE_IID,
    STAGE_TT_MOVE, STAGE_GEN_CAPTURES, STAGE_GOOD_CAPTURES,
    STAGE_KILLER_1, STAGE_KILLER_2, STAGE_COUNTER_MOVE, STAGE_GEN_QUIETS,
    STAGE_QUIETS, STAGE_BAD_CAPTURES, STAGE_DONE,
    ENABLE_SINGULAR_EXTENSIONS, MIN_SINGULAR_DEPTH, SINGULAR_EXTENSION_MARGIN,
    STOP_SEARCH_FLAG, MAX_HISTORY,
    SCORE_TT_MOVE, SCORE_GOOD_CAPTURE_BONUS, SCORE_KILLER_1,
    SCORE_KILLER_2, SCORE_COUNTER_MOVE, SCORE_BAD_CAPTURE_PENALTY, NMP_STATIC_MARGIN,
    ENABLE_SHALLOW_SEE_PRUNING, ENABLE_HISTORY_PRUNING, PRUNING_SHALLOW_DEPTH,
    PRUNING_CAPTURE_SEE_MARGIN, PRUNING_QUIET_SEE_MARGIN, PRUNING_HISTORY_THRESHOLD,
    WHITE, BLACK, PAWN_KEY_INDEX, CORRECTION_HISTORY_SIZE, CORRECTION_HISTORY_LIMIT,
    CONTINUATION_HISTORY_FACTOR, MAX_HISTORY,
    LMR_TABLE, ENABLE_MATE_DISTANCE_PRUNING
)
from chess_engine.bitboard_utils import find_piece_type_on_square, find_piece_type_on_square_side
from chess_engine.board_operations import find_piece_type_for_square
from chess_engine.debug_utils import log_info
from chess_engine.see import see, see_ge, get_pinned_pieces
from chess_engine.transposition_table import (
    probe_tt, store_tt, numba_tt_entry_type,
    TT_FLAG_NONE, TT_FLAG_EXACT, TT_FLAG_ALPHA, TT_FLAG_BETA
)
from chess_engine.engine_types import (
    piece_bbs_signature, occupancy_bbs_signature, game_state_signature,
    SearchContext, search_context_type
)
from .move import move_to_uci

from chess_engine.search_heuristics import (
    update_history, update_butterfly_history, update_capture_history,
    update_continuation_history, score_captures, score_quiets,
    get_lmr_reduction, score_moves
)

quiescence_search_return_type = numba.types.Tuple([
    numba.int32, numba.uint64
])

# @numba.njit(quiescence_search_return_type(
#     piece_bbs_signature, occupancy_bbs_signature, game_state_signature,
#     numba.int32, numba.int32, numba.int32, search_context_type
# ), cache=True)
@numba.njit(quiescence_search_return_type(piece_bbs_signature, occupancy_bbs_signature, game_state_signature, numba.int32, numba.int32, numba.int32, search_context_type, numba.int32), cache=True)
def quiescence_search(piece_bbs, occupancy_bbs, game_state, alpha, beta, ply, search_context, q_ply):
    q_nodes = np.uint64(1)
    search_context.nodes_searched += 1

    # Check for stop flag every 2048 nodes
    if (search_context.nodes_searched & 2047) == 0:
        if search_context.stop_flag[0]:
            return np.int32(0), q_nodes

        if search_context.end_time > 0.0:
            current_time = 0.0
            with numba.objmode(current_time='float64'):
                current_time = time.time()
            if current_time >= search_context.end_time:
                search_context.stop_flag[0] = True
                return np.int32(0), q_nodes

    if ply >= MAX_PLY:
        return evaluate_position(piece_bbs, occupancy_bbs, game_state, lazy=False), q_nodes

    # M4: TT Probe in QSearch — avoids re-evaluating positions already in TT
    zobrist_key = game_state[4]
    tt_entry = probe_tt(search_context.transposition_table, zobrist_key)
    if tt_entry['flag'] != TT_FLAG_NONE and tt_entry['depth'] >= 0:
        qs_tt_score = np.int32(tt_entry['score'])
        if qs_tt_score > MATE_IN_MAX_PLY: qs_tt_score -= ply
        elif qs_tt_score < -MATE_IN_MAX_PLY: qs_tt_score += ply
        
        should_cutoff = False
        if tt_entry['flag'] == TT_FLAG_EXACT:
            should_cutoff = True
        elif tt_entry['flag'] == TT_FLAG_ALPHA and qs_tt_score <= alpha:
            should_cutoff = True
        elif tt_entry['flag'] == TT_FLAG_BETA and qs_tt_score >= beta:
            should_cutoff = True
        if should_cutoff:
            return qs_tt_score, q_nodes

    is_currently_in_check = is_in_check(piece_bbs, occupancy_bbs, game_state)

    move_count = 0
    if is_currently_in_check:
        # If in check, we must evade. No stand_pat (can't stand pat in check).
        # H9 Fix: Add depth limit for check evasion in QSearch to prevent infinite loops.
        # Check evasions are very forcing, so we allow them to go deeper than normal QSearch.
        # (e.g., 2x MAX_QUIESCENCE_DEPTH)
        if q_ply >= MAX_QUIESCENCE_DEPTH * 2:
            return evaluate_position(piece_bbs, occupancy_bbs, game_state, lazy=False), q_nodes

        # We must generate ALL legal moves (evasions).
        move_count = generate_pseudo_legal_moves_buffer(piece_bbs, occupancy_bbs, game_state, search_context.moves_buffer, ply)
        if move_count == 0:
            # Checkmate
            return np.int32(-MATE_SCORE + ply), q_nodes
    else:
        # If not in check, we can stand pat (static evaluation)
        if q_ply >= MAX_QUIESCENCE_DEPTH:
            return evaluate_position(piece_bbs, occupancy_bbs, game_state, lazy=False), q_nodes

        stand_pat = evaluate_position(piece_bbs, occupancy_bbs, game_state, lazy=False)
        if stand_pat >= beta:
            return beta, q_nodes
        alpha = max(alpha, stand_pat)

        move_count = generate_pseudo_legal_captures_buffer(piece_bbs, occupancy_bbs, game_state, search_context.moves_buffer, ply)
        if move_count == 0:
            return stand_pat, q_nodes

    # --- Sort moves in QSearch ---
    moves = search_context.moves_buffer[ply]
    scores = search_context.move_scores[ply]

    # Safe access to killers:
    safe_ply = min(ply, MAX_PLY - 1)

    # Optimization: Calculate pinned pieces once for QS pruning logic and score_moves
    pinned_white = get_pinned_pieces(piece_bbs, occupancy_bbs, WHITE)
    pinned_black = get_pinned_pieces(piece_bbs, occupancy_bbs, BLACK)

    # B1: Use TT best move for QSearch ordering (since we already probe TT above)
    qs_tt_move = tt_entry['best_move'] if tt_entry['flag'] != TT_FLAG_NONE else NO_MOVE

    # Use score_moves to sort captures (SEE >= 0 first).
    score_moves(
        piece_bbs, occupancy_bbs, game_state, 
        moves, 
        scores, 
        move_count,
        qs_tt_move,  # B1: was NO_MOVE, now uses TT best move for better ordering
        search_context.killer_moves[safe_ply*2:safe_ply*2+2], 
        search_context.history_table, 
        NO_MOVE, # No counter move
        pinned_white,
        pinned_black,
        search_context,
        ply
    )

    legal_moves_tried = 0
    for i in range(move_count):
        # Lazy Selection Sort: find the best remaining move
        best_idx = i
        for j in range(i + 1, move_count):
            if scores[j] > scores[best_idx]:
                best_idx = j
        
        # Swap moves and scores in the pre-allocated buffers
        moves[i], moves[best_idx] = moves[best_idx], moves[i]
        scores[i], scores[best_idx] = scores[best_idx], scores[i]

        move = moves[i]
        score_val = scores[i]
        
        if not is_currently_in_check:
            if ENABLE_DELTA_PRUNING:
                is_promotion = get_special_move_flag(move) == SPECIAL_MOVE_FLAG_PROMOTION
                promotion_gain = MG_MATERIAL_VALUES[4] - MG_MATERIAL_VALUES[0] if is_promotion else 0
                side_to_move = game_state[0]
                victim_type = find_piece_type_on_square_side(piece_bbs, get_to_square(move), 1 - side_to_move)
                victim_value = MG_MATERIAL_VALUES[victim_type % 6] if victim_type != -1 else 0
                potential_gain = victim_value + promotion_gain
                
                if stand_pat + potential_gain + DELTA_PRUNING_MARGIN < alpha:
                    continue

        if not is_currently_in_check:
            if ENABLE_SEE_IN_QUIESCENCE:
                # Stockfish Alignment: Since SEE_THRESHOLD is 0, any move with score < SCORE_GOOD_CAPTURE_BONUS
                # has already failed see_ge(..., 0) inside score_moves. We can just prune it directly!
                if score_val < SCORE_GOOD_CAPTURE_BONUS:
                    continue

        unmake_info = make_move(piece_bbs, occupancy_bbs, game_state, move)
        
        # --- Lazy Legality Check ---
        king_bb_after_move = piece_bbs[5] if (1 - game_state[0]) == 0 else piece_bbs[11]
        king_sq = get_lsb_index(king_bb_after_move) if king_bb_after_move else 0
        if is_square_attacked(piece_bbs, occupancy_bbs, king_sq, game_state[0]):
            unmake_move(piece_bbs, occupancy_bbs, game_state, move, unmake_info)
            continue
            
        legal_moves_tried += 1

        score, child_q_nodes = quiescence_search(
            piece_bbs, occupancy_bbs, game_state, -beta, -alpha, ply + 1, search_context, q_ply + 1
        )
        unmake_move(piece_bbs, occupancy_bbs, game_state, move, unmake_info)

        if search_context.stop_flag[0]:
            return np.int32(0), q_nodes

        q_nodes += child_q_nodes
        score = -score

        if score >= beta:
            return beta, q_nodes
        alpha = max(alpha, score)

    if is_currently_in_check and legal_moves_tried == 0:
        return np.int32(-MATE_SCORE + ply), q_nodes

    return alpha, q_nodes

search_return_type = numba.types.Tuple([
    numba.int32, numba.uint16, numba.uint64, numba.uint64, numba.uint64
])

get_next_move_return_type = numba.uint16

@numba.njit(get_next_move_return_type(piece_bbs_signature, occupancy_bbs_signature, game_state_signature, search_context_type, numba.int32, numba.uint16, numba.uint16, numba.uint16, numba.uint16, numba.uint16, numba.uint64, numba.uint64), cache=True)
def get_next_move(piece_bbs, occupancy_bbs, game_state, search_context, ply, tt_move, excluded_move, killer_1, killer_2, counter_move, pinned_white, pinned_black):
    moves = search_context.moves_buffer[ply]
    scores = search_context.move_scores[ply]
    bad_captures = search_context.bad_captures[ply]

    while search_context.mp_stage[ply] < STAGE_DONE:
        move = NO_MOVE
        mp_stage = search_context.mp_stage[ply]
        
        if mp_stage == STAGE_TT_MOVE:
            if tt_move != NO_MOVE and tt_move != excluded_move:
                if is_move_pseudo_legal(piece_bbs, occupancy_bbs, game_state, tt_move):
                    move = tt_move
            search_context.mp_stage[ply] = STAGE_GEN_CAPTURES
            if move != NO_MOVE:
                return move
            
        elif mp_stage == STAGE_GEN_CAPTURES:
            search_context.mp_captures_end[ply] = generate_pseudo_legal_captures_buffer(piece_bbs, occupancy_bbs, game_state, search_context.moves_buffer, ply)
            score_captures(piece_bbs, occupancy_bbs, game_state, moves, scores, 0, search_context.mp_captures_end[ply], search_context)
            search_context.mp_current_idx[ply] = 0
            search_context.mp_stage[ply] = STAGE_GOOD_CAPTURES
            continue
            
        elif mp_stage == STAGE_GOOD_CAPTURES:
            captures_end = search_context.mp_captures_end[ply]
            current_idx = search_context.mp_current_idx[ply]
            if current_idx < captures_end:
                best_idx = current_idx
                for j in range(current_idx + 1, captures_end):
                    if scores[j] > scores[best_idx]:
                        best_idx = j
                moves[current_idx], moves[best_idx] = moves[best_idx], moves[current_idx]
                scores[current_idx], scores[best_idx] = scores[best_idx], scores[current_idx]
                
                candidate_move = moves[current_idx]
                search_context.mp_current_idx[ply] += 1
                
                if candidate_move == tt_move or candidate_move == excluded_move:
                    continue
                    
                from_sq = get_from_square(candidate_move)
                to_sq = get_to_square(candidate_move)
                side_to_move = game_state[0]
                
                if see_ge(piece_bbs, occupancy_bbs, side_to_move, from_sq, to_sq, 0, pinned_white, pinned_black):
                    move = candidate_move
                    return move
                else:
                    bad_captures[search_context.mp_bad_captures_count[ply]] = candidate_move
                    search_context.mp_bad_captures_count[ply] += 1
                    continue
            else:
                search_context.mp_stage[ply] = STAGE_KILLER_1
                continue
                
        elif mp_stage == STAGE_KILLER_1:
            search_context.mp_stage[ply] = STAGE_KILLER_2
            if killer_1 != NO_MOVE and killer_1 != tt_move and killer_1 != excluded_move:
                if is_move_pseudo_legal(piece_bbs, occupancy_bbs, game_state, killer_1):
                    to_sq = get_to_square(killer_1)
                    opponent_pieces_bb = occupancy_bbs[1] if game_state[0] == 0 else occupancy_bbs[0]
                    is_capture = (opponent_pieces_bb & BB_SQUARES[to_sq]) != 0
                    if not is_capture:
                        move = killer_1
                        return move
            
        elif mp_stage == STAGE_KILLER_2:
            search_context.mp_stage[ply] = STAGE_COUNTER_MOVE
            if killer_2 != NO_MOVE and killer_2 != tt_move and killer_2 != excluded_move and killer_2 != killer_1:
                if is_move_pseudo_legal(piece_bbs, occupancy_bbs, game_state, killer_2):
                    to_sq = get_to_square(killer_2)
                    opponent_pieces_bb = occupancy_bbs[1] if game_state[0] == 0 else occupancy_bbs[0]
                    is_capture = (opponent_pieces_bb & BB_SQUARES[to_sq]) != 0
                    if not is_capture:
                        move = killer_2
                        return move
                    
        elif mp_stage == STAGE_COUNTER_MOVE:
            search_context.mp_stage[ply] = STAGE_GEN_QUIETS
            if counter_move != NO_MOVE and counter_move != tt_move and counter_move != excluded_move and counter_move != killer_1 and counter_move != killer_2:
                if is_move_pseudo_legal(piece_bbs, occupancy_bbs, game_state, counter_move):
                    to_sq = get_to_square(counter_move)
                    opponent_pieces_bb = occupancy_bbs[1] if game_state[0] == 0 else occupancy_bbs[0]
                    is_capture = (opponent_pieces_bb & BB_SQUARES[to_sq]) != 0
                    if not is_capture:
                        move = counter_move
                        return move
                    
        elif mp_stage == STAGE_GEN_QUIETS:
            captures_end = search_context.mp_captures_end[ply]
            search_context.mp_quiets_end[ply] = generate_pseudo_legal_quiets_buffer(piece_bbs, occupancy_bbs, game_state, search_context.moves_buffer, ply, captures_end)
            score_quiets(piece_bbs, occupancy_bbs, game_state, moves, scores, captures_end, search_context.mp_quiets_end[ply], search_context, ply)
            search_context.mp_current_idx[ply] = captures_end
            search_context.mp_stage[ply] = STAGE_QUIETS
            continue
            
        elif mp_stage == STAGE_QUIETS:
            quiets_end = search_context.mp_quiets_end[ply]
            current_idx = search_context.mp_current_idx[ply]
            if current_idx < quiets_end:
                best_idx = current_idx
                for j in range(current_idx + 1, quiets_end):
                    if scores[j] > scores[best_idx]:
                        best_idx = j
                moves[current_idx], moves[best_idx] = moves[best_idx], moves[current_idx]
                scores[current_idx], scores[best_idx] = scores[best_idx], scores[current_idx]
                
                candidate_move = moves[current_idx]
                search_context.mp_current_idx[ply] += 1
                
                if candidate_move == tt_move or candidate_move == excluded_move or candidate_move == killer_1 or candidate_move == killer_2 or candidate_move == counter_move:
                    continue
                move = candidate_move
                return move
            else:
                search_context.mp_stage[ply] = STAGE_BAD_CAPTURES
                continue
                
        elif mp_stage == STAGE_BAD_CAPTURES:
            if search_context.mp_bad_captures_idx[ply] < search_context.mp_bad_captures_count[ply]:
                move = bad_captures[search_context.mp_bad_captures_idx[ply]]
                search_context.mp_bad_captures_idx[ply] += 1
                return move
            else:
                search_context.mp_stage[ply] = STAGE_DONE
                continue

    return NO_MOVE

search_return_type = numba.types.Tuple([
    numba.int32, numba.uint64
])

# @numba.njit(quiescence_search_return_type(
#     piece_bbs_signature, occupancy_bbs_signature, game_state_signature,
#     numba.int32, numba.int32, numba.int32, search_context_type
# ), cache=True)
@numba.njit(quiescence_search_return_type(piece_bbs_signature, occupancy_bbs_signature, game_state_signature, numba.int32, numba.int32, numba.int32, search_context_type, numba.int32), cache=True)
def quiescence_search(piece_bbs, occupancy_bbs, game_state, alpha, beta, ply, search_context, q_ply):
    q_nodes = np.uint64(1)
    search_context.nodes_searched += 1



    # Check for stop flag every 2048 nodes
    if (search_context.nodes_searched & 2047) == 0:
        if search_context.stop_flag[0]:
            return np.int32(0), q_nodes

        if search_context.end_time > 0.0:
            current_time = 0.0
            with numba.objmode(current_time='float64'):
                current_time = time.time()
            if current_time >= search_context.end_time:
                search_context.stop_flag[0] = True
                return np.int32(0), q_nodes

    if ply >= MAX_PLY:
        return evaluate_position(piece_bbs, occupancy_bbs, game_state, lazy=False), q_nodes

    # M4: TT Probe in QSearch — avoids re-evaluating positions already in TT
    zobrist_key = game_state[4]
    tt_entry = probe_tt(search_context.transposition_table, zobrist_key)
    if tt_entry['flag'] != TT_FLAG_NONE and tt_entry['depth'] >= 0:
        qs_tt_score = np.int32(tt_entry['score'])
        if qs_tt_score > MATE_IN_MAX_PLY: qs_tt_score -= ply
        elif qs_tt_score < -MATE_IN_MAX_PLY: qs_tt_score += ply
        
        should_cutoff = False
        if tt_entry['flag'] == TT_FLAG_EXACT:
            should_cutoff = True
        elif tt_entry['flag'] == TT_FLAG_ALPHA and qs_tt_score <= alpha:
            should_cutoff = True
        elif tt_entry['flag'] == TT_FLAG_BETA and qs_tt_score >= beta:
            should_cutoff = True
        if should_cutoff:
            return qs_tt_score, q_nodes

    is_currently_in_check = is_in_check(piece_bbs, occupancy_bbs, game_state)

    move_count = 0
    if is_currently_in_check:
        # If in check, we must evade. No stand_pat (can't stand pat in check).
        # H9 Fix: Add depth limit for check evasion in QSearch to prevent infinite loops.
        # Check evasions are very forcing, so we allow them to go deeper than normal QSearch.
        # (e.g., 2x MAX_QUIESCENCE_DEPTH)
        if q_ply >= MAX_QUIESCENCE_DEPTH * 2:
            return evaluate_position(piece_bbs, occupancy_bbs, game_state, lazy=False), q_nodes

        # We must generate ALL legal moves (evasions).
        move_count = generate_pseudo_legal_moves_buffer(piece_bbs, occupancy_bbs, game_state, search_context.moves_buffer, ply)
        if move_count == 0:
            # Checkmate
            return np.int32(-MATE_SCORE + ply), q_nodes
    else:
        # If not in check, we can stand pat (static evaluation)
        if q_ply >= MAX_QUIESCENCE_DEPTH:
            return evaluate_position(piece_bbs, occupancy_bbs, game_state, lazy=False), q_nodes

        stand_pat = evaluate_position(piece_bbs, occupancy_bbs, game_state, lazy=False)
        if stand_pat >= beta:
            return beta, q_nodes
        alpha = max(alpha, stand_pat)

        move_count = generate_pseudo_legal_captures_buffer(piece_bbs, occupancy_bbs, game_state, search_context.moves_buffer, ply)
        if move_count == 0:
            return stand_pat, q_nodes

    # --- Sort moves in QSearch ---
    moves = search_context.moves_buffer[ply]
    scores = search_context.move_scores[ply]

    # Safe access to killers:
    safe_ply = min(ply, MAX_PLY - 1)

    # Optimization: Calculate pinned pieces once for QS pruning logic and score_moves
    pinned_white = get_pinned_pieces(piece_bbs, occupancy_bbs, WHITE)
    pinned_black = get_pinned_pieces(piece_bbs, occupancy_bbs, BLACK)

    # B1: Use TT best move for QSearch ordering (since we already probe TT above)
    qs_tt_move = tt_entry['best_move'] if tt_entry['flag'] != TT_FLAG_NONE else NO_MOVE

    # Use score_moves to sort captures (SEE >= 0 first).
    score_moves(
        piece_bbs, occupancy_bbs, game_state, 
        moves, 
        scores, 
        move_count,
        qs_tt_move,  # B1: was NO_MOVE, now uses TT best move for better ordering
        search_context.killer_moves[safe_ply*2:safe_ply*2+2], 
        search_context.history_table, 
        NO_MOVE, # No counter move
        pinned_white,
        pinned_black,
        search_context,
        ply
    )

    legal_moves_tried = 0
    for i in range(move_count):
        # Lazy Selection Sort: find the best remaining move
        best_idx = i
        for j in range(i + 1, move_count):
            if scores[j] > scores[best_idx]:
                best_idx = j
        
        # Swap moves and scores in the pre-allocated buffers
        moves[i], moves[best_idx] = moves[best_idx], moves[i]
        scores[i], scores[best_idx] = scores[best_idx], scores[i]

        move = moves[i]
        score_val = scores[i]
        
        if not is_currently_in_check:
            if ENABLE_DELTA_PRUNING:
                is_promotion = get_special_move_flag(move) == SPECIAL_MOVE_FLAG_PROMOTION
                promotion_gain = MG_MATERIAL_VALUES[4] - MG_MATERIAL_VALUES[0] if is_promotion else 0
                side_to_move = game_state[0]
                victim_type = find_piece_type_on_square_side(piece_bbs, get_to_square(move), 1 - side_to_move)
                victim_value = MG_MATERIAL_VALUES[victim_type % 6] if victim_type != -1 else 0
                potential_gain = victim_value + promotion_gain
                
                if stand_pat + potential_gain + DELTA_PRUNING_MARGIN < alpha:
                    continue

        if not is_currently_in_check:
            if ENABLE_SEE_IN_QUIESCENCE:
                # Stockfish Alignment: Since SEE_THRESHOLD is 0, any move with score < SCORE_GOOD_CAPTURE_BONUS
                # has already failed see_ge(..., 0) inside score_moves. We can just prune it directly!
                if score_val < SCORE_GOOD_CAPTURE_BONUS:
                    continue

        unmake_info = make_move(piece_bbs, occupancy_bbs, game_state, move)
        
        # --- Lazy Legality Check ---
        king_bb_after_move = piece_bbs[5] if (1 - game_state[0]) == 0 else piece_bbs[11]
        king_sq = get_lsb_index(king_bb_after_move) if king_bb_after_move else 0
        if is_square_attacked(piece_bbs, occupancy_bbs, king_sq, game_state[0]):
            unmake_move(piece_bbs, occupancy_bbs, game_state, move, unmake_info)
            continue
            
        legal_moves_tried += 1

        score, child_q_nodes = quiescence_search(
            piece_bbs, occupancy_bbs, game_state, -beta, -alpha, ply + 1, search_context, q_ply + 1
        )
        unmake_move(piece_bbs, occupancy_bbs, game_state, move, unmake_info)

        if search_context.stop_flag[0]:
            return np.int32(0), q_nodes

        q_nodes += child_q_nodes
        score = -score

        if score >= beta:
            return beta, q_nodes
        alpha = max(alpha, score)

    if is_currently_in_check and legal_moves_tried == 0:
        return np.int32(-MATE_SCORE + ply), q_nodes

    return alpha, q_nodes


get_next_move_return_type = numba.uint16

@numba.njit(get_next_move_return_type(piece_bbs_signature, occupancy_bbs_signature, game_state_signature, search_context_type, numba.int32, numba.uint16, numba.uint16, numba.uint16, numba.uint16, numba.uint16, numba.uint64, numba.uint64), cache=True)
def get_next_move(piece_bbs, occupancy_bbs, game_state, search_context, ply, tt_move, excluded_move, killer_1, killer_2, counter_move, pinned_white, pinned_black):
    moves = search_context.moves_buffer[ply]
    scores = search_context.move_scores[ply]
    bad_captures = search_context.bad_captures[ply]

    while search_context.mp_stage[ply] < STAGE_DONE:
        move = NO_MOVE
        mp_stage = search_context.mp_stage[ply]
        
        if mp_stage == STAGE_TT_MOVE:
            if tt_move != NO_MOVE and tt_move != excluded_move:
                if is_move_pseudo_legal(piece_bbs, occupancy_bbs, game_state, tt_move):
                    move = tt_move
            search_context.mp_stage[ply] = STAGE_GEN_CAPTURES
            if move != NO_MOVE:
                return move
            
        elif mp_stage == STAGE_GEN_CAPTURES:
            search_context.mp_captures_end[ply] = generate_pseudo_legal_captures_buffer(piece_bbs, occupancy_bbs, game_state, search_context.moves_buffer, ply)
            score_captures(piece_bbs, occupancy_bbs, game_state, moves, scores, 0, search_context.mp_captures_end[ply], search_context)
            search_context.mp_current_idx[ply] = 0
            search_context.mp_stage[ply] = STAGE_GOOD_CAPTURES
            continue
            
        elif mp_stage == STAGE_GOOD_CAPTURES:
            captures_end = search_context.mp_captures_end[ply]
            current_idx = search_context.mp_current_idx[ply]
            if current_idx < captures_end:
                best_idx = current_idx
                for j in range(current_idx + 1, captures_end):
                    if scores[j] > scores[best_idx]:
                        best_idx = j
                moves[current_idx], moves[best_idx] = moves[best_idx], moves[current_idx]
                scores[current_idx], scores[best_idx] = scores[best_idx], scores[current_idx]
                
                candidate_move = moves[current_idx]
                search_context.mp_current_idx[ply] += 1
                
                if candidate_move == tt_move or candidate_move == excluded_move:
                    continue
                    
                from_sq = get_from_square(candidate_move)
                to_sq = get_to_square(candidate_move)
                side_to_move = game_state[0]
                
                if see_ge(piece_bbs, occupancy_bbs, side_to_move, from_sq, to_sq, 0, pinned_white, pinned_black):
                    move = candidate_move
                    return move
                else:
                    bad_captures[search_context.mp_bad_captures_count[ply]] = candidate_move
                    search_context.mp_bad_captures_count[ply] += 1
                    continue
            else:
                search_context.mp_stage[ply] = STAGE_KILLER_1
                continue
                
        elif mp_stage == STAGE_KILLER_1:
            search_context.mp_stage[ply] = STAGE_KILLER_2
            if killer_1 != NO_MOVE and killer_1 != tt_move and killer_1 != excluded_move:
                if is_move_pseudo_legal(piece_bbs, occupancy_bbs, game_state, killer_1):
                    to_sq = get_to_square(killer_1)
                    opponent_pieces_bb = occupancy_bbs[1] if game_state[0] == 0 else occupancy_bbs[0]
                    is_capture = (opponent_pieces_bb & BB_SQUARES[to_sq]) != 0
                    if not is_capture:
                        move = killer_1
                        return move
            
        elif mp_stage == STAGE_KILLER_2:
            search_context.mp_stage[ply] = STAGE_COUNTER_MOVE
            if killer_2 != NO_MOVE and killer_2 != tt_move and killer_2 != excluded_move and killer_2 != killer_1:
                if is_move_pseudo_legal(piece_bbs, occupancy_bbs, game_state, killer_2):
                    to_sq = get_to_square(killer_2)
                    opponent_pieces_bb = occupancy_bbs[1] if game_state[0] == 0 else occupancy_bbs[0]
                    is_capture = (opponent_pieces_bb & BB_SQUARES[to_sq]) != 0
                    if not is_capture:
                        move = killer_2
                        return move
                    
        elif mp_stage == STAGE_COUNTER_MOVE:
            search_context.mp_stage[ply] = STAGE_GEN_QUIETS
            if counter_move != NO_MOVE and counter_move != tt_move and counter_move != excluded_move and counter_move != killer_1 and counter_move != killer_2:
                if is_move_pseudo_legal(piece_bbs, occupancy_bbs, game_state, counter_move):
                    to_sq = get_to_square(counter_move)
                    opponent_pieces_bb = occupancy_bbs[1] if game_state[0] == 0 else occupancy_bbs[0]
                    is_capture = (opponent_pieces_bb & BB_SQUARES[to_sq]) != 0
                    if not is_capture:
                        move = counter_move
                        return move
                    
        elif mp_stage == STAGE_GEN_QUIETS:
            captures_end = search_context.mp_captures_end[ply]
            search_context.mp_quiets_end[ply] = generate_pseudo_legal_quiets_buffer(piece_bbs, occupancy_bbs, game_state, search_context.moves_buffer, ply, captures_end)
            score_quiets(piece_bbs, occupancy_bbs, game_state, moves, scores, captures_end, search_context.mp_quiets_end[ply], search_context, ply)
            search_context.mp_current_idx[ply] = captures_end
            search_context.mp_stage[ply] = STAGE_QUIETS
            continue
            
        elif mp_stage == STAGE_QUIETS:
            quiets_end = search_context.mp_quiets_end[ply]
            current_idx = search_context.mp_current_idx[ply]
            if current_idx < quiets_end:
                best_idx = current_idx
                for j in range(current_idx + 1, quiets_end):
                    if scores[j] > scores[best_idx]:
                        best_idx = j
                moves[current_idx], moves[best_idx] = moves[best_idx], moves[current_idx]
                scores[current_idx], scores[best_idx] = scores[best_idx], scores[current_idx]
                
                candidate_move = moves[current_idx]
                search_context.mp_current_idx[ply] += 1
                
                if candidate_move == tt_move or candidate_move == excluded_move or candidate_move == killer_1 or candidate_move == killer_2 or candidate_move == counter_move:
                    continue
                move = candidate_move
                return move
            else:
                search_context.mp_stage[ply] = STAGE_BAD_CAPTURES
                continue
                
        elif mp_stage == STAGE_BAD_CAPTURES:
            if search_context.mp_bad_captures_idx[ply] < search_context.mp_bad_captures_count[ply]:
                move = bad_captures[search_context.mp_bad_captures_idx[ply]]
                search_context.mp_bad_captures_idx[ply] += 1
                return move
            else:
                search_context.mp_stage[ply] = STAGE_DONE
                continue

    return NO_MOVE

search_return_type = numba.types.Tuple([
    numba.int32, numba.uint16, numba.uint64, numba.uint64, numba.uint64
])

@numba.njit(search_return_type(piece_bbs_signature, occupancy_bbs_signature, game_state_signature, numba.int32, numba.int32, numba.int32, search_context_type, numba.int32, numba.uint16, numba.boolean, numba.boolean), cache=True)
def _search(piece_bbs, occupancy_bbs, game_state, depth, alpha, beta, search_context, ply, excluded_move: np.uint16 = NO_MOVE, is_pv: bool = True, cut_node: bool = False):
    # ==========================================
    # PHASE 1: Initialize and Base Cases
    # ==========================================
    nodes_searched = np.uint64(1)
    search_context.nodes_searched += 1

    quiescence_nodes = np.uint64(0)
    tt_hits = np.uint64(0)

    # Check for stop flag every 2048 nodes
    if (search_context.nodes_searched & 2047) == 0:
        if search_context.stop_flag[0]:
            return (np.int32(0), NO_MOVE, nodes_searched, quiescence_nodes, tt_hits)

        if search_context.end_time > 0.0:
            current_time = 0.0
            with numba.objmode(current_time='float64'):
                current_time = time.time()
            if current_time >= search_context.end_time:
                search_context.stop_flag[0] = True
                return (np.int32(0), NO_MOVE, nodes_searched, quiescence_nodes, tt_hits)

    if ply >= MAX_PLY:
        return (evaluate_position(piece_bbs, occupancy_bbs, game_state, lazy=False), NO_MOVE, nodes_searched, quiescence_nodes, tt_hits)

    # --- Mate Distance Pruning ---
    if ENABLE_MATE_DISTANCE_PRUNING:
        # If we find a mate at 'ply', the score would be MATE_SCORE - ply.
        # We can't do better than mating at the current ply (MATE_SCORE - ply).
        # We can't do worse than being mated at the current ply (-MATE_SCORE + ply).
        
        # Lower bound: if we are mated in 'ply', score is -MATE_SCORE + ply.
        # Any score below this is impossible.
        alpha = max(alpha, -MATE_SCORE + ply)
        
        # Upper bound: if we mate in 'ply', score is MATE_SCORE - ply.
        # But we need to make a move to mate, so technically MATE_SCORE - (ply + 1)?
        # Actually standard logic is:
        # We are at ply. The best we can hope for is mate in 1 (at ply+1).
        # Score: MATE_SCORE - (ply + 1).
        beta = min(beta, MATE_SCORE - ply - 1)
        
        if alpha >= beta:
            return (np.int32(alpha), NO_MOVE, nodes_searched, quiescence_nodes, tt_hits)

    # --- Repetition Detection ---
    # Performance Optimization (Bolt): Use halfmove_clock to limit checks and skip odd plies.
    # Impact: ~1.8% NPS improvement on Kiwipete benchmark.
    zobrist_key = game_state[4]
    
    # Store current key in path stack
    search_context.ply_path_stack[ply] = zobrist_key
    
    repetition_count = 0
    
    # Check against Game History and Path, but only up to the last irreversible move
    # game_state[3] is the halfmove clock.
    halfmove_clock = int(game_state[3])
    
    if halfmove_clock >= 2:
        # Check Path (even plies only, as odd plies have different side-to-move)
        # Current position is at 'ply', so we check ply-2, ply-4, etc.
        for i in range(ply - 2, max(-1, ply - halfmove_clock - 1), -2):
            if search_context.ply_path_stack[i] == zobrist_key:
                repetition_count += 1
                if repetition_count >= 1: break
        
        if repetition_count < 1:
            # Check Game History
            # Number of moves in history that are within the halfmove clock
            # halfmove_clock includes moves from both path and history.
            hist_to_check = halfmove_clock - ply
            if hist_to_check >= 1:
                # Same side to move:
                # If ply is even, we match game_history[count-3, count-5, ...]
                # If ply is odd, we match game_history[count-2, count-4, ...]
                start_offset = 2 if (ply % 2) != 0 else 3
                start_idx = search_context.game_history_count - start_offset
                end_idx = max(-1, search_context.game_history_count - hist_to_check - 1)
                for i in range(start_idx, end_idx - 1, -2):
                    if i < 0: break
                    if search_context.game_history[i] == zobrist_key:
                        repetition_count += 1
                        if repetition_count >= 1: break

    # Avoid 1st repetition (2nd occurrence total) or 50-move rule
    # Crucial fix: Do not prune at the root (ply 0).
    if ply > 0 and (repetition_count >= 1 or halfmove_clock >= 100):
        search_context.pv_table[ply, ply] = NO_MOVE
        return (np.int32(0), NO_MOVE, nodes_searched, quiescence_nodes, tt_hits)
    
    # M3: 50-move rule scale-down — gradually reduce eval towards draw as halfmove_clock approaches 100
    # This prevents the "cliff effect" where depth-10 search sees halfmove_clock=100 at leaf nodes
    fifty_move_scale = 256  # Full scale (no reduction)
    if halfmove_clock >= 80 and ply > 0:
        # Linear scale-down: at hmc=80 → scale=256, at hmc=100 → scale=0
        fifty_move_scale = max(0, (100 - halfmove_clock) * 256 // 20)
    
    # Initialize PV for this ply to avoid ghost moves from previous searches
    search_context.pv_table[ply, ply] = NO_MOVE

    # --- DTR-LMR: Strategic Bitmask Pre-calculation ---
    side_to_move = game_state[0]
    friendly_king_bb = piece_bbs[5] if side_to_move == WHITE else piece_bbs[11]
    enemy_king_bb = piece_bbs[11] if side_to_move == WHITE else piece_bbs[5]
    
    enemy_king_zone = np.uint64(0)
    if enemy_king_bb:
        enemy_king_sq = get_lsb_index(enemy_king_bb)
        enemy_king_zone = BLACK_KING_ZONES[enemy_king_sq] if side_to_move == WHITE else WHITE_KING_ZONES[enemy_king_sq]
    
    friendly_danger_rays = np.uint64(0)
    if friendly_king_bb:
        friendly_king_sq = get_lsb_index(friendly_king_bb)
        enemy_offset = 6 if side_to_move == WHITE else 0
        enemy_rooks = piece_bbs[3 + enemy_offset] | piece_bbs[4 + enemy_offset]
        enemy_bishops = piece_bbs[2 + enemy_offset] | piece_bbs[4 + enemy_offset]
        
        # Orthogonal danger rays
        temp_rays = ROOK_RAYS[friendly_king_sq] & enemy_rooks
        while temp_rays:
            pinner_sq = get_lsb_index(temp_rays)
            friendly_danger_rays |= SQUARES_BETWEEN[friendly_king_sq, pinner_sq]
            temp_rays &= temp_rays - np.uint64(1)
            
        # Diagonal danger rays
        temp_rays = BISHOP_RAYS[friendly_king_sq] & enemy_bishops
        while temp_rays:
            pinner_sq = get_lsb_index(temp_rays)
            friendly_danger_rays |= SQUARES_BETWEEN[friendly_king_sq, pinner_sq]
            temp_rays &= temp_rays - np.uint64(1)

    original_alpha = alpha
    
    # --- Initialization ---
    move_count = 0
    # ==========================================
    # PHASE 2: Transposition Table Probe
    # ==========================================
    moves_generated = False

    tt_move = NO_MOVE
    tt_entry = probe_tt(search_context.transposition_table, zobrist_key)

    # 1. ALWAYS retrieve the best move if available (Critical for Move Ordering)
    if tt_entry['flag'] != TT_FLAG_NONE:
        tt_move = tt_entry['best_move']

        # 2. ONLY perform a Score Cutoff (Return) if:
        #    a. We are NOT at the Root Node (ply > 0)
        #    b. NOT a PV node (H9: PV nodes must not be cut off by TT bounds)
        #    c. The stored depth is sufficient
        #    d. The score bounds (Alpha/Beta) are valid for a cutoff
        #    e. We are NOT in a singular extension search (excluded_move == NO_MOVE)
        #       OR the TT move is NOT the excluded move.
        if ply > 0 and not is_pv and tt_entry['depth'] >= depth and (excluded_move == NO_MOVE or tt_move != excluded_move):
            tt_hits += np.uint64(1)
            tt_score = np.int32(tt_entry['score'])

            # Adjust mate scores relative to the current ply
            if tt_score > MATE_IN_MAX_PLY: tt_score -= ply
            elif tt_score < -MATE_IN_MAX_PLY: tt_score += ply

            # NEW: Apply 50-move scale down immediately to TT scores to avoid overlooking impending draws
            if fifty_move_scale < 256 and abs(tt_score) < MATE_IN_MAX_PLY:
                tt_score = tt_score * fifty_move_scale // 256

            should_cutoff = False
            if tt_entry['flag'] == TT_FLAG_EXACT:
                should_cutoff = True
            elif tt_entry['flag'] == TT_FLAG_ALPHA and tt_score <= alpha:
                should_cutoff = True
            elif tt_entry['flag'] == TT_FLAG_BETA and tt_score >= beta:
                should_cutoff = True

            if should_cutoff:
                search_context.pv_table[ply, ply] = NO_MOVE
                return (tt_score, tt_entry['best_move'], nodes_searched, quiescence_nodes, tt_hits)

    # ==========================================
    # PHASE 3: Static Evaluation and Pre-Search Pruning
    # ==========================================
    # --- H1: is_in_check MUST be computed before any node-level pruning/sub-searches ---
    is_currently_in_check = is_in_check(piece_bbs, occupancy_bbs, game_state)

    # --- Depth 0: drop into Quiescence Search ---
    if depth <= 0:
        search_context.pv_table[ply, ply] = NO_MOVE
        eval_score, q_nodes = quiescence_search(piece_bbs, occupancy_bbs, game_state, alpha, beta, ply, search_context, 0)
        return (eval_score, NO_MOVE, nodes_searched, q_nodes, tt_hits)

    # --- H2: Static Evaluation & Improving Flag (computed BEFORE any pruning) ---
    static_score = -INFINITY
    improving = False
    
    # Correction History Adjustment
    raw_static_eval = -INFINITY

    if not is_currently_in_check:
        # Retrieve from TT if available
        if tt_entry['flag'] != TT_FLAG_NONE and tt_entry['static_eval'] != 32767:
            raw_static_eval = np.int16(tt_entry['static_eval'])
        else:
            raw_static_eval = evaluate_position(piece_bbs, occupancy_bbs, game_state, lazy=True)
        
        # Apply Correction History
        pawn_key = game_state[PAWN_KEY_INDEX]
        correction = search_context.pawn_correction_history[pawn_key % CORRECTION_HISTORY_SIZE]
        
        # Clamp correction to limit its effect
        correction = min(max(correction, -CORRECTION_HISTORY_LIMIT), CORRECTION_HISTORY_LIMIT)
        
        static_score = raw_static_eval + correction
        
        # --- H2 Enhancement: Refine static_score using TT Score Bound ---
        if tt_entry['flag'] != TT_FLAG_NONE:
            tt_score = np.int32(tt_entry['score'])
            
            # Adjust mate scores relative to current ply
            if tt_score > MATE_IN_MAX_PLY: tt_score -= ply
            elif tt_score < -MATE_IN_MAX_PLY: tt_score += ply
            
            # If TT says score is at least tt_score (LOWER), and tt_score > static_score
            if tt_entry['flag'] == TT_FLAG_BETA and tt_score > static_score:
                static_score = tt_score
            # If TT says score is at most tt_score (UPPER), and tt_score < static_score
            elif tt_entry['flag'] == TT_FLAG_ALPHA and tt_score < static_score:
                static_score = tt_score
            # If TT has EXACT score, trust it completely
            elif tt_entry['flag'] == TT_FLAG_EXACT:
                static_score = tt_score
        
        search_context.static_eval_stack[ply] = static_score

        if ply >= 2 and static_score > search_context.static_eval_stack[ply - 2]:
            improving = True
    else:
        search_context.static_eval_stack[ply] = -INFINITY

    # --- Dissonance Calculation (DGP Algorithm) ---
    dissonance = 0
    if tt_entry['flag'] != TT_FLAG_NONE and not is_currently_in_check and static_score != -INFINITY:
        # Check if depth is reasonably close to current depth to trust the TT score
        if tt_entry['depth'] >= depth - 2:
            tt_score = np.int32(tt_entry['score'])
            # Avoid using mate scores for dissonance calculation
            if abs(tt_score) < MATE_IN_MAX_PLY and abs(static_score) < MATE_IN_MAX_PLY:
                dissonance = abs(tt_score - static_score)

    # =====================================================================
    # --- Node-Level Pruning (H3: ALL guarded by not is_currently_in_check) ---
    # =====================================================================
    if not is_currently_in_check and not is_pv:
        # --- M5: Reverse Futility Pruning (returns static_score, not beta) ---
        # C1: Tighter margin when not improving
        rfp_margin = depth * RFP_MARGIN_D1
        if not improving:
            rfp_margin = rfp_margin * 3 // 4  # 25% tighter when not improving
        if ENABLE_RFP and depth <= 5 and static_score - rfp_margin >= beta:
            return (np.int32(static_score), NO_MOVE, nodes_searched, quiescence_nodes, tt_hits)

    # --- Null Move Pruning (guarded by H3) ---
    # M2: Check for non-pawn material (any N/B/R/Q) to avoid zugzwang in pawn endgames
    has_non_pawn = False
    if not is_currently_in_check:
        stm = game_state[0]
        if stm == 0:  # WHITE
            has_non_pawn = (piece_bbs[1] | piece_bbs[2] | piece_bbs[3] | piece_bbs[4]) != 0
        else:  # BLACK
            has_non_pawn = (piece_bbs[7] | piece_bbs[8] | piece_bbs[9] | piece_bbs[10]) != 0
    if ENABLE_NMP and depth >= 3 and not is_currently_in_check and has_non_pawn and static_score >= beta:
        original_state_for_null = game_state.copy()
        make_null_move(game_state)
        
        # Dynamic NMP Reduction: R = 3 + depth / 6 + min(3, (static_score - beta) / 200)
        nmp_reduction = 3 + depth // 6 + min(3, (static_score - beta) // 200)
        # C1: Extra reduction when not improving
        if not improving:
            nmp_reduction += 1
        search_depth = max(0, depth - nmp_reduction)
        
        res_nm = _search(
            piece_bbs, occupancy_bbs, game_state, search_depth,
            -beta, -beta + 1, search_context, ply + 1, NO_MOVE, False, not cut_node
        )
        null_move_score = res_nm[0]
        child_nodes = res_nm[2]
        child_q_nodes = res_nm[3]
        child_tt_hits = res_nm[4]

        game_state[:] = original_state_for_null
        null_move_score = -null_move_score

        if search_context.stop_flag[0]:
            return (np.int32(0), NO_MOVE, nodes_searched, quiescence_nodes, tt_hits)

        nodes_searched += child_nodes; quiescence_nodes += child_q_nodes; tt_hits += child_tt_hits

        if null_move_score >= beta:
            # M6: Don't trust mate scores from NMP
            if null_move_score >= MATE_IN_MAX_PLY:
                null_move_score = beta
            search_context.pv_table[ply, ply] = NO_MOVE
            return (np.int32(null_move_score), NO_MOVE, nodes_searched, quiescence_nodes, tt_hits)

    # --- ProbCut (must be before NMP, guarded by H3) ---
    if depth >= 5 and abs(beta) < MATE_IN_MAX_PLY and not is_currently_in_check:
        probcut_beta = beta + PROBCUT_MARGIN
        
        # --- NEW: ProbCut TT Defense ---
        # If TT bound is UPPER (Alpha) and TT score is less than probcut_beta,
        # it is highly unlikely a reduced depth search will exceed probcut_beta. Skip ProbCut.
        skip_probcut = False
        if tt_entry['flag'] != TT_FLAG_NONE:
            tt_score_pc = np.int32(tt_entry['score'])
            if tt_score_pc > MATE_IN_MAX_PLY: tt_score_pc -= ply
            elif tt_score_pc < -MATE_IN_MAX_PLY: tt_score_pc += ply
            
            if tt_entry['flag'] == TT_FLAG_ALPHA and tt_score_pc < probcut_beta:
                skip_probcut = True
            elif tt_entry['flag'] == TT_FLAG_EXACT and tt_score_pc < probcut_beta:
                skip_probcut = True
                
        if not skip_probcut:
            # We must only try captures.
            pc_move_count = generate_pseudo_legal_captures_buffer(piece_bbs, occupancy_bbs, game_state, search_context.moves_buffer, ply)
            
            # Safe access to killers:
            safe_ply = min(ply, MAX_PLY - 1)
    
            # Use score_moves to get SEE order. NO killer/history since these are captures only.
            score_moves(
                piece_bbs, occupancy_bbs, game_state, 
                search_context.moves_buffer[ply], 
                search_context.move_scores[ply], 
                pc_move_count,
                tt_move, 
                search_context.killer_moves[safe_ply*2:safe_ply*2+2], # Pass normal slice to satisfy Numba compiler (killers won't match captures anyway)
                search_context.history_table, 
                NO_MOVE,
                0, 0, # Pinned pieces not critically needed for ordering here, or we can use 0
                search_context,
                ply
            )
            
            pc_moves = search_context.moves_buffer[ply]
            pc_scores = search_context.move_scores[ply]
            
            pc_side = game_state[0]
            pc_pinned_w = get_pinned_pieces(piece_bbs, occupancy_bbs, WHITE)
            pc_pinned_b = get_pinned_pieces(piece_bbs, occupancy_bbs, BLACK)
    
            for i in range(pc_move_count):
                # Selection sort
                best_idx = i
                for j in range(i + 1, pc_move_count):
                    if pc_scores[j] > pc_scores[best_idx]:
                        best_idx = j
                pc_moves[i], pc_moves[best_idx] = pc_moves[best_idx], pc_moves[i]
                pc_scores[i], pc_scores[best_idx] = pc_scores[best_idx], pc_scores[i]
    
                pc_move = pc_moves[i]
                pc_from = get_from_square(pc_move)
                pc_to = get_to_square(pc_move)
    
                # SEE filter: only try captures whose SEE >= probcut_beta - static_eval
                see_threshold_pc = probcut_beta - static_score if static_score != -INFINITY else 0
                if not see_ge(piece_bbs, occupancy_bbs, pc_side, pc_from, pc_to, see_threshold_pc, pc_pinned_w, pc_pinned_b):
                    continue
    
                pc_unmake = make_move(piece_bbs, occupancy_bbs, game_state, pc_move)
    
                # Legality check
                pc_king_bb = piece_bbs[5] if (1 - game_state[0]) == 0 else piece_bbs[11]
                pc_king_sq = get_lsb_index(pc_king_bb) if pc_king_bb else 0
                if is_square_attacked(piece_bbs, occupancy_bbs, pc_king_sq, game_state[0]):
                    unmake_move(piece_bbs, occupancy_bbs, game_state, pc_move, pc_unmake)
                    continue
    
                # Shallow verification search
                res_pc = _search(
                    piece_bbs, occupancy_bbs, game_state, depth - PROBCUT_R_PRIME,
                    -probcut_beta, -probcut_beta + 1, search_context, ply + 1, NO_MOVE, False, not cut_node
                )
                pc_score = -res_pc[0]
                
                # IMPORTANT: Must aggregate the search stats from ProbCut back into the parent
                child_pc_nodes = res_pc[2]; child_pc_q_nodes = res_pc[3]
                child_pc_tth = res_pc[4]
                
                nodes_searched += child_pc_nodes; quiescence_nodes += child_pc_q_nodes
                tt_hits += child_pc_tth

                unmake_move(piece_bbs, occupancy_bbs, game_state, pc_move, pc_unmake)

                if search_context.stop_flag[0]:
                    return (np.int32(0), NO_MOVE, nodes_searched, quiescence_nodes, tt_hits)
    
                if pc_score >= probcut_beta:
                    # Store in TT for future use
                    tt_store_score_pc = pc_score
                    if tt_store_score_pc > MATE_IN_MAX_PLY: tt_store_score_pc += ply
                    elif tt_store_score_pc < -MATE_IN_MAX_PLY: tt_store_score_pc -= ply
                    store_tt(search_context.transposition_table, zobrist_key, depth - 3, tt_store_score_pc, np.int16(32767), TT_FLAG_BETA, pc_move, search_context.tt_generation, False)
                    return (np.int32(pc_score), pc_move, nodes_searched, quiescence_nodes, tt_hits)

    # --- Razoring (must be after NMP, guarded by H3) ---
    if not is_currently_in_check and not is_pv and depth <= 7:
        if ENABLE_RAZORING and static_score + RAZORING_MARGIN < alpha and dissonance < 250:
            razor_score, child_q_nodes = quiescence_search(piece_bbs, occupancy_bbs, game_state, alpha, alpha + 1, ply, search_context, 0)
            quiescence_nodes += child_q_nodes
            if razor_score + RAZORING_MARGIN < alpha:
                return (np.int32(alpha), NO_MOVE, nodes_searched, quiescence_nodes, tt_hits)

    # --- M1: IIR (Internal Iterative Reduction) replaces IID ---
    # Instead of doing a costly sub-search, just reduce depth by 1 when we have no TT move
    if depth >= 4 and tt_move == NO_MOVE and not is_currently_in_check:
        depth -= 1

    # --- Move Ordering ---
    # Retrieve Counter Move if available
    counter_move = NO_MOVE
    if ply > 0:
        prev_move = search_context.move_stack[ply - 1]
        if prev_move != NO_MOVE:
            prev_from = get_from_square(prev_move)
            prev_to = get_to_square(prev_move)
            counter_move = search_context.counter_moves[prev_from, prev_to]

    moves = search_context.moves_buffer[ply]
    scores = search_context.move_scores[ply]

    # Safe access to killers:
    safe_ply = min(ply, MAX_PLY - 1)
    killer_1 = search_context.killer_moves[safe_ply*2]
    killer_2 = search_context.killer_moves[safe_ply*2+1]

    # Optimization: Calculate pinned pieces once for shallow pruning logic and score_moves
    pinned_white = get_pinned_pieces(piece_bbs, occupancy_bbs, WHITE)
    pinned_black = get_pinned_pieces(piece_bbs, occupancy_bbs, BLACK)

    best_move, max_eval = NO_MOVE, -INFINITY
    quiet_move_counter, searched_move_count = 0, 0
    legal_moves_tried = 0
    pruned_moves = 0
    
    # Track tried quiet moves for history malus
    quiet_moves_tried = search_context.quiet_moves_tried[ply]
    quiet_moves_tried_count = 0

    # Staged Move Generation State Init
    search_context.mp_stage[ply] = STAGE_TT_MOVE
    search_context.mp_current_idx[ply] = 0
    search_context.mp_captures_end[ply] = 0
    search_context.mp_quiets_end[ply] = 0
    search_context.mp_bad_captures_count[ply] = 0
    search_context.mp_bad_captures_idx[ply] = 0

    while True:
        move = get_next_move(
            piece_bbs, occupancy_bbs, game_state, search_context, ply,
            tt_move, excluded_move, killer_1, killer_2, counter_move,
            pinned_white, pinned_black
        )
        if move == NO_MOVE:
            break
        searched_move_count += 1
        
        opponent_pieces_bb = occupancy_bbs[1] if game_state[0] == 0 else occupancy_bbs[0]
        
        # --- Pre-move checks for extensions and move type ---
        from_sq = get_from_square(move)
        to_sq = get_to_square(move)
        is_capture = (opponent_pieces_bb & BB_SQUARES[to_sq]) != 0
        is_promotion = get_special_move_flag(move) == SPECIAL_MOVE_FLAG_PROMOTION
        is_pseudo_quiet = not is_capture and not is_promotion
        
        # --- NEW: Shallow Depth Pruning (Stockfish Step 14) ---
        if ENABLE_SHALLOW_SEE_PRUNING and depth <= PRUNING_SHALLOW_DEPTH and not is_currently_in_check and not is_pv:
             # Can't use is_giving_check_after_move here as it requires make_move.
             side_to_move = game_state[0]

             if is_capture:
                 # Capture Pruning
                 # Threshold: -200 * depth
                 threshold = PRUNING_CAPTURE_SEE_MARGIN * depth
                 if not see_ge(piece_bbs, occupancy_bbs, side_to_move, from_sq, to_sq, threshold, pinned_white, pinned_black):
                     pruned_moves += 1
                     continue
             elif is_pseudo_quiet:
                 # Quiet Move Pruning

                 # History Pruning
                 if ENABLE_HISTORY_PRUNING:
                     aggressor_type = find_piece_type_on_square_side(piece_bbs, from_sq, side_to_move)
                     history_score = search_context.history_table[aggressor_type, to_sq]
                     if history_score < PRUNING_HISTORY_THRESHOLD * depth:
                         pruned_moves += 1
                         continue

                 # Quiet SEE Pruning (e.g. moving into attack)
                 # Threshold: -100 * depth * depth
                 threshold = PRUNING_QUIET_SEE_MARGIN * depth * depth
                 if not see_ge(piece_bbs, occupancy_bbs, side_to_move, from_sq, to_sq, threshold, pinned_white, pinned_black):
                     pruned_moves += 1
                     continue
                  
        # Removed unconditional pre_see_check_ok to prevent massive performance waste (Lazy Evaluation)

        # --- Make the move ---
        unmake_info = make_move(piece_bbs, occupancy_bbs, game_state, move)
        
        # --- Lazy Legality Check ---
        king_bb_after_move = piece_bbs[5] if (1 - game_state[0]) == 0 else piece_bbs[11]
        king_sq = get_lsb_index(king_bb_after_move) if king_bb_after_move else 0
        if is_square_attacked(piece_bbs, occupancy_bbs, king_sq, game_state[0]):
            unmake_move(piece_bbs, occupancy_bbs, game_state, move, unmake_info)
            continue
            
        legal_moves_tried += 1

        search_context.move_stack[ply] = move # Record move in stack
        
        # --- Post-move checks ---
        is_giving_check_after_move = is_in_check(piece_bbs, occupancy_bbs, game_state)
        
        # Final determination of quiet move
        is_quiet_move = is_pseudo_quiet and not is_giving_check_after_move

        # --- DTR-LMR: Strategic Move Properties ---
        is_king_penetration = False
        is_proactive_masking = False
        if is_quiet_move:
            # Check for King Zone Penetration
            if BB_SQUARES[to_sq] & enemy_king_zone:
                is_king_penetration = True
            
            # Check for Proactive Masking (blocking an enemy ray to our king)
            if BB_SQUARES[to_sq] & friendly_danger_rays:
                is_proactive_masking = True

        if is_quiet_move:
            quiet_move_counter += 1
            # Add to tried list for potential malus
            quiet_moves_tried[quiet_moves_tried_count] = move
            quiet_moves_tried_count += 1
            
            # Late Move Pruning (LMP) - Disabled in PV nodes
            if ENABLE_LMP and not is_currently_in_check and not is_pv:
                limit = LMP_MOVE_COUNT[depth]
                if not improving: limit = limit // 2
                limit = max(limit, 2)

                if quiet_move_counter >= limit:
                    unmake_move(piece_bbs, occupancy_bbs, game_state, move, unmake_info)
                    continue  # H7: was 'break', changed to 'continue' to not skip bad captures

        # Futility Pruning (FP) - Disabled in PV nodes
        if ENABLE_FP and is_quiet_move and depth <= 8 and not is_currently_in_check and not is_pv and static_score != -INFINITY:
            # Dynamic Futility Margin: FP_BASE + FP_MULTIPLIER * depth
            margin = FP_BASE + FP_MULTIPLIER * depth
            # C1: Tighter margin when not improving
            if not improving:
                margin = margin * 3 // 4
            
            # DGP Adjustment
            if dissonance > 150:
                margin += dissonance // 2

            if margin > 0 and static_score + margin < alpha:
                unmake_move(piece_bbs, occupancy_bbs, game_state, move, unmake_info)
                continue
        
        # --- Determine total extension ---
        check_extension = 0
        if is_giving_check_after_move:
            # Lazy SEE Evaluation: Unmake to perfectly evaluate the pre-move position state
            unmake_move(piece_bbs, occupancy_bbs, game_state, move, unmake_info)
            # Use SEE_THRESHOLD (which is now 0) to ensure we don't extend on losing check sacrifices
            # Pass game_state[0] instead of side_to_move because we unmade the move, so side to move has reverted
            if see_ge(piece_bbs, occupancy_bbs, game_state[0], from_sq, to_sq, SEE_THRESHOLD, pinned_white, pinned_black):
                check_extension = 1
            # Remake move to restore state
            unmake_info = make_move(piece_bbs, occupancy_bbs, game_state, move)
        
        current_extension = check_extension

        # --- Singular Extension (inside loop, TT move only) ---
        if ENABLE_SINGULAR_EXTENSIONS and move == tt_move and depth >= MIN_SINGULAR_DEPTH and not is_currently_in_check:
            if tt_entry['flag'] != TT_FLAG_NONE and (tt_entry['flag'] == TT_FLAG_EXACT or tt_entry['flag'] == TT_FLAG_BETA) and tt_entry['depth'] >= depth - 3:
                se_tt_score = np.int32(tt_entry['score'])
                exclusion_beta = se_tt_score - SINGULAR_EXTENSION_MARGIN
                # Save MovePicker State to prevent Singular Extension from clobbering the parent
                saved_mp_stage = search_context.mp_stage[ply]
                saved_mp_idx = search_context.mp_current_idx[ply]
                saved_mp_captures = search_context.mp_captures_end[ply]
                saved_mp_quiets = search_context.mp_quiets_end[ply]
                saved_mp_bad_cap = search_context.mp_bad_captures_count[ply]
                saved_mp_bad_idx = search_context.mp_bad_captures_idx[ply]

                # Unmake to search position without this move
                unmake_move(piece_bbs, occupancy_bbs, game_state, move, unmake_info)
                res_ex = _search(
                    piece_bbs, occupancy_bbs, game_state, (depth - 1) // 2, exclusion_beta - 1, exclusion_beta, search_context, ply, tt_move, False, cut_node)
                exclusion_score = res_ex[0]
                
                nodes_searched += res_ex[2]; quiescence_nodes += res_ex[3]
                tt_hits += res_ex[4]
                
                # Restore MovePicker State
                search_context.mp_stage[ply] = saved_mp_stage
                search_context.mp_current_idx[ply] = saved_mp_idx
                search_context.mp_captures_end[ply] = saved_mp_captures
                search_context.mp_quiets_end[ply] = saved_mp_quiets
                search_context.mp_bad_captures_count[ply] = saved_mp_bad_cap
                search_context.mp_bad_captures_idx[ply] = saved_mp_bad_idx

                # Re-make the move
                unmake_info = make_move(piece_bbs, occupancy_bbs, game_state, move)

                if search_context.stop_flag[0]:
                    unmake_move(piece_bbs, occupancy_bbs, game_state, move, unmake_info)
                    return (np.int32(0), NO_MOVE, nodes_searched, quiescence_nodes, tt_hits)

                if exclusion_score < exclusion_beta:
                    current_extension = max(current_extension, 1)
                    # Double extension for big margin
                    if depth >= 8 and exclusion_score < exclusion_beta - SINGULAR_EXTENSION_MARGIN:
                        current_extension = max(current_extension, 2)

        search_depth = depth - 1 + current_extension

        evaluation = 0
        if legal_moves_tried == 1:
            res = _search(
                piece_bbs, occupancy_bbs, game_state, search_depth, -beta, -alpha, search_context, ply + 1, NO_MOVE, is_pv, False)
            evaluation = -res[0]
            child_nodes = res[2]; child_q_nodes = res[3]; child_tt_hits = res[4]
        else:
            # Dynamic LMR Logic
            lmr = 0
            if ENABLE_LMR and depth >= LMR_MIN_DEPTH and is_quiet_move and quiet_move_counter >= LMR_MIN_QUIET_MOVE_INDEX:
                # Get History Score for adjustment
                aggressor_type = find_piece_type_on_square_side(piece_bbs, from_sq, game_state[0])
                history_score = search_context.history_table[aggressor_type, to_sq]

                lmr = get_lmr_reduction(depth, legal_moves_tried, history_score, improving, is_pv)

                # CUT-node bonus: more aggressive reduction in expected CUT nodes
                if cut_node:
                    lmr += 1

                # No TT move bonus
                if tt_move == NO_MOVE:
                    lmr += 1

                # A2: Pawn push protection — don't reduce pawn pushes to 6th/7th rank
                is_pawn = (aggressor_type == 0 or aggressor_type == 6)
                if is_pawn:
                    to_rank = to_sq // 8
                    if (game_state[0] == 0 and to_rank >= 5) or (game_state[0] == 1 and to_rank <= 2):
                        lmr = max(0, lmr - 2)

                # A4: Continuation History adjustment in LMR
                if ply > 0:
                    prev_mv = search_context.move_stack[ply - 1]
                    if prev_mv != NO_MOVE:
                        p_to = get_to_square(prev_mv)
                        p_piece = find_piece_type_for_square(piece_bbs, p_to, 1 - game_state[0])
                        if p_piece != -1:
                            cont_score = search_context.continuation_history[p_piece, p_to, aggressor_type, to_sq]
                            # Scale: ±1 reduction for max history
                            lmr -= int(float(cont_score) / float(MAX_HISTORY) * 1.0)

                # Clamp LMR to avoid reducing below depth 1
                lmr = max(0, min(lmr, search_depth - 1))

            # A3: LMR for bad captures (only reduce when it's a bad capture, not a good one)
            elif ENABLE_LMR and depth >= LMR_MIN_DEPTH and is_capture and not is_promotion and legal_moves_tried > 1:
                if search_context.mp_stage[ply] == STAGE_BAD_CAPTURES or search_context.mp_stage[ply] == STAGE_DONE:
                    lmr = 1 + depth // 6
                    lmr = max(0, min(lmr, search_depth - 1))

            res = _search(
                piece_bbs, occupancy_bbs, game_state, search_depth - lmr, -alpha - 1, -alpha, search_context, ply + 1, NO_MOVE, False, True)
            evaluation = -res[0]
            child_nodes = res[2]; child_q_nodes = res[3]; child_tt_hits = res[4]

            if evaluation > alpha:
                do_full_pv_search = is_pv and evaluation < beta
                
                if lmr > 0:
                    # LMR failed high on zero window. Verify at full depth with zero window.
                    res = _search(
                        piece_bbs, occupancy_bbs, game_state, search_depth, -alpha - 1, -alpha, search_context, ply + 1, NO_MOVE, False, not cut_node)
                    evaluation = -res[0]
                    child_nodes += res[2]; child_q_nodes += res[3]; child_tt_hits += res[4]
                    
                    if is_pv and evaluation > alpha and evaluation < beta:
                        do_full_pv_search = True
                    else:
                        do_full_pv_search = False
                
                if do_full_pv_search:
                    # Re-search with full window
                    res = _search(
                        piece_bbs, occupancy_bbs, game_state, search_depth, -beta, -alpha, search_context, ply + 1, NO_MOVE, is_pv, False)
                    evaluation = -res[0]
                    child_nodes += res[2]; child_q_nodes += res[3]; child_tt_hits += res[4]

        unmake_move(piece_bbs, occupancy_bbs, game_state, move, unmake_info)

        if search_context.stop_flag[0]:
            return (np.int32(0), NO_MOVE, nodes_searched, quiescence_nodes, tt_hits)

        nodes_searched += child_nodes; quiescence_nodes += child_q_nodes; tt_hits += child_tt_hits

        if evaluation > max_eval:
            max_eval, best_move = evaluation, move
            search_context.pv_table[ply, ply] = move
            
            i = ply + 1
            if ply + 1 < MAX_PLY:
                j = ply + 1
                while j < MAX_PLY and search_context.pv_table[ply + 1, j] != NO_MOVE:
                    search_context.pv_table[ply, i] = search_context.pv_table[ply + 1, j]
                    i += 1; j += 1
            
            if i < MAX_PLY: search_context.pv_table[ply, i] = NO_MOVE

        alpha = max(alpha, evaluation)
        if alpha >= beta:
            bonus = depth * depth
            
            if is_capture:
                # Update Capture History
                victim_type = find_piece_type_on_square_side(piece_bbs, to_sq, 1 - game_state[0])
                if victim_type != -1:
                    aggressor_type = find_piece_type_on_square_side(piece_bbs, from_sq, game_state[0])
                    update_capture_history(search_context.capture_history, aggressor_type, to_sq, victim_type, bonus)
            
            if is_quiet_move:
                aggressor_type = find_piece_type_on_square_side(piece_bbs, get_from_square(move), game_state[0])
                to_sq = get_to_square(move)
                
                # Apply Gravity Bonus to the cutoff move
                update_history(search_context.history_table, aggressor_type, to_sq, bonus)
                update_butterfly_history(search_context.butterfly_history, get_from_square(move), to_sq, bonus)

                # --- Continuation History Update (Bonus) ---
                if ply > 0:
                    prev_move_played = search_context.move_stack[ply - 1]
                    if prev_move_played != NO_MOVE:
                        p_to = get_to_square(prev_move_played)
                        p_piece = find_piece_type_for_square(piece_bbs, p_to, 1 - game_state[0])
                        if p_piece != -1:
                            update_continuation_history(search_context, prev_move_played, p_piece, move, aggressor_type, bonus)

                # --- Update Counter Move ---
                if ply > 0:
                    prev_move_played = search_context.move_stack[ply - 1]
                    if prev_move_played != NO_MOVE:
                        p_from = get_from_square(prev_move_played)
                        p_to = get_to_square(prev_move_played)
                        search_context.counter_moves[p_from, p_to] = move
                
                # Apply History Malus to all previous quiet moves that failed low
                for q_idx in range(quiet_moves_tried_count - 1): # Exclude the current move (last one added)
                    bad_move = quiet_moves_tried[q_idx]
                    bad_from = get_from_square(bad_move)
                    bad_to = get_to_square(bad_move)
                    bad_aggressor = find_piece_type_on_square_side(piece_bbs, bad_from, game_state[0])
                    update_history(search_context.history_table, bad_aggressor, bad_to, -bonus)
                    update_butterfly_history(search_context.butterfly_history, bad_from, bad_to, -bonus)
                    
                    # --- Continuation History Update (Malus) ---
                    if ply > 0:
                        prev_move_played = search_context.move_stack[ply - 1]
                        if prev_move_played != NO_MOVE:
                            p_to = get_to_square(prev_move_played)
                            p_piece = find_piece_type_for_square(piece_bbs, p_to, 1 - game_state[0])
                            if p_piece != -1:
                                update_continuation_history(search_context, prev_move_played, p_piece, bad_move, bad_aggressor, -bonus)

                if move != search_context.killer_moves[ply * 2]:
                    search_context.killer_moves[ply * 2 + 1] = search_context.killer_moves[ply * 2]
                    search_context.killer_moves[ply * 2] = move
            break

    if searched_move_count == 0:
        if is_currently_in_check:
            return (np.int32(-MATE_SCORE + ply), NO_MOVE, nodes_searched, quiescence_nodes, tt_hits)
        else:
            return (np.int32(0), NO_MOVE, nodes_searched, quiescence_nodes, tt_hits)

    if legal_moves_tried == 0 and pruned_moves == 0:
        if is_currently_in_check:
            return (np.int32(-MATE_SCORE + ply), NO_MOVE, nodes_searched, quiescence_nodes, tt_hits)
        else:
            return (np.int32(0), NO_MOVE, nodes_searched, quiescence_nodes, tt_hits)

    final_flag = TT_FLAG_ALPHA if max_eval <= original_alpha else (TT_FLAG_BETA if max_eval >= beta else TT_FLAG_EXACT)
    if max_eval == -INFINITY:
        max_eval = original_alpha

    # M3: Apply 50-move scale-down to non-mate evaluations
    if fifty_move_scale < 256 and abs(max_eval) < MATE_IN_MAX_PLY:
        max_eval = max_eval * fifty_move_scale // 256
    if best_move == NO_MOVE and search_context.mp_captures_end[ply] + search_context.mp_quiets_end[ply] > 0:
        # Fallback to first move that is not excluded
        for i in range(search_context.mp_captures_end[ply] + search_context.mp_quiets_end[ply]):
            if moves[i] != excluded_move:
                best_move = moves[i]
                break
    
    # --- Update Correction History ---
    if not is_currently_in_check and best_move != NO_MOVE and abs(max_eval) < MATE_SCORE - MAX_PLY:
        # Only update if we have a valid static eval (raw_static_eval != -INFINITY)
        # And we are not in check
        if raw_static_eval != -INFINITY:
            diff = max_eval - raw_static_eval
            pawn_key = game_state[PAWN_KEY_INDEX]
            idx = pawn_key % CORRECTION_HISTORY_SIZE
            current_corr = search_context.pawn_correction_history[idx]
            
            # Simple gravity update towards the difference
            # new_corr = current_corr + (diff - current_corr) / GRAVITY
            # We use a shift or division for gravity. Using constant 16.
            # Avoid using floating point if possible, or cast.
            # Correction history is int16.
            
            # Limit diff to avoid extreme swings
            clamped_diff = min(max(diff, -CORRECTION_HISTORY_LIMIT), CORRECTION_HISTORY_LIMIT)
            
            new_corr = current_corr + (clamped_diff - current_corr) // 16
            
            # Clamp final result
            new_corr = min(max(new_corr, -CORRECTION_HISTORY_LIMIT), CORRECTION_HISTORY_LIMIT)
            
            search_context.pawn_correction_history[idx] = new_corr

    tt_score = max_eval
    if tt_score > MATE_IN_MAX_PLY: tt_score += ply
    elif tt_score < -MATE_IN_MAX_PLY: tt_score -= ply
    # Determine static eval to store logic
    tt_static_eval_to_store = np.int16(32767)
    if raw_static_eval != -INFINITY and abs(raw_static_eval) < MATE_SCORE - MAX_PLY:
        tt_static_eval_to_store = np.int16(raw_static_eval)

    store_tt(search_context.transposition_table, zobrist_key, depth, tt_score, tt_static_eval_to_store, final_flag, best_move, search_context.tt_generation, is_pv)

    return (max_eval, best_move, nodes_searched, quiescence_nodes, tt_hits)

def iterative_deepening_search(piece_bbs, occupancy_bbs, game_state, max_depth, time_config, search_context, game_history_list=None, tt_generation=0):
    start_time = time.time()
    
    # --- Decay History Tables ---
    # H11: Decay history to prioritize recent data and reduce stale information
    # Fix H8: Use truncation towards zero instead of floor division.
    # Python's // 2 is floor division: -1 // 2 = -1 (sticks forever).
    # np.sign * (abs // 2) truncates towards zero: -1 → 0, -3 → -1, etc.
    h = search_context.history_table
    search_context.history_table[:] = np.sign(h) * (np.abs(h) // 2)
    b = search_context.butterfly_history
    search_context.butterfly_history[:] = np.sign(b) * (np.abs(b) // 2)
    c = search_context.capture_history
    search_context.capture_history[:] = np.sign(c) * (np.abs(c) // 2)
    # H10: Continuation history decay (was missing entirely — caused cross-game state pollution)
    ch = search_context.continuation_history
    search_context.continuation_history[:] = np.sign(ch) * (np.abs(ch) // 2)

    # --- Setup Game History ---
    if game_history_list is not None:
        count = len(game_history_list)
        limit = min(count, 1024)
        for i in range(limit):
            search_context.game_history[i] = game_history_list[i]
        search_context.game_history_count = limit
    else:
        search_context.game_history_count = 0
        
    # --- Setup TT Generation ---
    search_context.tt_generation = tt_generation

    maximum_time_ms = time_config.get('maximum_time', 0)
    optimum_time_ms = time_config.get('optimum_time', 0)
    
    transposition_table = search_context.transposition_table
    
    if maximum_time_ms > 0:
        search_context.end_time = start_time + (maximum_time_ms / 1000.0)
    else:
        search_context.end_time = 0.0
    
    search_context.stop_flag[0] = False
    
    last_score, best_move_total = 0, NO_MOVE
    total_nodes, total_q_nodes, total_tt_hits = (np.uint64(v) for v in [0]*3)
    last_completed_depth = 0
    best_move_from_last_depth = NO_MOVE

    # Clear Counter Moves at start of search? Stockfish doesn't seem to reset them per search, 
    # but usually they are part of thread data. We can keep them or clear them.
    # Clearing them ensures no pollution from previous moves in different game contexts if not handled by generations.
    # However, for the same game, it might be useful. 
    # Let's clear them to be safe and consistent with "new search".
    # PRESERVED: search_context.counter_moves is NOT cleared here to prevent state pollution loss
    search_context.move_stack.fill(NO_MOVE)

    for current_depth in range(1, max_depth + 1):
        # Aspiration Window Logic
        alpha = -INFINITY
        beta = INFINITY
        delta = ASPIRATION_WINDOW_SIZE
        
        if current_depth > 1:
            alpha = max(-INFINITY, last_score - delta)
            beta = min(INFINITY, last_score + delta)

        search_context.nodes_searched = np.uint64(0)
        
        while True:
            res = _search(piece_bbs, occupancy_bbs, game_state, current_depth, alpha, beta, search_context, 0, NO_MOVE, True, False)
            score = res[0]
            
            # Always accumulate stats from the search (even if stopped or failed window)
            # Note: search_context.nodes_searched is reset per depth loop but accumulates inside the while loop effectively
            # Actually, to get correct NPS, we should probably accumulate into totals here.
            # But search_context.nodes_searched is reset at the top of the FOR loop, not the WHILE loop.
            # So if we re-search, search_context.nodes_searched will increase.
            # We just need to make sure we don't double count if we add to 'total_nodes' inside the loop?
            # The original code added search_context.nodes_searched to total_nodes AFTER the search call.
            # Here, we might search multiple times.
            
            # Let's accumulate non-node stats here immediately to safe-keep them.
            total_q_nodes += res[3]; total_tt_hits += res[4]

            if search_context.stop_flag[0]:
                break
            
            if score <= alpha:
                alpha = max(-INFINITY, alpha - delta)
                delta += delta // 2
                # log_info(f"depth {current_depth} fail low ({score} <= {alpha}), widening to [{alpha}, {beta}]")
            elif score >= beta:
                beta = min(INFINITY, beta + delta)
                delta += delta // 2
                # log_info(f"depth {current_depth} fail high ({score} >= {beta}), widening to [{alpha}, {beta}]")
            else:
                # Score is within window, we are done with this depth
                break
        
        total_nodes += search_context.nodes_searched

        if search_context.stop_flag[0]:
            log_info(f"Search stopped at depth {current_depth} due to time limit.")
            break

        # --- This block only runs if the search for the current depth was fully completed ---
        last_completed_depth = current_depth
        last_score = score

        tt_entry = probe_tt(transposition_table, game_state[4])
        if tt_entry['flag'] != TT_FLAG_NONE and tt_entry['best_move'] != NO_MOVE:
             best_move_from_last_depth = tt_entry['best_move']
        
        elapsed_time_ms = (time.time() - start_time) * 1000
        
        pv_moves = []
        for i in range(MAX_PLY):
            m = search_context.pv_table[0, i]
            if m == NO_MOVE:
                break
            pv_moves.append(move_to_uci(m))
        
        pv_string = " ".join(pv_moves)
        uci_score_string = format_score_for_uci(score)

        nps = int(total_nodes / (elapsed_time_ms / 1000)) if elapsed_time_ms > 0 else 0
        print(f"info depth {current_depth} score {uci_score_string} nodes {total_nodes} nps {nps} time {int(elapsed_time_ms)} pv {pv_string}")

        # Predictive soft time limit
        if maximum_time_ms > 0:
            if optimum_time_ms > 0 and elapsed_time_ms > optimum_time_ms:
                log_info(f"Optimum time reached at depth {current_depth}. Stopping.")
                break
            if elapsed_time_ms * 2.5 > maximum_time_ms:
                log_info(f"Predictive termination at depth {current_depth} to avoid timeout.")
                break

    final_best_move = best_move_from_last_depth
    return (final_best_move, last_score, total_nodes, total_q_nodes, total_tt_hits, last_completed_depth)

def format_score_for_uci(score):
    if abs(score) > MATE_IN_MAX_PLY:
        if score > 0:
            moves_to_mate = (MATE_SCORE - score + 1) // 2
            return f"mate {moves_to_mate}"
        else:
            moves_to_mate = (MATE_SCORE + score + 1) // 2
            return f"mate {-moves_to_mate}"
    return f"cp {score}"