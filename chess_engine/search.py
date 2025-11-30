# chess_engine/search.py
import time
import numba
import numpy as np
import threading

from chess_engine.evaluation import evaluate_position
from chess_engine.move_generator import (
    generate_legal_moves, is_in_check, has_sufficient_material, generate_captures
)
from chess_engine.zobrist import get_lsb_index
from chess_engine.board_operations import make_move, unmake_move, make_null_move
from chess_engine.move import (
    get_to_square, get_from_square, get_special_move_flag,
    SPECIAL_MOVE_FLAG_PROMOTION
)
from chess_engine.constants import (
    BB_SQUARES, MG_MATERIAL_VALUES, INFINITY, MAX_QUIESCENCE_DEPTH, ASPIRATION_WINDOW_SIZE,
    NULL_MOVE_REDUCTION, MAX_PLY, LMR_MIN_DEPTH, LMR_MIN_QUIET_MOVE_INDEX, LMR_REDUCTION, SEE_THRESHOLD,
    ENABLE_SEE_IN_QUIESCENCE, MATE_SCORE, MATE_IN_MAX_PLY, NO_MOVE,
    RAZORING_MARGIN, FP_MARGIN_D1, FP_MARGIN_D2, RFP_MARGIN_D1,
    ENABLE_DELTA_PRUNING, DELTA_PRUNING_MARGIN, LMP_MOVE_COUNT, ENABLE_LMP,
    ENABLE_PROBCUT, PROBCUT_R, PROBCUT_R_PRIME, PROBCUT_MARGIN,
    ENABLE_NMP, ENABLE_RAZORING, ENABLE_FP, ENABLE_RFP, ENABLE_LMR, ENABLE_IID,
    ENABLE_SINGULAR_EXTENSIONS, MIN_SINGULAR_DEPTH, SINGULAR_EXTENSION_MARGIN,
    STOP_SEARCH_FLAG, PAWN_PUSH_RANK_BONUS, PAWN_PUSH_ATTACK_BONUS
)
from chess_engine.bitboard_utils import find_piece_type_on_square, KING_ATTACK_ZONES
from chess_engine.debug_utils import log_info
from chess_engine.see import see
from chess_engine.transposition_table import (
    probe_tt, store_tt, numba_tt_entry_type,
    TT_FLAG_NONE, TT_FLAG_EXACT, TT_FLAG_ALPHA, TT_FLAG_BETA
)
from chess_engine.engine_types import (
    piece_bbs_signature, occupancy_bbs_signature, game_state_signature,
    SearchContext, search_context_type
)
from .move import move_to_uci

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def _partition(moves, scores, low, high):
    pivot_score = scores[high]
    i = low - 1
    for j in range(low, high):
        if scores[j] >= pivot_score:
            i += 1
            moves[i], moves[j] = moves[j], moves[i]
            scores[i], scores[j] = scores[j], scores[i]
    moves[i + 1], moves[high] = moves[high], moves[i + 1]
    scores[i + 1], scores[high] = scores[high], scores[i + 1]
    return i + 1

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def _quicksort_recursive(moves, scores, low, high):
    if low < high:
        pi = _partition(moves, scores, low, high)
        _quicksort_recursive(moves, scores, low, pi - 1)
        _quicksort_recursive(moves, scores, pi + 1, high)

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def score_moves(piece_bbs, occupancy_bbs, game_state, moves, tt_move, killer_moves_at_ply, history_table):
    scores = np.zeros(len(moves), dtype=np.int32)
    side_to_move = game_state[0]
    opponent_pieces_bb = occupancy_bbs[1] if side_to_move == 0 else occupancy_bbs[0]
    
    # Determine opponent king square for attack bonus
    opponent_king_bb = piece_bbs[11] if side_to_move == 0 else piece_bbs[5]
    opponent_king_sq = get_lsb_index(opponent_king_bb) if opponent_king_bb != 0 else -1
    
    for i in range(len(moves)):
        move = moves[i]
        score = 0
        if move == tt_move:
            score = 100_000
        else:
            to_square = get_to_square(move)
            is_capture = (opponent_pieces_bb & BB_SQUARES[to_square]) != 0
            if is_capture:
                aggressor_type = find_piece_type_on_square(piece_bbs, get_from_square(move))
                victim_type = find_piece_type_on_square(piece_bbs, to_square)
                if victim_type != -1:
                    score = 10_000 + (MG_MATERIAL_VALUES[victim_type % 6] - MG_MATERIAL_VALUES[aggressor_type % 6])
            else:
                if move == killer_moves_at_ply[0]:
                    score = 5_000
                elif move == killer_moves_at_ply[1]:
                    score = 4_000
                else:
                    aggressor_type = find_piece_type_on_square(piece_bbs, get_from_square(move))
                    score = history_table[aggressor_type, to_square]
                    
                    # --- Pawn Push Bonuses ---
                    if aggressor_type == 0 or aggressor_type == 6: # PAWN (White=0, Black=6)
                        # Rank Bonus (Rank 6/7)
                        rank = to_square // 8
                        is_advanced_pawn = False
                        if side_to_move == 0: # White
                            if rank >= 5: is_advanced_pawn = True 
                        else: # Black
                            if rank <= 2: is_advanced_pawn = True
                        
                        if is_advanced_pawn:
                            score += PAWN_PUSH_RANK_BONUS
                            
                        # King Attack Bonus
                        if opponent_king_sq != -1:
                             if (KING_ATTACK_ZONES[opponent_king_sq] & BB_SQUARES[to_square]) != 0:
                                score += PAWN_PUSH_ATTACK_BONUS

        scores[i] = score
    return scores

quiescence_search_return_type = numba.types.Tuple([
    numba.int32, numba.uint64, numba.uint64, numba.uint64
])

# @numba.njit(quiescence_search_return_type(
#     piece_bbs_signature, occupancy_bbs_signature, game_state_signature,
#     numba.int32, numba.int32, numba.int32, search_context_type
# ), cache=True)
@numba.njit(cache=True)
def quiescence_search(piece_bbs, occupancy_bbs, game_state, alpha, beta, ply, search_context):
    q_nodes = np.uint64(1)
    search_context.nodes_searched += 1

    delta_pruned = np.uint64(0)
    see_pruned = np.uint64(0)

    # Check for stop flag every 2048 nodes
    if (search_context.nodes_searched & 2047) == 0:
        if search_context.stop_flag[0]:
            return np.int32(0), q_nodes, delta_pruned, see_pruned

        if search_context.end_time > 0.0:
            current_time = 0.0
            with numba.objmode(current_time='float64'):
                current_time = time.time()
            if current_time >= search_context.end_time:
                search_context.stop_flag[0] = True
                return np.int32(0), q_nodes, delta_pruned, see_pruned

    if ply >= MAX_QUIESCENCE_DEPTH:
        return evaluate_position(piece_bbs, occupancy_bbs, game_state, lazy=False), q_nodes, delta_pruned, see_pruned

    stand_pat = evaluate_position(piece_bbs, occupancy_bbs, game_state, lazy=False)
    if stand_pat >= beta:
        return beta, q_nodes, delta_pruned, see_pruned
    alpha = max(alpha, stand_pat)

    capture_moves = generate_captures(piece_bbs, occupancy_bbs, game_state)
    if len(capture_moves) == 0:
        return stand_pat, q_nodes, delta_pruned, see_pruned

    for move in capture_moves:
        if ENABLE_DELTA_PRUNING:
            is_promotion = get_special_move_flag(move) == SPECIAL_MOVE_FLAG_PROMOTION
            promotion_gain = MG_MATERIAL_VALUES[4] - MG_MATERIAL_VALUES[0] if is_promotion else 0
            victim_type = find_piece_type_on_square(piece_bbs, get_to_square(move))
            victim_value = MG_MATERIAL_VALUES[victim_type % 6] if victim_type != -1 else 0
            potential_gain = victim_value + promotion_gain
            
            if stand_pat + potential_gain + DELTA_PRUNING_MARGIN < alpha:
                unmake_info = make_move(piece_bbs, occupancy_bbs, game_state, move)
                is_check = is_in_check(piece_bbs, occupancy_bbs, game_state)
                unmake_move(piece_bbs, occupancy_bbs, game_state, move, unmake_info)
                if not is_check:
                    delta_pruned += 1
                    continue

        if ENABLE_SEE_IN_QUIESCENCE:
            side_to_move = game_state[0]
            if see(piece_bbs, occupancy_bbs, side_to_move, get_from_square(move), get_to_square(move)) < SEE_THRESHOLD:
                see_pruned += 1
                continue

        unmake_info = make_move(piece_bbs, occupancy_bbs, game_state, move)
        score, child_q_nodes, child_delta_pruned, child_see_pruned = quiescence_search(
            piece_bbs, occupancy_bbs, game_state, -beta, -alpha, ply + 1, search_context
        )
        unmake_move(piece_bbs, occupancy_bbs, game_state, move, unmake_info)

        if search_context.stop_flag[0]:
            return np.int32(0), q_nodes, delta_pruned, see_pruned

        q_nodes += child_q_nodes
        delta_pruned += child_delta_pruned
        see_pruned += child_see_pruned
        score = -score

        if score >= beta:
            return beta, q_nodes, delta_pruned, see_pruned
        alpha = max(alpha, score)

    return alpha, q_nodes, delta_pruned, see_pruned

search_return_type = numba.types.Tuple([
    numba.int32, numba.uint16, numba.uint64, numba.uint64, numba.uint64, numba.uint64,
    numba.uint64, numba.uint64, numba.uint64, numba.uint64, numba.uint64, numba.uint64,
    numba.uint64, numba.uint64, numba.uint64, numba.uint64
])

# @numba.njit(search_return_type(
#     piece_bbs_signature, occupancy_bbs_signature, game_state_signature,
#     numba.int32, numba.int32, numba.int32, search_context_type, numba.int32, numba.uint16
# ), cache=True)
@numba.njit(cache=True)
def _search(piece_bbs, occupancy_bbs, game_state, depth, alpha, beta, search_context, ply, excluded_move: np.uint16 = NO_MOVE):
    nodes_searched = np.uint64(1)
    search_context.nodes_searched += 1

    quiescence_nodes = np.uint64(0)
    cutoffs = np.uint64(0)
    tt_hits = np.uint64(0)
    null_move_cutoffs = np.uint64(0)
    futility_pruned = np.uint64(0)
    razoring_used = np.uint64(0)
    rfp_pruned = np.uint64(0)
    lmp_pruned = np.uint64(0)
    probcut_pruned = np.uint64(0)
    qs_delta_pruned = np.uint64(0)
    qs_see_pruned = np.uint64(0)
    iid_searches = np.uint64(0)
    singular_extensions = np.uint64(0)

    # Check for stop flag every 2048 nodes
    if (search_context.nodes_searched & 2047) == 0:
        if search_context.stop_flag[0]:
            return (np.int32(0), NO_MOVE, nodes_searched, quiescence_nodes, cutoffs, tt_hits,
                    null_move_cutoffs, futility_pruned, razoring_used, rfp_pruned, lmp_pruned, probcut_pruned, qs_delta_pruned, qs_see_pruned,
                    iid_searches, singular_extensions)

        if search_context.end_time > 0.0:
            current_time = 0.0
            with numba.objmode(current_time='float64'):
                current_time = time.time()
            if current_time >= search_context.end_time:
                search_context.stop_flag[0] = True
                return (np.int32(0), NO_MOVE, nodes_searched, quiescence_nodes, cutoffs, tt_hits,
                        null_move_cutoffs, futility_pruned, razoring_used, rfp_pruned, lmp_pruned, probcut_pruned, qs_delta_pruned, qs_see_pruned,
                        iid_searches, singular_extensions)

    # --- Repetition Detection ---
    zobrist_key = game_state[4]
    
    # Store current key in path stack
    search_context.ply_path_stack[ply] = zobrist_key
    
    repetition_count = 0
    
    # Check against Game History
    for i in range(search_context.game_history_count):
        if search_context.game_history[i] == zobrist_key:
            repetition_count += 1
            
    # Check against Current Search Path (from root to ply-1)
    for i in range(ply):
        if search_context.ply_path_stack[i] == zobrist_key:
            repetition_count += 1
            
    # Avoid 2nd repetition
    # Crucial fix: Do not prune at the root (ply 0). If we are at the root, we must search for a move.
    if ply > 0 and repetition_count >= 1:
        search_context.pv_table[ply, :].fill(NO_MOVE)
        return (np.int32(0), NO_MOVE, nodes_searched, quiescence_nodes, cutoffs, tt_hits,
                null_move_cutoffs, futility_pruned, razoring_used, rfp_pruned, lmp_pruned, probcut_pruned, qs_delta_pruned, qs_see_pruned,
                iid_searches, singular_extensions)

    if ply >= MAX_PLY:
        search_context.pv_table[ply, :].fill(NO_MOVE)
        return (evaluate_position(piece_bbs, occupancy_bbs, game_state, lazy=False), NO_MOVE, nodes_searched, quiescence_nodes, cutoffs, tt_hits,
                null_move_cutoffs, futility_pruned, razoring_used, rfp_pruned, lmp_pruned, probcut_pruned, qs_delta_pruned, qs_see_pruned,
                iid_searches, singular_extensions)
    
    original_alpha = alpha
    
    # --- Initialization ---
    moves_generated = False
    moves = np.empty(0, dtype=np.uint16)

    tt_move = NO_MOVE
    tt_entry = probe_tt(search_context.transposition_table, zobrist_key)

    if tt_entry['flag'] != TT_FLAG_NONE and tt_entry['depth'] >= depth:
        
        # With corrected 64-bit Zobrist keys, collisions are virtually impossible.
        # We trust the TT move directly without expensive legal generation.
        tt_hits += 1
        tt_score = np.int32(tt_entry['score'])
        if tt_score > MATE_IN_MAX_PLY: tt_score -= ply
        elif tt_score < -MATE_IN_MAX_PLY: tt_score += ply

        if tt_entry['flag'] == TT_FLAG_EXACT:
            search_context.pv_table[ply, :].fill(NO_MOVE)
            return (tt_score, tt_entry['best_move'], nodes_searched, quiescence_nodes, cutoffs, tt_hits,
                    null_move_cutoffs, futility_pruned, razoring_used, rfp_pruned, lmp_pruned, probcut_pruned, qs_delta_pruned, qs_see_pruned,
                    iid_searches, singular_extensions)
        elif tt_entry['flag'] == TT_FLAG_ALPHA: beta = min(beta, tt_score)
        elif tt_entry['flag'] == TT_FLAG_BETA: alpha = max(alpha, tt_score)

        if alpha >= beta:
            search_context.pv_table[ply, :].fill(NO_MOVE)
            return (tt_score, tt_entry['best_move'], nodes_searched, quiescence_nodes, cutoffs, tt_hits,
                    null_move_cutoffs, futility_pruned, razoring_used, rfp_pruned, lmp_pruned, probcut_pruned, qs_delta_pruned, qs_see_pruned,
                    iid_searches, singular_extensions)
        tt_move = tt_entry['best_move']

    if ENABLE_IID and depth >= 8 and tt_move == NO_MOVE:
        iid_searches += 1
        _search(piece_bbs, occupancy_bbs, game_state, depth -5, -INFINITY, INFINITY, search_context, ply + 1, NO_MOVE)

        if search_context.stop_flag[0]:
            return (np.int32(0), NO_MOVE, nodes_searched, quiescence_nodes, cutoffs, tt_hits,
                    null_move_cutoffs, futility_pruned, razoring_used, rfp_pruned, lmp_pruned, probcut_pruned, qs_delta_pruned, qs_see_pruned,
                    iid_searches, singular_extensions)

        tt_entry = probe_tt(search_context.transposition_table, zobrist_key)
        if tt_entry['flag'] != TT_FLAG_NONE: tt_move = tt_entry['best_move']

    extension = 0
    if ENABLE_SINGULAR_EXTENSIONS and depth >= MIN_SINGULAR_DEPTH and tt_move != NO_MOVE and tt_entry['flag'] == TT_FLAG_EXACT:
        tt_score = np.int32(tt_entry['score'])
        exclusion_beta = tt_score - SINGULAR_EXTENSION_MARGIN
        exclusion_score, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = _search(
            piece_bbs, occupancy_bbs, game_state, depth - 3, exclusion_beta - 1, exclusion_beta, search_context, ply + 1, tt_move)

        if search_context.stop_flag[0]:
            return (np.int32(0), NO_MOVE, nodes_searched, quiescence_nodes, cutoffs, tt_hits,
                    null_move_cutoffs, futility_pruned, razoring_used, rfp_pruned, lmp_pruned, probcut_pruned, qs_delta_pruned, qs_see_pruned,
                    iid_searches, singular_extensions)

        if exclusion_score < exclusion_beta:
            extension = 1
            singular_extensions += 1

    if ENABLE_PROBCUT and depth >= 6 and tt_entry['flag'] != TT_FLAG_NONE:
        if tt_entry['depth'] >= depth - PROBCUT_R:
            tt_score = np.int32(tt_entry['score'])
            if tt_score + PROBCUT_MARGIN >= beta:
                probcut_score, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = _search(
                    piece_bbs, occupancy_bbs, game_state, depth - PROBCUT_R_PRIME,
                    beta - 1, beta, search_context, ply + 1, NO_MOVE
                )

                if search_context.stop_flag[0]:
                    return (np.int32(0), NO_MOVE, nodes_searched, quiescence_nodes, cutoffs, tt_hits,
                            null_move_cutoffs, futility_pruned, razoring_used, rfp_pruned, lmp_pruned, probcut_pruned, qs_delta_pruned, qs_see_pruned,
                            iid_searches, singular_extensions)

                if probcut_score >= beta:
                    probcut_pruned += 1
                    return (np.int32(beta), NO_MOVE, nodes_searched, quiescence_nodes, cutoffs, tt_hits,
                        null_move_cutoffs, futility_pruned, razoring_used, rfp_pruned, lmp_pruned, probcut_pruned, qs_delta_pruned, qs_see_pruned,
                        iid_searches, singular_extensions)

            elif tt_score - PROBCUT_MARGIN <= alpha:
                probcut_score, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = _search(
                    piece_bbs, occupancy_bbs, game_state, depth - PROBCUT_R_PRIME,
                    alpha, beta, search_context, ply + 1, NO_MOVE
                )

                if search_context.stop_flag[0]:
                    return (np.int32(0), NO_MOVE, nodes_searched, quiescence_nodes, cutoffs, tt_hits,
                            null_move_cutoffs, futility_pruned, razoring_used, rfp_pruned, lmp_pruned, probcut_pruned, qs_delta_pruned, qs_see_pruned,
                            iid_searches, singular_extensions)

                if probcut_score <= alpha:
                    probcut_pruned += 1
                    return (np.int32(alpha), NO_MOVE, nodes_searched, quiescence_nodes, cutoffs, tt_hits,
                        null_move_cutoffs, futility_pruned, razoring_used, rfp_pruned, lmp_pruned, probcut_pruned, qs_delta_pruned, qs_see_pruned,
                        iid_searches, singular_extensions)
                
    is_currently_in_check = is_in_check(piece_bbs, occupancy_bbs, game_state)
    
    if depth == 0:
        search_context.pv_table[ply, :].fill(NO_MOVE)
        eval_score, q_nodes, child_delta_pruned, child_see_pruned = quiescence_search(piece_bbs, occupancy_bbs, game_state, alpha, beta, 0, search_context)
        qs_delta_pruned += child_delta_pruned
        qs_see_pruned += child_see_pruned
        return (eval_score, NO_MOVE, nodes_searched, q_nodes, cutoffs, tt_hits,
                null_move_cutoffs, futility_pruned, razoring_used, rfp_pruned, lmp_pruned, probcut_pruned, qs_delta_pruned, qs_see_pruned,
                iid_searches, singular_extensions)

    if ENABLE_NMP and depth >= 3 and not is_currently_in_check and has_sufficient_material(piece_bbs, game_state[0]):
        original_state_for_null = game_state.copy()
        make_null_move(game_state)
        
        (null_move_score, _, child_nodes, child_q_nodes, child_cutoffs, child_tt_hits,
         child_nmc, child_fp, child_ru, child_rfp, child_lmp, child_pcp,
         child_qdp, child_qsp, child_iid, child_se) = _search(
            piece_bbs, occupancy_bbs, game_state, depth - 1 - NULL_MOVE_REDUCTION,
            -beta, -beta + 1, search_context, ply + 1, NO_MOVE
        )

        game_state[:] = original_state_for_null
        null_move_score = -null_move_score

        if search_context.stop_flag[0]:
            return (np.int32(0), NO_MOVE, nodes_searched, quiescence_nodes, cutoffs, tt_hits,
                    null_move_cutoffs, futility_pruned, razoring_used, rfp_pruned, lmp_pruned, probcut_pruned, qs_delta_pruned, qs_see_pruned,
                    iid_searches, singular_extensions)

        nodes_searched += child_nodes; quiescence_nodes += child_q_nodes; cutoffs += child_cutoffs; tt_hits += child_tt_hits
        null_move_cutoffs += child_nmc; futility_pruned += child_fp; razoring_used += child_ru; rfp_pruned += child_rfp
        lmp_pruned += child_lmp; probcut_pruned += child_pcp; qs_delta_pruned += child_qdp; qs_see_pruned += child_qsp
        iid_searches += child_iid; singular_extensions += child_se

        if null_move_score >= beta:
            null_move_cutoffs += 1
            search_context.pv_table[ply, :].fill(NO_MOVE)
            return (np.int32(beta), NO_MOVE, nodes_searched, quiescence_nodes, cutoffs, tt_hits,
                    null_move_cutoffs, futility_pruned, razoring_used, rfp_pruned, lmp_pruned, probcut_pruned, qs_delta_pruned, qs_see_pruned,
                    iid_searches, singular_extensions)

    if not moves_generated:
        moves = generate_legal_moves(piece_bbs, occupancy_bbs, game_state)

    static_score = -INFINITY
    if depth <= 2 and not is_currently_in_check:
        static_score = evaluate_position(piece_bbs, occupancy_bbs, game_state, lazy=True)

        if ENABLE_RFP and depth == 1 and static_score - RFP_MARGIN_D1 >= beta:
            rfp_pruned += 1
            return (np.int32(beta), NO_MOVE, nodes_searched, quiescence_nodes, cutoffs, tt_hits,
                    null_move_cutoffs, futility_pruned, razoring_used, rfp_pruned, lmp_pruned, probcut_pruned, qs_delta_pruned, qs_see_pruned,
                    iid_searches, singular_extensions)

        if ENABLE_RAZORING and static_score + RAZORING_MARGIN < alpha:
            razor_score, child_q_nodes, child_delta_pruned, child_see_pruned = quiescence_search(piece_bbs, occupancy_bbs, game_state, alpha, alpha + 1, 0, search_context)
            quiescence_nodes += child_q_nodes
            qs_delta_pruned += child_delta_pruned
            qs_see_pruned += child_see_pruned
            if razor_score + RAZORING_MARGIN < alpha:
                razoring_used += 1
                return (np.int32(alpha), NO_MOVE, nodes_searched, quiescence_nodes, cutoffs, tt_hits,
                        null_move_cutoffs, futility_pruned, razoring_used, rfp_pruned, lmp_pruned, probcut_pruned, qs_delta_pruned, qs_see_pruned,
                        iid_searches, singular_extensions)

    if len(moves) == 0:
        search_context.pv_table[ply, :].fill(NO_MOVE)
        if is_currently_in_check:
            return (np.int32(-MATE_SCORE + ply), NO_MOVE, nodes_searched, quiescence_nodes, cutoffs, tt_hits, null_move_cutoffs, futility_pruned, razoring_used, rfp_pruned, lmp_pruned, probcut_pruned, qs_delta_pruned, qs_see_pruned, iid_searches, singular_extensions)
        else:
            return (np.int32(0), NO_MOVE, nodes_searched, quiescence_nodes, cutoffs, tt_hits, null_move_cutoffs, futility_pruned, razoring_used, rfp_pruned, lmp_pruned, probcut_pruned, qs_delta_pruned, qs_see_pruned, iid_searches, singular_extensions)

    scores = score_moves(piece_bbs, occupancy_bbs, game_state, moves, tt_move, search_context.killer_moves[ply*2:ply*2+2], search_context.history_table)
    
    best_move, max_eval = NO_MOVE, -INFINITY
    quiet_move_counter, move_count = 0, 0

    for i in range(len(moves)):
        best_idx = i
        for j in range(i + 1, len(moves)):
            if scores[j] > scores[best_idx]: best_idx = j
        moves[i], moves[best_idx] = moves[best_idx], moves[i]
        scores[i], scores[best_idx] = scores[best_idx], scores[i]
        
        move = moves[i]
        if move == excluded_move: continue
        move_count += 1
        
        opponent_pieces_bb = occupancy_bbs[1] if game_state[0] == 0 else occupancy_bbs[0]
        
        # --- Pre-move checks for extensions and move type ---
        from_sq = get_from_square(move)
        to_sq = get_to_square(move)
        is_capture = (opponent_pieces_bb & BB_SQUARES[to_sq]) != 0
        is_promotion = get_special_move_flag(move) == SPECIAL_MOVE_FLAG_PROMOTION
        is_pseudo_quiet = not is_capture and not is_promotion

        # --- Make the move ---
        unmake_info = make_move(piece_bbs, occupancy_bbs, game_state, move)
        
        # --- Post-move checks ---
        is_giving_check_after_move = is_in_check(piece_bbs, occupancy_bbs, game_state)
        
        # Final determination of quiet move
        is_quiet_move = is_pseudo_quiet and not is_giving_check_after_move

        if is_quiet_move:
            quiet_move_counter += 1
            if ENABLE_LMP and not is_currently_in_check:
                if quiet_move_counter >= LMP_MOVE_COUNT[depth]:
                    lmp_pruned += 1
                    unmake_move(piece_bbs, occupancy_bbs, game_state, move, unmake_info)
                    break

        if ENABLE_FP and is_quiet_move and not is_currently_in_check and static_score != -INFINITY:
            margin = 0
            if depth == 1: margin = FP_MARGIN_D1
            elif depth == 2: margin = FP_MARGIN_D2

            if margin > 0 and static_score + margin < alpha:
                futility_pruned += 1
                unmake_move(piece_bbs, occupancy_bbs, game_state, move, unmake_info)
                continue
        
        # --- Determine total extension ---
        check_extension = 1 if is_giving_check_after_move else 0
        current_extension = check_extension
        if move == tt_move: current_extension = max(current_extension, extension)
        search_depth = depth - 1 + current_extension

        evaluation = 0
        if move_count == 1:
            evaluation, _, child_nodes, child_q_nodes, child_cutoffs, child_tt_hits, child_nmc, child_fp, child_ru, child_rfp, child_lmp, child_pcp, child_qdp, child_qsp, child_iid, child_se = _search(
                piece_bbs, occupancy_bbs, game_state, search_depth, -beta, -alpha, search_context, ply + 1, NO_MOVE)
            evaluation = -evaluation
        else:
            lmr = LMR_REDUCTION if ENABLE_LMR and depth >= LMR_MIN_DEPTH and is_quiet_move and quiet_move_counter >= LMR_MIN_QUIET_MOVE_INDEX else 0
            evaluation, _, child_nodes, child_q_nodes, child_cutoffs, child_tt_hits, child_nmc, child_fp, child_ru, child_rfp, child_lmp, child_pcp, child_qdp, child_qsp, child_iid, child_se = _search(
                piece_bbs, occupancy_bbs, game_state, search_depth - lmr, -alpha - 1, -alpha, search_context, ply + 1, NO_MOVE)
            evaluation = -evaluation
            if evaluation > alpha:
                evaluation, _, child_nodes, child_q_nodes, child_cutoffs, child_tt_hits, child_nmc, child_fp, child_ru, child_rfp, child_lmp, child_pcp, child_qdp, child_qsp, child_iid, child_se = _search(
                    piece_bbs, occupancy_bbs, game_state, search_depth, -beta, -alpha, search_context, ply + 1, NO_MOVE)
                evaluation = -evaluation

        unmake_move(piece_bbs, occupancy_bbs, game_state, move, unmake_info)

        if search_context.stop_flag[0]:
            return (np.int32(0), NO_MOVE, nodes_searched, quiescence_nodes, cutoffs, tt_hits,
                    null_move_cutoffs, futility_pruned, razoring_used, rfp_pruned, lmp_pruned, probcut_pruned, qs_delta_pruned, qs_see_pruned,
                    iid_searches, singular_extensions)

        nodes_searched += child_nodes; quiescence_nodes += child_q_nodes; cutoffs += child_cutoffs; tt_hits += child_tt_hits
        null_move_cutoffs += child_nmc; futility_pruned += child_fp; razoring_used += child_ru; rfp_pruned += child_rfp
        lmp_pruned += child_lmp; probcut_pruned += child_pcp; qs_delta_pruned += child_qdp; qs_see_pruned += child_qsp
        iid_searches += child_iid; singular_extensions += child_se

        if evaluation > max_eval:
            max_eval, best_move = evaluation, move
            search_context.pv_table[ply, ply] = move
            i, j = ply + 1, ply + 1
            while j < MAX_PLY and search_context.pv_table[ply + 1, j] != NO_MOVE:
                search_context.pv_table[ply, i] = search_context.pv_table[ply + 1, j]
                i += 1; j += 1
            if i < MAX_PLY: search_context.pv_table[ply, i] = NO_MOVE

        alpha = max(alpha, evaluation)
        if alpha >= beta:
            cutoffs += 1
            if is_quiet_move:
                bonus = depth * depth
                aggressor_type = find_piece_type_on_square(piece_bbs, get_from_square(move))
                search_context.history_table[aggressor_type, get_to_square(move)] += bonus
                if move != search_context.killer_moves[ply * 2]:
                    search_context.killer_moves[ply * 2 + 1] = search_context.killer_moves[ply * 2]
                    search_context.killer_moves[ply * 2] = move
            break

    final_flag = TT_FLAG_ALPHA if max_eval <= original_alpha else (TT_FLAG_BETA if max_eval >= beta else TT_FLAG_EXACT)
    if max_eval == -INFINITY:
        max_eval = original_alpha
    if best_move == NO_MOVE and len(moves) > 0: best_move = moves[0]
    
    tt_score = max_eval
    if tt_score > MATE_IN_MAX_PLY: tt_score += ply
    elif tt_score < -MATE_IN_MAX_PLY: tt_score -= ply
    store_tt(search_context.transposition_table, zobrist_key, depth, tt_score, final_flag, best_move, search_context.tt_generation)

    return (max_eval, best_move, nodes_searched, quiescence_nodes, cutoffs, tt_hits,
            null_move_cutoffs, futility_pruned, razoring_used, rfp_pruned, lmp_pruned, probcut_pruned, qs_delta_pruned, qs_see_pruned,
            iid_searches, singular_extensions)

def iterative_deepening_search(piece_bbs, occupancy_bbs, game_state, max_depth, time_config, search_context, game_history_list=None, tt_generation=0):
    start_time = time.time()

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
    total_nodes, total_q_nodes, total_cutoffs, total_tt_hits = (np.uint64(v) for v in [0]*4)
    total_nmc, total_fp, total_ru, total_rfp, total_lmp, total_pcp, total_qdp, total_qsp = (np.uint64(v) for v in [0]*8)
    total_iid, total_se = (np.uint64(v) for v in [0]*2)
    last_completed_depth = 0
    best_move_from_last_depth = NO_MOVE

    for current_depth in range(1, max_depth + 1):
        alpha, beta = (-INFINITY, INFINITY) if current_depth <= 1 else (last_score - ASPIRATION_WINDOW_SIZE, last_score + ASPIRATION_WINDOW_SIZE)

        search_context.nodes_searched = np.uint64(0)

        score, _, nodes, q_nodes, cutoffs, tt_hits, nmc, fp, ru, rfp, lmp, pcp, qdp, qsp, iid, se = _search(
            piece_bbs, occupancy_bbs, game_state, current_depth, alpha, beta, search_context, 0, NO_MOVE)

        if search_context.stop_flag[0]:
            log_info(f"Search stopped at depth {current_depth} due to time limit.")
            break

        if score <= alpha or score >= beta:
            log_info(f"depth {current_depth} aspiration window failed, re-searching...")
            alpha, beta = -INFINITY, INFINITY
            search_context.nodes_searched = np.uint64(0)
            score, _, nodes, q_nodes, cutoffs, tt_hits, nmc, fp, ru, rfp, lmp, pcp, qdp, qsp, iid, se = _search(
                piece_bbs, occupancy_bbs, game_state, current_depth, alpha, beta, search_context, 0, NO_MOVE)
            
            if search_context.stop_flag[0]:
                log_info(f"Search stopped during re-search at depth {current_depth} due to time limit.")
                break

        # --- This block only runs if the search for the current depth was completed ---
        last_completed_depth = current_depth
        last_score = score

        total_nodes += search_context.nodes_searched
        total_q_nodes += q_nodes
        total_cutoffs += cutoffs; total_tt_hits += tt_hits
        total_nmc += nmc; total_fp += fp; total_ru += ru; total_rfp += rfp; total_lmp += lmp; total_pcp += pcp; total_qdp += qdp; total_qsp += qsp
        total_iid += iid; total_se += se

        tt_entry = probe_tt(transposition_table, game_state[4])
        if tt_entry['flag'] != TT_FLAG_NONE and tt_entry['best_move'] != NO_MOVE:
             best_move_from_last_depth = tt_entry['best_move']
        
        elapsed_time_ms = (time.time() - start_time) * 1000
        pv_moves = [move_to_uci(search_context.pv_table[0, i]) for i in range(MAX_PLY) if search_context.pv_table[0, i] != NO_MOVE]
        pv_string = " ".join(pv_moves)
        uci_score_string = format_score_for_uci(score)

        nps = int(total_nodes / (elapsed_time_ms / 1000)) if elapsed_time_ms > 0 else 0
        print(f"info depth {current_depth} score {uci_score_string} nodes {total_nodes} nps {nps} time {int(elapsed_time_ms)} pv {pv_string}")

        if "mate" in uci_score_string:
            break

        # Predictive soft time limit
        if maximum_time_ms > 0:
            if optimum_time_ms > 0 and elapsed_time_ms > optimum_time_ms:
                log_info(f"Optimum time reached at depth {current_depth}. Stopping.")
                break
            if elapsed_time_ms * 2.5 > maximum_time_ms:
                log_info(f"Predictive termination at depth {current_depth} to avoid timeout.")
                break

    final_best_move = best_move_from_last_depth
    return (final_best_move, last_score, total_nodes, total_q_nodes, total_cutoffs, total_tt_hits,
            last_completed_depth, total_nmc, total_fp, total_ru, total_rfp, total_lmp, total_pcp, total_qdp, total_qsp,
            total_iid, total_se)

def format_score_for_uci(score):
    if abs(score) > MATE_IN_MAX_PLY:
        if score > 0:
            moves_to_mate = (MATE_SCORE - score + 1) // 2
            return f"mate {moves_to_mate}"
        else:
            moves_to_mate = (MATE_SCORE + score + 1) // 2
            return f"mate {-moves_to_mate}"
    return f"cp {score}"
