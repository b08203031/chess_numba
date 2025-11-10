# chess_engine/search.py
import time
import numba
import numpy as np

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
    ENABLE_SINGULAR_EXTENSIONS, MIN_SINGULAR_DEPTH, SINGULAR_EXTENSION_MARGIN
)
from chess_engine.bitboard_utils import find_piece_type_on_square
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


@numba.njit(cache=True)
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

@numba.njit(cache=True)
def _quicksort_recursive(moves, scores, low, high):
    if low < high:
        pi = _partition(moves, scores, low, high)
        _quicksort_recursive(moves, scores, low, pi - 1)
        _quicksort_recursive(moves, scores, pi + 1, high)

@numba.njit(cache=True)
def score_moves(piece_bbs, occupancy_bbs, game_state, moves, tt_move, killer_moves_at_ply, history_table):
    scores = np.zeros(len(moves), dtype=np.int32)
    opponent_pieces_bb = occupancy_bbs[1] if game_state[0] == 0 else occupancy_bbs[0]
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
        scores[i] = score
    return scores

quiescence_search_return_type = numba.types.Tuple([
    numba.int32, numba.uint64, numba.uint64, numba.uint64
])

@numba.njit(quiescence_search_return_type(
    piece_bbs_signature, occupancy_bbs_signature, game_state_signature,
    numba.int32, numba.int32, numba.int32
), cache=True)
def quiescence_search(piece_bbs, occupancy_bbs, game_state, alpha, beta, ply):
    q_nodes = np.uint64(1)
    delta_pruned = np.uint64(0)
    see_pruned = np.uint64(0)

    if ply >= MAX_QUIESCENCE_DEPTH:
        return evaluate_position(piece_bbs, occupancy_bbs, game_state), q_nodes, delta_pruned, see_pruned

    stand_pat = evaluate_position(piece_bbs, occupancy_bbs, game_state)
    if stand_pat >= beta:
        return beta, q_nodes, delta_pruned, see_pruned
    alpha = max(alpha, stand_pat)

    capture_moves = generate_captures(piece_bbs, occupancy_bbs, game_state)
    if len(capture_moves) == 0:
        return stand_pat, q_nodes, delta_pruned, see_pruned

    for move in capture_moves:
        # --- Delta Pruning ---
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
            if see(piece_bbs, occupancy_bbs, game_state, get_from_square(move), get_to_square(move)) < SEE_THRESHOLD:
                see_pruned += 1
                continue

        unmake_info = make_move(piece_bbs, occupancy_bbs, game_state, move)
        score, child_q_nodes, child_delta_pruned, child_see_pruned = quiescence_search(
            piece_bbs, occupancy_bbs, game_state, -beta, -alpha, ply + 1
        )
        unmake_move(piece_bbs, occupancy_bbs, game_state, move, unmake_info)

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

@numba.njit(search_return_type(
    piece_bbs_signature, occupancy_bbs_signature, game_state_signature,
    numba.int32, numba.int32, numba.int32, search_context_type, numba.int32, numba.uint16
), cache=True)
def _search(piece_bbs, occupancy_bbs, game_state, depth, alpha, beta, search_context, ply, excluded_move: np.uint16 = NO_MOVE):
    # (The body of this function is long, so it's omitted for brevity in this plan step)
    # The logic will be updated to use the make-unmake pattern as described.
    nodes_searched = np.uint64(1)
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

    if ply >= MAX_PLY:
        search_context.pv_table[ply, :].fill(NO_MOVE)
        return (evaluate_position(piece_bbs, occupancy_bbs, game_state), NO_MOVE, nodes_searched, quiescence_nodes, cutoffs, tt_hits,
                null_move_cutoffs, futility_pruned, razoring_used, rfp_pruned, lmp_pruned, probcut_pruned, qs_delta_pruned, qs_see_pruned,
                iid_searches, singular_extensions)

    original_alpha = alpha
    zobrist_key = game_state[4]

    tt_move = NO_MOVE
    tt_entry = probe_tt(search_context.transposition_table, zobrist_key)
    if tt_entry['flag'] != TT_FLAG_NONE and tt_entry['depth'] >= depth:
        tt_hits += 1
        tt_score = np.int32(tt_entry['score'])
        if tt_score > MATE_IN_MAX_PLY: tt_score += ply
        elif tt_score < -MATE_IN_MAX_PLY: tt_score -= ply

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

    if ENABLE_IID and depth >= 5 and tt_move == NO_MOVE:
        iid_searches += 1
        _search(piece_bbs, occupancy_bbs, game_state, depth - 3, -INFINITY, INFINITY, search_context, ply + 1, NO_MOVE)
        tt_entry = probe_tt(search_context.transposition_table, zobrist_key)
        if tt_entry['flag'] != TT_FLAG_NONE: tt_move = tt_entry['best_move']

    extension = 0
    if ENABLE_SINGULAR_EXTENSIONS and depth >= MIN_SINGULAR_DEPTH and tt_move != NO_MOVE and tt_entry['flag'] == TT_FLAG_EXACT:
        tt_score = np.int32(tt_entry['score'])
        exclusion_beta = tt_score - SINGULAR_EXTENSION_MARGIN
        exclusion_score, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = _search(
            piece_bbs, occupancy_bbs, game_state, depth - 3, exclusion_beta - 1, exclusion_beta, search_context, ply + 1, tt_move)
        if exclusion_score < exclusion_beta:
            extension = 1
            singular_extensions += 1

    is_currently_in_check = is_in_check(piece_bbs, occupancy_bbs, game_state)
    
    if depth == 0:
        search_context.pv_table[ply, :].fill(NO_MOVE)
        eval_score, q_nodes, child_delta_pruned, child_see_pruned = quiescence_search(piece_bbs, occupancy_bbs, game_state, alpha, beta, 0)
        qs_delta_pruned += child_delta_pruned
        qs_see_pruned += child_see_pruned
        return (eval_score, NO_MOVE, nodes_searched, q_nodes, cutoffs, tt_hits,
                null_move_cutoffs, futility_pruned, razoring_used, rfp_pruned, lmp_pruned, probcut_pruned, qs_delta_pruned, qs_see_pruned,
                iid_searches, singular_extensions)

    if ENABLE_NMP and depth >= 3 and not is_currently_in_check and has_sufficient_material(piece_bbs, game_state[0]):
        original_state_for_null = game_state.copy()
        make_null_move(game_state)
        
        # Correctly unpack all stats from the null move search
        (null_move_score, _, child_nodes, child_q_nodes, child_cutoffs, child_tt_hits,
         child_nmc, child_fp, child_ru, child_rfp, child_lmp, child_pcp,
         child_qdp, child_qsp, child_iid, child_se) = _search(
            piece_bbs, occupancy_bbs, game_state, depth - 1 - NULL_MOVE_REDUCTION,
            -beta, -beta + 1, search_context, ply + 1, NO_MOVE
        )

        game_state[:] = original_state_for_null # Restore state
        null_move_score = -null_move_score

        # Accumulate all stats from the null move search
        nodes_searched += child_nodes; quiescence_nodes += child_q_nodes; cutoffs += child_cutoffs; tt_hits += child_tt_hits
        null_move_cutoffs += child_nmc; futility_pruned += child_fp; razoring_used += child_ru; rfp_pruned += child_rfp
        lmp_pruned += child_lmp; probcut_pruned += child_pcp; qs_delta_pruned += child_qdp; qs_see_pruned += child_qsp
        iid_searches += child_iid; singular_extensions += child_se

        if null_move_score >= beta:
            null_move_cutoffs += 1 # Increment again for the actual cutoff
            search_context.pv_table[ply, :].fill(NO_MOVE)
            return (np.int32(beta), NO_MOVE, nodes_searched, quiescence_nodes, cutoffs, tt_hits,
                    null_move_cutoffs, futility_pruned, razoring_used, rfp_pruned, lmp_pruned, probcut_pruned, qs_delta_pruned, qs_see_pruned,
                    iid_searches, singular_extensions)

    moves = generate_legal_moves(piece_bbs, occupancy_bbs, game_state)

    # --- Shallow Pruning Techniques (RFP, Razoring) ---
    static_score = -INFINITY # Use a sentinel value
    if depth <= 2 and not is_currently_in_check:
        static_score = evaluate_position(piece_bbs, occupancy_bbs, game_state)

        # Reverse Futility Pruning (RFP)
        if ENABLE_RFP and depth == 1 and static_score - RFP_MARGIN_D1 >= beta:
            rfp_pruned += 1
            return (np.int32(beta), NO_MOVE, nodes_searched, quiescence_nodes, cutoffs, tt_hits,
                    null_move_cutoffs, futility_pruned, razoring_used, rfp_pruned, lmp_pruned, probcut_pruned, qs_delta_pruned, qs_see_pruned,
                    iid_searches, singular_extensions)

        # Razoring
        if ENABLE_RAZORING and static_score + RAZORING_MARGIN < alpha:
            # For Razoring, we need to perform a quick quiescence search.
            # This logic remains the same for now.
            razor_score, child_q_nodes, child_delta_pruned, child_see_pruned = quiescence_search(piece_bbs, occupancy_bbs, game_state, alpha, alpha + 1, 0)
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
        
        # 1. First, get the information we need from the CURRENT board state
        #    BEFORE we modify it.
        opponent_pieces_bb = occupancy_bbs[1] if game_state[0] == 0 else occupancy_bbs[0]

        # 2. Now, make the move and change the board state in-place.
        unmake_info = make_move(piece_bbs, occupancy_bbs, game_state, move)

        # 3. Finally, calculate is_capture and is_quiet_move using the information
        #    we saved from the original state.
        is_giving_check_after_move = is_in_check(piece_bbs, occupancy_bbs, game_state)
        is_capture = (opponent_pieces_bb & BB_SQUARES[get_to_square(move)]) != 0
        is_quiet_move = not is_capture and not (get_special_move_flag(move) == SPECIAL_MOVE_FLAG_PROMOTION) and not is_giving_check_after_move
        
        # --- Futility Pruning (F-Pruning) ---
        if ENABLE_FP and is_quiet_move and not is_currently_in_check:
            # Ensure static_score is computed if not already done for shallow depths
            # This reuses the static_score calculated in the shallow pruning section
            if static_score == -INFINITY:
                static_score = evaluate_position(piece_bbs, occupancy_bbs, game_state)

            margin = 0
            if depth == 1: margin = FP_MARGIN_D1
            elif depth == 2: margin = FP_MARGIN_D2

            if margin > 0 and static_score + margin < alpha:
                futility_pruned += 1
                # IMPORTANT: We MUST unmake the move before we continue the loop,
                # because make_move was called at the top of the loop.
                unmake_move(piece_bbs, occupancy_bbs, game_state, move, unmake_info)
                continue # Skip this quiet move

        current_extension = 1 if is_giving_check_after_move else 0
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
        if is_quiet_move: quiet_move_counter += 1

    final_flag = TT_FLAG_ALPHA if max_eval <= original_alpha else (TT_FLAG_BETA if max_eval >= beta else TT_FLAG_EXACT)
    if best_move == NO_MOVE and len(moves) > 0: best_move = moves[0]
    
    tt_score = max_eval
    if tt_score > MATE_IN_MAX_PLY: tt_score -= ply
    elif tt_score < -MATE_IN_MAX_PLY: tt_score += ply
    store_tt(search_context.transposition_table, zobrist_key, depth, tt_score, final_flag, best_move)

    return (max_eval, best_move, nodes_searched, quiescence_nodes, cutoffs, tt_hits,
            null_move_cutoffs, futility_pruned, razoring_used, rfp_pruned, lmp_pruned, probcut_pruned, qs_delta_pruned, qs_see_pruned,
            iid_searches, singular_extensions)

def iterative_deepening_search(piece_bbs, occupancy_bbs, game_state, max_depth, max_time_ms, transposition_table):
    from .move import move_to_uci
    start_time = time.time()

    pv_table = np.zeros((MAX_PLY, MAX_PLY), dtype=np.uint16)
    history_table = np.zeros((12, 64), dtype=np.int32)
    killer_moves = np.zeros(MAX_PLY * 2, dtype=np.uint16)
    search_context = SearchContext(transposition_table, killer_moves, pv_table, history_table)

    last_score, best_move_total = 0, NO_MOVE
    nodes_searched, quiescence_nodes, total_nodes_searched, total_cutoffs, total_tt_hits = np.uint64(0), np.uint64(0), np.uint64(0), np.uint64(0), np.uint64(0)
    total_nmc, total_fp, total_ru, total_rfp, total_lmp, total_pcp, total_qdp, total_qsp = np.uint64(0), np.uint64(0), np.uint64(0), np.uint64(0), np.uint64(0), np.uint64(0), np.uint64(0), np.uint64(0)
    total_iid, total_se = np.uint64(0), np.uint64(0)
    last_completed_depth = 0

    for current_depth in range(1, max_depth + 1):
        alpha, beta = (-INFINITY, INFINITY) if current_depth == 1 else (last_score - ASPIRATION_WINDOW_SIZE, last_score + ASPIRATION_WINDOW_SIZE)

        # We must pass copies as _search modifies the state
        p_bbs_copy, o_bbs_copy, g_state_copy = piece_bbs.copy(), occupancy_bbs.copy(), game_state.copy()
        pv_table.fill(0)

        score, _, nodes, q_nodes, cutoffs, tt_hits, nmc, fp, ru, rfp, lmp, pcp, qdp, qsp, iid, se = _search(
            p_bbs_copy, o_bbs_copy, g_state_copy, current_depth, alpha, beta, search_context, 0, NO_MOVE)

        if score <= alpha or score >= beta: # Aspiration window failed
            log_info(f"depth {current_depth} aspiration window failed, re-searching...")
            alpha, beta = -INFINITY, INFINITY
            p_bbs_copy, o_bbs_copy, g_state_copy = piece_bbs.copy(), occupancy_bbs.copy(), game_state.copy()
            pv_table.fill(0)
            score, _, nodes, q_nodes, cutoffs, tt_hits, nmc, fp, ru, rfp, lmp, pcp, qdp, qsp, iid, se = _search(
                p_bbs_copy, o_bbs_copy, g_state_copy, current_depth, alpha, beta, search_context, 0, NO_MOVE)

        last_score = score
        nodes_searched += nodes; quiescence_nodes += q_nodes; total_cutoffs += cutoffs; total_tt_hits += tt_hits
        total_nmc += nmc; total_fp += fp; total_ru += ru; total_rfp += rfp; total_lmp += lmp; total_pcp += pcp; total_qdp += qdp; total_qsp += qsp
        total_iid += iid; total_se += se

        tt_entry = probe_tt(transposition_table, game_state[4])
        if tt_entry['flag'] != TT_FLAG_NONE: best_move_total = tt_entry['best_move']

        elapsed_time = (time.time() - start_time) * 1000
        pv_moves = [move_to_uci(pv_table[0, i]) for i in range(MAX_PLY) if pv_table[0, i] != NO_MOVE]
        pv_string = " ".join(pv_moves)

        # --- Format Score for UCI ---
        uci_score_string = format_score_for_uci(score)

        # --- Print UCI Info String ---
        total_nodes_searched = nodes_searched + quiescence_nodes
        nps = int(total_nodes_searched / (elapsed_time / 1000)) if elapsed_time > 0 else 0
        print(f"info depth {current_depth} score {uci_score_string} nodes {total_nodes_searched} nps {nps} time {int(elapsed_time)} pv {pv_string}")

        last_completed_depth = current_depth
        if "mate" in uci_score_string: break
        if elapsed_time > max_time_ms: break
            
    return (best_move_total, last_score, nodes_searched, quiescence_nodes, total_cutoffs, total_tt_hits,
            last_completed_depth, total_nmc, total_fp, total_ru, total_rfp, total_lmp, total_pcp, total_qdp, total_qsp,
            total_iid, total_se)

def format_score_for_uci(score):
    """Formats the internal score for UCI, adjusting for mate distance."""
    if abs(score) > MATE_IN_MAX_PLY:
        if score > 0:
            moves_to_mate = (MATE_SCORE - score + 1) // 2
            return f"mate {moves_to_mate}"
        else:
            moves_to_mate = (MATE_SCORE + score + 1) // 2
            return f"mate {-moves_to_mate}"
    return f"cp {score}"
