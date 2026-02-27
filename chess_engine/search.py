# chess_engine/search.py
import time
import numba
import math
import numpy as np
import threading

from chess_engine.evaluation import evaluate_position
from chess_engine.move_generator import (
    generate_legal_moves, is_in_check, has_sufficient_material, generate_captures,
    generate_legal_moves_buffer, generate_captures_buffer
)
from chess_engine.bitboard_utils import get_lsb_index
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
    ENABLE_SINGULAR_EXTENSIONS, MIN_SINGULAR_DEPTH, SINGULAR_EXTENSION_MARGIN,
    STOP_SEARCH_FLAG, MAX_HISTORY,
    SCORE_TT_MOVE, SCORE_GOOD_CAPTURE_BONUS, SCORE_KILLER_1,
    SCORE_KILLER_2, SCORE_COUNTER_MOVE, SCORE_BAD_CAPTURE_PENALTY, NMP_STATIC_MARGIN,
    ENABLE_SHALLOW_SEE_PRUNING, ENABLE_HISTORY_PRUNING, PRUNING_SHALLOW_DEPTH,
    PRUNING_CAPTURE_SEE_MARGIN, PRUNING_QUIET_SEE_MARGIN, PRUNING_HISTORY_THRESHOLD,
    WHITE, BLACK, PAWN_KEY_INDEX, CORRECTION_HISTORY_SIZE, CORRECTION_HISTORY_LIMIT,
    CONTINUATION_HISTORY_FACTOR, MAX_HISTORY
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

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def update_history(history_table, piece_type, to_square, bonus):
    """
    Updates the history table using the gravity formula:
    history += bonus - history * abs(bonus) / MAX_HISTORY
    This automatically prevents overflow and scales updates.
    """
    current_value = history_table[piece_type, to_square]
    clamped_bonus = min(max(bonus, -MAX_HISTORY), MAX_HISTORY)
    
    # Gravity formula
    new_value = current_value + clamped_bonus - (current_value * abs(clamped_bonus)) // MAX_HISTORY
    history_table[piece_type, to_square] = new_value

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def update_butterfly_history(butterfly_table, from_sq, to_sq, bonus):
    """
    Updates the butterfly history table using the gravity formula.
    """
    current_value = butterfly_table[from_sq, to_sq]
    clamped_bonus = min(max(bonus, -MAX_HISTORY), MAX_HISTORY)
    
    # Gravity formula
    new_value = current_value + clamped_bonus - (current_value * abs(clamped_bonus)) // MAX_HISTORY
    butterfly_table[from_sq, to_sq] = new_value

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def update_capture_history(capture_history, piece_type, to_square, victim_type, bonus):
    """
    Updates the capture history table using the gravity formula.
    capture_history: [12, 64, 12] (aggressor_piece, to_square, victim_piece)
    """
    current_value = capture_history[piece_type, to_square, victim_type]
    clamped_bonus = min(max(bonus, -MAX_HISTORY), MAX_HISTORY)
    
    # Gravity formula
    new_value = current_value + clamped_bonus - (current_value * abs(clamped_bonus)) // MAX_HISTORY
    capture_history[piece_type, to_square, victim_type] = new_value

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def update_continuation_history(context, prev_move, prev_piece, curr_move, curr_piece, bonus):
    """
    Updates the continuation history table.
    """
    prev_to = get_to_square(prev_move)
    curr_to = get_to_square(curr_move)
    
    current_val = context.continuation_history[prev_piece, prev_to, curr_piece, curr_to]
    clamped_bonus = min(max(bonus, -MAX_HISTORY), MAX_HISTORY)
    
    new_val = current_val + clamped_bonus - (current_val * abs(clamped_bonus)) // MAX_HISTORY
    context.continuation_history[prev_piece, prev_to, curr_piece, curr_to] = new_val

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def get_lmr_reduction(depth, move_count, history_score, improving, is_pv):
    """
    Calculates LMR reduction based on depth, move_count, history score, improving flag and node type.
    """
    if depth < 2 or move_count < 2:
        return 0

    ld = math.log(float(depth))
    lmc = math.log(float(move_count))

    # Base reduction
    reduction = 0.5 + (ld * lmc) / 2.25
    
    # PV adjustment: PV nodes are searched more carefully
    if not is_pv:
        reduction += 0.5
    
    # Improving adjustment: If position is not improving, reduce more aggressively
    if not improving:
        reduction += 0.5

    # History Adjustment: Scale +/- 1.5 reduction for max history
    history_adjustment = (history_score / float(MAX_HISTORY)) * 1.5
    reduction -= history_adjustment

    return max(0, int(reduction))

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def score_moves(piece_bbs, occupancy_bbs, game_state, moves, scores, move_count, tt_move, killer_moves_at_ply, history_table, counter_move, pinned_white, pinned_black, search_context, ply):
    side_to_move = game_state[0]
    opponent_pieces_bb = occupancy_bbs[1] if side_to_move == 0 else occupancy_bbs[0]
    
    prev_move = NO_MOVE
    prev_piece = -1
    
    if ply > 0:
        prev_move = search_context.move_stack[ply-1]
        if prev_move != NO_MOVE:
            # We need to find what piece was moved in the previous move.
            # The piece is currently at prev_move.to_square (unless captured, but move_stack tracks moves made)
            # wait, move_stack tracks moves made on the board.
            # The previous move was made by the opponent (1-side_to_move).
            # The piece should be at get_to_square(prev_move).
            prev_to = get_to_square(prev_move)
            # Find piece type on square for opponent
            prev_piece = find_piece_type_for_square(piece_bbs, prev_to, 1 - side_to_move)

    for i in range(move_count):
        move = moves[i]
        score = 0
        if move == tt_move:
            score = SCORE_TT_MOVE
        else:
            to_square = get_to_square(move)
            is_capture = (opponent_pieces_bb & BB_SQUARES[to_square]) != 0
            if is_capture:
                # Use SEE to distinguish Good vs Bad captures
                from_sq = get_from_square(move)
                
                # Optimization: Lookup piece types once and reuse them in see_ge and MVV_LVA
                victim_type = find_piece_type_on_square_side(piece_bbs, to_square, 1 - side_to_move)
                aggressor_type = find_piece_type_on_square_side(piece_bbs, from_sq, side_to_move)

                # Optimization: Use see_ge(0) instead of full see()
                is_good_capture = see_ge(piece_bbs, occupancy_bbs, side_to_move, from_sq, to_square, 0, pinned_white, pinned_black, aggressor_type, victim_type)
                
                mvv_lva = 0
                if victim_type != -1:
                    mvv_lva = (MG_MATERIAL_VALUES[victim_type % 6] - MG_MATERIAL_VALUES[aggressor_type % 6])

                if is_good_capture:
                    score = SCORE_GOOD_CAPTURE_BONUS + mvv_lva
                    # Capture History for good captures
                    if victim_type != -1:
                        score += search_context.capture_history[aggressor_type, to_square, victim_type]
                else:
                    # Penalize bad captures. We could calculate exact see for sorting bad captures,
                    # but using MVV_LVA is acceptable for now to save performance.
                    score = SCORE_BAD_CAPTURE_PENALTY + mvv_lva
                    # Even bad captures might be better if they worked before
                    if victim_type != -1:
                        score += search_context.capture_history[aggressor_type, to_square, victim_type]
            else:
                if move == killer_moves_at_ply[0]:
                    score = SCORE_KILLER_1
                elif move == killer_moves_at_ply[1]:
                    score = SCORE_KILLER_2
                elif move == counter_move:
                    score = SCORE_COUNTER_MOVE
                else:
                    from_sq = get_from_square(move)
                    aggressor_type = find_piece_type_on_square_side(piece_bbs, from_sq, side_to_move)
                    score = history_table[aggressor_type, to_square]
                    
                    # Butterfly History
                    score += search_context.butterfly_history[from_sq, to_square]
                    
                    # Continuation History
                    if prev_move != NO_MOVE and prev_piece != -1:
                        cont_score = search_context.continuation_history[prev_piece, get_to_square(prev_move), aggressor_type, to_square]
                        score += cont_score * CONTINUATION_HISTORY_FACTOR

        scores[i] = score

quiescence_search_return_type = numba.types.Tuple([
    numba.int32, numba.uint64, numba.uint64, numba.uint64
])

# @numba.njit(quiescence_search_return_type(
#     piece_bbs_signature, occupancy_bbs_signature, game_state_signature,
#     numba.int32, numba.int32, numba.int32, search_context_type
# ), cache=True)
@numba.njit(cache=True)
def quiescence_search(piece_bbs, occupancy_bbs, game_state, alpha, beta, ply, search_context, q_ply):
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

    if ply >= MAX_PLY:
        return evaluate_position(piece_bbs, occupancy_bbs, game_state, lazy=False), q_nodes, delta_pruned, see_pruned

    is_currently_in_check = is_in_check(piece_bbs, occupancy_bbs, game_state)

    move_count = 0
    if is_currently_in_check:
        # If in check, we must evade. No stand_pat (can't stand pat in check).
        # We must generate ALL legal moves (evasions).
        move_count = generate_legal_moves_buffer(piece_bbs, occupancy_bbs, game_state, search_context.moves_buffer, ply)
        if move_count == 0:
            # Checkmate
            return np.int32(-MATE_SCORE + ply), q_nodes, delta_pruned, see_pruned
    else:
        # If not in check, we can stand pat (static evaluation)
        if q_ply >= MAX_QUIESCENCE_DEPTH:
            return evaluate_position(piece_bbs, occupancy_bbs, game_state, lazy=False), q_nodes, delta_pruned, see_pruned

        stand_pat = evaluate_position(piece_bbs, occupancy_bbs, game_state, lazy=False)
        if stand_pat >= beta:
            return beta, q_nodes, delta_pruned, see_pruned
        alpha = max(alpha, stand_pat)

        move_count = generate_captures_buffer(piece_bbs, occupancy_bbs, game_state, search_context.moves_buffer, ply)
        if move_count == 0:
            return stand_pat, q_nodes, delta_pruned, see_pruned

    # --- Sort moves in QSearch ---
    moves = search_context.moves_buffer[ply]
    scores = search_context.move_scores[ply]

    # Safe access to killers:
    safe_ply = min(ply, MAX_PLY - 1)

    # Optimization: Calculate pinned pieces once for QS pruning logic and score_moves
    pinned_white = get_pinned_pieces(piece_bbs, occupancy_bbs, WHITE)
    pinned_black = get_pinned_pieces(piece_bbs, occupancy_bbs, BLACK)

    # Use score_moves to sort captures (SEE >= 0 first).
    score_moves(
        piece_bbs, occupancy_bbs, game_state, 
        moves, 
        scores, 
        move_count,
        NO_MOVE, # No TT move in QSearch loop usually
        search_context.killer_moves[safe_ply*2:safe_ply*2+2], 
        search_context.history_table, 
        NO_MOVE, # No counter move
        pinned_white,
        pinned_black,
        search_context,
        ply
    )

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
                    unmake_info = make_move(piece_bbs, occupancy_bbs, game_state, move)
                    is_check = is_in_check(piece_bbs, occupancy_bbs, game_state)
                    unmake_move(piece_bbs, occupancy_bbs, game_state, move, unmake_info)
                    if not is_check:
                        delta_pruned += 1
                        continue

        if not is_currently_in_check:
            if ENABLE_SEE_IN_QUIESCENCE:
                # Optimization: If score indicates Good Capture (SEE >= 0), skip see check.
                # Only check SEE if it was classified as Bad Capture (or if logic changes).
                # Good Capture Score >= SCORE_GOOD_CAPTURE_BONUS (20000).
                # SEE_THRESHOLD is -100.
                # If SEE >= 0, then SEE >= -100 is always True.
                if score_val < SCORE_GOOD_CAPTURE_BONUS:
                    side_to_move = game_state[0]
                    if not see_ge(piece_bbs, occupancy_bbs, side_to_move, get_from_square(move), get_to_square(move), SEE_THRESHOLD, pinned_white, pinned_black):
                        see_pruned += 1
                        continue

        unmake_info = make_move(piece_bbs, occupancy_bbs, game_state, move)
        score, child_q_nodes, child_delta_pruned, child_see_pruned = quiescence_search(
            piece_bbs, occupancy_bbs, game_state, -beta, -alpha, ply + 1, search_context, q_ply + 1
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

@numba.njit(cache=True)
def _search(piece_bbs, occupancy_bbs, game_state, depth, alpha, beta, search_context, ply, excluded_move: np.uint16 = NO_MOVE, is_pv: bool = True):
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

    if ply >= MAX_PLY:
        return (evaluate_position(piece_bbs, occupancy_bbs, game_state, lazy=False), NO_MOVE, nodes_searched, quiescence_nodes, cutoffs, tt_hits,
                null_move_cutoffs, futility_pruned, razoring_used, rfp_pruned, lmp_pruned, probcut_pruned, qs_delta_pruned, qs_see_pruned,
                iid_searches, singular_extensions)

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
                if repetition_count >= 2: break
        
        if repetition_count < 2:
            # Check Game History
            # Number of moves in history that are within the halfmove clock
            # halfmove_clock includes moves from both path and history.
            hist_to_check = halfmove_clock - ply
            if hist_to_check >= 2:
                # History[count-1] is opponent's last move.
                # History[count-2] is our last move (same side-to-move as current).
                start_idx = search_context.game_history_count - 2
                end_idx = max(-1, search_context.game_history_count - hist_to_check - 1)
                for i in range(start_idx, end_idx, -2):
                    if search_context.game_history[i] == zobrist_key:
                        repetition_count += 1
                        if repetition_count >= 2: break

    # Avoid 2nd repetition (3rd occurrence total)
    # Crucial fix: Do not prune at the root (ply 0).
    if ply > 0 and repetition_count >= 2:
        search_context.pv_table[ply, ply] = NO_MOVE
        return (np.int32(0), NO_MOVE, nodes_searched, quiescence_nodes, cutoffs, tt_hits,
                null_move_cutoffs, futility_pruned, razoring_used, rfp_pruned, lmp_pruned, probcut_pruned, qs_delta_pruned, qs_see_pruned,
                iid_searches, singular_extensions)
    
    # Initialize PV for this ply to avoid ghost moves from previous searches
    search_context.pv_table[ply, ply] = NO_MOVE

    original_alpha = alpha
    
    # --- Initialization ---
    move_count = 0
    moves_generated = False

    tt_move = NO_MOVE
    tt_entry = probe_tt(search_context.transposition_table, zobrist_key)

    # 1. ALWAYS retrieve the best move if available (Critical for Move Ordering)
    if tt_entry['flag'] != TT_FLAG_NONE:
        tt_move = tt_entry['best_move']

        # 2. ONLY perform a Score Cutoff (Return) if:
        #    a. We are NOT at the Root Node (ply > 0)
        #    b. The stored depth is sufficient
        #    c. The score bounds (Alpha/Beta) are valid for a cutoff
        #    d. We are NOT in a singular extension search (excluded_move == NO_MOVE)
        #       OR the TT move is NOT the excluded move.
        if ply > 0 and tt_entry['depth'] >= depth and (excluded_move == NO_MOVE or tt_move != excluded_move):
            tt_hits += 1
            tt_score = np.int32(tt_entry['score'])

            # Adjust mate scores relative to the current ply
            if tt_score > MATE_IN_MAX_PLY: tt_score -= ply
            elif tt_score < -MATE_IN_MAX_PLY: tt_score += ply

            should_cutoff = False
            if tt_entry['flag'] == TT_FLAG_EXACT:
                should_cutoff = True
            elif tt_entry['flag'] == TT_FLAG_ALPHA and tt_score <= alpha:
                should_cutoff = True
                beta = min(beta, tt_score) # Technically not needed for return, but good for consistency
            elif tt_entry['flag'] == TT_FLAG_BETA and tt_score >= beta:
                should_cutoff = True
                alpha = max(alpha, tt_score)

            if should_cutoff:
                search_context.pv_table[ply, ply] = NO_MOVE
                return (tt_score, tt_entry['best_move'], nodes_searched, quiescence_nodes, cutoffs, tt_hits,
                        null_move_cutoffs, futility_pruned, razoring_used, rfp_pruned, lmp_pruned, probcut_pruned, qs_delta_pruned, qs_see_pruned,
                        iid_searches, singular_extensions)

    if ENABLE_IID and depth >= 8 and tt_move == NO_MOVE:
        iid_searches += 1
        # IID is non-PV search
        _search(piece_bbs, occupancy_bbs, game_state, depth - 5, -INFINITY, INFINITY, search_context, ply + 1, NO_MOVE, False)

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
        # Singular search is non-PV
        res_ex = _search(
            piece_bbs, occupancy_bbs, game_state, depth - 3, exclusion_beta - 1, exclusion_beta, search_context, ply + 1, tt_move, False)
        exclusion_score = res_ex[0]

        if search_context.stop_flag[0]:
            return (np.int32(0), NO_MOVE, nodes_searched, quiescence_nodes, cutoffs, tt_hits,
                    null_move_cutoffs, futility_pruned, razoring_used, rfp_pruned, lmp_pruned, probcut_pruned, qs_delta_pruned, qs_see_pruned,
                    iid_searches, singular_extensions)

        if exclusion_score < exclusion_beta:
            extension = 1
            singular_extensions += 1
            # Double Singular Extension: If it fails by a larger margin, extend more
            if depth >= 8 and exclusion_score < exclusion_beta - SINGULAR_EXTENSION_MARGIN:
                extension = 2

    if ENABLE_PROBCUT and depth >= 6 and tt_entry['flag'] != TT_FLAG_NONE:
        if tt_entry['depth'] >= depth - PROBCUT_R:
            tt_score = np.int32(tt_entry['score'])
            if tt_score + PROBCUT_MARGIN >= beta:
                res_pc = _search(
                    piece_bbs, occupancy_bbs, game_state, depth - PROBCUT_R_PRIME,
                    beta - 1, beta, search_context, ply + 1, NO_MOVE, False
                )
                probcut_score = res_pc[0]

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
                res_pc = _search(
                    piece_bbs, occupancy_bbs, game_state, depth - PROBCUT_R_PRIME,
                    alpha, beta, search_context, ply + 1, NO_MOVE, False
                )
                probcut_score = res_pc[0]

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
        search_context.pv_table[ply, ply] = NO_MOVE
        eval_score, q_nodes, child_delta_pruned, child_see_pruned = quiescence_search(piece_bbs, occupancy_bbs, game_state, alpha, beta, ply, search_context, 0)
        qs_delta_pruned += child_delta_pruned
        qs_see_pruned += child_see_pruned
        return (eval_score, NO_MOVE, nodes_searched, q_nodes, cutoffs, tt_hits,
                null_move_cutoffs, futility_pruned, razoring_used, rfp_pruned, lmp_pruned, probcut_pruned, qs_delta_pruned, qs_see_pruned,
                iid_searches, singular_extensions)

    # --- Static Evaluation & Improving Flag ---
    static_score = -INFINITY
    improving = False
    
    # Correction History Adjustment
    raw_static_eval = -INFINITY

    if not is_currently_in_check:
        raw_static_eval = evaluate_position(piece_bbs, occupancy_bbs, game_state, lazy=True)
        
        # Apply Correction History
        pawn_key = game_state[PAWN_KEY_INDEX]
        correction = search_context.pawn_correction_history[pawn_key % CORRECTION_HISTORY_SIZE]
        
        # Clamp correction to limit its effect
        correction = min(max(correction, -CORRECTION_HISTORY_LIMIT), CORRECTION_HISTORY_LIMIT)
        
        static_score = raw_static_eval + correction
        
        search_context.static_eval_stack[ply] = static_score

        if ply >= 2 and static_score > search_context.static_eval_stack[ply - 2]:
            improving = True
    else:
        search_context.static_eval_stack[ply] = -INFINITY

    # --- Null Move Pruning ---
    # Update: Dynamic Reduction and Safety Check
    if ENABLE_NMP and depth >= 3 and not is_currently_in_check and has_sufficient_material(piece_bbs, game_state[0]) and static_score >= beta:
        original_state_for_null = game_state.copy()
        make_null_move(game_state)
        
        # NOTE: Passing a large value or special flag for previous move might be needed if NMP impacts move ordering logic of sub-search. 
        # For now we pass NO_MOVE as we don't have a "previous move" for null move.
        res_nm = _search(
            piece_bbs, occupancy_bbs, game_state, depth - 1 - NULL_MOVE_REDUCTION,
            -beta, -beta + 1, search_context, ply + 1, NO_MOVE, False
        )
        null_move_score = res_nm[0]
        child_nodes = res_nm[2]
        child_q_nodes = res_nm[3]
        child_cutoffs = res_nm[4]
        child_tt_hits = res_nm[5]
        child_nmc = res_nm[6]
        child_fp = res_nm[7]
        child_ru = res_nm[8]
        child_rfp = res_nm[9]
        child_lmp = res_nm[10]
        child_pcp = res_nm[11]
        child_qdp = res_nm[12]
        child_qsp = res_nm[13]
        child_iid = res_nm[14]
        child_se = res_nm[15]

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
            search_context.pv_table[ply, ply] = NO_MOVE
            return (np.int32(beta), NO_MOVE, nodes_searched, quiescence_nodes, cutoffs, tt_hits,
                    null_move_cutoffs, futility_pruned, razoring_used, rfp_pruned, lmp_pruned, probcut_pruned, qs_delta_pruned, qs_see_pruned,
                    iid_searches, singular_extensions)

    if depth <= 2 and not is_currently_in_check:
        if ENABLE_RFP and depth == 1 and static_score - RFP_MARGIN_D1 >= beta:
            rfp_pruned += 1
            return (np.int32(beta), NO_MOVE, nodes_searched, quiescence_nodes, cutoffs, tt_hits,
                    null_move_cutoffs, futility_pruned, razoring_used, rfp_pruned, lmp_pruned, probcut_pruned, qs_delta_pruned, qs_see_pruned,
                    iid_searches, singular_extensions)

        if ENABLE_RAZORING and static_score + RAZORING_MARGIN < alpha:
            razor_score, child_q_nodes, child_delta_pruned, child_see_pruned = quiescence_search(piece_bbs, occupancy_bbs, game_state, alpha, alpha + 1, ply, search_context, 0)
            quiescence_nodes += child_q_nodes
            qs_delta_pruned += child_delta_pruned
            qs_see_pruned += child_see_pruned
            if razor_score + RAZORING_MARGIN < alpha:
                razoring_used += 1
                return (np.int32(alpha), NO_MOVE, nodes_searched, quiescence_nodes, cutoffs, tt_hits,
                        null_move_cutoffs, futility_pruned, razoring_used, rfp_pruned, lmp_pruned, probcut_pruned, qs_delta_pruned, qs_see_pruned,
                        iid_searches, singular_extensions)

    if not moves_generated:
        move_count = generate_legal_moves_buffer(piece_bbs, occupancy_bbs, game_state, search_context.moves_buffer, ply)

    if move_count == 0:
        search_context.pv_table[ply, ply] = NO_MOVE
        if is_currently_in_check:
            return (np.int32(-MATE_SCORE + ply), NO_MOVE, nodes_searched, quiescence_nodes, cutoffs, tt_hits, null_move_cutoffs, futility_pruned, razoring_used, rfp_pruned, lmp_pruned, probcut_pruned, qs_delta_pruned, qs_see_pruned, iid_searches, singular_extensions)
        else:
            return (np.int32(0), NO_MOVE, nodes_searched, quiescence_nodes, cutoffs, tt_hits, null_move_cutoffs, futility_pruned, razoring_used, rfp_pruned, lmp_pruned, probcut_pruned, qs_delta_pruned, qs_see_pruned, iid_searches, singular_extensions)

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

    # Optimization: Calculate pinned pieces once for shallow pruning logic and score_moves
    pinned_white = get_pinned_pieces(piece_bbs, occupancy_bbs, WHITE)
    pinned_black = get_pinned_pieces(piece_bbs, occupancy_bbs, BLACK)

    score_moves(
        piece_bbs, occupancy_bbs, game_state, 
        moves, 
        scores, 
        move_count,
        tt_move, 
        search_context.killer_moves[safe_ply*2:safe_ply*2+2], 
        search_context.history_table, 
        counter_move,
        pinned_white,
        pinned_black,
        search_context,
        ply
    )
    
    best_move, max_eval = NO_MOVE, -INFINITY
    quiet_move_counter, searched_move_count = 0, 0
    
    # Track tried quiet moves for history malus
    quiet_moves_tried = search_context.quiet_moves_tried[ply]
    quiet_moves_tried_count = 0

    for i in range(move_count):
        # Lazy Selection Sort
        best_idx = i
        for j in range(i + 1, move_count):
            if scores[j] > scores[best_idx]:
                best_idx = j
        
        # Swap moves and scores
        moves[i], moves[best_idx] = moves[best_idx], moves[i]
        scores[i], scores[best_idx] = scores[best_idx], scores[i]

        move = moves[i]
        if move == excluded_move: continue
        searched_move_count += 1
        
        opponent_pieces_bb = occupancy_bbs[1] if game_state[0] == 0 else occupancy_bbs[0]
        
        # --- Pre-move checks for extensions and move type ---
        from_sq = get_from_square(move)
        to_sq = get_to_square(move)
        is_capture = (opponent_pieces_bb & BB_SQUARES[to_sq]) != 0
        is_promotion = get_special_move_flag(move) == SPECIAL_MOVE_FLAG_PROMOTION
        is_pseudo_quiet = not is_capture and not is_promotion
        
        # --- NEW: Shallow Depth Pruning (Stockfish Step 14) ---
        if ENABLE_SHALLOW_SEE_PRUNING and depth <= PRUNING_SHALLOW_DEPTH and not is_currently_in_check:
             # Can't use is_giving_check_after_move here as it requires make_move.
             side_to_move = game_state[0]

             if is_capture:
                 # Capture Pruning
                 # Threshold: -200 * depth
                 threshold = PRUNING_CAPTURE_SEE_MARGIN * depth
                 if not see_ge(piece_bbs, occupancy_bbs, side_to_move, from_sq, to_sq, threshold, pinned_white, pinned_black):
                     search_context.see_pruned_captures += 1
                     continue
             elif is_pseudo_quiet:
                 # Quiet Move Pruning

                 # History Pruning
                 if ENABLE_HISTORY_PRUNING:
                     aggressor_type = find_piece_type_on_square_side(piece_bbs, from_sq, side_to_move)
                     history_score = search_context.history_table[aggressor_type, to_sq]
                     if history_score < PRUNING_HISTORY_THRESHOLD:
                         search_context.history_pruned += 1
                         continue

                 # Quiet SEE Pruning (e.g. moving into attack)
                 # Threshold: -100 * depth * depth
                 threshold = PRUNING_QUIET_SEE_MARGIN * depth * depth
                 if not see_ge(piece_bbs, occupancy_bbs, side_to_move, from_sq, to_sq, threshold, pinned_white, pinned_black):
                     search_context.see_pruned_quiets += 1
                     continue
                 
        # --- Make the move ---
        unmake_info = make_move(piece_bbs, occupancy_bbs, game_state, move)
        search_context.move_stack[ply] = move # Record move in stack
        
        # --- Post-move checks ---
        is_giving_check_after_move = is_in_check(piece_bbs, occupancy_bbs, game_state)
        
        # Final determination of quiet move
        is_quiet_move = is_pseudo_quiet and not is_giving_check_after_move

        if is_quiet_move:
            quiet_move_counter += 1
            # Add to tried list for potential malus
            quiet_moves_tried[quiet_moves_tried_count] = move
            quiet_moves_tried_count += 1
            
            if ENABLE_LMP and not is_currently_in_check:
                limit = LMP_MOVE_COUNT[depth]
                if not improving: limit = limit // 2
                limit = max(limit, 2)

                if quiet_move_counter >= limit:
                    lmp_pruned += 1
                    unmake_move(piece_bbs, occupancy_bbs, game_state, move, unmake_info)
                    break

        if ENABLE_FP and is_quiet_move and not is_currently_in_check and static_score != -INFINITY:
            # Dynamic Futility Margin: FP_BASE + FP_MULTIPLIER * depth
            margin = FP_BASE + FP_MULTIPLIER * depth

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
        if searched_move_count == 1:
            res = _search(
                piece_bbs, occupancy_bbs, game_state, search_depth, -beta, -alpha, search_context, ply + 1, NO_MOVE, is_pv)
            evaluation = -res[0]
            child_nodes = res[2]; child_q_nodes = res[3]; child_cutoffs = res[4]; child_tt_hits = res[5]
            child_nmc = res[6]; child_fp = res[7]; child_ru = res[8]; child_rfp = res[9]; child_lmp = res[10]
            child_pcp = res[11]; child_qdp = res[12]; child_qsp = res[13]; child_iid = res[14]; child_se = res[15]
        else:
            # Dynamic LMR Logic
            lmr = 0
            if ENABLE_LMR and depth >= LMR_MIN_DEPTH and is_quiet_move and quiet_move_counter >= LMR_MIN_QUIET_MOVE_INDEX:
                # Get History Score for adjustment
                aggressor_type = find_piece_type_on_square_side(piece_bbs, from_sq, game_state[0])
                history_score = search_context.history_table[aggressor_type, to_sq]

                lmr = get_lmr_reduction(depth, searched_move_count, history_score, improving, is_pv)

                # Reduce LMR for moves that improve King Tropism (Logic kept from previous version)
                side_to_move = game_state[0]
                opponent_king_bb = piece_bbs[11] if side_to_move == 0 else piece_bbs[5]
                if opponent_king_bb != 0:
                    opponent_king_sq = get_lsb_index(opponent_king_bb)
                    k_file = opponent_king_sq % 8
                    k_rank = opponent_king_sq // 8
                    
                    from_file = from_sq % 8
                    from_rank = from_sq // 8
                    
                    to_file = to_sq % 8
                    to_rank = to_sq // 8
                    
                    dist_before = abs(from_file - k_file) + abs(from_rank - k_rank)
                    dist_after = abs(to_file - k_file) + abs(to_rank - k_rank)
                    
                    if dist_after < dist_before:
                        lmr = max(0, lmr - 1)

            res = _search(
                piece_bbs, occupancy_bbs, game_state, search_depth - lmr, -alpha - 1, -alpha, search_context, ply + 1, NO_MOVE, False)
            evaluation = -res[0]
            child_nodes = res[2]; child_q_nodes = res[3]; child_cutoffs = res[4]; child_tt_hits = res[5]
            child_nmc = res[6]; child_fp = res[7]; child_ru = res[8]; child_rfp = res[9]; child_lmp = res[10]
            child_pcp = res[11]; child_qdp = res[12]; child_qsp = res[13]; child_iid = res[14]; child_se = res[15]

            if evaluation > alpha:
                # Re-search with full window
                res = _search(
                    piece_bbs, occupancy_bbs, game_state, search_depth, -beta, -alpha, search_context, ply + 1, NO_MOVE, is_pv)
                evaluation = -res[0]
                child_nodes += res[2]; child_q_nodes += res[3]; child_cutoffs += res[4]; child_tt_hits += res[5]
                child_nmc += res[6]; child_fp += res[7]; child_ru += res[8]; child_rfp += res[9]; child_lmp += res[10]
                child_pcp += res[11]; child_qdp += res[12]; child_qsp += res[13]; child_iid += res[14]; child_se += res[15]

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
            
            i = ply + 1
            if ply + 1 < MAX_PLY:
                j = ply + 1
                while j < MAX_PLY and search_context.pv_table[ply + 1, j] != NO_MOVE:
                    search_context.pv_table[ply, i] = search_context.pv_table[ply + 1, j]
                    i += 1; j += 1
            
            if i < MAX_PLY: search_context.pv_table[ply, i] = NO_MOVE

        alpha = max(alpha, evaluation)
        if alpha >= beta:
            cutoffs += 1
            if is_capture:
                # Update Capture History
                victim_type = find_piece_type_on_square_side(piece_bbs, to_sq, 1 - game_state[0])
                if victim_type != -1:
                    aggressor_type = find_piece_type_on_square_side(piece_bbs, from_sq, game_state[0])
                    bonus = depth * depth
                    update_capture_history(search_context.capture_history, aggressor_type, to_sq, victim_type, bonus)
            
            if is_quiet_move:
                bonus = depth * depth
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

    final_flag = TT_FLAG_ALPHA if max_eval <= original_alpha else (TT_FLAG_BETA if max_eval >= beta else TT_FLAG_EXACT)
    if max_eval == -INFINITY:
        max_eval = original_alpha
    if best_move == NO_MOVE and move_count > 0:
        # Fallback to first move that is not excluded
        for i in range(move_count):
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
    store_tt(search_context.transposition_table, zobrist_key, depth, tt_score, final_flag, best_move, search_context.tt_generation)

    return (max_eval, best_move, nodes_searched, quiescence_nodes, cutoffs, tt_hits,
            null_move_cutoffs, futility_pruned, razoring_used, rfp_pruned, lmp_pruned, probcut_pruned, qs_delta_pruned, qs_see_pruned,
            iid_searches, singular_extensions)

def iterative_deepening_search(piece_bbs, occupancy_bbs, game_state, max_depth, time_config, search_context, game_history_list=None, tt_generation=0):
    start_time = time.time()
    
    # --- Decay History Tables ---
    # Divide history values by 2 to prioritize recent successful moves and reduce impact of old history
    search_context.history_table[:] = search_context.history_table[:] // 2
    search_context.butterfly_history[:] = search_context.butterfly_history[:] // 2
    search_context.capture_history[:] = search_context.capture_history[:] // 2

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

    # Clear Counter Moves at start of search? Stockfish doesn't seem to reset them per search, 
    # but usually they are part of thread data. We can keep them or clear them.
    # Clearing them ensures no pollution from previous moves in different game contexts if not handled by generations.
    # However, for the same game, it might be useful. 
    # Let's clear them to be safe and consistent with "new search".
    search_context.counter_moves.fill(0)
    search_context.move_stack.fill(NO_MOVE)

    for current_depth in range(1, max_depth + 1):
        alpha, beta = (-INFINITY, INFINITY) if current_depth <= 1 else (last_score - ASPIRATION_WINDOW_SIZE, last_score + ASPIRATION_WINDOW_SIZE)

        search_context.nodes_searched = np.uint64(0)

        res = _search(piece_bbs, occupancy_bbs, game_state, current_depth, alpha, beta, search_context, 0, NO_MOVE, True)
        score = res[0]

        if not search_context.stop_flag[0] and (score <= alpha or score >= beta):
            log_info(f"depth {current_depth} aspiration window failed, re-searching...")
            
            # Accumulate stats from the failed attempt
            total_q_nodes += res[3]; total_cutoffs += res[4]; total_tt_hits += res[5]; total_nmc += res[6]; total_fp += res[7]; total_ru += res[8]; total_rfp += res[9]; total_lmp += res[10]; total_pcp += res[11]; total_qdp += res[12]; total_qsp += res[13]; total_iid += res[14]; total_se += res[15]
            
            alpha, beta = -INFINITY, INFINITY
            # Note: We do NOT reset search_context.nodes_searched here, let it accumulate for this depth.
            res = _search(piece_bbs, occupancy_bbs, game_state, current_depth, alpha, beta, search_context, 0, NO_MOVE, True)
            score = res[0]

        # Always accumulate stats from the search (even if stopped)
        total_nodes += search_context.nodes_searched
        total_q_nodes += res[3]; total_cutoffs += res[4]; total_tt_hits += res[5]; total_nmc += res[6]; total_fp += res[7]; total_ru += res[8]; total_rfp += res[9]; total_lmp += res[10]; total_pcp += res[11]; total_qdp += res[12]; total_qsp += res[13]; total_iid += res[14]; total_se += res[15]

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
