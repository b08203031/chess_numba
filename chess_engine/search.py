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

# Import TT components
from chess_engine.transposition_table import (
    probe_tt, store_tt, numba_tt_entry_type,
    TT_FLAG_NONE, TT_FLAG_EXACT, TT_FLAG_ALPHA, TT_FLAG_BETA
)

from chess_engine.engine_types import (
    piece_bbs_signature, occupancy_bbs_signature, game_state_signature,
    SearchContext, search_context_type
)

move_picker_spec = [
    ('moves', numba.uint16[::1]),
    ('scores', numba.int32[::1]),
    ('see_values', numba.int32[::1]),
    ('index', numba.int32),
    ('stage', numba.int32),
    ('tt_move', numba.uint16),
    ('killer_moves', numba.uint16[::1]),
    ('history_table', numba.int32[:, :]),
]

# MovePicker Stages
STAGE_TT = 0
STAGE_GOOD_CAPTURES = 1
STAGE_KILLERS = 2
STAGE_QUIETS = 3
STAGE_BAD_CAPTURES = 4
STAGE_DONE = 5

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

@numba.experimental.jitclass(move_picker_spec)
class MovePicker:
    def __init__(self, piece_bbs, occupancy_bbs, game_state, moves, tt_move, killer_moves_at_ply, history_table):
        self.moves = moves.copy()
        self.scores = np.zeros(len(moves), dtype=np.int32)
        self.index = 0

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
            self.scores[i] = score
        
    def next_move(self):
        if self.index >= len(self.moves):
            return np.uint16(NO_MOVE)

        best_score = -INFINITY
        best_idx = -1
        for i in range(self.index, len(self.moves)):
            if self.scores[i] > best_score:
                best_score = self.scores[i]
                best_idx = i
        
        # Swap the best move with the current move
        self.moves[self.index], self.moves[best_idx] = self.moves[best_idx], self.moves[self.index]
        self.scores[self.index], self.scores[best_idx] = self.scores[best_idx], self.scores[self.index]
        
        move = self.moves[self.index]
        self.index += 1
        return move


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
    
    legal_moves = generate_legal_moves(piece_bbs, occupancy_bbs, game_state)
    
    capture_moves = []
    opponent_pieces_bb = occupancy_bbs[1] if game_state[0] == 0 else occupancy_bbs[0]
    for move in legal_moves:
        to_sq = get_to_square(move)
        is_capture = (opponent_pieces_bb & BB_SQUARES[to_sq]) != 0
        is_en_passant = get_special_move_flag(move) == SPECIAL_MOVE_FLAG_EN_PASSANT

        if is_capture or is_en_passant:
            capture_moves.append(move)

    if not capture_moves:
        return stand_pat, q_nodes, delta_pruned, see_pruned
    
    for move in capture_moves:
        use_premade_board = False

        # --- Delta Pruning ---
        if ENABLE_DELTA_PRUNING:
            is_promotion = get_special_move_flag(move) == SPECIAL_MOVE_FLAG_PROMOTION
            
            # A better approach would be to get the promotion piece value, but for now, we assume Queen
            promotion_gain = MG_MATERIAL_VALUES[4] - MG_MATERIAL_VALUES[0] if is_promotion else 0
            
            # Find victim piece value
            victim_type = find_piece_type_on_square(piece_bbs, get_to_square(move))
            victim_value = 0
            if victim_type != -1:
                victim_value = MG_MATERIAL_VALUES[victim_type % 6]
            
            potential_gain = victim_value + promotion_gain
            
            if stand_pat + potential_gain + DELTA_PRUNING_MARGIN < alpha:
                # Before pruning, we must ensure the move does not give check.
                new_piece_bbs_temp, new_occupancy_bbs_temp, new_game_state_temp, _ = make_move(
                    piece_bbs, occupancy_bbs, game_state, move
                )
                if not is_in_check(new_piece_bbs_temp, new_occupancy_bbs_temp, new_game_state_temp):
                    delta_pruned += 1
                    continue  # 剪枝成功，跳過後續
                
                # 如果給將軍，這次結果可直接用，設定旗標
                use_premade_board = True
                new_piece_bbs, new_occupancy_bbs, new_game_state = new_piece_bbs_temp, new_occupancy_bbs_temp, new_game_state_temp

        if ENABLE_SEE_IN_QUIESCENCE and not use_premade_board:
            if see(piece_bbs, occupancy_bbs, game_state, get_from_square(move), get_to_square(move)) < SEE_THRESHOLD:
                see_pruned += 1
                continue
            
        if not use_premade_board:
            new_piece_bbs, new_occupancy_bbs, new_game_state, _ = make_move(
                piece_bbs, occupancy_bbs, game_state, move
            )
        score, child_q_nodes, child_delta_pruned, child_see_pruned = quiescence_search(
            new_piece_bbs, new_occupancy_bbs, new_game_state, -beta, -alpha, ply + 1
        )
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
    qs_delta_pruned = np.uint64(0) # Re-purposing this for the new delta pruning
    qs_see_pruned = np.uint64(0)
    iid_searches = np.uint64(0)
    singular_extensions = np.uint64(0)

    if ply >= MAX_PLY:
        for j in range(MAX_PLY):
            search_context.pv_table[ply, j] = np.uint16(NO_MOVE)
        return (evaluate_position(piece_bbs, occupancy_bbs, game_state), np.uint16(NO_MOVE), nodes_searched, quiescence_nodes, cutoffs, tt_hits,
                null_move_cutoffs, futility_pruned, razoring_used, rfp_pruned, lmp_pruned, probcut_pruned, qs_delta_pruned, qs_see_pruned,
                iid_searches, singular_extensions)

    original_alpha = alpha
    zobrist_key = game_state[4]

    tt_move = np.uint16(NO_MOVE)
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
            for j in range(MAX_PLY):
                search_context.pv_table[ply, j] = np.uint16(NO_MOVE)
            return (tt_score, tt_entry['best_move'], nodes_searched, quiescence_nodes, cutoffs, tt_hits,
                    null_move_cutoffs, futility_pruned, razoring_used, rfp_pruned, lmp_pruned, probcut_pruned, qs_delta_pruned, qs_see_pruned,
                    iid_searches, singular_extensions)
        elif tt_entry['flag'] == TT_FLAG_ALPHA:
            beta = min(beta, tt_score)
        elif tt_entry['flag'] == TT_FLAG_BETA:
            alpha = max(alpha, tt_score)

        if alpha >= beta:
            for j in range(MAX_PLY):
                search_context.pv_table[ply, j] = np.uint16(NO_MOVE)
            return (tt_score, tt_entry['best_move'], nodes_searched, quiescence_nodes, cutoffs, tt_hits,
                    null_move_cutoffs, futility_pruned, razoring_used, rfp_pruned, lmp_pruned, probcut_pruned, qs_delta_pruned, qs_see_pruned,
                    iid_searches, singular_extensions)

        tt_move = tt_entry['best_move']

    # --- Internal Iterative Deepening (IID) ---
    if ENABLE_IID and depth >= 5 and tt_move == NO_MOVE:
        iid_searches += 1
        iid_depth = max(2, depth - 3)
        # Note: This recursive call's return values are ignored. Its only purpose
        # is to populate the TT with a good move to improve move ordering.
        _search(
            piece_bbs, occupancy_bbs, game_state, iid_depth, -INFINITY, INFINITY,
            search_context, ply + 1, NO_MOVE
        )
        # Re-probe the TT to get the move found by IID
        tt_entry = probe_tt(search_context.transposition_table, zobrist_key)
        if tt_entry['flag'] != TT_FLAG_NONE:
            tt_move = tt_entry['best_move']


    # --- ProbCut ---
    # --- Singular Extensions ---
    extension = 0
    if ENABLE_SINGULAR_EXTENSIONS and depth >= MIN_SINGULAR_DEPTH and tt_move != NO_MOVE and tt_entry['flag'] == TT_FLAG_EXACT:
        # 1. Get the score of the best move from TT
        tt_score = np.int32(tt_entry['score'])

        # 2. Perform an exclusion search
        exclusion_beta = tt_score - SINGULAR_EXTENSION_MARGIN
        exclusion_score, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = _search(
            piece_bbs, occupancy_bbs, game_state, depth - 3,
            exclusion_beta - 1, exclusion_beta, search_context, ply + 1, tt_move
        )

        # 3. If the exclusion search fails low, it means the tt_move is singular
        if exclusion_score < exclusion_beta:
            extension = 1
            singular_extensions += 1


    # --- ProbCut ---
    if ENABLE_PROBCUT and depth >= 6 and tt_entry['flag'] != TT_FLAG_NONE:
        # Check if the TT entry is from a shallower search
        if tt_entry['depth'] >= depth - PROBCUT_R:
            tt_score = np.int32(tt_entry['score'])
            
            # Fail-high case
            if tt_score + PROBCUT_MARGIN >= beta:
                # Use a zero-window search to verify the fail-high
                probcut_score, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = _search(
                    piece_bbs, occupancy_bbs, game_state, depth - PROBCUT_R_PRIME,
                    beta - 1, beta, search_context, ply + 1, NO_MOVE  # This is effectively a zero-window search at the beta boundary
                )
                if probcut_score >= beta:
                    probcut_pruned += 1
                    return (np.int32(beta), np.uint16(NO_MOVE), nodes_searched, quiescence_nodes, cutoffs, tt_hits,
                           null_move_cutoffs, futility_pruned, razoring_used, rfp_pruned, lmp_pruned, probcut_pruned, qs_delta_pruned, qs_see_pruned,
                           iid_searches, singular_extensions)

            # Fail-low case
            elif tt_score - PROBCUT_MARGIN <= alpha:
                # Use a full-window search to verify the fail-low
                probcut_score, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = _search(
                    piece_bbs, occupancy_bbs, game_state, depth - PROBCUT_R_PRIME,
                    alpha, beta, search_context, ply + 1, NO_MOVE # As per user request, use full window for fail-low
                )
                if probcut_score <= alpha:
                    probcut_pruned += 1
                    return (np.int32(alpha), np.uint16(NO_MOVE), nodes_searched, quiescence_nodes, cutoffs, tt_hits,
                           null_move_cutoffs, futility_pruned, razoring_used, rfp_pruned, lmp_pruned, probcut_pruned, qs_delta_pruned, qs_see_pruned,
                           iid_searches, singular_extensions)

    is_currently_in_check = is_in_check(piece_bbs, occupancy_bbs, game_state)
    
    # Generate legal moves once
    moves = generate_legal_moves(piece_bbs, occupancy_bbs, game_state)

    # --- Shallow Pruning Techniques (RFP, Razoring) ---
    static_score = -INFINITY  # Use a sentinel value
    if depth <= 2 and not is_currently_in_check:
        static_score = evaluate_position(piece_bbs, occupancy_bbs, game_state)
        
        # Reverse Futility Pruning (RFP)
        if ENABLE_RFP and depth == 1 and static_score - RFP_MARGIN_D1 >= beta:
            rfp_pruned += 1
            return (np.int32(beta), np.uint16(NO_MOVE), nodes_searched, quiescence_nodes, cutoffs, tt_hits,
                    null_move_cutoffs, futility_pruned, razoring_used, rfp_pruned, lmp_pruned, probcut_pruned, qs_delta_pruned, qs_see_pruned,
                    iid_searches, singular_extensions)
        
        # Razoring
        if ENABLE_RAZORING and depth <= 2 and static_score + RAZORING_MARGIN < alpha:
            razor_score, child_q_nodes, child_delta_pruned, child_see_pruned = quiescence_search(piece_bbs, occupancy_bbs, game_state, alpha, alpha + 1, 0)
            quiescence_nodes += child_q_nodes
            qs_delta_pruned += child_delta_pruned
            qs_see_pruned += child_see_pruned
            if razor_score + RAZORING_MARGIN < alpha:
                razoring_used += 1
                return (np.int32(alpha), np.uint16(NO_MOVE), nodes_searched, quiescence_nodes, cutoffs, tt_hits,
                        null_move_cutoffs, futility_pruned, razoring_used, rfp_pruned, lmp_pruned, probcut_pruned, qs_delta_pruned, qs_see_pruned,
                        iid_searches, singular_extensions)


    # --- Null Move Pruning (NMP) ---
    if ENABLE_NMP and depth >= 3 and not is_currently_in_check and has_sufficient_material(piece_bbs, game_state[0]):
        null_move_piece_bbs, null_move_occupancy_bbs, null_move_game_state = make_null_move(piece_bbs, occupancy_bbs, game_state)
        
        null_move_score, _, _, _, _, child_tt_hits, _, _, _, _, _, _, _, _, _, _ = _search(
            null_move_piece_bbs, null_move_occupancy_bbs, null_move_game_state,
            depth - 1 - NULL_MOVE_REDUCTION,
            -beta, -beta + 1, search_context, ply + 1, NO_MOVE
        )
        null_move_score = -null_move_score
        tt_hits += child_tt_hits

        if null_move_score >= beta:
            null_move_cutoffs += 1
            for j in range(MAX_PLY):
                search_context.pv_table[ply, j] = np.uint16(NO_MOVE)
            return (np.int32(beta), np.uint16(NO_MOVE), nodes_searched, quiescence_nodes, cutoffs, tt_hits,
                    null_move_cutoffs, futility_pruned, razoring_used, rfp_pruned, lmp_pruned, probcut_pruned, qs_delta_pruned, qs_see_pruned,
                    iid_searches, singular_extensions)

    has_legal_moves = False
    for m in moves:
        if m != 0:
            has_legal_moves = True
            break

    if not has_legal_moves:
        for j in range(MAX_PLY):
            search_context.pv_table[ply, j] = np.uint16(NO_MOVE)
        if is_currently_in_check:
            # Checkmate
            return (np.int32(-MATE_SCORE + ply), np.uint16(NO_MOVE), nodes_searched, quiescence_nodes, cutoffs, tt_hits,
                    null_move_cutoffs, futility_pruned, razoring_used, rfp_pruned, lmp_pruned, probcut_pruned, qs_delta_pruned, qs_see_pruned,
                    iid_searches, singular_extensions)
        else:
            # Stalemate
            return (np.int32(0), np.uint16(NO_MOVE), nodes_searched, quiescence_nodes, cutoffs, tt_hits,
                    null_move_cutoffs, futility_pruned, razoring_used, rfp_pruned, lmp_pruned, probcut_pruned, qs_delta_pruned, qs_see_pruned,
                    iid_searches, singular_extensions)

    if depth == 0:
        for j in range(MAX_PLY):
            search_context.pv_table[ply, j] = np.uint16(NO_MOVE)
        
        eval_score, q_nodes, child_delta_pruned, child_see_pruned = quiescence_search(piece_bbs, occupancy_bbs, game_state, alpha, beta, 0)
        qs_delta_pruned += child_delta_pruned
        qs_see_pruned += child_see_pruned
        return (eval_score, np.uint16(NO_MOVE), nodes_searched, q_nodes, cutoffs, tt_hits,
                null_move_cutoffs, futility_pruned, razoring_used, rfp_pruned, lmp_pruned, probcut_pruned, qs_delta_pruned, qs_see_pruned,
                iid_searches, singular_extensions)

    move_picker = MovePicker(piece_bbs, occupancy_bbs, game_state, moves, tt_move, search_context.killer_moves[ply*2:ply*2+2], search_context.history_table)
    
    best_move = np.uint16(NO_MOVE)
    max_eval = -INFINITY
    quiet_move_counter = 0
    move_count = 0

    # Ensure static_score is computed if not already done for shallow depths
    if static_score == -INFINITY and depth <= 2 and not is_currently_in_check:
        static_score = evaluate_position(piece_bbs, occupancy_bbs, game_state)

    while True:
        move = move_picker.next_move()
        if move == np.uint16(NO_MOVE):
            break
        
        if move == excluded_move:
            continue
        
        move_count += 1
        
        new_piece_bbs, new_occupancy_bbs, new_game_state, _ = make_move(
            piece_bbs, occupancy_bbs, game_state, move
        )
        
        is_giving_check_after_move = is_in_check(new_piece_bbs, new_occupancy_bbs, new_game_state)
        opponent_pieces_bb = occupancy_bbs[1] if game_state[0] == 0 else occupancy_bbs[0]
        is_capture = (opponent_pieces_bb & BB_SQUARES[get_to_square(move)]) != 0
        is_promotion = get_special_move_flag(move) == SPECIAL_MOVE_FLAG_PROMOTION
        is_quiet_move = not is_capture and not is_promotion and not is_giving_check_after_move
        
        # --- Late Move Pruning (LMP) ---
        if ENABLE_LMP and is_quiet_move and not is_currently_in_check:
            if quiet_move_counter >= LMP_MOVE_COUNT[depth]:
                lmp_pruned += 1
                break # Stop searching quiet moves at this node

        # --- Futility Pruning (F-Pruning) ---
        if ENABLE_FP and is_quiet_move and not is_currently_in_check:
            margin = 0
            if depth == 1: margin = FP_MARGIN_D1
            elif depth == 2: margin = FP_MARGIN_D2

            if margin > 0 and static_score + margin < alpha:
                futility_pruned += 1
                continue # Skip this quiet move
        
        # --- Check Extensions ---
        current_extension = 0
        if is_giving_check_after_move:
            current_extension = 1
        
        # --- Singular Extension ---
        if move == tt_move:
            current_extension = max(current_extension, extension)

        search_depth = depth - 1 + current_extension

        evaluation = 0

        # --- Principal Variation Search (PVS) & Late Move Reductions (LMR) ---
        if move_count == 1:  # First move (PV node): Full window search
            evaluation, _, child_nodes, child_q_nodes, child_cutoffs, child_tt_hits, child_nmc, child_fp, child_ru, child_rfp, child_lmp, child_pcp, child_qdp, child_qsp, child_iid, child_se = _search(
                new_piece_bbs, new_occupancy_bbs, new_game_state, search_depth, -beta, -alpha, search_context, ply + 1, NO_MOVE
            )
            nodes_searched += child_nodes; quiescence_nodes += child_q_nodes; cutoffs += child_cutoffs; tt_hits += child_tt_hits
            null_move_cutoffs += child_nmc; futility_pruned += child_fp; razoring_used += child_ru; rfp_pruned += child_rfp
            lmp_pruned += child_lmp; probcut_pruned += child_pcp; qs_delta_pruned += child_qdp; qs_see_pruned += child_qsp
            iid_searches += child_iid; singular_extensions += child_se
            evaluation = -evaluation
        else:  # Subsequent moves (Non-PV nodes): Zero-window search
            lmr_reduction = 0
            if ENABLE_LMR and depth >= LMR_MIN_DEPTH and is_quiet_move and quiet_move_counter >= LMR_MIN_QUIET_MOVE_INDEX:
                lmr_reduction = LMR_REDUCTION
            
            reduced_search_depth = search_depth - lmr_reduction

            # 1. Zero-window search with potential LMR
            evaluation, _, child_nodes, child_q_nodes, child_cutoffs, child_tt_hits, child_nmc, child_fp, child_ru, child_rfp, child_lmp, child_pcp, child_qdp, child_qsp, child_iid, child_se = _search(
                new_piece_bbs, new_occupancy_bbs, new_game_state, reduced_search_depth, -alpha - 1, -alpha, search_context, ply + 1, NO_MOVE
            )
            nodes_searched += child_nodes; quiescence_nodes += child_q_nodes; cutoffs += child_cutoffs; tt_hits += child_tt_hits
            null_move_cutoffs += child_nmc; futility_pruned += child_fp; razoring_used += child_ru; rfp_pruned += child_rfp
            lmp_pruned += child_lmp; probcut_pruned += child_pcp; qs_delta_pruned += child_qdp; qs_see_pruned += child_qsp
            iid_searches += child_iid; singular_extensions += child_se
            evaluation = -evaluation

            # 2. If it fails high, re-search with a full window
            if evaluation > alpha: # Corrected PVS re-search condition
                evaluation, _, child_nodes, child_q_nodes, child_cutoffs, child_tt_hits, child_nmc, child_fp, child_ru, child_rfp, child_lmp, child_pcp, child_qdp, child_qsp, child_iid, child_se = _search(
                    new_piece_bbs, new_occupancy_bbs, new_game_state, search_depth, -beta, -alpha, search_context, ply + 1, NO_MOVE
                )
                nodes_searched += child_nodes; quiescence_nodes += child_q_nodes; cutoffs += child_cutoffs; tt_hits += child_tt_hits
                null_move_cutoffs += child_nmc; futility_pruned += child_fp; razoring_used += child_ru; rfp_pruned += child_rfp
                lmp_pruned += child_lmp; probcut_pruned += child_pcp; qs_delta_pruned += child_qdp; qs_see_pruned += child_qsp
                iid_searches += child_iid; singular_extensions += child_se
                evaluation = -evaluation

        if evaluation > max_eval:
            max_eval = evaluation
            best_move = move

            # --- PV Tracking ---
            search_context.pv_table[ply, ply] = move
            
            # Copy PV from child node
            i = ply + 1
            j = ply + 1
            while j < MAX_PLY and search_context.pv_table[ply + 1, j] != np.uint16(NO_MOVE):
                search_context.pv_table[ply, i] = search_context.pv_table[ply + 1, j]
                i += 1
                j += 1
            
            # Mark the end of the PV
            if i < MAX_PLY:
                search_context.pv_table[ply, i] = np.uint16(NO_MOVE)


        alpha = max(alpha, evaluation)
        if alpha >= beta:
            cutoffs += 1
            if is_quiet_move:
                bonus = depth * depth
                
                from_square = get_from_square(move)
                to_square = get_to_square(move)
                
                aggressor_type = find_piece_type_on_square(piece_bbs, from_square)
                
                search_context.history_table[aggressor_type, to_square] += bonus
                
                # Decay other moves to prevent unbounded growth
                # and prioritize recent history
                # search_context.history_table //= 2
                
                if move != search_context.killer_moves[ply * 2]:
                    search_context.killer_moves[ply * 2 + 1] = search_context.killer_moves[ply * 2]
                    search_context.killer_moves[ply * 2] = move
            break
        
        if is_quiet_move:
            quiet_move_counter += 1

    if max_eval <= original_alpha:
        final_flag = TT_FLAG_ALPHA
    elif max_eval >= beta:
        final_flag = TT_FLAG_BETA
    else:
        final_flag = TT_FLAG_EXACT
    
    if best_move == np.uint16(NO_MOVE) and len(moves) > 0:
        best_move = move_picker.moves[0]

    # Adjust mate score before storing
    tt_score = max_eval
    if tt_score > MATE_IN_MAX_PLY:
        tt_score -= ply
    elif tt_score < -MATE_IN_MAX_PLY:
        tt_score += ply

    store_tt(search_context.transposition_table, zobrist_key, depth, tt_score, final_flag, best_move)

    return (max_eval, best_move, nodes_searched, quiescence_nodes, cutoffs, tt_hits,
            null_move_cutoffs, futility_pruned, razoring_used, rfp_pruned, lmp_pruned, probcut_pruned, qs_delta_pruned, qs_see_pruned,
            iid_searches, singular_extensions)

@numba.njit(search_return_type(
    piece_bbs_signature, occupancy_bbs_signature, game_state_signature,
    numba.int32, numba.int32, numba.int32, numba.types.Array(numba_tt_entry_type, 1, 'C'),
    numba.types.Array(numba.uint16, 1, 'C'), numba.types.Array(numba.uint16, 2, 'C'),
    numba.types.Array(numba.int32, 2, 'C')
), cache=True)
def _search_wrapper(piece_bbs, occupancy_bbs, game_state, depth, alpha, beta, transposition_table, killer_moves, pv_table, history_table):
    search_context = SearchContext(transposition_table, killer_moves, pv_table, history_table)
    return _search(
        piece_bbs, occupancy_bbs, game_state, depth, alpha, beta, search_context, 0, NO_MOVE
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
    history_table = np.zeros((12, 64), dtype=np.int32)
    killer_moves = np.zeros(MAX_PLY * 2, dtype=np.uint16) # Changed to 1D array
    last_score = 0
    best_move_total = NO_MOVE
    total_nodes_searched, total_quiescence_nodes, total_cutoffs, total_tt_hits = np.uint64(0), np.uint64(0), np.uint64(0), np.uint64(0)
    total_nmc, total_fp, total_ru, total_rfp, total_lmp, total_pcp, total_qdp, total_qsp = np.uint64(0), np.uint64(0), np.uint64(0), np.uint64(0), np.uint64(0), np.uint64(0), np.uint64(0), np.uint64(0)
    total_iid, total_se = np.uint64(0), np.uint64(0)
    last_completed_depth = 0

    for current_depth in range(1, max_depth + 1):
        if current_depth > 1:
            alpha = last_score - ASPIRATION_WINDOW_SIZE
            beta = last_score + ASPIRATION_WINDOW_SIZE
        else:
            alpha, beta = -INFINITY, INFINITY

        pv_table.fill(0)
        score, _, nodes, q_nodes, cutoffs, tt_hits, nmc, fp, ru, rfp, lmp, pcp, qdp, qsp, iid, se = _search_wrapper(
            piece_bbs, occupancy_bbs, game_state, current_depth, alpha, beta, transposition_table, killer_moves, pv_table, history_table
        )

        if score <= alpha:
            log_info(f"depth {current_depth} aspiration window failed low, re-searching...")
            alpha, beta = -INFINITY, alpha + 1
            pv_table.fill(0)
            score, _, nodes, q_nodes, cutoffs, tt_hits, nmc, fp, ru, rfp, lmp, pcp, qdp, qsp, iid, se = _search_wrapper(
                piece_bbs, occupancy_bbs, game_state, current_depth, alpha, beta, transposition_table, killer_moves, pv_table, history_table
            )
        elif score >= beta:
            log_info(f"depth {current_depth} aspiration window failed high, re-searching...")
            alpha, beta = beta - 1, INFINITY
            pv_table.fill(0)
            score, _, nodes, q_nodes, cutoffs, tt_hits, nmc, fp, ru, rfp, lmp, pcp, qdp, qsp, iid, se = _search_wrapper(
                piece_bbs, occupancy_bbs, game_state, current_depth, alpha, beta, transposition_table, killer_moves, pv_table, history_table
            )

        last_score = score
        total_nodes_searched += nodes; total_quiescence_nodes += q_nodes; total_cutoffs += cutoffs; total_tt_hits += tt_hits
        total_nmc += nmc; total_fp += fp; total_ru += ru; total_rfp += rfp; total_lmp += lmp; total_pcp += pcp; total_qdp += qdp; total_qsp += qsp
        total_iid += iid; total_se += se

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
            last_completed_depth, total_nmc, total_fp, total_ru, total_rfp, total_lmp, total_pcp, total_qdp, total_qsp,
            total_iid, total_se)
