import numba
import numpy as np

from chess_engine.classical.move import (
    get_to_square, get_from_square, get_special_move_flag,
    SPECIAL_MOVE_FLAG_EN_PASSANT
)
from chess_engine.classical.constants import (
    SCORE_TT_MOVE, BB_SQUARES, NO_MOVE,
    HISTORY_MAX_MAIN, HISTORY_MAX_BUTTERFLY, HISTORY_MAX_CAPTURE, HISTORY_MAX_CONTINUATION,
    HISTORY_MAX_PAWN, HISTORY_WEIGHT_MAIN, HISTORY_WEIGHT_CONT_1,
    HISTORY_WEIGHT_CONT_2, HISTORY_WEIGHT_CONT_3, HISTORY_WEIGHT_CONT_4, HISTORY_WEIGHT_CONT_5,
    ORDERING_WEIGHT_MAIN, ORDERING_WEIGHT_PAWN, ORDERING_WEIGHT_BUTTERFLY,
    ORDERING_WEIGHT_CONT_1, ORDERING_WEIGHT_CONT_2, ORDERING_WEIGHT_CONT_3,
    ORDERING_WEIGHT_CONT_4, ORDERING_WEIGHT_CONT_5,
    REDUCTIONS, CHECK_BONUS,
    KNIGHT, BISHOP, ROOK, QUEEN, PAWN, MG_MATERIAL_VALUES,
    HISTORY_PAWN_BONUS_NUM, HISTORY_PAWN_MALUS_NUM, HISTORY_PAWN_SCALE_DEN,
    HISTORY_PAWN_MALUS_THRESHOLD, ENABLE_CHECK_SEE_GATE,
    LPH_ORDER_SCALE,
    TUNE_ORDER_KILLER_1, TUNE_ORDER_COUNTER, TUNE_CHECK_SEE_THR,
    LOW_PLY_HISTORY_SIZE, LMR_REDUCTION_BASE_ADD, LMR_NOT_IMP_DEN, LMR_NOT_IMP_NUM,
    LMR_TABLE_SCALE_PERCENT, PAWN_KEY_INDEX, PAWN_HISTORY_MASK,
    QS_EVASION_CAPTURE_BUCKET, QS_EVASION_QUIET_CLAMP,
    DIAG_MAIN_SCORE_GEOMETRY_REBUILD,
)
from chess_engine.classical.see import _see_ge_jit
from chess_engine.classical.bitboard_utils import find_piece_type_on_square_side, get_lsb_index
from chess_engine.classical.move_generator import (
    KNIGHT_ATTACKS, get_bishop_attacks, get_rook_attacks, PAWN_ATTACKS
)
from chess_engine.classical.engine_types import (
    piece_bbs_signature, occupancy_bbs_signature, game_state_signature,
    search_context_type
)

@numba.njit(cache=True, boundscheck=False, fastmath=True, inline='always')
def trunc_div(n, d):
    return n // d if n >= 0 else -(-n // d)

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def update_history(history_table, piece_type, to_square, bonus):
    """
    Updates the history table using the gravity formula:
    history += bonus - history * abs(bonus) / HISTORY_MAX_MAIN
    This automatically prevents overflow and scales updates.
    """
    current_value = history_table[piece_type, to_square]
    clamped_bonus = min(max(bonus, -HISTORY_MAX_MAIN), HISTORY_MAX_MAIN)
    
    # Gravity formula
    new_value = current_value + clamped_bonus - trunc_div(current_value * abs(clamped_bonus), HISTORY_MAX_MAIN)
    history_table[piece_type, to_square] = new_value

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def update_butterfly_history(butterfly_table, from_sq, to_sq, bonus):
    """
    Updates the butterfly history table using the gravity formula.
    """
    current_value = butterfly_table[from_sq, to_sq]
    clamped_bonus = min(max(bonus, -HISTORY_MAX_BUTTERFLY), HISTORY_MAX_BUTTERFLY)
    
    # Gravity formula
    new_value = current_value + clamped_bonus - trunc_div(current_value * abs(clamped_bonus), HISTORY_MAX_BUTTERFLY)
    butterfly_table[from_sq, to_sq] = new_value

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def update_capture_history(capture_history, piece_type, to_square, victim_type, bonus):
    """
    Updates the capture history table using the gravity formula.
    capture_history: [12, 64, 6] (aggressor_piece, to_square, victim_piece_type 0..5)
    victim_type must be relative piece type in 0..5 (SF CapturePieceToHistory).
    """
    victim_rel = victim_type % 6
    current_value = capture_history[piece_type, to_square, victim_rel]
    clamped_bonus = min(max(bonus, -HISTORY_MAX_CAPTURE), HISTORY_MAX_CAPTURE)
    
    # Gravity formula
    new_value = current_value + clamped_bonus - trunc_div(current_value * abs(clamped_bonus), HISTORY_MAX_CAPTURE)
    capture_history[piece_type, to_square, victim_rel] = new_value

@numba.njit(cache=True, boundscheck=False, fastmath=True, inline='always')
def capture_victim_index(victim_type):
    """Map absolute (0..11) or relative (0..5) piece id to capture-history victim slot 0..5."""
    if victim_type < 0:
        return np.int8(-1)
    return np.int8(victim_type % 6)

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def update_pawn_history(pawn_history, pawn_key_idx, piece_type, to_square, bonus):
    """
    Updates the pawn history table using the gravity formula and Stockfish asymmetric scaling.
    pawn_history: [8192, 12, 64] (pawn_key_idx, piece_type, to_square)
    """
    current_value = pawn_history[pawn_key_idx, piece_type, to_square]
    
    # SF-style asymmetric scaling (Phase 1a constants)
    if bonus > HISTORY_PAWN_MALUS_THRESHOLD:
        scaled_bonus = trunc_div(bonus * HISTORY_PAWN_BONUS_NUM, HISTORY_PAWN_SCALE_DEN)
    else:
        scaled_bonus = trunc_div(bonus * HISTORY_PAWN_MALUS_NUM, HISTORY_PAWN_SCALE_DEN)
    clamped_bonus = min(max(scaled_bonus, -HISTORY_MAX_PAWN), HISTORY_MAX_PAWN)
    
    # Gravity formula with explicit int32 casting to avoid int16 overflow under Numba
    new_value = np.int32(current_value) + clamped_bonus - trunc_div(np.int32(current_value) * abs(clamped_bonus), HISTORY_MAX_PAWN)
    pawn_history[pawn_key_idx, piece_type, to_square] = new_value

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def update_continuation_history(context, ply_offset, prev_move, prev_piece, curr_move, curr_piece, bonus):
    """
    Updates the continuation history table.
    """
    prev_to = get_to_square(prev_move)
    curr_to = get_to_square(curr_move)
    
    current_val = context.continuation_history[ply_offset, prev_piece, prev_to, curr_piece, curr_to]
    clamped_bonus = min(max(bonus, -HISTORY_MAX_CONTINUATION), HISTORY_MAX_CONTINUATION)
    
    new_val = np.int32(current_val) + clamped_bonus - trunc_div(np.int32(current_val) * abs(clamped_bonus), HISTORY_MAX_CONTINUATION)
    context.continuation_history[ply_offset, prev_piece, prev_to, curr_piece, curr_to] = new_val

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def update_quiet_stats_on_tt_hit(search_context, tt_move, tt_aggressor, tt_to, pawn_key_idx, tt_bonus, ply):
    """
    On TT fail-high, update quiet move history stats (Phase 1a linear bonus).
    Called only when tt_aggressor != -1.
    """
    tt_from = get_from_square(tt_move)
    update_history(search_context.history_table, tt_aggressor, tt_to, tt_bonus)
    update_butterfly_history(search_context.butterfly_history, tt_from, tt_to, tt_bonus)
    update_pawn_history(search_context.pawn_history, pawn_key_idx, tt_aggressor, tt_to, tt_bonus)

    # Continuation History
    if ply > 0:
        prev_move = search_context.move_stack[ply - 1]
        prev_piece = search_context.piece_stack[ply - 1]
        if prev_move != NO_MOVE and prev_piece != -1:
            update_continuation_history(search_context, 0, prev_move, prev_piece, tt_move, tt_aggressor, tt_bonus)
    if ply > 1:
        prev_move = search_context.move_stack[ply - 2]
        prev_piece = search_context.piece_stack[ply - 2]
        if prev_move != NO_MOVE and prev_piece != -1:
            update_continuation_history(search_context, 1, prev_move, prev_piece, tt_move, tt_aggressor, tt_bonus)
    if ply > 2:
        prev_move = search_context.move_stack[ply - 3]
        prev_piece = search_context.piece_stack[ply - 3]
        if prev_move != NO_MOVE and prev_piece != -1:
            update_continuation_history(search_context, 2, prev_move, prev_piece, tt_move, tt_aggressor, tt_bonus)
    if ply > 3:
        prev_move = search_context.move_stack[ply - 4]
        prev_piece = search_context.piece_stack[ply - 4]
        if prev_move != NO_MOVE and prev_piece != -1:
            update_continuation_history(search_context, 3, prev_move, prev_piece, tt_move, tt_aggressor, tt_bonus)
    if ply > 5:
        prev_move = search_context.move_stack[ply - 6]
        prev_piece = search_context.piece_stack[ply - 6]
        if prev_move != NO_MOVE and prev_piece != -1:
            update_continuation_history(search_context, 4, prev_move, prev_piece, tt_move, tt_aggressor, tt_bonus)

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def get_quiet_stat_score(
    search_context, ply, from_sq, to_sq, aggressor_type, pawn_key_idx,
    cont1_floor=-1000000000,
):
    score = search_context.history_table[aggressor_type, to_sq] * HISTORY_WEIGHT_MAIN
    score += search_context.pawn_history[pawn_key_idx, aggressor_type, to_sq] * 2
    score += search_context.butterfly_history[from_sq, to_sq]

    if ply > 0:
        prev_move = search_context.move_stack[ply - 1]
        prev_piece = search_context.piece_stack[ply - 1]
        if prev_move != NO_MOVE and prev_piece != -1:
            cont1 = (
                search_context.continuation_history[
                    0, prev_piece, get_to_square(prev_move), aggressor_type, to_sq
                ] * HISTORY_WEIGHT_CONT_1
            )
            if cont1 < cont1_floor:
                cont1 = cont1_floor
            score += cont1

    if ply > 1:
        prev_move = search_context.move_stack[ply - 2]
        prev_piece = search_context.piece_stack[ply - 2]
        if prev_move != NO_MOVE and prev_piece != -1:
            score += search_context.continuation_history[1, prev_piece, get_to_square(prev_move), aggressor_type, to_sq] * HISTORY_WEIGHT_CONT_2

    if ply > 2:
        prev_move = search_context.move_stack[ply - 3]
        prev_piece = search_context.piece_stack[ply - 3]
        if prev_move != NO_MOVE and prev_piece != -1:
            score += search_context.continuation_history[2, prev_piece, get_to_square(prev_move), aggressor_type, to_sq] * HISTORY_WEIGHT_CONT_3

    if ply > 3:
        prev_move = search_context.move_stack[ply - 4]
        prev_piece = search_context.piece_stack[ply - 4]
        if prev_move != NO_MOVE and prev_piece != -1:
            score += search_context.continuation_history[3, prev_piece, get_to_square(prev_move), aggressor_type, to_sq] * HISTORY_WEIGHT_CONT_4

    if ply > 5:
        prev_move = search_context.move_stack[ply - 6]
        prev_piece = search_context.piece_stack[ply - 6]
        if prev_move != NO_MOVE and prev_piece != -1:
            score += search_context.continuation_history[4, prev_piece, get_to_square(prev_move), aggressor_type, to_sq] * HISTORY_WEIGHT_CONT_5

    return score

@numba.njit(cache=True, boundscheck=False, fastmath=True, inline='always')
def get_quiet_ordering_score(
    search_context, ply, from_sq, to_sq, aggressor_type, pawn_key_idx,
):
    """
    Decoupled MovePicker quiet move scoring.
    Uses ORDERING_WEIGHT_* so move ordering can be tuned independently of history pruning.
    """
    score = search_context.history_table[aggressor_type, to_sq] * ORDERING_WEIGHT_MAIN
    score += search_context.pawn_history[pawn_key_idx, aggressor_type, to_sq] * ORDERING_WEIGHT_PAWN
    score += search_context.butterfly_history[from_sq, to_sq] * ORDERING_WEIGHT_BUTTERFLY

    if ply > 0:
        prev_move = search_context.move_stack[ply - 1]
        prev_piece = search_context.piece_stack[ply - 1]
        if prev_move != NO_MOVE and prev_piece != -1:
            score += (
                search_context.continuation_history[
                    0, prev_piece, get_to_square(prev_move), aggressor_type, to_sq
                ] * ORDERING_WEIGHT_CONT_1
            )

    if ply > 1:
        prev_move = search_context.move_stack[ply - 2]
        prev_piece = search_context.piece_stack[ply - 2]
        if prev_move != NO_MOVE and prev_piece != -1:
            score += search_context.continuation_history[1, prev_piece, get_to_square(prev_move), aggressor_type, to_sq] * ORDERING_WEIGHT_CONT_2

    if ply > 2:
        prev_move = search_context.move_stack[ply - 3]
        prev_piece = search_context.piece_stack[ply - 3]
        if prev_move != NO_MOVE and prev_piece != -1:
            score += search_context.continuation_history[2, prev_piece, get_to_square(prev_move), aggressor_type, to_sq] * ORDERING_WEIGHT_CONT_3

    if ply > 3:
        prev_move = search_context.move_stack[ply - 4]
        prev_piece = search_context.piece_stack[ply - 4]
        if prev_move != NO_MOVE and prev_piece != -1:
            score += search_context.continuation_history[3, prev_piece, get_to_square(prev_move), aggressor_type, to_sq] * ORDERING_WEIGHT_CONT_4

    if ply > 5:
        prev_move = search_context.move_stack[ply - 6]
        prev_piece = search_context.piece_stack[ply - 6]
        if prev_move != NO_MOVE and prev_piece != -1:
            score += search_context.continuation_history[4, prev_piece, get_to_square(prev_move), aggressor_type, to_sq] * ORDERING_WEIGHT_CONT_5

    return score

@numba.njit(cache=True, boundscheck=False, fastmath=True, inline='always')
def get_continuation_pruning_score(search_context, ply, see_piece, to_sq, pawn_key_idx):
    """
    Computes unweighted continuation history (1-ply, 2-ply) + pawn history
    for Stockfish 17 Step 14 continuation history pruning.
    """
    cont_hist = np.int32(search_context.pawn_history[pawn_key_idx, see_piece, to_sq])
    if ply > 0:
        prev_m1 = search_context.move_stack[ply - 1]
        prev_p1 = search_context.piece_stack[ply - 1]
        if prev_m1 != NO_MOVE and prev_p1 != -1:
            cont_hist += np.int32(search_context.continuation_history[0, prev_p1, get_to_square(prev_m1), see_piece, to_sq])
    if ply > 1:
        prev_m2 = search_context.move_stack[ply - 2]
        prev_p2 = search_context.piece_stack[ply - 2]
        if prev_m2 != NO_MOVE and prev_p2 != -1:
            cont_hist += np.int32(search_context.continuation_history[1, prev_p2, get_to_square(prev_m2), see_piece, to_sq])
    return cont_hist

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def get_quiet_stat_components(search_context, ply, from_sq, to_sq, aggressor_type, pawn_key_idx):
    """Return the exact eight additive quiet-history terms for diagnostics."""
    main = np.int64(search_context.history_table[aggressor_type, to_sq]) * HISTORY_WEIGHT_MAIN
    butterfly = np.int64(search_context.butterfly_history[from_sq, to_sq])
    pawn = np.int64(search_context.pawn_history[pawn_key_idx, aggressor_type, to_sq]) * 2
    cont1 = np.int64(0)
    cont2 = np.int64(0)
    cont3 = np.int64(0)
    cont4 = np.int64(0)
    cont6 = np.int64(0)

    if ply > 0:
        prev_move = search_context.move_stack[ply - 1]
        prev_piece = search_context.piece_stack[ply - 1]
        if prev_move != NO_MOVE and prev_piece != -1:
            cont1 = np.int64(search_context.continuation_history[
                0, prev_piece, get_to_square(prev_move), aggressor_type, to_sq
            ]) * HISTORY_WEIGHT_CONT_1
    if ply > 1:
        prev_move = search_context.move_stack[ply - 2]
        prev_piece = search_context.piece_stack[ply - 2]
        if prev_move != NO_MOVE and prev_piece != -1:
            cont2 = np.int64(search_context.continuation_history[
                1, prev_piece, get_to_square(prev_move), aggressor_type, to_sq
            ]) * HISTORY_WEIGHT_CONT_2
    if ply > 2:
        prev_move = search_context.move_stack[ply - 3]
        prev_piece = search_context.piece_stack[ply - 3]
        if prev_move != NO_MOVE and prev_piece != -1:
            cont3 = np.int64(search_context.continuation_history[
                2, prev_piece, get_to_square(prev_move), aggressor_type, to_sq
            ]) * HISTORY_WEIGHT_CONT_3
    if ply > 3:
        prev_move = search_context.move_stack[ply - 4]
        prev_piece = search_context.piece_stack[ply - 4]
        if prev_move != NO_MOVE and prev_piece != -1:
            cont4 = np.int64(search_context.continuation_history[
                3, prev_piece, get_to_square(prev_move), aggressor_type, to_sq
            ]) * HISTORY_WEIGHT_CONT_4
    if ply > 5:
        prev_move = search_context.move_stack[ply - 6]
        prev_piece = search_context.piece_stack[ply - 6]
        if prev_move != NO_MOVE and prev_piece != -1:
            cont6 = np.int64(search_context.continuation_history[
                4, prev_piece, get_to_square(prev_move), aggressor_type, to_sq
            ]) * HISTORY_WEIGHT_CONT_5

    return main, butterfly, pawn, cont1, cont2, cont3, cont4, cont6

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def score_captures_lazy(piece_bbs, occupancy_bbs, game_state, moves, scores, start_idx, end_idx, search_context, pinned_white, pinned_black, ply):
    side_to_move = game_state[0]
    
    for i in range(start_idx, end_idx):
        move = moves[i]
        to_square = get_to_square(move)
        from_sq = get_from_square(move)
        
        victim_type = find_piece_type_on_square_side(piece_bbs, to_square, 1 - side_to_move)
        aggressor_type = find_piece_type_on_square_side(piece_bbs, from_sq, side_to_move)
        
        if victim_type == -1 and get_special_move_flag(move) == SPECIAL_MOVE_FLAG_EN_PASSANT:
            victim_type = 0 # Pawn value for En Passant

        mvv_lva = 0
        victim_rel = capture_victim_index(victim_type)
        if victim_rel != -1:
            mvv_lva = MG_MATERIAL_VALUES[victim_rel] - MG_MATERIAL_VALUES[aggressor_type % 6]
        
        # Base score formula: mvv_lva + capture_history_bonus
        base_score = mvv_lva
        if victim_rel != -1:
            base_score += search_context.capture_history[aggressor_type, to_square, victim_rel]
            
        scores[i] = base_score

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def score_captures_with_tt_lazy(piece_bbs, occupancy_bbs, game_state, moves, scores, move_count, tt_move, pinned_white, pinned_black, search_context, ply):
    side_to_move = game_state[0]
    pawn_key_idx = game_state[PAWN_KEY_INDEX] & PAWN_HISTORY_MASK
    
    for i in range(move_count):
        move = moves[i]
        score = 0
        
        if move == tt_move:
            score = SCORE_TT_MOVE
        else:
            to_square = get_to_square(move)
            from_sq = get_from_square(move)
            
            victim_type = find_piece_type_on_square_side(piece_bbs, to_square, 1 - side_to_move)
            aggressor_type = find_piece_type_on_square_side(piece_bbs, from_sq, side_to_move)
            
            if victim_type == -1 and get_special_move_flag(move) == SPECIAL_MOVE_FLAG_EN_PASSANT:
                victim_type = 0 # Pawn value for En Passant

            mvv_lva = 0
            victim_rel = capture_victim_index(victim_type)
            if victim_rel != -1:
                mvv_lva = MG_MATERIAL_VALUES[victim_rel] - MG_MATERIAL_VALUES[aggressor_type % 6]
            
            base_score = mvv_lva
            if victim_rel != -1:
                base_score += (
                    QS_EVASION_CAPTURE_BUCKET
                    + search_context.capture_history[aggressor_type, to_square, victim_rel]
                )
            elif aggressor_type >= 0:
                # qsearch generates all evasions while in check.  Quiet evasions
                # used to tie at zero and were therefore searched in generator
                # order; reuse the main-search history stack to put previously
                # successful evasions first.
                base_score = get_quiet_stat_score(
                    search_context, ply, from_sq, to_square,
                    aggressor_type, pawn_key_idx,
                )
                base_score = max(
                    -QS_EVASION_QUIET_CLAMP,
                    min(QS_EVASION_QUIET_CLAMP, base_score),
                )
            score = base_score
                 
        scores[i] = score

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def score_quiets(
    piece_bbs, occupancy_bbs, game_state, moves, scores, start_idx, end_idx,
    search_context, ply, killer_1, killer_2, counter_move, pawn_key_idx,
    pinned_white, pinned_black,
    check_sq_knight=np.uint64(0), check_sq_bishop=np.uint64(0),
    check_sq_rook=np.uint64(0), check_sq_queen=np.uint64(0),
    check_sq_pawn=np.uint64(0),
):
    if start_idx >= end_idx:
        return
        
    side_to_move = game_state[0]
    all_occ = occupancy_bbs[2]
    
    check_sq_computed = (
        check_sq_knight != np.uint64(0)
        or check_sq_bishop != np.uint64(0)
        or check_sq_rook != np.uint64(0)
        or check_sq_queen != np.uint64(0)
        or check_sq_pawn != np.uint64(0)
    )

    for i in range(start_idx, end_idx):
        move = moves[i]
        to_square = get_to_square(move)
        from_sq = get_from_square(move)
        
        aggressor_type = find_piece_type_on_square_side(piece_bbs, from_sq, side_to_move)
        
        score = get_quiet_ordering_score(search_context, ply, from_sq, to_square, aggressor_type, pawn_key_idx)

        if ply < LOW_PLY_HISTORY_SIZE:
            lph_score = search_context.low_ply_history[ply, move]
            score += LPH_ORDER_SCALE * lph_score // (1 + ply)

        if move == killer_1:
            score += search_context.tune[TUNE_ORDER_KILLER_1]
        elif move == killer_2:
            score += (search_context.tune[TUNE_ORDER_KILLER_1] + search_context.tune[TUNE_ORDER_COUNTER]) // 2
        elif move == counter_move:
            score += search_context.tune[TUNE_ORDER_COUNTER]
        else:
            # Lazy Check Squares (baseline production path).
            if not check_sq_computed:
                enemy_king_sq = get_lsb_index(
                    piece_bbs[11] if side_to_move == 0 else piece_bbs[5]
                )
                check_sq_knight = KNIGHT_ATTACKS[enemy_king_sq]
                check_sq_bishop = get_bishop_attacks(enemy_king_sq, all_occ)
                check_sq_rook = get_rook_attacks(enemy_king_sq, all_occ)
                check_sq_queen = check_sq_bishop | check_sq_rook
                check_sq_pawn = PAWN_ATTACKS[side_to_move, enemy_king_sq]
                check_sq_computed = True
                if search_context.diag_enabled:
                    search_context.diag_stats[
                        DIAG_MAIN_SCORE_GEOMETRY_REBUILD
                    ] += np.uint64(1)

            piece_rel_type = aggressor_type % 6
            to_bb = BB_SQUARES[to_square]

            # --- Check Bonus ---
            is_check_candidate = False
            if piece_rel_type == KNIGHT:
                is_check_candidate = (check_sq_knight & to_bb) != 0
            elif piece_rel_type == BISHOP:
                is_check_candidate = (check_sq_bishop & to_bb) != 0
            elif piece_rel_type == ROOK:
                is_check_candidate = (check_sq_rook & to_bb) != 0
            elif piece_rel_type == QUEEN:
                is_check_candidate = (check_sq_queen & to_bb) != 0
            elif piece_rel_type == PAWN:
                is_check_candidate = (check_sq_pawn & to_bb) != 0

            if is_check_candidate:
                if (not ENABLE_CHECK_SEE_GATE) or _see_ge_jit(
                        piece_bbs, occupancy_bbs, side_to_move, from_sq, to_square,
                        search_context.tune[TUNE_CHECK_SEE_THR], pinned_white, pinned_black, aggressor_type, -2):
                    score += CHECK_BONUS

        scores[i] = score

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def compute_lmr_reduction_1024(
    depth,
    move_count,
    improving,
    table_scale_pct=LMR_TABLE_SCALE_PERCENT,
    not_imp_num=LMR_NOT_IMP_NUM,
):
    """
    Computes 1024-scale base reduction with improving flag.
    SF: reductionScale = reductions[d] * reductions[mn]
    r = reductionScale + !improving * reductionScale * NOT_IMP_NUM / NOT_IMP_DEN + BASE_ADD

    table_scale_pct: scales the REDUCTIONS product (production default LMR_TABLE_SCALE_PERCENT).
    not_imp_num: numerator for not-improving boost (default LMR_NOT_IMP_NUM).
    """
    d = min(depth, 255)
    mc = min(move_count, 255)
    if d < 1 or mc < 1:
        return 0
    reduction_scale = REDUCTIONS[d] * REDUCTIONS[mc]
    if table_scale_pct != 100:
        reduction_scale = reduction_scale * table_scale_pct // 100
    r = reduction_scale + LMR_REDUCTION_BASE_ADD
    if not improving:
        r += reduction_scale * not_imp_num // LMR_NOT_IMP_DEN
    return r

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def get_lmr_stat_score(search_context, ply, from_sq, to_sq, aggressor_type):
    """
    Computes a simplified statScore matching Stockfish range (max ~65,536) for 1024-scale LMR.
    SF: 2 * mainHistory + contHist[0] + contHist[1]
    """
    score = search_context.history_table[aggressor_type, to_sq] * 2
    
    if ply > 0:
        prev_move = search_context.move_stack[ply - 1]
        prev_piece = search_context.piece_stack[ply - 1]
        if prev_move != NO_MOVE and prev_piece != -1:
            score += search_context.continuation_history[0, prev_piece, get_to_square(prev_move), aggressor_type, to_sq]

    if ply > 1:
        prev_move = search_context.move_stack[ply - 2]
        prev_piece = search_context.piece_stack[ply - 2]
        if prev_move != NO_MOVE and prev_piece != -1:
            score += search_context.continuation_history[1, prev_piece, get_to_square(prev_move), aggressor_type, to_sq]

    return score

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def partial_insertion_sort_moves(moves, scores, start_idx, end_idx, limit):
    """
    Sorts moves in descending order up to and including a given score limit.
    Returns the index of the first element that is < limit (i.e. number of sorted elements + start_idx).
    Moves with score < limit are left unsorted at the end.
    """
    sorted_end = start_idx - 1
    
    for p in range(start_idx, end_idx):
        if scores[p] >= limit:
            tmp_score = scores[p]
            tmp_move = moves[p]
            
            sorted_end += 1
            if p != sorted_end:
                scores[p] = scores[sorted_end]
                moves[p] = moves[sorted_end]
            
            q = sorted_end
            while q > start_idx and scores[q - 1] < tmp_score:
                scores[q] = scores[q - 1]
                moves[q] = moves[q - 1]
                q -= 1
                
            scores[q] = tmp_score
            moves[q] = tmp_move
            
    return sorted_end + 1
