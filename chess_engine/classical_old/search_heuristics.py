import numba
import numpy as np

from chess_engine.classical_old.move import (
    get_to_square, get_from_square, get_special_move_flag,
    SPECIAL_MOVE_FLAG_PROMOTION, SPECIAL_MOVE_FLAG_EN_PASSANT
)
from chess_engine.classical_old.constants import SCORE_GOOD_CAPTURE_BONUS, SCORE_BAD_CAPTURE_PENALTY, SCORE_KILLER_1, SCORE_KILLER_2, SCORE_COUNTER_MOVE, MAX_HISTORY, LMR_TABLE, MAX_PLY, SCORE_TT_MOVE, BB_SQUARES, NO_MOVE, HISTORY_MAX_MAIN, HISTORY_MAX_BUTTERFLY, HISTORY_MAX_CAPTURE, HISTORY_MAX_CONTINUATION, HISTORY_MAX_PAWN, LMR_HISTORY_DIVISOR, HISTORY_WEIGHT_MAIN, HISTORY_WEIGHT_CONT_1, HISTORY_WEIGHT_CONT_2, HISTORY_WEIGHT_CONT_3, HISTORY_WEIGHT_CONT_4, HISTORY_WEIGHT_CONT_5, QS_SEE_THRESHOLD, REDUCTIONS, CHECK_BONUS, THREAT_MULTIPLIER, KNIGHT, BISHOP, ROOK, QUEEN, PAWN
from chess_engine.classical_old.constants import MG_MATERIAL_VALUES
from chess_engine.classical_old.see import _see_ge_jit
from chess_engine.classical_old.bitboard_utils import find_piece_type_on_square, find_piece_type_on_square_side, get_lsb_index
from chess_engine.classical_old.board_operations import find_piece_type_for_square
from chess_engine.classical_old.move_generator import (
    KNIGHT_ATTACKS, get_bishop_attacks, get_rook_attacks, get_queen_attacks, PAWN_ATTACKS, KING_ATTACKS,
    get_all_pawn_attacks, get_all_knight_attacks, get_all_bishop_attacks, get_all_rook_attacks
)
from chess_engine.classical_old.engine_types import (
    piece_bbs_signature, occupancy_bbs_signature, game_state_signature,
    search_context_type
)

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
    new_value = current_value + clamped_bonus - (current_value * abs(clamped_bonus)) // HISTORY_MAX_MAIN
    history_table[piece_type, to_square] = new_value

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def update_butterfly_history(butterfly_table, from_sq, to_sq, bonus):
    """
    Updates the butterfly history table using the gravity formula.
    """
    current_value = butterfly_table[from_sq, to_sq]
    clamped_bonus = min(max(bonus, -HISTORY_MAX_BUTTERFLY), HISTORY_MAX_BUTTERFLY)
    
    # Gravity formula
    new_value = current_value + clamped_bonus - (current_value * abs(clamped_bonus)) // HISTORY_MAX_BUTTERFLY
    butterfly_table[from_sq, to_sq] = new_value

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def update_capture_history(capture_history, piece_type, to_square, victim_type, bonus):
    """
    Updates the capture history table using the gravity formula.
    capture_history: [12, 64, 12] (aggressor_piece, to_square, victim_piece)
    """
    current_value = capture_history[piece_type, to_square, victim_type]
    clamped_bonus = min(max(bonus, -HISTORY_MAX_CAPTURE), HISTORY_MAX_CAPTURE)
    
    # Gravity formula
    new_value = current_value + clamped_bonus - (current_value * abs(clamped_bonus)) // HISTORY_MAX_CAPTURE
    capture_history[piece_type, to_square, victim_type] = new_value

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def update_pawn_history(pawn_history, pawn_key_idx, piece_type, to_square, bonus):
    """
    Updates the pawn history table using the gravity formula and Stockfish asymmetric scaling.
    pawn_history: [8192, 12, 64] (pawn_key_idx, piece_type, to_square)
    """
    current_value = pawn_history[pawn_key_idx, piece_type, to_square]
    
    # V2 asymmetric weighting: accept full positive bonus, but reduce the 1.5x malus by 2/3 for pawn history
    scaled_bonus = bonus if bonus > 0 else (bonus * 2) // 3
    clamped_bonus = min(max(scaled_bonus, -HISTORY_MAX_PAWN), HISTORY_MAX_PAWN)
    
    # Gravity formula with explicit int32 casting to avoid int16 overflow under Numba
    new_value = np.int32(current_value) + clamped_bonus - (np.int32(current_value) * abs(clamped_bonus)) // HISTORY_MAX_PAWN
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
    
    new_val = current_val + clamped_bonus - (current_val * abs(clamped_bonus)) // HISTORY_MAX_CONTINUATION
    context.continuation_history[ply_offset, prev_piece, prev_to, curr_piece, curr_to] = new_val

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def update_quiet_stats_on_tt_hit(search_context, tt_move, tt_aggressor, tt_to, pawn_key_idx, tt_bonus, ply):
    """
    On TT fail-high, optimally update quiet move history stats.
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
def get_quiet_stat_score(search_context, ply, from_sq, to_sq, aggressor_type, pawn_key_idx):
    score = search_context.history_table[aggressor_type, to_sq] * HISTORY_WEIGHT_MAIN
    score += search_context.pawn_history[pawn_key_idx, aggressor_type, to_sq] * 2
    score += search_context.butterfly_history[from_sq, to_sq]

    if ply > 0:
        prev_move = search_context.move_stack[ply - 1]
        prev_piece = search_context.piece_stack[ply - 1]
        if prev_move != NO_MOVE and prev_piece != -1:
            score += search_context.continuation_history[0, prev_piece, get_to_square(prev_move), aggressor_type, to_sq] * HISTORY_WEIGHT_CONT_1

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
            
        search_context.aggressor_cache[ply, i] = aggressor_type
        search_context.victim_cache[ply, i] = victim_type

        mvv_lva = 0
        if victim_type != -1:
            mvv_lva = MG_MATERIAL_VALUES[victim_type % 6] - MG_MATERIAL_VALUES[aggressor_type % 6]
        
        # Base score formula: mvv_lva + capture_history_bonus
        base_score = mvv_lva
        if victim_type != -1:
            base_score += search_context.capture_history[aggressor_type, to_square, victim_type]
            
        scores[i] = base_score

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def score_captures_with_tt_lazy(piece_bbs, occupancy_bbs, game_state, moves, scores, move_count, tt_move, pinned_white, pinned_black, search_context, ply):
    side_to_move = game_state[0]
    
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
                
            search_context.aggressor_cache[ply, i] = aggressor_type
            search_context.victim_cache[ply, i] = victim_type

            mvv_lva = 0
            if victim_type != -1:
                mvv_lva = MG_MATERIAL_VALUES[victim_type % 6] - MG_MATERIAL_VALUES[aggressor_type % 6]
            
            base_score = mvv_lva
            if victim_type != -1:
                base_score += search_context.capture_history[aggressor_type, to_square, victim_type]
            score = base_score
                 
        scores[i] = score

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def score_captures(piece_bbs, occupancy_bbs, game_state, moves, scores, start_idx, end_idx, search_context, pinned_white, pinned_black, ply):
    side_to_move = game_state[0]
    
    for i in range(start_idx, end_idx):
        move = moves[i]
        to_square = get_to_square(move)
        from_sq = get_from_square(move)
        
        victim_type = find_piece_type_on_square_side(piece_bbs, to_square, 1 - side_to_move)
        aggressor_type = find_piece_type_on_square_side(piece_bbs, from_sq, side_to_move)
        
        if victim_type == -1 and get_special_move_flag(move) == SPECIAL_MOVE_FLAG_EN_PASSANT:
            victim_type = 0 # Pawn value for En Passant
            
        search_context.aggressor_cache[ply, i] = aggressor_type
        search_context.victim_cache[ply, i] = victim_type

        mvv_lva = 0
        if victim_type != -1:
            mvv_lva = MG_MATERIAL_VALUES[victim_type % 6] - MG_MATERIAL_VALUES[aggressor_type % 6]
        
        # Identify good vs bad capture using SEE right here (R1 Optimization)
        # Using threshold 0 to divide good/bad captures
        is_good_capture = _see_ge_jit(piece_bbs, occupancy_bbs, side_to_move, from_sq, to_square, 0, pinned_white, pinned_black, aggressor_type, victim_type)
        
        score = 0
        if is_good_capture:
            score = SCORE_GOOD_CAPTURE_BONUS + mvv_lva
            if victim_type != -1:
                score += search_context.capture_history[aggressor_type, to_square, victim_type]
        else:
            score = SCORE_BAD_CAPTURE_PENALTY + mvv_lva
            if victim_type != -1:
                score += search_context.capture_history[aggressor_type, to_square, victim_type]
             
        scores[i] = score

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def score_captures_with_tt(piece_bbs, occupancy_bbs, game_state, moves, scores, move_count, tt_move, pinned_white, pinned_black, search_context, ply):
    side_to_move = game_state[0]
    
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
                
            search_context.aggressor_cache[ply, i] = aggressor_type
            search_context.victim_cache[ply, i] = victim_type

            mvv_lva = 0
            if victim_type != -1:
                mvv_lva = MG_MATERIAL_VALUES[victim_type % 6] - MG_MATERIAL_VALUES[aggressor_type % 6]
            
            is_good_capture = _see_ge_jit(piece_bbs, occupancy_bbs, side_to_move, from_sq, to_square, QS_SEE_THRESHOLD, pinned_white, pinned_black, aggressor_type, victim_type)
            
            if is_good_capture:
                score = SCORE_GOOD_CAPTURE_BONUS + mvv_lva
                if victim_type != -1:
                    score += search_context.capture_history[aggressor_type, to_square, victim_type]
                if score < SCORE_GOOD_CAPTURE_BONUS:
                    score = SCORE_GOOD_CAPTURE_BONUS
            else:
                score = SCORE_BAD_CAPTURE_PENALTY + mvv_lva
                if victim_type != -1:
                    score += search_context.capture_history[aggressor_type, to_square, victim_type]
                 
        scores[i] = score

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def score_quiets(piece_bbs, occupancy_bbs, game_state, moves, scores, start_idx, end_idx, search_context, ply, killer_1, killer_2, counter_move, pawn_key_idx, pinned_white, pinned_black):
    if start_idx >= end_idx:
        return
        
    side_to_move = game_state[0]
    all_occ = occupancy_bbs[2]
    enemy_side = 1 - side_to_move
    
    threats_computed = False
    check_sq_computed = False
    
    # Initialize check squares to 0
    check_sq_knight = np.uint64(0)
    check_sq_bishop = np.uint64(0)
    check_sq_rook   = np.uint64(0)
    check_sq_queen  = np.uint64(0)
    check_sq_pawn   = np.uint64(0)
    
    # Initialize threat bitboards to 0
    threat_knight = np.uint64(0)
    threat_bishop = np.uint64(0)
    threat_rook   = np.uint64(0)
    threat_queen  = np.uint64(0)

    for i in range(start_idx, end_idx):
        move = moves[i]
        to_square = get_to_square(move)
        from_sq = get_from_square(move)
        
        aggressor_type = find_piece_type_on_square_side(piece_bbs, from_sq, side_to_move)
        search_context.aggressor_cache[ply, i] = aggressor_type
        
        score = get_quiet_stat_score(search_context, ply, from_sq, to_square, aggressor_type, pawn_key_idx)

        if ply < 5:
            lph_score = search_context.low_ply_history[ply, move]
            score += 8 * lph_score // (1 + ply)

        if move == killer_1:
            score += 400000
        elif move == killer_2:
            score += 350000
        elif move == counter_move:
            score += 300000
        else:
            # Lazy Check Squares (Part A)
            if not check_sq_computed:
                enemy_king_sq = get_lsb_index(piece_bbs[11] if side_to_move == 0 else piece_bbs[5])
                check_sq_knight = KNIGHT_ATTACKS[enemy_king_sq]
                check_sq_bishop = get_bishop_attacks(enemy_king_sq, all_occ)
                check_sq_rook   = get_rook_attacks(enemy_king_sq, all_occ)
                check_sq_queen  = check_sq_bishop | check_sq_rook
                check_sq_pawn   = PAWN_ATTACKS[side_to_move, enemy_king_sq]
                check_sq_computed = True
                
            # Lazy Threat Bitboards (Part B)
            if not threats_computed:
                enemy_pawn_attacks = get_all_pawn_attacks(piece_bbs, enemy_side)
                enemy_knight_attacks = get_all_knight_attacks(piece_bbs, enemy_side)
                
                threat_knight = enemy_pawn_attacks
                threat_bishop = enemy_pawn_attacks
                threat_rook   = enemy_pawn_attacks | enemy_knight_attacks
                threat_queen  = threat_rook
                threat_computed = True

            piece_rel_type = aggressor_type % 6
            to_bb = BB_SQUARES[to_square]
            from_bb = BB_SQUARES[from_sq]

            # --- Check Bonus (Part A) ---
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
                score += CHECK_BONUS

            # --- Threat-Based Reordering (Part B) ---
            threat_bb = np.uint64(0)
            piece_value = 0
            if piece_rel_type == KNIGHT:
                threat_bb = threat_knight; piece_value = 320
            elif piece_rel_type == BISHOP:
                threat_bb = threat_bishop; piece_value = 330
            elif piece_rel_type == ROOK:
                threat_bb = threat_rook; piece_value = 500
            elif piece_rel_type == QUEEN:
                threat_bb = threat_queen; piece_value = 900
                
            if piece_value > 0:
                escape = 1 if (threat_bb & from_bb) != 0 else 0
                enter  = 1 if (threat_bb & to_bb) != 0 else 0
                score += piece_value * THREAT_MULTIPLIER * (escape - enter)

        scores[i] = score

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def compute_lmr_reduction_1024(depth, move_count, improving):
    """
    Computes 1024-scale base reduction with improving flag.
    SF: reductionScale = reductions[d] * reductions[mn]
    r = reductionScale + !improving * reductionScale * 194 / 512 + 1027
    """
    d = min(depth, 255)
    mc = min(move_count, 255)
    if d < 1 or mc < 1:
        return 0
    reduction_scale = REDUCTIONS[d] * REDUCTIONS[mc]
    r = reduction_scale + 1027
    if not improving:
        r += reduction_scale * 194 // 512
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
def get_lmr_reduction(depth, move_count, history_score, improving, is_pv):
    """
    Calculates LMR reduction based on depth, move_count, history score, improving flag and node type.
    Uses precomputed LMR_TABLE for base reduction.
    """
    if depth < 2 or move_count < 2:
        return 0
    
    # Clamp indices to table bounds
    d = min(depth, MAX_PLY - 1)
    mc = min(move_count, 255)
    
    # Base reduction from table
    reduction = float(LMR_TABLE[d, mc])
    
    # PV adjustment: PV nodes are searched more carefully
    if not is_pv:
        reduction += 0.5
    
    # Improving adjustment: If position is not improving, reduce more aggressively
    if not improving:
        reduction += 0.5

    # History Adjustment: V2-specific scaling for linear history
    history_adjustment = (history_score / LMR_HISTORY_DIVISOR)
    reduction -= history_adjustment

    return max(0, int(reduction))

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def score_moves(piece_bbs, occupancy_bbs, game_state, moves, scores, move_count, tt_move, killer_moves_at_ply, history_table, counter_move, pinned_white, pinned_black, search_context, ply, pawn_key_idx):
    if move_count <= 0:
        return
        
    side_to_move = game_state[0]
    opponent_pieces_bb = occupancy_bbs[1] if side_to_move == 0 else occupancy_bbs[0]
    all_occ = occupancy_bbs[2]
    enemy_side = 1 - side_to_move
    
    threats_computed = False
    check_sq_computed = False
    
    # Initialize check squares to 0
    check_sq_knight = np.uint64(0)
    check_sq_bishop = np.uint64(0)
    check_sq_rook   = np.uint64(0)
    check_sq_queen  = np.uint64(0)
    check_sq_pawn   = np.uint64(0)
    
    # Initialize threat bitboards to 0
    threat_knight = np.uint64(0)
    threat_bishop = np.uint64(0)
    threat_rook   = np.uint64(0)
    threat_queen  = np.uint64(0)

    prev_move_1 = NO_MOVE
    prev_piece_1 = -1
    prev_move_2 = NO_MOVE
    prev_piece_2 = -1
    prev_move_3 = NO_MOVE
    prev_piece_3 = -1
    prev_move_4 = NO_MOVE
    prev_piece_4 = -1
    prev_move_6 = NO_MOVE
    prev_piece_6 = -1
    
    if ply > 0:
        prev_move_1 = search_context.move_stack[ply-1]
        prev_piece_1 = search_context.piece_stack[ply-1]
        
    if ply > 1:
        prev_move_2 = search_context.move_stack[ply-2]
        prev_piece_2 = search_context.piece_stack[ply-2]
        
    if ply > 2:
        prev_move_3 = search_context.move_stack[ply-3]
        prev_piece_3 = search_context.piece_stack[ply-3]
        
    if ply > 3:
        prev_move_4 = search_context.move_stack[ply-4]
        prev_piece_4 = search_context.piece_stack[ply-4]

    if ply > 5:
        prev_move_6 = search_context.move_stack[ply-6]
        prev_piece_6 = search_context.piece_stack[ply-6]

    for i in range(move_count):
        move = moves[i]
        score = 0
        if move == tt_move:
            score = SCORE_TT_MOVE
        else:
            to_square = get_to_square(move)
            is_capture = ((opponent_pieces_bb & BB_SQUARES[to_square]) != 0) or (get_special_move_flag(move) == SPECIAL_MOVE_FLAG_EN_PASSANT)
            if is_capture:
                # Use SEE to distinguish Good vs Bad captures
                from_sq = get_from_square(move)
                
                # Optimization: Lookup piece types once and reuse them in see_ge and MVV_LVA
                victim_type = find_piece_type_on_square_side(piece_bbs, to_square, 1 - side_to_move)
                aggressor_type = find_piece_type_on_square_side(piece_bbs, from_sq, side_to_move)
                
                if victim_type == -1 and get_special_move_flag(move) == SPECIAL_MOVE_FLAG_EN_PASSANT:
                    victim_type = 0 # Pawn
 
                search_context.aggressor_cache[ply, i] = aggressor_type
                search_context.victim_cache[ply, i] = victim_type
 
                # Optimization: Use see_ge(0) instead of full see()
                is_good_capture = _see_ge_jit(piece_bbs, occupancy_bbs, side_to_move, from_sq, to_square, 0, pinned_white, pinned_black, aggressor_type, victim_type)
                
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
                from_sq = get_from_square(move)
                aggressor_type = find_piece_type_on_square_side(piece_bbs, from_sq, side_to_move)
                search_context.aggressor_cache[ply, i] = aggressor_type
 
                if move == killer_moves_at_ply[0]:
                    score = SCORE_KILLER_1
                elif move == killer_moves_at_ply[1]:
                    score = SCORE_KILLER_2
                elif move == counter_move:
                    score = SCORE_COUNTER_MOVE
                else:
                    score = history_table[aggressor_type, to_square]
                    
                    # Pawn History
                    score += search_context.pawn_history[pawn_key_idx, aggressor_type, to_square] * 2
                    
                    # Butterfly History
                    score += search_context.butterfly_history[from_sq, to_square]
                    
                    # Continuation History
                    if prev_move_1 != NO_MOVE and prev_piece_1 != -1:
                        cont_score = search_context.continuation_history[0, prev_piece_1, get_to_square(prev_move_1), aggressor_type, to_square]
                        if cont_score != 0:
                            score += cont_score
 
                    if prev_move_2 != NO_MOVE and prev_piece_2 != -1:
                        cont_score = search_context.continuation_history[1, prev_piece_2, get_to_square(prev_move_2), aggressor_type, to_square]
                        if cont_score != 0:
                            score += cont_score
 
                    if prev_move_3 != NO_MOVE and prev_piece_3 != -1:
                        cont_score = search_context.continuation_history[2, prev_piece_3, get_to_square(prev_move_3), aggressor_type, to_square]
                        if cont_score != 0:
                            score += cont_score
 
                    if prev_move_4 != NO_MOVE and prev_piece_4 != -1:
                        cont_score = search_context.continuation_history[3, prev_piece_4, get_to_square(prev_move_4), aggressor_type, to_square]
                        if cont_score != 0:
                            score += cont_score
 
                    if prev_move_6 != NO_MOVE and prev_piece_6 != -1:
                        cont_score = search_context.continuation_history[4, prev_piece_6, get_to_square(prev_move_6), aggressor_type, to_square]
                        if cont_score != 0:
                            score += cont_score
 
                    if ply < 5:
                        lph_score = search_context.low_ply_history[ply, move]
                        score += 8 * lph_score // (1 + ply)

                    # Lazy Check Squares (Part A)
                    if not check_sq_computed:
                        enemy_king_sq = get_lsb_index(piece_bbs[11] if side_to_move == 0 else piece_bbs[5])
                        check_sq_knight = KNIGHT_ATTACKS[enemy_king_sq]
                        check_sq_bishop = get_bishop_attacks(enemy_king_sq, all_occ)
                        check_sq_rook   = get_rook_attacks(enemy_king_sq, all_occ)
                        check_sq_queen  = check_sq_bishop | check_sq_rook
                        check_sq_pawn   = PAWN_ATTACKS[side_to_move, enemy_king_sq]
                        check_sq_computed = True
                        
                    # Lazy Threat Bitboards (Part B)
                    if not threats_computed:
                        enemy_pawn_attacks = get_all_pawn_attacks(piece_bbs, enemy_side)
                        enemy_knight_attacks = get_all_knight_attacks(piece_bbs, enemy_side)
                        
                        threat_knight = enemy_pawn_attacks
                        threat_bishop = enemy_pawn_attacks
                        threat_rook   = enemy_pawn_attacks | enemy_knight_attacks
                        threat_queen  = threat_rook
                        threat_computed = True

                    piece_rel_type = aggressor_type % 6
                    to_bb = BB_SQUARES[to_square]
                    from_bb = BB_SQUARES[from_sq]

                    # --- Check Bonus (Part A) ---
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
                        score += CHECK_BONUS

                    # --- Threat-Based Reordering (Part B) ---
                    threat_bb = np.uint64(0)
                    piece_value = 0
                    if piece_rel_type == KNIGHT:
                        threat_bb = threat_knight; piece_value = 320
                    elif piece_rel_type == BISHOP:
                        threat_bb = threat_bishop; piece_value = 330
                    elif piece_rel_type == ROOK:
                        threat_bb = threat_rook; piece_value = 500
                    elif piece_rel_type == QUEEN:
                        threat_bb = threat_queen; piece_value = 900
                        
                    if piece_value > 0:
                        escape = 1 if (threat_bb & from_bb) != 0 else 0
                        enter  = 1 if (threat_bb & to_bb) != 0 else 0
                        score += piece_value * THREAT_MULTIPLIER * (escape - enter)
 
        scores[i] = score

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
