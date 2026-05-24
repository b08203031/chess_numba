import numba
import numpy as np

from chess_engine.classical.move import (
    get_to_square, get_from_square, get_special_move_flag,
    SPECIAL_MOVE_FLAG_PROMOTION, SPECIAL_MOVE_FLAG_EN_PASSANT
)
from chess_engine.classical.constants import SCORE_GOOD_CAPTURE_BONUS, SCORE_BAD_CAPTURE_PENALTY, SCORE_KILLER_1, SCORE_KILLER_2, SCORE_COUNTER_MOVE, MAX_HISTORY, LMR_TABLE, MAX_PLY, SCORE_TT_MOVE, BB_SQUARES, NO_MOVE, HISTORY_MAX_MAIN, HISTORY_MAX_BUTTERFLY, HISTORY_MAX_CAPTURE, HISTORY_MAX_CONTINUATION, HISTORY_MAX_PAWN, LMR_HISTORY_DIVISOR, HISTORY_WEIGHT_MAIN, HISTORY_WEIGHT_CONT_1, HISTORY_WEIGHT_CONT_2, HISTORY_WEIGHT_CONT_4, QS_SEE_THRESHOLD
from chess_engine.classical.constants import MG_MATERIAL_VALUES
from chess_engine.classical.see import see_ge
from chess_engine.classical.bitboard_utils import find_piece_type_on_square, find_piece_type_on_square_side, get_lsb_index
from chess_engine.classical.board_operations import find_piece_type_for_square
from chess_engine.classical.move_generator import (
    KNIGHT_ATTACKS, get_bishop_attacks, get_rook_attacks, get_queen_attacks, PAWN_ATTACKS
)
from chess_engine.classical.engine_types import (
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
    if ply > 3:
        prev_move = search_context.move_stack[ply - 4]
        prev_piece = search_context.piece_stack[ply - 4]
        if prev_move != NO_MOVE and prev_piece != -1:
            update_continuation_history(search_context, 2, prev_move, prev_piece, tt_move, tt_aggressor, tt_bonus)

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

    if ply > 3:
        prev_move = search_context.move_stack[ply - 4]
        prev_piece = search_context.piece_stack[ply - 4]
        if prev_move != NO_MOVE and prev_piece != -1:
            score += search_context.continuation_history[2, prev_piece, get_to_square(prev_move), aggressor_type, to_sq] * HISTORY_WEIGHT_CONT_4

    return score

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def score_captures(piece_bbs, occupancy_bbs, game_state, moves, scores, start_idx, end_idx, search_context, pinned_white, pinned_black):
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
        if victim_type != -1:
            mvv_lva = MG_MATERIAL_VALUES[victim_type % 6] - MG_MATERIAL_VALUES[aggressor_type % 6]
        
        # Identify good vs bad capture using SEE right here (R1 Optimization)
        # Using threshold 0 to divide good/bad captures
        is_good_capture = see_ge(piece_bbs, occupancy_bbs, side_to_move, from_sq, to_square, 0, pinned_white, pinned_black, aggressor_type, victim_type)
        
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
def score_captures_with_tt(piece_bbs, occupancy_bbs, game_state, moves, scores, move_count, tt_move, pinned_white, pinned_black, search_context):
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
                
            mvv_lva = 0
            if victim_type != -1:
                mvv_lva = MG_MATERIAL_VALUES[victim_type % 6] - MG_MATERIAL_VALUES[aggressor_type % 6]
            
            is_good_capture = see_ge(piece_bbs, occupancy_bbs, side_to_move, from_sq, to_square, QS_SEE_THRESHOLD, pinned_white, pinned_black, aggressor_type, victim_type)
            
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
def score_quiets(piece_bbs, occupancy_bbs, game_state, moves, scores, start_idx, end_idx, search_context, ply, killer_1, killer_2, counter_move, pawn_key_idx):
    side_to_move = game_state[0]
    
    for i in range(start_idx, end_idx):
        move = moves[i]
        to_square = get_to_square(move)
        from_sq = get_from_square(move)
        
        aggressor_type = find_piece_type_on_square_side(piece_bbs, from_sq, side_to_move)
        score = get_quiet_stat_score(search_context, ply, from_sq, to_square, aggressor_type, pawn_key_idx)

        # Static Check Bonus
        opponent_king_bb = piece_bbs[11] if side_to_move == 0 else piece_bbs[5]
        if opponent_king_bb != 0:
            opp_king_sq = get_lsb_index(opponent_king_bb)
            all_pieces_bb = occupancy_bbs[2]
            is_check = False
            
            if aggressor_type == 0 or aggressor_type == 6: # PAWN
                if PAWN_ATTACKS[side_to_move, opp_king_sq] & BB_SQUARES[to_square]: is_check = True
            elif aggressor_type == 1 or aggressor_type == 7: # KNIGHT
                if KNIGHT_ATTACKS[opp_king_sq] & BB_SQUARES[to_square]: is_check = True
            elif aggressor_type == 2 or aggressor_type == 8: # BISHOP
                if get_bishop_attacks(opp_king_sq, all_pieces_bb) & BB_SQUARES[to_square]: is_check = True
            elif aggressor_type == 3 or aggressor_type == 9: # ROOK
                if get_rook_attacks(opp_king_sq, all_pieces_bb) & BB_SQUARES[to_square]: is_check = True
            elif aggressor_type == 4 or aggressor_type == 10: # QUEEN
                if get_queen_attacks(opp_king_sq, all_pieces_bb) & BB_SQUARES[to_square]: is_check = True

            if is_check:
                score += 15000

        # Threat Avoidance (New)
        opponent_side = 1 - side_to_move
        opp_offset = 6 if opponent_side == 1 else 0
        is_threatened_by_pawn = False
        is_threatened_by_minor = False
        
        # Check if destination is attacked by opponent pawns
        if PAWN_ATTACKS[opponent_side, to_square] & piece_bbs[opp_offset + 0]: 
            is_threatened_by_pawn = True
            
        if not is_threatened_by_pawn:
            # Check if destination is attacked by opponent minor pieces
            if KNIGHT_ATTACKS[to_square] & piece_bbs[opp_offset + 1]: 
                is_threatened_by_minor = True
            elif get_bishop_attacks(to_square, occupancy_bbs[2]) & piece_bbs[opp_offset + 2]: 
                is_threatened_by_minor = True
                
        # Only penalize if we are moving a piece into a lesser attacker
        ptype = aggressor_type % 6
        if is_threatened_by_pawn and ptype > 0: # Non-pawn threatened by pawn
            score -= 10000
        elif is_threatened_by_minor and ptype > 2: # Rook/Queen/King threatened by minor
            score -= 10000

        if move == killer_1:
            score += 400000
        elif move == killer_2:
            score += 350000
        elif move == counter_move:
            score += 300000

        scores[i] = score

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
    side_to_move = game_state[0]
    opponent_pieces_bb = occupancy_bbs[1] if side_to_move == 0 else occupancy_bbs[0]
    
    prev_move_1 = NO_MOVE
    prev_piece_1 = -1
    prev_move_2 = NO_MOVE
    prev_piece_2 = -1
    prev_move_4 = NO_MOVE
    prev_piece_4 = -1
    
    if ply > 0:
        prev_move_1 = search_context.move_stack[ply-1]
        prev_piece_1 = search_context.piece_stack[ply-1]
        
    if ply > 1:
        prev_move_2 = search_context.move_stack[ply-2]
        prev_piece_2 = search_context.piece_stack[ply-2]
        
    if ply > 3:
        prev_move_4 = search_context.move_stack[ply-4]
        prev_piece_4 = search_context.piece_stack[ply-4]

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

                    if prev_move_4 != NO_MOVE and prev_piece_4 != -1:
                        cont_score = search_context.continuation_history[2, prev_piece_4, get_to_square(prev_move_4), aggressor_type, to_square]
                        if cont_score != 0:
                            score += cont_score

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
