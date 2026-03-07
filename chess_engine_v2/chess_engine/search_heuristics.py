import numba
import numpy as np

from chess_engine.move import (
    get_to_square, get_from_square, get_special_move_flag,
    SPECIAL_MOVE_FLAG_PROMOTION
)
from chess_engine.constants import (
    MAX_HISTORY, CONTINUATION_HISTORY_FACTOR, LMR_TABLE,
    MG_MATERIAL_VALUES, NO_MOVE, MAX_PLY, BB_SQUARES,
    SCORE_TT_MOVE, SCORE_GOOD_CAPTURE_BONUS, SCORE_BAD_CAPTURE_PENALTY,
    SCORE_KILLER_1, SCORE_KILLER_2, SCORE_COUNTER_MOVE
)
from chess_engine.see import see_ge
from chess_engine.bitboard_utils import find_piece_type_on_square, find_piece_type_on_square_side, get_lsb_index
from chess_engine.board_operations import find_piece_type_for_square
from chess_engine.move_generator import (
    KNIGHT_ATTACKS, get_bishop_attacks, get_rook_attacks, get_queen_attacks, PAWN_ATTACKS
)
from chess_engine.engine_types import (
    piece_bbs_signature, occupancy_bbs_signature, game_state_signature,
    search_context_type
)

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
def score_captures(piece_bbs, occupancy_bbs, game_state, moves, scores, start_idx, end_idx, search_context):
    side_to_move = game_state[0]
    
    for i in range(start_idx, end_idx):
        move = moves[i]
        to_square = get_to_square(move)
        from_sq = get_from_square(move)
        
        victim_type = find_piece_type_on_square_side(piece_bbs, to_square, 1 - side_to_move)
        aggressor_type = find_piece_type_on_square_side(piece_bbs, from_sq, side_to_move)
        
        # MVV-LVA without SEE
        score = 0
        if victim_type != -1:
             score = MG_MATERIAL_VALUES[victim_type % 6] * 7
        
        # Add capture history
        if victim_type != -1:
             score += search_context.capture_history[aggressor_type, to_square, victim_type]
             
        scores[i] = score

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def score_quiets(piece_bbs, occupancy_bbs, game_state, moves, scores, start_idx, end_idx, search_context, ply):
    side_to_move = game_state[0]
    
    prev_move = NO_MOVE
    prev_piece = -1
    
    if ply > 0:
        prev_move = search_context.move_stack[ply-1]
        if prev_move != NO_MOVE:
            prev_to = get_to_square(prev_move)
            prev_piece = find_piece_type_for_square(piece_bbs, prev_to, 1 - side_to_move)

    for i in range(start_idx, end_idx):
        move = moves[i]
        to_square = get_to_square(move)
        from_sq = get_from_square(move)
        
        aggressor_type = find_piece_type_on_square_side(piece_bbs, from_sq, side_to_move)
        score = search_context.history_table[aggressor_type, to_square]
        
        # Butterfly History
        score += search_context.butterfly_history[from_sq, to_square]
        
        # Continuation History
        if prev_move != NO_MOVE and prev_piece != -1:
            cont_score = search_context.continuation_history[prev_piece, get_to_square(prev_move), aggressor_type, to_square]
            score += cont_score * CONTINUATION_HISTORY_FACTOR

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
