
import numba
import numpy as np
from chess_engine.constants import (
    BB_SQUARES, DE_BRUIJN_SEQUENCE, DE_BRUIJN_INDEX,
    WHITE, BLACK, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING
)
from chess_engine.engine_types import piece_bbs_signature, occupancy_bbs_signature
from chess_engine.bitboard_utils import find_piece_type_on_square
from chess_engine.move import (
    get_from_square, get_to_square, get_special_move_flag, get_promotion_piece,
    SPECIAL_MOVE_FLAG_PROMOTION, SPECIAL_MOVE_FLAG_EN_PASSANT,
    PROMO_KNIGHT, PROMO_BISHOP, PROMO_ROOK, PROMO_QUEEN
)
import numba.types as nbt

# Piece Values for SEE
PIECE_VALUES = np.array([100, 320, 330, 500, 900, 20000], dtype=np.int32)

@numba.njit(nbt.int32(nbt.uint64), cache=True)
def get_lsb_index(bitboard):
    if bitboard == 0:
        return -1
    return DE_BRUIJN_INDEX[((bitboard ^ (bitboard - 1)) * DE_BRUIJN_SEQUENCE) >> np.uint64(58)]

KNIGHT_OFFSETS = np.array([-17, -15, -10, -6, 6, 10, 15, 17], dtype=np.int32)
KING_OFFSETS = np.array([-9, -8, -7, -1, 1, 7, 8, 9], dtype=np.int32)
DIAGONAL_DIRECTIONS = np.array([-9, -7, 7, 9], dtype=np.int32)
ORTHOGONAL_DIRECTIONS = np.array([-8, -1, 1, 8], dtype=np.int32)

@numba.njit(cache=True)
def get_step_attacks(square, offsets):
    attacks = np.uint64(0)
    sq_rank = square // 8
    sq_file = square % 8
    for off in offsets:
        target = square + off
        if 0 <= target < 64:
            tr = target // 8
            tf = target % 8
            if abs(tr - sq_rank) <= 2 and abs(tf - sq_file) <= 2:
                attacks |= BB_SQUARES[int(target)]
    return attacks

@numba.njit(cache=True)
def get_sliding_attacks(square, occupied, is_diagonal):
    attacks = np.uint64(0)
    if is_diagonal:
        directions = DIAGONAL_DIRECTIONS
    else:
        directions = ORTHOGONAL_DIRECTIONS

    for d in directions:
        curr = square
        while True:
            curr_file = curr % 8
            if d == 1 and curr_file == 7: break
            if d == -1 and curr_file == 0: break
            if d == 9 and curr_file == 7: break
            if d == -7 and curr_file == 7: break
            if d == 7 and curr_file == 0: break
            if d == -9 and curr_file == 0: break

            curr += d
            if not (0 <= curr < 64):
                break

            attacks |= BB_SQUARES[int(curr)]
            if (occupied & BB_SQUARES[int(curr)]) != 0:
                break
    return attacks

@numba.njit(cache=True)
def get_pinners(piece_bbs, occupancy_bbs, side, king_sq):
    pinners = np.uint64(0)
    if king_sq == -1: return pinners

    occupied = occupancy_bbs[2]
    enemy_offset = 6 if side == WHITE else 0

    orth_sliders = piece_bbs[ROOK + enemy_offset] | piece_bbs[QUEEN + enemy_offset]
    diag_sliders = piece_bbs[BISHOP + enemy_offset] | piece_bbs[QUEEN + enemy_offset]

    friendly_occupancy = occupancy_bbs[side]

    for d in ORTHOGONAL_DIRECTIONS:
        curr = king_sq
        friend_sq = -1
        while True:
            curr_file = curr % 8
            if d == 1 and curr_file == 7: break
            if d == -1 and curr_file == 0: break
            if d == 8 and curr >= 56: break
            if d == -8 and curr <= 7: break

            curr += d
            if not (0 <= curr < 64): break

            sq_bb = BB_SQUARES[int(curr)]

            if (occupied & sq_bb):
                if friend_sq == -1:
                    if (friendly_occupancy & sq_bb):
                        friend_sq = curr
                    else:
                        break
                else:
                    if (sq_bb & orth_sliders):
                        pinners |= sq_bb
                    break

    for d in DIAGONAL_DIRECTIONS:
        curr = king_sq
        friend_sq = -1
        while True:
            curr_file = curr % 8
            if d == 9 and (curr_file == 7 or curr >= 56): break
            if d == -7 and (curr_file == 7 or curr <= 7): break
            if d == 7 and (curr_file == 0 or curr >= 56): break
            if d == -9 and (curr_file == 0 or curr <= 7): break

            curr += d
            if not (0 <= curr < 64): break

            sq_bb = BB_SQUARES[int(curr)]

            if (occupied & sq_bb):
                if friend_sq == -1:
                    if (friendly_occupancy & sq_bb):
                        friend_sq = curr
                    else:
                        break
                else:
                    if (sq_bb & diag_sliders):
                        pinners |= sq_bb
                    break
    return pinners

@numba.njit(cache=True)
def get_pinned_pieces(piece_bbs, occupancy_bbs, occupied, side, king_sq):
    pinned = np.uint64(0)
    if king_sq == -1: return pinned

    enemy_offset = 6 if side == WHITE else 0

    orth_pinners = piece_bbs[ROOK + enemy_offset] | piece_bbs[QUEEN + enemy_offset]
    diag_pinners = piece_bbs[BISHOP + enemy_offset] | piece_bbs[QUEEN + enemy_offset]

    for d in ORTHOGONAL_DIRECTIONS:
        curr = king_sq
        friend_sq = -1
        while True:
            curr_file = curr % 8
            if d == 1 and curr_file == 7: break
            if d == -1 and curr_file == 0: break
            if d == 8 and curr >= 56: break
            if d == -8 and curr <= 7: break

            curr += d
            if not (0 <= curr < 64): break

            sq_bb = BB_SQUARES[int(curr)]

            if (occupied & sq_bb):
                if friend_sq == -1:
                    if (occupancy_bbs[side] & sq_bb):
                        friend_sq = curr
                    else:
                        break
                else:
                    if (sq_bb & orth_pinners):
                        pinned |= BB_SQUARES[int(friend_sq)]
                    break

    for d in DIAGONAL_DIRECTIONS:
        curr = king_sq
        friend_sq = -1
        while True:
            curr_file = curr % 8
            if d == 9 and (curr_file == 7 or curr >= 56): break
            if d == -7 and (curr_file == 7 or curr <= 7): break
            if d == 7 and (curr_file == 0 or curr >= 56): break
            if d == -9 and (curr_file == 0 or curr <= 7): break

            curr += d
            if not (0 <= curr < 64): break

            sq_bb = BB_SQUARES[int(curr)]

            if (occupied & sq_bb):
                if friend_sq == -1:
                    if (occupancy_bbs[side] & sq_bb):
                        friend_sq = curr
                    else:
                        break
                else:
                    if (sq_bb & diag_pinners):
                        pinned |= BB_SQUARES[int(friend_sq)]
                    break
    return pinned

@numba.njit(cache=True)
def get_attackers_for_see(square, occupied, piece_bbs, side_mask):
    attackers = np.uint64(0)

    if side_mask == WHITE:
        if square >= 9 and (square % 8) > 0:
             if (piece_bbs[PAWN] & BB_SQUARES[int(square - 9)]): attackers |= BB_SQUARES[int(square - 9)]
        if square >= 7 and (square % 8) < 7:
             if (piece_bbs[PAWN] & BB_SQUARES[int(square - 7)]): attackers |= BB_SQUARES[int(square - 7)]
    else:
        if square <= 54 and (square % 8) < 7:
             if (piece_bbs[PAWN+6] & BB_SQUARES[int(square + 9)]): attackers |= BB_SQUARES[int(square + 9)]
        if square <= 56 and (square % 8) > 0:
             if (piece_bbs[PAWN+6] & BB_SQUARES[int(square + 7)]): attackers |= BB_SQUARES[int(square + 7)]

    offset = 0 if side_mask == WHITE else 6

    knights = piece_bbs[KNIGHT + offset]
    if knights:
        attackers |= (get_step_attacks(square, KNIGHT_OFFSETS) & knights)

    sliders_diag = piece_bbs[BISHOP + offset] | piece_bbs[QUEEN + offset]
    if sliders_diag:
        attackers |= (get_sliding_attacks(square, occupied, True) & sliders_diag)

    sliders_orth = piece_bbs[ROOK + offset] | piece_bbs[QUEEN + offset]
    if sliders_orth:
        attackers |= (get_sliding_attacks(square, occupied, False) & sliders_orth)

    kings = piece_bbs[KING + offset]
    if kings:
        attackers |= (get_step_attacks(square, KING_OFFSETS) & kings)

    return attackers

@numba.njit(cache=True)
def get_lva_and_remove(attackers, piece_bbs, side_mask):
    offset = 0 if side_mask == WHITE else 6
    for i in range(6): # P, N, B, R, Q, K
        subset = attackers & piece_bbs[i + offset]
        if subset:
            sq_bb = subset & -subset
            return PIECE_VALUES[i], sq_bb, i
    return 0, np.uint64(0), -1

@numba.njit(nbt.int32(piece_bbs_signature, occupancy_bbs_signature, nbt.int64, nbt.uint16), cache=True)
def see(piece_bbs, occupancy_bbs, side_to_move, move):
    from_sq = get_from_square(move)
    to_sq = get_to_square(move)
    special_flag = get_special_move_flag(move)

    is_promotion = (special_flag == SPECIAL_MOVE_FLAG_PROMOTION)
    is_en_passant = (special_flag == SPECIAL_MOVE_FLAG_EN_PASSANT)

    # 1. Initial Exchange Value

    if is_en_passant:
        value = PIECE_VALUES[PAWN]
    else:
        victim_type = find_piece_type_on_square(piece_bbs, to_sq)
        if victim_type != -1:
            value = PIECE_VALUES[victim_type % 6]
        else:
            value = 0

    if is_promotion:
        promo_code = get_promotion_piece(move)
        promo_type_idx = 0
        if promo_code == PROMO_KNIGHT: promo_type_idx = KNIGHT
        elif promo_code == PROMO_BISHOP: promo_type_idx = BISHOP
        elif promo_code == PROMO_ROOK: promo_type_idx = ROOK
        elif promo_code == PROMO_QUEEN: promo_type_idx = QUEEN

        value += PIECE_VALUES[promo_type_idx] - PIECE_VALUES[PAWN]

    scores = np.zeros(32, dtype=np.int32)
    scores[0] = value

    attacker_type = -1
    offset = 0 if side_to_move == WHITE else 6
    for i in range(6):
        if piece_bbs[i + offset] & BB_SQUARES[int(from_sq)]:
            attacker_type = i
            break

    if attacker_type == -1: return 0

    attacker_value = PIECE_VALUES[attacker_type]

    if is_promotion:
        promo_code = get_promotion_piece(move)
        promo_type_idx = 0
        if promo_code == PROMO_KNIGHT: promo_type_idx = KNIGHT
        elif promo_code == PROMO_BISHOP: promo_type_idx = BISHOP
        elif promo_code == PROMO_ROOK: promo_type_idx = ROOK
        elif promo_code == PROMO_QUEEN: promo_type_idx = QUEEN
        attacker_value = PIECE_VALUES[promo_type_idx]

    occupied = occupancy_bbs[2]
    occupied &= ~BB_SQUARES[int(from_sq)]
    occupied |= BB_SQUARES[int(to_sq)]

    king_sq_white = get_lsb_index(piece_bbs[KING])
    king_sq_black = get_lsb_index(piece_bbs[KING + 6])

    pinners_white = get_pinners(piece_bbs, occupancy_bbs, WHITE, king_sq_white)
    pinners_black = get_pinners(piece_bbs, occupancy_bbs, BLACK, king_sq_black)

    attackers = get_attackers_for_see(to_sq, occupied, piece_bbs, WHITE) | \
                get_attackers_for_see(to_sq, occupied, piece_bbs, BLACK)

    current_side = side_to_move
    current_attacker_val = attacker_value

    depth = 1

    while True:
        current_side = 1 - current_side
        attackers &= occupied

        scores[depth] = current_attacker_val - scores[depth - 1]

        stm_attackers = attackers & (piece_bbs[0] | piece_bbs[1] | piece_bbs[2] | piece_bbs[3] | piece_bbs[4] | piece_bbs[5]) \
                        if current_side == WHITE else \
                        attackers & (piece_bbs[6] | piece_bbs[7] | piece_bbs[8] | piece_bbs[9] | piece_bbs[10] | piece_bbs[11])

        relevant_pinners = pinners_white if current_side == WHITE else pinners_black
        if (relevant_pinners & occupied):
             relevant_king = king_sq_white if current_side == WHITE else king_sq_black
             pinned_mask = get_pinned_pieces(piece_bbs, occupancy_bbs, occupied, current_side, relevant_king)
             stm_attackers &= ~pinned_mask

        if stm_attackers == 0:
            break

        val, bb, type_idx = get_lva_and_remove(stm_attackers, piece_bbs, current_side)

        if (type_idx == KING or type_idx == KING + 6):
             opp_mask = (piece_bbs[6] | piece_bbs[7] | piece_bbs[8] | piece_bbs[9] | piece_bbs[10] | piece_bbs[11]) \
                        if current_side == WHITE else \
                        (piece_bbs[0] | piece_bbs[1] | piece_bbs[2] | piece_bbs[3] | piece_bbs[4] | piece_bbs[5])
             if (attackers & opp_mask):
                 break
             val = 0

        occupied &= ~bb

        if type_idx == PAWN or type_idx == BISHOP or type_idx == QUEEN or \
           type_idx == PAWN+6 or type_idx == BISHOP+6 or type_idx == QUEEN+6:
             sliders_diag = piece_bbs[BISHOP] | piece_bbs[QUEEN] | piece_bbs[BISHOP+6] | piece_bbs[QUEEN+6]
             if sliders_diag:
                 attackers |= (get_sliding_attacks(to_sq, occupied, True) & sliders_diag)

        if type_idx == ROOK or type_idx == QUEEN or \
           type_idx == ROOK+6 or type_idx == QUEEN+6:
             sliders_orth = piece_bbs[ROOK] | piece_bbs[QUEEN] | piece_bbs[ROOK+6] | piece_bbs[QUEEN+6]
             if sliders_orth:
                 attackers |= (get_sliding_attacks(to_sq, occupied, False) & sliders_orth)

        current_attacker_val = val
        depth += 1
        if depth >= 32: break

    while depth > 1:
        depth -= 1
        scores[depth-1] = -max(-scores[depth-1], scores[depth])

    return scores[0]


@numba.njit(nbt.boolean(piece_bbs_signature, occupancy_bbs_signature, nbt.int64, nbt.uint16, nbt.int32), cache=True)
def see_ge(piece_bbs, occupancy_bbs, side_to_move, move, threshold):
    from_sq = get_from_square(move)
    to_sq = get_to_square(move)
    special_flag = get_special_move_flag(move)
    is_promotion = (special_flag == SPECIAL_MOVE_FLAG_PROMOTION)
    is_en_passant = (special_flag == SPECIAL_MOVE_FLAG_EN_PASSANT)

    if is_en_passant:
        value = PIECE_VALUES[PAWN]
    else:
        victim_type = find_piece_type_on_square(piece_bbs, to_sq)
        value = PIECE_VALUES[victim_type % 6] if victim_type != -1 else 0

    if is_promotion:
        promo_code = get_promotion_piece(move)
        promo_type_idx = 0
        if promo_code == PROMO_KNIGHT: promo_type_idx = KNIGHT
        elif promo_code == PROMO_BISHOP: promo_type_idx = BISHOP
        elif promo_code == PROMO_ROOK: promo_type_idx = ROOK
        elif promo_code == PROMO_QUEEN: promo_type_idx = QUEEN
        value += PIECE_VALUES[promo_type_idx] - PIECE_VALUES[PAWN]

    balance = value - threshold
    if balance < 0: return False

    attacker_type = -1
    offset = 0 if side_to_move == WHITE else 6
    for i in range(6):
        if piece_bbs[i + offset] & BB_SQUARES[int(from_sq)]:
            attacker_type = i
            break

    attacker_val = PIECE_VALUES[attacker_type]

    if is_promotion:
        promo_code = get_promotion_piece(move)
        promo_type_idx = 0
        if promo_code == PROMO_KNIGHT: promo_type_idx = KNIGHT
        elif promo_code == PROMO_BISHOP: promo_type_idx = BISHOP
        elif promo_code == PROMO_ROOK: promo_type_idx = ROOK
        elif promo_code == PROMO_QUEEN: promo_type_idx = QUEEN
        attacker_val = PIECE_VALUES[promo_type_idx]

    balance = attacker_val - balance
    if balance <= 0: return True

    occupied = occupancy_bbs[2]
    occupied &= ~BB_SQUARES[int(from_sq)]
    occupied |= BB_SQUARES[int(to_sq)]

    king_sq_white = get_lsb_index(piece_bbs[KING])
    king_sq_black = get_lsb_index(piece_bbs[KING + 6])
    pinners_white = get_pinners(piece_bbs, occupancy_bbs, WHITE, king_sq_white)
    pinners_black = get_pinners(piece_bbs, occupancy_bbs, BLACK, king_sq_black)

    attackers = get_attackers_for_see(to_sq, occupied, piece_bbs, WHITE) | \
                get_attackers_for_see(to_sq, occupied, piece_bbs, BLACK)

    current_side = side_to_move

    while True:
        current_side = 1 - current_side
        attackers &= occupied

        stm_attackers = attackers & (piece_bbs[0] | piece_bbs[1] | piece_bbs[2] | piece_bbs[3] | piece_bbs[4] | piece_bbs[5]) \
                        if current_side == WHITE else \
                        attackers & (piece_bbs[6] | piece_bbs[7] | piece_bbs[8] | piece_bbs[9] | piece_bbs[10] | piece_bbs[11])

        relevant_pinners = pinners_white if current_side == WHITE else pinners_black
        if (relevant_pinners & occupied):
             relevant_king = king_sq_white if current_side == WHITE else king_sq_black
             pinned_mask = get_pinned_pieces(piece_bbs, occupancy_bbs, occupied, current_side, relevant_king)
             stm_attackers &= ~pinned_mask

        if stm_attackers == 0:
            return (current_side != side_to_move)

        val, bb, type_idx = get_lva_and_remove(stm_attackers, piece_bbs, current_side)

        if (type_idx == KING or type_idx == KING + 6):
             opp_mask = (piece_bbs[6] | piece_bbs[7] | piece_bbs[8] | piece_bbs[9] | piece_bbs[10] | piece_bbs[11]) \
                        if current_side == WHITE else \
                        (piece_bbs[0] | piece_bbs[1] | piece_bbs[2] | piece_bbs[3] | piece_bbs[4] | piece_bbs[5])
             if (attackers & opp_mask):
                 return (current_side != side_to_move)
             val = 0

        occupied &= ~bb

        if type_idx == PAWN or type_idx == BISHOP or type_idx == QUEEN or \
           type_idx == PAWN+6 or type_idx == BISHOP+6 or type_idx == QUEEN+6:
             sliders_diag = piece_bbs[BISHOP] | piece_bbs[QUEEN] | piece_bbs[BISHOP+6] | piece_bbs[QUEEN+6]
             if sliders_diag:
                 attackers |= (get_sliding_attacks(to_sq, occupied, True) & sliders_diag)

        if type_idx == ROOK or type_idx == QUEEN or \
           type_idx == ROOK+6 or type_idx == QUEEN+6:
             sliders_orth = piece_bbs[ROOK] | piece_bbs[QUEEN] | piece_bbs[ROOK+6] | piece_bbs[QUEEN+6]
             if sliders_orth:
                 attackers |= (get_sliding_attacks(to_sq, occupied, False) & sliders_orth)

        balance = val - balance
        if balance <= 0:
            return (current_side == side_to_move)

    return True
