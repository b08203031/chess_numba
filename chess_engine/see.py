
import numba
import numpy as np
from chess_engine.constants import (
    BB_SQUARES, DE_BRUIJN_SEQUENCE, DE_BRUIJN_INDEX,
    WHITE, BLACK, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING,
    MG_MATERIAL_VALUES
)
from chess_engine.engine_types import piece_bbs_signature, occupancy_bbs_signature
from chess_engine.bitboard_utils import find_piece_type_on_square
from chess_engine.move import (
    get_from_square, get_to_square, get_promotion_piece,
    get_special_move_flag, SPECIAL_MOVE_FLAG_PROMOTION,
    PROMOTION_PIECE_SHIFT
)
import numba.types as nbt

# Piece Values for SEE (based on Stockfish's internal values for SEE)
# P=100, N=320, B=330, R=500, Q=900, K=20000
# Aligned with engine constants where possible, but optimized for SEE stability.
PIECE_VALUES = np.array([100, 320, 330, 500, 900, 20000], dtype=np.int32)

@numba.njit(nbt.int32(nbt.uint64), cache=True)
def get_lsb_index(bitboard):
    """Returns the index (0-63) of the least significant bit set."""
    if bitboard == 0:
        return -1
    return DE_BRUIJN_INDEX[((bitboard ^ (bitboard - 1)) * DE_BRUIJN_SEQUENCE) >> np.uint64(58)]

# Basic Attack Tables
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
                attacks |= BB_SQUARES[target]
    return attacks

@numba.njit(cache=True)
def get_sliding_attacks(square, occupied, is_diagonal):
    attacks = np.uint64(0)
    directions = DIAGONAL_DIRECTIONS if is_diagonal else ORTHOGONAL_DIRECTIONS

    for d in directions:
        curr = square
        while True:
            curr_file = curr % 8
            # Boundary checks for 8x8 board
            if d == 1 and curr_file == 7: break
            if d == -1 and curr_file == 0: break
            if d == 9 and curr_file == 7: break
            if d == -7 and curr_file == 7: break
            if d == 7 and curr_file == 0: break
            if d == -9 and curr_file == 0: break

            curr += d
            if not (0 <= curr < 64):
                break

            # Explicitly cast index to int64 to avoid float inference
            idx = np.int64(curr)
            attacks |= BB_SQUARES[idx]
            if (occupied & BB_SQUARES[idx]) != 0:
                break
    return attacks

@numba.njit(cache=True)
def get_pinned_pieces(piece_bbs, occupancy_bbs, side):
    """
    Returns a bitboard of all pieces of 'side' that are pinned to their King
    by enemy sliding pieces.
    """
    pinned = np.uint64(0)
    king_idx = KING if side == WHITE else (KING + 6)
    king_sq = get_lsb_index(piece_bbs[king_idx])

    if king_sq == -1: return pinned

    occupied = occupancy_bbs[2]
    enemy_offset = 6 if side == WHITE else 0

    orth_pinners = piece_bbs[ROOK + enemy_offset] | piece_bbs[QUEEN + enemy_offset]
    diag_pinners = piece_bbs[BISHOP + enemy_offset] | piece_bbs[QUEEN + enemy_offset]

    # Check Orthogonal Rays
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

            sq_bb = BB_SQUARES[curr]

            if (occupied & sq_bb):
                if friend_sq == -1:
                    is_us = False
                    if side == WHITE:
                         if (occupancy_bbs[WHITE] & sq_bb): is_us = True
                    else:
                         if (occupancy_bbs[BLACK] & sq_bb): is_us = True

                    if is_us:
                        friend_sq = curr
                    else:
                        break # Enemy piece blocks
                else:
                    if (sq_bb & orth_pinners):
                        pinned |= BB_SQUARES[friend_sq]
                    break

    # Check Diagonal Rays
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

            sq_bb = BB_SQUARES[curr]

            if (occupied & sq_bb):
                if friend_sq == -1:
                    is_us = False
                    if side == WHITE:
                         if (occupancy_bbs[WHITE] & sq_bb): is_us = True
                    else:
                         if (occupancy_bbs[BLACK] & sq_bb): is_us = True

                    if is_us:
                        friend_sq = curr
                    else:
                        break
                else:
                    if (sq_bb & diag_pinners):
                        pinned |= BB_SQUARES[friend_sq]
                    break

    return pinned

@numba.njit(cache=True)
def get_attackers_for_see(square, occupied, piece_bbs, side_mask):
    """
    Get bitboard of all pieces of 'side_mask' (0=White, 1=Black) attacking 'square'.
    """
    attackers = np.uint64(0)

    # 1. Pawns
    if side_mask == WHITE:
        if square >= 9 and (square % 8) > 0:
             if (piece_bbs[PAWN] & BB_SQUARES[square - 9]): attackers |= BB_SQUARES[square - 9]
        if square >= 7 and (square % 8) < 7:
             if (piece_bbs[PAWN] & BB_SQUARES[square - 7]): attackers |= BB_SQUARES[square - 7]
    else:
        if square <= 54 and (square % 8) < 7:
             if (piece_bbs[PAWN+6] & BB_SQUARES[square + 9]): attackers |= BB_SQUARES[square + 9]
        if square <= 56 and (square % 8) > 0:
             if (piece_bbs[PAWN+6] & BB_SQUARES[square + 7]): attackers |= BB_SQUARES[square + 7]

    offset = 0 if side_mask == WHITE else 6

    # 2. Knights
    knights = piece_bbs[KNIGHT + offset]
    if knights:
        attackers |= (get_step_attacks(square, KNIGHT_OFFSETS) & knights)

    # 3. Sliders (Magic Bitboards replacement logic)
    sliders_diag = piece_bbs[BISHOP + offset] | piece_bbs[QUEEN + offset]
    if sliders_diag:
        attackers |= (get_sliding_attacks(square, occupied, True) & sliders_diag)

    sliders_orth = piece_bbs[ROOK + offset] | piece_bbs[QUEEN + offset]
    if sliders_orth:
        attackers |= (get_sliding_attacks(square, occupied, False) & sliders_orth)

    # 4. King
    kings = piece_bbs[KING + offset]
    if kings:
        attackers |= (get_step_attacks(square, KING_OFFSETS) & kings)

    return attackers

@numba.njit(cache=True)
def get_lva_and_remove(attackers, piece_bbs, side_mask):
    """
    Finds LVA, returns (value, bitboard, type_idx).
    """
    offset = 0 if side_mask == WHITE else 6
    # Check in order: P, N, B, R, Q, K
    for i in range(6):
        subset = attackers & piece_bbs[i + offset]
        if subset:
            sq_bb = subset & -subset
            return PIECE_VALUES[i], sq_bb, i
    return 0, np.uint64(0), -1

@numba.njit(nbt.int32(piece_bbs_signature, occupancy_bbs_signature, nbt.int64, nbt.uint16), cache=True)
def see(piece_bbs, occupancy_bbs, side_to_move, move):
    """
    Static Exchange Evaluation (SEE).
    Updated to handle:
    - Moves with Promotion (via encoded 'move' uint16)
    - En Passant
    - Pins
    - X-Rays
    """
    from_sq = get_from_square(move)
    to_sq = get_to_square(move)
    special_flag = get_special_move_flag(move)

    # 1. Identify Initial Attacker Type
    attacker_type = -1
    offset = 0 if side_to_move == WHITE else 6
    for i in range(6):
        if piece_bbs[i + offset] & BB_SQUARES[from_sq]:
            attacker_type = i
            break

    if attacker_type == -1: return 0

    # 2. Identify Initial Victim Value
    # Handle Promotion: If promoting, we "capture" the difference in value
    # But usually SEE is about the capture on 'to_sq'.
    # Stockfish logic: swap = PieceValue[to] + (Promo - Pawn) if promotion.

    victim_type = -1
    enemy_offset = 6 if side_to_move == WHITE else 0
    for i in range(6):
        if piece_bbs[i + enemy_offset] & BB_SQUARES[to_sq]:
            victim_type = i
            break

    if victim_type == -1:
        # En Passant
        if attacker_type == PAWN and abs((from_sq % 8) - (to_sq % 8)) == 1 and special_flag == 0:
             # Heuristic check for EP if flag not available/reliable in all contexts?
             # But we should rely on caller.
             pass

        # If move has EN_PASSANT flag (2)
        # Note: current `encode_move` might set EN_PASSANT flag
        # We check special_flag == 2
        if special_flag == 2: # SPECIAL_MOVE_FLAG_EN_PASSANT
            value = PIECE_VALUES[PAWN]
        else:
            value = 0 # Quiet move to empty square
    else:
        value = PIECE_VALUES[victim_type]

    if special_flag == SPECIAL_MOVE_FLAG_PROMOTION:
        promo_type_encoded = get_promotion_piece(move)
        # Map 0->N, 1->B, 2->R, 3->Q
        # promo_type_idx: 1, 2, 3, 4
        promo_type_idx = promo_type_encoded + 1
        value += PIECE_VALUES[promo_type_idx] - PIECE_VALUES[PAWN]

    # Scores stack
    scores = np.zeros(32, dtype=np.int32)
    scores[0] = value

    # Attacker value for the next recapture
    if special_flag == SPECIAL_MOVE_FLAG_PROMOTION:
        promo_type_encoded = get_promotion_piece(move)
        scores[1] = PIECE_VALUES[promo_type_encoded + 1] # Promoted piece value
    else:
        scores[1] = PIECE_VALUES[attacker_type]

    # Setup Occupied Bitboard
    occupied = occupancy_bbs[2]
    occupied &= ~BB_SQUARES[from_sq]

    # Calculate Pinned Pieces
    pinned_white = get_pinned_pieces(piece_bbs, occupancy_bbs, WHITE)
    pinned_black = get_pinned_pieces(piece_bbs, occupancy_bbs, BLACK)

    # Find all attackers to `to_sq`
    # Note: For EP, the pawn on to_sq is 'ghost', real pawn is captured elsewhere.
    # But SEE targets 'to_sq'. EP capture happens on to_sq effectively for the swap sequence on that square.
    # Stockfish handles EP by shifting 'to_sq' for pawn attacks?
    # Stockfish: "If m.type_of() == EN_PASSANT... swap = PawnValue... capsq = to - pawn_push..."
    # But the swap loop happens on 'to'.

    attackers = get_attackers_for_see(to_sq, occupied, piece_bbs, WHITE) | \
                get_attackers_for_see(to_sq, occupied, piece_bbs, BLACK)

    current_side = side_to_move
    current_attacker_val = scores[1]
    depth = 1

    while True:
        current_side = 1 - current_side # Swap side

        attackers &= occupied # Remove captured pieces from attackers

        # Calculate gain
        # scores[depth] = value_of_piece_just_captured
        # scores[depth] comes from previous iteration's `current_attacker_val` which was the attacker.
        # Wait, the logic is: gain[i] = val_captured - see(...).
        # scores[0] is initial capture gain.
        # scores[1] is attacker value (penalty for next guy).
        # We need to store gains.
        # Stockfish:
        #   values[0] = swap (initial capture val)
        #   values[1] = attacker_val
        #   loop:
        #      values[d] = new_attacker_val
        #      d++
        #   backprop: gains[i] = values[i] - gains[i-1] ... wait.
        #   Stockfish backprop:
        #     gains[0] = values[0]
        #     for i=1..d: gains[i] = values[i] - gains[i-1] ? NO.
        #     Stockfish backprop is:
        #       gains[i-1] = -max(-gains[i-1], gains[i])
        #     It stores raw piece values in `values` array?
        #     Yes: values[d] = PieceValue[attacker]

        # Let's fix our structure to match Stockfish exactly.

        stm_attackers = attackers & (piece_bbs[0] | piece_bbs[1] | piece_bbs[2] | piece_bbs[3] | piece_bbs[4] | piece_bbs[5]) \
                        if current_side == WHITE else \
                        attackers & (piece_bbs[6] | piece_bbs[7] | piece_bbs[8] | piece_bbs[9] | piece_bbs[10] | piece_bbs[11])

        pinned_mask = pinned_white if current_side == WHITE else pinned_black
        stm_attackers &= ~pinned_mask

        if stm_attackers == 0:
            break

        val, bb, type_idx = get_lva_and_remove(stm_attackers, piece_bbs, current_side)

        # King Logic
        if (type_idx == KING or type_idx == KING + 6):
             opp_mask = (piece_bbs[6] | piece_bbs[7] | piece_bbs[8] | piece_bbs[9] | piece_bbs[10] | piece_bbs[11]) \
                        if current_side == WHITE else \
                        (piece_bbs[0] | piece_bbs[1] | piece_bbs[2] | piece_bbs[3] | piece_bbs[4] | piece_bbs[5])
             if (attackers & opp_mask):
                 break # King cannot capture into check
             val = 0 # Safe king capture

        depth += 1
        scores[depth] = val # Store the value of the piece that just captured (the new victim for next ply)

        occupied &= ~bb # Remove attacker from board (it is now on 'to_sq' conceptually, but for X-ray we clear source)

        # Add X-Ray attackers (behind the piece that just moved)
        if type_idx in (PAWN, BISHOP, QUEEN, PAWN+6, BISHOP+6, QUEEN+6):
             sliders_diag = piece_bbs[BISHOP] | piece_bbs[QUEEN] | piece_bbs[BISHOP+6] | piece_bbs[QUEEN+6]
             if sliders_diag:
                 attackers |= (get_sliding_attacks(to_sq, occupied, True) & sliders_diag)

        if type_idx in (ROOK, QUEEN, ROOK+6, QUEEN+6):
             sliders_orth = piece_bbs[ROOK] | piece_bbs[QUEEN] | piece_bbs[ROOK+6] | piece_bbs[QUEEN+6]
             if sliders_orth:
                 attackers |= (get_sliding_attacks(to_sq, occupied, False) & sliders_orth)

        if depth >= 31: break

    # Backpropagation
    # scores array contains: [CaptureVal, Attacker1Val, Attacker2Val, ...]
    # Gain[0] = scores[0]
    # Gain[1] = scores[0] - scores[1] (Net gain for side 1)
    # Gain[2] = scores[0] - scores[1] + scores[2] (Net gain for side 0)
    #
    # Stockfish implementation:
    # int gains[32];
    # gains[0] = values[0];
    # for (int i=1; i<=d; ++i) gains[i] = values[i] - gains[i-1]; (Wait, this is prefix sum?)
    # Stockfish code: `gains[i] = values[i] - gains[i-1]` is equivalent to `values[i] - (values[i-1] - ...)`
    # essentially alternating sum.
    # Then `gains[i-1] = -max(-gains[i-1], gains[i])` minimax.

    # Let's implement Stockfish's array logic
    gains = np.zeros(32, dtype=np.int32)
    gains[0] = scores[0]
    for i in range(1, depth + 1):
        gains[i] = scores[i] - gains[i-1]

    for i in range(depth - 1, 0, -1):
        gains[i-1] = -max(-gains[i-1], gains[i])

    return gains[0]


@numba.njit(nbt.boolean(piece_bbs_signature, occupancy_bbs_signature, nbt.int64, nbt.uint16, nbt.int32), cache=True)
def see_ge(piece_bbs, occupancy_bbs, side_to_move, move, threshold):
    """
    SEE >= Threshold optimization.
    """
    from_sq = get_from_square(move)
    to_sq = get_to_square(move)
    special_flag = get_special_move_flag(move)

    attacker_type = -1
    offset = 0 if side_to_move == WHITE else 6
    for i in range(6):
        if piece_bbs[i + offset] & BB_SQUARES[from_sq]:
            attacker_type = i
            break
    if attacker_type == -1: return False

    victim_type = -1
    enemy_offset = 6 if side_to_move == WHITE else 0
    for i in range(6):
        if piece_bbs[i + enemy_offset] & BB_SQUARES[to_sq]:
            victim_type = i
            break

    if victim_type == -1:
        if special_flag == 2: # En Passant
            value = PIECE_VALUES[PAWN]
        else:
            value = 0
    else:
        value = PIECE_VALUES[victim_type]

    if special_flag == SPECIAL_MOVE_FLAG_PROMOTION:
        promo_type_encoded = get_promotion_piece(move)
        value += PIECE_VALUES[promo_type_encoded + 1] - PIECE_VALUES[PAWN]

    # First check
    balance = value - threshold
    if balance < 0: return False

    # Next attacker value
    if special_flag == SPECIAL_MOVE_FLAG_PROMOTION:
        promo_type_encoded = get_promotion_piece(move)
        attacker_val = PIECE_VALUES[promo_type_encoded + 1]
    else:
        attacker_val = PIECE_VALUES[attacker_type]

    balance = attacker_val - balance
    if balance <= 0: return True

    occupied = occupancy_bbs[2]
    occupied &= ~BB_SQUARES[from_sq]

    pinned_white = get_pinned_pieces(piece_bbs, occupancy_bbs, WHITE)
    pinned_black = get_pinned_pieces(piece_bbs, occupancy_bbs, BLACK)

    attackers = get_attackers_for_see(to_sq, occupied, piece_bbs, WHITE) | \
                get_attackers_for_see(to_sq, occupied, piece_bbs, BLACK)

    current_side = side_to_move

    while True:
        current_side = 1 - current_side
        attackers &= occupied

        stm_attackers = attackers & (piece_bbs[0] | piece_bbs[1] | piece_bbs[2] | piece_bbs[3] | piece_bbs[4] | piece_bbs[5]) \
                        if current_side == WHITE else \
                        attackers & (piece_bbs[6] | piece_bbs[7] | piece_bbs[8] | piece_bbs[9] | piece_bbs[10] | piece_bbs[11])

        pinned_mask = pinned_white if current_side == WHITE else pinned_black
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

        # X-Ray
        if type_idx in (PAWN, BISHOP, QUEEN, PAWN+6, BISHOP+6, QUEEN+6):
             sliders_diag = piece_bbs[BISHOP] | piece_bbs[QUEEN] | piece_bbs[BISHOP+6] | piece_bbs[QUEEN+6]
             if sliders_diag:
                 attackers |= (get_sliding_attacks(to_sq, occupied, True) & sliders_diag)

        if type_idx in (ROOK, QUEEN, ROOK+6, QUEEN+6):
             sliders_orth = piece_bbs[ROOK] | piece_bbs[QUEEN] | piece_bbs[ROOK+6] | piece_bbs[QUEEN+6]
             if sliders_orth:
                 attackers |= (get_sliding_attacks(to_sq, occupied, False) & sliders_orth)

        balance = val - balance
        if balance <= 0:
            return (current_side == side_to_move)

    return True
