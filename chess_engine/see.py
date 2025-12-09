
import numba
import numpy as np
from chess_engine.constants import (
    BB_SQUARES, DE_BRUIJN_SEQUENCE, DE_BRUIJN_INDEX
)
from chess_engine.engine_types import piece_bbs_signature, occupancy_bbs_signature
from chess_engine.bitboard_utils import find_piece_type_on_square
import numba.types as nbt

# Piece Values for SEE (based on Stockfish's internal values for SEE)
# P=100, N=320, B=330, R=500, Q=900, K=20000
# Note: King is given a very high value to prevent it from being "captured" in simulation logic
PIECE_VALUES = np.array([100, 320, 330, 500, 900, 20000], dtype=np.int32)

@numba.njit(nbt.int32(nbt.uint64), cache=True)
def get_lsb_index(bitboard):
    """Returns the index (0-63) of the least significant bit set."""
    if bitboard == 0:
        return -1
    return DE_BRUIJN_INDEX[((bitboard ^ (bitboard - 1)) * DE_BRUIJN_SEQUENCE) >> np.uint64(58)]

# Basic Attack Tables (Initialize once if possible, or use logic)
# Numba caches function compilation, so we can embed logic.

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
            # Check for wrap-around
            if abs(tr - sq_rank) <= 2 and abs(tf - sq_file) <= 2:
                attacks |= BB_SQUARES[target]
    return attacks

@numba.njit(cache=True)
def get_sliding_attacks(square, occupied, is_diagonal):
    # Hyperbola Quintessence or similar would be fast, but simple ray casting for now
    # to avoid complex dependencies. Stockfish uses Magic Bitboards.
    # We will use a simplified loop for Numba.
    attacks = np.uint64(0)

    if is_diagonal:
        directions = DIAGONAL_DIRECTIONS
    else:
        directions = ORTHOGONAL_DIRECTIONS

    for d in directions:
        curr = square
        while True:
            # Check file wrap
            curr_file = curr % 8
            if d == 1 and curr_file == 7: break
            if d == -1 and curr_file == 0: break
            if d == 9 and curr_file == 7: break # Up-Right
            if d == -7 and curr_file == 7: break # Down-Right
            if d == 7 and curr_file == 0: break # Up-Left
            if d == -9 and curr_file == 0: break # Down-Left

            curr += d
            if not (0 <= curr < 64):
                break

            attacks |= BB_SQUARES[curr]
            if (occupied & BB_SQUARES[curr]) != 0:
                break
    return attacks

@numba.njit(cache=True)
def get_all_attackers(square, occupied, piece_bbs, side_mask):
    """
    Get bitboard of all pieces of 'side_mask' (0=White, 1=Black) attacking 'square'.
    """
    attackers = np.uint64(0)

    # Pawn Attacks (from inverse perspective)
    # If checking for White attackers, we check Black pawn attack patterns from square
    pawn_attacks = np.uint64(0)
    if side_mask == 0: # White attackers
        # Target square attacked by White Pawn if White Pawn is on (sq-7) or (sq-9)
        # This is equivalent to Black Pawn attacks from square
        # White P on A1 attacks B2. B2(9) - 9 = 0(A1).
        if square >= 9 and (square % 8) > 0: # Attacker at sq-9 (Left-Down)
             if (piece_bbs[0] & BB_SQUARES[square - 9]): attackers |= BB_SQUARES[square - 9]
        if square >= 7 and (square % 8) < 7: # Attacker at sq-7 (Right-Down)
             if (piece_bbs[0] & BB_SQUARES[square - 7]): attackers |= BB_SQUARES[square - 7]

        offset = 0
    else: # Black attackers
        # Black P on A8 attacks B7. B7(54) + 9? No.
        # Black P moves down (-8). Attacks -7, -9.
        # Target sq(48, A7). Attacker B8(57)? 57-9=48.
        if square <= 54 and (square % 8) < 7: # Attacker at sq+9 (Right-Up)
             if (piece_bbs[6] & BB_SQUARES[square + 9]): attackers |= BB_SQUARES[square + 9]
        if square <= 56 and (square % 8) > 0: # Attacker at sq+7 (Left-Up)
             if (piece_bbs[6] & BB_SQUARES[square + 7]): attackers |= BB_SQUARES[square + 7]

        offset = 6

    # Knights
    knights = piece_bbs[1 + offset]
    if knights:
        attackers |= (get_step_attacks(square, KNIGHT_OFFSETS) & knights)

    # Bishops + Queens (Diagonal)
    sliders_diag = piece_bbs[2 + offset] | piece_bbs[4 + offset]
    if sliders_diag:
        attackers |= (get_sliding_attacks(square, occupied, True) & sliders_diag)

    # Rooks + Queens (Straight)
    sliders_orth = piece_bbs[3 + offset] | piece_bbs[4 + offset]
    if sliders_orth:
        attackers |= (get_sliding_attacks(square, occupied, False) & sliders_orth)

    # King
    kings = piece_bbs[5 + offset]
    if kings:
        attackers |= (get_step_attacks(square, KING_OFFSETS) & kings)

    return attackers


@numba.njit(cache=True)
def get_least_valuable_attacker(attackers, piece_bbs, side_mask):
    """
    Finds the LVA from the attackers bitboard.
    Returns (piece_value, piece_bitboard, type_index)
    """
    offset = 0 if side_mask == 0 else 6

    # Order: P, N, B, R, Q, K
    for i in range(6):
        subset = attackers & piece_bbs[i + offset]
        if subset:
            # Found least valuable piece type
            sq_bb = subset & -subset # Extract LSB
            return PIECE_VALUES[i], sq_bb, i # Value, BB, Type Index (0-5)

    return 0, np.uint64(0), -1


@numba.njit(nbt.int32(piece_bbs_signature, occupancy_bbs_signature, nbt.int64, nbt.int64, nbt.int64), cache=True)
def see(piece_bbs, occupancy_bbs, side_to_move, from_sq, to_sq):
    """
    Static Exchange Evaluation (SEE).
    Returns the score in centipawns.
    """

    # 1. Initial Capture
    # Value of piece being captured
    victim_type = find_piece_type_on_square(piece_bbs, to_sq)
    attacker_type = find_piece_type_on_square(piece_bbs, from_sq)

    if victim_type == -1:
        # En Passant logic: If attacker is Pawn and moves diagonally to empty square
        if (attacker_type == 0 or attacker_type == 6) and abs((from_sq % 8) - (to_sq % 8)) == 1:
            value = PIECE_VALUES[0]
        else:
            value = 0
    else:
        value = PIECE_VALUES[victim_type % 6]

    # Initial swap
    if attacker_type == -1: return 0

    attacker_value = PIECE_VALUES[attacker_type % 6]

    scores = np.zeros(32, dtype=np.int32)
    scores[0] = value

    # Occupied board
    occupied = occupancy_bbs[2]

    # Remove start piece
    current_attacker_sq_bb = BB_SQUARES[from_sq]
    current_attacker_val = attacker_value
    current_side = side_to_move
    occupied &= ~current_attacker_sq_bb

    # Initial Attackers
    attackers = get_all_attackers(to_sq, occupied, piece_bbs, 0) | \
                get_all_attackers(to_sq, occupied, piece_bbs, 1)

    depth = 1

    while True:
        # Next side
        current_side = 1 - current_side

        # Value of the piece that just captured (the previous attacker)
        scores[depth] = current_attacker_val - scores[depth - 1]

        # Filter attackers by current_side

        # --- Pin Check Logic (Simplified Ray-Trace) ---
        valid_attacker_found = False
        next_attacker_val = 0
        next_attacker_bb = np.uint64(0)

        while True:
            attackers &= occupied

            val, bb, type_idx = get_least_valuable_attacker(attackers, piece_bbs, current_side)
            if bb == 0:
                break

            # Check Pin: If attacker is removed, is King in check?
            attacker_sq = get_lsb_index(bb)
            potential_occ = occupied & ~bb
            king_idx = 5 if current_side == 0 else 11
            king_sq = get_lsb_index(piece_bbs[king_idx])

            is_pinned = False

            opp_offset = 0 if current_side == 1 else 6
            opp_sliders_diag = piece_bbs[2 + opp_offset] | piece_bbs[4 + opp_offset]
            opp_sliders_orth = piece_bbs[3 + opp_offset] | piece_bbs[4 + opp_offset]

            # Diagonal
            if (get_sliding_attacks(king_sq, potential_occ, True) & opp_sliders_diag):
                is_pinned = True
            # Orthogonal
            elif (get_sliding_attacks(king_sq, potential_occ, False) & opp_sliders_orth):
                is_pinned = True

            if not is_pinned:
                valid_attacker_found = True
                next_attacker_val = val
                next_attacker_bb = bb
                break
            else:
                attackers &= ~bb

        if not valid_attacker_found:
            break

        current_attacker_val = next_attacker_val
        current_attacker_sq_bb = next_attacker_bb

        occupied &= ~current_attacker_sq_bb

        # Add X-Ray attackers
        sliders_all_diag = piece_bbs[2] | piece_bbs[4] | piece_bbs[8] | piece_bbs[10]
        sliders_all_orth = piece_bbs[3] | piece_bbs[4] | piece_bbs[9] | piece_bbs[10]

        if sliders_all_diag:
            attackers |= (get_sliding_attacks(to_sq, occupied, True) & sliders_all_diag)
        if sliders_all_orth:
            attackers |= (get_sliding_attacks(to_sq, occupied, False) & sliders_all_orth)

        depth += 1
        if depth >= 32: break

    # Propagate scores back (Minimax)
    # Start from end of chain.
    # scores[i] is the gain for side moving at i.
    # They choose max(scores[i], -scores[i+1]).
    # We update scores[i] in place.
    # Root node (0) is FORCED.
    # So we stop loop at 1.

    while depth > 2:
        depth -= 1
        scores[depth-1] = max(scores[depth-1], -scores[depth])

    if depth == 2:
        # At depth 1 (Opponent's first choice).
        # Opponent compares scores[1] (Stop) vs -scores[2] (Continue).
        # scores[1] is already updated from loop if depth was > 2.
        # Opponent outcome is scores[1].
        # Our outcome is scores[0] - scores[1].
        # Wait, if scores[1] is opponent's GAIN.
        # scores[0] is OUR gain from first capture.
        # Total = scores[0] - scores[1].
        return scores[0] - scores[1]

    return scores[0]
