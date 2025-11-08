# chess_engine/see.py
import numba
import numpy as np

from chess_engine.engine_types import (
    piece_bbs_signature,
    occupancy_bbs_signature,
    game_state_signature,
)
from chess_engine.constants import BB_SQUARES, MG_MATERIAL_VALUES
from chess_engine.move_generator import (
    get_bishop_attacks,
    get_rook_attacks,
    get_queen_attacks,
    KNIGHT_ATTACKS,
    KING_ATTACKS,
    NOT_A_FILE,
    NOT_H_FILE,
)
from chess_engine.zobrist import get_lsb_index
from chess_engine.bitboard_utils import find_piece_type_on_square


WHITE, BLACK = 0, 1
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 0, 1, 2, 3, 4, 5


@numba.njit(
    numba.types.Tuple((numba.int8, numba.int8))(
        piece_bbs_signature,
        occupancy_bbs_signature,
        numba.uint8,
        numba.uint8,
    ),
    cache=True,
)
def get_least_valuable_attacker(
    piece_bbs, occupancy_bbs, target_sq, attacking_side
):
    """
    Finds the least valuable attacker of a given square for a given side.
    Iterates through piece types from least to most valuable (P, N, B, R, Q, K).
    The first attacker found is guaranteed to be the least valuable.
    For sliding pieces, this function considers X-ray attacks.
    Returns:
        A tuple (from_square, piece_type index 0-11), or (-1, -1) if no attacker is found.
    """
    all_pieces_bb = occupancy_bbs[0] | occupancy_bbs[1]
    
    # For sliding piece attacks, we need to see "through" the piece on the
    # target square. We create an occupancy board where the target square is empty.
    xray_occupancy = all_pieces_bb & ~BB_SQUARES[target_sq]

    # --- 1. Check for Pawns ---
    if attacking_side == WHITE:
        pawn_bb = piece_bbs[PAWN]
        pawn_attackers = (((BB_SQUARES[target_sq] & NOT_A_FILE) >> 9) |
                          ((BB_SQUARES[target_sq] & NOT_H_FILE) >> 7))
        attacker = pawn_attackers & pawn_bb
        if attacker:
            return get_lsb_index(attacker), PAWN
    else:  # BLACK
        pawn_bb = piece_bbs[PAWN + 6]
        pawn_attackers = (((BB_SQUARES[target_sq] & NOT_H_FILE) << 9) |
                          ((BB_SQUARES[target_sq] & NOT_A_FILE) << 7))
        attacker = pawn_attackers & pawn_bb
        if attacker:
            return get_lsb_index(attacker), PAWN + 6

    # --- 2. Check for Knights ---
    knight_idx = KNIGHT + (6 if attacking_side == BLACK else 0)
    knight_bb = piece_bbs[knight_idx]
    knight_attackers = KNIGHT_ATTACKS[target_sq] & knight_bb
    if knight_attackers:
        return get_lsb_index(knight_attackers), knight_idx

    # --- 3. Check for Bishops ---
    bishop_idx = BISHOP + (6 if attacking_side == BLACK else 0)
    bishop_bb = piece_bbs[bishop_idx]
    bishop_attackers = get_bishop_attacks(target_sq, xray_occupancy) & bishop_bb
    if bishop_attackers:
        return get_lsb_index(bishop_attackers), bishop_idx

    # --- 4. Check for Rooks ---
    rook_idx = ROOK + (6 if attacking_side == BLACK else 0)
    rook_bb = piece_bbs[rook_idx]
    rook_attackers = get_rook_attacks(target_sq, xray_occupancy) & rook_bb
    if rook_attackers:
        return get_lsb_index(rook_attackers), rook_idx

    # --- 5. Check for Queens ---
    queen_idx = QUEEN + (6 if attacking_side == BLACK else 0)
    queen_bb = piece_bbs[queen_idx]
    queen_attackers = get_queen_attacks(target_sq, xray_occupancy) & queen_bb
    if queen_attackers:
        return get_lsb_index(queen_attackers), queen_idx

    # --- 6. Check for King ---
    king_idx = KING + (6 if attacking_side == BLACK else 0)
    king_bb = piece_bbs[king_idx]
    king_attackers = KING_ATTACKS[target_sq] & king_bb
    if king_attackers:
        return get_lsb_index(king_attackers), king_idx

    return -1, -1 # No attacker found


@numba.njit(
    numba.int32(
        piece_bbs_signature,
        occupancy_bbs_signature,
        game_state_signature,
        numba.uint8,
        numba.uint8,
    ),
    cache=True,
)
def see(piece_bbs, occupancy_bbs, game_state, from_sq, to_sq):
    """
    Static Exchange Evaluation (SEE).
    Calculates the score of a sequence of captures on a single square.
    Uses an iterative approach to avoid recursion overhead in Numba.
    """
    gain = np.zeros(32, dtype=np.int32)
    depth = 0
    side_to_move = game_state[0]

    # --- Initial move ---
    attacker_piece_type = find_piece_type_on_square(piece_bbs, from_sq)
    victim_piece_type = find_piece_type_on_square(piece_bbs, to_sq)

    # If victim is somehow not found (e.g., en passant), assume it's a pawn
    if victim_piece_type == -1:
         victim_piece_type = PAWN + (6 if side_to_move == WHITE else 0)

    # The first gain is the value of the piece being captured
    gain[depth] = MG_MATERIAL_VALUES[victim_piece_type % 6]

    # --- Simulate the capture for the rest of the sequence ---
    temp_piece_bbs = piece_bbs
    temp_occupancy_bbs = occupancy_bbs
    
    # Update piece bitboards
    from_bb = BB_SQUARES[from_sq]
    to_bb = BB_SQUARES[to_sq]
    
    temp_piece_bbs_list = list(temp_piece_bbs)
    temp_piece_bbs_list[attacker_piece_type] ^= (from_bb | to_bb)
    temp_piece_bbs_list[victim_piece_type] ^= to_bb
    temp_piece_bbs = (
        temp_piece_bbs_list[0], temp_piece_bbs_list[1], temp_piece_bbs_list[2],
        temp_piece_bbs_list[3], temp_piece_bbs_list[4], temp_piece_bbs_list[5],
        temp_piece_bbs_list[6], temp_piece_bbs_list[7], temp_piece_bbs_list[8],
        temp_piece_bbs_list[9], temp_piece_bbs_list[10], temp_piece_bbs_list[11]
    )
    
    # Update occupancy bitboards
    temp_occupancy_bbs_list = list(temp_occupancy_bbs)
    attacker_color = 0 if attacker_piece_type < 6 else 1
    victim_color = 0 if victim_piece_type < 6 else 1
    temp_occupancy_bbs_list[attacker_color] ^= (from_bb | to_bb)
    temp_occupancy_bbs_list[victim_color] ^= to_bb
    temp_occupancy_bbs_list[2] = temp_occupancy_bbs_list[0] | temp_occupancy_bbs_list[1]
    temp_occupancy_bbs = (
        temp_occupancy_bbs_list[0],
        temp_occupancy_bbs_list[1],
        temp_occupancy_bbs_list[2]
    )

    side_to_move = 1 - side_to_move

    # --- Iteratively find attackers ---
    while True:
        depth += 1
        
        aggressor_sq, aggressor_type = get_least_valuable_attacker(
            temp_piece_bbs, temp_occupancy_bbs, to_sq, side_to_move
        )

        if aggressor_sq == -1:
            break

        # The next gain is the current victim's value minus the previous gain
        victim_piece_type = find_piece_type_on_square(temp_piece_bbs, to_sq)
        gain[depth] = MG_MATERIAL_VALUES[victim_piece_type % 6] - gain[depth - 1]
        
        # Simulate this next capture
        from_bb = BB_SQUARES[aggressor_sq]
        to_bb = BB_SQUARES[to_sq]

        temp_piece_bbs_list = list(temp_piece_bbs)
        
        temp_piece_bbs_list[aggressor_type] ^= (from_bb | to_bb)
        temp_piece_bbs_list[victim_piece_type] ^= to_bb
        temp_piece_bbs = (
            temp_piece_bbs_list[0], temp_piece_bbs_list[1], temp_piece_bbs_list[2],
            temp_piece_bbs_list[3], temp_piece_bbs_list[4], temp_piece_bbs_list[5],
            temp_piece_bbs_list[6], temp_piece_bbs_list[7], temp_piece_bbs_list[8],
            temp_piece_bbs_list[9], temp_piece_bbs_list[10], temp_piece_bbs_list[11]
        )

        # Update occupancy bitboards
        temp_occupancy_bbs_list = list(temp_occupancy_bbs)
        aggressor_color = 0 if aggressor_type < 6 else 1
        victim_color = 0 if victim_piece_type < 6 else 1
        temp_occupancy_bbs_list[aggressor_color] ^= (from_bb | to_bb)
        temp_occupancy_bbs_list[victim_color] ^= to_bb
        temp_occupancy_bbs_list[2] = temp_occupancy_bbs_list[0] | temp_occupancy_bbs_list[1]
        temp_occupancy_bbs = (
            temp_occupancy_bbs_list[0],
            temp_occupancy_bbs_list[1],
            temp_occupancy_bbs_list[2]
        )
        
        side_to_move = 1 - side_to_move

    # --- Calculate SEE score from the gain array (minimax) ---
    while depth > 1:
        depth -= 1
        gain[depth - 1] = -max(-gain[depth - 1], gain[depth])
        
    return gain[0]
