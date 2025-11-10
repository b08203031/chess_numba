# chess_engine/see.py
import numba
import numpy as np

from chess_engine.engine_types import (
    piece_bbs_signature,
    occupancy_bbs_signature,
    game_state_signature,
)
from chess_engine.constants import BB_SQUARES, MG_MATERIAL_VALUES, NO_MOVE
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
from chess_engine.board_operations import make_move, unmake_move
from chess_engine.move import encode_move

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
    """
    all_pieces_bb = occupancy_bbs[0] | occupancy_bbs[1]
    xray_occupancy = all_pieces_bb & ~BB_SQUARES[target_sq]

    # --- Pawns ---
    pawn_idx = PAWN + (6 if attacking_side == BLACK else 0)
    pawn_bb = piece_bbs[pawn_idx]
    if attacking_side == WHITE:
        pawn_attackers = (((BB_SQUARES[target_sq] & NOT_A_FILE) >> 9) |
                          ((BB_SQUARES[target_sq] & NOT_H_FILE) >> 7))
    else:
        pawn_attackers = (((BB_SQUARES[target_sq] & NOT_H_FILE) << 9) |
                          ((BB_SQUARES[target_sq] & NOT_A_FILE) << 7))
    attacker = pawn_attackers & pawn_bb
    if attacker:
        return get_lsb_index(attacker), pawn_idx

    # --- Knights ---
    knight_idx = KNIGHT + (6 if attacking_side == BLACK else 0)
    knight_bb = piece_bbs[knight_idx]
    knight_attackers = KNIGHT_ATTACKS[target_sq] & knight_bb
    if knight_attackers:
        return get_lsb_index(knight_attackers), knight_idx

    # --- Bishops ---
    bishop_idx = BISHOP + (6 if attacking_side == BLACK else 0)
    bishop_bb = piece_bbs[bishop_idx]
    bishop_attackers = get_bishop_attacks(target_sq, xray_occupancy) & bishop_bb
    if bishop_attackers:
        return get_lsb_index(bishop_attackers), bishop_idx

    # --- Rooks ---
    rook_idx = ROOK + (6 if attacking_side == BLACK else 0)
    rook_bb = piece_bbs[rook_idx]
    rook_attackers = get_rook_attacks(target_sq, xray_occupancy) & rook_bb
    if rook_attackers:
        return get_lsb_index(rook_attackers), rook_idx

    # --- Queens ---
    queen_idx = QUEEN + (6 if attacking_side == BLACK else 0)
    queen_bb = piece_bbs[queen_idx]
    queen_attackers = get_queen_attacks(target_sq, xray_occupancy) & queen_bb
    if queen_attackers:
        return get_lsb_index(queen_attackers), queen_idx

    # --- King ---
    king_idx = KING + (6 if attacking_side == BLACK else 0)
    king_bb = piece_bbs[king_idx]
    king_attackers = KING_ATTACKS[target_sq] & king_bb
    if king_attackers:
        return get_lsb_index(king_attackers), king_idx

    return -1, -1


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
    Static Exchange Evaluation (SEE), refactored for a mutable board state.
    """
    gain = np.zeros(32, dtype=np.int32)
    depth = 0

    # --- Make copies of the board state to simulate on ---
    p_bbs_copy = piece_bbs.copy()
    o_bbs_copy = occupancy_bbs.copy()
    g_state_copy = game_state.copy()

    # --- Initial move ---
    side_to_move = g_state_copy[0]
    victim_piece_type = find_piece_type_on_square(p_bbs_copy, to_sq)
    if victim_piece_type == -1: # En-passant case
        victim_piece_type = PAWN + (6 if side_to_move == WHITE else 0)
    
    gain[depth] = MG_MATERIAL_VALUES[victim_piece_type % 6]
    
    # Simulate the initial capture
    initial_move = encode_move(from_sq, to_sq, 0, 0)
    unmake_info = make_move(p_bbs_copy, o_bbs_copy, g_state_copy, initial_move)

    # --- Iteratively find attackers ---
    while True:
        depth += 1
        side_to_move = g_state_copy[0]
        
        aggressor_sq, aggressor_type = get_least_valuable_attacker(
            p_bbs_copy, o_bbs_copy, to_sq, side_to_move
        )

        if aggressor_sq == -1:
            break

        gain[depth] = MG_MATERIAL_VALUES[aggressor_type % 6] - gain[depth - 1]
        
        # Simulate this next capture
        move = encode_move(aggressor_sq, to_sq, 0, 0)
        unmake_info = make_move(p_bbs_copy, o_bbs_copy, g_state_copy, move)
        
    # --- Calculate SEE score from the gain array (minimax) ---
    while depth > 1:
        depth -= 1
        gain[depth - 1] = -max(-gain[depth - 1], gain[depth])
        
    return gain[0]
