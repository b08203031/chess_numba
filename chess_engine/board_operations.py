# chess_engine/board_operations.py

import numpy as np
import numba
from chess_engine.move import (
    get_from_square, get_to_square, get_special_move_flag, get_promotion_piece,
    SPECIAL_MOVE_FLAG_PROMOTION, SPECIAL_MOVE_FLAG_EN_PASSANT, SPECIAL_MOVE_FLAG_CASTLING
)
from chess_engine.zobrist import (
    PIECE_SQUARE_KEYS, SIDE_TO_MOVE_KEY, EN_PASSANT_FILE_KEYS, CASTLING_RIGHTS_KEYS
)
from chess_engine.engine_types import (
    piece_bbs_signature, occupancy_bbs_signature, game_state_signature, unmake_info_signature
)

# --- Piece Type Constants ---
PAWN, ROOK, KING = 0, 3, 5
WHITE, BLACK = 0, 1

# --- Castling Rights Constants ---
WK_CASTLE, WQ_CASTLE, BK_CASTLE, BQ_CASTLE = 1, 2, 4, 8
ALL_CASTLING_RIGHTS = 15

# --- Pre-computed Castling Update Masks ---
CASTLING_UPDATE_MASK = np.full(64, ALL_CASTLING_RIGHTS, dtype=np.uint64)
CASTLING_UPDATE_MASK[0]  -= WQ_CASTLE
CASTLING_UPDATE_MASK[4]  -= (WK_CASTLE | WQ_CASTLE)
CASTLING_UPDATE_MASK[7]  -= WK_CASTLE
CASTLING_UPDATE_MASK[56] -= BQ_CASTLE
CASTLING_UPDATE_MASK[60] -= (BK_CASTLE | BQ_CASTLE)
CASTLING_UPDATE_MASK[63] -= BK_CASTLE

@numba.jit(numba.int8(piece_bbs_signature, numba.uint8, numba.uint8), nopython=True, inline='always')
def find_piece_type_for_square(piece_bbs: np.ndarray, square: int, color: int) -> int:
    """Finds which piece type occupies a given square."""
    bit = np.uint64(1) << square
    start_index = color * 6
    for i in range(6):
        if piece_bbs[start_index + i] & bit:
            return i
    return -1

@numba.jit(unmake_info_signature(piece_bbs_signature, occupancy_bbs_signature, game_state_signature, numba.uint16), nopython=True, cache=True, boundscheck=False, fastmath=True)
def make_move(piece_bbs: np.ndarray, occupancy_bbs: np.ndarray, game_state: np.ndarray, move: np.uint16):
    """
    Applies a move by MUTATING the board state arrays in-place.
    Returns unmake_info needed to reverse the move.
    """
    side = game_state[0]
    current_castling_rights = game_state[1]
    current_ep_square = game_state[2]
    current_halfmove_clock = game_state[3]
    zobrist_key = game_state[4]

    original_zobrist_key = zobrist_key
    key = zobrist_key

    from_sq = get_from_square(move)
    to_sq = get_to_square(move)
    flag = get_special_move_flag(move)

    moving_piece_type = find_piece_type_for_square(piece_bbs, from_sq, side)
    moving_piece_bb_idx = side * 6 + moving_piece_type

    captured_piece_type = np.int8(-1)

    # --- Piece Movement ---
    move_mask = (np.uint64(1) << from_sq) | (np.uint64(1) << to_sq)
    key ^= PIECE_SQUARE_KEYS[moving_piece_bb_idx, from_sq]
    key ^= PIECE_SQUARE_KEYS[moving_piece_bb_idx, to_sq]
    piece_bbs[moving_piece_bb_idx] ^= move_mask
    occupancy_bbs[side] ^= move_mask

    is_capture = False
    # --- Standard Capture ---
    if occupancy_bbs[1 - side] & (np.uint64(1) << to_sq):
        is_capture = True
        opponent_color = 1 - side
        captured_piece_type = find_piece_type_for_square(piece_bbs, to_sq, opponent_color)
        captured_piece_bb_idx = opponent_color * 6 + captured_piece_type

        capture_mask = np.uint64(1) << to_sq
        piece_bbs[captured_piece_bb_idx] &= ~capture_mask
        occupancy_bbs[opponent_color] &= ~capture_mask
        key ^= PIECE_SQUARE_KEYS[captured_piece_bb_idx, to_sq]

    # --- Handle Special Moves ---
    if flag == SPECIAL_MOVE_FLAG_PROMOTION:
        promo_piece_type = get_promotion_piece(move) + 1
        promo_piece_bb_idx = side * 6 + promo_piece_type

        piece_bbs[moving_piece_bb_idx] &= ~(np.uint64(1) << to_sq)
        piece_bbs[promo_piece_bb_idx] |= (np.uint64(1) << to_sq)

        key ^= PIECE_SQUARE_KEYS[moving_piece_bb_idx, to_sq]
        key ^= PIECE_SQUARE_KEYS[promo_piece_bb_idx, to_sq]

    elif flag == SPECIAL_MOVE_FLAG_EN_PASSANT:
        is_capture = True
        captured_piece_type = np.int8(PAWN)
        opponent_color = 1 - side
        captured_pawn_bb_idx = opponent_color * 6 + PAWN
        captured_pawn_sq = to_sq + (8 if side == BLACK else -8)
        capture_mask = np.uint64(1) << captured_pawn_sq

        piece_bbs[captured_pawn_bb_idx] &= ~capture_mask
        occupancy_bbs[opponent_color] &= ~capture_mask
        key ^= PIECE_SQUARE_KEYS[captured_pawn_bb_idx, captured_pawn_sq]

    elif flag == SPECIAL_MOVE_FLAG_CASTLING:
        king_side_castle = to_sq > from_sq
        rook_from_sq, rook_to_sq = ((7, 5) if king_side_castle else (0, 3)) if side == WHITE else ((63, 61) if king_side_castle else (56, 59))
        rook_bb_idx = side * 6 + ROOK
        rook_move_mask = (np.uint64(1) << rook_from_sq) | (np.uint64(1) << rook_to_sq)

        piece_bbs[rook_bb_idx] ^= rook_move_mask
        occupancy_bbs[side] ^= rook_move_mask
        key ^= PIECE_SQUARE_KEYS[rook_bb_idx, rook_from_sq]
        key ^= PIECE_SQUARE_KEYS[rook_bb_idx, rook_to_sq]

    # --- Recalculate Combined Occupancy (Robust Fix) ---
    occupancy_bbs[2] = occupancy_bbs[0] | occupancy_bbs[1]

    # --- Update Game State (Castling, EP, etc.) ---
    new_castling_rights = current_castling_rights & CASTLING_UPDATE_MASK[from_sq] & CASTLING_UPDATE_MASK[to_sq]
    if new_castling_rights != current_castling_rights:
        key ^= CASTLING_RIGHTS_KEYS[current_castling_rights]
        key ^= CASTLING_RIGHTS_KEYS[new_castling_rights]

    new_ep_square = np.uint64(64)
    if moving_piece_type == PAWN:
        if side == WHITE and (to_sq - from_sq) == 16:
            new_ep_square = np.uint64(from_sq + 8)
        elif side == BLACK and (from_sq - to_sq) == 16:
            new_ep_square = np.uint64(from_sq - 8)

    if new_ep_square != current_ep_square:
        if current_ep_square != 64:
            key ^= EN_PASSANT_FILE_KEYS[np.uint64(current_ep_square) % 8]
        if new_ep_square != 64:
            key ^= EN_PASSANT_FILE_KEYS[new_ep_square % 8]

    key ^= SIDE_TO_MOVE_KEY
    new_halfmove_clock = np.uint64(0) if (moving_piece_type == PAWN or is_capture) else current_halfmove_clock + 1

    game_state[0] = 1 - side
    game_state[1] = new_castling_rights
    game_state[2] = new_ep_square
    game_state[3] = new_halfmove_clock
    game_state[4] = key

    return (
        captured_piece_type,
        np.uint8(current_castling_rights),
        np.uint8(current_ep_square),
        np.uint8(current_halfmove_clock),
        original_zobrist_key
    )

@numba.jit(numba.void(
    piece_bbs_signature, occupancy_bbs_signature, game_state_signature, numba.uint16, unmake_info_signature
), nopython=True, cache=True, boundscheck=False, fastmath=True)
def unmake_move(piece_bbs: np.ndarray, occupancy_bbs: np.ndarray, game_state: np.ndarray, move: np.uint16, unmake_info: tuple):
    """
    Reverts a move by MUTATING the board state arrays in-place.
    """
    captured_piece_type, old_castling_rights, old_ep_square, old_halfmove_clock, old_zobrist_key = unmake_info

    side = 1 - game_state[0]

    from_sq = get_from_square(move)
    to_sq = get_to_square(move)
    flag = get_special_move_flag(move)

    moving_piece_type = find_piece_type_for_square(piece_bbs, to_sq, side)
    if flag == SPECIAL_MOVE_FLAG_PROMOTION:
        moving_piece_type = PAWN

    moving_piece_bb_idx = side * 6 + moving_piece_type

    # --- Restore Piece Movement ---
    move_mask = (np.uint64(1) << from_sq) | (np.uint64(1) << to_sq)
    # For promotions, the standard XOR trick is incorrect because the piece type changes.
    # It is handled entirely within the promotion block below.
    if flag != SPECIAL_MOVE_FLAG_PROMOTION:
        piece_bbs[moving_piece_bb_idx] ^= move_mask
        occupancy_bbs[side] ^= move_mask

    # --- Restore Special Moves ---
    if flag == SPECIAL_MOVE_FLAG_PROMOTION:
        promo_piece_type = get_promotion_piece(move) + 1
        promo_piece_bb_idx = side * 6 + promo_piece_type

        # Manually reverse the promotion:
        # 1. Remove the promoted piece from the to_square.
        promo_mask = np.uint64(1) << to_sq
        piece_bbs[promo_piece_bb_idx] &= ~promo_mask

        # 2. Place the pawn back on the from_square.
        pawn_mask = np.uint64(1) << from_sq
        piece_bbs[moving_piece_bb_idx] |= pawn_mask

        # 3. Update occupancy accordingly.
        occupancy_bbs[side] &= ~promo_mask  # The bit for the 'to' square is removed
        occupancy_bbs[side] |= pawn_mask   # The bit for the 'from' square is added

    elif flag == SPECIAL_MOVE_FLAG_EN_PASSANT:
        opponent_color = 1 - side
        captured_pawn_bb_idx = opponent_color * 6 + PAWN
        captured_pawn_sq = to_sq + (8 if side == BLACK else -8)
        capture_mask = np.uint64(1) << captured_pawn_sq

        piece_bbs[captured_pawn_bb_idx] |= capture_mask
        occupancy_bbs[opponent_color] |= capture_mask

    elif flag == SPECIAL_MOVE_FLAG_CASTLING:
        king_side_castle = to_sq > from_sq
        rook_from_sq, rook_to_sq = ((7, 5) if king_side_castle else (0, 3)) if side == WHITE else ((63, 61) if king_side_castle else (56, 59))
        rook_bb_idx = side * 6 + ROOK
        rook_move_mask = (np.uint64(1) << rook_from_sq) | (np.uint64(1) << rook_to_sq)

        piece_bbs[rook_bb_idx] ^= rook_move_mask
        occupancy_bbs[side] ^= rook_move_mask

    # --- Restore Captured Piece ---
    if captured_piece_type != -1 and flag != SPECIAL_MOVE_FLAG_EN_PASSANT:
        opponent_color = 1 - side
        captured_piece_bb_idx = opponent_color * 6 + captured_piece_type
        capture_mask = np.uint64(1) << to_sq

        piece_bbs[captured_piece_bb_idx] |= capture_mask
        occupancy_bbs[opponent_color] |= capture_mask

    # --- Recalculate Combined Occupancy (Robust Fix) ---
    occupancy_bbs[2] = occupancy_bbs[0] | occupancy_bbs[1]

    # --- Restore Game State ---
    game_state[0] = side
    game_state[1] = old_castling_rights
    game_state[2] = old_ep_square
    game_state[3] = old_halfmove_clock
    game_state[4] = old_zobrist_key

@numba.jit(numba.void(game_state_signature), nopython=True, cache=True, boundscheck=False, fastmath=True)
def make_null_move(game_state: np.ndarray):
    """
    Performs a null move in-place: flips side to move, clears EP, increments clock.
    """
    side_to_move = game_state[0]
    en_passant_square = game_state[2]
    zobrist_key = game_state[4]

    new_key = zobrist_key ^ SIDE_TO_MOVE_KEY
    if en_passant_square != 64:
        new_key ^= EN_PASSANT_FILE_KEYS[en_passant_square % 8]

    game_state[0] = 1 - side_to_move
    game_state[2] = 64
    game_state[3] += 1
    game_state[4] = new_key
