# chess_engine/board_operations.py

import numba
import numpy as np
from chess_engine.move import (
    get_from_square, get_to_square, get_move_flag, get_promotion_piece,
    FLAG_NORMAL, FLAG_PROMOTION, FLAG_EN_PASSANT, FLAG_CASTLING,
    PROMO_KNIGHT, PROMO_BISHOP, PROMO_ROOK, PROMO_QUEEN
)

# Piece type constants
WHITE_PAWN, WHITE_KNIGHT, WHITE_BISHOP, WHITE_ROOK, WHITE_QUEEN, WHITE_KING = 0, 1, 2, 3, 4, 5
BLACK_PAWN, BLACK_KNIGHT, BLACK_BISHOP, BLACK_ROOK, BLACK_QUEEN, BLACK_KING = 6, 7, 8, 9, 10, 11

# Castling Rights Constants
WK_CASTLE, WQ_CASTLE, BK_CASTLE, BQ_CASTLE = 1, 2, 4, 8

@numba.jit(nopython=True)
def make_move(board_state, move):
    # 1. Unpack the board state tuple
    (
        wp_bb, wn_bb, wb_bb, wr_bb, wq_bb, wk_bb,
        bp_bb, bn_bb, bb_bb, br_bb, bq_bb, bk_bb,
        white_pieces_bb, black_pieces_bb, all_pieces_bb,
        castling_rights, en_passant_square, side_to_move, halfmove_clock
    ) = board_state

    # Create mutable copies of the bitboards
    piece_bbs = [
        wp_bb, wn_bb, wb_bb, wr_bb, wq_bb, wk_bb,
        bp_bb, bn_bb, bb_bb, br_bb, bq_bb, bk_bb
    ]

    # 2. Decode the move and initialize variables
    from_sq = get_from_square(move)
    to_sq = get_to_square(move)
    flag = get_move_flag(move)
    from_bb = np.uint64(1) << from_sq
    to_bb = np.uint64(1) << to_sq
    move_bb = from_bb | to_bb
    is_capture = False

    # 3. Identify moving piece
    moving_piece_type = -1
    for i in range(12):
        if (piece_bbs[i] & from_bb):
            moving_piece_type = i
            break

    # Reset halfmove clock if a pawn is moved
    new_halfmove_clock = halfmove_clock + 1
    if moving_piece_type == WHITE_PAWN or moving_piece_type == BLACK_PAWN:
        new_halfmove_clock = 0

    # Move the piece
    piece_bbs[moving_piece_type] ^= move_bb

    # 4. Handle captures
    if (all_pieces_bb & to_bb):
        if flag != FLAG_EN_PASSANT:
            is_capture = True
            opponent_start_idx = 6 if side_to_move == 0 else 0
            for i in range(6):
                piece_idx = opponent_start_idx + i
                if (piece_bbs[piece_idx] & to_bb):
                    piece_bbs[piece_idx] ^= to_bb # Remove captured piece
                    break

    if is_capture:
        new_halfmove_clock = 0

    # 5. Handle special move types
    if flag == FLAG_PROMOTION:
        # Remove the pawn and add the promoted piece
        pawn_idx = WHITE_PAWN if side_to_move == 0 else BLACK_PAWN
        piece_bbs[pawn_idx] ^= to_bb
        promo_piece_code = get_promotion_piece(move)
        promo_piece_offset = 1 # Knight
        if promo_piece_code == PROMO_BISHOP: promo_piece_offset = 2
        elif promo_piece_code == PROMO_ROOK: promo_piece_offset = 3
        elif promo_piece_code == PROMO_QUEEN: promo_piece_offset = 4
        base_piece_idx = WHITE_KNIGHT if side_to_move == 0 else BLACK_KNIGHT
        piece_bbs[base_piece_idx + promo_piece_offset - 1] |= to_bb

    elif flag == FLAG_EN_PASSANT:
        new_halfmove_clock = 0
        pawn_to_remove_sq = (en_passant_square - 8) if side_to_move == 0 else (en_passant_square + 8)
        pawn_to_remove_bb = np.uint64(1) << pawn_to_remove_sq
        captured_pawn_idx = BLACK_PAWN if side_to_move == 0 else WHITE_PAWN
        piece_bbs[captured_pawn_idx] ^= pawn_to_remove_bb

    elif flag == FLAG_CASTLING:
        if to_sq == 6: # White kingside (g1)
            piece_bbs[WHITE_ROOK] ^= (np.uint64(1) << 7 | np.uint64(1) << 5) # h1 -> f1
        elif to_sq == 2: # White queenside (c1)
            piece_bbs[WHITE_ROOK] ^= (np.uint64(1) << 0 | np.uint64(1) << 3) # a1 -> d1
        elif to_sq == 62: # Black kingside (g8)
            piece_bbs[BLACK_ROOK] ^= (np.uint64(1) << 63 | np.uint64(1) << 61) # h8 -> f8
        elif to_sq == 58: # Black queenside (c8)
            piece_bbs[BLACK_ROOK] ^= (np.uint64(1) << 56 | np.uint64(1) << 59) # a8 -> d8

    # 6. Update En Passant square
    new_en_passant_square = -1
    if moving_piece_type == WHITE_PAWN and (to_sq - from_sq == 16):
        new_en_passant_square = from_sq + 8
    elif moving_piece_type == BLACK_PAWN and (from_sq - to_sq == 16):
        new_en_passant_square = from_sq - 8

    # 7. Update Castling Rights
    new_castling_rights = castling_rights
    # King moves
    if moving_piece_type == WHITE_KING: new_castling_rights &= ~(WK_CASTLE | WQ_CASTLE)
    if moving_piece_type == BLACK_KING: new_castling_rights &= ~(BK_CASTLE | BQ_CASTLE)
    # Rook moves or is captured
    if from_sq == 0 or to_sq == 0: new_castling_rights &= ~WQ_CASTLE # a1
    if from_sq == 7 or to_sq == 7: new_castling_rights &= ~WK_CASTLE # h1
    if from_sq == 56 or to_sq == 56: new_castling_rights &= ~BQ_CASTLE # a8
    if from_sq == 63 or to_sq == 63: new_castling_rights &= ~BK_CASTLE # h8

    # 8. Recalculate occupancy bitboards
    new_white_pieces_bb = piece_bbs[0] | piece_bbs[1] | piece_bbs[2] | piece_bbs[3] | piece_bbs[4] | piece_bbs[5]
    new_black_pieces_bb = piece_bbs[6] | piece_bbs[7] | piece_bbs[8] | piece_bbs[9] | piece_bbs[10] | piece_bbs[11]
    new_all_pieces_bb = new_white_pieces_bb | new_black_pieces_bb

    # 9. Pack and return the new state
    return (
        piece_bbs[0], piece_bbs[1], piece_bbs[2], piece_bbs[3], piece_bbs[4], piece_bbs[5],
        piece_bbs[6], piece_bbs[7], piece_bbs[8], piece_bbs[9], piece_bbs[10], piece_bbs[11],
        new_white_pieces_bb, new_black_pieces_bb, new_all_pieces_bb,
        new_castling_rights,
        np.int8(new_en_passant_square),
        np.uint8(1 - side_to_move),
        np.uint8(new_halfmove_clock)
    )
