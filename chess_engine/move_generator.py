# chess_engine/move_generator.py
import numpy
import numba as nb

from chess_engine.move import (
    encode_move, SPECIAL_MOVE_FLAG_NORMAL, SPECIAL_MOVE_FLAG_PROMOTION, SPECIAL_MOVE_FLAG_EN_PASSANT, SPECIAL_MOVE_FLAG_CASTLING,
    PROMO_QUEEN, PROMO_ROOK, PROMO_BISHOP, PROMO_KNIGHT
)
from chess_engine.board_operations import make_move
from chess_engine.engine_types import board_state_flat_signature
import numba.types as nbt

# =============================================================================
# Constants and Pre-computation
# =============================================================================
NOT_A_FILE = ~numpy.uint64(0x0101010101010101)
NOT_H_FILE = ~numpy.uint64(0x8080808080808080)
NOT_AB_FILE = ~numpy.uint64(0x0303030303030303)
NOT_GH_FILE = ~numpy.uint64(0xC0C0C0C0C0C0C0C0)
RANK_1 = numpy.uint64(0xFF)
RANK_2 = numpy.uint64(0xFF00)
RANK_3 = numpy.uint64(0xFF0000)
RANK_4 = numpy.uint64(0xFF000000)
RANK_5 = numpy.uint64(0xFF00000000)
RANK_6 = numpy.uint64(0xFF0000000000)
RANK_7 = numpy.uint64(0xFF000000000000)
RANK_8 = numpy.uint64(0xFF00000000000000)
WHITE, BLACK = 0, 1
BB_SQUARES = numpy.array([numpy.uint64(1) << i for i in range(64)], dtype=numpy.uint64)
EMPTY = numpy.uint64(0)

# =============================================================================
# Magic Bitboard Constants (Commented Out)
# =============================================================================
# BISHOP_RELEVANT_BITS = numpy.array([...])
# ROOK_RELEVANT_BITS = numpy.array([...])
# BISHOP_MAGIC_NUMBERS = numpy.array([...])
# ROOK_MAGIC_NUMBERS = numpy.array([...])

# =============================================================================
# Helper Functions
# =============================================================================

@nb.njit(nb.int32(nb.uint64), cache=True)
def count_bits(bb):
    c=0
    while bb > 0:
        bb &= (bb - numpy.uint64(1))
        c += 1
    return c

@nb.njit(nb.uint8(nb.uint64), cache=True)
def get_ls1b_index(bb):
    return count_bits((bb & -bb) - numpy.uint64(1))

# =============================================================================
# Attack Generation
# =============================================================================

# @nb.njit(nb.uint64(nb.uint8), cache=True)
# def mask_bishop_attacks(sq):
#     ...

# @nb.njit(nb.uint64(nb.uint8), cache=True)
# def mask_rook_attacks(sq):
#     ...

@nb.njit(nb.uint64(nb.uint8, nb.uint64), cache=True)
def bishop_attacks_on_the_fly(sq, block):
    attacks, tr, tf = EMPTY, sq // 8, sq % 8
    for r, f in zip(range(tr + 1, 8), range(tf + 1, 8)):
        attacks |= BB_SQUARES[r*8+f];
        if BB_SQUARES[r*8+f] & block: break
    for r, f in zip(range(tr - 1, -1, -1), range(tf + 1, 8)):
        attacks |= BB_SQUARES[r*8+f]
        if BB_SQUARES[r*8+f] & block: break
    for r, f in zip(range(tr + 1, 8), range(tf - 1, -1, -1)):
        attacks |= BB_SQUARES[r*8+f]
        if BB_SQUARES[r*8+f] & block: break
    for r, f in zip(range(tr - 1, -1, -1), range(tf - 1, -1, -1)):
        attacks |= BB_SQUARES[r*8+f]
        if BB_SQUARES[r*8+f] & block: break
    return attacks

@nb.njit(nb.uint64(nb.uint8, nb.uint64), cache=True)
def rook_attacks_on_the_fly(sq, block):
    attacks, tr, tf = EMPTY, sq // 8, sq % 8
    for r in range(tr + 1, 8):
        attacks |= BB_SQUARES[r*8+tf]
        if BB_SQUARES[r*8+tf] & block: break
    for r in range(tr - 1, -1, -1):
        attacks |= BB_SQUARES[r*8+tf]
        if BB_SQUARES[r*8+tf] & block: break
    for f in range(tf + 1, 8):
        attacks |= BB_SQUARES[tr*8+f]
        if BB_SQUARES[tr*8+f] & block: break
    for f in range(tf - 1, -1, -1):
        attacks |= BB_SQUARES[tr*8+f]
        if BB_SQUARES[tr*8+f] & block: break
    return attacks

# @nb.njit(nb.uint64(nb.int32, nb.uint8, nb.uint64), cache=True)
# def set_occupancy(index, bits_in_mask, attack_mask):
#     ...

# def init_sliders_attacks():
#     ...

# BISHOP_MASKS, ROOK_MASKS, BISHOP_ATTACKS, ROOK_ATTACKS = init_sliders_attacks()

@nb.njit(nb.uint64(nb.uint8, nb.uint64), cache=True)
def get_bishop_attacks(sq, occ):
    return bishop_attacks_on_the_fly(sq, occ)
@nb.njit(nb.uint64(nb.uint8, nb.uint64), cache=True)
def get_rook_attacks(sq, occ):
    return rook_attacks_on_the_fly(sq, occ)
@nb.njit(nb.uint64(nb.uint8, nb.uint64), cache=True)
def get_queen_attacks(sq, occ): return get_rook_attacks(sq, occ) | get_bishop_attacks(sq, occ)

KNIGHT_ATTACKS, KING_ATTACKS = (numpy.empty(64, dtype=numpy.uint64), numpy.empty(64, dtype=numpy.uint64))
def _precompute_leaper_attacks():
    for sq in range(64):
        bb = BB_SQUARES[sq]
        KNIGHT_ATTACKS[sq] = (((bb << 17) & NOT_A_FILE)|((bb << 15) & NOT_H_FILE)|((bb << 10) & NOT_AB_FILE)|((bb << 6) & NOT_GH_FILE)|((bb >> 6) & NOT_AB_FILE)|((bb >> 10) & NOT_GH_FILE)|((bb >> 15) & NOT_A_FILE)|((bb >> 17) & NOT_H_FILE))
        KING_ATTACKS[sq] = (((bb << 1)|(bb << 9)|(bb >> 7)) & NOT_A_FILE)|(((bb >> 1)|(bb >> 9)|(bb << 7)) & NOT_H_FILE)|(bb << 8)|(bb >> 8)
_precompute_leaper_attacks()

@nb.njit(cache=True)
def is_square_attacked(sq, attacker_side, board_state):
    (wp_bb, wn_bb, wb_bb, wr_bb, wq_bb, wk_bb, bp_bb, bn_bb, bb_bb, br_bb, bq_bb, bk_bb, _, _, all_pieces_bb, _, _, _) = board_state
    if attacker_side == WHITE:
        pawn_attacks = (((BB_SQUARES[sq] & NOT_A_FILE) >> 9) | ((BB_SQUARES[sq] & NOT_H_FILE) >> 7))
        if pawn_attacks & wp_bb: return True
        return (KING_ATTACKS[sq] & wk_bb) or (KNIGHT_ATTACKS[sq] & wn_bb) or (get_bishop_attacks(sq, all_pieces_bb) & (wb_bb|wq_bb)) or (get_rook_attacks(sq, all_pieces_bb) & (wr_bb|wq_bb))
    else:
        pawn_attacks = (((BB_SQUARES[sq] & NOT_H_FILE) << 9) | ((BB_SQUARES[sq] & NOT_A_FILE) << 7))
        if pawn_attacks & bp_bb: return True
        return (KING_ATTACKS[sq] & bk_bb) or (KNIGHT_ATTACKS[sq] & bn_bb) or (get_bishop_attacks(sq, all_pieces_bb) & (bb_bb|bq_bb)) or (get_rook_attacks(sq, all_pieces_bb) & (br_bb|bq_bb))

@nb.njit(nbt.uint16[:](board_state_flat_signature), cache=True)
def generate_pseudo_legal_moves(board_state):
    moves = numpy.zeros(256, dtype=numpy.uint16)
    move_count = 0

    (wp_bb, wn_bb, wb_bb, wr_bb, wq_bb, wk_bb,
     bp_bb, bn_bb, bb_bb, br_bb, bq_bb, bk_bb,
     white_pieces_bb, black_pieces_bb, all_pieces_bb,
     castling_rights, en_passant_square, side_to_move) = board_state

    own_pieces_bb = white_pieces_bb if side_to_move == WHITE else black_pieces_bb
    opponent_pieces_bb = black_pieces_bb if side_to_move == WHITE else white_pieces_bb
    empty_squares = ~all_pieces_bb
    WK, WQ, BK, BQ = 1, 2, 4, 8

    if side_to_move == WHITE:
        single_pushes = (wp_bb << 8) & empty_squares
        double_pushes = ((single_pushes & RANK_3) << 8) & empty_squares
        captures_west = ((wp_bb & NOT_A_FILE) << 7) & opponent_pieces_bb
        captures_east = ((wp_bb & NOT_H_FILE) << 9) & opponent_pieces_bb

        pushes = single_pushes
        while pushes:
            to_sq = get_ls1b_index(pushes)
            from_sq = to_sq - 8
            if to_sq >= 56:
                for p_type in [PROMO_QUEEN, PROMO_ROOK, PROMO_BISHOP, PROMO_KNIGHT]: moves[move_count] = encode_move(from_sq, to_sq, p_type, SPECIAL_MOVE_FLAG_PROMOTION); move_count += 1
            else:
                moves[move_count] = encode_move(from_sq, to_sq, 0, SPECIAL_MOVE_FLAG_NORMAL); move_count += 1
            pushes &= (pushes - numpy.uint64(1))

        pushes = double_pushes
        while pushes:
            to_sq = get_ls1b_index(pushes)
            from_sq = to_sq - 16
            moves[move_count] = encode_move(from_sq, to_sq, 0, SPECIAL_MOVE_FLAG_NORMAL); move_count += 1
            pushes &= (pushes - numpy.uint64(1))

        caps = captures_west
        while caps:
            to_sq = get_ls1b_index(caps)
            from_sq = to_sq - 7
            if to_sq >= 56:
                for p_type in [PROMO_QUEEN, PROMO_ROOK, PROMO_BISHOP, PROMO_KNIGHT]: moves[move_count] = encode_move(from_sq, to_sq, p_type, SPECIAL_MOVE_FLAG_PROMOTION); move_count += 1
            else:
                moves[move_count] = encode_move(from_sq, to_sq, 0, SPECIAL_MOVE_FLAG_NORMAL); move_count += 1
            caps &= (caps - numpy.uint64(1))

        caps = captures_east
        while caps:
            to_sq = get_ls1b_index(caps)
            from_sq = to_sq - 9
            if to_sq >= 56:
                for p_type in [PROMO_QUEEN, PROMO_ROOK, PROMO_BISHOP, PROMO_KNIGHT]: moves[move_count] = encode_move(from_sq, to_sq, p_type, SPECIAL_MOVE_FLAG_PROMOTION); move_count += 1
            else:
                moves[move_count] = encode_move(from_sq, to_sq, 0, SPECIAL_MOVE_FLAG_NORMAL); move_count += 1
            caps &= (caps - numpy.uint64(1))

        if en_passant_square != -1:
            ep_target_bb = BB_SQUARES[numpy.int8(en_passant_square)]
            west_attackers = (wp_bb & NOT_H_FILE) << 9
            if west_attackers & ep_target_bb:
                from_sq = numpy.int8(en_passant_square - 9)
                moves[move_count] = encode_move(from_sq, numpy.int8(en_passant_square), 0, SPECIAL_MOVE_FLAG_EN_PASSANT); move_count += 1
            east_attackers = (wp_bb & NOT_A_FILE) << 7
            if east_attackers & ep_target_bb:
                from_sq = numpy.int8(en_passant_square - 7)
                moves[move_count] = encode_move(from_sq, numpy.int8(en_passant_square), 0, SPECIAL_MOVE_FLAG_EN_PASSANT); move_count += 1

        if (castling_rights & WK) and not(all_pieces_bb & 0x60) and not is_square_attacked(4,BLACK,board_state) and not is_square_attacked(5,BLACK,board_state) and not is_square_attacked(6,BLACK,board_state): moves[move_count]=encode_move(4,6,0,SPECIAL_MOVE_FLAG_CASTLING); move_count+=1
        if (castling_rights & WQ) and not(all_pieces_bb & 0xe) and not is_square_attacked(4,BLACK,board_state) and not is_square_attacked(3,BLACK,board_state) and not is_square_attacked(2,BLACK,board_state): moves[move_count]=encode_move(4,2,0,SPECIAL_MOVE_FLAG_CASTLING); move_count+=1

    else: # BLACK
        single_pushes = (bp_bb >> 8) & empty_squares
        double_pushes = ((single_pushes & RANK_6) >> 8) & empty_squares
        captures_west = ((bp_bb & NOT_A_FILE) >> 9) & opponent_pieces_bb
        captures_east = ((bp_bb & NOT_H_FILE) >> 7) & opponent_pieces_bb

        pushes = single_pushes
        while pushes:
            to_sq = get_ls1b_index(pushes)
            from_sq = to_sq + 8
            if to_sq <= 7:
                for p_type in [PROMO_QUEEN, PROMO_ROOK, PROMO_BISHOP, PROMO_KNIGHT]: moves[move_count] = encode_move(from_sq, to_sq, p_type, SPECIAL_MOVE_FLAG_PROMOTION); move_count += 1
            else:
                moves[move_count] = encode_move(from_sq, to_sq, 0, SPECIAL_MOVE_FLAG_NORMAL); move_count += 1
            pushes &= (pushes - numpy.uint64(1))

        pushes = double_pushes
        while pushes:
            to_sq = get_ls1b_index(pushes)
            from_sq = to_sq + 16
            moves[move_count] = encode_move(from_sq, to_sq, 0, SPECIAL_MOVE_FLAG_NORMAL); move_count += 1
            pushes &= (pushes - numpy.uint64(1))

        caps = captures_west
        while caps:
            to_sq = get_ls1b_index(caps)
            from_sq = to_sq + 9
            if to_sq <= 7:
                for p_type in [PROMO_QUEEN, PROMO_ROOK, PROMO_BISHOP, PROMO_KNIGHT]: moves[move_count] = encode_move(from_sq, to_sq, p_type, SPECIAL_MOVE_FLAG_PROMOTION); move_count += 1
            else:
                moves[move_count] = encode_move(from_sq, to_sq, 0, SPECIAL_MOVE_FLAG_NORMAL); move_count += 1
            caps &= (caps - numpy.uint64(1))

        caps = captures_east
        while caps:
            to_sq = get_ls1b_index(caps)
            from_sq = to_sq + 7
            if to_sq <= 7:
                for p_type in [PROMO_QUEEN, PROMO_ROOK, PROMO_BISHOP, PROMO_KNIGHT]: moves[move_count] = encode_move(from_sq, to_sq, p_type, SPECIAL_MOVE_FLAG_PROMOTION); move_count += 1
            else:
                moves[move_count] = encode_move(from_sq, to_sq, 0, SPECIAL_MOVE_FLAG_NORMAL); move_count += 1
            caps &= (caps - numpy.uint64(1))

        if en_passant_square != -1:
            ep_target_bb = BB_SQUARES[numpy.int8(en_passant_square)]
            west_attackers = (bp_bb & NOT_H_FILE) >> 7
            if west_attackers & ep_target_bb:
                from_sq = numpy.int8(en_passant_square + 7)
                moves[move_count] = encode_move(from_sq, numpy.int8(en_passant_square), 0, SPECIAL_MOVE_FLAG_EN_PASSANT); move_count += 1
            east_attackers = (bp_bb & NOT_A_FILE) >> 9
            if east_attackers & ep_target_bb:
                from_sq = numpy.int8(en_passant_square + 9)
                moves[move_count] = encode_move(from_sq, numpy.int8(en_passant_square), 0, SPECIAL_MOVE_FLAG_EN_PASSANT); move_count += 1

        if (castling_rights & BK) and not(all_pieces_bb & 0x6000000000000000) and not is_square_attacked(60,WHITE,board_state) and not is_square_attacked(61,WHITE,board_state) and not is_square_attacked(62,WHITE,board_state): moves[move_count]=encode_move(60,62,0,SPECIAL_MOVE_FLAG_CASTLING); move_count+=1
        if (castling_rights & BQ) and not(all_pieces_bb & 0xe00000000000000) and not is_square_attacked(60,WHITE,board_state) and not is_square_attacked(59,WHITE,board_state) and not is_square_attacked(58,WHITE,board_state): moves[move_count]=encode_move(60,58,0,SPECIAL_MOVE_FLAG_CASTLING); move_count+=1

    piece_bbs = (wn_bb, wb_bb, wr_bb, wq_bb, wk_bb) if side_to_move == WHITE else (bn_bb, bb_bb, br_bb, bq_bb, bk_bb)
    for piece_type in range(5):
        bb = piece_bbs[piece_type]
        while bb:
            from_sq = get_ls1b_index(bb)
            if piece_type == 0: targets = KNIGHT_ATTACKS[from_sq] & ~own_pieces_bb
            elif piece_type == 1: targets = get_bishop_attacks(from_sq, all_pieces_bb) & ~own_pieces_bb
            elif piece_type == 2: targets = get_rook_attacks(from_sq, all_pieces_bb) & ~own_pieces_bb
            elif piece_type == 3: targets = get_queen_attacks(from_sq, all_pieces_bb) & ~own_pieces_bb
            else: targets = KING_ATTACKS[from_sq] & ~own_pieces_bb

            while targets:
                to_sq = get_ls1b_index(targets)
                moves[move_count] = encode_move(from_sq, to_sq, 0, SPECIAL_MOVE_FLAG_NORMAL); move_count+=1
                targets &= (targets - numpy.uint64(1))
            bb &= (bb - numpy.uint64(1))

    return moves[:move_count]

@nb.njit(nbt.uint16[:](board_state_flat_signature), cache=True)
def generate_legal_moves(board_state):
    """
    Generates all fully legal moves by filtering pseudo-legal moves.
    This is the core logic for the Perft loop.
    """
    pseudo_legal_moves = generate_pseudo_legal_moves(board_state)
    legal_moves = numpy.zeros(256, dtype=numpy.uint16)
    legal_move_count = 0

    side_that_moved = board_state[17]
    original_piece_bbs = board_state[0:12]
    original_occupancy_bbs = board_state[12:15]
    original_game_state = (board_state[17], board_state[15], numpy.int8(board_state[16]), 0, numpy.uint64(0))

    for i in range(len(pseudo_legal_moves)):
        move = pseudo_legal_moves[i]
        new_piece_bbs, new_occupancy_bbs, new_game_state, _ = make_move(
            original_piece_bbs, original_occupancy_bbs, original_game_state, move
        )

        new_board_state_flat = new_piece_bbs + new_occupancy_bbs + (new_game_state[1], new_game_state[2], new_game_state[0])
        king_bb = new_piece_bbs[5] if side_that_moved == WHITE else new_piece_bbs[11]

        if king_bb == 0:
            continue

        king_sq = get_ls1b_index(king_bb)
        if not is_square_attacked(king_sq, new_game_state[0], new_board_state_flat):
            legal_moves[legal_move_count] = move
            legal_move_count += 1

    return legal_moves[:legal_move_count]
