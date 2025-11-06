# chess_engine/move_generator.py
import numpy
import numba as nb

from chess_engine.move import (
    encode_move, FLAG_NORMAL, FLAG_PROMOTION, FLAG_EN_PASSANT, FLAG_CASTLING,
    PROMO_QUEEN, PROMO_ROOK, PROMO_BISHOP, PROMO_KNIGHT
)
from chess_engine.board_operations import make_move

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

BISHOP_RELEVANT_BITS = numpy.array([6,5,5,5,5,5,5,6,5,5,5,5,5,5,5,5,5,5,7,7,7,7,5,5,5,5,7,9,9,7,5,5,5,5,7,9,9,7,5,5,5,5,7,7,7,7,5,5,5,5,5,5,5,5,5,5,6,5,5,5,5,5,5,6], dtype=numpy.uint8)
ROOK_RELEVANT_BITS = numpy.array([12,11,11,11,11,11,11,12,11,10,10,10,10,10,10,11,11,10,10,10,10,10,10,11,11,10,10,10,10,10,10,11,11,10,10,10,10,10,10,11,11,10,10,10,10,10,10,11,11,10,10,10,10,10,10,11,12,11,11,11,11,11,11,12], dtype=numpy.uint8)
ROOK_MAGIC_NUMBERS = numpy.array([0x8a80104000800020,0x140002000100040,0x2801880a0017001,0x100081001000420,0x200020010080420,0x3001c0002010008,0x8480008002000100,0x2080088004402900,0x800098204000,0x2024401000200040,0x100802000801000,0x120800800801000,0x208808088000400,0x2802200800400,0x2200800100020080,0x801000060821100,0x80044006422000,0x100808020004000,0x12108a0010204200,0x140848010000802,0x481828014002800,0x8094004002004100,0x4010040010010802,0x20008806104,0x100400080208000,0x2040002120081000,0x21200680100081,0x20100080080080,0x2000a00200410,0x20080800400,0x80088400100102,0x80004600042881,0x4040008040800020,0x440003000200801,0x4200011004500,0x188020010100100,0x14800401802800,0x2080040080800200,0x124080204001001,0x200046502000484,0x480400080088020,0x1000422010034000,0x30200100110040,0x100021010009,0x2002080100110004,0x202008004008002,0x20020004010100,0x2048440040820001,0x101002200408200,0x40802000401080,0x4008142004410100,0x2060820c0120200,0x1001004080100,0x20c020080040080,0x2935610830022400,0x44440041009200,0x280001040802101,0x2100190040002085,0x80c0084100102001,0x4024081001000421,0x20030a0244872,0x12001008414402,0x2006104900a0804,0x1004081002402], dtype=numpy.uint64)
BISHOP_MAGIC_NUMBERS = numpy.array([0x40040844404084,0x2004208a004208,0x10190041080202,0x108060845042010,0x581104180800210,0x2112080446200010,0x1080820820060210,0x3c0808410220200,0x4050404440404,0x21001420088,0x24d0080801082102,0x1020a0a020400,0x40308200402,0x4011002100800,0x401484104104005,0x801010402020200,0x400210c3880100,0x404022024108200,0x810018200204102,0x4002801a02003,0x85040820080400,0x810102c808880400,0xe900410884800,0x8002020480840102,0x220200865090201,0x2010100a02021202,0x152048408022401,0x20080002081110,0x4001001021004000,0x800040400a011002,0xe4004081011002,0x1c004001012080,0x8004200962a00220,0x8422100208500202,0x2000402200300c08,0x8646020080080080,0x80020a0200100808,0x2010004880111000,0x623000a080011400,0x42008c0340209202,0x209188240001000,0x400408a884001800,0x110400a6080400,0x1840060a44020800,0x90080104000041,0x201011000808101,0x1a2208080504f080,0x8012020600211212,0x500861011240000,0x180806108200800,0x4000020e01040044,0x300000261044000a,0x802241102020002,0x20906061210001,0x5a84841004010310,0x4010801011c04,0xa010109502200,0x4a02012000,0x500201010098b028,0x8040002811040900,0x28000010020204,0x6000020202d0240,0x8918844842082200,0x401001102902002], dtype=numpy.uint64)

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

@nb.njit(nb.uint64(nb.uint8), cache=True)
def mask_bishop_attacks(sq):
    attacks, tr, tf = EMPTY, sq // 8, sq % 8
    for r, f in zip(range(tr + 1, 7), range(tf + 1, 7)): attacks |= BB_SQUARES[r * 8 + f]
    for r, f in zip(range(tr - 1, 0, -1), range(tf + 1, 7)): attacks |= BB_SQUARES[r * 8 + f]
    for r, f in zip(range(tr + 1, 7), range(tf - 1, 0, -1)): attacks |= BB_SQUARES[r * 8 + f]
    for r, f in zip(range(tr - 1, 0, -1), range(tf - 1, 0, -1)): attacks |= BB_SQUARES[r * 8 + f]
    return attacks
@nb.njit(nb.uint64(nb.uint8), cache=True)
def mask_rook_attacks(sq):
    attacks, tr, tf = EMPTY, sq // 8, sq % 8
    for r in range(tr + 1, 7): attacks |= BB_SQUARES[r * 8 + tf]
    for r in range(tr - 1, 0, -1): attacks |= BB_SQUARES[r * 8 + tf]
    for f in range(tf + 1, 7): attacks |= BB_SQUARES[tr * 8 + f]
    for f in range(tf - 1, 0, -1): attacks |= BB_SQUARES[tr * 8 + f]
    return attacks
BISHOP_MASKS, ROOK_MASKS = (numpy.fromiter((mask_bishop_attacks(sq) for sq in range(64)), dtype=numpy.uint64), numpy.fromiter((mask_rook_attacks(sq) for sq in range(64)), dtype=numpy.uint64))
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
@nb.njit(nb.uint64(nb.int32, nb.uint8, nb.uint64), cache=True)
def set_occupancy(index, bits_in_mask, attack_mask):
    occupancy = EMPTY
    for count in range(bits_in_mask):
        sq = get_ls1b_index(attack_mask)
        attack_mask &= ~BB_SQUARES[sq]
        if index & (1 << count): occupancy |= BB_SQUARES[sq]
    return occupancy
def init_sliders_attacks():
    bishop_attacks, rook_attacks = numpy.empty((64, 512), dtype=numpy.uint64), numpy.empty((64, 4096), dtype=numpy.uint64)
    for sq in range(64):
        attack_mask, relevant_bits_count = BISHOP_MASKS[sq], count_bits(BISHOP_MASKS[sq])
        for i in range(1 << relevant_bits_count):
            occ = set_occupancy(i, relevant_bits_count, attack_mask)
            magic_index = (occ * BISHOP_MAGIC_NUMBERS[sq]) >> numpy.uint64(64 - BISHOP_RELEVANT_BITS[sq])
            bishop_attacks[sq][magic_index] = bishop_attacks_on_the_fly(sq, occ)
        attack_mask, relevant_bits_count = ROOK_MASKS[sq], count_bits(ROOK_MASKS[sq])
        for i in range(1 << relevant_bits_count):
            occ = set_occupancy(i, relevant_bits_count, attack_mask)
            magic_index = (occ * ROOK_MAGIC_NUMBERS[sq]) >> numpy.uint64(64 - ROOK_RELEVANT_BITS[sq])
            rook_attacks[sq][magic_index] = rook_attacks_on_the_fly(sq, occ)
    return bishop_attacks, rook_attacks
BISHOP_ATTACKS, ROOK_ATTACKS = init_sliders_attacks()

@nb.njit(nb.uint64(nb.uint8, nb.uint64), cache=True)
def get_bishop_attacks(sq, occ):
    occ &= BISHOP_MASKS[sq]; occ *= BISHOP_MAGIC_NUMBERS[sq]; occ >>= numpy.uint64(64-BISHOP_RELEVANT_BITS[sq])
    return BISHOP_ATTACKS[sq][occ]
@nb.njit(nb.uint64(nb.uint8, nb.uint64), cache=True)
def get_rook_attacks(sq, occ):
    occ &= ROOK_MASKS[sq]; occ *= ROOK_MAGIC_NUMBERS[sq]; occ >>= numpy.uint64(64-ROOK_RELEVANT_BITS[sq])
    return ROOK_ATTACKS[sq][occ]
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
    (wp_bb, wn_bb, wb_bb, wr_bb, wq_bb, wk_bb, bp_bb, bn_bb, bb_bb, br_bb, bq_bb, bk_bb, _, _, all_pieces_bb, _, _, _, _) = board_state
    if attacker_side == WHITE:
        pawn_attacks = (((BB_SQUARES[sq] & NOT_A_FILE) >> 9) | ((BB_SQUARES[sq] & NOT_H_FILE) >> 7))
        if pawn_attacks & wp_bb: return True
        return (KING_ATTACKS[sq] & wk_bb) or (KNIGHT_ATTACKS[sq] & wn_bb) or (get_bishop_attacks(sq, all_pieces_bb) & (wb_bb|wq_bb)) or (get_rook_attacks(sq, all_pieces_bb) & (wr_bb|wq_bb))
    else:
        pawn_attacks = (((BB_SQUARES[sq] & NOT_H_FILE) << 9) | ((BB_SQUARES[sq] & NOT_A_FILE) << 7))
        if pawn_attacks & bp_bb: return True
        return (KING_ATTACKS[sq] & bk_bb) or (KNIGHT_ATTACKS[sq] & bn_bb) or (get_bishop_attacks(sq, all_pieces_bb) & (bb_bb|bq_bb)) or (get_rook_attacks(sq, all_pieces_bb) & (br_bb|bq_bb))

@nb.njit(cache=True)
def generate_legal_moves(board_state):
    moves = numpy.zeros(256, dtype=numpy.uint16)
    move_count = 0
    (wp_bb, wn_bb, wb_bb, wr_bb, wq_bb, wk_bb, bp_bb, bn_bb, bb_bb, br_bb, bq_bb, bk_bb,
     white_pieces_bb, black_pieces_bb, all_pieces_bb, castling_rights, en_passant_square, side_to_move, _) = board_state

    own_pieces_bb = white_pieces_bb if side_to_move == WHITE else black_pieces_bb
    opponent_pieces_bb = black_pieces_bb if side_to_move == WHITE else white_pieces_bb
    empty_squares = ~all_pieces_bb
    WK,WQ,BK,BQ = 1,2,4,8

    if side_to_move == WHITE:
        king_sq = get_ls1b_index(wk_bb)
        single_pushes, double_pushes = (wp_bb << 8) & empty_squares, ((wp_bb & RANK_2) << 16) & empty_squares & (empty_squares << 8)
        captures_east, captures_west = ((wp_bb & NOT_H_FILE) << 9) & opponent_pieces_bb, ((wp_bb & NOT_A_FILE) << 7) & opponent_pieces_bb

        for to_sq in range(64):
            if (BB_SQUARES[to_sq] & single_pushes):
                from_sq = to_sq - 8
                if to_sq >= 56:
                    for p_type in [PROMO_QUEEN, PROMO_ROOK, PROMO_BISHOP, PROMO_KNIGHT]:
                        moves[move_count] = encode_move(from_sq, to_sq, FLAG_PROMOTION, p_type)
                        move_count += 1
                else: moves[move_count] = encode_move(from_sq, to_sq, FLAG_NORMAL); move_count += 1
            if (BB_SQUARES[to_sq] & double_pushes): moves[move_count] = encode_move(to_sq - 16, to_sq, FLAG_NORMAL); move_count += 1
            if (BB_SQUARES[to_sq] & captures_east):
                from_sq = to_sq - 9
                if to_sq >= 56:
                    for p_type in [PROMO_QUEEN, PROMO_ROOK, PROMO_BISHOP, PROMO_KNIGHT]:
                        moves[move_count] = encode_move(from_sq, to_sq, FLAG_PROMOTION, p_type)
                        move_count += 1
                else: moves[move_count] = encode_move(from_sq, to_sq, FLAG_NORMAL); move_count += 1
            if (BB_SQUARES[to_sq] & captures_west):
                from_sq = to_sq - 7
                if to_sq >= 56:
                    for p_type in [PROMO_QUEEN, PROMO_ROOK, PROMO_BISHOP, PROMO_KNIGHT]:
                        moves[move_count] = encode_move(from_sq, to_sq, FLAG_PROMOTION, p_type)
                        move_count += 1
                else: moves[move_count] = encode_move(from_sq, to_sq, FLAG_NORMAL); move_count += 1
        if en_passant_square != -1:
            ep_captures = (((wp_bb & NOT_A_FILE) << 7) | ((wp_bb & NOT_H_FILE) << 9)) & BB_SQUARES[en_passant_square]
            if ep_captures:
                to_sq = en_passant_square
                if (wp_bb << 7) & BB_SQUARES[to_sq]: moves[move_count] = encode_move(to_sq-7, to_sq, FLAG_EN_PASSANT); move_count+=1
                if (wp_bb << 9) & BB_SQUARES[to_sq]: moves[move_count] = encode_move(to_sq-9, to_sq, FLAG_EN_PASSANT); move_count+=1
        if (castling_rights & WK) and not(all_pieces_bb & (BB_SQUARES[5]|BB_SQUARES[6])) and not is_square_attacked(4,BLACK,board_state) and not is_square_attacked(5,BLACK,board_state): moves[move_count]=encode_move(4,6,FLAG_CASTLING); move_count+=1
        if (castling_rights & WQ) and not(all_pieces_bb & (BB_SQUARES[1]|BB_SQUARES[2]|BB_SQUARES[3])) and not is_square_attacked(4,BLACK,board_state) and not is_square_attacked(3,BLACK,board_state): moves[move_count]=encode_move(4,2,FLAG_CASTLING); move_count+=1
    else: # BLACK
        king_sq = get_ls1b_index(bk_bb)
        single_pushes, double_pushes = (bp_bb >> 8) & empty_squares, ((bp_bb & RANK_7) >> 16) & empty_squares & (empty_squares >> 8)
        captures_east, captures_west = ((bp_bb & NOT_H_FILE) >> 7) & opponent_pieces_bb, ((bp_bb & NOT_A_FILE) >> 9) & opponent_pieces_bb
        for to_sq in range(64):
            if (BB_SQUARES[to_sq] & single_pushes):
                from_sq = to_sq + 8
                if to_sq <= 7:
                    for p_type in [PROMO_QUEEN, PROMO_ROOK, PROMO_BISHOP, PROMO_KNIGHT]:
                        moves[move_count] = encode_move(from_sq, to_sq, FLAG_PROMOTION, p_type)
                        move_count += 1
                else: moves[move_count] = encode_move(from_sq, to_sq, FLAG_NORMAL); move_count += 1
            if (BB_SQUARES[to_sq] & double_pushes): moves[move_count] = encode_move(to_sq + 16, to_sq, FLAG_NORMAL); move_count += 1
            if (BB_SQUARES[to_sq] & captures_east):
                from_sq = to_sq + 7
                if to_sq <= 7:
                    for p_type in [PROMO_QUEEN, PROMO_ROOK, PROMO_BISHOP, PROMO_KNIGHT]:
                        moves[move_count] = encode_move(from_sq, to_sq, FLAG_PROMOTION, p_type)
                        move_count += 1
                else: moves[move_count] = encode_move(from_sq, to_sq, FLAG_NORMAL); move_count += 1
            if (BB_SQUARES[to_sq] & captures_west):
                from_sq = to_sq + 9
                if to_sq <= 7:
                    for p_type in [PROMO_QUEEN, PROMO_ROOK, PROMO_BISHOP, PROMO_KNIGHT]:
                        moves[move_count] = encode_move(from_sq, to_sq, FLAG_PROMOTION, p_type)
                        move_count += 1
                else: moves[move_count] = encode_move(from_sq, to_sq, FLAG_NORMAL); move_count += 1
        if en_passant_square != -1:
            ep_captures = (((bp_bb & NOT_A_FILE) >> 9) | ((bp_bb & NOT_H_FILE) >> 7)) & BB_SQUARES[en_passant_square]
            if ep_captures:
                to_sq = en_passant_square
                if (bp_bb >> 7) & BB_SQUARES[to_sq]: moves[move_count] = encode_move(to_sq+7, to_sq, FLAG_EN_PASSANT); move_count+=1
                if (bp_bb >> 9) & BB_SQUARES[to_sq]: moves[move_count] = encode_move(to_sq+9, to_sq, FLAG_EN_PASSANT); move_count+=1
        if (castling_rights & BK) and not(all_pieces_bb & (BB_SQUARES[61]|BB_SQUARES[62])) and not is_square_attacked(60,WHITE,board_state) and not is_square_attacked(61,WHITE,board_state): moves[move_count]=encode_move(60,62,FLAG_CASTLING); move_count+=1
        if (castling_rights & BQ) and not(all_pieces_bb & (BB_SQUARES[57]|BB_SQUARES[58]|BB_SQUARES[59])) and not is_square_attacked(60,WHITE,board_state) and not is_square_attacked(59,WHITE,board_state): moves[move_count]=encode_move(60,58,FLAG_CASTLING); move_count+=1

    piece_bbs = (wn_bb, wb_bb, wr_bb, wq_bb, wk_bb) if side_to_move == WHITE else (bn_bb, bb_bb, br_bb, bq_bb, bk_bb)
    for piece_type in range(5):
        bb = piece_bbs[piece_type]
        while bb:
            from_sq = get_ls1b_index(bb)
            if piece_type==0: targets = KNIGHT_ATTACKS[from_sq] & ~own_pieces_bb
            elif piece_type==1: targets = get_bishop_attacks(from_sq, all_pieces_bb) & ~own_pieces_bb
            elif piece_type==2: targets = get_rook_attacks(from_sq, all_pieces_bb) & ~own_pieces_bb
            elif piece_type==3: targets = get_queen_attacks(from_sq, all_pieces_bb) & ~own_pieces_bb
            else: targets = KING_ATTACKS[from_sq] & ~own_pieces_bb
            while targets:
                to_sq = get_ls1b_index(targets)
                moves[move_count] = encode_move(from_sq, to_sq, FLAG_NORMAL); move_count+=1
                targets &= ~BB_SQUARES[to_sq]
            bb &= ~BB_SQUARES[from_sq]

    legal_moves, legal_count = numpy.zeros(256, dtype=numpy.uint16), 0
    king_sq_original = get_ls1b_index(wk_bb if side_to_move == WHITE else bk_bb)
    for i in range(move_count):
        move = moves[i]
        temp_board_state = make_move(board_state, move)
        temp_king_bb = temp_board_state[5] if side_to_move == WHITE else temp_board_state[11]
        if not temp_king_bb: continue
        current_king_sq = get_ls1b_index(temp_king_bb)
        if not is_square_attacked(current_king_sq, 1 - side_to_move, temp_board_state):
            legal_moves[legal_count] = move
            legal_count += 1
    return legal_moves[:legal_count]
