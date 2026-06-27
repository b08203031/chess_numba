# chess_engine/move_generator.py
import numpy as np
import numba

from chess_engine.classical.move import (
    encode_move, SPECIAL_MOVE_FLAG_NORMAL, SPECIAL_MOVE_FLAG_PROMOTION, SPECIAL_MOVE_FLAG_EN_PASSANT, SPECIAL_MOVE_FLAG_CASTLING,
    PROMO_QUEEN, PROMO_ROOK, PROMO_BISHOP, PROMO_KNIGHT,
    get_from_square, get_to_square, get_special_move_flag
)
from chess_engine.classical.board_operations import make_move, unmake_move
from chess_engine.classical.engine_types import (
    piece_bbs_signature, occupancy_bbs_signature, game_state_signature
)
from chess_engine.classical.constants import BB_SQUARES, ROOK, QUEEN, BISHOP, KING, KNIGHT, PAWN
import numba.types as nbt

# =============================================================================
# Constants and Pre-computation
# =============================================================================
NOT_A_FILE = ~np.uint64(0x0101010101010101)
NOT_H_FILE = ~np.uint64(0x8080808080808080)
NOT_AB_FILE = ~np.uint64(0x0303030303030303)
NOT_GH_FILE = ~np.uint64(0xC0C0C0C0C0C0C0C0)
RANK_1 = np.uint64(0xFF)
RANK_2 = np.uint64(0xFF00)
RANK_3 = np.uint64(0xFF0000)
RANK_4 = np.uint64(0xFF000000)
RANK_5 = np.uint64(0xFF00000000)
RANK_6 = np.uint64(0xFF0000000000)
RANK_7 = np.uint64(0xFF000000000000)
RANK_8 = np.uint64(0xFF00000000000000)
WHITE, BLACK = 0, 1
EMPTY = np.uint64(0)

BISHOP_RELEVANT_BITS = np.array([6,5,5,5,5,5,5,6,5,5,5,5,5,5,5,5,5,5,7,7,7,7,5,5,5,5,7,9,9,7,5,5,5,5,7,9,9,7,5,5,5,5,7,7,7,7,5,5,5,5,5,5,5,5,5,5,6,5,5,5,5,5,5,6], dtype=np.uint8)
ROOK_RELEVANT_BITS = np.array([12,11,11,11,11,11,11,12,11,10,10,10,10,10,10,11,11,10,10,10,10,10,10,11,11,10,10,10,10,10,10,11,11,10,10,10,10,10,10,11,11,10,10,10,10,10,10,11,11,10,10,10,10,10,10,11,12,11,11,11,11,11,11,12], dtype=np.uint8)
BISHOP_MAGIC_NUMBERS = np.array([
    0x440010805110220, 0x8200800b3004085, 0x4140082091440, 0x2110c0089003808,
    0xc20404a101100040, 0x1008804c1008009, 0xc002008220118002, 0x26820a0104111408,
    0x2110600404080041, 0x1300201840090, 0x40410401184315, 0x8000020a02020042,
    0x800c840c20010000, 0x1140a0804040884, 0x20018404924028, 0x201009049101000,
    0x1004c042080140, 0x20282304040082, 0x402004420240100, 0xa094000124008010,
    0x401004890400644, 0x200e802148044008, 0x24040021008824c0, 0x20861a0200820e,
    0x2030130104202200, 0x81150020144400, 0xc60880010004410, 0x42008008048002,
    0x489840020822001, 0x404002031000, 0x4084008484020501, 0x1062412049c00,
    0x8054040101210, 0x4d04022080080d82, 0x1044004100080200, 0x60180080080,
    0xc002288400020020, 0x300828100120300, 0x8404080208094100, 0x8214110030020080,
    0x8013084240001024, 0x200b451010000881, 0x103808000400, 0xa0022018002100,
    0x1013049536004400, 0x3008030802016120, 0x10500081010182, 0x8a682600a10080,
    0x10022804040c0001, 0x2044404840001, 0x280010088440002, 0x6880003484040205,
    0x3222020c00, 0xa00a00549020412, 0x8020a04201810042, 0x6300142058011,
    0x6401440488011008, 0x14248082900, 0x420000284048810, 0x82009000840400,
    0x2400000a10021608, 0x4010104044088880, 0x48304a018010100, 0x420040082004200,
], dtype=np.uint64)

ROOK_MAGIC_NUMBERS = np.array([
    0x1c80001080c00220, 0x2040001000402000, 0x2080082000801002, 0x8080042801801001,
    0x80080080228400, 0x9100010034008826, 0x3600080200010084, 0x6200010202468224,
    0x8100800340088822, 0x400050002000, 0x2001028420080, 0x12000932016040,
    0x101001025002800, 0x8200808004000200, 0x200800100800200, 0x4002000042008104,
    0x101c308000400c82, 0x210004000200040, 0x81010040122000, 0x2201010010000820,
    0x48808004000800, 0x801010018020400, 0x40010210a48, 0x1008020000842849,
    0x816080004010, 0x200180400081, 0x2001004100200810, 0x2000210100089001,
    0x14008080480104, 0x240040080020080, 0x8508050080800200, 0x90001000c8042,
    0x684400081800030, 0x491000c004402000, 0x40018c2000801000, 0x20010a0012004020,
    0x4009001005000800, 0x8082005042000814, 0x1205000401000200, 0x88801044800500,
    0x10080400020800d, 0x4440004020008080, 0x2004183120020, 0x211001000210008,
    0x804880011010004, 0x420025102a0028, 0x421100200040208, 0xc10282000c,
    0x1153248000400080, 0x2009441002600, 0x81b000200880, 0x4000821000080080,
    0x608441188010100, 0x1208082004c0080, 0x1013a2080400, 0x800808409055200,
    0x80a00211080c102, 0x8621008091204606, 0x8c0102001000a41, 0x1a00412090040a,
    0x8220020041009aa, 0x201000208040041, 0x8006010850008204, 0x1094004093002402,
], dtype=np.uint64)

from chess_engine.classical.bitboard_utils import get_lsb_index, count_bits, SQUARES_BETWEEN, ROOK_RAYS, BISHOP_RAYS

@numba.njit(numba.uint64(numba.uint8), cache=True, boundscheck=False, fastmath=True)
def mask_bishop_attacks(sq):
    attacks, tr, tf = EMPTY, sq // 8, sq % 8
    for r, f in zip(range(tr + 1, 7), range(tf + 1, 7)): attacks |= BB_SQUARES[r * 8 + f]
    for r, f in zip(range(tr - 1, 0, -1), range(tf + 1, 7)): attacks |= BB_SQUARES[r * 8 + f]
    for r, f in zip(range(tr + 1, 7), range(tf - 1, 0, -1)): attacks |= BB_SQUARES[r * 8 + f]
    for r, f in zip(range(tr - 1, 0, -1), range(tf - 1, 0, -1)): attacks |= BB_SQUARES[r * 8 + f]
    return attacks
@numba.njit(numba.uint64(numba.uint8), cache=True, boundscheck=False, fastmath=True)
def mask_rook_attacks(sq):
    attacks, tr, tf = EMPTY, sq // 8, sq % 8
    for r in range(tr + 1, 7): attacks |= BB_SQUARES[r * 8 + tf]
    for r in range(tr - 1, 0, -1): attacks |= BB_SQUARES[r * 8 + tf]
    for f in range(tf + 1, 7): attacks |= BB_SQUARES[tr * 8 + f]
    for f in range(tf - 1, 0, -1): attacks |= BB_SQUARES[tr * 8 + f]
    return attacks
BISHOP_MASKS, ROOK_MASKS = (np.fromiter((mask_bishop_attacks(sq) for sq in range(64)), dtype=np.uint64), np.fromiter((mask_rook_attacks(sq) for sq in range(64)), dtype=np.uint64))
@numba.njit(numba.uint64(numba.uint8, numba.uint64), cache=True, boundscheck=False, fastmath=True)
def bishop_attacks_on_the_fly(sq, block):
    attacks, tr, tf = EMPTY, sq // 8, sq % 8
    for r, f in zip(range(tr + 1, 8), range(tf + 1, 8)):
        attacks |= BB_SQUARES[r*8+f]
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
@numba.njit(numba.uint64(numba.uint8, numba.uint64), cache=True, boundscheck=False, fastmath=True)
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
@numba.njit(numba.uint64(numba.int32, numba.uint8, numba.uint64), cache=True, boundscheck=False, fastmath=True)
def set_occupancy(index, bits_in_mask, attack_mask):
    occupancy = EMPTY
    for count in range(bits_in_mask):
        sq = get_lsb_index(attack_mask)
        attack_mask &= ~BB_SQUARES[sq]
        if index & (1 << count): occupancy |= BB_SQUARES[sq]
    return occupancy

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def init_sliders_attacks():
    bishop_attacks = np.zeros((64, 512), dtype=np.uint64)
    rook_attacks = np.zeros((64, 4096), dtype=np.uint64)
    for sq in range(64):
        attack_mask, relevant_bits_count = BISHOP_MASKS[sq], count_bits(BISHOP_MASKS[sq])
        indices = 1 << relevant_bits_count
        for i in range(indices):
            occ = set_occupancy(i, relevant_bits_count, attack_mask)
            magic_index = (np.uint64(occ) * BISHOP_MAGIC_NUMBERS[sq]) >> np.uint64(64 - BISHOP_RELEVANT_BITS[sq])
            bishop_attacks[sq, magic_index] = bishop_attacks_on_the_fly(sq, occ)
            
        attack_mask, relevant_bits_count = ROOK_MASKS[sq], count_bits(ROOK_MASKS[sq])
        indices = 1 << relevant_bits_count
        for i in range(indices):
            occ = set_occupancy(i, relevant_bits_count, attack_mask)
            magic_index = (np.uint64(occ) * ROOK_MAGIC_NUMBERS[sq]) >> np.uint64(64 - ROOK_RELEVANT_BITS[sq])
            rook_attacks[sq, magic_index] = rook_attacks_on_the_fly(sq, occ)
    return bishop_attacks, rook_attacks

BISHOP_ATTACKS, _rook_attacks_temp = init_sliders_attacks()
ROOK_ATTACKS_1 = np.ascontiguousarray(_rook_attacks_temp[:16])
ROOK_ATTACKS_2 = np.ascontiguousarray(_rook_attacks_temp[16:32])
ROOK_ATTACKS_3 = np.ascontiguousarray(_rook_attacks_temp[32:48])
ROOK_ATTACKS_4 = np.ascontiguousarray(_rook_attacks_temp[48:])
# BISHOP_ATTACKS = np.empty((64, 512), dtype=np.uint64)
# ROOK_ATTACKS = np.empty((64, 4096), dtype=np.uint64)

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def get_bishop_attacks(sq, occ):
    """
    使用魔法位元棋盤查表獲取主教的攻擊。
    """
    occ &= BISHOP_MASKS[sq]
    occ *= BISHOP_MAGIC_NUMBERS[sq]
    occ >>= np.uint64(64-BISHOP_RELEVANT_BITS[sq])
    return BISHOP_ATTACKS[sq, occ]

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def get_rook_attacks(sq, occ):
    """
    使用魔法位元棋盤查表獲取城堡的攻擊。
    """
    occ &= ROOK_MASKS[sq]
    occ *= ROOK_MAGIC_NUMBERS[sq]
    occ >>= np.uint64(64-ROOK_RELEVANT_BITS[sq])
    if sq < 16:
        return ROOK_ATTACKS_1[sq, occ]
    elif sq < 32:
        return ROOK_ATTACKS_2[sq - 16, occ]
    elif sq < 48:
        return ROOK_ATTACKS_3[sq - 32, occ]
    else:
        return ROOK_ATTACKS_4[sq - 48, occ]

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def get_queen_attacks(sq, occ): return get_rook_attacks(sq, occ) | get_bishop_attacks(sq, occ)

KNIGHT_ATTACKS, KING_ATTACKS = (np.empty(64, dtype=np.uint64), np.empty(64, dtype=np.uint64))
def _precompute_leaper_attacks():
    for sq in range(64):
        bb = BB_SQUARES[sq]
        KNIGHT_ATTACKS[sq] = (((bb << 17) & NOT_A_FILE)|((bb << 15) & NOT_H_FILE)|((bb << 10) & NOT_AB_FILE)|((bb << 6) & NOT_GH_FILE)|((bb >> 6) & NOT_AB_FILE)|((bb >> 10) & NOT_GH_FILE)|((bb >> 15) & NOT_A_FILE)|((bb >> 17) & NOT_H_FILE))
        KING_ATTACKS[sq] = (((bb << 1)|(bb << 9)|(bb >> 7)) & NOT_A_FILE)|(((bb >> 1)|(bb >> 9)|(bb << 7)) & NOT_H_FILE)|(bb << 8)|(bb >> 8)
_precompute_leaper_attacks()

PAWN_ATTACKS = np.zeros((2, 64), dtype=np.uint64)
def _precompute_pawn_attacks():
    for sq in range(64):
        bb_sq = BB_SQUARES[sq]
        # White pawn attackers for square sq
        PAWN_ATTACKS[WHITE, sq] = ((bb_sq & NOT_A_FILE) >> 9) | ((bb_sq & NOT_H_FILE) >> 7)
        # Black pawn attackers for square sq
        PAWN_ATTACKS[BLACK, sq] = ((bb_sq & NOT_H_FILE) << 9) | ((bb_sq & NOT_A_FILE) << 7)
_precompute_pawn_attacks()



GLOBAL_ATTACK_TABLES = (
    PAWN_ATTACKS, KNIGHT_ATTACKS, KING_ATTACKS,
    BISHOP_MASKS, ROOK_MASKS, BISHOP_MAGIC_NUMBERS, ROOK_MAGIC_NUMBERS,
    BISHOP_RELEVANT_BITS, ROOK_RELEVANT_BITS,
    BISHOP_ATTACKS, ROOK_ATTACKS_1, ROOK_ATTACKS_2, ROOK_ATTACKS_3, ROOK_ATTACKS_4,
    SQUARES_BETWEEN, ROOK_RAYS, BISHOP_RAYS
)


@numba.njit(cache=True, boundscheck=False, fastmath=True)
def is_square_attacked(piece_bbs, occupancy_bbs, sq, attacker_side):
    """
    Checks if a given square is attacked by the specified side.
    """
    all_pieces_bb = occupancy_bbs[2]

    if attacker_side == WHITE:
        wp, wn, wb, wr, wq, wk = piece_bbs[0], piece_bbs[1], piece_bbs[2], piece_bbs[3], piece_bbs[4], piece_bbs[5]
        if PAWN_ATTACKS[WHITE, sq] & wp: return True
        if KING_ATTACKS[sq] & wk: return True
        if KNIGHT_ATTACKS[sq] & wn: return True
        if (wb | wq):
            if get_bishop_attacks(sq, all_pieces_bb) & (wb | wq): return True
        if (wr | wq):
            if get_rook_attacks(sq, all_pieces_bb) & (wr | wq): return True
    else: # Attacker is BLACK
        bp, bn, bb, br, bq, bk = piece_bbs[6], piece_bbs[7], piece_bbs[8], piece_bbs[9], piece_bbs[10], piece_bbs[11]
        if PAWN_ATTACKS[BLACK, sq] & bp: return True
        if KING_ATTACKS[sq] & bk: return True
        if KNIGHT_ATTACKS[sq] & bn: return True
        if (bb | bq):
            if get_bishop_attacks(sq, all_pieces_bb) & (bb | bq): return True
        if (br | bq):
            if get_rook_attacks(sq, all_pieces_bb) & (br | bq): return True
        
    return False

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def check_legality_and_gives_check(piece_bbs, occupancy_bbs, our_king_sq, their_king_sq, our_side):
    """
    Combined function to check:
    1. Is the move legal? (our_king_sq is NOT attacked by 1 - our_side)
    2. Does the move give check? (their_king_sq IS attacked by our_side)
    Returns (is_legal, gives_check) as explicit booleans.
    """
    all_pieces_bb = occupancy_bbs[2]
    
    if our_side == WHITE:
        # Our side is WHITE, opponent is BLACK
        bp, bn, bb, br, bq, bk = piece_bbs[6], piece_bbs[7], piece_bbs[8], piece_bbs[9], piece_bbs[10], piece_bbs[11]
        
        # 1. Check Legality (Is our king attacked by black?)
        is_legal = True
        if PAWN_ATTACKS[BLACK, our_king_sq] & bp:
            is_legal = False
        elif KING_ATTACKS[our_king_sq] & bk:
            is_legal = False
        elif KNIGHT_ATTACKS[our_king_sq] & bn:
            is_legal = False
        elif (bb | bq) and (get_bishop_attacks(our_king_sq, all_pieces_bb) & (bb | bq)):
            is_legal = False
        elif (br | bq) and (get_rook_attacks(our_king_sq, all_pieces_bb) & (br | bq)):
            is_legal = False
            
        if not is_legal:
            return False, False
        
        # 2. Check Gives Check (Is their king attacked by white?)
        wp, wn, wb, wr, wq, wk = piece_bbs[0], piece_bbs[1], piece_bbs[2], piece_bbs[3], piece_bbs[4], piece_bbs[5]
        gives_check = False
        if PAWN_ATTACKS[WHITE, their_king_sq] & wp:
            gives_check = True
        elif KING_ATTACKS[their_king_sq] & wk:
            gives_check = True
        elif KNIGHT_ATTACKS[their_king_sq] & wn:
            gives_check = True
        elif (wb | wq) and (get_bishop_attacks(their_king_sq, all_pieces_bb) & (wb | wq)):
            gives_check = True
        elif (wr | wq) and (get_rook_attacks(their_king_sq, all_pieces_bb) & (wr | wq)):
            gives_check = True
            
        return True, gives_check
    else:
        # Our side is BLACK, opponent is WHITE
        wp, wn, wb, wr, wq, wk = piece_bbs[0], piece_bbs[1], piece_bbs[2], piece_bbs[3], piece_bbs[4], piece_bbs[5]
        
        # 1. Check Legality (Is our king attacked by white?)
        is_legal = True
        if PAWN_ATTACKS[WHITE, our_king_sq] & wp:
            is_legal = False
        elif KING_ATTACKS[our_king_sq] & wk:
            is_legal = False
        elif KNIGHT_ATTACKS[our_king_sq] & wn:
            is_legal = False
        elif (wb | wq) and (get_bishop_attacks(our_king_sq, all_pieces_bb) & (wb | wq)):
            is_legal = False
        elif (wr | wq) and (get_rook_attacks(our_king_sq, all_pieces_bb) & (wr | wq)):
            is_legal = False
            
        if not is_legal:
            return False, False
        
        # 2. Check Gives Check (Is their king attacked by black?)
        bp, bn, bb, br, bq, bk = piece_bbs[6], piece_bbs[7], piece_bbs[8], piece_bbs[9], piece_bbs[10], piece_bbs[11]
        gives_check = False
        if PAWN_ATTACKS[BLACK, their_king_sq] & bp:
            gives_check = True
        elif KING_ATTACKS[their_king_sq] & bk:
            gives_check = True
        elif KNIGHT_ATTACKS[their_king_sq] & bn:
            gives_check = True
        elif (bb | bq) and (get_bishop_attacks(their_king_sq, all_pieces_bb) & (bb | bq)):
            gives_check = True
        elif (br | bq) and (get_rook_attacks(their_king_sq, all_pieces_bb) & (br | bq)):
            gives_check = True
            
        return True, gives_check

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def get_pinned_pieces(piece_bbs, occupancy_bbs, side):
    """
    Returns a bitboard of all pieces of 'side' that are pinned to their King
    by enemy sliding pieces.
    Optimized using bitboard operations and precomputed SQUARES_BETWEEN.
    """
    king_idx = KING if side == WHITE else (KING + 6)
    king_bb = piece_bbs[king_idx]
    if not king_bb: return np.uint64(0)
    king_sq = get_lsb_index(king_bb)

    pinned = np.uint64(0)
    occupied = occupancy_bbs[2]
    own_pieces = occupancy_bbs[side]
    enemy_offset = 6 if side == WHITE else 0

    enemy_rooks = piece_bbs[ROOK + enemy_offset] | piece_bbs[QUEEN + enemy_offset]
    enemy_bishops = piece_bbs[BISHOP + enemy_offset] | piece_bbs[QUEEN + enemy_offset]

    # Orthogonal pinners: enemy rooks/queens on same rank or file as king
    pinners = ROOK_RAYS[king_sq] & enemy_rooks
    while pinners:
        pinner_sq = get_lsb_index(pinners)
        between = SQUARES_BETWEEN[king_sq, pinner_sq]
        blockers = between & occupied
        # If exactly one piece between king and pinner, and it belongs to side, it's pinned
        if count_bits(blockers) == 1 and (blockers & own_pieces):
            pinned |= blockers
        pinners &= pinners - np.uint64(1)

    # Diagonal pinners: enemy bishops/queens on same diagonal as king
    pinners = BISHOP_RAYS[king_sq] & enemy_bishops
    while pinners:
        pinner_sq = get_lsb_index(pinners)
        between = SQUARES_BETWEEN[king_sq, pinner_sq]
        blockers = between & occupied
        if count_bits(blockers) == 1 and (blockers & own_pieces):
            pinned |= blockers
        pinners &= pinners - np.uint64(1)

    return pinned

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def is_move_pseudo_legal(piece_bbs, occupancy_bbs, game_state, move):
    """
    Checks if a given move is pseudo-legal in the current position.
    This is much faster than generating all moves and checking if it's in the list.
    """
    if move == 0:
        return False

    from_sq = get_from_square(move)
    to_sq = get_to_square(move)
    special_flag = get_special_move_flag(move)
    side_to_move = game_state[0]

    # Optimization: Unpack occupancy
    white_pieces_bb = occupancy_bbs[0]
    black_pieces_bb = occupancy_bbs[1]
    all_pieces_bb = occupancy_bbs[2]

    own_pieces_bb = white_pieces_bb if side_to_move == WHITE else black_pieces_bb
    opponent_pieces_bb = black_pieces_bb if side_to_move == WHITE else white_pieces_bb

    from_bb = BB_SQUARES[from_sq]
    to_bb = BB_SQUARES[to_sq]

    # The piece must belong to the side to move
    if not (from_bb & own_pieces_bb):
        return False

    # Cannot capture own piece
    if to_bb & own_pieces_bb:
        return False

    # Find the piece type
    piece_type = -1
    offset = 0 if side_to_move == WHITE else 6
    for pt in range(6):
        if from_bb & piece_bbs[pt + offset]:
            piece_type = pt
            break
            
    if piece_type == -1:
        return False

    # --- Castling ---
    if special_flag == SPECIAL_MOVE_FLAG_CASTLING:
        if piece_type != KING:
            return False
            
        castling_rights = game_state[1]
        if side_to_move == WHITE:
            if from_sq != 4: return False
            if to_sq == 6: # Kingside
                if not (castling_rights & 1): return False
                if all_pieces_bb & 0x60: return False
                if is_square_attacked(piece_bbs, occupancy_bbs, 4, BLACK): return False
                if is_square_attacked(piece_bbs, occupancy_bbs, 5, BLACK): return False
                if is_square_attacked(piece_bbs, occupancy_bbs, 6, BLACK): return False
                return True
            elif to_sq == 2: # Queenside
                if not (castling_rights & 2): return False
                if all_pieces_bb & 0xe: return False
                if is_square_attacked(piece_bbs, occupancy_bbs, 4, BLACK): return False
                if is_square_attacked(piece_bbs, occupancy_bbs, 3, BLACK): return False
                if is_square_attacked(piece_bbs, occupancy_bbs, 2, BLACK): return False
                return True
            else:
                return False
        else: # BLACK
            if from_sq != 60: return False
            if to_sq == 62: # Kingside
                if not (castling_rights & 4): return False
                if all_pieces_bb & 0x6000000000000000: return False
                if is_square_attacked(piece_bbs, occupancy_bbs, 60, WHITE): return False
                if is_square_attacked(piece_bbs, occupancy_bbs, 61, WHITE): return False
                if is_square_attacked(piece_bbs, occupancy_bbs, 62, WHITE): return False
                return True
            elif to_sq == 58: # Queenside
                if not (castling_rights & 8): return False
                if all_pieces_bb & 0xe00000000000000: return False
                if is_square_attacked(piece_bbs, occupancy_bbs, 60, WHITE): return False
                if is_square_attacked(piece_bbs, occupancy_bbs, 59, WHITE): return False
                if is_square_attacked(piece_bbs, occupancy_bbs, 58, WHITE): return False
                return True
            else:
                return False

    # --- Pawns ---
    if piece_type == PAWN:
        if side_to_move == WHITE:
            dir = 8
            start_rank = 1
            promo_rank = 7
            attacks = PAWN_ATTACKS[WHITE, to_sq] # These are backwards attacks, meaning from where pawn could have come
        else:
            dir = -8
            start_rank = 6
            promo_rank = 0
            attacks = PAWN_ATTACKS[BLACK, to_sq]
            
        is_promo = (to_sq // 8 == promo_rank)
        
        # Check promotion flag matches destination rank
        if is_promo and special_flag != SPECIAL_MOVE_FLAG_PROMOTION: return False
        if not is_promo and special_flag == SPECIAL_MOVE_FLAG_PROMOTION: return False

        if special_flag == SPECIAL_MOVE_FLAG_EN_PASSANT:
            ep_sq = game_state[2]
            if to_sq != ep_sq: return False
            # Check if pawn can attack ep square
            return (attacks & from_bb) != 0

        # Push
        if from_sq + dir == to_sq:
            if to_bb & all_pieces_bb: return False
            return True
            
        # Double push
        if from_sq // 8 == start_rank and from_sq + dir * 2 == to_sq:
            if (BB_SQUARES[from_sq + dir] | to_bb) & all_pieces_bb: return False
            return True

        # Capture
        if (attacks & from_bb):
            if to_bb & opponent_pieces_bb: return True
            return False

        return False

    # --- En Passant flag on non-pawn ---
    if special_flag == SPECIAL_MOVE_FLAG_EN_PASSANT:
        return False
        
    # --- Promotion flag on non-pawn ---
    if special_flag == SPECIAL_MOVE_FLAG_PROMOTION:
        return False

    # --- Leapers ---
    if piece_type == KNIGHT:
        return (KNIGHT_ATTACKS[from_sq] & to_bb) != 0
    if piece_type == KING:
        return (KING_ATTACKS[from_sq] & to_bb) != 0

    # --- Sliders ---
    if piece_type == BISHOP:
        return (get_bishop_attacks(from_sq, all_pieces_bb) & to_bb) != 0
    if piece_type == ROOK:
        return (get_rook_attacks(from_sq, all_pieces_bb) & to_bb) != 0
    if piece_type == QUEEN:
        return (get_queen_attacks(from_sq, all_pieces_bb) & to_bb) != 0

    return False

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def generate_pseudo_legal_moves_buffer(piece_bbs, occupancy_bbs, game_state, moves_buffer, ply):
    """
    Generates all pseudo-legal moves for the current position into the provided buffer at the given ply.
    Returns the count of legal moves.
    """
    move_count = 0
    
    side_to_move = game_state[0]
    castling_rights = game_state[1]
    en_passant_square = game_state[2]
    
    (wp_bb, wn_bb, wb_bb, wr_bb, wq_bb, wk_bb, 
     bp_bb, bn_bb, bb_bb, br_bb, bq_bb, bk_bb) = piece_bbs
     
    # Optimization: Unpack all 3 elements. occupancy_bbs[2] is the pre-calculated union of white and black pieces.
    white_pieces_bb, black_pieces_bb, all_pieces_bb = occupancy_bbs
 
    own_pieces_bb = white_pieces_bb if side_to_move == WHITE else black_pieces_bb
    opponent_pieces_bb = black_pieces_bb if side_to_move == WHITE else white_pieces_bb
    empty_squares = ~all_pieces_bb
    WK, WQ, BK, BQ = 1, 2, 4, 8
 
    # --- Pseudo-legal move generation ---
    
    # (The pseudo-legal move generation logic is copied here and adapted)
    if side_to_move == WHITE:
        # --- Pawn Moves ---
        single_pushes = (wp_bb << 8) & empty_squares
        double_pushes = ((single_pushes & RANK_3) << 8) & empty_squares
        captures_west = ((wp_bb & NOT_A_FILE) << 7) & opponent_pieces_bb
        captures_east = ((wp_bb & NOT_H_FILE) << 9) & opponent_pieces_bb
 
        # Single and Double Pushes
        pushes = single_pushes
        while pushes:
            to_sq = get_lsb_index(pushes)
            from_sq = to_sq - 8
            if to_sq >= 56: # Promotion
                for p_type in [PROMO_QUEEN, PROMO_ROOK, PROMO_BISHOP, PROMO_KNIGHT]: moves_buffer[ply, move_count] = encode_move(from_sq, to_sq, p_type, SPECIAL_MOVE_FLAG_PROMOTION); move_count += 1
            else:
                moves_buffer[ply, move_count] = encode_move(from_sq, to_sq, 0, SPECIAL_MOVE_FLAG_NORMAL); move_count += 1
            pushes &= (pushes - np.uint64(1))
        
        pushes = double_pushes
        while pushes:
            to_sq = get_lsb_index(pushes)
            from_sq = to_sq - 16
            moves_buffer[ply, move_count] = encode_move(from_sq, to_sq, 0, SPECIAL_MOVE_FLAG_NORMAL); move_count += 1
            pushes &= (pushes - np.uint64(1))
 
        # Captures
        caps_west = captures_west
        while caps_west:
            to_sq = get_lsb_index(caps_west)
            from_sq = to_sq - 7
            if to_sq >= 56: # Promotion
                for p_type in [PROMO_QUEEN, PROMO_ROOK, PROMO_BISHOP, PROMO_KNIGHT]: moves_buffer[ply, move_count] = encode_move(from_sq, to_sq, p_type, SPECIAL_MOVE_FLAG_PROMOTION); move_count += 1
            else:
                moves_buffer[ply, move_count] = encode_move(from_sq, to_sq, 0, SPECIAL_MOVE_FLAG_NORMAL); move_count += 1
            caps_west &= (caps_west - np.uint64(1))
 
        caps_east = captures_east
        while caps_east:
            to_sq = get_lsb_index(caps_east)
            from_sq = to_sq - 9
            if to_sq >= 56: # Promotion
                for p_type in [PROMO_QUEEN, PROMO_ROOK, PROMO_BISHOP, PROMO_KNIGHT]: moves_buffer[ply, move_count] = encode_move(from_sq, to_sq, p_type, SPECIAL_MOVE_FLAG_PROMOTION); move_count += 1
            else:
                moves_buffer[ply, move_count] = encode_move(from_sq, to_sq, 0, SPECIAL_MOVE_FLAG_NORMAL); move_count += 1
            caps_east &= (caps_east - np.uint64(1))
 
        # En Passant
        if en_passant_square != 64:
            ep_target_bb = BB_SQUARES[np.uint8(en_passant_square)]
            west_attackers = ((wp_bb & NOT_A_FILE) << 7) & ep_target_bb
            if west_attackers:
                from_sq = np.uint8(en_passant_square - 7)
                moves_buffer[ply, move_count] = encode_move(from_sq, np.uint8(en_passant_square), 0, SPECIAL_MOVE_FLAG_EN_PASSANT); move_count += 1
            east_attackers = ((wp_bb & NOT_H_FILE) << 9) & ep_target_bb
            if east_attackers:
                from_sq = np.uint8(en_passant_square - 9)
                moves_buffer[ply, move_count] = encode_move(from_sq, np.uint8(en_passant_square), 0, SPECIAL_MOVE_FLAG_EN_PASSANT); move_count += 1
 
        # --- Castling ---
        if (castling_rights & WK) and not(all_pieces_bb & 0x60) and not is_square_attacked(piece_bbs, occupancy_bbs, 4, BLACK) and not is_square_attacked(piece_bbs, occupancy_bbs, 5, BLACK) and not is_square_attacked(piece_bbs, occupancy_bbs, 6, BLACK): moves_buffer[ply, move_count]=encode_move(4,6,0,SPECIAL_MOVE_FLAG_CASTLING); move_count+=1
        if (castling_rights & WQ) and not(all_pieces_bb & 0xe) and not is_square_attacked(piece_bbs, occupancy_bbs, 4, BLACK) and not is_square_attacked(piece_bbs, occupancy_bbs, 3, BLACK) and not is_square_attacked(piece_bbs, occupancy_bbs, 2, BLACK): moves_buffer[ply, move_count]=encode_move(4,2,0,SPECIAL_MOVE_FLAG_CASTLING); move_count+=1
 
    else: # BLACK
        # --- Pawn Moves ---
        single_pushes = (bp_bb >> 8) & empty_squares
        double_pushes = ((single_pushes & RANK_6) >> 8) & empty_squares
        captures_west = ((bp_bb & NOT_H_FILE) >> 7) & opponent_pieces_bb
        captures_east = ((bp_bb & NOT_A_FILE) >> 9) & opponent_pieces_bb
 
        pushes = single_pushes
        while pushes:
            to_sq = get_lsb_index(pushes)
            from_sq = to_sq + 8
            if to_sq <= 7: # Promotion
                for p_type in [PROMO_QUEEN, PROMO_ROOK, PROMO_BISHOP, PROMO_KNIGHT]: moves_buffer[ply, move_count] = encode_move(from_sq, to_sq, p_type, SPECIAL_MOVE_FLAG_PROMOTION); move_count += 1
            else:
                moves_buffer[ply, move_count] = encode_move(from_sq, to_sq, 0, SPECIAL_MOVE_FLAG_NORMAL); move_count += 1
            pushes &= (pushes - np.uint64(1))
        
        pushes = double_pushes
        while pushes:
            to_sq = get_lsb_index(pushes)
            from_sq = to_sq + 16
            moves_buffer[ply, move_count] = encode_move(from_sq, to_sq, 0, SPECIAL_MOVE_FLAG_NORMAL); move_count += 1
            pushes &= (pushes - np.uint64(1))
 
        caps_west = captures_west
        while caps_west:
            to_sq = get_lsb_index(caps_west)
            from_sq = to_sq + 7
            if to_sq <= 7: # Promotion
                for p_type in [PROMO_QUEEN, PROMO_ROOK, PROMO_BISHOP, PROMO_KNIGHT]: moves_buffer[ply, move_count] = encode_move(from_sq, to_sq, p_type, SPECIAL_MOVE_FLAG_PROMOTION); move_count += 1
            else:
                moves_buffer[ply, move_count] = encode_move(from_sq, to_sq, 0, SPECIAL_MOVE_FLAG_NORMAL); move_count += 1
            caps_west &= (caps_west - np.uint64(1))
 
        caps_east = captures_east
        while caps_east:
            to_sq = get_lsb_index(caps_east)
            from_sq = to_sq + 9
            if to_sq <= 7: # Promotion
                for p_type in [PROMO_QUEEN, PROMO_ROOK, PROMO_BISHOP, PROMO_KNIGHT]: moves_buffer[ply, move_count] = encode_move(from_sq, to_sq, p_type, SPECIAL_MOVE_FLAG_PROMOTION); move_count += 1
            else:
                moves_buffer[ply, move_count] = encode_move(from_sq, to_sq, 0, SPECIAL_MOVE_FLAG_NORMAL); move_count += 1
            caps_east &= (caps_east - np.uint64(1))
 
        # En Passant
        if en_passant_square != 64:
            ep_target_bb = BB_SQUARES[np.uint8(en_passant_square)]
            west_attackers = ((bp_bb & NOT_H_FILE) >> 7) & ep_target_bb
            if west_attackers:
                from_sq = np.uint8(en_passant_square + 7)
                moves_buffer[ply, move_count] = encode_move(from_sq, np.uint8(en_passant_square), 0, SPECIAL_MOVE_FLAG_EN_PASSANT); move_count += 1
            east_attackers = ((bp_bb & NOT_A_FILE) >> 9) & ep_target_bb
            if east_attackers:
                from_sq = np.uint8(en_passant_square + 9)
                moves_buffer[ply, move_count] = encode_move(from_sq, np.uint8(en_passant_square), 0, SPECIAL_MOVE_FLAG_EN_PASSANT); move_count += 1
 
        # --- Castling ---
        if (castling_rights & BK) and not(all_pieces_bb & 0x6000000000000000) and not is_square_attacked(piece_bbs, occupancy_bbs, 60, WHITE) and not is_square_attacked(piece_bbs, occupancy_bbs, 61, WHITE) and not is_square_attacked(piece_bbs, occupancy_bbs, 62, WHITE): moves_buffer[ply, move_count]=encode_move(60,62,0,SPECIAL_MOVE_FLAG_CASTLING); move_count+=1
        if (castling_rights & BQ) and not(all_pieces_bb & 0xe00000000000000) and not is_square_attacked(piece_bbs, occupancy_bbs, 60, WHITE) and not is_square_attacked(piece_bbs, occupancy_bbs, 59, WHITE) and not is_square_attacked(piece_bbs, occupancy_bbs, 58, WHITE): moves_buffer[ply, move_count]=encode_move(60,58,0,SPECIAL_MOVE_FLAG_CASTLING); move_count+=1
 
    # --- Leaper Moves (Knights, Bishops, Rooks, Queens, Kings) ---
    leaper_bbs = (wn_bb, wb_bb, wr_bb, wq_bb, wk_bb) if side_to_move == WHITE else (bn_bb, bb_bb, br_bb, bq_bb, bk_bb)
    for piece_type in range(5):
        bb = leaper_bbs[piece_type]
        while bb:
            from_sq = get_lsb_index(bb)
            if piece_type == 0: targets = KNIGHT_ATTACKS[from_sq] & ~own_pieces_bb
            elif piece_type == 1: targets = get_bishop_attacks(from_sq, all_pieces_bb) & ~own_pieces_bb
            elif piece_type == 2: targets = get_rook_attacks(from_sq, all_pieces_bb) & ~own_pieces_bb
            elif piece_type == 3: targets = get_queen_attacks(from_sq, all_pieces_bb) & ~own_pieces_bb
            else: targets = KING_ATTACKS[from_sq] & ~own_pieces_bb
            
            while targets:
                to_sq = get_lsb_index(targets)
                moves_buffer[ply, move_count] = encode_move(from_sq, to_sq, 0, SPECIAL_MOVE_FLAG_NORMAL); move_count+=1
                targets &= (targets - np.uint64(1))
            bb &= (bb - np.uint64(1))
     
    return move_count
 
@numba.njit(cache=True, boundscheck=False, fastmath=True)
def generate_legal_moves_buffer(piece_bbs, occupancy_bbs, game_state, moves_buffer, ply):
    """
    Generates all fully legal moves for the current position into the provided buffer at the given ply.
    Returns the count of legal moves.
    """
    move_count = generate_pseudo_legal_moves_buffer(piece_bbs, occupancy_bbs, game_state, moves_buffer, ply)
    side_to_move = game_state[0]
 
    # --- Filter for legality ---
    legal_move_count = 0
 
    king_bb = piece_bbs[5] if side_to_move == WHITE else piece_bbs[11]
    original_king_sq = get_lsb_index(king_bb) if king_bb else 0
 
    # Optimization: Pre-calculate pinned pieces and check status to skip expensive make/unmake
    opponent_side = 1 - side_to_move
    in_check = is_square_attacked(piece_bbs, occupancy_bbs, original_king_sq, opponent_side)
    pinned = get_pinned_pieces(piece_bbs, occupancy_bbs, side_to_move) if not in_check else np.uint64(0)
 
    for i in range(move_count):
        move = moves_buffer[ply, i]
        from_sq = get_from_square(move)
        
        # Fast path: if not in check, not pinned, not a king move, and not en passant, it MUST be legal.
        if not in_check and not (pinned & BB_SQUARES[from_sq]) and from_sq != original_king_sq and get_special_move_flag(move) != SPECIAL_MOVE_FLAG_EN_PASSANT:
            moves_buffer[ply, legal_move_count] = move
            legal_move_count += 1
            continue
 
        # Slow path: full legality check via make/unmake
        unmake_info = make_move(piece_bbs, occupancy_bbs, game_state, move)
 
        # After the move, the 'side_to_move' in game_state is the opponent.
        # We need to check if the ORIGINAL side's king is attacked.
        king_bb_after_move = piece_bbs[5] if side_to_move == WHITE else piece_bbs[11]
        king_sq = get_lsb_index(king_bb_after_move) if king_bb_after_move else original_king_sq
 
        # Check if the king is attacked by the new side to move (the opponent)
        if not is_square_attacked(piece_bbs, occupancy_bbs, king_sq, game_state[0]):
            moves_buffer[ply, legal_move_count] = move
            legal_move_count += 1
 
        # Unmake the move to restore the board state for the next iteration
        unmake_move(piece_bbs, occupancy_bbs, game_state, move, unmake_info)
                 
    return legal_move_count
 
 
def generate_legal_moves(piece_bbs, occupancy_bbs, game_state):
    """
    Wrapper for generate_legal_moves_buffer to maintain compatibility.
    Allocates a new array.
    """
    moves = np.zeros((1, 256), dtype=np.uint16)
    count = generate_legal_moves_buffer(piece_bbs, occupancy_bbs, game_state, moves, 0)
    return moves[0, :count]


@numba.njit(cache=True, boundscheck=False, fastmath=True)
def _generate_tactical_moves_jit(piece_bbs, occupancy_bbs, game_state):
    """
    Generates pseudo-legal tactical moves (all captures and promotions).
    """
    moves = np.zeros(128, dtype=np.uint16)
    move_count = 0

    side_to_move = game_state[0]
    en_passant_square = game_state[2]
    
    (wp_bb, wn_bb, wb_bb, wr_bb, wq_bb, wk_bb, 
     bp_bb, bn_bb, bb_bb, br_bb, bq_bb, bk_bb) = piece_bbs
     
    # Optimization: Unpack all 3 elements. occupancy_bbs[2] is the pre-calculated union of white and black pieces.
    white_pieces_bb, black_pieces_bb, all_pieces_bb = occupancy_bbs

    opponent_pieces_bb = black_pieces_bb if side_to_move == WHITE else white_pieces_bb
    own_pieces_bb = white_pieces_bb if side_to_move == WHITE else black_pieces_bb

    if side_to_move == WHITE:
        # --- Pawn Captures & Promotions ---
        single_pushes = (wp_bb << 8) & ~all_pieces_bb
        captures_west = ((wp_bb & NOT_A_FILE) << 7) & opponent_pieces_bb
        captures_east = ((wp_bb & NOT_H_FILE) << 9) & opponent_pieces_bb

        # Promotion via single push
        promo_pushes = single_pushes & RANK_8
        while promo_pushes:
            to_sq = get_lsb_index(promo_pushes)
            from_sq = to_sq - 8
            for p_type in [PROMO_QUEEN, PROMO_ROOK, PROMO_BISHOP, PROMO_KNIGHT]:
                moves[move_count] = encode_move(from_sq, to_sq, p_type, SPECIAL_MOVE_FLAG_PROMOTION)
                move_count += 1
            promo_pushes &= (promo_pushes - np.uint64(1))

        # Capture Promotions
        promo_caps_west = captures_west & RANK_8
        while promo_caps_west:
            to_sq = get_lsb_index(promo_caps_west)
            from_sq = to_sq - 7
            for p_type in [PROMO_QUEEN, PROMO_ROOK, PROMO_BISHOP, PROMO_KNIGHT]:
                moves[move_count] = encode_move(from_sq, to_sq, p_type, SPECIAL_MOVE_FLAG_PROMOTION)
                move_count += 1
            promo_caps_west &= (promo_caps_west - np.uint64(1))
            
        promo_caps_east = captures_east & RANK_8
        while promo_caps_east:
            to_sq = get_lsb_index(promo_caps_east)
            from_sq = to_sq - 9
            for p_type in [PROMO_QUEEN, PROMO_ROOK, PROMO_BISHOP, PROMO_KNIGHT]:
                moves[move_count] = encode_move(from_sq, to_sq, p_type, SPECIAL_MOVE_FLAG_PROMOTION)
                move_count += 1
            promo_caps_east &= (promo_caps_east - np.uint64(1))

        # Regular Captures
        caps_west = captures_west & ~RANK_8
        while caps_west:
            to_sq = get_lsb_index(caps_west)
            from_sq = to_sq - 7
            moves[move_count] = encode_move(from_sq, to_sq, 0, SPECIAL_MOVE_FLAG_NORMAL)
            move_count += 1
            caps_west &= (caps_west - np.uint64(1))

        caps_east = captures_east & ~RANK_8
        while caps_east:
            to_sq = get_lsb_index(caps_east)
            from_sq = to_sq - 9
            moves[move_count] = encode_move(from_sq, to_sq, 0, SPECIAL_MOVE_FLAG_NORMAL)
            move_count += 1
            caps_east &= (caps_east - np.uint64(1))

        # En Passant
        if en_passant_square != 64:
            ep_target_bb = BB_SQUARES[np.uint8(en_passant_square)]
            west_attackers = ((wp_bb & NOT_A_FILE) << 7) & ep_target_bb
            if west_attackers:
                from_sq = np.uint8(en_passant_square - 7)
                moves[move_count] = encode_move(from_sq, np.uint8(en_passant_square), 0, SPECIAL_MOVE_FLAG_EN_PASSANT); move_count += 1
            east_attackers = ((wp_bb & NOT_H_FILE) << 9) & ep_target_bb
            if east_attackers:
                from_sq = np.uint8(en_passant_square - 9)
                moves[move_count] = encode_move(from_sq, np.uint8(en_passant_square), 0, SPECIAL_MOVE_FLAG_EN_PASSANT); move_count += 1
    
    else: # BLACK
        # --- Pawn Captures & Promotions ---
        single_pushes = (bp_bb >> 8) & ~all_pieces_bb
        captures_west = ((bp_bb & NOT_H_FILE) >> 7) & opponent_pieces_bb
        captures_east = ((bp_bb & NOT_A_FILE) >> 9) & opponent_pieces_bb

        # Promotion via single push
        promo_pushes = single_pushes & RANK_1
        while promo_pushes:
            to_sq = get_lsb_index(promo_pushes)
            from_sq = to_sq + 8
            for p_type in [PROMO_QUEEN, PROMO_ROOK, PROMO_BISHOP, PROMO_KNIGHT]:
                moves[move_count] = encode_move(from_sq, to_sq, p_type, SPECIAL_MOVE_FLAG_PROMOTION)
                move_count += 1
            promo_pushes &= (promo_pushes - np.uint64(1))

        # Capture Promotions
        promo_caps_west = captures_west & RANK_1
        while promo_caps_west:
            to_sq = get_lsb_index(promo_caps_west)
            from_sq = to_sq + 7
            for p_type in [PROMO_QUEEN, PROMO_ROOK, PROMO_BISHOP, PROMO_KNIGHT]:
                moves[move_count] = encode_move(from_sq, to_sq, p_type, SPECIAL_MOVE_FLAG_PROMOTION)
                move_count += 1
            promo_caps_west &= (promo_caps_west - np.uint64(1))
            
        promo_caps_east = captures_east & RANK_1
        while promo_caps_east:
            to_sq = get_lsb_index(promo_caps_east)
            from_sq = to_sq + 9
            for p_type in [PROMO_QUEEN, PROMO_ROOK, PROMO_BISHOP, PROMO_KNIGHT]:
                moves[move_count] = encode_move(from_sq, to_sq, p_type, SPECIAL_MOVE_FLAG_PROMOTION)
                move_count += 1
            promo_caps_east &= (promo_caps_east - np.uint64(1))

        # Regular Captures
        caps_west = captures_west & ~RANK_1
        while caps_west:
            to_sq = get_lsb_index(caps_west)
            from_sq = to_sq + 7
            moves[move_count] = encode_move(from_sq, to_sq, 0, SPECIAL_MOVE_FLAG_NORMAL)
            move_count += 1
            caps_west &= (caps_west - np.uint64(1))

        caps_east = captures_east & ~RANK_1
        while caps_east:
            to_sq = get_lsb_index(caps_east)
            from_sq = to_sq + 9
            moves[move_count] = encode_move(from_sq, to_sq, 0, SPECIAL_MOVE_FLAG_NORMAL)
            move_count += 1
            caps_east &= (caps_east - np.uint64(1))

        # En Passant
        if en_passant_square != 64:
            ep_target_bb = BB_SQUARES[np.uint8(en_passant_square)]
            west_attackers = ((bp_bb & NOT_H_FILE) >> 7) & ep_target_bb
            if west_attackers:
                from_sq = np.uint8(en_passant_square + 7)
                moves[move_count] = encode_move(from_sq, np.uint8(en_passant_square), 0, SPECIAL_MOVE_FLAG_EN_PASSANT); move_count += 1
            east_attackers = ((bp_bb & NOT_A_FILE) >> 9) & ep_target_bb
            if east_attackers:
                from_sq = np.uint8(en_passant_square + 9)
                moves[move_count] = encode_move(from_sq, np.uint8(en_passant_square), 0, SPECIAL_MOVE_FLAG_EN_PASSANT); move_count += 1

    # --- Leaper Captures (Knights, Bishops, Rooks, Queens, Kings) ---
    leaper_bbs = (wn_bb, wb_bb, wr_bb, wq_bb, wk_bb) if side_to_move == WHITE else (bn_bb, bb_bb, br_bb, bq_bb, bk_bb)
    for piece_type in range(5):
        bb = leaper_bbs[piece_type]
        while bb:
            from_sq = get_lsb_index(bb)
            if piece_type == 0: targets = KNIGHT_ATTACKS[from_sq]
            elif piece_type == 1: targets = get_bishop_attacks(from_sq, all_pieces_bb)
            elif piece_type == 2: targets = get_rook_attacks(from_sq, all_pieces_bb)
            elif piece_type == 3: targets = get_queen_attacks(from_sq, all_pieces_bb)
            else: targets = KING_ATTACKS[from_sq]
            
            # --- This is the key difference: only consider captures ---
            capture_targets = targets & opponent_pieces_bb
            
            while capture_targets:
                to_sq = get_lsb_index(capture_targets)
                moves[move_count] = encode_move(from_sq, to_sq, 0, SPECIAL_MOVE_FLAG_NORMAL)
                move_count += 1
                capture_targets &= (capture_targets - np.uint64(1))
            bb &= (bb - np.uint64(1))
            
    return moves[:move_count]

def generate_tactical_moves(piece_bbs, occupancy_bbs, game_state):
    """
    Generates pseudo-legal tactical moves (all captures and promotions).
    """
    return _generate_tactical_moves_jit(piece_bbs, occupancy_bbs, game_state)

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def _is_in_check_jit(piece_bbs, occupancy_bbs, game_state):
    """
    Checks if the current side to move is in check.
    """
    side_to_move = game_state[0]
    king_bb = piece_bbs[5] if side_to_move == WHITE else piece_bbs[11]
    if king_bb == 0: # Should not happen in a legal position
        return False
    king_sq = get_lsb_index(king_bb)
    return is_square_attacked(piece_bbs, occupancy_bbs, king_sq, 1 - side_to_move)

def is_in_check(piece_bbs, occupancy_bbs, game_state):
    return _is_in_check_jit(piece_bbs, occupancy_bbs, game_state)

@numba.njit(numba.int32(numba.uint64), cache=True, inline='always')
def count_bits_local(bb: np.uint64) -> numba.int32:
    # Standard software popcount that compiles to the hardware popcnt instruction in LLVM
    bb = bb - ((bb >> np.uint64(1)) & np.uint64(0x5555555555555555))
    bb = (bb & np.uint64(0x3333333333333333)) + ((bb >> np.uint64(2)) & np.uint64(0x3333333333333333))
    bb = (bb + (bb >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    bb = bb + (bb >> np.uint64(8))
    bb = bb + (bb >> np.uint64(16))
    bb = bb + (bb >> np.uint64(32))
    return numba.int32(bb & np.uint64(0x7F))

@numba.njit(numba.boolean(piece_bbs_signature), cache=True, boundscheck=False, fastmath=True)
def has_sufficient_material(piece_bbs):
    """
    Checks if either side has sufficient mating material.
    If not, the position is a draw by insufficient material.
    """
    # Pawns, Rooks, Queens are sufficient material.
    if (piece_bbs[0] | piece_bbs[3] | piece_bbs[4] |
        piece_bbs[6] | piece_bbs[9] | piece_bbs[10]) != np.uint64(0):
        return True

    # Count minor pieces (Knights, Bishops) using hardware-accelerated local popcount
    w_knights = piece_bbs[1]
    w_bishops = piece_bbs[2]
    b_knights = piece_bbs[7]
    b_bishops = piece_bbs[8]

    wn_cnt = count_bits_local(w_knights)
    wb_cnt = count_bits_local(w_bishops)
    bn_cnt = count_bits_local(b_knights)
    bb_cnt = count_bits_local(b_bishops)

    # Either side has at least two minor pieces -> sufficient.
    if (wn_cnt + wb_cnt > 1) or (bn_cnt + bb_cnt > 1):
        return True

    # One bishop each: opposite colors -> sufficient, same color -> insufficient.
    if wb_cnt == 1 and bb_cnt == 1 and wn_cnt == 0 and bn_cnt == 0:
        light_squares = np.uint64(0x55AA55AA55AA55AA)
        w_light = (w_bishops & light_squares) != np.uint64(0)
        b_light = (b_bishops & light_squares) != np.uint64(0)
        if w_light == b_light:
            return False  # Same color bishops
        return True  # Opposite color bishops

    # King + Bishop vs King, King + Knight vs King, King vs King -> insufficient.
    return False


@numba.njit(cache=True, boundscheck=False, fastmath=True)
def generate_pseudo_legal_quiets_buffer(piece_bbs, occupancy_bbs, game_state, moves_buffer, ply, start_idx):
    """
    Generates all pseudo-legal quiet moves (non-captures, non-promotions) for the current position into the provided buffer starting at start_idx.
    Returns the updated total move count.
    """
    move_count = start_idx
    
    side_to_move = game_state[0]
    castling_rights = game_state[1]
    en_passant_square = game_state[2]
    
    (wp_bb, wn_bb, wb_bb, wr_bb, wq_bb, wk_bb, 
     bp_bb, bn_bb, bb_bb, br_bb, bq_bb, bk_bb) = piece_bbs
     
    # Optimization: Unpack all 3 elements. occupancy_bbs[2] is the pre-calculated union of white and black pieces.
    white_pieces_bb, black_pieces_bb, all_pieces_bb = occupancy_bbs

    own_pieces_bb = white_pieces_bb if side_to_move == WHITE else black_pieces_bb
    empty_squares = ~all_pieces_bb
    WK, WQ, BK, BQ = 1, 2, 4, 8

    # --- Pseudo-legal quiet move generation ---
    if side_to_move == WHITE:
        # --- Pawn Moves ---
        single_pushes = (wp_bb << 8) & empty_squares
        double_pushes = ((single_pushes & RANK_3) << 8) & empty_squares

        # Single Pushes (including quiet promotions)
        pushes = single_pushes
        while pushes:
            to_sq = get_lsb_index(pushes)
            from_sq = to_sq - 8
            if to_sq >= 56: # Promotion
                for p_type in [PROMO_QUEEN, PROMO_ROOK, PROMO_BISHOP, PROMO_KNIGHT]:
                    moves_buffer[ply, move_count] = encode_move(from_sq, to_sq, p_type, SPECIAL_MOVE_FLAG_PROMOTION)
                    move_count += 1
            else:
                moves_buffer[ply, move_count] = encode_move(from_sq, to_sq, 0, SPECIAL_MOVE_FLAG_NORMAL)
                move_count += 1
            pushes &= (pushes - np.uint64(1))
        
        # Double Pushes
        pushes = double_pushes
        while pushes:
            to_sq = get_lsb_index(pushes)
            from_sq = to_sq - 16
            moves_buffer[ply, move_count] = encode_move(from_sq, to_sq, 0, SPECIAL_MOVE_FLAG_NORMAL); move_count += 1
            pushes &= (pushes - np.uint64(1))

        # --- Castling ---
        if (castling_rights & WK) and not(all_pieces_bb & 0x60) and not is_square_attacked(piece_bbs, occupancy_bbs, 4, BLACK) and not is_square_attacked(piece_bbs, occupancy_bbs, 5, BLACK) and not is_square_attacked(piece_bbs, occupancy_bbs, 6, BLACK): moves_buffer[ply, move_count]=encode_move(4,6,0,SPECIAL_MOVE_FLAG_CASTLING); move_count+=1
        if (castling_rights & WQ) and not(all_pieces_bb & 0xe) and not is_square_attacked(piece_bbs, occupancy_bbs, 4, BLACK) and not is_square_attacked(piece_bbs, occupancy_bbs, 3, BLACK) and not is_square_attacked(piece_bbs, occupancy_bbs, 2, BLACK): moves_buffer[ply, move_count]=encode_move(4,2,0,SPECIAL_MOVE_FLAG_CASTLING); move_count+=1

    else: # BLACK
        # --- Pawn Moves ---
        single_pushes = (bp_bb >> 8) & empty_squares
        double_pushes = ((single_pushes & RANK_6) >> 8) & empty_squares

        # Single Pushes (including quiet promotions)
        pushes = single_pushes
        while pushes:
            to_sq = get_lsb_index(pushes)
            from_sq = to_sq + 8
            if to_sq <= 7: # Promotion
                for p_type in [PROMO_QUEEN, PROMO_ROOK, PROMO_BISHOP, PROMO_KNIGHT]:
                    moves_buffer[ply, move_count] = encode_move(from_sq, to_sq, p_type, SPECIAL_MOVE_FLAG_PROMOTION)
                    move_count += 1
            else:
                moves_buffer[ply, move_count] = encode_move(from_sq, to_sq, 0, SPECIAL_MOVE_FLAG_NORMAL)
                move_count += 1
            pushes &= (pushes - np.uint64(1))
        
        # Double Pushes
        pushes = double_pushes
        while pushes:
            to_sq = get_lsb_index(pushes)
            from_sq = to_sq + 16
            moves_buffer[ply, move_count] = encode_move(from_sq, to_sq, 0, SPECIAL_MOVE_FLAG_NORMAL); move_count += 1
            pushes &= (pushes - np.uint64(1))

        # --- Castling ---
        if (castling_rights & BK) and not(all_pieces_bb & 0x6000000000000000) and not is_square_attacked(piece_bbs, occupancy_bbs, 60, WHITE) and not is_square_attacked(piece_bbs, occupancy_bbs, 61, WHITE) and not is_square_attacked(piece_bbs, occupancy_bbs, 62, WHITE): moves_buffer[ply, move_count]=encode_move(60,62,0,SPECIAL_MOVE_FLAG_CASTLING); move_count+=1
        if (castling_rights & BQ) and not(all_pieces_bb & 0xe00000000000000) and not is_square_attacked(piece_bbs, occupancy_bbs, 60, WHITE) and not is_square_attacked(piece_bbs, occupancy_bbs, 59, WHITE) and not is_square_attacked(piece_bbs, occupancy_bbs, 58, WHITE): moves_buffer[ply, move_count]=encode_move(60,58,0,SPECIAL_MOVE_FLAG_CASTLING); move_count+=1

    # --- Leaper Moves (Knights, Bishops, Rooks, Queens, Kings) ---
    leaper_bbs = (wn_bb, wb_bb, wr_bb, wq_bb, wk_bb) if side_to_move == WHITE else (bn_bb, bb_bb, br_bb, bq_bb, bk_bb)
    for piece_type in range(5):
        bb = leaper_bbs[piece_type]
        while bb:
            from_sq = get_lsb_index(bb)
            if piece_type == 0: targets = KNIGHT_ATTACKS[from_sq] & empty_squares
            elif piece_type == 1: targets = get_bishop_attacks(from_sq, all_pieces_bb) & empty_squares
            elif piece_type == 2: targets = get_rook_attacks(from_sq, all_pieces_bb) & empty_squares
            elif piece_type == 3: targets = get_queen_attacks(from_sq, all_pieces_bb) & empty_squares
            else: targets = KING_ATTACKS[from_sq] & empty_squares
            
            while targets:
                to_sq = get_lsb_index(targets)
                moves_buffer[ply, move_count] = encode_move(from_sq, to_sq, 0, SPECIAL_MOVE_FLAG_NORMAL); move_count+=1
                targets &= (targets - np.uint64(1))
            bb &= (bb - np.uint64(1))
    
    return move_count

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def generate_pseudo_legal_captures_buffer(piece_bbs, occupancy_bbs, game_state, moves_buffer, ply):
    """
    Generates all pseudo-legal capture and promotion moves for the current position into the provided buffer at the given ply.
    This is used in quiescence search.
    Returns the count of legal moves.
    """
    move_count = 0

    side_to_move = game_state[0]
    en_passant_square = game_state[2]

    (wp_bb, wn_bb, wb_bb, wr_bb, wq_bb, wk_bb,
     bp_bb, bn_bb, bb_bb, br_bb, bq_bb, bk_bb) = piece_bbs

    # Optimization: Unpack all 3 elements. occupancy_bbs[2] is the pre-calculated union of white and black pieces.
    white_pieces_bb, black_pieces_bb, all_pieces_bb = occupancy_bbs

    opponent_pieces_bb = black_pieces_bb if side_to_move == WHITE else white_pieces_bb
    own_pieces_bb = white_pieces_bb if side_to_move == WHITE else black_pieces_bb

    if side_to_move == WHITE:
        # --- Pawn Captures & Promotions ---
        single_pushes = (wp_bb << 8) & ~all_pieces_bb
        captures_west = ((wp_bb & NOT_A_FILE) << 7) & opponent_pieces_bb
        captures_east = ((wp_bb & NOT_H_FILE) << 9) & opponent_pieces_bb

        # Promotion via single push (quiet promotion)
        promo_pushes = single_pushes & RANK_8
        while promo_pushes:
            to_sq = get_lsb_index(promo_pushes)
            from_sq = to_sq - 8
            for p_type in [PROMO_QUEEN, PROMO_ROOK, PROMO_BISHOP, PROMO_KNIGHT]:
                moves_buffer[ply, move_count] = encode_move(from_sq, to_sq, p_type, SPECIAL_MOVE_FLAG_PROMOTION)
                move_count += 1
            promo_pushes &= (promo_pushes - np.uint64(1))

        # Capture Promotions
        promo_caps_west = captures_west & RANK_8
        while promo_caps_west:
            to_sq = get_lsb_index(promo_caps_west)
            from_sq = to_sq - 7
            for p_type in [PROMO_QUEEN, PROMO_ROOK, PROMO_BISHOP, PROMO_KNIGHT]:
                moves_buffer[ply, move_count] = encode_move(from_sq, to_sq, p_type, SPECIAL_MOVE_FLAG_PROMOTION)
                move_count += 1
            promo_caps_west &= (promo_caps_west - np.uint64(1))

        promo_caps_east = captures_east & RANK_8
        while promo_caps_east:
            to_sq = get_lsb_index(promo_caps_east)
            from_sq = to_sq - 9
            for p_type in [PROMO_QUEEN, PROMO_ROOK, PROMO_BISHOP, PROMO_KNIGHT]:
                moves_buffer[ply, move_count] = encode_move(from_sq, to_sq, p_type, SPECIAL_MOVE_FLAG_PROMOTION)
                move_count += 1
            promo_caps_east &= (promo_caps_east - np.uint64(1))

        # Regular Captures (non-promotion)
        caps_west = captures_west & ~RANK_8
        while caps_west:
            to_sq = get_lsb_index(caps_west)
            from_sq = to_sq - 7
            moves_buffer[ply, move_count] = encode_move(from_sq, to_sq, 0, SPECIAL_MOVE_FLAG_NORMAL)
            move_count += 1
            caps_west &= (caps_west - np.uint64(1))

        caps_east = captures_east & ~RANK_8
        while caps_east:
            to_sq = get_lsb_index(caps_east)
            from_sq = to_sq - 9
            moves_buffer[ply, move_count] = encode_move(from_sq, to_sq, 0, SPECIAL_MOVE_FLAG_NORMAL)
            move_count += 1
            caps_east &= (caps_east - np.uint64(1))

        # En Passant
        if en_passant_square != 64:
            ep_target_bb = BB_SQUARES[np.uint8(en_passant_square)]
            west_attackers = ((wp_bb & NOT_A_FILE) << 7) & ep_target_bb
            if west_attackers:
                from_sq = np.uint8(en_passant_square - 7)
                moves_buffer[ply, move_count] = encode_move(from_sq, np.uint8(en_passant_square), 0, SPECIAL_MOVE_FLAG_EN_PASSANT); move_count += 1
            east_attackers = ((wp_bb & NOT_H_FILE) << 9) & ep_target_bb
            if east_attackers:
                from_sq = np.uint8(en_passant_square - 9)
                moves_buffer[ply, move_count] = encode_move(from_sq, np.uint8(en_passant_square), 0, SPECIAL_MOVE_FLAG_EN_PASSANT); move_count += 1

    else: # BLACK
        # --- Pawn Captures & Promotions ---
        single_pushes = (bp_bb >> 8) & ~all_pieces_bb
        captures_west = ((bp_bb & NOT_H_FILE) >> 7) & opponent_pieces_bb
        captures_east = ((bp_bb & NOT_A_FILE) >> 9) & opponent_pieces_bb

        # Promotion via single push (quiet promotion)
        promo_pushes = single_pushes & RANK_1
        while promo_pushes:
            to_sq = get_lsb_index(promo_pushes)
            from_sq = to_sq + 8
            for p_type in [PROMO_QUEEN, PROMO_ROOK, PROMO_BISHOP, PROMO_KNIGHT]:
                moves_buffer[ply, move_count] = encode_move(from_sq, to_sq, p_type, SPECIAL_MOVE_FLAG_PROMOTION)
                move_count += 1
            promo_pushes &= (promo_pushes - np.uint64(1))

        # Capture Promotions
        promo_caps_west = captures_west & RANK_1
        while promo_caps_west:
            to_sq = get_lsb_index(promo_caps_west)
            from_sq = to_sq + 7
            for p_type in [PROMO_QUEEN, PROMO_ROOK, PROMO_BISHOP, PROMO_KNIGHT]:
                moves_buffer[ply, move_count] = encode_move(from_sq, to_sq, p_type, SPECIAL_MOVE_FLAG_PROMOTION)
                move_count += 1
            promo_caps_west &= (promo_caps_west - np.uint64(1))

        promo_caps_east = captures_east & RANK_1
        while promo_caps_east:
            to_sq = get_lsb_index(promo_caps_east)
            from_sq = to_sq + 9
            for p_type in [PROMO_QUEEN, PROMO_ROOK, PROMO_BISHOP, PROMO_KNIGHT]:
                moves_buffer[ply, move_count] = encode_move(from_sq, to_sq, p_type, SPECIAL_MOVE_FLAG_PROMOTION)
                move_count += 1
            promo_caps_east &= (promo_caps_east - np.uint64(1))

        # Regular Captures (non-promotion)
        caps_west = captures_west & ~RANK_1
        while caps_west:
            to_sq = get_lsb_index(caps_west)
            from_sq = to_sq + 7
            moves_buffer[ply, move_count] = encode_move(from_sq, to_sq, 0, SPECIAL_MOVE_FLAG_NORMAL)
            move_count += 1
            caps_west &= (caps_west - np.uint64(1))

        caps_east = captures_east & ~RANK_1
        while caps_east:
            to_sq = get_lsb_index(caps_east)
            from_sq = to_sq + 9
            moves_buffer[ply, move_count] = encode_move(from_sq, to_sq, 0, SPECIAL_MOVE_FLAG_NORMAL)
            move_count += 1
            caps_east &= (caps_east - np.uint64(1))

        # En Passant
        if en_passant_square != 64:
            ep_target_bb = BB_SQUARES[np.uint8(en_passant_square)]
            west_attackers = ((bp_bb & NOT_H_FILE) >> 7) & ep_target_bb
            if west_attackers:
                from_sq = np.uint8(en_passant_square + 7)
                moves_buffer[ply, move_count] = encode_move(from_sq, np.uint8(en_passant_square), 0, SPECIAL_MOVE_FLAG_EN_PASSANT); move_count += 1
            east_attackers = ((bp_bb & NOT_A_FILE) >> 9) & ep_target_bb
            if east_attackers:
                from_sq = np.uint8(en_passant_square + 9)
                moves_buffer[ply, move_count] = encode_move(from_sq, np.uint8(en_passant_square), 0, SPECIAL_MOVE_FLAG_EN_PASSANT); move_count += 1

    # --- Leaper Captures (Knights, Bishops, Rooks, Queens, Kings) ---
    leaper_bbs = (wn_bb, wb_bb, wr_bb, wq_bb, wk_bb) if side_to_move == WHITE else (bn_bb, bb_bb, br_bb, bq_bb, bk_bb)
    for piece_type in range(5):
        bb = leaper_bbs[piece_type]
        while bb:
            from_sq = get_lsb_index(bb)
            if piece_type == 0: targets = KNIGHT_ATTACKS[from_sq]
            elif piece_type == 1: targets = get_bishop_attacks(from_sq, all_pieces_bb)
            elif piece_type == 2: targets = get_rook_attacks(from_sq, all_pieces_bb)
            elif piece_type == 3: targets = get_queen_attacks(from_sq, all_pieces_bb)
            else: targets = KING_ATTACKS[from_sq]

            capture_targets = targets & opponent_pieces_bb

            while capture_targets:
                to_sq = get_lsb_index(capture_targets)
                moves_buffer[ply, move_count] = encode_move(from_sq, to_sq, 0, SPECIAL_MOVE_FLAG_NORMAL)
                move_count += 1
                capture_targets &= (capture_targets - np.uint64(1))
            bb &= (bb - np.uint64(1))

    return move_count

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def generate_captures_buffer(piece_bbs, occupancy_bbs, game_state, moves_buffer, ply):
    """
    Generates all fully legal capture and promotion moves for the current position into the provided buffer at the given ply.
    Returns the count of legal moves.
    """
    move_count = generate_pseudo_legal_captures_buffer(piece_bbs, occupancy_bbs, game_state, moves_buffer, ply)
    side_to_move = game_state[0]

    # --- Filter for legality ---
    legal_move_count = 0

    king_bb = piece_bbs[5] if side_to_move == WHITE else piece_bbs[11]
    original_king_sq = get_lsb_index(king_bb) if king_bb else 0

    # Optimization: Pre-calculate pinned pieces and check status to skip expensive make/unmake
    opponent_side = 1 - side_to_move
    in_check = is_square_attacked(piece_bbs, occupancy_bbs, original_king_sq, opponent_side)
    pinned = get_pinned_pieces(piece_bbs, occupancy_bbs, side_to_move) if not in_check else np.uint64(0)

    for i in range(move_count):
        move = moves_buffer[ply, i]
        from_sq = get_from_square(move)
        
        # Fast path: if not in check, not pinned, not a king move, and not en passant, it MUST be legal.
        if not in_check and not (pinned & BB_SQUARES[from_sq]) and from_sq != original_king_sq and get_special_move_flag(move) != SPECIAL_MOVE_FLAG_EN_PASSANT:
            moves_buffer[ply, legal_move_count] = move
            legal_move_count += 1
            continue

        unmake_info = make_move(piece_bbs, occupancy_bbs, game_state, move)

        king_bb_after_move = piece_bbs[5] if side_to_move == WHITE else piece_bbs[11]
        king_sq = get_lsb_index(king_bb_after_move) if king_bb_after_move else original_king_sq

        if not is_square_attacked(piece_bbs, occupancy_bbs, king_sq, game_state[0]):
            moves_buffer[ply, legal_move_count] = move
            legal_move_count += 1

        unmake_move(piece_bbs, occupancy_bbs, game_state, move, unmake_info)

    return legal_move_count

def generate_captures(piece_bbs, occupancy_bbs, game_state):
    """
    Wrapper for generate_captures_buffer.
    """
    moves = np.zeros((1, 128), dtype=np.uint16)
    count = generate_captures_buffer(piece_bbs, occupancy_bbs, game_state, moves, 0)
    return moves[0, :count]

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def get_all_pawn_attacks(piece_bbs, side):
    pawn_bb = piece_bbs[0 if side == 0 else 6]
    attacks = np.uint64(0)
    if side == 0:
        attacks |= ((pawn_bb & NOT_A_FILE) << 7) | ((pawn_bb & NOT_H_FILE) << 9)
    else:
        attacks |= ((pawn_bb & NOT_H_FILE) >> 7) | ((pawn_bb & NOT_A_FILE) >> 9)
    return attacks

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def get_all_knight_attacks(piece_bbs, side):
    knight_bb = piece_bbs[1 if side == 0 else 7]
    attacks = np.uint64(0)
    bb = knight_bb
    while bb:
        sq = get_lsb_index(bb)
        attacks |= KNIGHT_ATTACKS[sq]
        bb &= bb - np.uint64(1)
    return attacks

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def get_all_bishop_attacks(piece_bbs, occupancy_bbs, side):
    bishop_bb = piece_bbs[2 if side == 0 else 8]
    attacks = np.uint64(0)
    all_occ = occupancy_bbs[2]
    bb = bishop_bb
    while bb:
        sq = get_lsb_index(bb)
        attacks |= get_bishop_attacks(sq, all_occ)
        bb &= bb - np.uint64(1)
    return attacks

@numba.njit(cache=True, boundscheck=False, fastmath=True)
def get_all_rook_attacks(piece_bbs, occupancy_bbs, side):
    rook_bb = piece_bbs[3 if side == 0 else 9]
    attacks = np.uint64(0)
    all_occ = occupancy_bbs[2]
    bb = rook_bb
    while bb:
        sq = get_lsb_index(bb)
        attacks |= get_rook_attacks(sq, all_occ)
        bb &= bb - np.uint64(1)
    return attacks
