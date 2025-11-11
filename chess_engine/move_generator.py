# chess_engine/move_generator.py
import numpy as np
import numba

from chess_engine.move import (
    encode_move, SPECIAL_MOVE_FLAG_NORMAL, SPECIAL_MOVE_FLAG_PROMOTION, SPECIAL_MOVE_FLAG_EN_PASSANT, SPECIAL_MOVE_FLAG_CASTLING,
    PROMO_QUEEN, PROMO_ROOK, PROMO_BISHOP, PROMO_KNIGHT
)
from chess_engine.board_operations import make_move, unmake_move
# from chess_engine.engine_types import board_state_flat_signature # This is no longer needed
from chess_engine.constants import BB_SQUARES
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
ROOK_MAGIC_NUMBERS = np.array([0x8a80104000800020,0x140002000100040,0x2801880a0017001,0x100081001000420,0x200020010080420,0x3001c0002010008,0x8480008002000100,0x2080088004402900,0x800098204000,0x2024401000200040,0x100802000801000,0x120800800801000,0x208808088000400,0x2802200800400,0x2200800100020080,0x801000060821100,0x80044006422000,0x100808020004000,0x12108a0010204200,0x140848010000802,0x481828014002800,0x8094004002004100,0x4010040010010802,0x20008806104,0x100400080208000,0x2040002120081000,0x21200680100081,0x20100080080080,0x2000a00200410,0x20080800400,0x80088400100102,0x80004600042881,0x4040008040800020,0x440003000200801,0x4200011004500,0x188020010100100,0x14800401802800,0x2080040080800200,0x124080204001001,0x200046502000484,0x480400080088020,0x1000422010034000,0x30200100110040,0x100021010009,0x2002080100110004,0x202008004008002,0x20020004010100,0x2048440040820001,0x101002200408200,0x40802000401080,0x4008142004410100,0x2060820c0120200,0x1001004080100,0x20c020080040080,0x2935610830022400,0x44440041009200,0x280001040802101,0x2100190040002085,0x80c0084100102001,0x4024081001000421,0x20030a0244872,0x12001008414402,0x2006104900a0804,0x1004081002402], dtype=np.uint64)
BISHOP_MAGIC_NUMBERS = np.array([0x40040844404084,0x2004208a004208,0x10190041080202,0x108060845042010,0x581104180800210,0x2112080446200010,0x1080820820060210,0x3c0808410220200,0x4050404440404,0x21001420088,0x24d0080801082102,0x1020a0a020400,0x40308200402,0x4011002100800,0x401484104104005,0x801010402020200,0x400210c3880100,0x404022024108200,0x810018200204102,0x4002801a02003,0x85040820080400,0x810102c808880400,0xe900410884800,0x8002020480840102,0x220200865090201,0x2010100a02021202,0x152048408022401,0x20080002081110,0x4001001021004000,0x800040400a011002,0xe4004081011002,0x1c004001012080,0x8004200962a00220,0x8422100208500202,0x2000402200300c08,0x8646020080080080,0x80020a0200100808,0x2010004880111000,0x623000a080011400,0x42008c0340209202,0x209188240001000,0x400408a884001800,0x110400a6080400,0x1840060a44020800,0x90080104000041,0x201011000808101,0x1a2208080504f080,0x8012020600211212,0x500861011240000,0x180806108200800,0x4000020e01040044,0x300000261044000a,0x802241102020002,0x20906061210001,0x5a84841004010310,0x4010801011c04,0xa010109502200,0x4a02012000,0x500201010098b028,0x8040002811040900,0x28000010020204,0x6000020202d0240,0x8918844842082200,0x401001102902002], dtype=np.uint64)

from chess_engine.zobrist import get_lsb_index
from chess_engine.bitboard_utils import count_bits

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
def init_sliders_attacks():
    bishop_attacks, rook_attacks = np.empty((64, 512), dtype=np.uint64), np.empty((64, 4096), dtype=np.uint64)
    for sq in range(64):
        attack_mask, relevant_bits_count = BISHOP_MASKS[sq], count_bits(BISHOP_MASKS[sq])
        for i in range(1 << relevant_bits_count):
            occ = set_occupancy(i, relevant_bits_count, attack_mask)
            product = int(occ) * int(BISHOP_MAGIC_NUMBERS[sq])
            magic_index = (product & 0xFFFFFFFFFFFFFFFF) >> (64 - BISHOP_RELEVANT_BITS[sq])
            bishop_attacks[sq][magic_index] = bishop_attacks_on_the_fly(sq, occ)
        attack_mask, relevant_bits_count = ROOK_MASKS[sq], count_bits(ROOK_MASKS[sq])
        for i in range(1 << relevant_bits_count):
            occ = set_occupancy(i, relevant_bits_count, attack_mask)
            product = int(occ) * int(ROOK_MAGIC_NUMBERS[sq])
            magic_index = (product & 0xFFFFFFFFFFFFFFFF) >> (64 - ROOK_RELEVANT_BITS[sq])
            rook_attacks[sq][magic_index] = rook_attacks_on_the_fly(sq, occ)
    return bishop_attacks, rook_attacks
# BISHOP_ATTACKS, ROOK_ATTACKS = init_sliders_attacks()
BISHOP_ATTACKS = np.empty((64, 512), dtype=np.uint64)
ROOK_ATTACKS = np.empty((64, 4096), dtype=np.uint64)

@numba.njit(numba.uint64(numba.uint8, numba.uint64), cache=True, boundscheck=False, fastmath=True)
def get_bishop_attacks(sq, occ):
    return bishop_attacks_on_the_fly(sq, occ)
@numba.njit(numba.uint64(numba.uint8, numba.uint64), cache=True, boundscheck=False, fastmath=True)
def get_rook_attacks(sq, occ):
    return rook_attacks_on_the_fly(sq, occ)
@numba.njit(numba.uint64(numba.uint8, numba.uint64), cache=True, boundscheck=False, fastmath=True)
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


from chess_engine.engine_types import piece_bbs_signature, occupancy_bbs_signature, game_state_signature


@numba.njit(numba.boolean(piece_bbs_signature, occupancy_bbs_signature, game_state_signature, numba.uint8, numba.uint8), cache=True, boundscheck=False, fastmath=True)
def is_square_attacked(piece_bbs, occupancy_bbs, game_state, sq, attacker_side):
    """
    Checks if a given square is attacked by the specified side.
    """
    all_pieces_bb = occupancy_bbs[0] | occupancy_bbs[1]
    (wp_bb, wn_bb, wb_bb, wr_bb, wq_bb, wk_bb, 
     bp_bb, bn_bb, bb_bb, br_bb, bq_bb, bk_bb) = piece_bbs

    if attacker_side == WHITE:
        if PAWN_ATTACKS[WHITE, sq] & wp_bb: return True
        if KING_ATTACKS[sq] & wk_bb: return True
        if KNIGHT_ATTACKS[sq] & wn_bb: return True
        if get_bishop_attacks(sq, all_pieces_bb) & (wb_bb | wq_bb): return True
        if get_rook_attacks(sq, all_pieces_bb) & (wr_bb | wq_bb): return True
    else: # Attacker is BLACK
        if PAWN_ATTACKS[BLACK, sq] & bp_bb: return True
        if KING_ATTACKS[sq] & bk_bb: return True
        if KNIGHT_ATTACKS[sq] & bn_bb: return True
        if get_bishop_attacks(sq, all_pieces_bb) & (bb_bb | bq_bb): return True
        if get_rook_attacks(sq, all_pieces_bb) & (br_bb | bq_bb): return True
        
    return False

@numba.njit(numba.uint16[:](piece_bbs_signature, occupancy_bbs_signature, game_state_signature), cache=True, boundscheck=False, fastmath=True)
def generate_legal_moves(piece_bbs, occupancy_bbs, game_state):
    """
    Generates all fully legal moves for the current position.
    This version is refactored to use the new board state representation.
    """
    moves = np.zeros(256, dtype=np.uint16)
    move_count = 0
    
    side_to_move, castling_rights, en_passant_square, _, _ = game_state
    
    (wp_bb, wn_bb, wb_bb, wr_bb, wq_bb, wk_bb, 
     bp_bb, bn_bb, bb_bb, br_bb, bq_bb, bk_bb) = piece_bbs
     
    white_pieces_bb, black_pieces_bb, _ = occupancy_bbs
    all_pieces_bb = white_pieces_bb | black_pieces_bb

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
                for p_type in [PROMO_QUEEN, PROMO_ROOK, PROMO_BISHOP, PROMO_KNIGHT]: moves[move_count] = encode_move(from_sq, to_sq, p_type, SPECIAL_MOVE_FLAG_PROMOTION); move_count += 1
            else:
                moves[move_count] = encode_move(from_sq, to_sq, 0, SPECIAL_MOVE_FLAG_NORMAL); move_count += 1
            pushes &= (pushes - np.uint64(1))
        
        pushes = double_pushes
        while pushes:
            to_sq = get_lsb_index(pushes)
            from_sq = to_sq - 16
            moves[move_count] = encode_move(from_sq, to_sq, 0, SPECIAL_MOVE_FLAG_NORMAL); move_count += 1
            pushes &= (pushes - np.uint64(1))

        # Captures
        caps = captures_west
        while caps:
            to_sq = get_lsb_index(caps)
            from_sq = to_sq - 7
            if to_sq >= 56: # Promotion
                for p_type in [PROMO_QUEEN, PROMO_ROOK, PROMO_BISHOP, PROMO_KNIGHT]: moves[move_count] = encode_move(from_sq, to_sq, p_type, SPECIAL_MOVE_FLAG_PROMOTION); move_count += 1
            else:
                moves[move_count] = encode_move(from_sq, to_sq, 0, SPECIAL_MOVE_FLAG_NORMAL); move_count += 1
            caps &= (caps - np.uint64(1))

        caps = captures_east
        while caps:
            to_sq = get_lsb_index(caps)
            from_sq = to_sq - 9
            if to_sq >= 56: # Promotion
                for p_type in [PROMO_QUEEN, PROMO_ROOK, PROMO_BISHOP, PROMO_KNIGHT]: moves[move_count] = encode_move(from_sq, to_sq, p_type, SPECIAL_MOVE_FLAG_PROMOTION); move_count += 1
            else:
                moves[move_count] = encode_move(from_sq, to_sq, 0, SPECIAL_MOVE_FLAG_NORMAL); move_count += 1
            caps &= (caps - np.uint64(1))

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

        # --- Castling ---
        if (castling_rights & WK) and not(all_pieces_bb & 0x60) and not is_square_attacked(piece_bbs, occupancy_bbs, game_state, 4, BLACK) and not is_square_attacked(piece_bbs, occupancy_bbs, game_state, 5, BLACK) and not is_square_attacked(piece_bbs, occupancy_bbs, game_state, 6, BLACK): moves[move_count]=encode_move(4,6,0,SPECIAL_MOVE_FLAG_CASTLING); move_count+=1
        if (castling_rights & WQ) and not(all_pieces_bb & 0xe) and not is_square_attacked(piece_bbs, occupancy_bbs, game_state, 4, BLACK) and not is_square_attacked(piece_bbs, occupancy_bbs, game_state, 3, BLACK) and not is_square_attacked(piece_bbs, occupancy_bbs, game_state, 2, BLACK): moves[move_count]=encode_move(4,2,0,SPECIAL_MOVE_FLAG_CASTLING); move_count+=1

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
                for p_type in [PROMO_QUEEN, PROMO_ROOK, PROMO_BISHOP, PROMO_KNIGHT]: moves[move_count] = encode_move(from_sq, to_sq, p_type, SPECIAL_MOVE_FLAG_PROMOTION); move_count += 1
            else:
                moves[move_count] = encode_move(from_sq, to_sq, 0, SPECIAL_MOVE_FLAG_NORMAL); move_count += 1
            pushes &= (pushes - np.uint64(1))
        
        pushes = double_pushes
        while pushes:
            to_sq = get_lsb_index(pushes)
            from_sq = to_sq + 16
            moves[move_count] = encode_move(from_sq, to_sq, 0, SPECIAL_MOVE_FLAG_NORMAL); move_count += 1
            pushes &= (pushes - np.uint64(1))

        caps = captures_west
        while caps:
            to_sq = get_lsb_index(caps)
            from_sq = to_sq + 7
            if to_sq <= 7: # Promotion
                for p_type in [PROMO_QUEEN, PROMO_ROOK, PROMO_BISHOP, PROMO_KNIGHT]: moves[move_count] = encode_move(from_sq, to_sq, p_type, SPECIAL_MOVE_FLAG_PROMOTION); move_count += 1
            else:
                moves[move_count] = encode_move(from_sq, to_sq, 0, SPECIAL_MOVE_FLAG_NORMAL); move_count += 1
            caps &= (caps - np.uint64(1))

        caps = captures_east
        while caps:
            to_sq = get_lsb_index(caps)
            from_sq = to_sq + 9
            if to_sq <= 7: # Promotion
                for p_type in [PROMO_QUEEN, PROMO_ROOK, PROMO_BISHOP, PROMO_KNIGHT]: moves[move_count] = encode_move(from_sq, to_sq, p_type, SPECIAL_MOVE_FLAG_PROMOTION); move_count += 1
            else:
                moves[move_count] = encode_move(from_sq, to_sq, 0, SPECIAL_MOVE_FLAG_NORMAL); move_count += 1
            caps &= (caps - np.uint64(1))

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

        # --- Castling ---
        if (castling_rights & BK) and not(all_pieces_bb & 0x6000000000000000) and not is_square_attacked(piece_bbs, occupancy_bbs, game_state, 60, WHITE) and not is_square_attacked(piece_bbs, occupancy_bbs, game_state, 61, WHITE) and not is_square_attacked(piece_bbs, occupancy_bbs, game_state, 62, WHITE): moves[move_count]=encode_move(60,62,0,SPECIAL_MOVE_FLAG_CASTLING); move_count+=1
        if (castling_rights & BQ) and not(all_pieces_bb & 0xe00000000000000) and not is_square_attacked(piece_bbs, occupancy_bbs, game_state, 60, WHITE) and not is_square_attacked(piece_bbs, occupancy_bbs, game_state, 59, WHITE) and not is_square_attacked(piece_bbs, occupancy_bbs, game_state, 58, WHITE): moves[move_count]=encode_move(60,58,0,SPECIAL_MOVE_FLAG_CASTLING); move_count+=1

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
                moves[move_count] = encode_move(from_sq, to_sq, 0, SPECIAL_MOVE_FLAG_NORMAL); move_count+=1
                targets &= (targets - np.uint64(1))
            bb &= (bb - np.uint64(1))
    
    # --- Filter for legality ---
    legal_moves_final = np.zeros(256, dtype=np.uint16)
    legal_move_count = 0

    king_bb = piece_bbs[5] if side_to_move == WHITE else piece_bbs[11]
    original_king_sq = get_lsb_index(king_bb) if king_bb else 0

    for i in range(move_count):
        move = moves[i]

        # Make the move on the board
        unmake_info = make_move(piece_bbs, occupancy_bbs, game_state, move)

        # After the move, the 'side_to_move' in game_state is the opponent.
        # We need to check if the ORIGINAL side's king is attacked.
        king_bb_after_move = piece_bbs[5] if side_to_move == WHITE else piece_bbs[11]
        king_sq = get_lsb_index(king_bb_after_move) if king_bb_after_move else original_king_sq

        # Check if the king is attacked by the new side to move (the opponent)
        if not is_square_attacked(piece_bbs, occupancy_bbs, game_state, king_sq, game_state[0]):
            legal_moves_final[legal_move_count] = move
            legal_move_count += 1

        # Unmake the move to restore the board state for the next iteration
        unmake_move(piece_bbs, occupancy_bbs, game_state, move, unmake_info)
                
    return legal_moves_final[:legal_move_count]


@numba.njit(nbt.uint16[:](piece_bbs_signature, occupancy_bbs_signature, game_state_signature), cache=True, boundscheck=False, fastmath=True)
def generate_tactical_moves(piece_bbs, occupancy_bbs, game_state):
    """
    Generates pseudo-legal tactical moves (all captures and promotions).
    """
    moves = np.zeros(128, dtype=np.uint16)
    move_count = 0

    side_to_move, _, en_passant_square, _, _ = game_state
    
    (wp_bb, wn_bb, wb_bb, wr_bb, wq_bb, wk_bb, 
     bp_bb, bn_bb, bb_bb, br_bb, bq_bb, bk_bb) = piece_bbs
     
    white_pieces_bb, black_pieces_bb, _ = occupancy_bbs
    all_pieces_bb = white_pieces_bb | black_pieces_bb

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

@numba.njit(numba.boolean(piece_bbs_signature, occupancy_bbs_signature, game_state_signature), cache=True, boundscheck=False, fastmath=True)
def is_in_check(piece_bbs, occupancy_bbs, game_state):
    """
    Checks if the current side to move is in check.
    """
    side_to_move = game_state[0]
    king_bb = piece_bbs[5] if side_to_move == WHITE else piece_bbs[11]
    if king_bb == 0: # Should not happen in a legal position
        return False
    king_sq = get_lsb_index(king_bb)
    return is_square_attacked(piece_bbs, occupancy_bbs, game_state, king_sq, 1 - side_to_move)

@numba.njit(numba.boolean(piece_bbs_signature, numba.uint8), cache=True, boundscheck=False, fastmath=True)
def has_sufficient_material(piece_bbs, side_to_move):
    """
    Checks if the side to move has major pieces (Rook or Queen).
    Used as a condition for Null Move Pruning.
    """
    if side_to_move == WHITE:
        return (piece_bbs[3] | piece_bbs[4]) != 0
    else: # BLACK
        return (piece_bbs[9] | piece_bbs[10]) != 0


@numba.njit(nbt.uint16[:](piece_bbs_signature, occupancy_bbs_signature, game_state_signature), cache=True, boundscheck=False, fastmath=True)
def generate_captures(piece_bbs, occupancy_bbs, game_state):
    """
    Generates all fully legal capture and promotion moves for the current position.
    This is used in quiescence search.
    """
    moves = np.zeros(128, dtype=np.uint16)
    move_count = 0

    side_to_move, _, en_passant_square, _, _ = game_state

    (wp_bb, wn_bb, wb_bb, wr_bb, wq_bb, wk_bb,
     bp_bb, bn_bb, bb_bb, br_bb, bq_bb, bk_bb) = piece_bbs

    white_pieces_bb, black_pieces_bb, _ = occupancy_bbs
    all_pieces_bb = white_pieces_bb | black_pieces_bb

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

        # Regular Captures (non-promotion)
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

        # Promotion via single push (quiet promotion)
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

        # Regular Captures (non-promotion)
        caps_west = captures_west & ~RANK_1
        while caps_west:
            to_sq = get_lsb_index(caps_west)
            from_sq = to_sq + 7
            moves[move_count] = encode_move(from_sq, to_sq, 0, SPECIAL_MOVE_FLAG_NORMAL)
            move_count += 1
            caps_west &= (caps_west - np.uint64(1))

        caps_east = caps_east & ~RANK_1
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

            capture_targets = targets & opponent_pieces_bb

            while capture_targets:
                to_sq = get_lsb_index(capture_targets)
                moves[move_count] = encode_move(from_sq, to_sq, 0, SPECIAL_MOVE_FLAG_NORMAL)
                move_count += 1
                capture_targets &= (capture_targets - np.uint64(1))
            bb &= (bb - np.uint64(1))

    # --- Filter for legality ---
    legal_moves_final = np.zeros(128, dtype=np.uint16)
    legal_move_count = 0

    king_bb = piece_bbs[5] if side_to_move == WHITE else piece_bbs[11]
    original_king_sq = get_lsb_index(king_bb) if king_bb else 0

    for i in range(move_count):
        move = moves[i]

        unmake_info = make_move(piece_bbs, occupancy_bbs, game_state, move)

        king_bb_after_move = piece_bbs[5] if side_to_move == WHITE else piece_bbs[11]
        king_sq = get_lsb_index(king_bb_after_move) if king_bb_after_move else original_king_sq

        if not is_square_attacked(piece_bbs, occupancy_bbs, game_state, king_sq, game_state[0]):
            legal_moves_final[legal_move_count] = move
            legal_move_count += 1

        unmake_move(piece_bbs, occupancy_bbs, game_state, move, unmake_info)

    return legal_moves_final[:legal_move_count]