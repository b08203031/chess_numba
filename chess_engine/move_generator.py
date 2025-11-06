# chess_engine/move_generator.py
# This file will contain functions for generating piece moves using bitboards.
# Magic Bitboard constants and logic adapted from the black_numba project by Avo-k.
import numpy
import numba as nb

# =============================================================================
# BITBOARD CONSTANTS
# =============================================================================
NOT_A_FILE = ~numpy.uint64(0x0101010101010101)
NOT_H_FILE = ~numpy.uint64(0x8080808080808080)
NOT_AB_FILE = ~numpy.uint64(0x0303030303030303)
NOT_GH_FILE = ~numpy.uint64(0xC0C0C0C0C0C0C0C0)
RANK_2 = numpy.uint64(0xFF00)
RANK_3 = numpy.uint64(0xFF0000)
RANK_4 = numpy.uint64(0xFF000000)
RANK_5 = numpy.uint64(0xFF00000000)
RANK_6 = numpy.uint64(0xFF0000000000)
RANK_7 = numpy.uint64(0xFF000000000000)
WHITE, BLACK = 0, 1

BB_SQUARES = numpy.array([numpy.uint64(1) << i for i in range(64)], dtype=numpy.uint64)
BB_RANK_1 = numpy.uint64(0xff)
BB_RANK_8 = numpy.uint64(0xff00000000000000)
BB_FILE_A = numpy.uint64(0x0101010101010101)
BB_FILE_H = numpy.uint64(0x8080808080808080)
BB_RANKS = numpy.array([numpy.uint64(0xff) << (8 * i) for i in range(8)], dtype=numpy.uint64)
BB_FILES = numpy.array([numpy.uint64(0x0101010101010101) << i for i in range(8)], dtype=numpy.uint64)
EMPTY = numpy.uint64(0)
BIT = numpy.uint64(1)

# =============================================================================
# MAGIC BITBOARD CONSTANTS (from black_numba)
# =============================================================================
BISHOP_RELEVANT_BITS = numpy.array(
    [6, 5, 5, 5, 5, 5, 5, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 7, 7, 7, 7, 5, 5, 5, 5, 7, 9, 9, 7, 5, 5,
     5, 5, 7, 9, 9, 7, 5, 5, 5, 5, 7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 5, 5, 5, 5, 5, 5, 6],
    dtype=numpy.uint8)

ROOK_RELEVANT_BITS = numpy.array(
    [12, 11, 11, 11, 11, 11, 11, 12, 11, 10, 10, 10, 10, 10, 10, 11, 11, 10, 10, 10, 10, 10, 10, 11,
     11, 10, 10, 10, 10, 10, 10, 11, 11, 10, 10, 10, 10, 10, 10, 11, 11, 10, 10, 10, 10, 10, 10, 11,
     11, 10, 10, 10, 10, 10, 10, 11, 12, 11, 11, 11, 11, 11, 11, 12],
    dtype=numpy.uint8)

ROOK_MAGIC_NUMBERS = numpy.array([0x8a80104000800020, 0x140002000100040, 0x2801880a0017001,
                                  0x100081001000420, 0x200020010080420, 0x3001c0002010008,
                                  0x8480008002000100, 0x2080088004402900, 0x800098204000,
                                  0x2024401000200040, 0x100802000801000, 0x120800800801000,
                                  0x208808088000400, 0x2802200800400, 0x2200800100020080,
                                  0x801000060821100, 0x80044006422000, 0x100808020004000,
                                  0x12108a0010204200, 0x140848010000802, 0x481828014002800,
                                  0x8094004002004100, 0x4010040010010802, 0x20008806104,
                                  0x100400080208000, 0x2040002120081000, 0x21200680100081,
                                  0x20100080080080, 0x2000a00200410, 0x20080800400,
                                  0x80088400100102, 0x80004600042881, 0x4040008040800020,
                                  0x440003000200801, 0x4200011004500, 0x188020010100100,
                                  0x14800401802800, 0x2080040080800200, 0x124080204001001,
                                  0x200046502000484, 0x480400080088020, 0x1000422010034000,
                                  0x30200100110040, 0x100021010009, 0x2002080100110004,
                                  0x202008004008002, 0x20020004010100, 0x2048440040820001,
                                  0x101002200408200, 0x40802000401080, 0x4008142004410100,
                                  0x2060820c0120200, 0x1001004080100, 0x20c020080040080,
                                  0x2935610830022400, 0x44440041009200, 0x280001040802101,
                                  0x2100190040002085, 0x80c0084100102001, 0x4024081001000421,
                                  0x20030a0244872, 0x12001008414402, 0x2006104900a0804,
                                  0x1004081002402], dtype=numpy.uint64)

BISHOP_MAGIC_NUMBERS = numpy.array([0x40040844404084, 0x2004208a004208, 0x10190041080202,
                                  0x108060845042010, 0x581104180800210, 0x2112080446200010,
                                  0x1080820820060210, 0x3c0808410220200, 0x4050404440404,
                                  0x21001420088, 0x24d0080801082102, 0x1020a0a020400,
                                  0x40308200402, 0x4011002100800, 0x401484104104005,
                                  0x801010402020200, 0x400210c3880100, 0x404022024108200,
                                  0x810018200204102, 0x4002801a02003, 0x85040820080400,
                                  0x810102c808880400, 0xe900410884800, 0x8002020480840102,
                                  0x220200865090201, 0x2010100a02021202, 0x152048408022401,
                                  0x20080002081110, 0x4001001021004000, 0x800040400a011002,
                                  0xe4004081011002, 0x1c004001012080, 0x8004200962a00220,
                                  0x8422100208500202, 0x2000402200300c08, 0x8646020080080080,
                                  0x80020a0200100808, 0x2010004880111000, 0x623000a080011400,
                                  0x42008c0340209202, 0x209188240001000, 0x400408a884001800,
                                  0x110400a6080400, 0x1840060a44020800, 0x90080104000041,
                                  0x201011000808101, 0x1a2208080504f080, 0x8012020600211212,
                                  0x500861011240000, 0x180806108200800, 0x4000020e01040044,
                                  0x300000261044000a, 0x802241102020002, 0x20906061210001,
                                  0x5a84841004010310, 0x4010801011c04, 0xa010109502200,
                                  0x4a02012000, 0x500201010098b028, 0x8040002811040900,
                                  0x28000010020204, 0x6000020202d0240, 0x8918844842082200,
                                  0x401001102902002], dtype=numpy.uint64)

# =============================================================================
# HELPER FUNCTIONS (for attack table generation)
# =============================================================================

@nb.njit(nb.int32(nb.uint64), cache=True)
def count_bits(bb: numpy.uint64) -> int:
    """
    Counts the number of set bits in a bitboard (popcount).
    This implementation is Numba-compatible.
    """
    count = 0
    bb_int = numpy.uint64(bb)
    while bb_int > 0:
        bb_int = bb_int & (bb_int - numpy.uint64(1))
        count += 1
    return count

@nb.njit(nb.uint8(nb.uint64), cache=True)
def get_ls1b_index(bb):
    return count_bits((bb & -bb) - numpy.uint64(1))

@nb.njit(nb.uint64(nb.uint64, nb.uint8), cache=True)
def pop_bit(bb, sq):
    return bb & ~BB_SQUARES[sq]

# =============================================================================
# SLIDING PIECE ATTACK TABLE GENERATION (Magic Bitboards)
# =============================================================================

@nb.njit(nb.uint64(nb.uint8), cache=True)
def mask_bishop_attacks(sq):
    attacks = EMPTY
    tr, tf = sq // 8, sq % 8
    for r, f in zip(range(tr + 1, 7), range(tf + 1, 7)): attacks |= BB_SQUARES[r * 8 + f]
    for r, f in zip(range(tr - 1, 0, -1), range(tf + 1, 7)): attacks |= BB_SQUARES[r * 8 + f]
    for r, f in zip(range(tr + 1, 7), range(tf - 1, 0, -1)): attacks |= BB_SQUARES[r * 8 + f]
    for r, f in zip(range(tr - 1, 0, -1), range(tf - 1, 0, -1)): attacks |= BB_SQUARES[r * 8 + f]
    return attacks

@nb.njit(nb.uint64(nb.uint8), cache=True)
def mask_rook_attacks(sq):
    attacks = EMPTY
    tr, tf = sq // 8, sq % 8
    for r in range(tr + 1, 7): attacks |= BB_SQUARES[r * 8 + tf]
    for r in range(tr - 1, 0, -1): attacks |= BB_SQUARES[r * 8 + tf]
    for f in range(tf + 1, 7): attacks |= BB_SQUARES[tr * 8 + f]
    for f in range(tf - 1, 0, -1): attacks |= BB_SQUARES[tr * 8 + f]
    return attacks

BISHOP_MASKS = numpy.fromiter((mask_bishop_attacks(sq) for sq in range(64)), dtype=numpy.uint64)
ROOK_MASKS = numpy.fromiter((mask_rook_attacks(sq) for sq in range(64)), dtype=numpy.uint64)

@nb.njit(nb.uint64(nb.uint8, nb.uint64), cache=True)
def bishop_attacks_on_the_fly(sq, block):
    attacks = EMPTY
    tr, tf = sq // 8, sq % 8
    for r, f in zip(range(tr + 1, 8), range(tf + 1, 8)):
        attacks |= BB_SQUARES[r * 8 + f]
        if BB_SQUARES[r * 8 + f] & block: break
    for r, f in zip(range(tr - 1, -1, -1), range(tf + 1, 8)):
        attacks |= BB_SQUARES[r * 8 + f]
        if BB_SQUARES[r * 8 + f] & block: break
    for r, f in zip(range(tr + 1, 8), range(tf - 1, -1, -1)):
        attacks |= BB_SQUARES[r * 8 + f]
        if BB_SQUARES[r * 8 + f] & block: break
    for r, f in zip(range(tr - 1, -1, -1), range(tf - 1, -1, -1)):
        attacks |= BB_SQUARES[r * 8 + f]
        if BB_SQUARES[r * 8 + f] & block: break
    return attacks

@nb.njit(nb.uint64(nb.uint8, nb.uint64), cache=True)
def rook_attacks_on_the_fly(sq, block):
    attacks = EMPTY
    tr, tf = sq // 8, sq % 8
    for r in range(tr + 1, 8):
        attacks |= BB_SQUARES[r * 8 + tf]
        if BB_SQUARES[r * 8 + tf] & block: break
    for r in range(tr - 1, -1, -1):
        attacks |= BB_SQUARES[r * 8 + tf]
        if BB_SQUARES[r * 8 + tf] & block: break
    for f in range(tf + 1, 8):
        attacks |= BB_SQUARES[tr * 8 + f]
        if BB_SQUARES[tr * 8 + f] & block: break
    for f in range(tf - 1, -1, -1):
        attacks |= BB_SQUARES[tr * 8 + f]
        if BB_SQUARES[tr * 8 + f] & block: break
    return attacks

@nb.njit(nb.uint64(nb.uint16, nb.uint8, nb.uint64), cache=True)
def set_occupancy(index, bits_in_mask, attack_mask):
    occupancy = EMPTY
    for count in range(bits_in_mask):
        square = get_ls1b_index(attack_mask)
        attack_mask = pop_bit(attack_mask, square)
        if index & (1 << count):
            occupancy |= BIT << square
    return occupancy

def init_sliders_attacks():
    bishop_attacks = numpy.empty((64, 512), dtype=numpy.uint64)
    rook_attacks = numpy.empty((64, 4096), dtype=numpy.uint64)
    for sq in range(64):
        # Bishop
        attack_mask = BISHOP_MASKS[sq]
        relevant_bits_count = count_bits(attack_mask)
        occupancy_indices = 1 << relevant_bits_count
        for index in range(occupancy_indices):
            occupancy = set_occupancy(index, relevant_bits_count, attack_mask)
            magic_index = (occupancy * BISHOP_MAGIC_NUMBERS[sq]) >> numpy.uint64(64 - BISHOP_RELEVANT_BITS[sq])
            bishop_attacks[sq][magic_index] = bishop_attacks_on_the_fly(sq, occupancy)
        # Rook
        attack_mask = ROOK_MASKS[sq]
        relevant_bits_count = count_bits(attack_mask)
        occupancy_indices = 1 << relevant_bits_count
        for index in range(occupancy_indices):
            occupancy = set_occupancy(index, relevant_bits_count, attack_mask)
            magic_index = (occupancy * ROOK_MAGIC_NUMBERS[sq]) >> numpy.uint64(64 - ROOK_RELEVANT_BITS[sq])
            rook_attacks[sq][magic_index] = rook_attacks_on_the_fly(sq, occupancy)

    return bishop_attacks, rook_attacks

BISHOP_ATTACKS, ROOK_ATTACKS = init_sliders_attacks()

# =============================================================================
# SLIDING PIECE ATTACK QUERY FUNCTIONS (Numba JIT Compiled)
# =============================================================================

@nb.njit(nb.uint64(nb.uint8, nb.uint64), cache=True)
def get_bishop_attacks(sq, occ):
    """
    Gets the pre-computed bishop attacks for a given square and board occupancy,
    using Magic Bitboard lookup.
    """
    occ &= BISHOP_MASKS[sq]
    occ *= BISHOP_MAGIC_NUMBERS[sq]
    occ >>= numpy.uint64(64 - BISHOP_RELEVANT_BITS[sq])
    return BISHOP_ATTACKS[sq][occ]


@nb.njit(nb.uint64(nb.uint8, nb.uint64), cache=True)
def get_rook_attacks(sq, occ):
    """
    Gets the pre-computed rook attacks for a given square and board occupancy,
    using Magic Bitboard lookup.
    """
    occ &= ROOK_MASKS[sq]
    occ *= ROOK_MAGIC_NUMBERS[sq]
    occ >>= numpy.uint64(64 - ROOK_RELEVANT_BITS[sq])
    return ROOK_ATTACKS[sq][occ]

# =============================================================================
# HIGH-LEVEL MOVE GENERATION
# =============================================================================

def get_rook_moves(square, occupied, own_pieces):
    """
    Generates all legal moves for a rook on a given square.
    This filters out moves that land on friendly pieces.
    """
    attacks = get_rook_attacks(square, occupied)
    return attacks & ~own_pieces

def get_bishop_moves(square, occupied, own_pieces):
    """
    Generates all legal moves for a bishop on a given square.
    This filters out moves that land on friendly pieces.
    """
    attacks = get_bishop_attacks(square, occupied)
    return attacks & ~own_pieces

def get_queen_moves(square, occupied, own_pieces):
    """
    Generates all legal moves for a queen on a given square.
    This filters out moves that land on friendly pieces.
    """
    rook_moves = get_rook_moves(square, occupied, own_pieces)
    bishop_moves = get_bishop_moves(square, occupied, own_pieces)
    return rook_moves | bishop_moves

# =============================================================================
# KNIGHT MOVE GENERATION (PRE-COMPUTED)
# =============================================================================
KNIGHT_ATTACKS = numpy.empty(64, dtype=numpy.uint64)

def _precompute_knight_attacks():
    for square in range(64):
        knight_bb = numpy.uint64(1) << numpy.uint64(square)
        attacks = numpy.uint64(0)
        if (knight_bb << 17) & NOT_A_FILE: attacks |= (knight_bb << 17)
        if (knight_bb << 15) & NOT_H_FILE: attacks |= (knight_bb << 15)
        if (knight_bb << 10) & NOT_AB_FILE: attacks |= (knight_bb << 10)
        if (knight_bb << 6) & NOT_GH_FILE: attacks |= (knight_bb << 6)
        if (knight_bb >> 6) & NOT_AB_FILE: attacks |= (knight_bb >> 6)
        if (knight_bb >> 10) & NOT_GH_FILE: attacks |= (knight_bb >> 10)
        if (knight_bb >> 15) & NOT_A_FILE: attacks |= (knight_bb >> 15)
        if (knight_bb >> 17) & NOT_H_FILE: attacks |= (knight_bb >> 17)
        KNIGHT_ATTACKS[square] = attacks

_precompute_knight_attacks()

def get_knight_attacks(square):
    return KNIGHT_ATTACKS[square]


# =============================================================================
# PAWN MOVE GENERATION
# =============================================================================
def _get_white_pawn_moves(pawns_bb, all_pieces_bb, opponent_pieces_bb):
    empty_squares = ~all_pieces_bb
    single_push = (pawns_bb << 8) & empty_squares
    double_push_candidates = (single_push & RANK_3)
    double_push = (double_push_candidates << 8) & empty_squares & RANK_4
    capture_right = (pawns_bb << 9) & opponent_pieces_bb & NOT_A_FILE
    capture_left = (pawns_bb << 7) & opponent_pieces_bb & NOT_H_FILE
    return single_push | double_push | capture_right | capture_left

def _get_black_pawn_moves(pawns_bb, all_pieces_bb, opponent_pieces_bb):
    empty_squares = ~all_pieces_bb
    single_push = (pawns_bb >> 8) & empty_squares
    double_push_candidates = (single_push & RANK_6)
    double_push = (double_push_candidates >> 8) & empty_squares & RANK_5
    capture_right = (pawns_bb >> 7) & opponent_pieces_bb & NOT_A_FILE
    capture_left = (pawns_bb >> 9) & opponent_pieces_bb & NOT_H_FILE
    return single_push | double_push | capture_right | capture_left

def get_pawn_moves(pawns_bb, color, all_pieces_bb, opponent_pieces_bb):
    if color == WHITE:
        return _get_white_pawn_moves(pawns_bb, all_pieces_bb, opponent_pieces_bb)
    else: # BLACK
        return _get_black_pawn_moves(pawns_bb, all_pieces_bb, opponent_pieces_bb)
