# chess_engine/move_generator.py
# This file will contain functions for generating piece moves using bitboards,
# following the style of the 'python-chess' library.
import numpy

# =============================================================================
# BITBOARD CONSTANTS
# =============================================================================
# These constants are pre-calculated bitmasks that are useful for move generation.
# Using these helps to avoid "wrap-around" issues where a piece on one side of
# the board incorrectly appears on the other side after a bit shift.

# File masks (prevent wrap-around)
# numpy.uint64 is essential for correct bitwise operations on 64-bit integers.
NOT_A_FILE = ~numpy.uint64(0x0101010101010101) # Masks out the 'a' file
NOT_H_FILE = ~numpy.uint64(0x8080808080808080) # Masks out the 'h' file
NOT_AB_FILE = ~numpy.uint64(0x0303030303030303) # Masks out the 'a' and 'b' files
NOT_GH_FILE = ~numpy.uint64(0xC0C0C0C0C0C0C0C0) # Masks out the 'g' and 'h' files

# Rank masks (useful for pawn moves)
RANK_2 = numpy.uint64(0xFF00)            # All squares on the 2nd rank
RANK_3 = numpy.uint64(0xFF0000)          # All squares on the 3rd rank
RANK_4 = numpy.uint64(0xFF000000)        # All squares on the 4th rank (white pawn double move target)
RANK_5 = numpy.uint64(0xFF00000000)      # All squares on the 5th rank (black pawn double move target)
RANK_6 = numpy.uint64(0xFF0000000000)    # All squares on the 6th rank
RANK_7 = numpy.uint64(0xFF000000000000)  # All squares on the 7th rank


# =============================================================================
# KNIGHT MOVE GENERATION (PRE-COMPUTED)
# =============================================================================
# Knight moves are pre-computed for all 64 squares and stored in a lookup table.
# This is extremely fast as it only requires a single array access at runtime.

# Initialize the lookup table
KNIGHT_ATTACKS = [numpy.uint64(0)] * 64

def _precompute_knight_attacks():
    """
    Computes the attack bitboards for a knight on each square of the board
    and populates the KNIGHT_ATTACKS lookup table.
    This function is intended to be called once at program startup.
    """
    for square in range(64):
        knight_bb = numpy.uint64(1) << numpy.uint64(square)
        attacks = numpy.uint64(0)

        # Generate all possible knight moves from the current square
        # The conditions check for board wrap-around.

        # Up 2, Right 1
        if (knight_bb << 17) & NOT_A_FILE: attacks |= (knight_bb << 17)
        # Up 2, Left 1
        if (knight_bb << 15) & NOT_H_FILE: attacks |= (knight_bb << 15)
        # Up 1, Right 2
        if (knight_bb << 10) & NOT_AB_FILE: attacks |= (knight_bb << 10)
        # Up 1, Left 2
        if (knight_bb << 6) & NOT_GH_FILE: attacks |= (knight_bb << 6)
        # Down 1, Right 2
        if (knight_bb >> 6) & NOT_AB_FILE: attacks |= (knight_bb >> 6)
        # Down 1, Left 2
        if (knight_bb >> 10) & NOT_GH_FILE: attacks |= (knight_bb >> 10)
        # Down 2, Right 1
        if (knight_bb >> 15) & NOT_A_FILE: attacks |= (knight_bb >> 15)
        # Down 2, Left 1
        if (knight_bb >> 17) & NOT_H_FILE: attacks |= (knight_bb >> 17)

        KNIGHT_ATTACKS[square] = attacks

# --- Execute pre-computation at import time ---
_precompute_knight_attacks()

def get_knight_attacks(square):
    """
    Returns the pre-computed attack bitboard for a knight on a given square.

    Args:
        square (int): The square index (0-63) where the knight is located.

    Returns:
        numpy.uint64: A bitboard representing all squares the knight can attack.
    """
    return KNIGHT_ATTACKS[square]


# =============================================================================
# PAWN MOVE GENERATION
# =============================================================================
# Pawn moves are more complex as they depend on the board state (other pieces).
# These functions calculate pawn moves in a parallel fashion using bitwise ops.

# Constants for color, making the code more readable
WHITE, BLACK = 0, 1

def _get_white_pawn_moves(pawns_bb, all_pieces_bb, opponent_pieces_bb):
    """Calculates all possible moves for white pawns."""
    empty_squares = ~all_pieces_bb

    # 1. Single Push: Pawns move one step forward to an empty square.
    single_push = (pawns_bb << 8) & empty_squares

    # 2. Double Push: Pawns on rank 2 can move two steps if both squares are empty.
    double_push_candidates = (single_push & RANK_3)
    double_push = (double_push_candidates << 8) & empty_squares & RANK_4

    # 3. Captures: Pawns capture diagonally.
    capture_right = (pawns_bb << 9) & opponent_pieces_bb & NOT_A_FILE
    capture_left = (pawns_bb << 7) & opponent_pieces_bb & NOT_H_FILE

    return single_push | double_push | capture_right | capture_left

def _get_black_pawn_moves(pawns_bb, all_pieces_bb, opponent_pieces_bb):
    """Calculates all possible moves for black pawns."""
    empty_squares = ~all_pieces_bb

    # 1. Single Push
    single_push = (pawns_bb >> 8) & empty_squares

    # 2. Double Push
    double_push_candidates = (single_push & RANK_6)
    double_push = (double_push_candidates >> 8) & empty_squares & RANK_5

    # 3. Captures
    capture_right = (pawns_bb >> 7) & opponent_pieces_bb & NOT_A_FILE
    capture_left = (pawns_bb >> 9) & opponent_pieces_bb & NOT_H_FILE

    return single_push | double_push | capture_right | capture_left

def get_pawn_moves(pawns_bb, color, all_pieces_bb, opponent_pieces_bb):
    """
    Generates all possible moves for a set of pawns of a given color.
    """
    if color == WHITE:
        return _get_white_pawn_moves(pawns_bb, all_pieces_bb, opponent_pieces_bb)
    else: # BLACK
        return _get_black_pawn_moves(pawns_bb, all_pieces_bb, opponent_pieces_bb)

# =============================================================================
# SLIDING PIECE MOVE GENERATION (PYTHON-CHESS STYLE)
# =============================================================================
# This approach pre-calculates sliding piece attacks for every possible
# blocker configuration on each square and stores them in a lookup table.
# Instead of using Magic Numbers, it uses the blocker bitboard itself as the
# key to a dictionary, which is simpler to understand and implement.

# More bitboard constants, inspired by `python-chess`
BB_SQUARES = [numpy.uint64(1) << i for i in range(64)]
BB_RANK_1 = numpy.uint64(0xff)
BB_RANK_8 = numpy.uint64(0xff00000000000000)
BB_FILE_A = numpy.uint64(0x0101010101010101)
BB_FILE_H = numpy.uint64(0x8080808080808080)
BB_RANKS = [numpy.uint64(0xff) << (8 * i) for i in range(8)]
BB_FILES = [numpy.uint64(0x0101010101010101) << i for i in range(8)]

def _sliding_attacks(square, occupied, deltas):
    """
    Computes sliding attacks from a square, considering blockers.
    This is used to populate the attack tables.
    """
    attacks = numpy.uint64(0)
    for delta in deltas:
        sq = square
        while True:
            sq += delta
            # Check for leaving the board or wrapping around
            if not (0 <= sq < 64) or (abs((sq % 8) - ((sq - delta) % 8)) > 2):
                break

            attacks |= BB_SQUARES[sq]

            # Stop if a piece is hit
            if occupied & BB_SQUARES[sq]:
                break
    return attacks

def _edges(square):
    """
    Returns a bitboard of the edges of the board, relative to a square.
    Used to calculate the attack mask by excluding squares "behind" the piece.
    """
    rank, file = square // 8, square % 8
    return ((BB_RANK_1 | BB_RANK_8) & ~BB_RANKS[rank]) | \
           ((BB_FILE_A | BB_FILE_H) & ~BB_FILES[file])

def _carry_rippler(mask):
    """
    Iterates over all subsets of a given mask (bitboard).
    This is a classic bit manipulation trick.
    """
    subset = numpy.uint64(0)
    while True:
        yield subset
        subset = (subset - mask) & mask
        if subset == 0:
            break

def _generate_attack_table(deltas):
    """
    Generates the attack table for a set of sliding directions (deltas).
    Returns a tuple of (mask_table, attack_table).
    """
    mask_table = [numpy.uint64(0)] * 64
    attack_table = [{} for _ in range(64)]

    for square in range(64):
        # 1. Calculate the mask of relevant blockers
        mask = _sliding_attacks(square, numpy.uint64(0), deltas) & ~_edges(square)
        mask_table[square] = mask

        # 2. Iterate through all blocker combinations (subsets of the mask)
        for subset in _carry_rippler(mask):
            # 3. For each combination, calculate the resulting attacks
            attacks = _sliding_attacks(square, subset, deltas)
            # 4. Store it in the dictionary
            attack_table[square][subset] = attacks

    return mask_table, attack_table

# --- Pre-computation at import time ---
BISHOP_DELTAS = [-9, -7, 7, 9]
ROOK_DELTAS = [-8, -1, 1, 8]

BISHOP_MASKS, BISHOP_ATTACKS = _generate_attack_table(BISHOP_DELTAS)
ROOK_MASKS, ROOK_ATTACKS = _generate_attack_table(ROOK_DELTAS)

def get_rook_attacks(square, occupied):
    """
    Gets the pre-computed rook attacks for a given square and board occupancy,
    using the python-chess style lookup.
    """
    blockers = occupied & ROOK_MASKS[square]
    return ROOK_ATTACKS[square][blockers]

def get_bishop_attacks(square, occupied):
    """
    Gets the pre-computed bishop attacks for a given square and board occupancy,
    using the python-chess style lookup.
    """
    blockers = occupied & BISHOP_MASKS[square]
    return BISHOP_ATTACKS[square][blockers]
