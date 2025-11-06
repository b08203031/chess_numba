# chess_engine/evaluation.py

import numba
import numpy as np

# =============================================================================
# EVALUATION CONSTANTS
# =============================================================================

# Piece values in centipawns
PAWN_VALUE = 100
KNIGHT_VALUE = 320
BISHOP_VALUE = 330
ROOK_VALUE = 500
QUEEN_VALUE = 900
KING_VALUE = 20000

MATERIAL_VALUES = np.array([
    PAWN_VALUE, KNIGHT_VALUE, BISHOP_VALUE, ROOK_VALUE, QUEEN_VALUE, KING_VALUE,
    -PAWN_VALUE, -KNIGHT_VALUE, -BISHOP_VALUE, -ROOK_VALUE, -QUEEN_VALUE, -KING_VALUE
], dtype=np.int32)

# Piece-Square Tables (PSTs)
# These tables are for White's perspective. Black's scores are mirrored.
# The values are chosen to encourage good development and control of the center.

PAWN_PST = np.array([
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
], dtype=np.int32)

KNIGHT_PST = np.array([
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
], dtype=np.int32)

BISHOP_PST = np.array([
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
], dtype=np.int32)

ROOK_PST = np.array([
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0
], dtype=np.int32)

QUEEN_PST = np.array([
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
], dtype=np.int32)

KING_PST = np.array([
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20
], dtype=np.int32)

PIECE_PSTS = (
    PAWN_PST, KNIGHT_PST, BISHOP_PST, ROOK_PST, QUEEN_PST, KING_PST
)

# Numba helper to get LSB index
@numba.njit(numba.int32(numba.uint64), cache=True)
def count_bits(bb: np.uint64) -> int:
    count = 0
    while bb > 0:
        bb &= (bb - np.uint64(1))
        count += 1
    return count

@numba.njit(numba.uint8(numba.uint64), cache=True)
def get_ls1b_index(bb):
    return count_bits((bb & -bb) - np.uint64(1))

# =============================================================================
# CORE EVALUATION FUNCTION
# =============================================================================

@numba.jit(nopython=True, cache=True)
def evaluate_position(board_state):
    """
    Evaluates the board state and returns a score from White's perspective.
    Positive score = White's advantage.
    Negative score = Black's advantage.
    """
    (wp_bb, wn_bb, wb_bb, wr_bb, wq_bb, wk_bb,
     bp_bb, bn_bb, bb_bb, br_bb, bq_bb, bk_bb,
     _, _, _, _, _, side_to_move, _) = board_state

    score = 0

    # Pack bitboards into arrays for easier iteration
    white_bbs = (wp_bb, wn_bb, wb_bb, wr_bb, wq_bb, wk_bb)
    black_bbs = (bp_bb, bn_bb, bb_bb, br_bb, bq_bb, bk_bb)

    # 1. Material and Positional Score
    # White pieces
    for piece_type in range(6):
        bb = np.uint64(white_bbs[piece_type])
        pst = PIECE_PSTS[piece_type]
        while bb:
            sq = get_ls1b_index(bb)
            score += MATERIAL_VALUES[piece_type]
            score += pst[sq]
            bb &= (bb - np.uint64(1)) # Clear LSB

    # Black pieces
    for piece_type in range(6):
        bb = np.uint64(black_bbs[piece_type])
        pst = PIECE_PSTS[piece_type]
        while bb:
            sq = get_ls1b_index(bb)
            score += MATERIAL_VALUES[piece_type + 6] # Black piece values
            # For black, the PST index is flipped vertically
            score -= pst[sq ^ 56]
            bb &= (bb - np.uint64(1)) # Clear LSB

    # 2. Return score relative to the current player
    # The search function expects the score from the perspective of the side to move.
    if side_to_move == 0: # WHITE
        return score
    else: # BLACK
        return -score
