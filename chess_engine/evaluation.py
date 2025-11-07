# chess_engine/evaluation.py
import numba as nb
import numpy as np

from .constants import (
    MG_MATERIAL_VALUES, EG_MATERIAL_VALUES,
    PST_MG, PST_EG,
    PHASE_WEIGHTS, MAX_PHASE
)
from .bitboard_utils import get_ls1b_index, count_bits

# --- Piece Type Constants (for indexing) ---
# PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING
#  0  ,   1   ,   2   ,  3  ,   4  ,  5
WHITE_PAWN, WHITE_KNIGHT, WHITE_BISHOP, WHITE_ROOK, WHITE_QUEEN, WHITE_KING = 0, 1, 2, 3, 4, 5
BLACK_PAWN, BLACK_KNIGHT, BLACK_BISHOP, BLACK_ROOK, BLACK_QUEEN, BLACK_KING = 6, 7, 8, 9, 10, 11

# =============================================================================
# CORE EVALUATION FUNCTION
# =============================================================================

@nb.jit(nopython=True, cache=True)
def evaluate_position(board_state_flat):
    """
    Evaluates the board state using a tapered evaluation and returns a score
    from the perspective of the side to move.
    """
    piece_bbs = board_state_flat[0:12]
    side_to_move = board_state_flat[17]

    # --- Game Phase Calculation ---
    game_phase = 0
    # Loop over knights, bishops, rooks, queens for both sides
    for piece_type in range(1, 5): # KNIGHT to QUEEN
        # White pieces (index 1-4)
        game_phase += PHASE_WEIGHTS[piece_type] * count_bits(piece_bbs[piece_type])
        # Black pieces (index 7-10)
        game_phase += PHASE_WEIGHTS[piece_type] * count_bits(piece_bbs[piece_type + 6])

    # Ensure phase is within bounds [0, MAX_PHASE]
    game_phase = min(game_phase, MAX_PHASE)

    # --- Tapered Score Calculation ---
    mg_score = 0
    eg_score = 0

    # Evaluate White pieces
    for piece_type in range(6): # PAWN to KING
        bb = piece_bbs[piece_type]
        while bb > 0:
            sq = get_ls1b_index(bb)
            mg_score += MG_MATERIAL_VALUES[piece_type] + PST_MG[piece_type, sq]
            eg_score += EG_MATERIAL_VALUES[piece_type] + PST_EG[piece_type, sq]
            bb &= (bb - np.uint64(1))

    # Evaluate Black pieces
    for piece_type in range(6): # PAWN to KING
        bb = piece_bbs[piece_type + 6] # Black piece bitboards
        while bb > 0:
            sq = get_ls1b_index(bb)
            # Subtract black's score
            mg_score -= MG_MATERIAL_VALUES[piece_type] + PST_MG[piece_type, sq ^ 56]
            eg_score -= EG_MATERIAL_VALUES[piece_type] + PST_EG[piece_type, sq ^ 56]
            bb &= (bb - np.uint64(1))

    # --- Interpolation ---
    # Final score is interpolated between middlegame and endgame scores
    final_score = ((mg_score * game_phase) + (eg_score * (MAX_PHASE - game_phase))) / MAX_PHASE

    # Return score from the perspective of the side to move
    return final_score if side_to_move == 0 else -final_score
