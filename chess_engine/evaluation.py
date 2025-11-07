# chess_engine/evaluation.py

import numba
import numpy as np

from chess_engine.constants import (
    MG_MATERIAL_VALUES, EG_MATERIAL_VALUES,
    PST_MG, PST_EG,
    PHASE_WEIGHTS, MAX_PHASE
)

from chess_engine.bitboard_utils import count_bits, get_ls1b_index
from chess_engine.engine_types import piece_bbs_signature, occupancy_bbs_signature, game_state_signature


@numba.njit(numba.int32(piece_bbs_signature, occupancy_bbs_signature, game_state_signature), cache=True)
def evaluate_position(piece_bbs, occupancy_bbs, game_state):
    """
    Evaluates the board state using a tapered evaluation and returns a score
    from the perspective of the current player.

    The evaluation includes:
    1. Material value (middlegame and endgame).
    2. Piece-square tables (middlegame and endgame).
    3. Tapered evaluation based on the game phase.

    Returns:
        np.int32: The score in centipawns from the current player's perspective.
                  A positive score indicates an advantage for the current player.
    """
    side_to_move = game_state[0]

    # --- 1. Calculate Game Phase ---
    # The phase determines the balance between middlegame and endgame evaluation.
    # It starts at MAX_PHASE and decreases as pieces are traded off.
    phase = np.int32(0)
    # Piece types are indexed 0-5 for White, 6-11 for Black.
    # We only consider non-pawn, non-king pieces for phase calculation.
    # White pieces (Knight, Bishop, Rook, Queen)
    phase += count_bits(piece_bbs[1]) * PHASE_WEIGHTS[1]
    phase += count_bits(piece_bbs[2]) * PHASE_WEIGHTS[2]
    phase += count_bits(piece_bbs[3]) * PHASE_WEIGHTS[3]
    phase += count_bits(piece_bbs[4]) * PHASE_WEIGHTS[4]
    # Black pieces (Knight, Bishop, Rook, Queen)
    phase += count_bits(piece_bbs[7]) * PHASE_WEIGHTS[1]
    phase += count_bits(piece_bbs[8]) * PHASE_WEIGHTS[2]
    phase += count_bits(piece_bbs[9]) * PHASE_WEIGHTS[3]
    phase += count_bits(piece_bbs[10]) * PHASE_WEIGHTS[4]

    # Clamp the phase to the maximum value
    phase = min(phase, MAX_PHASE)

    # --- 2. Calculate Middlegame and Endgame Scores ---
    mg_score = np.int32(0)
    eg_score = np.int32(0)

    # Evaluate White pieces
    for piece_type in range(6): # 0-5: P, N, B, R, Q, K
        bb = piece_bbs[piece_type]
        while bb:
            sq = get_ls1b_index(bb)
            mg_score += MG_MATERIAL_VALUES[piece_type] + PST_MG[piece_type][sq]
            eg_score += EG_MATERIAL_VALUES[piece_type] + PST_EG[piece_type][sq]
            bb &= bb - np.uint64(1) # Clear LSB

    # Evaluate Black pieces
    for piece_type in range(6): # 0-5 for piece type, but access bb at 6-11
        bb = piece_bbs[piece_type + 6]
        while bb:
            sq = get_ls1b_index(bb)
            # Scores are subtracted for the opponent
            mg_score -= MG_MATERIAL_VALUES[piece_type] + PST_MG[piece_type][sq ^ 56]
            eg_score -= EG_MATERIAL_VALUES[piece_type] + PST_EG[piece_type][sq ^ 56]
            bb &= bb - np.uint64(1) # Clear LSB

    # --- 3. Interpolate Scores based on Game Phase ---
    # This blends the MG and EG scores.
    # As the game progresses (phase decreases), the EG score becomes more influential.
    final_score = (mg_score * phase + eg_score * (MAX_PHASE - phase)) // MAX_PHASE

    # --- 4. Return Score from the Perspective of the Side to Move ---
    if side_to_move == 0: # White to move
        return np.int32(final_score)
    else: # Black to move
        return np.int32(-final_score)
