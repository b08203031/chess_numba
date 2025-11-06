# chess_engine/evaluation.py

import numba
import numpy as np
from chess_engine.constants import (
    MG_MATERIAL_VALUES, EG_MATERIAL_VALUES,
    PST_MG, PST_EG,
    PHASE_WEIGHTS, MAX_PHASE,
    DE_BRUIJN_SEQUENCE, DE_BRUIJN_INDEX
)

# =============================================================================
# --- Core Evaluation Function ---
# =============================================================================

# Explicit signature to ensure type stability.
# Takes a 1D uint64 array and three int64 scalars, returns an int64.
@numba.jit("i8(u8[:], i8, i8, i8)", nopython=True)
def evaluate_position(bitboards, castling, ep_square, side_to_move):
    """
    Evaluates a board position using a tapered evaluation model.
    The logic for popcount and bit-scanning is inlined to avoid Numba
    type inference issues across function calls.
    """
    # Initialize scores as 64-bit integers to prevent potential overflow.
    mg_material_score = np.int64(0)
    eg_material_score = np.int64(0)
    mg_positional_score = np.int64(0)
    eg_positional_score = np.int64(0)

    current_phase = 0

    # --- Evaluate White's pieces (indices 0-5) ---
    for piece_type in range(6):
        bb = bitboards[piece_type]
        num_pieces = 0

        # Inlined popcount and bit-scanning logic
        temp_bb = bb
        while temp_bb > 0:
            num_pieces += 1
            lsb = temp_bb & -temp_bb

            # High-performance bitscan using De Bruijn sequence
            hash_val = (lsb * DE_BRUIJN_SEQUENCE) >> np.uint64(58)
            sq = DE_BRUIJN_INDEX[hash_val]

            # Add positional score for the current piece
            mg_positional_score += PST_MG[piece_type][sq]
            eg_positional_score += PST_EG[piece_type][sq]

            temp_bb ^= lsb

        current_phase += num_pieces * PHASE_WEIGHTS[piece_type]
        mg_material_score += num_pieces * MG_MATERIAL_VALUES[piece_type]
        eg_material_score += num_pieces * EG_MATERIAL_VALUES[piece_type]

    # --- Evaluate Black's pieces (indices 6-11) ---
    for piece_type in range(6):
        bb = bitboards[piece_type + 6]
        num_pieces = 0

        # Inlined popcount and bit-scanning logic
        temp_bb = bb
        while temp_bb > 0:
            num_pieces += 1
            lsb = temp_bb & -temp_bb

            # High-performance bitscan using De Bruijn sequence
            hash_val = (lsb * DE_BRUIJN_SEQUENCE) >> np.uint64(58)
            sq = DE_BRUIJN_INDEX[hash_val]

            # Subtract positional score for the opponent's piece
            mirrored_sq = sq ^ 56
            mg_positional_score -= PST_MG[piece_type][mirrored_sq]
            eg_positional_score -= PST_EG[piece_type][mirrored_sq]

            temp_bb ^= lsb

        current_phase += num_pieces * PHASE_WEIGHTS[piece_type]
        mg_material_score -= num_pieces * MG_MATERIAL_VALUES[piece_type]
        eg_material_score -= num_pieces * EG_MATERIAL_VALUES[piece_type]

    # --- Tapered Evaluation Calculation ---
    phase = min(current_phase, MAX_PHASE)

    mg_score = mg_material_score + mg_positional_score
    eg_score = eg_material_score + eg_positional_score

    # Use integer division and check for MAX_PHASE > 0 to avoid division by zero.
    if MAX_PHASE > 0:
        final_score = (mg_score * phase + eg_score * (MAX_PHASE - phase)) // MAX_PHASE
    else:
        final_score = mg_score

    # --- Final Adjustment ---
    perspective_multiplier = np.int64(1) if side_to_move == 0 else np.int64(-1)

    return final_score * perspective_multiplier
