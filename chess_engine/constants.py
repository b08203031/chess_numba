# chess_engine/constants.py

import numpy as np

# =============================================================================
# --- Material Values (Tapered) ---
# =============================================================================
# Values are in centipawns

# Index mapping for pieces
# PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING
#  0  ,   1   ,   2   ,  3  ,   4  ,  5

MG_MATERIAL_VALUES = np.array([100, 320, 330, 500, 900, 0], dtype=np.int32)
EG_MATERIAL_VALUES = np.array([120, 310, 340, 530, 950, 0], dtype=np.int32)

# =============================================================================
# --- Game Phase Calculation ---
# =============================================================================
# Weights for each piece type to determine the game phase
PHASE_WEIGHTS = np.array([0, 1, 1, 2, 4, 0], dtype=np.int32)
MAX_PHASE = np.sum(PHASE_WEIGHTS * np.array([8, 2, 2, 2, 1, 1])) * 2 # Pawns are not counted

# =============================================================================
# --- Piece-Square Tables (PSTs) ---
# =============================================================================
# All tables are from White's perspective.
# For Black, the board is flipped vertically (sq ^ 56).
# The arrays are flattened 8x8 matrices, index 0 is A1, 63 is H8.

def _create_pst(values):
    """Helper to create a flattened 8x8 PST from a 2D list."""
    return np.array([val for row in reversed(values) for val in row], dtype=np.int32)

# --- Middlegame PSTs ---

PAWN_PST_MG = _create_pst([
    [0,  0,  0,  0,  0,  0,  0,  0],
    [50, 50, 50, 50, 50, 50, 50, 50],
    [10, 10, 20, 30, 30, 20, 10, 10],
    [5,  5, 10, 25, 25, 10,  5,  5],
    [0,  0,  0, 20, 20,  0,  0,  0],
    [5, -5,-10,  0,  0,-10, -5,  5],
    [5, 10, 10,-20,-20, 10, 10,  5],
    [0,  0,  0,  0,  0,  0,  0,  0]
])

KNIGHT_PST_MG = _create_pst([
    [-50,-40,-30,-30,-30,-30,-40,-50],
    [-40,-20,  0,  0,  0,  0,-20,-40],
    [-30,  0, 10, 15, 15, 10,  0,-30],
    [-30,  5, 15, 20, 20, 15,  5,-30],
    [-30,  0, 15, 20, 20, 15,  0,-30],
    [-30,  5, 10, 15, 15, 10,  5,-30],
    [-40,-20,  0,  5,  5,  0,-20,-40],
    [-50,-40,-30,-30,-30,-30,-40,-50]
])

BISHOP_PST_MG = _create_pst([
    [-20,-10,-10,-10,-10,-10,-10,-20],
    [-10,  0,  0,  0,  0,  0,  0,-10],
    [-10,  0,  5, 10, 10,  5,  0,-10],
    [-10,  5,  5, 10, 10,  5,  5,-10],
    [-10,  0, 10, 10, 10, 10,  0,-10],
    [-10, 10, 10, 10, 10, 10, 10,-10],
    [-10,  5,  0,  0,  0,  0,  5,-10],
    [-20,-10,-10,-10,-10,-10,-10,-20]
])

ROOK_PST_MG = _create_pst([
    [0,  0,  0,  0,  0,  0,  0,  0],
    [5, 10, 10, 10, 10, 10, 10,  5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [0,  0,  0,  5,  5,  0,  0,  0]
])

QUEEN_PST_MG = _create_pst([
    [-20,-10,-10, -5, -5,-10,-10,-20],
    [-10,  0,  0,  0,  0,  0,  0,-10],
    [-10,  0,  5,  5,  5,  5,  0,-10],
    [-5,  0,  5,  5,  5,  5,  0, -5],
    [0,  0,  5,  5,  5,  5,  0, -5],
    [-10,  5,  5,  5,  5,  5,  0,-10],
    [-10,  0,  5,  0,  0,  0,  0,-10],
    [-20,-10,-10, -5, -5,-10,-10,-20]
])

KING_PST_MG = _create_pst([
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-20,-30,-30,-40,-40,-30,-30,-20],
    [-10,-20,-20,-20,-20,-20,-20,-10],
    [20, 20,  0,  0,  0,  0, 20, 20],
    [20, 30, 10,  0,  0, 10, 30, 20]
])


# --- Endgame PSTs ---

PAWN_PST_EG = _create_pst([
    [0,  0,  0,  0,  0,  0,  0,  0],
    [80, 80, 80, 80, 80, 80, 80, 80],
    [50, 50, 50, 50, 50, 50, 50, 50],
    [30, 30, 30, 30, 30, 30, 30, 30],
    [20, 20, 20, 20, 20, 20, 20, 20],
    [10, 10, 10, 10, 10, 10, 10, 10],
    [10, 10, 10, 10, 10, 10, 10, 10],
    [0,  0,  0,  0,  0,  0,  0,  0]
])

KNIGHT_PST_EG = _create_pst([
    [-50,-40,-30,-30,-30,-30,-40,-50],
    [-40,-20,  0,  5,  5,  0,-20,-40],
    [-30,  5, 10, 15, 15, 10,  5,-30],
    [-30,  0, 15, 20, 20, 15,  0,-30],
    [-30,  5, 15, 20, 20, 15,  5,-30],
    [-30,  0, 10, 15, 15, 10,  0,-30],
    [-40,-20,  0,  0,  0,  0,-20,-40],
    [-50,-40,-30,-30,-30,-30,-40,-50]
])

BISHOP_PST_EG = _create_pst([
    [-20,-10,-10,-10,-10,-10,-10,-20],
    [-10,  0,  0,  0,  0,  0,  0,-10],
    [-10,  0,  5, 10, 10,  5,  0,-10],
    [-10,  5,  5, 10, 10,  5,  5,-10],
    [-10,  0, 10, 10, 10, 10,  0,-10],
    [-10, 10, 10, 10, 10, 10, 10,-10],
    [-10,  5,  0,  0,  0,  0,  5,-10],
    [-20,-10,-10,-10,-10,-10,-10,-20]
])

ROOK_PST_EG = _create_pst([
    [0,  0,  0,  0,  0,  0,  0,  0],
    [5, 10, 10, 10, 10, 10, 10,  5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [0,  0,  0,  5,  5,  0,  0,  0]
])

QUEEN_PST_EG = _create_pst([
    [-20,-10,-10, -5, -5,-10,-10,-20],
    [-10,  0,  0,  0,  0,  0,  0,-10],
    [-10,  0,  5,  5,  5,  5,  0,-10],
    [-5,  0,  5,  5,  5,  5,  0, -5],
    [0,  0,  5,  5,  5,  5,  0, -5],
    [-10,  5,  5,  5,  5,  5,  0,-10],
    [-10,  0,  5,  0,  0,  0,  0,-10],
    [-20,-10,-10, -5, -5,-10,-10,-20]
])

KING_PST_EG = _create_pst([
    [-50,-40,-30,-20,-20,-30,-40,-50],
    [-30,-20,-10,  0,  0,-10,-20,-30],
    [-30,-10, 20, 30, 30, 20,-10,-30],
    [-30,-10, 30, 40, 40, 30,-10,-30],
    [-30,-10, 30, 40, 40, 30,-10,-30],
    [-30,-10, 20, 30, 30, 20,-10,-30],
    [-30,-30,  0,  0,  0,  0,-30,-30],
    [-50,-30,-30,-30,-30,-30,-30,-50]
])


# --- Aggregated PSTs for easier access ---
# The order must match the piece index mapping
# PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING

PST_MG = np.array([
    PAWN_PST_MG, KNIGHT_PST_MG, BISHOP_PST_MG, ROOK_PST_MG, QUEEN_PST_MG, KING_PST_MG
])

PST_EG = np.array([
    PAWN_PST_EG, KNIGHT_PST_EG, BISHOP_PST_EG, ROOK_PST_EG, QUEEN_PST_EG, KING_PST_EG
])


# =============================================================================
# --- King Safety Constants ---
# =============================================================================
# These values are added to the middlegame score. Endgame scores are all 0.

# --- Pawn Shield ---
# Bonus for having pawns in front of the king.
# Index 0: Pawn on its starting rank (e.g., g2 for white king on g1)
# Index 1: Pawn pushed one square (e.g., g3 for white king on g1)
PAWN_SHIELD_BONUS = np.array([
    [ 25, 10],  # MG, EG for perfect shield
    [ 12,  5]   # MG, EG for advanced shield
], dtype=np.int32)


# --- Semi-Open Files near King ---
# Penalty for semi-open files in the king's zone (king's file and adjacent files).
SEMI_OPEN_FILE_PENALTY = np.array([-15, -5], dtype=np.int32) # MG, EG


# --- Attacker Proximity ---
# Penalty based on which enemy pieces are in the king's 5x5 zone.
# Weights are for: Queen, Rook, Bishop, Knight, Pawn
ATTACKER_WEIGHTS = np.array([
    [10, 4],  # Queen
    [ 5, 2],  # Rook
    [ 3, 1],  # Bishop
    [ 3, 1],  # Knight
    [ 2, 1]   # Pawn
], dtype=np.int32)


# =============================================================================
# --- Search Constants ---
# =============================================================================

INFINITY = 32000
MATE_SCORE = 30000
MAX_PLY = 128
MATE_IN_MAX_PLY = MATE_SCORE - MAX_PLY
NO_MOVE = np.uint16(0)


# --- Aspiration Windows ---
ASPIRATION_WINDOW_SIZE = 100 # centipawns

# --- Pruning Techniques ---
NULL_MOVE_REDUCTION = 2
MAX_QUIESCENCE_DEPTH = 6

# Razoring
RAZORING_MARGIN = 250

# Futility Pruning
FP_MARGIN_D1 = 100
FP_MARGIN_D2 = 250

# Reverse Futility Pruning
RFP_MARGIN_D1 = 100

# Late Move Reductions (LMR)
LMR_MIN_DEPTH = 3           # Minimum depth to apply LMR
LMR_MIN_QUIET_MOVE_INDEX = 4 # Minimum number of quiet moves before LMR
LMR_REDUCTION = 2           # Depth reduction for LMR

# --- Static Exchange Evaluation (SEE) Threshold ---
SEE_THRESHOLD = 0  # centipawns

# =============================================================================
# --- Bitboard Utilities Constants ---
# =============================================================================

BB_SQUARES = np.array([np.uint64(1) << i for i in range(64)], dtype=np.uint64)
"""
An array of bitboards, where each bitboard has a single bit set at the corresponding square index.
"""

# De Bruijn sequence for fast bit scanning
DE_BRUIJN_SEQUENCE = np.uint64(0x03f79d71b4cb0a89)

# Lookup table for mapping the De Bruijn hash to a square index (0-63)
DE_BRUIJN_INDEX = np.array([
     0,  1, 48,  2, 57, 49, 28,  3,
    61, 58, 50, 42, 38, 29, 17,  4,
    62, 55, 59, 36, 53, 51, 43, 22,
    45, 39, 33, 30, 24, 18, 12,  5,
    63, 47, 56, 27, 60, 41, 37, 16,
    54, 35, 52, 21, 44, 32, 23, 11,
    46, 26, 40, 15, 34, 20, 31, 10,
    25, 14, 19,  9, 13,  8,  7,  6
], dtype=np.int32)


# =============================================================================
# --- Transposition Table Constants ---
# =============================================================================

TT_SIZE_MB = 64
