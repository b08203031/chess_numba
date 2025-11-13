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
# --- Other Evaluation Constants ---
# =============================================================================

# --- Initiative Bonus ---
# A small bonus awarded to the side to move, acknowledging the advantage of having the turn.
# This bonus is tapered and disappears in the endgame.
INITIATIVE_BONUS = 10 # centipawns
INITIATIVE_PHASE_THRESHOLD = MAX_PHASE * 0.4 # Apply only when phase is above 40% of max


# =============================================================================
# --- Mobility Constants ---
# =============================================================================
# Bonus for piece mobility, calculated as: (move_count - base_moves) * weight.
# This rewards active pieces and penalizes pieces that are blocked or restricted.

# --- Knight Mobility ---
KNIGHT_MOBILITY_BASE_MOVES = 4
KNIGHT_MOBILITY_WEIGHT = np.array([4, 2], dtype=np.int32) # MG, EG

# --- Bishop Mobility ---
BISHOP_MOBILITY_BASE_MOVES = 5
BISHOP_MOBILITY_WEIGHT = np.array([4, 2], dtype=np.int32) # MG, EG

# --- Rook Mobility ---
ROOK_MOBILITY_BASE_MOVES = 6
ROOK_MOBILITY_WEIGHT = np.array([3, 1], dtype=np.int32) # MG, EG

# --- Queen Mobility ---
QUEEN_MOBILITY_BASE_MOVES = 8
QUEEN_MOBILITY_WEIGHT = np.array([2, 1], dtype=np.int32) # MG, EG


# =============================================================================
# --- Piece Coordination Constants ---
# =============================================================================

# --- Bishop Pair ---
# Bonus for having both bishops. This bonus is generally stronger in open positions.
BISHOP_PAIR_BONUS = np.array([10, 20], dtype=np.int32) # MG, EG

# --- Rook on Open/Semi-Open File ---
# Bonus for a rook on a file with no friendly pawns (semi-open)
# or no pawns at all (open).
ROOK_ON_SEMI_OPEN_FILE_BONUS = np.array([15, 10], dtype=np.int32) # MG, EG
ROOK_ON_OPEN_FILE_BONUS = np.array([25, 15], dtype=np.int32) # MG, EG


# =============================================================================
# --- Pawn Structure Constants ---
# =============================================================================

# --- Passed Pawns ---
# Bonus for having a passed pawn, scaled by its rank. (Values increased significantly)
# Index corresponds to the pawn's rank (1-8, though rank 1 and 8 are not used for pawns).
PASSED_PAWN_BONUS = np.array([
    # MG, EG
    [  0,   0], # Rank 1
    [ 10,  20], # Rank 2
    [ 20,  40], # Rank 3
    [ 35,  70], # Rank 4
    [ 50, 100], # Rank 5
    [ 80, 180], # Rank 6
    [150, 300], # Rank 7
    [  0,   0]  # Rank 8
], dtype=np.int32)

# --- Isolated Pawns ---
# Penalty for each isolated pawn on a file.
ISOLATED_PAWN_PENALTY = np.array([-10, -15], dtype=np.int32) # MG, EG

# --- Doubled Pawns ---
# Penalty for each doubled pawn on a file.
DOUBLED_PAWN_PENALTY = np.array([-15, -20], dtype=np.int32) # MG, EG


# =============================================================================
# --- King Safety Constants (NEW - based on Chessprogramming Wiki) ---
# =============================================================================

# --- Phase 2: Attacking the King Zone ---
KING_SAFETY_ATTACK_UNITS = np.array([2, 2, 3, 5], dtype=np.int32) # N, B, R, Q
KING_SAFETY_TABLE = np.array([
    0, 0, 1, 2, 4, 6, 9, 12, 16, 20, 25, 30, 36, 42, 49, 56,
    64, 72, 81, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
    210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340,
    350, 360, 370, 380, 390, 400
] + [400] * 50, dtype=np.int32)

# --- Phase 3: King Tropism ---
KING_TROPISM_MAX_DISTANCE = 14 # Max MANHATTAN distance
KING_TROPISM_WEIGHTS = np.array([1, 2, 2, 3, 5], dtype=np.int32) # P, N, B, R, Q

# --- Phase 4: Advanced & Dynamic ---
PAWN_STORM_PENALTY = -5 # Penalty for each enemy pawn near the king
SCALING_WEIGHTS = np.array([0, 4, 4, 6, 10], dtype=np.int32) # N, B, R, Q - for scaling factor
MAX_SCALING_MATERIAL = (2*4 + 2*4 + 2*6 + 1*10) # Sum of all weights for one side

EG_SAFETY_SCALE = 0.5 # Scale down endgame king safety impact

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
# Master switches for new pruning techniques
ENABLE_LMP = True           # Late Move Pruning
ENABLE_PROBCUT = True       # ProbCut
ENABLE_DELTA_PRUNING = True # Delta Pruning in Quiescence Search
ENABLE_IID = True           # Internal Iterative Deepening
ENABLE_SINGULAR_EXTENSIONS = True # Singular Extensions


# IID and Singular Extension Parameters
MIN_SINGULAR_DEPTH = 6
SINGULAR_EXTENSION_MARGIN = 150 # centipawns


# Master switches for existing pruning techniques
ENABLE_NMP = True           # Null Move Pruning
ENABLE_RAZORING = True      # Razoring
ENABLE_FP = True            # Futility Pruning
ENABLE_RFP = True           # Reverse Futility Pruning
ENABLE_LMR = True           # Late Move Reductions

NULL_MOVE_REDUCTION = 2
MAX_QUIESCENCE_DEPTH = 5

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

# Late Move Pruning (LMP) - Prune moves after a certain number of quiet moves have been searched
LMP_MOVE_COUNT = np.array([
 # depth: 0  1  2   3   4   5   6   7   8   9  10 ...
          0, 6, 10, 14, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 90, 90, 90
] + [90] * (MAX_PLY - 17), dtype=np.int32)


# ProbCut
PROBCUT_R = 2
PROBCUT_R_PRIME = 4
PROBCUT_MARGIN = 150 # centipawns

# Delta Pruning
DELTA_PRUNING_MARGIN = 500

# --- Static Exchange Evaluation (SEE) Threshold ---
SEE_THRESHOLD = 0  # centipawns
ENABLE_SEE_IN_QUIESCENCE = True # Master switch to enable/disable SEE in quiescence search

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

TT_SIZE_MB = 256
