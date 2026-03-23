# chess_engine/constants.py

import numpy as np
import math

"""
此模組定義了西洋棋引擎中使用的所有常量。
包含了棋盤表示、棋子值、位置分數（PST）、搜尋參數、剪枝開關以及位元棋盤的輔助常量。
"""

# =============================================================================
# --- Basic Definitions / 基本定義 ---
# =============================================================================

# --- MovePicker Stages ---
STAGE_TT_MOVE       = 0
STAGE_GEN_CAPTURES  = 1
STAGE_GOOD_CAPTURES = 2
STAGE_GEN_QUIETS    = 3
STAGE_GOOD_QUIETS   = 4
STAGE_BAD_CAPTURES  = 5
STAGE_BAD_QUIETS    = 6
STAGE_DONE          = 7

WHITE, BLACK = 0, 1
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 0, 1, 2, 3, 4, 5

# =============================================================================
# --- Material Values (Tapered) / 棋子材質值（漸進式） ---
# =============================================================================
# Values are in centipawns / 單位為分（centipawns）

# Index mapping for pieces / 棋子索引映射
# PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING
#  0  ,   1   ,   2   ,  3  ,   4  ,  5

# 中局（Middle Game）材質值
MG_MATERIAL_VALUES = np.array([100, 342, 342, 470, 902, 0], dtype=np.int32)
# 殘局（End Game）材質值
EG_MATERIAL_VALUES = np.array([120, 320, 345, 496, 953, 0], dtype=np.int32)

# =============================================================================
# --- Game Phase Calculation / 遊戲階段計算 ---
# =============================================================================
# Weights for each piece type to determine the game phase / 每個棋子類型的權重，用於確定遊戲階段
PHASE_WEIGHTS = np.array([0, 1, 1, 2, 4, 0], dtype=np.int32)
# 最大階段值（初始局面所有非兵棋子的權重總和）
MAX_PHASE = np.sum(PHASE_WEIGHTS * np.array([8, 2, 2, 2, 1, 1])) * 2 # Pawns are not counted

# =============================================================================
# --- Piece-Square Tables (PSTs) / 棋子位置分數表 ---
# =============================================================================
# All tables are from White's perspective.
# For Black, the board is flipped vertically (sq ^ 56).
# The arrays are flattened 8x8 matrices, index 0 is A1, 63 is H8.
# 所有表格均以白方視角定義。黑方使用時需垂直翻轉棋盤。
# 陣列為展平的 8x8 矩陣，索引 0 為 A1，63 為 H8。

# def _create_pst(values):
#     """
#     輔助函式：從 2D 列表創建展平的 8x8 PST 陣列。
    
#     Args:
#         values (list of list of int): 8x8 的分數列表。
        
#     Returns:
#         np.array: 展平的一維 numpy 陣列。
#     """
#     return np.array([val for row in reversed(values) for val in row], dtype=np.int32)

# # --- Middlegame PSTs / 中局位置分數表 ---

# PAWN_PST_MG = _create_pst([
#     [0,  0,  0,  0,  0,  0,  0,  0],
#     [50, 50, 50, 50, 50, 50, 50, 50],
#     [10, 10, 20, 30, 30, 20, 10, 10],
#     [5,  5, 10, 25, 25, 10,  5,  5],
#     [0,  0,  0, 20, 20,  0,  0,  0],
#     [5, -5,-10,  0,  0,-10, -5,  5],
#     [5, 10, 10,-20,-20, 10, 10,  5],
#     [0,  0,  0,  0,  0,  0,  0,  0]
# ])

# KNIGHT_PST_MG = _create_pst([
#     [-50,-40,-30,-30,-30,-30,-40,-50],
#     [-40,-20,  0,  0,  0,  0,-20,-40],
#     [-30,  0, 10, 15, 15, 10,  0,-30],
#     [-30,  5, 15, 20, 20, 15,  5,-30],
#     [-30,  0, 15, 20, 20, 15,  0,-30],
#     [-30,  5, 10, 15, 15, 10,  5,-30],
#     [-40,-20,  0,  5,  5,  0,-20,-40],
#     [-50,-40,-30,-30,-30,-30,-40,-50]
# ])

# BISHOP_PST_MG = _create_pst([
#     [-20,-10,-10,-10,-10,-10,-10,-20],
#     [-10,  0,  0,  0,  0,  0,  0,-10],
#     [-10,  0,  5, 10, 10,  5,  0,-10],
#     [-10,  5,  5, 10, 10,  5,  5,-10],
#     [-10,  0, 10, 10, 10, 10,  0,-10],
#     [-10, 10, 10, 10, 10, 10, 10,-10],
#     [-10,  5,  0,  0,  0,  0,  5,-10],
#     [-20,-10,-10,-10,-10,-10,-10,-20]
# ])

# ROOK_PST_MG = _create_pst([
#     [0,  0,  0,  0,  0,  0,  0,  0],
#     [5, 10, 10, 10, 10, 10, 10,  5],
#     [-5,  0,  0,  0,  0,  0,  0, -5],
#     [-5,  0,  0,  0,  0,  0,  0, -5],
#     [-5,  0,  0,  0,  0,  0,  0, -5],
#     [-5,  0,  0,  0,  0,  0,  0, -5],
#     [-5,  0,  0,  0,  0,  0,  0, -5],
#     [0,  0,  0,  5,  5,  0,  0,  0]
# ])

# QUEEN_PST_MG = _create_pst([
#     [-20,-10,-10, -5, -5,-10,-10,-20],
#     [-10,  0,  0,  0,  0,  0,  0,-10],
#     [-10,  0,  5,  5,  5,  5,  0,-10],
#     [-5,  0,  5,  5,  5,  5,  0, -5],
#     [0,  0,  5,  5,  5,  5,  0, -5],
#     [-10,  5,  5,  5,  5,  5,  0,-10],
#     [-10,  0,  5,  0,  0,  0,  0,-10],
#     [-20,-10,-10, -5, -5,-10,-10,-20]
# ])

# KING_PST_MG = _create_pst([
#     [-30,-40,-40,-50,-50,-40,-40,-30],
#     [-30,-40,-40,-50,-50,-40,-40,-30],
#     [-30,-40,-40,-50,-50,-40,-40,-30],
#     [-30,-40,-40,-50,-50,-40,-40,-30],
#     [-20,-30,-30,-40,-40,-30,-30,-20],
#     [-10,-20,-20,-20,-20,-20,-20,-10],
#     [20, 20,  0,  0,  0,  0, 20, 20],
#     [20, 30, 10,  0,  0, 10, 30, 20]
# ])


# # --- Endgame PSTs / 殘局位置分數表 ---

# PAWN_PST_EG = _create_pst([
#     [0,  0,  0,  0,  0,  0,  0,  0],
#     [80, 80, 80, 80, 80, 80, 80, 80],
#     [50, 50, 50, 50, 50, 50, 50, 50],
#     [30, 30, 30, 30, 30, 30, 30, 30],
#     [20, 20, 20, 20, 20, 20, 20, 20],
#     [10, 10, 10, 10, 10, 10, 10, 10],
#     [10, 10, 10, 10, 10, 10, 10, 10],
#     [0,  0,  0,  0,  0,  0,  0,  0]
# ])

# KNIGHT_PST_EG = _create_pst([
#     [-50,-40,-30,-30,-30,-30,-40,-50],
#     [-40,-20,  0,  5,  5,  0,-20,-40],
#     [-30,  5, 10, 15, 15, 10,  5,-30],
#     [-30,  0, 15, 20, 20, 15,  0,-30],
#     [-30,  5, 15, 20, 20, 15,  5,-30],
#     [-30,  0, 10, 15, 15, 10,  0,-30],
#     [-40,-20,  0,  0,  0,  0,-20,-40],
#     [-50,-40,-30,-30,-30,-30,-40,-50]
# ])

# # SF11-inspired EG: bishops prefer activity and central diagonals in endgame
# BISHOP_PST_EG = _create_pst([
#     [-46,-24,-30,-10,-10,-30,-24,-46],
#     [-30,-10,-14,  0,  0,-14,-10,-30],
#     [-13,  0,  0,  8,  8,  0,  0,-13],
#     [-16,  0, 14, 13, 13, 14,  0,-16],
#     [-14,  0, 12, 13, 13, 12,  0,-14],
#     [-24,  5,  3,  5,  5,  3,  5,-24],
#     [-25,-16,  0,  1,  1,  0,-16,-25],
#     [-37,-34,-30,-19,-19,-30,-34,-37]
# ])

# # SF11-inspired EG: rooks more active, penalize confinement less, reward 7th-rank
# ROOK_PST_EG = _create_pst([
#     [ 18,  0, 19, 13, 13, 19,  0, 18],
#     [  4,  5, 20, -5, -5, 20,  5,  4],
#     [  6, -8, -2,  8,  8, -2, -8,  6],
#     [ -6,  1, -9,  7,  7, -9,  1, -6],
#     [ -5,  8,  7, -6, -6,  7,  8, -5],
#     [  6,  1, -7, 10, 10, -7,  1,  6],
#     [ -12, -9, -1, -2, -2, -1, -9,-12],
#     [ -9,-13,-10, -9, -9,-10,-13, -9]
# ])

# # SF11-inspired EG: queens centralize more aggressively, avoid corners
# QUEEN_PST_EG = _create_pst([
#     [-75,-52,-43,-36,-36,-43,-52,-75],
#     [-57,-31,-22, -4, -4,-22,-31,-57],
#     [-47,-18, -9,  3,  3, -9,-18,-47],
#     [-26, -3, 13, 24, 24, 13, -3,-26],
#     [-29, -6,  9, 21, 21,  9, -6,-29],
#     [-39,-18,-12,  1,  1,-12,-18,-39],
#     [-55,-27,-24, -8, -8,-24,-27,-55],
#     [-69,-57,-47,-36,-36,-47,-57,-69]
# ])

# KING_PST_EG = _create_pst([
#     [-50,-40,-30,-20,-20,-30,-40,-50],
#     [-30,-20,-10,  0,  0,-10,-20,-30],
#     [-30,-10, 20, 30, 30, 20,-10,-30],
#     [-30,-10, 30, 40, 40, 30,-10,-30],
#     [-30,-10, 30, 40, 40, 30,-10,-30],
#     [-30,-10, 20, 30, 30, 20,-10,-30],
#     [-30,-30,  0,  0,  0,  0,-30,-30],
#     [-50,-30,-30,-30,-30,-30,-30,-50]
# ])


# --- Aggregated PSTs for easier access / 聚合 PST 以便於訪問 ---
# The order must match the piece index mapping / 順序必須與棋子索引映射匹配
# PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING

# PST_MG = np.array([
#     PAWN_PST_MG, KNIGHT_PST_MG, BISHOP_PST_MG, ROOK_PST_MG, QUEEN_PST_MG, KING_PST_MG
# ])

# PST_EG = np.array([
#     PAWN_PST_EG, KNIGHT_PST_EG, BISHOP_PST_EG, ROOK_PST_EG, QUEEN_PST_EG, KING_PST_EG
# ])

PST_MG = np.array([
    [3, 2, 2, 2, 0, -6, 4, -1, -6, -7, -5, -17, -5, 1, 2, -9, -10, -8, -7, -14, 0, 1, 4, -1, -1, -7, -1, -11, 9, 0, -9, -4, 7, 0, -1, 5, 10, 4, 0, 2, 5, 1, 18, 16, 28, 15, 3, 8, 47, 51, 44, 47, 49, 50, 51, 45, -3, -4, 6, -1, 3, 3, 1, 7],
    [-53, -21, -24, -25, -21, -22, -33, -56, -40, -15, 0, 5, 3, -3, -16, -33, -19, 5, 5, 17, 17, 2, 3, -23, -13, -3, 14, 22, 4, 14, -11, -21, -22, 4, 15, 13, -16, -3, -1, -25, -34, 0, 15, 16, 15, 14, -5, -33, -38, -14, 1, 1, 0, 2, -17, -40, -55, -43, -28, -25, -26, -30, -36, -51],
    [-21, -4, 4, -3, -5, 7, -10, -21, -6, 13, -1, -4, 10, 3, 2, -11, -7, 9, 7, 1, -2, 14, 6, -4, -6, 5, 3, 6, 3, -1, 2, -5, -13, 14, 8, 5, 5, -5, 6, -14, -8, 6, 3, 7, 7, 0, -6, -12, -1, 9, 3, -3, 7, -2, 2, -11, -24, -9, -10, -10, -12, -9, -7, -19],
    [-4, 0, 0, 3, 4, -5, 10, 6, 1, -5, 1, 0, -4, -3, 1, -6, -1, -3, -6, -2, -1, -2, -1, -3, -6, 1, 1, 7, 1, -3, -2, -5, -12, 3, 2, -6, 3, -2, 3, -5, -4, -5, 0, -3, -1, -4, 1, -8, -2, 7, 4, 7, 5, 14, 11, -1, -1, -2, -4, 3, 3, -4, 1, -3],
    [-13, -1, -3, 13, 1, -6, -7, -19, -10, -2, 8, 4, 10, 3, 2, -7, -5, 10, -1, -4, 0, -5, 0, -11, 2, -1, 6, 0, 1, 2, -2, -3, -8, -1, 8, 4, 1, 2, -1, -5, -10, 7, 4, -1, 0, 9, 3, -10, -10, -5, 1, -1, 0, -1, -1, -11, -21, -9, -6, -3, -4, -8, -8, -20],
    [21, 25, 15, 6, 7, 13, 16, 16, 17, 26, 8, 2, 4, 5, 29, 20, -8, -17, -17, -23, -17, -25, -24, -4, -20, -27, -33, -40, -35, -33, -29, -23, -30, -36, -41, -50, -50, -43, -35, -35, -31, -40, -43, -46, -47, -39, -37, -30, -34, -35, -39, -50, -47, -43, -44, -27, -31, -45, -37, -53, -54, -44, -39, -32],
], dtype=np.int32)

PST_EG = np.array([
    [4, 2, 0, -8, -2, -1, 2, 2, 3, 13, 13, 11, 8, 10, 9, 4, 8, 11, 9, 19, 10, 8, 11, 8, 17, 13, 10, 5, -4, -6, 0, 13, 19, 18, 13, 5, 11, 8, 15, 15, 37, 40, 38, 42, 40, 44, 44, 39, 75, 75, 74, 78, 76, 78, 73, 72, -4, 1, -2, 3, 0, -4, 2, 6],
    [-50, -35, -29, -27, -25, -25, -35, -48, -40, -23, 6, -1, 1, 2, -19, -36, -27, 1, 5, 15, 16, 6, -4, -26, -29, 8, 16, 19, 17, 10, 7, -26, -25, 3, 13, 19, 16, 11, 3, -31, -31, 9, 9, 14, 15, 11, 10, -31, -35, -27, 3, 10, 4, 5, -17, -37, -49, -44, -33, -27, -36, -29, -38, -45],
    [-39, -31, -19, -13, -16, -18, -38, -31, -23, -11, -7, -3, 2, -7, -14, -25, -23, 4, 4, -3, -1, 7, 6, -22, -11, 3, 10, 6, 9, 6, -1, -18, -14, 4, 9, 7, 11, 11, -7, -19, -9, 0, 2, 10, 5, 3, 1, -12, -27, -8, -15, 5, -10, -14, -10, -34, -47, -21, -29, -10, -3, -33, -26, -44],
    [-3, -11, -8, -9, -12, -8, -15, -11, -9, -6, 1, -3, 2, -4, -10, -18, 5, -4, -5, 8, 4, -11, -4, 8, 3, 8, 11, -9, -8, 10, 7, -6, -2, 0, -11, 1, 6, -10, 0, -7, 2, -6, -2, 4, -2, -2, -9, 3, -5, -4, 8, -4, -11, 18, -1, 2, 18, 1, 23, 8, 18, 21, 1, 18],
    [-70, -50, -43, -30, -36, -47, -57, -71, -50, -31, -21, -1, -10, -32, -27, -56, -40, -14, -8, 2, 1, -8, -20, -38, -23, -10, 9, 21, 16, 12, -7, -26, -26, -12, 11, 24, 27, 13, 1, -29, -51, -12, -14, 1, 0, -13, -18, -53, -54, -33, -15, -6, 0, -19, -32, -55, -72, -54, -43, -36, -34, -48, -58, -78],
    [-51, -32, -36, -24, -21, -13, -36, -40, -29, -27, 0, 13, 4, 6, -25, -23, -29, -12, 16, 29, 24, 19, -5, -19, -29, -11, 29, 38, 32, 21, -15, -36, -29, -19, 30, 35, 34, 26, -6, -28, -26, -12, 20, 25, 20, 18, -12, -29, -32, -23, -10, 2, 4, -5, -18, -25, -50, -35, -37, -20, -18, -32, -37, -45],
], dtype=np.int32)
# =============================================================================
# --- Other Evaluation Constants / 其他評估常量 ---
# =============================================================================

# --- Initiative Bonus / 主動權獎勵 ---
# A small bonus awarded to the side to move, acknowledging the advantage of having the turn.
# This bonus is tapered and disappears in the endgame.
# 給予輪到行棋一方的小獎勵，承認擁有下棋權的優勢。此獎勵是漸進的，在殘局中會消失。
INITIATIVE_BONUS = 21 # centipawns
INITIATIVE_PHASE_THRESHOLD = MAX_PHASE * 0.4 # Apply only when phase is above 40% of max / 僅在階段值高於最大值的 40% 時應用


# =============================================================================
# --- Mobility Constants / 機動性常量 ---
# =============================================================================
# Bonus for piece mobility, calculated as: (move_count - base_moves) * weight.
# This rewards active pieces and penalizes pieces that are blocked or restricted.
# 棋子機動性獎勵，原本的計算方式為：(移動次數 - 基礎移動數) * 權重，現已改為查表。
# 這獎勵活躍的棋子，懲罰被阻擋或受限的棋子。

# --- Knight Mobility (SF11 MobilityBonus × Pawn ratio MG×0.585 EG×0.42) --- max 8 squares
KNIGHT_MOBILITY_BONUS = np.array([
    [-28, -29],
    [1, -16],
    [18, -4],
    [19, 7],
    [19, 7],
    [22, 7],
    [22, 7],
    [22, 8],
    [23, 10],
], dtype=np.int32)  # index 0..8

# --- Bishop Mobility (SF11 MobilityBonus × Pawn ratio MG×0.585 EG×0.42) --- max 13 squares
BISHOP_MOBILITY_BONUS = np.array([
    [-7, -23],
    [10, 0],
    [22, 4],
    [26, 9],
    [27, 11],
    [27, 21],
    [27, 22],
    [27, 25],
    [29, 28],
    [35, 31],
    [42, 32],
    [49, 41],
    [52, 44],
    [55, 46],
], dtype=np.int32)  # index 0..13

# --- Rook Mobility (SF11 MobilityBonus × Pawn ratio MG×0.585 EG×0.42) --- max 14 squares
ROOK_MOBILITY_BONUS = np.array([
    [-8, -27],
    [17, -4],
    [27, 20],
    [27, 33],
    [27, 36],
    [27, 40],
    [28, 48],
    [29, 48],
    [29, 49],
    [29, 50],
    [30, 56],
    [30, 60],
    [31, 69],
    [31, 70],
    [32, 71],
], dtype=np.int32)  # index 0..14

# --- Queen Mobility (SF11 MobilityBonus × Pawn ratio MG×0.585 EG×0.42) --- max 27 squares
QUEEN_MOBILITY_BONUS = np.array([
    [-27, -17],
    [-5, -11],
    [23, 8],
    [23, 10],
    [23, 18],
    [32, 26],
    [33, 29],
    [34, 33],
    [36, 34],
    [36, 38],
    [36, 41],
    [36, 43],
    [36, 47],
    [37, 47],
    [37, 53],
    [38, 53],
    [47, 63],
    [50, 65],
    [50, 66],
    [53, 66],
    [57, 66],
    [62, 73],
    [66, 75],
    [67, 76],
    [68, 77],
    [68, 81],
    [71, 87],
    [71, 92],
], dtype=np.int32)  # index 0..27


# =============================================================================
# --- Piece Coordination Constants / 棋子協同常量 ---
# =============================================================================

# --- Bishop Pair / 雙象優勢 ---
# Bonus for having both bishops. This bonus is generally stronger in open positions.
# 擁有雙象的獎勵。這個獎勵在開放局面中通常更強。
BISHOP_PAIR_BONUS = np.array([26, 26], dtype=np.int32) # MG, EG

# --- Rook on Open/Semi-Open File / 車在開放線/半開放線 ---
# Bonus for a rook on a file with no friendly pawns (semi-open)
# or no pawns at all (open).
# 車在沒有己方兵（半開放線）或完全沒有兵（開放線）的直線上的獎勵。
ROOK_ON_SEMI_OPEN_FILE_BONUS = np.array([12, 8], dtype=np.int32) # MG, EG
ROOK_ON_OPEN_FILE_BONUS = np.array([15, 4], dtype=np.int32) # MG, EG

ROOK_ON_SEVENTH_BONUS = np.array([0, 12], dtype=np.int32) # MG, EG

# =============================================================================
# --- Pawn Structure Constants / 兵型結構常量 ---
# =============================================================================

# --- Passed Pawns / 通路兵 ---
# Bonus for having a passed pawn, scaled by its rank. (Values increased significantly)
# Index corresponds to the pawn's rank (1-8, though rank 1 and 8 are not used for pawns).
# 通路兵的獎勵，根據其橫排進行縮放。（數值已顯著增加）
# 索引對應於兵的橫排（1-8，雖然 1 和 8 橫排不用於兵）。
PASSED_PAWN_BONUS = np.array([
    [0, 0], # Rank 1
    [0, 0], # Rank 2
    [1, 2], # Rank 3
    [7, 12], # Rank 4
    [22, 24], # Rank 5
    [54, 98], # Rank 6
    [143, 229], # Rank 7
    [0, 0], # Rank 8
], dtype=np.int32)

# Candidate Passed Pawns Bonus by Rank (0-7).
# Pawns that are not passed yet, but can become passed easily (e.g. facing only one enemy pawn that is at the same rank or ahead on adjacent file).
# Bonus is roughly half that of a true passed pawn.
CANDIDATE_PASSED_PAWN_BONUS = np.array([
    [0, 0], # Rank 1
    [0, 0], # Rank 2
    [0, 0], # Rank 3
    [0, 0], # Rank 4
    [11, 13], # Rank 5
    [35, 71], # Rank 6
    [80, 130], # Rank 7
    [0, 0], # Rank 8
], dtype=np.int32)

# --- Isolated Pawns / 孤兵 ---
# Penalty for each isolated pawn on a file.
# 每一個孤兵的懲罰。
ISOLATED_PAWN_PENALTY = np.array([-5, -19], dtype=np.int32) # MG, EG

# --- Doubled Pawns / 重疊兵 ---
# Penalty for each doubled pawn on a file.
# 每一個重疊兵的懲罰。
DOUBLED_PAWN_PENALTY = np.array([-16, -28], dtype=np.int32) # MG, EG

# --- Backward Pawns / 後兵 ---
# Penalty for a backward pawn.
# 後兵的懲罰。
# Negative values, applied with += (consistent with ISOLATED_PAWN_PENALTY and DOUBLED_PAWN_PENALTY)
BACKWARD_PAWN_PENALTY = np.array([-1, -5], dtype=np.int32) # MG, EG

# --- Connected Pawn Bonus by Rank (SF11) / SF11 連結兵獎勵（按橫排） ---
# Applied to ALL pawns that are in phalanx (side-by-side) or supported (diagonally behind).
# 適用於所有處於並列或受支撐狀態的兵。
CONNECTED_BONUS = np.array([0, 0, 1, 3, 5, 7, 35, 0], dtype=np.int32)
CONNECTED_SUPPORT_WEIGHT = np.int32(10)  # SF11 value was 21

# =============================================================================
# --- Outpost Constants / 前哨常量 ---
# =============================================================================
# Bonus for knights and bishops on outpost squares (supported by pawn, not attacked by pawn).
# 前哨方格上的騎士和主教的獎勵（由兵支持，且未被兵攻擊）。

# Outpost Bonuses by Rank (0-7).
# Indices: Rank 0 to Rank 7.
# Values: [MG, EG]
# 騎士前哨獎勵
OUTPOST_BONUS_KNIGHT = np.array([
    [0, 0], # Rank 1
    [0, 0], # Rank 2
    [0, 0], # Rank 3
    [20, 10], # Rank 4
    [44, 34], # Rank 5
    [53, 42], # Rank 6
    [54, 44], # Rank 7
    [0, 0], # Rank 8
], dtype=np.int32)

# 主教前哨獎勵
OUTPOST_BONUS_BISHOP = np.array([
    [0, 0], # Rank 1
    [0, 0], # Rank 2
    [0, 1], # Rank 3
    [15, 6], # Rank 4
    [30, 19], # Rank 5
    [37, 26], # Rank 6
    [38, 34], # Rank 7
    [0, 0], # Rank 8
], dtype=np.int32)

# Bonus if the outpost is a "Hole" (cannot be attacked by enemy pawns at all).
# 如果前哨是“洞”（完全無法被敵方兵攻擊），則給予額外獎勵。
OUTPOST_HOLE_BONUS = np.array([10, 0], dtype=np.int32) # MG, EG

# =============================================================================
# --- King Safety Constants (NEW - based on Chessprogramming Wiki) / 王的安全常量 ---
# =============================================================================

# --- Phase 2: Attacking the King Zone (Non-Linear Model) / 攻擊王翼區域（非線性模型） ---
# Attack units for each piece type. Order: P, N, B, R, Q
# 每個棋子類型的攻擊單位。順序：兵、馬、象、車、后
KING_SAFETY_ATTACK_UNITS = np.array([1, 4, 4, 4, 5], dtype=np.int32) # P, N, B, R, Q

# --- kingDanger Linear Formula Weights (Inspired by Stockfish 11) ---
# These contribute to a kingDanger score that is then squared.
# 這些值貢獻到 kingDanger 分數，最後進行二次轉換。

# Weight per weak square in king zone (attacked by enemy, not defended by us except K)
# 王圈內弱格（被敵攻、只被王守或不守）每個的 danger 貢獻
KING_DANGER_WEAK_SQ = np.int32(3)

# Weight per unsafe check square (enemy can check but not safely)
# 不安全將軍格每個的 danger 貢獻
KING_DANGER_UNSAFE_CHECK = np.int32(2)

# Weight per enemy attack on squares adjacent to king (KING_ATTACKS[ksq])
# 敵方攻擊到王鄰格每次的 danger 貢獻
KING_DANGER_ATTACK_ON_KING_SQ = np.int32(1)

# Flat deduction from kingDanger when enemy has no queen
# 敵方無后時的固定 danger 扣除
KING_DANGER_NO_QUEEN = np.int32(5)

# Penalty per pinned piece on the defending side (SF11-inspired)
# 防守方被牽制棋子的 danger 懲罰（直接計算成數分數，會乘以材質 scaling factor）
KING_DANGER_PINNED = np.int32(10)

# Divisor for kingDanger² conversion (controls overall penalty magnitude)
# kingDanger² 的除數（i²/2 與舊 KING_SAFETY_TABLE 等價）
KING_DANGER_DIVISOR = np.int32(2)

# --- Safe Check Penalties (in attack units, added to total_attack_units) ---
# Represent danger of enemy pieces being able to safely give check.
# Scaled for our system: SF11 values (780-1080) mapped to our table-index units (~5-7).
SAFE_CHECK_KNIGHT = np.int32(6)
SAFE_CHECK_BISHOP = np.int32(5)
SAFE_CHECK_ROOK = np.int32(7)
SAFE_CHECK_QUEEN = np.int32(6)

# 攻擊王翼的棋子數量的最大值
MAX_KING_ATTACKERS = np.int32(1000)

# --- Phase 3: King Tropism / 王的向性 ---
KING_TROPISM_MAX_DISTANCE = 14 # Max MANHATTAN distance / 最大曼哈頓距離
KING_TROPISM_WEIGHTS = np.array([0, 1, 1, 2, 3], dtype=np.int32) # P, N, B, R, Q

# --- Phase 4: Advanced & Dynamic / 進階與動態 ---
# REMOVED: Flat penalty
# PAWN_STORM_PENALTY = -5 

# NEW: Rank-based pawn storm penalty
# Penalty for enemy pawns on files adjacent to the king, based on their rank relative to the defender's back rank.
# Index 0: 8th rank (impossible/promotion), Index 1: 7th rank... Index 2: 2nd rank (close).
# We interpret index as "distance from defender's back rank" or simply "relative rank".
# For White King (Rank 0), enemy Black pawn at Rank 2 is index 2.
# For Black King (Rank 7), enemy White pawn at Rank 5 is index 2 (7-5=2).
# Values: [Dummy, Dummy, Rank2, Rank3, Rank4, Rank5, Rank6, Rank7]
PAWN_STORM_PENALTY_BY_RANK = np.array([0, 121, 73, 25, 13, 4, 3, 0], dtype=np.int32)

SCALING_WEIGHTS = np.array([0, 4, 4, 6, 10], dtype=np.int32) # N, B, R, Q - for scaling factor / 用於縮放因子的權重
MAX_SCALING_MATERIAL = (2*4 + 2*4 + 2*6 + 1*10) # Sum of all weights for one side / 一方所有權重的總和

PAWN_SHIELD_MISSING_PENALTY = 15
PAWN_SHIELD_INTACT_BONUS = 10
PAWN_SHIELD_ADVANCED_BONUS = 5
PAWN_SHIELD_PUSHED_PENALTY = 10
KING_OPEN_FILE_PENALTY = 10
KING_SEMI_OPEN_FILE_PENALTY = 5
KING_SAFETY_WEAK_SQUARE_PENALTY = 20

EG_SAFETY_SCALE = 0.5 # Scale down endgame king safety impact / 縮減殘局王的安全影響

# =============================================================================
# --- Threat Evaluation Constants / 威脅評估常量 ---
# =============================================================================
# Derived from Stockfish 11 but simplified and scaled.

# =============================================================================
# --- Threat Evaluation Constants (SF11-inspired, scaled by Pawn ratio) ---
# =============================================================================
# All values scaled from SF11 by Pawn ratio: MG×0.78, EG×0.56

# ThreatBySafePawn: Friendly safe pawn attacks enemy non-pawn piece.
# 安全兵的威脅：己方安全兵攻擊敵方非兵棋子。
THREAT_SAFE_PAWN = np.array([57, 33], dtype=np.int32)  # SF11: (173, 94)

# ThreatByMinor[target_piece_type]: Minor (N/B) attacks piece of given type.
# Index: 0=Pawn, 1=Knight, 2=Bishop, 3=Rook, 4=Queen, 5=King
# 輕子威脅：馬/象攻擊對應類型棋子的獎勵。
THREAT_BY_MINOR = np.array([
    [0, 19], # vs Pawn
    [22, 16], # vs Knight
    [33, 19], # vs Bishop
    [60, 27], # vs Rook
    [66, 64], # vs Queen
    [0, 0], # vs King
], dtype=np.int32)

# ThreatByRook[target_piece_type]: Rook attacks piece of given type (only if weak).
# Index: 0=Pawn, 1=Knight, 2=Bishop, 3=Rook, 4=Queen, 5=King
# 車威脅：車攻擊對應類型棋子的獎勵。
THREAT_BY_ROOK = np.array([
    [2, 30], # vs Pawn
    [18, 30], # vs Knight
    [18, 30], # vs Bishop
    [3, 24], # vs Rook
    [40, 20], # vs Queen
    [0, 0], # vs King
], dtype=np.int32)

# ThreatByKing: King attacks a weakly defended enemy piece.
# 王攻擊弱子：王攻擊敵方弱子的獎勵。
THREAT_BY_KING = np.array([34, 60], dtype=np.int32)  # SF11: (24, 89)

# Hanging: Enemy piece is attacked + not strongly protected.
# 懸掛子：敵方棋子被攻擊且未被強力保護。
THREAT_HANGING = np.array([17, 5], dtype=np.int32)  # SF11: (69, 36)

# RestrictedPiece: Enemy piece moves are restricted by our attacks.
# 限制棋子行動力：敵方棋子的走子受到我方攻擊限制。
THREAT_RESTRICTED_PIECE = np.array([5, 4], dtype=np.int32)  # SF11: (7, 7)

# ThreatByPawnPush: Pawn push threatens enemy pieces on next move.
# 兵推威脅：兵推進後能威脅敵子。
THREAT_PAWN_PUSH = np.array([19, 4], dtype=np.int32) # SF11: (48, 39)

# =============================================================================
# --- Per-Piece Bonus/Penalty Constants (SF11-inspired, scaled by Pawn ratio) ---
# =============================================================================

# KingProtector: Penalty for minor piece being far from own king.
# 王保護者：輕子距離己方王越遠，懲罰越重（每格切比雪夫距離）。
# SF11: (7, 8). Scaled to 0.75× of our previous value (which itself was already below SF11)
# to compensate for mobility being scaled down 0.75×.
KING_PROTECTOR = np.array([4, 3], dtype=np.int32)  # mg, eg — per distance unit

# MinorBehindPawn: Bonus for minor piece sheltered behind a pawn.
# 輕子藏兵後：輕子站在兵後方的獎勵。
# BishopPawns: Penalty per own pawn on same color as bishop.
# 壞象懲罰：象同色上的己方兵數量懲罰（含封閉中心加重）。
BISHOP_PAWNS_PENALTY = np.array([0, 6], dtype=np.int32)  # SF11: (3, 7)

# TrappedRook: Penalty for rook with mobility <= 3 trapped by own king.
# 困車懲罰：車移動格數 ≤ 3 且在己方王同側。
TRAPPED_ROOK = np.array([21, 0], dtype=np.int32)  # SF11: (52, 10)

# --- Endgame Scale Factors ---
SCALE_FACTOR_NORMAL = 64
SCALE_FACTOR_DRAW = 0
SCALE_FACTOR_OCB_ONE_PAWN = 16
SCALE_FACTOR_OCB_TWO_PAWNS = 32
SCALE_FACTOR_OCB_MULTIPLE_PAWNS = 48

# =============================================================================
# --- Search Constants / 搜尋常量 ---
# =============================================================================

INFINITY = 32000
MATE_SCORE = 30000
MAX_PLY = 128
MATE_IN_MAX_PLY = MATE_SCORE - MAX_PLY
NO_MOVE = np.uint16(0)

# Move Ordering Bonuses
SCORE_TT_MOVE = 100000
SCORE_GOOD_CAPTURE_BONUS = 20000
SCORE_KILLER_1 = 15000
SCORE_KILLER_2 = 14000
SCORE_COUNTER_MOVE = 10000
SCORE_BAD_CAPTURE_PENALTY = -5000

# Contempt Factor: Score penalty for draws when the engine is winning.
# This encourages the engine to prefer winning lines over draws.
CONTEMPT = 20 # centipawns

PAWN_PUSH_RANK_BONUS = 2000
PAWN_PUSH_ATTACK_BONUS = 3000
KING_TROPISM_BONUS = 2500 # Bonus for improving king tropism
KING_ATTACK_BONUS = 2000 # Bonus for quiet moves attacking the opponent's King zone
ROOK_QUEEN_BATTERY_BONUS = 5000 # Bonus for Rook moving to same file/rank as Queen

# History Heuristics Constants
MAX_HISTORY = 16384 # Max value for history table to prevent overflow and saturation
HISTORY_MAX_MAIN = 7183
HISTORY_MAX_BUTTERFLY = 7183
HISTORY_MAX_CAPTURE = 10692
HISTORY_MAX_CONTINUATION = 30000

# 50-move rule constants
FIFTY_MOVE_RULE_LIMIT = 100
FIFTY_MOVE_SCALE_THRESHOLD = 80
FIFTY_MOVE_MAX_SCALE = 256

# History tuning in Search
SEE_HISTORY_DIVISOR = 512
LMR_CONT_HISTORY_MULT = 1.0

CORRECTION_HISTORY_SIZE = 16384
CORRECTION_HISTORY_MASK = 16383
CORRECTION_HISTORY_LIMIT = 1024
CORRECTION_HISTORY_DIVISOR = 131072
CORRECTION_HISTORY_PAWN_WEIGHT = 11433
CORRECTION_HISTORY_MINOR_WEIGHT = 8823
CORRECTION_HISTORY_NON_PAWN_WEIGHT = 12749
CORRECTION_HISTORY_UPDATE_DEPTH = 4
SCORE_MIN = -30000
SCORE_MAX = 30000
CONTINUATION_HISTORY_FACTOR = 4

# Special value to indicate that the search was stopped due to timeout
# 特殊值，表示搜尋因超時而停止
STOP_SEARCH_FLAG = 66666

PAWN_KEY_INDEX = 5
MINOR_KEY_INDEX = 6
NON_PAWN_KEY_WHITE_INDEX = 7
NON_PAWN_KEY_BLACK_INDEX = 8

# --- Aspiration Windows / 期望窗口 ---
ASPIRATION_WINDOW_SIZE = 25 # centipawns (H8: tightened from 100 to detect score instability)

# --- Pruning Techniques / 剪枝技術 ---
# Master switches for new pruning techniques / 新剪枝技術的總開關
ENABLE_LMP = True           # Late Move Pruning
ENABLE_PROBCUT = True       # ProbCut
ENABLE_DELTA_PRUNING = True # Delta Pruning in Quiescence Search
ENABLE_IIR = True           # Internal Iterative Reduction (Replaces old IID)
ENABLE_SINGULAR_EXTENSIONS = True # Singular Extensions
ENABLE_MATE_DISTANCE_PRUNING = True # Mate Distance Pruning
ENABLE_MULTICUT = False  # Multi-Cut Pruning (disabled — causes regression, needs redesign)

# Multi-Cut Parameters
MULTICUT_MIN_DEPTH = 5  # Only apply Multi-Cut at this depth or higher
MULTICUT_M = 3          # Number of fail-highs required to trigger prune
MULTICUT_C = 6          # Number of moves to probe


# IID and Singular Extension Parameters
MIN_SINGULAR_DEPTH = 6
SINGULAR_EXTENSION_MARGIN = 150 # centipawns

# NEW: Pruning Switches (based on Stockfish Analysis)
ENABLE_SHALLOW_SEE_PRUNING = True  # Enable SEE pruning for captures/quiets at shallow depth
ENABLE_HISTORY_PRUNING = True      # Enable pruning based on History Score

# NEW: Pruning Parameters (Tightened for Performance/Strength Balance)
PRUNING_SHALLOW_DEPTH = 8         # Prune moves only if depth is below this
PRUNING_CAPTURE_SEE_MARGIN = -150 # Tightened from -200 (More pruning)
PRUNING_QUIET_SEE_MARGIN = -80    # Tightened from -100 (More pruning)
PRUNING_HISTORY_THRESHOLD = -1000 # Tightened from -1500 (More pruning, threshold is higher/closer to 0)

# Master switches for existing pruning techniques / 現有剪枝技術的總開關
ENABLE_NMP = True           # Null Move Pruning
ENABLE_RAZORING = True      # Razoring
ENABLE_FP = True            # Futility Pruning
ENABLE_RFP = True           # Reverse Futility Pruning
ENABLE_LMR = True           # Late Move Reductions

NULL_MOVE_REDUCTION = 2
MAX_QUIESCENCE_DEPTH = 5

# Razoring
RAZORING_MARGIN = 600 # Tightened from 700

# Futility Pruning
FP_BASE = 150      # Reduced base
FP_MULTIPLIER = 180 # Reduced multiplier

# Reverse Futility Pruning
RFP_MARGIN_D1 = 200 # Tightened from 250

NMP_STATIC_MARGIN = 150

# Late Move Reductions (LMR)
LMR_MIN_DEPTH = 3           # Minimum depth to apply LMR (H4: lowered from 4 to match Stockfish)
LMR_MIN_QUIET_MOVE_INDEX = 3 # Minimum number of quiet moves before LMR (H4: lowered from 4)
LMR_REDUCTION = 1           # Depth reduction for LMR / LMR 的深度減少值

# Late Move Pruning (LMP) - Prune moves after a certain number of quiet moves have been searched
# 晚期移動剪枝（LMP） - 在搜尋了一定數量的寧靜步後剪枝
LMP_MOVE_COUNT = np.array([
    0 if d == 0 else 3 + 2 * d * d for d in range(8)
], dtype=np.int32)

# LMR Table (Precomputed)
# A1: Formula tuned — divisor 2.25→2.0, offset 0.5→0.77 (matching modern Stockfish)
# Using 256 as max move count (enough for almost all positions)
LMR_TABLE = np.zeros((MAX_PLY, 256), dtype=np.int32)
for d in range(MAX_PLY):
    for mc in range(256):
        if d < 2 or mc < 2:
            LMR_TABLE[d, mc] = 0
        else:
            LMR_TABLE[d, mc] = int(0.77 + math.log(d) * math.log(mc) / 2.0)


# ProbCut
PROBCUT_R = 4
PROBCUT_MARGIN = 150 # centipawns

# Delta Pruning
DELTA_PRUNING_MARGIN = 400

# --- Static Exchange Evaluation (SEE) Threshold / SEE 閾值 ---
SEE_THRESHOLD = 0  # centipawns (H6: tightened from -200, only search non-losing captures in QSearch)
ENABLE_SEE_IN_QUIESCENCE = True # Master switch to enable/disable SEE in quiescence search / 啟用/禁用靜態搜尋中 SEE 的總開關

# =============================================================================
# --- Bitboard Utilities Constants / 位元棋盤工具常量 ---
# =============================================================================

BB_SQUARES = np.array([np.uint64(1) << i for i in range(64)], dtype=np.uint64)
"""
位元棋盤陣列，其中每個位元棋盤在對應的方格索引處設置了一個位元。
"""

# Rank Masks / 橫排掩碼
RANK_MASKS = np.array([np.uint64(0xFF) << np.uint64(i * 8) for i in range(8)], dtype=np.uint64)

# De Bruijn sequence for fast bit scanning / 用於快速位掃描的 De Bruijn 序列
DE_BRUIJN_SEQUENCE = np.uint64(0x03f79d71b4cb0a89)

# Lookup table for mapping the De Bruijn hash to a square index (0-63)
# 將 De Bruijn 哈希映射到方格索引（0-63）的查找表
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
# --- Transposition Table Constants / 置換表常量 ---
# =============================================================================

TT_SIZE_MB = 256

# File constants
NOT_A_FILE = ~np.uint64(0x0101010101010101)
NOT_H_FILE = ~np.uint64(0x8080808080808080)

# === 新增的可調魔法數字 (New Tunable Magic Numbers) ===
PROXIMITY_ENEMY_WEIGHT = np.int32(5)
PROXIMITY_FRIENDLY_WEIGHT = np.int32(2)
MAX_PROXIMITY_BONUS = np.int32(150)
KING_DANGER_SINGLE_ATTACKER_DIVISOR = np.int32(2)
UNSTOPPABLE_PAWN_BONUS = np.int32(800)
EG_KING_PAWN_PROXIMITY_WEIGHT = np.int32(5)
BLOCKED_PASSER_DIVISOR = np.int32(2)



MG_MATERIAL_VALUES = np.array([100, 329, 344, 481, 898, 0], dtype=np.int32)

EG_MATERIAL_VALUES = np.array([120, 312, 339, 484, 942, 0], dtype=np.int32)

PST_MG = np.array([
    [17, 2, -20, -13, -10, -13, 12, 2, -4, -2, -4, -12, -4, 5, -7, -10, -12, -4, -12, 2, 1, 2, -2, 0, 0, -9, 2, -13, 11, -3, -19, -8, 8, -2, -12, 17, 5, 17, -14, 3, -9, -8, 10, 25, 28, 12, 3, 0, 46, 61, 28, 35, 48, 51, 51, 47, 7, -9, 0, -4, 0, 2, -1, 20],
    [-44, -32, -36, -27, -31, -13, -36, -45, -55, -26, 2, 1, 5, -1, -17, -12, -18, 6, -11, 16, 5, -9, 8, -13, -11, 6, 4, 2, 11, 17, -20, -24, -39, 6, 10, 20, -7, 5, 0, -33, -25, -3, -3, 22, 4, 6, 2, -48, -64, 4, 11, -7, -4, -10, -41, -49, -57, -52, -34, -11, -22, -42, -44, -72],
    [-30, -1, 6, -1, -8, 16, -1, -19, -15, 3, 0, -10, 6, 5, -3, -5, -7, 11, 12, 1, 0, -1, 3, -6, -5, 9, -10, 4, -9, -11, 15, -16, -12, -7, -5, 12, 26, 8, -11, -27, -21, -6, -9, -12, 17, 6, 1, -8, 2, -11, 15, 4, 7, 12, 3, -3, -31, 13, -9, 0, -23, 6, -31, -7],
    [-10, 9, 0, 9, -3, -7, 11, -2, -4, 3, -15, -8, -9, -10, -13, -12, 1, -1, -4, 9, -15, 0, 3, -25, -6, -23, 2, 8, 10, 25, 2, -23, -7, 7, 4, -9, 8, -12, -10, -2, -22, 5, 12, 0, 3, -11, 27, 5, 14, 7, 19, 1, 11, 20, 19, -10, 5, 2, 14, 11, -7, 18, -3, -7],
    [-10, -3, -11, 1, -2, -14, -22, -15, -4, -8, 6, -3, 6, -3, -22, 4, -22, 4, 12, -6, -20, 0, -13, -16, -21, 16, 8, -5, -10, -1, 14, 5, 23, 19, 3, 18, 0, 16, -8, -7, -2, -6, -13, -7, 5, 8, 27, -19, -22, -21, 8, 5, 7, 1, -3, -4, -32, 5, 12, 3, 13, -12, -15, -11],
    [30, 25, 7, 10, 4, 15, 30, 21, 19, 29, 8, 9, 6, -8, 16, 30, -15, -29, 2, -14, -30, -36, -42, -10, -16, -39, -56, -37, -33, -22, -13, -41, -25, -42, -54, -65, -60, -47, -40, -33, -44, -24, -34, -39, -44, -36, -40, -29, -40, -51, -43, -58, -32, -38, -37, -18, -40, -56, -26, -48, -47, -59, -49, -47],
], dtype=np.int32)

PST_EG = np.array([
    [8, 3, -7, -9, -9, -10, -1, -16, 5, 26, 20, -12, 11, 12, 12, 0, 20, 13, 13, 21, 1, -4, 10, 0, 22, 9, 9, -7, -1, -1, 0, 7, 39, 9, 15, 5, 18, -8, 27, 19, 37, 40, 31, 38, 32, 33, 31, 46, 90, 70, 70, 68, 97, 89, 86, 91, -9, -2, 3, 5, 4, 4, -26, 18],
    [-41, -45, -21, -36, -19, -30, -50, -55, -57, -27, -3, -12, -2, 12, -14, -46, -21, -21, -4, 4, 16, -7, -22, -13, -20, 6, 13, 27, 20, 37, -1, -31, -39, -5, 35, 10, 7, 7, -11, -49, -25, 19, 33, 21, 22, -10, 3, -36, -20, -16, 10, 13, 2, 0, -35, -31, -66, -42, -22, -22, -21, -40, -11, -31],
    [-36, -39, -19, -7, -22, -16, -40, -32, -27, -12, 3, 5, 15, -26, -25, -31, -50, 8, 7, 15, 1, 10, 0, -5, -4, -27, 4, 0, 2, -17, 1, -5, -28, -1, 0, 18, -1, 23, -13, -28, -6, -4, 21, 7, 16, 1, 15, -5, -36, -2, -28, -20, -19, 16, 15, -30, -42, -31, -24, -18, 3, -27, -30, -33],
    [1, -10, -17, -6, -7, -11, -15, 7, -23, -3, -3, -18, -16, 14, -6, -15, -8, 12, -18, 8, -12, -14, -9, 5, -22, 21, 15, 1, 0, 8, 14, -2, -3, -7, -11, -14, 14, -26, -10, -3, -3, -13, -15, -2, 0, 2, -26, 3, 19, -1, 16, -18, -4, 5, -3, -11, 6, -6, 40, 5, 20, 15, 1, 17],
    [-71, -57, -52, -14, -22, -52, -52, -89, -28, -39, -46, -9, -31, -47, -29, -61, -30, -19, -14, -15, -8, 3, -4, -37, -20, -21, 3, 1, 15, 20, -3, -29, -15, -16, 33, -2, 4, 8, -8, -30, -44, -15, -31, -4, 13, 5, -22, -49, -71, -43, -27, -1, -14, -25, -28, -41, -65, -49, -47, -39, -44, -43, -65, -69],
    [-39, -23, -31, -17, -6, -26, -43, -33, -43, -26, -2, 11, -2, 3, -17, -14, -24, -2, 8, 13, 29, 28, 2, -17, -41, 10, 23, 30, 11, 14, -8, -43, -41, -3, 14, 49, 47, 19, -8, -33, -5, 4, 29, 19, 36, 6, -7, -41, -3, -27, -5, 7, 19, -19, -12, -40, -40, -36, -29, -32, -35, -28, -38, -37],
], dtype=np.int32)

KNIGHT_MOBILITY_BONUS = np.array([
    [-42, -41],
    [18, -11],
    [34, -6],
    [42, 4],
    [44, 10],
    [48, 20],
    [50, 23],
    [54, 27],
    [58, 30],
], dtype=np.int32)

BISHOP_MOBILITY_BONUS = np.array([
    [-25, -24],
    [3, -14],
    [20, 6],
    [28, 6],
    [33, 19],
    [37, 30],
    [39, 41],
    [41, 50],
    [46, 53],
    [49, 58],
    [52, 61],
    [56, 62],
    [62, 66],
    [66, 68],
], dtype=np.int32)

ROOK_MOBILITY_BONUS = np.array([
    [-3, -20],
    [17, -8],
    [31, 30],
    [31, 32],
    [31, 38],
    [36, 45],
    [36, 51],
    [42, 52],
    [48, 55],
    [53, 55],
    [55, 62],
    [56, 66],
    [59, 75],
    [62, 85],
    [63, 87],
], dtype=np.int32)

QUEEN_MOBILITY_BONUS = np.array([
    [-20, 8],
    [-2, 9],
    [16, 11],
    [35, 19],
    [36, 19],
    [47, 29],
    [49, 37],
    [55, 38],
    [57, 49],
    [62, 50],
    [63, 57],
    [66, 60],
    [67, 64],
    [68, 69],
    [68, 76],
    [72, 79],
    [73, 81],
    [74, 87],
    [76, 95],
    [78, 96],
    [79, 97],
    [81, 99],
    [84, 100],
    [90, 100],
    [91, 102],
    [96, 102],
    [101, 103],
    [101, 109],
], dtype=np.int32)

BISHOP_PAIR_BONUS = np.array([30, 16], dtype=np.int32)

ROOK_ON_SEMI_OPEN_FILE_BONUS = np.array([13, 10], dtype=np.int32)

ROOK_ON_OPEN_FILE_BONUS = np.array([10, 0], dtype=np.int32)

ROOK_ON_SEVENTH_BONUS = np.array([0, 23], dtype=np.int32)

PASSED_PAWN_BONUS = np.array([
    [0, 0],
    [0, 0],
    [0, 0],
    [1, 23],
    [23, 46],
    [61, 99],
    [153, 250],
    [0, 0],
], dtype=np.int32)

CANDIDATE_PASSED_PAWN_BONUS = np.array([
    [0, 0],
    [0, 0],
    [0, 0],
    [3, 2],
    [16, 9],
    [28, 65],
    [58, 137],
    [0, 0],
], dtype=np.int32)

ISOLATED_PAWN_PENALTY = np.array([-9, -10], dtype=np.int32)

DOUBLED_PAWN_PENALTY = np.array([-15, -34], dtype=np.int32)

BACKWARD_PAWN_PENALTY = np.array([-1, -11], dtype=np.int32)

CONNECTED_BONUS = np.array([0, 0, 0, 2, 3, 19, 30, 0], dtype=np.int32)

CONNECTED_SUPPORT_WEIGHT = 0

OUTPOST_BONUS_KNIGHT = np.array([
    [0, 0],
    [0, 0],
    [0, 12],
    [10, 20],
    [41, 48],
    [66, 48],
    [76, 63],
    [0, 0],
], dtype=np.int32)

OUTPOST_BONUS_BISHOP = np.array([
    [0, 0],
    [0, 0],
    [2, 10],
    [20, 21],
    [32, 23],
    [48, 24],
    [52, 29],
    [0, 0],
], dtype=np.int32)

OUTPOST_HOLE_BONUS = np.array([2, 2], dtype=np.int32)

KING_SAFETY_ATTACK_UNITS = np.array([4, 8, 8, 10, 12], dtype=np.int32)

KING_DANGER_WEAK_SQ = 4

KING_DANGER_UNSAFE_CHECK = 5

KING_DANGER_ATTACK_ON_KING_SQ = 4

KING_DANGER_NO_QUEEN = 13

KING_DANGER_PINNED = 13

KING_DANGER_DIVISOR = 48

SAFE_CHECK_KNIGHT = 24

SAFE_CHECK_BISHOP = 23

SAFE_CHECK_ROOK = 22

SAFE_CHECK_QUEEN = 17

KING_TROPISM_WEIGHTS = np.array([0, 0, 0, 0, 0], dtype=np.int32)

PAWN_STORM_PENALTY_BY_RANK = np.array([7, 104, 70, 24, 11, 3, 3, 0], dtype=np.int32)

SCALING_WEIGHTS = np.array([0, 4, 4, 6, 10], dtype=np.int32)

PAWN_SHIELD_MISSING_PENALTY = 0

PAWN_SHIELD_INTACT_BONUS = 14

PAWN_SHIELD_ADVANCED_BONUS = 3

PAWN_SHIELD_PUSHED_PENALTY = 1

KING_OPEN_FILE_PENALTY = 22

KING_SEMI_OPEN_FILE_PENALTY = 0

THREAT_SAFE_PAWN = np.array([62, 35], dtype=np.int32)

THREAT_BY_MINOR = np.array([
    [1, 17],
    [19, 46],
    [32, 20],
    [48, 35],
    [69, 55],
    [0, 0],
], dtype=np.int32)

THREAT_BY_ROOK = np.array([
    [4, 16],
    [14, 26],
    [27, 43],
    [16, 37],
    [42, 25],
    [0, 0],
], dtype=np.int32)

THREAT_BY_KING = np.array([35, 47], dtype=np.int32)

THREAT_HANGING = np.array([7, 8], dtype=np.int32)

THREAT_PAWN_PUSH = np.array([9, 10], dtype=np.int32)

KING_PROTECTOR = np.array([0, 0], dtype=np.int32)

BISHOP_PAWNS_PENALTY = np.array([0, 7], dtype=np.int32)

TRAPPED_ROOK = np.array([11, 9], dtype=np.int32)

INITIATIVE_BONUS = 24

MAX_KING_ATTACKERS = 1016

PROXIMITY_ENEMY_WEIGHT = 3

PROXIMITY_FRIENDLY_WEIGHT = 6

MAX_PROXIMITY_BONUS = 147

KING_DANGER_SINGLE_ATTACKER_DIVISOR = 1

UNSTOPPABLE_PAWN_BONUS = 781

EG_KING_PAWN_PROXIMITY_WEIGHT = 0

BLOCKED_PASSER_DIVISOR = 5