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
MG_MATERIAL_VALUES = np.array([100, 338, 338, 482, 898, 0], dtype=np.int32)
# 殘局（End Game）材質值
EG_MATERIAL_VALUES = np.array([120, 317, 348, 510, 947, 0], dtype=np.int32)

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
    [2, 3, 0, 2, 2, -4, 5, 0, 0, 0, 6, -15, -9, 8, 3, 2, -4, -5, 0, 4, 5, -3, 6, 0, 1, -8, -2, -8, 11, -3, -11, -6, 4, -2, 2, -2, 8, 4, 1, 5, 5, 3, 17, 25, 30, 19, 6, 10, 51, 55, 47, 48, 49, 51, 53, 47, -1, -4, 6, 0, 5, 4, 1, 5],
    [-55, -29, -26, -26, -21, -22, -34, -53, -40, -18, 1, 9, 5, -1, -16, -37, -22, 6, 0, 15, 15, 3, 5, -24, -20, -2, 14, 21, 9, 15, -7, -23, -25, 2, 12, 17, -3, 1, 1, -25, -35, 0, 14, 14, 13, 12, -5, -32, -41, -16, 3, 1, -1, -1, -17, -42, -56, -42, -28, -28, -31, -29, -39, -53],
    [-25, -9, 3, -3, -7, 2, -12, -21, -10, 14, -3, -2, 11, -2, 9, -13, -8, 10, 9, -5, -4, 13, 10, -8, -12, 4, 4, 8, 6, 0, 0, -5, -13, 10, 6, 8, 7, -3, 4, -13, -9, 4, 1, 8, 7, 3, -4, -10, -5, 3, 0, -2, 3, -1, 0, -11, -21, -6, -13, -11, -10, -6, -9, -20],
    [2, 2, 1, 0, 2, -2, 8, 9, 0, -3, 0, 1, -7, 1, 1, -5, -3, -5, -5, 0, -1, -3, -1, -4, -8, 3, 2, 4, 1, -2, -1, -5, -12, 4, 3, -6, 3, -1, 2, -5, -6, -3, 2, -3, 1, -2, 4, -9, 1, 8, 10, 8, 7, 15, 11, 0, 0, -1, -2, 1, 1, -5, -1, -3],
    [-15, -3, -7, 13, 0, -8, -6, -22, -10, -3, 4, -1, 7, 3, 1, -6, -6, 11, 1, -2, 2, -2, 1, -8, 1, -1, 4, 1, 1, 2, 1, -1, -10, -1, 7, 5, 2, 1, -1, -1, -10, 4, 4, 0, 2, 8, 0, -11, -9, -5, 0, -1, 0, -1, 1, -14, -19, -10, -5, -3, -3, -7, -5, -21],
    [21, 24, 13, 5, 7, 14, 18, 21, 19, 24, 4, 1, 3, 5, 26, 21, -10, -17, -17, -22, -19, -22, -24, -6, -18, -30, -33, -37, -34, -32, -29, -22, -30, -37, -40, -47, -50, -39, -36, -34, -33, -41, -42, -47, -47, -40, -38, -31, -33, -34, -39, -52, -48, -41, -41, -28, -32, -44, -38, -52, -50, -43, -39, -30],
], dtype=np.int32)

PST_EG = np.array([
    [6, 3, 3, -5, -3, -2, -1, 0, 5, 12, 13, 9, 8, 10, 13, 9, 11, 12, 12, 19, 11, 7, 15, 13, 16, 13, 13, 9, 4, 1, 2, 17, 22, 20, 18, 12, 17, 16, 20, 19, 43, 44, 41, 46, 42, 48, 45, 46, 77, 78, 76, 80, 76, 83, 74, 76, -1, -2, -1, 3, -1, -3, 1, 5],
    [-50, -38, -30, -28, -26, -26, -37, -45, -41, -23, 5, -1, 4, 1, -19, -36, -28, -1, 6, 14, 15, 9, -3, -25, -30, 9, 15, 21, 20, 13, 8, -29, -25, 1, 17, 21, 16, 14, 1, -29, -31, 9, 14, 15, 17, 12, 8, -30, -34, -26, 2, 10, 4, 4, -18, -38, -53, -43, -31, -29, -36, -29, -37, -46],
    [-39, -32, -22, -16, -17, -20, -35, -32, -23, -13, -4, -3, -1, -5, -11, -28, -24, 4, 4, -1, -1, 5, 6, -24, -14, 1, 13, 9, 14, 8, -1, -16, -15, 3, 11, 9, 10, 13, -5, -20, -12, -2, 1, 9, 2, 4, 0, -12, -27, -8, -13, 4, -8, -14, -12, -33, -49, -21, -28, -9, -6, -31, -23, -45],
    [-6, -11, -12, -8, -14, -7, -15, -9, -9, -6, 0, -1, 0, -1, -9, -16, 4, -4, -5, 9, 2, -10, -1, 8, 1, 7, 10, -7, -12, 8, 9, -3, -4, -1, -11, 2, 5, -10, -1, -8, 2, -6, -1, 6, 2, -1, -10, 3, -2, -1, 14, -5, -6, 19, 2, 1, 18, 1, 22, 9, 16, 22, -2, 18],
    [-71, -54, -46, -32, -34, -47, -58, -70, -50, -31, -22, -6, -9, -30, -29, -56, -41, -15, -11, 0, -1, -8, -17, -41, -26, -10, 9, 20, 15, 10, -6, -29, -26, -11, 13, 23, 25, 13, -1, -26, -52, -15, -14, 2, 1, -8, -18, -50, -55, -35, -17, -7, -2, -20, -32, -56, -72, -55, -43, -40, -33, -45, -58, -78],
    [-50, -31, -35, -24, -25, -18, -37, -42, -29, -29, 1, 10, 2, 2, -31, -24, -30, -10, 17, 30, 25, 19, -9, -22, -31, -8, 31, 39, 35, 24, -13, -35, -30, -16, 29, 37, 38, 25, -7, -27, -26, -10, 21, 25, 22, 17, -12, -29, -31, -23, -11, 3, 1, -6, -17, -24, -49, -37, -32, -20, -17, -32, -39, -48],
], dtype=np.int32)
# =============================================================================
# --- Other Evaluation Constants / 其他評估常量 ---
# =============================================================================

# --- Initiative Bonus / 主動權獎勵 ---
# A small bonus awarded to the side to move, acknowledging the advantage of having the turn.
# This bonus is tapered and disappears in the endgame.
# 給予輪到行棋一方的小獎勵，承認擁有下棋權的優勢。此獎勵是漸進的，在殘局中會消失。
INITIATIVE_BONUS = 39 # centipawns
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
    [-29, -30],
    [-5, -16],
    [18, -7],
    [18, 3],
    [18, 4],
    [18, 4],
    [18, 7],
    [18, 9],
    [19, 11],
], dtype=np.int32)  # index 0..8

# --- Bishop Mobility (SF11 MobilityBonus × Pawn ratio MG×0.585 EG×0.42) --- max 13 squares
BISHOP_MOBILITY_BONUS = np.array([
    [-11, -24],
    [10, -1],
    [22, 3],
    [22, 9],
    [22, 10],
    [22, 19],
    [22, 23],
    [25, 25],
    [31, 28],
    [37, 33],
    [45, 34],
    [50, 42],
    [53, 44],
    [55, 46],
], dtype=np.int32)  # index 0..13

# --- Rook Mobility (SF11 MobilityBonus × Pawn ratio MG×0.585 EG×0.42) --- max 14 squares
ROOK_MOBILITY_BONUS = np.array([
    [-18, -27],
    [7, -5],
    [24, 18],
    [24, 30],
    [24, 36],
    [24, 36],
    [24, 48],
    [24, 49],
    [24, 50],
    [24, 52],
    [24, 56],
    [24, 62],
    [24, 68],
    [28, 71],
    [30, 72],
], dtype=np.int32)  # index 0..14

# --- Queen Mobility (SF11 MobilityBonus × Pawn ratio MG×0.585 EG×0.42) --- max 27 squares
QUEEN_MOBILITY_BONUS = np.array([
    [-24, -18],
    [-8, -9],
    [21, 6],
    [21, 10],
    [21, 19],
    [29, 24],
    [29, 29],
    [29, 31],
    [29, 35],
    [29, 38],
    [29, 40],
    [30, 44],
    [32, 46],
    [33, 48],
    [37, 51],
    [39, 53],
    [46, 61],
    [48, 62],
    [48, 62],
    [53, 64],
    [55, 65],
    [62, 73],
    [64, 74],
    [65, 76],
    [66, 78],
    [67, 82],
    [67, 88],
    [67, 92],
], dtype=np.int32)  # index 0..27


# =============================================================================
# --- Piece Coordination Constants / 棋子協同常量 ---
# =============================================================================

# --- Bishop Pair / 雙象優勢 ---
# Bonus for having both bishops. This bonus is generally stronger in open positions.
# 擁有雙象的獎勵。這個獎勵在開放局面中通常更強。
BISHOP_PAIR_BONUS = np.array([22, 25], dtype=np.int32) # MG, EG

# --- Rook on Open/Semi-Open File / 車在開放線/半開放線 ---
# Bonus for a rook on a file with no friendly pawns (semi-open)
# or no pawns at all (open).
# 車在沒有己方兵（半開放線）或完全沒有兵（開放線）的直線上的獎勵。
ROOK_ON_SEMI_OPEN_FILE_BONUS = np.array([10, 5], dtype=np.int32) # MG, EG
ROOK_ON_OPEN_FILE_BONUS = np.array([20, 10], dtype=np.int32) # MG, EG

ROOK_ON_SEVENTH_BONUS = np.array([0, 27], dtype=np.int32) # MG, EG

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
    [2, 7], # Rank 3
    [15, 23], # Rank 4
    [30, 40], # Rank 5
    [63, 123], # Rank 6
    [149, 241], # Rank 7
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
    [17, 24], # Rank 5
    [37, 71], # Rank 6
    [80, 128], # Rank 7
    [0, 0], # Rank 8
], dtype=np.int32)

# --- Isolated Pawns / 孤兵 ---
# Penalty for each isolated pawn on a file.
# 每一個孤兵的懲罰。
ISOLATED_PAWN_PENALTY = np.array([0, -13], dtype=np.int32) # MG, EG

# --- Doubled Pawns / 重疊兵 ---
# Penalty for each doubled pawn on a file.
# 每一個重疊兵的懲罰。
DOUBLED_PAWN_PENALTY = np.array([-13, -23], dtype=np.int32) # MG, EG

# --- Connected Pawn Bonus by Rank (SF11) / SF11 連結兵獎勵（按橫排） ---
# Applied to ALL pawns that are in phalanx (side-by-side) or supported (diagonally behind).
# 適用於所有處於並列或受支撐狀態的兵。
CONNECTED_BONUS = np.array([0, 4, 7, 12, 19, 28, 39, 0], dtype=np.int32)
CONNECTED_SUPPORT_WEIGHT = np.int32(10)  # SF11 value was 21

# --- Connected Passed Pawns / 連結通路兵 --- (legacy, kept for reference)
CONNECTED_PASSED_PAWN_BONUS = np.array([15, 35], dtype=np.int32) # MG, EG (not used in evaluation)

# --- Backward Pawns / 後兵 ---
# Penalty for a backward pawn.
# 後兵的懲罰。
# Negative values, applied with += (consistent with ISOLATED_PAWN_PENALTY and DOUBLED_PAWN_PENALTY)
BACKWARD_PAWN_PENALTY = np.array([0, -14], dtype=np.int32) # MG, EG

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
    [23, 14], # Rank 4
    [46, 37], # Rank 5
    [53, 42], # Rank 6
    [54, 45], # Rank 7
    [0, 0], # Rank 8
], dtype=np.int32)

# 主教前哨獎勵
OUTPOST_BONUS_BISHOP = np.array([
    [0, 0], # Rank 1
    [0, 0], # Rank 2
    [1, 4], # Rank 3
    [18, 8], # Rank 4
    [32, 21], # Rank 5
    [36, 26], # Rank 6
    [37, 33], # Rank 7
    [0, 0], # Rank 8
], dtype=np.int32)

# Bonus if the outpost is a "Hole" (cannot be attacked by enemy pawns at all).
# 如果前哨是“洞”（完全無法被敵方兵攻擊），則給予額外獎勵。
OUTPOST_HOLE_BONUS = np.array([15, 3], dtype=np.int32) # MG, EG

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
PAWN_STORM_PENALTY_BY_RANK = np.array([0, 122, 78, 30, 13, 13, 11, 0], dtype=np.int32)

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
THREAT_SAFE_PAWN = np.array([58, 37], dtype=np.int32)  # SF11: (173, 94)

# ThreatByMinor[target_piece_type]: Minor (N/B) attacks piece of given type.
# Index: 0=Pawn, 1=Knight, 2=Bishop, 3=Rook, 4=Queen, 5=King
# 輕子威脅：馬/象攻擊對應類型棋子的獎勵。
THREAT_BY_MINOR = np.array([
    [0, 17], # vs Pawn
    [27, 18], # vs Knight
    [36, 22], # vs Bishop
    [61, 29], # vs Rook
    [67, 64], # vs Queen
    [0, 0], # vs King
], dtype=np.int32)

# ThreatByRook[target_piece_type]: Rook attacks piece of given type (only if weak).
# Index: 0=Pawn, 1=Knight, 2=Bishop, 3=Rook, 4=Queen, 5=King
# 車威脅：車攻擊對應類型棋子的獎勵。
THREAT_BY_ROOK = np.array([
    [1, 26], # vs Pawn
    [24, 32], # vs Knight
    [22, 28], # vs Bishop
    [2, 24], # vs Rook
    [40, 20], # vs Queen
    [0, 0], # vs King
], dtype=np.int32)

# ThreatByKing: King attacks a weakly defended enemy piece.
# 王攻擊弱子：王攻擊敵方弱子的獎勵。
THREAT_BY_KING = np.array([27, 61], dtype=np.int32)  # SF11: (24, 89)

# Hanging: Enemy piece is attacked + not strongly protected.
# 懸掛子：敵方棋子被攻擊且未被強力保護。
THREAT_HANGING = np.array([24, 3], dtype=np.int32)  # SF11: (69, 36)

# RestrictedPiece: Enemy piece moves are restricted by our attacks.
# 限制棋子行動力：敵方棋子的走子受到我方攻擊限制。
THREAT_RESTRICTED_PIECE = np.array([5, 4], dtype=np.int32)  # SF11: (7, 7)

# ThreatByPawnPush: Pawn push threatens enemy pieces on next move.
# 兵推威脅：兵推進後能威脅敵子。
THREAT_PAWN_PUSH = np.array([15, 9], dtype=np.int32) # SF11: (48, 39)

# --- Legacy aliases (kept for backward compatibility, now unused in threats) ---
# Minor Attacking Major (replaced by THREAT_BY_MINOR)
THREAT_MINOR_ON_MAJOR = np.array([25, 15], dtype=np.int32)  # kept for reference
# Rook Attacking Queen (replaced by THREAT_BY_ROOK)
THREAT_ROOK_ON_QUEEN = np.array([20, 10], dtype=np.int32)   # kept for reference

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
BISHOP_PAWNS_PENALTY = np.array([2, 13], dtype=np.int32)  # SF11: (3, 7)

# TrappedRook: Penalty for rook with mobility <= 3 trapped by own king.
# 困車懲罰：車移動格數 ≤ 3 且在己方王同側。
TRAPPED_ROOK = np.array([26, 3], dtype=np.int32)  # SF11: (52, 10)

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

MAX_KING_ATTACKERS = 1000

PROXIMITY_ENEMY_WEIGHT = 5
PROXIMITY_FRIENDLY_WEIGHT = 2
MAX_PROXIMITY_BONUS = 150
KING_DANGER_SINGLE_ATTACKER_DIVISOR = 2
UNSTOPPABLE_PAWN_BONUS = 800
EG_KING_PAWN_PROXIMITY_WEIGHT = 5
BLOCKED_PASSER_DIVISOR = 2
