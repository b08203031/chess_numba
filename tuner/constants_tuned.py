# chess_engine/constants.py
#所有的參數

import numpy as np

"""
此模組定義了西洋棋引擎中使用的所有常量。
包含了棋盤表示、棋子值、位置分數（PST）、搜尋參數、剪枝開關以及位元棋盤的輔助常量。
"""

# =============================================================================
# --- Material Values (Tapered) / 棋子材質值（漸進式） ---
# =============================================================================
# Values are in centipawns / 單位為分（centipawns）

# Index mapping for pieces / 棋子索引映射
# PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING
#  0  ,   1   ,   2   ,  3  ,   4  ,  5

MG_MATERIAL_VALUES = np.array([100, 319, 331, 500, 901, 1], dtype=np.int32)

EG_MATERIAL_VALUES = np.array([119, 310, 336, 531, 949, 1], dtype=np.int32)

# =============================================================================
# --- Game Phase Calculation / 遊戲階段計算 ---
# =============================================================================
# Weights for each piece type to determine the game phase / 每個棋子類型的權重，用於確定遊戲階段
PHASE_WEIGHTS = np.array([0, 1, 1, 2, 4, 0], dtype=np.int32)
# 最大階段值（初始局面所有非兵棋子的權重總和）
MAX_PHASE = np.sum(PHASE_WEIGHTS * np.array([8, 2, 2, 2, 1, 1])) * 2 # Pawns are not counted

PST_MG = np.array([
    [0, 0, -2, 0, -1, 0, 1, -1, 3, 9, 10, -22, -18, 11, 9, 5, 6, -6, -10, 1, -2, -11, -7, 6, -2, 0, 4, 19, 20, 0, 4, -1, 5, 6, 8, 23, 23, 9, 5, 4, 11, 7, 19, 29, 31, 19, 9, 8, 50, 48, 50, 49, 49, 51, 48, 51, 1, -1, 1, 0, -2, 0, -1, -1],
    [-53, -40, -30, -28, -29, -31, -41, -49, -39, -20, -1, 5, 7, 0, -18, -38, -28, 3, 10, 14, 17, 12, 5, -32, -30, 1, 15, 19, 21, 17, -1, -30, -31, 4, 15, 21, 20, 14, 5, -30, -30, -1, 10, 15, 16, 9, 1, -30, -39, -20, -1, -1, 1, 0, -18, -40, -50, -40, -30, -30, -30, -31, -40, -48],
    [-19, -10, -10, -8, -9, -11, -10, -21, -9, 7, 0, -1, -1, 1, 5, -10, -10, 9, 12, 11, 12, 11, 9, -11, -12, -1, 9, 11, 11, 11, -2, -11, -8, 8, 5, 12, 9, 5, 5, -12, -8, -2, 6, 9, 8, 2, -1, -9, -12, 1, 3, -1, -1, -2, 2, -8, -19, -10, -10, -12, -11, -10, -13, -20],
    [1, -1, -1, 3, 7, 0, -1, -1, -5, -1, -1, 0, -1, 1, 0, -4, -6, 2, 0, 1, 0, 1, 1, -3, -6, 0, 3, -1, 2, 1, 1, -5, -5, 1, -1, -1, 0, 2, 2, -5, -5, 2, 0, -1, -2, 2, 1, -6, 4, 10, 12, 9, 8, 8, 11, 4, 0, -1, -1, 3, 1, 1, 2, 1],
    [-17, -10, -10, -4, -8, -12, -11, -19, -13, -2, 4, 1, -1, 2, 0, -11, -10, 6, 6, 3, 5, 5, -2, -11, -2, 0, 5, 3, 3, 6, 1, -4, -2, -1, 5, 4, 5, 5, -1, -6, -9, 0, 4, 7, 5, 5, 0, -8, -7, -1, 0, 1, -1, 1, -1, -10, -18, -10, -10, -7, -5, -10, -10, -21],
    [21, 30, 10, 1, -1, 9, 32, 20, 18, 19, 0, -1, -2, -1, 22, 20, -11, -19, -21, -21, -21, -20, -22, -12, -20, -31, -28, -41, -39, -32, -31, -19, -30, -40, -39, -49, -51, -41, -39, -31, -29, -39, -41, -49, -52, -40, -41, -30, -29, -40, -39, -49, -51, -40, -39, -29, -30, -40, -40, -51, -49, -41, -39, -31],
], dtype=np.int32)

PST_EG = np.array([
    [-2, 0, 0, 1, 1, 1, -2, -2, 8, 9, 11, 11, 11, 9, 12, 9, 9, 11, 11, 10, 11, 11, 11, 9, 19, 20, 20, 19, 19, 20, 17, 20, 29, 30, 30, 31, 30, 30, 30, 30, 47, 50, 49, 49, 51, 51, 51, 50, 78, 80, 82, 82, 80, 81, 82, 79, 0, -1, 1, -1, 0, -3, -2, 0],
    [-50, -40, -29, -29, -29, -31, -41, -49, -41, -20, 1, 0, 0, -1, -21, -41, -29, 1, 8, 13, 13, 9, -2, -31, -32, 4, 13, 21, 18, 15, 5, -31, -29, 0, 14, 21, 21, 17, 1, -30, -30, 5, 10, 19, 15, 11, 4, -32, -40, -21, 0, 6, 5, -1, -20, -40, -49, -39, -29, -32, -30, -30, -38, -50],
    [-21, -8, -12, -11, -11, -11, -9, -21, -13, 6, 0, -1, 1, -2, 5, -12, -5, 11, 9, 9, 9, 11, 12, -10, -12, 0, 11, 11, 9, 9, 1, -11, -8, 5, 5, 11, 11, 5, 3, -7, -9, -1, 6, 10, 9, 4, 0, -12, -10, 2, 1, -1, 0, 2, 0, -11, -20, -10, -9, -10, -9, -10, -12, -20],
    [1, -1, 1, 6, 6, -1, 1, 0, -7, 1, -1, -2, -1, 0, -1, -4, -5, -1, -1, 0, 1, 1, -1, -5, -6, 1, -1, -1, 0, -1, 0, -5, -4, -1, -1, 0, 1, 2, -2, -6, -5, 0, -2, 1, -1, -1, 1, -6, 4, 11, 8, 12, 9, 11, 9, 5, 0, 1, 3, -2, 1, -2, 0, 2],
    [-20, -11, -8, -4, -6, -9, -9, -22, -10, -3, 3, -1, 0, 1, 0, -6, -10, 2, 5, 5, 4, 4, 1, -9, 1, 0, 6, 4, 5, 7, -1, -6, -5, 0, 5, 7, 6, 3, 0, -3, -9, 0, 6, 3, 7, 6, 0, -11, -10, -1, 1, -1, 2, -2, -2, -11, -19, -10, -10, -7, -4, -8, -10, -20],
    [-53, -29, -30, -30, -31, -31, -30, -50, -32, -28, -1, -1, 0, 0, -29, -30, -30, -8, 20, 30, 29, 19, -8, -30, -27, -9, 30, 41, 40, 32, -10, -30, -31, -11, 32, 40, 40, 30, -9, -30, -29, -11, 22, 33, 29, 20, -9, -30, -29, -19, -8, 1, 0, -8, -20, -31, -49, -40, -29, -19, -20, -31, -40, -52],
], dtype=np.int32)
# =============================================================================
# --- Other Evaluation Constants / 其他評估常量 ---
# =============================================================================

# --- Initiative Bonus / 主動權獎勵 ---
# A small bonus awarded to the side to move, acknowledging the advantage of having the turn.
# This bonus is tapered and disappears in the endgame.
# 給予輪到行棋一方的小獎勵，承認擁有下棋權的優勢。此獎勵是漸進的，在殘局中會消失。
INITIATIVE_BONUS = 10 # centipawns
INITIATIVE_PHASE_THRESHOLD = MAX_PHASE * 0.4 # Apply only when phase is above 40% of max / 僅在階段值高於最大值的 40% 時應用


# =============================================================================
# --- Mobility Constants / 機動性常量 ---
# =============================================================================
# Bonus for piece mobility, calculated as: (move_count - base_moves) * weight.
# This rewards active pieces and penalizes pieces that are blocked or restricted.
# 棋子機動性獎勵，計算方式為：(移動次數 - 基礎移動數) * 權重。
# 這獎勵活躍的棋子，懲罰被阻擋或受限的棋子。

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
# --- Piece Coordination Constants / 棋子協同常量 ---
# =============================================================================

# --- Bishop Pair / 雙象優勢 ---
# Bonus for having both bishops. This bonus is generally stronger in open positions.
# 擁有雙象的獎勵。這個獎勵在開放局面中通常更強。
BISHOP_PAIR_BONUS = np.array([10, 20], dtype=np.int32) # MG, EG

# --- Rook on Open/Semi-Open File / 車在開放線/半開放線 ---
# Bonus for a rook on a file with no friendly pawns (semi-open)
# or no pawns at all (open).
# 車在沒有己方兵（半開放線）或完全沒有兵（開放線）的直線上的獎勵。
ROOK_ON_SEMI_OPEN_FILE_BONUS = np.array([15, 10], dtype=np.int32) # MG, EG
ROOK_ON_OPEN_FILE_BONUS = np.array([25, 15], dtype=np.int32) # MG, EG


# =============================================================================
# --- Pawn Structure Constants / 兵型結構常量 ---
# =============================================================================

# --- Passed Pawns / 通路兵 ---
# Bonus for having a passed pawn, scaled by its rank. (Values increased significantly)
# Index corresponds to the pawn's rank (1-8, though rank 1 and 8 are not used for pawns).
# 通路兵的獎勵，根據其橫排進行縮放。（數值已顯著增加）
# 索引對應於兵的橫排（1-8，雖然 1 和 8 橫排不用於兵）。
PASSED_PAWN_BONUS = np.array([
    # MG, EG
    [  0,   0], # Rank 1
    [ 10,  20], # Rank 2
    [ 20,  40], # Rank 3
    [ 35,  80], # Rank 4
    [ 50, 120], # Rank 5
    [ 80, 200], # Rank 6
    [150, 350], # Rank 7
    [  0,   0]  # Rank 8
], dtype=np.int32)

# --- Isolated Pawns / 孤兵 ---
# Penalty for each isolated pawn on a file.
# 每一個孤兵的懲罰。
ISOLATED_PAWN_PENALTY = np.array([-10, -5], dtype=np.int32) # MG, EG

# --- Doubled Pawns / 重疊兵 ---
# Penalty for each doubled pawn on a file.
# 每一個重疊兵的懲罰。
DOUBLED_PAWN_PENALTY = np.array([-15, -10], dtype=np.int32) # MG, EG

# --- Connected Passed Pawns / 連結通路兵 ---
# Bonus for each passed pawn that is connected to another passed pawn.
# 每個連結通路兵的獎勵。
CONNECTED_PASSED_PAWN_BONUS = np.array([40, 80], dtype=np.int32) # MG, EG


# =============================================================================
# --- King Safety Constants (NEW - based on Chessprogramming Wiki) / 王的安全常量 ---
# =============================================================================

# --- Phase 2: Attacking the King Zone (Non-Linear Model) / 攻擊王翼區域（非線性模型） ---
# Attack units for each piece type. Order: P, N, B, R, Q
# 每個棋子類型的攻擊單位。順序：兵、馬、象、車、后
# Updated: Aggressive weights for R and Q
KING_SAFETY_ATTACK_UNITS = np.array([1, 3, 4, 7, 11], dtype=np.int32) # P, N, B, R, Q

# A non-linear table where the index is the sum of attack units, and the value is the penalty.
# The penalty grows exponentially, rewarding multi-piece attacks.
# 一個非線性表格，索引是攻擊單位的總和，值是懲罰分數。懲罰呈指數增長，獎勵多子協同攻擊。
# Updated: Steeper, quadratic-plus growth curve
KING_SAFETY_TABLE = np.array([
    min(int((i**2) / 2.5), 1000) for i in range(100)
], dtype=np.int32)

# --- Phase 3: King Tropism / 王的向性 ---
KING_TROPISM_MAX_DISTANCE = 14 # Max MANHATTAN distance / 最大曼哈頓距離
KING_TROPISM_WEIGHTS = np.array([1, 2, 3, 5, 8], dtype=np.int32) # P, N, B, R, Q

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
PAWN_STORM_PENALTY_BY_RANK = np.array([0, 0, 80, 50, 30, 10, 5, 0], dtype=np.int32)

SCALING_WEIGHTS = np.array([0, 4, 4, 6, 10], dtype=np.int32) # N, B, R, Q - for scaling factor / 用於縮放因子的權重
MAX_SCALING_MATERIAL = (2*4 + 2*4 + 2*6 + 1*10) # Sum of all weights for one side / 一方所有權重的總和

EG_SAFETY_SCALE = 0.5 # Scale down endgame king safety impact / 縮減殘局王的安全影響
