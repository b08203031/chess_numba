# chess_engine/constants.py

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

# 中局（Middle Game）材質值
# MG_MATERIAL_VALUES = np.array([49, 351, 355, 460, 946], dtype=np.int32)
# # 殘局（End Game）材質值
# EG_MATERIAL_VALUES = np.array([120, 310, 340, 530, 950], dtype=np.int32)

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

def _create_pst(values):
    """
    輔助函式：從 2D 列表創建展平的 8x8 PST 陣列。
    
    Args:
        values (list of list of int): 8x8 的分數列表。
        
    Returns:
        np.array: 展平的一維 numpy 陣列。
    """
    return np.array([val for row in reversed(values) for val in row], dtype=np.int32)


# --- Endgame PSTs / 殘局位置分數表 ---

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

# BISHOP_PST_EG = _create_pst([
#     [-20,-10,-10,-10,-10,-10,-10,-20],
#     [-10,  0,  0,  0,  0,  0,  0,-10],
#     [-10,  0,  5, 10, 10,  5,  0,-10],
#     [-10,  5,  5, 10, 10,  5,  5,-10],
#     [-10,  0, 10, 10, 10, 10,  0,-10],
#     [-10, 10, 10, 10, 10, 10, 10,-10],
#     [-10,  5,  0,  0,  0,  0,  5,-10],
#     [-20,-10,-10,-10,-10,-10,-10,-20]
# ])

# ROOK_PST_EG = _create_pst([
#     [0,  0,  0,  0,  0,  0,  0,  0],
#     [5, 10, 10, 10, 10, 10, 10,  5],
#     [-5,  0,  0,  0,  0,  0,  0, -5],
#     [-5,  0,  0,  0,  0,  0,  0, -5],
#     [-5,  0,  0,  0,  0,  0,  0, -5],
#     [-5,  0,  0,  0,  0,  0,  0, -5],
#     [-5,  0,  0,  0,  0,  0,  0, -5],
#     [0,  0,  0,  5,  5,  0,  0,  0]
# ])

# QUEEN_PST_EG = _create_pst([
#     [-20,-10,-10, -5, -5,-10,-10,-20],
#     [-10,  0,  0,  0,  0,  0,  0,-10],
#     [-10,  0,  5,  5,  5,  5,  0,-10],
#     [-5,  0,  5,  5,  5,  5,  0, -5],
#     [0,  0,  5,  5,  5,  5,  0, -5],
#     [-10,  5,  5,  5,  5,  5,  0,-10],
#     [-10,  0,  5,  0,  0,  0,  0,-10],
#     [-20,-10,-10, -5, -5,-10,-10,-20]
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
#     [-1, 3, -2, -1, -4, 2, 1, 3, -4, -2, -10, -18, -8, -8, -14, 0, 8, -5, -1, 3, 33, 12, 14, 1, -4, 4, 10, 6, 4, 3, -1, 1, 0, 0, 0, 2, -1, -3, 3, 4, 7, 9, 17, 23, 27, 16, 9, 3, 50, 52, 53, 49, 48, 50, 51, 49, -2, 2, 6, -2, -4, 1, -2, -2],
#     [-51, -32, -28, -24, -23, -22, -35, -48, -33, -20, 8, 10, 13, 0, -20, -35, -25, 12, 0, 17, 21, 13, 5, -26, -26, 1, 18, 13, 10, 9, 0, -30, -31, 1, 11, 4, -1, 8, 2, -30, -29, -2, 4, 8, 12, 10, 0, -30, -39, -22, -2, -1, 3, 1, -19, -39, -49, -37, -31, -31, -29, -24, -43, -52],
#     [-14, -11, -4, -10, -10, -4, -9, -14, -9, 16, 2, 8, 9, 3, 5, -7, -11, 14, 11, -4, 4, 12, 6, -11, -12, 0, 3, 5, 12, -6, -1, -18, -10, 3, 1, 7, 6, 8, -2, -11, -10, -1, 4, 4, 7, 5, -4, -13, -11, 2, -1, -1, 0, 2, 2, -7, -20, -7, -6, -10, -9, -10, -7, -19],
#     [12, 13, 5, -6, -6, -7, 6, 21, 0, 8, 8, -5, 0, -1, -1, -3, -4, 0, 0, -1, -2, 1, -5, -4, -6, 0, -1, 2, 0, -5, 2, -5, -5, 1, -4, -6, -2, -2, -4, -1, -6, -6, 3, -3, -3, 1, -1, -8, 3, 6, 6, 7, 8, 9, 10, 4, -3, -2, 3, -5, -4, 0, -2, -4],
#     [-17, -6, -5, 34, 3, -5, -12, -18, -9, 1, 19, -3, 18, 7, 1, -11, -12, 1, 10, 8, 0, -6, -9, -14, 3, -2, 3, 0, 4, -1, -9, -12, -5, -2, 0, 4, 5, 8, -1, -12, -13, -1, 2, 1, 5, 2, 1, -12, -10, -4, 2, -1, 1, -1, -1, -8, -21, -9, -11, -7, -6, -12, -8, -19],
#     [24, 24, 10, -2, 16, 12, -1, 10, 18, 18, 4, 0, 3, 6, 29, 30, -8, -22, -17, -19, -22, -19, -19, -10, -22, -30, -30, -39, -42, -27, -32, -18, -29, -39, -41, -49, -52, -42, -39, -31, -31, -40, -39, -48, -48, -39, -47, -26, -30, -40, -41, -50, -50, -40, -37, -30, -28, -41, -43, -49, -54, -43, -41, -27],
# ], dtype=np.int32)

# PST_EG = np.array([
#     PAWN_PST_EG, KNIGHT_PST_EG, BISHOP_PST_EG, ROOK_PST_EG, QUEEN_PST_EG, KING_PST_EG
# ])


# =============================================================================
# --- Other Evaluation Constants / 其他評估常量 ---
# =============================================================================

# --- Initiative Bonus / 主動權獎勵 ---
# A small bonus awarded to the side to move, acknowledging the advantage of having the turn.
# This bonus is tapered and disappears in the endgame.
# 給予輪到行棋一方的小獎勵，承認擁有下棋權的優勢。此獎勵是漸進的，在殘局中會消失。
# INITIATIVE_BONUS = 10 # centipawns
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
# KNIGHT_MOBILITY_WEIGHT = np.array([4, 2], dtype=np.int32) # MG, EG

# --- Bishop Mobility ---
BISHOP_MOBILITY_BASE_MOVES = 5
# BISHOP_MOBILITY_WEIGHT = np.array([4, 2], dtype=np.int32) # MG, EG

# --- Rook Mobility ---
ROOK_MOBILITY_BASE_MOVES = 6
# ROOK_MOBILITY_WEIGHT = np.array([3, 1], dtype=np.int32) # MG, EG

# --- Queen Mobility ---
QUEEN_MOBILITY_BASE_MOVES = 8
# QUEEN_MOBILITY_WEIGHT = np.array([2, 1], dtype=np.int32) # MG, EG


# =============================================================================
# --- Piece Coordination Constants / 棋子協同常量 ---
# =============================================================================

# --- Bishop Pair / 雙象優勢 ---
# Bonus for having both bishops. This bonus is generally stronger in open positions.
# 擁有雙象的獎勵。這個獎勵在開放局面中通常更強。
# BISHOP_PAIR_BONUS = np.array([10, 20], dtype=np.int32) # MG, EG

# --- Rook on Open/Semi-Open File / 車在開放線/半開放線 ---
# Bonus for a rook on a file with no friendly pawns (semi-open)
# or no pawns at all (open).
# 車在沒有己方兵（半開放線）或完全沒有兵（開放線）的直線上的獎勵。
# ROOK_ON_SEMI_OPEN_FILE_BONUS = np.array([15, 10], dtype=np.int32) # MG, EG
# ROOK_ON_OPEN_FILE_BONUS = np.array([25, 15], dtype=np.int32) # MG, EG

# ROOK_ON_SEVENTH_BONUS = np.array([20, 50], dtype=np.int32) # MG, EG
# =============================================================================
# --- Pawn Structure Constants / 兵型結構常量 ---
# =============================================================================

# --- Passed Pawns / 通路兵 ---
# Bonus for having a passed pawn, scaled by its rank. (Values increased significantly)
# Index corresponds to the pawn's rank (1-8, though rank 1 and 8 are not used for pawns).
# 通路兵的獎勵，根據其橫排進行縮放。（數值已顯著增加）
# 索引對應於兵的橫排（1-8，雖然 1 和 8 橫排不用於兵）。
# PASSED_PAWN_BONUS = np.array([
#     # MG, EG
#     [  0,   0], # Rank 1
#     [ 10,  20], # Rank 2
#     [ 20,  40], # Rank 3
#     [ 35,  80], # Rank 4
#     [ 50, 120], # Rank 5
#     [ 80, 200], # Rank 6
#     [150, 350], # Rank 7
#     [  0,   0]  # Rank 8
# ], dtype=np.int32)

# --- Isolated Pawns / 孤兵 ---
# Penalty for each isolated pawn on a file.
# # 每一個孤兵的懲罰。
# ISOLATED_PAWN_PENALTY = np.array([-10, -5], dtype=np.int32) # MG, EG

# # --- Doubled Pawns / 重疊兵 ---
# # Penalty for each doubled pawn on a file.
# # 每一個重疊兵的懲罰。
# DOUBLED_PAWN_PENALTY = np.array([-15, -10], dtype=np.int32) # MG, EG

# # --- Connected Passed Pawns / 連結通路兵 ---
# # Bonus for each passed pawn that is connected to another passed pawn.
# # 每個連結通路兵的獎勵。
# CONNECTED_PASSED_PAWN_BONUS = np.array([40, 80], dtype=np.int32) # MG, EG


# =============================================================================
# --- Outpost Constants / 前哨常量 ---
# =============================================================================
# Outpost Bonuses by Rank (0-7).
# Indices: Rank 0 to Rank 7.
# Values: [MG, EG]
# 騎士前哨獎勵
# OUTPOST_BONUS_KNIGHT = np.array([
#     [0, 0],    # Rank 1
#     [0, 0],    # Rank 2
#     [10, 5],   # Rank 3
#     [30, 15],  # Rank 4
#     [50, 40],  # Rank 5
#     [40, 30],  # Rank 6 (Octopus)
#     [20, 10],  # Rank 7
#     [0, 0]     # Rank 8
# ], dtype=np.int32)

# # 主教前哨獎勵
# OUTPOST_BONUS_BISHOP = np.array([
#     [0, 0],    # Rank 1
#     [0, 0],    # Rank 2
#     [10, 5],   # Rank 3
#     [20, 15],  # Rank 4
#     [30, 25],  # Rank 5
#     [20, 15],  # Rank 6
#     [10, 5],   # Rank 7
#     [0, 0]     # Rank 8
# ], dtype=np.int32)

# Bonus if the outpost is a "Hole" (cannot be attacked by enemy pawns at all).
# 如果前哨是“洞”（完全無法被敵方兵攻擊），則給予額外獎勵。
# OUTPOST_HOLE_BONUS = np.array([25, 15], dtype=np.int32) # MG, EG


# =============================================================================
# --- King Safety Constants (NEW - based on Chessprogramming Wiki) / 王的安全常量 ---
# =============================================================================

# --- Phase 2: Attacking the King Zone (Non-Linear Model) / 攻擊王翼區域（非線性模型） ---
# Attack units for each piece type. Order: P, N, B, R, Q
# 每個棋子類型的攻擊單位。順序：兵、馬、象、車、后
# Updated: Aggressive weights for R and Q
# KING_SAFETY_WEAK_UNITS = np.array([2, 5, 5, 10, 16], dtype=np.int32) # P, N, B, R, Q
# KING_SAFETY_ATTACK_UNITS = np.array([1, 4, 4, 9, 15], dtype=np.int32) # P, N, B, R, Q

# A non-linear table where the index is the sum of attack units, and the value is the penalty.
# The penalty grows exponentially, rewarding multi-piece attacks.
# 一個非線性表格，索引是攻擊單位的總和，值是懲罰分數。懲罰呈指數增長，獎勵多子協同攻擊。
# Updated: Steeper, quadratic-plus growth curve
# KING_SAFETY_TABLE = np.array([0, 0, 2, 4, 7, 12, 18, 24, 32, 41, 50, 60, 71, 85, 98, 111, 128, 143, 160, 180, 200, 219, 242, 264, 288, 312, 338, 364, 392, 420, 450, 481, 511, 545, 578, 612, 648, 684, 723, 760, 799, 840, 883, 925, 968, 1000, 1001, 1001, 1001, 1001, 1002, 1002, 1002, 1002, 1002, 1002, 1002, 1003, 1003, 1003, 1003, 1003, 1003, 1003, 1003, 1003, 1003, 1003, 1004, 1004, 1004, 1004, 1004, 1004, 1004, 1004, 1004, 1004, 1004, 1004, 1004, 1005, 1005, 1005, 1005, 1005, 1005, 1005, 1005, 1005, 1005, 1005, 1005, 1005, 1005, 1005, 1005, 1005, 1005, 1005], dtype=np.int32)


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
# PAWN_STORM_PENALTY_BY_RANK = np.array([0, 0, 80, 50, 30, 10, 5, 0], dtype=np.int32)

SCALING_WEIGHTS = np.array([0, 4, 4, 6, 10], dtype=np.int32) # N, B, R, Q - for scaling factor / 用於縮放因子的權重
MAX_SCALING_MATERIAL = (2*4 + 2*4 + 2*6 + 1*10) # Sum of all weights for one side / 一方所有權重的總和

# PAWN_SHIELD_MISSING_PENALTY = 60
# PAWN_SHIELD_INTACT_BONUS = 20
# PAWN_SHIELD_ADVANCED_BONUS = 10
# PAWN_SHIELD_PUSHED_PENALTY = 20
# KING_OPEN_FILE_PENALTY = 50
# KING_SEMI_OPEN_FILE_PENALTY = 25

EG_SAFETY_SCALE = 0.5 # Scale down endgame king safety impact / 縮減殘局王的安全影響

# # =============================================================================
# # --- Threat Evaluation Constants / 威脅評估常量 ---
# # =============================================================================
# # Derived from Stockfish 11 but simplified and scaled.

# # Threat By Safe Pawn: Friendly pawn attacks enemy piece (N, B, R, Q).
# # 兵的威脅：己方兵攻擊敵方棋子（N, B, R, Q）。
# THREAT_SAFE_PAWN = np.array([60, 60], dtype=np.int32) # MG, EG

# # Minor Attacking Major: Knight/Bishop attacking Rook/Queen.
# # 輕子攻擊重子：馬/象攻擊車/后。
# THREAT_MINOR_ON_MAJOR = np.array([25, 15], dtype=np.int32) # MG, EG

# # Rook Attacking Queen: Rook attacking Queen.
# # 車捉后：車攻擊后。
# THREAT_ROOK_ON_QUEEN = np.array([20, 10], dtype=np.int32) # MG, EG

# # Hanging Pieces: Enemy piece is attacked and undefended.
# # 懸掛子：敵方棋子被攻擊且未被防守。
# # This is a bonus for the ATTACKER.
# # 這是給攻擊者的獎勵。
# THREAT_HANGING = np.array([35, 20], dtype=np.int32) # MG, EG


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

# File constants
NOT_A_FILE = ~np.uint64(0x0101010101010101)
NOT_H_FILE = ~np.uint64(0x8080808080808080)

# Backward Pawn Penalty
# BACKWARD_PAWN_PENALTY = np.array([10, 25], dtype=np.int32) # MG, EG

MG_MATERIAL_VALUES = np.array([92, 600, 602, 758, 974], dtype=np.int32)

EG_MATERIAL_VALUES = np.array([71, 75, 267, 449, 695], dtype=np.int32)

PST_MG = np.array([
    [-12, 4, 10, 16, -1, 14, 13, 12, 15, 35, -9, -7, -6, -21, -31, -4, 9, -24, 5, 23, 33, -1, -1, -10, 7, -31, 6, 5, 25, -9, -2, -8, -12, -19, -34, 2, -3, -17, -2, 36, 28, -13, 31, 28, -15, 45, 26, -22, 61, 84, 62, 46, 62, 39, 30, 47, 4, 30, 48, 33, -28, -26, -26, -43],
    [-78, -24, -6, -14, -19, -33, -38, -61, -25, -19, 21, 11, -6, -2, 19, -48, 5, 13, 2, 13, 0, 38, 5, -27, 4, 21, 10, 20, 21, -28, 22, -17, -32, -30, 7, -1, -10, -5, 49, -27, -37, -3, 24, 26, 5, 13, -7, -49, -47, -23, 13, -2, 14, -10, -10, -71, -48, -46, -16, -17, -9, -27, -54, -50],
    [-16, 1, 18, -63, -16, -11, -2, -26, -22, 21, -8, 8, 8, 9, -26, -17, 14, 30, 23, 10, 3, -22, 19, -13, 1, 1, 1, -13, -10, -40, 36, -30, -18, 6, 0, 19, -8, 33, 23, -48, 5, -17, -6, 10, -8, -32, 3, -13, -33, 5, 10, -23, 12, 13, -7, -25, -30, -11, 10, 1, 8, -9, 11, -27],
    [43, -19, 18, -2, -5, -17, -5, 3, 0, -1, -15, 1, 21, 9, 20, -17, 34, 4, -7, -8, -4, 3, -13, 0, 20, 21, 11, -4, 47, -2, -34, -4, 17, 33, 9, -5, -31, -7, -12, -46, 34, 4, 35, 15, 11, -5, -41, 31, -8, 18, -1, 14, 12, -16, -7, 13, 2, -18, 10, -36, -13, -15, 9, 22],
    [-24, -13, -22, 51, -8, 10, 25, 19, -28, -6, 36, 26, 31, 36, 0, 5, 1, -7, -9, 11, -31, 30, -7, 23, -6, -10, -14, 19, -46, 16, -25, -44, 14, 30, 17, 19, -13, 6, -22, -21, -3, -18, -17, -36, -23, 18, -19, -21, -17, -17, -8, 28, -1, 10, -2, 19, 13, -49, -20, -64, -15, 22, -16, 1],
    [9, 41, -8, 26, 48, 32, -12, -3, 30, 27, 20, -20, -27, 23, 16, 15, 2, -8, -13, -14, -21, 11, 27, -22, -4, -17, -2, -48, -82, -36, -32, -28, -18, -12, -38, -41, -4, -56, -21, -23, -15, -11, -37, -50, -69, -39, -45, 21, -25, -94, -30, -16, -68, -25, -59, -28, -5, -22, -16, -27, -80, -10, -23, -46],
], dtype=np.int32)

PST_EG = np.array([
    [2, 9, -53, 10, -24, -18, -12, 38, -31, 4, 27, 22, 8, -14, 4, 17, 0, 34, -18, -18, -14, -16, -5, 22, -27, 17, -5, -14, 5, 38, 19, 36, 40, 46, 30, 29, 19, -5, 45, 70, 29, 50, 25, 31, 64, 53, 28, 53, 56, 82, 73, 80, 53, 95, 103, 94, 19, -31, 16, 10, -14, 0, 2, -19],
    [-62, -84, -25, -22, -42, -16, -5, -100, -43, -16, -25, -16, -29, -5, -1, -59, -23, -13, 44, 49, 24, 23, -50, -20, -19, 14, 31, 47, -5, -6, 34, -22, -13, -33, 60, 18, 40, -22, 4, -41, -20, -8, -12, 1, 30, -29, -39, -27, -30, -37, -19, 0, 20, 36, -34, -14, -33, -53, -23, -18, -48, -33, -60, -57],
    [9, 7, -29, -25, 5, 13, -21, 7, 7, -6, -23, -42, -19, -41, 9, 1, -2, 61, 32, 9, 6, -10, 33, -43, -55, 6, 2, 21, -1, -1, 28, -24, -37, 18, 22, 38, 26, -11, 2, -1, -4, -12, 2, 12, 5, 18, 25, 2, 1, -11, -21, 12, -4, 2, -42, -12, -1, -41, 0, 2, 1, -14, -17, 26],
    [20, 8, -23, -14, 15, -26, -9, -12, 11, 34, 16, 5, 4, -22, 0, -42, -25, -5, 7, 19, 20, -5, 19, -8, -15, -2, 4, -17, -6, 6, -13, -34, -25, 4, 20, -4, 10, 15, -27, -18, -11, 5, -2, -50, 21, -5, -14, -24, -1, 21, 18, -21, 2, 24, 15, -16, 11, -16, -10, -14, -9, 3, -27, -17],
    [10, 10, -12, -32, 18, -31, 13, -12, 25, -21, 3, -5, 26, -14, 4, -12, -44, 29, 13, 27, -5, -5, -7, -35, 14, -26, 26, -28, 39, 32, 17, 18, -29, -10, -19, 16, -23, -4, -14, -13, -7, 22, 28, 6, -6, 6, 13, -18, -9, -3, -11, -30, 4, -29, -16, 11, -31, 4, -1, -7, 19, -63, -11, 2],
    [-59, -20, -29, -34, -52, -26, -10, -17, -64, 1, -4, 0, -22, -13, -54, 3, -31, -54, 50, 15, 27, 11, -39, -16, -4, -4, 16, 24, 27, 27, 6, -26, -36, -12, 46, 34, 25, 22, -4, -34, -41, -18, 15, 32, 20, -21, -29, -54, -55, -56, -7, 11, 3, -15, -14, -27, -36, -36, -30, -7, -12, -40, -54, -78],
], dtype=np.int32)


KNIGHT_MOBILITY_WEIGHT = np.array([1, 42], dtype=np.int32)

BISHOP_MOBILITY_WEIGHT = np.array([0, 21], dtype=np.int32)

ROOK_MOBILITY_WEIGHT = np.array([2, 0], dtype=np.int32)

QUEEN_MOBILITY_WEIGHT = np.array([0, 0], dtype=np.int32)

BISHOP_PAIR_BONUS = np.array([0, 9], dtype=np.int32)

ROOK_ON_SEMI_OPEN_FILE_BONUS = np.array([26, 14], dtype=np.int32)

ROOK_ON_OPEN_FILE_BONUS = np.array([48, 12], dtype=np.int32)

ROOK_ON_SEVENTH_BONUS = np.array([9, 36], dtype=np.int32)

PASSED_PAWN_BONUS = np.array([
    [0, 0],
    [27, 50],
    [29, 84],
    [31, 105],
    [44, 107],
    [66, 213],
    [145, 323],
    [0, 0],
], dtype=np.int32)

ISOLATED_PAWN_PENALTY = np.array([-30, -16], dtype=np.int32)

DOUBLED_PAWN_PENALTY = np.array([-24, -15], dtype=np.int32)

CONNECTED_PASSED_PAWN_BONUS = np.array([19, 60], dtype=np.int32)

BACKWARD_PAWN_PENALTY = np.array([0, 10], dtype=np.int32)

OUTPOST_BONUS_KNIGHT = np.array([
    [0, 0],
    [0, 0],
    [20, 0],
    [45, 34],
    [56, 16],
    [6, 17],
    [34, 20],
    [0, 0],
], dtype=np.int32)

OUTPOST_BONUS_BISHOP = np.array([
    [0, 0],
    [0, 0],
    [7, 8],
    [1, 31],
    [17, 8],
    [39, 0],
    [6, 4],
    [0, 0],
], dtype=np.int32)

OUTPOST_HOLE_BONUS = np.array([50, 13], dtype=np.int32)

KING_SAFETY_WEAK_UNITS = np.array([1, 5, 5, 8, 12], dtype=np.int32)

KING_SAFETY_ATTACK_UNITS = np.array([1, 5, 5, 8, 12], dtype=np.int32)

KING_SAFETY_TABLE = np.array([0, 0, 2, 4, 8, 16, 22, 32, 44, 60, 75, 93, 102, 112, 118, 128, 131, 164, 164, 178, 196, 215, 256, 292, 292, 323, 366, 390, 406, 441, 447, 497, 521, 534, 567, 575, 644, 698, 743, 768, 788, 858, 882, 932, 973, 980, 992, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000], dtype=np.int32)

KING_TROPISM_WEIGHTS = np.array([15, 18, 21, 24, 37], dtype=np.int32)

PAWN_STORM_PENALTY_BY_RANK = np.array([13, 6, 93, 37, 9, 39, 34, 0], dtype=np.int32)

SCALING_WEIGHTS = np.array([11, 14, 0, 0, 0], dtype=np.int32)

PAWN_SHIELD_MISSING_PENALTY = 0

PAWN_SHIELD_INTACT_BONUS = 47

PAWN_SHIELD_ADVANCED_BONUS = 44

PAWN_SHIELD_PUSHED_PENALTY = 27

KING_OPEN_FILE_PENALTY = 38

KING_SEMI_OPEN_FILE_PENALTY = 10

THREAT_SAFE_PAWN = np.array([50, 64], dtype=np.int32)

THREAT_MINOR_ON_MAJOR = np.array([13, 10], dtype=np.int32)

THREAT_ROOK_ON_QUEEN = np.array([6, 28], dtype=np.int32)

THREAT_HANGING = np.array([28, 40], dtype=np.int32)

INITIATIVE_BONUS = 26







