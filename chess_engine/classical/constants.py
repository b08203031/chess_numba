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
MG_MATERIAL_VALUES = np.array([100, 320, 330, 500, 900, 0], dtype=np.int32)
# 殘局（End Game）材質值
EG_MATERIAL_VALUES = np.array([120, 310, 340, 530, 950, 0], dtype=np.int32)

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

# --- Middlegame PSTs / 中局位置分數表 ---

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


# --- Endgame PSTs / 殘局位置分數表 ---

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

# SF11-inspired EG: bishops prefer activity and central diagonals in endgame
BISHOP_PST_EG = _create_pst([
    [-46,-24,-30,-10,-10,-30,-24,-46],
    [-30,-10,-14,  0,  0,-14,-10,-30],
    [-13,  0,  0,  8,  8,  0,  0,-13],
    [-16,  0, 14, 13, 13, 14,  0,-16],
    [-14,  0, 12, 13, 13, 12,  0,-14],
    [-24,  5,  3,  5,  5,  3,  5,-24],
    [-25,-16,  0,  1,  1,  0,-16,-25],
    [-37,-34,-30,-19,-19,-30,-34,-37]
])

# SF11-inspired EG: rooks more active, penalize confinement less, reward 7th-rank
ROOK_PST_EG = _create_pst([
    [ 18,  0, 19, 13, 13, 19,  0, 18],
    [  4,  5, 20, -5, -5, 20,  5,  4],
    [  6, -8, -2,  8,  8, -2, -8,  6],
    [ -6,  1, -9,  7,  7, -9,  1, -6],
    [ -5,  8,  7, -6, -6,  7,  8, -5],
    [  6,  1, -7, 10, 10, -7,  1,  6],
    [ -12, -9, -1, -2, -2, -1, -9,-12],
    [ -9,-13,-10, -9, -9,-10,-13, -9]
])

# SF11-inspired EG: queens centralize more aggressively, avoid corners
QUEEN_PST_EG = _create_pst([
    [-75,-52,-43,-36,-36,-43,-52,-75],
    [-57,-31,-22, -4, -4,-22,-31,-57],
    [-47,-18, -9,  3,  3, -9,-18,-47],
    [-26, -3, 13, 24, 24, 13, -3,-26],
    [-29, -6,  9, 21, 21,  9, -6,-29],
    [-39,-18,-12,  1,  1,-12,-18,-39],
    [-55,-27,-24, -8, -8,-24,-27,-55],
    [-69,-57,-47,-36,-36,-47,-57,-69]
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


# --- Aggregated PSTs for easier access / 聚合 PST 以便於訪問 ---
# The order must match the piece index mapping / 順序必須與棋子索引映射匹配
# PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING

PST_MG = np.array([
    PAWN_PST_MG, KNIGHT_PST_MG, BISHOP_PST_MG, ROOK_PST_MG, QUEEN_PST_MG, KING_PST_MG
])

PST_EG = np.array([
    PAWN_PST_EG, KNIGHT_PST_EG, BISHOP_PST_EG, ROOK_PST_EG, QUEEN_PST_EG, KING_PST_EG
])


# =============================================================================
# --- Other Evaluation Constants / 其他評估常量 ---
# =============================================================================

# --- Initiative / Complexity Constants (SF11-inspired) ---
# Dynamic complexity-based score dampening, prevents draw-drift in simple endgames.
# 動態複雜度主動權修正，防止簡單殘局中的和棋漂移。

# Middlegame (MG) weights - scaled by Pawn ratio MG = 100/128 ≈ 0.78 / 中局主動權/複雜度係數
INITIATIVE_PASSED_WEIGHT_MG = np.int32(7)      # 9  * 0.78 ≈ 7 / 通路兵權重
INITIATIVE_PAWN_WEIGHT_MG = np.int32(9)        # 11 * 0.78 ≈ 9 / 兵數量權重
INITIATIVE_OUTFLANKING_WEIGHT_MG = np.int32(7) # 9  * 0.78 ≈ 7 / 側翼國王相對位置權重
INITIATIVE_INFILTRATION_WEIGHT_MG = np.int32(9)  # 12 * 0.78 ≈ 9 / 國王滲透權重
INITIATIVE_BOTH_FLANKS_WEIGHT_MG = np.int32(16)  # 21 * 0.78 ≈ 16 / 雙翼皆有兵權重
INITIATIVE_PAWN_ENDGAME_WEIGHT_MG = np.int32(40) # 51 * 0.78 ≈ 40 / 純王兵殘局權重
INITIATIVE_ALMOST_UNWIN_WEIGHT_MG = np.int32(34) # 43 * 0.78 ≈ 34 / 幾乎不可獲勝局面懲罰權重
INITIATIVE_OFFSET_MG = np.int32(-78)             # -100 * 0.78 ≈ -78 / 複雜度基礎偏置
INITIATIVE_MG_OFFSET = np.int32(39)              # +50 * 0.78 ≈ 39 / 主動權分數修正偏置

# Endgame (EG) weights - scaled by Pawn ratio EG = 120/213 ≈ 0.563 / 殘局主動權/複雜度係數
INITIATIVE_PASSED_WEIGHT_EG = np.int32(5)      # 9  * 0.563 ≈ 5 / 通路兵權重
INITIATIVE_PAWN_WEIGHT_EG = np.int32(6)        # 11 * 0.563 ≈ 6 / 兵數量權重
INITIATIVE_OUTFLANKING_WEIGHT_EG = np.int32(5) # 9  * 0.563 ≈ 5 / 側翼國王相對位置權重
INITIATIVE_INFILTRATION_WEIGHT_EG = np.int32(7)  # 12 * 0.563 ≈ 7 / 國王滲透權重
INITIATIVE_BOTH_FLANKS_WEIGHT_EG = np.int32(12)  # 21 * 0.563 ≈ 12 / 雙翼皆有兵權重
INITIATIVE_PAWN_ENDGAME_WEIGHT_EG = np.int32(29) # 51 * 0.563 ≈ 29 / 純王兵殘局權重
INITIATIVE_ALMOST_UNWIN_WEIGHT_EG = np.int32(24) # 43 * 0.563 ≈ 24 / 幾乎不可獲勝局面懲罰權重
INITIATIVE_OFFSET_EG = np.int32(-56)             # -100 * 0.563 ≈ -56 / 複雜度基礎偏置

TEMPO_BONUS = np.int32(22)                       # Positional bonus for the side to move / 輪行方（先手/Tempo）的位置分數獎勵 (SF11 S(28) scaled by 100/128)
LAZY_EVAL_THRESHOLD = np.int32(1100)             # Dynamic evaluation exit threshold / 動態評估提早結束閥值 (SF11 1400 scaled by 100/128)

# --- King-Pawn Endgame Specifics / 王兵殘局特定常數 ---
UNSTOPPABLE_PAWN_BONUS = np.int32(800)       # 王兵殘局中不可阻擋通路兵的額外獎勵（接近一個后）/ Unstoppable passed pawn bonus in king-pawn endgame
KING_PAWN_PROXIMITY_FACTOR = np.int32(5)     # 王兵殘局中，國王親近所有兵的加分係數（每格曼哈頓距離）/ King proximity to all pawns factor

# --- Space Evaluation Constants / 空間控制評估常數 ---
# SPACE_THRESHOLD scaled from SF11's 12222: 6400 * (12222 / 16604) ≈ 4710
SPACE_THRESHOLD = np.int32(4700)             # 啟用空間評估的最小非兵子力閥值 / Minimum non-pawn material to enable space eval
SPACE_BONUS_DIVISOR = np.int32(16)           # 空間控制基礎分數除數 / Space bonus divisor
SPACE_SCALE_DIVISOR = np.int32(4)            # 空間控制縮放除數 / Space scale divisor


# =============================================================================
# --- Mobility Constants / 機動性常量 ---
# =============================================================================
# Bonus for piece mobility, calculated as: (move_count - base_moves) * weight.
# This rewards active pieces and penalizes pieces that are blocked or restricted.
# 棋子機動性獎勵，原本的計算方式為：(移動次數 - 基礎移動數) * 權重，現已改為查表。
# 這獎勵活躍的棋子，懲罰被阻擋或受限的棋子。

# --- Knight Mobility (SF11 MobilityBonus × Pawn ratio MG×0.585 EG×0.42) --- max 8 squares
KNIGHT_MOBILITY_BONUS = np.array([
    [-36, -34], [-31, -23], [ -7, -13], [ -2,  -6],
    [  2,   3], [  8,   6], [ 13,  10], [ 17,  11], [ 20,  14]
], dtype=np.int32)  # index 0..8

# --- Bishop Mobility (SF11 MobilityBonus × Pawn ratio MG×0.585 EG×0.42) --- max 13 squares
BISHOP_MOBILITY_BONUS = np.array([
    [-28, -25], [-12, -10], [  9,  -2], [ 15,   5],
    [ 23,  10], [ 30,  18], [ 32,  23], [ 37,  24],
    [ 37,  27], [ 40,  31], [ 47,  33], [ 47,  36],
    [ 53,  37], [ 57,  41]
], dtype=np.int32)  # index 0..13

# --- Rook Mobility (SF11 MobilityBonus × Pawn ratio MG×0.585 EG×0.42) --- max 14 squares
ROOK_MOBILITY_BONUS = np.array([
    [-34, -32], [-16,  -8], [ -9,  12], [ -6,  23],
    [ -3,  29], [ -2,  35], [  5,  47], [  9,  50],
    [ 17,  56], [ 17,  60], [ 19,  65], [ 23,  69],
    [ 27,  70], [ 28,  71], [ 34,  72]
], dtype=np.int32)  # index 0..14

# --- Queen Mobility (SF11 MobilityBonus × Pawn ratio MG×0.585 EG×0.42) --- max 27 squares
QUEEN_MOBILITY_BONUS = np.array([
    [-23, -15], [-12,  -6], [  2,   3], [  2,   8],
    [  8,  14], [ 13,  23], [ 17,  26], [ 24,  31],
    [ 26,  33], [ 28,  39], [ 33,  40], [ 35,  44],
    [ 35,  47], [ 38,  50], [ 39,  52], [ 41,  53],
    [ 41,  56], [ 43,  57], [ 47,  59], [ 52,  60],
    [ 52,  62], [ 58,  70], [ 60,  71], [ 60,  74],
    [ 62,  77], [ 64,  80], [ 66,  86], [ 68,  89]
], dtype=np.int32)  # index 0..27


# =============================================================================
# --- Piece Coordination Constants / 棋子協同常量 ---
# =============================================================================

# --- Bishop Pair / 雙象優勢 ---
# Bonus for having both bishops. This bonus is generally stronger in open positions.
# 擁有雙象的獎勵。這個獎勵在開放局面中通常更強。
BISHOP_PAIR_BONUS = np.array([20, 30], dtype=np.int32) # MG, EG

# --- Rook on Open/Semi-Open File / 車在開放線/半開放線 ---
# Bonus for a rook on a file with no friendly pawns (semi-open)
# or no pawns at all (open).
# 車在沒有己方兵（半開放線）或完全沒有兵（開放線）的直線上的獎勵。
ROOK_ON_SEMI_OPEN_FILE_BONUS = np.array([15, 10], dtype=np.int32) # MG, EG
ROOK_ON_OPEN_FILE_BONUS = np.array([25, 15], dtype=np.int32) # MG, EG

# Rook on 7th rank (confining king or attacking pawns) bonus / 車在第 7 橫排（限制國王或攻擊兵）的獎勵
ROOK_ON_SEVENTH_BONUS = np.array([20, 50], dtype=np.int32) # MG, EG

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
    [  6,  20], # Rank 2
    [ 10,  23], # Rank 3
    [ 12,  30], # Rank 4
    [ 37,  50], # Rank 5
    [100, 124], # Rank 6
    [166, 182], # Rank 7
    [  0,   0]  # Rank 8
], dtype=np.int32)

# PassedFile adjustment constants (SF11: PassedFile = S(11, 8), scaled by edge_distance)
PASSED_FILE_BONUS = np.array([7, 5], dtype=np.int32) # MG, EG

# --- Passed Pawn Evaluation Refinements / 通路兵動態安全與比例評估參數 ---
PASSED_SCALE_FULL = np.int32(4)          # 材質充足時的完整加成比例 / Full scale when material is sufficient
PASSED_SCALE_HALF = np.int32(2)          # 材質較少時的減半加成比例 / Half scale when material is less
PASSED_SCALE_MIN = np.int32(1)           # 極少材質時的最低加成比例 / Minimum scale when material is very scarce
CANDIDATE_PASSER_DIVISOR = np.int32(2)    # 候選通路兵（非真正通路兵）的加成折半除數 / Divisor for candidate passed pawn bonus

# Path Safety coefficients (SF11 dynamic bonus) / 推進路徑安全性係數
PASSED_PATH_SAFE_NONE_ATTACK = np.int32(21)  # 前進路徑無人控制 / Safe path weight (no enemy attacks)
PASSED_PATH_SAFE_EDGE_ATTACK = np.int32(12)  # 僅路徑邊緣受控 / Safe path weight (attacks on adjacent files only)
PASSED_PATH_SAFE_BLOCK_ONLY = np.int32(5)    # 僅前方阻擋格安全 / Safe path weight (attacks on blocker square only)
PASSED_PATH_SUPPORT_BONUS = np.int32(3)      # 友方車后在後方支援或控制阻擋格的額外安全係數 / Support from behind bonus

# Passed pawn rank-based scaling formula / 通路兵動態加分成長公式參數 (5 * rank - 13)
PASSED_DYNAMICS_MULT = np.int32(5)           # 行動加分排數乘數 / Rank multiplier
PASSED_DYNAMICS_OFFSET = np.int32(-13)       # 行動加分偏置 / Rank offset

# King Proximity in Passed Pawn Endgames / 通路兵殘局國王親近度參數
KING_PROXIMITY_MAX_DIST = np.int32(5)        # 距離計算上限格數 / Cap for Chebyshev distance
KING_PROX_ENEMY_MULT = np.int32(19)          # 敵方王距離權重乘數 / Enemy king distance weight
KING_PROX_ENEMY_DIV = np.int32(4)            # 敵方王距離權重除數 / Enemy king distance divisor (19/4)
KING_PROX_FRIENDLY_MULT = np.int32(2)        # 己方王距離權重乘數 / Friendly king distance weight
KING_PROX_MIN_BONUS = np.int32(-150)         # 國王親近度最小懲罰 / Proximity bonus lower bound
KING_PROX_MAX_BONUS = np.int32(150)          # 國王親近度最大加分 / Proximity bonus upper bound

# --- Isolated Pawns / 孤兵 ---
# Penalty for each isolated pawn on a file.
# 每一個孤兵的懲罰。
ISOLATED_PAWN_PENALTY = np.array([-4, -8], dtype=np.int32) # MG, EG (SF11 S(5, 15) * scale)

# --- Doubled Pawns / 重疊兵 ---
# Penalty for each doubled pawn on a file.
# 每一個重疊兵的懲罰。
DOUBLED_PAWN_PENALTY = np.array([-8, -31], dtype=np.int32) # MG, EG (SF11 S(11, 56) * scale)

# --- Weak Lever & Weak Unopposed / 弱對決兵與開放線弱兵 ---
# Penalty for a weak pawn under pressure (lever) / 弱兵對決時的懲罰
WEAK_LEVER_PENALTY = np.array([0, -31], dtype=np.int32)      # MG, EG (SF11 S(0, 56) * scale)
# Penalty for an unopposed weak pawn on an open/semi-open file / 開放線或半開放線上無敵兵對決的弱兵懲罰
WEAK_UNOPPOSED_PENALTY = np.array([-10, -15], dtype=np.int32) # MG, EG (SF11 S(13, 27) * scale)

# --- Connected Pawn Bonus by Rank (SF11) / SF11 連結兵獎勵（按橫排） ---
# Applied to ALL pawns that are in phalanx (side-by-side) or supported (diagonally behind).
# 適用於所有處於並列或受支撐狀態 of the pawns.
CONNECTED_BONUS = np.array([0, 5, 6, 9, 23, 37, 67, 0], dtype=np.int32) # SF11 S(Connected) * 0.78
CONNECTED_BONUS_EG = np.array([0, 4, 5, 7, 16, 27, 48, 0], dtype=np.int32) # SF11 * 0.563
CONNECTED_SUPPORT_WEIGHT = np.int32(16)  # SF11 21 * 0.78 ≈ 16
CONNECTED_SUPPORT_WEIGHT_EG = np.int32(12)  # SF11 21 * 0.563 ≈ 12

# --- Connected Passed Pawns / 連結通路兵 --- (legacy, kept for reference)
CONNECTED_PASSED_PAWN_BONUS = np.array([15, 35], dtype=np.int32) # MG, EG (not used in evaluation)

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
    [0, 0],    # Rank 1
    [0, 0],    # Rank 2
    [10, 5],   # Rank 3
    [30, 15],  # Rank 4
    [50, 40],  # Rank 5
    [40, 30],  # Rank 6 (Octopus)
    [20, 10],  # Rank 7
    [0, 0]     # Rank 8
], dtype=np.int32)

# 主教前哨獎勵
OUTPOST_BONUS_BISHOP = np.array([
    [0, 0],    # Rank 1
    [0, 0],    # Rank 2
    [10, 5],   # Rank 3
    [20, 15],  # Rank 4
    [30, 25],  # Rank 5
    [20, 15],  # Rank 6
    [10, 5],   # Rank 7
    [0, 0]     # Rank 8
], dtype=np.int32)

# Bonus if the outpost is a "Hole" (cannot be attacked by enemy pawns at all).
# 如果前哨是“洞”（完全無法被敵方兵攻擊），則給予額外獎勵。
OUTPOST_HOLE_BONUS = np.array([25, 15], dtype=np.int32) # MG, EG
REACHABLE_OUTPOST_BONUS = np.array([16, 5], dtype=np.int32) # MG, EG (SF11: S(32, 10) scaled)

# =============================================================================
# --- King Safety Constants (NEW - based on Chessprogramming Wiki) / 王的安全常量 ---
# =============================================================================

# --- Phase 2: Attacking the King Zone (Non-Linear Model) / 攻擊王翼區域（非線性模型） ---
# Attack units for each piece type. Order: P, N, B, R, Q
# 每個棋子類型的攻擊單位。順序：兵、馬、象、車、后
KING_SAFETY_ATTACK_UNITS = np.array([1, 4, 3, 3, 5], dtype=np.int32) # P, N, B, R, Q

# --- kingDanger Linear Formula Weights (Inspired by Stockfish 11) ---
# These contribute to a kingDanger score that is then squared.
# 這些值貢獻到 kingDanger 分數，最後進行二次轉換。

# Weight per weak square in king zone (attacked by enemy, not defended by us except K)
# 王圈內弱格（被敵攻、只被王守 or 不守）每個的 danger 貢獻
KING_DANGER_WEAK_SQ = np.int32(3)

# Weight per pinned piece/blocker in king danger calculation
# 國王防禦者中牽制/阻擋棋子的 danger 貢獻 (SF11 popcount(blockers_for_king(Us)) scaled)
KING_DANGER_BLOCKERS = np.int32(2)

# Flat offset in king danger formula when any danger exists
# 國王遭受攻擊時的固定 offset
KING_DANGER_OFFSET = np.int32(1)

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
SCALING_WEIGHTS = np.array([0, 4, 4, 6, 10], dtype=np.int32) # N, B, R, Q - for scaling factor / 用於縮放因子的權重
MAX_SCALING_MATERIAL = (2*4 + 2*4 + 2*6 + 1*10) # Sum of all weights for one side / 一方所有權重的總和

EG_SAFETY_SCALE = 0.5 # Scale down endgame king safety impact / 縮減殘局王的安全影響

# --- SF11-aligned King Shelter & Storm Tables ---
# ShelterStrength: friendly pawn shield defense values (MG)
SHELTER_STRENGTH = np.array([
    [ -5,  63,  72,  45,  30,  14,  19,   0], # d=0 (A/H file)
    [-33,  47,  27, -38, -22,  -8, -49,   0], # d=1 (B/G file)
    [ -7,  58,  17,  -1,  24,   2, -35,   0], # d=2 (C/F file)
    [-30, -10, -22, -40, -37, -52, -129,  0]  # d=3 (D/E file)
], dtype=np.int32)

# UnblockedStorm: enemy pawn storm threat penalties (MG)
UNBLOCKED_STORM = np.array([
    [ -66,  225,  129,  -75,  -39,  -35,  -39,   0], # d=0
    [ -35,   19,  -95,  -35,  -28,    7,  -15,   0], # d=1
    [   4,  -39, -131,  -26,    1,   17,   10,   0], # d=2
    [  11,    8,  -78,   -3,   -8,   11,   22,   0]  # d=3
], dtype=np.int32)

BLOCKED_STORM = np.int32(-64) # SF11: -82 * 0.78 ≈ -64
BLOCKED_STORM_EG = np.int32(-46) # SF11: -82 * 0.563 ≈ -46
SHELTER_BASE_MG = np.int32(4) # SF11: 5 * 0.78 ≈ 4
SHELTER_BASE_EG = np.int32(3) # SF11: 5 * 0.563 ≈ 3
KING_PAWN_DIST_PENALTY_EG = np.int32(-9) # SF11: -16 * 0.563 ≈ -9

# --- King Danger Calculation Refinement Constants / 王危險度計算精細化常數 ---
KING_FLANK_ATTACK_FACTOR = 3.0       # 翼部攻擊強度加權因子 / Flank attack weight factor
KING_FLANK_ATTACK_DIVISOR = 360.0    # 翼部攻擊強度除數 / Flank attack divisor
KING_SAFETY_MOBILITY_SCALE = 45.0    # 機動性對王安全影響的縮放比例 / Mobility scale for king safety danger
KING_DEFENDER_KNIGHT_BONUS = 2       # 守護馬防禦王鄰格的 danger 扣除 / Knight protector danger bonus
KING_SAFETY_SHIELD_SCALE = 60.0      # 兵盾反饋對王安全危險度的縮放除數 / Shelter shield scale for danger
KING_SAFETY_FLANK_DEF_SCALE = 11.25  # 翼部防禦子力對王安全危險度的縮放除數 / Flank defense scale for danger
KING_DANGER_THRESHOLD = 2            # 王危險度二次方懲罰起步閥值 / Danger quadratic penalty start threshold
KING_DANGER_QUAD_DIVISOR = 2         # 王危險度二次方懲罰除數 / Danger quadratic penalty divisor

# --- Pawnless Flank & Flank Attacks (SF11-inspired) ---
# Penalty when own king flank has no pawns / 己方國王側翼完全無兵時的防禦缺失懲罰（中局與殘局）
PAWNLESS_FLANK = np.array([-13, -53], dtype=np.int32) # MG, EG (SF11 S(17, 95) * scale)
# Penalty factor per flank attack square when king flank is attacked / 國王側翼遭受攻擊時，每格受控格的危險度加成懲罰（中局與殘局）
FLANK_ATTACKS = np.array([-6, 0], dtype=np.int32) # MG, EG (SF11 S(8, 0) * scale)


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
THREAT_SAFE_PAWN = np.array([70, 45], dtype=np.int32)  # SF11: (173, 94)

# KnightOnQueen: Knight attacks squares that attack queen.
# 騎士攻擊能攻擊后的方格的獎勵。
THREAT_KNIGHT_ON_QUEEN = np.array([12, 7], dtype=np.int32)

# SliderOnQueen: Bishop/Rook attacks squares that attack queen.
# 滑動棋子（象/車）攻擊能攻擊后的方格的獎勵。
THREAT_SLIDER_ON_QUEEN = np.array([46, 10], dtype=np.int32)

# ThreatByMinor[target_piece_type]: Minor (N/B) attacks piece of given type.
# Index: 0=Pawn, 1=Knight, 2=Bishop, 3=Rook, 4=Queen, 5=King
# 輕子威脅：馬/象攻擊對應類型棋子的獎勵。
THREAT_BY_MINOR = np.array([
    [ 5, 18],  # vs Pawn    (SF11: 6, 32)
    [46, 23],  # vs Knight  (SF11: 59, 41)
    [46, 23],  # vs Bishop  (SF11: 59, 41)
    [62, 32],  # vs Rook    (SF11: 79, 56)
    [70, 67],  # vs Queen   (SF11: 90, 119)
    [ 0,  0],  # vs King    (should not occur)
], dtype=np.int32)

# ThreatByRook[target_piece_type]: Rook attacks piece of given type (only if weak).
# Index: 0=Pawn, 1=Knight, 2=Bishop, 3=Rook, 4=Queen, 5=King
# 車威脅：車攻擊對應類型棋子的獎勵。
THREAT_BY_ROOK = np.array([
    [ 2, 25],  # vs Pawn    (SF11: 3, 44)
    [30, 40],  # vs Knight  (SF11: 38, 71)
    [30, 34],  # vs Bishop  (SF11: 38, 61)
    [ 0, 21],  # vs Rook    (SF11: 0, 38)
    [40, 21],  # vs Queen   (SF11: 51, 38)
    [ 0,  0],  # vs King    (should not occur)
], dtype=np.int32)

# ThreatByKing: King attacks a weakly defended enemy piece.
# 王攻擊弱子：王攻擊敵方弱子的獎勵。
THREAT_BY_KING = np.array([19, 50], dtype=np.int32)  # SF11: (24, 89)

# Hanging: Enemy piece is attacked + not strongly protected.
# 懸掛子：敵方棋子被攻擊且未被強力保護。
THREAT_HANGING = np.array([54, 20], dtype=np.int32)  # SF11: (69, 36)

# RestrictedPiece: Enemy piece moves are restricted by our attacks.
# 限制棋子行動力：敵方棋子的走子受到我方攻擊限制。
THREAT_RESTRICTED_PIECE = np.array([5, 4], dtype=np.int32)  # SF11: (7, 7)

# ThreatByPawnPush: Pawn push threatens enemy pieces on next move.
# 兵推威脅：兵推進後能威脅敵子。
THREAT_PAWN_PUSH = np.array([37, 22], dtype=np.int32)  # SF11: (48, 39)

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
MINOR_BEHIND_PAWN = np.array([14, 2], dtype=np.int32)  # SF11: S(18, 3) * scale

# BishopPawns: Penalty per own pawn on same color as bishop.
# 壞象懲罰：象同色上的己方兵數量懲罰（含封閉中心加重）。
BISHOP_PAWNS_PENALTY = np.array([2, 4], dtype=np.int32)  # SF11: (3, 7)
BISHOP_PAWNS_CENTER_BLOCKED_FACTOR = np.int32(1)  # 中心鎖死時的壞象懲罰加重因子 / Bad bishop penalty multiplier in blocked center

# TrappedRook: Penalty for rook with mobility <= 3 trapped by own king.
# 困車懲罰：車移動格數 ≤ 3 且在己方王同側。
TRAPPED_ROOK = np.array([41, 6], dtype=np.int32)  # SF11: (52, 10)

# LongDiagonalBishop: Bishop on long diagonal seeing both center squares.
# 長對角線象：象在長對角線上，且能穿過兵阻擋看到兩個中心方格。
LONG_DIAGONAL_BISHOP = np.array([35, 0], dtype=np.int32)

# RookOnQueenFile: Rook on same file as queen (both colors).
# 車后同列：車在與任何一方的后同一列上。
ROOK_ON_QUEEN_FILE = np.array([5, 3], dtype=np.int32)

# WeakQueen: Queen in relative pin or discovered attack line.
# 弱勢后：后處於被牽制或發現攻擊射線上。
WEAK_QUEEN = np.array([38, 8], dtype=np.int32)

# --- Endgame Scale Factors / 殘局縮放因子 ---
SCALE_FACTOR_NORMAL = 64              # Normal scaling (64/64 = 1.0) / 正常殘局縮放因子（無縮減）
SCALE_FACTOR_DRAW = 0                 # Scaling for forced draw positions (0/64 = 0.0) / 強制和棋局面縮放因子
SCALE_FACTOR_OCB_ONE_PAWN = 16        # Opposite-colored bishops endgame with 1 pawn / 異色象殘局且僅有 1 兵時的縮放因子
SCALE_FACTOR_OCB_TWO_PAWNS = 32       # Opposite-colored bishops endgame with 2 pawns / 異色象殘局且有 2 兵時的縮放因子
SCALE_FACTOR_OCB_MULTIPLE_PAWNS = 48  # Opposite-colored bishops endgame with multiple pawns / 異色象殘局且有多兵時的縮放因子
SCALE_FACTOR_KXK = 64                 # King vs King or other basic endgames / 基本國王殘局縮放因子
SCALE_FACTOR_KBNK = 64                # King + Bishop + Knight vs King / 王象馬對單王殘局縮放因子
SCALE_FACTOR_KBPSK_FORTRESS = 8       # King + Bishop + Pawn vs King (fortress) / 王象兵對單王堡壘局面縮放因子
SCALE_FACTOR_KRPKR_FORTRESS = 8       # King + Rook + Pawn vs King + Rook (fortress) / 王車兵對王車堡壘局面縮放因子
SCALE_FACTOR_KQKR_FORTRESS = 64       # King + Queen vs King + Rook / 王后對王車殘局縮放因子
SCALE_FACTOR_KQKRPs_FORTRESS = 8      # King + Queen vs King + Rook + Pawns (fortress) / 王后對王車兵堡壘局面縮放因子

# =============================================================================
# --- Material Imbalance Polynomial Matrix (SF11 material.cpp) ---
# --- 材質不平衡多項式矩陣 ---
# =============================================================================
# SF11 uses a quadratic polynomial to evaluate material imbalance beyond simple
# piece values. The index order is: [BishopPair, Pawn, Knight, Bishop, Rook, Queen].
# This models cross-piece-type interactions (e.g., knights are worth more with many pawns,
# rooks are worth less when queens are present on the same side, etc.).
#
# Reference: stockfish_11/src/material.cpp lines 33-53
# 公式: for pt1=0..5: bonus += count[Us][pt1] * Σ(Ours[pt1][pt2]*count[Us][pt2] + Theirs[pt1][pt2]*count[Them][pt2])
# 最終不平衡 = (white_bonus - black_bonus) / 16

IMBALANCE_QUADRATIC_OURS = np.array([
    #  BP   Pawn  Knight Bishop  Rook  Queen
    [1438,    0,    0,    0,    0,    0],  # Bishop pair
    [  40,   38,    0,    0,    0,    0],  # Pawn
    [  32,  255,  -62,    0,    0,    0],  # Knight
    [   0,  104,    4,    0,    0,    0],  # Bishop
    [ -26,   -2,   47,  105, -208,    0],  # Rook
    [-189,   24,  117,  133, -134,   -6],  # Queen
], dtype=np.int32)

IMBALANCE_QUADRATIC_THEIRS = np.array([
    #  BP   Pawn  Knight Bishop  Rook  Queen
    [   0,    0,    0,    0,    0,    0],  # Bishop pair
    [  36,    0,    0,    0,    0,    0],  # Pawn
    [   9,   63,    0,    0,    0,    0],  # Knight
    [  59,   65,   42,    0,    0,    0],  # Bishop
    [  46,   39,   24,  -24,    0,    0],  # Rook
    [  97,  100,  -42,  137,  268,    0],  # Queen
], dtype=np.int32)

IMBALANCE_DIVISOR = np.int32(16)  # SF11 divides raw imbalance by 16

# Scaling from SF11's internal centipawn scale (PawnMg=128) to our scale (PawnMg=100)
IMBALANCE_SCALE_MG = np.int32(78)   # 100/128 ≈ 0.78, stored as percentage for integer math
IMBALANCE_SCALE_EG = np.int32(56)   # 120/213 ≈ 0.563, stored as percentage for integer math

# =============================================================================
# --- Search Constants / 搜尋常量 ---
# =============================================================================

INFINITY = 32000
MATE_SCORE = 30000
MAX_PLY = 128
MATE_IN_MAX_PLY = MATE_SCORE - MAX_PLY
VALUE_KNOWN_WIN = 10000
NO_MOVE = np.uint16(0)

# Move Ordering Bonuses
SCORE_TT_MOVE = 100000
SCORE_GOOD_CAPTURE_BONUS = 20000
SCORE_KILLER_1 = 15000
SCORE_KILLER_2 = 14000
SCORE_COUNTER_MOVE = 10000
SCORE_BAD_CAPTURE_PENALTY = -5000
GOOD_QUIET_THRESHOLD = -1200

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
HISTORY_MAX_MAIN = 16384
HISTORY_MAX_BUTTERFLY = 16384
HISTORY_MAX_CAPTURE = 16384
HISTORY_MAX_CONTINUATION = 16384
HISTORY_MAX_PAWN = 8192

PAWN_HISTORY_SIZE = 8192
PAWN_HISTORY_MASK = 8191

# 50-move rule constants
FIFTY_MOVE_RULE_LIMIT = 100
FIFTY_MOVE_SCALE_THRESHOLD = 80
FIFTY_MOVE_MAX_SCALE = 256

# History tuning in Search
SEE_HISTORY_DIVISOR = 512
LMR_HISTORY_DIVISOR = 10240

# History Score Weighting in LMR (must be int)
HISTORY_WEIGHT_MAIN = 2
HISTORY_WEIGHT_CONT_1 = 4
HISTORY_WEIGHT_CONT_2 = 2
HISTORY_WEIGHT_CONT_3 = 1
HISTORY_WEIGHT_CONT_4 = 2

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
MIN_SINGULAR_DEPTH = 7
SINGULAR_MARGIN_MULTIPLIER = 2          # Margin = 2 * depth
SINGULAR_DOUBLE_EXT_MULTIPLIER = 2     # Double margin = 2 * depth (total offset 4 * depth for double ext check)

# NEW: Pruning Switches (based on Stockfish Analysis)
ENABLE_SHALLOW_SEE_PRUNING = True  # Enable SEE pruning for captures/quiets at shallow depth
ENABLE_HISTORY_PRUNING = True      # Enable pruning based on History Score

# NEW: Pruning Parameters (Tightened for Performance/Strength Balance)
PRUNING_SHALLOW_DEPTH = 12         # Prune moves only if depth is below this (was 8)
PRUNING_CAPTURE_SEE_MARGIN = -100 # Stockfish dynamic margin: -100 * depth
PRUNING_QUIET_SEE_MARGIN = -25    # Stockfish dynamic margin: -25 * depth^2
PRUNING_HISTORY_THRESHOLD = -4000 # Adjusted from -1000 to match V2 linear history scale

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
RFP_MAX_DEPTH = 12           # V4: Extended from 9 to 12 based on parameter analysis
RFP_BASE_MULT = 170           # V3: Reduced from 180 to 170 for deeper RFP reach
RFP_NO_TT_PENALTY = 40       # Extra margin multiplier if TT info is missing

NMP_STATIC_MARGIN = 150
NMP_MIN_SIDE_NON_PAWNS = 2
NMP_VERIFICATION_DEPTH = 8
LOW_MATERIAL_PRUNING_PIECE_COUNT = 7

# Late Move Reductions (LMR)
LMR_MIN_DEPTH = 3           # Minimum depth to apply LMR (H4: lowered from 4 to match Stockfish)
LMR_MIN_QUIET_MOVE_INDEX = 3 # Minimum number of quiet moves before LMR (Lowered to 2 for aggressive LMR)
LMR_REDUCTION = 1           # Depth reduction for LMR / LMR 的深度減少值

# Late Move Pruning (LMP) - Prune moves after a certain number of quiet moves have been searched
# 晚期移動剪枝（LMP） - 在搜尋了一定數量的寧靜步後剪枝
LMP_MOVE_COUNT = np.array([
    0 if d == 0 else 3 + 2 * d * d for d in range(MAX_PLY)
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

# --- 1024-scale LMR (Phase B) ---
REDUCTIONS = np.zeros(256, dtype=np.int32)
for i in range(1, 256):
    REDUCTIONS[i] = int(20.0 * math.log(i))

LMR_BASE_OFFSET = 512          # r += 512
LMR_TTPV_INCREASE = 768        # r += 768 (ttPv nodes)
LMR_TTPV_DECREASE_BASE = 2048  # r -= 2048 (ttPv decrease)
LMR_TTPV_PV_BONUS = 768        # r -= PvNode * 768
LMR_CUTNODE_BONUS = 1024       # r += 1024 * cutNode
LMR_TTCAPTURE_BONUS = 768      # r += 768 * ttCapture
LMR_MOVECOUNT_FACTOR = 40      # r -= moveCount * 40
LMR_HISTORY_SCALE = 150        # r -= statScore * 150 / 4096
LMR_CUTOFF_CNT_BASE = 128      # r += 128
LMR_CUTOFF_CNT_EXTRA = 512     # r += 512 * (cutoffCnt > 2)
LMR_ALLNODE_EXTRA = 512        # r += 512 * allNode
LMR_TTMOVE_REDUCTION = 1024    # r -= 1024 for ttMove
LMR_NO_TTMOVE_BONUS = 512      # r += 512 * !ttMove (cutNode)
LMR_CORRECTION_DIVISOR = 32768 # r -= abs(correctionValue) / 32768

LMR_ALLNODE_SCALE_NUM = 150    # r += r * 150 / (256*depth + 285)
LMR_ALLNODE_SCALE_DENOM_BASE = 256
LMR_ALLNODE_SCALE_DENOM_OFFSET = 285


# ProbCut
PROBCUT_R = 4
PROBCUT_MARGIN = 150 # centipawns

# Delta Pruning
DELTA_PRUNING_MARGIN = 400

# --- Static Exchange Evaluation (SEE) Threshold / SEE 閾值 ---
SEE_THRESHOLD = 0  # centipawns (H6: tightened from -200, only search non-losing captures in QSearch)
QS_SEE_THRESHOLD = -80  # Stockfish-style qsearch tolerance: keep slightly losing tactical captures
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

# --- Backward Pawns / 後兵 ---
# Penalty for a backward pawn.
# 後兵的懲罰。
# Negative values, applied with += (consistent with ISOLATED_PAWN_PENALTY and DOUBLED_PAWN_PENALTY)
BACKWARD_PAWN_PENALTY = np.array([-7, -13], dtype=np.int32) # MG, EG (SF11 S(9, 24) * scale)

# File constants
NOT_A_FILE = ~np.uint64(0x0101010101010101)
NOT_H_FILE = ~np.uint64(0x8080808080808080)
QUEEN_SIDE_BB = np.uint64(0x0F0F0F0F0F0F0F0F)
KING_SIDE_BB = np.uint64(0xF0F0F0F0F0F0F0F0)

# Exact King Flank bitboards by king file (0-7), matching Stockfish 11
KING_FLANKS = np.array([
    np.uint64(0x0707070707070707), # Files A, B, C
    np.uint64(0x0F0F0F0F0F0F0F0F), # Files A, B, C, D
    np.uint64(0x0F0F0F0F0F0F0F0F), # Files A, B, C, D
    np.uint64(0x3C3C3C3C3C3C3C3C), # Files C, D, E, F
    np.uint64(0x3C3C3C3C3C3C3C3C), # Files C, D, E, F
    np.uint64(0xF0F0F0F0F0F0F0F0), # Files E, F, G, H
    np.uint64(0xF0F0F0F0F0F0F0F0), # Files E, F, G, H
    np.uint64(0xE0E0E0E0E0E0E0E0), # Files F, G, H
], dtype=np.uint64)



