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

# --- Initiative Bonus / 主動權獎勵 ---
# A small bonus awarded to the side to move, acknowledging the advantage of having the turn.
# This bonus is tapered and disappears in the endgame.
# 給予輪到行棋一方的小獎勵，承認擁有下棋權的優勢。此獎勵是漸進的，在殘局中會消失。
INITIATIVE_BONUS = 10 # centipawns
INITIATIVE_PHASE_THRESHOLD = MAX_PHASE * 0.4 # Apply only when phase is above 40% of max / 僅在階段值高於最大值的 40% 時應用


# =============================================================================
# --- Mobility Constants / 機動性常量 ---
# =============================================================================
# Non-linear lookup tables for piece mobility bonuses.
# Derived from Stockfish 11, scaled by ~0.78 (our pawn=100cp vs SF's ~128cp).
# Index = number of safe squares attacked. Columns = [MG, EG].
# 基於 Stockfish 11 的非線性機動性查表。索引 = 安全攻擊格數，列 = [MG, EG]。
# 這比線性公式更準確：被困棋子受嚴厲懲罰，高機動性遞減邊際效益。

# --- Knight Mobility (9 entries, 0-8 squares) ---
KNIGHT_MOBILITY_BONUS = np.array([
    [-48, -63], [-41, -44], [-9, -23], [-3, -11], [2, 6],
    [10, 12], [17, 18], [22, 21], [26, 26]
], dtype=np.int32)

# --- Bishop Mobility (14 entries, 0-13 squares) ---
BISHOP_MOBILITY_BONUS = np.array([
    [-37, -46], [-16, -18], [12, -2], [20, 10], [30, 19],
    [40, 33], [43, 42], [49, 44], [49, 51], [53, 57],
    [63, 61], [63, 67], [71, 69], [76, 76]
], dtype=np.int32)

# --- Rook Mobility (15 entries, 0-14 squares) ---
ROOK_MOBILITY_BONUS = np.array([
    [-45, -59], [-21, -14], [-12, 22], [-8, 43], [-4, 54],
    [-2, 64], [7, 87], [12, 92], [23, 103], [23, 111],
    [25, 121], [30, 129], [36, 130], [37, 132], [45, 133]
], dtype=np.int32)

# --- Queen Mobility (28 entries, 0-27 squares) ---
QUEEN_MOBILITY_BONUS = np.array([
    [-30, -28], [-16, -12], [2, 6], [2, 14], [11, 27],
    [17, 42], [22, 48], [32, 57], [34, 62], [37, 72],
    [44, 73], [47, 81], [47, 88], [51, 94], [52, 96],
    [55, 98], [55, 104], [57, 106], [62, 109], [69, 112],
    [69, 115], [77, 130], [80, 133], [80, 137], [83, 144],
    [85, 149], [88, 161], [90, 165]
], dtype=np.int32)


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

ROOK_ON_SEVENTH_BONUS = np.array([20, 50], dtype=np.int32) # MG, EG

# --- Trapped Rook / 受困車 ---
# Penalty when a rook has <= 3 mobility and is trapped by its own king on the same side.
# Doubled if the king has no castling rights. (SF11: S(52,10) → scaled)
# 受困車懲罰：車機動性<=3且被國王困在同一側。國王無法易位時加倍。
TRAPPED_ROOK_PENALTY = np.array([40, 8], dtype=np.int32) # MG, EG

# --- Minor Behind Pawn / 輕子在兵後方 ---
# Bonus for a knight or bishop standing directly behind a pawn (protected).
# (SF11: S(18,3) → scaled)
MINOR_BEHIND_PAWN_BONUS = np.array([14, 2], dtype=np.int32) # MG, EG

# --- Bad Bishop / 壞象懲罰 ---
# Penalty for each friendly pawn on the same color square as the bishop.
# Worse when center files are blocked. (SF11: S(3,7) → scaled)
BISHOP_PAWNS_PENALTY = np.array([2, 5], dtype=np.int32) # MG, EG per pawn

# --- Space Evaluation / 空間評估 ---
# Only computed when total non-pawn material > threshold.
# Measures safe squares in center files (D,E,F,G) on ranks 2-4 for pieces.
SPACE_THRESHOLD = 12222  # Minimum non-pawn material to compute space

# --- Passed Pawn File Correction / 通路兵邊線修正 ---
# Penalty for passed pawns far from center (edge pawns are less valuable).
PASSED_FILE_PENALTY = np.array([9, 6], dtype=np.int32) # MG, EG per file distance from center

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
    [ 0,  0], # Rank 2
    [ 10,  20], # Rank 3
    [ 30, 50], # Rank 4
    [ 50, 80], # Rank 5
    [ 80, 150], # Rank 6
    [150, 250], # Rank 7
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
CONNECTED_PASSED_PAWN_BONUS = np.array([15, 35], dtype=np.int32) # MG, EG

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

# =============================================================================
# --- King Safety Constants (Enhanced - based on Stockfish 11) / 王的安全常量 ---
# =============================================================================

# --- Safe Check Penalties (Added to kingDanger when enemy can safely check king) ---
# These are the highest-impact king safety parameters in Stockfish.
# Scaled from SF11 by ~0.78 for our value system (100cp pawn vs SF's ~128cp).
# 安全將軍懲罰：當敵方能在安全的格子上將軍時加入 kingDanger。
QUEEN_SAFE_CHECK  = 608   # SF11: 780
ROOK_SAFE_CHECK   = 842   # SF11: 1080
BISHOP_SAFE_CHECK = 495   # SF11: 635
KNIGHT_SAFE_CHECK = 616   # SF11: 790

# --- KingAttackWeights: Weight per attacking piece type (SF11 × 0.78) ---
# Higher = more dangerous attacker. Accumulated as attackers_count * weight.
# 每種攻擊棋子的權重。
KING_ATTACK_WEIGHTS = np.array([0, 63, 41, 34, 8], dtype=np.int32)  # P, N, B, R, Q

# --- KingDanger Formula Parameters (SF11-inspired, scaled) ---
KING_DANGER_WEAK_SQUARE = 144   # SF11: 185, scaled
KING_DANGER_UNSAFE_CHECK = 115  # SF11: 148, scaled
KING_DANGER_ATTACK_COUNT = 54   # SF11: 69, scaled
KING_DANGER_NO_QUEEN = 680      # SF11: 873, scaled
KING_DANGER_KNIGHT_DEFENSE = 78 # SF11: 100, scaled
KING_DANGER_OFFSET = 29         # SF11: 37, scaled
KING_DANGER_THRESHOLD = 100     # Below this, no penalty applied (SF11 threshold)

# --- Old parameters retained for pawn shield / pawn storm ---
KING_SAFETY_WEAK_UNITS = np.array([0, 1, 1, 2, 3], dtype=np.int32) # P, N, B, R, Q (retained for compatibility)

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
PAWN_STORM_PENALTY_BY_RANK = np.array([0, 120, 80, 50, 30, 10, 5, 0], dtype=np.int32)

SCALING_WEIGHTS = np.array([0, 4, 4, 6, 10], dtype=np.int32) # N, B, R, Q - for scaling factor / 用於縮放因子的權重
MAX_SCALING_MATERIAL = (2*4 + 2*4 + 2*6 + 1*10) # Sum of all weights for one side / 一方所有權重的總和

PAWN_SHIELD_MISSING_PENALTY = 15
PAWN_SHIELD_INTACT_BONUS = 10
PAWN_SHIELD_ADVANCED_BONUS = 5
PAWN_SHIELD_PUSHED_PENALTY = 10
KING_OPEN_FILE_PENALTY = 10
KING_SEMI_OPEN_FILE_PENALTY = 5
KING_SAFETY_WEAK_SQUARE_PENALTY = 20

EG_SAFETY_SCALE = 0.3 # Scale down endgame king safety impact / 縮減殘局王的安全影響

# =============================================================================
# --- Threat Evaluation Constants / 威脅評估常量 ---
# =============================================================================
# Calibrated from Stockfish 11, scaled to our value system (~0.78x).
# 基於 Stockfish 11 校準，按比例縮放到我們的材質系統。

# ThreatByMinor[attacked_piece_type]: Bonus when a Knight or Bishop attacks
# an enemy piece of the given type. Index: 0=None, 1=Pawn, 2=Knight, 3=Bishop, 4=Rook, 5=Queen
# 輕子（馬/象）攻擊敵方棋子的獎勵，按被攻擊棋子類型細分
THREAT_BY_MINOR_MG = np.array([0, 5, 46, 62, 70, 62], dtype=np.int32)
THREAT_BY_MINOR_EG = np.array([0, 25, 32, 44, 93, 126], dtype=np.int32)

# ThreatByRook[attacked_piece_type]: Bonus when a Rook attacks an enemy piece.
# 車攻擊敵方棋子的獎勵，按被攻擊棋子類型細分
THREAT_BY_ROOK_MG = np.array([0, 2, 30, 30, 0, 40], dtype=np.int32)
THREAT_BY_ROOK_EG = np.array([0, 34, 55, 48, 30, 30], dtype=np.int32)

# Threat By Safe Pawn: Friendly pawn attacks enemy non-pawn piece.
# 安全兵威脅：己方兵攻擊敵方非兵棋子。(SF11: S(173,94) → scaled)
THREAT_SAFE_PAWN = np.array([135, 73], dtype=np.int32) # MG, EG

# Hanging Pieces: Enemy piece is attacked and undefended.
# 懸掛子：敵方棋子被攻擊且未被防守。(SF11: S(69,36) → scaled)
THREAT_HANGING = np.array([54, 28], dtype=np.int32) # MG, EG

# Threat By Pawn Push: Safe pawn push threatens enemy non-pawn piece.
# 兵推進威脅：安全的兵推進威脅敵方非兵子。(SF11: S(48,39) → scaled)
THREAT_BY_PAWN_PUSH = np.array([37, 30], dtype=np.int32) # MG, EG

# Restricted Piece: Enemy piece movement is restricted (attacked by us, not strongly protected).
# 限制棋子：敵方棋子的移動受限。(SF11: S(7,7) → scaled)
THREAT_RESTRICTED_PIECE = np.array([5, 5], dtype=np.int32) # MG, EG

# Threat By King: King attacks weak enemy piece.
# 王攻擊弱子。(SF11: S(24,89) → scaled)
THREAT_BY_KING = np.array([19, 69], dtype=np.int32) # MG, EG


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

# History Heuristic Constants
MAX_HISTORY = 16384 # Max value for history table to prevent overflow and saturation
CORRECTION_HISTORY_SIZE = 16384
CORRECTION_HISTORY_LIMIT = 400
CORRECTION_HISTORY_GRAVITY = 16
CONTINUATION_HISTORY_FACTOR = 4

# Special value to indicate that the search was stopped due to timeout
# 特殊值，表示搜尋因超時而停止
STOP_SEARCH_FLAG = 66666

PAWN_KEY_INDEX = 5

# --- Aspiration Windows / 期望窗口 ---
ASPIRATION_WINDOW_SIZE = 25 # centipawns (H8: tightened from 100 to detect score instability)

# --- Pruning Techniques / 剪枝技術 ---
# Master switches for new pruning techniques / 新剪枝技術的總開關
ENABLE_LMP = True           # Late Move Pruning
ENABLE_PROBCUT = True       # ProbCut
ENABLE_DELTA_PRUNING = True # Delta Pruning in Quiescence Search
ENABLE_IID = True           # Internal Iterative Deepening
ENABLE_SINGULAR_EXTENSIONS = True # Singular Extensions
ENABLE_MATE_DISTANCE_PRUNING = True # Mate Distance Pruning


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
FP_MARGIN_D1 = 350 # Tightened from 400
FP_MARGIN_D2 = 650 # Tightened from 700
FP_BASE = 150      # Reduced base
FP_MULTIPLIER = 180 # Reduced multiplier

# Reverse Futility Pruning
RFP_MARGIN_D1 = 200 # Tightened from 250

NMP_STATIC_MARGIN = 150

# Late Move Reductions (LMR)
LMR_MIN_DEPTH = 4           # Minimum depth to apply LMR (H4: lowered from 4 to match Stockfish)
LMR_MIN_QUIET_MOVE_INDEX = 4 # Minimum number of quiet moves before LMR (H4: lowered from 4)
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
PROBCUT_R = 2
PROBCUT_R_PRIME = 4
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

# --- Backward Pawns / 後兵 ---
# Penalty for a backward pawn.
# 後兵的懲罰。
BACKWARD_PAWN_PENALTY = np.array([10, 25], dtype=np.int32) # MG, EG

# File constants
NOT_A_FILE = ~np.uint64(0x0101010101010101)
NOT_H_FILE = ~np.uint64(0x8080808080808080)

