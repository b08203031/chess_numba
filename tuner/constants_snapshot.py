# AUTO-GENERATED snapshot of chess_engine/classical/constants.py for HCE SPSA.
# Do not edit by hand. Regenerate with: python -m tuner.sync_constants_snapshot
# Source of truth remains classical/constants.py

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

# 中局（Middle Game）材質值 / 殘局（End Game）材質值
# Pawn: keep ~100cp human anchor (MG=100, EG=120).
# N/B/R/Q: SF11 display scale = internal * 100 / PawnValueEg(213)
#   SF types.h → N 367/401, B 387/430, R 599/648, Q 1192/1259.
# Passed / path / prox use the same *100/213 so pass/N matches SF relative strength.
# Index: PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING
MG_MATERIAL_VALUES = np.array([100, 347, 364, 572, 1170, 0], dtype=np.int32)
EG_MATERIAL_VALUES = np.array([117, 422, 437, 690, 1284, 0], dtype=np.int32)
# =============================================================================
# --- Game Phase Calculation (SF11 npm taper) / 遊戲階段計算 ---
# =============================================================================
# SF11: phase = ((clamp(npm, EndgameLimit, MidgameLimit) - EndgameLimit) * 128)
#              / (MidgameLimit - EndgameLimit)
# Keep the phase scale coupled to the tuned MG N/B/R/Q values:
#   MidgameLimit = full two-side starting non-pawn material
#   EndgameLimit = MidgameLimit * SF11(3915 / 15258), rounded
# This preserves phase(startpos)=128 when Texel/SPSA changes material values.
PHASE_MIDGAME = np.int32(128)
MAX_PHASE = PHASE_MIDGAME  # alias used by tapered blend
PHASE_ENDGAME_RATIO_NUM = np.int32(3915)
PHASE_ENDGAME_RATIO_DEN = np.int32(15258)
MIDGAME_LIMIT = np.int32(2 * (
    2 * int(MG_MATERIAL_VALUES[KNIGHT])
    + 2 * int(MG_MATERIAL_VALUES[BISHOP])
    + 2 * int(MG_MATERIAL_VALUES[ROOK])
    + int(MG_MATERIAL_VALUES[QUEEN])
))
ENDGAME_LIMIT = np.int32(
    (int(MIDGAME_LIMIT) * int(PHASE_ENDGAME_RATIO_NUM)
     + int(PHASE_ENDGAME_RATIO_DEN) // 2)
    // int(PHASE_ENDGAME_RATIO_DEN)
)
# Legacy piece-count weights (kept for reference / any external tools)
PHASE_WEIGHTS = np.array([0, 1, 1, 2, 4, 0], dtype=np.int32)

# =============================================================================
# --- Piece-Square Tables (PSTs) / 棋子位置分數表 ---
# =============================================================================
# White's perspective; Black uses sq ^ 56. Flat 8x8, index 0=A1, 63=H8.
# Shape (6, 64): PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING.
# Single source of truth for evaluation (no per-piece PAWN_PST_* tables).
# 白方視角；黑方 sq^56。形狀 (6,64)。評估唯一 PST 來源。

PST_MG = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, -9, -10, -7, -6, -7, -8, 10, 4, -14, -23, -18, -11, -11, -5, -12, -6, -6, -16, -5, 6, 9, 0, 11, -11, 7, -1, -10, 16, 19, 10, 1, 11, 9, 11, 17, 27, 32, 16, 9, 16, 51, 49, 51, 49, 53, 45, 50, 49, 0, 0, 0, 0, 0, 0, 0, 0],
    [-45, -33, -33, -31, -24, -31, -41, -47, -42, -22, -7, -10, -7, 2, -19, -34, -27, -4, -2, 6, 12, 6, 8, -26, -11, 5, 14, 13, 12, 18, -4, -25, -27, 5, 12, 19, 11, 18, 9, -26, -25, 4, 12, 20, 13, 4, 0, -18, -48, -25, -3, 0, -2, -3, -20, -44, -38, -44, -40, -25, -32, -29, -46, -48],
    [-12, 6, -13, -10, -8, -18, -9, -18, -7, -10, 8, -14, -1, 4, 6, -13, -2, 14, -4, 8, 4, -3, 11, -3, -8, -2, 5, 11, 11, 8, -3, -5, -11, 3, 0, 8, 10, 8, 2, -6, -6, 10, 8, 13, 9, 1, -2, -6, -5, 1, 5, -3, 2, -2, -5, -8, -26, -8, -12, -6, -14, 0, -6, -20],
    [-11, 0, 0, 6, 3, -8, -3, 7, -5, 1, -4, 1, 1, 0, -1, -9, -7, -2, 0, -1, 4, -2, 0, -9, -6, 0, 5, 0, -1, 6, -4, -13, -8, 7, 2, 4, 1, 2, 2, -12, -3, 5, -1, -5, 5, -3, 2, -5, 7, 13, 10, 12, 17, 9, 12, 12, 5, -2, 1, 1, -2, -4, 3, -1],
    [-13, -8, -5, -10, -7, -13, -8, -22, -8, 0, 0, -1, -1, 6, 1, -6, -8, 4, 0, -6, 0, 12, 12, -1, 7, 0, 5, -5, 0, 6, 11, -4, -4, 2, 10, -3, 4, 2, 2, 4, -7, -2, 7, 5, 7, 5, -4, -6, -6, -3, 6, 2, -3, 3, 3, -9, -17, -10, -6, -7, -4, -15, -11, -20],
    [18, 35, 19, 2, 19, 9, 26, 18, 20, 24, -2, 0, -12, -8, 22, 18, -5, -17, -23, -20, -16, -24, -21, -10, -16, -30, -29, -44, -42, -36, -23, -17, -26, -41, -42, -51, -51, -38, -36, -31, -24, -34, -41, -54, -48, -38, -43, -27, -31, -37, -41, -50, -51, -40, -41, -24, -25, -45, -39, -52, -50, -37, -36, -31],
], dtype=np.int32)
PST_EG = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 11, 12, 15, 12, 25, 20, 10, -1, 9, 9, 10, 15, 11, 12, 6, 1, 10, 9, 6, 3, 4, 2, 4, 5, 26, 16, 14, 17, 8, 12, 17, 13, 48, 42, 43, 54, 50, 39, 43, 46, 75, 73, 85, 76, 80, 87, 74, 74, 0, 0, 0, 0, 0, 0, 0, 0],
    [-49, -36, -33, -28, -26, -34, -35, -50, -38, -21, -10, -4, -2, 1, -21, -37, -30, -6, 2, 11, 14, -5, -3, -28, -16, -2, 16, 15, 17, 16, 7, -21, -27, -2, 15, 20, 24, 17, 4, -23, -30, 1, 9, 14, 13, 7, 5, -39, -37, -16, -7, 14, 6, -6, -23, -38, -49, -43, -33, -31, -26, -27, -37, -56],
    [-35, -29, -22, -15, -18, -16, -35, -27, -20, -24, -9, -8, -3, -6, -22, -21, -17, 2, 0, 2, 2, -3, -2, -20, -17, -1, 4, 10, 10, 3, 2, -17, -18, 2, 13, 15, 6, 18, -3, -9, -18, 0, 4, 3, 3, 2, -8, 0, -29, -8, -17, 6, 4, -7, -17, -31, -44, -20, -31, -11, -12, -27, -25, -42],
    [-10, -17, -12, -13, -15, -8, -17, -19, -15, -10, -4, -4, -4, -3, -9, -15, 2, 1, -1, 1, -1, -6, 5, 4, -2, 5, 14, 2, -1, 8, 0, -5, -1, 6, 2, 10, 8, -1, -1, -4, 12, -2, 6, 12, 11, 0, -9, 9, 2, 2, 16, 1, 1, 21, 6, 7, 10, 5, 19, 16, 10, 19, 5, 18],
    [-65, -55, -47, -33, -34, -45, -54, -70, -56, -34, -22, -11, -13, -28, -30, -61, -42, -17, -12, -7, -4, -12, -21, -41, -32, -7, -1, 12, 18, 4, -6, -26, -27, 2, 13, 25, 24, 14, -2, -20, -44, -15, -8, 0, 4, -7, -21, -38, -50, -31, -19, -6, 0, -19, -26, -54, -75, -54, -43, -35, -38, -42, -44, -77],
    [-50, -25, -23, -28, -29, -28, -29, -52, -30, -24, -9, -8, -11, -12, -17, -31, -33, -12, 13, 9, 7, 0, -4, -28, -27, -7, 25, 30, 22, 17, -3, -26, -27, -7, 33, 38, 40, 33, -4, -25, -24, -7, 22, 33, 28, 20, -8, -24, -28, -9, -3, -2, -4, -7, -14, -30, -55, -38, -30, -21, -16, -27, -35, -48],
], dtype=np.int32)
# =============================================================================
# --- Other Evaluation Constants / 其他評估常量 ---
# =============================================================================

# --- Initiative / Complexity Constants (SF11-inspired) ---
# Dynamic complexity-based score dampening, prevents draw-drift in simple endgames.
# 動態複雜度主動權修正，防止簡單殘局中的和棋漂移。

# Middlegame (MG) weights - scaled by Pawn ratio MG = 100/128 ≈ 0.78 / 中局主動權/複雜度係數
INITIATIVE_PASSED_WEIGHT_MG = 7
INITIATIVE_PAWN_WEIGHT_MG = 10
INITIATIVE_OUTFLANKING_WEIGHT_MG = 7
INITIATIVE_INFILTRATION_WEIGHT_MG = 9
INITIATIVE_BOTH_FLANKS_WEIGHT_MG = 15
INITIATIVE_PAWN_ENDGAME_WEIGHT_MG = 41
INITIATIVE_ALMOST_UNWIN_WEIGHT_MG = 33
INITIATIVE_OFFSET_MG = -78
INITIATIVE_MG_OFFSET = np.int32(39)              # +50 * 0.78 ≈ 39 / 主動權分數修正偏置

# Endgame (EG) weights - scaled by Pawn ratio EG = 120/213 ≈ 0.563 / 殘局主動權/複雜度係數
INITIATIVE_PASSED_WEIGHT_EG = np.int32(5)      # 9  * 0.563 ≈ 5 / 通路兵權重
INITIATIVE_PAWN_WEIGHT_EG = 4
INITIATIVE_OUTFLANKING_WEIGHT_EG = 3
INITIATIVE_INFILTRATION_WEIGHT_EG = 7
INITIATIVE_BOTH_FLANKS_WEIGHT_EG = 15
INITIATIVE_PAWN_ENDGAME_WEIGHT_EG = 28
INITIATIVE_ALMOST_UNWIN_WEIGHT_EG = 24
INITIATIVE_OFFSET_EG = -53
TEMPO_BONUS = 19
LAZY_EVAL_THRESHOLD = np.int32(1100)             # Dynamic evaluation exit threshold / 動態評估提早結束閥值 (SF11 1400 scaled by 100/128)
PINNED_UNCOMPUTED_SENTINEL = np.uint64(18446744073709551615)  # Sentinel when evaluation exits early without computing pins

# --- King-Pawn Endgame Specifics / 王兵殘局特定常數 ---
UNSTOPPABLE_PAWN_BONUS = 800
KING_PAWN_PROXIMITY_FACTOR = 5
SPACE_THRESHOLD = 4701
SPACE_BONUS_DIVISOR = 15
SPACE_SCALE_DIVISOR = 6
# =============================================================================
# --- Mobility Constants / 機動性常量 ---
# =============================================================================
# Bonus for piece mobility, calculated as: (move_count - base_moves) * weight.
# This rewards active pieces and penalizes pieces that are blocked or restricted.
# 棋子機動性獎勵，原本的計算方式為：(移動次數 - 基礎移動數) * 權重，現已改為查表。
# 這獎勵活躍的棋子，懲罰被阻擋或受限的棋子。

# --- Knight Mobility (SF11 MobilityBonus × Pawn ratio MG×0.585 EG×0.42) --- max 8 squares
KNIGHT_MOBILITY_BONUS = np.array([
    [-17, -26],
    [-15, -20],
    [1, -5],
    [3, 11],
    [3, 12],
    [6, 12],
    [10, 13],
    [11, 13],
    [13, 13],
], dtype=np.int32)
BISHOP_MOBILITY_BONUS = np.array([
    [-10, -18],
    [3, -11],
    [19, 3],
    [24, 10],
    [27, 23],
    [29, 29],
    [32, 33],
    [34, 37],
    [34, 37],
    [34, 37],
    [46, 37],
    [49, 39],
    [52, 41],
    [61, 42],
], dtype=np.int32)
ROOK_MOBILITY_BONUS = np.array([
    [-27, -32],
    [-4, -1],
    [10, 22],
    [13, 32],
    [13, 38],
    [15, 45],
    [16, 53],
    [19, 55],
    [24, 59],
    [26, 63],
    [26, 66],
    [27, 67],
    [28, 69],
    [28, 71],
    [33, 72],
], dtype=np.int32)
QUEEN_MOBILITY_BONUS = np.array([
    [-26, -10],
    [-14, -3],
    [5, 6],
    [12, 7],
    [21, 17],
    [25, 23],
    [26, 33],
    [29, 35],
    [32, 37],
    [34, 43],
    [37, 45],
    [38, 46],
    [41, 51],
    [41, 51],
    [42, 54],
    [42, 58],
    [43, 62],
    [44, 66],
    [45, 68],
    [54, 69],
    [57, 69],
    [59, 69],
    [61, 72],
    [63, 74],
    [65, 75],
    [66, 87],
    [74, 90],
    [74, 95],
], dtype=np.int32)
# =============================================================================
# --- Piece Coordination Constants / 棋子協同常量 ---
# =============================================================================

# --- Bishop Pair / 雙象優勢 ---
# Bonus for having both bishops. This bonus is generally stronger in open positions.
# 擁有雙象的獎勵。這個獎勵在開放局面中通常更強。
BISHOP_PAIR_BONUS = np.array([9, 15], dtype=np.int32)
BISHOP_PAIR_PAWN_SCALE = np.array([19, 33], dtype=np.int32)
ROOK_ON_SEMI_OPEN_FILE_BONUS = np.array([20, 3], dtype=np.int32)
ROOK_ON_OPEN_FILE_BONUS = np.array([24, 12], dtype=np.int32)
# =============================================================================
# --- Pawn Structure Constants / 兵型結構常量 ---
# =============================================================================

# --- Passed Pawns / 通路兵 ---
# SF11 PassedRank * 100/PawnValueEg(213) — same display scale as N/B/R/Q material.
# Index = rank (0..7); ranks 1 and 8 unused for pawns.
PASSED_PAWN_BONUS = np.array([
    [0, 0],
    [1, 3],
    [0, 0],
    [5, 22],
    [27, 41],
    [75, 75],
    [132, 109],
    [0, 0],
], dtype=np.int32)
PASSED_FILE_BONUS = np.array([16, 4], dtype=np.int32)
CANDIDATE_PASSER_DIVISOR = 2

# Path Safety (SF11 35/20/9/+5 * 100/213) — was *3/5; now same display scale as material
PASSED_PATH_SAFE_NONE_ATTACK = 11
PASSED_PATH_SAFE_EDGE_ATTACK = 7
PASSED_PATH_SAFE_BLOCK_ONLY = 3
PASSED_PATH_SUPPORT_BONUS = 4
PASSED_DYNAMICS_MULT = 4
PASSED_DYNAMICS_OFFSET = -11
KING_PROXIMITY_MAX_DIST = 6
KING_PROX_ENEMY_MULT = 7
KING_PROX_ENEMY_DIV = 4
KING_PROX_FRIENDLY_MULT = 1
KING_PROX_MIN_BONUS = -74
KING_PROX_MAX_BONUS = 79
ISOLATED_PAWN_PENALTY = np.array([-1, -6], dtype=np.int32)
DOUBLED_PAWN_PENALTY = np.array([-19, -37], dtype=np.int32)
WEAK_LEVER_PENALTY = np.array([-4, -27], dtype=np.int32)
WEAK_UNOPPOSED_PENALTY = np.array([-15, -9], dtype=np.int32)
CONNECTED_BONUS = np.array([0, 1, 4, 4, 8, 38, 69, 0], dtype=np.int32)
CONNECTED_BONUS_EG = np.array([0, 2, 3, 11, 14, 24, 37, 0], dtype=np.int32)
CONNECTED_SUPPORT_WEIGHT = 11
CONNECTED_SUPPORT_WEIGHT_EG = 8
CONNECTED_PASSED_PAWN_BONUS = np.array([15, 35], dtype=np.int32) # MG, EG (legacy)
PASSED_CONNECTED_BONUS = np.array([0, 2, 3, 7, 17, 23, 58, 0], dtype=np.int32)
PASSED_PHALANX_BONUS = np.array([0, 1, 3, 12, 26, 42, 75, 0], dtype=np.int32)
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
    [0, 0],
    [0, 0],
    [16, 4],
    [28, 16],
    [40, 24],
    [35, 24],
    [24, 5],
    [0, 0],
], dtype=np.int32)
OUTPOST_BONUS_BISHOP = np.array([
    [0, 0],
    [0, 0],
    [0, 3],
    [18, 13],
    [24, 17],
    [26, 20],
    [9, 4],
    [0, 0],
], dtype=np.int32)
REACHABLE_OUTPOST_BONUS = np.array([16, 14], dtype=np.int32)
# =============================================================================
# --- King Safety Constants (NEW - based on Chessprogramming Wiki) / 王的安全常量 ---
# =============================================================================

# --- King danger (SF11 structure, OUR ~100cp score space) ---
#
# SF11 builds kingDanger in an intermediate unit space where attack weights
# (81, 1080, 185, ...) were co-tuned with SF internal S() scores (pawn MG=128).
# Our engine uses ~100cp material/mobility/shelter. To avoid mixing units:
#
#   1. Scale ALL pure SF danger coefficients by MG_SCORE_SCALE = 100/128.
#   2. Feed pure mobility MG and shelter MG already in our score units (no lift).
#   3. Convert kd -> penalty with algebra for kd_us = (100/128)*kd_sf:
#        pen_mg = kd² * 128 / (4096 * 100)
#        pen_eg = kd * 120 * 128 / (16 * 213 * 100)
#        threshold = 100 * 100 / 128
#
# Product term: attackersCount * attackersWeight (pawns seed count only, weight 0).
# Indices: NO_PIECE, PAWN, KNIGHT, BISHOP, ROOK, QUEEN  (SF KingAttackWeights)

_MG_SCALE_NUM = 100  # our pawn MG
_MG_SCALE_DEN = 128  # SF pawn MG

def _sf_danger_to_ours(sf_val: int) -> np.int32:
    return np.int32((int(sf_val) * _MG_SCALE_NUM) // _MG_SCALE_DEN)

KING_ATTACK_WEIGHTS = np.array([0, 0, 44, 12, 14, 1], dtype=np.int32)
KING_SAFETY_ATTACK_UNITS = KING_ATTACK_WEIGHTS[1:].copy()  # legacy alias P,N,B,R,Q

# Safe checks (SF raw -> our danger space)
# G4 match-SPSA best (L2 accept iter 19, score 0.526 vs prior; frozen 2026-07-20).
# Source: tournament_analysis/spsa_eval best_params / G4_FROZEN_INT.
SAFE_CHECK_QUEEN = 565
SAFE_CHECK_ROOK = 824
SAFE_CHECK_BISHOP = 477
SAFE_CHECK_KNIGHT = 609
KING_DANGER_WEAK_SQ = 111
KING_DANGER_UNSAFE_CHECK = 99
KING_DANGER_BLOCKERS = 70
KING_DANGER_ATTACK_ON_KING_SQ = 16
KING_DANGER_NO_QUEEN = 560
# Rookless attackers need substantially more no-queen suppression than rook
# attackers.  This single split removes the opposite Texel gradients of rook
# endings and minor-only endings without material-combination special cases.
KING_DANGER_NO_QUEEN_ROOKLESS = 810
KING_DANGER_KNIGHT_DEF = 98
KING_DANGER_OFFSET = 0
KING_DANGER_THRESHOLD = 69
KING_DANGER_QUAD_DIV = np.int32(4096)
KING_DANGER_EG_DIV = np.int32(16)
# Match-SPSA A-block best (L2 iter 74, score 0.534; source spsa_eval_a123/best_params.json).
# Was _MG_SCALE_DEN (=128). Passed/material/passed_rank blocks had no L2 accept.
KING_DANGER_OUT_MG_NUM = np.int32(131)
KING_DANGER_OUT_MG_DEN = np.int32(4096 * _MG_SCALE_NUM)   # 409600
KING_DANGER_OUT_EG_NUM = np.int32(_MG_SCALE_DEN * 120)    # 128*120
KING_DANGER_OUT_EG_DEN = np.int32(_MG_SCALE_NUM * 16 * 213)  # 100*16*213

# Deprecated aliases
KING_DANGER_PINNED = np.int32(0)
KING_DANGER_DIVISOR = np.int32(2)
KING_DANGER_QUAD_DIVISOR = KING_DANGER_QUAD_DIV
KING_DANGER_MG_NUM = KING_DANGER_OUT_MG_NUM
KING_DANGER_MG_DEN = KING_DANGER_OUT_MG_DEN
KING_DANGER_EG_NUM = KING_DANGER_OUT_EG_NUM
KING_DANGER_EG_DEN = KING_DANGER_OUT_EG_DEN

# King tropism removed (was dead code: computed but never added to final score;
# kingDanger already captures approach-to-king via ring attacks / safe checks).
# --- Phase 4: Advanced & Dynamic / 進階與動態 ---
SCALING_WEIGHTS = np.array([0, 4, 4, 6, 10], dtype=np.int32)  # legacy
MAX_SCALING_MATERIAL = (2 * 4 + 2 * 4 + 2 * 6 + 1 * 10)
# Match-SPSA A: weight 96/100 → float scale (was 1.0)
EG_SAFETY_SCALE = 0.96

# --- SF11-aligned King Shelter & Storm Tables ---
# ShelterStrength: friendly pawn shield defense values (MG)
SHELTER_STRENGTH = np.array([
    [11, 49, 56, 51, 32, 21, 23, 0],
    [-35, 23, 22, -30, -25, -3, -48, 0],
    [3, 62, 24, 9, 25, -5, -39, 0],
    [-20, -4, -8, -17, -30, -51, -130, 0],
], dtype=np.int32)
UNBLOCKED_STORM = np.array([
    [-67, 219, 129, -70, -40, -36, -37, 0],
    [-28, 16, -91, -34, -28, -8, -18, 0],
    [14, -35, -129, -20, 1, 16, 27, 0],
    [6, 7, -78, -2, 4, 21, 15, 0],
], dtype=np.int32)
BLOCKED_STORM = -61
BLOCKED_STORM_EG = -41
SHELTER_BASE_MG = 20
SHELTER_BASE_EG = 5
KING_PAWN_DIST_PENALTY_EG = -6
KING_FLANK_ATTACK_NUM = np.int32(3)
KING_FLANK_ATTACK_DEN = np.int32(8)
KING_SHELTER_FEEDBACK_NUM = np.int32(6)
KING_SHELTER_FEEDBACK_DEN = np.int32(8)
KING_FLANK_DEFENSE_MULT = np.int32(4)

# --- Pawnless Flank & Flank Attacks (SF11-inspired) ---
# Match-SPSA A best: was [-14,-51] / [-2,0]
PAWNLESS_FLANK = np.array([-13, -55], dtype=np.int32)
FLANK_ATTACKS = np.array([-1, 0], dtype=np.int32)
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
# SF11 S(173, 94). Full 0.78/0.56 scale (~135,53) regressed in match play → keep damped values.
THREAT_SAFE_PAWN = np.array([69, 41], dtype=np.int32)
THREAT_KNIGHT_ON_QUEEN = np.array([9, 0], dtype=np.int32)
THREAT_SLIDER_ON_QUEEN = np.array([19, 0], dtype=np.int32)
THREAT_BY_MINOR = np.array([
    [4, 16],
    [23, 23],
    [52, 28],
    [73, 56],
    [50, 81],
    [0, 0],
], dtype=np.int32)
THREAT_BY_ROOK = np.array([
    [3, 21],
    [20, 29],
    [27, 25],
    [0, 22],
    [40, 22],
    [0, 0],
], dtype=np.int32)
THREAT_BY_KING = np.array([18, 38], dtype=np.int32)
THREAT_HANGING = np.array([12, 21], dtype=np.int32)
THREAT_RESTRICTED_PIECE = np.array([2, 0], dtype=np.int32)
THREAT_PAWN_PUSH = np.array([11, 12], dtype=np.int32)
THREAT_MINOR_ON_MAJOR = np.array([25, 15], dtype=np.int32)  # kept for reference
# Rook Attacking Queen (replaced by THREAT_BY_ROOK)
THREAT_ROOK_ON_QUEEN = np.array([20, 10], dtype=np.int32)   # kept for reference

# =============================================================================
# --- Per-Piece Bonus/Penalty Constants (SF11-inspired, scaled by Pawn ratio) ---
# =============================================================================

# KingProtector: Penalty for minor piece being far from own king.
# 王保護者：輕子距離己方王越遠，懲罰越重（每格切比雪夫距離）。
# SF11 S(7, 8). P3a full scale (5,4) not kept after neutral/slightly-negative match → (4,3).
KING_PROTECTOR = np.array([0, 0], dtype=np.int32)
MINOR_BEHIND_PAWN = np.array([11, 3], dtype=np.int32)
BISHOP_PAWNS_PENALTY = np.array([1, 2], dtype=np.int32)
BISHOP_PAWNS_CENTER_BLOCKED_FACTOR = 1
TRAPPED_ROOK = np.array([11, 2], dtype=np.int32)
LONG_DIAGONAL_BISHOP = np.array([18, 6], dtype=np.int32)
ROOK_ON_QUEEN_FILE = np.array([0, 2], dtype=np.int32)
WEAK_QUEEN = np.array([28, 1], dtype=np.int32)
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
    [1435, 0, 0, 0, 0, 0],
    [32, 21, 0, 0, 0, 0],
    [32, 249, -62, 0, 0, 0],
    [0, 95, 4, 0, 0, 0],
    [-23, -3, 44, 110, -207, 0],
    [-186, 19, 118, 137, -136, -6],
], dtype=np.int32)
IMBALANCE_QUADRATIC_THEIRS = np.array([
    [0, 0, 0, 0, 0, 0],
    [35, 0, 0, 0, 0, 0],
    [10, 69, 0, 0, 0, 0],
    [58, 72, 49, 0, 0, 0],
    [50, 45, 27, -23, 0, 0],
    [98, 105, -39, 138, 272, 0],
], dtype=np.int32)
IMBALANCE_DIVISOR = 21
IMBALANCE_SCALE_MG = 59
IMBALANCE_SCALE_EG = 52
# =============================================================================
# --- Search Constants / 搜尋常量 ---
# =============================================================================

INFINITY = 32000
MATE_SCORE = 30000
MAX_PLY = 128
MATE_IN_MAX_PLY = MATE_SCORE - MAX_PLY
VALUE_KNOWN_WIN = 10000
NO_MOVE = np.uint16(0)

# --- Search tree efficiency diagnostics (uint64 counters on SearchContext.diag_stats) ---
DIAG_CUT_NODES = 0          # beta cutoffs from move loop (non-root tracked too)
DIAG_CUT_FIRST = 1          # beta cutoff on first legal move tried
DIAG_CUT_MOVE_SUM = 2       # sum of searched_legal_moves at cutoff (avg = sum/cuts)
DIAG_LMR_TRY = 3            # reduced-depth LMR searches launched
DIAG_LMR_RESEARCH = 4       # LMR fail-high re-searches
DIAG_TT_CUT = 5             # TT direct cutoffs
DIAG_NMP_CUT = 6            # null-move cutoffs
DIAG_RFP_CUT = 7            # reverse futility cutoffs
DIAG_RAZOR_CUT = 8          # razoring cutoffs
DIAG_LMP_SKIP = 9           # late-move prunes (quiet skips)
DIAG_FP_SKIP = 10           # futility prunes (quiet skips)
DIAG_PROBCUT = 11           # ProbCut cutoffs
# NMP funnel (tournament_analysis/NMP_EXPERIMENT_PLAN.md)
DIAG_NMP_ELIGIBLE = 12      # hard guards passed; about to evaluate gate
DIAG_NMP_GATE_PASS = 13     # static gate passed
DIAG_NMP_NULL_TRY = 14      # null-move child search launched
DIAG_NMP_NULL_FH = 15       # null score >= beta
DIAG_NMP_VERIFY_TRY = 16    # verification search launched
DIAG_NMP_VERIFY_FAIL = 17   # verification failed (score < beta)
# History infrastructure diagnostics (Phase full history)
DIAG_HIST_QUIET_BONUS = 18  # quiet cutoff bonus applications (best move)
DIAG_HIST_QUIET_MALUS = 19  # quiet malus applications (failed quiets)
DIAG_HIST_CAP_BONUS = 20    # capture history bonus
DIAG_HIST_CAP_MALUS = 21    # capture history malus
DIAG_HIST_CONT_UPD = 22     # continuation gravity writes (any layer)
DIAG_HIST_FAIL_LOW = 23     # fail-low opponent-move feedback events
DIAG_HIST_TT_HIT = 24       # TT quiet fail-high history updates
DIAG_HIST_REVERSE = 25      # reverse-quiet main-history writes
DIAG_HIST_PRUNE = 26        # history-based quiet prunes
DIAG_HIST_AGE = 27          # history aging runs (per ID root entry / iteration)
DIAG_HIST_MAIN_UPD = 28     # main_history gravity writes
DIAG_HIST_PIECE_TO_UPD = 29 # piece-to (history_table) gravity writes
DIAG_HIST_PAWN_UPD = 30     # pawn_history gravity writes
DIAG_HIST_LPH_UPD = 31      # low-ply history gravity writes
# QSearch / SEE / extension funnel. These counters are opt-in at runtime;
# production UCI searches keep diagnostics disabled to avoid hot-path writes.
DIAG_Q_TT_CUT = 32          # qsearch TT direct cutoffs
DIAG_Q_STANDPAT_CUT = 33    # qsearch stand-pat beta cutoffs
DIAG_Q_IN_CHECK = 34        # qsearch nodes entered while in check
DIAG_Q_MOVES = 35           # pseudo-legal qsearch moves generated
DIAG_Q_DELTA_SKIP = 36      # qsearch delta-pruned captures
DIAG_Q_SEE_SKIP = 37        # qsearch SEE-pruned captures
DIAG_Q_ILLEGAL_SKIP = 38    # qsearch pseudo-legal moves rejected as illegal
DIAG_Q_BETA_CUT = 39        # qsearch move-loop beta cutoffs
DIAG_SEE_CAP_SKIP = 40      # main-search capture SEE prunes
DIAG_SEE_QUIET_SKIP = 41    # main-search quiet SEE prunes
DIAG_CHECK_EXT = 42         # check extensions applied
DIAG_SINGULAR_TRY = 43      # singular exclusion searches launched
DIAG_SINGULAR_EXT = 44      # singular extensions applied
DIAG_DOUBLE_EXT = 45        # double singular extensions applied
DIAG_Q_SEE_TRY = 46         # qsearch SEE calls
DIAG_PICKER_SEE_TRY = 47    # main MovePicker good/bad capture SEE calls
DIAG_MAIN_CAP_SEE_TRY = 48  # shallow capture-pruning SEE calls
DIAG_MAIN_QUIET_SEE_TRY = 49 # shallow quiet-pruning SEE calls
DIAG_LMR_CAP_SEE_TRY = 50   # lazy capture-LMR SEE calls
DIAG_PROBCUT_SEE_TRY = 51   # ProbCut SEE filter calls
DIAG_CHECK_GATE_SEE_TRY = 52 # non-PV check-extension SEE calls
DIAG_Q_CHECK_PROTECT = 53  # checking captures rescued from qsearch delta pruning
DIAG_Q_EVASION_PREFILTER = 54 # in-check qsearch evasions rejected before make/unmake
DIAG_MAIN_CHECK_SEE_ROUTE = 55 # quiet checks routed through capture/check SEE margin
# P1 main quiet SEE / legality / check-geometry classification.
DIAG_MAIN_QUIET_SEE_PRECHECK = 56       # quiet SEE candidates classified before SEE
DIAG_MAIN_QUIET_PRE_PIN_ILLEGAL = 57    # pre-SEE pinned move is provably illegal
DIAG_MAIN_QUIET_PRE_KING_ILLEGAL = 58   # pre-SEE king destination is attacked
DIAG_MAIN_QUIET_SEE_PASS = 59           # quiet/checking quiet SEE passed
DIAG_MAIN_QUIET_POST_PIN_ILLEGAL = 60   # SEE-pass candidate rejected by pin test
DIAG_MAIN_QUIET_POST_KING_ILLEGAL = 61  # SEE-pass candidate rejected by king test
DIAG_MAIN_QUIET_FALLBACK_TRY = 62       # SEE-pass candidate requiring make fallback
DIAG_MAIN_QUIET_FALLBACK_REJECT = 63    # fallback candidate rejected after make
DIAG_MAIN_GEOMETRY_NODE = 64            # main node precomputed check geometry
DIAG_MAIN_SCORE_GEOMETRY_REBUILD = 65   # score_quiets rebuilt the same geometry

# P2 aspiration-window diagnostics.  These are kept in the same opt-in
# uint64 buffer as the hot-path counters, but use one fixed slot per search
# depth so the Python iterative-deepening loop can expose where re-searches
# occur.  MAX_PLY is also the legal depth bound used by the recursive search.
DIAG_ASPIRATION_DEPTH_SLOTS = MAX_PLY + 1
DIAG_ASPIRATION_ITERATIONS_BASE = 66       # completed aspiration iterations
DIAG_ASPIRATION_FAIL_LOW_BASE = (
    DIAG_ASPIRATION_ITERATIONS_BASE + DIAG_ASPIRATION_DEPTH_SLOTS
)
DIAG_ASPIRATION_FAIL_HIGH_BASE = (
    DIAG_ASPIRATION_FAIL_LOW_BASE + DIAG_ASPIRATION_DEPTH_SLOTS
)
DIAG_ASPIRATION_RESEARCH_BASE = (
    DIAG_ASPIRATION_FAIL_HIGH_BASE + DIAG_ASPIRATION_DEPTH_SLOTS
)
DIAG_ASPIRATION_MAX_DELTA_BASE = (
    DIAG_ASPIRATION_RESEARCH_BASE + DIAG_ASPIRATION_DEPTH_SLOTS
)
DIAG_ASPIRATION_WASTED_NODES_BASE = (
    DIAG_ASPIRATION_MAX_DELTA_BASE + DIAG_ASPIRATION_DEPTH_SLOTS
)

# P3 deep-cost diagnostics.  These counters remain opt-in and deliberately
# do not change the qsearch/main-search/TT decision path.  Q cap forcing counts
# are pseudo-candidate upper bounds: the cap path does not pay make/unmake just
# to prove legality.  Main in-check counters classify the picker output with
# the same single/double-check geometry used by P0; EP/unknown geometry is a
# conservative fallback.  TT verify ``skip`` means the existing guard could
# not validate the move (for example pseudo-illegal/king-illegal), while the
# legacy cutoff is preserved for A/B safety.
DIAG_Q_CAP_NONCHECK = (
    DIAG_ASPIRATION_WASTED_NODES_BASE + DIAG_ASPIRATION_DEPTH_SLOTS
)
DIAG_Q_CAP_CHECK = DIAG_Q_CAP_NONCHECK + 1
DIAG_Q_CAP_FORCING = DIAG_Q_CAP_NONCHECK + 2
DIAG_Q_CAP_FORCING_MOVES = DIAG_Q_CAP_NONCHECK + 3
DIAG_MAIN_IN_CHECK_NODE = DIAG_Q_CAP_NONCHECK + 4
DIAG_MAIN_IN_CHECK_PICK_TRY = DIAG_Q_CAP_NONCHECK + 5
DIAG_MAIN_IN_CHECK_PRE_ILLEGAL = DIAG_Q_CAP_NONCHECK + 6
DIAG_MAIN_IN_CHECK_PRE_PASS = DIAG_Q_CAP_NONCHECK + 7
DIAG_MAIN_IN_CHECK_FALLBACK = DIAG_Q_CAP_NONCHECK + 8
DIAG_TT_VERIFY_TRY = DIAG_Q_CAP_NONCHECK + 9
DIAG_TT_VERIFY_PASS = DIAG_Q_CAP_NONCHECK + 10
DIAG_TT_VERIFY_REJECT = DIAG_Q_CAP_NONCHECK + 11
DIAG_TT_VERIFY_SAVED_CUT = DIAG_Q_CAP_NONCHECK + 12
DIAG_TT_VERIFY_SKIP = DIAG_Q_CAP_NONCHECK + 13

# P4 decision-quality diagnostics.  Source counters use the MovePicker stage
# observed when a legal move is actually searched.  All writes are guarded by
# ``diag_enabled`` and must not affect ordering, history, LMR, or re-search.
MOVE_ORDER_SOURCE_TT = 0
MOVE_ORDER_SOURCE_GOOD_CAPTURE = 1
MOVE_ORDER_SOURCE_GOOD_QUIET = 2
MOVE_ORDER_SOURCE_BAD_CAPTURE = 3
MOVE_ORDER_SOURCE_BAD_QUIET = 4
MOVE_ORDER_SOURCE_OTHER = 5
DIAG_ORDER_SOURCE_COUNT = 6
DIAG_ORDER_TRY_BASE = DIAG_TT_VERIFY_SKIP + 1
DIAG_ORDER_CUT_BASE = DIAG_ORDER_TRY_BASE + DIAG_ORDER_SOURCE_COUNT
DIAG_ORDER_KILLER1_TRY = DIAG_ORDER_CUT_BASE + DIAG_ORDER_SOURCE_COUNT
DIAG_ORDER_KILLER1_CUT = DIAG_ORDER_KILLER1_TRY + 1
DIAG_ORDER_KILLER2_TRY = DIAG_ORDER_KILLER1_TRY + 2
DIAG_ORDER_KILLER2_CUT = DIAG_ORDER_KILLER1_TRY + 3
DIAG_ORDER_COUNTER_TRY = DIAG_ORDER_KILLER1_TRY + 4
DIAG_ORDER_COUNTER_CUT = DIAG_ORDER_KILLER1_TRY + 5
DIAG_ORDER_QUIET_CUT = DIAG_ORDER_KILLER1_TRY + 6
DIAG_ORDER_QUIET_CUT_RANK_SUM = DIAG_ORDER_KILLER1_TRY + 7
DIAG_ORDER_NODES_BEFORE_CUT_SUM = DIAG_ORDER_KILLER1_TRY + 8

DIAG_LMR_R1_TRY = DIAG_ORDER_KILLER1_TRY + 9
DIAG_LMR_R2_TRY = DIAG_LMR_R1_TRY + 1
DIAG_LMR_R3P_TRY = DIAG_LMR_R1_TRY + 2
DIAG_LMR_R1_FAIL_HIGH = DIAG_LMR_R1_TRY + 3
DIAG_LMR_R2_FAIL_HIGH = DIAG_LMR_R1_TRY + 4
DIAG_LMR_R3P_FAIL_HIGH = DIAG_LMR_R1_TRY + 5
DIAG_LMR_QUIET_TRY = DIAG_LMR_R1_TRY + 6
DIAG_LMR_QUIET_FAIL_HIGH = DIAG_LMR_R1_TRY + 7
DIAG_LMR_CAPTURE_TRY = DIAG_LMR_R1_TRY + 8
DIAG_LMR_CAPTURE_FAIL_HIGH = DIAG_LMR_R1_TRY + 9
DIAG_LMR_RESEARCH_KEEP = DIAG_LMR_R1_TRY + 10
DIAG_LMR_RESEARCH_REJECT = DIAG_LMR_R1_TRY + 11
DIAG_LMR_RESEARCH_DEEPER = DIAG_LMR_R1_TRY + 12
DIAG_LMR_RESEARCH_SHALLOWER = DIAG_LMR_R1_TRY + 13
DIAG_LMR_FAIL_HIGH_UNVERIFIED = DIAG_LMR_R1_TRY + 14

# Counterfactual SF11-style post-LMR continuation-history samples.  These do
# not update a table: bonus/malus mean the verified result that a future
# experiment would have learned from.  Sign counters compare the current LMR
# statScore with the missing move-specific butterfly signal.
DIAG_POST_LMR_BONUS_SAMPLE = DIAG_LMR_R1_TRY + 15
DIAG_POST_LMR_MALUS_SAMPLE = DIAG_LMR_R1_TRY + 16
DIAG_POST_LMR_UNVERIFIED_SAMPLE = DIAG_LMR_R1_TRY + 17
DIAG_POST_LMR_BONUS_BUTTERFLY_NONNEG = DIAG_LMR_R1_TRY + 18
DIAG_POST_LMR_MALUS_BUTTERFLY_NONNEG = DIAG_LMR_R1_TRY + 19
DIAG_POST_LMR_BONUS_STAT_NONNEG = DIAG_LMR_R1_TRY + 20
DIAG_POST_LMR_MALUS_STAT_NONNEG = DIAG_LMR_R1_TRY + 21

# P5 history-prune attribution.  Layer order is deliberately stable because
# reports persist the numeric counters: piece-to, butterfly, pawn, then
# continuation offsets 1/2/3/4/6 ply.  Attribution is shadow-only and runs
# only after the production prune predicate has already evaluated true.
HIST_PRUNE_LAYER_MAIN = 0
HIST_PRUNE_LAYER_BUTTERFLY = 1
HIST_PRUNE_LAYER_PAWN = 2
HIST_PRUNE_LAYER_CONT_1 = 3
HIST_PRUNE_LAYER_CONT_2 = 4
HIST_PRUNE_LAYER_CONT_3 = 5
HIST_PRUNE_LAYER_CONT_4 = 6
HIST_PRUNE_LAYER_CONT_6 = 7
HIST_PRUNE_LAYER_COUNT = 8
DIAG_HP_NEG_COUNT_BASE = DIAG_POST_LMR_MALUS_STAT_NONNEG + 1
DIAG_HP_NEG_ABS_SUM_BASE = DIAG_HP_NEG_COUNT_BASE + HIST_PRUNE_LAYER_COUNT
DIAG_HP_DECISIVE_BASE = DIAG_HP_NEG_ABS_SUM_BASE + HIST_PRUNE_LAYER_COUNT
DIAG_HP_DOMINANT_BASE = DIAG_HP_DECISIVE_BASE + HIST_PRUNE_LAYER_COUNT
DIAG_HP_CONT_COALITION_DECISIVE = DIAG_HP_DOMINANT_BASE + HIST_PRUNE_LAYER_COUNT
DIAG_HP_NONCONT_ALREADY_PRUNES = DIAG_HP_CONT_COALITION_DECISIVE + 1
DIAG_HP_LMR_ELIGIBLE = DIAG_HP_CONT_COALITION_DECISIVE + 2
DIAG_HP_MAIN_GATE_NEAR_ZERO = DIAG_HP_CONT_COALITION_DECISIVE + 3
DIAG_HP_MARGIN_NEAR = DIAG_HP_CONT_COALITION_DECISIVE + 4
DIAG_HP_MARGIN_MID = DIAG_HP_CONT_COALITION_DECISIVE + 5
DIAG_HP_MARGIN_FAR = DIAG_HP_CONT_COALITION_DECISIVE + 6
DIAG_HP_MARGIN_SUM = DIAG_HP_CONT_COALITION_DECISIVE + 7
DIAG_HP_DEPTH_SUM = DIAG_HP_CONT_COALITION_DECISIVE + 8
DIAG_HP_MOVE_INDEX_SUM = DIAG_HP_CONT_COALITION_DECISIVE + 9
DIAG_HP_CONT1_CAP_HIT = DIAG_HP_MOVE_INDEX_SUM + 1
DIAG_HP_CONT1_CAP_RELIEF_SUM = DIAG_HP_MOVE_INDEX_SUM + 2
DIAG_HP_CONT1_CAP_RESCUE = DIAG_HP_MOVE_INDEX_SUM + 3
DIAG_SIZE = DIAG_HP_CONT1_CAP_RESCUE + 1

# Move Ordering Bonuses
SCORE_TT_MOVE = 100000
SCORE_GOOD_CAPTURE_BONUS = 20000
SCORE_KILLER_1 = 15000
SCORE_KILLER_2 = 14000
SCORE_COUNTER_MOVE = 10000
SCORE_BAD_CAPTURE_PENALTY = -5000
GOOD_QUIET_THRESHOLD = -1200

# QSearch check-evasion ordering.  Captures keep a separate score bucket so
# large accumulated quiet-history values cannot jump ahead of every capture.
QS_EVASION_CAPTURE_BUCKET = 20000
QS_EVASION_QUIET_CLAMP = 5000

# Contempt Factor: Score penalty for draws when the engine is winning.
# This encourages the engine to prefer winning lines over draws.
CONTEMPT = 20 # centipawns

PAWN_PUSH_RANK_BONUS = 2000
PAWN_PUSH_ATTACK_BONUS = 3000
# KING_TROPISM_BONUS removed (unused move-ordering stub; eval tropism also removed)
KING_ATTACK_BONUS = 2000 # Bonus for quiet moves attacking the opponent's King zone
ROOK_QUEEN_BATTERY_BONUS = 5000 # Bonus for Rook moving to same file/rank as Queen

# History Heuristics Constants (full history reform — HCE-conservative SF structure)
MAX_HISTORY = 16384  # legacy alias; prefer HISTORY_MAX_*
# Compatibility mode: Phase-1a read/write semantics on modern tables.
# 2026-07-16: full/soft full hist ~-30 Elo vs Old1a → default back to 1a behavior.
# Tables keep 7D cont / main shape for future experiments; ic/cap forced to 0 when True.
HISTORY_COMPAT_1A = True
# Restored Phase 1a ceilings (match classical_old gravity saturation)
HISTORY_MAX_MAIN = 16384
HISTORY_MAX_PIECE_TO = 16384
# E10 kept 10240; E14 kept 12288 (marginal +5 Elo; stop deeper BF for now).
HISTORY_MAX_BUTTERFLY = 12288
# E15 kept: capture ceiling 8192→10240.
HISTORY_MAX_CAPTURE = 10240
# E16 14336 rolled back (48.25% / −12 Elo). Keep 12288.
HISTORY_MAX_CONTINUATION = 12288
HISTORY_MAX_PAWN = 4096

PAWN_HISTORY_SIZE = 8192
PAWN_HISTORY_MASK = 8191

# Continuation: [in_check][is_capture][ply_idx 0..4][prev_pc][prev_to][curr_pc][curr_to]
CONT_PLY_OFFSETS = (1, 2, 3, 4, 6)  # ply_idx i ↔ ply - CONT_PLY_OFFSETS[i]
CONT_NUM_PLY = 5
# Update weights (SF-ish, sum style); applied at write, reads are 1:1
CONT_WEIGHT_0 = 1040
CONT_WEIGHT_1 = 780
CONT_WEIGHT_2 = 300
CONT_WEIGHT_3 = 537
CONT_WEIGHT_4 = 423
CONT_WEIGHT_SUM_DIV = 131072  # cont_bonus * w * cmhc // this
CONT_NEAR_BIAS = 71           # +bias for ply_idx < 2
# CMHC multipliers (1024 = neutral); index by agreement count 0..6 (clamped)
CONT_CMHC_TABLE = (96, 113, 101, 105, 127, 121, 126)

# 50-move rule constants
FIFTY_MOVE_RULE_LIMIT = 100
FIFTY_MOVE_SCALE_THRESHOLD = 80
FIFTY_MOVE_MAX_SCALE = 256

# History tuning in Search
SEE_HISTORY_DIVISOR = 512
LMR_HISTORY_DIVISOR = 10240

# Quiet ordering weights (MovePicker) - Decoupled from history pruning
ORDERING_WEIGHT_MAIN = 3
ORDERING_WEIGHT_PAWN = 2
ORDERING_WEIGHT_BUTTERFLY = 1
ORDERING_WEIGHT_CONT_1 = 5
ORDERING_WEIGHT_CONT_2 = 2
ORDERING_WEIGHT_CONT_3 = 1
ORDERING_WEIGHT_CONT_4 = 2
ORDERING_WEIGHT_CONT_5 = 1

# History pruning weights & thresholds (Step 14b History Pruning)
# 1a-compat: piece-to * 2 + main (as butterfly) * 1 + cont read-weights 4,2,1,2,1
# full mode: 2*main + piece_to*W/1024 + unweighted cont sum
# E8/H1 kept: main/piece-to weight 2→3 (1a-compat read path).
HISTORY_WEIGHT_MAIN = 3          # full: main*2; 1a-compat: piece-to uses this scale
HISTORY_WEIGHT_PIECE_TO = 256
HISTORY_WEIGHT_PIECE_TO_DEN = 1024
# E9/H2 kept: cont-1 weight 4→5 (1a-compat).
HISTORY_WEIGHT_CONT_1 = 5        # 1a-compat read weights (full mode ignores, uses 1)
HISTORY_WEIGHT_CONT_2 = 2
HISTORY_WEIGHT_CONT_3 = 1
HISTORY_WEIGHT_CONT_4 = 2
HISTORY_WEIGHT_CONT_5 = 1
# P6: history pruning alone must not let one sparse cont-1 cell contribute an
# arbitrarily large negative veto.  This is a weighted-score floor; ordering,
# LMR statScore, and the underlying continuation table remain unchanged.
HISTORY_PRUNE_CONT1_FLOOR = -4096
# 1a write: full bonus to piece-to and main (butterfly), cont raw bonus per layer
HISTORY_QUIET_MAIN_SCALE_NUM_1A = 1024
HISTORY_QUIET_PIECE_TO_SCALE_NUM_1A = 1024
HISTORY_QUIET_CONT_SCALE_NUM_1A = 1024

LOW_PLY_HISTORY_SIZE = 5
LOW_PLY_HISTORY_MAX = 7183
LOW_PLY_HISTORY_FILL = 100       # soft age: reset low-ply toward this each iteration

# --- Write scales (HCE soft full — after -31 Elo @200k nodes) ---
HISTORY_QUIET_MAIN_SCALE_NUM = 824
HISTORY_QUIET_MAIN_SCALE_DEN = 1024
HISTORY_QUIET_PIECE_TO_SCALE_NUM = 384   # was 512; less piece-to churn
HISTORY_QUIET_PIECE_TO_SCALE_DEN = 1024
HISTORY_QUIET_CONT_SCALE_NUM = 512       # was 820; damp cont writes (long-game pollution)
HISTORY_QUIET_CONT_SCALE_DEN = 1024
HISTORY_QUIET_MALUS_SCALE_NUM = 1024     # was 1136; milder failed-quiet penalty
HISTORY_QUIET_MALUS_SCALE_DEN = 1024
HISTORY_QUIET_MALUS_DECAY_NUM = 956
HISTORY_QUIET_MALUS_DECAY_DEN = 1024
HISTORY_CAP_BONUS_SCALE_NUM = 1152       # was 1280
HISTORY_CAP_BONUS_SCALE_DEN = 1024
HISTORY_CAP_MALUS_SCALE_NUM = 1126       # back toward Phase 1a
HISTORY_CAP_MALUS_SCALE_DEN = 1024
HISTORY_REFUTE_SCALE_NUM = 512           # was 680
HISTORY_REFUTE_SCALE_DEN = 1024
HISTORY_TT_MOVE_BONUS_EXTRA = 0          # was 280; TT quiet bump noisy on HCE
HISTORY_PARENT_STAT_DIV = 64             # was 40; weaker parent-stat feed
# Malus only updates shallow cont (ply offsets 1–2), reduces deep-history pollution
HISTORY_MALUS_SHALLOW_CONT = True

# Pawn history asymmetric gravity input (SF: 1038 if bonus > -7 else 525)
HISTORY_PAWN_BONUS_NUM = 1038
HISTORY_PAWN_MALUS_NUM = 525
HISTORY_PAWN_SCALE_DEN = 1024
HISTORY_PAWN_MALUS_THRESHOLD = -7

# Low-ply history write scale (SF)
HISTORY_LPH_SCALE_NUM = 663
HISTORY_LPH_SCALE_DEN = 1024

# Reverse quiet: off after match regression (pollutes main in HCE)
ENABLE_REVERSE_QUIET = False

# Quiet check ordering
# E1 (2026-07-16): experiment CHECK_SEE_GATE=True (SEE >= -75 before CHECK_BONUS)
CHECK_BONUS = 20000
CHECK_SEE_THRESHOLD = -75
ENABLE_CHECK_SEE_GATE = True

# Threat reordering — disabled (regressed when enabled)
ENABLE_THREAT_REORDERING = False
THREAT_MULTIPLIER = 20

# Capture history victim dimension: piece type 0..5 (SF CapturePieceToHistory)
CAPTURE_HISTORY_VICTIM_TYPES = 6

# History prune threshold. More negative = harder to prune.
# E3: -4000→-4500 kept; E12/H5: -4500→-5000 kept (2026-07-17).
PRUNING_HISTORY_THRESHOLD = -6500  # E23 keep; E25B −7000 aborted ~negative @n10k

# Continuation history prune threshold (SF19 Step 14: -4136 * depth, tuned from -1800 to -1500)
ENABLE_CONTINUATION_HISTORY_PRUNING = True
PRUNING_CONTINUATION_THRESHOLD = -1500

CORRECTION_HISTORY_SIZE = 16384
CORRECTION_HISTORY_MASK = 16383
CORRECTION_HISTORY_LIMIT = 1024
CORRECTION_HISTORY_DIVISOR = 131072
CORRECTION_HISTORY_PAWN_WEIGHT = 11433
CORRECTION_HISTORY_MINOR_WEIGHT = 8823
CORRECTION_HISTORY_NON_PAWN_WEIGHT = 12749
CORRECTION_HISTORY_UPDATE_DEPTH = 4
CORRECTION_HISTORY_CNTCV_FALLBACK = 0
CORRECTION_HISTORY_OUTER_SCALE = 1024
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
ASPIRATION_WINDOW_SIZE = 70 # centipawns (tuned via sweep experiment for optimal search efficiency)
ASPIRATION_WINDOW_MIN = 15         # HCE 適配最低初始窗口
ASPIRATION_WINDOW_BASE = 20        # Match baseline / SF11-style HCE starting window
ASPIRATION_WINDOW_SCALE_DIV = 120  # 分數縮放除數，數值越小，分數高時窗口擴張越快

# --- Pruning Techniques / 剪枝技術 ---
# Master switches for new pruning techniques / 新剪枝技術的總開關
ENABLE_LMP = True           # Late Move Pruning
ENABLE_PROBCUT = True       # default on; runtime knob / sweep may disable via SearchContext
ENABLE_DELTA_PRUNING = True # Delta Pruning in Quiescence Search

# --- Runtime tune indices (SearchContext.tune[int32]) for one-compile sweeps ---
# Values initialized from the scalar constants below; scripts mutate ctx.tune in Python.
TUNE_RFP_MULT = 0
TUNE_RAZOR_MARGIN = 1
TUNE_FP_BASE = 2
TUNE_FP_MULT = 3
TUNE_SEE_CAP_MARGIN = 4
TUNE_SEE_QUIET_MARGIN = 5
TUNE_LMR_BASE_OFFSET = 6
TUNE_LMR_HIST_SCALE = 7
TUNE_DELTA_MARGIN = 8
TUNE_PROBCUT_MARGIN = 9
TUNE_BAD_CAP_BONUS = 10
TUNE_GOOD_CAP_RELIEF = 11
TUNE_LMP_SCALE = 12          # percent: 100 = SF11-table limit; lower → prune earlier
TUNE_KILLER_RELIEF = 13      # LMR r subtract for killer/counter (1024 scale)
TUNE_QS_SEE = 14             # QS SEE threshold (more negative = prune more losing caps)
# Round-3 structural LMR (previously hard-coded; still SF11-capped in grids)
TUNE_LMR_CUTNODE = 15        # +r on cut nodes (default LMR_CUTNODE_BONUS)
TUNE_LMR_NO_TTMOVE = 16      # +r on cut when no TT move
TUNE_LMR_TTCAP = 17          # +r when TT move is capture
TUNE_LMR_MC_FACTOR = 18      # r -= mc * factor (higher → softer LMR)
TUNE_LMR_TTMOVE_RED = 19     # r subtract for researching TT move
# Structural experiment modes (runtime; see NMP_* production defaults below)
# NMP_SCOPE: 0=cut only, 1=all non-PV, 2=non-PV&d>=mind, 3=cut OR (non-PV&d>=mind)
TUNE_NMP_SCOPE = 20
# NMP_GATE: 0 = legacy, 1 = SF11 raw, 2 = SF11*78, 3 = custom G_* tune slots
TUNE_NMP_GATE = 21
# NMP_R: 0 = legacy 7+d//3, 1 = SF11, 2 = custom R_BASE + d//R_DIV + eval_margin
TUNE_NMP_R = 22
# PROBCUT_STYLE: 0 = flat TUNE_PROBCUT_MARGIN, 1 = SF 189-45*imp (*78/100), 2 = force off
TUNE_PROBCUT_STYLE = 23
# LMR base table (E2): scale REDUCTIONS product; not-improving numerator (SF uses 194/512)
TUNE_LMR_TABLE_SCALE = 24   # percent on REDUCTIONS[d]*REDUCTIONS[mc]
TUNE_LMR_NOT_IMP = 25       # not-improving numerator (higher → more reduction)
# NMP continuous params (gate=3 / r=2); defaults match legacy so gate=0 path still primary
TUNE_NMP_G_BASE = 26        # custom: threshold = beta - G_DEPTH*d - G_IMP*imp + G_BASE
TUNE_NMP_G_DEPTH = 27
TUNE_NMP_G_IMP = 28
TUNE_NMP_NEED_BETA = 29     # 0/1: also require static >= beta
TUNE_NMP_R_BASE = 30        # custom R base (legacy 7)
TUNE_NMP_R_DIV = 31         # custom R: base + depth//div (legacy 3)
TUNE_NMP_VERIFY_D = 32      # verification min depth (legacy 8; 999 = off)
TUNE_NMP_SCOPE_MIND = 33    # for scope 2/3: min depth for non-PV NMP
TUNE_SIZE = 34
# E2 tried 70 (puzzle +6 pass, nodes +25%); match @200k nodes: depth↓, no Elo.
LMR_TABLE_SCALE_PERCENT = 113  # E60 tuned (was 100; SPSA L1 12k + L2 600@300k +15.6 Elo)
LMR_NOT_IMP_NUM = 197  # E57 (was 194; SPSA L1+L2 400@300k +5.2 Elo)
ENABLE_IIR = True           # Internal Iterative Reduction (Replaces old IID)
ENABLE_SINGULAR_EXTENSIONS = True # Singular Extensions
ENABLE_MATE_DISTANCE_PRUNING = True # Mate Distance Pruning
ENABLE_MULTICUT = True  # Multi-Cut Pruning (Phase 2: Enabled with corrected exclusion_score >= beta)

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

# Pruning parameters — SF11 is a *reference*, not a hard aggression ceiling
# (policy lifted 2026-07-19). Prefer Elo/pass gates over "must not exceed SF11".
# SEE: SF11 capture ~-194 internal → ~-151 @0.78; current values stay until A/B.
PRUNING_SHALLOW_DEPTH = 12
PRUNING_CAPTURE_SEE_MARGIN = -100  # SF11 ~-194 internal; currently milder
PRUNING_QUIET_SEE_MARGIN = -25

# Master switches for existing pruning techniques / 現有剪枝技術的總開關
ENABLE_NMP = True           # Null Move Pruning
ENABLE_RAZORING = True      # Razoring
ENABLE_FP = True            # Futility Pruning
ENABLE_RFP = True           # Reverse Futility Pruning
ENABLE_LMR = True           # Late Move Reductions
ENABLE_ALPHA_RAISE_DEPTH_REDUCTION = False  # rejected 2026-07-20: δ1≈0 Elo, δ2≈−30 @n30k
ENABLE_FOLLOW_PV = False  # E27 rejected mid-match (unstable / not clearly +); re-test later if needed
# SF-style: quiet FP / quiet SEE use history-adjusted lmrDepth instead of raw depth
ENABLE_HIST_LMR_DEPTH_PRUNING = True  # E28 adopted (≈50% @n30k; keep; production ON)
# SF18 capture futility + captHist term on capture SEE margin (Phase 4 refactored)
ENABLE_CAPTURE_FUTILITY = True

# Capture futility (HCE cp; SF-inspired). fut ≈ static + BASE + MULT*lmrDepth + victim + hist
CAP_FP_MAX_LMR_DEPTH = 6       # apply only if pruning_lmr_depth < this (SF: lmrDepth < 7)
CAP_FP_BASE = 110              # SF: 231 (scaled to HCE 100cp pawn)
CAP_FP_LMR_MULT = 100          # SF: 232 (scaled to HCE 100cp pawn)
CAP_FP_CAPTHIST_NUM = 131      # + captHist * NUM // DEN
CAP_FP_CAPTHIST_DEN = 1024
# Capture SEE: threshold = SEE_CAP_MARGIN * depth - captHist * NUM // DEN
# (good/high captHist → more negative threshold → harder to prune; matches SF sign)
CAP_SEE_CAPTHIST_NUM = 34
CAP_SEE_CAPTHIST_DEN = 1024

# Alpha-raise depth reduction (kept for re-enable experiments; production OFF).
# SF18: depth>2 && depth<13 && depth-=2. HCE A/B: band 5–12, ply>0; δ1 neutral, δ2 negative.
ALPHA_RAISE_DEPTH_LO = 4
ALPHA_RAISE_DEPTH_HI = 13
ALPHA_RAISE_DEPTH_DELTA = 1
ALPHA_RAISE_MIN_IMPROVEMENT = 0

NULL_MOVE_REDUCTION = 2
MAX_QUIESCENCE_DEPTH = 5

# --- Prune margins (HCE cp; SF11 formulas remain useful baselines) ---

# Razoring — SF19 Step 8: eval < alpha - RAZORING_COEFF * depth * depth
RAZORING_MARGIN = 250  # Retained for runtime tune slot compatibility
RAZORING_MAX_DEPTH = 3
RAZORING_COEFF = 482

# Futility: larger mult → larger margin → harder to skip (softer FP).
# E25A 130→140 failed 1000@n10k 49.0%/−7 Elo → keep 130.
FP_BASE = 180
FP_MULTIPLIER = 125

RFP_MAX_DEPTH = 8
RFP_BASE_MULT = 170  # E24 180 aborted ~900@n10k ~49.6%/Elo−3 → rollback
RFP_NO_TT_PENALTY = 30

NMP_STATIC_MARGIN = 150      # not used as primary NMP gate
NMP_MIN_SIDE_NON_PAWNS = 2
NMP_VERIFICATION_DEPTH = 8   # verification from this depth (tunable via TUNE_NMP_VERIFY_D)
LOW_MATERIAL_PRUNING_PIECE_COUNT = 7
# Production NMP structure: cut_node only (legacy).
# N1 tried scope=3 (cut OR non-PV&d>=6): 1000@n10k 49.0%/Elo−6.6 → rollback.
# 0=cut only, 1=all non-PV (S1a pass−7), 2=non-PV&d>=mind only, 3=cut∨(nonPV&d>=mind)
NMP_SCOPE_MODE = 0
NMP_SCOPE_MIN_DEPTH = 6
# Runtime mode defaults.  These used to be hard-coded as zero while the
# corresponding SearchContext slots were already mutable.  Keeping the
# defaults in constants lets the UCI/SPSA adapter select a mode before the
# first Numba compilation without changing the production defaults.
NMP_GATE_MODE = 0          # 0=legacy, 1=SF11 raw, 2=SF11*78, 3=custom G slots
NMP_R_MODE = 0             # 0=legacy, 1=SF11, 2=custom R slots
PROBCUT_STYLE_MODE = 0     # 0=flat, 1=SF-style, 2=off
NMP_NEED_BETA = 0          # 0/1: custom NMP gate also requires static >= beta

# Late Move Reductions (LMR)
LMR_MIN_DEPTH = 3
# SF11: LMR when moveCount > 1 (+ root extras). Quiet index 2 ≈ not on first quiet.
LMR_MIN_QUIET_MOVE_INDEX = 2
LMR_REDUCTION = 1

# Late Move Pruning — SF11: (5 + depth^2) * (1+improving) / 2 - 1
# improving factor applied at search-time (// (2 - improving))
LMP_MOVE_COUNT = np.array([
    0 if d == 0 else max(1, (5 + d * d) // 2 - 1) for d in range(MAX_PLY)
], dtype=np.int32)
# Percent scale on LMP limit (100=table). Higher → later prune (fatter tree).
# E4: 100→105 kept (+15.7 Elo @100k). E57: 105→107 (+5.2 Elo @300k).
LMP_SCALE_PERCENT = 107

# LMR Table (legacy integer plies; live path uses 1024-scale REDUCTIONS)
LMR_TABLE = np.zeros((MAX_PLY, 256), dtype=np.int32)
for d in range(MAX_PLY):
    for mc in range(256):
        if d < 2 or mc < 2:
            LMR_TABLE[d, mc] = 0
        else:
            LMR_TABLE[d, mc] = int(0.77 + math.log(d) * math.log(mc) / 2.0)

# --- 1024-scale LMR — base table matches SF11 (24.8 * log), not above ---
REDUCTIONS = np.zeros(256, dtype=np.int32)
for i in range(1, 256):
    REDUCTIONS[i] = int(24.8 * math.log(i))

# LMR_BASE: E7 kept 480 (was 512). E20 keep 460. E56 tuned to 463 (+9.6 Elo). E60 tuned to 437 (+15.6 Elo).
LMR_BASE_OFFSET = 437  # E60 (was 463; SPSA L1 12k + L2 600@300k +15.6 Elo)
LMR_TTPV_INCREASE = 768
LMR_TTPV_DECREASE_BASE = 2048
LMR_TTPV_PV_BONUS = 768
LMR_CUTNODE_BONUS = 1839     # E60 (was 2117; SPSA L1 12k + L2 600@300k +15.6 Elo)
LMR_TTCAPTURE_BONUS = 1191   # E60 (was 1035; SPSA L1 12k + L2 600@300k +15.6 Elo)
# Match classical_old hard-coded LMR (±1024). P1 1536 withdrawn 2026-07-16.
LMR_BAD_CAPTURE_BONUS = 1046    # E57 (was 1024; SPSA L1+L2 400@300k +5.2 Elo)
LMR_GOOD_CAPTURE_RELIEF = 1030  # E57 (was 1024; SPSA L1+L2 400@300k +5.2 Elo)
LMR_LATE_CAPTURE_BONUS = 0
LMR_KILLER_COUNTER_RELIEF = 1024
LMR_MOVECOUNT_FACTOR = 35    # E60 (was 38; SPSA L1 12k + L2 600@300k +15.6 Elo)
LMR_HISTORY_SCALE = 204      # E60 (was 167; SPSA L1 12k + L2 600@300k +15.6 Elo)
LMR_CUTOFF_CNT_BASE = 128
LMR_CUTOFF_CNT_EXTRA = 512
LMR_ALLNODE_EXTRA = 512
LMR_TTMOVE_REDUCTION = 1024
LMR_NO_TTMOVE_BONUS = 1068     # E57 (was 1024; SPSA L1+L2 400@300k +5.2 Elo)
LMR_CORRECTION_DIVISOR = 32768

# SF19 Step 18: Scale up reductions for expected ALL nodes: r += r * 276 / (256 * depth + 268)
LMR_ALLNODE_SCALE_NUM = 276
LMR_ALLNODE_SCALE_DENOM_BASE = 256
LMR_ALLNODE_SCALE_DENOM_OFFSET = 268


# ProbCut — multi-pack 180 regressed with stack → keep 150 default
PROBCUT_R = 4  # Retained for tuning registry backwards compatibility
PROBCUT_MARGIN = 150
PROBCUT_MIN_DEPTH = 5  # Retained for tuning registry backwards compatibility
# SF19 Step 12: Dynamic ProbCut reduction based on improving flag
PROBCUT_R_IMPROVING = 5
PROBCUT_R_NOT_IMPROVING = 3
# ProbCut style=1 (SF11-ish): raisedBeta = beta + (BASE - IMP*improving) * HCE_SCALE
PROBCUT_SF_BASE = 189
PROBCUT_SF_IMPROVING = 45
PROBCUT_TT_DEPTH_OFFSET = 3  # store TT at depth - 3 after ProbCut cut

# Step 12: TT-based ProbCut Shortcut (SF 18 Step 12: 428 * 78 // 100 ≈ 334)
ENABLE_PROBCUT_TT_SHORTCUT = True
PROBCUT_TT_SHORTCUT_MARGIN = 334


# Delta Pruning (QS) — keep milder than aggressive node-exp values
# QS delta (P0 try 300 regressed fixed-depth suite nodes → keep 350)
DELTA_PRUNING_MARGIN = 350

# --- Static Exchange Evaluation (SEE) Threshold / SEE 閾值 ---
SEE_THRESHOLD = 0
# QS SEE (P0 try -60: puzzle OK but no clear node win on suite → keep -80)
QS_SEE_THRESHOLD = -80
ENABLE_SEE_IN_QUIESCENCE = True

# =============================================================================
# --- Search internals (formerly hard-coded in search.py) ---
# Tunable margins / formula coeffs only — not structural sentinels (TT empty, piece ids).
# =============================================================================

# HCE scale used when adapting SF internal cp → our HCE (≈100/128 MG)
HCE_SCALE_NUM = 78
HCE_SCALE_DEN = 100

# --- Hindsight / depth adjustment (parent reduction + opponent worsening) ---
HINDSIGHT_REDUCE_MIN_PRIOR = 3   # prior_reduction >= this and not worsening → depth += 1
HINDSIGHT_INCREASE_MIN_PRIOR = 2  # prior_reduction >= this and depth>=2 → maybe depth -= 1
HINDSIGHT_EVAL_SUM_MARGIN = 200   # static + parent_eval > this → shallower

# --- Dissonance (TT score vs static) ---
DISSONANCE_TT_DEPTH_SLACK = 2    # trust TT when tt.depth >= depth - slack
RAZOR_DISSONANCE_MAX = 250       # skip razor if |tt-static| above this
FP_DISSONANCE_THRESHOLD = 190    # add dissonance//2 to FP margin when above

# Force full HCE when lazy/TT static near known-win band
FULL_EVAL_KNOWN_WIN_MARGIN = 500  # abs(static) >= VALUE_KNOWN_WIN - this

# --- TT cutoff gates ---
TT_CUTOFF_HALFMOVE_MAX = 96      # skip TT score cut near 50-move GHI
TT_CONSISTENCY_MIN_DEPTH = 4     # cutNode consistency: or depth > this
TT_DEEP_VERIFY_DEPTH = 7         # child-TT verification at depth >= this
TT_PENALIZE_MIN_DEPTH = 5        # penalize mismatched bound when depth > this

# --- History bonus / malus on cutoffs ---
# 2026-07-16: quadratic regressed (~-31 Elo). Prefer Phase-1a linear for HCE.
# USE_LINEAR: stat = min(SCALE * depth, CAP); else SF11 quadratic skeleton.
HISTORY_USE_LINEAR_BONUS = True
HISTORY_BONUS_QUAD_A = 16
HISTORY_BONUS_QUAD_B = 96
HISTORY_MALUS_QUAD_A = 18
HISTORY_MALUS_QUAD_B = 88
HISTORY_BONUS_DEPTH_CLAMP = 14
HISTORY_BONUS_SCALE = 120
HISTORY_BONUS_CAP = 1800
HISTORY_MALUS_CAP = 1600         # was 2000; match 1a malus cap
HISTORY_NODE_WIDTH_DIV = 256     # non-PV: bonus += bonus * (q+c) // div
CAPTURE_MALUS_SCALE_NUM = 1126
CAPTURE_MALUS_SCALE_DEN = 1024
FAIL_LOW_MIN_LEGAL_MOVES = 4     # gate fail-low feedback (1a); reduce noise
# Fail-low structured feedback (conservative vs SF)
FAIL_LOW_PARENT_STAT_DIV = 120
FAIL_LOW_DEPTH_BONUS_SCALE = 40
FAIL_LOW_DEPTH_BONUS_CAP = 320
FAIL_LOW_MOVECOUNT_THR = 8
FAIL_LOW_MOVECOUNT_BONUS = 120
FAIL_LOW_STATIC_MARGIN = 90
FAIL_LOW_STATIC_BONUS = 90
FAIL_LOW_PARENT_STATIC_MARGIN = 70
FAIL_LOW_PARENT_STATIC_BONUS = 90
FAIL_LOW_BASE_DEPTH_MULT = 100
FAIL_LOW_BASE_OFFSET = 60
FAIL_LOW_BASE_CAP = 1000
FAIL_LOW_SCALE_DIV = 512
FAIL_LOW_CONT_NUM = 200
FAIL_LOW_CONT_DEN = 16384
FAIL_LOW_MAIN_NUM = 180
FAIL_LOW_MAIN_DEN = 32768
FAIL_LOW_PAWN_NUM = 260
FAIL_LOW_PAWN_DEN = 8192
FAIL_LOW_CAP_DEPTH_MULT = 40
FAIL_LOW_CAP_BASE = 200
FAIL_LOW_CAP_CAP = 700

# --- Quiet move ordering (score_quiets additive bonuses) ---
# Live path adds these on top of history; SCORE_KILLER_* are absolute scores elsewhere.
QUIET_ORDER_KILLER_1 = 400000
QUIET_ORDER_KILLER_2 = 350000
QUIET_ORDER_COUNTER = 300000
LPH_ORDER_SCALE = 8              # score += scale * lph // (1+ply)

# --- NMP formula coeffs ---
NMP_MIN_DEPTH = 3
# Legacy gate: static >= beta - DEPTH_COEF*d - IMP*improving + BASE
NMP_LEGACY_DEPTH_COEF = 14
NMP_LEGACY_IMPROVING_COEF = 45
# Legacy gate: static >= beta - 14*d - 45*imp + BASE
# d=10 !imp: BASE200 → β+60; BASE160 → β+20; BASE80 was match-negative → avoid.
NMP_LEGACY_BASE = 200
# SF11 gate margin: -DEPTH_COEF*d + BASE - IMP*improving (optionally * HCE scale)
NMP_SF_MARGIN_DEPTH_COEF = 32
NMP_SF_MARGIN_BASE = 292
NMP_SF_MARGIN_IMPROVING = 30
# R: SF (BASE + DEPTH*d) // DIV + eval_margin; legacy BASE + d//DEPTH_DIV + eval_margin
NMP_EVAL_MARGIN_DIV = 150
NMP_EVAL_MARGIN_MAX = 3
NMP_SF_R_BASE = 854
NMP_SF_R_DEPTH = 68
NMP_SF_R_DIV = 258
NMP_LEGACY_R_BASE = 7
NMP_LEGACY_R_DEPTH_DIV = 3

# --- IIR / FP / LMP / check extension depth gates ---
IIR_MIN_DEPTH = 4
FP_MAX_DEPTH = 6
LMP_MIN_LIMIT = 2                # floor after table * scale
# SF19 Step 17: Check extension eliminated to prevent check explosion
ENABLE_CHECK_EXTENSION = False
CHECK_EXT_NON_PV_MAX_DEPTH = 5   # non-PV non-TT: extend only if depth <= this (when enabled)

# Search/timer synchronization. Must remain 2^n - 1 because hot paths use
# ``nodes & mask`` instead of modulo. This bounds stop latency to about 4k
# nodes instead of the former 32k-node staircase.
STOP_CHECK_MASK = 4095

# --- Singular extension ---
SINGULAR_TT_DEPTH_SLACK = 3      # tt.depth >= depth - slack
SINGULAR_MARGIN_BASE = 60        # margin = (BASE + ttpv_bonus) * depth // DIV
SINGULAR_TTPV_BONUS = 70
SINGULAR_MARGIN_DIV = 59
SINGULAR_DOUBLE_EXT_MIN_DEPTH = 10

# --- LMR (remaining hard-codes → named) ---
LMR_REDUCTION_BASE_ADD = 1027    # r = scale + BASE_ADD (+ not-imp term)
LMR_NOT_IMP_DEN = 512            # not-imp: scale * num // DEN
LMR_TT_SCORE_GT_ALPHA = 838      # ttPv && tt_score > alpha → r -=
LMR_TT_DEPTH_GE = 923            # tt.depth >= depth → r -=
LMR_TT_DEPTH_GE_CUT = 955        # extra when cut_node
LMR_ADVANCED_PAWN_RELIEF = 1536  # -1.5 ply on advanced pawn push
LMR_ADVANCED_PAWN_WHITE_RANK = 5 # white to_rank >= this
LMR_ADVANCED_PAWN_BLACK_RANK = 2 # black to_rank <= this
LMR_LATE_CAPTURE_MAX_DEPTH = 8   # when LMR_LATE_CAPTURE_BONUS > 0
LMR_LATE_CAPTURE_MIN_MC = 2
LMR_TTMOVE_CUTNODE_EXTRA = 150   # tt-move LMR: +extra on cut nodes
LMR_R_FLOOR = -10                # clamp r before ply conversion
LMR_D_MAX_EXTRA = 2              # d <= search_depth + this
LMR_CAPTURE_VICTIM_SCALE = 809   # SF18: 809 * PieceValue / 128 + captHist
LMR_CAPTURE_VICTIM_DIV = 128
LMR_STAT_SCORE_DIV = 4096        # r -= stat * hist_scale // DIV
LMR_RESEARCH_DEEPER_MARGIN = 32  # eval > max_eval + this → +1 depth research
LMR_RESEARCH_SHALLOWER_MARGIN = 15

# --- QS fail-high / stand-pat smoothing (a*score + b*beta) // div ---
QS_STANDPAT_SMOOTH_A = 467
QS_STANDPAT_SMOOTH_B = 557
QS_FAILHIGH_SMOOTH_A = 481
QS_FAILHIGH_SMOOTH_B = 543
QS_SMOOTH_DIV = 1024

# --- Correction history write scales ---
CORRECTION_CNTCV_WEIGHT = 10000  # HCE: weight * (val_2 + val_4); SF18 ~8363
CORRECTION_BONUS_HAS_MOVE_DIV = 10
CORRECTION_BONUS_NO_MOVE_DIV = 8
CORRECTION_MINOR_SCALE_NUM = 155
CORRECTION_MINOR_SCALE_DEN = 128
CORRECTION_NP_SCALE_NUM = 181
CORRECTION_NP_SCALE_DEN = 128
CORRECTION_CONT_2PLY_NUM = 136
CORRECTION_CONT_4PLY_NUM = 68
CORRECTION_CONT_SCALE_DEN = 128

# =============================================================================
# --- Bitboard Utilities Constants / 位元棋盤工具常量 ---
# =============================================================================

BB_SQUARES = np.array([np.uint64(1) << i for i in range(64)], dtype=np.uint64)
"""
位元棋盤陣列，其中每個位元棋盤在對應的方格索引處設置了一個位元。
"""

# Rank Masks / 橫排掩碼
RANK_MASKS = np.array([np.uint64(0xFF) << np.uint64(i * 8) for i in range(8)], dtype=np.uint64)

# Square Color Masks / 方格顏色掩碼 (a1=sq 0 is dark)
DARK_SQUARES = np.uint64(0xAA55AA55AA55AA55)
LIGHT_SQUARES = np.uint64(0x55AA55AA55AA55AA)

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

TT_SIZE_MB = 128

# --- Backward Pawns / 後兵 ---
# Penalty for a backward pawn.
# 後兵的懲罰。
# Negative values, applied with += (consistent with ISOLATED_PAWN_PENALTY and DOUBLED_PAWN_PENALTY)
BACKWARD_PAWN_PENALTY = np.array([0, -5], dtype=np.int32)
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



