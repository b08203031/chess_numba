# chess_engine/evaluation.py

import numba
import numpy as np

from chess_engine.constants import (
    MG_MATERIAL_VALUES, EG_MATERIAL_VALUES,
    PST_MG, PST_EG,
    PHASE_WEIGHTS, MAX_PHASE,
    PAWN_SHIELD_BONUS, SEMI_OPEN_FILE_PENALTY, ATTACKER_WEIGHTS,
    BB_SQUARES
)

from chess_engine.zobrist import get_lsb_index
from chess_engine.engine_types import piece_bbs_signature, occupancy_bbs_signature, game_state_signature
from chess_engine.bitboard_utils import count_bits, KING_ATTACK_ZONES, FILE_MASKS
from chess_engine.core import generate_legal_moves, is_king_in_check
from chess_engine.constants import MATE_SCORE


@numba.njit(numba.types.UniTuple(numba.int32, 2)(piece_bbs_signature), cache=True)
def evaluate_king_safety(piece_bbs):
    """
    評估雙方的國王安全，並分別返回中局（MG）和殘局（EG）的分數差異。
    分數是從白方的角度計算的（正分對白方有利，負分對黑方有利）。
    """
    mg_safety_score = np.int32(0)
    eg_safety_score = np.int32(0)

    # --- 1. 兵盾評估 (Pawn Shield Evaluation) ---
    # 預先定義好國王在不同位置時，其前方兵盾的位元板遮罩。
    # "完美"兵盾指兵在初始位置，"推進"兵盾指兵向前一格。
    W_KS_SHIELD_PERFECT = BB_SQUARES[13] | BB_SQUARES[14] | BB_SQUARES[15]  # f2, g2, h2
    W_KS_SHIELD_ADVANCED = BB_SQUARES[21] | BB_SQUARES[22] | BB_SQUARES[23] # f3, g3, h3
    W_QS_SHIELD_PERFECT = BB_SQUARES[8] | BB_SQUARES[9] | BB_SQUARES[10]    # a2, b2, c2
    W_QS_SHIELD_ADVANCED = BB_SQUARES[16] | BB_SQUARES[17] | BB_SQUARES[18]  # a3, b3, c3

    B_KS_SHIELD_PERFECT = BB_SQUARES[53] | BB_SQUARES[54] | BB_SQUARES[55]  # f7, g7, h7
    B_KS_SHIELD_ADVANCED = BB_SQUARES[45] | BB_SQUARES[46] | BB_SQUARES[47] # f6, g6, h6
    B_QS_SHIELD_PERFECT = BB_SQUARES[48] | BB_SQUARES[49] | BB_SQUARES[50]  # a7, b7, c7
    B_QS_SHIELD_ADVANCED = BB_SQUARES[40] | BB_SQUARES[41] | BB_SQUARES[42]  # a6, b6, c6

    # 獲取雙方國王和兵的位置
    white_king_sq = get_lsb_index(piece_bbs[5])
    black_king_sq = get_lsb_index(piece_bbs[11])
    white_pawns = piece_bbs[0]
    black_pawns = piece_bbs[6]

    # --- 白方兵盾評估 ---
    if white_king_sq == 6:  # 王翼易位後的國王在 g1
        perfect = count_bits(white_pawns & W_KS_SHIELD_PERFECT)
        advanced = count_bits(white_pawns & W_KS_SHIELD_ADVANCED)
        mg_safety_score += perfect * PAWN_SHIELD_BONUS[0][0] + advanced * PAWN_SHIELD_BONUS[1][0]
        eg_safety_score += perfect * PAWN_SHIELD_BONUS[0][1] + advanced * PAWN_SHIELD_BONUS[1][1]
    elif white_king_sq == 2:  # 后翼易位後的國王在 c1
        perfect = count_bits(white_pawns & W_QS_SHIELD_PERFECT)
        advanced = count_bits(white_pawns & W_QS_SHIELD_ADVANCED)
        mg_safety_score += perfect * PAWN_SHIELD_BONUS[0][0] + advanced * PAWN_SHIELD_BONUS[1][0]
        eg_safety_score += perfect * PAWN_SHIELD_BONUS[0][1] + advanced * PAWN_SHIELD_BONUS[1][1]
    elif white_king_sq == 4:  # 國王未易位，在 e1
        W_CENTER_SHIELD_PERFECT = BB_SQUARES[11] | BB_SQUARES[12] | BB_SQUARES[13] # d2, e2, f2
        W_CENTER_SHIELD_ADVANCED = BB_SQUARES[19] | BB_SQUARES[20] | BB_SQUARES[21] # d3, e3, f3
        perfect = count_bits(white_pawns & W_CENTER_SHIELD_PERFECT)
        advanced = count_bits(white_pawns & W_CENTER_SHIELD_ADVANCED)
        # 未易位時的兵盾獎勵減半
        mg_safety_score += (perfect * PAWN_SHIELD_BONUS[0][0] + advanced * PAWN_SHIELD_BONUS[1][0]) // 2
        eg_safety_score += (perfect * PAWN_SHIELD_BONUS[0][1] + advanced * PAWN_SHIELD_BONUS[1][1]) // 2

    # --- 黑方兵盾評估 ---
    if black_king_sq == 62:  # 王翼易位後的國王在 g8
        perfect = count_bits(black_pawns & B_KS_SHIELD_PERFECT)
        advanced = count_bits(black_pawns & B_KS_SHIELD_ADVANCED)
        mg_safety_score -= perfect * PAWN_SHIELD_BONUS[0][0] + advanced * PAWN_SHIELD_BONUS[1][0]
        eg_safety_score -= perfect * PAWN_SHIELD_BONUS[0][1] + advanced * PAWN_SHIELD_BONUS[1][1]
    elif black_king_sq == 58:  # 后翼易位後的國王在 c8
        perfect = count_bits(black_pawns & B_QS_SHIELD_PERFECT)
        advanced = count_bits(black_pawns & B_QS_SHIELD_ADVANCED)
        mg_safety_score -= perfect * PAWN_SHIELD_BONUS[0][0] + advanced * PAWN_SHIELD_BONUS[1][0]
        eg_safety_score -= perfect * PAWN_SHIELD_BONUS[0][1] + advanced * PAWN_SHIELD_BONUS[1][1]
    elif black_king_sq == 60:  # 國王未易位，在 e8
        B_CENTER_SHIELD_PERFECT = BB_SQUARES[51] | BB_SQUARES[52] | BB_SQUARES[53] # d7, e7, f7
        B_CENTER_SHIELD_ADVANCED = BB_SQUARES[43] | BB_SQUARES[44] | BB_SQUARES[45] # d6, e6, f6
        perfect = count_bits(black_pawns & B_CENTER_SHIELD_PERFECT)
        advanced = count_bits(black_pawns & B_CENTER_SHIELD_ADVANCED)
        # 未易位時的兵盾獎勵減半
        mg_safety_score -= (perfect * PAWN_SHIELD_BONUS[0][0] + advanced * PAWN_SHIELD_BONUS[1][0]) // 2
        eg_safety_score -= (perfect * PAWN_SHIELD_BONUS[0][1] + advanced * PAWN_SHIELD_BONUS[1][1]) // 2

    # --- 2. 半開放線路懲罰 (Semi-Open Files Penalty) ---
    # 懲罰指向國王及其相鄰線路的、沒有我方兵防守的線路。
    
    # --- 白方國王 ---
    white_king_file = white_king_sq % 8
    # 遍歷國王所在以及左右相鄰的線路
    for f in range(max(0, white_king_file - 1), min(7, white_king_file + 1) + 1):
        file_mask = FILE_MASKS[f]
        # 如果該線路上沒有白兵
        if not (white_pawns & file_mask):
            # 計算該線路上有多少黑方的車和后
            attackers = count_bits(piece_bbs[9] & file_mask) + count_bits(piece_bbs[10] & file_mask)
            # 根據攻擊者數量施加懲罰
            mg_safety_score += attackers * SEMI_OPEN_FILE_PENALTY[0]
            eg_safety_score += attackers * SEMI_OPEN_FILE_PENALTY[1]
            
    # --- 黑方國王 ---
    black_king_file = black_king_sq % 8
    # 遍歷國王所在以及左右相鄰的線路
    for f in range(max(0, black_king_file - 1), min(7, black_king_file + 1) + 1):
        file_mask = FILE_MASKS[f]
        # 如果該線路上沒有黑兵
        if not (black_pawns & file_mask):
            # 計算該線路上有多少白方的車和后
            attackers = count_bits(piece_bbs[3] & file_mask) + count_bits(piece_bbs[4] & file_mask)
            # 根據攻擊者數量施加懲罰（從白方角度，所以是減去懲罰值，相當於加分）
            mg_safety_score -= attackers * SEMI_OPEN_FILE_PENALTY[0]
            eg_safety_score -= attackers * SEMI_OPEN_FILE_PENALTY[1]

    # --- 3. 攻擊者鄰近度懲罰 (Attacker Proximity Penalty) ---
    # 懲罰位於國王 5x5 區域內的對方棋子。
    
    # --- 白方國王區域內的黑棋 ---
    white_king_zone = KING_ATTACK_ZONES[white_king_sq]
    # 根據棋子類型和數量，乘以對應的威脅權重並累加懲罰。
    mg_safety_score -= count_bits(piece_bbs[10] & white_king_zone) * ATTACKER_WEIGHTS[0][0] # 黑后
    eg_safety_score -= count_bits(piece_bbs[10] & white_king_zone) * ATTACKER_WEIGHTS[0][1]
    mg_safety_score -= count_bits(piece_bbs[9] & white_king_zone) * ATTACKER_WEIGHTS[1][0]  # 黑車
    eg_safety_score -= count_bits(piece_bbs[9] & white_king_zone) * ATTACKER_WEIGHTS[1][1]
    mg_safety_score -= count_bits(piece_bbs[8] & white_king_zone) * ATTACKER_WEIGHTS[2][0]  # 黑象
    eg_safety_score -= count_bits(piece_bbs[8] & white_king_zone) * ATTACKER_WEIGHTS[2][1]
    mg_safety_score -= count_bits(piece_bbs[7] & white_king_zone) * ATTACKER_WEIGHTS[3][0]  # 黑馬
    eg_safety_score -= count_bits(piece_bbs[7] & white_king_zone) * ATTACKER_WEIGHTS[3][1]
    mg_safety_score -= count_bits(piece_bbs[6] & white_king_zone) * ATTACKER_WEIGHTS[4][0]  # 黑兵
    eg_safety_score -= count_bits(piece_bbs[6] & white_king_zone) * ATTACKER_WEIGHTS[4][1]

    # --- 黑方國王區域內的白棋 ---
    black_king_zone = KING_ATTACK_ZONES[black_king_sq]
    # 邏輯同上，但分數是正的（對白方有利）
    mg_safety_score += count_bits(piece_bbs[4] & black_king_zone) * ATTACKER_WEIGHTS[0][0] # 白后
    eg_safety_score += count_bits(piece_bbs[4] & black_king_zone) * ATTACKER_WEIGHTS[0][1]
    mg_safety_score += count_bits(piece_bbs[3] & black_king_zone) * ATTACKER_WEIGHTS[1][0]  # 白車
    eg_safety_score += count_bits(piece_bbs[3] & black_king_zone) * ATTACKER_WEIGHTS[1][1]
    mg_safety_score += count_bits(piece_bbs[2] & black_king_zone) * ATTACKER_WEIGHTS[2][0]  # 白象
    eg_safety_score += count_bits(piece_bbs[2] & black_king_zone) * ATTACKER_WEIGHTS[2][1]
    mg_safety_score += count_bits(piece_bbs[1] & black_king_zone) * ATTACKER_WEIGHTS[3][0]  # 白馬
    eg_safety_score += count_bits(piece_bbs[1] & black_king_zone) * ATTACKER_WEIGHTS[3][1]
    mg_safety_score += count_bits(piece_bbs[0] & black_king_zone) * ATTACKER_WEIGHTS[4][0]  # 白兵
    eg_safety_score += count_bits(piece_bbs[0] & black_king_zone) * ATTACKER_WEIGHTS[4][1]

    return mg_safety_score, eg_safety_score


@numba.njit(numba.int32(piece_bbs_signature, occupancy_bbs_signature, game_state_signature), cache=True)
def evaluate_position(piece_bbs, occupancy_bbs, game_state):
    """
    使用 Tapered Evaluation (加權評估) 模型評估目前局面，並從當前執棋方的角度返回分數。
    評估包含：
    1. 物質價值 (Material value) - MG 和 EG
    2. 棋子位置表 (Piece-square tables) - MG 和 EG
    3. 國王安全 (King Safety) - MG 和 EG
    4. 根據遊戲階段 (Game Phase) 進行加權插值。
    
    返回:
        np.int32: 以百分之一兵為單位的分數。正分表示當前執棋方有優勢。
    """
    side_to_move = game_state[0]

    # --- 0. 檢查遊戲是否結束 (Check for Game Over) ---
    # 在進行靜態評估之前，首先檢查是否存在任何合法走法。
    legal_moves = generate_legal_moves(piece_bbs, occupancy_bbs, game_state)
    has_legal_moves = False
    for move in legal_moves:
        if move != 0:
            has_legal_moves = True
            break

    if not has_legal_moves:
        # 如果沒有合法走法，則判斷是將死還是逼和
        if is_king_in_check(piece_bbs, occupancy_bbs, game_state):
            # 當前玩家被將死，這是一個極大的負分
            return np.int32(-MATE_SCORE)
        else:
            # 逼和，分數為 0
            return np.int32(0)

    # --- 1. 計算遊戲階段 (Game Phase) ---
    # Game Phase 用於決定中局和殘局評估的權重。
    # 它的初始值為 MAX_PHASE，並隨著棋子（不含兵和王）的兌換而減少。
    phase = np.int32(0)
    # 棋子類型索引 1-4 分別為 馬、象、車、后
    # 白棋
    phase += count_bits(piece_bbs[1]) * PHASE_WEIGHTS[1]
    phase += count_bits(piece_bbs[2]) * PHASE_WEIGHTS[2]
    phase += count_bits(piece_bbs[3]) * PHASE_WEIGHTS[3]
    phase += count_bits(piece_bbs[4]) * PHASE_WEIGHTS[4]
    # 黑棋
    phase += count_bits(piece_bbs[7]) * PHASE_WEIGHTS[1]
    phase += count_bits(piece_bbs[8]) * PHASE_WEIGHTS[2]
    phase += count_bits(piece_bbs[9]) * PHASE_WEIGHTS[3]
    phase += count_bits(piece_bbs[10]) * PHASE_WEIGHTS[4]

    # 確保 phase 不會超過最大值（例如，在一些特殊開局或升變後）
    phase = min(phase, MAX_PHASE)

    # --- 2. 計算中局和殘局的基礎分數（物質 + 位置） ---
    mg_score = np.int32(0)
    eg_score = np.int32(0)

    # 遍歷白棋 (索引 0-5: 兵, 馬, 象, 車, 后, 王)
    for piece_type in range(6):
        bb = piece_bbs[piece_type]
        while bb:
            sq = get_lsb_index(bb)
            # 累加物質價值和棋子在該位置的價值
            mg_score += MG_MATERIAL_VALUES[piece_type] + PST_MG[piece_type][sq]
            eg_score += EG_MATERIAL_VALUES[piece_type] + PST_EG[piece_type][sq]
            bb &= bb - np.uint64(1)  # 清除 LSB，繼續處理下一個棋子

    # 遍歷黑棋 (bb 索引 6-11)
    for piece_type in range(6):
        bb = piece_bbs[piece_type + 6]
        while bb:
            sq = get_lsb_index(bb)
            # 對手的分數要減去
            # 黑棋的位置需要垂直翻轉 (sq ^ 56) 來對應白方的 PST
            mg_score -= MG_MATERIAL_VALUES[piece_type] + PST_MG[piece_type][sq ^ 56]
            eg_score -= EG_MATERIAL_VALUES[piece_type] + PST_EG[piece_type][sq ^ 56]
            bb &= bb - np.uint64(1)

    # --- 3. 加入國王安全分數 ---
    # 調用 evaluate_king_safety 獲取 MG 和 EG 的國王安全分差
    mg_king_safety, eg_king_safety = evaluate_king_safety(piece_bbs)
    mg_score += mg_king_safety
    eg_score += eg_king_safety

    # --- 4. 根據遊戲階段進行插值計算 ---
    # 這個公式混合了 MG 和 EG 的分數。
    # 隨著遊戲進行 (phase 減少)，EG 分數的影響力會越來越大。
    final_score = (mg_score * phase + eg_score * (MAX_PHASE - phase)) // MAX_PHASE

    # --- 5. 從當前執棋方的角度返回最終分數 ---
    if side_to_move == 0:  # 白方回合
        return np.int32(final_score)
    else:  # 黑方回合
        return np.int32(-final_score)
