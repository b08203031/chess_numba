import numba
import numpy as np
from chess_engine.nnue.engine_types import piece_bbs_signature, occupancy_bbs_signature, game_state_signature, search_context_type
from chess_engine.nnue.ml_eval.inference import nnue_forward_incremental

# The NNUE evaluation function needs to be a drop-in replacement for the
# original evaluate_position in evaluation.py, which has this signature:
# @numba.njit(numba.int32(piece_bbs_signature, occupancy_bbs_signature, game_state_signature, search_context_type, numba.boolean))

@numba.njit(numba.int32(piece_bbs_signature, occupancy_bbs_signature, game_state_signature, search_context_type, numba.int32, numba.boolean), cache=True, boundscheck=False, fastmath=True)
def evaluate_position(piece_bbs, occupancy_bbs, game_state, search_context, ply, lazy: bool = False):
    """
    Evaluates a chess position using the NNUE network.
    
    Args:
        piece_bbs (np.ndarray): 12 個棋子的位元棋盤。
        occupancy_bbs (np.ndarray): 3 個佔用位元棋盤 (白, 黑, 全)。
        game_state (np.ndarray): 遊戲狀態封裝 [side_to_move, castling_rights, en_passant_sq, halfmove_clock, ...]。
        ply (int): 節點目前的深度。
        lazy (bool): 是否為懶惰評估。對於 NNUE 來說沒有區別，我們直接回傳完整計算結果。
        
    Returns:
        int: 從行棋方視角的評估分數（Centipawn）。
    """
    # 1. 取得行棋方 (STM: 0=White, 1=Black)
    side_to_move = numba.int32(game_state[0])
    
    # 2. 計算盤面全部棋子數量 (piece_count) 用作 NNUE 的 bucket 索引
    piece_count = 0
    for p_idx in range(12):
        bb = piece_bbs[p_idx]
        while bb:
            piece_count += 1
            bb &= bb - np.uint64(1)
    
    # 3. 以 STM 視角取得神經網路評估值 logit (已轉換為 Centipawn)
    stm_score = nnue_forward_incremental(ply, side_to_move, search_context.accumulator_stack, piece_count)
    
    # 4. Initiative Bonus
    initiative_bonus = np.int32(15)
    stm_score += initiative_bonus
    
    # The neural network naturally outputs the score relative to the side to move
    return stm_score
