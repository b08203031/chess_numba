# chess_engine/engine_types.py
import numba
import numpy as np
from numba.experimental import jitclass
from chess_engine.classical_old.transposition_table import numba_tt_entry_type
from chess_engine.classical_old.constants import (
    MAX_PLY, CORRECTION_HISTORY_SIZE,
    ENABLE_NMP, ENABLE_RFP, ENABLE_RAZORING,
    ENABLE_LMR, ENABLE_SHALLOW_SEE_PRUNING, ENABLE_LMP, ENABLE_FP,
    ENABLE_PROBCUT, ENABLE_MULTICUT
)

"""
此模組定義了西洋棋引擎中使用的 Numba 類型和類別。
它包含了用於搜尋上下文的 jitclass 定義以及棋盤狀態的類型簽名。
"""

# --- Color Constants / 顏色常量 ---
WHITE, BLACK = 0, 1

# --- Piece Type Constants / 棋子類型常量 ---
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 0, 1, 2, 3, 4, 5


# --- Numba Type Signatures for the Refactored Board State / 重構後棋盤狀態的 Numba 類型簽名 ---
piece_bbs_signature = numba.uint64[::1]
occupancy_bbs_signature = numba.uint64[::1]
game_state_signature = numba.uint64[::1]
piece_counts_signature = numba.types.UniTuple(numba.int32, 12)

unmake_info_signature = numba.types.Tuple([
    numba.int8, numba.int8, numba.uint8, numba.uint8, numba.uint8, numba.uint64, numba.uint64,
    numba.uint64, numba.uint64, numba.uint64
])

# --- Search Context / 搜尋上下文 ---
search_context_spec = [
    ('transposition_table', numba.types.Array(numba_tt_entry_type, 1, 'C')),
    ('killer_moves', numba.uint16[::1]),
    ('pv_table', numba.uint16[:, :]),
    ('history_table', numba.int32[:, :]),
    ('nodes_searched', numba.uint64),
    ('end_time', numba.float64),
    ('stop_flag', numba.boolean[:]),
    ('game_history', numba.uint64[::1]),  # Array of Zobrist keys for game history
    ('game_history_count', numba.int32),  # Valid number of entries in game_history
    ('ply_path_stack', numba.uint64[::1]), # Stack of Zobrist keys for current search path
    ('tt_generation', numba.uint8), # Current generation for TT aging
    ('counter_moves', numba.uint16[:, :]), # Counter moves table [64][64]
    ('move_stack', numba.uint16[::1]), # Stack of moves for current search path
    ('piece_stack', numba.int8[::1]), # Stack of piece types for current search path
    ('static_eval_stack', numba.int32[::1]), # Stack of static evaluations for improving heuristic

    ('move_scores', numba.int32[:, :]),
    ('moves_buffer', numba.uint16[:, :]),
    ('quiet_moves_tried', numba.uint16[:, :]),
    ('quiet_pieces_tried', numba.int8[:, :]),
    ('aggressor_cache', numba.int8[:, :]),
    ('victim_cache', numba.int8[:, :]),
    ('bad_captures', numba.uint16[:, :]),
    ('mp_stage', numba.int32[::1]),
    ('mp_current_idx', numba.int32[::1]),
    ('mp_captures_end', numba.int32[::1]),
    ('mp_quiets_end', numba.int32[::1]),
    ('mp_bad_captures_count', numba.int32[::1]),
    ('mp_bad_captures_idx', numba.int32[::1]),
    ('continuation_history', numba.int16[:, :, :, :, :]),
    ('pawn_history', numba.int16[:, :, :]),
    ('pawn_correction_history', numba.int16[:]),
    ('minor_correction_history', numba.int16[:]),
    ('non_pawn_correction_history_white', numba.int16[:]),
    ('non_pawn_correction_history_black', numba.int16[:]),
    ('butterfly_history', numba.int32[:, :]),
    ('capture_history', numba.int32[:, :, :]),
    ('enable_nmp', numba.boolean),
    ('enable_rfp', numba.boolean),
    ('enable_razoring', numba.boolean),
    ('enable_lmr', numba.boolean),
    ('enable_see_pruning', numba.boolean),
    ('enable_lmp', numba.boolean),
    ('enable_fp', numba.boolean),
    ('enable_probcut', numba.boolean),
    ('enable_multicut', numba.boolean),
    ('cutoff_cnt', numba.int32[::1]),
    ('reduction_stack', numba.int32[::1]),
    ('tt_pv_stack', numba.boolean[::1]),
    ('pawn_table_keys', numba.uint64[::1]),
    ('pawn_table_w_king_sq', numba.int8[::1]),
    ('pawn_table_b_king_sq', numba.int8[::1]),
    ('pawn_table_mg', numba.int32[::1]),
    ('pawn_table_eg', numba.int32[::1]),
    ('pawn_table_w_tropism', numba.int32[::1]),
    ('pawn_table_b_tropism', numba.int32[::1]),
    ('pawn_table_w_storm', numba.int32[::1]),
    ('pawn_table_b_storm', numba.int32[::1]),
]

@jitclass(search_context_spec)
class _SearchContextJIT:
    """
    用於在遞歸搜尋函數之間傳遞共享數據和狀態的上下文類別。
    
    Note: accumulator_stack removed (unused in Classical engine).
    Note: aggressor_cache/victim_cache shrunk from 65536 to 256 columns
          (indexed by move position in buffer, not move encoding).

    Attributes:
        transposition_table (numba.types.Array): 置換表陣列。
        killer_moves (numba.uint16[::1]): 殺手步表。
        pv_table (numba.uint16[:, :]): 主要變例（PV）表。
        history_table (numba.int32[:, :]): 歷史啟發表。
        nodes_searched (numba.uint64): 已搜尋的節點總數。
        end_time (numba.float64): 搜尋應結束的目標時間戳。
        stop_flag (numba.boolean[:]): 用於信號通知搜尋停止的標誌陣列（作為引用傳遞）。
        game_history (numba.uint64[::1]): 遊戲歷史記錄的 Zobrist 鍵列表。
        game_history_count (numba.int32): game_history 的有效條目數量。
        ply_path_stack (numba.uint64[::1]): 當前搜尋路徑的 Zobrist 鍵堆疊。
        tt_generation (numba.uint8): 當前 TT 世代。
        counter_moves (numba.uint16[:, :]): 反制走法表，索引為 [prev_src][prev_dst]。
        move_stack (numba.uint16[::1]): 當前搜尋路徑的走法堆疊，用於查找上一手棋。
        piece_stack (numba.int8[::1]): 當前搜尋路徑的棋子堆疊，用於歷史紀錄。
        static_eval_stack (numba.int32[::1]): 靜態評估值堆疊，用於判斷 Improving。

        continuation_history (numba.int16[:, :, :, :, :]): 連續歷史表。
        pawn_history (numba.int16[:, :, :]): 兵結構歷史表。
        pawn_correction_history (numba.int16[:]): 兵型修正歷史表。
        minor_correction_history (numba.int16[:]): 輕子修正歷史表。
        non_pawn_correction_history_white (numba.int16[:]): 白方非兵修正歷史表。
        non_pawn_correction_history_black (numba.int16[:]): 黑方非兵修正歷史表。
    """
    def __init__(self, transposition_table, killer_moves, pv_table, history_table, butterfly_history, continuation_history, capture_history, pawn_history, pawn_correction_history, minor_correction_history, non_pawn_correction_history_white, non_pawn_correction_history_black):
        """
        初始化搜尋上下文。

        Args:
            transposition_table: 預先分配的置換表。
            killer_moves: 預先分配的殺手步表。
            pv_table: 預先分配的 PV 表。
            history_table: 預先分配的歷史表。
            butterfly_history: 預先分配的蝴蝶歷史表。
            continuation_history: 預先分配的連續歷史表。
            capture_history: 預先分配的吃子歷史表。
            pawn_history: 預先分配的兵結構歷史表。
            pawn_correction_history: 預先分配的兵型修正歷史表。
            minor_correction_history: 預先分配的輕子修正歷史表。
            non_pawn_correction_history_white: 預先分配的白方非兵修正歷史表。
            non_pawn_correction_history_black: 預先分配的黑方非兵修正歷史表。
        """
        self.transposition_table = transposition_table
        self.killer_moves = killer_moves
        self.pv_table = pv_table
        self.history_table = history_table
        self.butterfly_history = butterfly_history
        self.continuation_history = continuation_history
        self.capture_history = capture_history
        self.pawn_history = pawn_history
        self.pawn_correction_history = pawn_correction_history
        self.minor_correction_history = minor_correction_history
        self.non_pawn_correction_history_white = non_pawn_correction_history_white
        self.non_pawn_correction_history_black = non_pawn_correction_history_black
        # accumulator_stack removed — not used in Classical engine
        self.nodes_searched = np.uint64(0)
        self.end_time = 0.0
        # 使用陣列來包裝布林值，以便可以作為引用傳遞並在外部修改
        self.stop_flag = np.array([False], dtype=np.bool_)
        
        # Repetition Detection and TT Aging
        self.game_history = np.zeros(1024, dtype=np.uint64) # Max 1024 moves in history
        self.game_history_count = 0
        self.ply_path_stack = np.zeros(MAX_PLY, dtype=np.uint64)
        self.tt_generation = 0
        
        # Counter Moves and Move Stack
        self.counter_moves = np.zeros((64, 64), dtype=np.uint16)
        self.move_stack = np.zeros(MAX_PLY, dtype=np.uint16)
        self.piece_stack = np.full(MAX_PLY, -1, dtype=np.int8)
        self.static_eval_stack = np.zeros(MAX_PLY, dtype=np.int32)
        
        self.move_scores = np.zeros((MAX_PLY, 256), dtype=np.int32)
        self.moves_buffer = np.zeros((MAX_PLY, 256), dtype=np.uint16)
        self.quiet_moves_tried = np.zeros((MAX_PLY, 256), dtype=np.uint16)
        self.quiet_pieces_tried = np.zeros((MAX_PLY, 256), dtype=np.int8)
        self.aggressor_cache = np.zeros((MAX_PLY, 256), dtype=np.int8)
        self.victim_cache = np.zeros((MAX_PLY, 256), dtype=np.int8)
        self.bad_captures = np.zeros((MAX_PLY, 256), dtype=np.uint16)
        # Move Picker state arrays
        self.mp_stage = np.zeros(MAX_PLY, dtype=np.int32)
        self.mp_current_idx = np.zeros(MAX_PLY, dtype=np.int32)
        self.mp_captures_end = np.zeros(MAX_PLY, dtype=np.int32)
        self.mp_quiets_end = np.zeros(MAX_PLY, dtype=np.int32)
        self.mp_bad_captures_count = np.zeros(MAX_PLY, dtype=np.int32)
        self.mp_bad_captures_idx = np.zeros(MAX_PLY, dtype=np.int32)
        self.enable_nmp = ENABLE_NMP
        self.enable_rfp = ENABLE_RFP
        self.enable_razoring = ENABLE_RAZORING
        self.enable_lmr = ENABLE_LMR
        self.enable_see_pruning = ENABLE_SHALLOW_SEE_PRUNING
        self.enable_lmp = ENABLE_LMP
        self.enable_fp = ENABLE_FP
        self.enable_probcut = ENABLE_PROBCUT
        self.enable_multicut = ENABLE_MULTICUT
        
        self.cutoff_cnt = np.zeros(MAX_PLY, dtype=np.int32)
        self.reduction_stack = np.zeros(MAX_PLY, dtype=np.int32)
        self.tt_pv_stack = np.zeros(MAX_PLY, dtype=np.bool_)
        
        # Pawn Hash Table arrays initialization (65536 entries, ~2.1MB total memory footprint)
        self.pawn_table_keys = np.full(65536, np.uint64(0xFFFFFFFFFFFFFFFF), dtype=np.uint64)
        self.pawn_table_w_king_sq = np.full(65536, -1, dtype=np.int8)
        self.pawn_table_b_king_sq = np.full(65536, -1, dtype=np.int8)
        self.pawn_table_mg = np.zeros(65536, dtype=np.int32)
        self.pawn_table_eg = np.zeros(65536, dtype=np.int32)
        self.pawn_table_w_tropism = np.zeros(65536, dtype=np.int32)
        self.pawn_table_b_tropism = np.zeros(65536, dtype=np.int32)
        self.pawn_table_w_storm = np.zeros(65536, dtype=np.int32)
        self.pawn_table_b_storm = np.zeros(65536, dtype=np.int32)




search_context_type = _SearchContextJIT.class_type.instance_type


def SearchContext(
    transposition_table, killer_moves, pv_table, history_table, butterfly_history,
    continuation_history, capture_history, pawn_history, pawn_correction_history,
    minor_correction_history, non_pawn_correction_history_white, non_pawn_correction_history_black
):
    return _SearchContextJIT(
        transposition_table, killer_moves, pv_table, history_table,
        butterfly_history, continuation_history, capture_history, pawn_history,
        pawn_correction_history, minor_correction_history,
        non_pawn_correction_history_white, non_pawn_correction_history_black
    )
