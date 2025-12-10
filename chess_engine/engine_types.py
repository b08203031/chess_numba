# chess_engine/engine_types.py
import numba
import numpy as np
from chess_engine.transposition_table import numba_tt_entry_type
from chess_engine.constants import MAX_PLY

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

unmake_info_signature = numba.types.Tuple([
    numba.int8, numba.uint8, numba.uint8, numba.uint8, numba.uint64
])

# --- Search Context / 搜尋上下文 ---
from numba.experimental import jitclass

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
    ('static_eval_stack', numba.int32[::1]), # Stack of static evaluations for improving heuristic
]

@jitclass(search_context_spec)
class SearchContext:
    """
    用於在遞歸搜尋函數之間傳遞共享數據和狀態的上下文類別。
    
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
        static_eval_stack (numba.int32[::1]): 靜態評估值堆疊，用於判斷 Improving。
    """
    def __init__(self, transposition_table, killer_moves, pv_table, history_table):
        """
        初始化搜尋上下文。

        Args:
            transposition_table: 預先分配的置換表。
            killer_moves: 預先分配的殺手步表。
            pv_table: 預先分配的 PV 表。
            history_table: 預先分配的歷史表。
        """
        self.transposition_table = transposition_table
        self.killer_moves = killer_moves
        self.pv_table = pv_table
        self.history_table = history_table
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
        self.static_eval_stack = np.zeros(MAX_PLY, dtype=np.int32)

search_context_type = SearchContext.class_type.instance_type
