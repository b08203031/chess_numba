# chess_engine/transposition_table.py
import numpy as np
import numba as nb
from chess_engine.constants import TT_SIZE_MB

# 1. Define constants for TT entry flags / 定義置換表項目標誌的常量
TT_FLAG_NONE = 0
TT_FLAG_EXACT = 1  # Exact score (score is between alpha and beta) / 精確分數（分數在 alpha 和 beta 之間）
TT_FLAG_ALPHA = 2  # Upper bound (score <= alpha), an ALL-node / 上界（分數 <= alpha），所有節點（ALL-node）
TT_FLAG_BETA = 3   # Lower bound (score >= beta), a CUT-node / 下界（分數 >= beta），截斷節點（CUT-node）

# 2. Define the data type (dtype) for a transposition table entry / 定義置換表項目的數據類型
# Total size: 8 (key) + 2 (score) + 1 (depth) + 1 (flag) + 1 (generation) + 2 (best_move) + 1 (padding) = 16 bytes
# Using 16 bytes ensures optimal memory alignment on 64-bit systems.
tt_entry_dtype = np.dtype([
    ('key', np.uint64),       # Zobrist hash key / Zobrist 哈希鍵值
    ('score', np.int16),      # Evaluation score / 評估分數
    ('depth', np.uint8),      # Search depth / 搜尋深度
    ('flag', np.uint8),       # Node type flag (Exact, Alpha, Beta) / 節點類型標誌
    ('generation', np.uint8), # Generation ID for aging / 用於老化的世代 ID
    ('best_move', np.uint16), # Best move found at this node / 此節點找到的最佳移動
    ('padding', np.uint8)     # Padding for 16-byte alignment / 用於 16 字節對齊的填充
])

# Convert the NumPy dtype to a Numba-compatible type / 將 NumPy dtype 轉換為 Numba 兼容類型
numba_tt_entry_type = nb.from_dtype(tt_entry_dtype)

# Create a global, "empty" entry to return on a TT miss for Numba type stability
# 創建一個全局的「空」項目，以便在未命中置換表時返回，確保 Numba 類型穩定性
_EMPTY_TT_ENTRY = np.zeros(1, dtype=tt_entry_dtype)[0]
_EMPTY_TT_ENTRY['flag'] = TT_FLAG_NONE

def create_transposition_table(size_mb):
    """
    根據指定的 MB 大小初始化置換表。
    為了優化效能，將項目數量限制為 2 的冪次方，以便使用位元運算進行索引。
    
    Args:
        size_mb (int): 置換表的大小（MB）。
        
    Returns:
        np.ndarray: 置換表陣列。
    """
    entry_size_bytes = tt_entry_dtype.itemsize
    total_bytes = size_mb * 1024 * 1024
    max_entries = total_bytes // entry_size_bytes
    
    # Find the largest power of 2 less than or equal to max_entries
    # 找到小於或等於 max_entries 的最大 2 的冪次方
    if max_entries <= 0:
        num_entries = 0
    else:
        num_entries = 1 << (max_entries.bit_length() - 1)
        
    transposition_table = np.zeros(num_entries, dtype=tt_entry_dtype)
    return transposition_table

@nb.njit(cache=True)
def clear_transposition_table(tt):
    """
    清除置換表中的所有項目，將每個欄位設置為零。
    
    Args:
        tt (np.ndarray): 置換表。
    """
    for i in range(len(tt)):
        tt[i]['key'] = np.uint64(0)
        tt[i]['score'] = np.int16(0)
        tt[i]['depth'] = np.uint8(0)
        tt[i]['flag'] = np.uint8(0)
        tt[i]['generation'] = np.uint8(0)
        tt[i]['best_move'] = np.uint16(0)

@nb.njit(cache=True)
def probe_tt(tt, zobrist_key):
    """
    在置換表中查找項目。如果鍵值匹配，則返回該項目。
    使用位元與運算 (&) 代替取模運算 (%) 以提高效能。
    
    Args:
        tt (np.ndarray): 置換表。
        zobrist_key (np.uint64): 當前局面的 Zobrist 哈希鍵值。
        
    Returns:
        tuple: 置換表項目。如果未命中，則返回空項目。
    """
    if len(tt) == 0:
        return _EMPTY_TT_ENTRY
        
    index = zobrist_key & np.uint64(len(tt) - 1)
    entry = tt[index]
    if entry['key'] == zobrist_key:
        return entry
    else:
        return _EMPTY_TT_ENTRY

@nb.njit(cache=True)
def store_tt(tt, zobrist_key, depth, score, flag, best_move, current_generation):
    """
    使用深度優先替換策略將項目存儲在置換表中。
    
    Args:
        tt (np.ndarray): 置換表。
        zobrist_key (np.uint64): Zobrist 哈希鍵值。
        depth (int): 當前搜尋深度。
        score (int): 評估分數。
        flag (int): 節點標誌（EXACT, ALPHA, BETA）。
        best_move (int): 最佳移動。
        current_generation (int): 當前搜尋世代。
    """
    if len(tt) == 0:
        return
        
    index = zobrist_key & np.uint64(len(tt) - 1)
    existing_entry = tt[index]

    # Replacement Strategy / 替換策略:
    # 1. If the key matches (update same position), replace if new depth is >= existing depth OR existing entry is old.
    # 2. If the key is different (collision), replace if new depth is >= existing depth OR existing entry is old.
    # Simply put: If existing entry is from an old generation, we always replace it (it's effectively empty/stale).
    
    replace = False
    
    if existing_entry['generation'] != current_generation:
        # Existing entry is old, replace it!
        replace = True
    else:
        # Entry is from current generation, apply standard depth check
        if depth >= existing_entry['depth']:
            replace = True

    if replace:
        # Best move preservation: If we are not storing a new best move, but the keys match,
        # we keep the best move already in the table.
        final_best_move = np.uint16(best_move)
        if final_best_move == 0 and existing_entry['key'] == zobrist_key:
            final_best_move = existing_entry['best_move']

        tt[index]['key'] = zobrist_key
        tt[index]['depth'] = np.uint8(depth)
        tt[index]['score'] = np.int16(score)
        tt[index]['flag'] = np.uint8(flag)
        tt[index]['generation'] = np.uint8(current_generation)
        tt[index]['best_move'] = final_best_move
