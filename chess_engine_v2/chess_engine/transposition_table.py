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
    在置換表 (Bucket=4) 中查找項目。
    使用位元與運算 (&) 提取雜湊桶索引以提高效能。
    
    Args:
        tt (np.ndarray): 置換表。
        zobrist_key (np.uint64): 當前局面的 Zobrist 哈希鍵值。
        
    Returns:
        tuple: 置換表項目。如果未命中，則返回空項目。
    """
    if len(tt) < 4:
        return _EMPTY_TT_ENTRY
        
    num_buckets = len(tt) // 4
    base_index = (zobrist_key & np.uint64(num_buckets - 1)) * 4
    
    for i in range(4):
        entry = tt[base_index + i]
        if entry['key'] == zobrist_key:
            return entry
            
    return _EMPTY_TT_ENTRY

@nb.njit(cache=True)
def store_tt(tt, zobrist_key, depth, score, flag, best_move, current_generation):
    """
    使用 Stockfish 風格的 4-Way Set Associative (四路組相聯) 策略存取置換表，並套用智能替換決策。
    
    Args:
        tt (np.ndarray): 置換表。
        zobrist_key (np.uint64): Zobrist 哈希鍵值。
        depth (int): 當前搜尋深度。
        score (int): 評估分數。
        flag (int): 節點標誌（EXACT, ALPHA, BETA）。
        best_move (int): 最佳移動。
        current_generation (int): 當前搜尋世代。
    """
    if len(tt) < 4:
        return
        
    num_buckets = len(tt) // 4
    base_index = (zobrist_key & np.uint64(num_buckets - 1)) * 4
    
    # 1. 尋找完全相同的局面 (Exact Match)
    for i in range(4):
        idx = base_index + i
        existing_entry = tt[idx]
        if existing_entry['key'] == zobrist_key:
            # 決定是否覆寫：舊世代或當前世代但深度更深
            replace = False
            if existing_entry['generation'] != current_generation:
                replace = True
            elif depth >= existing_entry['depth']:
                replace = True
                
            if replace:
                final_best_move = np.uint16(best_move)
                if final_best_move == 0:
                    final_best_move = existing_entry['best_move']

                tt[idx]['key'] = zobrist_key
                tt[idx]['depth'] = np.uint8(depth)
                tt[idx]['score'] = np.int16(score)
                tt[idx]['flag'] = np.uint8(flag)
                tt[idx]['generation'] = np.uint8(current_generation)
                tt[idx]['best_move'] = final_best_move
            return # 找到並處理完相符局面，直接退出
            
    # 2. 如果沒有找到完全相同的局 (雜湊碰撞 Collision)，進入替換評估 (Replacement Strategy)
    replace_idx = base_index
    min_depth = 255
    
    for i in range(4):
        idx = base_index + i
        existing_entry = tt[idx]
        
        # 優先權 1: 尋找完全沒有資料的空位
        if existing_entry['key'] == 0:
            replace_idx = idx
            break
            
        # 優先權 2: 尋找舊世代 (過期) 的資料
        if existing_entry['generation'] != current_generation:
            replace_idx = idx
            break
            
        # 優先權 3: 同世代中尋找深度最淺的拿來犧牲
        if existing_entry['depth'] < min_depth:
            min_depth = existing_entry['depth']
            replace_idx = idx

    # 執行最終替換 (因為非吻合點，不保留舊的最佳步)
    tt[replace_idx]['key'] = zobrist_key
    tt[replace_idx]['depth'] = np.uint8(depth)
    tt[replace_idx]['score'] = np.int16(score)
    tt[replace_idx]['flag'] = np.uint8(flag)
    tt[replace_idx]['generation'] = np.uint8(current_generation)
    tt[replace_idx]['best_move'] = np.uint16(best_move)
