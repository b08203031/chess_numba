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
    ('key', np.uint32),        # 32-bit Zobrist hash key (compressed) / 32位 Zobrist 哈希鍵值
    ('score', np.int16),       # Evaluation score / 評估分數
    ('static_eval', np.int16), # Static evaluation / 靜態評估分數
    ('depth', np.uint8),       # Search depth / 搜尋深度
    ('flag', np.uint8),        # Node type flag (Exact, Alpha, Beta) / 節點類型標誌
    ('generation', np.uint8),  # Generation ID for aging / 用於老化的世代 ID
    ('best_move', np.uint16),  # Best move found at this node / 此節點找到的最佳移動
    ('is_pv', np.bool_),       # PV node flag / 主變例節點標誌
    ('padding', np.uint8, 2)   # Padding for exactly 16-byte alignment / 用於完全 16 字節對齊的填充 (2 bytes)
])

# Convert the NumPy dtype to a Numba-compatible type / 將 NumPy dtype 轉換為 Numba 兼容類型
numba_tt_entry_type = nb.from_dtype(tt_entry_dtype)

# Create a global, "empty" entry to return on a TT miss for Numba type stability
# 創建一個全局的「空」項目，以便在未命中置換表時返回，確保 Numba 類型穩定性
_EMPTY_TT_ENTRY = np.zeros(1, dtype=tt_entry_dtype)[0]
_EMPTY_TT_ENTRY['flag'] = TT_FLAG_NONE
_EMPTY_TT_ENTRY['static_eval'] = 32767 # Use maximum int16 to represent "no static eval recorded"
_EMPTY_TT_ENTRY['is_pv'] = False

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
        tt[i]['key'] = np.uint32(0)
        tt[i]['score'] = np.int16(0)
        tt[i]['static_eval'] = np.int16(32767)
        tt[i]['depth'] = np.uint8(0)
        tt[i]['flag'] = np.uint8(0)
        tt[i]['generation'] = np.uint8(0)
        tt[i]['best_move'] = np.uint16(0)
        tt[i]['is_pv'] = False

@nb.njit(cache=True)
def mul_hi64(a, b):
    a = np.uint64(a)
    b = np.uint64(b)
    aL = np.uint32(a)
    aH = np.uint32(a >> np.uint64(32))
    bL = np.uint32(b)
    bH = np.uint32(b >> np.uint64(32))
    c1 = np.uint64(aL) * np.uint64(bL) >> np.uint64(32)
    c2 = np.uint64(aH) * np.uint64(bL) + c1
    c3 = np.uint64(aL) * np.uint64(bH) + np.uint64(np.uint32(c2))
    return np.uint64(aH) * np.uint64(bH) + (c2 >> np.uint64(32)) + (c3 >> np.uint64(32))

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
    base_index = mul_hi64(zobrist_key, np.uint64(num_buckets)) * 4
    
    key32 = np.uint32(zobrist_key)
    for i in range(4):
        entry = tt[base_index + i]
        if entry['key'] == key32:
            return entry
            
    return _EMPTY_TT_ENTRY

@nb.njit(cache=True)
def store_tt(tt, zobrist_key, depth, score, static_eval, flag, best_move, current_generation, is_pv=False):
    """
    使用 Stockfish 風格的 4-Way Set Associative (四路組相聯) 策略存取置換表，並套用智能替換決策。
    
    Args:
        tt (np.ndarray): 置換表。
        zobrist_key (np.uint64): Zobrist 哈希鍵值。
        depth (int): 當前搜尋深度。
        score (int): 評估分數。
        static_eval (int): 靜態評估分數。
        flag (int): 節點標誌（EXACT, ALPHA, BETA）。
        best_move (int): 最佳移動。
        current_generation (int): 當前搜尋世代。
        is_pv (bool): 是否為主變例節點。
    """
    if len(tt) < 4:
        return
        
    num_buckets = len(tt) // 4
    base_index = mul_hi64(zobrist_key, np.uint64(num_buckets)) * 4
    key32 = np.uint32(zobrist_key)
    
    # 1. 尋找完全相同的局面 (Exact Match)
    for i in range(4):
        idx = base_index + i
        existing_entry = tt[idx]
        if existing_entry['key'] == key32:
            # 優先保留舊有最佳步，如果新的沒有的話
            final_best_move = np.uint16(best_move)
            if final_best_move == 0:
                final_best_move = existing_entry['best_move']
                
            # 決定是否覆寫：Stockfish 覆寫規則
            replace = False
            if flag == TT_FLAG_EXACT:
                replace = True
            elif existing_entry['generation'] != current_generation:
                replace = True
            else:
                # PV nodes get a +2 virtual depth bonus to protect them from being overwritten
                existing_pv_bonus = 2 if existing_entry['is_pv'] else 0
                new_pv_bonus = 2 if is_pv else 0
                if depth + new_pv_bonus > existing_entry['depth'] + existing_pv_bonus - 4:
                    replace = True
                
            if replace:
                tt[idx]['key'] = key32
                tt[idx]['depth'] = np.uint8(depth)
                tt[idx]['score'] = np.int16(score)
                if static_eval != 32767:
                    tt[idx]['static_eval'] = np.int16(static_eval)
                tt[idx]['flag'] = np.uint8(flag)
                tt[idx]['generation'] = np.uint8(current_generation)
                tt[idx]['best_move'] = final_best_move
                tt[idx]['is_pv'] = is_pv
            return # 找到並處理完相符局面，直接退出
            
    # 2. 如果沒有找到完全相同的局面 (雜湊碰撞 Collision)，進入替換評估 (Replacement Strategy)
    replace_idx = base_index
    min_replace_value = 999999
    
    for i in range(4):
        idx = base_index + i
        existing_entry = tt[idx]
        
        # 計算相對老化程度
        age = (current_generation - existing_entry['generation'] + 256) % 256
        if age > 15:
            age = 15  # 簡化的世代老化權重
            
        # PV nodes get a virtual depth bonus of 2, making them harder to replace
        existing_pv_bonus = 2 if existing_entry['is_pv'] else 0
        
        # 由於深度是無符號整數(uint8)，計算 replace_value 時需轉為有符號整數(int)處理負數情況
        replace_value = int(existing_entry['depth']) + existing_pv_bonus - age * 8
        
        # 找到取代價值最小的項目（包含空位，因其深度為 0，能自然獲得極低的 replace_value）
        if replace_value < min_replace_value:
            min_replace_value = replace_value
            replace_idx = idx

    # 執行最終替換 (因為非吻合點，不保留舊的最佳步)
    tt[replace_idx]['key'] = key32
    tt[replace_idx]['depth'] = np.uint8(depth)
    tt[replace_idx]['score'] = np.int16(score)
    tt[replace_idx]['static_eval'] = np.int16(static_eval)
    tt[replace_idx]['flag'] = np.uint8(flag)
    tt[replace_idx]['generation'] = np.uint8(current_generation)
    tt[replace_idx]['best_move'] = np.uint16(best_move)
    tt[replace_idx]['is_pv'] = is_pv
