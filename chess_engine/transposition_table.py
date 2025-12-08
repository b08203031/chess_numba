# chess_engine/transposition_table.py
import numpy as np
import numba as nb
from chess_engine.constants import TT_SIZE_MB

# 1. Define constants for TT entry flags / 定義置換表項目標誌的常量
TT_FLAG_NONE = 0
TT_FLAG_EXACT = 1  # Exact score (score is between alpha and beta) / 精確分數（分數在 alpha 和 beta 之間）
TT_FLAG_ALPHA = 2  # Upper bound (score <= alpha), an ALL-node / 上界（分數 <= alpha），所有節點（ALL-node）
TT_FLAG_BETA = 3   # Lower bound (score >= beta), a CUT-node / 下界（分數 >= beta），截斷節點（CUT-node）

# TT Bucketing Constants
TT_CLUSTER_SIZE = 4 # 4 entries per cluster to reduce collisions

# 2. Define the data type (dtype) for a transposition table entry / 定義置換表項目的數據類型
# Size: 8 (key) + 2 (score) + 2 (move) + 1 (depth) + 1 (flag) + 1 (gen) + 1 (pad) = 16 bytes
tt_entry_dtype = np.dtype([
    ('key', np.uint64),       # Zobrist hash key / Zobrist 哈希鍵值
    ('score', np.int16),      # Evaluation score / 評估分數
    ('best_move', np.uint16), # Best move found at this node / 此節點找到的最佳移動
    ('depth', np.uint8),      # Search depth / 搜尋深度
    ('flag', np.uint8),       # Node type flag (Exact, Alpha, Beta) / 節點類型標誌
    ('generation', np.uint8), # Generation ID for aging / 用於老化的世代 ID
    ('pad', np.uint8)         # Padding for alignment (unused)
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
    Ensure total entries is a multiple of TT_CLUSTER_SIZE.
    
    Args:
        size_mb (int): 置換表的大小（MB）。
        
    Returns:
        np.ndarray: 置換表陣列。
    """
    entry_size_bytes = tt_entry_dtype.itemsize # 16 bytes
    num_entries = (size_mb * 1024 * 1024) // entry_size_bytes

    # Align to cluster size
    num_entries = (num_entries // TT_CLUSTER_SIZE) * TT_CLUSTER_SIZE

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
        tt[i]['best_move'] = np.uint16(0)
        tt[i]['depth'] = np.uint8(0)
        tt[i]['flag'] = np.uint8(0)
        tt[i]['generation'] = np.uint8(0)

@nb.njit(cache=True)
def relative_age(entry_gen, current_gen):
    """
    Calculate the age of an entry relative to the current generation.
    Handles wrapping logic (0-255).
    """
    diff = (int(current_gen) - int(entry_gen)) & 0xFF
    return diff

@nb.njit(cache=True)
def probe_tt(tt, zobrist_key):
    """
    在置換表中查找項目。
    Uses Bucket System (Cluster of 4).
    
    Args:
        tt (np.ndarray): 置換表。
        zobrist_key (np.uint64): 當前局面的 Zobrist 哈希鍵值。
        
    Returns:
        tuple: 置換表項目。如果未命中，則返回空項目。
    """
    num_clusters = len(tt) // TT_CLUSTER_SIZE
    cluster_idx = (zobrist_key % num_clusters) * TT_CLUSTER_SIZE

    # Check all 4 entries in the cluster
    for i in range(TT_CLUSTER_SIZE):
        entry = tt[cluster_idx + i]
        if entry['key'] == zobrist_key:
            return entry

    return _EMPTY_TT_ENTRY

@nb.njit(cache=True)
def store_tt(tt, zobrist_key, depth, score, flag, best_move, current_generation):
    """
    Store an entry in the transposition table.
    Uses Bucket System (Cluster of 4).
    Prioritizes preserving deep search results over new shallow results to prevent 'Amnesia'.
    
    Args:
        tt (np.ndarray): 置換表。
        zobrist_key (np.uint64): Zobrist 哈希鍵值。
        depth (int): 當前搜尋深度。
        score (int): 評估分數。
        flag (int): 節點標誌（EXACT, ALPHA, BETA）。
        best_move (int): 最佳移動。
        current_generation (int): 當前搜尋世代。
    """
    num_clusters = len(tt) // TT_CLUSTER_SIZE
    cluster_idx = (zobrist_key % num_clusters) * TT_CLUSTER_SIZE

    target_idx = -1
    found_existing = False

    # Phase 1: Try to find exact match
    for i in range(TT_CLUSTER_SIZE):
        if tt[cluster_idx + i]['key'] == zobrist_key:
            target_idx = i
            found_existing = True
            break

    # Phase 2: If no match, find best replacement candidate (Victim)
    if target_idx == -1:
        best_replace_val = 10000 # Large initial value
        victim_idx = 0

        for i in range(TT_CLUSTER_SIZE):
            entry = tt[cluster_idx + i]

            # If empty, use it immediately
            if entry['key'] == 0:
                victim_idx = i
                break

            age = relative_age(entry['generation'], current_generation)
            # Replacement Score: High Depth = Valuable. High Age = Disposable.
            # Value = Depth - (Age * 2)
            # We want to replace the entry with MIN Value.
            replace_val = int(entry['depth']) - (age * 2)

            if replace_val < best_replace_val:
                best_replace_val = replace_val
                victim_idx = i

        target_idx = victim_idx

    # Phase 3: Write data to target_idx
    
    entry = tt[cluster_idx + target_idx]
    should_replace = False
    
    if not found_existing:
        # It's a collision replacement (victim) or empty slot -> Always write
        should_replace = True
    else:
        # Same key update (found_existing == True)
        # CRITICAL FIX: Only replace if the new depth is better or equal.
        # Do NOT replace just because it's a new generation (Age > 0), as this causes "Amnesia"
        # of deep search results when visited by shallow searches in new frames.

        if depth >= entry['depth']:
            should_replace = True
        elif flag == TT_FLAG_EXACT and entry['flag'] != TT_FLAG_EXACT:
             # Prefer Exact score if depths are close?
             # Safety: Stick to depth. If depth is lower, we trust the deeper search more.
             pass

    if should_replace:
        tt[cluster_idx + target_idx]['key'] = zobrist_key
        tt[cluster_idx + target_idx]['depth'] = np.uint8(depth)
        tt[cluster_idx + target_idx]['score'] = np.int16(score)
        tt[cluster_idx + target_idx]['flag'] = np.uint8(flag)
        tt[cluster_idx + target_idx]['generation'] = np.uint8(current_generation)
        tt[cluster_idx + target_idx]['best_move'] = np.uint16(best_move)
