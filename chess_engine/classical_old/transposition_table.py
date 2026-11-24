# chess_engine/transposition_table.py
import numpy as np
import numba as nb
from chess_engine.classical_old.constants import TT_SIZE_MB, MATE_IN_MAX_PLY

# 1. Define constants for TT entry flags / 定義置換表項目標誌的常量
TT_FLAG_NONE = 0
TT_FLAG_EXACT = 1  # Exact score (score is between alpha and beta) / 精確分數（分數在 alpha 和 beta 之間）
TT_FLAG_ALPHA = 2  # Upper bound (score <= alpha), an ALL-node / 上界（分數 <= alpha），所有節點（ALL-node）
TT_FLAG_BETA = 3   # Lower bound (score >= beta), a CUT-node / 下界（分數 >= beta），截斷節點（CUT-node）

# 2. Define the data type (dtype) for a transposition table entry / 定義置換表項目的數據類型
# Total size: 4 (key) + 2 (score) + 2 (static_eval) + 1 (depth) + 1 (flag) + 1 (generation) + 1 (is_pv) + 2 (best_move) + 2 (padding) = 16 bytes
# Using 16 bytes ensures optimal memory alignment on 64-bit systems.
tt_entry_dtype = np.dtype([
    ('key', np.uint32),        # 32-bit Zobrist hash key (compressed) / 32位 Zobrist 哈希鍵值
    ('score', np.int16),       # Evaluation score / 評估分數
    ('static_eval', np.int16), # Static evaluation / 靜態評估分數
    ('depth', np.uint8),       # Search depth / 搜尋深度
    ('flag', np.uint8),        # Node type flag (Exact, Alpha, Beta) / 節點類型標誌
    ('generation', np.uint8),  # Generation ID for aging / 用於老化的世代 ID
    ('is_pv', np.bool_),       # PV node flag / 主變例節點標誌
    ('best_move', np.uint16),  # Best move found at this node / 此節點找到的最佳移動
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
    transposition_table['static_eval'] = 32767
    return transposition_table

@nb.njit(cache=True, nogil=True, boundscheck=False, fastmath=True, inline='always')
def _compute_tt_checksum(key32, score, static_eval, depth, flag, best_move, is_pv):
    """
    Compute 16-bit lockless XOR checksum for a TT entry.
    Stored in padding[0] (low byte) and padding[1] (high byte) to detect torn reads
    under concurrent Lazy SMP access without changing the 16-byte entry layout.
    """
    k = np.uint32(key32)
    k_fold = np.uint16(k & np.uint32(0xFFFF)) ^ np.uint16(k >> np.uint32(16))
    s_u16 = np.uint16(np.int16(score))
    se_u16 = np.uint16(np.int16(static_eval))
    df_u16 = np.uint16(np.uint8(depth)) | (np.uint16(np.uint8(flag)) << np.uint16(8))
    bm_u16 = np.uint16(best_move)
    pv_u16 = np.uint16(1) if is_pv else np.uint16(0)
    return k_fold ^ s_u16 ^ se_u16 ^ df_u16 ^ bm_u16 ^ pv_u16 ^ np.uint16(0x9E37)


@nb.njit(cache=True, nogil=True)
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
        tt[i]['padding'][0] = np.uint8(0)
        tt[i]['padding'][1] = np.uint8(0)

@nb.njit(cache=True, nogil=True, boundscheck=False, fastmath=True)
def hashfull(tt, current_generation):
    """
    計算置換表的使用率（以千分比表示，0 到 1000）。
    掃描前 1000 個 bucket (共 4000 個 entry) 中，
    屬於當前世代且已被佔用的項目數量。
    """
    num_buckets = len(tt) >> 2
    scan_buckets = min(1000, num_buckets)
    if scan_buckets <= 0:
        return 0
        
    cnt = 0
    for i in range(scan_buckets):
        base = i << 2
        for j in range(4):
            entry = tt[base + j]
            if entry['flag'] != TT_FLAG_NONE:
                if entry['generation'] == current_generation:
                    cnt += 1
                    
    return (cnt * 1000) // (scan_buckets << 2)

@nb.njit(cache=True, nogil=True, boundscheck=False, fastmath=True)
def penalize_tt(tt, zobrist_key, penalty):
    """
    降低與 zobrist_key 匹配的置換表項目的深度。
    """
    if len(tt) < 4:
        return
    num_buckets_minus_1 = np.uint64((len(tt) >> 2) - 1)
    base_index = np.int64(((zobrist_key >> np.uint64(32)) & num_buckets_minus_1) << np.uint64(2))
    key32 = np.uint32(zobrist_key)
    for i in range(4):
        idx = base_index + i
        e_key = np.uint32(tt[idx]['key'])
        if e_key == key32:
            e_score = np.int16(tt[idx]['score'])
            e_static_eval = np.int16(tt[idx]['static_eval'])
            e_depth = np.uint8(tt[idx]['depth'])
            e_flag = np.uint8(tt[idx]['flag'])
            e_gen = np.uint8(tt[idx]['generation'])
            e_best_move = np.uint16(tt[idx]['best_move'])
            e_is_pv = bool(tt[idx]['is_pv'])
            p0 = np.uint8(tt[idx]['padding'][0])
            p1 = np.uint8(tt[idx]['padding'][1])
            if e_flag == TT_FLAG_NONE:
                continue
            if e_flag > TT_FLAG_BETA:
                break
            stored_chk = np.uint16(p0) | (np.uint16(p1) << np.uint16(8))
            expected_chk = _compute_tt_checksum(
                key32,
                e_score,
                e_static_eval,
                e_depth,
                e_flag,
                e_best_move,
                e_is_pv,
            )
            if stored_chk != expected_chk:
                break
            new_depth_i = int(e_depth) - int(penalty)
            if new_depth_i < 0:
                new_depth_i = 0
            new_depth = np.uint8(new_depth_i)
            if new_depth != e_depth:
                chk = _compute_tt_checksum(
                    key32,
                    e_score,
                    e_static_eval,
                    new_depth,
                    e_flag,
                    e_best_move,
                    e_is_pv,
                )
                tt[idx]['key'] = key32
                tt[idx]['depth'] = new_depth
                tt[idx]['score'] = e_score
                tt[idx]['static_eval'] = e_static_eval
                tt[idx]['flag'] = e_flag
                tt[idx]['generation'] = e_gen
                tt[idx]['best_move'] = e_best_move
                tt[idx]['is_pv'] = e_is_pv
                tt[idx]['padding'][0] = np.uint8(chk & np.uint16(0xFF))
                tt[idx]['padding'][1] = np.uint8(chk >> np.uint16(8))
            break

@nb.njit(cache=True, nogil=True, boundscheck=False, fastmath=True, inline='always')
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
        
    num_buckets_minus_1 = np.uint64((len(tt) >> 2) - 1)
    # The TT is allocated as a power-of-two number of 4-entry buckets.
    # Use the high 32 bits for the bucket and the low 32 bits as the lock key:
    # this avoids the expensive 64x64 mul_hi index computation on every probe.
    base_index = np.int64(((zobrist_key >> np.uint64(32)) & num_buckets_minus_1) << np.uint64(2))
    
    key32 = np.uint32(zobrist_key)
    for i in range(4):
        entry = tt[base_index + i]
        if entry['key'] == key32:
            flag = entry['flag']
            if flag == TT_FLAG_NONE:
                return entry
            if flag > TT_FLAG_BETA:
                return _EMPTY_TT_ENTRY
            stored_chk = np.uint16(entry['padding'][0]) | (np.uint16(entry['padding'][1]) << np.uint16(8))
            expected_chk = _compute_tt_checksum(
                key32,
                entry['score'],
                entry['static_eval'],
                entry['depth'],
                flag,
                entry['best_move'],
                entry['is_pv'],
            )
            if stored_chk != expected_chk:
                return _EMPTY_TT_ENTRY
            return entry

    return _EMPTY_TT_ENTRY


@nb.njit(cache=True, nogil=True, boundscheck=False, fastmath=True, inline='always')
def probe_tt_snapshot(tt, zobrist_key):
    """
    Lockless thread-safe TT probe that snapshots entry fields into local scalar registers
    and verifies the 16-bit XOR checksum before returning.
    Returns (flag, depth, score, static_eval, is_pv, best_move).
    """
    if len(tt) < 4:
        return np.uint8(TT_FLAG_NONE), np.int32(0), np.int32(0), np.int16(32767), False, np.uint16(0)

    num_buckets_minus_1 = np.uint64((len(tt) >> 2) - 1)
    base_index = np.int64(((zobrist_key >> np.uint64(32)) & num_buckets_minus_1) << np.uint64(2))
    key32 = np.uint32(zobrist_key)

    for i in range(4):
        idx = base_index + i
        e_key = np.uint32(tt[idx]['key'])
        if e_key == key32:
            e_score = np.int16(tt[idx]['score'])
            e_static_eval = np.int16(tt[idx]['static_eval'])
            e_depth = np.uint8(tt[idx]['depth'])
            e_flag = np.uint8(tt[idx]['flag'])
            e_best_move = np.uint16(tt[idx]['best_move'])
            e_is_pv = bool(tt[idx]['is_pv'])
            p0 = np.uint8(tt[idx]['padding'][0])
            p1 = np.uint8(tt[idx]['padding'][1])
            if e_flag == TT_FLAG_NONE:
                continue
            if e_flag > TT_FLAG_BETA:
                return np.uint8(TT_FLAG_NONE), np.int32(0), np.int32(0), np.int16(32767), False, np.uint16(0)
            stored_chk = np.uint16(p0) | (np.uint16(p1) << np.uint16(8))
            expected_chk = _compute_tt_checksum(
                key32,
                e_score,
                e_static_eval,
                e_depth,
                e_flag,
                e_best_move,
                e_is_pv,
            )
            if stored_chk != expected_chk:
                return np.uint8(TT_FLAG_NONE), np.int32(0), np.int32(0), np.int16(32767), False, np.uint16(0)
            return e_flag, np.int32(e_depth), np.int32(e_score), e_static_eval, e_is_pv, e_best_move

    return np.uint8(TT_FLAG_NONE), np.int32(0), np.int32(0), np.int16(32767), False, np.uint16(0)


@nb.njit(cache=True, nogil=True, boundscheck=False, fastmath=True)
def probe_tt_best_move(tt, zobrist_key):
    """Pure-Python-safe TT lookup returning scalar (flag, best_move) without np.record boxing."""
    if len(tt) < 4:
        return np.uint8(TT_FLAG_NONE), np.uint16(0)

    zk = np.uint64(zobrist_key)
    num_buckets_minus_1 = np.uint64((len(tt) >> 2) - 1)
    base_index = np.int64(((zk >> np.uint64(32)) & num_buckets_minus_1) << np.uint64(2))
    key32 = np.uint32(zk)

    for i in range(4):
        idx = base_index + i
        e_key = np.uint32(tt[idx]['key'])
        if e_key == key32:
            e_score = np.int16(tt[idx]['score'])
            e_static_eval = np.int16(tt[idx]['static_eval'])
            e_depth = np.uint8(tt[idx]['depth'])
            e_flag = np.uint8(tt[idx]['flag'])
            e_best_move = np.uint16(tt[idx]['best_move'])
            e_is_pv = bool(tt[idx]['is_pv'])
            p0 = np.uint8(tt[idx]['padding'][0])
            p1 = np.uint8(tt[idx]['padding'][1])
            if e_flag == TT_FLAG_NONE:
                return np.uint8(TT_FLAG_NONE), e_best_move
            if e_flag > TT_FLAG_BETA:
                return np.uint8(TT_FLAG_NONE), np.uint16(0)
            stored_chk = np.uint16(p0) | (np.uint16(p1) << np.uint16(8))
            expected_chk = _compute_tt_checksum(
                key32,
                e_score,
                e_static_eval,
                e_depth,
                e_flag,
                e_best_move,
                e_is_pv,
            )
            if stored_chk != expected_chk:
                return np.uint8(TT_FLAG_NONE), np.uint16(0)
            return e_flag, e_best_move

    return np.uint8(TT_FLAG_NONE), np.uint16(0)

@nb.njit(cache=True, nogil=True, boundscheck=False, fastmath=True)
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
        
    num_buckets_minus_1 = np.uint64((len(tt) >> 2) - 1)
    # Match probe_tt(): power-of-two bucket mask using high key bits.
    base_index = np.int64(((zobrist_key >> np.uint64(32)) & num_buckets_minus_1) << np.uint64(2))
    key32 = np.uint32(zobrist_key)
    gen_u8 = np.uint8(current_generation)
    
    # 1. 尋找完全相同的局面 (Exact Match)
    for i in range(4):
        idx = base_index + i
        e_key = np.uint32(tt[idx]['key'])
        e_flag = np.uint8(tt[idx]['flag'])
        if e_key == key32 and e_flag != TT_FLAG_NONE:
            e_score = np.int16(tt[idx]['score'])
            e_static_eval = np.int16(tt[idx]['static_eval'])
            e_depth = np.uint8(tt[idx]['depth'])
            e_gen = np.uint8(tt[idx]['generation'])
            e_best_move = np.uint16(tt[idx]['best_move'])
            e_is_pv = bool(tt[idx]['is_pv'])
            p0 = np.uint8(tt[idx]['padding'][0])
            p1 = np.uint8(tt[idx]['padding'][1])

            stored_chk = np.uint16(p0) | (np.uint16(p1) << np.uint16(8))
            expected_chk = _compute_tt_checksum(
                key32,
                e_score,
                e_static_eval,
                e_depth,
                e_flag,
                e_best_move,
                e_is_pv,
            )
            e_valid = (e_flag <= TT_FLAG_BETA) and (stored_chk == expected_chk)

            if not e_valid:
                replace = True
                final_best_move = np.uint16(best_move)
            else:
                # 優先保留舊有最佳步，如果新的沒有的話
                final_best_move = np.uint16(best_move)
                if final_best_move == 0:
                    final_best_move = e_best_move
                    
                # 決定是否覆寫：Stockfish 覆寫規則（Bug #3: depth=0 不可覆寫深層有效條目）
                replace = False
                if flag == TT_FLAG_EXACT and (depth > 0 or e_depth == 0):
                    replace = True
                elif e_gen != gen_u8:
                    replace = True
                else:
                    # PV nodes get a +2 virtual depth bonus to protect them from being overwritten
                    existing_pv_bonus = 2 if e_is_pv else 0
                    new_pv_bonus = 2 if is_pv else 0
                    if int(depth) + new_pv_bonus > int(e_depth) + existing_pv_bonus - 4:
                        replace = True
                
            if replace:
                d_u8 = np.uint8(depth)
                s_i16 = np.int16(score)
                if static_eval != 32767:
                    se_i16 = np.int16(static_eval)
                elif e_valid:
                    se_i16 = e_static_eval
                else:
                    se_i16 = np.int16(32767)
                f_u8 = np.uint8(flag)
                pv_b = bool(is_pv)
                chk = _compute_tt_checksum(key32, s_i16, se_i16, d_u8, f_u8, final_best_move, pv_b)
                tt[idx]['key'] = key32
                tt[idx]['depth'] = d_u8
                tt[idx]['score'] = s_i16
                tt[idx]['static_eval'] = se_i16
                tt[idx]['flag'] = f_u8
                tt[idx]['generation'] = gen_u8
                tt[idx]['best_move'] = final_best_move
                tt[idx]['is_pv'] = pv_b
                tt[idx]['padding'][0] = np.uint8(chk & np.uint16(0xFF))
                tt[idx]['padding'][1] = np.uint8(chk >> np.uint16(8))
            else:
                new_se = e_static_eval
                new_depth = e_depth
                new_bm = e_best_move
                updated = False
                if static_eval != 32767 and e_static_eval == 32767:
                    new_se = np.int16(static_eval)
                    updated = True
                bm_u16 = np.uint16(best_move)
                if bm_u16 != 0 and flag != TT_FLAG_ALPHA:
                    if e_best_move == 0 or (depth > 0 and e_flag == TT_FLAG_ALPHA):
                        if new_bm != bm_u16:
                            new_bm = bm_u16
                            updated = True
                # Secondary aging. Important for elementary mate finding.
                if e_depth >= 8 and e_flag != TT_FLAG_EXACT:
                    if abs(np.int32(e_score)) >= MATE_IN_MAX_PLY:
                        aged_depth = np.uint8(int(e_depth) - 1)
                        if aged_depth != new_depth:
                            new_depth = aged_depth
                            updated = True
                if updated:
                    chk = _compute_tt_checksum(
                        key32,
                        e_score,
                        new_se,
                        new_depth,
                        e_flag,
                        new_bm,
                        e_is_pv,
                    )
                    tt[idx]['key'] = key32
                    tt[idx]['depth'] = new_depth
                    tt[idx]['score'] = e_score
                    tt[idx]['static_eval'] = new_se
                    tt[idx]['flag'] = e_flag
                    tt[idx]['generation'] = e_gen
                    tt[idx]['best_move'] = new_bm
                    tt[idx]['is_pv'] = e_is_pv
                    tt[idx]['padding'][0] = np.uint8(chk & np.uint16(0xFF))
                    tt[idx]['padding'][1] = np.uint8(chk >> np.uint16(8))
            return # 找到並處理完相符局面，直接退出
            
    # 2. 如果沒有找到完全相同的局面 (雜湊碰撞 Collision)，進入替換評估 (Replacement Strategy)
    replace_idx = base_index
    min_replace_value = 999999
    
    for i in range(4):
        idx = base_index + i
        c_flag = np.uint8(tt[idx]['flag'])
        
        # 空位無條件優先使用，避免 QSearch (depth=0) 或跨世代時誤覆寫有效項目
        if c_flag == TT_FLAG_NONE:
            replace_idx = idx
            break
        
        # 計算相對老化程度
        c_gen = np.uint8(tt[idx]['generation'])
        age = (int(gen_u8) - int(c_gen) + 256) & 255
        if age > 15:
            age = 15  # 簡化的世代老化權重
            
        # PV nodes get a virtual depth bonus of 2, making them harder to replace
        existing_pv_bonus = 2 if bool(tt[idx]['is_pv']) else 0
        
        # 由於深度是無符號整數(uint8)，計算 replace_value 時需轉為有符號整數(int)處理負數情況
        replace_value = int(tt[idx]['depth']) + existing_pv_bonus - age * 8
        
        # 找到取代價值最小的項目（包含空位，因其深度為 0，能自然獲得極低的 replace_value）
        if replace_value < min_replace_value:
            min_replace_value = replace_value
            replace_idx = idx

    # 執行最終替換 (因為非吻合點，不保留舊的最佳步)
    d_u8 = np.uint8(depth)
    s_i16 = np.int16(score)
    se_i16 = np.int16(static_eval)
    f_u8 = np.uint8(flag)
    gen_u8 = np.uint8(current_generation)
    bm_u16 = np.uint16(best_move)
    pv_b = bool(is_pv)
    chk = _compute_tt_checksum(key32, s_i16, se_i16, d_u8, f_u8, bm_u16, pv_b)
    tt[replace_idx]['key'] = key32
    tt[replace_idx]['depth'] = d_u8
    tt[replace_idx]['score'] = s_i16
    tt[replace_idx]['static_eval'] = se_i16
    tt[replace_idx]['flag'] = f_u8
    tt[replace_idx]['generation'] = gen_u8
    tt[replace_idx]['best_move'] = bm_u16
    tt[replace_idx]['is_pv'] = pv_b
    tt[replace_idx]['padding'][0] = np.uint8(chk & np.uint16(0xFF))
    tt[replace_idx]['padding'][1] = np.uint8(chk >> np.uint16(8))
