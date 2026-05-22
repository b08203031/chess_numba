import os
import time
import math
import random
import numpy as np
import pyarrow.parquet as pq
import numba

try:
    from huggingface_hub import list_repo_files, hf_hub_download, login
    import pandas as pd
except ImportError:
    print("❌ 請先安裝必要的高效套件：")
    print("pip install huggingface_hub pandas pyarrow")
    exit(1)

# ==========================================
# ⚙️ 設定區
# ==========================================
HF_TOKEN = "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" 

TARGET_SAMPLES = 40_000_000 
SKIP_SAMPLES = 1_000_000     
OUTPUT_FILE = "tuner/ultimate_halfka.npz"
REPO_ID = "mateuszgrzyb/lichess-stockfish-normalized"

# K = 1 / 400.0 (Official Stockfish calibration scale)
K = 0.0025
PIECE_MAP = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}

if HF_TOKEN:
    login(token=HF_TOKEN, add_to_git_credential=False)
else:
    print("⚠️ 尚未設定 HF_TOKEN，將以訪客模式下載。")

def cp_to_wdl(cp):
    cp = max(-10000, min(10000, cp))
    wdl = 1.0 / (1.0 + math.exp(-K * cp))
    return max(0.001, min(0.999, wdl))

def parse_fen_fast(fen):
    parts = fen.split(' ')
    board_part = parts[0]
    stm_part = parts[1]
    
    bbs = np.zeros(12, dtype=np.uint64)
    ranks = board_part.split('/')
    for r_idx, rank in enumerate(ranks):
        rank_val = 7 - r_idx
        file_val = 0
        for char in rank:
            if char.isdigit():
                file_val += int(char)
            else:
                sq = rank_val * 8 + file_val
                p_idx = PIECE_MAP[char]
                bbs[p_idx] |= np.uint64(1 << sq)
                file_val += 1
                
    stm = 0 if stm_part == 'w' else 1
    return bbs, stm

# ==========================================
# ⚡ Numba 加速轉換邏輯 (從 convert_dataset.py 移植)
# ==========================================

@numba.njit(numba.int32[:](numba.uint64[:], numba.boolean), cache=True, fastmath=True)
def get_halfka_indices(bbs, is_black_perspective):
    """將 12x64 位元棋盤轉換為 32 個 HalfKA 特徵索引 (King Mirroring)"""
    indices = np.full(32, 22528, dtype=np.int32)
    idx_count = 0
    if is_black_perspective:
        king_bb = bbs[11]
        king_sq = 0
        if king_bb:
            lsb = king_bb & (~king_bb + np.uint64(1))
            tmp = lsb >> np.uint64(1)
            while tmp:
                king_sq += 1
                tmp >>= np.uint64(1)
        king_sq = king_sq ^ 56
        flip_h = (king_sq % 8) >= 4
        if flip_h:
            king_sq ^= 7
        bucket = (king_sq // 8) * 4 + (king_sq % 8)
        for p_idx in range(12):
            if p_idx == 11: continue 
            mapped_type = (p_idx + 5) if p_idx < 6 else (p_idx - 6)
            bb = bbs[p_idx]
            while bb:
                lsb = bb & (~bb + np.uint64(1))
                sq = 0
                tmp = lsb >> np.uint64(1)
                while tmp:
                    sq += 1
                    tmp >>= np.uint64(1)
                sq = sq ^ 56
                if flip_h:
                    sq ^= 7
                if idx_count < 32:
                    indices[idx_count] = bucket * 704 + mapped_type * 64 + sq
                    idx_count += 1
                bb &= bb - np.uint64(1)
    else:
        king_bb = bbs[5]
        king_sq = 0
        if king_bb:
            lsb = king_bb & (~king_bb + np.uint64(1))
            tmp = lsb >> np.uint64(1)
            while tmp:
                king_sq += 1
                tmp >>= np.uint64(1)
        flip_h = (king_sq % 8) >= 4
        if flip_h:
            king_sq ^= 7
        bucket = (king_sq // 8) * 4 + (king_sq % 8)
        for p_idx in range(12):
            if p_idx == 5: continue 
            mapped_type = p_idx if p_idx < 5 else (p_idx - 1)
            bb = bbs[p_idx]
            while bb:
                lsb = bb & (~bb + np.uint64(1))
                sq = 0
                tmp = lsb >> np.uint64(1)
                while tmp:
                    sq += 1
                    tmp >>= np.uint64(1)
                if flip_h:
                    sq ^= 7
                if idx_count < 32:
                    indices[idx_count] = bucket * 704 + mapped_type * 64 + sq
                    idx_count += 1
                bb &= bb - np.uint64(1)
    return indices

@numba.njit(parallel=False, fastmath=True)
def process_chunk_indices(bbs_array, stm_array, wdl_array, out_stm, out_nstm, out_wdl, out_piece_counts):
    """處理一個分塊的樣本轉換"""
    N = bbs_array.shape[0]
    for i in range(N):
        bbs = bbs_array[i]
        stm = stm_array[i]
        
        f_white = get_halfka_indices(bbs, False)
        f_black = get_halfka_indices(bbs, True)
        
        if stm == 0: 
            for j in range(32):
                out_stm[i, j] = f_white[j]
                out_nstm[i, j] = f_black[j]
            out_wdl[i] = wdl_array[i]
        else: 
            for j in range(32):
                out_stm[i, j] = f_black[j]
                out_nstm[i, j] = f_white[j]
            out_wdl[i] = 1.0 - wdl_array[i]
            
        total_p = np.int8(0)
        for p_idx in range(12):
            bb = bbs[p_idx]
            while bb:
                total_p += np.int8(1)
                bb &= bb - np.uint64(1)
        out_piece_counts[i] = total_p

def main():
    print(f"🚀 開始向 Hugging Face 伺服器請求資料表清單...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    all_features_stm  = np.zeros((TARGET_SAMPLES, 32), dtype=np.uint16)
    all_features_nstm = np.zeros((TARGET_SAMPLES, 32), dtype=np.uint16)
    all_targets       = np.zeros(TARGET_SAMPLES,       dtype=np.float32)
    all_piece_counts  = np.zeros(TARGET_SAMPLES,       dtype=np.int8)
    
    all_files = list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    parquet_files = [f for f in all_files if f.endswith(".parquet")]
    print(f"✅ 成功找到 {len(parquet_files)} 個分塊檔案！")
    
    count = 0
    skipped_records = 0
    total_start_time = time.time()
    
    for file_idx, file_path in enumerate(parquet_files):
        if count >= TARGET_SAMPLES:
            break
            
        print(f"\n📥 正在儲存區塊 [{file_idx+1}/{len(parquet_files)}]: {file_path}")
        local_path = hf_hub_download(repo_id=REPO_ID, filename=file_path, repo_type="dataset")
        
        df = pd.read_parquet(local_path)
        df_len = len(df)
        
        if skipped_records + df_len <= SKIP_SAMPLES:
            skipped_records += df_len
            print(f"⏩ 已跳過前 {skipped_records:,} 筆...")
            continue
            
        start_row = 0
        if skipped_records < SKIP_SAMPLES:
            start_row = SKIP_SAMPLES - skipped_records
            skipped_records = SKIP_SAMPLES
            
        effective_df = df.iloc[start_row:]
        fens  = effective_df['fen' if 'fen' in df.columns else 'FEN'].values
        cps   = effective_df['cp'].values
        mates = effective_df['mate'].values
        
        batch_size = len(fens)
        chunk_bbs = np.zeros((batch_size, 12), dtype=np.uint64)
        chunk_stm = np.zeros(batch_size, dtype=np.uint8)
        chunk_wdl = np.zeros(batch_size, dtype=np.float32)
        
        valid_chunk_count = 0
        chunk_start_t = time.time()
        for i in range(batch_size):
            if i % 100000 == 0 and i > 0:
                elapsed = time.time() - chunk_start_t
                fps = i / elapsed
                print(f"  ... FEN 解析中 [{i:,} / {batch_size:,}] | 速度: {fps:,.0f} 局/秒")
            
            if fens[i] is None or not math.isfinite(cps[i]):
                continue
            if abs(mates[i]) > 0: 
                continue
            
            bbs, stm = parse_fen_fast(fens[i])
            chunk_bbs[valid_chunk_count] = bbs
            chunk_stm[valid_chunk_count] = stm
            chunk_wdl[valid_chunk_count] = cp_to_wdl(cps[i])
            valid_chunk_count += 1
            
            if count + valid_chunk_count >= TARGET_SAMPLES:
                break
            
        if valid_chunk_count > 0:
            out_stm_view  = all_features_stm[count : count + valid_chunk_count]
            out_nstm_view = all_features_nstm[count : count + valid_chunk_count]
            out_wdl_view  = all_targets[count : count + valid_chunk_count]
            out_pc_view   = all_piece_counts[count : count + valid_chunk_count]
            
            process_chunk_indices(
                chunk_bbs[:valid_chunk_count], 
                chunk_stm[:valid_chunk_count], 
                chunk_wdl[:valid_chunk_count],
                out_stm_view, out_nstm_view, out_wdl_view, out_pc_view
            )
            count += valid_chunk_count
            print(f"✨ 已處理總量: {count:,} / {TARGET_SAMPLES:,} (本次 +{valid_chunk_count})")

        # Free memory
        del df

    print(f"\n✅ 下載與預先轉換完成！耗時: {(time.time()-total_start_time)/60:.1f} 分鐘")
    print(f"💾 正在儲存至 {OUTPUT_FILE} (啟用壓縮)...")
    
    save_start = time.time()
    np.savez_compressed(
        OUTPUT_FILE,
        features_stm=all_features_stm[:count],
        features_nstm=all_features_nstm[:count],
        targets=all_targets[:count],
        piece_counts=all_piece_counts[:count]
    )
    print(f"🏁 存檔成功！耗時: {time.time()-save_start:.2f}s")

if __name__ == '__main__':
    main()
