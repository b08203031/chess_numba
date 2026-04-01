import os
import time
import math
import random
import numpy as np
import pyarrow.parquet as pq

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
# 填入您的 Hugging Face Access Token (如果沒有也不會報錯，但有登入可以提速/解除限制)
HF_TOKEN = "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" # ← 填入您的 Token (例如 "hf_xxxx...")

TARGET_SAMPLES = 40_000_000  # 本次抓取目標數量 (一次抓滿 4000 萬筆平滑過濾後的健康資料)
SKIP_SAMPLES = 1_000_000     # 跳過前 100 萬筆 (因為原資料集開頭的 100 萬筆 FEN 格式損壞/異常)
OUTPUT_FILE = "tuner/ultimate_dataset.npz"
REPO_ID = "mateuszgrzyb/lichess-stockfish-normalized"

K = 0.00368208
PIECE_MAP = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}

# 若有提供 Token 則進行自動登入
if HF_TOKEN:
    login(token=HF_TOKEN, add_to_git_credential=False)
else:
    print("⚠️ 尚未設定 HF_TOKEN，將以訪客模式下載。")

def cp_to_wdl(cp):
    cp = max(-10000, min(10000, cp))
    wdl = 1.0 / (1.0 + math.exp(-K * cp))
    return max(0.001, min(0.999, wdl))  # Target Bounding: 防止 logit → ±∞ 導致 INT16 量化崩潰
# 當 CP 值達到約 1876 時，勝率會精確轉化為 0.999
def parse_fen_fast(fen):
    """維持您寫的極速字節級 FEN 解析器"""
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

def main():
    print(f"🚀 開始向 Hugging Face 伺服器請求資料表清單...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # 預先分配記憶體陣列
    all_bbs = np.zeros((TARGET_SAMPLES, 12), dtype=np.uint64)
    all_stm = np.zeros((TARGET_SAMPLES, 1), dtype=np.uint8)
    all_results = np.zeros(TARGET_SAMPLES, dtype=np.float32)
    
    # 1. 直接取得資源庫裡所有 parquet 檔案的清單
    all_files = list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    parquet_files = [f for f in all_files if f.endswith(".parquet")]
    print(f"✅ 成功找到 {len(parquet_files)} 個分塊 (Shard) 檔案！")
    
    count = 0
    skipped_records = 0
    start_time = time.time()
    
    # 2. 採用「逐塊下載 -> 載入記憶體 -> 轉換 -> 釋放記憶體」的官方推薦高效做法
    for file_idx, file_path in enumerate(parquet_files):
        if count >= TARGET_SAMPLES:
            break
            
        print(f"\n📥 正在以最快速度下載並快取區塊 [{file_idx+1}/{len(parquet_files)}]: {file_path} ...")
        
        # hf_hub_download 會把檔案下載到系統快取，如果已經下載過就會瞬間載入！
        local_path = hf_hub_download(
            repo_id=REPO_ID, 
            filename=file_path, 
            repo_type="dataset",
        )
        
        # 使用 Pandas + PyArrow 瞬間讀取這幾百萬筆資料
        df = pd.read_parquet(local_path)
        print(f"📊 成功載入區塊，包含 {len(df):,} 筆盤面，開始高速轉換...")
        
        # 轉換核心迴圈
        # 針對 mateuszgrzyb/lichess-stockfish-normalized 的專屬欄位
        fen_col = 'FEN' if 'FEN' in df.columns else 'fen'
        if 'cp' not in df.columns or 'mate' not in df.columns:
            print(f"❌ 找不到 cp 或 mate 欄位！略過此區塊。現有欄位：{df.columns.tolist()}")
            continue
            
        fens = df[fen_col].values
        cps = df['cp'].values
        mates = df['mate'].values
        
        # --- 瞬間略過已下載的歷史資料邏輯 ---
        df_len = len(df)
        if skipped_records + df_len <= SKIP_SAMPLES:
            skipped_records += df_len
            print(f"⏩ 本區塊 ({df_len:,} 筆) 屬於舊資料，光速略過...")
            del df
            continue
            
        start_row = 0
        if skipped_records < SKIP_SAMPLES:
            start_row = SKIP_SAMPLES - skipped_records
            skipped_records += start_row
            print(f"⏩ 本區塊跳過前 {start_row:,} 筆舊資料，從中間開始收割全新資料...")
        
        # 只取我們需要的全新片段
        fens = fens[start_row:]
        cps = cps[start_row:]
        mates = mates[start_row:]
        
        chunk_count = 0
        for i in range(len(fens)):
            if count >= TARGET_SAMPLES:
                break
                
            fen_val = fens[i]
            cp_val = cps[i]
            mate_val = mates[i]
            
            if fen_val is None:
                continue
                
            try:
                # 處理分數與 MATE (智能洗滌 - Exponential Decay Resampling)
                is_mate = False
                abs_cp = 0.0
                
                if not math.isnan(mate_val):
                    is_mate = True
                    wdl = 0.999 if mate_val > 0 else 0.001  # Target Bounding: 不用精確 1.0/0.0
                elif not math.isnan(cp_val):
                    wdl = cp_to_wdl(cp_val)  # cp_to_wdl 內部已做 Target Bounding
                    abs_cp = abs(cp_val)
                else:
                    # 兩個都是 NaN，發生異常，略過
                    continue
                    
                # ------ 核心降採樣邏輯 (簡化版) ------
                # 所有 CP 局面 100% 保留，只過濾 Mate 到 5%
                if is_mate:
                    keep_prob = 0.05  # 5% 保留，確保量化安全但有足夠方向感
                else:
                    keep_prob = 1.0   # 所有 CP 局面全部保留
                    
                if random.random() > keep_prob:
                    continue
                # ---------------------------
                    
                # 處理 FEN
                bbs, stm = parse_fen_fast(fen_val)
                
                all_bbs[count] = bbs
                all_stm[count, 0] = stm
                all_results[count] = wdl
                
                count += 1
                chunk_count += 1
                
                if count % 100_000 == 0:
                    elapsed = time.time() - start_time
                    fps = count / elapsed
                    print(f"✓ 已光速處理 {count:,} / {TARGET_SAMPLES:,} 筆... (均速: {fps:,.0f} 盤面/秒)")
                    
            except Exception as e:
                continue
                
        # 主動刪除 pandas Dataframe，釋放記憶體
        del df
        print(f"♻️ 區塊 {file_idx+1} 處理完畢，已轉換 {chunk_count:,} 筆。釋放記憶體。")

    print(f"\n✅ 全部收集完成！總共獲得 {count:,} 筆高品質資料，開始壓縮並寫入硬碟...")
    # === [新增：全局洗牌片段] ===
    print(f"\n🌪️ 正在對 {count:,} 筆高品質資料進行全域洗牌...")
    shuffle_start = time.time()
    
    # 1. 生成隨機索引序列
    # 使用固定種子確保實驗可重複性，若想每次都不同可改為 None
    np.random.seed(42) 
    perm = np.random.permutation(count)
    
    # 2. 依照隨機索引重排所有陣列
    # 註：這會暫時消耗較多 RAM，請確保系統有 16GB 以上記憶體
    all_bbs[:count] = all_bbs[perm]
    all_stm[:count] = all_stm[perm]
    all_results[:count] = all_results[perm]
    
    print(f"✅ 洗牌完成！耗時: {time.time() - shuffle_start:.2f}s")
    # ==========================

    print(f"\n開始壓縮並寫入硬碟...")
    np.savez_compressed(
        OUTPUT_FILE,
        piece_bbs=all_bbs[:count],
        game_states=all_stm[:count],
        results=all_results[:count]
    )
    
    
    print(f"🎉 儲存至 {OUTPUT_FILE}！")

if __name__ == "__main__":
    main()
