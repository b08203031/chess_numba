import numpy as np
import os
import time

def merge():
    existing_ultimate = "tuner/hf_dataset_next_40M.npz"
    new_part = "tuner/hf_dataset_next_20M.npz"
    
    if not os.path.exists(existing_ultimate):
        print(f"❌ 錯誤：找不到 {existing_ultimate}。")
        return
        
    if not os.path.exists(new_part):
        print(f"❌ 錯誤：找不到 {new_part}。請先執行 download_lichess.py 下載新資料！")
        return
        
    start_time = time.time()
    
    # 讀取原本的 2,000 萬筆資料
    print(f"Loading Existing Ultimate ({existing_ultimate})...")
    data_old = np.load(existing_ultimate)
    bbs1 = data_old['piece_bbs']
    stm1 = data_old['game_states']
    ref1 = data_old['results']
    print(f" -> 已讀取舊資料 {len(ref1):,} 筆")
    
    # 讀取剛下載的 1,000 萬筆資料
    print(f"Loading New Part ({new_part})...")
    data_new = np.load(new_part)
    bbs2 = data_new['piece_bbs']
    stm2 = data_new['game_states']
    ref2 = data_new['results']
    print(f" -> 已讀取新資料 {len(ref2):,} 筆")
    
    total_len = len(ref1) + len(ref2)
    
    # 大規模合併 (Concatenation)
    print(f"正在接合資料集... (預期總長度: {total_len:,})")
    merged_bbs = np.concatenate((bbs1, bbs2), axis=0)
    merged_stm = np.concatenate((stm1, stm2), axis=0)
    merged_ref = np.concatenate((ref1, ref2), axis=0)
    
    data_old.close()
    data_new.close()
    
    # 儲存全新的終極資料庫 (這會覆蓋原本的 20M 版本，升級為 30M 版本)
    print(f"✅ 融合成功！新資料庫總長度為：{len(merged_ref):,} 筆！")
    print(f"正在覆蓋保存為 {existing_ultimate} ... (需要寫入超過 2 GB 的檔案，請耐心等待)")
    
    np.savez_compressed(
        existing_ultimate,
        piece_bbs=merged_bbs,
        game_states=merged_stm,
        results=merged_ref
    )
    
    elapsed = time.time() - start_time
    print(f"🎉 大功告成！全自動融合完成，耗時 {elapsed:.2f} 秒。")
    print(f"如果你想要節省硬碟空間，現在可以安全地刪除 `hf_dataset_20M.npz` 了。")
    print(f"接下來，請務必重新執行 `python tuner/convert_dataset.py`，提煉出 HalfKA 的 NPZ！")

if __name__ == '__main__':
    merge()
