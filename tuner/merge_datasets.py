import numpy as np
import os
import time

def merge():
    part1_path = "tuner/dataset_part1.npz"
    part2_path = "tuner/dataset_part2.npz"
    out_path = "tuner/ultimate_dataset.npz"

    if not os.path.exists(part1_path):
        print(f"❌ 錯誤：找不到 {part1_path}。你有先將舊資料庫改名嗎？")
        return
        
    if not os.path.exists(part2_path):
        print(f"❌ 錯誤：找不到 {part2_path}。請先執行 download_lichess.py 下載新資料！")
        return
        
    start_time = time.time()
    
    # 讀取第 1 份資料
    print("Loading Part 1 (10,000,000 筆舊資料)...")
    data1 = np.load(part1_path)
    bbs1 = data1['piece_bbs']
    stm1 = data1['game_states']
    ref1 = data1['results']
    
    # 讀取第 2 份資料
    print("Loading Part 2 (10,000,000 筆全新下載資料)...")
    data2 = np.load(part2_path)
    bbs2 = data2['piece_bbs']
    stm2 = data2['game_states']
    ref2 = data2['results']
    
    print(f"Part 1 length: {len(ref1):,}")
    print(f"Part 2 length: {len(ref2):,}")
    
    # 大規模合併 (Concatenation) - 這裡會暫時消耗較多 RAM
    print("正在將兩千萬大軍強行接合... 這可能需要幾十秒，視你的 RAM 大小而定。")
    merged_bbs = np.concatenate((bbs1, bbs2), axis=0)
    merged_stm = np.concatenate((stm1, stm2), axis=0)
    merged_ref = np.concatenate((ref1, ref2), axis=0)
    
    total_len = len(merged_ref)
    
    # 儲存全新的終極資料庫
    print(f"✅ 融合成功！新資料庫總長度為：{total_len:,} 筆！")
    print(f"正在保存為 {out_path} ... (需要寫入約 1.7 GB 的檔案)")
    
    np.savez_compressed(
        out_path,
        piece_bbs=merged_bbs,
        game_states=merged_stm,
        results=merged_ref
    )
    
    data1.close()
    data2.close()
    
    elapsed = time.time() - start_time
    print(f"🎉 大功告成！全自動融合完成，耗時 {elapsed:.2f} 秒。")
    print(f"接下來，請務必重新執行 `python tuner/convert_dataset.py`，提煉出足以讓 RTX 4050 運作的 HalfKA Numpy 矩陣！")

if __name__ == '__main__':
    merge()
