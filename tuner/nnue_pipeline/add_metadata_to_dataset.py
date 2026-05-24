"""
add_metadata_to_dataset.py — 為舊格式的 HalfKA .npz 資料集添加 metadata_json
避免訓練時出現 "WARNING: Dataset does not contain metadata_json" 的警告。
"""

import os
import json
import time
import numpy as np

def main():
    dataset_path = "tuner/ultimate_halfka_farseerT75.npz"
    if not os.path.exists(dataset_path):
        # 嘗試向上尋找
        levels = ['../', '../../']
        found = False
        for lv in levels:
            alt_path = os.path.join(lv, 'tuner/ultimate_halfka_farseerT75.npz')
            if os.path.exists(alt_path):
                dataset_path = alt_path
                found = True
                break
        if not found:
            print("❌ 錯誤：找不到 tuner/ultimate_halfka_farseerT75.npz 檔案。")
            return

    print(f"正在載入資料集：{dataset_path} ... (可能需要數十秒，請耐心等待)")
    start_time = time.time()
    
    # 讀取原本的資料
    data = np.load(dataset_path)
    
    if 'metadata_json' in data:
        print("ℹ️ 資料集已經包含 metadata_json，無須重複添加。")
        data.close()
        return

    features_stm = data['features_stm']
    features_nstm = data['features_nstm']
    targets = data['targets']
    piece_counts = data['piece_counts']
    
    print(f" -> 已成功載入 {len(targets):,} 筆資料。")
    print("正在寫入 metadata_json...")
    
    # 建構符合新版 convert_bullet_bin.py 規範的 metadata
    metadata = {
        "feature_encoding_version": "halfkav2_hm_stockfish_official_v1",
        "target_k": 0.0025,
        "pawn_value_eg": 208.0,
        "schema_version": "1.0.0"
    }
    metadata_json = json.dumps(metadata)
    
    print("正在重新儲存資料集至原本路徑 (需要重新壓縮與寫入，請稍後)...")
    save_start = time.time()
    
    np.savez_compressed(
        dataset_path,
        features_stm=features_stm,
        features_nstm=features_nstm,
        targets=targets,
        piece_counts=piece_counts,
        metadata_json=np.array(metadata_json)
    )
    
    # 關閉資源
    data.close()
    
    print(f"✅ 成功添加 metadata_json！")
    print(f"   - 載入耗時：{save_start - start_time:.2f} 秒")
    print(f"   - 儲存耗時：{time.time() - save_start:.2f} 秒")
    print(f"   - 總耗時：{time.time() - start_time:.2f} 秒")

if __name__ == '__main__':
    main()
