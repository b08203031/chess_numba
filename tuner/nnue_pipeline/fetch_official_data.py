"""
fetch_official_data.py — 自動下載與轉換官方 Stockfish binpack 訓練資料

步驟:
1. 從 Hugging Face 下載最高品質的官方 binpack 檔案 (直接截斷前 200MB)
2. 呼叫 primer.exe 進行戰術過濾與解壓縮 (Primer遇到截斷處會自動報錯但保留前面的結果)
3. 呼叫 convert_bullet_bin.py 將資料轉為 Numba 訓練器所需的 .npz 格式
"""

import os
import sys
import subprocess
import requests

# ==========================================
# ⚙️ 設定區
# ==========================================
# 選擇官方最高品質的資料集
DATASET_URL = "https://huggingface.co/datasets/official-stockfish/master-binpacks/resolve/main/farseerT75.binpack"
BINPACK_FILE = "tuner/official_data_farseerT75.binpack"
BULLET_FILE = "tuner/official_data_farseerT75.bin"
OUTPUT_NPZ = "tuner/ultimate_halfka_farseerT75.npz"

# Paths relative to project root (chess_numba/)
# primer.exe is in external/
# convert_bullet_bin.py is in tuner/nnue_pipeline/

TARGET_MB = 200

def download_binpack_fast(url, out_path, target_mb):
    print(f"🌍 正在連接到 Hugging Face 以下載官方 binpack...")
    print(f"   目標 URL: {url}")
    print(f"   目標大小: {target_mb} MB")
    
    target_bytes = target_mb * 1024 * 1024
    downloaded_bytes = 0
    
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(out_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024 * 10):
                if chunk:
                    f.write(chunk)
                    downloaded_bytes += len(chunk)
                    print(f"  📥 已下載: {downloaded_bytes / (1024*1024):.1f} MB ...", end='\r')
                if downloaded_bytes >= target_bytes:
                    break
                    
    print(f"\n✅ 下載完成！總大小: {downloaded_bytes / (1024*1024):.2f} MB")

def run_primer(binpack_path, bin_path):
    print(f"⚙️ 呼叫 primer.exe 進行戰術將軍過濾與解壓縮...")
    cmd = [
        "external/primer.exe", 
        "convert", 
        binpack_path, 
        bin_path, 
        "--max-score", "10000", 
        "--min-ply", "16"
    ]
    
    if os.path.exists(bin_path):
        os.remove(bin_path)
        
    # check=False 因為截斷的 binpack 在尾端會拋出 parsing 錯誤並退出 code 1, 但這沒關係，因為前面的檔案已成功產出!
    subprocess.run(cmd, check=False)
    
    if os.path.exists(bin_path) and os.path.getsize(bin_path) > 0:
        print("✅ primer.exe 轉換成功 (尾端截斷報錯請忽略)！")
    else:
        print("❌ primer.exe 轉換失敗！")
        sys.exit(1)

def run_converter(bin_path, npz_path):
    print(f"🐍 呼叫 convert_bullet_bin.py 轉換為 NumPy (NPZ) 格式...")
    cmd = [
        sys.executable,
        "tuner/nnue_pipeline/convert_bullet_bin.py",
        bin_path,
        "--output", npz_path
    ]
    try:
        subprocess.run(cmd, check=True)
        print("\n🎉 全部處理流程完成！官方純淨資料已準備就緒！")
    except subprocess.CalledProcessError as e:
        print(f"❌ convert_bullet_bin.py 執行失敗: {e}")
        sys.exit(1)

if __name__ == '__main__':
    download_binpack_fast(DATASET_URL, BINPACK_FILE, TARGET_MB)
    run_primer(BINPACK_FILE, BULLET_FILE)
    run_converter(BULLET_FILE, OUTPUT_NPZ)
