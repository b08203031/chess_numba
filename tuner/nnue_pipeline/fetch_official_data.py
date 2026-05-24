"""
fetch_official_data.py — 自動下載與轉換官方 Stockfish binpack 訓練資料

步驟:
1. 從 Hugging Face 下載最高品質的官方 binpack 檔案 (預設截斷前 1GB，適合 16GB RAM 訓練機)
2. 呼叫 primer.exe 進行戰術過濾與解壓縮 (Primer遇到截斷處會自動報錯但保留前面的結果)
3. 呼叫 convert_bullet_bin.py 將資料轉為 Numba 訓練器所需的 .npz 格式
"""

import os
import sys
import argparse
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

# 1GB binpack is enough source material for the default cap. Primer is also
# passed --limit-positions so the intermediate Bullet .bin stays near 1GB.
TARGET_MB = 1024
DEFAULT_CONVERT_LIMIT = 50_000_000

def _format_size(num_bytes):
    return f"{num_bytes / (1024 ** 2):.1f} MiB"


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

def run_primer(binpack_path, bin_path, limit):
    print(f"⚙️ 呼叫 primer.exe 進行戰術將軍過濾與解壓縮...")
    cmd = [
        "external/primer.exe", 
        "convert", 
        binpack_path, 
        bin_path, 
        "--max-score", "5000", 
        "--min-ply", "16",
        "--limit-positions", str(limit),
    ]
    
    if os.path.exists(bin_path):
        os.remove(bin_path)
        
    # check=False 因為截斷的 binpack 在尾端會拋出 parsing 錯誤並退出 code 1, 但這沒關係，因為前面的檔案已成功產出!
    # 捕捉輸出並限制其輸出頻率，避免洗版
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    
    last_printed_million = 0
    for line in process.stdout:
        if "Processed" in line and "positions." in line:
            try:
                parts = line.split()
                pos_idx = parts.index("positions.") if "positions." in parts else -1
                if pos_idx > 0:
                    pos_count = int(parts[pos_idx - 1])
                    current_million = pos_count // 5_000_000
                    if current_million > last_printed_million:
                        print(f"   [primer.exe] 已過濾並解壓: {pos_count:,} 筆局面...")
                        last_printed_million = current_million
            except Exception:
                pass
        else:
            print(line.strip())
            
    process.wait()
    
    if os.path.exists(bin_path) and os.path.getsize(bin_path) > 0:
        binpack_size = os.path.getsize(binpack_path)
        bin_size = os.path.getsize(bin_path)
        records = bin_size // 32
        print("✅ primer.exe 轉換成功 (尾端截斷報錯請忽略)！")
        print(f"   binpack: {_format_size(binpack_size)} -> Bullet .bin: {_format_size(bin_size)}")
        if records >= limit:
            print(f"   records: {records:,} (已達限制上限 {limit:,}，未完全解壓 binpack，不計算壓縮比)")
        else:
            ratio = bin_size / binpack_size if binpack_size else 0.0
            print(f"   records: {records:,} | .bin/.binpack ratio: {ratio:.2f}x")
    else:
        print("❌ primer.exe 轉換失敗！")
        sys.exit(1)

def run_converter(bin_path, npz_path, limit):
    print(f"🐍 呼叫 convert_bullet_bin.py 轉換為 NumPy (NPZ) 格式...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    convert_script = os.path.join(script_dir, "convert_bullet_bin.py")
    cmd = [
        sys.executable,
        convert_script,
        bin_path,
        "--output", npz_path,
        "--limit", str(limit),
    ]
    try:
        subprocess.run(cmd, check=True)
        print("\n🎉 全部處理流程完成！官方純淨資料已準備就緒！")
    except subprocess.CalledProcessError as e:
        print(f"❌ convert_bullet_bin.py 執行失敗: {e}")
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download official Stockfish binpack and convert it to HalfKAv2_hm NPZ.")
    parser.add_argument("--target-mb", type=int, default=TARGET_MB, help=f"Binpack download cap in MB (default: {TARGET_MB})")
    parser.add_argument("--limit", type=int, default=DEFAULT_CONVERT_LIMIT, help=f"Max Bullet records to convert (default: {DEFAULT_CONVERT_LIMIT:,})")
    args = parser.parse_args()

    # 確保 tuner 資料夾存在
    os.makedirs(os.path.dirname(os.path.abspath(BULLET_FILE)), exist_ok=True)

    if os.path.exists(BULLET_FILE) and os.path.getsize(BULLET_FILE) > 0:
        print(f"ℹ️ 偵測到已存在 {BULLET_FILE}，跳過下載與 primer 戰術過濾。")
    else:
        if os.path.exists(BINPACK_FILE) and os.path.getsize(BINPACK_FILE) > 0:
            print(f"ℹ️ 偵測到已存在 {BINPACK_FILE}，跳過下載。")
        else:
            download_binpack_fast(DATASET_URL, BINPACK_FILE, args.target_mb)
        run_primer(BINPACK_FILE, BULLET_FILE, args.limit)

    run_converter(BULLET_FILE, OUTPUT_NPZ, args.limit)
