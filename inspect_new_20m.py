import numpy as np
import os
import sys

dataset_path = r'tuner/ultimate_dataset.npz'

def print_usage():
    print("使用方法: python inspect_new_40m.py [起始索引] [結束索引]")
    print("例如: python inspect_new_40m.py 10000 10100")
    print("預設顯示前 100 筆資料。")

if not os.path.exists(dataset_path):
    print(f"❌ 找不到資料集：{dataset_path}")
    exit(1)

# 取得命令列參數
start_idx = 1000000
end_idx = 1100000

if len(sys.argv) > 1:
    try:
        start_idx = int(sys.argv[1])
        if len(sys.argv) > 2:
            end_idx = int(sys.argv[2])
        else:
            end_idx = start_idx + 100
    except ValueError:
        print_usage()
        exit(1)

print(f"🚀 載入資料集：{dataset_path} ...")
data = np.load(dataset_path)

total_count = len(data['results'])
if start_idx < 0 or start_idx >= total_count:
    print(f"❌ 起始索引 {start_idx} 超出範圍 (總數: {total_count})")
    exit(1)
if end_idx > total_count:
    end_idx = total_count

results = data['results'][start_idx:end_idx]
game_states = data['game_states'][start_idx:end_idx]

print(f"\n=== 資料集品質檢查 (Index {start_idx} - {end_idx}) ===")
print(f"{'Index':<10} | {'WDL (Target)':<12} | {'STM':<5} | {'Mate Check'}")
print("-" * 55)

mate_count = 0
mate_count_white = 0
mate_count_black = 0
for i in range(len(results)):
    current_abs_idx = start_idx + i
    wdl = results[i]
    stm = "White" if game_states[i] == 0 else "Black"
    is_mate = "YES (1.0/0.0)" if wdl == 1.0 or wdl == 0.0 else "Normal"
    if wdl == 1.0 or wdl == 0.0:
        mate_count += 1
        if wdl == 1.0:
            mate_count_white += 1
        else:
            mate_count_black += 1
    
    # print(f"{current_abs_idx:<10} | {wdl:<12.4f} | {stm:<5} | {is_mate}")

print(f"\n📊 該範圍中偵測到 {mate_count} 筆疑似殺局 (WDL 為精準 1.0 或 0.0)")
print(f"白方殺局: {mate_count_white}")
print(f"黑方殺局: {mate_count_black}")
data.close()
