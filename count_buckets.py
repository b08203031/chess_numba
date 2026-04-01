import numpy as np
import os

dataset_path = r'tuner/ultimate_halfka.npz'

if not os.path.exists(dataset_path):
    print(f"❌ 找不到資料集：{dataset_path}")
    print("請先執行：python tuner/convert_dataset.py")
    exit(1)

print(f"🚀 正在讀取資料集：{dataset_path} 以計算桶子分佈...")
data = np.load(dataset_path)

if 'piece_counts' not in data:
    print("❌ 資料集中找不到 'piece_counts' 欄位。請重新執行 convert_dataset.py！")
    exit(1)

piece_counts = data['piece_counts']
total_samples = len(piece_counts)
print(f"\n✅ 成功載入 {total_samples:,} 筆資料。")

# SF18 Bucket 公式：bucket_id = (piece_count - 1) // 4
bucket_ids = (piece_counts.astype(np.int32) - 1) // 4
bucket_ids = np.clip(bucket_ids, 0, 7)

counts = np.bincount(bucket_ids, minlength=8)
percentages = counts / total_samples

print("\n=== 🧠 8-Bucket 分佈統計 ===")
print(f"{'Bucket':<8} | {'Piece Counts':<15} | {'Samples':<12} | {'Percentage':<10}")
print("-" * 55)

for i in range(8):
    count = counts[i]
    percentage = percentages[i] * 100
    piece_range = f"{(i*4)+1} - {(i*4)+4}"
    print(f"B{i:<7} | {piece_range:<15} | {count:<12,} | {percentage:>6.2f}%")

print(f"\n💡 總計: {total_samples:,} 筆")

# =====================================================
# 自動計算反比權重並輸出可直接貼進 train.py 的程式碼
# =====================================================
print("\n" + "="*60)
print("📋 請將以下 BUCKET_WEIGHTS 貼入 train.py 取代原有的版本：")
print("="*60)
print("    BUCKET_WEIGHTS = np.array([")
for i in range(8):
    pct = percentages[i]
    # avoid division by zero
    if pct == 0:
        weight = 1.0
        note = f"B{i}: 0 樣本，給予預設權重 1.0"
    else:
        weight = 1.0 / pct
        piece_range = f"{(i*4)+1}-{(i*4)+4} 子"
        note = f"B{i} ({piece_range}): {pct*100:.2f}% -> 權重 {weight:.1f}"
    comma = "," if i < 7 else " "  # no comma on last
    print(f"        1.0 / {pct:.4f}{comma}  # {note}")
print("    ], dtype=np.float64)")
print("="*60)

data.close()
