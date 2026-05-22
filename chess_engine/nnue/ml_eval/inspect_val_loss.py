import os
import sys
import torch
import numpy as np
import torch.nn as nn
import time

# --- 路徑修正 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from train import NNUE 
except ImportError:
    from train import LayerStackNNUE as NNUE

ROOT_DIR = os.path.abspath(os.path.join(current_dir, "../../../")) 
INPUT_FILE = os.path.join(ROOT_DIR, "tuner/ultimate_halfka_farseerT75.npz")
CHECKPOINT_PATH = os.path.join(current_dir, "weights/latest_model.pth")

def inspect_bucket_loss_full_ram():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 啟動「全記憶體 + GPU」深度診斷...")

    # 1. 載入模型
    model = NNUE().to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    model.eval()

    # 2. 一次性載入全量數據到 RAM
    print(f"📦 正在將 4000 萬筆數據載入 RAM (請稍候，正在解壓縮)...")
    start_load = time.time()
    # 移除 mmap_mode，直接載入
    with np.load(INPUT_FILE) as data:
        all_features_stm = data['features_stm']
        all_features_nstm = data['features_nstm']
        all_targets = data['targets']
        all_piece_counts = data['piece_counts']
    print(f"✅ 載入完成！耗時: {time.time() - start_load:.2f}s")

    total_samples = len(all_targets)
    val_size = int(total_samples * 0.1) # 4,000,000 筆驗證集
    start_idx = total_samples - val_size
    
    bucket_sums = torch.zeros(8, device=device)
    bucket_counts = torch.zeros(8, device=device)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    # 3. 極速批次處理
    batch_size = 65536 # 既然數據都在 RAM，我們可以把 Batch 開到最大
    print(f"🔥 開始 GPU 全力輸出 (Batch: {batch_size})...")
    
    with torch.no_grad():
        for i in range(start_idx, total_samples, batch_size):
            end = min(i + batch_size, total_samples)
            
            # 直接從 RAM 切片並轉為 Tensor
            indices_stm = torch.from_numpy(all_features_stm[i:end].astype(np.int64)).to(device)
            indices_nstm = torch.from_numpy(all_features_nstm[i:end].astype(np.int64)).to(device)
            targets = torch.from_numpy(all_targets[i:end]).unsqueeze(1).to(device)
            
            # 分桶計算 (SF18 邏輯)
            pc = all_piece_counts[i:end]
            buckets = np.clip((pc - 1) // 4, 0, 7).astype(np.int64)
            buckets_torch = torch.from_numpy(buckets).to(device)
            
            outputs = model(indices_stm, indices_nstm, buckets_torch)
            loss_tensor = criterion(outputs, targets).squeeze()

            bucket_sums.scatter_add_(0, buckets_torch, loss_tensor)
            bucket_counts.scatter_add_(0, buckets_torch, torch.ones_like(loss_tensor))

            if ((i - start_idx) // batch_size) % 10 == 0:
                prog = 100 * (i - start_idx) / val_size
                print(f"進度: {prog:.1f}% ...")

    # 4. 輸出報告
    final_sums = bucket_sums.cpu().numpy()
    final_counts = bucket_counts.cpu().numpy()
    # ... (後續輸出邏輯與之前相同)
    print("\n" + "="*65)
    print(f"📊 NNUE 全量驗證集分桶報告")
    print("="*65)
    ranges = ["1-4", "5-8", "9-12", "13-16", "17-20", "21-24", "25-28", "29-32"]
    for b in range(8):
        if final_counts[b] > 0:
            print(f"B{b:<7} | {ranges[b]:<10} | {int(final_counts[b]):<12,} | {final_sums[b]/final_counts[b]:.6f}")
    print("="*65)

if __name__ == "__main__":
    inspect_bucket_loss_full_ram()