import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# 物理常數與配置
FC1_BOUNDARY = 4.0
NUM_BUCKETS = 8
X_LIM_DENSE = (-2.0, 2.0) # 稠密層觀察區間

def inspect_weight(checkpoint_path="./chess_engine/nnue/ml_eval/weights/latest_model.pth"):
    if not os.path.exists(checkpoint_path):
        print(f"錯誤：找不到權重檔案 {checkpoint_path}")
        return

    print(f"🚀 啟動全網路流形深度診斷: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    except Exception as e:
        print(f"載入失敗: {e}")
        return

    state_dict = checkpoint.get('model_state_dict', checkpoint)
    epoch = checkpoint.get('epoch', 'Unknown')

    # --- 1. 輔助函數：提取層資料 ---
    def get_layer_data(layer_name):
        fact = state_dict[f'fact_{layer_name}.weight'].float().numpy()
        eff_list = []
        delta_list = []
        for b in range(NUM_BUCKETS):
            delta = state_dict[f'delta_{layer_name}.{b}.weight'].float().numpy()
            eff_list.append(fact + delta)
            delta_list.append(delta)
        all_eff = np.concatenate([e.flatten() for e in eff_list])
        return fact, eff_list, delta_list, all_eff

    print("正在解構各層流形...")
    fc1_w = state_dict['fc1.weight'].float().numpy()
    f2_base, f2_effs, f2_deltas, f2_all = get_layer_data("fc2")
    f3_base, f3_effs, f3_deltas, f3_all = get_layer_data("fc3")
    f4_base, f4_effs, f4_deltas, f4_all = get_layer_data("fc4")

    # --- 2. 終端機統計報告 ---
    print("\n" + "="*60)
    print(f"📊 NNUE 權重動力學診斷報告 (Epoch: {epoch})")
    print("="*60)
    
    fc1_sat = 100.0 * np.sum(np.abs(fc1_w) >= (FC1_BOUNDARY - 1e-4)) / fc1_w.size
    print(f"【FC1 稀疏層】飽和率: {fc1_sat:.4f}% | Max: {np.max(fc1_w):.4f}")
    
    for name, deltas in [("FC2", f2_deltas), ("FC3", f3_deltas), ("FC4", f4_deltas)]:
        mabs = [np.mean(np.abs(d)) for d in deltas]
        print(f"【{name} 決策層】Δ MeanAbs 區間: [{min(mabs):.5f} ~ {max(mabs):.5f}]")
    print("="*60)

    # --- 3. 繪圖函數：產出 3x3 桶子解構圖 ---
    def plot_bucket_breakdown(base, effs, deltas, layer_label, filename, color_map):
        fig, axs = plt.subplots(3, 3, figsize=(18, 14))
        fig.suptitle(f"{layer_label} Bucket Breakdown (Epoch {epoch})", fontsize=20, fontweight='bold')
        
        # [0,0] Factorizer
        axs[0,0].hist(base.flatten(), bins=100, range=X_LIM_DENSE, color='dimgray', alpha=0.9, log=True)
        axs[0,0].set_title(f"Global Factorizer ($W_{{fact}}$)", fontweight='bold', color='red')
        axs[0,0].set_facecolor('#fff5f5')
        axs[0,0].grid(axis='y', alpha=0.3)

        colors = color_map(np.linspace(0, 0.8, NUM_BUCKETS))
        for b in range(NUM_BUCKETS):
            r, c = (b + 1) // 3, (b + 1) % 3
            ax = axs[r, c]
            ax.hist(effs[b].flatten(), bins=100, range=X_LIM_DENSE, color=colors[b], alpha=0.8, log=True)
            d_mean = np.mean(np.abs(deltas[b]))
            ax.set_title(f"Bucket {b} (Effective)\n$\Delta$ MeanAbs: {d_mean:.5f}")
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"./chess_engine/nnue/ml_eval/weights/{filename}", dpi=150)
        plt.close()

    # --- 4. 執行繪圖任務 ---
    # 圖表 1: 全網路宏觀圖 (2x2)
    print("🎨 產出 macro_overview.png...")
    fig_m, axs_m = plt.subplots(2, 2, figsize=(14, 10))
    fig_m.suptitle(f"Global Overview (Epoch {epoch})", fontsize=16, fontweight='bold')
    axs_m[0,0].hist(fc1_w.flatten(), bins=150, range=(-4.5, 4.5), color='steelblue', log=True)
    axs_m[0,0].set_title("FC1 (Sparse)")
    axs_m[0,1].hist(f2_all, bins=100, color='mediumseagreen', log=True); axs_m[0,1].set_title("FC2 (Dense)")
    axs_m[1,0].hist(f3_all, bins=100, color='coral', log=True); axs_m[1,0].set_title("FC3 (Dense)")
    axs_m[1,1].hist(f4_all, bins=100, color='mediumpurple', log=True); axs_m[1,1].set_title("FC4 (Output)")
    plt.savefig("./chess_engine/nnue/ml_eval/weights/all_layers_histogram.png", dpi=150); plt.close()

    # 圖表 2, 3, 4: 各層桶子解構
    print("🎨 產出 fc2_buckets_histogram.png...")
    plot_bucket_breakdown(f2_base, f2_effs, f2_deltas, "FC2", "fc2_buckets_histogram.png", plt.cm.viridis)
    
    print("🎨 產出 fc3_buckets_histogram.png...")
    plot_bucket_breakdown(f3_base, f3_effs, f3_deltas, "FC3", "fc3_buckets_histogram.png", plt.cm.plasma)
    
    print("🎨 產出 fc4_buckets_histogram.png...")
    plot_bucket_breakdown(f4_base, f4_effs, f4_deltas, "FC4", "fc4_buckets_histogram.png", plt.cm.cool)

    print(f"✅ 診斷完畢！共產出 4 張分析圖表於 weights/ 目錄。")

if __name__ == "__main__":
    inspect_weight()