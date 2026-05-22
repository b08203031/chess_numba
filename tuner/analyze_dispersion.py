import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from chess_engine.nnue.ml_eval.train import LayerStackNNUE, TrainingConfig

def analyze_dispersion():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset_path = 'tuner/ultimate_halfka.npz'
    if not os.path.exists(dataset_path):
        print(f"Dataset {dataset_path} not found.")
        sys.exit(1)
        
    print("Loading 50,000 validation samples...")
    data = np.load(dataset_path)
    
    n_samples = 50000
    features_stm = data['features_stm'][-n_samples:].astype(np.int64)
    features_nstm = data['features_nstm'][-n_samples:].astype(np.int64)
    targets = data['targets'][-n_samples:].astype(np.float32)
    piece_counts = data['piece_counts'][-n_samples:].astype(np.int32)
    del data 
    
    model = LayerStackNNUE()
    weights_path = 'chess_engine/nnue/ml_eval/weights/best_model.pth'
    if not os.path.exists(weights_path):
        weights_path = 'chess_engine/nnue/ml_eval/weights/latest_model.pth'
        
    if not os.path.exists(weights_path):
        print("No PyTorch model weights found. Please run training first.")
        sys.exit(1)
        
    print(f"Loading weights from {weights_path}...")
    checkpoint = torch.load(weights_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    
    print("Running inference...")
    batch_size = 5000
    preds_list = []
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            # 使用高效搬運模式
            f_stm = torch.from_numpy(features_stm[i:end]).to(device, non_blocking=True)
            f_nstm = torch.from_numpy(features_nstm[i:end]).to(device, non_blocking=True)
            
            # 計算桶子 ID (SF18 邏輯: (pc-1)//4)
            pc = piece_counts[i:end]
            bids = torch.from_numpy(np.clip((pc - 1) // 4, 0, 7).astype(np.int64)).to(device, non_blocking=True)
            
            # 傳遞三個參數滿足 LayerStackNNUE 介面
            p = model(f_stm, f_nstm, bids).squeeze(-1)
            preds_list.append(p.cpu().numpy())
            
    preds_logit = np.concatenate(preds_list)
    
    K = 0.00368208
    preds_cp = preds_logit / K
    
    eps = 1e-6
    targets_clamped = np.clip(targets, eps, 1.0 - eps)
    targets_logit = -np.log(1.0 / targets_clamped - 1.0)
    
    # We use 0.0025 or whatever K was used to create the target.
    # In Stockfish it's sigmoid(score / 400). Here, we evaluate what our CP score means vs Stockfish CP score.
    # Let's derive it directly as CP.
    targets_cp = targets_logit / K
    
    abs_err = np.abs(targets_cp - preds_cp)
    
    print("\n" + "="*50)
    print("=== NNUE Evaluation Dispersion Summary ===")
    print("="*50)
    
    valid = np.abs(targets_cp) < 2000
    if np.sum(valid) > 0:
        overall_mae = np.mean(abs_err[valid])
        print(f"Overall MAE (|Target| < 2000 CP): {overall_mae:.2f} CP")
        
        bins = [
            ("0 - 100 CP   (Equal)    ", (np.abs(targets_cp) <= 100)),
            ("100 - 300 CP (Slight Adv)", (np.abs(targets_cp) > 100) & (np.abs(targets_cp) <= 300)),
            ("300 - 600 CP (Winning)   ", (np.abs(targets_cp) > 300) & (np.abs(targets_cp) <= 600)),
            ("> 600 CP     (Crushing)  ", (np.abs(targets_cp) > 600) & valid)
        ]
        
        print("\n--- Error Breakdown by Complexity (MAE) ---")
        for name, mask in bins:
            if np.sum(mask) > 0:
                bucket_mae = np.mean(abs_err[mask])
                print(f"{name}: MAE = {bucket_mae:6.2f} CP (N={np.sum(mask):<5})")
                
    else:
        print("No valid scores below 2000 CP found.")
        
    try:
        plt.figure(figsize=(10, 8))
        plt.scatter(targets_cp[valid], preds_cp[valid], alpha=0.1, color='#1f77b4', s=2)
        plt.plot([-1500, 1500], [-1500, 1500], color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
        plt.xlim(-1500, 1500)
        plt.ylim(-1500, 1500)
        plt.xlabel('Ground Truth Stockfish Score (Centipawns)', fontsize=12)
        plt.ylabel('NNUE Predicted Score (Centipawns)', fontsize=12)
        plt.title('Evaluation Dispersion Check: Target vs NNUE Prediction', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plot_path = os.path.dirname(os.path.abspath(__file__)) + '/dispersion_plot.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\n=> 📊 Scatter plot saved to {plot_path}")
    except Exception as e:
        print(f"Could not generate plot: {e}")

if __name__ == '__main__':
    analyze_dispersion()
