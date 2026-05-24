import os
import sys
import numpy as np
import torch

workspace_dir = r"c:\Users\ren cian\OneDrive\桌面\chess\chess_numba\chess_numba"
sys.path.insert(0, workspace_dir)

from chess_engine.nnue.ml_eval.train import LayerStackNNUE, QuantConfig

def main():
    device = torch.device('cpu')
    
    # 1. Load best_model.pth
    best_model_path = os.path.join(workspace_dir, 'chess_engine/nnue/ml_eval/weights/best_model.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model_state = checkpoint['model_state_dict']
        fc1_weight_pth = model_state['fc1.weight'].numpy()[:22528]
        print(f"Loaded best_model.pth fc1.weight mean absolute: {np.mean(np.abs(fc1_weight_pth)):.6f}")
        print(f"Loaded best_model.pth fc1.weight std: {np.std(fc1_weight_pth):.6f}")
    else:
        print("best_model.pth not found")
        return

    # 2. Load exported fc1_weight.npy
    exported_path = os.path.join(workspace_dir, 'chess_engine/nnue/ml_eval/weights/fc1_weight.npy')
    if os.path.exists(exported_path):
        fc1_weight_npy = np.load(exported_path)
        print(f"Loaded fc1_weight.npy mean absolute: {np.mean(np.abs(fc1_weight_npy)):.6f}")
        print(f"Loaded fc1_weight.npy std: {np.std(fc1_weight_npy):.6f}")
    else:
        print("fc1_weight.npy not found")

    # 3. Initialize a fresh random model
    torch.manual_seed(12345)
    fresh_model = LayerStackNNUE(num_buckets=8)
    fresh_fc1 = fresh_model.fc1.weight.detach().numpy()[:22528]
    print(f"Fresh random model fc1.weight mean absolute: {np.mean(np.abs(fresh_fc1)):.6f}")
    print(f"Fresh random model fc1.weight std: {np.std(fresh_fc1):.6f}")
    
    # Check if the weights in best_model.pth differ from the fresh model
    # (i.e. has the model actually updated its weights during training?)
    diff_from_fresh = np.mean(np.abs(fc1_weight_pth - fresh_fc1))
    print(f"Mean absolute difference from fresh random model: {diff_from_fresh:.6f}")
    
    # Let's check if the weights have changed from a different random initialization seed
    torch.manual_seed(42) # The seed used in train.py
    seed_42_model = LayerStackNNUE(num_buckets=8)
    seed_42_fc1 = seed_42_model.fc1.weight.detach().numpy()[:22528]
    diff_from_seed_42 = np.mean(np.abs(fc1_weight_pth - seed_42_fc1))
    print(f"Mean absolute difference from seed 42 random model: {diff_from_seed_42:.6f}")

if __name__ == "__main__":
    main()
