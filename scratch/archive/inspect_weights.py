import torch
import os

weights_path = r"c:\Users\ren cian\OneDrive\桌面\chess\chess_numba\chess_numba\chess_engine\nnue\ml_eval\weights\best_model.pth"
if os.path.exists(weights_path):
    checkpoint = torch.load(weights_path, map_location="cpu")
    sd = checkpoint['model_state_dict']
    for k, v in sd.items():
        if 'weight' in k or 'bias' in k:
            print(f"{k:<30} | shape: {str(list(v.shape)):<15} | mean: {v.mean().item():+.6f} | std: {v.std().item():.6f} | min: {v.min().item():+.6f} | max: {v.max().item():+.6f}")
else:
    print("Weights not found.")
