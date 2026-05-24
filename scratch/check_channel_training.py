import os
import sys
import numpy as np
import torch

workspace_dir = r"c:\Users\ren cian\OneDrive\桌面\chess\chess_numba\chess_numba"
sys.path.insert(0, workspace_dir)

from chess_engine.nnue.ml_eval.train import LayerStackNNUE

def main():
    device = torch.device('cpu')
    
    # Load best_model.pth
    best_model_path = os.path.join(workspace_dir, 'chess_engine/nnue/ml_eval/weights/best_model.pth')
    if not os.path.exists(best_model_path):
        print("best_model.pth not found")
        return
        
    checkpoint = torch.load(best_model_path, map_location=device)
    model_state = checkpoint['model_state_dict']
    fc1_weight_pth = model_state['fc1.weight'].numpy()[:22528]
    
    # Initialize seed 42 model (initial weights)
    torch.manual_seed(42)
    init_model = LayerStackNNUE(num_buckets=8)
    init_fc1 = init_model.fc1.weight.detach().numpy()[:22528]
    
    channel_names = {
        0: "own Pawn", 1: "opp Pawn",
        2: "own Knight", 3: "opp Knight",
        4: "own Bishop", 5: "opp Bishop",
        6: "own Rook", 7: "opp Rook",
        8: "own Queen", 9: "opp Queen",
        10: "King"
    }
    
    print(f"{'Channel':<15} | {'Mean Abs Change':<18} | {'Max Abs Change':<18}")
    print("-" * 57)
    
    for c in range(11):
        indices = []
        for b in range(32):
            for sq in range(64):
                idx = b * 704 + c * 64 + sq
                indices.append(idx)
                
        c_init = init_fc1[indices]
        c_trained = fc1_weight_pth[indices]
        
        diff = np.abs(c_trained - c_init)
        mean_diff = np.mean(diff)
        max_diff = np.max(diff)
        
        name = channel_names.get(c)
        print(f"{name:<15} | {mean_diff:>18.6f} | {max_diff:>18.6f}")

if __name__ == "__main__":
    main()
