import os
import sys
import numpy as np

workspace_dir = r"c:\Users\ren cian\OneDrive\桌面\chess\chess_numba\chess_numba"
sys.path.insert(0, workspace_dir)

def main():
    weights_path = os.path.join(workspace_dir, "chess_engine/nnue/ml_eval/weights/fc1_weight.npy")
    if not os.path.exists(weights_path):
        print("fc1_weight.npy not found")
        return
        
    print("Loading fc1_weight...")
    fc1_weight = np.load(weights_path) # Shape: (22528, 512)
    print(f"Shape: {fc1_weight.shape}")
    
    # Feature size: 22528 = 32 buckets * 704 (11 channels * 64 squares)
    # Channel mapping:
    # 0: own P, 1: opp P
    # 2: own N, 3: opp N
    # 4: own B, 5: opp B
    # 6: own R, 7: opp R
    # 8: own Q, 9: opp Q
    # 10: King
    channel_names = {
        0: "own Pawn", 1: "opp Pawn",
        2: "own Knight", 3: "opp Knight",
        4: "own Bishop", 5: "opp Bishop",
        6: "own Rook", 7: "opp Rook",
        8: "own Queen", 9: "opp Queen",
        10: "King"
    }
    
    print("\n--- Weight magnitudes per channel ---")
    print(f"{'Channel':<15} | {'Mean L2 Norm':<15} | {'Max L2 Norm':<15} | {'Mean Abs Weight':<18}")
    print("-" * 70)
    
    for c in range(11):
        # Gather all indices across all 32 buckets and 64 squares for channel c
        indices = []
        for b in range(32):
            for sq in range(64):
                idx = b * 704 + c * 64 + sq
                indices.append(idx)
                
        c_weights = fc1_weight[indices] # Shape: (32*64, 512)
        
        # Calculate magnitudes
        l2_norms = np.linalg.norm(c_weights, axis=1) # Shape: (2048,)
        mean_l2 = np.mean(l2_norms)
        max_l2 = np.max(l2_norms)
        mean_abs = np.mean(np.abs(c_weights))
        
        name = channel_names.get(c)
        print(f"{name:<15} | {mean_l2:>15.4f} | {max_l2:>15.4f} | {mean_abs:>18.4f}")

if __name__ == "__main__":
    main()
