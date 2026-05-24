import os
import sys
import numpy as np
import math

workspace_dir = r"c:\Users\ren cian\OneDrive\桌面\chess\chess_numba\chess_numba"
sys.path.insert(0, workspace_dir)

K = 0.0025

def wdl_to_cp(wdl):
    # wdl = 1.0 / (1.0 + exp(-K * cp))
    # 1.0 / wdl - 1 = exp(-K * cp)
    # -K * cp = log(1.0 / wdl - 1)
    # cp = -log(1.0 / wdl - 1) / K
    if wdl <= 0.001:
        return -10000.0
    if wdl >= 0.999:
        return 10000.0
    return -math.log(1.0 / wdl - 1.0) / K

def main():
    npz_path = os.path.join(workspace_dir, "tuner/ultimate_halfka_farseerT75.npz")
    if not os.path.exists(npz_path):
        print("NPZ not found")
        return
        
    print("Loading NPZ targets...")
    data = np.load(npz_path)
    targets = data['targets']
    
    print(f"Total samples: {len(targets)}")
    print(f"Min target (WDL): {np.min(targets):.6f}")
    print(f"Max target (WDL): {np.max(targets):.6f}")
    print(f"Mean target (WDL): {np.mean(targets):.6f}")
    print(f"Std target (WDL): {np.std(targets):.6f}")
    
    # Sample some targets and convert to centipawns
    cp_scores = []
    # Take a subset of 1 million samples to speed up
    subset = targets[:1000000]
    for w in subset:
        cp_scores.append(wdl_to_cp(w))
        
    cp_scores = np.array(cp_scores)
    print(f"\nCentipawn equivalent stats (on 1M sample subset):")
    print(f"Min CP: {np.min(cp_scores):.1f}")
    print(f"Max CP: {np.max(cp_scores):.1f}")
    print(f"Mean CP: {np.mean(cp_scores):.1f}")
    print(f"Std CP: {np.std(cp_scores):.1f}")
    
    print("\nFirst 10 WDL targets and their CP equivalents:")
    for i in range(10):
        print(f"  Sample {i+1}: WDL = {targets[i]:.4f} -> CP = {wdl_to_cp(targets[i]):.1f}")

if __name__ == "__main__":
    main()
