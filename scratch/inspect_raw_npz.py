import os
import sys
import numpy as np

workspace_dir = r"c:\Users\ren cian\OneDrive\桌面\chess\chess_numba\chess_numba"
sys.path.insert(0, workspace_dir)

def main():
    npz_path = os.path.join(workspace_dir, "tuner/ultimate_halfka_farseerT75.npz")
    if not os.path.exists(npz_path):
        print("NPZ not found")
        return
        
    print("Loading NPZ...")
    data = np.load(npz_path)
    
    # We want to check what is in the features array.
    # In train.py, the npz dataset has 'features_stm' and 'features_nstm'.
    # These are already feature indices (integers from 0 to 22528), not raw bullet records!
    # Ah! The NPZ file doesn't contain raw bullet records; it contains precomputed feature indices!
    # So the features_stm and features_nstm were already converted using convert_bullet_bin.py.
    
    # Let's check the first sample features_stm list:
    f_stm = data['features_stm'][0]
    print("Sample 1 features_stm:", f_stm)

if __name__ == "__main__":
    main()
