import os
import sys
import numpy as np

workspace_dir = r"c:\Users\ren cian\OneDrive\桌面\chess\chess_numba\chess_numba"
sys.path.insert(0, workspace_dir)

# Helper to decode feature
def decode_feature(f):
    if f == 22528:
        return "PADDING"
    bucket = f // 704
    rem = f % 704
    channel = rem // 64
    sq = rem % 64
    
    channel_names = {
        0: "own P", 1: "opp P",
        2: "own N", 3: "opp N",
        4: "own B", 5: "opp B",
        6: "own R", 7: "opp R",
        8: "own Q", 9: "opp Q",
        10: "King"
    }
    
    sq_name = f"{chr(ord('a') + sq % 8)}{sq // 8 + 1}"
    return f"{channel_names.get(channel, f'Ch{channel}')}@{sq_name}"

def main():
    npz_path = os.path.join(workspace_dir, "tuner/ultimate_halfka_farseerT75.npz")
    if not os.path.exists(npz_path):
        print(f"NPZ dataset not found at {npz_path}")
        return
        
    print("Loading NPZ dataset...")
    data = np.load(npz_path)
    
    features_stm = data['features_stm']
    features_nstm = data['features_nstm']
    targets = data['targets']
    piece_counts = data['piece_counts']
    
    print(f"Total samples: {len(targets)}")
    
    # Reconstruct positions from features
    for idx in range(3):
        print(f"\n--- Sample {idx+1} ---")
        print(f"Target (WDL): {targets[idx]:.4f}, Piece Count: {piece_counts[idx]}")
        
        # stm features
        stm_feats = features_stm[idx]
        print("STM active features:")
        for f in stm_feats:
            if f != 22528:
                print(f"  {f}: {decode_feature(f)}")
                
        nstm_feats = features_nstm[idx]
        print("NSTM active features:")
        for f in nstm_feats:
            if f != 22528:
                print(f"  {f}: {decode_feature(f)}")

if __name__ == "__main__":
    main()
