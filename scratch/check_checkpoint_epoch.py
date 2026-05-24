import os
import sys
import torch

workspace_dir = r"c:\Users\ren cian\OneDrive\桌面\chess\chess_numba\chess_numba"
sys.path.insert(0, workspace_dir)

def check_checkpoint(path):
    if not os.path.exists(path):
        print(f"{os.path.basename(path)} not found")
        return
        
    try:
        checkpoint = torch.load(path, map_location='cpu')
        print(f"=== {os.path.basename(path)} ===")
        print(f"  Epoch: {checkpoint.get('epoch')}")
        print(f"  Best Val Loss: {checkpoint.get('best_val_loss')}")
        print(f"  Feature Version: {checkpoint.get('feature_encoding_version')}")
    except Exception as e:
        print(f"  Error loading {os.path.basename(path)}: {e}")

def main():
    weights_dir = os.path.join(workspace_dir, 'chess_engine/nnue/ml_eval/weights')
    check_checkpoint(os.path.join(weights_dir, 'latest_model.pth'))
    check_checkpoint(os.path.join(weights_dir, 'best_model.pth'))

if __name__ == "__main__":
    main()
