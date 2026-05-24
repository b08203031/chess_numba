import torch
import os
import numpy as np
from train import LayerStackNNUE, TrainingConfig, export_weights

def quantize():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LayerStackNNUE().to(device)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(script_dir, 'weights')
    best_model_path = os.path.join(weights_dir, 'best_model.pth')
    
    if os.path.exists(best_model_path):
        print(f"Loading existing best model from {best_model_path}...")
        checkpoint = torch.load(best_model_path, map_location=device)
        checkpoint_version = checkpoint.get('feature_encoding_version')
        if checkpoint_version != TrainingConfig.FEATURE_ENCODING_VERSION:
            print(
                f"Error: feature encoding mismatch: checkpoint={checkpoint_version}, "
                f"current={TrainingConfig.FEATURE_ENCODING_VERSION}"
            )
            return
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded. Starting quantization export...")
        export_weights(model)
        print("Quantization complete! Performing post-validation check...")
        
        # Post-validation check
        meta_path = os.path.join(weights_dir, 'metadata.json')
        if not os.path.exists(meta_path):
            raise FileNotFoundError("Post-validation failed: metadata.json was not created.")
            
        import json
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
            
        if meta.get("feature_encoding_version") != TrainingConfig.FEATURE_ENCODING_VERSION:
            raise ValueError(f"Post-validation failed: feature_encoding_version mismatch in metadata.json. Expected: {TrainingConfig.FEATURE_ENCODING_VERSION}, Got: {meta.get('feature_encoding_version')}")
            
        # Verify file existence and SHA256 hashes
        import hashlib
        def get_sha256(filepath):
            h = hashlib.sha256()
            with open(filepath, 'rb') as f:
                h.update(f.read())
            return h.hexdigest()
            
        for filename, expected_hash in meta.get("weights_sha256", {}).items():
            filepath = os.path.join(weights_dir, filename)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Post-validation failed: weight file {filename} is missing.")
            current_hash = get_sha256(filepath)
            if current_hash != expected_hash:
                raise ValueError(f"Post-validation failed: SHA256 hash mismatch for {filename}.")
                
        print("Post-validation verification successful! All weight files are intact and match metadata.")
    else:
        print(f"Error: {best_model_path} not found.")

if __name__ == "__main__":
    quantize()
