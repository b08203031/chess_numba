import torch
import os
import numpy as np
from train import LayerStackNNUE, export_weights

def quantize():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LayerStackNNUE().to(device)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(script_dir, 'weights')
    best_model_path = os.path.join(weights_dir, 'best_model.pth')
    
    if os.path.exists(best_model_path):
        print(f"Loading existing best model from {best_model_path}...")
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded. Starting quantization export...")
        export_weights(model)
        print("Quantization complete!")
    else:
        print(f"Error: {best_model_path} not found.")

if __name__ == "__main__":
    quantize()
