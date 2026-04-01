import os
import torch
import numpy as np
from chess_engine.ml_eval.train import SimpleNNUE, export_weights

def reexport():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNNUE().to(device)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(script_dir, 'weights')
    best_model_path = os.path.join(weights_dir, 'best_model.pth')
    
    if os.path.exists(best_model_path):
        print(f"Loading {best_model_path}...")
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully.")
        export_weights(model)
        print("Weights re-exported in new INT16 format.")
    else:
        print(f"Error: {best_model_path} not found.")

if __name__ == '__main__':
    reexport()
