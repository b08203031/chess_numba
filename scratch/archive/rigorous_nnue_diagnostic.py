import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chess_engine.nnue.ml_eval.train import LayerStackNNUE, TrainingConfig

def run_diagnostic():
    device = torch.device("cpu")
    
    # 1. Load dataset
    dataset_path = "./tuner/ultimate_halfka_farseerT75.npz"
    if not os.path.exists(dataset_path):
        print(f"Error: dataset not found at {dataset_path}")
        return
        
    print(f"Loading dataset from {dataset_path}...")
    data = np.load(dataset_path)
    features_stm = data['features_stm']
    features_nstm = data['features_nstm']
    piece_counts = data['piece_counts']
    targets = data['targets']
    
    total_samples = len(targets)
    print(f"Total samples in dataset: {total_samples:,}")
    
    # Select 20,000 random validation samples
    np.random.seed(42)
    val_indices = np.random.choice(total_samples, size=20000, replace=False)
    
    val_stm = torch.tensor(features_stm[val_indices], dtype=torch.long, device=device)
    val_nstm = torch.tensor(features_nstm[val_indices], dtype=torch.long, device=device)
    val_pc = piece_counts[val_indices]
    val_bkt = torch.tensor(np.clip((val_pc - 1) // 4, 0, 7), dtype=torch.long, device=device)
    
    print(f"Selected {len(val_indices):,} validation samples for diagnostics.")
    
    # 2. Instantiate Fresh Model (Epoch 0 Baseline)
    torch.manual_seed(42)
    model_fresh = LayerStackNNUE(num_buckets=8).to(device)
    model_fresh.eval()
    
    # 3. Instantiate Trained Model
    model_trained = LayerStackNNUE(num_buckets=8).to(device)
    weights_path = "./chess_engine/nnue/ml_eval/weights/best_model.pth"
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device)
        model_trained.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded trained model from {weights_path} (Epoch {checkpoint.get('epoch', 'unknown')})")
    else:
        print(f"Warning: trained model weights not found at {weights_path}, running only fresh baseline.")
        model_trained = None

    def analyze_model(model, label):
        print(f"\n==========================================")
        print(f" Analyzing Model: {label}")
        print(f"==========================================")
        
        with torch.no_grad():
            # Pass 1: FC1 Accumulator
            acc_stm = model.crelu(model.fc1(val_stm) + model.fc1_bias)
            acc_nstm = model.crelu(model.fc1(val_nstm) + model.fc1_bias)
            x = torch.cat((acc_stm, acc_nstm), dim=1) # (N, 1024)
            
            # Print FC1 stats
            fc1_active_per_sample = (acc_stm > 0).float().sum(dim=1)
            print(f"FC1 (512 dims, STM accumulator):")
            print(f"  Active features per sample: mean={fc1_active_per_sample.mean().item():.2f}, min={fc1_active_per_sample.min().item():.0f}, max={fc1_active_per_sample.max().item():.0f}")
            print(f"  Values of active features: mean={acc_stm[acc_stm > 0].mean().item():.4f}, std={acc_stm[acc_stm > 0].std().item():.4f}, max={acc_stm.max().item():.4f}")
            
            # We will inspect Bucket 7 (Middle-to-Opening game, piece count 29-32)
            # and Bucket 0 (Endgame, piece count 2-4)
            for b in [7, 0]:
                mask = (val_bkt == b)
                n_bkt_samples = mask.sum().item()
                if n_bkt_samples == 0:
                    print(f"\nNo samples in Bucket {b}")
                    continue
                    
                xb = x[mask]
                
                # FC2 layer for bucket b
                w2 = model.fact_fc2.weight + model.delta_fc2[b].weight
                b2 = model.fact_fc2.bias + model.delta_fc2[b].bias
                h2_pre = F.linear(xb, w2, b2) # (N_bkt, 32)
                h2 = model.crelu(h2_pre)
                
                # FC3 layer for bucket b
                w3 = model.fact_fc3.weight + model.delta_fc3[b].weight
                b3 = model.fact_fc3.bias + model.delta_fc3[b].bias
                h3_pre = F.linear(h2, w3, b3) # (N_bkt, 32)
                h3 = model.crelu(h3_pre)
                
                print(f"\n--- Bucket {b} Analysis ({n_bkt_samples} samples) ---")
                
                # Analyze FC2 Neurons
                fc2_max_pre = h2_pre.max(dim=0)[0] # max pre-activation per neuron
                fc2_mean_pre = h2_pre.mean(dim=0)
                fc2_dead = (fc2_max_pre <= 0).sum().item()
                
                fc2_active_ratio_per_sample = (h2 > 0).float().mean(dim=1)
                
                print(f"FC2 Layer (32 neurons):")
                print(f"  Ever-active neurons (max pre-act > 0): {32 - fc2_dead}/32")
                print(f"  Dead neurons (max pre-act <= 0): {fc2_dead}/32")
                print(f"  Active neurons per sample: mean={fc2_active_ratio_per_sample.mean().item()*32:.2f}/32 ({(fc2_active_ratio_per_sample.mean().item()*100):.1f}%)")
                print(f"  Neuron max pre-activations: min={fc2_max_pre.min().item():.4f}, max={fc2_max_pre.max().item():.4f}, mean={fc2_max_pre.mean().item():.4f}")
                print(f"  Neuron mean pre-activations: min={fc2_mean_pre.min().item():.4f}, max={fc2_mean_pre.max().item():.4f}, mean={fc2_mean_pre.mean().item():.4f}")
                
                # Analyze FC3 Neurons
                fc3_max_pre = h3_pre.max(dim=0)[0]
                fc3_mean_pre = h3_pre.mean(dim=0)
                fc3_dead = (fc3_max_pre <= 0).sum().item()
                
                fc3_active_ratio_per_sample = (h3 > 0).float().mean(dim=1)
                
                print(f"FC3 Layer (32 neurons):")
                print(f"  Ever-active neurons (max pre-act > 0): {32 - fc3_dead}/32")
                print(f"  Dead neurons (max pre-act <= 0): {fc3_dead}/32")
                print(f"  Active neurons per sample: mean={fc3_active_ratio_per_sample.mean().item()*32:.2f}/32 ({(fc3_active_ratio_per_sample.mean().item()*100):.1f}%)")
                print(f"  Neuron max pre-activations: min={fc3_max_pre.min().item():.4f}, max={fc3_max_pre.max().item():.4f}, mean={fc3_max_pre.mean().item():.4f}")
                print(f"  Neuron mean pre-activations: min={fc3_mean_pre.min().item():.4f}, max={fc3_mean_pre.max().item():.4f}, mean={fc3_mean_pre.mean().item():.4f}")

    analyze_model(model_fresh, "Fresh Initialization (Epoch 0)")
    if model_trained is not None:
        analyze_model(model_trained, "Trained Model (best_model.pth)")

if __name__ == "__main__":
    run_diagnostic()
