#!/usr/bin/env python3
"""
NNUE Comprehensive Diagnostic Tool
Performs:
1. Parity checks (Float PyTorch vs Quantized Numba Inference) on typical positions.
2. Layer weight statistics (means, ranges, delta bucket updates).
3. (Optional) Dataset-wide activation analysis to detect dead neurons on validation data.
"""

import os
import sys
import numpy as np

# Adjust paths to import from chess_engine
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../chess_engine/nnue/ml_eval")))

try:
    import torch
    import torch.nn.functional as F
    from train import LayerStackNNUE, TrainingConfig, fen_to_halfka_indices
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from chess_engine.nnue.ml_eval.inference import (
    init_accumulator,
    nnue_forward_incremental,
    weights_loaded,
    warmup_numba_kernels
)
from chess_engine.classical.fen_parser import parse_fen

def print_section(title):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)

def evaluate_via_numba(fen):
    """Evaluates a FEN position using the compiled Numba inference engine."""
    piece_bbs, _, game_state = parse_fen(fen)
    side_to_move = int(game_state[0])
    
    accumulator_stack = np.zeros((1, 2, 512), dtype=np.int32)
    init_accumulator(piece_bbs, accumulator_stack)
    
    piece_count = 0
    for p_idx in range(12):
        bb = piece_bbs[p_idx]
        while bb:
            piece_count += 1
            bb &= bb - np.uint64(1)
            
    val = nnue_forward_incremental(0, side_to_move, accumulator_stack, piece_count)
    # Return from side_to_move perspective
    return val

def run_diagnostics():
    if not TORCH_AVAILABLE:
        print("Error: PyTorch is required to run the full diagnostics.")
        sys.exit(1)
        
    device = torch.device('cpu')
    model = LayerStackNNUE(num_buckets=8).to(device)
    
    weights_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../chess_engine/nnue/ml_eval/weights'))
    best_model_path = os.path.join(weights_dir, 'best_model.pth')
    
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'N/A')
        best_val_loss = checkpoint.get('best_val_loss', 'N/A')
        print(f"Loaded trained model weights from: {best_model_path}")
        print(f"Model Epoch: {epoch} | Best Validation Loss: {best_val_loss}")
    else:
        print(f"WARNING: Trained model checkpoint ({best_model_path}) not found.")
        print("Running diagnostics on freshly initialized (untrained) model weights.")
        
    model.eval()
    
    # --- 1. Parity Check Table ---
    print_section("PyTorch vs Quantized JIT Inference Parity Check")
    
    test_positions = [
        ("Starting pos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Missing a2 pawn", "rnbqkbnr/pppppppp/8/8/8/8/1PPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Missing e2 pawn", "rnbqkbnr/pppppppp/8/8/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"),
        ("Missing b1 knight", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/R1BQKBNR w KQkq - 0 1"),
        ("Missing c1 bishop", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RN1QKBNR w KQkq - 0 1"),
        ("Missing a1 rook", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/1NBQKBNR w KQkq - 0 1"),
        ("Missing d1 queen", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1"),
        ("White Queen Advantage", "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Double King Endgame", "8/8/8/4k3/8/8/3K4/8 w - - 0 1")
    ]
    
    print(f"{'Position Description':<25} | {'PC':>3} | {'Bkt':>3} | {'PyTorch Logit':>14} | {'Float CP':>10} | {'Quant CP (JIT)':>15} | {'Diff':>6}")
    print("-" * 90)
    
    with torch.no_grad():
        for name, fen in test_positions:
            stm_idx, nstm_idx, pc = fen_to_halfka_indices(fen)
            bucket_id = max(0, min(7, (pc - 1) // 4))
            
            inputs_stm = torch.tensor([stm_idx], dtype=torch.long, device=device)
            inputs_nstm = torch.tensor([nstm_idx], dtype=torch.long, device=device)
            bucket_ids = torch.tensor([bucket_id], dtype=torch.long, device=device)
            
            logit = model(inputs_stm, inputs_nstm, bucket_ids).item()
            float_cp = logit / 0.0025
            
            # Simulated quantized value from PyTorch logit
            int_output = logit * 16256
            if int_output >= 0:
                py_quant_cp = int(int_output) // 41
            else:
                py_quant_cp = -(int(-int_output) // 41)
                
            # Actual JIT quantized value
            jit_quant_cp = evaluate_via_numba(fen)
            
            diff = jit_quant_cp - py_quant_cp
            
            print(f"{name:<25} | {pc:>3} | {bucket_id:>3} | {logit:>14.6f} | {float_cp:>+10.1f} | {jit_quant_cp:>+15d} | {diff:>+6d}")

    # --- 2. Weight Statistics Analysis ---
    print_section("NNUE Model Weight Statistics")
    with torch.no_grad():
        sd = model.state_dict()
        fc1_w = sd['fc1.weight']
        print(f"FC1 Weight Space: shape={list(fc1_w.shape)}, range=[{fc1_w.min().item():.4f}, {fc1_w.max().item():.4f}], std={fc1_w.std().item():.4f}")
        print(f"FC1 Bias Space  : range=[{model.fc1_bias.min().item():.4f}, {model.fc1_bias.max().item():.4f}], mean={model.fc1_bias.mean().item():.4f}")
        print("-" * 60)
        
        for lname in ['fc2', 'fc3', 'fc4']:
            fact_layer = getattr(model, f'fact_{lname}')
            w_std = fact_layer.weight.std().item()
            b_min = fact_layer.bias.min().item()
            b_max = fact_layer.bias.max().item()
            print(f"FC{lname[-1]} Fact (Global)  : weight_std={w_std:.6f}, bias_range=[{b_min:.4f}, {b_max:.4f}]")
            
            # Buckets Delta analysis
            delta_layer_list = getattr(model, f'delta_{lname}')
            delta_means = [delta_layer_list[b].weight.abs().mean().item() for b in range(8)]
            delta_bias_means = [delta_layer_list[b].bias.abs().mean().item() for b in range(8)]
            print(f"FC{lname[-1]} Delta (Buckets) : mean_abs_weight range=[{min(delta_means):.6f} ~ {max(delta_means):.6f}], mean_abs_bias range=[{min(delta_bias_means):.6f} ~ {max(delta_bias_means):.6f}]")
            print("-" * 60)

    # --- 3. Dataset-wide Activation Diagnostics (Optional) ---
    dataset_path = "./tuner/ultimate_halfka_farseerT75.npz"
    if os.path.exists(dataset_path):
        print_section("Dataset-wide Activation Diagnostics (10,000 Validation Samples)")
        print(f"Loading validation slice from {dataset_path}...")
        try:
            data = np.load(dataset_path)
            features_stm = data['features_stm']
            features_nstm = data['features_nstm']
            piece_counts = data['piece_counts']
            targets = data['targets']
            
            # Sample 10,000 random samples
            np.random.seed(42)
            val_size = len(targets)
            sample_size = min(10000, val_size)
            indices = np.random.choice(val_size, size=sample_size, replace=False)
            
            val_stm = torch.tensor(features_stm[indices], dtype=torch.long, device=device)
            val_nstm = torch.tensor(features_nstm[indices], dtype=torch.long, device=device)
            val_bkt = torch.tensor(np.clip((piece_counts[indices] - 1) // 4, 0, 7), dtype=torch.long, device=device)
            
            with torch.no_grad():
                acc_stm = model.crelu(model.fc1(val_stm) + model.fc1_bias)
                acc_nstm = model.crelu(model.fc1(val_nstm) + model.fc1_bias)
                x = torch.cat((acc_stm, acc_nstm), dim=1) # (N, 1024)
                
                fc1_active_per_sample = (acc_stm > 0).float().sum(dim=1).mean().item()
                print(f"FC1 STM Accumulator (512 dims): average active dimensions per sample = {fc1_active_per_sample:.2f}/512 ({(fc1_active_per_sample/5.12):.1f}%)")
                
                # Analyze Bucket 7 (Opening/Middlegame) and Bucket 0 (Endgame)
                for b in [0,1,2,3,4,5,6,7]:
                    mask = (val_bkt == b)
                    bkt_samples = mask.sum().item()
                    if bkt_samples == 0:
                        continue
                        
                    xb = x[mask]
                    
                    # FC2
                    w2 = model.fact_fc2.weight + model.delta_fc2[b].weight
                    b2 = model.fact_fc2.bias + model.delta_fc2[b].bias
                    h2_pre = F.linear(xb, w2, b2)
                    h2 = model.crelu(h2_pre)
                    fc2_dead = (h2_pre.max(dim=0)[0] <= 0).sum().item()
                    fc2_act = (h2 > 0).float().mean(dim=1).mean().item() * 32
                    
                    # FC3
                    w3 = model.fact_fc3.weight + model.delta_fc3[b].weight
                    b3 = model.fact_fc3.bias + model.delta_fc3[b].bias
                    h3_pre = F.linear(h2, w3, b3)
                    h3 = model.crelu(h3_pre)
                    fc3_dead = (h3_pre.max(dim=0)[0] <= 0).sum().item()
                    fc3_act = (h3 > 0).float().mean(dim=1).mean().item() * 32
                    
                    print(f"\n--- Bucket {b} ({bkt_samples} samples) ---")
                    print(f"  FC2 Layer: {32 - fc2_dead}/32 ever-active | {fc2_dead}/32 dead | avg active/sample = {fc2_act:.2f}/32 ({(fc2_act*100/32):.1f}%)")
                    print(f"  FC3 Layer: {32 - fc3_dead}/32 ever-active | {fc3_dead}/32 dead | avg active/sample = {fc3_act:.2f}/32 ({(fc3_act*100/32):.1f}%)")
                    
        except Exception as e:
            print(f"Could not load/analyze dataset: {e}")
    else:
        print_section("Dataset-wide Activation Diagnostics")
        print(f"Dataset not found at {dataset_path}. Skipping dataset-wide diagnostics.")

if __name__ == "__main__":
    run_diagnostics()
