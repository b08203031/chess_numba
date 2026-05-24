import os
import sys
import torch
import torch.nn as nn

workspace_dir = r"c:\Users\ren cian\OneDrive\桌面\chess\chess_numba\chess_numba"
sys.path.insert(0, workspace_dir)

from chess_engine.nnue.ml_eval.train import LayerStackNNUE, fen_to_halfka_indices

def main():
    device = torch.device('cpu')
    model = LayerStackNNUE(num_buckets=8).to(device)
    
    # Enable gradients
    model.train()
    
    # Prepare a batch with the starting position
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    stm_idx, nstm_idx, pc = fen_to_halfka_indices(fen)
    bucket_id = max(0, min(7, (pc - 1) // 4))
    
    inputs_stm = torch.tensor([stm_idx], dtype=torch.long, device=device)
    inputs_nstm = torch.tensor([nstm_idx], dtype=torch.long, device=device)
    bucket_ids = torch.tensor([bucket_id], dtype=torch.long, device=device)
    
    target = torch.tensor([[0.5]], dtype=torch.float, device=device) # Draw target
    
    # Forward pass
    outputs = model(inputs_stm, inputs_nstm, bucket_ids)
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(outputs, target)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    fc1_grad = model.fc1.weight.grad
    fc1_bias_grad = model.fc1_bias.grad
    fact_fc2_grad = model.fact_fc2.weight.grad
    
    print("=== Gradient Diagnostics ===")
    if fc1_grad is None:
        print("model.fc1.weight.grad is None! (Gradient not flowing to FC1!)")
    else:
        # Get gradients of active features (indices in stm_idx and nstm_idx)
        active_grads = []
        for idx in stm_idx + nstm_idx:
            if idx != 22528:
                grad_norm = torch.norm(fc1_grad[idx]).item()
                active_grads.append(grad_norm)
        
        print(f"FC1 weight gradient shape: {fc1_grad.shape}")
        print(f"Active FC1 features gradient norms (mean): {sum(active_grads)/len(active_grads):.6f}")
        print(f"Active FC1 features gradient norms (max): {max(active_grads):.6f}")
        
        # Get gradient of inactive features
        inactive_indices = [i for i in range(22528) if i not in stm_idx and i not in nstm_idx]
        inactive_grads = [torch.norm(fc1_grad[i]).item() for i in inactive_indices[:100]]
        print(f"Inactive FC1 features gradient norms (mean): {sum(inactive_grads)/len(inactive_grads):.6f}")

    if fc1_bias_grad is None:
        print("model.fc1_bias.grad is None!")
    else:
        print(f"FC1 bias gradient norm: {torch.norm(fc1_bias_grad).item():.6f}")
        
    if fact_fc2_grad is None:
        print("model.fact_fc2.weight.grad is None!")
    else:
        print(f"Fact FC2 weight gradient norm: {torch.norm(fact_fc2_grad).item():.6f}")

if __name__ == "__main__":
    main()
