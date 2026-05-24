import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Add paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../chess_engine/nnue/ml_eval")))

from train import LayerStackNNUE, TrainingConfig, QuantConfig, fen_to_halfka_indices

def run_experiment(loss_type="bce", num_epochs=3):
    print(f"\n=== Running Experiment: Loss Type = {loss_type.upper()} for {num_epochs} epochs ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    dataset_path = "./tuner/ultimate_halfka_farseerT75.npz"
    data = np.load(dataset_path)
    
    # Use contiguous slice of 100,000 samples to keep it fast
    num_samples = 100000
    
    features_stm = data['features_stm'][:num_samples]
    features_nstm = data['features_nstm'][:num_samples]
    targets = data['targets'][:num_samples]
    piece_counts = data['piece_counts'][:num_samples]
    
    # Split into train/val
    train_size = int(0.8 * num_samples)
    
    # Model
    torch.manual_seed(42)
    model = LayerStackNNUE(num_buckets=8).to(device)
    # Initialize FC1 weights with a smaller, standard initialization scale
    nn.init.normal_(model.fc1.weight, mean=0.0, std=0.01)
    # Set the padding index row to zero
    with torch.no_grad():
        model.fc1.weight[22528].zero_()
    
    # Optimizer
    fc1_params = list(model.fc1.parameters()) + [model.fc1_bias]
    dense_params = (
        list(model.fact_fc2.parameters()) + list(model.fact_fc3.parameters()) + list(model.fact_fc4.parameters()) +
        list(model.delta_fc2.parameters()) + list(model.delta_fc3.parameters()) + list(model.delta_fc4.parameters())
    )
    optimizer = optim.AdamW([
        {'params': fc1_params,    'lr': 1.0e-4, 'weight_decay': 0.0},
        {'params': dense_params,  'lr': 8.75e-4, 'weight_decay': 1.0e-4},
    ])
    
    # Loss functions
    bce_criterion = nn.BCEWithLogitsLoss()
    mse_criterion = nn.MSELoss()
    
    # Training Loop
    batch_size = 4096
    num_batches = (train_size + batch_size - 1) // batch_size
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        # Shuffle
        perm = np.random.permutation(train_size)
        
        for b in range(num_batches):
            start = b * batch_size
            end = min(start + batch_size, train_size)
            batch_slice = perm[start:end]
            
            inputs_stm = torch.from_numpy(features_stm[batch_slice]).long().to(device)
            inputs_nstm = torch.from_numpy(features_nstm[batch_slice]).long().to(device)
            batch_targets = torch.from_numpy(targets[batch_slice]).to(device).unsqueeze(1).clamp(0.001, 0.999)
            pc = piece_counts[batch_slice].astype(np.int32)
            bucket_ids = torch.from_numpy(np.clip((pc - 1) // 4, 0, 7).astype(np.int64)).to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs_stm, inputs_nstm, bucket_ids)
            
            if loss_type == "bce":
                loss = bce_criterion(outputs, batch_targets)
            else:
                # WDL MSE Loss
                loss = mse_criterion(torch.sigmoid(outputs), batch_targets)
                
            loss.backward()
            optimizer.step()
            
            # Clamp FC1 weights as in train.py
            with torch.no_grad():
                model.fc1.weight.clamp_(-QuantConfig.FC1_MAX_WEIGHT, QuantConfig.FC1_MAX_WEIGHT)
                
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{num_epochs} | Avg Loss: {total_loss / num_batches:.6f}")
        
    # Evaluate weight statistics
    with torch.no_grad():
        fc1_w = model.fc1.weight.detach().cpu()
        print(f"FC1 weights: range=[{fc1_w.min():.4f}, {fc1_w.max():.4f}], mean={fc1_w.mean():.4f}, std={fc1_w.std():.4f}")
        
        # Evaluate "Black missing d8 queen" position
        fen = "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        stm_idx, nstm_idx, pc = fen_to_halfka_indices(fen)
        bucket_id = max(0, min(7, (pc - 1) // 4))
        
        inputs_stm = torch.tensor([stm_idx], dtype=torch.long, device=device)
        inputs_nstm = torch.tensor([nstm_idx], dtype=torch.long, device=device)
        bucket_ids = torch.tensor([bucket_id], dtype=torch.long, device=device)
        
        model.eval()
        logit = model(inputs_stm, inputs_nstm, bucket_ids).item()
        cp = logit / 0.0025
        print(f"Evaluation of FEN (Black missing d8 queen): Logit={logit:.4f}, CP={cp:+.1f}")

if __name__ == "__main__":
    # Run both BCE and MSE to compare
    run_experiment(loss_type="bce", num_epochs=10)
    run_experiment(loss_type="mse", num_epochs=10)
