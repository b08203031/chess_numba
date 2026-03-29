import os
import numpy as np
import time
import numba

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim.lr_scheduler import StepLR
    # Using modern torch.amp for future compatibility
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch is not installed. You will need to install PyTorch to run this script: pip install torch")

try:
    import matplotlib.pyplot as plt
    from IPython import display
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Matplotlib is not installed. Real-time plotting will be disabled. pip install matplotlib")


                                      


if TORCH_AVAILABLE:
    class ClippedReLU(nn.Module):
        def __init__(self, upper_bound=127.0):
            super(ClippedReLU, self).__init__()
            self.upper_bound = upper_bound
            
        def forward(self, x):
            return torch.clamp(x, 0.0, self.upper_bound)

    class SimpleNNUE(nn.Module):
        """
        HalfKA architecture:
        45056 Sparse -> 256x2 (ClippedReLU) -> 512 -> 32 -> 32 -> 1
        """
        def __init__(self):
            super(SimpleNNUE, self).__init__()
            # Sparse input layer (EmbeddingBag mode=sum is mathematically identical to Sparse Linear)
            # 45056 is the padding index (it will output 0 vectors)
            self.fc1 = nn.EmbeddingBag(45057, 256, mode="sum", padding_idx=45056)
            self.fc1_bias = nn.Parameter(torch.zeros(256))
            
            self.fc2 = nn.Linear(512, 32)
            self.fc3 = nn.Linear(32, 32)
            self.fc4 = nn.Linear(32, 1)
            self.crelu = ClippedReLU(127.0)
            
        def forward(self, x_stm, x_nstm):
            # Pass both perspectives through the same EmbeddingBag + Bias
            acc_stm = self.crelu(self.fc1(x_stm) + self.fc1_bias)
            acc_nstm = self.crelu(self.fc1(x_nstm) + self.fc1_bias)
            
            # Concatenate them: [STM (256), NSTM (256)] -> (512,)
            x = torch.cat((acc_stm, acc_nstm), dim=1)
            
            x = self.crelu(self.fc2(x))
            x = self.crelu(self.fc3(x))
            x = self.fc4(x)
            return x

def plot_loss(epoch, train_losses, val_losses):
    if not PLOTTING_AVAILABLE:
        return
        
    plt.ion() # Enable interactive mode
    plt.clf()
    plt.plot(range(1, epoch + 2), train_losses, label='Train Loss', color='blue', marker='o')
    plt.plot(range(1, epoch + 2), val_losses, label='Validation Loss', color='orange', marker='o')
    plt.title('NNUE Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (BCEWithLogitsLoss)')
    plt.legend()
    plt.grid(True)
    plt.draw()
    plt.pause(0.1)

# --- 訓練超參數集中區 (Training Hyperparameters) ---
class TrainingConfig:
    DATASET_PATH = './tuner/ultimate_halfka.npz'
    BATCH_SIZE = 16384 # 提升 Batch Size 減少任務分發開銷
    LEARNING_RATE = 0.0001 #從頭的話：0.01
    
    # 學習率遞減設定 (StepLR)
    SCHEDULER_STEP = 3
    SCHEDULER_GAMMA = 0.5 # 減緩下降速度
    
    # 訓練與停止邏輯
    TOTAL_EPOCHS = 200
    EARLY_STOPPING_PATIENCE = 20
    RESUME_FROM_CHECKPOINT = False # False: 僅讀取上次權重，但不接續 Epoch 數字與學習率
    
    # 資料分割比例
    VAL_SPLIT = 0.1

def export_weights(model):
    """Export weights to NumPy format for Numba inference."""
    print("Exporting weights to NumPy format...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(script_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    
    # 處理 fc1 (EmbeddingBag)
    fc1_weight = model.fc1.weight.detach().cpu().numpy()[:45056].T  # 捨棄 padding_idx 並轉置為 (256, 45056)
    fc1_bias = model.fc1_bias.detach().cpu().numpy()
    np.save(os.path.join(weights_dir, 'fc1_weight.npy'), fc1_weight)
    np.save(os.path.join(weights_dir, 'fc1_bias.npy'), fc1_bias)
    print(f"Exported fc1 -> {fc1_weight.shape}")

    # 處理後端層
    layers = [(model.fc2, 2), (model.fc3, 3), (model.fc4, 4)]
    for layer, num in layers:
        weight = layer.weight.detach().cpu().numpy()
        bias = layer.bias.detach().cpu().numpy()
        np.save(os.path.join(weights_dir, f'fc{num}_weight.npy'), weight)
        np.save(os.path.join(weights_dir, f'fc{num}_bias.npy'), bias)
        print(f"Exported fc{num} -> {weight.shape}")
    print(f"All weights exported successfully to: {weights_dir}")

def train():
    if not TORCH_AVAILABLE:
        print("Cannot run training without PyTorch.")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- Dataset Path Logic ---
    dataset_path = TrainingConfig.DATASET_PATH
    if not os.path.exists(dataset_path):
        # 嘗試向上搜尋
        levels = ['../', '../../', '../../../']
        found = False
        for lv in levels:
            alt_path = os.path.join(lv, 'tuner/ultimate_halfka.npz')
            if os.path.exists(alt_path):
                dataset_path = alt_path
                found = True
                break
        if not found:
            print(f"Error: Dataset not found. Please check TrainingConfig.DATASET_PATH.")
            return

    print(f"Loading precomputed dataset into RAM from {dataset_path}...")
    # Load fully into RAM for maximum slicing speed!
    data = np.load(dataset_path)
    features_stm_full = data['features_stm']
    features_nstm_full = data['features_nstm']
    targets_full = data['targets']
    
    total_size = len(targets_full)
    val_size = int(TrainingConfig.VAL_SPLIT * total_size)
    train_size = total_size - val_size
    
    print(f"Dataset successfully mapped directly: {total_size} samples.")
    print(f"Train/Val split: {train_size} training samples, {val_size} validation samples.")
    
    # Static Validation Slices (Only slice pointers are copied)
    val_f_stm = features_stm_full[train_size:]
    val_f_nstm = features_nstm_full[train_size:]
    val_tgt = targets_full[train_size:]
    
    # Static Train Slices
    train_f_stm_cpu = features_stm_full[:train_size]
    train_f_nstm_cpu = features_nstm_full[:train_size]
    train_tgt_cpu = targets_full[:train_size]

    print(f"Uploading ENTIRE {train_size:,} training dataset to GPU VRAM...")
    # Using torch.int32 instead of torch.long saves 50% VRAM (4-bytes vs 8-bytes)
    # 18M samples * 32 indices * 4 bytes * 2 (stm/nstm) = ~4.6 GB. Fits in 6GB 4050!
    train_f_stm_cuda = torch.tensor(train_f_stm_cpu, dtype=torch.int32, device=device)
    train_f_nstm_cuda = torch.tensor(train_f_nstm_cpu, dtype=torch.int32, device=device)
    train_tgt_cuda = torch.tensor(train_tgt_cpu, dtype=torch.float32, device=device).unsqueeze(1)
    print("VRAM Upload Complete! Ready for extreme speed training.")
    
    model = SimpleNNUE().to(device)
    
    # --- Resume & Global Best Logic ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(script_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(weights_dir, 'latest_model.pth')
    best_model_path = os.path.join(weights_dir, 'best_model.pth')
    
    best_val_loss = float('inf')
    start_epoch = 0
    epochs_since_best = 0
    patience = TrainingConfig.EARLY_STOPPING_PATIENCE
    
    # --- 1. 永遠嘗試讀取歷史「最高分紀錄」，作為本次訓練的起點 ---
    if os.path.exists(best_model_path) and not TrainingConfig.RESUME_FROM_CHECKPOINT:
        try:
            best_checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(best_checkpoint['model_state_dict'])
            best_val_loss = best_checkpoint.get('best_val_loss', float('inf'))
            # best_val_loss = float('inf')
            print("="*60)
            print(f"🚀 成功讀取上次訓練好的權重 (Val Loss: {best_val_loss:.6f})！")
            print("🚀 本次訓練將以此權重為基礎，但重置 Epoch 與學習率。")
            print("="*60)
        except Exception as e:
            print(f"Skipping incompatible historic checkpoint: architecture changed. Resetting to scratch.")
            best_val_loss = float('inf')

    criterion = nn.BCEWithLogitsLoss()
    # 加入 weight_decay=1e-5 以抑制模型死背資料 (Overfitting)
    # Use Fused=True for RTX 4050 hardware acceleration
    optimizer = optim.Adam(model.parameters(), lr=TrainingConfig.LEARNING_RATE, 
                           weight_decay=1e-5, fused=True if device.type == 'cuda' else False)
    scheduler = StepLR(optimizer, step_size=TrainingConfig.SCHEDULER_STEP, gamma=TrainingConfig.SCHEDULER_GAMMA)
    
    # 使用最新的 torch.amp 語法避開 FutureWarning
    scaler = torch.amp.GradScaler('cuda') 


    # --- 2. 決定是否接續完整訓練 ---
    if TrainingConfig.RESUME_FROM_CHECKPOINT and os.path.exists(checkpoint_path):
        # 模式 A：完整接續（大腦 + 優化器 + LR + 輪次 全部還原）
        print(f"Full resume from {checkpoint_path}...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            start_epoch = checkpoint.get('epoch', -1) + 1
            print(f"Resumed from Epoch {start_epoch + 1} with Val Loss {best_val_loss:.6f}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}. Starting from scratch.")
            best_val_loss = float('inf')
    elif not os.path.exists(best_model_path):
        print("No checkpoint found. Starting from scratch.")

    total_epochs = TrainingConfig.TOTAL_EPOCHS
    print(f"Maximum allowed epochs: {total_epochs}")
    
    train_loss_history = []
    val_loss_history = []
    
    if PLOTTING_AVAILABLE:
        plt.ion()
        fig = plt.figure(figsize=(10, 6))

    for epoch_idx in range(start_epoch, total_epochs):
        start_time = time.time()
        
        # --- Training Phase ---
        model.train()
        total_train_loss = 0.0
        
        # In-memory slice-based shuffling (Ultra Fast)
        perm = np.random.permutation(train_size)
        num_train_batches = (train_size + TrainingConfig.BATCH_SIZE - 1) // TrainingConfig.BATCH_SIZE
        
        for batch_idx in range(num_train_batches):
            start = batch_idx * TrainingConfig.BATCH_SIZE
            end = min(start + TrainingConfig.BATCH_SIZE, train_size)
            batch_slice = perm[start:end]
            
            # Pure GPU Slicing. Zero CPU-GPU transfer overhead!
            # Casting to long inside GPU is extremely fast and required for Embedding layers
            inputs_stm = train_f_stm_cuda[batch_slice].long()
            inputs_nstm = train_f_nstm_cuda[batch_slice].long()
            targets = train_tgt_cuda[batch_slice]
            
            optimizer.zero_grad()
            
            # AMP Forward Pass (Modern Syntax)
            with torch.amp.autocast('cuda'):
                outputs = model(inputs_stm, inputs_nstm)
                loss = criterion(outputs, targets)
            
            # AMP Scaled Backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_train_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f" > Batch [{batch_idx}/{num_train_batches}] | Current Loss: {loss.item():.6f}")
            
        avg_train_loss = total_train_loss / num_train_batches
        
        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0.0
        num_val_batches = (val_size + TrainingConfig.BATCH_SIZE - 1) // TrainingConfig.BATCH_SIZE
        
        with torch.no_grad():
            for batch_idx in range(num_val_batches):
                start = batch_idx * TrainingConfig.BATCH_SIZE
                end = min(start + TrainingConfig.BATCH_SIZE, val_size)
                
                # Keep validation on CPU RAM and only move batch-by-batch to save VRAM for training
                inputs_stm = torch.tensor(val_f_stm[start:end], dtype=torch.long, device=device)
                inputs_nstm = torch.tensor(val_f_nstm[start:end], dtype=torch.long, device=device)
                targets = torch.tensor(val_tgt[start:end], dtype=torch.float32, device=device).unsqueeze(1)
                
                outputs = model(inputs_stm, inputs_nstm)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / num_val_batches
        
        # Step the scheduler
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)
        
        elapsed = time.time() - start_time
        print(f"Epoch [{epoch_idx+1}/{total_epochs}] | LR: {current_lr:.6f} | Time: {elapsed:.1f}s | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        # --- Checkpoint Saving (Using Binary Stream to bypass Unicode path bugs) ---
        checkpoint_data = {
            'epoch': epoch_idx,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': min(best_val_loss, avg_val_loss),
        }
        with open(checkpoint_path, 'wb') as f:
            torch.save(checkpoint_data, f)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_since_best = 0
            best_model_path = os.path.join(weights_dir, 'best_model.pth')
            with open(best_model_path, 'wb') as f:
                torch.save(checkpoint_data, f)
            print(f" NEW BEST! Saved to best_model.pth")
            export_weights(model)
        else:
            epochs_since_best += 1
            if epochs_since_best >= patience:
                print(f"Early stopping triggered! No improvement for {patience} epochs.")
                break

        if PLOTTING_AVAILABLE:
            plot_loss(len(train_loss_history)-1, train_loss_history, val_loss_history)
        
    print("Training finished.")
    if PLOTTING_AVAILABLE:
        plt.ioff()
        plt.show(block=False)
        plt.savefig(os.path.join(weights_dir, 'training_loss.png'))
        print(f"Loss chart saved to {os.path.join(weights_dir, 'training_loss.png')}")


if __name__ == '__main__':
    train()
