import os
import numpy as np
import time

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, random_split
    from torch.optim.lr_scheduler import StepLR
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

class ChessDataset(Dataset):
    """
    A simple dataset to load bitboards and evaluate results.
    """
    def __init__(self, npz_path):
        if not TORCH_AVAILABLE:
            return
            
        print(f"Loading dataset from {npz_path}...")
        data = np.load(npz_path)
        
        raw_bbs = data['piece_bbs'] # Shape: (N, 12)
        raw_results = data['results'] # Shape: (N,)
        
        # Some older datasets might not have 'stm' saved. Assume White (0) to move if missing.
        if 'stm' in data:
            raw_stm = data['stm']
        else:
            raw_stm = np.zeros(raw_results.shape[0], dtype=np.int8)
        
        N = raw_bbs.shape[0]
        # 1. 向量化提取：先提取白方視角下的所有特徵 (N, 768)
        features_all = np.zeros((N, 768), dtype=np.float32)
        print(f"Extracting features from {N} bitboards using vectorized NumPy...")
        for piece_idx in range(12):
            bb = raw_bbs[:, piece_idx]
            for sq in range(64):
                is_set = (bb >> np.uint64(sq)) & np.uint64(1)
                features_all[:, piece_idx * 64 + sq] = is_set.astype(np.float32)
        
        # 2. 建立鏡像映射地圖 (Mirror indices map)
        mirror_map = np.zeros(768, dtype=np.int32)
        for i in range(768):
            p_idx = i // 64
            sq = i % 64
            mirror_map[i] = ((p_idx + 6) % 12) * 64 + (sq ^ 56)
        
        features_mirrored = features_all[:, mirror_map]

        # 3. 根據行棋方 (STM) 分配
        self.features_stm = np.zeros((N, 768), dtype=np.float32)
        self.features_nstm = np.zeros((N, 768), dtype=np.float32)
        
        white_mask = (raw_stm == 0)
        black_mask = (raw_stm == 1)
        
        # 當白方行棋：STM 視角 = 原始, NSTM 視角 = 鏡像
        self.features_stm[white_mask] = features_all[white_mask]
        self.features_nstm[white_mask] = features_mirrored[white_mask]
        
        # 當黑方行棋：STM 視角 = 鏡像, NSTM 視角 = 原始
        self.features_stm[black_mask] = features_mirrored[black_mask]
        self.features_nstm[black_mask] = features_all[black_mask]
        
        # 結果轉換為行棋方視角 (Label inversion if Black to move)
        self.results = np.where(raw_stm == 0, raw_results, 1.0 - raw_results).astype(np.float32)
        print(f"Dataset loaded. Total samples: {N}")
        
    def __len__(self):
        return len(self.results)
        
    def __getitem__(self, idx):
        return self.features_stm[idx], self.features_nstm[idx], self.results[idx]

if TORCH_AVAILABLE:
    class ClippedReLU(nn.Module):
        def __init__(self, upper_bound=127.0):
            super(ClippedReLU, self).__init__()
            self.upper_bound = upper_bound
            
        def forward(self, x):
            return torch.clamp(x, 0.0, self.upper_bound)

    class SimpleNNUE(nn.Module):
        """
        4-layer dual-accumulator network (HalfKP style):
        768 -> 256x2 (ClippedReLU) -> 512 -> 32 (ClippedReLU) -> 32 (ClippedReLU) -> 1
        """
        def __init__(self):
            super(SimpleNNUE, self).__init__()
            # FC1 acts as the accumulator (processes 768 features for one perspective)
            self.fc1 = nn.Linear(768, 256)
            # FC2 takes the concatenated accumulators (STM + NSTM)
            self.fc2 = nn.Linear(512, 32)
            self.fc3 = nn.Linear(32, 32)
            self.fc4 = nn.Linear(32, 1)
            self.crelu = ClippedReLU(127.0)
            
        def forward(self, x_stm, x_nstm):
            # Pass both perspectives through the same FC1 weights
            acc_stm = self.crelu(self.fc1(x_stm))
            acc_nstm = self.crelu(self.fc1(x_nstm))
            
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
    DATASET_PATH = './training_data/stockfish_t80_jan2024.npz'
    BATCH_SIZE = 4096
    LEARNING_RATE = 0.0001
    
    # 學習率遞減設定 (StepLR)
    SCHEDULER_STEP = 2
    SCHEDULER_GAMMA = 0.8
    
    # 訓練與停止邏輯
    TOTAL_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 20
    RESUME_FROM_CHECKPOINT = False # 是否從上次的 .pth 檔案接續訓練
    
    # 資料分割比例
    VAL_SPLIT = 0.1

def export_weights(model):
    """Export weights to NumPy format for Numba inference."""
    print("Exporting weights to NumPy format...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(script_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    
    layers = [model.fc1, model.fc2, model.fc3, model.fc4]
    for i, layer in enumerate(layers):
        layer_num = i + 1
        weight = layer.weight.detach().cpu().numpy()
        bias = layer.bias.detach().cpu().numpy()
        np.save(os.path.join(weights_dir, f'fc{layer_num}_weight.npy'), weight)
        np.save(os.path.join(weights_dir, f'fc{layer_num}_bias.npy'), bias)
        print(f"Exported fc{layer_num} -> {weight.shape}")
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
            alt_path = os.path.join(lv, 'tuner/ultimate_dataset.npz')
            if os.path.exists(alt_path):
                dataset_path = alt_path
                found = True
                break
        if not found:
            print(f"Error: Dataset not found. Please check TrainingConfig.DATASET_PATH.")
            return

    # Load dataset
    full_dataset = ChessDataset(dataset_path)
    
    # Train/Validation Split
    total_size = len(full_dataset)
    val_size = int(TrainingConfig.VAL_SPLIT * total_size)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Train/Val split: {train_size} training samples, {val_size} validation samples.")

    train_loader = DataLoader(train_dataset, batch_size=TrainingConfig.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=TrainingConfig.BATCH_SIZE, shuffle=False)
    
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
    
    # --- 1. 永遠嘗試讀取歷史「最高分紀錄」，防止被新訓練覆蓋掉 ---
    if os.path.exists(best_model_path) and not TrainingConfig.RESUME_FROM_CHECKPOINT:
        try:
            best_checkpoint = torch.load(best_model_path, map_location=device)
            # 先確認架構相符（舊的 256 維度權重不能套用在 512 維度上）
            model.load_state_dict(best_checkpoint['model_state_dict'])
            best_val_loss = best_checkpoint.get('best_val_loss', float('inf'))
            print(f"Loaded compatible historic best Val Loss: {best_val_loss:.6f} and best weights.")
        except Exception as e:
            print(f"Skipping incompatible historic checkpoint: architecture changed. Resetting to scratch.")
            best_val_loss = float('inf')

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=TrainingConfig.LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=TrainingConfig.SCHEDULER_STEP, gamma=TrainingConfig.SCHEDULER_GAMMA)

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
        
        for inputs_stm, inputs_nstm, targets in train_loader:
            inputs_stm = inputs_stm.to(device)
            inputs_nstm = inputs_nstm.to(device)
            targets = targets.unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs_stm, inputs_nstm)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for inputs_stm, inputs_nstm, targets in val_loader:
                inputs_stm = inputs_stm.to(device)
                inputs_nstm = inputs_nstm.to(device)
                targets = targets.unsqueeze(1).to(device)
                outputs = model(inputs_stm, inputs_nstm)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        
        # Step the scheduler
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)
        
        elapsed = time.time() - start_time
        print(f"Epoch [{epoch_idx+1}/{total_epochs}] | LR: {current_lr:.6f} | Time: {elapsed:.1f}s | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        # --- Checkpoint Saving ---
        checkpoint_data = {
            'epoch': epoch_idx,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': min(best_val_loss, avg_val_loss),
        }
        torch.save(checkpoint_data, checkpoint_path)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_since_best = 0
            torch.save(checkpoint_data, os.path.join(weights_dir, 'best_model.pth'))
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
