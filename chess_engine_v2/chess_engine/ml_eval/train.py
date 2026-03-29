import os
import numpy as np
import time
import numba

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
    HalfKA 優化版：儲存原始位元棋盤，在 `__getitem__` 即時利用 numba 解碼成稀疏表示 (Sparse Indices)。
    以節省大量記憶體並大幅加速模型訓練。
    """
    def __init__(self, npz_path):
        if not TORCH_AVAILABLE:
            return
            
        print(f"Loading dataset from {npz_path}...")
        data = np.load(npz_path)
        
        self.raw_bbs = data['piece_bbs']  # (N, 12) uint64
        self.raw_results = data['results'] # (N,) float32
        
        if 'game_states' in data:
            # game_states[0] 是 side_to_move (0=White, 1=Black)
            self.raw_stm = data['game_states'][:, 0].astype(np.int8)
        else:
            self.raw_stm = np.zeros(self.raw_results.shape[0], dtype=np.int8)

        # 建立鏡像映射地圖 (用於黑方視角)
        self.mirror_map = np.zeros(768, dtype=np.int32)
        for i in range(768):
            p_idx = i // 64
            sq = i % 64
            # 棋子換色 (p_idx + 6) % 12, 格子上下翻轉 (sq ^ 56)
            self.mirror_map[i] = ((p_idx + 6) % 12) * 64 + (sq ^ 56)

        print(f"Dataset indexed. Total samples: {len(self.raw_results)}")
        
    def __len__(self):
        return len(self.raw_results)

    def __getitem__(self, idx):
        bbs = self.raw_bbs[idx]
        stm = self.raw_stm[idx]
        
        # 1. 直接取得 HalfKA 視角的稀疏索引
        f_white = get_halfka_indices(bbs, False)
        f_black = get_halfka_indices(bbs, True)
        
        # 2. 視角映射
        if stm == 0:
            features_stm = f_white
            features_nstm = f_black
            result = self.raw_results[idx]
        else:
            features_stm = f_black
            features_nstm = f_white
            result = 1.0 - self.raw_results[idx] # 黑方勝 = 視角下的 1.0
            
        return torch.tensor(features_stm, dtype=torch.long), torch.tensor(features_nstm, dtype=torch.long), torch.tensor(result, dtype=torch.float32)

@numba.njit(numba.int32[:](numba.types.Array(numba.uint64, 1, 'C', readonly=False), numba.boolean), cache=True, fastmath=True)
def get_halfka_indices(bbs, is_black_perspective):
    indices = np.full(32, 45056, dtype=np.int32) # padding index = 45056
    idx_count = 0
    if is_black_perspective:
        king_bb = bbs[11]
        king_sq = 0
        if king_bb:
            lsb = king_bb & (~king_bb + np.uint64(1))
            tmp = lsb >> np.uint64(1)
            while tmp:
                king_sq += 1
                tmp >>= np.uint64(1)
        king_sq = king_sq ^ 56
        bucket = king_sq
        for p_idx in range(12):
            if p_idx == 11: continue # Skip Black King
            mapped_type = (p_idx + 5) if p_idx < 6 else (p_idx - 6)
            bb = bbs[p_idx]
            while bb:
                lsb = bb & (~bb + np.uint64(1))
                sq = 0
                tmp = lsb >> np.uint64(1)
                while tmp:
                    sq += 1
                    tmp >>= np.uint64(1)
                sq = sq ^ 56
                indices[idx_count] = bucket * 704 + mapped_type * 64 + sq
                idx_count += 1
                bb &= bb - np.uint64(1)
    else:
        king_bb = bbs[5]
        king_sq = 0
        if king_bb:
            lsb = king_bb & (~king_bb + np.uint64(1))
            tmp = lsb >> np.uint64(1)
            while tmp:
                king_sq += 1
                tmp >>= np.uint64(1)
        bucket = king_sq
        for p_idx in range(12):
            if p_idx == 5: continue # Skip White King
            mapped_type = p_idx if p_idx < 5 else (p_idx - 1)
            bb = bbs[p_idx]
            while bb:
                lsb = bb & (~bb + np.uint64(1))
                sq = 0
                tmp = lsb >> np.uint64(1)
                while tmp:
                    sq += 1
                    tmp >>= np.uint64(1)
                indices[idx_count] = bucket * 704 + mapped_type * 64 + sq
                idx_count += 1
                bb &= bb - np.uint64(1)
    return indices
                                      


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
    DATASET_PATH = './tuner/ultimate_dataset.npz'
    BATCH_SIZE = 16384 # 提升 Batch Size 減少任務分發開銷
    LEARNING_RATE = 0.0005
    
    # 學習率遞減設定 (StepLR)
    SCHEDULER_STEP = 10
    SCHEDULER_GAMMA = 0.5
    
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

    # 修正：Windows 下 num_workers 不宜過高，設為 4 以避免分頁檔不足的錯誤 (WinError 1455)
    num_workers = 4
    print(f"Assigning {num_workers} CPU workers for data pre-fetching (Safe Mode).")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=TrainingConfig.BATCH_SIZE, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=TrainingConfig.BATCH_SIZE, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
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
            print("="*60)
            print(f"🚀 成功讀取上次訓練好的權重 (Val Loss: {best_val_loss:.6f})！")
            print("🚀 本次訓練將以此權重為基礎，但重置 Epoch 與學習率。")
            print("="*60)
        except Exception as e:
            print(f"Skipping incompatible historic checkpoint: architecture changed. Resetting to scratch.")
            best_val_loss = float('inf')

    criterion = nn.BCEWithLogitsLoss()
    # 加入 weight_decay=1e-5 以抑制過擬合
    optimizer = optim.Adam(model.parameters(), lr=TrainingConfig.LEARNING_RATE, weight_decay=1e-4)
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
