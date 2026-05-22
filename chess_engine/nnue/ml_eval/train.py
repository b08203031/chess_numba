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

    class LayerStackNNUE(nn.Module):
        """
        HalfKA + LayerStacks (SF18-style) with Structural Re-parameterization:
        45056 Sparse -> 512x2 (ClippedReLU) -> Concat(1024)
            -> [8 x CReLU((W_fact + ΔW_b) · x) -> ... -> scalar]
        At export time: W_export[b] = W_fact + ΔW_b  (algebraically exact fusion)
        Bucket selection: bucket = (piece_count - 1) // 4  [same as SF18]
        """
        def __init__(self, num_buckets=8):
            super(LayerStackNNUE, self).__init__()
            self.num_buckets = num_buckets

            # --- Shared FC1 (the big accumulator, stays unchanged) ---
            self.fc1 = nn.EmbeddingBag(45057, 512, mode="sum", padding_idx=45056)
            self.fc1_bias = nn.Parameter(torch.zeros(512))
            self.crelu = ClippedReLU(127.0)

            # --- Global Base Manifold (Factorizer): sees ALL data gradients ---
            self.fact_fc2 = nn.Linear(1024, 32)
            self.fact_fc3 = nn.Linear(32, 32)
            self.fact_fc4 = nn.Linear(32, 1)

            # --- Per-bucket Residuals (ΔW): zero-initialized, bucket-specific ---
            # Receives gradients only from its own bucket's samples
            self.delta_fc2 = nn.ModuleList([nn.Linear(1024, 32) for _ in range(num_buckets)])
            self.delta_fc3 = nn.ModuleList([nn.Linear(32, 32) for _ in range(num_buckets)])
            self.delta_fc4 = nn.ModuleList([nn.Linear(32, 1) for _ in range(num_buckets)])

            # CRITICAL: residuals must start at zero so Factorizer dominates early training
            for b in range(num_buckets):
                nn.init.zeros_(self.delta_fc2[b].weight)
                nn.init.zeros_(self.delta_fc2[b].bias)
                nn.init.zeros_(self.delta_fc3[b].weight)
                nn.init.zeros_(self.delta_fc3[b].bias)
                nn.init.zeros_(self.delta_fc4[b].weight)
                nn.init.zeros_(self.delta_fc4[b].bias)

        def forward(self, x_stm, x_nstm, bucket_ids):
            import torch.nn.functional as F
            # Shared accumulator (same for all buckets)
            acc_stm  = self.crelu(self.fc1(x_stm)  + self.fc1_bias)
            acc_nstm = self.crelu(self.fc1(x_nstm) + self.fc1_bias)
            x = torch.cat((acc_stm, acc_nstm), dim=1)  # (batch, 1024)

            # Route each sample to its bucket
            # Effective weight = Global Base (fact) + Bucket Residual (delta)
            # Fusion happens BEFORE nonlinearity → algebraically exact, inferrable
            output = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
            for b in range(self.num_buckets):
                mask = (bucket_ids == b)
                if mask.any():
                    xb = x[mask]

                    w2 = self.fact_fc2.weight + self.delta_fc2[b].weight
                    b2 = self.fact_fc2.bias   + self.delta_fc2[b].bias
                    xb = self.crelu(F.linear(xb, w2, b2))

                    w3 = self.fact_fc3.weight + self.delta_fc3[b].weight
                    b3 = self.fact_fc3.bias   + self.delta_fc3[b].bias
                    xb = self.crelu(F.linear(xb, w3, b3))

                    w4 = self.fact_fc4.weight + self.delta_fc4[b].weight
                    b4 = self.fact_fc4.bias   + self.delta_fc4[b].bias
                    output[mask] = F.linear(xb, w4, b4).to(output.dtype)

            return output

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
    NUM_BUCKETS = 8  # SF18-style LayerStacks
    BATCH_SIZE = 8192
    
    # --- 差異化學習率 (Layerwise LR) ---
    # 統一基準：由 0.001 開始，透過 Gamma 平滑冷卻至 200 Epochs
    LR_FC1 = 0.00004
    LR_DENSE = 0.00004
    #正常要從0.001開始，但因為我們有resume_from_checkpoint，所以可以從0.0008開始
    LR_GAMMA = 0.99  # 每個 Epoch 衰減 1%
    
    # --- 差異化 Weight Decay (防止資訊熱寂) ---
    # FC1: wd=0 (稀疏層禁止密集衰減，否則罕見特徵會被殺死)
    WD_FC1 = 0.0
    # Dense layers: 輕微懲罰，防止量化溢出
    WD_DENSE = 0.001
    
    # 訓練與停止邏輯
    TOTAL_EPOCHS = 200
    EARLY_STOPPING_PATIENCE = 20
    RESUME_FROM_CHECKPOINT = True  # False: 拋棄受污染的舊大腦，使用新資料集完全從零重生
    
    # 資料分割比例
    VAL_SPLIT = 0.1

# --- Quantization Constants ---
class QuantConfig:
    # Layer 1: Scale by 127 to align with ClippedReLU(127)
    Q1 = 127
    # Subsequent layers: Scale by 128 for high precision and ultra-efficient bit-shifts
    Q_HIDDEN = 128
    
    # [新增] FC1 的物理極限防爆閾值
    # 假設盤面最多約有 25~31 個啟動特徵，確保總和不會輕易超過 ClippedReLU 的 127.0
    FC1_MAX_WEIGHT = 127.0 / Q1 * 4.0

def export_weights(model):
    """Export LayerStackNNUE weights: FC1 shared + 8x FC2/FC3/FC4 buckets."""
    print(f"Exporting quantized weights (Q1={QuantConfig.Q1}, Q_HIDDEN={QuantConfig.Q_HIDDEN})...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(script_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    
    def q_round(x):
        return np.round(x).astype(np.int32)

    # 1. FC1 (Shared EmbeddingBag)
    # No .T — store as (45056, 512) so inference.py accesses FC1_WEIGHT[f, :]
    # This makes each feature index map to a contiguous 512-dim vector (cache-friendly)
    fc1_weight = model.fc1.weight.detach().cpu().numpy()[:45056]  # (45056, 512)
    fc1_bias = model.fc1_bias.detach().cpu().numpy()
    fc1_weight_int = np.ascontiguousarray(
        np.clip(q_round(fc1_weight * QuantConfig.Q1), -32768, 32767).astype(np.int16)
    )
    fc1_bias_int = q_round(fc1_bias * QuantConfig.Q1).astype(np.int32)
    with open(os.path.join(weights_dir, 'fc1_weight.npy'), 'wb') as f: np.save(f, fc1_weight_int)
    with open(os.path.join(weights_dir, 'fc1_bias.npy'),   'wb') as f: np.save(f, fc1_bias_int)
    print(f"Exported FC1 (INT16) -> {fc1_weight_int.shape}  (should be (45056, 512))")

    # 2. 8x LayerStack buckets: W_export[b] = W_fact + ΔW_b (algebraically exact)
    num_buckets = model.num_buckets

    for b in range(num_buckets):
        # FC2: Global base + Bucket residual (fusion is valid: addition before nonlinearity)
        w    = model.fact_fc2.weight.detach().cpu().numpy() + model.delta_fc2[b].weight.detach().cpu().numpy()
        bias = model.fact_fc2.bias.detach().cpu().numpy()   + model.delta_fc2[b].bias.detach().cpu().numpy()
        w_int    = np.clip(q_round(w * QuantConfig.Q_HIDDEN), -32768, 32767).astype(np.int16)
        bias_int = q_round(bias * QuantConfig.Q_HIDDEN * QuantConfig.Q1).astype(np.int32)
        with open(os.path.join(weights_dir, f'fc2_weight_b{b}.npy'), 'wb') as f: np.save(f, w_int)
        with open(os.path.join(weights_dir, f'fc2_bias_b{b}.npy'),   'wb') as f: np.save(f, bias_int)

        # FC3: Global base + Bucket residual
        w    = model.fact_fc3.weight.detach().cpu().numpy() + model.delta_fc3[b].weight.detach().cpu().numpy()
        bias = model.fact_fc3.bias.detach().cpu().numpy()   + model.delta_fc3[b].bias.detach().cpu().numpy()
        w_int    = np.clip(q_round(w * QuantConfig.Q_HIDDEN), -32768, 32767).astype(np.int16)
        bias_int = q_round(bias * QuantConfig.Q_HIDDEN * QuantConfig.Q1).astype(np.int32)
        with open(os.path.join(weights_dir, f'fc3_weight_b{b}.npy'), 'wb') as f: np.save(f, w_int)
        with open(os.path.join(weights_dir, f'fc3_bias_b{b}.npy'),   'wb') as f: np.save(f, bias_int)

        # FC4: Global base + Bucket residual
        w    = model.fact_fc4.weight.detach().cpu().numpy() + model.delta_fc4[b].weight.detach().cpu().numpy()
        bias = model.fact_fc4.bias.detach().cpu().numpy()   + model.delta_fc4[b].bias.detach().cpu().numpy()
        w_int    = q_round(w * QuantConfig.Q_HIDDEN).astype(np.int16)
        bias_int = q_round(bias * QuantConfig.Q_HIDDEN * QuantConfig.Q1).astype(np.int32)
        with open(os.path.join(weights_dir, f'fc4_weight_b{b}.npy'), 'wb') as f: np.save(f, w_int)
        with open(os.path.join(weights_dir, f'fc4_bias_b{b}.npy'),   'wb') as f: np.save(f, bias_int)

        print(f"  Exported bucket {b}: FC2{w_int.shape if b==0 else ''} FC3 FC4")

    print(f"All INT weights exported successfully to: {weights_dir}")

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
        levels = ['../', '../../../', '../../../']
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
    # piece_counts: SF18 bucket = (count - 1) // 4
    if 'piece_counts' in data:
        piece_counts_full = data['piece_counts'].astype(np.int8)
    else:
        print("WARNING: 'piece_counts' not found in dataset. Please re-run convert_dataset.py!")
        piece_counts_full = np.full(len(targets_full), 32, dtype=np.int8)  # fallback: all bucket 7
    
    total_size = len(targets_full)
    val_size = int(TrainingConfig.VAL_SPLIT * total_size)
    train_size = total_size - val_size
    
    print(f"Dataset successfully mapped directly: {total_size} samples.")
    
    # --- Global Shuffle with Fixed Seed ---
    # Ensures Validation set is representative while remaining "Fixed" across runs
    print("Performing global shuffle with fixed seed (42)...")
    indices = np.arange(total_size)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    features_stm_full = features_stm_full[indices]
    features_nstm_full = features_nstm_full[indices]
    targets_full = targets_full[indices]
    piece_counts_full = piece_counts_full[indices]
    
    print(f"Train/Val split: {train_size} training samples, {val_size} validation samples.")
    
    # Static Validation Slices (Only slice pointers are copied)
    val_f_stm = features_stm_full[train_size:]
    val_f_nstm = features_nstm_full[train_size:]
    val_tgt = targets_full[train_size:]
    val_pc = piece_counts_full[train_size:]
    
    # Static Train Slices
    train_f_stm_cpu = features_stm_full[:train_size]
    train_f_nstm_cpu = features_nstm_full[:train_size]
    train_tgt_cpu = targets_full[:train_size]
    train_pc_cpu = piece_counts_full[:train_size]
    
    model = LayerStackNNUE(num_buckets=TrainingConfig.NUM_BUCKETS).to(device)
    
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
    
    # --- 1. FC1 Transfer Learning: Load old FC1 weights if available ---
    # Architecture changed (SimpleNNUE -> LayerStackNNUE), so we do partial load:
    # only transfer fc1 weights (23M params) to give the new model a head start.
    if os.path.exists(best_model_path) and not TrainingConfig.RESUME_FROM_CHECKPOINT:
        try:
            old_checkpoint = torch.load(best_model_path, map_location=device)
            old_state = old_checkpoint['model_state_dict']
            new_state = model.state_dict()
            transferred = []
            for k in ['fc1.weight', 'fc1_bias']:
                if k in old_state and old_state[k].shape == new_state[k].shape:
                    new_state[k] = old_state[k]
                    transferred.append(k)
            model.load_state_dict(new_state)
            print("=" * 60)
            print(f"🚀 FC1 Transfer Learning: carried over {transferred}")
            print("🚀 FC2/FC3/FC4 LayerStacks initialised from scratch (8 buckets).")
            print("=" * 60)
        except Exception as e:
            print(f"No compatible FC1 to transfer ({e}). Starting fully from scratch.")

    criterion = nn.BCEWithLogitsLoss()
    
    # --- 分離式參數群組 (Parameter Groups) ---
    # 基於稀疏動力學理論：FC1 (稀疏輸入層) 與 Dense 層需要完全不同的正則化策略
    # FC1: wd=0 防止「資訊熱寂」(未被啟動的罕見特徵不會被衰減殺死)
    # Dense: wd=0.0001 輕微懲罰，控制權重幅度以適應 INT16 量化
    fc1_params = list(model.fc1.parameters()) + [model.fc1_bias]
    dense_params = (
        list(model.fact_fc2.parameters()) + list(model.fact_fc3.parameters()) + list(model.fact_fc4.parameters()) +
        list(model.delta_fc2.parameters()) + list(model.delta_fc3.parameters()) + list(model.delta_fc4.parameters())
    )
    
    optimizer = optim.AdamW([
        {'params': fc1_params,    'lr': TrainingConfig.LR_FC1,   'weight_decay': TrainingConfig.WD_FC1},
        {'params': dense_params,  'lr': TrainingConfig.LR_DENSE, 'weight_decay': TrainingConfig.WD_DENSE},
    ], fused=True if device.type == 'cuda' else False)
    
    print(f"Optimizer: FC1 LR={TrainingConfig.LR_FC1}, WD={TrainingConfig.WD_FC1} | "
          f"Dense LR={TrainingConfig.LR_DENSE}, WD={TrainingConfig.WD_DENSE}")
    
    # --- 指數衰減排程器 (ExponentialLR) ---
    # 官方基準：解耦驗證集，將排程器轉變為單純的時間觀測者
    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer, 
        gamma=TrainingConfig.LR_GAMMA
    )
    
    # 使用最新的 torch.amp 語法避開 FutureWarning
    scaler = torch.amp.GradScaler('cuda') 


    # --- 2. 決定是否接續完整訓練 ---
    if TrainingConfig.RESUME_FROM_CHECKPOINT and os.path.exists(best_model_path):
        # 模式 B：載入模型權重但使用全新優化器（適用於修改了優化器結構的情況）
        print(f"Loading model weights from {best_model_path} (fresh optimizer)...")
        try:
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            # 不還原 optimizer / scheduler — 因為參數群組結構已改變
            print(f"✅ Loaded best model (Val Loss {best_val_loss:.6f}). Optimizer reset with new param groups.")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}. Starting from scratch.")
            best_val_loss = float('inf')
    elif not os.path.exists(best_model_path):
        print("No checkpoint found. Starting from scratch.")

    total_epochs = TrainingConfig.TOTAL_EPOCHS
    print(f"Maximum allowed epochs: {total_epochs}")
    
    train_loss_history = []
    val_loss_history = []
    
    # --- 分數階測度平滑 (Fractional Power Smoothing / Square Root Weighting) ---
    # 權重 w_i = 1 / (p_i ** tao)。這在照顧少數樣本與尊重中局複雜度之間取得最佳平衡。
    tao = 0.25
    BUCKET_WEIGHTS = np.array([
        1.0 / (0.0102 ** tao),   # B0 (1-4 子): 9.9  -> 3.14 (大幅降溫)
        1.0 / (0.0411 ** tao),   # B1 (5-8 子): 4.9  -> 2.22
        1.0 / (0.0342 ** tao),   # B2 (9-12 子): 5.4 -> 2.32
        1.0 / (0.0878 ** tao),   # B3 (13-16 子): 3.4 -> 1.84
        1.0 / (0.1905 ** tao),   # B4 (17-20 子): 2.3 -> 1.51
        1.0 / (0.3010 ** tao),   # B5 (21-24 子): 1.8 -> 1.35
        1.0 / (0.2602 ** tao),   # B6 (25-28 子): 2.0 -> 1.40
        1.0 / (0.0749 ** tao)    # B7 (29-32 子): 3.7 -> 1.91
    ], dtype=np.float64)
    train_bucket_ids_np = np.clip((train_pc_cpu.astype(np.int32) - 1) // 4, 0, 7)
    sample_weights_np = BUCKET_WEIGHTS[train_bucket_ids_np].astype(np.float64)
    sample_weights_np /= sample_weights_np.sum()  # 正規化為機率
    # 使用 NumPy Generator (Alias Method) 沿快速採樣
    _rng = np.random.default_rng()
    print(f"✅ 加權採樣權重已預先計算，B0權重={BUCKET_WEIGHTS[0]:.1f}x, B5權重={BUCKET_WEIGHTS[5]:.1f}x")
    
    if PLOTTING_AVAILABLE:
        plt.ion()
        fig = plt.figure(figsize=(10, 6))

    for epoch_idx in range(start_epoch, total_epochs):
        start_time = time.time()
        
        # --- Training Phase ---
        model.train()
        total_train_loss = 0.0
        
        # 加權採樣 (使用 NumPy Generator Alias Method，速度跟 permutation 相同級)
        perm = _rng.choice(train_size, size=train_size, replace=True, p=sample_weights_np)
        num_train_batches = (train_size + TrainingConfig.BATCH_SIZE - 1) // TrainingConfig.BATCH_SIZE
        
        for batch_idx in range(num_train_batches):
            start = batch_idx * TrainingConfig.BATCH_SIZE
            end = min(start + TrainingConfig.BATCH_SIZE, train_size)
            batch_slice = perm[start:end]
            
            # Dynamic GPU Transfer: Only move the current batch to save VRAM
            inputs_stm = torch.tensor(train_f_stm_cpu[batch_slice], dtype=torch.long, device=device)
            inputs_nstm = torch.tensor(train_f_nstm_cpu[batch_slice], dtype=torch.long, device=device)
            targets = torch.tensor(train_tgt_cpu[batch_slice], dtype=torch.float32, device=device).unsqueeze(1).clamp(0.001, 0.999)  # Target Bounding: INT16 安全防線
            # SF18 bucket: (piece_count - 1) // 4, clamped to [0, 7]
            pc = train_pc_cpu[batch_slice].astype(np.int32)
            bucket_ids = torch.tensor(np.clip((pc - 1) // 4, 0, 7), dtype=torch.long, device=device)
            
            optimizer.zero_grad()
            
            # AMP Forward Pass (Modern Syntax)
            with torch.amp.autocast('cuda'):
                outputs = model(inputs_stm, inputs_nstm, bucket_ids)
                loss = criterion(outputs, targets)
            
            # AMP Scaled Backward
            scaler.scale(loss).backward()
            
            # 梯度的 Unscale
            scaler.unscale_(optimizer)
            
            # --- 新增：梯度裁剪 (Gradient Clipping) 防爆 ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20.0)
            
            # 權重更新
            scaler.step(optimizer)
            scaler.update()
            # [新增] 嚴謹動力學約束：物理極限防爆
            # 必須在 no_grad() 區塊內執行，避免破壞 Autograd 的計算圖
            with torch.no_grad():
                model.fc1.weight.clamp_(-QuantConfig.FC1_MAX_WEIGHT, QuantConfig.FC1_MAX_WEIGHT)

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
                
                # 高效資料搬運：使用 from_numpy + non_blocking=True
                inputs_stm  = torch.from_numpy(val_f_stm[start:end]).long().to(device, non_blocking=True)
                inputs_nstm = torch.from_numpy(val_f_nstm[start:end]).long().to(device, non_blocking=True)
                targets     = torch.from_numpy(val_tgt[start:end]).to(device, non_blocking=True).unsqueeze(1).clamp(0.001, 0.999)
                
                pc = val_pc[start:end].astype(np.int32)
                bucket_ids = torch.from_numpy(np.clip((pc - 1) // 4, 0, 7).astype(np.int64)).to(device, non_blocking=True)
                
                outputs = model(inputs_stm, inputs_nstm, bucket_ids)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / num_val_batches

        # --- 分桶診斷 (Bucket-level Val Loss Diagnostic) ---
        # 使用 reduction='none' 計算每個樣本的獨立 Loss，再用 scatter_add_ 歸併
        # 零記憶體增量：直接復用已在 RAM 中的 val_f_* 陣列
        _criterion_none = nn.BCEWithLogitsLoss(reduction='none')
        _bucket_sums   = torch.zeros(8, device=device)
        _bucket_counts = torch.zeros(8, device=device)
        with torch.no_grad():
            for _s in range(0, val_size, TrainingConfig.BATCH_SIZE):
                _e = min(_s + TrainingConfig.BATCH_SIZE, val_size)
                _stm  = torch.from_numpy(val_f_stm[_s:_e]).long().to(device, non_blocking=True)
                _nstm = torch.from_numpy(val_f_nstm[_s:_e]).long().to(device, non_blocking=True)
                _tgt  = torch.from_numpy(val_tgt[_s:_e]).to(device, non_blocking=True).unsqueeze(1).clamp(0.001, 0.999)
                _pc   = val_pc[_s:_e].astype(np.int32)
                _bid  = torch.from_numpy(np.clip((_pc - 1) // 4, 0, 7).astype(np.int64)).to(device, non_blocking=True)
                _out  = model(_stm, _nstm, _bid)
                _loss = _criterion_none(_out, _tgt).squeeze()
                _bucket_sums.scatter_add_(0, _bid, _loss)
                _bucket_counts.scatter_add_(0, _bid, torch.ones(_e - _s, device=device))
        _RANGES = ["1-4 ", "5-8 ", "9-12", "13-16", "17-20", "21-24", "25-28", "29-32"]
        print(f"  {'Bucket':<8} {'Pieces':<8} {'Samples':>10}  {'Val Loss':>10}")
        print(f"  {'-'*42}")
        for _b in range(8):
            if _bucket_counts[_b] > 0:
                _bl = (_bucket_sums[_b] / _bucket_counts[_b]).item()
                print(f"  B{_b:<7} {_RANGES[_b]:<8} {int(_bucket_counts[_b].item()):>10,}  {_bl:>10.6f}")
        del _criterion_none, _bucket_sums, _bucket_counts
        # -------------------------------------------------------

        # Step the scheduler (純粹時間函數，不依賴 avg_val_loss)
        lr_fc1 = optimizer.param_groups[0]['lr']
        lr_dense = optimizer.param_groups[1]['lr']
        scheduler.step()
        
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)
        
        elapsed = time.time() - start_time
        print(f"Epoch [{epoch_idx+1}/{total_epochs}] | LR_FC1: {lr_fc1:.6f} LR_Dense: {lr_dense:.6f} | Time: {elapsed:.1f}s | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
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

        # --- 權重動力學診斷 (Weight Dynamics Diagnostic) ---
        with torch.no_grad():
            _sd = model.state_dict()
            
            # 直接在 GPU 上進行張量運算，避免龐大矩陣的 PCIe 傳輸
            _fc1_w = _sd['fc1.weight']
            # 使用 .item() 僅將最終的純量數值拉回 CPU
            _fc1_sat_count = (_fc1_w.abs() >= (QuantConfig.FC1_MAX_WEIGHT - 1e-4)).sum().item()
            _fc1_sat = 100.0 * _fc1_sat_count / _fc1_w.numel()
            _fc1_max = _fc1_w.abs().max().item()
            
            print(f"\n{'='*60}")
            print(f"📊 NNUE 權重動力學診斷報告 (Epoch: {epoch_idx + 1})")
            print(f"{'='*60}")
            print(f"【FC1 稀疏層】飽和率: {_fc1_sat:.4f}% | Max: {_fc1_max:.4f}")
            
            for _lname in ['fc2', 'fc3', 'fc4']:
                # 移除未使用的 fact_*.weight 提取
                
                # 直接在 GPU 上計算每個 bucket 的 Delta MeanAbs
                _mabs = [
                    _sd[f'delta_{_lname}.{_b}.weight'].abs().mean().item()
                    for _b in range(model.num_buckets)
                ]
                print(f"【{_lname.upper()} 決策層】Δ MeanAbs 區間: [{min(_mabs):.5f} ~ {max(_mabs):.5f}]")
                
            print(f"{'='*60}\n")
            del _sd, _fc1_w
        # -------------------------------------------------------

    print("Training finished.")
    if PLOTTING_AVAILABLE:
        plt.ioff()
        plt.show(block=False)
        plt.savefig(os.path.join(weights_dir, 'training_loss.png'))
        print(f"Loss chart saved to {os.path.join(weights_dir, 'training_loss.png')}")


if __name__ == '__main__':
    train()
