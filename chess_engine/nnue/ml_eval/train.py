import os
import numpy as np
import time

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    # Using modern torch.amp for future compatibility
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch is not installed. You will need to install PyTorch to run this script: pip install torch")

try:
    import matplotlib
    matplotlib.use('Agg')  # Prevents GUI errors on headless/Colab environments
    import matplotlib.pyplot as plt
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
        HalfKAv2_hm + LayerStacks (SF18-style) with Structural Re-parameterization:
        22528 Sparse -> 512x2 (ClippedReLU) -> Concat(1024)
            -> [8 x CReLU((W_fact + ΔW_b) · x) -> ... -> scalar]
        At export time: W_export[b] = W_fact + ΔW_b  (algebraically exact fusion)
        Bucket selection: bucket = (piece_count - 1) // 4  [same as SF18]
        """
        def __init__(self, num_buckets=8, fc2_bias_init=0.50, fc3_bias_init=0.25):
            super(LayerStackNNUE, self).__init__()
            self.num_buckets = num_buckets

            # --- Shared FC1 (the big accumulator, stays unchanged) ---
            self.fc1 = nn.EmbeddingBag(22529, 512, mode="sum", padding_idx=22528)
            self.fc1_bias = nn.Parameter(torch.zeros(512))
            self.crelu = ClippedReLU(127.0)

            # --- Global Base Manifold (Factorizer): sees ALL data gradients ---
            self.fact_fc2 = nn.Linear(1024, 32)
            self.fact_fc3 = nn.Linear(32, 32)
            self.fact_fc4 = nn.Linear(32, 1)
            nn.init.constant_(self.fact_fc2.bias, fc2_bias_init)
            nn.init.constant_(self.fact_fc3.bias, fc3_bias_init)

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
    plt.ylabel('Loss (MSELoss on Sigmoid)')
    plt.legend()
    plt.grid(True)
    plt.draw()
    plt.pause(0.1)

# --- 訓練超參數集中區 (Training Hyperparameters) ---
class TrainingConfig:
    FEATURE_ENCODING_VERSION = 'halfkav2_hm_stockfish_official_v1'
    DATASET_PATH = './tuner/ultimate_halfka_farseerT75.npz'
    NUM_BUCKETS = 8  # SF18-style LayerStacks
    BATCH_SIZE = 16384
    SAMPLER_SEED = 42

    # --- Dense CReLU Bias Initialization ---
    # Positive initial biases keep FC2/FC3 neurons alive during early training.
    INIT_FC2_BIAS = 0.50
    INIT_FC3_BIAS = 0.25

    # --- Dense Activation Health Diagnostics ---
    # Samples inspected per epoch for FC2/FC3 pre-activation statistics.
    # Set to 0 to disable; keep capped to avoid doubling validation cost.
    PREACT_DIAG_SAMPLES = 20000
    PREACT_DIAG_EVERY = 1
    
    # --- 差異化學習率 (Layerwise LR) ---
    # Fresh training: Dense LR 高於 FC1，加速收斂
    LR_FC1_FRESH = 1.0e-4
    LR_DENSE_FRESH = 8.75e-4
    # Resume / Fine-tune: 統一低 LR 微調
    LR_FC1_RESUME = 0.000045
    LR_DENSE_RESUME = 0.000395
    LR_GAMMA = 0.992  # 每個 Epoch 衰減 0.8%
    
    # --- 差異化 Weight Decay (防止資訊熱寂) ---
    # FC1: wd=0 (稀疏層禁止密集衰減，否則罕見特徵會被殺死)
    WD_FC1 = 0.0
    # Dense layers: wd=1.0e-4 輕微懲罰，防止量化溢出
    WD_DENSE = 1.0e-4
    
    # 損失函數指數 (2.0=標準 MSE，2.5/2.6=官方最優指數)
    LOSS_EXPONENT = 2.0
    
    # 訓練與停止邏輯
    TOTAL_EPOCHS = 200
    EARLY_STOPPING_PATIENCE = 10
    RESUME_FROM_CHECKPOINT = True  # False: 使用新資料集完全從零重生
    
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

def fen_to_halfka_indices(fen):
    parts = fen.split(' ')
    board_part = parts[0]
    stm_part = parts[1]
    
    pieces = {}
    ranks = board_part.split('/')
    for r_idx, rank in enumerate(ranks):
        rank_val = 7 - r_idx
        file_val = 0
        for char in rank:
            if char.isdigit():
                file_val += int(char)
            else:
                sq = rank_val * 8 + file_val
                pieces[sq] = char
                file_val += 1
                
    piece_count = len(pieces)
    stm = 0 if stm_part == 'w' else 1
    
    wk_sq = next(sq for sq, char in pieces.items() if char == 'K')
    bk_sq = next(sq for sq, char in pieces.items() if char == 'k')
    
    def get_indices(active_side):
        if active_side == 0:
            k_sq = wk_sq
            opp_k_sq = bk_sq
        else:
            k_sq = bk_sq
            opp_k_sq = wk_sq
            
        if active_side == 1:
            k_sq_sym = k_sq ^ 56
        else:
            k_sq_sym = k_sq
            
        flip_h = (k_sq_sym & 7) < 4
        if flip_h:
            king_mapped = k_sq_sym ^ 7
        else:
            king_mapped = k_sq_sym
            
        bucket = (7 - (king_mapped >> 3)) * 4 + (7 - (king_mapped & 7))
        
        indices = []
        # Own king
        sq_own_mapped = k_sq ^ (56 if active_side == 1 else 0)
        if flip_h: sq_own_mapped ^= 7
        indices.append(bucket * 704 + 10 * 64 + sq_own_mapped)
        
        # Opp king
        sq_opp_mapped = opp_k_sq ^ (56 if active_side == 1 else 0)
        if flip_h: sq_opp_mapped ^= 7
        indices.append(bucket * 704 + 10 * 64 + sq_opp_mapped)
        
        PIECE_CHARS = ['P', 'N', 'B', 'R', 'Q'] if active_side == 0 else ['p', 'n', 'b', 'r', 'q']
        OPP_CHARS = ['p', 'n', 'b', 'r', 'q'] if active_side == 0 else ['P', 'N', 'B', 'R', 'Q']
        
        for sq, char in pieces.items():
            if char in ['K', 'k']:
                continue
            if char in PIECE_CHARS:
                channel = PIECE_CHARS.index(char)
                mt = channel * 2
            elif char in OPP_CHARS:
                channel = OPP_CHARS.index(char)
                mt = channel * 2 + 1
            else:
                continue
                
            sq_mapped = sq ^ (56 if active_side == 1 else 0)
            if flip_h: sq_mapped ^= 7
            indices.append(bucket * 704 + mt * 64 + sq_mapped)
            
        while len(indices) < 32:
            indices.append(22528)
        return indices[:32]
        
    if stm == 0:
        stm_indices = get_indices(0)
        nstm_indices = get_indices(1)
    else:
        stm_indices = get_indices(1)
        nstm_indices = get_indices(0)
        
    return stm_indices, nstm_indices, piece_count

def validate_typical_positions(model, device):
    fens = {
        "Initial Position": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "White Queen Advantage": "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "Double King Endgame": "8/8/8/4k3/8/8/3K4/8 w - - 0 1"
    }
    print("\n--- Validation Positions Diagnostic ---")
    model.eval()
    with torch.no_grad():
        for name, fen in fens.items():
            stm_idx, nstm_idx, pc = fen_to_halfka_indices(fen)
            
            inputs_stm = torch.tensor([stm_idx], dtype=torch.long, device=device)
            inputs_nstm = torch.tensor([nstm_idx], dtype=torch.long, device=device)
            bucket_id = max(0, min(7, (pc - 1) // 4))
            bucket_ids = torch.tensor([bucket_id], dtype=torch.long, device=device)
            
            logit = model(inputs_stm, inputs_nstm, bucket_ids).item()
            float_cp = logit / 0.0025
            
            int_output = logit * 16256
            if int_output >= 0:
                int_cp = int(int_output) // 41
            else:
                int_cp = -int(-int_output) // 41
                
            print(f"  {name:<25} | FEN: {fen} | Piece Count: {pc} | Float CP: {float_cp:+.1f} | Quant CP: {int_cp:+d}")
            
            # Print layer activity diagnostics for the initial position
            if name == "Initial Position":
                import torch.nn.functional as F
                # Run manual forward to inspect intermediates
                acc_stm = model.crelu(model.fc1(inputs_stm) + model.fc1_bias)
                acc_nstm = model.crelu(model.fc1(inputs_nstm) + model.fc1_bias)
                x = torch.cat((acc_stm, acc_nstm), dim=1)
                
                w2 = model.fact_fc2.weight + model.delta_fc2[bucket_id].weight
                b2 = model.fact_fc2.bias + model.delta_fc2[bucket_id].bias
                h2 = model.crelu(F.linear(x, w2, b2))
                
                w3 = model.fact_fc3.weight + model.delta_fc3[bucket_id].weight
                b3 = model.fact_fc3.bias + model.delta_fc3[bucket_id].bias
                h3 = model.crelu(F.linear(h2, w3, b3))
                
                fc1_alive = (acc_stm > 0).sum().item()
                fc2_alive = (h2 > 0).sum().item()
                fc3_alive = (h3 > 0).sum().item()
                print(f"    [Neuron Activity] FC1 Active: {fc1_alive}/512 | FC2 Active: {fc2_alive}/32 | FC3 Active: {fc3_alive}/32")
    print("-" * 40)

def diagnose_dense_pre_activations(model, device, val_f_stm, val_f_nstm, val_pc, sample_limit):
    """Report FC2/FC3 pre-activation health on a capped validation slice."""
    if sample_limit <= 0:
        return

    import torch.nn.functional as F

    model.eval()
    sample_count = min(sample_limit, len(val_pc))
    if sample_count <= 0:
        return

    num_buckets = model.num_buckets
    num_layers = 2
    hidden_width = 32
    upper_bound = float(model.crelu.upper_bound)
    batch_size = TrainingConfig.BATCH_SIZE

    layer_samples = torch.zeros(num_buckets, dtype=torch.float64, device=device)
    layer_elems = torch.zeros((num_layers, num_buckets), dtype=torch.float64, device=device)
    layer_pre_le0 = torch.zeros((num_layers, num_buckets), dtype=torch.float64, device=device)
    layer_pre_hi = torch.zeros((num_layers, num_buckets), dtype=torch.float64, device=device)
    layer_active = torch.zeros((num_layers, num_buckets), dtype=torch.float64, device=device)
    layer_sums = torch.zeros((num_layers, num_buckets), dtype=torch.float64, device=device)
    layer_sqs = torch.zeros((num_layers, num_buckets), dtype=torch.float64, device=device)
    layer_mins = torch.full((num_layers, num_buckets), float('inf'), device=device)
    layer_maxs = torch.full((num_layers, num_buckets), float('-inf'), device=device)
    layer_ever = torch.zeros((num_layers, num_buckets, hidden_width), dtype=torch.bool, device=device)

    def update_stats(layer_idx, bucket, pre, act):
        pre_f = pre.float()
        active = act > 0.0
        elems = pre_f.numel()

        layer_elems[layer_idx, bucket] += elems
        layer_pre_le0[layer_idx, bucket] += (pre_f <= 0.0).sum()
        layer_pre_hi[layer_idx, bucket] += (pre_f >= upper_bound).sum()
        layer_active[layer_idx, bucket] += active.sum()
        layer_sums[layer_idx, bucket] += pre_f.double().sum()
        layer_sqs[layer_idx, bucket] += pre_f.double().square().sum()
        layer_mins[layer_idx, bucket] = torch.minimum(layer_mins[layer_idx, bucket], pre_f.min())
        layer_maxs[layer_idx, bucket] = torch.maximum(layer_maxs[layer_idx, bucket], pre_f.max())
        layer_ever[layer_idx, bucket] = layer_ever[layer_idx, bucket] | active.any(dim=0)

    with torch.no_grad():
        for start in range(0, sample_count, batch_size):
            end = min(start + batch_size, sample_count)
            inputs_stm = torch.from_numpy(val_f_stm[start:end]).long().to(device, non_blocking=True)
            inputs_nstm = torch.from_numpy(val_f_nstm[start:end]).long().to(device, non_blocking=True)
            pc = val_pc[start:end].astype(np.int32)
            bucket_ids = torch.from_numpy(np.clip((pc - 1) // 4, 0, 7).astype(np.int64)).to(device, non_blocking=True)

            acc_stm = model.crelu(model.fc1(inputs_stm) + model.fc1_bias)
            acc_nstm = model.crelu(model.fc1(inputs_nstm) + model.fc1_bias)
            x = torch.cat((acc_stm, acc_nstm), dim=1)

            for bucket in range(num_buckets):
                mask = bucket_ids == bucket
                bucket_sample_count = int(mask.sum().item())
                if bucket_sample_count == 0:
                    continue

                xb = x[mask]
                layer_samples[bucket] += bucket_sample_count

                w2 = model.fact_fc2.weight + model.delta_fc2[bucket].weight
                b2 = model.fact_fc2.bias + model.delta_fc2[bucket].bias
                pre2 = F.linear(xb, w2, b2)
                h2 = model.crelu(pre2)
                update_stats(0, bucket, pre2, h2)

                w3 = model.fact_fc3.weight + model.delta_fc3[bucket].weight
                b3 = model.fact_fc3.bias + model.delta_fc3[bucket].bias
                pre3 = F.linear(h2, w3, b3)
                h3 = model.crelu(pre3)
                update_stats(1, bucket, pre3, h3)

    print(f"\n--- Dense Pre-Activation Diagnostic ({sample_count:,} validation samples) ---")
    print(f"  {'Bucket':<6} {'Layer':<4} {'Samples':>8} {'<=0%':>8} {'>=127%':>8} {'Mean':>9} {'Std':>9} {'Min':>9} {'Max':>9} {'Ever':>7} {'Act/Smp':>8}")
    for bucket in range(num_buckets):
        samples = int(layer_samples[bucket].item())
        if samples == 0:
            continue

        for layer_idx, layer_name in enumerate(('FC2', 'FC3')):
            elems = layer_elems[layer_idx, bucket].item()
            if elems == 0:
                continue
            mean = (layer_sums[layer_idx, bucket] / layer_elems[layer_idx, bucket]).item()
            variance = (layer_sqs[layer_idx, bucket] / layer_elems[layer_idx, bucket]).item() - mean * mean
            std = max(variance, 0.0) ** 0.5
            dead_or_zero = 100.0 * layer_pre_le0[layer_idx, bucket].item() / elems
            saturated = 100.0 * layer_pre_hi[layer_idx, bucket].item() / elems
            ever_active = int(layer_ever[layer_idx, bucket].sum().item())
            active_per_sample = layer_active[layer_idx, bucket].item() / samples
            min_val = layer_mins[layer_idx, bucket].item()
            max_val = layer_maxs[layer_idx, bucket].item()
            print(
                f"  B{bucket:<5} {layer_name:<4} {samples:>8,} "
                f"{dead_or_zero:>7.2f}% {saturated:>7.2f}% "
                f"{mean:>9.3f} {std:>9.3f} {min_val:>9.3f} {max_val:>9.3f} "
                f"{ever_active:>2}/{hidden_width:<2} {active_per_sample:>8.2f}"
            )
    print("-" * 72)

def export_weights(model):
    """Export LayerStackNNUE weights: FC1 shared + 8x FC2/FC3/FC4 buckets."""
    print(f"Exporting quantized weights (Q1={QuantConfig.Q1}, Q_HIDDEN={QuantConfig.Q_HIDDEN})...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(script_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    
    def q_round(x):
        return np.round(x).astype(np.int32)

    # 1. FC1 (Shared EmbeddingBag) — HalfKAv2_hm: 22528 features
    # No .T — store as (22528, 512) so inference.py accesses FC1_WEIGHT[f, :]
    NUM_FEATURES = 22528
    fc1_weight = model.fc1.weight.detach().cpu().numpy()[:NUM_FEATURES]  # (22528, 512)
    fc1_bias = model.fc1_bias.detach().cpu().numpy()
    fc1_weight_int = np.ascontiguousarray(
        np.clip(q_round(fc1_weight * QuantConfig.Q1), -32768, 32767).astype(np.int16)
    )
    fc1_bias_int = q_round(fc1_bias * QuantConfig.Q1).astype(np.int32)
    assert fc1_weight_int.shape == (NUM_FEATURES, 512), f"FC1 shape mismatch: {fc1_weight_int.shape}"
    with open(os.path.join(weights_dir, 'fc1_weight.npy'), 'wb') as f: np.save(f, fc1_weight_int)
    with open(os.path.join(weights_dir, 'fc1_bias.npy'),   'wb') as f: np.save(f, fc1_bias_int)
    # Saturation diagnostics for FC1
    fc1_sat = 100.0 * np.sum(np.abs(fc1_weight_int) >= 32767) / fc1_weight_int.size
    print(f"Exported FC1 (INT16) -> {fc1_weight_int.shape}  (sat: {fc1_sat:.4f}%)")

    # 2. 8x LayerStack buckets: W_export[b] = W_fact + ΔW_b (algebraically exact)
    num_buckets = model.num_buckets

    for b in range(num_buckets):
        # FC2: Global base + Bucket residual (fusion is valid: addition before nonlinearity)
        w    = model.fact_fc2.weight.detach().cpu().numpy() + model.delta_fc2[b].weight.detach().cpu().numpy()
        bias = model.fact_fc2.bias.detach().cpu().numpy()   + model.delta_fc2[b].bias.detach().cpu().numpy()
        w_int    = np.clip(q_round(w * QuantConfig.Q_HIDDEN), -32768, 32767).astype(np.int16)
        fc2_shape = w_int.shape
        bias_int = q_round(bias * QuantConfig.Q_HIDDEN * QuantConfig.Q1).astype(np.int32)
        with open(os.path.join(weights_dir, f'fc2_weight_b{b}.npy'), 'wb') as f: np.save(f, w_int)
        with open(os.path.join(weights_dir, f'fc2_bias_b{b}.npy'),   'wb') as f: np.save(f, bias_int)

        # FC3: Global base + Bucket residual
        w    = model.fact_fc3.weight.detach().cpu().numpy() + model.delta_fc3[b].weight.detach().cpu().numpy()
        bias = model.fact_fc3.bias.detach().cpu().numpy()   + model.delta_fc3[b].bias.detach().cpu().numpy()
        w_int    = np.clip(q_round(w * QuantConfig.Q_HIDDEN), -32768, 32767).astype(np.int16)
        fc3_shape = w_int.shape
        bias_int = q_round(bias * QuantConfig.Q_HIDDEN * QuantConfig.Q1).astype(np.int32)
        with open(os.path.join(weights_dir, f'fc3_weight_b{b}.npy'), 'wb') as f: np.save(f, w_int)
        with open(os.path.join(weights_dir, f'fc3_bias_b{b}.npy'),   'wb') as f: np.save(f, bias_int)

        # FC4: Global base + Bucket residual — with clipping to prevent int16 wrap
        w    = model.fact_fc4.weight.detach().cpu().numpy() + model.delta_fc4[b].weight.detach().cpu().numpy()
        bias = model.fact_fc4.bias.detach().cpu().numpy()   + model.delta_fc4[b].bias.detach().cpu().numpy()
        w_int    = np.clip(q_round(w * QuantConfig.Q_HIDDEN), -32768, 32767).astype(np.int16)
        fc4_shape = w_int.shape
        bias_int = q_round(bias * QuantConfig.Q_HIDDEN * QuantConfig.Q1).astype(np.int32)
        with open(os.path.join(weights_dir, f'fc4_weight_b{b}.npy'), 'wb') as f: np.save(f, w_int)
        with open(os.path.join(weights_dir, f'fc4_bias_b{b}.npy'),   'wb') as f: np.save(f, bias_int)

        if b == 0:
            print(f"  Exported bucket {b}: FC2{fc2_shape} FC3{fc3_shape} FC4{fc4_shape}")
        else:
            print(f"  Exported bucket {b}: FC2 FC3 FC4")

    import hashlib
    import json
    def get_sha256(filepath):
        h = hashlib.sha256()
        with open(filepath, 'rb') as f:
            h.update(f.read())
        return h.hexdigest()

    meta = {
        "feature_encoding_version": TrainingConfig.FEATURE_ENCODING_VERSION,
        "q1": int(QuantConfig.Q1),
        "q_hidden": int(QuantConfig.Q_HIDDEN),
        "int_activation_max": 16129,
        "output_divisor": 41,
        "target_k": 0.0025,
        "pawn_value_eg": 208.0,
        "layers": {
            "fc1": [22528, 512],
            "fc2": [32, 1024],
            "fc3": [32, 32],
            "fc4": [1, 32]
        },
        "weights_sha256": {}
    }
    
    filenames = ['fc1_weight.npy', 'fc1_bias.npy']
    for b in range(num_buckets):
        filenames.extend([f'fc2_weight_b{b}.npy', f'fc2_bias_b{b}.npy',
                          f'fc3_weight_b{b}.npy', f'fc3_bias_b{b}.npy',
                          f'fc4_weight_b{b}.npy', f'fc4_bias_b{b}.npy'])
                          
    for filename in filenames:
        meta["weights_sha256"][filename] = get_sha256(os.path.join(weights_dir, filename))
        
    with open(os.path.join(weights_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=4)

    print(f"All INT weights and metadata.json exported successfully to: {weights_dir}")

def validate_dataset_schema(data, dataset_path="dataset"):
    # Dataset Schema Validator
    required_keys = ['features_stm', 'features_nstm', 'targets', 'piece_counts']
    for k in required_keys:
        if k not in data:
            raise ValueError(f"Dataset schema error: missing required key '{k}' in {dataset_path}")
            
    # Shape consistency verification
    n_samples = len(data['targets'])
    if len(data['features_stm']) != n_samples or len(data['features_nstm']) != n_samples or len(data['piece_counts']) != n_samples:
        raise ValueError(
            f"Dataset array shapes are inconsistent! "
            f"targets: {n_samples}, "
            f"features_stm: {len(data['features_stm'])}, "
            f"features_nstm: {len(data['features_nstm'])}, "
            f"piece_counts: {len(data['piece_counts'])}"
        )
        
    # Value boundary constraints verification
    p_min = data['piece_counts'].min()
    p_max = data['piece_counts'].max()
    if p_min < 2 or p_max > 32:
        raise ValueError(f"Dataset piece_counts boundaries violated: expected [2, 32], got range [{p_min}, {p_max}]")
        
    f_stm_min = data['features_stm'].min()
    f_stm_max = data['features_stm'].max()
    f_nstm_min = data['features_nstm'].min()
    f_nstm_max = data['features_nstm'].max()
    if f_stm_min < 0 or f_stm_max > 22528 or f_nstm_min < 0 or f_nstm_max > 22528:
        raise ValueError(
            f"Dataset features boundaries violated: expected [0, 22528], "
            f"got features_stm range [{f_stm_min}, {f_stm_max}], "
            f"features_nstm range [{f_nstm_min}, {f_nstm_max}]"
        )
        
    # Verify metadata if present
    if 'metadata_json' in data:
        try:
            import json
            meta = json.loads(str(data['metadata_json']))
            print(f"Dataset metadata verification successful: {meta}")
            if meta.get("feature_encoding_version") != TrainingConfig.FEATURE_ENCODING_VERSION:
                print(f"WARNING: Dataset feature_encoding_version ({meta.get('feature_encoding_version')}) "
                      f"does not match TrainingConfig.FEATURE_ENCODING_VERSION ({TrainingConfig.FEATURE_ENCODING_VERSION})")
        except Exception as e:
            print(f"WARNING: Failed to parse dataset metadata: {e}")
    else:
        print("WARNING: Dataset does not contain metadata_json. Assuming legacy format.")

def train():
    if not TORCH_AVAILABLE:
        print("Cannot run training without PyTorch.")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # --- Dataset Path Logic ---
    dataset_path = TrainingConfig.DATASET_PATH
    if not os.path.exists(dataset_path):
        # 嘗試向上搜尋
        levels = ['../', '../../../', '../../../']
        found = False
        for lv in levels:
            alt_path = os.path.join(lv, 'tuner/ultimate_halfka_farseerT75.npz')
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
    
    validate_dataset_schema(data, dataset_path)
        
    features_stm_full = data['features_stm']
    features_nstm_full = data['features_nstm']
    targets_full = data['targets']
    piece_counts_full = data['piece_counts'].astype(np.int8)
    
    total_size = len(targets_full)
    val_size = int(TrainingConfig.VAL_SPLIT * total_size)
    train_size = total_size - val_size
    
    print(f"Dataset successfully mapped directly: {total_size} samples.")
    
    # --- Global Shuffle with Fixed Seed (Index-Only to save RAM) ---
    # Only shuffle index array, avoiding full-array copy that would double RAM usage.
    print("Performing index-only global shuffle with fixed seed (42)...")
    shuffle_indices = np.arange(total_size)
    np.random.seed(42)
    np.random.shuffle(shuffle_indices)
    
    train_indices = shuffle_indices[:train_size]
    val_indices = shuffle_indices[train_size:]
    
    print(f"Train/Val split: {train_size} training samples, {val_size} validation samples.")
    
    # Validation: copy once (small fraction of data)
    val_f_stm = features_stm_full[val_indices]
    val_f_nstm = features_nstm_full[val_indices]
    val_tgt = targets_full[val_indices]
    val_pc = piece_counts_full[val_indices]
    
    # Training: keep indices for double-indexing in batch loop (no copy)
    train_pc_cpu = piece_counts_full[train_indices]  # small: N x 1 byte
    
    model = LayerStackNNUE(
        num_buckets=TrainingConfig.NUM_BUCKETS,
        fc2_bias_init=TrainingConfig.INIT_FC2_BIAS,
        fc3_bias_init=TrainingConfig.INIT_FC3_BIAS,
    ).to(device)
    
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
            old_version = old_checkpoint.get('feature_encoding_version')
            if old_version != TrainingConfig.FEATURE_ENCODING_VERSION:
                raise ValueError(
                    f"feature encoding mismatch: checkpoint={old_version}, "
                    f"current={TrainingConfig.FEATURE_ENCODING_VERSION}"
                )
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

    criterion = nn.MSELoss()
    
    # --- 分離式參數群組 (Parameter Groups) ---
    # FC1: wd=0 防止「資訊熱寂」(罕見特徵不會被衰減殺死)
    # Dense: wd=0.001 輕微懲罰，控制權重幅度以適應 INT16 量化
    fc1_params = list(model.fc1.parameters()) + [model.fc1_bias]
    dense_params = (
        list(model.fact_fc2.parameters()) + list(model.fact_fc3.parameters()) + list(model.fact_fc4.parameters()) +
        list(model.delta_fc2.parameters()) + list(model.delta_fc3.parameters()) + list(model.delta_fc4.parameters())
    )
    
    # Dynamic LR selection based on training mode
    lr_fc1 = TrainingConfig.LR_FC1_RESUME if TrainingConfig.RESUME_FROM_CHECKPOINT else TrainingConfig.LR_FC1_FRESH
    lr_dense = TrainingConfig.LR_DENSE_RESUME if TrainingConfig.RESUME_FROM_CHECKPOINT else TrainingConfig.LR_DENSE_FRESH
    
    optimizer = optim.AdamW([
        {'params': fc1_params,    'lr': lr_fc1,   'weight_decay': TrainingConfig.WD_FC1},
        {'params': dense_params,  'lr': lr_dense, 'weight_decay': TrainingConfig.WD_DENSE},
    ], fused=True if device.type == 'cuda' else False)
    
    print(f"Optimizer: FC1 LR={lr_fc1}, WD={TrainingConfig.WD_FC1} | "
          f"Dense LR={lr_dense}, WD={TrainingConfig.WD_DENSE}")
    
    # --- 指數衰減排程器 (ExponentialLR) ---
    # 官方基準：解耦驗證集，將排程器轉變為單純的時間觀測者
    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer, 
        gamma=TrainingConfig.LR_GAMMA
    )
    
    # AMP safety: disable scaler on CPU to prevent warnings/errors
    amp_enabled = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)


    # --- 2. 決定是否接續完整訓練 ---
    if TrainingConfig.RESUME_FROM_CHECKPOINT and os.path.exists(best_model_path):
        # 模式 B：載入模型權重但使用全新優化器（適用於修改了優化器結構的情況）
        print(f"Loading model weights from {best_model_path} (fresh optimizer)...")
        try:
            checkpoint = torch.load(best_model_path, map_location=device)
            checkpoint_version = checkpoint.get('feature_encoding_version')
            if checkpoint_version != TrainingConfig.FEATURE_ENCODING_VERSION:
                raise ValueError(
                    f"feature encoding mismatch: checkpoint={checkpoint_version}, "
                    f"current={TrainingConfig.FEATURE_ENCODING_VERSION}"
                )
            model.load_state_dict(checkpoint['model_state_dict'])
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            
            # 檢查並重置不相容的損失函數尺度的 best_val_loss
            loaded_loss_type = checkpoint.get('loss_type', 'BCE')
            if loaded_loss_type != 'MSE':
                print(f"⚠️ 檢測到不同的損失函數格式 (從 {loaded_loss_type} 切換至 MSE)。重置 best_val_loss 為 inf。")
                best_val_loss = float('inf')
                
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
    
    # --- Empirical Bucket Sampling (data-driven, not hardcoded) ---
    # 權重 w_i = 1 / (p_i ** tao)。從資料集自動計算 empirical probability。
    tao = 0.25
    train_bucket_ids_np = np.clip((train_pc_cpu.astype(np.int32) - 1) // 4, 0, 7)
    bucket_counts = np.bincount(train_bucket_ids_np, minlength=8).astype(np.float64)
    bucket_probs = bucket_counts / bucket_counts.sum()
    
    # 計算每個分桶的加權權重
    bucket_weights = 1.0 / np.maximum(bucket_probs, 1e-12) ** tao
    # 正規化各分桶被抽樣的機率
    normalized_bucket_probs = (bucket_weights * bucket_probs)
    normalized_bucket_probs /= normalized_bucket_probs.sum()
    
    # 預先分組分桶索引，避免在大數組上建立 27M 的自定義機率 Alias Table
    bucket_indices = [np.where(train_bucket_ids_np == b)[0] for b in range(8)]
    
    _rng = np.random.default_rng(TrainingConfig.SAMPLER_SEED)
    print(f"✅ Empirical bucket sampling: B0={bucket_probs[0]:.4f} w={bucket_weights[0]:.1f}x, "
          f"B5={bucket_probs[5]:.4f} w={bucket_weights[5]:.1f}x")
    
    if PLOTTING_AVAILABLE:
        plt.ion()
        fig = plt.figure(figsize=(10, 6))

    for epoch_idx in range(start_epoch, total_epochs):
        start_time = time.time()
        
        # --- Training Phase ---
        model.train()
        total_train_loss = 0.0
        
        # 快速分層均勻抽樣 (Stratified Uniform Sampling)，速度極快且記憶體消耗為 0
        bucket_sample_sizes = np.round(train_size * normalized_bucket_probs).astype(np.int32)
        diff = train_size - bucket_sample_sizes.sum()
        if diff != 0:
            largest_bucket = np.argmax(bucket_sample_sizes)
            bucket_sample_sizes[largest_bucket] += diff
        print(f"Epoch {epoch_idx+1} bucket sample sizes: {bucket_sample_sizes}")
            
        perm_parts = []
        for b in range(8):
            if len(bucket_indices[b]) > 0 and bucket_sample_sizes[b] > 0:
                sampled = _rng.choice(bucket_indices[b], size=bucket_sample_sizes[b], replace=True)
                perm_parts.append(sampled)
        perm = np.concatenate(perm_parts)
        _rng.shuffle(perm) # 隨機打亂以混合不同分桶的批次
        
        num_train_batches = (train_size + TrainingConfig.BATCH_SIZE - 1) // TrainingConfig.BATCH_SIZE
        
        for batch_idx in range(num_train_batches):
            start = batch_idx * TrainingConfig.BATCH_SIZE
            end = min(start + TrainingConfig.BATCH_SIZE, train_size)
            batch_slice = perm[start:end]
            real_indices = train_indices[batch_slice]
            
            # Dynamic GPU/CPU Transfer: load batches on the fly from the full dataset arrays
            inputs_stm = torch.from_numpy(features_stm_full[real_indices]).long().to(device, non_blocking=True)
            inputs_nstm = torch.from_numpy(features_nstm_full[real_indices]).long().to(device, non_blocking=True)
            targets = torch.from_numpy(targets_full[real_indices]).to(device, non_blocking=True).unsqueeze(1).clamp(0.001, 0.999)  # Target Bounding: INT16 安全防線
            # SF18 bucket: (piece_count - 1) // 4, clamped to [0, 7]
            pc = train_pc_cpu[batch_slice].astype(np.int32)
            bucket_ids = torch.from_numpy(np.clip((pc - 1) // 4, 0, 7).astype(np.int64)).to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # AMP Forward Pass (Modern Syntax, safe for CPU and CUDA)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(inputs_stm, inputs_nstm, bucket_ids)
                probs = torch.sigmoid(outputs)
                if TrainingConfig.LOSS_EXPONENT == 2.0:
                    loss = criterion(probs, targets)
                else:
                    loss = torch.mean(torch.abs(probs - targets) ** TrainingConfig.LOSS_EXPONENT)
            
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
        
        # --- Validation Phase & Bucket-level Diagnostic ---
        model.eval()
        total_val_loss = 0.0
        total_val_samples = 0
        num_val_batches = (val_size + TrainingConfig.BATCH_SIZE - 1) // TrainingConfig.BATCH_SIZE
        
        _criterion_none = nn.MSELoss(reduction='none')
        _bucket_sums   = torch.zeros(8, device=device)
        _bucket_counts = torch.zeros(8, device=device)
        
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
                
                # 評估與分桶統計合併，避免兩次 forward pass 帶來的翻倍開銷
                probs = torch.sigmoid(outputs)
                if TrainingConfig.LOSS_EXPONENT == 2.0:
                    loss_elements = _criterion_none(probs, targets)
                else:
                    loss_elements = torch.abs(probs - targets) ** TrainingConfig.LOSS_EXPONENT
                total_val_loss += loss_elements.sum().item()
                total_val_samples += end - start
                
                loss_elements_squeezed = loss_elements.squeeze(1)
                _bucket_sums.scatter_add_(0, bucket_ids, loss_elements_squeezed)
                _bucket_counts.scatter_add_(0, bucket_ids, torch.ones(end - start, device=device))
                
        avg_val_loss = total_val_loss / total_val_samples

        _RANGES = ["2-4 ", "5-8 ", "9-12", "13-16", "17-20", "21-24", "25-28", "29-32"]
        print(f"  {'Bucket':<8} {'Pieces':<8} {'Samples':>10}  {'Val Loss':>10}")
        print(f"  {'-'*42}")
        for _b in range(8):
            if _bucket_counts[_b] > 0:
                _bl = (_bucket_sums[_b] / _bucket_counts[_b]).item()
                print(f"  B{_b:<7} {_RANGES[_b]:<8} {int(_bucket_counts[_b].item()):>10,}  {_bl:>10.6f}")
        del _criterion_none, _bucket_sums, _bucket_counts
        # -------------------------------------------------------
        validate_typical_positions(model, device)
        if (
            TrainingConfig.PREACT_DIAG_SAMPLES > 0
            and TrainingConfig.PREACT_DIAG_EVERY > 0
            and (epoch_idx + 1) % TrainingConfig.PREACT_DIAG_EVERY == 0
        ):
            diagnose_dense_pre_activations(
                model,
                device,
                val_f_stm,
                val_f_nstm,
                val_pc,
                TrainingConfig.PREACT_DIAG_SAMPLES,
            )

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
            'feature_encoding_version': TrainingConfig.FEATURE_ENCODING_VERSION,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': min(best_val_loss, avg_val_loss),
            'loss_type': 'MSE',
            'loss_exponent': TrainingConfig.LOSS_EXPONENT,
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
