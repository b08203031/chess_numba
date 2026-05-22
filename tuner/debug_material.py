"""
NNUE Comprehensive Diagnosis Probe v3
Produces train_diagnosis.txt with actionable diagnostics grounded in
the quantization chain: Q1=255, Q_HIDDEN=256, SCALE_OUT=400.0.
"""
import sys, os, torch, numpy as np, math
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

script_dir = os.path.dirname(os.path.abspath(__file__))
weights_dir = os.path.abspath(os.path.join(script_dir, '..', 'chess_engine', 'nnue', 'ml_eval', 'weights'))

latest_model_path = os.path.join(weights_dir, 'latest_model.pth')
best_model_path   = os.path.join(weights_dir, 'best_model.pth')

if os.path.exists(latest_model_path):
    model_path = latest_model_path
    model_source = 'latest_model.pth (current epoch)'
elif os.path.exists(best_model_path):
    model_path = best_model_path
    model_source = 'best_model.pth (fallback)'
else:
    print("ERROR: No model found in weights/. Has training started?")
    sys.exit(1)

from chess_engine.nnue.ml_eval.train import LayerStackNNUE

model = LayerStackNNUE(num_buckets=8)
with open(model_path, 'rb') as fh:
    checkpoint = torch.load(fh, map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

import datetime
now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Quantization constants (must match train.py / inference.py)
Q1 = 255
QH = 256
SCALE_OUT = 400.0  # Unified scaling factor
FC1_CLAMP = 128.5  # True INT16 physical limit (32767 / 255)

out_path = os.path.join(os.path.dirname(__file__), 'train_diagnosis.txt')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write("=" * 72 + "\n")
    f.write("NNUE Comprehensive Diagnosis Probe v3\n")
    f.write(f"Source: {model_source} | Time: {now_str}\n")
    f.write("=" * 72 + "\n\n")

    epoch_num = checkpoint.get('epoch', 0)
    best_val = checkpoint.get('best_val_loss', 'unknown')
    f.write(f"Epoch: {epoch_num} | Best Val Loss: {best_val}\n\n")

    # ================================================================
    # 1. FC1 Sparse Layer Health
    # ================================================================
    # FC1 weights are sliced to 22528 because King Mirroring reduces 
    # the feature space from 45056 to 22528 (32 king squares * 704 features per king).
    # FC1: (22528, 520) = 512 accumulator + 8 PSQT columns
    fc1_w_kp = model.fc1.weight.detach()[:22528].numpy()  # (22528, 520)
    if hasattr(model, 'fc1_p'):
        fc1_w_p = model.fc1_p.weight.detach()[:704].numpy()
        fc1_w_full = fc1_w_kp + np.tile(fc1_w_p, (32, 1))
    else:
        fc1_w_full = fc1_w_kp
    fc1_w    = fc1_w_full[:, :512]   # accumulator part used by CReLU/FC2/FC3/FC4
    fc1_psqt = fc1_w_full[:, 512:]   # PSQT shortcut part (22528, 8)
    fc1_b = model.fc1_bias.detach().numpy()  # (520,)

    fc1_abs = np.abs(fc1_w)
    fc1_sat = 100.0 * np.mean(fc1_abs >= (FC1_CLAMP - 1e-4))
    fc1_near_zero = 100.0 * np.mean(fc1_abs < 1e-4)

    f.write("=" * 72 + "\n")
    f.write("1. FC1 SPARSE FEATURE TRANSFORMER\n")
    f.write("=" * 72 + "\n")
    f.write(f"  Shape: {fc1_w.shape} | dtype: float32 → quantized INT16 × Q1({Q1})\n")
    f.write(f"  Weight stats: mean={np.mean(fc1_w):.5f}, std={np.std(fc1_w):.5f}\n")
    f.write(f"  Weight range: [{np.min(fc1_w):.4f}, {np.max(fc1_w):.4f}]\n")
    f.write(f"  Saturation (|w| >= {FC1_CLAMP}): {fc1_sat:.4f}%\n")
    f.write(f"  Near-zero (|w| < 1e-4): {fc1_near_zero:.2f}%\n")
    f.write(f"  Bias range: [{np.min(fc1_b):.4f}, {np.max(fc1_b):.4f}], mean={np.mean(fc1_b):.4f}\n")

    # Accumulator simulation: estimate pre-CReLU distribution
    # Typical position: ~31 active features.  acc = sum(31 random weights) + bias
    # Central Limit Theorem: acc ~ N(31*mean + bias_mean, 31*std^2)
    n_active = 31
    acc_mean = n_active * np.mean(fc1_w) + np.mean(fc1_b)
    acc_std  = np.sqrt(n_active) * np.std(fc1_w)
    # Fraction expected above CReLU(1.0) upper bound
    if acc_std > 1e-8:
        from scipy import stats as sp_stats
        frac_above_1 = 1.0 - sp_stats.norm.cdf(1.0, loc=acc_mean, scale=acc_std)
        frac_below_0 = sp_stats.norm.cdf(0.0, loc=acc_mean, scale=acc_std)
    else:
        frac_above_1 = 0.0
        frac_below_0 = 0.0

    f.write(f"\n  Accumulator Estimate (CLT, {n_active} features):\n")
    f.write(f"    Pre-CReLU mean={acc_mean:.3f}, std={acc_std:.3f}\n")
    f.write(f"    Expected clipped above 1.0: {100*frac_above_1:.1f}%\n")
    f.write(f"    Expected clipped below 0.0 (dead): {100*frac_below_0:.1f}%\n")
    f.write(f"    Effective utilization of [0, 1.0] range: {100*(1-frac_above_1-frac_below_0):.1f}%\n")

    # INT16 overflow check using actual current weight maximum
    actual_max_w = np.max(fc1_abs)
    worst_w_sum_int = n_active * int(actual_max_w * Q1)
    worst_bias_int  = int(np.max(np.abs(fc1_b)) * Q1)
    worst_total = worst_w_sum_int + worst_bias_int
    f.write(f"\n  INT16 Overflow Check (worst case {n_active} features using actual max |w| = {actual_max_w:.4f}):\n")
    f.write(f"    Max weight sum (INT): {worst_w_sum_int}\n")
    f.write(f"    Max bias (INT): {worst_bias_int}\n")
    f.write(f"    Worst total: {worst_total} / 32767 → {'SAFE' if worst_total < 32767 else 'OVERFLOW RISK!'}\n")

    if fc1_sat < 0.01:
        f.write("  [STATUS] FC1 weights well within bounds. ✓\n\n")
    elif fc1_sat < 1.0:
        f.write("  [STATUS] Healthy weight utilization. ✓\n\n")
    else:
        f.write("  [STATUS] WARNING: Significant saturation at clamp boundary!\n\n")

    # ================================================================
    # 2. Per-Bucket FC2/FC3/FC4 Analysis
    # ================================================================
    f.write("=" * 72 + "\n")
    f.write("2. LAYERSTACK BUCKET ANALYSIS (Independent Layers)\n")
    f.write("=" * 72 + "\n")

    for layer_name in ['fc2', 'fc3', 'fc4']:
        layers = getattr(model, layer_name)
        # Use bucket 0 to get dimensions
        out_dim = layers[0].weight.shape[0]
        in_dim  = layers[0].weight.shape[1]

        f.write(f"\n  --- {layer_name.upper()} ({out_dim}×{in_dim}) ---\n")

        # Weight clamping limit for expanded capacity
        if layer_name == 'fc4':
            max_float = (float(Q1) * float(Q1)) / (SCALE_OUT * float(QH))
        else:
            max_float = float(Q1) / float(QH)
            
        f.write(f"  QAT weight limit: ±{max_float:.4f}\n")

        for b in range(8):
            eff_w = layers[b].weight.detach().numpy()
            eff_b = layers[b].bias.detach().numpy()

            # Effective weight stats
            eff_max = np.max(np.abs(eff_w))
            eff_utilization = 100.0 * eff_max / max_float

            # Quantization round-trip error
            w_int = np.clip(np.round(eff_w * QH), -32768, 32767)
            w_roundtrip = w_int / QH
            quant_err = np.mean(np.abs(eff_w - w_roundtrip))

            # Effective rank (how many independent directions the weight matrix uses)
            try:
                s = np.linalg.svd(eff_w, compute_uv=False)
                s_norm = s / (s.sum() + 1e-8)
                eff_rank = np.exp(-np.sum(s_norm * np.log(s_norm + 1e-12)))
            except:
                eff_rank = 0.0

            # Neuron survival (for layers with CReLU: fc2, fc3)
            if layer_name in ('fc2', 'fc3'):
                alive = np.sum(eff_b > -0.5)  # neurons with reasonable chance of activation
                survival = f"alive={alive}/{out_dim}"
            else:
                survival = "N/A (output layer)"

            f.write(f"    B{b}: max|w|={eff_max:.4f} ({eff_utilization:.0f}% of limit)"
                    f" | Q_err={quant_err:.5f}"
                    f" | rank={eff_rank:.1f}/{min(out_dim, in_dim)}"
                    f" | {survival}\n")

    # ================================================================
    # 3. FC4 Output Layer Diagnostics
    # ================================================================
    f.write("=" * 72 + "\n")
    f.write("3. FC4 OUTPUT LAYER (Gradient Valve)\n")
    f.write("=" * 72 + "\n")

    for b in range(8):
        w4 = model.fc4[b].weight.detach().numpy()
        b4 = model.fc4[b].bias.detach().numpy()

        # Expected output range: with 32 inputs in [0, 1.0]
        # max_out ≈ sum(max(w4, 0)) * 1.0 + bias
        max_out = np.sum(np.maximum(w4, 0)) + b4[0]
        min_out = np.sum(np.minimum(w4, 0)) + b4[0]
        # Convert to centipawns via SCALE_OUT
        max_cp = max_out * SCALE_OUT
        min_cp = min_out * SCALE_OUT

        f.write(f"  B{b}: logit range=[{min_out:.3f}, {max_out:.3f}]"
                f" → CP range=[{min_cp:.0f}, {max_cp:.0f}]"
                f" | bias={b4[0]:.5f}\n")

    # ================================================================
    # 4. Per-Piece-Type Feature Weight Analysis
    # ================================================================
    f.write("\n" + "=" * 72 + "\n")
    f.write("4. PER-PIECE-TYPE FEATURE WEIGHT ANALYSIS\n")
    f.write("=" * 72 + "\n")
    f.write("  HalfKA index: king_sq(64) × 704 + piece_type(11) × 64 + sq(64)\n")
    f.write("  Piece types: 0=Pawn(W) 1=Knight(W) 2=Bishop(W) 3=Rook(W) 4=Queen(W)\n")
    f.write("               5=Pawn(B) 6=Knight(B) 7=Bishop(B) 8=Rook(B) 9=Queen(B) 10=King(B)\n\n")

    piece_names = ['Pawn(W)', 'Knight(W)', 'Bishop(W)', 'Rook(W)', 'Queen(W)',
                   'Pawn(B)', 'Knight(B)', 'Bishop(B)', 'Rook(B)', 'Queen(B)', 'King(B)']

    # Average across all valid king squares to get per-piece-type weight magnitude.
    # Mirrored HalfKA uses king_sq 0-31 (File A-D).
    for pt_idx, pt_name in enumerate(piece_names):
        piece_weights = []
        for king_sq in range(32): # Corrected range for mirrored King positions
            start_idx = king_sq * 704 + pt_idx * 64
            end_idx = start_idx + 64
            if end_idx <= 22528: # Corrected boundary for 22,528 dimensions
                piece_weights.append(fc1_w_kp[start_idx:end_idx, :512])

        if piece_weights:
            pw = np.concatenate(piece_weights)
            pw_reshaped = pw.reshape(-1, 512)
            per_neuron_importance = np.mean(np.abs(pw_reshaped), axis=0)
            
            # Virtual Layer base values
            p_str = ""
            if hasattr(model, 'fc1_p'):
                p_start = pt_idx * 64
                p_end = p_start + 64
                if p_end <= 704:
                    pw_p = fc1_w_p[p_start:p_end, :512]
                    p_str = f" | Virtual(fc1_p) mean|w|={np.mean(np.abs(pw_p)):.5f} max|w|={np.max(np.abs(pw_p)):.4f}"

            f.write(f"  {pt_name:>10}: mean|w|={np.mean(np.abs(pw)):.5f}"
                    f" | std={np.std(pw):.5f}"
                    f" | max|w|={np.max(np.abs(pw)):.4f}"
                    f" | top-10 neurons: {np.mean(np.sort(per_neuron_importance)[-10:]):.5f}{p_str}\n")

    # ================================================================
    # 4b. PSQT Per-Piece-Type Analysis (via FC1 PSQT columns)
    # ================================================================
    f.write("\n" + "=" * 72 + "\n")
    f.write("4b. PSQT SHORTCUT ANALYSIS (Multiple PSQT, 8 Buckets)\n")
    f.write("=" * 72 + "\n")
    f.write(f"  CP formula per perspective: psqt_sum × 0.5 × {SCALE_OUT}\n")
    f.write(f"  Net CP output: (psqt_stm - psqt_nstm) × 0.5 × {SCALE_OUT}\n\n")

    for b in range(8):
        col = fc1_psqt[:, b]  # (22528,)
        max_abs = float(np.max(np.abs(col)))
        mean_abs = float(np.mean(np.abs(col)))
        cp_range = 32 * max_abs * 0.5 * SCALE_OUT
        f.write(f"  Bucket {b}: mean|w|={mean_abs:.5f} | max|w|={max_abs:.5f}"
                f" | theoretical CP range ±{cp_range:.0f}\n")

    psqt_bias = fc1_b[512:]  # (8,)
    f.write("\n  PSQT bias (direct offset):\n")
    for b in range(8):
        cp = float(psqt_bias[b]) * 0.5 * SCALE_OUT
        f.write(f"    B{b}: bias={psqt_bias[b]:.6f} -> {cp:+.2f} CP\n")

    # Per-piece PSQT contribution (average over all king squares)
    f.write("\n  Per-piece-type PSQT weight (averaged over all king squares, all buckets):\n")
    for pt_idx, pt_name in enumerate(piece_names):
        piece_psqt = []
        for king_sq in range(32):
            start_idx = king_sq * 704 + pt_idx * 64
            end_idx = start_idx + 64
            if end_idx <= 22528:
                piece_psqt.append(fc1_psqt[start_idx:end_idx, :])  # (64, 8)
        if piece_psqt:
            pp = np.concatenate(piece_psqt, axis=0)  # (64*32, 8)
            mean_by_bucket = np.mean(pp, axis=0)     # (8,) - mean weight per bucket
            cp_by_bucket   = mean_by_bucket * 0.5 * SCALE_OUT
            cp_str = ' '.join([f'B{b}:{cp_by_bucket[b]:+.1f}' for b in range(8)])
            f.write(f"    {pt_name:>10}: {cp_str} CP\n")

    if hasattr(model, 'fc1_p'):
        f.write("\n  [8x8 ASCII PSQT BOARDS] (Pure static CP values extracted from Virtual Layer fc1_p, averaged over 8 buckets)\n")
        
        # We compute the avg PSQT value for each square independently of king.
        fc1_p_psqt = fc1_w_p[:, 512:] # (704, 8)
        
        for pt_idx, pt_name in enumerate(piece_names):
            p_start = pt_idx * 64
            p_end = p_start + 64
            if p_end <= 704:
                # (64, 8)
                sq_psqt_weights = fc1_p_psqt[p_start:p_end]
                # Average over the 8 buckets
                avg_sq_weights = np.mean(sq_psqt_weights, axis=1) # (64,)
                # Convert to CP
                sq_cp = avg_sq_weights * 0.5 * SCALE_OUT
                
                # Reshape to 8x8. Index sq = rank * 8 + file (A1 = 0, H8 = 63)
                board_2d = sq_cp.reshape(8, 8)
                
                f.write(f"\n    === {pt_name} ===\n")
                # Print from rank 8 down to rank 1
                for rank in range(7, -1, -1):
                    row_str = " | ".join([f"{board_2d[rank, file]:+5.1f}" for file in range(8)])
                    f.write(f"    {row_str}\n")

    # ================================================================
    # 5. Quantization Round-Trip Fidelity (Full Pipeline)
    # ================================================================
    f.write("\n" + "=" * 72 + "\n")
    f.write("5. QUANTIZATION ROUND-TRIP FIDELITY\n")
    f.write("=" * 72 + "\n")
    f.write(f"  Q1={Q1}, Q_HIDDEN={QH}, SCALE_OUT={SCALE_OUT}\n\n")

    # FC1
    fc1_int = np.clip(np.round(fc1_w * Q1), -32768, 32767)
    fc1_rt = fc1_int / Q1
    fc1_qerr = np.mean(np.abs(fc1_w - fc1_rt))
    fc1_max_err = np.max(np.abs(fc1_w - fc1_rt))
    f.write(f"  FC1: mean_err={fc1_qerr:.6f} | max_err={fc1_max_err:.6f} | step=1/{Q1}={1/Q1:.5f}\n")

    # FC2/FC3: weight scale = QH
    for layer_name in ['fc2', 'fc3']:
        eff_w = getattr(model, layer_name)[0].weight.detach().numpy()
        w_int = np.clip(np.round(eff_w * QH), -32768, 32767)
        w_rt  = w_int / QH
        qerr  = np.mean(np.abs(eff_w - w_rt))
        max_err = np.max(np.abs(eff_w - w_rt))
        f.write(f"  {layer_name.upper()} (B0): mean_err={qerr:.6f} | max_err={max_err:.6f} | step=1/{QH}={1/QH:.6f}\n")

    # FC4: DIFFERENT quantization scale = QH * SCALE_OUT / Q1
    fc4_w_scale = QH * SCALE_OUT / Q1  # ≈ 201.57
    eff_w4 = model.fc4[0].weight.detach().numpy()
    w4_int = np.clip(np.round(eff_w4 * fc4_w_scale), -32768, 32767)
    w4_rt  = w4_int / fc4_w_scale
    qerr4  = np.mean(np.abs(eff_w4 - w4_rt))
    max_err4 = np.max(np.abs(eff_w4 - w4_rt))
    f.write(f"  FC4 (B0): mean_err={qerr4:.6f} | max_err={max_err4:.6f}"
            f" | step=1/{fc4_w_scale:.2f}={1/fc4_w_scale:.6f}  (scale=QH*s_O/Q1)\n")

    # ================================================================
    # 6. Checkpoint Metadata
    # ================================================================
    f.write("\n" + "=" * 72 + "\n")
    f.write("6. CHECKPOINT METADATA\n")
    f.write("=" * 72 + "\n")
    f.write(f"  Epoch: {epoch_num}\n")
    f.write(f"  Best val loss: {best_val}\n")

    if 'current_bucket_val_losses' in checkpoint:
        losses = checkpoint['current_bucket_val_losses']
        f.write("  Per-bucket val losses:\n")
        for i, l in enumerate(losses):
            f.write(f"    B{i}: {l:.6f}\n")

    # Print keys in checkpoint for reference
    f.write(f"\n  Checkpoint keys: {sorted(checkpoint.keys())}\n")

print("Done. Output: " + out_path)