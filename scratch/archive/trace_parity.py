import torch
import torch.nn.functional as F
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../chess_engine/nnue/ml_eval")))

from train import LayerStackNNUE, fen_to_halfka_indices
from chess_engine.nnue.ml_eval.inference import (
    init_accumulator,
    _sf_hm_king_bucket,
    FC1_WEIGHT,
    FC1_BIAS,
    FC2_WEIGHTS_I32,
    FC2_BIASES,
    FC3_WEIGHTS_I32,
    FC3_BIASES,
    FC4_WEIGHTS_I32,
    FC4_BIASES
)
from chess_engine.classical.fen_parser import parse_fen

def trace():
    device = torch.device('cpu')
    model = LayerStackNNUE(num_buckets=8).to(device)
    
    weights_path = "./chess_engine/nnue/ml_eval/weights/best_model.pth"
    if not os.path.exists(weights_path):
        print("Error: best_model.pth not found")
        return
        
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    fen = "8/8/8/4k3/8/8/3K4/8 w - - 0 1"
    print(f"Tracing FEN: {fen}")
    
    # 1. Float forward pass
    stm_idx, nstm_idx, pc = fen_to_halfka_indices(fen)
    bucket_id = max(0, min(7, (pc - 1) // 4))
    
    inputs_stm = torch.tensor([stm_idx], dtype=torch.long)
    inputs_nstm = torch.tensor([nstm_idx], dtype=torch.long)
    bucket_ids = torch.tensor([bucket_id], dtype=torch.long)
    
    with torch.no_grad():
        # FC1
        acc_stm_fl = model.fc1(inputs_stm) + model.fc1_bias
        acc_nstm_fl = model.fc1(inputs_nstm) + model.fc1_bias
        
        acc_stm_crelu_fl = model.crelu(acc_stm_fl)
        acc_nstm_crelu_fl = model.crelu(acc_nstm_fl)
        x_fl = torch.cat((acc_stm_crelu_fl, acc_nstm_crelu_fl), dim=1) # (1, 1024)
        
        # FC2
        w2_fl = model.fact_fc2.weight + model.delta_fc2[bucket_id].weight
        b2_fl = model.fact_fc2.bias + model.delta_fc2[bucket_id].bias
        h2_pre_fl = F.linear(x_fl, w2_fl, b2_fl)
        h2_fl = model.crelu(h2_pre_fl)
        
        # FC3
        w3_fl = model.fact_fc3.weight + model.delta_fc3[bucket_id].weight
        b3_fl = model.fact_fc3.bias + model.delta_fc3[bucket_id].bias
        h3_pre_fl = F.linear(h2_fl, w3_fl, b3_fl)
        h3_fl = model.crelu(h3_pre_fl)
        
        # FC4
        w4_fl = model.fact_fc4.weight + model.delta_fc4[bucket_id].weight
        b4_fl = model.fact_fc4.bias + model.delta_fc4[bucket_id].bias
        out_fl = F.linear(h3_fl, w4_fl, b4_fl)
        
        logit = out_fl.item()
        float_cp = logit / 0.0025
        print(f"Float logit: {logit:.6f} | Float CP: {float_cp:.2f}")

    # 2. Integer (Quantized) JIT forward pass
    piece_bbs, _, game_state = parse_fen(fen)
    side_to_move = int(game_state[0])
    
    accumulator_stack = np.zeros((1, 2, 512), dtype=np.int32)
    init_accumulator(piece_bbs, accumulator_stack)
    
    # Trace values
    acc_stm_qt = accumulator_stack[0, 0, :]
    acc_nstm_qt = accumulator_stack[0, 1, :]
    
    print("\n--- FC1 Accumulator Comparison ---")
    print(f"Float Acc STM range: [{acc_stm_fl.min().item():.4f}, {acc_stm_fl.max().item():.4f}], mean={acc_stm_fl.mean().item():.4f}")
    print(f"Quant Acc STM range: [{acc_stm_qt.min():.4f}, {acc_stm_qt.max():.4f}], mean={acc_stm_qt.mean():.4f}")
    print(f"Scale parity: Quant Acc STM / 127 mean: {acc_stm_qt.mean()/127.0:.4f}")
    
    # Layer 1 CReLU
    layer1 = np.empty(1024, dtype=np.int32)
    for i in range(512):
        layer1[i]       = max(0, min(16129, acc_stm_qt[i]))
        layer1[512 + i] = max(0, min(16129, acc_nstm_qt[i]))
        
    print(f"Float CReLU STM range: [{acc_stm_crelu_fl.min().item():.4f}, {acc_stm_crelu_fl.max().item():.4f}]")
    print(f"Quant CReLU STM range: [{layer1[:512].min():.4f}, {layer1[:512].max():.4f}] (scale / 127: [{layer1[:512].min()/127:.4f}, {layer1[:512].max()/127:.4f}])")
    
    x_fl_np = x_fl.detach().cpu().numpy()[0]
    layer1_fl = layer1 / 127.0
    l1_diff = np.abs(x_fl_np - layer1_fl)
    print(f"Layer 1 CReLU max diff: {l1_diff.max():.6f}, mean diff: {l1_diff.mean():.6f}")
    large_diff_indices = np.where(l1_diff > 0.05)[0]
    print(f"Number of indices with diff > 0.05: {len(large_diff_indices)}")
    if len(large_diff_indices) > 0:
        for idx in large_diff_indices[:5]:
            print(f"  idx {idx}: float={x_fl_np[idx]:.6f}, quant={layer1[idx]} ({layer1_fl[idx]:.6f}), diff={l1_diff[idx]:.6f}")
    
    # Layer 2 FC2
    layer2 = np.empty(32, dtype=np.int32)
    w2 = FC2_WEIGHTS_I32[bucket_id]
    b2 = FC2_BIASES[bucket_id]
    
    # Let's compare PyTorch w2 vs Quantized w2
    w2_fl_np = w2_fl.detach().cpu().numpy()
    b2_fl_np = b2_fl.detach().cpu().numpy()
    w2_qt_scaled = np.round(w2_fl_np * 128).astype(np.int32)
    b2_qt_scaled = np.round(b2_fl_np * 128 * 127).astype(np.int32)
    
    w2_diff = np.abs(w2 - w2_qt_scaled).max()
    b2_diff = np.abs(b2 - b2_qt_scaled).max()
    print(f"\n--- FC2 Weight Comparison ---")
    print(f"Max Weight diff: {w2_diff}")
    print(f"Max Bias diff: {b2_diff}")
    
    for i in range(32):
        acc = np.int64(b2[i])
        w_row = w2[i]
        for j in range(1024):
            acc += np.int64(w_row[j]) * np.int64(layer1[j])
        shifted = acc >> np.int64(7)
        if shifted <= np.int64(0):
            layer2[i] = 0
        elif shifted >= np.int64(16129):
            layer2[i] = 16129
        else:
            layer2[i] = np.int32(shifted)
            
    print(f"\n--- FC2 Activation Comparison ---")
    print(f"Float FC2 range: [{h2_fl.min().item():.4f}, {h2_fl.max().item():.4f}], mean={h2_fl.mean().item():.4f}")
    print(f"Quant FC2 range: [{layer2.min():.4f}, {layer2.max():.4f}], mean={layer2.mean():.4f} (scale / 127: [{layer2.min()/127.0:.4f}, {layer2.max()/127.0:.4f}], mean={layer2.mean()/127.0:.4f})")
    
    h2_fl_np = h2_fl.detach().cpu().numpy()[0]
    print(f"{'Idx':>3} | {'Float H2':>10} | {'Quant H2':>8} (H/127) | {'Diff':>10}")
    print("-" * 50)
    for i in range(32):
        fl_h = h2_fl_np[i]
        qt_h = layer2[i] / 127.0
        print(f"{i:>3} | {fl_h:>10.6f} | {layer2[i]:>8d} ({qt_h:>+6.3f}) | {qt_h - fl_h:>+10.6f}")
        
    # Analyze Index 16 detailed terms
    print("\n--- FC2 Index 16 Detailed Terms ---")
    fl_bias_16 = b2_fl_np[16]
    qt_bias_16 = b2[16]
    print(f"Float bias: {fl_bias_16:.6f}")
    print(f"Quant bias: {qt_bias_16} (bias/16256 = {qt_bias_16/16256.0:.6f})")
    
    fl_dot = 0.0
    qt_dot = 0
    large_term_diffs = []
    for j in range(1024):
        fl_term = w2_fl_np[16, j] * x_fl_np[j]
        qt_term = int(w2[16, j]) * int(layer1[j])
        fl_dot += fl_term
        qt_dot += qt_term
        term_diff = (qt_term / 16256.0) - fl_term
        if abs(term_diff) > 0.005:
            large_term_diffs.append((j, w2_fl_np[16, j], w2[16, j], x_fl_np[j], layer1[j], fl_term, qt_term, term_diff))
            
    print(f"Float dot product: {fl_dot:.6f}")
    print(f"Quant dot product: {qt_dot} (dot/16256 = {qt_dot/16256.0:.6f})")
    print(f"Float pre-activation: {fl_dot + fl_bias_16:.6f}")
    print(f"Quant pre-activation: {qt_dot + qt_bias_16} (pre/16256 = {(qt_dot + qt_bias_16)/16256.0:.6f}) (shifted >> 7 = {(qt_dot + qt_bias_16)>>7})")
    print(f"Number of terms with diff > 0.005: {len(large_term_diffs)}")
    for item in large_term_diffs[:5]:
        print(f"  idx {item[0]}: w_fl={item[1]:.6f}, w_qt={item[2]}, x_fl={item[3]:.6f}, x_qt={item[4]}, fl_term={item[5]:.6f}, qt_term={item[6]} (qt/16256={item[6]/16256.0:.6f}), diff={item[7]:+6.4f}")
        
    # Layer 3 FC3
    layer3 = np.empty(32, dtype=np.int32)
    w3 = FC3_WEIGHTS_I32[bucket_id]
    b3 = FC3_BIASES[bucket_id]
    for i in range(32):
        acc = np.int64(b3[i])
        w_row = w3[i]
        for j in range(32):
            acc += np.int64(w_row[j]) * np.int64(layer2[j])
        shifted = acc >> np.int64(7)
        if shifted <= np.int64(0):
            layer3[i] = 0
        elif shifted >= np.int64(16129):
            layer3[i] = 16129
        else:
            layer3[i] = np.int32(shifted)
            
    print(f"\n--- FC3 Activation Comparison ---")
    print(f"Float FC3 range: [{h3_fl.min().item():.4f}, {h3_fl.max().item():.4f}], mean={h3_fl.mean().item():.4f}")
    print(f"Quant FC3 range: [{layer3.min():.4f}, {layer3.max():.4f}], mean={layer3.mean():.4f} (scale / 127: [{layer3.min()/127.0:.4f}, {layer3.max()/127.0:.4f}], mean={layer3.mean()/127.0:.4f})")
    
    h3_fl_np = h3_fl.detach().cpu().numpy()[0]
    print(f"{'Idx':>3} | {'Float H3':>10} | {'Quant H3':>8} (H/127) | {'Diff':>10}")
    print("-" * 50)
    for i in range(32):
        fl_h = h3_fl_np[i]
        qt_h = layer3[i] / 127.0
        print(f"{i:>3} | {fl_h:>10.6f} | {layer3[i]:>8d} ({qt_h:>+6.3f}) | {qt_h - fl_h:>+10.6f}")
    
    # Layer 4 FC4
    w4 = FC4_WEIGHTS_I32[bucket_id, 0]
    output = np.int64(FC4_BIASES[bucket_id, 0])
    
    print("\n--- FC4 Detailed Comparison ---")
    w4_fl_np = w4_fl.detach().cpu().numpy()[0]
    h3_fl_np = h3_fl.detach().cpu().numpy()[0]
    b4_fl_val = b4_fl.item()
    
    print(f"Float FC4 bias: {b4_fl_val:.6f}")
    print(f"Quant FC4 bias: {FC4_BIASES[bucket_id, 0]} (bias / 16256 = {FC4_BIASES[bucket_id, 0] / 16256.0:.6f})")
    
    float_sum = b4_fl_val
    quant_sum = np.int64(FC4_BIASES[bucket_id, 0])
    
    print(f"{'Idx':>3} | {'Float W4':>10} | {'Quant W4':>8} (W/128) | {'Float H3':>10} | {'Quant H3':>8} (H/127) | {'Float Prod':>10} | {'Quant Prod':>10} (P/16256)")
    print("-" * 110)
    for i in range(32):
        fl_p = w4_fl_np[i] * h3_fl_np[i]
        qt_p = np.int64(w4[i]) * np.int64(layer3[i])
        float_sum += fl_p
        quant_sum += qt_p
        print(f"{i:>3} | {w4_fl_np[i]:>10.6f} | {w4[i]:>8d} ({w4[i]/128.0:>+6.3f}) | {h3_fl_np[i]:>10.6f} | {layer3[i]:>8d} ({layer3[i]/127.0:>+6.3f}) | {fl_p:>10.6f} | {qt_p:>10d} ({qt_p/16256.0:>+10.6f})")
        
    print("-" * 110)
    print(f"Cumulative Float Sum: {float_sum:.6f}")
    print(f"Cumulative Quant Sum: {quant_sum} (sum / 16256 = {quant_sum / 16256.0:.6f})")
    
    print(f"\n--- FC4 Output Comparison ---")
    print(f"Float FC4 raw output: {out_fl.item():.6f}")
    print(f"Quant FC4 raw output: {output + sum(np.int64(w4[i]) * np.int64(layer3[i]) for i in range(32))} (scale / (128*127): {quant_sum / (128.0*127.0):.6f})")
    
    OUTPUT_DIVISOR = np.int64(41)
    if quant_sum >= np.int64(0):
        jit_quant_cp = np.int32(quant_sum // OUTPUT_DIVISOR)
    else:
        jit_quant_cp = np.int32(-((-quant_sum) // OUTPUT_DIVISOR))
        
    print(f"Quant JIT CP: {jit_quant_cp}")

if __name__ == "__main__":
    import sys
    out_file = "./scratch/trace_output.txt"
    print(f"Redirecting output to {out_file}")
    with open(out_file, "w", encoding="utf-8") as f:
        sys.stdout = f
        trace()
