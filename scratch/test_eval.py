import os
import sys
import numpy as np
import torch

workspace_dir = r"c:\Users\ren cian\OneDrive\桌面\chess\chess_numba\chess_numba"
sys.path.insert(0, workspace_dir)

from chess_engine.nnue.ml_eval.train import LayerStackNNUE, fen_to_halfka_indices
from chess_engine.classical.fen_parser import parse_fen
from chess_engine.nnue.ml_eval.inference import init_accumulator, nnue_forward_incremental, weights_loaded

def evaluate_pytorch(model, fen):
    device = torch.device('cpu')
    stm_idx, nstm_idx, pc = fen_to_halfka_indices(fen)
    bucket_id = max(0, min(7, (pc - 1) // 4))
    
    inputs_stm = torch.tensor([stm_idx], dtype=torch.long, device=device)
    inputs_nstm = torch.tensor([nstm_idx], dtype=torch.long, device=device)
    bucket_ids = torch.tensor([bucket_id], dtype=torch.long, device=device)
    
    with torch.no_grad():
        logit = model(inputs_stm, inputs_nstm, bucket_ids).item()
    # Convert to centipawns
    return logit / 0.0025

def evaluate_jit(fen):
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
    return val

def main():
    device = torch.device('cpu')
    model = LayerStackNNUE(num_buckets=8).to(device)
    
    best_model_path = os.path.join(workspace_dir, 'chess_engine/nnue/ml_eval/weights/best_model.pth')
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    positions = {
        "Starting pos": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "White missing b1 knight": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/R1BQKBNR w KQkq - 0 1",
        "White missing d1 queen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1"
    }
    
    print(f"{'Position':<25} | {'PyTorch Eval (cp)':<18} | {'JIT Eval (cp)':<15}")
    print("-" * 65)
    for name, fen in positions.items():
        py_val = evaluate_pytorch(model, fen)
        jit_val = evaluate_jit(fen)
        print(f"{name:<25} | {py_val:>18.2f} | {jit_val:>15d}")

if __name__ == "__main__":
    main()
