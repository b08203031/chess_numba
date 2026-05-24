import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chess_engine.classical.fen_parser import parse_fen
from chess_engine.nnue.ml_eval.train import fen_to_halfka_indices

# Inference feature extraction rewritten simply to collect feature indices
def get_inference_indices(fen):
    piece_bbs, occupancy_bbs, game_state = parse_fen(fen)
    
    # Extract king squares
    king_w_sq = 0
    if piece_bbs[5]:
        bb = piece_bbs[5]
        king_w_sq = int(np.log2(bb & -bb))

    king_b_sq = 0
    if piece_bbs[11]:
        bb = piece_bbs[11]
        king_b_sq = int(np.log2(bb & -bb))

    # White Perspective
    flip_h_w = (king_w_sq & 7) < 4
    # sf_hm_king_bucket logic
    oriented_king_sq_w = king_w_sq
    if (oriented_king_sq_w & 7) < 4:
        oriented_king_sq_w ^= 7
    bucket_w = (7 - (oriented_king_sq_w >> 3)) * 4 + (7 - (oriented_king_sq_w & 7))

    white_indices = []
    for p_idx in range(12):
        if p_idx < 5:
            mt_w = p_idx * 2
        elif p_idx == 5:
            mt_w = 10
        elif p_idx < 11:
            mt_w = (p_idx - 6) * 2 + 1
        else:
            mt_w = 10

        bb = piece_bbs[p_idx]
        while bb:
            sq = int(np.log2(bb & -bb))
            sq_w = sq ^ 7 if flip_h_w else sq
            f_w = bucket_w * 704 + mt_w * 64 + sq_w
            white_indices.append(f_w)
            bb &= bb - np.uint64(1)

    # Black Perspective
    king_b_flip = king_b_sq ^ 56
    flip_h_b = (king_b_flip & 7) < 4
    oriented_king_sq_b = king_b_flip
    if (oriented_king_sq_b & 7) < 4:
        oriented_king_sq_b ^= 7
    bucket_b = (7 - (oriented_king_sq_b >> 3)) * 4 + (7 - (oriented_king_sq_b & 7))

    black_indices = []
    for p_idx in range(12):
        if p_idx < 5:
            mt_b = p_idx * 2 + 1
        elif p_idx == 5:
            mt_b = 10
        elif p_idx < 11:
            mt_b = (p_idx - 6) * 2
        else:
            mt_b = 10

        bb = piece_bbs[p_idx]
        while bb:
            sq = int(np.log2(bb & -bb))
            sq_b = sq ^ 56
            if flip_h_b:
                sq_b ^= 7
            f_b = bucket_b * 704 + mt_b * 64 + sq_b
            black_indices.append(f_b)
            bb &= bb - np.uint64(1)

    return sorted(white_indices), sorted(black_indices)

# Compare for "Initial Position" and the "Queen missing" positions
fens = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", # Black missing d8 queen
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1", # White missing d1 queen
]

for fen in fens:
    print(f"\nFEN: {fen}")
    train_stm, train_nstm, pc = fen_to_halfka_indices(fen)
    # Remove padding 22528 from train features
    train_stm = sorted([f for f in train_stm if f != 22528])
    train_nstm = sorted([f for f in train_nstm if f != 22528])
    
    inf_w, inf_b = get_inference_indices(fen)
    
    # STM and NSTM depending on side to move
    parts = fen.split()
    stm = 0 if parts[1] == 'w' else 1
    
    if stm == 0:
        inf_stm, inf_nstm = inf_w, inf_b
    else:
        inf_stm, inf_nstm = inf_b, inf_w
        
    print(f"STM Match: {train_stm == inf_stm}")
    if train_stm != inf_stm:
        print(f"  Train STM ({len(train_stm)}): {train_stm}")
        print(f"  Inference STM ({len(inf_stm)}): {inf_stm}")
        
    print(f"NSTM Match: {train_nstm == inf_nstm}")
    if train_nstm != inf_nstm:
        print(f"  Train NSTM ({len(train_nstm)}): {train_nstm}")
        print(f"  Inference NSTM ({len(inf_nstm)}): {inf_nstm}")
