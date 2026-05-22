import os
import json
import time
import numpy as np
import numba
import chess

INPUT_JSONL = "tuner/tactical_data.jsonl"
TARGET_NPZ = "tuner/ultimate_halfka.npz"

# ========== 1. Numba HalfKA 解析器 (完全沿用) ==========
@numba.njit(numba.int32[:](numba.types.Array(numba.uint64, 1, 'C', readonly=True), numba.boolean), cache=True, fastmath=True)
def get_halfka_indices(bbs, is_black_perspective):
    indices = np.full(32, 45056, dtype=np.int32)
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
            if p_idx == 11: continue
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
                if idx_count < 32:
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
            if p_idx == 5: continue
            mapped_type = p_idx if p_idx < 5 else (p_idx - 1)
            bb = bbs[p_idx]
            while bb:
                lsb = bb & (~bb + np.uint64(1))
                sq = 0
                tmp = lsb >> np.uint64(1)
                while tmp:
                    sq += 1
                    tmp >>= np.uint64(1)
                if idx_count < 32:
                    indices[idx_count] = bucket * 704 + mapped_type * 64 + sq
                    idx_count += 1
                bb &= bb - np.uint64(1)
    return indices

@numba.njit(parallel=True, fastmath=True)
def convert_to_halfka(bbs_array, stm_array, wdl_array, out_stm, out_nstm, out_wdl):
    N = bbs_array.shape[0]
    for i in numba.prange(N):
        bbs = bbs_array[i]
        stm = stm_array[i]
        
        f_white = get_halfka_indices(bbs, False)
        f_black = get_halfka_indices(bbs, True)
        
        if stm == 0:
            for j in range(32):
                out_stm[i, j] = f_white[j]
                out_nstm[i, j] = f_black[j]
            out_wdl[i] = wdl_array[i]
        else:
            for j in range(32):
                out_stm[i, j] = f_black[j]
                out_nstm[i, j] = f_white[j]
            out_wdl[i] = 1.0 - wdl_array[i]

# ========== 2. 解析 JSONL 為 Bitboards ==========
def parse_tactical_data():
    print(f"Reading {INPUT_JSONL}...")
    fens = []
    wdls = []
    
    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            data = json.loads(line)
            fens.append(data['fen'])
            wdls.append(data['result'])
            
    N = len(fens)
    print(f"Loaded {N} tactical positions. Converting FEN to bitboards...")
    
    bbs_array = np.zeros((N, 12), dtype=np.uint64)
    stm_array = np.zeros(N, dtype=np.int8)
    wdl_array = np.array(wdls, dtype=np.float32)
    
    for i, fen in enumerate(fens):
        board = chess.Board(fen)
        stm_array[i] = 0 if board.turn == chess.WHITE else 1
        for p_type in range(1, 7):
            bbs_array[i, p_type - 1] = np.uint64(int(board.pieces(p_type, chess.WHITE)))
            bbs_array[i, p_type + 5] = np.uint64(int(board.pieces(p_type, chess.BLACK)))
            
    print("Bitboards parsed safely.")
    return bbs_array, stm_array, wdl_array

# ========== 3. 處理並寫入母庫 ==========
def main():
    start = time.time()
    bbs, stms, wdls = parse_tactical_data()
    N = len(bbs)
    
    new_stm = np.zeros((N, 32), dtype=np.int32)
    new_nstm = np.zeros((N, 32), dtype=np.int32)
    new_tgt = np.zeros(N, dtype=np.float32)
    
    print("Computing HalfKA features via Numba...")
    # Numba 第一圈編譯可能需要 1~2 秒，但之後會瞬間跑完 13499 筆
    convert_to_halfka(bbs, stms, wdls, new_stm, new_nstm, new_tgt)
    
    print(f"HalfKA computed in {time.time() - start:.2f}s. Loading ultimate dataset...")
    
    data = np.load(TARGET_NPZ)
    old_stm = data['features_stm']
    old_nstm = data['features_nstm']
    old_tgt = data['targets']
    
    old_len = len(old_tgt)
    print(f"Original dataset size: {old_len:,}")
    
    print("Concatenating arrays... (This requires ~3GB RAM briefly)")
    fused_stm = np.concatenate((old_stm, new_stm), axis=0)
    fused_nstm = np.concatenate((old_nstm, new_nstm), axis=0)
    fused_tgt = np.concatenate((old_tgt, new_tgt), axis=0)
    
    new_len = len(fused_tgt)
    print(f"Fusion complete! New dataset size: {new_len:,}")
    
    print(f"Overwriting {TARGET_NPZ}...")
    np.savez_compressed(
        TARGET_NPZ,
        features_stm=fused_stm,
        features_nstm=fused_nstm,
        targets=fused_tgt
    )
    data.close()
    
    print(f"Successfully injected {N:,} tactical antibodies! Total time: {time.time() - start:.2f}s")
    
if __name__ == '__main__':
    main()
