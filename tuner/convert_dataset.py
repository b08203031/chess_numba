import os
import time
import numpy as np
import numba

INPUT_FILE = "tuner/ultimate_dataset.npz"
OUTPUT_FILE = "tuner/ultimate_halfka.npz"

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
                if idx_count < 32:
                    indices[idx_count] = bucket * 704 + mapped_type * 64 + sq
                    idx_count += 1
                bb &= bb - np.uint64(1)
    return indices

@numba.njit(parallel=True, fastmath=True)
def process_all_samples(bbs_array, stm_array, wdl_array, out_stm, out_nstm, out_wdl):
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

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found!")
        return
        
    print(f"Loading {INPUT_FILE} into memory...")
    start_time = time.time()
    data = np.load(INPUT_FILE)
    bbs = data['piece_bbs']
    wdl = data['results']
    stm = data['game_states'][:, 0].astype(np.int8) if 'game_states' in data else np.zeros(len(bbs), dtype=np.int8)
    data.close()
    
    N = len(bbs)
    print(f"Loaded {N} samples in {time.time() - start_time:.2f}s.")
    
    print("Allocating memory for HalfKA indices (this will use ~2.5 GB RAM)...")
    out_stm = np.zeros((N, 32), dtype=np.int32)
    out_nstm = np.zeros((N, 32), dtype=np.int32)
    out_wdl = np.zeros(N, dtype=np.float32)
    
    print("Pre-computing HalfKA using multi-threaded Numba prange...")
    compute_start = time.time()
    process_all_samples(bbs, stm, wdl, out_stm, out_nstm, out_wdl)
    print(f"Computation finished in {time.time() - compute_start:.2f}s!")
    
    print(f"Saving to {OUTPUT_FILE}...")
    save_start = time.time()
    np.savez_compressed(
        OUTPUT_FILE,
        features_stm=out_stm,
        features_nstm=out_nstm,
        targets=out_wdl
    )
    print(f"Saved successfully in {time.time() - save_start:.2f}s!")
    
if __name__ == '__main__':
    main()
