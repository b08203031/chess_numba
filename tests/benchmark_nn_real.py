import os
import sys
import numpy as np
import time
import numba

# ? å…¥è·¯å?ä»¥ä¾¿å°å…¥å¼•æ?
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chess_engine.nnue.ml_eval.inference import init_accumulator, update_accumulator, nnue_forward_incremental

def benchmark_incremental():
    print("="*60)
    print("?? æ¨¡æ“¬?Ÿå¯¦?œå??´æ™¯ï¼šå??æ›´??(Incremental Update) æ¸¬è©¦")
    print("="*60)
    
    # 1. æº–å??å?å±€??(èµ·é?)
    test_bbs = np.zeros(12, dtype=np.uint64)
    test_bbs[0] = np.uint64(0x000000000000FF00) # White Pawns
    test_bbs[5] = np.uint64(0x0000000000000010) # White King
    test_bbs[6] = np.uint64(0x00FF000000000000) # Black Pawns
    test_bbs[11] = np.uint64(0x1000000000000000) # Black King
    
    # ?å??ç©º??(? é€Ÿæ?å°‹æ??„å…¸?‹é?ç½?
    accumulator_stack = np.zeros((128, 2, 512), dtype=np.int32)
    
    # --- A. ç¬¬ä?æ¬¡å†·?Ÿå? (?? ---
    print("\n[Step 1] ?å??–èµ·å§‹å???(init_accumulator)...")
    init_accumulator(test_bbs, accumulator_stack)
    print("?å??–å??ã€?)

    # --- B. æ¨¡æ“¬å¢é??´æ–° (å¿? ---
    # æ¨¡æ“¬é¦¬å? B1 èµ°åˆ° C3 ?„å?ä½?    piece_idx = 1 # White Knight
    from_sq = 1   # B1
    to_sq = 18    # C3
    
    @numba.njit()
    def test_loop(n, piece_bbs, piece_idx, from_sq, to_sq, acc_stack, add_buf, rem_buf, layer1, layer2, layer3):
        for i in range(n):
            # æ¨¡æ“¬èµ°ä?æ­?(Ply 1)
            update_accumulator(piece_bbs, add_buf, rem_buf, np.int32(1), acc_stack)
            # ?²è?è©•ä¼°
            _ = nnue_forward_incremental(np.int32(1), np.int32(0), acc_stack, np.int32(32), layer1, layer2, layer3)
        return n

    # ?å??†é?ç·©è??€
    add_buf = np.array([piece_idx * 64 + to_sq], dtype=np.int32)
    rem_buf = np.array([piece_idx * 64 + from_sq], dtype=np.int32)
    layer1 = np.zeros(1024, dtype=np.int32)
    layer2 = np.zeros(32, dtype=np.int32)
    layer3 = np.zeros(32, dtype=np.int32)

    # ?ç†± JIT
    test_loop(1, test_bbs, piece_idx, from_sq, to_sq, accumulator_stack, add_buf, rem_buf, layer1, layer2, layer3)
    
    iters = 200000 # å¢å?æ¬¡æ•¸è®“æ¸¬è©¦æ›´ç©©å?
    print(f"\n[Step 2] ?·è??å?å¢é??´æ–°æ¸¬è©¦ ({iters:,} æ¬?...")
    
    start = time.perf_counter()
    test_loop(iters, test_bbs, piece_idx, from_sq, to_sq, accumulator_stack, add_buf, rem_buf, layer1, layer2, layer3)
    end = time.perf_counter()
    
    total_time = end - start
    nps = iters / total_time
    
    print("-" * 30)
    print(f"ç¸½è€—æ?: {total_time * 1000:.2f} ms")
    print(f"å¹³å?è©•ä¼°?—æ?: {total_time / iters * 1000000:.3f} å¾®ç? (Î¼s)")
    print(f"?”¥ å¯¦æˆ°å¹³å? NPS: {int(nps):,} nodes/sec")
    print("-" * 30)
    print("\nçµè?ï¼šå??æ›´?°ç??Ÿåº¦?æ˜¯?œå?å¼•æ??Ÿæ­£?„å¯¦?›ï?")

if __name__ == "__main__":
    benchmark_incremental()
