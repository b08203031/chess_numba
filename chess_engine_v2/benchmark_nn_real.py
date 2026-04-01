import os
import sys
import numpy as np
import time
import numba

# 加入路徑以便導入引擎
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chess_engine.ml_eval.inference import init_accumulator, update_accumulator, nnue_forward_incremental

def benchmark_incremental():
    print("="*60)
    print("🚀 模擬真實搜尋場景：增量更新 (Incremental Update) 測試")
    print("="*60)
    
    # 1. 準備初始局面 (起點)
    test_bbs = np.zeros(12, dtype=np.uint64)
    test_bbs[0] = np.uint64(0x000000000000FF00) # White Pawns
    test_bbs[5] = np.uint64(0x0000000000000010) # White King
    test_bbs[6] = np.uint64(0x00FF000000000000) # Black Pawns
    test_bbs[11] = np.uint64(0x1000000000000000) # Black King
    
    # 預分配空間 (加速搜尋時的典型配置)
    accumulator_stack = np.zeros((128, 2, 512), dtype=np.int32)
    
    # --- A. 第一次冷啟動 (慢) ---
    print("\n[Step 1] 初始化起始局面 (init_accumulator)...")
    init_accumulator(test_bbs, accumulator_stack)
    print("初始化完成。")

    # --- B. 模擬增量更新 (快) ---
    # 模擬馬從 B1 走到 C3 的動作
    piece_idx = 1 # White Knight
    from_sq = 1   # B1
    to_sq = 18    # C3
    
    @numba.njit()
    def test_loop(n, piece_bbs, piece_idx, from_sq, to_sq, acc_stack, add_buf, rem_buf):
        for i in range(n):
            # 模擬走一步 (Ply 1)
            update_accumulator(piece_bbs, add_buf, rem_buf, np.int32(1), acc_stack)
            # 進行評估
            _ = nnue_forward_incremental(np.int32(1), np.int32(0), acc_stack)
        return n

    # 預先分配緩衝區
    add_buf = np.array([piece_idx * 64 + to_sq], dtype=np.int32)
    rem_buf = np.array([piece_idx * 64 + from_sq], dtype=np.int32)

    # 預熱 JIT
    test_loop(1, test_bbs, piece_idx, from_sq, to_sq, accumulator_stack, add_buf, rem_buf)
    
    iters = 200000 # 增加次數讓測試更穩定
    print(f"\n[Step 2] 執行量化增量更新測試 ({iters:,} 次)...")
    
    start = time.perf_counter()
    test_loop(iters, test_bbs, piece_idx, from_sq, to_sq, accumulator_stack, add_buf, rem_buf)
    end = time.perf_counter()
    
    total_time = end - start
    nps = iters / total_time
    
    print("-" * 30)
    print(f"總耗時: {total_time * 1000:.2f} ms")
    print(f"平均評估耗時: {total_time / iters * 1000000:.3f} 微秒 (μs)")
    print(f"🔥 實戰平均 NPS: {int(nps):,} nodes/sec")
    print("-" * 30)
    print("\n結論：增量更新的速度才是搜尋引擎真正的實力！")

if __name__ == "__main__":
    benchmark_incremental()
