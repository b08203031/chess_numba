import numpy as np
import numba
import time

# 假裝的 FC1 權重 (22528, 520) - Int16
FC1_WEIGHT = np.random.randint(-500, 500, size=(22528, 520), dtype=np.int16)

@numba.njit(fastmath=True)
def init_accumulator_int16(piece_bbs, accumulator_stack):
    # 簡化的 FC1 初始化，假裝存取多個特徵
    for i in range(16): # 假裝找到 16 顆棋子
        f = (i * 704 + 13) % 22528
        accumulator_stack[0, :] += FC1_WEIGHT[f, :]

@numba.njit(fastmath=True)
def init_accumulator_int32(piece_bbs, accumulator_stack):
    # 同上的簡化邏輯
    for i in range(16):
        f = (i * 704 + 13) % 22528
        accumulator_stack[0, :] += FC1_WEIGHT[f, :]

@numba.njit(fastmath=True)
def benchmark_int16(runs, dummy_bbs, accumulator_stack):
    for _ in range(runs):
        accumulator_stack[0, :] = 0  # 重設
        init_accumulator_int16(dummy_bbs, accumulator_stack)

@numba.njit(fastmath=True)
def benchmark_int32(runs, dummy_bbs, accumulator_stack):
    for _ in range(runs):
        accumulator_stack[0, :] = 0  # 重設
        init_accumulator_int32(dummy_bbs, accumulator_stack)


if __name__ == "__main__":
    runs = 3_000_000
    print(f"正在預熱編譯 (Numba JIT)...")
    
    # 預熱
    acc_16 = np.zeros((1, 520), dtype=np.int16)
    acc_32 = np.zeros((1, 520), dtype=np.int32)
    dummy_bbs = np.zeros(12, dtype=np.uint64)
    benchmark_int16(10, dummy_bbs, acc_16)
    benchmark_int32(10, dummy_bbs, acc_32)
    
    print(f"\n[Starts] FC1 accumulation benchmark [{runs:,} runs]")
    print("-" * 50)
    
    # 測試 Int16 (L1 Cache 友善)
    start_16 = time.perf_counter()
    benchmark_int16(runs, dummy_bbs, acc_16)
    end_16 = time.perf_counter()
    t16 = end_16 - start_16
    nps16 = runs / t16
    print(f"[INT16] Time: {t16:.3f} s | Speed: {nps16:,.0f} ops/sec")
    
    # 測試 Int32 (若因 Q1 過大而被迫升降級)
    start_32 = time.perf_counter()
    benchmark_int32(runs, dummy_bbs, acc_32)
    end_32 = time.perf_counter()
    t32 = end_32 - start_32
    nps32 = runs / t32
    print(f"[INT32] Time: {t32:.3f} s | Speed: {nps32:,.0f} ops/sec")
    
    print("-" * 50)
    diff = nps16 / nps32
    slowdown = (t32 - t16) / t16 * 100
    print(f"[Result] INT16 is about {slowdown:.1f}% faster than INT32!")
