"""
效能實驗：驗證三個效能主張的真實性
  Claim 1: accumulator_stack INT32 → INT16 可節省 50% 記憶體、提升 20-30% NPS
  Claim 2: FC2/FC3 計算保持 INT16 * INT16 accumulate INT32 比 INT32*INT32 快
  Claim 3: Swish LUT 可以 O(1) 替換 CReLU 而無額外成本

執行：python bench_claims.py
"""
import numpy as np
import numba
import time

# ─── 共用常數 ──────────────────────────────────────────────────────────────────
MAX_PLY   = 128
FC1_TOTAL = 520
FC1_DIM   = 512
FC2_IN    = 1024   # 2 × FC1_DIM (stm + nstm)
FC2_OUT   = 32
FC3_OUT   = 32
N_ITERS   = 5000   # 每個測試的迭代次數（模擬 NPS）

rng = np.random.default_rng(0)

# ─── CLAIM 1: Accumulator stack dtype ─────────────────────────────────────────
print("=" * 60)
print("CLAIM 1: accumulator_stack INT32 vs INT16 memory & speed")
print("=" * 60)

acc_i32 = np.zeros((MAX_PLY, 2, FC1_TOTAL), dtype=np.int32)
acc_i16 = np.zeros((MAX_PLY, 2, FC1_TOTAL), dtype=np.int16)

# Weights INT16 in range ±255 (Q1=255, fc1 weights ≤ 2.0 float → ≤510 INT)
w_i16 = rng.integers(-510, 510, size=(22528, FC1_TOTAL), dtype=np.int16)

print(f"  INT32 accumulator size per search: {acc_i32.nbytes / 1024:.1f} KB")
print(f"  INT16 accumulator size per search: {acc_i16.nbytes / 1024:.1f} KB")
print(f"  Memory reduction: {(1 - acc_i16.nbytes/acc_i32.nbytes)*100:.0f}%")

@numba.njit(cache=True)
def update_acc_i32(acc, weights, ply, perspective, feat_indices):
    for fi in feat_indices:
        for i in range(520):
            acc[ply, perspective, i] += weights[fi, i]

@numba.njit(cache=True)
def update_acc_i16(acc, weights, ply, perspective, feat_indices):
    for fi in feat_indices:
        for i in range(520):
            acc[ply, perspective, i] += weights[fi, i]

feats = np.array([100, 500, 1000, 5000, 10000, 15000, 20000], dtype=np.int32)

# Warm up JIT
update_acc_i32(acc_i32, w_i16, 0, 0, feats)
acc_i16_dummy = np.zeros((MAX_PLY, 2, FC1_TOTAL), dtype=np.int16)
update_acc_i16(acc_i16_dummy, w_i16, 0, 0, feats)

t0 = time.perf_counter()
for _ in range(N_ITERS):
    update_acc_i32(acc_i32, w_i16, 0, 0, feats)
t_i32 = time.perf_counter() - t0

t0 = time.perf_counter()
for _ in range(N_ITERS):
    update_acc_i16(acc_i16_dummy, w_i16, 0, 0, feats)
t_i16 = time.perf_counter() - t0

print(f"\n  Accumulator update speed ({N_ITERS} iters, 7 features each):")
print(f"    INT32: {t_i32*1e6/N_ITERS:.2f} µs/iter")
print(f"    INT16: {t_i16*1e6/N_ITERS:.2f} µs/iter")
speedup = t_i32 / t_i16
print(f"    Speedup (INT32/INT16): {speedup:.3f}x  ← {'INT16 faster' if speedup > 1 else 'INT32 faster or equal'}")

# Overflow check
max_acc_i16 = int(np.iinfo(np.int16).max)
max_features = 32  # pieces on board
max_weight_int = 510  # Q1=255 × max_float_weight=2.0
theoretical_max = max_features * max_weight_int
print(f"\n  INT16 overflow check:")
print(f"    Theoretical max accumulator value: {theoretical_max}")
print(f"    INT16 limit: {max_acc_i16}")
print(f"    Safe? {'✅ YES' if theoretical_max < max_acc_i16 else '❌ NO — OVERFLOW RISK'}")

# ─── CLAIM 2: INT16*INT16→INT32 vs INT32*INT32→INT32 in FC2 inner loop ────────
print("\n" + "=" * 60)
print("CLAIM 2: FC2/FC3 dtype INT16*INT16 vs INT32*INT32 inner loop")
print("=" * 60)

w2_i16 = rng.integers(-512, 512, size=(FC2_OUT, FC2_IN), dtype=np.int16)
w2_i32 = w2_i16.astype(np.int32)
layer1_i16 = rng.integers(0, 255, size=FC2_IN, dtype=np.int16)
layer1_i32 = layer1_i16.astype(np.int32)
b2_i32 = np.zeros(FC2_OUT, dtype=np.int32)
layer2_out = np.zeros(FC2_OUT, dtype=np.int32)

@numba.njit(cache=True)
def fc2_i32_i32(weights_i32, biases, inp_i32, out):
    for i in range(32):
        acc = biases[i]
        for j in range(1024):
            acc += weights_i32[i, j] * inp_i32[j]
        out[i] = max(0, min(255, acc >> 8))

@numba.njit(cache=True)
def fc2_i16_i16(weights_i16, biases, inp_i16, out):
    for i in range(32):
        acc = numba.int32(biases[i])
        for j in range(1024):
            acc += numba.int32(weights_i16[i, j]) * numba.int32(inp_i16[j])
        out[i] = max(0, min(255, acc >> 8))

# Warm up
fc2_i32_i32(w2_i32, b2_i32, layer1_i32, layer2_out)
fc2_i16_i16(w2_i16, b2_i32, layer1_i16, layer2_out)

N2 = 50000
t0 = time.perf_counter()
for _ in range(N2):
    fc2_i32_i32(w2_i32, b2_i32, layer1_i32, layer2_out)
t_fc2_i32 = time.perf_counter() - t0

t0 = time.perf_counter()
for _ in range(N2):
    fc2_i16_i16(w2_i16, b2_i32, layer1_i16, layer2_out)
t_fc2_i16 = time.perf_counter() - t0

print(f"  FC2 forward ({N2} iters, 1024 inputs × 32 outputs):")
print(f"    INT32*INT32: {t_fc2_i32*1e6/N2:.2f} µs/iter")
print(f"    INT16*INT16: {t_fc2_i16*1e6/N2:.2f} µs/iter")
speedup2 = t_fc2_i32 / t_fc2_i16
print(f"    Speedup: {speedup2:.3f}x  ← {'INT16 faster' if speedup2 > 1 else 'INT32 faster or equal (claim is wrong)'}")

# ─── CLAIM 3: Swish LUT vs CReLU ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("CLAIM 3: Swish LUT O(1) vs CReLU — training accuracy impact")
print("=" * 60)

# Build Swish LUT for INT16 range [0, 32767] (only positive, since acc can be negative pre-clip)
# Swish(x) = x * sigmoid(x), mapped from float to int
LUT_SIZE = 32768
lut_swish = np.zeros(LUT_SIZE, dtype=np.int16)
for i in range(LUT_SIZE):
    x_float = i / 255.0  # dequantize (Q1=255)
    swish_val = x_float / (1.0 + np.exp(-x_float))  # Swish
    lut_swish[i] = np.clip(round(swish_val * 255), 0, 255)

print(f"  Swish LUT size: {lut_swish.nbytes / 1024:.1f} KB")
print(f"  L1 Cache size (typical): 32-64 KB")
print(f"  LUT fits in L1? {'✅ Yes (~2KB)' if lut_swish.nbytes < 32768 else '⚠️  Check: might overflow L1 in search context'}")

# How different is Swish vs CReLU in the [0,1] range?
x = np.linspace(0, 1, 100)
crelu = np.clip(x, 0, 1)
swish = x / (1 + np.exp(-x))
diff = np.abs(crelu - swish)
print(f"\n  CReLU vs Swish in [0,1]:")
print(f"    Max absolute difference: {diff.max():.4f}")
print(f"    Mean absolute difference: {diff.mean():.4f}")
print(f"    Note: Swish(0)=0, Swish(1)≈0.731 (vs CReLU(1)=1.0)")
print(f"    The effective output range of Swish in [0,1] input is [0, 0.731]")
print(f"    This changes the effective Q1 scaling! Training must be aware of this.")

# LUT lookup speed
@numba.njit(cache=True)
def apply_crelu(acc_vals, q1):
    out = np.empty(512, dtype=numba.int32)
    for i in range(512):
        out[i] = max(0, min(q1, acc_vals[i]))
    return out

@numba.njit(cache=True)
def apply_lut(acc_vals, lut):
    out = np.empty(512, dtype=numba.int16)
    for i in range(512):
        v = acc_vals[i]
        if v <= 0:
            out[i] = 0
        elif v >= 32767:
            out[i] = lut[32767]
        else:
            out[i] = lut[v]
    return out

acc_test = rng.integers(-200, 1000, size=512, dtype=np.int32)

# Warm up
apply_crelu(acc_test, 255)
apply_lut(acc_test, lut_swish)

N3 = 100000
t0 = time.perf_counter()
for _ in range(N3):
    apply_crelu(acc_test, 255)
t_crelu = time.perf_counter() - t0

t0 = time.perf_counter()
for _ in range(N3):
    apply_lut(acc_test, lut_swish)
t_lut = time.perf_counter() - t0

print(f"\n  CReLU vs LUT-Swish speed ({N3} iters, 512 neurons):")
print(f"    CReLU:     {t_crelu*1e6/N3:.3f} µs/iter")
print(f"    LUT-Swish: {t_lut*1e6/N3:.3f} µs/iter")
speedup3 = t_crelu / t_lut
print(f"    Speedup: {speedup3:.3f}x  ← {'LUT faster' if speedup3 > 1 else 'CReLU faster or equal'}")

# ─── SUMMARY ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Claim 1 (INT16 acc): {speedup:.2f}x speedup | Memory ↓50% | Overflow-safe={theoretical_max < max_acc_i16}")
print(f"  Claim 2 (INT16 FC2): {speedup2:.2f}x speedup")
print(f"  Claim 3 (LUT Swish): {speedup3:.2f}x speedup | But changes activation semantics")
