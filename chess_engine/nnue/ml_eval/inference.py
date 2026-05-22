import os
import sys
import numpy as np
import numba

# Allow running as a script from within the ml_eval folder
if __name__ == "__main__":
    # Add chess_engine_v2 to path so chess_engine can be found
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from chess_engine.nnue.constants import MAX_PLY
from chess_engine.nnue.bitboard_utils import get_lsb_index  # O(1) lookup

# =============================================================================
# HalfKAv2_hm Inference Engine (Stockfish-style)
#
# Feature encoding: interleaved channels + horizontal mirror
# Feature space: 22528 = 32 king buckets × 11 channels × 64 squares
#
# Channel mapping (from each perspective):
#   Own P/N/B/R/Q → even channels 0,2,4,6,8
#   Opp P/N/B/R/Q → odd channels  1,3,5,7,9
#   Both Kings    → merged channel 10
#
# Bucket = (king_mapped >> 3) * 4 + (king_mapped & 7)
#   king_mapped = king_sq ^ 7 if king is on files e-h, else king_sq
#   (For Black perspective: king_sq is vertically flipped first)
# =============================================================================

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), 'weights')

NUM_BUCKETS = 8       # SF18-style LayerStacks
NUM_FEATURES = 22528  # 32 king buckets × 704 (11 channels × 64 squares)
PADDING_INDEX = 22528 # First index beyond valid feature range

# Shared FC1 (HalfKA accumulator)
FC1_WEIGHT = np.zeros((NUM_FEATURES, 512), dtype=np.int16)
FC1_BIAS   = np.zeros((512,),              dtype=np.int32)

# 8x LayerStack weights: FC2 (32x1024), FC3 (32x32), FC4 (1x32)
FC2_WEIGHTS     = np.zeros((NUM_BUCKETS, 32, 1024), dtype=np.int16)
FC2_BIASES      = np.zeros((NUM_BUCKETS, 32),       dtype=np.int32)
FC3_WEIGHTS     = np.zeros((NUM_BUCKETS, 32, 32),   dtype=np.int16)
FC3_BIASES      = np.zeros((NUM_BUCKETS, 32),       dtype=np.int32)
FC4_WEIGHTS     = np.zeros((NUM_BUCKETS,  1, 32),   dtype=np.int16)
FC4_BIASES      = np.zeros((NUM_BUCKETS,  1),       dtype=np.int32)
# Pre-cast INT32 versions for SIMD-friendly Numba forward pass
FC2_WEIGHTS_I32 = np.zeros((NUM_BUCKETS, 32, 1024), dtype=np.int32)
FC3_WEIGHTS_I32 = np.zeros((NUM_BUCKETS, 32, 32),   dtype=np.int32)
FC4_WEIGHTS_I32 = np.zeros((NUM_BUCKETS,  1, 32),   dtype=np.int32)

def load_weights():
    """Loads FC1 (shared) + 8x LayerStack weights from disk."""
    global FC1_WEIGHT, FC1_BIAS
    global FC2_WEIGHTS, FC2_BIASES, FC3_WEIGHTS, FC3_BIASES, FC4_WEIGHTS, FC4_BIASES
    global FC2_WEIGHTS_I32, FC3_WEIGHTS_I32, FC4_WEIGHTS_I32
    try:
        if not os.path.exists(WEIGHTS_DIR):
            print("Weights directory not found. Please run ml_eval/train.py first.")
            return False

        FC1_WEIGHT = np.ascontiguousarray(
            np.load(os.path.join(WEIGHTS_DIR, 'fc1_weight.npy')).astype(np.int16)
        )
        FC1_BIAS   = np.load(os.path.join(WEIGHTS_DIR, 'fc1_bias.npy')).astype(np.int32)

        # Validate FC1 shape matches the HalfKAv2_hm feature space
        if FC1_WEIGHT.shape[0] != NUM_FEATURES:
            print(f"WARNING: FC1 weight shape {FC1_WEIGHT.shape} does not match "
                  f"expected ({NUM_FEATURES}, 512). Weights may use an older encoding.")
            return False

        for b in range(NUM_BUCKETS):
            FC2_WEIGHTS[b] = np.load(os.path.join(WEIGHTS_DIR, f'fc2_weight_b{b}.npy')).astype(np.int16)
            FC2_BIASES[b]  = np.load(os.path.join(WEIGHTS_DIR, f'fc2_bias_b{b}.npy')).astype(np.int32)
            FC3_WEIGHTS[b] = np.load(os.path.join(WEIGHTS_DIR, f'fc3_weight_b{b}.npy')).astype(np.int16)
            FC3_BIASES[b]  = np.load(os.path.join(WEIGHTS_DIR, f'fc3_bias_b{b}.npy')).astype(np.int32)
            FC4_WEIGHTS[b] = np.load(os.path.join(WEIGHTS_DIR, f'fc4_weight_b{b}.npy')).astype(np.int16)
            FC4_BIASES[b]  = np.load(os.path.join(WEIGHTS_DIR, f'fc4_bias_b{b}.npy')).astype(np.int32)

        # Pre-cast once at load time (eliminates per-iteration casts in Numba hot loop)
        FC2_WEIGHTS_I32 = FC2_WEIGHTS.astype(np.int32)
        FC3_WEIGHTS_I32 = FC3_WEIGHTS.astype(np.int32)
        FC4_WEIGHTS_I32 = FC4_WEIGHTS.astype(np.int32)

        print(f"Loaded LayerStack NNUE weights ({NUM_BUCKETS} buckets, "
              f"{NUM_FEATURES} features, FC2=32x1024, FC3=32x32, FC4=1x32).")
        return True
    except Exception as e:
        print(f"Failed to load neural network weights: {e}")
        return False

# Initialize weights on import
weights_loaded = load_weights()

# =============================================================================
# HalfKAv2_hm Accumulator Functions
# =============================================================================

@numba.njit(fastmath=True, cache=True)
def init_accumulator(piece_bbs, accumulator_stack):
    """
    Initializes the accumulator stack at ply 0 from scratch using the full bitboards.
    Uses HalfKAv2_hm encoding: interleaved channels + horizontal mirror.
    Computes both White and Black perspectives. Accumulator entries are INT32.
    """
    accumulator_stack[0, 0, :] = FC1_BIAS[:]
    accumulator_stack[0, 1, :] = FC1_BIAS[:]

    # Extract king squares
    king_w_sq = 0
    if piece_bbs[5]:
        king_w_sq = get_lsb_index(piece_bbs[5] & (~piece_bbs[5] + np.uint64(1)))

    king_b_sq = 0
    if piece_bbs[11]:
        king_b_sq = get_lsb_index(piece_bbs[11] & (~piece_bbs[11] + np.uint64(1)))

    # --- White Perspective: own=White(0-5), opp=Black(6-11) ---
    flip_h_w = (king_w_sq & 7) >= 4
    king_w_m = king_w_sq ^ 7 if flip_h_w else king_w_sq
    bucket_w = (king_w_m >> 3) * 4 + (king_w_m & 7)

    for p_idx in range(12):
        if p_idx < 5:
            mt_w = p_idx * 2              # own P=0, N=2, B=4, R=6, Q=8
        elif p_idx == 5:
            mt_w = 10                     # own king
        elif p_idx < 11:
            mt_w = (p_idx - 6) * 2 + 1    # opp P=1, N=3, B=5, R=7, Q=9
        else:
            mt_w = 10                     # opp king

        bb = piece_bbs[p_idx]
        while bb:
            lsb = bb & (~bb + np.uint64(1))
            sq = get_lsb_index(lsb)
            sq_w = sq ^ 7 if flip_h_w else sq
            f_w = bucket_w * 704 + mt_w * 64 + sq_w
            accumulator_stack[0, 0, :] += FC1_WEIGHT[f_w, :]
            bb &= bb - np.uint64(1)

    # --- Black Perspective: own=Black(6-11), opp=White(0-5) ---
    king_b_flip = king_b_sq ^ 56  # vertical flip
    flip_h_b = (king_b_flip & 7) >= 4
    king_b_m = king_b_flip ^ 7 if flip_h_b else king_b_flip
    bucket_b = (king_b_m >> 3) * 4 + (king_b_m & 7)

    for p_idx in range(12):
        if p_idx < 5:
            mt_b = p_idx * 2 + 1          # opp P=1, N=3, B=5, R=7, Q=9
        elif p_idx == 5:
            mt_b = 10                     # opp king
        elif p_idx < 11:
            mt_b = (p_idx - 6) * 2        # own P=0, N=2, B=4, R=6, Q=8
        else:
            mt_b = 10                     # own king

        bb = piece_bbs[p_idx]
        while bb:
            lsb = bb & (~bb + np.uint64(1))
            sq = get_lsb_index(lsb)
            sq_b = sq ^ 56
            if flip_h_b:
                sq_b ^= 7
            f_b = bucket_b * 704 + mt_b * 64 + sq_b
            accumulator_stack[0, 1, :] += FC1_WEIGHT[f_b, :]
            bb &= bb - np.uint64(1)

@numba.njit(fastmath=True, cache=True)
def update_accumulator(piece_bbs, added_features, removed_features, ply, accumulator_stack):
    """
    Incrementally updates both White and Black accumulators from ply-1 to ply.
    If a king moves, fires a full refresh for its side.
    added_features/removed_features contain raw (piece_idx * 64 + sq) values, padded with -1.
    """
    white_king_moved = False
    black_king_moved = False

    for f in added_features:
        if f != -1:
            p_idx = f // 64
            if p_idx == 5: white_king_moved = True
            elif p_idx == 11: black_king_moved = True

    # Extract king squares (post-move)
    king_w_sq = 0
    if piece_bbs[5]:
        king_w_sq = get_lsb_index(piece_bbs[5] & (~piece_bbs[5] + np.uint64(1)))

    king_b_sq = 0
    if piece_bbs[11]:
        king_b_sq = get_lsb_index(piece_bbs[11] & (~piece_bbs[11] + np.uint64(1)))

    # White perspective setup
    flip_h_w = (king_w_sq & 7) >= 4
    king_w_m = king_w_sq ^ 7 if flip_h_w else king_w_sq
    bucket_w = (king_w_m >> 3) * 4 + (king_w_m & 7)

    # Black perspective setup
    king_b_flip = king_b_sq ^ 56
    flip_h_b = (king_b_flip & 7) >= 4
    king_b_m = king_b_flip ^ 7 if flip_h_b else king_b_flip
    bucket_b = (king_b_m >> 3) * 4 + (king_b_m & 7)

    # --- White Accumulator Update ---
    if white_king_moved:
        # Full refresh: king position changed, all feature indices shift
        accumulator_stack[ply, 0, :] = FC1_BIAS[:]
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
                lsb = bb & (~bb + np.uint64(1))
                sq = get_lsb_index(lsb)
                sq_w = sq ^ 7 if flip_h_w else sq
                f_w = bucket_w * 704 + mt_w * 64 + sq_w
                accumulator_stack[ply, 0, :] += FC1_WEIGHT[f_w, :]
                bb &= bb - np.uint64(1)
    else:
        # Incremental update
        accumulator_stack[ply, 0, :] = accumulator_stack[ply - 1, 0, :]
        for f in added_features:
            if f != -1:
                p_idx = f // 64
                sq = f % 64
                if p_idx < 5:
                    mt_w = p_idx * 2
                elif p_idx == 5:
                    mt_w = 10
                elif p_idx < 11:
                    mt_w = (p_idx - 6) * 2 + 1
                else:
                    mt_w = 10
                sq_w = sq ^ 7 if flip_h_w else sq
                f_w = bucket_w * 704 + mt_w * 64 + sq_w
                accumulator_stack[ply, 0, :] += FC1_WEIGHT[f_w, :]
        for f in removed_features:
            if f != -1:
                p_idx = f // 64
                sq = f % 64
                if p_idx < 5:
                    mt_w = p_idx * 2
                elif p_idx == 5:
                    mt_w = 10
                elif p_idx < 11:
                    mt_w = (p_idx - 6) * 2 + 1
                else:
                    mt_w = 10
                sq_w = sq ^ 7 if flip_h_w else sq
                f_w = bucket_w * 704 + mt_w * 64 + sq_w
                accumulator_stack[ply, 0, :] -= FC1_WEIGHT[f_w, :]

    # --- Black Accumulator Update ---
    if black_king_moved:
        accumulator_stack[ply, 1, :] = FC1_BIAS[:]
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
                lsb = bb & (~bb + np.uint64(1))
                sq = get_lsb_index(lsb)
                sq_b = sq ^ 56
                if flip_h_b:
                    sq_b ^= 7
                f_b = bucket_b * 704 + mt_b * 64 + sq_b
                accumulator_stack[ply, 1, :] += FC1_WEIGHT[f_b, :]
                bb &= bb - np.uint64(1)
    else:
        accumulator_stack[ply, 1, :] = accumulator_stack[ply - 1, 1, :]
        for f in added_features:
            if f != -1:
                p_idx = f // 64
                sq = f % 64
                if p_idx < 5:
                    mt_b = p_idx * 2 + 1
                elif p_idx == 5:
                    mt_b = 10
                elif p_idx < 11:
                    mt_b = (p_idx - 6) * 2
                else:
                    mt_b = 10
                sq_b = sq ^ 56
                if flip_h_b:
                    sq_b ^= 7
                f_b = bucket_b * 704 + mt_b * 64 + sq_b
                accumulator_stack[ply, 1, :] += FC1_WEIGHT[f_b, :]
        for f in removed_features:
            if f != -1:
                p_idx = f // 64
                sq = f % 64
                if p_idx < 5:
                    mt_b = p_idx * 2 + 1
                elif p_idx == 5:
                    mt_b = 10
                elif p_idx < 11:
                    mt_b = (p_idx - 6) * 2
                else:
                    mt_b = 10
                sq_b = sq ^ 56
                if flip_h_b:
                    sq_b ^= 7
                f_b = bucket_b * 704 + mt_b * 64 + sq_b
                accumulator_stack[ply, 1, :] -= FC1_WEIGHT[f_b, :]

@numba.njit(fastmath=True, cache=True)
def copy_accumulator(ply, accumulator_stack):
    accumulator_stack[ply, 0, :] = accumulator_stack[ply - 1, 0, :]
    accumulator_stack[ply, 1, :] = accumulator_stack[ply - 1, 1, :]

# =============================================================================
# Quantized Forward Pass (int64 accumulators to prevent overflow)
# =============================================================================

@numba.njit(fastmath=True, cache=True)
def nnue_forward_incremental(ply, stm, accumulator_stack, piece_count):
    nstm = stm ^ 1

    # SF18 bucket: (piece_count - 1) // 4, clamped to [0, 7]
    bucket = (piece_count - 1) // 4
    if bucket < 0: bucket = 0
    if bucket > 7: bucket = 7

    # Layer 1: ClippedReLU — shared across all buckets
    layer1 = np.empty(1024, dtype=np.int32)
    for i in range(512):
        layer1[i]       = max(0, min(16129, accumulator_stack[ply, stm,  i]))
        layer1[512 + i] = max(0, min(16129, accumulator_stack[ply, nstm, i]))

    # Layer 2: FC2[bucket] (32 outputs) — int64 accumulator to prevent overflow
    layer2 = np.empty(32, dtype=np.int32)
    w2 = FC2_WEIGHTS_I32[bucket]
    b2 = FC2_BIASES[bucket]
    for i in range(32):
        acc = np.int64(b2[i])
        w_row = w2[i]
        for j in range(1024):
            acc += np.int64(w_row[j]) * np.int64(layer1[j])
        layer2[i] = max(0, min(16129, np.int32(acc >> 7)))

    # Layer 3: FC3[bucket] (32 outputs) — int64 accumulator
    layer3 = np.empty(32, dtype=np.int32)
    w3 = FC3_WEIGHTS_I32[bucket]
    b3 = FC3_BIASES[bucket]
    for i in range(32):
        acc = np.int64(b3[i])
        w_row = w3[i]
        for j in range(32):
            acc += np.int64(w_row[j]) * np.int64(layer2[j])
        layer3[i] = max(0, min(16129, np.int32(acc >> 7)))

    # Layer 4: FC4[bucket] (scalar output) — int64 accumulator
    w4 = FC4_WEIGHTS_I32[bucket, 0]
    output = np.int64(FC4_BIASES[bucket, 0])
    for i in range(32):
        output += np.int64(w4[i]) * np.int64(layer3[i])

    # Convert to centipawns — sign-aware integer truncation (toward zero)
    OUTPUT_DIVISOR = np.int64(38)
    if output >= np.int64(0):
        return np.int32(output // OUTPUT_DIVISOR)
    return np.int32(-((-output) // OUTPUT_DIVISOR))

# =============================================================================
# Bitboard Diff (unchanged — returns raw piece_idx*64+sq descriptors)
# =============================================================================

@numba.njit(numba.types.Tuple((numba.int32[:], numba.int32[:]))(numba.types.Array(numba.uint64, 1, 'A'), numba.types.Array(numba.uint64, 1, 'A')), fastmath=True, cache=True)
def get_bb_differences(old_piece_bbs, new_piece_bbs):
    """
    Computes the added and removed feature indices by diffing two piece_bbs arrays.
    Returns (added_features, removed_features) padded with -1 up to capacity 8.
    """
    added = np.full(8, -1, dtype=np.int32)
    removed = np.full(8, -1, dtype=np.int32)
    add_idx = 0
    rem_idx = 0

    for piece_idx in range(12):
        diff = old_piece_bbs[piece_idx] ^ new_piece_bbs[piece_idx]
        offset = piece_idx * 64
        while diff:
            lsb = diff & (~diff + np.uint64(1))
            sq = np.int32(get_lsb_index(lsb))

            if old_piece_bbs[piece_idx] & lsb:
                removed[rem_idx] = offset + sq
                rem_idx += 1
            else:
                added[add_idx] = offset + sq
                add_idx += 1

            diff &= diff - np.uint64(1)

    return added, removed

# =============================================================================
# Standalone evaluation (for speed testing)
# =============================================================================

@numba.njit(numba.int32(numba.types.Array(numba.uint64, 1, 'C')), fastmath=True, cache=True)
def evaluate_position_nn(piece_bbs):
    """
    Evaluates a chess position using the neural network directly from bitboards.
    This is mainly used in speed testing.
    """
    accumulator_stack = np.zeros((1, 2, 512), dtype=np.int32)
    init_accumulator(piece_bbs, accumulator_stack)

    piece_count = 0
    for p_idx in range(12):
        bb = piece_bbs[p_idx]
        while bb:
            piece_count += 1
            bb &= bb - np.uint64(1)

    return nnue_forward_incremental(0, 0, accumulator_stack, piece_count)

if __name__ == '__main__':
    if not weights_loaded:
        print("Cannot test inference without training weights. Please run train.py first.")
    else:
        # Simple test to verify compilation and execution
        print("Testing Numba Inference Engine (HalfKAv2_hm)...")
        import time

        # Initial Position test
        test_bbs = np.zeros(12, dtype=np.uint64)
        # White
        test_bbs[0] = np.uint64(0x000000000000FF00) # Pawns
        test_bbs[1] = np.uint64(0x0000000000000042) # Knights
        test_bbs[2] = np.uint64(0x0000000000000024) # Bishops
        test_bbs[3] = np.uint64(0x0000000000000081) # Rooks
        test_bbs[4] = np.uint64(0x0000000000000008) # Queen
        test_bbs[5] = np.uint64(0x0000000000000010) # King
        # Black
        test_bbs[6] = np.uint64(0x00FF000000000000) # Pawns
        test_bbs[7] = np.uint64(0x4200000000000000) # Knights
        test_bbs[8] = np.uint64(0x2400000000000000) # Bishops
        test_bbs[9] = np.uint64(0x8100000000000000) # Rooks
        test_bbs[10] = np.uint64(0x0800000000000000) # Queen
        test_bbs[11] = np.uint64(0x1000000000000000) # King

        print("Running NN eval on starting position (first run will trigger JIT compilation)...")
        start = time.perf_counter()
        score = evaluate_position_nn(test_bbs)
        end = time.perf_counter()
        print(f"Initial Pos Raw Centipawn score: {score}")
        print(f"Compilation + Execution time: {(end-start)*1000:.2f} ms")

        # Speed test
        print("Running speed test (10000 runs)...")
        start = time.perf_counter()
        for _ in range(10000):
            _ = evaluate_position_nn(test_bbs)
        end = time.perf_counter()
        print(f"10000 evals time: {(end-start)*1000:.2f} ms")
        print(f"Average eval time: {(end-start)*1000/10000:.3f} ms/eval")
        print(f"NPS upper bound equivalent: {int(10000/(end-start)):,} nodes/sec")

        print("Inference Engine is ready!")