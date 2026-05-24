---
trigger: glob
description: "NNUE evaluation neural network architecture guidelines and inference specifications."
---

---

description: "NNUE evaluation neural network architecture guidelines and inference specifications."
trigger: glob
glob: "**/nnue/**/*.py"
---

# NNUE Architecture & Inference Specification

This rule defines the core architectural layout, mathematical invariants, and JIT-compiled inference rules for the **HalfKA + LayerStackNNUE** evaluation network. All modifications to the NNUE inference pipeline (`inference.py`, `evaluation.py`, `core.py`) **MUST** strictly adhere to these specifications.

---

## 1. Network Topology & Bucket Structure

The engine utilizes a custom neural network evaluation structured with **8 LayerStack Buckets** (Stockfish 18 inspired) and Structural Re-parameterization.

```
              [ 2 × 22528 Sparse HalfKAv2_hm Features ]
                                       │
                                       ▼
                  [ Shared FC1 Feature Transformer ] -> 512 dims
                                       │
                                       ▼ ClippedReLU(16129)
                        [ Concatenation: 1024 dims ]
                                       │
                                       ▼
                 [ Route to 1 of 8 FC2-FC4 LayerStack Buckets ]
                       - FC2: 1024 -> 32  (ClippedReLU)
                       - FC3: 32 -> 32    (ClippedReLU)
                       - FC4: 32 -> 1     (Linear)
                                       │
                                       ▼
                       [ Centipawn Output (cp = trunc(x / 41)) ]
```

### Bucket Selection (Piece Count)

The network routes a given position to one of eight LayerStack dense buckets based on the total active piece count:
$$\text{bucket} = \text{clamp}\left(\left\lfloor\frac{\text{piece\_count} - 1}{4}\right\rfloor, 0, 7\right)$$

* `piece_count` includes both Kings. Range: `[2, 32]`.
* Buckets divide the game phases smoothly (e.g., Bucket 0 handles micro-endgames; Bucket 7 handles complex opening/middle game structures).

---

## 2. HalfKA Feature Encoding & Mirroring (HalfKAv2_hm)

The input feature space consists of $22,528$ sparse indices per perspective ($32 \text{ King buckets} \times 11 \text{ piece channels} \times 64 \text{ target squares}$). The engine maintains two accumulators over the same table: one for White/color-0 perspective and one for Black/color-1 perspective, then concatenates them in STM/NSTM order at evaluation time.

### Interleaved Piece Channels

Each perspective maps active pieces to 11 channels:

* `0, 2, 4, 6, 8`: the perspective side's own Pawn, Knight, Bishop, Rook, Queen.
* `1, 3, 5, 7, 9`: the opponent's Pawn, Knight, Bishop, Rook, Queen.
* `10`: Both Kings (merged channel).

### Horizontal Mirroring

* Following Stockfish's HalfKAv2_hm feature order, if the oriented King file is $< 4$ (files a-d), the entire perspective (including King square and all other pieces) is horizontally mirrored (e.g., `sq ^= 7`) so the King lands on files e-h.
* This folds 64 possible King squares into 32, reducing the per-perspective feature table from 45,056 possible non-folded indices to 22,528 deployed indices.
* The king bucket order is the official vertically reversed, horizontally folded Stockfish order:
    $$\text{bucket} = (7 - (\text{king\_mapped} \gg 3)) \times 4 + (7 - (\text{king\_mapped} \ \& \ 7))$$

### Coordinate Perspectives & Formulas

1. **White / Color-0 Perspective**:
    * King Square: $\text{king\_w\_sq}$
    * If King is on files a-d: $\text{flip\_h} = \text{True}$; $\text{king\_mapped} = \text{king\_w\_sq} \oplus 7$.
    * Else: $\text{flip\_h} = \text{False}$; $\text{king\_mapped} = \text{king\_w\_sq}$.
    * $\text{bucket} = (7 - (\text{king\_mapped} \gg 3)) \times 4 + (7 - (\text{king\_mapped} \ \& \ 7))$
    * Piece channels ($\text{mt}$): own $P/N/B/R/Q \to 0,2,4,6,8$, opp $P/N/B/R/Q \to 1,3,5,7,9$, both kings $\to 10$.
    * Target square: $\text{sq\_mapped} = \text{sq} \oplus 7$ if $\text{flip\_h}$ else $\text{sq}$.
    * **Feature Index Formula**:
        $$f_w = \text{bucket} \times 704 + \text{mt} \times 64 + \text{sq\_mapped}$$

2. **Black / Color-1 Perspective**:
    * King Square: $\text{king\_b\_sq} \oplus 56$ (Vertically flipped first)
    * If flipped King is on files a-d: $\text{flip\_h} = \text{True}$; $\text{king\_mapped} = (\text{king\_b\_sq} \oplus 56) \oplus 7$.
    * Else: $\text{flip\_h} = \text{False}$; $\text{king\_mapped} = \text{king\_b\_sq} \oplus 56$.
    * $\text{bucket} = (7 - (\text{king\_mapped} \gg 3)) \times 4 + (7 - (\text{king\_mapped} \ \& \ 7))$
    * Piece channels ($\text{mt}$): opp (White) $P/N/B/R/Q \to 1,3,5,7,9$, own (Black) $P/N/B/R/Q \to 0,2,4,6,8$, both kings $\to 10$.
    * Target square: $\text{sq\_mapped} = (\text{sq} \oplus 56) \oplus 7$ if $\text{flip\_h}$ else $\text{sq} \oplus 56$.
    * **Feature Index Formula**:
        $$f_b = \text{bucket} \times 704 + \text{mt} \times 64 + \text{sq\_mapped}$$

---

## 3. Accumulator Invariants & Stack Updates

To avoid redundant evaluation passes over the deep search tree, the first network layer (FC1) is incrementally maintained in a JIT-compiled stack.

* **Stack Representation**: `accumulator_stack` must have shape `(MAX_PLY, 2, 512)` and a data type of `np.int32`.
* **Initialization (Ply = 0)**: Computes the full accumulators for both perspectives by iterating over all active pieces on the bitboard.
* **Incremental Update (Ply > 0)**:
  * **King Move Trigger**: If a King moves on a given side, its king-relative feature indices shift completely. You **must** trigger a full refresh of that side's accumulator from the bitboard.
  * **Non-King Moves**: Accumulators are updated incrementally using the diff vectors of added and removed feature indices:
        $$\text{acc}[ply] = \text{acc}[ply - 1] + \sum \text{FC1\_WEIGHT}[added] - \sum \text{FC1\_WEIGHT}[removed]$$
* **Special Moves**: Generate diffs from the complete old/new piece bitboards. Castling must include both King and Rook coordinate changes. En-passant captures must remove the captured pawn from its original captured square, not from the moving pawn's destination square.

---

## 4. Quantized Forward Pass Arithmetic

The JIT inference engine is fully integer-based, avoiding slow floating-point operations.

### Scale Parameters & Quantization Constants

* **Layer 1 (Sparse Embedding) Scale ($Q_1$)**: $127$ (aligns with $127.0$ ClippedReLU limit).
* **Dense Layer Scale ($Q_{\text{hidden}}$)**: $128$ (enables division via bit-shifting `>> 7`).
* **Integer ClippedReLU Activation Threshold ($T_{\text{clip}}$)**:
    $$T_{\text{clip}} = 127 \times 127 = 16,129$$

### Inference Operations (Layer-by-Layer)

1. **Layer 1 Output (Clipped Activation)**:
    For $i \in [0, 511]$:
    $$\text{layer1}[i] = \max(0, \min(16129, \text{accumulator}[ply, \text{stm}, i]))$$
    $$\text{layer1}[512 + i] = \max(0, \min(16129, \text{accumulator}[ply, \text{nstm}, i]))$$

2. **Layer 2 Output (FC2 with Right-Shift)**:
    $$\text{acc}_2[i] = \text{FC2\_BIAS}[\text{bucket}, i] + \sum_{j=0}^{1023} \text{FC2\_WEIGHT}[\text{bucket}, i, j] \times \text{layer1}[j]$$
    $$\text{layer2}[i] = \max\left(0, \min\left(16129, \text{acc}_2[i] \gg 7\right)\right)$$

3. **Layer 3 Output (FC3 with Right-Shift)**:
    $$\text{acc}_3[i] = \text{FC3\_BIAS}[\text{bucket}, i] + \sum_{j=0}^{31} \text{FC3\_WEIGHT}[\text{bucket}, i, j] \times \text{layer2}[j]$$
    $$\text{layer3}[i] = \max\left(0, \min\left(16129, \text{acc}_3[i] \gg 7\right)\right)$$

4. **Layer 4 Output (FC4 Linear Logit)**:
    $$\text{output} = \text{FC4\_BIAS}[\text{bucket}, 0] + \sum_{i=0}^{31} \text{FC4\_WEIGHT}[\text{bucket}, 0, i] \times \text{layer3}[i]$$

5. **Final Score Scaling**:
    Convert the quantized raw logit output to Centipawns (cp) using integer, sign-aware truncation toward zero (not floor division, which behaves incorrectly for negative numbers):
    $$\text{cp} = \operatorname{trunc}\left(\frac{\text{output}}{41}\right)$$
    In Python JIT code, this is implemented as `output // 41` for `output >= 0` and `-((-output) // 41)` for `output < 0`. Here `output` is still scaled by $Q_1 \times Q_{\text{hidden}} = 16,256$. With the training target transform $\sigma(0.0025 \times \text{cp})$, the mathematically aligned divisor is $16,256 \times 0.0025 \approx 40.64$, rounded to `41` for integer inference.

---

## 5. Weight Directory Layout & Types

All compiled models must export static binary files inside `weights/` matching the following properties:

* **`fc1_weight.npy`**: Shape `(22528, 512)`, dtype `int16`, row-major contiguous memory for SIMD cache-friendliness.
* **`fc1_bias.npy`**: Shape `(512,)`, dtype `int32`.
* **`fc2_weight_b{b}.npy`** ($b \in [0, 7]$): Shape `(32, 1024)`, dtype `int16`.
* **`fc2_bias_b{b}.npy`** ($b \in [0, 7]$): Shape `(32,)`, dtype `int32`.
* **`fc3_weight_b{b}.npy`** ($b \in [0, 7]$): Shape `(32, 32)`, dtype `int16`.
* **`fc3_bias_b{b}.npy`** ($b \in [0, 7]$): Shape `(32,)`, dtype `int32`.
* **`fc4_weight_b{b}.npy`** ($b \in [0, 7]$): Shape `(1, 32)`, dtype `int16`.
* **`fc4_bias_b{b}.npy`** ($b \in [0, 7]$): Shape `(1,)`, dtype `int32`.
