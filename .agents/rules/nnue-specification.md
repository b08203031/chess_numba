---
description: "NNUE topology, HalfKA, accumulator, quantized forward MUST invariants."
trigger: glob
glob: "**/chess_engine/nnue/**/*.py"
---

# NNUE Inference — Hard Constraints

Full math & feature formulas: **`chess_engine/nnue/ARCHITECTURE.md`**.  
Training/export: **`.agents/rules/nnue-training.md`** + `ml_eval` docs.

When editing `nnue/**` (including `ml_eval/inference.py`), **MUST** preserve:

---

## 1. Topology (MUST NOT silently change)

```
HalfKAv2_hm sparse 22528 × 2 perspectives
  → shared FC1 accumulator 512
  → ClippedReLU → concat 1024
  → 8 LayerStack buckets × (FC2 1024→32 → FC3 32→32 → FC4 32→1)
  → cp = trunc_toward_zero(output / 41)
```

* Bucket: `clamp((piece_count - 1) // 4, 0, 7)` with `piece_count` including both kings, range `[2, 32]`
* Changing width/bucket count/divisor requires coordinated retrain + export + doc update

---

## 2. Features (MUST)

* Per perspective: **22528** indices (32 king buckets × 11 channels × 64 squares), HalfKAv2_hm horizontal fold
* Channels: own P/N/B/R/Q on even indices 0..8; opp on 1..9; both kings channel **10**
* Black perspective: vertical flip `sq ^ 56` before file fold rules (see ARCHITECTURE)
* Index: `bucket * 704 + mt * 64 + sq_mapped`

---

## 3. Accumulator (MUST)

* Stack shape `(MAX_PLY, 2, 512)`, `int32`
* Non-king moves: incremental add/remove FC1 rows
* **King move** (that side): full refresh of that perspective
* Castling: king + rook diffs; EP: remove captured pawn on **from-file** capture square, not only `to`

---

## 4. Quantized forward (MUST)

| Symbol | Value |
| :--- | ---: |
| \(Q_1\) | 127 |
| \(Q_{\mathrm{hidden}}\) | 128 |
| Clip threshold | 16129 (= 127×127) |
| Hidden shift | `>> 7` after FC2/FC3 matmul |
| Output → cp | trunc toward zero by **41** (not floor for negatives) |

* Inference is **integer**; no FP on the JIT hot path
* Export layout under `ml_eval/weights/`: `fc1_*`, `fc{2,3,4}_*_{b0..b7}.npy` with dtypes/shapes in ARCHITECTURE

---

## 5. After changes

```bash
python -m unittest tests.test_nnue_validation
# Optional: python chess_engine/nnue/ml_eval/inference.py
```
