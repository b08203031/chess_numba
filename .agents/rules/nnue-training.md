---
description: "NNUE train/export MUST: factorizer fusion, loss, LR groups, sampling, clamps."
trigger: glob
glob: "**/chess_engine/nnue/ml_eval/**/*.py"
---

# NNUE Training Pipeline — Hard Constraints

Ops manual: **`chess_engine/nnue/ml_eval/ml_eval_documentation_and_architecture.md`**  
Data pipeline: **`tuner/README.md`** / `tuner/nnue_pipeline/`  
Architecture: **`chess_engine/nnue/ARCHITECTURE.md`**

---

## 1. Structural re-parameterization (MUST)

* Train with **global factorizer** + **per-bucket deltas** (\(\Delta W, \Delta b\)), \(b \in [0,7]\)
* Deltas **initialize to 0**
* Forward: \(W_{\mathrm{eff}} = W_{\mathrm{fact}} + \Delta W_b\) (same for bias) **before** activation
* Export **fuses** factorizer + delta into plain `fc*_b{b}.npy` — JIT inference has **no** factorizer

---

## 2. Loss & dataset (MUST)

* Targets WDL in \([0,1]\); clamp to **`[0.001, 0.999]`** before BCE
* Criterion: `BCEWithLogitsLoss` on raw logits
* NPZ: `features_stm` / `features_nstm` uint16 sparse indices, pad **22528**; `targets` float32; `piece_counts` for buckets
* Primer `Value` → UCI cp: `cp = value * 100 / 208`, then \(\sigma(0.0025 \cdot cp)\); inference divisor **41** is tied to this — change only as a coordinated set

---

## 3. Optimizers (MUST)

| Group | Weight decay | LR (typical) |
| :--- | ---: | :--- |
| FC1 sparse | **0** | lower (~4e-5 … 1e-4) |
| Dense (factorizer + deltas) | **0.001** | higher (~5e-4 … 1e-3) |

* **MUST NOT** apply weight decay on FC1 (kills rare features)
* AMP + `GradScaler` on CUDA; `clip_grad_norm_(..., max_norm=20.0)` for large batches

---

## 4. Sampling & physical clamps (MUST)

* Bucket balance: weights \(w_i = 1 / p_i^{\tau}\) with project \(\tau = 0.25\) (or document a deliberate change)
* After each step: clamp FC1 weights in `no_grad` to physical INT16-safe bounds (see train script / ARCHITECTURE; ~±4.0 scale convention)

---

## 5. After changes

```bash
python -m unittest tests.test_nnue_validation
# Full train only with user OK and GPU ready
```
