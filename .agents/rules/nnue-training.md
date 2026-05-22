---
description: "NNUE offline training pipeline conventions, hyperparameter defaults, and model re-parameterization invariants."
trigger: glob
glob: "**/nnue/ml_eval/**/*.py"
---

# NNUE Training Pipeline Specifications

This rule outlines the training conventions, structural constraints, and hyperparameter standards for optimization of the `LayerStackNNUE` model. All modifications to the training scripts (`train.py`, `quantize_weights.py`, etc.) **MUST** comply with these requirements.

---

## 1. Structural Re-Parameterization (Factorizer & Deltas)

To prevent underfitting while training on large-scale datasets, the model implements **Structural Re-parameterization (SF18-style)** during training and collapses to pure single-bucket inference weights at export time.

*   **Global Base Manifold (Factorizer)**:
    *   Shared weights: `fact_fc2` ($1024 \to 32$), `fact_fc3` ($32 \to 32$), `fact_fc4` ($32 \to 1$).
    *   Factorizer parameters accumulate gradients from **all** data samples across all game phases.
*   **Per-Bucket Residuals ($\Delta W$, $\Delta b$)**:
    *   Specific sub-layers: `delta_fc2[b]`, `delta_fc3[b]`, `delta_fc4[b]` for $b \in [0, 7]$.
    *   **In-Place Zero Initialization**: All residual weights and biases **must** be initialized to $0.0$. This ensures the global Factorizer dominates early training, avoiding random gradients in sparsely populated buckets.
*   **Forward Flow**:
    *   Add weights and biases **prior** to non-linear activations to ensure mathematically exact fusion:
        $$W_{\text{eff}} = W_{\text{fact}} + \Delta W_b$$
        $$b_{\text{eff}} = b_{\text{fact}} + \Delta b_b$$
*   **Algebraic Export Fusion**:
    *   Exported inference files (`weights/fc2_weight_b{b}.npy`, etc.) **must** perform this sum statically. The JIT inference engine has no awareness of Factorizers or delta structures.

---

## 2. Loss Functions & Dataset Schema

*   **WDL Target Boundary**:
    *   Targets represent Win/Draw/Loss probabilities in the range $[0, 1]$.
    *   To prevent infinite gradients in Binary Cross Entropy and protect INT16 quantization limits, target probabilities must be clamped:
        $$\text{target}_{\text{clamped}} = \text{clamp}(\text{target}, 0.001, 0.999)$$
*   **Loss Criterion**:
    *   Model outputs raw logits. Use `torch.nn.BCEWithLogitsLoss()` to compute standard entropy.
*   **NPZ Dataset Layout**:
    *   `features_stm`: INT16 list of sparse HalfKA indices for STM.
    *   `features_nstm`: INT16 list of sparse HalfKA indices for NSTM.
    *   `targets`: Float32 array containing target WDL values.
    *   `piece_counts`: Int8 array containing the active piece counts (for bucket index selection).

---

## 3. Layerwise Learning Rate & Weight Decay

Due to the extreme sparsity of the first layer (FC1) compared to the dense layers (FC2-FC4), optimization parameters **must** be segregated into independent parameter groups.

*   **FC1 Parameter Group**:
    *   **Weight Decay ($WD$)**: Must be **exactly** $0.0$. Applying weight decay to the sparse embedding layer triggers "information heat-death," where rare feature weights (such as complex pawn promotion setups) are decayed to zero before receiving enough training gradients.
    *   **Learning Rate ($LR$)**: Lower than dense parameters (typically $0.00004$ or $0.0001$).
*   **Dense Layer Parameter Group (Factorizer + Deltas)**:
    *   **Weight Decay ($WD$)**: Set to $0.001$ to regularize dense mapping and prevent extreme weight bounds that would overflow INT16 quantization limits.
    *   **Learning Rate ($LR$)**: Higher than sparse parameters (typically $0.001$ or $0.0005$, decaying via `ExponentialLR` with $\gamma=0.99$).

---

## 4. Balanced Sampling via Fractional Power Smoothing

To prevent highly populated middle-game positions from overwhelming rare endgame samples, you **must** apply a weighted sampling filter on piece counts.

*   **Fractional Weight Formula**:
    $$w_i = \frac{1}{p_i^{\tau}}$$
    *   $p_i$: Empirical probability of a sample falling into bucket $i$.
    *   $\tau$: Power smoothing constant. The project standard is $\tau = 0.25$.
*   **Sampling Strategy**:
    *   Use the **NumPy Alias Method** (`np.random.default_rng().choice`) with the normalized bucket weights. This delivers $O(1)$ sampling speeds identical to uniform random indices while ensuring representation of rare endgames.

---

## 5. Numerical Safeguards (AMP & Physical Clamping)

*   **Mixed Precision (AMP)**:
    *   Always utilize `torch.amp.autocast('cuda')` and `torch.amp.GradScaler('cuda')` during CUDA execution to maintain high throughput on modern tensor-core GPUs (like RTX 4050).
*   **Gradient Clipping**:
    *   Apply `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20.0)` to stabilize training under large batch sizes (e.g., 8192).
*   **Physical Accumulator Clamping**:
    *   To prevent accumulator overflow under severe piece pile-ups, clamp FC1 weights within `torch.no_grad()` at the end of every optimization step:
        $$\text{FC1\_MAX\_WEIGHT} = \frac{127.0}{Q_1} \times 4.0 \approx 4.0$$
        $$\text{model.fc1.weight.clamp\_}(-\text{FC1\_MAX\_WEIGHT}, \text{FC1\_MAX\_WEIGHT})$$
