---
description: "Texel/SPSA tuner invariants: per-dataset K* calibration (no K=1.000), dev/holdout K* decoupling under SF labels, sub-bucket guards, and notebook sync."
trigger: glob
glob: "**/tuner/**/*.{py,ipynb}"
---

# Texel / SPSA Tuner Constraints

Manual: [`tuner/README.md`](../../tuner/README.md).  
Verification suite: `tests/test_texel_parameterization.py`.

---

## 1. Calibration & K* Isolation (MUST)

* **No raw `K=1.000` reporting or gating**: `run_paired_validation` and CLI `--k-factor` default to `"auto"`. Validation splits (`holdout`, `val_pgn`, `val_zurichess`) must evaluate at their own baseline-calibrated `K*` (`fit_k=False` with cached/pre-fitted `K*`). Do **not** restore `mse_k1`, `delta@K=1`, or default `k_factor=1.0` gates.
* **Decouple mixed-label `dev_K*` from pure-WDL `K*`**: When `USE_SF_LABELS = True`, SPSA, `scan_grid`, and `dev` use `_step_dev_k_star`, while unseen pure-WDL gates (`holdout`, `val_pgn`, `val_zurichess`) must each use their own baseline-fitted `K*` (`_step_holdout_k_star`, `_step_pgn_k_star`, `_step_zurichess_k_star`; `_initial_*_k_star` in `cumulative_report`). Never leak holdout scores/K* into `dev`, and never score pure-WDL holdout at mixed-label `dev_K*`.

---

## 2. Gate & Sub-bucket Safety (MUST)

* Maintain noise-floored guards in `TexelTuningSession._gate_summary` against localized regressions masked by overall averages:
  * Holdout: `pure king+pawns` (`z > 3.5` & `delta > 2e-5` -> `REJECT`), `phase=0` (`z > 3.5` & `delta > 1e-5` -> `REJECT`).
  * Independent (`val_pgn` / `val_zurichess`): `pure king+pawns` and extreme `castling!=0` collapses (`z > 4.0` & `delta > 2e-5` -> `REJECT`), keeping milder caution bounds.

---

## 3. Notebook (`texel_tuner_runner.ipynb`) Parity & Portability (MUST)

* When editing constants or SPSA hyperparameters (`alpha`, `c`, `A`, `iters`, `batch_size`, `patience`): keep `texel_tuner_runner.ipynb` Grid bounds centered on live `chess_engine/classical/constants.py`, keep Markdown descriptions synced with cell code, and prevent duplicate block cells.
* Runnable post-tuning/tournament/report cells must auto-resolve the repo root (`chess_engine` + `tools` / parent dir / `/content/chess_numba`) rather than relying on bare Colab-only `%cd /content/chess_numba`.

---

## 4. Required Verification

```bash
python -m unittest tests/test_texel_parameterization.py -v
```
