# Match SPSA for HCE eval weights

In-process, **single-process / single JIT** self-play SPSA for classical HCE
parameters that live on `SearchContext.eval_weights`.

## Why not UCI subprocess?

Numba freezes module-level globals. Restarting a process re-JITs (minutes).
Mutating a **StructRef field array** is visible inside nopython, so we keep
long-lived `SearchContext`s and only rewrite `eval_weights` between games.

## Parameter blocks (`--block`) & Roadmap

| `--block` | Dims | Content | Status |
| :--- | ---: | :--- | :--- |
| `g4` | - | Tempo / safe-check / king-danger linear | **Done** → Synced to `constants.py` |
| `a` (default) | 12 | King-safety neighbours (flank / shelter / OUT_MG / …) | **Done** (iter 74, score 0.534) → Synced |
| `passed` | 10 | Passed path/dynamics/king-prox/file scalars | **Done** (no L2 accept → defaults kept) |
| `passed_rank` | 8 | Passed ranks 3–6 MG+EG | **Done** (defaults kept) |
| `material` | 10 | N/B/R/Q MG+EG (pawn frozen) + 2 imbalance scales | **Available** |

Pipeline order: **G4 → A → Passed → Passed Rank → Material**.

## Protocol Defaults

| Knob | Default |
| :--- | :--- |
| L1 | 250 games @ **n20k**, `-c 10`, TT 4MB (fast SPSA gradient games) |
| L2 | Every **25** iters, 250 @ **n200k**; accept score ≥ 0.52 (deeper validation) |
| L2 skip | If `quantize(θ) == quantize(best)` → another 25 L1 |
| L2 accept | `best` = L2 ints; working θ **keeps float** (no re-center) |
| Stop | 2 consecutive L2 rejects |

Search flags stay frozen.

## Usage

```bash
# Example: Material block run
python -u tune_eval_match/spsa_match.py --block material \
  --out-dir tournament_analysis/spsa_eval_material \
  -c 10 --games 250 --nodes 20000 --l2-every 25 \
  --l2-games 250 --l2-nodes 200000 --l2-stop-after-fails 2 \
  --no-phase2-after-l2 --no-verify

# Smoke test
python -u tune_eval_match/spsa_match.py --block material --smoke

# Autonomous pipeline runner:
python -u tune_eval_match/pipeline_after_a.py --poll 180
```

## Post-Block Synchronization

When an L2 stage accepts a new parameter set:
1. Copy `best_params.json` (`theta_int`) into `chess_engine/classical/constants.py`
2. Sync baseline: `python tools/sync_classical_old.py --sync`
3. Verify: `python tools/sync_classical_old.py --diff` (must be identical)
4. Start next block with fresh `--out-dir` (defaults load new constants via `default_eval_weights`)

## Wiring

- All tunable ints live in `SearchContext.eval_weights` (`eval_weights.py` indices).
- Evaluation (`evaluation.py`), pawns (`pawns.py`), and material (`material.py`) read via `_ew_get` when `search_context` is set.
- Search still uses module `MG_MATERIAL_VALUES` for promotion SEE etc. (search frozen).

## Artifacts

Each run outputs to `--out-dir`:
- `spsa_log.jsonl`: per-iteration step records and LLR.
- `checkpoint.json`: current running float parameters.
- `best_params.json`: accepted integer configuration and score.
- `run_meta.json`: metadata, seeds, and runtime settings.
