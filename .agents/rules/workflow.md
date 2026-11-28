---
description: "Pre/post task docs workflow, Stockfish refs, light vs heavy tests, JIT compile patience."
trigger: always_on
---

# Workflow Guidelines

Apply on every development, refactor, debug, or documentation task.

---

## 1. Pre-task

* **MUST** open root `README.md` → section **專案文件導覽 (Documentation Map)** and load any design/analysis doc for the module you will touch.
* **MUST NOT** invent parallel designs that contradict indexed docs (e.g. HCE audit, SEARCH_ANALYSIS, NNUE ARCHITECTURE).
* Align algorithms with reference trees (repo-relative paths only):
  * Search / modern engine ideas → `stockfish_repo/` (SF 19)
  * Hand-crafted eval (HCE) → `stockfish_11/` (SF 11)

---

## 2. Post-task

* Major features, refactors, or invariant fixes → **MUST** update the matching tech doc under `chess_engine/classical/` or `chess_engine/nnue/` (or create one if none exists).
* Any add / delete / rename of a tech markdown file → **MUST** update the Documentation Map in root `README.md`.
* **MUST NOT** dump session notes or one-off plans in the repo root; put archaeology under `archive/` (not in the main index).
* **Battle baseline protection (`classical_old/`) (MUST NOT)**:
  * `classical_old/` is the frozen Elo tournament baseline.
  * **MUST NOT** run `python tools/sync_classical_old.py --sync` or `--sync-code` without explicit tournament/match verification (對戰驗證) and user approval.
  * During development, only read-only diffs (`--diff` / `--diff-code`) are permitted to observe divergence.
  * `classical_old/` is a **.py-only** battle baseline — **no markdown maintenance** there.
* **Diff sanity check (MUST)**: Run `git diff --stat` before finishing. If a small logic fix results in hundreds or thousands of unexpected line insertions/deletions, inspect immediately for newline doubling or line-ending corruption (`\r\r\n`).
* **Selective Git staging around active notebooks (MUST)**: Active/dirty Colab notebooks (`tournament_colab_runner_se_multicut.ipynb`, `tuner/texel_tuner_runner.ipynb`, `copy*.ipynb`) frequently reside in the workspace. Never use `git add -A` or `git add .` — stage code, test, and doc paths explicitly so multi-thousand-line notebook outputs are not committed unintentionally.

---

## 3. Testing policy

| Allowed without asking | Requires explicit user approval |
| :--- | :--- |
| Light unit tests (`unittest`) with little/no heavy JIT; standard perft verification (incl. Startpos & Kiwipete depth 5) | Elo tournaments, long benchmarks, ultra-deep perft mass runs (depth ≥ 6), SPSA games |

* First JIT compile of search/eval can take **several minutes** (order-of-magnitude: cold ~2–3+ min after 2026-07 cache work; still long). Wait for completion; **MUST NOT** spam restarts or re-launch the same compile. Details and re-measure protocol: `chess_engine/classical/JIT_COMPILE_CACHE_REPORT.md`.
* **Batching & environment isolation for `pytest`**:
  * Run related engine/contract/tuner suites in **one** `pytest` process so `cache=False` search functions compile at most once (~5–6 min) instead of once per file.
  * Keep Tkinter/GUI tests (`tests/test_vision_and_assistant.py` — fails with `_tkinter.TclError` in headless or Tk-less shells) and cache-wiping tests (`tests/test_search_param_sweep.py`) out of batch engine runs and report any excluded/env-specific suites explicitly.
* **Small-gain SPRT evaluation**: When reviewing a small-magnitude fix against the default `SPRT[0, +10]` runner, compute `[-5, 0]` (non-regression) and `[-3, +3]` LLR alongside 95% Elo CI via `tools.match_core.SPRTTest` before proposing `--sync` acceptance.
* When board/make/unmake/movegen change: **MUST** run perft verification directly (see `board-integrity` rule), including **Kiwipete depth 5**.

---

## 4. Cross-tool note

`.agents/rules/*.md` frontmatter (`trigger` / `glob`) is for agents that support modular rules. Tools that only load `AGENTS.md` **MUST** still open the rule file listed for the files being edited.

---

## 5. Line-Ending & Formatting Hygiene (MUST)

* **No spurious blank lines**: Follow PEP 8 spacing strictly. Maximum 2 blank lines between top-level definitions, 1 blank line between methods or logical sections. **MUST NOT** have blank lines between decorators and `def`, inside single expressions, or between consecutive constant declarations.
* **Prevent `\r\r\n` (CRCRLF) on Windows**:
  * When writing Python files via scripts or tools, always normalize line endings to `\n` (`open(..., 'w', encoding='utf-8', newline='\n')` or write normalized bytes). Let Git's `core.autocrlf` manage workspace checkouts cleanly.
  * Verify clean file writes: `\r\r\n` must never be introduced into committed or modified files.
* **Sidecar module synchronization**:
  * When `chess_engine/classical/constants.py` changes: run `python -m tuner.sync_constants_snapshot` to regenerate tuner snapshot and clear stale Numba cache.
  * When `chess_engine/classical/{evaluation,pawns,material}.py` changes: run `python -m tuner.codegen.generate_tunable_eval` to regenerate `tuner/tunable_eval.py` and verify with `tests/test_tunable_eval_parity.py`.

---

## 6. Open / Running Notebook (`.ipynb`) Safety (MUST)

* **Do NOT silently overwrite an active/running `.ipynb`**: Before editing a `.ipynb` on disk, check whether it is the user's open `Active Document` or executing a long run (tuning/tournament). Finish `.py`/`.md`/test edits first and defer `.ipynb` writes until the user saves/closes or confirms, avoiding IDE `[Overwrite] / [Revert]` save-conflict races.
* **Lossless JSON & syntax check**: Match existing `.ipynb` JSON formatting (`indent=1, ensure_ascii=False`, trailing `\n`, `LF` without `\r\r\n`), verify round-trip preservation of untouched cell `outputs`/`execution_count`, and `ast.parse` all code cells after patching.
