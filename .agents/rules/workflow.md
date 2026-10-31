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

---

## 3. Testing policy

| Allowed without asking | Requires explicit user approval |
| :--- | :--- |
| Light unit tests (`unittest`) with little/no heavy JIT | Elo tournaments, long benchmarks, deep perft mass runs, SPSA games |

* First JIT compile of search/eval can take **several minutes** (order-of-magnitude: cold ~2–3+ min after 2026-07 cache work; still long). Wait for completion; **MUST NOT** spam restarts or re-launch the same compile. Details and re-measure protocol: `chess_engine/classical/JIT_COMPILE_CACHE_REPORT.md`.
* When board/make/unmake/movegen change: **recommend** the user run perft (see `board-integrity` rule) rather than auto-running heavy perft.

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
