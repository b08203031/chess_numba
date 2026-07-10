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
  * Search / modern engine ideas → `stockfish_repo/` (SF 18)
  * Hand-crafted eval (HCE) → `stockfish_11/` (SF 11)

---

## 2. Post-task

* Major features, refactors, or invariant fixes → **MUST** update the matching tech doc under `chess_engine/classical/` or `chess_engine/nnue/` (or create one if none exists).
* Any add / delete / rename of a tech markdown file → **MUST** update the Documentation Map in root `README.md`.
* **MUST NOT** dump session notes or one-off plans in the repo root; put archaeology under `archive/` (not in the main index).
* `classical_old/` is a **.py-only** battle baseline via `tools/sync_classical_old.py` — **no markdown maintenance** there.

---

## 3. Testing policy

| Allowed without asking | Requires explicit user approval |
| :--- | :--- |
| Light unit tests (`unittest`) with little/no heavy JIT | Elo tournaments, long benchmarks, deep perft mass runs, SPSA games |

* First JIT compile of search/eval can take **several minutes** (>5). Wait for completion; **MUST NOT** spam restarts or re-launch the same compile.
* When board/make/unmake/movegen change: **recommend** the user run perft (see `board-integrity` rule) rather than auto-running heavy perft.

---

## 4. Cross-tool note

`.agents/rules/*.md` frontmatter (`trigger` / `glob`) is for agents that support modular rules. Tools that only load `AGENTS.md` **MUST** still open the rule file listed for the files being edited.
