---
description: "Negamax symmetry, mate distance, repetition/50-move, root and in-check pruning safety."
trigger: glob
glob: "**/chess_engine/**/{search,search_heuristics,time_manager,transposition_table}.py"
---

# Search Correctness Constraints

Applies when editing search / move ordering / TT.  
Algorithm narrative: `chess_engine/classical/SEARCH_ANALYSIS.md` (and NNUE twin if relevant).

---

## 1. Negamax (MUST)

* Scores are **side-to-move relative**
* Child call: `score = -search(...)` with bounds `(-beta, -alpha)`
* No asymmetric “side bonuses” inside the recursive score path

---

## 2. Mate distance (MUST)

* Mating side: `MATE_SCORE - ply`
* Being mated: `-MATE_SCORE + ply`
* TT store/load: convert mate scores to/from path-relative form so TT entries stay valid across plies

---

## 3. Repetition & draws (MUST match engine policy — Stockfish-aligned)

* `_is_repetition_jit` scopes `ply_path_stack` and `game_history` (most recent 1024 keys) by `min(halfmove_clock, pliesFromNull)` — never scan across a null move (`NO_MOVE` in `move_stack`)
* At `ply > 0`:
  * **2-fold strictly inside the search tree** (`1 <= i <= ply - 2`, strictly after root) **or** `halfmove_clock >= 50-move limit` (guarded by `not in_check` in `_search` and `quiescence_search`): treat as **draw (0)**
  * **Crossing into root (`i == 0`) or pre-root `game_history`**: require **3-fold** (`>= 2` prior occurrences) so losing positions cannot hallucinate `0` on a single past repetition
* **MUST NOT** apply draw/repetition cutoffs at **root** (`ply == 0`)
* Near 50-move limit: scale eval toward draw (avoid cliff at the leaf)

---

## 4. Pruning safety (MUST)

* **No** heuristic prune / early fail-soft that skips full root move resolution at `ply == 0` (all legal root moves must be considered for bestmove)
* If **in check**: **forbid** NMP, RFP, and other static-eval-based prunes that skip evasion search; search check evasions full-width as designed
* Changing prune margins/constants: prefer constants in `constants.py`; keep PV / check / mate lines protected

---

## 5. After changes (recommend)

```bash
python -m tests.test_search
# Heavy: only with user OK — tools/tournament.py, tests/benchmark.py
```
