---
description: "Search algorithm mathematical correctness, Negamax symmetry, draw, repetition, and pruning safety constraints."
trigger: glob
glob: "**/{search,search_heuristics}.py"
---

# Search Correctness Constraints

When modifying or optimizing the core search engine (`search.py`, `search_heuristics.py`), the AI assistant **MUST** ensure the mathematical correctness of the search algorithms and the safety of pruning boundaries.

---

## 1. Negamax Symmetry Invariants

The engine operates under the Negamax framework.

*   **Score Symmetry**:
    *   All evaluation scores and search returns **must** be relative to the active player (**Side to Move**).
    *   When recursively calling child searches, you **must** negate the return value (e.g., `score = -search(...)`) and invert the alpha-beta bounds (`-beta`, `-alpha`).
    *   Any asymmetric score calculations or reduction offsets will break the alpha-beta pruning mathematics, resulting in catastrophic tactical blunders.

---

## 2. Checkmate Score Handling (Mate Distance Invariants)

When checkmate is discovered, the return score **must** be adjusted dynamically based on the **search depth (ply)**. This guides the engine to select the fastest mate path when winning, and the longest defense path when losing.

*   **Ply-Distance Adjustment Formulas**:
    *   **Mating the opponent**: Return `MATE_SCORE - ply`
    *   **Being mated by the opponent**: Return `-MATE_SCORE + ply`
*   **Transposition Table (TT) Safeguard**:
    *   When writing mate scores into or reading them from the TT, you **must** translate/normalize them (removing the ply offset relative to the current node) to ensure cache validity across different search branches.

---

## 3. Repetition & Draw Invariants

To prevent the engine from falling into infinite loops or ignoring upcoming draw conditions:

*   **Stack and History Dual-Check**:
    *   Repetition checks **must** compare both the active search path stack (`ply_path_stack`) and the global game history (`game_history`).
    *   The lookup scope is strictly limited by the **halfmove clock limit** (no pawn moves, no captures).
*   **Draw Scores**:
    *   A **draw (0.0 score)** must be triggered on the **first repetition** (i.e., the second time a position is reached) or when the **50-move halfmove clock limit** is reached.
    *   **Root Safeguard**: **Strictly forbid draw pruning at the root node (ply = 0)**. The root must always evaluate legal moves fully to avoid illegal moves or instant resignations.
*   **50-Move Scale-down**:
    *   As the halfmove clock approaches 50, scale down evaluations smoothly to prevent search "cliff effects" where a forced draw is ignored until it is too late.

---

## 4. Pruning Safety Boundaries

Search heuristics (such as Null Move Pruning (NMP), Reverse Futility Pruning (RFP), Late Move Pruning (LMP), and Razoring) are highly effective at node reduction but must strictly respect safety bounds:

*   **No Root Pruning**: **Never** prune or return early at the root node (`ply == 0`). All legal moves at the root must be fully searched.
*   **In-Check Protection**: If the side-to-move is in check (`is_in_check` is True), **strictly forbid** NMP, RFP, or any other heuristic pruning. You must perform a full-width search of all check evasions.
