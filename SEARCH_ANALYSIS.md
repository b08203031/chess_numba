# Search Analysis: Spirit, Meaning, Details, and Methodology

This document provides a comprehensive analysis of the search implementation in `chess_engine/search.py` and related files (`constants.py`, `transposition_table.py`). It explains the underlying logic ("spirit"), the significance of the parameters ("meaning"), the specific implementation choices ("details"), and the overall algorithmic approach ("methodology").

## 1. Overview

The engine employs a **Principal Variation Search (PVS)** within an **Iterative Deepening** framework. This is the standard approach for modern chess engines, allowing for efficient exploration of the game tree by prioritizing the most promising moves (Principal Variation) and proving that other moves are inferior with minimal effort (Zero Window Search).

**Key Files:**
*   `chess_engine/search.py`: Core search logic (Iterative Deepening, PVS, Quiescence).
*   `chess_engine/constants.py`: Tuning parameters, pruning margins, and weights.
*   `chess_engine/transposition_table.py`: Hash table implementation.

## 2. Core Algorithms

### 2.1 Iterative Deepening
**Spirit:** Incrementally search deeper (depth 1, 2, 3...) rather than searching to a fixed depth immediately.
**Meaning:** Allows the engine to always have a "best move" ready if time runs out. Crucially, it populates the Transposition Table (TT) and History Heuristics from shallow searches, which massively improves move ordering for deeper searches.
**Details:**
*   Located in `iterative_deepening_search`.
*   Uses Aspiration Windows (`alpha`, `beta`) centered around the previous iteration's score to narrow the search window and increase cutoffs.
*   Handles time management checks between iterations.

### 2.2 Principal Variation Search (PVS) / Negamax
**Spirit:** Assume the first move (ordered best) is the Principal Variation (PV). Search it with a full window `(alpha, beta)`. Assume all subsequent moves are worse and search them with a "Zero Window" or "Null Window" `(alpha, alpha+1)` to prove they fail low.
**Meaning:** Drastically reduces the effective branching factor if move ordering is good.
**Details:**
*   Implemented in `_search`.
*   If `searched_move_count > 1`, it performs a Zero Window search (`-alpha - 1, -alpha`).
*   If the Zero Window search fails high (returns score > alpha), it means the assumption was wrong, and a full re-search is required (`-beta, -alpha`).

### 2.3 Quiescence Search (QS)
**Spirit:** extend the search at leaf nodes (depth 0) to "quiet" positions where no captures or major threats exist, mitigating the "Horizon Effect".
**Meaning:** prevents the engine from stopping the search in the middle of a capture sequence (e.g., stopping after QxP but before the recapture PxQ).
**Details:**
*   Implemented in `quiescence_search`.
*   Only considers captures (and potentially checking moves if in check).
*   Includes **Delta Pruning**: If the position is so bad that even a huge capture (Queen) wouldn't restore the balance, prune immediately.
*   Includes **SEE Pruning**: Prunes bad captures (SEE < 0) in QS to save time.

## 3. Move Ordering

Move ordering is the single most critical factor for Alpha-Beta performance. The engine uses the following hierarchy:

1.  **Transposition Table (TT) Move:** The best move found from a previous iteration or search. This is always tried first.
2.  **Captures:**
    *   Sorted by **MVV-LVA** (Most Valuable Victim - Least Valuable Aggressor).
    *   **Good Captures:** Captures with SEE >= 0 are prioritized.
    *   **Bad Captures:** Captures with SEE < 0 are searched later (penalty applied).
3.  **Killer Moves:** Two moves that caused a beta-cutoff at the same ply in sibling nodes.
4.  **Counter Move:** The move that historically caused a cutoff in response to the opponent's previous move.
5.  **History Heuristic:** Quiet moves ordered by their historical success (frequency of causing cutoffs relative to depth).
    *   Uses a "Gravity" formula to decay old history and bound values: `history += bonus - history * abs(bonus) / MAX_HISTORY`.

## 4. Pruning and Reductions

The engine implements a suite of aggressive pruning techniques to reduce the search space.

### 4.1 Null Move Pruning (NMP)
**Spirit:** If we pass our turn (do a null move) and the opponent *still* cannot improve their position above beta, we are winning by a large margin and can cut off.
**Logic:**
*   Condition: `static_eval >= beta` and `depth >= 3`.
*   Action: Search with reduced depth (`R = 2`).
*   Verification: If the returned score is `>= beta`, return beta (cutoff).
*   **Safety:** Not applied in check or endgame (Zugzwang risk).

### 4.2 Reverse Futility Pruning (RFP) / Static Null Move Pruning
**Spirit:** If the static evaluation is so high that even subtracting a large margin leaves us above beta, we can likely cutoff immediately.
**Logic:**
*   Applied at `depth <= 2`.
*   Condition: `static_eval - margin >= beta`.

### 4.3 Razoring
**Spirit:** If the static evaluation is very low at low depths, assume we can't recover and drop into Quiescence Search early.
**Logic:**
*   Applied at low depths.
*   Condition: `static_eval + margin < alpha`.
*   Action: Check with Quiescence Search. If it fails low, return alpha.

### 4.4 Futility Pruning (FP)
**Spirit:** Prune quiet moves at low depths if the static evaluation plus a margin (based on depth) is still below alpha.
**Logic:**
*   Condition: `static_eval + margin < alpha`.
*   Action: Skip the current move.

### 4.5 Late Move Pruning (LMP)
**Spirit:** If we have searched many quiet moves at a node and haven't found a better move, it's unlikely the remaining (worse-ordered) quiet moves will help.
**Logic:**
*   Condition: `quiet_move_count > LMP_THRESHOLD(depth)`.
*   Action: Stop searching quiet moves at this node.

### 4.6 ProbCut
**Spirit:** A probabilistic version of Null Move. Perform a shallow search with a wider window to test if a strong cutoff is likely.
**Logic:**
*   Uses data from TT.
*   If a shallow search confirms the cutoff, prune the main search.

### 4.7 Late Move Reduction (LMR)
**Spirit:** Search moves that are ordered late (likely worse) with reduced depth.
**Logic:**
*   Condition: `depth >= 3`, move is quiet, move index is high.
*   Reduction: `R = 1 + log(depth) * log(move_count) / 3`.
*   Adjustment: History score modifies R (good history reduces reduction, bad history increases it).
*   Extensions: Moves giving check or promoting are usually not reduced.

### 4.8 Singular Extensions (SE)
**Spirit:** If the TT move is significantly better than all other moves (singular), extend the search depth for this node to ensure accuracy.
**Logic:**
*   Perform a shallow search excluding the TT move.
*   If all other moves fail low by a large margin, the TT move is "singular".
*   Action: Extend search depth by 1.

### 4.9 Internal Iterative Deepening (IID)
**Spirit:** If no TT move is present (e.g., new node), perform a shallow search to get a good move for ordering before the full search.
**Logic:**
*   Condition: `depth >= 8` and no TT move.
*   Action: Search at `depth - 5` first.

## 5. Transposition Table (TT)

**Spirit:** A hash map to store search results for positions (nodes) encountered.
**Implementation Details:**
*   **Key:** Zobrist Hash (64-bit).
*   **Structure:** `key`, `score`, `depth`, `flag` (Exact/Alpha/Beta), `generation`, `best_move`.
*   **Replacement Strategy:**
    1.  **Generation:** Always replace entries from previous searches (generations).
    2.  **Depth:** Within the same generation, prefer entries with higher depth.
*   **Optimization:** Uses `numba` optimized structure for fast access.
*   **Score Adjustment:** Mate scores are stored relative to the root, but adjusted relative to the current ply when retrieving/storing.

## 6. Time Management

**Spirit:** Allocate time based on the remaining time and the game phase.
**Logic:**
*   **Soft Limit:** `optimum_time`. If exceeded, stop search at next safe point (depth end).
*   **Hard Limit:** `maximum_time`. If exceeded, stop immediately (set `stop_flag`).
*   **Predictive Termination:** If `elapsed * 2.5 > max_time`, stop early to avoid wasting time on a depth iteration that likely won't finish.

## 7. Repetition Detection

**Spirit:** Detect 3-fold repetition to correctly score draws (0.00).
**Logic:**
*   Checks `ply_path_stack` (moves in current search) and `game_history` (moves played before search).
*   **Optimization:** Only checks positions with the same side-to-move (plies `ply, ply-2, ...`).
*   **Limit:** Checks are bounded by the `halfmove_clock` (rule of 50 / pawn move / capture resets repetition).

## 8. Analysis and Suggestions

### Correctness Checks
1.  **PVS Logic:** Implementation is standard and correct.
2.  **LMR Formula:** `1.0 + (ld * lmc) / 3.0` is a reasonable approximation of the standard `0.5 + ... / 2.0` formula, though slightly more aggressive at the base.
3.  **Mate Score:** Correctly handles `+/- ply` adjustment.
4.  **Repetition:** Logic covers both search path and game history correctly.

### Potential Issues & Improvements
1.  **Typo in LMR:**
    Line: `lmr = lmr = max(0, lmr - 1)`
    Status: Harmless but sloppy. Should be `lmr = max(0, lmr - 1)`.

2.  **Redundant NMP Check:**
    Code:
    ```python
    if static_score >= beta - NMP_STATIC_MARGIN:
        if static_score >= beta:
            # ...
    ```
    Analysis: `NMP_STATIC_MARGIN` is 600. `beta - 600` is smaller than `beta`. The outer check is looser than the inner check. The inner check dominates. The outer check is redundant but harmless.

3.  **Predictive Time Termination:**
    Code: `if elapsed_time_ms * 2.5 > maximum_time_ms: break`
    Analysis: This is a heuristic. In some endgame positions or forced lines, the branching factor is small (< 1.5). This might cause the engine to stop at depth `N` when it could easily finish `N+1`.
    Suggestion: Consider a dynamic branching factor estimate or just rely on the soft/hard limits. However, for safety, it is acceptable.

4.  **TT Probe Safety:**
    The `probe_tt` returns a copy of the entry or a view. Since it's Numba compiled `njit`, returning a structured array element usually returns a value (tuple-like) or a reference. Given `store_tt` writes to the array directly by index, this is safe.

### Conclusion
The search implementation is robust, modern, and follows standard chess programming practices. It includes all major strength-increasing heuristics. The identified issues are minor and cosmetic.

