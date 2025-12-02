# Stockfish SEE Pruning Logic Analysis

## 1. Main Search Pruning (Shallow Depth)

Located in `search.cpp` inside `search()` function, step 14 (Pruning at shallow depth).

### 1.1 Captures and Checks (Bad Capture/Check Pruning)
```cpp
if (capture || givesCheck) {
    // ...
    // SEE based pruning for captures and checks
    int seeHist = std::clamp(captHist / 32, -138 * depth, 135 * depth);
    if (!pos.see_ge(move, -154 * depth - seeHist))
        continue;
}
```
*   **Condition**: If it's a capture or check.
*   **Threshold**: `-154 * depth - seeHist`.
    *   `seeHist` is derived from capture history (Move Ordering history for captures), scaled.
    *   Base threshold is linear with depth (`-154 * depth`).
    *   This effectively prunes captures/checks that lose significant material (e.g. Queen takes protected Pawn at depth 1 -> SEE -800. Threshold -154. Pruned).
*   **Application**: Prunes moves that are statically bad exchanges.

### 1.2 Quiet Moves (History Guard)
```cpp
else {
    // ...
    // Prune moves with negative SEE
    if (!pos.see_ge(move, -27 * lmrDepth * lmrDepth))
        continue;
}
```
*   **Condition**: Quiet move (not capture, not check).
*   **Threshold**: `-27 * lmrDepth * lmrDepth`.
    *   Quadratic with LMR depth.
    *   Checks if the quiet move is safe. If `see` is very negative (piece hangs), it's pruned.
*   **Application**: "History Guard". Prevents searching quiet moves that hang a piece, even if History Heuristic likes them.

## 2. Quiescence Search Pruning

Located in `qsearch()`.

### 2.1 Futility Pruning (Captures)
```cpp
// Futility pruning and moveCount pruning
if (!givesCheck && move.to_sq() != prevSq && !is_loss(futilityBase) && move.type_of() != PROMOTION) {
    // ...
    // If static exchange evaluation is low enough we can prune this move.
    if (!pos.see_ge(move, alpha - futilityBase)) {
        bestValue = std::min(alpha, futilityBase);
        continue;
    }
}
```
*   **Condition**: Not check, not recapture (maybe?), not promotion.
*   **Threshold**: `alpha - futilityBase`.
    *   `futilityBase = staticEval + 359`.
    *   Checks if `gain >= alpha - (eval + 359)`.
    *   i.e. `eval + 359 + gain >= alpha`.
    *   If even with the gain from SEE, we don't beat alpha, prune.
*   **Application**: Futility pruning for captures in QS.

### 2.2 Bad Capture Pruning (General)
```cpp
// Do not search moves with bad enough SEE values
if (!pos.see_ge(move, -75))
    continue;
```
*   **Condition**: All QS moves (captures/checks).
*   **Threshold**: `-75` (approx -0.75 pawns).
*   **Application**: Prunes any capture that loses material (e.g. N x P protected).

## 3. Plan Update

I need to implement:
1.  **QS Futility Pruning**: `see < alpha - (eval + margin)`.
2.  **Main Search Bad Capture/Check Pruning**: `see < -154 * depth`. (Ignoring `seeHist` for now as I don't have capture history implemented yet, or I can use simple history?).
3.  **Quiet Move Guard**: `see < -27 * depth * depth` (already partially implemented, but need to ensure `see` works for quiet moves).

Crucially, **fix `see.py`** to handle Quiet Moves correctly first.
