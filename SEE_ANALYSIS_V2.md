# SEE Analysis & Improvement Plan

## 1. Introduction
This document analyzes the `see` (Static Exchange Evaluation) implementation in Stockfish and compares it with the current engine's implementation in `chess_engine/see.py`. The goal is to identify gaps and improve the engine's SEE to match Stockfish's capabilities.

## 2. Stockfish SEE Analysis

### 2.1. Core Logic (`position.cpp`)
Stockfish implements `see` and `see_ge` (greater-or-equal optimization).

*   **Exact SEE (`see`)**: Calculates the exact score of the exchange.
*   **SEE GE (`see_ge`)**: Returns `true` if the SEE value is >= threshold, optimized for pruning.

### 2.2. Key Features
1.  **Promotion Handling**:
    *   Explicitly accounts for material gain during promotion.
    *   `swap += PieceValue[promo] - PawnValue`.
    *   The piece remaining on the square is the *promoted* piece.
2.  **Pinned Pieces**:
    *   Uses `pinners` bitboard (enemy sliders pinning friendly pieces).
    *   Checks `if (pinners(~stm) & occupied)` to verify if the pinner is still on the board.
    *   If the pinner is present, pinned pieces are removed from the attackers list (`stmAttackers &= ~blockers_for_king(stm)`).
    *   This ensures pinned pieces are freed if the pinner is captured in the exchange.
3.  **X-Ray Attacks**:
    *   Updates `occupied` bitboard incrementally.
    *   When a piece is removed, it adds "X-Ray" attackers behind it to the `attackers` bitboard.
    *   This avoids recalculating all attacks from scratch.
4.  **En Passant**:
    *   Handles EP capture value (`PawnValue`).
    *   Does *not* explicitly remove the captured pawn from `occupied` (an approximation or optimization).

### 2.3. Constants
Piece values in Stockfish (`types.h`):
*   Pawn: 208
*   Knight: 781
*   Bishop: 825
*   Rook: 1276
*   Queen: 2538

## 3. Current Engine Gap Analysis

### 3.1. Missing Promotion Logic
*   **Current**: `see` infers capture value from `to_sq`. It treats promotions as normal pawn moves (value = Pawn or captured piece).
*   **Gap**: Misses the massive material gain from promotion (e.g., Pawn -> Queen).
*   **Impact**: Severe underestimation of promotion captures, leading to bad pruning or ordering.

### 3.2. Incorrect Pinned Piece Logic
*   **Current**: Calculates pinned pieces *once* based on the initial board.
*   **Gap**: Does not account for pinners being captured. If a pinner is captured during the SEE sequence, the pinned piece should be free to recapture.
*   **Impact**: Pessimistic evaluation (falsely believes a recapture is impossible).

### 3.3. Piece Values
*   **Current**: `[100, 320, 330, 500, 900, 20000]`.
*   **Gap**: Values are normalized/simplified. Stockfish uses tuned values.
*   **Action**: Adopt Stockfish's relative values or keep consistent with engine's internal evaluation. For SEE specifically, relative values matter most. The engine's values are fine but `100` vs `208` scale difference exists. *Decision: Stick to engine's values for consistency with evaluation, or scale them.*

### 3.4. Function Signature
*   **Current**: `see(..., from_sq, to_sq)`.
*   **Gap**: Cannot access move flags (Promotion, EP).
*   **Action**: Change to `see(..., move)`.

## 4. Improvement Plan

### 4.1. Refactor `chess_engine/see.py`
1.  **Update Signature**: `see` and `see_ge` will take `move` (uint16) as input.
2.  **Implement Promotion**: Parse `move` to check for promotion and adjust values.
3.  **Implement Advanced Pin Logic**:
    *   Implement `get_pinners` to return the bitboard of enemy sliding pieces that are pinning.
    *   In the loop, check `if (pinners & occupied)` before filtering pinned pieces.
4.  **Optimize X-Ray**: Ensure `get_sliding_attacks` correctly finds new attackers behind removed pieces.

### 4.2. Update Call Sites
*   Update `chess_engine/search.py` (`score_moves`, `quiescence_search`) to pass the `move` object.

### 4.3. Verification
*   Create a test script comparing `see` output for complex positions (Pins, Promotions).
