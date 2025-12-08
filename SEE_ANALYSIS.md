# Stockfish 與 Antares 引擎 SEE (Static Exchange Evaluation) 分析報告

本報告詳細分析了 Stockfish 引擎中 SEE 函數的實現原理、應用場景及其參數，並對比 Antares 引擎的現有實現，最後提出改進方案。

## 1. Stockfish SEE 分析

### 1.1 核心算法 (`Position::see_ge`)

Stockfish 的 SEE 實現位於 `src/position.cpp` 中的 `Position::see_ge`。

*   **功能**: 判斷一個移動後的靜態交換評估值是否大於或等於給定的 `threshold`。
*   **返回值**: `bool` (若 `>= threshold` 則返回 `true`)。
*   **主要邏輯**:
    1.  **初始交換**: 計算被吃子（victim）的價值減去 `threshold`。如果小於 0，直接返回 `false`（快速失敗）。
    2.  **攻擊者迭代**:
        *   從價值最低的攻擊者開始（P -> N -> B -> R -> Q -> K）。
        *   **X-Ray 處理**: 當一個攻擊者被移除後，檢查是否有滑子（R/B/Q）透過該方格進行 X-Ray 攻擊，並將其加入攻擊者列表。
        *   **Pinned Piece (牽制)**: Stockfish 明確檢查攻擊者是否被牽制 (`pinners(~stm) & occupied`)。如果被牽制且無法沿著牽制線移動，則該攻擊者無法參與交換。
    3.  **剪枝**: 使用 `threshold` 進行剪枝。如果當前積累的交換值已經無法逆轉結果，則提前終止循環。
    4.  **國王特例**: 如果國王參與交換，必須確保國王吃子後不會仍被攻擊（即不能送王）。

### 1.2 應用場景與參數

Stockfish 在多個關鍵搜索環節使用了 SEE，而不僅僅是 Move Ordering。

1.  **Move Ordering (MovePick)**:
    *   **Good/Bad Captures**: `pos.see_ge(move, -value / 18)`。
        *   這裡使用了一個寬鬆的負閾值，意味著即使虧一點點也可能被視為 Good Capture (或者是為了容錯)。
    *   **用途**: 將攻擊分為 Good Captures (排在殺手棋之前) 和 Bad Captures (排在最後)。

2.  **Main Search Pruning (Search)**:
    *   **SEE Pruning (Quiet Moves)**:
        *   條件: `!pos.see_ge(move, -154 * depth - seeHist)`
        *   邏輯: 如果一個安靜移動的 SEE 值非常差（負值，且隨深度增加容忍度變大，但仍是負的），則直接剪枝。這通常用於過濾掉那些明顯送子的安靜移動。
    *   **LMR (Late Move Reduction) Gating**:
        *   條件: `!pos.see_ge(move, -27 * lmrDepth * lmrDepth)`
        *   邏輯: 如果移動的 SEE 值太差，可能會遭受更嚴重的 LMR 或被剪枝。
    *   **Futility Pruning**:
        *   條件: `!pos.see_ge(move, alpha - futilityBase)`
        *   邏輯: 如果 SEE 值無法提升 alpha，則進行 Futility Pruning。
    *   **History Pruning**: 使用 `seeHist` 調整剪枝閾值。

3.  **Quiescence Search (QSearch)**:
    *   Stockfish 在 QSearch 中會剪掉 SEE < 0 的捕獲（除了某些特例）。

## 2. Antares SEE 現狀分析

### 2.1 核心算法 (`chess_engine/see.py`)

*   **功能**: 計算精確的 SEE 值。
*   **返回值**: `int` (分值)。
*   **不足之處**:
    1.  **Pinned Pieces (缺失)**: Antares 的 SEE **沒有檢查牽制**。這意味著被牽制的棋子（例如絕對牽制的兵）在 SEE 中會被視為合法的防禦者或攻擊者，導致評估錯誤。
    2.  **效率**: 雖然使用了 Numba，但在循環內部重複調用 `get_bishop_attacks` 和 `get_rook_attacks` 來處理 X-Ray，這在 Python/Numba 層面可能不如位運算增量更新高效（儘管邏輯是正確的）。
    3.  **無閾值優化**: 總是計算完整值，無法像 `see_ge` 那樣快速退出。
    4.  **En Passant**: 註釋提到 "current see implementation cannot evaluate... returning 0"。這是一個明顯的功能缺失。

### 2.2 應用場景

*   **Move Ordering**: 用於區分 Good/Bad Captures (`score_moves`)。
*   **QSearch Pruning**: `see < SEE_THRESHOLD` 則剪枝。
*   **Main Search**: **完全沒有使用 SEE 進行剪枝**。這是一個巨大的優化空間。Stockfish 利用 SEE 大膽地剪除那些看起來是送子的安靜移動（例如錯誤的棄子），從而節省大量節點。

## 3. 改進計劃

### 3.1 算法改進 (`chess_engine/see.py`)

1.  **引入 Pinned Piece 檢測**:
    *   在 SEE 循環中，如果輪到的攻擊者是被牽制的（絕對牽制），且移動方向不在牽制線上，則跳過該攻擊者。
2.  **支持 En Passant**:
    *   正確處理吃過路兵的價值計算（受害者在不同方格）。
3.  **API 調整**:
    *   雖然保持返回 `int` 對於 Move Ordering 的排序很方便，但我們可以增加一個可選的 `threshold` 參數（默認為 None 或極小值）。如果調用者提供了 `threshold`，則可以在內部實現類似 `see_ge` 的剪枝邏輯，提高性能。

### 3.2 搜索應用改進 (`chess_engine/search.py`)

1.  **Main Search SEE Pruning**:
    *   在 `_search` 的安靜移動循環中，加入 SEE 剪枝。
    *   **用戶要求**: "剪枝的條件必須要寬鬆許多"。
    *   Stockfish: `-154 * depth`。
    *   Antares 建議: `-200 * depth` 或固定值 `-100` (視測試而定)。這能防止引擎搜索那些顯然是送子的步法。
2.  **History Pruning**:
    *   可以考慮將 SEE 結合到 History Pruning 的邏輯中，但這比較複雜，初期先實現基本的 SEE Pruning。

## 4. 具體代碼變更預覽

### SEE 函數簽名更新
```python
def see(piece_bbs, occupancy_bbs, side_to_move, from_sq, to_sq, threshold=None):
    # ...
```

### Search 邏輯更新
```python
# In _search loop for quiet moves:
if ENABLE_SEE_PRUNING and depth <= SEE_PRUNING_DEPTH:
    # Relaxed margin
    if see(..., threshold=-100 * depth) < -100 * depth:
        pruned += 1
        continue
```
