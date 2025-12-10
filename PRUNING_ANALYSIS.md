# Stockfish 剪枝策略與 SEE 分析報告

本報告分析了 Stockfish 的靜態交換評估 (SEE) 實作及其在搜尋剪枝中的應用，並對比目前的 `chess_engine`，提出改進方案。

## 1. Stockfish 的 SEE (Static Exchange Evaluation) 實作分析

### 1.1 原理與邏輯 (`position.cpp`)
Stockfish 的 `see` 函數主要用於評估一個移動在連續交換後的靜態價值。
*   **核心演算法**: 模擬雙方輪流在目標格上的捕捉，生成一個分數列表 (`gain[]`)，最後通過 Minimax 反向傳播計算最終得分。
*   **優化**:
    *   **X-Ray (透視攻擊)**: 當一個滑動棋子 (R/B/Q) 移開後，會將其後方的攻擊者加入攻擊列表。Stockfish 通過 bitboard 操作 (`attackers |= attacks_bb<...>(to, occupied) & pieces(...)`) 高效實現。
    *   **Pinned Pieces (釘子)**: 絕對釘子 (Pinned pieces) 在 SEE 中通常不被視為合法的攻擊者，或者在 Stockfish 的簡化版 `see_ge` 中，如果移除釘子會導致國王被攻擊，則該攻擊者被視為無效。
    *   **Early Exit (提早退出)**: `see_ge` (SEE >= Threshold) 是一個高度優化的版本，一旦當前累積的交換值已經無法滿足閾值 (或者必定滿足)，就提早返回。

### 1.2 `chess_engine/see.py` 現狀與改進
目前的 `see.py` 已經實作了基本的 Swap Algorithm 和 X-Ray 處理，以及 `see_ge` 的優化。
**改進空間**:
*   **攻擊者尋找**: Stockfish 使用預計算的 `attackers_to` 配合位運算，目前的 Python/Numba 實作雖然邏輯正確，但在尋找 "Least Valuable Attacker (LVA)" 時的迴圈效率可能不如 Stockfish 的位元操作直接。
*   **Pinned Pieces**: Stockfish 在 `see` 中會檢查潛在的釘子。目前的實作有 `get_pinned_pieces`，但確保其邏輯與 Stockfish 的 "假裝移開後檢查 Check" 一致是關鍵。
*   **Capture 模擬**: 確保 En Passant 和 Promotion 的價值計算準確。

## 2. Stockfish 的剪枝策略 (重點關注 Step 14)

在 `stockfish/src/search.cpp` 中，"Step 14: Pruning at shallow depth" 是一個非常強力的剪枝階段，主要依賴 SEE 和歷史啟發 (History Heuristic)。

### 2.1 邏輯結構
此階段只在非根節點 (`!rootNode`) 且非殺棋分數 (`!is_loss`) 時執行。

1.  **Skip Quiet Moves (略過靜止移動)**:
    *   如果已經搜尋的移動數 (`moveCount`) 超過了基於深度和改進狀態的閾值 (`futility_move_count`)，則直接跳過剩餘的靜止移動 (Quiet Moves)。這依賴於 Move Ordering 已經將好棋排在前面的假設。

2.  **Captures & Checks (捕捉與將軍)**:
    *   **Futility Pruning (Captures)**: 如果深度很淺 (`lmrDepth < 7`) 且 `staticEval + gain + history < alpha`，則剪枝。
    *   **SEE Pruning**: 如果 `see(move) < threshold`，則剪枝。
        *   Stockfish 閾值: `-154 * depth - (history_bonus / 32)`。也就是說，隨著深度增加，允許更差的 SEE；歷史分數高的移動允許更差的 SEE。

3.  **Quiet Moves (靜止移動)**:
    *   **History Pruning**: 如果該移動的歷史分數極低 (`history < -4348 * depth`)，直接剪枝。
    *   **Futility Pruning (Parent)**: 基於 `staticEval` 和歷史分數預估改進後的深度，如果評估值仍低於 `alpha`，則剪枝。
    *   **SEE Pruning**: 即使是靜止移動，如果 SEE 負得太多 (通常發生在自殺移動或受威脅的格子)，也會剪枝。
        *   Stockfish 閾值: `-27 * lmrDepth * lmrDepth` (非線性寬鬆)。

### 2.2 `chess_engine/search.py` 現狀與改進
目前的 `search.py` 缺乏這種整合性的淺層剪枝模組。我們有獨立的 `Futility Pruning` (Step 8/14 混合) 和 `LMP`，但缺乏基於 SEE 的強力過濾。

**改進方案**:
在 `_search` 的 Move Loop 中引入類似 Stockfish Step 14 的邏輯塊。

## 3. 改進計畫

### 3.1 `see.py` 優化
*   微調 `see_ge` 邏輯，確保與 Stockfish 一致的 Pinned 處理。
*   確認數值體系 (`PIECE_VALUES`) 與搜尋中的評分一致。

### 3.2 `search.py` 新增剪枝
在 Move Loop 內部，`make_move` 之前，新增以下邏輯：

1.  **Capture SEE Pruning**:
    *   對於捕捉移動，如果 `see_ge` 低於某個負閾值 (例如 `-100 * depth`)，則剪枝。
    *   *寬鬆設定*: Stockfish 用 `-154 * depth`，我們初期可以用 `-200 * depth` 或固定 `-100` (目前已有但只在 QS?)，讓它更寬鬆以免剪掉棄子戰術。

2.  **Quiet Move SEE Pruning**:
    *   對於靜止移動，如果有明顯的 SEE 虧損 (例如走到被攻擊的格子)，剪枝。
    *   閾值公式參考: `-50 * depth * depth` (比 Stockfish 的 `-27` 寬鬆)。

3.  **History Pruning**:
    *   利用 `history_table`，如果分數過低 (例如 `< -10000`)，剪枝。

### 3.3 參數設定 (寬鬆版)
為了符合 "一開始我們先寬鬆一點，不要剪太多" 的要求：
*   **Capture SEE Margin**: `-200 * depth` (容忍深度越深越大的虧損)。
*   **Quiet SEE Margin**: `-100 * depth * depth`.
*   **History Threshold**: `-16000` (接近 `MIN_HISTORY` 的一半)。

## 4. 預期效果
*   **NPS 提升**: 減少對明顯壞棋 (送子、無效捕捉) 的搜尋。
*   **棋力提升**: 將算力集中在更有希望的分支。

---
報告完畢。接下來將依照此報告進行程式碼修改。
