# Stockfish SEE (Static Exchange Evaluation) 分析報告

## 1. 原理與實現方式

Stockfish 的 SEE 函數用於靜態評估一個連續交換序列的結果。它模擬雙方在同一個目標格（Target Square）上不斷進行吃子，直到一方停止或被吃光。

### 核心機制
*   **Swap Algorithm**: 使用經典的 Swap List 算法。計算每次吃子後的 `gain[i] = captured_piece_value - gain[i-1]`。
*   **Minimax Propagation**: 序列結束後，從後往前進行 Minimax 傳播：`score[i-1] = -max(-score[i-1], score[i])`。這代表如果下一步對手能獲得更好的結果，當前這一步就會被駁回。
*   **see_ge 優化**: 大部分應用場景只需要知道 SEE 是否大於等於某個閾值 (`threshold`)。`see_ge` 函數在模擬過程中，如果發現當前餘額 (`balance`) 已經低於 0（且輪到對手吃子），或者餘額極高對手無法挽回，就會提前終止並返回結果，省去了後續不必要的計算。

### 實現細節
*   **Bitboards**: 利用 Bitboard 操作高效查找攻擊者。
*   **X-Ray (透視攻擊)**: SEE 需要處理 X-Ray 攻擊（例如 R-Q-P，R 吃 P 後，Q 成為新的攻擊者）。Stockfish 的做法是維護一個 `occupied` bitboard，每當一個攻擊者「移除」（移到目標格）後，將其從 `occupied` 中清除，並利用 `attacks_bb(to_sq, occupied)` 重新生成滑動棋子的攻擊，從而「透視」到原本被遮擋的棋子。
*   **LVA (Least Valuable Attacker)**: 總是優先選擇價值最低的攻擊者（Pawn -> Knight -> Bishop -> Rook -> Queen -> King）進行吃子。
*   **Pinned Pieces (釘子)**: Stockfish 在 SEE 中有一個重要的簡化/優化：如果某一方的棋子被釘住（Pinned），它完全被禁止參與 SEE 交換。這雖然是近似（在釘子方向上的吃子其實是合法的），但避免了复杂的合法性檢查，且在静態評估中通常是安全的假設。

### 特殊移動處理
*   **En Passant (吃過路兵)**: 作為第一步吃子時，被吃價值設為 `PawnValue`。
*   **Promotion (升變)**: 如果移動是升變，第一步的價值會加上 `PromotionValue - PawnValue`。這一點在當前引擎中被忽略了。
*   **Castling (王車易位)**: 返回 0（中性）。

## 2. 應用場景與參數

Stockfish 在搜索的多個階段極其依賴 SEE 來進行剪枝和排序。

### 參數體系 (Piece Values)
Stockfish 使用的棋子價值（Midgame）：
*   Pawn: 208
*   Knight: 781
*   Bishop: 825
*   Rook: 1276
*   Queen: 2538

這與傳統的 100/300/320/500/900 體系不同，比例約為 2 倍左右。

### 主要應用場景與閾值

1.  **Move Ordering (走法排序)**
    *   **Good Captures**: `see_ge(move, -value / 18)`。允許微小的負收益（犧牲），但過於糟糕的吃子被歸類為 Bad Captures 並排在最後。

2.  **Quiescence Search (靜態搜索)**
    *   **Pruning**: `if (!see_ge(move, -75)) prune`。在 QS 中直接剪枝掉 SEE 結果很差的吃子（-75 約為 0.36 個兵）。
    *   **Futility**: 用於剪枝那些即便 SEE 獲利也無法提升 alpha 的平靜走法。

3.  **Shallow Pruning (淺層剪枝)**
    *   在主搜索的淺層（Step 14），使用動態閾值剪枝：
        *   **Captures**: `threshold = -154 * depth - history`。深度越深，容忍度越高。
        *   **Quiets**: `threshold = -27 * lmrDepth * lmrDepth`。

4.  **ProbCut**: 使用 SEE 作為預剪枝的過濾器。

## 3. 當前引擎的不足之處分析

經過對比分析，我的引擎在 SEE 實現上存在以下問題：

### 1. 性能瓶頸：Pinned Pieces 重複計算
*   **問題**: 我的 `see` 函數中，每次調用都會運行 `get_pinned_pieces`，這需要進行多次射線掃描（Ray Casting）。
*   **Stockfish**: 使用增量更新或緩存的 `pinners` 信息，檢查開銷極低。
*   **影響**: SEE 是高頻調用函數，這裡的開銷會嚴重拖慢 NPS。

### 2. 功能缺失：升變處理 (Promotion)
*   **問題**: 我的 `see` 函數簽名為 `(..., from_sq, to_sq)`，沒有傳入 `move` 或 `promotion_type`。因此，升變走法被視為普通的兵吃子或移動，嚴重低估了升變的價值（少了 Q-P 的巨大分值）。
*   **影響**: 導致升變走法在排序中可能靠後，或在 QS 中被錯誤剪枝。

### 3. 數值體系差異與閾值適配
*   **問題**: 我的引擎使用 P=100 體系，而 Stockfish 是 P=208 體系。
*   **影響**: 直接照搬 Stockfish 的閾值（如 -75）會導致剪枝過於激進（-75 在你是 0.75 個兵，在 Stockfish 僅 0.36 個兵）。需要按比例（約 0.5 倍）縮放閾值。

### 4. X-Ray 攻擊生成效率
*   **問題**: 使用循環遍歷生成滑動攻擊 (`get_sliding_attacks`)，而非 Magic Bitboards 或查表。
*   **影響**: 計算速度不如 Stockfish，但在 Numba 優化下可能尚可接受，優先級低於 Pinned Pieces 優化。

## 4. 改進計劃

1.  **修改 `see` 接口**: 增加 `promotion_type` 參數，正確計算升變價值。
2.  **優化 Pinned 邏輯**: 在 SEE 中移除全局 `get_pinned_pieces` 計算。改為僅當「攻擊者與 King 在同一條射線上」時，才進行局部的 Pin 檢查，或者在 `SearchContext` 中維護 Pinned Bitboard（若架構允許）。考慮到目前架構，採用「局部檢查」或「忽略 Pin（僅依賴合法性檢查）」可能是短期更優解。
3.  **調整閾值**: 將所有涉及 SEE 的閾值乘以 0.5 以適應 P=100 的價值體系。
