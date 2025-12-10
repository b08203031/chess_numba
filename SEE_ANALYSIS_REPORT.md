# Stockfish SEE 分析與引擎改進報告

## 1. Stockfish SEE 原理分析

### 1.1 演算法核心 (Algorithm)
Stockfish 使用標準的 **Swap Algorithm** (交換算法) 進行靜態交換評估 (Static Exchange Evaluation, SEE)。該算法模擬在一個方格上的一系列捕捉 (captures)，以確定在該方格上進行交換是否對發起方有利。

- **Minimax 邏輯**:
  - 算法維護一個 `gain` 數組，記錄每一層捕捉後的材質變化。
  - `gains[i] = value_captured - see(next_position)`
  - 最後通過回溯 (Backpropagation) 計算最終分數：`score = -max(-score_next_ply, current_gain)`。這本質上是一個 Minimax 過程：如果對手在下一層能獲得更好的結果，他就會選擇那條路，所以當前層的得分是對手最佳回應的負值。

- **處理細節**:
  - **En Passant (吃過路兵)**: 特殊處理，受害者是兵，但被捕捉的方格是兵後方的方格。捕捉價值為兵的價值。
  - **Promotion (升變)**: 如果移動是升變，捕捉價值會加上 (升變棋子價值 - 兵價值)。
  - **Castling (王車易位)**: 返回 0 (雖然 SEE 通常不對易位調用，但有防禦性代碼)。
  - **X-Ray (X 射線攻擊)**: 當一個棋子移動（進行捕捉）後，算法會檢查它原來的方格是否阻擋了後方滑動棋子 (Rook, Bishop, Queen) 的攻擊線。如果是，則將後方的滑動棋子加入攻擊者列表。Stockfish 利用 Magic Bitboards 高效地完成這一步。
  - **Pin (牽制) 處理**: 這是 Stockfish 的一個關鍵優化。它在 SEE 內部不允許**被絕對牽制**的棋子進行移動。如果一個棋子移動會導致己方王被將軍（即它被牽制），則該棋子被從攻擊者列表中移除。

### 1.2 實現方式 (Implementation)
- **輸入**: `Position` 對象和 `Move` 對象。
- **效率優化**:
  - `see_ge` (Greater or Equal): 這是一個優化版本，只返回布林值。它利用 `balance` 變量進行剪枝。如果當前積累的材質差加上剩餘攻擊者的價值都無法達到閾值，或者已經超過閾值且對手無法翻盤，則提前終止。
  - **位元運算**: 大量使用 Bitboards 操作，如 `pop_lsb`, `attacks_bb`。

### 1.3 參數 (Parameters)
Stockfish 在 SEE 中使用的棋子價值 (`types.h`):
- Pawn: 208
- Knight: 781
- Bishop: 825
- Rook: 1276
- Queen: 2538
- King: 0 (但在邏輯中作為無窮大處理，一旦王進行捕捉，如果安全則結束，不安全則視為非法)。

這些值與其中局/殘局評估值略有不同，是專門為 SEE 調優的內部單位。

### 1.4 應用場景 (Usage)
1.  **Move Ordering (移動排序)**:
    - **Good Captures**: `see_ge(move, -threshold)`. 好的捕捉被排在前面。
    - **Bad Captures**: SEE 值為負的捕捉被排在後面（甚至在寧靜步之後）。
2.  **Quiescence Search (靜態搜尋)**:
    - 用於剪枝：如果一個捕捉的 SEE 值非常低（例如由大子吃小子且被保護），可能被直接剪枝（`ENABLE_SEE_IN_QUIESCENCE`）。
3.  **ProbCut**:
    - 在主搜尋中，利用強烈剪枝技術。如果一個非捕捉移動或淺層搜尋的 SEE 值很高，可能觸發 ProbCut。
4.  **Pruning**:
    - **MCP (Multicut Pruning)** / **LMP**: 參考 SEE 值。

---

## 2. 當前引擎的不足 (Deficiencies)

經過對比分析 `chess_engine/see.py` 與 Stockfish 源碼，發現以下不足：

1.  **升變處理缺失 (Promotion Bug)**:
    - 當前 `see` 函數簽名為 `see(..., from_sq, to_sq)`，丟失了 `Move` 對象中的 `promotion` 信息。
    - 結果：兵升變吃子時，被視為普通兵吃子，嚴重低估了升變帶來的材質增益。

2.  **牽制計算效率低 (Inefficient Pin Handling)**:
    - 當前引擎在 `see` 函數開頭調用 `get_pinned_pieces`，這是一個昂貴的射線投射操作，且對所有方向重新計算。
    - Stockfish 利用已有的 `pinners` 資訊或更高效的位運算。雖然 Python 中難以完全複製 C++ 指針結構，但可以優化邏輯，僅在相關方向檢查，或利用 `SearchContext` 中的資訊（如果可靠）。

3.  **X-Ray 邏輯實現差異**:
    - 當前引擎在循環末尾重新掃描所有滑動棋子。Stockfish 的做法更精確：當移除一個阻擋者後，直接將該阻擋者後方的潛在攻擊者加入列表。

4.  **王的安全檢查**:
    - 當前引擎在王進行捕捉時，雖有檢查邏輯，但不如 Stockfish 嚴謹（Stockfish 確保王捕捉後不會被任何敵方棋子攻擊，包括被牽制的棋子）。

5.  **輸入參數**:
    - 缺乏 `Move` 對象輸入，導致無法處理特殊移動標誌（如升變）。

---

## 3. 改進計畫 (Improvement Plan)

### 3.1 重構 `see.py`
1.  **更新函數簽名**: 修改 `see` 和 `see_ge` 接受 `move` (uint16) 參數，或者至少傳入 `promotion_type`。考慮到 Numba 性能，直接傳入解包後的 `from`, `to`, `promo`, `flag` 或許更快，或者在內部解碼。為了接口清晰，建議傳入 `move` (uint16) 並在內部解碼。
2.  **修復升變邏輯**: 在初始化 `value` 時，如果是升變移動，加上 `(Value(Promo) - Value(Pawn))`。
3.  **優化 X-Ray**: 採用類似 Stockfish 的邏輯，當從 `occupied` 移除一個棋子後，立即檢查該方格是否在滑動棋子的攻擊線上，若是，則補充攻擊者。
4.  **優化 Pin**: 嘗試簡化 `get_pinned_pieces`，或者僅在必要時計算。

### 3.2 參數調整
- 雖然 Stockfish 使用 208/781 等數值，但考慮到我們引擎的基礎評分體系（P=100），我們將保持現有比例，但確保邏輯一致。
- `PIECE_VALUES` 應與 `constants.py` 中的 `MG_MATERIAL_VALUES` 保持一致。

### 3.3 驗證
- 使用 `test_puzzle.py` 中的 FEN 進行對比測試。
- 確保 `see_ge` 的行為與 `see` 一致。
