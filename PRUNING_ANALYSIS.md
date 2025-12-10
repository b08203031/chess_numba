# Stockfish 剪枝與搜尋策略分析報告 (Analysis of Stockfish Pruning & Search Strategy)

本報告針對 Stockfish (SF) 引擎的原始碼進行了深入分析，重點在於其剪枝 (Pruning)、縮減 (Reductions) 與延伸 (Extensions) 的實作細節，並與目前的 Antares 引擎進行對比，找出結構性差異與改進方向。

## 1. 前向剪枝 (Forward Pruning)

### 1.1 Null Move Pruning (NMP)
*   **Stockfish 實作 (Step 9):**
    *   **縮減公式:** 動態計算。`R = (eval - beta) / 232 + depth / 3 + 5`。這意味著如果 Static Eval 超出 Beta 越多，縮減的深度越多。
    *   **條件:** `eval >= beta` 且 `staticEval >= beta - 19 * depth + 418`。這裡 SF 增加了一個基於 staticEval 的額外安全檢查。
    *   **驗證搜尋:** 如果 NMP 成功 (返回值 >= beta)，且 `depth >= 16` 或曾經被驗證過，則直接返回。否則，SF 會進行一次「驗證搜尋」(Verification Search)，即以降低的深度但不做 NMP 再搜一次，以避免 Zugzwang (逼走劣著) 的誤判。
*   **Antares 現狀:**
    *   使用固定的 `NULL_MOVE_REDUCTION` (通常為 3)。
    *   缺乏基於 `eval - beta` 的動態縮減。
    *   缺乏 `staticEval` 的額外安全檢查。
*   **改進建議:**
    *   將 NMP 的縮減深度改為動態公式：`R = 3 + depth / 3 + (eval - beta) / 200` (係數可調整，保持比 SF 寬鬆)。
    *   引入 Static Eval 的安全邊界檢查。

### 1.2 Futility Pruning (FP) & Reverse Futility Pruning (RFP/Static NMP)
*   **Stockfish 實作 (Step 8 & 14):**
    *   **Child Node (Step 8):** 使用公式 `eval - margin >= beta`。Margin 是 `(110 - 25 * noTtCutNode) * depth` 相關的線性/二次函數。
    *   **Parent Node (Step 14):** 這是傳統的 Static Null Move Pruning (RFP) 位置。SF 使用 `lmrDepth` 和 `history` 來調整判定。
    *   **Move Count Pruning:** 如果 `moveCount` 超過閾值 (基於深度的公式)，則跳過 Quiet Moves (Step 14)。
*   **Antares 現狀:**
    *   使用固定的 `FP_MARGIN_D1`, `FP_MARGIN_D2` 等常數。
    *   RFP (Static NMP) 實作較為簡單。
*   **改進建議:**
    *   改用基於深度的線性或二次公式來計算 Margin，而非針對特定深度 (D1, D2) 寫死。例如 `margin = base + multiplier * depth`。

### 1.3 Razoring
*   **Stockfish 實作 (Step 7):**
    *   如果 Static Eval 非常低 (`eval < alpha - margin`)，則直接進入 QSearch。
    *   Margin 公式：`461 + 315 * depth * depth`。
*   **Antares 現狀:**
    *   有 Razoring，但需確認 Margin 是否足夠寬鬆且隨深度適當增長。

### 1.4 ProbCut
*   **Stockfish 實作 (Step 11):**
    *   使用 `SEE` (Static Exchange Evaluation) 來篩選「好」的吃子 (Captures)。
    *   以 `depth - 4` 進行搜尋。
    *   如果 `value >= probCutBeta`，則剪枝。
*   **Antares 現狀:**
    *   有 ProbCut，邏輯類似。
*   **結構差異:** SF 在 ProbCut 內部使用了專門的 `MovePicker` 並進行了一次初步的 QSearch 驗證 (`qsearch` return `-beta` to verify hold) 之後才做 `search`。Antares 直接搜。

## 2. 搜尋縮減 (Search Reductions)

### 2.1 Late Move Reductions (LMR)
这是雙方差距最大的部分。

*   **Stockfish 實作 (Step 17):**
    *   **基礎公式:** `reductions[depth][moveCount]`。SF 預計算了一個對數表：`log(depth) * log(moveCount)`。
    *   **動態調整:** LMR 的幅度 `r` 會根據極多因素調整：
        *   **History Score:** `r -= statScore / 16384`。歷史分數高的步法，減少縮減；歷史分數低的，增加縮減。
        *   **PvNode / CutNode:** PV 節點縮減較少。
        *   **Capture / Check:** 根據是否吃子或叫將調整。
        *   **TT Move:** 如果是 TT Move 則減少縮減。
*   **Antares 現狀:**
    *   使用固定的 `LMR_REDUCTION` 常數。
    *   雖然有根據 King Tropism 調整，但缺乏與 **History Heuristic** 的深度整合。
*   **改進建議:**
    *   **結構性升級:** 實作基於 `log(depth)` 和 `log(move_count)` 的查表法或公式。
    *   **整合歷史:** 將 `history_table` 的分數正規化後，用於調整 LMR 的幅度 (減少或增加 1 ply)。

## 3. 搜尋延伸 (Extensions)

### 3.1 Singular Extensions (SE)
*   **Stockfish 實作 (Step 15):**
    *   如果 TT Move 的深度足夠且是 Lower Bound (或 Exact)，則嘗試排除該步法進行搜尋 (`excludedMove`)。
    *   如果排除後的搜尋結果仍低於 TT Value (減去 Margin)，代表 TT Move 是「唯一好棋」(Singular)，因此對其進行延伸 (`extension = 1` 或更多)。
    *   SF 還實作了 "Negative Extensions" (Multi-cut pruning)，如果排除後仍 Fail High，則可能直接剪枝。
*   **Antares 現狀:**
    *   有 SE。
*   **改進建議:**
    *   確認 Margin 的計算方式。SF 的 Margin 與 Depth 相關：`depth * constant`。Antares 應確保 Margin 足夠寬鬆。

## 4. 結構性差異總結 (Structural Gaps)

1.  **歷史啟發 (History Heuristics) 的複雜度:**
    *   SF 擁有 `CaptureHistory` (吃子歷史), `ContinuationHistory` (續著歷史 - 即前一兩步棋對應的歷史), `CorrectionHistory` (靜態評估修正歷史)。
    *   Antares 目前主要是 Butterfly History (`history_table`)。
    *   **影響:** 這些歷史表不僅影響排序，還直接影響 LMR 的縮減量。缺了這些，LMR 的精準度會下降。

2.  **Move Picker (步法選擇器):**
    *   SF 使用 `MovePicker` 類別按階段 (Stage) 生成步法 (TT -> Captures -> Good Quiets -> Bad Quiets)。這允許在特定階段 (如 Bad Quiets) 直接配合 LMP (Late Move Pruning) 跳過生成，效率極高。
    *   Antares 是一次生成所有步法，然後排序。這在大量 Quiet Moves 的局面下較慢。

3.  **Static Eval Correction:**
    *   SF 會根據搜尋結果動態修正 Static Eval (`CorrectionHistory`)，這有助於處理 Static Eval 長期高估或低估的局面。

## 5. 執行計畫 (Action Plan)

為了在不破壞穩定性的前提下提升棋力，我建議優先進行以下修改（保持參數寬鬆）：

1.  **改進 LMR (Late Move Reductions):**
    *   廢除單一常數 `LMR_REDUCTION`。
    *   引入基於 `depth` 和 `move_count` 的對數公式：`reduction = 1 + log(depth) * log(move_count) / 2` (係數可調，比 SF 寬鬆)。
    *   將 `history_table` 的分數整合進 LMR：歷史分數好則 `reduction - 1`，差則 `reduction + 1`。

2.  **改進 NMP (Null Move Pruning):**
    *   引入動態縮減公式：`R = 3 + depth / 6 + (eval - beta) / 200`。
    *   加入 Static Eval 的安全檢查：確保 `static_eval >= beta - margin` 才執行 NMP。

3.  **改進 Futility Pruning (FP) & Razoring:**
    *   將固定的 Margin 改為隨深度增長的公式，例如 `margin = 150 * depth`。

這些改動屬於「結構性調整」，能讓引擎的搜尋行為更像 Stockfish，但參數上我們保持保守，避免過度剪枝。
