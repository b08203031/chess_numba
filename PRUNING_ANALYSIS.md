# Stockfish 剪枝技術與搜尋優化分析報告

本報告分析了 Stockfish 16 (基於提供的原始碼) 的剪枝與搜尋優化技術，並與目前的 Antares 引擎 (`chess_engine/search.py`) 進行對比。報告旨在指出我方引擎的不足之處，並提出改進建議。

## 1. 核心機制差異

### 1.1 Improving 旗標 (Improving Flag)
*   **Stockfish**: 在搜尋每一層節點時，會計算 `improving = ss->staticEval > (ss - 2)->staticEval`。這代表目前的局面評估比上一步（己方上一次移動後）更好。
*   **應用**: `improving` 旗標廣泛應用於幾乎所有剪枝決策：
    *   **放寬剪枝**: 如果局面在變好，我們可以更大膽地剪枝（例如 LMP 剪更多步）。
    *   **收緊剪枝**: 如果局面在變差（`!improving`），則某些剪枝會被禁用或限制，以免漏掉翻盤的機會。
*   **Antares**: **尚未實作**。這是一個重大缺失，導致剪枝缺乏對局面趨勢的動態感知。

### 1.2 歷史啟發與剪枝的交互 (History Interaction)
*   **Stockfish**: 在決定是否進行剪枝（如 Futility Pruning, LMP, SEE Pruning）時，會檢查該步棋的 History 分數。如果某步棋歷史表現很好，即使其他條件滿足剪枝，也可能被保留。
*   **Antares**: 目前剪枝邏輯大多只看 Static Eval 或 Capture 狀態，較少結合 History 表格進行判斷。

---

## 2. 剪枝技術分析

### 2.1 Null Move Pruning (NMP) - 空步剪枝
*   **Stockfish**:
    *   **動態縮減 (Dynamic Reduction)**: `R` 值不是固定的，而是基於 `(eval - beta)` 的深度和 `depth` 計算：`R = (eval - beta)/232 + depth/3 + 5` (大致公式)。
    *   **適應性條件**: 要求 `staticEval >= beta - adaptive_margin`，確保靜態評估也足夠好。
    *   **驗證搜尋 (Verification Search)**: 在高深度或特定條件下，如果 NMP 剪枝了，會進行一個降低深度的驗證搜尋，確保不是 False Positive。
*   **Antares**:
    *   使用固定的 `NULL_MOVE_REDUCTION = 2`。
    *   缺乏 Static Eval 的保護條件。
    *   無驗證搜尋機制。

### 2.2 Late Move Pruning (LMP) - 晚期移動剪枝
*   **Stockfish**:
    *   **公式**: `moveCount > (3 + depth*depth) / (2 - improving)`。
    *   **邏輯**: 深度越深，搜尋的步數越多；但如果局面正在變好 (`improving=True`)，分母變為 1，允許搜尋的步數變少（剪枝更積極）。
*   **Antares**:
    *   使用查表 `LMP_MOVE_COUNT` (`20 + 20*depth`)。
    *   邏輯較為線性且寬鬆，且未考慮 `improving`。

### 2.3 Late Move Reduction (LMR) - 晚期移動縮減
*   **Stockfish**:
    *   **對數公式**: 縮減值基於 `log(depth)` 和 `log(moveCount)` 的乘積表。這意味著隨著步數索引增加，縮減幅度會呈現非線性增長。
    *   **動態調整**:
        *   `improving` 影響縮減量。
        *   `Cut Node` (預期會發生 Beta Cutoff 的節點) 會增加縮減量。
        *   `History` 表現好的棋子減少縮減量。
        *   `Capture` / `Check` 通常不縮減。
*   **Antares**:
    *   使用固定的 `LMR_REDUCTION = 1`。
    *   條件較為單一 (`depth >= LMR_MIN_DEPTH` 等)。

### 2.4 Futility Pruning (FP) & Razoring
*   **Stockfish**:
    *   **Razoring**: 當評估值極低時直接進入 QSearch。
    *   **Futility**: 區分為 Parent Node (Quiet moves) 和 Child Node。
    *   **Margin**: 使用線性公式 `k * depth`，但會根據 `improving` 旗標動態調整 Margin 大小。
*   **Antares**:
    *   有實作 Razoring 和 Futility。
    *   但 Margin 是固定的常數 (`FP_MARGIN_D1` 等)，缺乏深度縮放（除了針對不同深度定義不同常數）與動態調整。

### 2.5 Static Exchange Evaluation (SEE) Pruning
*   **Stockfish**:
    *   對 Quiet Moves 也進行 SEE 檢查（通常檢查是否 `< 0` 或更低的負閾值）。
    *   閾值是動態的：`-27 * lmrDepth^2`。
*   **Antares**:
    *   主要用於 Quiescence Search 和 Move Ordering。
    *   主搜尋中的 SEE Pruning 閾值是固定的 `-100`。

### 2.6 Singular Extensions (SE)
*   **Stockfish**:
    *   邏輯非常複雜，包含 Negative Extensions（當搜尋失敗時減少深度）。
    *   延伸深度 (`newDepth`) 和搜尋邊界 (`singularBeta`) 都有精細的公式。
*   **Antares**:
    *   有實作基本的 SE。
    *   目前的 `depth - 3` 和固定 Margin 是一個簡化的版本。

---

## 3. 改進建議 (Action Plan)

根據您的要求「參數先寬鬆」、「不更動現有參數常數」，建議採取以下改進策略：

### 第一階段：引入核心機制與邏輯補強
1.  **實作 `improving` 旗標**：
    *   在 `_search` 中追蹤 `ss->staticEval` 與 `(ss-2)->staticEval` 的關係。
    *   這不需要新參數，但需要調整 `SearchContext` 或遞迴傳遞結構。

2.  **優化 LMR 邏輯 (不改動常數，改動公式結構)**：
    *   保留 `LMR_REDUCTION` 作為基底。
    *   引入 `History` 權重：如果歷史分數低，增加縮減；歷史分數高，減少縮減。
    *   引入 `improving`：如果 `!improving`，減少縮減量（更謹慎）。

3.  **優化 NMP 邏輯**：
    *   加入 Static Eval 保護條件：`static_eval >= beta - variable_margin`。
    *   這個 `variable_margin` 可以設定得較寬鬆。

4.  **優化 LMP 邏輯**：
    *   改用公式計算 `move_count` 閾值，納入 `depth` 與 `improving`。
    *   建議公式：`(const_base + depth * depth) / (2 if improving else 1)`。

### 第二階段：細節參數調整 (未來)
*   待邏輯架構完善後，再將固定的 `FP_MARGIN` 等常數改為基於深度的公式。

## 4. 結論
Antares 目前的剪枝架構是完整的，但缺乏**動態性**。Stockfish 的強大之處在於它能根據局面的趨勢 (`improving`) 和歷史資訊 (`history`) 動態調整剪枝的力度。本次改進的重點將放在引入這些動態判斷因子。
