# Stockfish SEE 分析與改進報告

本報告詳細分析了 Stockfish 的靜態交換評估 (Static Exchange Evaluation, SEE) 函數的原理、實現方式、應用場景，並對比了目前引擎的實現，提出了重寫與改進方案。

## 1. Stockfish SEE 分析

### 1.1 算法原理與實現 (`position.cpp`: `see_ge`)

Stockfish 使用一個名為 `see_ge` (Greater or Equal) 的函數，它不直接返回具體的分數，而是檢查 SEE 分數是否大於等於某個閾值 (`threshold`)。這種設計允許在某些情況下提早終止計算（雖然主要算法是迭代式的）。

核心算法是一個迭代的 "Swap List" (交換列表) 方法：

1.  **初始交換**: 計算 `swap = 價值(被吃子) - threshold`。如果小於 0，直接返回 false（這是一個快速檢查，但邏輯上是因為如果連拿掉被吃子的價值都不夠抵扣閾值，那肯定不行）。
2.  **迭代模擬**: 模擬每一輪的吃子 (Capture)。
3.  **找最小攻擊者 (LVA)**: 使用 `stmAttackers & pieces(PAWN)` 等位運算，按 P->N->B->R->Q->K 的順序尋找當前行棋方的最小價值攻擊者。
4.  **X-Ray 處理**: 當一個滑子 (R/B/Q) 移動後，它背後的滑子可能會成為新的攻擊者。Stockfish 通過 `occupancy ^= lsb` 更新佔用位元板，並增量更新 `attackers` (只為滑子)。
5.  **Pin (牽制) 處理 (關鍵)**:
    *   Stockfish 明確檢查攻擊者是否被牽制：`if (pinners(~stm) & occupied)`。
    *   如果攻擊者被牽制且移動會暴露國王，則該攻擊者被從候選列表中移除 (`stmAttackers &= ~blockers_for_king(stm)`)。
    *   這是目前我方引擎最欠缺的部分，導致 SEE 可能使用非法移動進行吃子。
6.  **國王安全**: 如果國王參與吃子，但對方仍有攻擊者，則判定該分支失敗（因為國王不能被吃）。

### 1.2 性能優化

*   **位元板操作**: 大量使用位元板運算 (`&`, `^`, `|`) 而非列表迭代。
*   **增量更新**: 不會在每一步都重新掃描整個棋盤的攻擊者。它維護一個 `attackers` 位元板，並只在滑子移動時更新 X-Ray 攻擊。
*   **LSB 提取**: 使用 `least_significant_square_bb` 快速定位棋子。

### 1.3 應用場景

Stockfish 在多個地方使用了 SEE，主要是為了剪枝和排序：

1.  **Move Ordering (著法排序)**:
    *   **Good Captures**: `see_ge(move, -threshold)` (閾值動態調整)。SEE 值高的吃子被優先搜索。
    *   **Bad Captures**: SEE 值過低的吃子被推遲到最後搜索。
2.  **Quiescence Search (靜態搜索)**:
    *   **Pruning**: 如果 `!see_ge(move, -75)` (非常差的吃子)，則直接剪枝，不進行搜索。
    *   **Futility**: 如果 `!see_ge(move, alpha - futilityBase)`，則剪枝。
3.  **Shallow Pruning (淺層剪枝)**:
    *   在主搜索中，對於捕捉和將軍的移動，如果 SEE 值非常低 (`!see_ge(move, -154 * depth - seeHist)`)，則進行剪枝。
4.  **LMR (Late Move Reduction)**:
    *   在高深度下，如果移動的 SEE 為負 (`!see_ge(move, -27 * depth^2)`)，則可能被剪枝或減少搜索。

---

## 2. 當前引擎 SEE 分析

### 2.1 現狀 (`chess_engine/see.py`)

*   **算法**: 使用類似的 Minimax 遞歸思想（但展開為循環），計算 `gain` 數組。
*   **返回類型**: 返回精確的整數分數。
*   **缺陷**:
    1.  **缺少 Pin 處理**: 這是最大的正確性問題。被牽制的棋子（如絕對牽制）不應能移動去吃子，但目前 SEE 會將其視為有效攻擊者。這會導致高估防守方的能力。
    2.  **缺少國王安全檢查**: 沒有像 Stockfish 那樣處理 "國王吃子後仍被攻擊" 的情況。
    3.  **效率較低**: 每次調用 `_get_attackers_to_square` 都會重新計算所有棋子的攻擊，且在循環內部處理 X-Ray 時使用了較重的 `get_bishop_attacks` 全量查詢。
    4.  **En Passant**: 目前直接返回 0，處理較粗糙。

---

## 3. 改進與重寫計畫

目標是重寫 `chess_engine/see.py`，在 Python/Numba 架構下儘可能模仿 Stockfish 的邏輯與精確度。

### 3.1 核心 SEE 函數重寫

將實現一個新的 `see` 函數（返回分數）和 `see_ge` (可選，若為了性能)。鑑於 Python 調用開銷，我們主要優化 `see` 函數本身。

**新邏輯特點**:
1.  **引入 `pinners`**: 在 `see` 內部（或由調用者傳入）需要獲取牽制信息。考慮到計算成本，我們將在 `see` 內部根據需要計算，或者利用 `bitboard_utils` 優化計算。
    *   *技術難點*: Stockfish 的 `Position` 對象緩存了 `blockers_for_king`。我們的引擎是無狀態的 bitboard 傳遞。
    *   *解決方案*: 在 `see` 開始時，如果涉及滑子防守，快速計算或估算牽制。或者，更準確地，我們需要實現 `get_pinners`。為了性能，我們可以只在 "國王在攻擊線" 上時才進行精確檢查。
2.  **迭代 Swap 算法**: 放棄 `gain` 數組的遞歸回溯寫法（雖然等價），改用 Stockfish 的 `swap` 變量迭代寫法，這樣更易於理解和加入 `see_ge` 的閾值邏輯。
3.  **增量 X-Ray**: 優化 X-Ray 的更新邏輯，避免全量重算。

### 3.2 擴展應用場景 (Pruning)

將在 `chess_engine/search.py` 中加入以下基於 SEE 的剪枝（參數將比 Stockfish 寬鬆，以適應我們引擎的評估尺度）：

1.  **QSearch Pruning**:
    *   剪枝 SEE < -100 (centipawns) 的移動。
    *   (可選) 基於 Alpha 的動態剪枝。
2.  **Main Search Pruning**:
    *   在淺層搜索中，剪枝 SEE 極差的吃子 (例如 -200 * depth)。

### 3.3 代碼結構變更

1.  **`chess_engine/bitboard_utils.py`**:
    *   新增 `BETWEEN_BB` 查找表（用於判斷兩格之間是否有阻擋，輔助 Pin 計算）。
    *   新增 `get_blockers_and_pinners` 函數。
2.  **`chess_engine/see.py`**:
    *   完全重寫。

---

## 4. 具體參數調整

Stockfish 使用的參數非常精細，我們的引擎目前評估分數尺度約為 centipawns。
*   **Good Capture Bonus**: Stockfish 動態計算，我們保留現有的 `SCORE_GOOD_CAPTURE_BONUS` 但引入 SEE 值作為微調。
*   **Pruning Margin**:
    *   Stockfish: `-75` (QSearch bad capture).
    *   Antares (Proposed): `-100` (稍微保守一點).

準備開始執行代碼修改。
