# 搜尋算法分析：理念、意義、細節與方法論

本文件對分拆包中實現的搜尋算法（[search.py](search.py) 與 [nnue/search.py](../nnue/search.py)）及相關模組（如 [constants.py](constants.py)、[transposition_table.py](transposition_table.py)）進行了全面的分析。它闡明了底層邏輯（理念）、參數的意義、具體的實現選擇以及整體的算法方法論。

---

## 1. 總覽

引擎在**迭代加深 (Iterative Deepening)** 框架內採用了**主要變例搜尋 (Principal Variation Search, PVS)**。這是現代西洋棋引擎的標準做法，通過優先嘗試最優可能的移動（主要變例，PV），並以極低的開銷證明其他移動皆劣於當前最優解（零窗口搜尋，Zero Window Search），從而高效地探索博弈樹。

**核心檔案：**
*   [search.py](search.py) / [nnue/search.py](../nnue/search.py)：核心搜尋邏輯（迭代加深、PVS、靜態搜尋）。
*   [constants.py](constants.py) / [nnue/constants.py](../nnue/constants.py)：調優參數、剪枝邊界與權重。
*   [transposition_table.py](transposition_table.py) / [nnue/transposition_table.py](../nnue/transposition_table.py)：置換表（哈希表）的實現。

---

## 2. 核心算法

### 2.1 迭代加深 (Iterative Deepening)
*   **理念**：以遞增的深度（深度 1, 2, 3...）逐步進行搜尋，而非直接搜尋到固定深度。
*   **意義**：使引擎在時間耗盡時能隨時返回一個「最佳移動」。更重要的是，它通過淺層搜尋填充了置換表 (TT) 和歷史啟發 (History Heuristic)，這極大地改善了深層搜尋時的移動排序（Move Ordering）。
*   **細節**：
    *   實現於 `iterative_deepening_search` 函數。
    *   使用以先前迭代分數為中心的**抱負窗口 (Aspiration Windows)** (`alpha`, `beta`) 來收窄搜尋窗口並增加截斷。
    *   在迭代之間進行時間管理檢查。

### 2.2 主要變例搜尋 (Principal Variation Search, PVS) / Negamax
*   **理念**：假設第一步棋（排序最好的一步）是主要變例 (PV)，使用全窗口 `(alpha, beta)` 對其進行搜尋。隨後假設所有其他移動皆較差，使用「零窗口」或「空窗口」`(alpha, alpha+1)` 進行搜尋，以低成本證明它們會低於預期（Fail Low）。
*   **意義**：如果移動排序品質高，這將大幅降低博弈樹的有效分支因子。
*   **細節**：
    *   實現於遞迴搜尋函數 `_search` 中。
    *   如果 `searched_move_count > 1`，則執行零窗口搜尋 (`-alpha - 1, -alpha`)。
    *   如果零窗口搜尋意外產生高出邊界（Fail High，返回分數 > alpha），說明最初的假設錯誤，必須使用全窗口重新搜尋 (`-beta, -alpha`)。

### 2.3 靜態搜尋 (Quiescence Search, QS)
*   **理念**：在葉子節點（深度 0）擴展搜尋以抵達「靜態局面」（即不存在吃子或重大威脅的局面），以緩解「地平線效應 (Horizon Effect)」。
*   **意義**：防止引擎在吃子交換序列的中途停止搜尋（例如在 QxP 之後但 recapturing PxQ 之前停下，導致對局勢產生嚴重誤判）。
*   **細節**：
    *   實現於 `quiescence_search`。
    *   僅考慮吃子（若被將軍，則額外考慮逃避將軍的偽合法移動）。
    *   引入**增益剪枝 (Delta Pruning)**：如果當前評估極差，即使吃掉對手最大的子（如后）也無法挽回，則直接剪枝。
    *   引入**SEE 剪枝 (SEE Pruning)**：在靜態搜尋中直接剪除不划算的吃子移動 (SEE < 0) 以節省時間。

---

## 3. 移動排序 (Move Ordering)

移動排序是決定 Alpha-Beta 搜尋效率的單一最關鍵因素。引擎採用以下排序結構：

1.  **置換表移動 (TT Move)**：先前搜尋或迭代中記錄的最佳移動。此移動一律最先嘗試。
2.  **吃子移動 (Captures)**：
    *   根據 **MVV-LVA**（最有價值受害者 - 最無價值攻擊者）進行排序。
    *   **划算吃子 (Good Captures)**：優先考慮 SEE >= 0 的吃子移動。
    *   **不划算吃子 (Bad Captures)**：SEE < 0 的吃子移動會被降級，稍後再搜尋。
3.  **殺手步 (Killer Moves)**：在兄弟節點中，在同一深度下產生 Beta 截斷的兩個安靜移動。
4.  **反擊步 (Counter Move)**：記錄對手前一步棋在歷史上最常引起截斷的回應步。
5.  **歷史啟發 (History Heuristic)**：按歷史成功率（引起截斷的頻率相對於深度）排序的安靜移動。
    *   使用「重力公式」衰減舊歷史並限制最大範圍：`history += bonus - history * abs(bonus) / MAX_HISTORY`。

---

## 4. 剪枝與歸約 (Pruning and Reductions)

引擎實現了一套激進的剪枝技術來減少搜尋空間。

### 4.1 空著剪枝 (Null Move Pruning, NMP)
*   **理念**：如果我們跳過自己的一步棋（下空著），而對手**仍然**無法將局面評分提升到 Beta 以上，說明我們佔據了巨大的優勢，可以直接進行截斷。
*   **邏輯**：
    *   條件：`static_eval >= beta` 且 `depth >= 3`。
    *   操作：以降低的深度進行搜尋。
    *   驗證：如果返回分數 `>= beta`，則直接返回 Beta（截斷）。
    *   **安全性**：被將軍時或殘局中不套用（以防雙向無子限制 zugzwang 危害）。

### 4.2 反向徒勞剪枝 (Reverse Futility Pruning, RFP) / 靜態空著剪枝
*   **理念**：如果靜態評估值非常高，以至於減去一個很大的安全邊界後依然高於 Beta，我們可以直接進行截斷。
*   **邏輯**：
    *   適用於 `depth <= 2`。
    *   條件：`static_eval - margin >= beta`。

### 4.3 剃刀剪枝 (Razoring)
*   **理念**：如果在極淺的深度下靜態估值顯著低於 Alpha，我們假設該分支無法挽回，提前進入靜態搜尋以節省時間。
*   **邏輯**：
    *   適用於淺深度。
    *   條件：`static_eval + margin < alpha`。
    *   操作：調用靜態搜尋驗證，若仍低於 Alpha 則直接返回 Alpha。

### 4.4 徒勞剪枝 (Futility Pruning, FP)
*   **理念**：在淺層深度搜尋時，如果靜態評估加上與深度正相關的邊界後仍然低於 Alpha，則直接跳過剩餘的安靜移動。
*   **邏輯**：
    *   條件：`static_eval + margin < alpha`。
    *   操作：跳過當前的安靜移動。

### 4.5 晚期移動剪枝 (Late Move Pruning, LMP)
*   **理念**：如果在一個節點中已經搜尋了大量的安靜移動卻仍未找到更好的移動，那麼剩餘排序極後的安靜移動不太可能有所幫助。
*   **邏輯**：
    *   條件：已嘗試的安靜移動數超過特定深度的閾值。
    *   操作：直接忽略該節點剩餘的所有安靜移動。

### 4.6 概率剪枝 (ProbCut)
*   **理念**：空著剪枝的概率版。使用一個更寬的窗口和較淺的深度進行搜尋，以預測當前移動是否極有可能引發強烈的截斷。

### 4.7 晚期移動歸約 (Late Move Reduction, LMR)
*   **理念**：對於排序靠後（可能較差）的移動，以降低的深度進行搜尋。
*   **邏輯**：
    *   條件：`depth >= 3`，移動是安靜移動，且移動索引較大。
    *   歸約：基於 precomputed LMR 表格查表。
    *   調整：受歷史分數調節（歷史表現好的移動減少歸約，表現差的增加歸約）。
    *   例外：給予將軍、升變等強迫性移動通常不進行歸約。

### 4.8 單一擴展 (Singular Extensions, SE)
*   **理念**：如果 TT 移動顯著好於其他所有候選移動（單一優勢），則將該節點的搜尋深度擴展 1 層，以確保計算精準度。

### 4.9 內部迭代加深 (Internal Iterative Deepening, IID)
*   **理念**：如果當前節點沒有 TT 移動（例如是新節點），則先以較淺的深度搜尋該節點，以獲得一個合適的移動進行排序，再進行完整搜尋。

---

## 5. 置換表 (Transposition Table, TT)

*   **理念**：一個哈希表，用於存儲搜尋過的節點結果，以便在不同路徑到達相同局面時直接復用。
*   **實現細節**：
    *   **鍵值**：Zobrist Hash (64位)。
    *   **結構**：`key`, `score`, `depth`, `flag` (Exact/Alpha/Beta), `generation`, `best_move`, `static_eval`。
    *   **替換策略**：
        1.  **世代 (Generation)**：優先替換來自舊搜尋世代的條目。
        2.  **深度 (Depth)**：在相同世代中，優先保留深度較深的條目。
    *   **優化**：在 Numba 內以一維結構體數組的形式高效訪問。
    *   **分數調整**：將軍殺死的分數在寫入表前會轉換為相對於根節點的絕對值，在讀出時再轉換回相對於當前層數 (ply) 的分數。

---

## 6. 時間管理

*   **理念**：根據剩餘時間、增量時間以及局面的複雜程度分配搜尋時間。
*   **邏輯**：
    *   **軟限制 (Soft Limit)**：`optimum_time`。若超出，在下一個搜尋層級結束時退出。
    *   **硬限制 (Hard Limit)**：`maximum_time`。一旦超出，立即中斷搜尋並設置 `stop_flag`。
    *   **預測性退出**：如果在當前深度迭代中已經耗費了大量時間（例如已達極限的 40% 以上），預測下一層迭代無法在硬限制前完成，則提前中斷，避免浪費時間。

---

## 7. 重複偵測 (Repetition Detection)

*   **理念**：檢測三手重複或 50 步規則以正確將和棋局面評分為 0.00。
*   **邏輯**：
    *   檢查搜尋路徑棧 `ply_path_stack` 以及搜尋前的歷史對局記錄 `game_history`。
    *   **優化**：僅檢查行棋方相同的局面（每隔兩步檢查一次：`ply-2, ply-4` 等）。
    *   **限制**：檢查範圍受半步計數器 (halfmove clock) 限制，任何兵的推進或吃子操作都會重置重複檢查起點。

---

## 8. 分析與結論

Antares 引擎的搜尋算法架構現代且完備，包含了當今最先進的剪枝和歸約啟發式方法。通過 Numba (JIT) 進行底層編譯後，搜尋邏輯在 Python 環境中能展現出極佳的每秒搜尋節點數 (NPS)。
