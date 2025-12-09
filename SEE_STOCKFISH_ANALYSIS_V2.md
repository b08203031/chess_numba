# Stockfish SEE (Static Exchange Evaluation) 分析報告

## 1. 原理與實現方式

Stockfish 的 SEE 是一種用於靜態評估交換（Exchange）結果的算法，主要用於判斷一個特定的移動（通常是捕捉）是否在材質交換上獲利。

### 核心機制：Swap Algorithm
Stockfish 使用經典的 Swap Algorithm，模擬雙方輪流在目標格（Target Square）上進行捕捉，直到一方無法捕捉或選擇停止為止。

*   **遞歸/迭代模擬**：計算當前捕捉者的價值，減去隨後對手最佳回擊的價值（Minimax 原理）。
*   **數據結構**：使用一個 `values` 陣列（或堆疊）來存儲每一步捕捉後的材質得失（gain）。
    *   `values[0]`：初始捕捉的受害者價值。
    *   `values[i]`：第 `i` 步捕捉者的價值（即下一位受害者的價值）。
*   **最終分數計算**：
    ```cpp
    // 逆向傳播 Minimax
    for (int i = d - 1; i >= 1; --i)
        gains[i-1] = -std::max(-gains[i-1], gains[i]);
    return gains[0];
    ```

### 關鍵細節處理

1.  **X-Ray (透視攻擊) 處理**：
    *   當一個棋子（如車、象、后）移動後，它原來佔據的位置變為空，可能會暴露後方（Hidden）的遠程攻擊者。
    *   Stockfish 在移除當前攻擊者後，會立即檢查該攻擊者背後是否有同方向的 sliding piece，並將其加入攻擊者列表 (`attackers`)。
    *   *實現方式*：利用 `attacks_bb<BISHOP/ROOK>(to, occupied)` 快速查找新暴露的攻擊者。

2.  **Pin (牽制) 處理**：
    *   Stockfish 在 SEE 中採取了一種簡化的嚴格策略：**被牽制的棋子（Pinned Pieces）不參與捕捉**。
    *   它利用預先計算好的 `blockers_for_king` 位棋盤（Bitboard），在尋找攻擊者時直接過濾掉這些棋子：
        ```cpp
        stmAttackers &= ~blockers_for_king(stm);
        ```
    *   這比在 SEE 循環中每次進行射線檢測（Ray-Tracing）要快得多，雖然可能會漏掉沿著牽制線（Pin Ray）的合法捕捉，但換取了極高的效能。

3.  **En Passant (吃過路兵) 處理**：
    *   在 `see_ge`（快速版）中，Stockfish 通常直接忽略 EP 或將其視為平手/特殊情況處理，因為 EP 較罕見。
    *   在 `see`（精確版）中，它將被吃掉的兵視為 `PawnValue`。
    *   *特別注意*：Stockfish 的 SEE **不會** 從 `occupied` 位棋盤中移除被吃掉的過路兵（因為它不在目標格 `to` 上）。這意味著它**不考慮**穿過被吃掉兵的 X-Ray 攻擊。這是一個已知的近似處理。

4.  **國王處理**：
    *   如果在交換序列中輪到國王捕捉，且捕捉後國王會被攻擊（送吃），則該分支立即終止（不進行該捕捉）。

## 2. 應用場景

Stockfish 在引擎的多個關鍵部分使用了 SEE，主要分為「精確值使用 (`see`)」和「閾值檢查 (`see_ge`)」。

### A. Move Ordering (著法排序)
在 `movepick.cpp` 中，SEE 用於區分好壞捕捉（Good vs Bad Captures）：
*   **Good Captures**：SEE >= 0（或某個負閾值）。優先搜索。
*   **Bad Captures**：SEE < 0。推遲搜索（Losing Captures）。
*   這能顯著減少 Alpha-Beta 搜索的分支因子。

### B. Pruning & Reduction (剪枝與縮減)
在 `search.cpp` 中，SEE 被大量用於決定是否放棄搜索某個著法：

1.  **ProbCut (Probability Cut)**：
    *   只對 SEE >= `threshold` 的捕捉進行快速的淺層搜索驗證。
2.  **Static NMP (Null Move Pruning)**：
    *   如果靜態評估極差，可能會用到 SEE 輔助判斷（較少見，主要是 Eval）。
3.  **LMR (Late Move Reduction)**：
    *   對於 SEE 為負的 Quiet Moves 或 Captures，可能會增加縮減深度。
4.  **Quiescence Search (靜止搜索)**：
    *   **SEE Pruning**：如果一個捕捉的 SEE 遠低於 Alpha（例如 `see < alpha - margin`），則直接剪枝，不進行搜索。
    *   **Stand Pat**：在 QS 中也使用 SEE 來過濾掉明顯虧損的捕捉（如用后吃被保護的兵）。
5.  **Futility Pruning**：
    *   使用 SEE 判斷歷史著法是否值得嘗試。

### C. Evaluation (評估)
*   雖然主要在搜索中，但 SEE 的概念有時也會用於評估某些戰術威脅的嚴重性（雖然 Stockfish 的 Eval 目前主要依賴 NNUE，但在舊的手寫 Eval 中有使用）。

## 3. Stockfish 與 我們引擎 (Antares) 的對比與改進建議

### 目前引擎 (`chess_engine/see.py`) 的不足：

1.  **Pin 檢測效率低**：
    *   **現狀**：目前是在 `while` 循環中，每次選出 LVA (Least Valuable Attacker) 後，都執行一次射線檢測 (`get_sliding_attacks`) 來判斷該棋子是否被牽制。
    *   **問題**：這涉及重複的循環運算，對於 Numba 雖然有優化，但仍是熱點（Hotspot）。
    *   **改進**：雖然我們沒有像 Stockfish 那樣維護全局 `blockers_for_king`，但可以在 SEE 函數開頭（或循環外）一次性計算相關的 Pin 資訊，或者優化檢測邏輯（例如只檢測與國王在同一條線上的棋子）。鑑於 `see` 是純函數，傳入全局狀態可能複雜，我們將維持局部計算但優化其觸發條件。

2.  **Piece Type Lookup (棋子類型查找) 效率**：
    *   **現狀**：使用 `find_piece_type_on_square`，需要遍歷 6-12 個位棋盤來確定一個格子上的棋子類型。
    *   **問題**：這是 O(N) 操作，在 SEE 的每一步都要執行（查找 LVA）。
    *   **改進**：利用 `occupancy_bbs` 先判斷顏色，減少一半的檢查次數。

3.  **En Passant 精確度**：
    *   **現狀**：目前的邏輯判斷了 EP，但可能未完全對齊 Stockfish 的「不移除過路兵阻擋」的近似邏輯（這其實是優化）。
    *   **改進**：確保邏輯簡單且與 Stockfish 一致（即只將目標格 `to` 視為捕捉價值來源，不處理特殊的 `to - pawn_push` 阻擋位）。

4.  **數值系統 (Values)**：
    *   **現狀**：使用 `[100, 320, 330, 500, 900]`。
    *   **Stockfish**：使用 `[208, 781, 825, 1276, 2538]`。
    *   **建議**：暫時保持我們的數值系統以配合 Evaluation，但在 SEE 內部計算時，比例差異不大，不會嚴重影響正負判斷。重點是算法邏輯。

### 改進計畫

1.  **重構 `see` 函數**：
    *   優化 LVA 選擇循環。
    *   優化 Pin 檢測：只對與 King 處於 Ray 上的棋子進行詳細 Pin Check。
    *   優化 X-Ray 添加邏輯。
    *   實現 `see_ge`：Stockfish 有一個專門的 `see_ge` 函數，它在達到閾值時會提前終止（Early Exit），這比算出精確值再比較要快得多。我們應該實現這個版本並用於 Move Ordering 和 Pruning。

2.  **代碼移植策略**：
    *   保留 Numba `@njit`。
    *   盡量減少 NumPy 陣列分配（使用固定大小數組或標量）。
    *   引入 `see_ge` 用於搜索剪枝。

---

接下來我將依照此報告，對 `chess_engine/see.py` 進行代碼優化。
