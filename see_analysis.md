# Static Exchange Evaluation (SEE) 分析與改進報告

本報告詳細分析了 Stockfish 的 `see_ge` 函數與當前引擎 `see` 函數的差異，並探討 Stockfish 如何利用 SEE 進行搜索剪枝。

## 1. Stockfish SEE 函數分析 (`see_ge`)

Stockfish 的 SEE 實作位於 `position.cpp` 中的 `Position::see_ge`。

### 核心特點
1.  **Boolean 回傳值與閾值 (Threshold)**:
    *   Stockfish 不直接計算 SEE 的確切分數，而是檢查 SEE 分數是否 **大於等於** 某個閾值 (`threshold`)。
    *   這允許算法在確定結果滿足或不滿足條件時 **提前終止 (Early Cutoff)**，無需計算完整的交換序列。
2.  **迭代 Swap 算法**:
    *   使用一個 `swap` 變數來追蹤當前的材質平衡。
    *   邏輯類似於：`swap = 當前被吃子價值 - 之前的 swap`。
    *   如果 `swap` 已經無法影響最終結果（例如已經小於閾值且輪到對手，或者大於閾值且輪到自己），則提前返回。
3.  **顯式處理釘子 (Pins)**:
    *   在生成攻擊者時，Stockfish 會檢查攻擊者是否被釘住 (`pinners(~stm) & occupied`)。
    *   如果攻擊者被釘住且移動會暴露國王，則該攻擊者被從攻擊列表中移除，視為無法參與交換。
4.  **高效的攻擊者查找**:
    *   不使用迴圈遍歷所有兵種，而是依序檢查 P -> N -> B -> R -> Q -> K。
    *   使用 `if ((bb = stmAttackers & pieces(PAWN)))` 的方式，利用位運算快速找到價值最低的攻擊者。
5.  **動態 X-Ray 更新**:
    *   當一個棋子移動後，立即更新該位置的滑動棋子攻擊（X-Ray）。

## 2. 當前引擎 SEE 函數分析 (`see.py`)

### 核心特點
1.  **確切分數計算**:
    *   計算完整的交換序列，將每一步的材質增益存儲在 `gain` 陣列中。
    *   最後透過 Minimax 算法（反向傳播）計算最終分數。
2.  **無提前終止**:
    *   即使結果已經很明顯（例如丟后換兵），程式仍會嘗試模擬後續所有可能的吃子，直到沒有攻擊者或達到最大深度。
3.  **缺乏釘子處理**:
    *   目前的 `see` 函數僅考慮幾何上的攻擊關係，未考慮絕對牽制（Pin）。這可能導致引擎高估某些因為被釘住而實際上不可行的吃子或防守。
4.  **迴圈查找攻擊者**:
    *   使用 `for piece_type in range(PAWN, KING + 1)` 迴圈，這在 Numba 中雖然優化過，但相比展開的 `if` 判斷仍有額外開銷。

## 3. 差異與不足總結

| 特性 | Stockfish (`see_ge`) | 當前引擎 (`see`) | 差異影響 |
| :--- | :--- | :--- | :--- |
| **目標** | 判斷 Score >= Threshold | 計算 Exact Score | Stockfish 更適合剪枝判斷，速度更快。 |
| **效率** | 高 (Early Cutoff) | 中 (Full Evaluation) | 當前引擎在長序列交換中浪費計算資源。 |
| **釘子 (Pin)** | 處理 (考慮 Pin) | 未處理 (忽略 Pin) | 當前引擎可能高估被釘住棋子的防守能力，導致誤判。 |
| **數據結構** | 純量變數 (Scalar) | 陣列 (Array) | 陣列操作增加記憶體開銷與存取時間。 |
| **代碼結構** | 展開式 (Unrolled) | 迴圈 (Loop) | 展開式通常對 CPU 分支預測更友好。 |

## 4. Stockfish 的 SEE 剪枝應用

Stockfish 在搜索中廣泛使用 SEE 進行剪枝與排序。根據源碼分析 (`search.cpp`, `movepick.cpp`)：

### 4.1. Move Ordering (移動排序)
在 `movepick.cpp` 中：
*   **Good Captures vs Bad Captures**: 使用 `see_ge(move, threshold)` 來區分好壞吃子。通常 `threshold` 設為 0 或一個小的負值。
    *   `see >= 0`: Good Capture (優先搜索)
    *   `see < 0`: Bad Capture (推遲搜索)

### 4.2. Pruning (搜索剪枝)
在 `search.cpp` 中發現了以下應用條件（基於變數名推斷）：

1.  **Quiet Move Pruning / History Pruning Guard**:
    ```cpp
    if (!pos.see_ge(move, -154 * depth - seeHist))
        // Prune move
    ```
    *   **條件**: 如果一個安靜移動的 SEE 分數 **低於** 一個與深度和歷史分數相關的負閾值（即這個移動看起來虧損很多），則剪枝。
    *   **限制**: 這通常用於避免搜索那些明顯送子的移動。

2.  **LMR (Late Move Reduction) 相關**:
    ```cpp
    if (!pos.see_ge(move, -27 * lmrDepth * lmrDepth))
        // Do not reduce or Prune? (需確認上下文，通常是作為剪枝條件)
    ```
    *   這看起來像是在深層搜索時，如果移動虧損太多（即便考慮了 LMR 的寬容度），則直接剪枝或進行更重的減枝。

3.  **Futility Pruning (無效剪枝)**:
    ```cpp
    if (!pos.see_ge(move, alpha - futilityBase))
        // Prune
    ```
    *   **條件**: 確保移動的靜態交換價值至少能接近 Alpha。如果連 SEE 都遠低於 Alpha，則該節點不太可能產生 Cutoff。

4.  **Bad Capture Pruning**:
    ```cpp
    if (!pos.see_ge(move, -75))
        // Prune
    ```
    *   **條件**: 如果一個吃子看起來虧損超過 0.75 個兵（例如用輕子吃兵被吃回），則在某些階段（如 QS）直接剪枝。

## 5. 改進計畫

根據上述分析，我建議對 `chess_engine/see.py` 進行以下改進：

1.  **重寫 `see` 函數邏輯**:
    *   採用 Stockfish 的 "Swap List" 思想，但為了保持兼容性，初期仍回傳分數（或提供一個 `see_ge` 版本）。
    *   鑑於 Python/Numba 的特性，我們可以保留回傳確切分數的功能，但內部實現改為類似 Stockfish 的迭代更新，去除 `gain` 陣列，改用純量計算。
    *   **優化**: 移除 `minimax` 迴圈，改為在查找攻擊者的過程中直接維護 `balance`。

2.  **優化攻擊者查找**:
    *   將 `for` 迴圈展開為針對 P, N, B, R, Q, K 的獨立 `if` 塊。

3.  **關於釘子 (Pin) 的處理**:
    *   目前引擎架構中 `see` 函數是靜態方法，未傳入 `SearchContext` 或 `blockers` 位元棋盤。
    *   **決策**: 暫時 **不加入** 複雜的 Pin 檢查，因為重新計算 `blockers_for_king` 開銷過大。Stockfish 之所以能做是因為它在 `Position` 物件中維護了該狀態。
    *   **替代方案**: 保持目前的純幾何 SEE，但在調用 SEE 的地方（如搜索），通常已經過濾了非法移動（對於 Root Moves）。對於內部節點的偽合法移動，接受輕微的不準確性以換取速度。

4.  **引入 SEE 剪枝**:
    *   在 `search.py` 中引入 **Bad Capture Pruning** (在 QS 中)。
    *   在 `search.py` 中引入 **SEE Pruning for Quiet Moves** (如果 SEE < -Threshold * Depth 則剪枝)。

本報告建議優先優化 `see` 函數的寫法，使其更高效。
