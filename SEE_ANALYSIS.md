# 分析 Stockfish SEE (Static Exchange Evaluation) 函數與改進報告

## 1. Stockfish SEE 算法分析

### 1.1 核心邏輯
Stockfish 的 SEE 實現主要位於 `Position::see_ge` 函數中。它不是計算精確的分數，而是判斷交換後的結果是否大於等於某個閾值 (`threshold`)。

**算法步驟：**

1.  **快速剪枝：** 計算 `swap = victim_value - threshold`。如果 `swap < 0`，則直接返回 `false`。
2.  **初始交換：** `swap = attacker_value - swap`。如果 `swap <= 0`，則初始攻擊方直接獲利（或虧損在可接受範圍內），返回 `true`。
3.  **迭代模擬：**
    *   **尋找最小攻擊者 (Least Valuable Attacker)：** 按照 Pawn -> Knight -> Bishop -> Rook -> Queen -> King 的順序尋找己方攻擊者。
    *   **Pin 檢測 (關鍵)：** 這是 Stockfish 精確度的核心。它會檢查攻擊者是否被 Pin 住（即移動後會暴露己方 King 於攻擊之下）。
        *   `if (pinners(~stm) & occupied)`: 如果存在敵方的 Pin 棋子。
        *   `stmAttackers &= ~blockers_for_king(stm)`: 將被 Pin 住的棋子從攻擊者列表中移除。
    *   **移除攻擊者：** 將攻擊者從 `occupied` 位元板中移除。
    *   **X-Ray 處理：** 當一個棋子移動後，它後方的滑子（Rook/Bishop/Queen）可能會成為新的攻擊者。
        *   `attackers |= attacks_bb<BISHOP/ROOK>(to, occupied) & pieces(BISHOP/ROOK, QUEEN)`。
        *   Stockfish 動態地將這些新暴露的攻擊者加入 `attackers` 位元板。
    *   **更新 Swap 值：** `swap = piece_value - swap`。
    *   **剪枝：** 如果 `swap < res` (當前最優結果)，則停止搜尋。

### 1.2 應用場景

Stockfish 在以下場景使用 SEE：

1.  **Move Ordering (Good vs Bad Captures):**
    *   在 `MovePicker` 中，Stockfish 使用 `see_ge` 來區分 "Good Captures" 和 "Bad Captures"。
    *   `pos.see_ge(*cur, -cur->value / 18)` (這是一個啟發式的閾值，並非單純的 0)。
    *   Good Captures 會在 Killer Moves 之前被搜尋，Bad Captures 則在 Quiet Moves 之後。

2.  **Pruning (剪枝):**
    *   **ProbCut:** 在 `search.cpp` 中，ProbCut 只對 `see_ge(move, threshold)` 為真的 Capture 進行嘗試。
    *   **Quiescence Search (靜態搜尋):** 雖然現代 Stockfish 主要依賴 `MovePicker` 的排序，但在某些版本或變體中，SEE < 0 的 Capture 可能會被直接剪枝（SEE Pruning）。

3.  **LMR (Late Move Reduction):**
    *   SEE 的結果會影響 LMR 的深度縮減量。

---

## 2. 現有引擎的不足

經過分析，我們的 Python/Numba 引擎在 SEE 方面存在以下不足：

1.  **代碼缺失：** 雖然 `search.py` 引用了 `see`，但 `chess_engine/see.py` 文件實際上不存在（可能在重構中遺失），導致無法運行或潛在錯誤。
2.  **Pin 處理缺失 (假設)：** 舊的 SEE 實現（根據記憶）通常缺乏對 Pinned Pieces 的處理。如果一個被 Pin 的 Pawn 參與交換，SEE 會錯誤地認為它可以移動，導致評估過於樂觀。
3.  **X-Ray 效率：** 在 Python/Numba 中，動態計算 X-Ray 需要高效的位元運算。簡單的循環可能效率低落。
4.  **接口不匹配：** Stockfish 使用 `see_ge` (Boolean)，我們目前在 `search.py` 中依賴返回具體數值的 `see` 函數來進行排序 (`score_moves`)。

---

## 3. 改進方案

為了在 Numba 架構下達到接近 Stockfish 的精確度，同時保持高效，我將重新實現 `chess_engine/see.py`。

### 3.1 算法移植策略

1.  **實現 `see` (返回數值)：** 為了兼容現有的 `score_moves` 邏輯，我將實現返回具體值的版本，而不是 `see_ge`。
2.  **精確的 Pin 處理：**
    *   由於我們沒有像 Stockfish 那樣實時維護 `blockers_for_king`，我將在 `see` 函數內部進行簡化的 Pin 檢測。
    *   **優化：** 只在涉及滑子攻擊線上的棋子時才檢查 Pin，或者利用現有的 `is_square_attacked` 輔助函數。
    *   或者，更高效的做法是傳入或計算 `pinners` 位元板。考慮到 Numba 的限制，我將使用 `get_pinners_and_blockers` (如果存在) 或在函數開頭快速計算一次 Pin 資訊（如果 King 被攻擊或參與交換）。
    *   *修正策略：* 為了效能，我們暫時不對每個 SEE 調用都計算全局 Pin。Stockfish 是因為增量更新才這麼快。我們將採用 **Ray-Tracing Check**：當輪到某個棋子攻擊時，檢查它是否與自己的 King 在同一條線上，且移動後是否會暴露 King。這比全局計算快。

3.  **X-Ray 處理：**
    *   利用 `magic_bitboards` (如果可用) 或 `sliding_attacks` 來動態更新攻擊者。
    *   當移除一個棋子後，使用 `attacks_bb` 查詢該方格，並與 Slider 位元板做 `AND` 運算，找出新暴露的攻擊者。

4.  **數值常量：** 使用與 Stockfish 一致的棋子價值 (P=100, N=320, B=330, R=500, Q=900, K=20000)。

### 3.2 具體改進代碼結構

我將創建 `chess_engine/see.py`，包含以下內容：

*   `see(piece_bbs, occupancy_bbs, side_to_move, from_sq, to_sq)`: 主函數。
*   使用 `numba.njit` 進行加速。
*   內部維護 `occupied` 位元板並動態更新。

## 4. 結論

本次改進將修復缺失的 `see.py`，並引入更精確的交換評估邏輯（特別是 X-Ray 和隱式 Pin 檢測），這將直接提升搜尋的 Move Ordering 質量，從而提高引擎的棋力。
