# 「無用剪枝」（Futility Pruning）問題分析與解決方案

本文檔詳細說明了先前版本中「無用剪枝」（Futility Pruning）邏輯存在的嚴重問題，以及我們是如何修正它的。

## 1. 無用剪枝的原理回顧

無用剪枝是一種在西洋棋引擎淺層搜索中（通常是 `depth` 較小時）使用的優化技巧。其核心思想是：

> 如果**當前局面**的靜態評估分數（`static_eval`）已經非常差，遠低於目前搜索窗口的下限（`alpha`），那麼再去遍歷那些通常不會大幅改變局面的「安靜棋步」（quiet moves），很可能是在浪費計算資源。

因此，剪枝的判斷條件大致為：

`static_eval + 一個邊界值 (Margin) < alpha`

如果此條件成立，引擎就會直接「剪掉」（跳過）後續所有安靜棋步的搜索。

**最關鍵的一點**：這個決策必須完全基於**做出任何棋步之前**的父節點局面評估。

## 2. 舊程式碼中的問題分析

在舊版本的 `chess_engine/search.py` 中，這段邏輯的實現存在一個根本性的錯誤。問題出在 `_search` 函數的 `for` 迴圈內部：

```python
# --- Futility Pruning (F-Pruning) ---
if ENABLE_FP and is_quiet_move and not is_currently_in_check:
    # 錯誤點：如果 static_score 未被計算，它會在此處用一個已經走過一步棋的新局面來計算
    if static_score == -INFINITY:
        static_score = evaluate_position(piece_bbs, occupancy_bbs, game_state, lazy=False)

    # ... 接下來才是剪枝判斷
    if margin > 0 and static_score + margin < alpha:
        # ... 剪枝 ...
        continue
```

這段程式碼在 `make_move(piece_bbs, occupancy_bbs, game_state, move)` **之後**才去檢查 `static_score`。如果 `static_score` 未被計算（在 `depth` 較深時就是這種情況），它會用**已經走出一步棋之後的新局面**去評估分數，然後用這個「未來」的分數來判斷是否應該在「過去」剪枝。

這完全違背了無用剪枝的原理，相當於在問一個邏輯顛倒的問題：

-   **正確的邏輯**：「我當前的局面太差了，還有必要看下一步安靜棋步嗎？」
-   **錯誤的邏輯**：「我先走出下一步安靜棋步，然後看看這個新局面是不是值得我當初去探索它？」

這個邏輯缺陷是導致剪枝計數異常，並且可能在某些情況下導致引擎做出錯誤決策的根源。

## 3. 解決方案

我的解決方案是將邏輯修正回其最純粹、最正確的定義，確保剪枝判斷的時機和依據都是正確的。

1.  **確保評估時機正確**：
    無用剪枝與其他淺層剪枝（如 Razoring）一樣，都依賴於在進入遍歷所有棋步的 `for` 迴圈**之前**，對當前局面進行的一次靜態評估。在程式碼中，`static_score` 這個變數就是在這個時間點被計算的（僅在 `depth <= 2` 時）。

2.  **移除錯誤的邏輯分支**：
    我完全刪除了在 `for` 迴圈內部、`make_move` 之後試圖去「補算」`static_score` 的那段錯誤程式碼。

3.  **增加嚴格的條件判斷**：
    我將剪枝的條件語句修改為：
    ```python
    if ENABLE_FP and is_quiet_move and not is_currently_in_check and static_score != -INFINITY:
    ```
    其中 `static_score != -INFINITY` 這個檢查至關重要。`-INFINITY` 是 `static_score` 的初始值。這個新條件確保了**只有在我們已經（在迴圈外）成功計算出當前父節點局面的靜態評估分數時，無用剪枝才會被觸發**。如果在 `depth` 較深，沒有觸發淺層評估的情況下，`static_score` 依然是 `-INFINITY`，那麼剪枝邏輯就根本不會執行，從而避免了所有錯誤判斷。

### 結論

通過以上修正，我們確保了無用剪枝只在正確的時機、基於正確的資訊來執行。這不僅修正了一個潛在的嚴重 bug，也使得剪枝統計數據恢復到了正常、合理的範圍，驗證了此修正的有效性。
