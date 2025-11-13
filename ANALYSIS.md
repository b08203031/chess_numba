# 引擎搜尋與剪枝技術分析報告

本文檔詳細分析了 `search.py` 中實現的各種西洋棋引擎搜尋與剪枝技術。每一章節都涵蓋了特定技術的作用、實現方式、程式碼邏輯、預期效果以及正確性評估。

---

## 總結與建議

### 綜合評估

這個西洋棋引擎的搜尋架構非常現代化且實現得相當出色。它整合了當前主流引擎所使用的大部分高級剪枝與搜尋優化技術。程式碼結構清晰，特別是 `_search` 函數中的邏輯層次分明，各種技術的啟用和停用都由 `constants.py` 中的開關控制，便於測試和調試。

**優點：**
1.  **技術全面：** 涵蓋了從基礎的 Aspiration Windows 到高級的 Singular Extensions，技術棧非常完整。
2.  **實現標準：** 大多數技術的實現都遵循了 Chess Programming Wiki 或學術論文中的標準模型，顯示出開發者對演算法的深刻理解。
3.  **細節處理到位：** 在很多地方，例如 NMP 的統計數據累加、Futility Pruning 中對 `unmake_move` 的正確處理，都體現了高品質的程式碼實踐。

**待改進之處：**
在分析過程中，我發現了兩個潛在的邏輯錯誤，如果修正，可能會進一步提升引擎的穩定性和性能。

### 具體修改建議

#### 1. (嚴重) Late Move Pruning (LMP) 的計數器邏輯錯誤

*   **問題描述：** `quiet_move_counter` 在走法循環的**末尾**才被更新，而 LMP 的剪枝判斷在循環的**開頭**。這導致剪枝判斷所使用的 `quiet_move_counter` 值是過時的（少算了當前這一步），使得剪枝觸發比預期晚了一步，甚至在某些情況下完全失效。
*   **風險：** 高。此 Bug 嚴重削弱了 LMP 的剪枝效率，導致引擎在搜尋後期著法時浪費了大量不必要的計算。
*   **建議修改 (`_search` 函數):**

    ```python
    # <<<<<<< SEARCH
            if alpha >= beta:
                cutoffs += 1
                if is_quiet_move:
                    bonus = depth * depth
                    aggressor_type = find_piece_type_on_square(piece_bbs, get_from_square(move))
                    search_context.history_table[aggressor_type, get_to_square(move)] += bonus
                    if move != search_context.killer_moves[ply * 2]:
                        search_context.killer_moves[ply * 2 + 1] = search_context.killer_moves[ply * 2]
                        search_context.killer_moves[ply * 2] = move
                break
            if is_quiet_move: quiet_move_counter += 1
    =======
            if alpha >= beta:
                cutoffs += 1
                if is_quiet_move:
                    bonus = depth * depth
                    aggressor_type = find_piece_type_on_square(piece_bbs, get_from_square(move))
                    search_context.history_table[aggressor_type, get_to_square(move)] += bonus
                    if move != search_context.killer_moves[ply * 2]:
                        search_context.killer_moves[ply * 2 + 1] = search_context.killer_moves[ply * 2]
                        search_context.killer_moves[ply * 2] = move
                break
    >>>>>>> REPLACE
    ```
    應將 `quiet_move_counter += 1` 的邏輯移到 LMP 判斷之前，並確保它只在 `is_quiet_move` 為 true 時執行一次。

    **正確的邏輯應如下所示：**
    ```python
    # ... inside move loop, after is_quiet_move is determined ...
    if is_quiet_move:
        quiet_move_counter += 1 # <<<< 立即更新
        if ENABLE_LMP and not is_currently_in_check:
            if quiet_move_counter >= LMP_MOVE_COUNT[depth]:
                lmp_pruned += 1
                unmake_move(...)
                break

    # ... Futility Pruning, LMR, etc. ...

    # ... at the end of the loop, remove the old "if is_quiet_move: quiet_move_counter += 1"
    ```

#### 2. (中等) Delta Pruning 的邏輯不夠穩健

*   **問題描述：** Delta Pruning 目前在靜態搜尋（Quiescence Search）中被應用於所有的吃子著法。然而，Delta Pruning 的理論基礎是基於**靜態**評估，它在處理**動態**的戰術交換時非常不可靠。一個根據靜態評估看似虧本的吃子（例如，兵吃兵），可能是避免被將死或實現其他戰術目標的唯一途徑。目前的實現雖然檢查了 "spite check"（送吃將軍），但沒有完全涵蓋所有戰術可能性。
*   **風險：** 中等。在某些複雜的戰術局面下，此剪枝可能會錯誤地剪掉一個關鍵的吃子著法，導致引擎犯下大錯。
*   **建議修改 (`quiescence_search` 函數):**

    最簡單且最安全的修改是**在靜態搜尋中完全禁用 Delta Pruning**，完全依賴 SEE 來過濾壞的吃子著法。SEE 是專為此類戰術計算而設計的，遠比 Delta Pruning 更可靠。

    ```python
    # in quiescence_search
    # <<<<<<< SEARCH
            # --- Delta Pruning ---
            if ENABLE_DELTA_PRUNING:
                is_promotion = get_special_move_flag(move) == SPECIAL_MOVE_FLAG_PROMOTION
                promotion_gain = MG_MATERIAL_VALUES[4] - MG_MATERIAL_VALUES[0] if is_promotion else 0
                victim_type = find_piece_type_on_square(piece_bbs, get_to_square(move))
                victim_value = MG_MATERIAL_VALUES[victim_type % 6] if victim_type != -1 else 0
                potential_gain = victim_value + promotion_gain

                if stand_pat + potential_gain + DELTA_PRUNING_MARGIN < alpha:
                    unmake_info = make_move(piece_bbs, occupancy_bbs, game_state, move)
                    is_check = is_in_check(piece_bbs, occupancy_bbs, game_state)
                    unmake_move(piece_bbs, occupancy_bbs, game_state, move, unmake_info)
                    if not is_check:
                        delta_pruned += 1
                        continue
    =======
    >>>>>>> REPLACE
    ```
    直接移除或通過 `ENABLE_DELTA_PRUNING` 開關禁用此邏輯塊，是更穩健的選擇。

---

## 各項技術詳細分析

### 1. Aspiration Windows (Aspiration 窗)
- **作用：** 利用前一深度的分數來預測當前深度的分數範圍，從而設定一個更窄的 Alpha-Beta 視窗以加速搜尋。
- **實現方式：** 在 `iterative_deepening_search` 中，每次迭代時，將 `alpha` 和 `beta` 設定為 `last_score ± ASPIRATION_WINDOW_SIZE`。如果搜尋結果超出了這個視窗（fail-high 或 fail-low），則以無限大視窗重新進行搜尋。
- **正確性評估：** **正確**。實現完全符合標準演算法，邏輯清晰。

### 2. Null Move Pruning (NMP / 空著剪枝)
- **作用：** 基於「如果讓對手多走一步，局面依然非常好，那當前局面本身就已經足夠好」的假設，提前剪枝。
- **實現方式：** 在滿足特定條件下（非將軍、深度足夠、有足夠棋子），引擎會模擬一次「空著」（只交換走棋方），並以削減後的深度 `(depth - 1 - R)` 和零窗口 `(-beta, -beta + 1)` 進行搜尋。如果結果仍然 `> beta`，則直接剪枝。
- **正確性評估：** **正確**。實現非常標準，所有前提條件和遞迴搜尋的參數都設定無誤。

### 3. Razoring (剃刀剪枝)
- **作用：** 在淺層搜尋中，如果靜態評估值已經遠低於 `alpha`，則可以提前剪掉整個節點。
- **實現方式：** 在 `depth <= 2` 且非將軍時，若 `static_score + MARGIN < alpha`，則進行一次快速的靜態搜尋作為驗證。如果驗證結果仍然低於 `alpha`，則剪枝。
- **正確性評估：** **正確**。兩階段驗證（靜態評估 + 靜態搜尋）的實現方式非常穩健。

### 4. Futility Pruning (無用剪枝)
- **作用：** 在淺層搜尋中，如果靜態評估值很低，則可以剪掉所有後續的「安靜著法」。
- **實現方式：** 在 `depth <= 2` 的走法循環中，若 `static_score + MARGIN < alpha`，則跳過 (`continue`) 當前處理的安靜著法。
- **正確性評估：** **正確**。邏輯無誤，特別是在 `continue` 前正確處理了 `unmake_move`，避免了狀態污染。

### 5. Reverse Futility Pruning (RFP / 逆無用剪枝)
- **作用：** 在極淺層 (`depth=1`)，如果靜態評估值已經非常高 (`>= beta`)，則可以提前剪掉整個節點。
- **實現方式：** 在 `depth == 1` 且非將軍時，若 `static_score - MARGIN >= beta`，則直接回傳 `beta`。
- **正確性評估：** **正確**。實現符合標準定義，且條件嚴格，風險低。

### 6. Late Move Reductions (LMR / 遲著削減)
- **作用：** 對著法列表中排在後面的安靜著法，使用一個削減過的深度進行搜尋，以節省時間。
- **實現方式：** 與 PVS 結合。對於非第一個著法，在進行零窗口搜尋時，如果滿足 LMR 條件（深度、著法順序），則削減搜尋深度。如果削減深度的搜尋結果依然很好，再用完整深度複搜。
- **正確性評估：** **正確**。是 PVS + LMR 的標準範例，邏輯清晰無誤。

### 7. Late Move Pruning (LMP / 遲著剪枝)
- **作用：** 比 LMR 更激進。當檢查過的安靜著法數量達到閾值後，完全剪掉後續所有著法。
- **實現方式：** 在走法循環中，若 `quiet_move_counter >= THRESHOLD`，則 `break` 跳出循環。
- **正確性評估：** **存在嚴重錯誤**。詳見上方「總結與建議」。

### 8. ProbCut (概率剪枝)
- **作用：** 利用淺層的 TT 分數來預測深層搜尋的結果。如果預測結果極好或極差，則進行一次更淺的驗證性搜尋，成功後直接剪枝。
- **實現方式：** 在滿足深度和 TT 條件時，若 `tt_score` 遠超 `beta` 或遠低於 `alpha`，則呼叫一次削減深度的 `_search` 來驗證，成功後剪枝。
- **正確性評估：** **正確**。實現符合 ProbCut 的核心思想，兩個分支（fail-high/fail-low）的處理都很標準。

### 9. Quiescence Search (靜態搜尋)
- **作用：** 在常規搜尋的葉子節點，繼續向下只搜尋戰術性著法（吃子、升變），直到局面平靜為止，以避免「水平線效應」。
- **實現方式：** 一個迷你的 Alpha-Beta 搜尋，但只生成吃子和升變著法，並包含 "stand-pat" 評估優化。
- **正確性評估：** **正確**。框架和邏輯都符合標準實現。

### 10. Delta Pruning (Delta 剪枝)
- **作用：** 在靜態搜尋中，剪掉那些即使算上最大物質收益，也無法提高分數的吃子著法。
- **實現方式：** 若 `stand_pat + potential_gain < alpha`，則剪枝，但會額外檢查是否為 "spite check"。
- **正確性評估：** **存在邏輯風險**。詳見上方「總結與建議」。

### 11. Internal Iterative Deepening (IID / 內部迭代加深)
- **作用：** 當節點在 TT 中找不到最佳著法時，進行一次快速的淺層搜尋來找到一個引導著法，以改善著法排序。
- **實現方式：** 在 `tt_move == NO_MOVE` 時，遞迴呼叫一次削減深度的 `_search`，然後重新查詢 TT 以獲取 `tt_move`。
- **正確性評估：** **正確**。實現符合標準定義，觸發時機和處理方式都無誤。

### 12. Singular Extensions (奇異擴展)
- **作用：** 識別出那些「唯一且超級好」的著法，並對其增加搜尋深度。
- **實現方式：** 通過一次「排除性搜尋」（搜尋時排除掉候選著法），如果其他所有著法都遠不如該候選著法，則確認其為「奇異」，並在主搜尋中給予深度擴展。
- **正確性評估：** **正確**。實現複雜但準確，完全符合現代引擎的做法。

### 13. Static Exchange Evaluation (SEE)
- **作用：** 預估一個格子上的交換鏈所導致的淨物質得失，用於剪掉「虧本」的吃子著法。
- **實現方式：** 在靜態搜尋中，對每個吃子著法呼叫 `see()` 函數。如果回傳值 `< 0`，則跳過該著法。
- **正確性評估：** **正確**。是 SEE 最經典、最有效的應用場景，邏輯無誤。
