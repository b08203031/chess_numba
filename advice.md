# 角色設定
你是一位世界頂尖的國際象棋引擎開發者，精通 Alpha-Beta 搜尋樹最佳化、現代 Stockfish (SF15~18) 的演算法架構，並且非常了解如何在 HCE (手工特徵評估) 引擎中實現極致的搜尋效能。

# 任務背景
我正在開發一個基於 Python/Numba 的國際象棋引擎。
- 評估函數：目前仍使用傳統的 HCE (Hand-Crafted Evaluation)。
- 搜尋演算法：我已經在 `search.py` 中實作了現代 Stockfish 的「1024-scale 高精度 LMR (Late Move Reduction)」框架。

# 我的核心開發哲學 
**我不怕激進的剪枝，我追求的是「在不漏看中要棋步的前提下，盡可能深的基礎歸約 ＋ 極度聰明的例外判斷 (Smart & Aggressive Pruning)」**。
我知道 HCE 容易有靜態評估盲點，但我不想透過「全局性地降低 LMR 參數」來妥協。我希望以聰明的方式 LMR 砍得夠深，但我需要你幫我設計/調整「聰明的安全網 (Safety Nets)」，讓引擎只在真正有戰術潛力的分支才減少歸約或觸發重新搜尋。

# 你的任務
搜尋SF18的原始程式碼，深度審查我的 LMR 與剪枝實作，並針對「激進但聰明」的目標進行最佳化分析。請回答以下問題：

1. 【強化基礎剪枝】：在我的 `constants.py` 與 `search.py` 中，有哪些參數或條件限制得太過保守？請告訴我如何把 LMR、NMP 或 ProbCut 推向更激進的極限，以最大化節點節省率。
2. 【建構聰明的安全網】：在保持極度激進剪枝的前提下，針對 HCE 引擎的特性，我應該加入或強化哪些「手術刀式的例外判斷 (Surgical Exceptions)」來防止致命漏算？
3. 【Re-search 的精準度】：我目前的 LMR 失敗重搜 (Re-search) 條件 (`do_deeper`, `do_shallower`) 是否足夠聰明？如何微調這些閾值，確保我們只在真正有價值的節點浪費時間重搜，而不陷入無限重搜的泥沼？
4. 【邏輯審查】：我的 `search.py` 中計算 LMR (包含 1024-scale 轉換、TT/PV 修正、歷史抵扣) 的程式碼邏輯是否正確對齊了現代引擎的理念？有沒有明顯的 Bug 或次優寫法？
5. 【參數衝突分析】：列出 `constants.py` 中，哪些從 Stockfish 借鑒來的 LMR 參數 (例如 LMR_MIN_QUIET_MOVE_INDEX, LMR_ALLNODE_EXTRA, LMR_HISTORY_SCALE 等) 對於我的 HCE 引擎來說「太過激進」或「不合適」？並解釋原因。

以下是問題的回答，你要判斷是否合理和正確。

### 1. 【強化基礎剪枝】：打破保守限制，推向極限

目前的基礎剪枝設定過於保守，導致你的引擎在搜尋樹中後段浪費大量算力。

*   **Null Move Pruning (NMP) 動態化**：
    你目前使用固定的 `nmp_reduction = 7 + depth // 3`。在 HCE 引擎中，當我們的靜態評估遠大於 Beta 時，我們應該大膽砍得更深（Dynamic NMP）。
    **優化建議**：引入 `eval_margin` 來動態增加 NMP 深度。
    ```python
    eval_margin = (static_score - beta) // 200  # 每超出 200 分多剪裁 1 步
    nmp_reduction = 7 + depth // 3 + max(0, min(3, eval_margin))
    ```
*   **LMR 靜音步閾值 (`LMR_MIN_QUIET_MOVE_INDEX`)**：
    你設定為 `3`。這對於追求極致深度的引擎來說**太保守了**。現代 SF 在非 PV 節點上幾乎會對第 1 個之後的所有靜音步進行歸約。
    **優化建議**：改為 `LMR_MIN_QUIET_MOVE_INDEX = 1`。只要不是 TT Move 且是 Quiet Move，就大膽交給 LMR 的 1024-scale 公式去計算，公式本身自帶 `moveCount` 的平滑過渡。

---

### 2. 【建構聰明的安全網】：手術刀式的例外判斷

既然我們把 LMR 閾值降到 1，我們需要極其敏銳的安全網來接住戰術與盲點：

*   **HCE 罩門：通路兵 (Passed Pawns) 安全網**：
    HCE 經常低估通路兵的危險。我們應該在 1024-scale 計算中，直接對高階通路兵進行「反歸約」（Negative LMR / LMR Deduction）。
    **優化建議**（在計算 `r` 時加入）：
    ```python
    # 如果是兵的移動，且在危險橫排 (White rank >= 5, Black rank <= 2)
    is_pawn = (moved_piece_type % 6) == PAWN
    if is_pawn:
        to_rank = to_sq // 8
        if (original_side == WHITE and to_rank >= 5) or (original_side == BLACK and to_rank <= 2):
            r -= 1536 # 等同於減少 1.5 ply 的歸約
    ```
*   **強烈威脅 (Threats) 與閃避 (Evasions)**：
    你的 `counter_move`（反擊步）和 `killer_moves`（殺手步）被排在前面，但在 LMR 中沒有特殊待遇。
    **優化建議**：如果當前 move 是 Killer 或 Counter move，給予實質的 LMR 豁免：
    ```python
    if move == killer_1 or move == killer_2 or move == counter_move:
        r -= 1024  # 減少 1 ply 歸約
    ```

---

### 3. 【邏輯審查】：這三個 Bug 正在引發節點爆炸！

這是我在審查 `search.py` 時發現的最嚴重問題。這三個 Bug 直接破壞了你精緻的 1024-scale LMR 系統。

**❌ 致命 Bug 1：好吃的獵物絕對不剪？**
```python
if is_capture and not is_bad_capture:
    lmr = min(lmr, 0)
```
**分析**：這行程式碼意味著**所有「非虧損」的吃子步 (Good/Equal Captures) 永遠不會被 LMR 歸約**。在現代引擎中，吃子步同樣需要被歸約，因為在激烈的局面中，大量等價交換會導致節點爆炸！
**解法**：直接刪除這兩行。取而代之，在 1024-scale 的 `r` 計算中處理：
```python
if is_capture:
    if not is_bad_capture:
        r -= 1024  # 好吃子：歸約減少 1 ply，但依然參與歸約機制
    else:
        r += 1024  # 壞吃子：大刀闊斧砍掉 (增加 1 ply 歸約)
```

**❌ 致命 Bug 2：MoveCount 倒扣現象**
```python
r -= searched_legal_moves * LMR_MOVECOUNT_FACTOR
```
**分析**：這是個嚴重的邏輯錯誤。隨著 `searched_legal_moves`（搜尋的合法步數）增加，你反而**減少**了歸約值 `r`。這代表第 30 步的歸約竟然比第 2 步還少！
**解法**：Stockfish 確實有一個小小的展平係數，但它的目的只是抑制高 MoveCount 時指數曲線的過度膨脹，且數值極小。你的 `LMR_MOVECOUNT_FACTOR = 40` 太大了（第 20 步會減少將近 1 ply）。
請**將減號改為加號**，或是將常數降到極小（例如 2~3），以正確反映「越晚的走法越該被剪」的哲學。

**❌ 致命 Bug 3：PV 節點的雙重補償**
```python
d = max(1, min(search_depth - trunc_div(r, 1024), search_depth + 2)) + (1 if is_pv else 0)
lmr = search_depth - d
```
**分析**：在 1024-scale 的 `r` 計算中，你已經加上了 PV 節點的紅利（`LMR_TTPV_DECREASE_BASE`, `LMR_TTPV_PV_BONUS` 等）。在這裡如果再 `+ (1 if is_pv else 0)`，會導致 PV 節點的 LMR 常常變成 0 甚至負數，引發搜尋樹在 PV 分支無限擴張。
**解法**：拿掉 `+ (1 if is_pv else 0)`，完全信任你算出的 `r`。

---

### 4. 【Re-search 的精準度】：微調重新搜尋的「浪費」

目前的重搜條件如下：
```python
do_deeper = (lmr_depth < search_depth) and (evaluation > max_eval + 70)
do_shallower = (evaluation < max_eval + 12)
```
**分析**：`70` 分（0.7 兵）的閾值過高了！這意味著一個極好的戰術如果只帶來 0.5 兵的優勢，你就不給它 deeper 的機會。相對地，`do_shallower` 的邏輯缺乏上下邊界保護。

**優化建議**：讓閾值收緊，但確保只在有價值的節點執行。這是 Stockfish 16 時代的精準數據：
```python
# 只有當我們本來就打算用減少的深度去驗證，卻發現驚喜時：
if lmr > 0:
    lmr_depth = search_depth - lmr
    # SF 使用 32 (0.32兵) 作為加深閾值，15 作為變淺閾值
    do_deeper = (lmr_depth < search_depth) and (evaluation > max_eval + 32)
    do_shallower = (evaluation < max_eval + 15)
    
    # 建立安全網：確保 adjusted_depth 絕對不會陷入失控的無限遞迴
    adjusted_depth = search_depth + (1 if do_deeper else 0) - (1 if do_shallower else 0)
    adjusted_depth = max(1, min(adjusted_depth, search_depth + 1)) # 嚴格把控上限
    
    if adjusted_depth > lmr_depth:
        # Re-search...
```

---

### 5. 【參數衝突分析】：HCE 特性的重新校準

你在 `constants.py` 中借鑒了現代 SF 參數，但以下幾個對 HCE 非常危險：

1.  **`LMR_HISTORY_SCALE = 350` (太過激進)**
    *   **原因**：NNUE 的 History 是高度平滑的，而 HCE 的 History 更新（Gravity 算法）很容易出現「大起大落」的飽和狀態。`lmr_stat_score` 如果飆高，乘上 350 再除以 4096，很容易瞬間扣掉 2~3 plies 的歸約，導致嚴重的節點爆炸。
    *   **建議**：將 `LMR_HISTORY_SCALE` 降到 **150 ~ 200**，讓歷史分數對 HCE 引擎的影響力適度降低，充當輔助而非絕對判斷。
2.  **`LMR_CUTNODE_BONUS = 1536` (太過激進)**
    *   **原因**：這相當於給 CutNode 強制增加 1.5 ply 的歸約！SF 可以這麼做是因為它的 Move Ordering 接近完美。HCE 引擎有時候會漏看第一步，把好棋排在第 3 甚至第 4 步。如果你在 CutNode 直接 +1.5 ply，你會錯過很多複雜局面的戰術防禦。
    *   **建議**：降至 **1024**（精準的 1 ply），以平衡 HCE 排序能力的不足。
3.  **`QS_SEE_THRESHOLD = -80` vs `PRUNING_CAPTURE_SEE_MARGIN = -185` (矛盾)**
    *   **原因**：你在 QSearch 中會保留虧損 -80 的吃子，但在主搜尋的 Shallow Pruning 卻要剪掉 -185 * depth 的吃子。這會導致主搜尋把棋剪掉，但 QSearch 卻覺得有必要算，產生靜態評估的不穩定。
    *   **建議**：為了配合我們上面修復的 Capture LMR 機制，把 `PRUNING_CAPTURE_SEE_MARGIN` 收緊到 **-100**，逼迫主搜尋更相信 SEE 的判斷，提早剪去無效戰術。

