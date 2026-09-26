# Classical 搜尋優化候選提案 (E47–E50)

> **建立日期：** 2026-09-25  
> **最後更新：** 2026-09-26  
> **基準狀態：** **E50 已採納進生產基線**（廢除將軍延伸，400@n300k, 53.25%, Elo +22.62 ±20.50, 淨勝 +26 局, 深度 +1.20 ply, 耗時 -13.4ms）  
> **參考來源：** 官方最新發行標籤 `stockfish_repo/src/search.cpp` (Stockfish 19)

---

## 提案背景與核心原則

自 E43（RFP 深度 8 + TT 安靜步守護，+6.08 Elo）、E46（Razoring 現代化，+6.08 Elo）與 E47（ProbCut 深度動態自適應，+12.17 Elo）成功納入生產基線後，古典引擎在固定節點（300,000 nodes）對抗中的深層算力效率顯著提升。
然而，E44（IIR 門檻提升至 6，-20.46 Elo）與 E45（LMR Eval 差值調節，-25.26 Elo）的教訓表明：
1. **嚴禁破壞淺層快速試探**：過度延後減深會導致分支因子在淺層膨脹，揮霍固定算力。
2. **古典 HCE 必須免疫局部估值波動**：神經網路（NNUE）具有平滑無雜訊的估值，但古典 HCE 具有 50~100 cp 的局部波動，任何直接引入靜態估值微小差值（如 `alpha - eval`）至減深公式的機制，都會引發對防守安靜步的「致命誤剪」。
3. **優化方向必須著眼於「結構性剪枝對齊」與「節點算力再分配」**。

---

## 方案 A：ProbCut 深度動態自適應（對齊 Stockfish 19 Step 12）—【E47 已採納】

> **對戰驗證結果 (E47)：** 400 盤正式對局 **94W 226D 80L（得分率 51.75%，淨勝 +14 局）**，相對 Elo **+12.17 ± 19.88**，每步耗時提速 -1.4ms，NPS 提升 +0.3%。中局 (+2)、長局 (+5)、超長殘局 (+8) 全面獲勝。已正式同步至生產基線。

### 1. 現狀分析與問題
本作在 E35 引入了 TT-based ProbCut Shortcut（`ENABLE_PROBCUT_TT_SHORTCUT = True`），成功斬獲 **+11.59 Elo**。
然而，先前常規 ProbCut 搜尋依然保留舊版固定減深模式：
* 深度門檻固定為 `depth >= PROBCUT_MIN_DEPTH`（預設為 5）；
* 減深幅度一律固定為 `PROBCUT_R = 4`，即不論局面走向如何，均統一執行 `depth - 4` 的子搜尋；
* 未結合當前局面是否正在改善（`improving`）進行自適應調節。

### 2. Stockfish 19 Step 12 機制對齊
Stockfish 19（`stockfish_repo/src/search.cpp:1057-1063`）的核心設計：
```cpp
// Step 12. ProbCut
probCutBeta = beta + 241 - 64 * improving;
if (depth >= 3 && !is_decisive(beta) && !(is_valid(ttData.value) && ttData.value < probCutBeta))
{
    MovePicker mp(pos, ttData.move, probCutBeta - ss->staticEval, &captureHistory);
    Depth probCutDepth = depth - (improving ? 5 : 3);
    ...
```
* **動態自適應減深**：
  * 當前盤面正在改善（`improving` 為真）：局面樂觀且處於上升趨勢，採用更激進的減深 **5** 層（`depth - 5`），大幅加快驗證速度並釋放算力；
  * 當前盤面未改善（`improving` 為假）：局面停滯或受壓，採用保守減深 **3** 層（`depth - 3`），防止在劣勢局中因過度減深而漏算對手戰術反撲。
* **深度門檻調適**：
  * 在 `improving` 時要求 `depth >= 6`（保證子搜尋深度至少為 1）；在 `not improving` 時放寬至 `depth >= 4`，讓優勢局更早透過 ProbCut 提前截斷。

### 3. 預期效益
* 與已採納的 E35 深度協同，在優勢局大幅節省節點耗損，在危險局提高搜尋安全性，預期帶來穩定正向 Elo。

---

## 方案 B：LMR ALL-Node 放大比率對齊（對齊 Stockfish 19 Step 18）—【E48 已採納】

> **對戰驗證結果 (E48)：** 400 盤正式對局 **102W 204D 94L（得分率 51.00%，淨勝 +8 局）**，相對 Elo **+6.95 ± 23.83**。平均深度提升 **+0.08 ply**（深層比例 +0.9%），每步耗時提速 **-3.8ms**，持黑勝率高達 **52.50%**，中局 (+7 局) 與長局 (+4 局) 明顯勝出。符合通過門檻，建議同步採納。

### 1. 現狀分析與問題
在晚期走步減深（LMR）中，節點分為 PV 節點、Cut 節點（預期第 1 步截斷）與 ALL 節點（預期所有走步皆無法突破 $\alpha$ 的防守節點）。
在 ALL 節點上，子節點預期全軍覆沒，應該大幅放大減深幅度以快速驗證防守成立。
本作原有的公式：
```python
all_node = not is_pv and not cut_node
if all_node:
    r += trunc_div(r * LMR_ALLNODE_SCALE_NUM, LMR_ALLNODE_SCALE_DENOM_BASE * depth + LMR_ALLNODE_SCALE_DENOM_OFFSET)
```
其中原版 `LMR_ALLNODE_SCALE_NUM = 150`, `LMR_ALLNODE_SCALE_DENOM_OFFSET = 285`。

### 2. Stockfish 19 Step 18 機制對齊
Stockfish 19（`stockfish_repo/src/search.cpp:1358-1359`）：
```cpp
// Scale up reductions for expected ALL nodes
if (allNode)
    r += r * 276 / (256 * depth + 268);
```
* 原版分子 `150` 僅為 SF19（`276`）的 **54.3%**！
* 這導致本作在 ALL 節點上的減深力度嚴重不足，耗費大量節點在明知無法突破的防守走步上。

### 3. 改動與預期效益
* 將 `LMR_ALLNODE_SCALE_NUM` 由 150 調至 **276**；`LMR_ALLNODE_SCALE_DENOM_OFFSET` 由 285 調至 **268**。
* 純常數精準對齊，邏輯完全受控，在防守節點安全回收海量無效算力。
* 對戰驗證全面達成預期，中局與長局勝率顯著改善，防守效率大幅提升。

---

## 方案 C：LMR CutNode 減深加強（對齊 Stockfish 19 Step 18）—【E49 否決回滾】

> **對戰驗證結果 (E49)：** 400 盤正式對局 **87W 224D 89L（得分率 49.75%，淨負 -2 局）**，相對 Elo **-1.74 ± 22.59**。全場平均深度倒退 **-0.13 ply**（深層比例下滑 -2.5%），持黑勝率暴跌至 **42.50%**（淨負 -30 局）。
> **根因分析：** 古典 HCE 的走步排序不如神經網路精準，將 CutNode 減深一口氣倍增至 4026 導致後續次優步過度減深，反覆觸發二度搜尋（LMR Re-search），嚴重揮霍 nodes 預算並引發深度萎縮。未達通過門檻，已**全數回滾至 E48 生產基線**。

### 1. 現狀分析與問題
在 CutNode（預期第一步即達成 $\beta$-Cutoff 的節點）上，如果第一步（通常是 TT 著法或殺手步）未能截斷，進入後續走步（`moveCount > 1`）時，這些走步在統計上超過 95% 全為劣步。
原版在 CutNode 上給予的減深補貼：
`LMR_CUTNODE_BONUS = 2048`（SF11 舊世代，在 1024 尺度下僅相當於 2 層減深），`LMR_NO_TTMOVE_BONUS = 1024`。

### 2. Stockfish 19 Step 18 機制對齊
Stockfish 19（`stockfish_repo/src/search.cpp:1327-1328`）：
```cpp
// Increase reduction for cut nodes
if (cutNode)
    r += 4026 + 933 * !ttData.move;
```
* SF19 在 CutNode 上直接增加高達 **4026**（接近 4 層減深），若連 TT 著法都沒有更是增加近 5 層（$4026 + 933 = 4959$）。
* 然而在缺少 NNUE 平滑精準排序的古典引擎架構下，激進減深反而招致 Re-search 算力風暴。

### 3. 結論
* 400 盤實測證實直接跳躍至 4026 對古典引擎過於激進，引發深度萎縮與黑方防守崩盤。
* 已嚴格執行回滾，常數維持 `LMR_CUTNODE_BONUS = 2048`, `LMR_NO_TTMOVE_BONUS = 1024`。

---

## 方案 D：廢除無效將軍延伸（Check Extension 限制）（對齊 Stockfish 19 Step 17）—【E50 已採納】

> **對戰驗證結果 (E50)：** 400 盤正式對局 **96W 234D 70L（得分率 53.25%，淨勝高達 +26 局！）**，相對 Elo **+22.62 ± 20.50 (95% CI: [2.20, 43.19]，CI 下界 > 0 統計顯著勝出！)**。全場平均深度暴增 **+1.20 ply**（深層比例暴增 +17.3%），每步耗時提速 **-13.4ms**，長局勝率 **56.14%**。完全符合「明確正向」最高通過門檻，**已正式同步進生產基線**！

### 1. 現狀分析與問題
現代西洋棋引擎的一大演進是**徹底移除 Check Extension（將軍延伸）**。
* 過去引擎只要走出將軍就盲目加深 1 層（`depth - 1 + 1 = depth`），極易引發「將軍爆炸（Check Explosion）」：在無效逃王與連環將軍的死胡同中浪費巨量節點。
* 本作在 E1 曾藉由收緊 Check Extension 的 SEE gate（`ENABLE_CHECK_SEE_GATE = True`）斬獲 **+15.7 Elo**，已充分驗證「限制無效將軍加深」的巨大價值。

### 2. Stockfish 19 Step 17 機制對齊
* Stockfish 19（`stockfish_repo/src/search.cpp:1147, 1311`）完全沒有 `check_extension`，將軍著法不再賦予整層深度延伸。
* 將軍威脅完全交由 Move Ordering（將軍步優先嘗試）、QSearch（靜態搜尋處理將軍）以及 LMR（將軍步減少減深）自然處理。

### 3. 改動與預期效益
* 設置 `ENABLE_CHECK_EXTENSION = False`，徹底消除將軍延伸引發的將軍爆炸，促使平均名義深度真實轉化為有效棋力。
* 對戰驗證全面大捷，深度暴增 +1.20 ply，全場提速 -13.4ms，斬獲 +22.62 Elo。

---

## 優先執行順序建議

| 順序 | 提案 | 狀態 / 結果 | 預期 Elo | 實作難度 | 風險級別 | 說明 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **已採納** | **方案 A (ProbCut 深度動態自適應)** | **E47 採納 (+12.17 Elo)** | +8 ~ +15 | 低 | 極低 | 400 盤淨勝 +14 局，全場提速 -1.4ms，已納入基線 |
| **已採納** | **方案 B (LMR ALL-Node 放大比率對齊)** | **E48 採納 (+6.95 Elo)** | +5 ~ +10 | 極低 | 極低 | 400 盤淨勝 +8 局，深度 +0.08，已納入基線 |
| **已回滾** | **方案 C (LMR CutNode 減深加強)** | **E49 回滾 (-1.74 Elo)** | +5 ~ +10 | 低 | 低 | 4026 過度減深引發 Re-search 算力風暴與深度萎縮 |
| **已採納** | **方案 D (廢除將軍延伸)** | **E50 採納 (+22.62 Elo)** | +10 ~ +20 | 低 | 低 | 徹底消除深層將軍爆炸，深度 +1.20 ply，狂勝 +26 局 |


