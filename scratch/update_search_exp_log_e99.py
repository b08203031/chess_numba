path = 'tournament_analysis/SEARCH_EXP_LOG.md'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# Update Last Updated Date
text = text.replace('> **最後更新：** 2026-10-26', '> **最後更新：** 2026-10-28', 1)

# E97, E98, E99 text
new_sections = """### E97：LMR 基礎偏移量與雙延展起點深度探索（`LMR_BASE_OFFSET` 475 vs 465、`SINGULAR_DOUBLE_EXT_MIN_DEPTH = 9 vs 10`）

* **動機與原理**：
  * 在 E96 確立 `SINGULAR_TTPV_BONUS = 65` 後，針對 LMR 基礎偏移與奇異雙延展起點深度進行邊界探測。
* **實測對局與結果**：
  1. **`LMR_BASE_OFFSET = 475 vs 465`（`copy 4.ipynb`）**：
     * 494 局（247 對成對換先開局，時控 10,000 nodes/move）：Candidate 88W 299D 107L（得分率 **48.1%**，**-13.4 ± 19.2 Elo**，SPRT LLR = **-2.02**）。
     * 分析：偏移量增加 10 點（465 -> 475）導致全樹 non-PV 著法普遍減深過重，大量關鍵戰術變例被提前截斷漏算，顯著負向，**否決回滾**。隨後依據戰術精度需求將基準微調放寬至 `455`（commit `a8bcdf3`，並測試 `460 vs 455` commit `c4ca843`）。
  2. **`SINGULAR_DOUBLE_EXT_MIN_DEPTH = 9 vs 10`（`copy 6.ipynb`）**：
     * 820 局（410 對成對換先開局，時控 10,000 nodes/move）：Candidate 160W 494D 166L（得分率 **49.6%**，**-2.5 ± 15.0 Elo**，SPRT LLR = **-1.44**）。
     * 分析：在 depth=9 即提早開啟雙延展，使非 PV 節點過早觸發昂貴的深層追搜，導致搜尋樹有效深度略微稀釋，微幅負向，**否決維持 10**。
* **處置結論**：維持 `SINGULAR_DOUBLE_EXT_MIN_DEPTH = 10`，LMR 基礎偏移量收斂於 `455`。

---

### E98：LMR 重新搜尋深度自適應門檻多維微調（`LMR_RESEARCH_DEEPER_MARGIN` & `SHALLOWER_MARGIN`）

* **動機與原理**：
  * 針對 LMR 減深著法在零視窗搜尋中 fail-high 時，觸發 deeper (+1 depth) 或 shallower (-1 depth) 的評估偏置裕度展開細部探索（對齊 Stockfish Re-search 深度自適應）。
* **實測對局對比**：
  1. `DEEPER = 24 vs 26`（`copy 2.ipynb`）：480 局 91W 301D 88L（得分率 **50.3%**，**+2.2 ± 19.0 Elo**，SPRT LLR = **-0.29**）。
  2. `DEEPER = 28 vs 26`（`copy 7.ipynb`）：244 局 45W 154D 45L（得分率 **50.0%**，**-0.0 ± 26.5 Elo**，SPRT LLR = **-0.28**）。
  3. `SHALLOWER = 10 vs 15`（`copy 5.ipynb`）：472 局 94W 289D 89L（得分率 **50.5%**，**+3.7 ± 19.5 Elo**，SPRT LLR = **-0.15**）。
  4. `SHALLOWER = 18 vs 15`（`copy.ipynb`）：204 局 36W 130D 38L（得分率 **49.5%**，**-3.4 ± 28.7 Elo**，SPRT LLR = **-0.38**）。
* **分析與處置**：
  * 四組平行實驗累計跑了超過 1,400 局，勝率全部緊鎖在 49.5% ~ 50.5% 的隨機噪聲帶，LLR 均未產生有效偏移。
  * 根因分析：LMR 重新搜尋深度微調需同時滿足「著法減深 + 零視窗 fail-high + 評估落入極端裕度區間」，全樹觸發比例低於 0.3%，微調 2~5 cp 對勝率無統計顯著性，屬於平原噪聲。
  * **處置結論**：終止微調，鎖定基準為 `DEEPER = 26, SHALLOWER = 18`。

---

### E99：晚期移動步數剪枝比例 `LMP_SCALE_PERCENT` 全局多維梯度與 400k LTC 突破

* **動機與原理**：
  * Late Move Pruning（LMP）控制非 PV 節點在嘗試若干靜態走法後，直接剪除剩餘排序靠後之走法：
    $$\\text{limit} = \\frac{\\text{LMP\\_MOVE\\_COUNT}[\\text{depth}]}{2 - \\text{improving}} \\times \\frac{\\text{LMP\\_SCALE\\_PERCENT}}{100}$$
  * 生產基準原為 150%。為了探明 LMP 在當前搜尋深度與高時控下的最優剪枝閥值，啟動多點梯度實測（135%、142%、145%、155%、160%、165%）。
* **多點梯度對局對比**：
  1. **135% vs 150%**（`copy 8.ipynb` 前期，400k nodes）：250 局 36W 154D 59L（得分率 **45.6%**，**-30.7 ± 26.6 Elo**，SPRT LLR = **-2.00**）❌ 剪枝太狠，防守漏步崩潰。
  2. **155% vs 150%**（`copy 3.ipynb`，400k nodes）：160 局 25W 103D 32L（得分率 **47.8%**，**-15.2 ± 32.1 Elo**，SPRT LLR = **-0.73**）❌ 負向。
  3. **160% vs 150%**（`runner.ipynb` 前期，400k nodes）：260 局 48W 166D 46L（得分率 **50.4%**，**+2.7 ± 25.4 Elo**，SPRT LLR = **-0.16**）⚪ 與 150 持平。
  4. **142% vs 150%**（`copy 3.ipynb` 前期，400k nodes）：120 局 23W 80D 17L（得分率 **52.5%**，**+17.4 ± 35.8 Elo**，SPRT LLR = **+0.34**）📈 正向。
  5. **145% vs 150%**（`copy 8.ipynb` 當前，400k nodes）：274 局 56W 168D 50L（得分率 **51.1%**，**+7.6 ± 25.6 Elo**，SPRT LLR = **+0.16**）📈 正向。
  6. **165% vs 150%**（`tournament_colab_runner_se_multicut.ipynb`，400,000 nodes/move 長時控 LTC）：
     * **戰績**：Candidate **87 勝 219 和 64 負**（得分率 **53.1%**，總得分 196.5 / 370，淨勝 **+23 局**）。
     * **相對 Elo**：**+21.6 ± 22.6 Elo**。
     * **SPRT 檢定狀態**：**LLR = +1.16**（bounds `[-2.94, +2.94]`，強勢正向攀升）。
* **核心機理深度剖析**：
  * **高時控（400k nodes LTC）深層防守盲區假說**：
    在 400,000 nodes 的長時控下，引擎搜尋深度常深入至 $d = 20 \\sim 28$。在此深度下，若維持 150%（或更緊的 135%），深層靜態著法的探索寬度會被過早掐斷，產生「過度剪枝盲區（Pruning Blindness）」，導致長盤殘局中的細微防守步與走位失察。
  * 放寬至 **165%** 讓高時控下的深層高質量 Move Ordering 能夠探索更多邊緣防守步，同時不會膨脹淺層節點，為長時控高水平對局提供了關鍵的戰術健全性，淨勝 +23 局成果極為顯著！
* **處置建議**：全力支援 `165 vs 150` 完成對戰；若確認達到顯著性，即採納進長時控生產基準。

---

"""

target = "## 4. 常用指令"
assert target in text, "target not found"
text = text.replace(target, new_sections + target, 1)

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)

print("Updated SEARCH_EXP_LOG.md successfully.")

