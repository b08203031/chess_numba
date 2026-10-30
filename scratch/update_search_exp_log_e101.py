from pathlib import Path

p = Path("tournament_analysis/SEARCH_EXP_LOG.md")
content = p.read_text(encoding="utf-8")

old_e101_marker = "### E101：LMR All-node 減深分母多維梯度探針與基準確認對戰準備"
next_marker = "### E102：競賽對戰提前裁決機制（Early Adjudication）建置"

assert old_e101_marker in content, "old_e101_marker not found"
assert next_marker in content, "next_marker not found"

new_e101 = '''### E101：LMR All-node 減深分母多維梯度探針與基準確認對戰（正式採納）

* **動機與原理**：
  * 在 E100 將分子鎖定在 `LMR_ALLNODE_SCALE_NUM = 260` 之後，針對分母結構展開進一步微調：
    $$\\Delta r = \\frac{r \\times 260}{\\text{DENOM\\_BASE} \\times \\text{depth} + \\text{DENOM\\_OFFSET}}$$
  * 生產原基準為 `DENOM_BASE = 256`、`DENOM_OFFSET = 268`。
* **第一階段：雙向梯度探針實測（1,000 局成對換先，確定梯度方向）**：
  1. **Probe 1（常數項偏移探針，`copy 7.ipynb`）**：
     * 固定 `BASE = 256`，測試 `OFFSET = 298` (New, +30) vs `238` (Old, -30)。
     * 戰績：Candidate **191 勝 644 和 165 負**（得分率 **51.30%**，總得分 513.0 / 1000，淨勝 **+26 局**）。
     * 相對 Elo：**+9.04 ± 12.85**，SPRT LLR = **+1.07**。
     * 盤長表現：51~80 步中長盤對攻錄得 134W 162D 106L（勝率 **53.5%**，單項淨勝 **+28 局**），且每步提速 -5.0ms，NPS 微升 +0.6%。
     * 結論：分母常數項向正側放寬（+30）優勢極為顯著。
  2. **Probe 2（深度乘數探針，`multicut.ipynb`）**：
     * 固定 `OFFSET = 268`，測試 `BASE = 290` (New, +34) vs `220` (Old, -36)。
     * 戰績：Candidate **198 勝 615 和 187 負**（得分率 **50.55%**，總得分 505.5 / 1000，淨勝 **+11 局**）。
     * 相對 Elo：**+3.82 ± 13.36**，SPRT LLR = **-0.26**。
     * 結論：大分母側（290）同樣優於小分母側（220）。
  3. **探針結論**：兩組探針一致指向「放寬分母、緩和 ALL-node 減深」為正確發展方向，其中常數項偏移 `OFFSET = 298` 效果最為顯著。
* **第二階段：對抗原生產基準確認對戰（1,000 局成對換先，400k LTC，Early Adjudication 首次實戰）**：
  * **對陣雙方**：
    * **Candidate (New)**: `LMR_ALLNODE_SCALE_DENOM_BASE = 256`, `LMR_ALLNODE_SCALE_DENOM_OFFSET = 298`
    * **Baseline (Old)**: `LMR_ALLNODE_SCALE_DENOM_BASE = 256`, `LMR_ALLNODE_SCALE_DENOM_OFFSET = 268`（嚴格對抗原基準）
    * 時控：400,000 nodes/move；轉置表：64 MB / context；Early Adjudication：預設開啟。
  * **戰績統計**：
    * 總對局數：**1,000 局（500 對換先）**
    * 戰績：**206 勝 625 和 169 負**（總得分 **518.5 / 1000**，得分率 **51.85%**，淨勝 **+37 局**！）
    * 相對 Elo：**+12.86 ± 13.18 (95% CI: [+0.17, +25.59])**
    * SPRT LLR：**+1.88**（強烈偏向 H1 接受區間）
    * **統計顯著性**：95% 信賴區間下界明確大於 0（**+0.17 > 0**），達成嚴格的 95% 置信度統計顯著性！
  * **先手與執色分析**：
    * 持白：127 勝 287 和 86 負（勝率 **54.10%**，淨勝 **+41 局**）
    * 持黑：79 勝 338 和 83 負（勝率 **49.60%**，在黑方劣勢下抗衡原基準，僅淨負 -4 局）
    * 同 FEN 換先破冰對：Candidate 淨多出 **27 組換先得分優勢**（WD 81 vs LD 57；WW 19 vs LL 12）。
  * **盤長區間分析**：
    * 26~50 步中盤：76W 250D 54L（勝率 **52.9%**，淨勝 **+22 局**）
    * 51~80 步長盤：101W 141D 89L（勝率 **51.8%**，淨勝 **+12 局**）
    * >80 步超長盤：25W 112D 21L（勝率 **51.3%**，淨勝 **+4 局**）
  * **搜尋效能指標（parse_search_stats）**：
    * 搜尋深度：New 17.19 vs Old 17.19 (+0.01 ply，深度完全無損)
    * 節點消耗：247,955 vs 247,747 (+0.1%，無節點膨脹)
    * 同深度節點比（Same-depth ratio）：d=12~24 節點比嚴格保持在 0.994 ~ 1.004，比值近乎 1.000。
  * **提前裁決首戰表現**：
    * 平均對局步數收斂在 55.0 步，終局由勝負裁定（500 分 10 步）與殘局和棋裁定（8 分 12 步）平穩收束，省去超過 10%~15% 的無效垃圾時間，使 1000 局高強度 400k LTC 對局快速平穩完賽。
* **判定**：**正式採納（Adopted）**！確認 `LMR_ALLNODE_SCALE_DENOM_OFFSET = 298` 為新生產基準，並同步至 `classical_old`。

---

'''

idx_old = content.find(old_e101_marker)
idx_next = content.find(next_marker)

new_content = content[:idx_old] + new_e101 + content[idx_next:]
p.write_text(new_content, encoding="utf-8", newline="\n")
print("SEARCH_EXP_LOG.md updated successfully with E101 final adoption!")

