path = 'tournament_analysis/SEARCH_EXP_LOG.md'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# Update Last Updated Date
text = text.replace('> **最後更新：** 2026-10-28', '> **最後更新：** 2026-10-29', 1)

e101_text = """### E101：LMR All-node 減深分母多維梯度探針與基準確認對戰準備

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
* **第二階段：對抗原生產基準確認對局配置（Setup for Baseline Confirmation Battle）**：
  * **關鍵原則**：探針戰勝弱側基準（-30）不等於實打實相對原基準的純 Elo 收益。只有在原基準上勝出，才是可正式採納的生產收益。
  * **確認對戰環境設定**：
    * **Candidate (New)**: `LMR_ALLNODE_SCALE_DENOM_BASE = 256`, `LMR_ALLNODE_SCALE_DENOM_OFFSET = 298`
    * **Baseline (Old)**: `LMR_ALLNODE_SCALE_DENOM_BASE = 256`, `LMR_ALLNODE_SCALE_DENOM_OFFSET = 268`（嚴格對抗原基準）
  * 完成代碼、差分驗證與合約測試，靜候使用者排程執行。

---

"""

target = "## 4. 常用指令"
assert target in text, "target not found"
text = text.replace(target, e101_text + target, 1)

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)

print("Updated SEARCH_EXP_LOG.md with E101.")

