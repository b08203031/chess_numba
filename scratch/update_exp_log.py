import re

path = 'tournament_analysis/SEARCH_EXP_LOG.md'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Update top summary
old_top = "> 8. **E95（PROBCUT_TT_SHORTCUT_MARGIN = 290）已採納進生產基線**（1000 局 510.0/1000，勝率 51.00%，淨勝 +20 局，+6.95 Elo，>80 步殘局淨勝 +13 局；同期 300 探索項亦得 +5.56 Elo 淨勝 +16 局，確立最優平原）；"
new_top = """> 8. **E95（PROBCUT_TT_SHORTCUT_MARGIN = 290）已採納進生產基線**（1000 局 510.0/1000，勝率 51.00%，淨勝 +20 局，+6.95 Elo，>80 步殘局淨勝 +13 局；同期 300 探索項亦得 +5.56 Elo 淨勝 +16 局，確立最優平原）；
> 9. **E96（奇異擴充 TT PV 獎勵選定 `SINGULAR_TTPV_BONUS = 65`）已採納進生產基線**（1000 局 516.0/1000，勝率 51.60%，淨勝 +32 局，+11.12 Elo，95% CI: `[-1.23, +23.50]`，LLR=+1.55；51~80 步中長盤淨勝 +35 局；同深度節點比完美持平 1.000）；"""

assert old_top in text, "old_top not found"
text = text.replace(old_top, new_top, 1)

# 2. Update E96 section
old_e96_section_start = "### E96：前向剪枝修訂回退與奇異擴充 TT PV Bonus 探索 SINGULAR_TTPV_BONUS = 65 vs 62"
idx = text.find(old_e96_section_start)
assert idx != -1, "old_e96 not found"

end_marker = "## 4. 常用指令"
end_idx = text.find(end_marker, idx)
assert end_idx != -1, "end_marker not found"

new_e96_text = """### E96：前向剪枝修訂回退與奇異擴充 TT PV Bonus 探索 `SINGULAR_TTPV_BONUS = 65` vs `62`（採納進生產基線 ADOPTED）

* **動機與原理**：
  * **前向剪枝探索（Tasks A1~A5）實測總結**：
    * 針對升變、Step 14 淺層剪枝、Capture Futility 及 Quiescence Search 之修正（包含 A3/A4 變體），在 Colab 1000 局高強度對戰（`copy 8.ipynb`）中錄得 176W 637D 187L（得分率 49.5%，-3.82 Elo，LLR=-2.41）。
    * 實測證實在無全局大規模重調參支援下，局部前向剪枝與歷史表補償容易引發葉節點震盪與剪枝失衡，判定全數回退至 commit `17f9138` 前乾淨基準。
  * **奇異擴充獎勵 `SINGULAR_TTPV_BONUS` 參數景觀探索**：
    * `54 vs 62`（`copy 5.ipynb`，1000 局）：193W 596D 211L，得分率 49.1%，-6.25 Elo，LLR=-2.59（裕度過寬導致分支膨脹，負向顯著）。
    * `62`：舊基準。
    * `70 vs 62`（`copy 6.ipynb`，1000 局）：188W 634D 178L，得分率 50.5%，+3.47 Elo，淨勝 +10 局，正向增益。
    * 依梯度內插選定 65（介於基準 62 與 70 之中間值），展開 1000 局嚴格成對換先 A/B 對局驗證。
* **實測對局戰績（1000 局成對換先開局，HEAD = e55d4e4，New=65 vs Old=62，`copy 4.ipynb`）**：
  * **總戰績**：Candidate **203 勝 626 和 171 負**（得分率 **51.60%**，總得分 516.0 / 1000，淨勝 **+32 局**）。
  * **相對 Elo**：**+11.12 Elo**（95% CI: `[-1.23, +23.50]`）。
  * **SPRT 檢定狀態**：LLR=+1.55（bounds: `[-2.94, +2.94]`，強勢正向走勢，完整跑滿 1,000 局）。
  * **分執色表現**：
    * 持白：115W 322D 63L（勝率 55.20%，淨勝 +52）
    * 持黑：88W 304D 108L（勝率 48.00%，淨負 -20）
  * **分盤長勝率**：
    * <= 25 步：86 局 0W 86D 0L（50.0%，淨勝 0）
    * 26 ~ 50 步：255 局 12W 226D 17L（49.0%，淨負 -5）
    * **51 ~ 80 步（中長盤對攻）：401 局 140W 156D 105L（勝率 54.40%，淨勝 +35 局！）**
    * > 80 步（超長殘局）：258 局 51W 158D 49L（勝率 50.40%，淨勝 +2 局）
  * **搜尋樹品質**：
    * 平均深度 19.62 vs 19.61（+0.01 ply，Welch t=+0.084, p=0.933，完全無深度衰退）。
    * 每步耗時 873.2ms vs 869.5ms（+3.7ms）。
    * 每步平均節點 249,351 vs 249,287（+0.0%），NPS 349,988 vs 350,923（-0.3%）。
    * **同深度平均節點比（d=9~24）**：比值嚴格落在 0.980 ~ 1.018，主要常規深度（d=14~20）均為 0.999 ~ 1.002，證實 SE 擴充極度精準，無任何多餘節點浪費。
* **參數景觀匯總（1000 局 scale）**：
  * `54`：-6.25 Elo (LLR=-2.59)
  * `62`：0.00 Elo (基準)
  * `65`：**+11.12 Elo (LLR=+1.55, 淨勝 +32 局)**  🏆 最佳最優點
  * `70`：+3.47 Elo (LLR=-0.37, 淨勝 +10 局)
* **處置結論**：**正式採納進生產基線 (ADOPTED / PRODUCTION BASELINE UPDATED)**，`SINGULAR_TTPV_BONUS = 65` 已同步至 `classical_old` 及 `tuner/constants_snapshot.py`。

---

"""

text = text[:idx] + new_e96_text + text[end_idx:]

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)

print("Updated SEARCH_EXP_LOG.md successfully.")

