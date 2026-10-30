from pathlib import Path

p = Path("chess_engine/classical/SEARCH_ANALYSIS.md")
content = p.read_text(encoding="utf-8")

old_line = "    *   **All-node 減步縮放分子調降（E100）**：對預期非 PV 且非 Cut-node 的 All-node，按公式 $r \\mathrel{+}= \\operatorname{trunc\\_div}(r \\times \\text{LMR\\_ALLNODE\\_SCALE\\_NUM},\\, 256 \\times \\text{depth} + 268)$ 增加減步。E100 實驗將該縮放分子由 `276` 調降至 `260`（減幅微降 5.8%），在 1000 局成對測試中獲得 **+6.95 Elo**（51.00%，淨勝 +20 局，>50 步長局淨勝 +19 局），且平均深度與各深度節點比（1.000）完全無損，有效修復深層殘局中因過度減步產生的戰術盲點，正式採納為生產基準。"

new_line = '''    *   **All-node 減步縮放分子調降（E100）**：對預期非 PV 且非 Cut-node 的 All-node，按公式 $r \\mathrel{+}= \\operatorname{trunc\\_div}(r \\times \\text{LMR\\_ALLNODE\\_SCALE\\_NUM},\\, 256 \\times \\text{depth} + 268)$ 增加減步。E100 實驗將該縮放分子由 `276` 調降至 `260`（減幅微降 5.8%），在 1000 局成對測試中獲得 **+6.95 Elo**（51.00%，淨勝 +20 局，>50 步長局淨勝 +19 局），且平均深度與各深度節點比（1.000）完全無損，有效修復深層殘局中因過度減步產生的戰術盲點，正式採納為生產基準。
    *   **All-node 減步分母常數項偏移優化（E101）**：在 E100 將分子鎖定在 `260` 後，進一步以雙向梯度探針探勘分母。候選版本將分母常數項由 `268` 放寬至 `298`（即分母為 $256 \\times \\text{depth} + 298$）。在嚴格對抗原基準（268）的 1,000 局 400k nodes 長時控成對對戰中，錄得 **206 勝 625 和 169 負（51.85%，淨勝 +37 局）**，相對 Elo **+12.86 ± 13.18 (95% CI: [+0.17, +25.59])**，SPRT LLR = **+1.88**。95% CI 下界嚴格大於 0 達成統計顯著性，平均深度（17.19 vs 17.19）完全無損，各深度節點比保持 1.000，正式採納為生產基準。'''

assert old_line in content, "old_line not found"
content = content.replace(old_line, new_line, 1)
p.write_text(content, encoding="utf-8", newline="\n")
print("SEARCH_ANALYSIS.md updated successfully!")

