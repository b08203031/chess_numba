from pathlib import Path

p = Path("tournament_analysis/SEARCH_EXP_LOG.md")
content = p.read_text(encoding="utf-8")

e102_text = '''### E102：競賽對戰提前裁決機制（Early Adjudication）建置

* **動機與目標**：
  * 在大批量對抗與自動調參中，大量對局在優勢明朗（如領先 > 500 cp）或殘局死均勢後，仍需空轉數十甚至上百步直至 50 步規則或單王被殺，拖慢整體 A/B 迭代效率。
  * 對標 Cutechess-cli、TCEC 與 CCRL 競賽標準，在裁判核心 `tools/match_core.py` 與對戰入口 `tools/tournament.py` 建置提前裁決機制。
* **歷史數據回測（基於 894 局真實對局、116,751 個半步/plies）**：
  1. **勝負裁定（Win/Resign Adjudication）**：
     * 規則：雙方連續 5 回合（10 個半步/plies）同時認定差距 $\ge 500\text{ cp}$（領先方視角 $\ge +500$，落後方視角 $\le -500$）。
     * 回測結果：共裁定 367 場對局，**準確率 100.0%（0 次翻盤、0 次誤判）**，直接省下 11,345 個走步（**整體對抗時間節省約 10%~15%**）。
  2. **和棋裁定（Draw Adjudication）**：
     * 若直接判定「嚴格 0 cp 達 10 plies」，實測發現在第 15~25 步中局大子活躍期，引擎局部戰術重複威脅會出現暫時性 `eval=0.00`，但在變招後引發勝負搏殺，導致 15 場誤判為和棋。
     * 解決方案：對標 Cutechess 標準（`-draw movenumber=40 movecount=6 score=8`），引入開局/中局保護期（第 40 回合以後啟用）、殘局微小位置分容限（$\le 8\text{ cp}$）、持續 6 回合（12 plies），徹底杜絕中局誤判。
* **實作落地**：
  * `tools/match_core.py`：
    * `InProcessEngine` 與 `UciEngine` 新增 `last_score_cp` 欄位與 `extract_score_from_info` 雙層容錯解析。
    * `play_game` 實作走子方視角轉 White POV、連續勝負計數、殘局和棋門檻判斷與 PGN `Termination` header 標註。
    * `run_stage`、`play_pair` 完整貫通參數傳遞。
  * `tools/tournament.py`：
    * **一鍵開關**：`--adjudicate` / `--no-adjudicate`，**預設開啟**。
    * 支援自訂門檻：`--adjudicate-win-score`（預設 500）、`--adjudicate-win-moves`（預設 5）、`--adjudicate-draw-score`（預設 8）、`--adjudicate-draw-min-move`（預設 40）、`--adjudicate-draw-moves`（預設 6）。
  * `tests/test_adjudication.py`：新增 7 項單元合約測試，包含雙邊勝負裁定、和棋保護期門檻、一鍵開關、streak 中斷重置與 CLI 解析，全部 100% 通過。

---

'''

target = "### E101：LMR All-node 減深分母多維梯度探針與基準確認對戰準備"
assert target in content, "target not found"

parts = content.split("### E101：LMR All-node 減深分母多維梯度探針與基準確認對戰準備")
# Find where Section 4 begins
sec4_split = parts[1].split("\n## 4. 常用指令")
new_part = parts[0] + target + sec4_split[0] + "\n" + e102_text + "## 4. 常用指令" + sec4_split[1]

p.write_text(new_part, encoding="utf-8", newline="\n")
print("SEARCH_EXP_LOG.md updated!")

