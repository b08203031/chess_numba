# 引擎對戰與分析框架 (`tournament_analysis/`)

本目錄存放**對戰棋譜 (`tournament_results.pgn`)、局級與著法級統計分析腳本，以及搜尋實驗紀錄**。  
對戰執行參數、in-process JIT 池、SPRT 與 `match_runner.py` 完整操作請見 **[tools/TOURNAMENT_TUTORIAL.md](../tools/TOURNAMENT_TUTORIAL.md)**。

> **目錄收斂原則**：舊版一次性圖表與歷史腳本已移至 [`archive/`](../archive/)（或隨 PGN 輪替封存）；本目錄僅保留當期 PGN、分析腳本、SPRT 教材與實驗主日誌（[`SEARCH_EXP_LOG.md`](SEARCH_EXP_LOG.md)）。

---

## 1. 對戰與分析實務流程

```bash
# 1) 程式碼同步檢查（驗證純評估 A/B 時，應與 classical_old 代碼一致、僅 constants 不同）
python tools/sync_classical_old.py --diff-code

# 2) 固定節點成對對戰（長時控常採 300k~400k nodes；高併發時 --tt-mb 64 可減少多 context 記憶體壓力）
python tools/tournament.py -c 8 --games 1000 --nodes 300000 --tt-mb 64 --name1 New --name2 Old

# 3) 產出局級報告與圖表（自動依 Round 重排並行紀錄，並剔除註解干擾計算正確回合數）
python tournament_analysis/統計數據.py
python tournament_analysis/統計數據.py file1.pgn file2.pgn       # 多檔合併 + 跨檔開局查重
python tournament_analysis/統計數據.py --target New --peek      # 僅看 PGN 摘要

# 4) 產出著法級搜尋指標報告與圖表
python tournament_analysis/parse_search_stats.py --name1 New --name2 Old
```

* 系統預設關閉開局書（`OwnBook false`）、使用 `data/openings.epd` 成對換先、並啟用 Cutechess/CCRL 等級的提前裁決（`--adjudicate`：連續差距 $\ge 500\text{ cp}$ 判勝；第 40 回合後連 12 plies 在 $\pm 8\text{ cp}$ 內判和）。若對戰發生 crash 寫入 `Failure` header，統計時應剔除該組換先配對。

---

## 2. 分析腳本產出與圖表索引

| 腳本 | 文字報告 | 核心產出圖表 |
| :--- | :--- | :--- |
| [`統計數據.py`](統計數據.py) | `game_stats_report.txt` | `cumulative_wins.png`、`cumulative_score.png`、`elo_ci.png`（以完整換先對為獨立樣本計算 95% CI）、`wdl_by_color.png`、`pair_outcomes.png`、`score_by_length.png`、`length_by_result.png`、`openings_score.png`、`promotions_by_result.png`、`score_halves.png` |
| [`parse_search_stats.py`](parse_search_stats.py) | `search_stats_report.txt` | `depth_per_move.png`、`nodes_per_move.png`、`time_per_move.png`、`nps_per_move.png`、`avg_nodes_vs_depth.png`、`same_depth_node_ratio.png`、`depth_histogram.png`、`nps_histogram.png`、`nodes_histogram.png`、`depth_by_color.png`、`depth_cdf.png` |

---

## 3. 換先九宮格與 SPRT 快速判讀

**同開局交換執色九宮格（Target 持白結果 + 持黑結果）**：

| 配對 | 判讀意義 | 配對 | 判讀意義 |
| :--- | :--- | :--- | :--- |
| **WW** | 雙色皆勝，在該開局主題具明顯優勢 | **DD** | 雙方攻守均衡，衡量殘局防守韌性 |
| **WD** | 白棋贏下先手、黑棋穩守（最理想增益模式） | **DL** | 白棋僅守和、黑棋潰敗（常見於殘局或防衛評估偏弱） |
| **WL** | 高度偏向白先勝的極端開局主題 | **LW / LL** | 雙敗或反轉敗，常提示嚴重戰術盲區或開局主題劣勢 |
| **DW** | 執黑反擊強但白棋未擴大優勢 | **LD** | 執白敗、執黑和（警訊：白方進攻或時間管理有缺陷） |

**SPRT 狀態**：`Continue`（LLR 尚未碰觸 `[-2.94, +2.94]`，原理見 [`sprt_tutorial.md`](sprt_tutorial.md)）、`H1 Accepted`（達到目標 Elo 增益，可停機）、`H0 Accepted`（無顯著進步或退步，提早止損）。
