# 引擎對戰與測試框架 (Tournament Framework)

本目錄存放**對戰勝果、統計腳本與分析報告**。對戰執行器在 `tools/tournament.py`，設計理念參考 Fishtest 的 SPRT / 成對換先方法。

完整操作參數見 **[tools/TOURNAMENT_TUTORIAL.md](../tools/TOURNAMENT_TUTORIAL.md)**。

---

## 核心機制（摘要）

1. **SPRT**：動態樣本數，LLR 觸界即停（進步 / 無進步）。
2. **Match-Pair**：同一開局換先各一局，降低白先偏差。
3. **關閉開局書**：`OwnBook false`，比的是搜尋與評估，不是背譜。

開局庫預設路徑：`data/openings.epd`（專案根目錄下）。

---

## 如何執行

在**專案根目錄**：

```bash
python tools/tournament.py --time 1000 --games 100
```

### 常用參數（與 `tools/tournament.py` 預設一致）

| 參數 | 預設 | 說明 |
| :--- | :--- | :--- |
| `--engine1` | `main.py` | New（現行 `classical`） |
| `--engine2` | `main_old.py` | Old（`classical_old` 基準） |
| `--name1` / `--name2` | `New` / `Old` | PGN 標籤 |
| `--games` | `100` | 最大局數（偶數；成對換先） |
| `--time` | `1000` | 每步毫秒 |
| `--depth` / `--nodes` | 無 | 固定深度或節點（會覆蓋時間） |
| `-c` / `--concurrency` | `1` | 並行 worker 數 |

同步 Old 程式碼（僅 `.py`）：

```bash
python tools/sync_classical_old.py --diff
```

---

## 對戰結果分析腳本

在**專案根目錄**對 `tournament_results.pgn` 跑：

```bash
# 局級：WDL / Elo / 執色 / 長度分桶 / 對局對 / 升變 / 累積得分
python tournament_analysis/統計數據.py
python tournament_analysis/統計數據.py --target New --peek   # 只看 PGN 摘要

# 著法級：深度 / 節點 / 時間 / NPS、分執色、深度直方、同深度節點比 + 圖表
python tournament_analysis/parse_search_stats.py
python tournament_analysis/parse_search_stats.py --name1 New --name2 Old
```

| 腳本 | 產出 |
| :--- | :--- |
| `統計數據.py` | 終端報告、`game_stats_report.txt` + 局級圖表（見下） |
| `parse_search_stats.py` | `search_stats_report.txt` + 著法級圖表（見下） |

### 局級圖表（`統計數據.py`）

| 檔案 | 內容 |
| :--- | :--- |
| `cumulative_wins.png` | 累積勝場 vs 理想斜率 |
| `cumulative_score.png` | 累積得分 + 滾動勝率 |
| `wdl_by_color.png` | 持白/持黑 WDL 分組長條 |
| `score_by_length.png` | 局長分桶得分率 |
| `length_by_result.png` | 勝/負/和 局長直方圖 |
| `pair_outcomes.png` | 成對結果（WW…LL） |
| `elo_ci.png` | Elo 差 ± 95% CI |
| `openings_score.png` | 主流開局得分率 |
| `promotions_by_result.png` | 升變次數 by 結果 |
| `score_halves.png` | 前/後半場得分率 |

### 著法級圖表（`parse_search_stats.py`）

| 檔案 | 內容 |
| :--- | :--- |
| `depth_per_move.png` | 每步平均深度（含 SEM） |
| `nodes_per_move.png` | 每步平均節點 |
| `time_per_move.png` | 每步平均耗時 |
| `nps_per_move.png` | 每步平均 NPS |
| `avg_nodes_vs_depth.png` | 同深度平均節點曲線 |
| `depth_histogram.png` | 深度分箱 % 對比 |
| `nps_histogram.png` | NPS 分佈 |
| `nodes_histogram.png` | 節點分佈 |
| `depth_by_color.png` | 引擎×執色平均深度 |
| `same_depth_node_ratio.png` | 同深度節點比 New/Old |
| `depth_cdf.png` | 深度累積分佈（≤40） |

（舊的 `scratch/analyze_tournament_phase_*.py`、`_extra_analysis.py`、`inspect_pgn.py` 已併入上述兩支腳本並刪除。）

---

## 本目錄內容

| 檔案 | 用途 |
| :--- | :--- |
| `tournament_results.pgn` | 最近一輪（或累積）對戰棋譜 |
| `統計數據.py` | 局級統計（WDL / Elo / 分桶 / pair…） |
| `parse_search_stats.py` | 著法級搜尋統計與圖表 |
| `sprt_tutorial.md` | SPRT 原理補充 |
| `pair_outcomes_meaning.md` | 成對結果語意 |

HCE 對齊與階段總覽仍以 [HCE_SF11_GAP_AUDIT.md](../chess_engine/classical/HCE_SF11_GAP_AUDIT.md) 為主。

---

## SPRT 輸出判讀（摘要）

```text
After 10 pairs (20 games):
...
SPRT LLR: 0.85 bounds: [-2.94, 2.94]
SPRT Status: Continue
```

| 狀態 | 意義 |
| :--- | :--- |
| **Continue** | 尚未觸界，繼續測 |
| **H1 Accepted** | 判定 New 相對 Old 有目標水準的進步 |
| **H0 Accepted** | 判定無進步或退步，可提前中止 |

棋譜可用 Lichess / ChessBase 等開啟 `tournament_results.pgn` 複盤。
