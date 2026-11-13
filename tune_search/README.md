# 搜尋參數調優器 (`tune_search/`) 使用指南

本目錄負責 **Classical 搜尋常數（剪枝邊界、LMR、History、SEE、Correction）的對局式 SPSA 與驗證**。搜尋參數會改變搜尋樹形狀，無法靠靜態 MSE 衡量，必須透過實戰 fixed-node 對弈驗證。

---

## 1. 核心架構與設計（In-Process vs UCI）

* **單一參數清冊**：定義集中在 [`search_param_registry.py`](search_param_registry.py)（共 **96 個搜尋標量**，其中 **44 個**已映射到 `SearchContext.tune[]`，其餘為需重新 JIT 的 compile-time 軸）。`param_config.py` 僅作舊腳本相容 facade。
* **Backend 自動分流 (`--backend auto`)**：
  * **Runtime 路徑 (`inprocess`)**：SPSA 開始時配置長駐 `SearchContext` worker pool，並以 **Kiwipete Depth-1** 預先暖機一次；整個 campaign 共用同一組已 JIT 編譯機器碼，每回合只寫入 `SearchContext.tune[]` 並重置 TT / history 狀態。
  * **Compile-time 路徑 (`uci`)**：選到未進 `tune[]` 的常數時回退多進程 UCI adapter（冷/暖 JIT 成本高，不適合高迭代 SPSA）。
* **Windows 防死鎖**：adapter 與啟動指令預設使用 `-u` unbuffered I/O。

---

## 2. 硬體併發、Fishtest SPSA 與選參數原則

* **併發執行緒正確認知**：`play_match_pair` 在同一個 pair 內**依序**下完互換先後手的兩局，因此 `--concurrency N` 代表**同時進行 N 個搜尋**（而非 `2N`）。在 12 核 / 16 執行緒工作站（如 i5-12500H）上，可直接使用 `--concurrency 6..12` 量測實際 games/min，並依散熱與核心穩定度調整。
* **參數尺度與更新公式**：預設 `--scale-mode integer` 在 `SearchContext.tune[]` 邊界做 1 單位整數量化，內部 `theta` 維持浮點累積（避免小梯度證據被截斷）。依各參數自身 `c_end`、`canonical_r_end` 與 Fishtest decay schedule（`A = spsa_a_ratio * paired_games`）進行更新並寫入 JSONL / `search_spsa_state.json`。
* **常數覆蓋審查（395 個搜尋相關常數不可全塞進 SPSA）**：
  1. **結構/索引常數（約 136 個，禁止調參）**：`DIAG_*`、`STAGE_*`、`MOVE_ORDER_SOURCE_*`、棋子/分數邊界。
  2. **無序離散模式（約 5 個，禁止 SPSA）**：如 `PROBCUT_STYLE_MODE`（`0=flat, 1=SF-style, 2=off`），必須單獨以 `run_search_param_match.py` 做 A/B 對測（SPSA 可加 `--profile runtime_continuous` 自動排除）。
  3. **核心即時軸（44 個）**：主要 L1 runtime SPSA 對象，包含 Phase 7 新增槽位 39–41（`HISTORY_QUIET_MALUS_DECAY_NUM`、`LMR_DIVISOR_DEPTH_STEP`、`HISTORY_QUIET_MALUS_SCALE_NUM`），每次僅挑同一機制的 3–6 軸（Profile：`runtime`、`see`、`pruning`、`nmp`、`lmr`、`lmr_runtime`、`lmr_compile`、`qsearch`、`history`、`ordering`、`safeguards`、`aspiration`）。
  4. **Compile-time 候補軸（約 90 個）與非對稱峭壁經驗**：根據長期實驗紀錄（`SEARCH_ANALYSIS.md` Phase 22–29），多數剪枝/歷史除數門檻位於**極陡的非對稱凸優化峰頂**（例如 `HISTORY_MALUS_CAP=1500`、`PRUNING_CONTINUATION_THRESHOLD=-1400`、`LMR_DIVISOR_DEPTH_STEP=128`、`LMR_STAT_SCORE_DIV=3900`、`CORRECTION_MINOR_SCALE_NUM=170`、`LMR_CAPTURE_VICTIM_SCALE=796`），再往緊側跨出一小步常造成 -20 至 -40 Elo 斷崖。這類軸**應先用單軸/雙向邊界 A/B 掃描**（`run_search_param_match.py` 於 `200k–400k` nodes），確定值得高頻調優後再安全映射到 `SearchContext.tune[]`。

---

## 3. 標準四階段流程與指令

1. **L0 基線**：跑 `--list-params` 或固定預設做一次 warmup 確認 NPS 與深度穩定（不把 warmup 計入 NPS）。
2. **L1 SPSA 方向搜尋**（單次挑 3–6 個 runtime 軸，固定 `30k` nodes 防時間排程噪音）：

```bash
# 查看可用 profile 與參數
python -m tune_search.search_tuner --list-params

# L1 runtime SPSA（例：LMR runtime 子集）
python -m tune_search.search_tuner \
  --params LMR_BASE_OFFSET,LMR_HISTORY_SCALE,LMR_CUTNODE_BONUS,LMR_TTCAPTURE_BONUS,LMR_MOVECOUNT_FACTOR,LMR_TABLE_SCALE_PERCENT \
  --backend inprocess --scale-mode integer \
  --iter 24 --schedule-iterations 24 --games 50 --nodes 30000 --concurrency 8 \
  --openings data/openings.epd --openings-limit 5000 --spsa-a-ratio 0.1 --seed 20260804 \
  --state tune_search/states/lmr_l1.json --log tune_search/logs/lmr_l1.jsonl
```

* 中斷後保留相同配置加上 `--resume` 即可接續（`--iter` 表示「再增加幾次」；切換 `--scale-mode` 或 `--spsa-a-ratio` 須更換 `--state`）。
* SEE 專用 5 軸可用 `--profile see`（Colab 版見 `tune_colab_runner.ipynb` / Phase 7 見 `tune_colab_runner_p7.ipynb`）。

3. **日誌信號檢驗與候選匯出**（勿直接取最後一個隨機游走點，改取後半段平均）：

```bash
python tune_search/summarize_run.py tune_search/logs/lmr_l1.jsonl --tail 0.5 --out tune_search/candidate_l1.json
```
* **判讀門檻**：`observed_sd / expected_sd < 1.3` 代表結果仍在純隨機游走噪聲內（需加局數或 iter）；關注 `|drift_z| >= 2.0` 的參數。

4. **L2 獨立高節點 SPRT 驗證與回寫**：

```bash
python tools/run_search_param_match.py \
  --params-json tune_search/candidate_l1.json \
  --nodes 200000 --games 300 --concurrency 10 \
  --sprt 0,3 --batch-games 50 --adjudicate \
  --out tournament_analysis/l2_verify_cand.json
```
* L2 與使用者核准之長局對戰確認淨勝後，再更新 `chess_engine/classical/constants.py`（並執行 `python -m tuner.sync_constants_snapshot` 與合約測試）。

---

## 4. 重要選項速查與常見問題

* **開局庫**：預設讀取 `tune_search/polyglot.bin` 或 `tuner/data/pgn/twic1613.pgn`；長 campaign 請傳 `--openings-limit >= iter * games / 2` 並在 `--resume` 時維持不變，防止開局序號重複或重洗牌。
* **`--nodes` vs `--tc`**：`inprocess` 用 `--nodes`（L1 建議 `30000`，L2 `200000`）；`uci` compile-time 僅適合少量 `--iter 3 --games 18 --tc "0.1"` 短對比。
* **為何啟動要等 1–2 分鐘？** Numba 初次 JIT warmup 時間；`inprocess` 整個 campaign 只編譯一次，勿在 iteration 之間清空 cache。按 `Ctrl+C` 會優雅關閉 worker pool。
