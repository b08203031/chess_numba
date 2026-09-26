# Python Chess Engine

這是一個使用 Python 和 Numba 開發的高性能西洋棋引擎。它利用位元棋盤（Bitboards）表示法和先進的搜尋演算法，旨在提供快速且強大的分析能力。

## 特色

* **位元棋盤 (Bitboards):** 使用 64 位元整數高效表示棋盤狀態，利用位元運算加速棋子移動生成和攻擊計算。
* **Numba 加速:** 核心搜尋和評估函數使用 `@numba.njit` 進行即時編譯 (JIT)，大幅提升執行速度，接近 C++ 引擎的性能。
* **先進搜尋演算法:**
  * 迭代加深搜尋 (Iterative Deepening)
  * Alpha-Beta 剪枝
  * 主要變例搜尋 (Principal Variation Search, PVS)
  * 靜態搜尋 (Quiescence Search)
  * 置換表 (Transposition Table)
* **剪枝與歸約技術:**
  * 空著剪枝 (Null Move Pruning)
  * 晚期移動歸約 (Late Move Reductions, LMR)
  * 殺手步啟發 (Killer Heuristic)
  * 歷史啟發 (History Heuristic)
  * 徒勞剪枝 (Futility Pruning)
  * 反向徒勞剪枝 (Reverse Futility Pruning)
  * Razoring
  * 概率剪枝 (ProbCut)
  * 晚期移動剪枝 (Late Move Pruning, LMP)
  * Delta 剪枝 (Delta Pruning)
  * 靜態交換評估 (Static Exchange Evaluation, SEE)
  * 內部迭代加深 (Internal Iterative Deepening, IID)
  * 奇異延展 (Singular Extensions)
* **評估函數:**
  * 漸進式評估 (Tapered Evaluation)
  * 材質平衡
  * 棋子位置分數表 (Piece-Square Tables, PST)
  * 兵型結構 (通路兵、孤兵、重疊兵)
  * 國王安全 (兵盾、攻擊者、兵風暴、材質縮放)
  * 棋子機動性
  * 棋子協同性
* **UCI 協議:** 支援通用西洋棋介面 (UCI) 協議，可與任何標準西洋棋 GUI (如 Arena, CuteChess) 配合使用。
* **Polyglot 開局書:** 支援讀取 `.bin` 格式的 Polyglot 開局書。

## 環境需求

* Python 3.10
* NumPy
* Numba
* （NNUE 訓練另需 PyTorch）

## 安裝

1. 克隆此倉庫：

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2. 安裝依賴套件：

    ```bash
    pip install numpy numba
    ```

## 使用方法

### 作為 UCI 引擎運行

您可以運行對應的進入點腳本來啟動引擎（會等待標準 UCI 命令輸入）：

* **經典評估版本 (Classical Engine)**:

    ```bash
    python main.py
    ```

* **NNUE 評估版本 (NNUE Engine)**:

    ```bash
    python main_nn.py
    ```

在西洋棋 GUI（如 Arena、CuteChess）中，將引擎可執行命令分別設置為 `python main.py` 或 `python main_nn.py`。

### 運行螢幕輔助工具 (GUI Assistant)

您可以使用螢幕輔助程式在網頁棋盤上自動偵測並標示推薦走步：

* **經典評估版本**:

    ```bash
    python assistant.py
    ```

* **NNUE 評估版本**:

    ```bash
    python assistant_nn.py
    ```

### 運行基準測試與 Puzzle 測試

要測試引擎的性能與編譯正確性：

```bash
# 經典/NNUE 引擎搜尋正確性測試
python -m tests.test_search

# NNUE 引擎 JIT、量化與數據管線安全性單元測試
python -m unittest tests.test_nnue_validation

# 經典評估版本戰術謎題測試
python -m tests.test_puzzle

# NNUE 評估版本戰術謎題測試
python -m tests.test_puzzle_nn

# 經典/NNUE 引擎效能基準測試
python -m tests.benchmark
python -m tests.benchmark_nn_real
```

### 運行 Perft 測試

Perft (Performance Test) 用於驗證移動生成器的正確性：

```bash
python -m tests.perft perft --depth 5
```

您還可以指定 FEN 字串：

```bash
python -m tests.perft perft --fen "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1" --depth 4
```

使用 `divide` 命令查看每個首步的節點數詳細信息：

```bash
python -m tests.perft divide --depth 4
```

## 專案結構

本專案結構清晰，模組分工明確：

* `main.py`: 經典評估引擎的 UCI 進入點。
* `main_nn.py`: NNUE 評估引擎的 UCI 進入點。
* `assistant.py`: 經典評估引擎螢幕輔助 GUI 進入點。
* `assistant_nn.py`: NNUE 評估引擎螢幕輔助 GUI 進入點。
* `main_old.py`: 對戰用舊版 Classical 進入點（`chess_engine.classical_old`）。
* `chess_engine/`: 引擎核心套件。
  * `polyglot.bin`: 共享的二進位開局庫檔案。
  * `classical/`: 現行經典評估（HCE）與搜尋；各模組自洽。
    * `constants.py`, `evaluation.py`, `pawns.py`, `material.py`, `endgame.py`, `search.py`, `move_generator.py` 等。
  * `classical_old/`: 對戰基準快照；純參數 A/B 以 `tools/sync_classical_old.py --sync-code` 同步程式與搜尋常數、保留舊評估常數（**不維護 md**）。
  * `nnue/`: NNUE 評估版本，包含推理邏輯。
    * `constants.py`, `evaluation.py`, `search.py` 等。
    * `ml_eval/`: 機器學習訓練、量化及推理模組。
      * `train.py`: 訓練 `LayerStackNNUE` 模型。
      * `quantize_weights.py`: 將 PyTorch 權重導出為量化 `.npy` 格式。
      * `inference.py`: 基於 Numba JIT 的前向推理引擎。
      * `weights/`: 儲存 `*.pth` 與 `*.npy` 權重檔案。
* `tests/`: 單元測試與效能基準測試目錄。
  * `test_nnue_validation.py`: NNUE JIT、量化與數據管線安全性單元測試。
  * `test_puzzle.py` & `test_puzzle_nn.py`: 戰術謎題測試。
  * `test_search.py`: 搜尋邏輯測試。
  * `perft.py`: 移動生成器性能與正確性測試。
  * `benchmark*.py`: 引擎效能基準測試。
* `tools/`: 工具與對決管理目錄。
  * `screen_recognizer.py`: 螢幕棋盤影像識別模組。
  * `match_core.py`: 共用 in-process context 池、UCI adapter、裁判、PGN 與 SPRT。
  * `tournament.py` & `match_runner.py`: 單進程共享 JIT 的引擎對決，以及對 Stockfish 的混合式 match runner。
  * `filter_book.py`: 開局庫過濾工具。
* `data/`: 資料儲存目錄。
  * `templates/`: 棋子識別影像範本。
  * `openings.epd`: 聯賽開局庫。
* `external/`: 外部工具。
  * `stockfish/`: 包含官方 Stockfish 執行檔，用於聯賽對手或資料標記。
* `tuner/`: Classical HCE 評估參數 SPSA（`data_pipeline/`、`data/` 大檔）、以及 `nnue_pipeline/` 訓練資料。
* `tune_search/`: 搜尋參數 SPSA 調優器（單一 registry、runtime/compile-time 分層、runtime 單次 JIT fixed-node match、compile-time UCI match）。
* `tune_eval_match/`: **對局式 HCE 權重 SPSA**（單進程 / 單次 JIT，`SearchContext.eval_weights` 即時改權）。

## 📂 專案文件導覽 (Documentation Map)

技術設計、演算法說明與操作指南放在對應子目錄。**本節為專案 Markdown 的唯一主索引**；請勿在倉庫根目錄堆積 session 筆記或一次性分析報告（歷史稿可放 `archive/`）。

### 1. 引擎整體與神經網路 (NNUE)

* **[引擎核心架構說明書](chess_engine/README.md)**：UCI、搜尋、評估、移動生成、棋盤表示與 `classical` / `classical_old` / `nnue` 分工。
* **[NNUE 架構設計說明](chess_engine/nnue/ARCHITECTURE.md)**：HalfKAv2_hm、8 分桶 LayerStack、CReLU、結構重參數化與整數推理。
* **[NNUE 訓練與量化流程](chess_engine/nnue/ml_eval/ml_eval_documentation_and_architecture.md)**：資料管線、訓練、量化、Numba 推理操作說明。
* **[NNUE 搜尋分析](chess_engine/nnue/SEARCH_ANALYSIS.md)**：PVS、剪枝與 `SearchContext` UCI 開關。

### 2. 經典 (Classical) 評估與搜尋

* **[搜尋算法分析](chess_engine/classical/SEARCH_ANALYSIS.md)**：PVS、NMP / LMR / Futility / Singular 等剪枝與 move ordering。
* **[2026-08-04 搜尋優化診斷與交接](chess_engine/classical/SEARCH_OPTIMIZATION_DIAGNOSTIC_20260804.md)**：500 戰術 + 500 開局／安靜盤面的 n30k 複驗；P0–P4 diagnostics/rollback、P5 暖 history attribution，以及 P6 history-prune cont1 負尾保護。
* **[Numba JIT 編譯與快取實驗報告](chess_engine/classical/JIT_COMPILE_CACHE_REPORT.md)**（**JIT / disk cache 主文件**）：1 MB 表上限、packed rook、StructRef `SearchContext`、特化爆炸、cold/warm 量測與 checklist；後續 JIT 工作請參考並擴寫。
* **[2026-07-15/16 搜尋戰役報告](tournament_analysis/REPORT_2026-07-15_16_search_campaign.md)**：座標下降 R1–R3、結構 A/B、E1/E2、雙對戰 scale 100/85 結論與接納決策。
* **[Classical 搜尋單項實驗日誌](tournament_analysis/SEARCH_EXP_LOG.md)**：E1+ 連續 A/B（200@100k）、留下/回滾門檻、early-trash、累積基線；**E14 起診斷隨對戰**（可選 `tools/hist_diag_bench.py` 事後彙總）。
* **[NMP 實驗計畫](tournament_analysis/NMP_EXPERIMENT_PLAN.md)**：~15% 節點貢獻診斷、S1 失敗複盤、漏斗 KPI 與 Phase 0–3 中間態實驗閘道。
* **[搜尋 ↔ 評估介面](chess_engine/classical/SEARCH_EVAL_INTERFACE.md)**：KNOWN_WIN 護欄（**PR-A+B+C 已定稿**；122 局 50%）、NMP/RFP/ProbCut/Razor 與 SF11/SF18 對照。
* **[SEE 專題](chess_engine/classical/SEE_ANALYSIS.md)**：Swap 演算法、射線過濾、過路兵與升變。
* **[HCE 與 SF11 對齊審核](chess_engine/classical/HCE_SF11_GAP_AUDIT.md)**（**評估對齊主文件**）：階段 A/B/B3/B4/B5/P2、對戰結論；評估 P3 回滾；搜尋護欄交叉引用。
* **[殘局 vs Stockfish 驗證](chess_engine/classical/ENDGAME_SF_VERIFICATION.md)**：專用殘局說明；日常測 `tests/test_endgame_conformance.py`，可選 oracle `tools/verify_endgame_vs_sf.py`。
* **[歷史啟發](chess_engine/classical/HISTORY_HEURISTICS_CN.md)**：主 / 蝴蝶 / 吃子 / 延續 / 糾錯歷史與重力公式；**現用基線 Phase 1a**。
* **[歷史啟發重構工程計畫](chess_engine/classical/HISTORY_REFORM_PLAN.md)**：五階段閘門紀錄；**1a 凍結、P2 已回退、P3–5 暫緩**；下一段方向見 §11（timeman / 剪枝尺度 / HCE 單項）。
* **[搜尋優化候選提案 (E47–E50)](tournament_analysis/SEARCH_PROPOSALS_E47_E50.md)**：ProbCut 深度動態自適應、LMR ALL-Node / CutNode 減深對齊、廢除將軍延伸方案規劃。

### 3. 對戰、聯賽與調參

* **[引擎對戰指南](tools/TOURNAMENT_TUTORIAL.md)**：`tools/tournament.py`、SPRT、併發與 JIT 暖機。
* **[對戰框架說明](tournament_analysis/README.md)**：本目錄角色、成對賽果九宮格、局級/著法級統計圖表與分析腳本。
* **[SPRT 統計原理推導](tournament_analysis/sprt_tutorial.md)**：瓦爾德序貫機率比檢定（SPRT）數學推導與 LLR 邊界解析。
* **[Classical HCE 調參手冊](tuner/README.md)**（靜態資料集 SPSA、資料標籤、與 classical 同步）、**[搜尋參數 SPSA](tune_search/README.md)**、**[對局式 HCE 權重 SPSA](tune_eval_match/README.md)**（單進程 / 單次 JIT；king danger / tempo 等 runtime 權重與調參路線圖）。
* **[AI 助手入口](AGENTS.md)** 與 **`.agents/rules/`**（工作流、Numba、棋盤完整性、搜尋約束、NNUE 規範）。

### 4. 隔離區（不進日常索引）

* `archive/`、`tuner/archive/`（含 `tuned_dumps/` 歷史 SPSA 輸出）、`archive/scratch/`：舊腳本與歷史產物，僅考古用。
* `stockfish_repo/`、`stockfish_11/`、`external/`：上游原始碼與工具，非本引擎文件。

## 開發者注意事項

* **Numba JIT:** 計算密集型函數使用 `@numba.njit`；修改時須保持 Numba 相容型別（NumPy 陣列與基本型別）。首次 / 清 cache 後編譯可能需數分鐘；disk cache、1 MB 全域表上限、StructRef 與遞迴 cache 策略見 **[JIT 編譯與快取實驗報告](chess_engine/classical/JIT_COMPILE_CACHE_REPORT.md)**。
* **Make-Unmake 架構:** 搜尋樹以原地 make/unmake 遍歷，須維護佔用位元棋盤與狀態完整性。
* **Zobrist 哈希:** 每次移動與撤銷都必須正確增量更新，置換表才可靠。
* **文件維護:** 重大功能或重構後更新對應技術文件，並同步本節導覽索引。

## 授權

[MIT License](LICENSE)
