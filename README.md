# Chess Numba Engine

使用 **Python 3.10 + NumPy + Numba `@njit`** 編寫的高效能西洋棋引擎，具備 **Classical HCE（對齊 SF11）** 與 **NNUE（LayerStack，PyTorch 訓練）** 雙評估系統、Alpha-Beta / PVS 搜尋、UCI 協定、螢幕輔助工具與自動化 SPSA / 對戰調參管線。

---

## 快速開始

### 環境需求
```bash
pip install numpy numba      # NNUE 訓練另需 PyTorch
```

### 啟動引擎與輔助工具
| 用途 | Classical HCE | Classical Old（對戰基準） | NNUE |
| :--- | :--- | :--- | :--- |
| **UCI 引擎** | `python main.py` | `python main_old.py` | `python main_nn.py` |
| **螢幕輔助 GUI** | `python assistant.py` | — | `python assistant_nn.py` |

### 常用測試與檢查
```bash
python -m tests.test_search
python -m unittest tests.test_texel_parameterization tests.test_endgame_conformance tests.test_nnue_validation
python -m tests.perft perft --depth 5
python -m tests.test_puzzle      # NNUE 版：python -m tests.test_puzzle_nn
python -m tests.benchmark        # NNUE 版：python -m tests.benchmark_nn_real
```

---

## 倉庫結構總覽

| 目錄 | 職責說明 |
| :--- | :--- |
| `chess_engine/` | 核心引擎套件：`classical/`（現行 HCE + 搜尋）、`classical_old/`（凍結對戰快照，僅用 `tools/sync_classical_old.py` 同步 `.py`）、`nnue/`（HalfKAv2_hm + `ml_eval/` 訓練/量化/推理）、`polyglot.bin`。模組詳情見 [chess_engine/README.md](chess_engine/README.md) |
| `tuner/` | Classical HCE 靜態 Texel / SPSA 調參工作台（`texel_tuner_runner.ipynb`、`data_pipeline/`、`nnue_pipeline/`） |
| `tune_search/` | 搜尋常數 SPSA 調優器（44 軸 runtime `SearchContext.tune[]` 單次 JIT + UCI compile-time 路徑） |
| `tune_eval_match/` | 對局式 HCE 評估權重 SPSA（單進程共享 JIT、`SearchContext.eval_weights`） |
| `tools/` | `tournament.py`（New vs Old 成對換先 + SPRT）、`match_runner.py`（對外部 Stockfish）、`match_core.py`、螢幕棋盤辨識 CNN |
| `tournament_analysis/` | 對戰棋譜 `tournament_results.pgn`、局級/著法級統計分析腳本與實驗日誌 |
| `tests/`、`data/`、`external/` | 單元/謎題/效能測試、開局庫與 CNN 模型、外部 Stockfish 執行檔 |

---

## 📂 專案文件導覽 (Documentation Map)

**本節為全專案技術 Markdown 的唯一主索引**（不於倉庫根目錄堆積一次性筆記；歷史檔案歸入 `archive/`）。

### 1. 核心手冊與 AI 規範（日常優先閱讀）
* **[AI 助手入口 (`AGENTS.md`)](AGENTS.md)** 與 **`.agents/rules/`**（工作流、Numba JIT、棋盤完整性、搜尋約束、HCE、NNUE、Texel 調參規範）
* **[引擎核心架構說明書](chess_engine/README.md)**：UCI、分層架構與 `classical` / `classical_old` / `nnue` 模組清單
* **[Classical HCE 調參手冊](tuner/README.md)**：20M EPD / `val_pgn` / `SF19` 混合標籤、per-dataset `K*` 分離、SPSA `scales^2` 與子桶守衛
* **[搜尋參數 SPSA 手冊](tune_search/README.md)**：96/44 軸清冊、Fishtest `inprocess` SPSA、L0→L2 SPRT 流程與非對稱峭壁經驗
* **[對局式 HCE 權重 SPSA](tune_eval_match/README.md)**：單進程共享 JIT 的 `eval_weights` 對局 SPSA
* **[引擎對戰指南](tools/TOURNAMENT_TUTORIAL.md)**、**[對戰分析框架](tournament_analysis/README.md)**、**[SPRT 統計原理](tournament_analysis/sprt_tutorial.md)**
* **[NNUE 架構設計說明](chess_engine/nnue/ARCHITECTURE.md)** 與 **[NNUE 訓練及量化說明書](chess_engine/nnue/ml_eval/ml_eval_documentation_and_architecture.md)**

### 2. 現行系統分析、審計與專題報告
* **[HCE 與 SF11 對齊審核](chess_engine/classical/HCE_SF11_GAP_AUDIT.md)**（HCE 對齊主文件）與 **[殘局 vs Stockfish 驗證](chess_engine/classical/ENDGAME_SF_VERIFICATION.md)**
* **[2026-10-31 經典評估審計與修復計畫](chess_engine/classical/EVAL_AUDIT_AND_FIX_PLAN_20261031.md)**：`evaluation.py` Bug 修復與多輪 `USE_PGN_VAL` 調參／400k 對戰驗收
* **[古典搜尋算法分析](chess_engine/classical/SEARCH_ANALYSIS.md)** 與 **[NNUE 搜尋分析](chess_engine/nnue/SEARCH_ANALYSIS.md)**：PVS、QS、剪枝及 Phase 18–29 調參結論
* **[Numba JIT 編譯與快取報告](chess_engine/classical/JIT_COMPILE_CACHE_REPORT.md)**：1 MB 表上限、packed rook、StructRef 與冷/暖快取量測
* **[搜尋 ↔ 評估介面](chess_engine/classical/SEARCH_EVAL_INTERFACE.md)**（KNOWN_WIN 護欄）、**[SEE 專題](chess_engine/classical/SEE_ANALYSIS.md)**、**[歷史啟發](chess_engine/classical/HISTORY_HEURISTICS_CN.md)**
* **[Classical 搜尋單項實驗日誌](tournament_analysis/SEARCH_EXP_LOG.md)**（E1+ 連續 A/B 實驗流水紀錄）

### 3. 已結案階段計畫與歷史戰役報告（歸檔候選）
* **[搜尋演算法深度審計與 SF19 提案 (2026-10-19)](chess_engine/classical/SEARCH_AUDIT_AND_SF19_PROPOSALS_20261019.md)**、**[搜尋優化診斷 (2026-08-04)](chess_engine/classical/SEARCH_OPTIMIZATION_DIAGNOSTIC_20260804.md)**
* **[歷史啟發重構工程計畫](chess_engine/classical/HISTORY_REFORM_PLAN.md)**、**[走步排序與歷史剪枝解耦計畫](chess_engine/classical/QUIET_ORDERING_DECOUPLING_PLAN.md)**
* **[Phase 7 歷史動力鏈與 LMR 對齊計畫](chess_engine/classical/PHASE7_HISTORY_LMR_ALIGNMENT_PLAN.md)**、**[Phase 8 空步剪枝對齊計畫](chess_engine/classical/PHASE8_NMP_SF19_ALIGNMENT_PLAN.md)**
* **[2026-07-15/16 搜尋戰役報告](tournament_analysis/REPORT_2026-07-15_16_search_campaign.md)**、**[2026-10-14 Phase 8 NMP 戰役報告](tournament_analysis/REPORT_2026-10-14_nmp_alignment_campaign.md)**、**[NMP 實驗計畫](tournament_analysis/NMP_EXPERIMENT_PLAN.md)**、**[搜尋候選提案 E47–E50](tournament_analysis/SEARCH_PROPOSALS_E47_E50.md)**

### 4. 隔離區（不進日常維護）
* `archive/`、`tuner/archive/`：歷史分析稿、2026-07 舊戰役 PGN 與早期 tuned dumps。
* `stockfish_repo/`（SF 19）、`stockfish_11/`（SF 11）、`external/`：上游參考原始碼與執行檔。

---

## 開發規範摘要

* **Numba JIT**：熱路徑保持 `@numba.njit` 相容（固定全域表 `<= 1 MB`、`SearchContext` 使用 `StructRef`）；見 [numba-jit.md](.agents/rules/numba-jit.md)。
* **棋盤與搜尋不變量**：維護 `piece_bbs` / `occupancy_bbs` / Zobrist 增量同步與根節點剪枝邊界；見 [board-integrity.md](.agents/rules/board-integrity.md)、[search-constraints.md](.agents/rules/search-constraints.md)。
* **文件與基準同步**：任何新增/刪除/更名技術 `.md` 須同步更新本頁導覽；未經對戰驗證與明確同意勿同步 `classical_old/`。

## 授權
[MIT License](LICENSE)
