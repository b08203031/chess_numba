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
  * `classical_old/`: 對戰基準快照（僅同步 `.py`，見 `tools/sync_classical_old.py`；**不維護 md**）。
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
  * `tournament.py` & `match_runner.py`: UCI 引擎對決與聯賽管理程式。
  * `filter_book.py`: 開局庫過濾工具。
* `data/`: 資料儲存目錄。
  * `templates/`: 棋子識別影像範本。
  * `openings.epd`: 聯賽開局庫。
* `external/`: 外部工具。
  * `stockfish/`: 包含官方 Stockfish 執行檔，用於聯賽對手或資料標記。
* `tuner/`: 經典評估函數參數 SPSA 調優器與 NNUE 訓練數據管線。
* `tune_search/`: 搜尋參數 SPSA 自動調優器。

## 📂 專案文件導覽 (Documentation Map)

技術設計、演算法說明與操作指南放在對應子目錄。**本節為專案 Markdown 的唯一主索引**；請勿在倉庫根目錄堆積 session 筆記或一次性分析報告（歷史稿可放 `archive/`）。

### 1. 引擎整體與神經網路 (NNUE)

* **[引擎核心架構說明書](chess_engine/README.md)**：UCI、搜尋、評估、移動生成、棋盤表示與 `classical` / `classical_old` / `nnue` 分工。
* **[NNUE 架構設計說明](chess_engine/nnue/ARCHITECTURE.md)**：HalfKAv2_hm、8 分桶 LayerStack、CReLU、結構重參數化與整數推理。
* **[NNUE 訓練與量化流程](chess_engine/nnue/ml_eval/ml_eval_documentation_and_architecture.md)**：資料管線、訓練、量化、Numba 推理操作說明。
* **[NNUE 搜尋分析](chess_engine/nnue/SEARCH_ANALYSIS.md)**：PVS、剪枝與 `SearchContext` UCI 開關。

### 2. 經典 (Classical) 評估與搜尋

* **[搜尋算法分析](chess_engine/classical/SEARCH_ANALYSIS.md)**：PVS、NMP / LMR / Futility / Singular 等剪枝與 move ordering。
* **[SEE 專題](chess_engine/classical/SEE_ANALYSIS.md)**：Swap 演算法、射線過濾、過路兵與升變。
* **[HCE 與 SF11 對齊審核](chess_engine/classical/HCE_SF11_GAP_AUDIT.md)**（**評估對齊主文件**）：階段 A/B/B3/B4、kingDanger、通路兵 scale 與對戰結論。
* **[殘局 vs Stockfish 驗證](chess_engine/classical/ENDGAME_SF_VERIFICATION.md)**：專用殘局說明；日常測 `tests/test_endgame_conformance.py`，可選 oracle `tools/verify_endgame_vs_sf.py`。
* **[歷史啟發](chess_engine/classical/HISTORY_HEURISTICS_CN.md)**：主 / 蝴蝶 / 吃子 / 延續 / 糾錯歷史與重力公式。

### 3. 對戰、聯賽與調參

* **[引擎對戰指南](tools/TOURNAMENT_TUTORIAL.md)**：`tools/tournament.py`、SPRT、併發與 JIT 暖機。
* **[對戰框架說明](tournament_analysis/README.md)**：本目錄角色、開局庫與 SPRT 判讀。
* **[階段 A](tournament_analysis/PHASE_A_TOURNAMENT_ANALYSIS.md)** / **[階段 B](tournament_analysis/PHASE_B_TOURNAMENT_ANALYSIS.md)** / **[階段 B4](tournament_analysis/PHASE_B4_TOURNAMENT_ANALYSIS.md)** 對戰分析。
* **[SPSA 與 NNUE 數據管線](tuner/README.md)**、**[搜尋參數 SPSA](tune_search/README.md)**。
* **[AI 助手入口](AGENTS.md)** 與 **`.agents/rules/`**（工作流、Numba、棋盤完整性、搜尋約束、NNUE 規範）。

### 4. 隔離區（不進日常索引）

* `archive/`、`tuner/archive_old_scripts/`：舊腳本與歷史分析，僅考古用。
* `stockfish_repo/`、`stockfish_11/`、`external/`：上游原始碼與工具，非本引擎文件。

## 開發者注意事項

* **Numba JIT:** 計算密集型函數使用 `@numba.njit`；修改時須保持 Numba 相容型別（NumPy 陣列與基本型別）。首次編譯可能需數分鐘。
* **Make-Unmake 架構:** 搜尋樹以原地 make/unmake 遍歷，須維護佔用位元棋盤與狀態完整性。
* **Zobrist 哈希:** 每次移動與撤銷都必須正確增量更新，置換表才可靠。
* **文件維護:** 重大功能或重構後更新對應技術文件，並同步本節導覽索引。

## 授權

[MIT License](LICENSE)
