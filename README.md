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

* Python 3.8+
* NumPy
* Numba

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
* `chess_engine/`: 引擎核心套件。
  * `polyglot.bin`: 共享的二進位開局庫檔案。
  * `classical/`: 經典評估版本，各模組完全自洽。
    * `constants.py`, `evaluation.py`, `search.py`, `move_generator.py` 等。
  * `nnue/`: NNUE 評估版本，包含推理邏輯。
    * `constants.py`, `evaluation.py`, `search.py` 等。
    * `ml_eval/`: 機器學習訓練、量化及推導模組。
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

為了確保專案結構清晰，所有的技術設計、演算法研析報告與配置指南均已模組化並放置於對應的子目錄下。以下為專案中所有的 Markdown 文件的完整索引與作用說明：

### 1. 引擎整體與神經網路 (NNUE) 架構

* **[引擎核心架構說明書](chess_engine/README.md)**：
  * 介紹引擎整體各個模組（UCI、搜尋、評估、移動生成、棋盤表示）的設計理念與檔案功能說明。
* **[NNUE V2 架構設計說明](chess_engine/nnue/ARCHITECTURE.md)**：
  * 詳細解析神經網絡評估（NNUE）架構，包括特徵提取 (HalfKA)、8 分桶 (LayerStack Bucketing) 設計、CReLU/SqrCReLU 激活函數與 Numba JIT 的前向推論加速。
* **[NNUE 訓練與量化流程說明](chess_engine/nnue/ml_eval/ml_eval_documentation_and_architecture.md)**：
  * 介紹 NNUE 模型特徵生成、離線數據清洗、PyTorch 模型訓練、權重量化導出與測試指標。
* **[NNUE 可行性與評估報告](chess_engine/nnue/ml_eval/ml_eval_feasibility_report.md)**：
  * 分析機器學習 (CNN 與 NNUE 路線) 取代手工評估的可行性，並評估在消費級 GPU (如 RTX 4050) 下的訓練可行性。
* **[NNUE 搜尋算法最佳化報告](chess_engine/nnue/SEARCH_ANALYSIS.md)**：
  * 解析 NNUE 搜尋引擎的 PVS 搜尋結構、多重剪枝優化與基於 `SearchContext` 的動態 UCI 開關調參設計。

### 2. 經典 (Classical) 評估與搜尋演算法

* **[搜尋算法最佳化報告](chess_engine/classical/SEARCH_ANALYSIS.md)**：
  * 深入解析主要變例搜尋 (PVS)、各種剪枝技術（Null Move, Futility, LMP, Singular Extensions）以及歷史啟發（History Heuristics）等核心搜尋機制的設計細節。
* **[靜態交換評估 (SEE) 專題報告](chess_engine/classical/SEE_ANALYSIS.md)**：
  * 詳細記錄靜態交換評估（Static Exchange Evaluation, SEE）的 Swap 演算法、射線過濾、過路兵與升變的邊界處理及性能優化。
* **[經典評估函數研析](chess_engine/classical/EVALUATION_ANALYSIS.md)**：
  * 分析 Stockfish 11 與本引擎手工評估函數（Tapered Evaluation、兵型結構、王安全）的對比與改進點。
* **[搜尋歷史啟發技術解析](chess_engine/classical/HISTORY_HEURISTICS_CN.md)**：
  * 深入探討主歷史、蝴蝶歷史、吃子歷史、延續歷史與糾錯歷史（Correction History）的原理與重力更新公式。
* **[評估函數研究與對比報告](chess_engine/classical/RESEARCH_REPORT.md)**：
  * 深度對比 Stockfish 11 評估架構（包含材質不平衡矩陣、非線性機動性表、kingDanger 系統、空間評估與主動權修正）與本引擎經典評估函數的對照分析。

### 3. 對戰、聯賽與調參工具 (Tuner)

* **[引擎對戰測試指南](tools/TOURNAMENT_TUTORIAL.md)**：
  * 使用 `tools/tournament.py` 運行引擎版本聯賽、自我對局、Elo 估算與 95% 信賴區間計算的操作與配置指南。
* **[SPSA 參數調參與 NNUE 數據管線說明](tuner/README.md)**：
  * 詳述如何使用 SPSA 調優傳統手工評估常數，以及獲取、清洗並提煉 HalfKA 特徵 NPZ 資料庫的完整步驟。
* **[搜尋參數 SPSA 調優系統說明](tune_search/README.md)**：
  * 詳述如何利用自我對弈 (Match Playing) 和進程重用、智能暖機技術來優化引擎搜尋參數。
* **[歷史診斷與訓練分析 (Legacy)](tuner/archive_old_scripts/training_analysis.md)** 與 **[失誤分析報告 (Legacy)](tuner/archive_old_scripts/blunder_summary.md)**：
  * 記錄了早期調參訓練分析與特定盲點對局的失誤排除日誌。

## 開發者注意事項

* **Numba JIT:** 大多數計算密集型函數都使用了 `@numba.njit` 裝飾器。這意味著它們會被編譯成機器碼。在修改這些函數時，必須確保所有變量類型都與 Numba 兼容（通常是 NumPy 陣列和基本數據類型）。
* **Make-Unmake 架構:** 引擎使用 "Make-Unmake" 模式來遍歷搜尋樹，這比複製整個棋盤狀態更高效。然而，這意味著必須小心維護棋盤狀態的完整性。
* **Zobrist 哈希:** 必須確保 Zobrist 哈希在每次移動和撤銷移動時都正確地增量更新，這對於置換表的正確性至關重要。

## 授權

[MIT License](LICENSE)
