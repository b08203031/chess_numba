# Python Chess Engine

這是一個使用 Python 和 Numba 開發的高性能西洋棋引擎。它利用位元棋盤（Bitboards）表示法和先進的搜尋演算法，旨在提供快速且強大的分析能力。

## 特色

*   **位元棋盤 (Bitboards):** 使用 64 位元整數高效表示棋盤狀態，利用位元運算加速棋子移動生成和攻擊計算。
*   **Numba 加速:** 核心搜尋和評估函數使用 `@numba.njit` 進行即時編譯 (JIT)，大幅提升執行速度，接近 C++ 引擎的性能。
*   **先進搜尋演算法:**
    *   迭代加深搜尋 (Iterative Deepening)
    *   Alpha-Beta 剪枝
    *   主要變例搜尋 (Principal Variation Search, PVS)
    *   靜態搜尋 (Quiescence Search)
    *   置換表 (Transposition Table)
*   **剪枝與歸約技術:**
    *   空著剪枝 (Null Move Pruning)
    *   晚期移動歸約 (Late Move Reductions, LMR)
    *   殺手步啟發 (Killer Heuristic)
    *   歷史啟發 (History Heuristic)
    *   徒勞剪枝 (Futility Pruning)
    *   反向徒勞剪枝 (Reverse Futility Pruning)
    *   Razoring
    *   概率剪枝 (ProbCut)
    *   晚期移動剪枝 (Late Move Pruning, LMP)
    *   Delta 剪枝 (Delta Pruning)
    *   靜態交換評估 (Static Exchange Evaluation, SEE)
    *   內部迭代加深 (Internal Iterative Deepening, IID)
    *   奇異延展 (Singular Extensions)
*   **評估函數:**
    *   漸進式評估 (Tapered Evaluation)
    *   材質平衡
    *   棋子位置分數表 (Piece-Square Tables, PST)
    *   兵型結構 (通路兵、孤兵、重疊兵)
    *   國王安全 (兵盾、攻擊者、兵風暴、材質縮放)
    *   棋子機動性
    *   棋子協同性
*   **UCI 協議:** 支援通用西洋棋介面 (UCI) 協議，可與任何標準西洋棋 GUI (如 Arena, CuteChess) 配合使用。
*   **Polyglot 開局書:** 支援讀取 `.bin` 格式的 Polyglot 開局書。

## 環境需求

*   Python 3.8+
*   NumPy
*   Numba

## 安裝

1.  克隆此倉庫：
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  安裝依賴套件：
    ```bash
    pip install numpy numba
    ```

## 使用方法

### 作為 UCI 引擎運行

您可以直接運行 `main.py` 來啟動引擎，它將等待 UCI 命令。

```bash
python main.py
```

在西洋棋 GUI 中，將引擎路徑設置為 `python main.py` 的執行指令（或者您可以將其打包為執行檔）。

### 運行基準測試

要測試引擎的性能，請運行 `benchmark.py`：

```bash
python benchmark.py
```

這將對標準起始局面和 "Kiwipete" 局面進行固定深度的搜尋，並報告節點數 (Nodes)、每秒節點數 (NPS) 和搜尋時間。

### 運行 Perft 測試

Perft (Performance Test) 用於驗證移動生成器的正確性。

```bash
python perft.py perft --depth 5
```

您還可以指定 FEN 字串：

```bash
python perft.py perft --fen "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1" --depth 4
```

使用 `divide` 命令查看每個首步的節點數詳細信息：

```bash
python perft.py divide --depth 4
```

### 運行單元測試

項目包含多個單元測試腳本，用於驗證各個模組的功能。

*   `test_evaluation_features.py`: 測試評估函數的各個特徵（兵盾、國王安全等）。
*   `test_zobrist.py`: 驗證 Zobrist 哈希的增量更新是否正確。
*   `test_search.py`: 運行一個完整的搜尋測試並打印詳細統計。
*   `test_puzzle.py`: 運行一組戰術謎題來測試引擎的求解能力。
*   `test_time_limit.py`: 測試搜尋時間限制功能。

## 專案結構

*   `chess_engine/`
    *   `main.py`: 程式入口點，UCI 循環。
    *   `bitboard_utils.py`: 位元棋盤操作的輔助函數。
    *   `board_operations.py`: 棋盤狀態修改 (make_move/unmake_move)。
    *   `constants.py`: 遊戲常量 (PST, 材質值, 搜尋參數)。
    *   `core.py`: 核心功能 (Perft)。
    *   `debug_utils.py`: 調試工具。
    *   `engine_types.py`: Numba 類型定義和 SearchContext 類別。
    *   `evaluation.py`: 靜態局面評估函數。
    *   `fen_parser.py`: FEN 字串解析。
    *   `move.py`: 移動編碼和解碼邏輯。
    *   `move_generator.py`: 合法移動和戰術移動生成器。
    *   `opening_book.py`: Polyglot 開局書讀取。
    *   `search.py`: 搜尋演算法 (Iterative Deepening, Alpha-Beta, Quiescence)。
    *   `see.py`: 靜態交換評估 (SEE)。
    *   `time_manager.py`: 搜尋時間管理。
    *   `transposition_table.py`: 置換表實現。
    *   `zobrist.py`: Zobrist 哈希計算。

## 開發者注意事項

*   **Numba JIT:** 大多數計算密集型函數都使用了 `@numba.njit` 裝飾器。這意味著它們會被編譯成機器碼。在修改這些函數時，必須確保所有變量類型都與 Numba 兼容（通常是 NumPy 陣列和基本數據類型）。
*   **Make-Unmake 架構:** 引擎使用 "Make-Unmake" 模式來遍歷搜尋樹，這比複製整個棋盤狀態更高效。然而，這意味著必須小心維護棋盤狀態的完整性。
*   **Zobrist 哈希:** 必須確保 Zobrist 哈希在每次移動和撤銷移動時都正確地增量更新，這對於置換表的正確性至關重要。

## 授權

[MIT License](LICENSE)
