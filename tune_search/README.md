# 搜尋參數調優器 (Search Parameter Tuner) 使用說明

本目錄 (`tune_search/`) 包含了一套專為搜尋參數設計的自動化調優系統。與靜態評估調參不同，搜尋參數（如剪枝邊界、LMR 深度等）無法通過靜態數據集 (MSE) 進行優化，必須通過**實戰對局 (Match Playing)** 來驗證優劣。

本系統採用 **SPSA (Simultaneous Perturbation Stochastic Approximation)** 演算法，透過讓引擎自我對弈來不斷調整參數，以提升棋力。

---

## 1. 核心原理 (Principles)

### 為什麼需要獨立的搜尋調優器？
*   **評估 (Evaluation)**：目標是讓靜態分數接近真實勝率。可以使用大師對局數據計算 MSE (均方誤差) 來訓練，速度極快。
*   **搜尋 (Search)**：目標是修剪無效分支並保留關鍵路徑。參數（例如 "RFP Margin = 250"）直接影響搜尋樹的形狀。唯一的驗證方法是**實際下棋**，看是否能贏過舊參數的引擎。

### SPSA for Match Playing
1.  **基準 (Base)**：當前最佳的一組參數。
2.  **擾動 (Perturbation)**：隨機生成一個擾動向量 $\Delta$（例如 +50, -1, +25...）。
3.  **對弈 (Match)**：
    *   生成兩組參數：$\theta_{plus} = \theta + c\Delta$ 與 $\theta_{minus} = \theta - c\Delta$。
    *   讓 $\theta_{plus}$ 與 $\theta_{minus}$ 互相比賽（例如 10 局）。
4.  **更新 (Update)**：
    *   如果 $\theta_{plus}$ 勝率高，則參數往 $+\Delta$ 方向移動。
    *   如果 $\theta_{minus}$ 勝率高，則參數往 $-\Delta$ 方向移動。
    *   移動步長由學習率 (Learning Rate) 決定，隨迭代次數衰減。

---

## 2. 系統架構 (Architecture)

本系統由三個主要部分組成：

### 1. 參數配置 (`param_config.py`)
定義了所有需要調整的參數、預設值、範圍 (Min/Max) 以及步長 (Step)。
*   **用途**：集中管理參數定義。
*   **修改**：若要新增調優參數（如新的剪枝常數），請在此檔案中添加。

### 2. 引擎適配器 (`engine_adapter.py`)
這是引擎的啟動入口。
*   **功能**：它接受命令行參數（例如 `--RAZORING_MARGIN 600`），並在**導入主引擎邏輯之前**，動態修改 `chess_engine.constants` 中的全域變數。
*   **關鍵技術**：由於引擎使用 Numba JIT 編譯，常數通常會在編譯時被「燒錄」進去。此適配器確保在 JIT 編譯發生前完成參數替換，從而讓優化後的機器碼生效。

### 3. 調優控制器 (`search_tuner.py`)
這是執行調優的主程式。
*   **功能**：
    *   實作 SPSA 演算法。
    *   內建 Python 對弈執行器 (Match Runner)，使用 `python-chess` 進行並發對局。
    *   自動啟動兩個子進程 (Engine Process)，分別載入不同的參數進行對戰。
    *   記錄並更新最佳參數。

---

## 3. 操作步驟 (Operation Steps)

### 步驟 1: 配置參數 (可選)
打開 `tune_search/param_config.py`，確認 `SEARCH_PARAMS` 字典中包含了你想要調整的參數。
```python
SEARCH_PARAMS = {
    "RAZORING_MARGIN": (700, 300, 1200, 50), # 格式: (預設值, 最小值, 最大值, 推薦步長)
    ...
}
```

### 步驟 2: 準備開局庫
為了確保對局的多樣性，建議準備一個 PGN 開局庫。系統預設尋找 `tuner/twic1613.pgn`。若無開局庫，系統將從標準起始局面開始（容易導致重複和棋）。

### 步驟 3: 開始調優
在專案根目錄執行以下指令：

```bash
# 範例：執行 100 次迭代，每次迭代對弈 20 局 (雙核並行)
# 時間控制設為每步 0.05 秒 (極快棋，用於快速驗證)
python tune_search/search_tuner.py --iter 100 --games 20 --concurrency 2 --tc "0.05+0"
```

**參數說明：**
*   `--iter <N>`: 總迭代次數 (SPSA Iterations)。
*   `--games <N>`: 每次迭代中，Plus 與 Minus 參數互相比賽的局數。越多越準確，但越慢。建議至少 10-20 局。
*   `--concurrency <N>`: 同時進行的對局數 (建議設為 CPU 核心數的一半)。
*   `--tc <Time>`: 時間控制 (Time Control)。格式為 `秒+增量` (例如 `0.1+0.01`) 或純秒數 (例如 `0.1`)。
    *   **注意**：由於這是 Python 控制的極快棋，建議使用純秒數（如 `0.05` 或 `0.1`），視為 "Movetime" 模式。

### 步驟 4: 查看結果
*   **即時輸出**：程式會顯示每次迭代的勝率 (Score) 以及參數的更新方向。
*   **保存檔案**：當前最佳參數會即時寫入 `tune_search/current_params.txt`。

---

## 4. 常見問題與注意事項

1.  **啟動緩慢 (Warm-up)**：
    *   由於 Numba 需要 JIT 編譯，引擎啟動後的第一個搜尋動作會非常慢（約 10-30 秒）。
    *   `engine_adapter.py` 已設置較長的超時時間來容忍此現象。
    *   **建議**：不要將 `tc` 設得太短（低於 0.01s），否則首步可能會超時敗判。

2.  **如何應用結果？**
    *   調優結束後，打開 `tune_search/current_params.txt`，將裡面的數值手動更新回 `chess_engine/constants.py` 中對應的常數定義。

3.  **無法找到 `c-chess-cli`？**
    *   本系統內建了 Python 對弈功能，**不需要** 安裝外部的 `c-chess-cli` 工具，開箱即用。
