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

為了優化執行效率並支援 Numba JIT 的特性，本系統採用以下架構：

### 1. 進程重用與生命週期管理
*   **舊版問題**：每盤棋重新啟動引擎 -> 觸發 Numba 重新編譯 -> 浪費 30~60 秒。
*   **新版設計 (Process Reuse)**：
    *   **一次性啟動**：在每次迭代（Iteration）開始時，根據並發數（Concurrency）啟動對應數量的引擎對（Pair）。
    *   **持續運行**：同一組引擎進程會連續執行多盤對局，直到該次迭代的所有對局完成。
    *   **狀態重置**：每盤棋之間發送 `ucinewgame` 指令清除雜湊表 (TT) 和歷史紀錄，確保對局獨立性。

### 2. 智能暖機 (Smart Warm-up)
*   引擎啟動後，會自動執行一次 **Kiwipete 深度 1 (Depth 1)** 的搜尋。
*   **目的**：Kiwipete 局面極其複雜，能強制 Numba 編譯絕大多數的搜尋路徑（如進攻、防禦、叫殺檢查等）。這確保了正式對局開始時，JIT 編譯已完成，引擎能立即全速運作。
*   **使用者體驗**：啟動時會顯示 `Warming up...` 提示，約需等待 1 分鐘，之後的對局將非常流暢。

### 3. 輸出緩衝處理
*   強制使用 Python 的 `-u` (Unbuffered) 模式啟動引擎，防止在 Windows 環境下因 I/O 緩衝導致的通訊死鎖 (Hang)。

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
建議準備一個 PGN 或 Polyglot (.bin) 開局庫。系統預設尋找 `tune_search/polyglot.bin` 或 `tuner/twic1613.pgn`。
*   若使用 Polyglot 書，系統會隨機漫步產生開局。
*   若使用 PGN 書，系統會隨機選取對局的前 8 半步 (4 回合) 作為開局。

### 步驟 3: 開始調優
在專案根目錄執行以下指令：

```bash
# 範例：執行 3 次迭代，每次迭代對弈 18 局 (六核並行)
# 時間控制設為每步 0.1 秒 (極快棋，用於快速驗證)
python -m tune_search.search_tuner --iter 3 --games 18 --concurrency 6 --tc "0.1"
```

**參數說明：**
*   `--iter <N>`: 總迭代次數。
*   `--games <N>`: 每次迭代的對局數。
*   `--concurrency <N>`: 同時運行的引擎**對 (Pair)** 數量。例如設為 2，則會有 2 對引擎（共 4 個進程）同時下棋。建議設為 `實體核心數 / 2`。
*   `--tc <Time>`: 時間控制。格式為 `秒+增量` (例如 `0.1+0.01`)。

### 步驟 4: 查看結果
系統會提供詳細的即時輸出：
*   **暖機進度**：顯示引擎啟動與 Kiwipete 暖機狀態。
*   **單局結果**：每當一對引擎完成一組對局（互換先後手），會顯示 `Game Pair Finished: Test Won | Base Won`。
*   **累計分數**：即時顯示當前迭代的測試組勝率與勝/負/和局數。
*   **參數更新**：迭代結束後，顯示參數的調整方向與新數值。

結果會即時保存至 `tune_search/current_params.txt`。

---

## 4. 常見問題

1.  **為什麼一開始要等很久？**
    *   這是正常的 JIT 編譯過程（暖機）。請耐心等待約 1 分鐘，看到 "Warmup complete" 後，對局速度會大幅提升。

2.  **如何終止調優？**
    *   直接按 `Ctrl+C`。由於使用了進程池，系統會嘗試優雅地關閉所有引擎進程。

3.  **如何應用最佳參數？**
    *   手動將 `tune_search/current_params.txt` 中的數值更新回 `chess_engine/classical/constants.py`。
