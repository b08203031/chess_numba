# 神經網路評估 (NNUE) 系統：使用說明書、架構表與未來展望

本文件詳細說明了 `ml_eval` 目錄下提供的機器學習（神經網路）評估原型的系統架構、如何使用您的 RTX 4050 進行訓練，以及未來將其深度整合至西洋棋引擎的發展藍圖。

---

## 一、 系統架構表 (Architecture Diagram)

目前的實作採用了簡化的 NNUE (Efficiently Updatable Neural Network) 架構，專為與您的 Numba 位元棋盤 (Bitboards) 引擎兼容而設計。

```text
[ 棋盤狀態 (Bitboards) ]
          │
          ▼
[ 1. 特徵提取 (Feature Extraction) ]
  - 將 12 個 64-bit 的位元棋盤 (Piece Bitboards) 展開。
  - 形成 768 維 (12 棋種 x 64 格子) 的稀疏二進位向量。
  - 1 表示有棋子，0 表示無。
          │
          ▼
[ 2. 神經網路輸入層 (Input Layer) ] -> 維度: 768
          │
          ▼ (全連接權重矩陣 W1: 768 -> 256, 加上偏差 B1)
          │
[ 3. 隱藏層 (Hidden Layer) ]       -> 維度: 256
          │
          ▼ (ReLU 激活函數: max(0, x))
          │
[ 4. 神經網路輸出層 (Output Layer) ]-> 維度: 1
          │
          ▼ (全連接權重矩陣 W2: 256 -> 1, 加上偏差 B2)
          │
[ 5. 分數轉換 (Score Conversion) ]
  - 網路輸出為 Logit 值 (即 Sigmoid 的反函數)。
  - 將 Logit 除以常數 K (約 0.005756) 轉換為 Centipawn (百分之一兵) 分數。
          │
          ▼
[ 最終評估分數 (Centipawns) ]
```

---

## 二、 使用說明書 (User Manual)

這個系統被設計為兩個分離的階段：**GPU 離線訓練** 和 **CPU/Numba 即時推理**。

### 階段 1：使用 RTX 4050 訓練神經網路
*這個步驟需要安裝 PyTorch。*

1.  **環境準備**：
    打開您的命令列 (終端機)，確保您在 Python 虛擬環境中，並且安裝了 PyTorch：
    ```bash
    pip install torch numpy
    ```
2.  **確認數據集**：
    確保 `tuner/dataset.npz` 存在。這是由引擎的調參系統 (Tuner) 所生成的數萬個局面與 Stockfish 分數標籤。
3.  **開始訓練**：
    在專案根目錄執行以下指令：
    ```bash
    python ml_eval/train.py
    ```
    - 腳本會自動偵測您的 RTX 4050 GPU (`cuda`)。
    - 它會讀取 `dataset.npz`，將其轉換為 768 維特徵，並進行 10 個 Epoch 的訓練（您可以在程式碼中修改 `epochs` 變數）。
    - 訓練完成後，網路的權重會被萃取出來，並儲存為 `.npy` 檔案在 `ml_eval/weights/` 目錄下。

### 階段 2：在 Numba 引擎中進行推理
*這個步驟不需要 PyTorch，完全相容於您現有的 Numba 引擎。*

1.  **測試推理速度與正確性**：
    確保 `ml_eval/weights/` 目錄下有剛剛生成的 `.npy` 檔案。
    ```bash
    python ml_eval/inference.py
    ```
    - 這會載入權重，編譯 Numba `@njit` 函數，並對一個測試局面（白兵在 A2）進行評估，印出 Centipawn 分數。

2.  **整合至主引擎 (範例)**：
    在您的 `chess_engine_v2/chess_engine/evaluation.py` 中，您可以這樣呼叫：
    ```python
    from ml_eval.inference import evaluate_position_nn

    # 在您的 evaluate_position 函數內：
    # 如果您想完全使用神經網路：
    # return evaluate_position_nn(piece_bbs)
    
    # 或者是將神經網路分數與您的手工評估 (HCE) 混合 (Ensemble)：
    # nn_score = evaluate_position_nn(piece_bbs)
    # hce_score = ... 原本的評估代碼 ...
    # return (nn_score + hce_score) // 2
    ```

---

## 三、 未來展望與進階升級藍圖 (Roadmap)

雖然目前的實作已經證明了可行性，但要讓它在正式比賽（或高難度戰術題）中完全超越您現有的手工評估，還需要進行以下關鍵升級：

### 1. 實作增量更新 (Incremental Updates) - 核心效能突破
**現狀**：每次評估，`inference.py` 都會重新計算 256 個隱藏神經元與 768 個特徵的乘積。
**未來**：真正的 NNUE 架構會將隱藏層的狀態緩存起來（通常放在搜尋樹的節點或 `SearchContext` 中）。
- 當白方將兵從 e2 走到 e4 時：
  `新隱藏層 = 舊隱藏層 - W1[:, e2兵的索引] + W1[:, e4兵的索引]`
- 這將第一層的計算量從 **「矩陣乘法」** 降維打擊成 **「簡單的陣列加減法」**，推理速度 (NPS) 將提升 100 倍以上。

### 2. 半卡特爾特徵 (Half-KP 或 Half-KA)
**現狀**：輸入特徵只有 768 維，這意味著網路不知道國王的位置，只能憑空猜測。
**未來**：改用基於國王位置的特徵（例如 Stockfish 的 Half-KP）。
- 輸入特徵擴充為 `41024` 維 (64 個國王位置 * 641 個棋子狀態)。
- 這樣網路就能非常精確地理解「白馬在 f3 對位於 g8 的黑王有什麼威脅」。

### 3. 從監督式學習走向強化學習 (Reinforcement Learning)
**現狀**：我們使用的是 `tuner/dataset.npz`，這些標籤是 Stockfish 給的。這叫做**監督式學習 (Supervised Learning)**，網路的上限就是 Stockfish（甚至會學到 Stockfish 的盲點）。
**未來**：
- **自對弈 (Self-Play)**：讓您的引擎使用這個神經網路自己跟自己下棋，下幾百萬盤。
- **TD-Learning (時序差分學習)**：根據最終勝負（1, 0, 0.5）來回溯更新網路的權重，而不是依賴外部引擎的評分。
- 這正是 AlphaZero 和 Leela Chess Zero 變強的核心秘密。您的 RTX 4050 完全有能力在幾天內完成數十萬盤自對弈並訓練出專屬於您引擎的獨特棋風。