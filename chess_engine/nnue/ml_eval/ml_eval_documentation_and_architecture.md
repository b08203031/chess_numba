# 神經網路評估 (NNUE) 系統：使用說明書、架構表與未來展望

本文件詳細說明了在 `chess_engine/nnue/ml_eval/` 底下已實作的機器學習評估系統架構、如何使用您的 RTX 4050 進行訓練，以及引擎的推理效能設計。

---

## 一、 實作系統架構

詳細的神經網路架構（8 分桶 LayerStackNNUE）、HalfKA 特徵定義與結構重參數化數學，請參閱上一層的核心說明：**[ARCHITECTURE.md](../ARCHITECTURE.md)**。

本手冊將專注於離線訓練管線的操作與資料準備。

---

## 二、 使用說明書 (User Manual)

這個系統被設計為兩個分離的階段：**GPU 離線訓練** 和 **CPU/Numba 即時增量推理**。

### 階段 1：使用 RTX 4050 訓練神經網路
*這個步驟需要安裝 PyTorch。*

1.  **環境準備**：
    確保您在 Python 虛擬環境中，並且安裝了 PyTorch 與 NumPy：
    ```bash
    pip install torch numpy
    ```
2.  **確認數據集**：
    確保數據集 `tuner/ultimate_halfka_farseerT75.npz` 存在（可以藉由數據下載與轉換腳本生成）。
3.  **開始訓練**：
    執行以下指令：
    ```bash
    python chess_engine/nnue/ml_eval/train.py
    ```
    - 腳本會自動偵測您的 RTX 4050 GPU (`cuda`)。
    - 它會以大 Batch Size (如 8192) 載入數據，並實作差異化學習率（LR）和防爆梯度裁剪。
    - 訓練進程會將最優模型儲存於 `weights/best_model.pth`，並自動呼叫 `export_weights` 匯出量化檔案。

### 階段 2：量化導出與 Numba 推理
*這個步驟不需要 PyTorch，完全相容於您現有的 Numba 引擎。*

1.  **手動量化與導出**：
    若要手動將訓練好的 `best_model.pth` 量化導出為推理用的整數 `.npy` 檔案：
    ```bash
    python chess_engine/nnue/ml_eval/quantize_weights.py
    ```
    這將在 `weights/` 底下生成 `fc1_weight.npy`, `fc1_bias.npy` 及 8 個分桶的 `fc2/fc3/fc4` 權重與偏置檔案。

2.  **測試推理速度與正確性**：
    ```bash
    python chess_engine/nnue/ml_eval/inference.py
    ```
    - 這會載入所有量化權重，編譯 Numba `@njit` 函數，對起始局面進行評估，並進行 10,000 次速度基準測試，回報每秒節點數 (NPS)。

3.  **增量更新的核心效率設計**：
    - 推理引擎已與搜尋樹 `SearchContext` 高度整合。
    - 每次 make/unmake move 時，皆是調用增量更新 `update_accumulator` 計算特徵差異，只進行極快速的 `int16`/`int32` 向量加減，將推理時間降至最低。

---

## 三、 展望與調優

1.  **數據庫擴充**：
    可透過 `tuner/nnue_pipeline/` 目錄下的多個轉檔與合併工具，將大師棋譜與戰術謎題進行聯集，擴增模型的泛化能力。
2.  **參數微調**：
    網路各層的量化縮放係數 $Q_1=127$, $Q_{hidden}=128$ 已經過優化，既可防止溢出，又可完全利用二進位位移 (Bit-Shift) 加速乘除法運算。