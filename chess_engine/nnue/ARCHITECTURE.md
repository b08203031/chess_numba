# NNUE 架構說明文件 (Deployed LayerStackNNUE)

本文檔詳細說明了本引擎目前實際部署運行的神經網路評價函數（NNUE）架構及其設計理念。

---

## 1. 架構總覽 (Architecture Overview)

目前部署運行的網路是一個多層前饋神經網路，具備 **8 個獨立的決策分桶 (LayerStack Buckets)** 與 **結構重參數化 (Structural Re-parameterization)** 技術，專門為西洋棋的稀疏特徵與 JIT 即時推論進行了極限優化。

**拓撲結構：**
`22528 (Sparse) -> 512 (Accumulator) x 2 -> 1024 (Concatenated) -> [8 Buckets x (FC2: 32 -> FC3: 32 -> FC4: 1)]`

### 第一層：共享特徵累積層 (Accumulator / Feature Transformer)
*   **輸入特徵**：HalfKAv2_hm (King-Relative Piece Square, featuring interleaved channels and horizontal mirroring). 考慮了己方國王位置（經水平鏡像映射到 a-d 檔案）與全場其餘棋子位置的組合，共 **22,528** 種特徵。
*   **維度**：**512**。能精準捕捉細微的局面結構差異。
*   **計算方式**：採用增量更新（Incremental Update）。在搜尋樹遍歷中，每走一步只需對 Accumulator 進行加減法向量運算（除了國王移動觸發全重構外），運算複雜度為 $O(1)$。
*   **激活函數**：`ClippedReLU(0.0, 127.0)`。
*   **特徵拼接 (Concat)**：將行棋方 STM 的 512 維向量與非行棋方 NSTM 的 512 維向量拼接，形成 **1024** 維向量作為決策層的輸入。

### 第二層至第四層：決策分桶層 (8 LayerStack Buckets)
決策層共有 **8 個分桶 (Buckets)**。系統依據棋盤上當前的**總子力數 (Piece Count)** 自動路由至對應的決策分桶：
$$\text{bucket} = \text{clamp}\left(\frac{\text{piece\_count} - 1}{4}, 0, 7\right)$$

每個分桶內部包含三層全連接層：
*   **FC2 (特徵融合層)**：$1024 \to 32$。經由 `ClippedReLU(0.0, 127.0)` 激活。
*   **FC3 (特徵壓縮層)**：$32 \to 32$。經由 `ClippedReLU(0.0, 127.0)` 激活。
*   **FC4 (決策輸出層)**：$32 \to 1$。線性輸出（Logit），隨後經量化縮放轉換為 Centipawn (cp) 分數。

---

## 2. 核心設計理念 (Design Philosophy)

### A. 結構重參數化 (Structural Re-parameterization)
這是本架構在工程與訓練上的核心突破：
1.  **訓練期 (Training Phase)**：
    *   為了防止分桶過多導致罕見局面（如少子殘局 Bucket 0）訓練不足，模型引入了**全域基礎流形 (Factorizer)** 加上**每個分桶獨立的殘差偏置 ($\Delta W, \Delta b$)**。
    *   全域基礎層接受全數據集的梯度流更新，而各分桶殘差只接收該分桶樣本的梯度。分桶殘差在初始化時被設為零，使 Factorizer 在訓練早期主導學習。
    *   計算公式：
        $$W_{\text{effective}} = W_{\text{fact}} + \Delta W_b$$
        $$b_{\text{effective}} = b_{\text{fact}} + \Delta b_b$$
2.  **推論期 (Inference Phase / Export)**：
    *   在訓練完成後，我們在代數上將 Factorizer 權重與各分桶殘差**精確融合**（$W_{\text{export}}[b] = W_{\text{fact}} + \Delta W_b$），直接將其導出為獨立的 8 組量化參數。
    *   因此，Numba 即時推論端**完全不需要**做任何多餘的分支相加，推理耗時降為零，保持極限 NPS。

### B. 零記憶體開銷與 JIT 優化 (Zero-Memory Overhead & JIT Optimization)
*   **整數化量化 (Integer Quantization)**：
    *   模型透過 $Q_1 = 127$ 與 $Q_{\text{hidden}} = 128$ 進行整數化。
    *   在 Numba 推理中，所有的除法與激活操作被轉化為極快的二進位位移運算（`>> 7`）和 `max/min` 邊界限制，完美契合 CPU 緩存與 SIMD 自動向量化。
*   **物理極限防爆 (Physical Weight Bounds)**：
    *   在訓練期對 FC1 稀疏層進行物理上限夾逼（Clamping），確保在多子密集的複雜局面下，累加器激活值不會輕易溢出 INT16/INT32 的數值安全區。

---

## 3. 性能指標與參數

*   **單桶決策參數**：約 3.4 萬個 dense 參數（總共 8 個桶約 27 萬個）。
*   **累加器參數 (FC1)**：約 2,300 萬個參數（由 8 個分桶共享）。
*   **訓練目標**：BCEWithLogitsLoss (WDL 勝平負機率預測)。
*   **適用資料集**：`ultimate_halfka_farseerT75.npz` (包含子力分桶特徵與 WDL 標籤)。
