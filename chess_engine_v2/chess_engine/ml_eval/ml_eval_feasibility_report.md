# 西洋棋引擎機器學習與神經網路評估 (NNUE) 可行性報告

## 一、 背景與目標

您目前擁有一個基於 Numba (JIT) 加速的高效能 Python 西洋棋引擎。它已經具備了完整的 Alpha-Beta 搜尋樹、置換表 (Transposition Table) 以及基於專家知識的手工評估函數 (Hand-Crafted Evaluation, HCE)，包括 Tapered Evaluation、兵型結構、國王安全等。

**目標：** 探討如何利用機器學習 (Machine Learning) 或強化學習 (Reinforcement Learning) 取代或增強現有的手工評估函數，並評估您的硬體（NVIDIA RTX 4050 GPU）是否適合進行這項訓練。

---

## 二、 機器學習在西洋棋評估的應用方式

目前最頂尖的西洋棋引擎（如 Stockfish 16+、Leela Chess Zero）都已經採用了神經網路來進行局面評估。主要有兩種路線：

### 1. 深度卷積神經網路 (CNN) - AlphaZero/LC0 路線
*   **原理**：將 8x8 的棋盤轉換為多個圖像通道（例如：白兵通道、黑車通道），輸入給深層的卷積神經網路（類似影像辨識）。
*   **優點**：評估極其精準，對位置的理解非常深刻，擅長長期戰略。
*   **缺點**：**推理速度極慢 (NPS 極低)**。必須依賴強大的 GPU 才能在搜尋樹中實時計算。如果用純 CPU 或 Python 呼叫，每秒只能搜尋幾百到幾千個節點，這與您目前引擎的設計架構（高 NPS 的 Alpha-Beta 搜尋）完全衝突。

### 2. 高效能可更新神經網路 (NNUE) - Stockfish 路線 (✨ 推薦)
*   **原理**：**N**efficiently **U**pdatable **N**eural **N**etwork. 使用非常淺層但極寬的全連接神經網路 (Fully Connected Network)。輸入通常是 768 個特徵 (64 個格子 x 12 種棋子)。
*   **優點**：**推理速度極快 (NPS 幾乎不降)**。NNUE 的核心優勢在於**增量更新 (Incremental Update)**。當走一步棋時，只有兩個特徵發生改變（原來的格子變空，新格子有了棋子），網路的第一層不需要全部重新計算，只需進行簡單的向量加減法。
*   **架構**：適合與您現有的 Alpha-Beta 搜尋樹完美結合。

---

## 三、 硬體可行性評估：RTX 4050 GPU

**結論：非常適合且綽綽有餘！**

您的電腦配備了 NVIDIA RTX 4050 Laptop/Desktop GPU。對於訓練一個中小型規模的西洋棋評估神經網路（NNUE 架構）來說，這是一張非常優秀的顯示卡。

### 1. 顯存 (VRAM) 需求
*   RTX 4050 通常配備 6GB 的 GDDR6 VRAM。
*   NNUE 網路的參數量其實非常小。一個典型的 `768 -> 256 -> 32 -> 32 -> 1` 的架構，參數總大小不到 2MB。
*   即使在訓練時使用極大的 Batch Size (例如 16384 或 32768) 將數百萬個局面同時載入 GPU 記憶體，6GB 的 VRAM 也絕對足夠。

### 2. 算力 (CUDA Cores & Tensor Cores)
*   RTX 40 系列採用 Ada Lovelace 架構，具備強大的 FP32 和 FP16 計算能力，並且支援 Tensor Cores 進行矩陣乘法加速。
*   訓練一個 NNUE 網路本質上就是大量的矩陣相乘與反向傳播，RTX 4050 可以在幾分鐘到幾十分鐘內完成幾百萬個局面的多個 Epoch 訓練。

### 3. 數據準備
您目前的程式庫中已經有一個極好的基礎：`tuner/dataset.npz` (約 3MB)。這個資料集包含了數以萬計的高品質局面，並且已經標註了 Stockfish 的評估分數（Sigmoid 標籤）。我們可以直接拿這些數據來訓練神經網路！

---

## 四、 實作策略與步驟 (Implementation Plan)

為您的引擎引入 ML 評估，我們將採用 **「PyTorch 訓練 -> NumPy 導出 -> Numba 推理」** 的三步走策略。這樣既能利用 RTX 4050 進行高速訓練，又能保證引擎運行時不需要依賴笨重的 PyTorch 庫。

### 步驟 1：定義 NNUE 架構 (PyTorch)
我們將建立一個簡單的兩層網路：
*   **輸入層**：768 個神經元 (代表 64 個格子 x 12 種棋子，1 表示有棋子，0 表示無)。
*   **隱藏層**：256 個神經元，使用 ReLU 或 Clipped ReLU 激活函數。
*   **輸出層**：1 個神經元，輸出勝率 (Sigmoid)，再轉換為 Centipawn 分數。

### 步驟 2：利用 RTX 4050 進行訓練 (PyTorch)
編寫 `ml_eval/train.py`。
1.  讀取 `tuner/dataset.npz` 中的 `piece_bbs` (位元棋盤) 和 `results` (Stockfish 分數標籤)。
2.  將位元棋盤解碼為 `(Batch, 768)` 的張量。
3.  將模型移動到 `cuda:0` (您的 RTX 4050)。
4.  使用 MSE Loss 和 Adam 優化器進行訓練。
5.  訓練完成後，提取 PyTorch 模型的權重矩陣 (`Weight` 和 `Bias`)，保存為 NumPy 的 `.npy` 檔案。

### 步驟 3：在 Numba 引擎中實作快速推理 (Numba)
編寫 `ml_eval/inference.py`。
1.  這是在搜尋過程中被頻繁呼叫的部分，**絕對不能使用 PyTorch**，否則速度會慢幾千倍。
2.  讀取之前保存的 `.npy` 權重檔案。
3.  使用 `@numba.njit(fastmath=True)` 編寫神經網路的前向傳播 (Forward Pass) 函數。
4.  實作矩陣向量乘法 (`np.dot` 或手動循環)，並套用 ReLU 激活函數。

*(未來的進階優化：實作增量更新 (Incremental Update) 架構，將第一層的計算從搜尋樹中剝離，每次落子只更新差異，將推理速度提升至極致。)*

---

## 五、 預期成果與後續行動

這份報告隨後會附上兩個基礎實作腳本：
1.  `ml_eval/train.py`: 展示如何提取您的資料集並在 GPU 上訓練一個評估網路。
2.  `ml_eval/inference.py`: 展示如何將訓練好的權重匯入 Numba 環境進行高速計算。

一旦這個框架建立起來，您可以隨時用更多、更高品質的資料（例如自對弈資料，這是強化學習的核心）來重新訓練您的網路，讓引擎的棋力不斷進化！