# 神經網路評估 (NNUE) 系統：使用說明書、架構表與未來展望

本文件詳細說明了在 `chess_engine/nnue/ml_eval/` 底下已實作的機器學習評估系統架構、如何使用您的 RTX 4050 進行訓練，以及引擎的推理效能設計。

---

## 一、 實作系統架構

詳細的神經網路架構（8 分桶 LayerStackNNUE）、HalfKA 特徵定義與結構重參數化數學，請參閱上一層的核心說明：**[ARCHITECTURE.md](../ARCHITECTURE.md)**。

本手冊將專注於離線訓練管線的操作與資料準備。

---

## 二、 使用說明書 (User Manual)

目前這個 workspace 的 NNUE 腳本位於根目錄；所有命令請從 workspace 根目錄執行。完整流程分為：**資料生成 → GPU 離線訓練 → 量化導出 → Numba 推理測試**。

### 階段 0：確認外部工具

`fetch_official_data.py` 會呼叫 `external/primer.exe`。如果只有 Primer 原始碼，請先自行編譯或把可執行檔放到 `external/primer.exe`；否則可先手動準備 Bullet `.bin`，再直接跑 `convert_bullet_bin.py`。

### 階段 1：生成 HalfKAv2_hm 訓練資料

推薦使用官方 Stockfish binpack 資料。預設設定針對 16GB RAM / 6GB VRAM：下載 `1024MB` binpack，Primer 最多輸出 `32,000,000` 筆 Bullet positions，轉出的未壓縮訓練陣列約 `3.96 GiB`。

```bash
python fetch_official_data.py --target-mb 1024 --limit 32000000
```

流程會依序產生：

- `tuner/official_data_farseerT75.binpack`：截斷下載的官方來源資料。
- `tuner/official_data_farseerT75.bin`：Primer 輸出的 Bullet 32-byte records。
- `tuner/ultimate_halfka_farseerT75.npz`：`train.py` 實際使用的 HalfKAv2_hm sparse feature dataset。

`convert_bullet_bin.py` 會使用 Stockfish 官方 HalfKAv2_hm feature order（每個視角 22,528 個索引，白/黑兩個 accumulator 共用同一張 FC1 表），並把 Primer raw `Value` 先轉成 UCI centipawns：`cp = value * 100 / 208`，再套用 `sigma(0.0025 * cp)` 生成 WDL target。

> [!IMPORTANT]
> **數據特徵提取中的國王格子視角歸一化 (King Square Perspective Normalization)**：
> - **Bullet 二進位格式**：由 `primer.exe` 輸出的 32 節點結構中，`king_square` 與 `opp_king_square` 分別代表行棋方（STM）與非行棋方（NSTM）的國王格子。
> - **Primer 預先歸一化設計**：為了加速推理與簡化計算，`primer` 寫入 `.bin` 時，已將 `opp_king_square` 預先映射至 NSTM 自己的視角。
>   - *白方行棋時*：白王（STM）正常儲存，黑王（NSTM）則寫入垂直翻轉後的座標：`opp_king_square = blackKing.flippedVertically()` (即 `bk_sq_board ^ 56`)。
>   - *黑方行棋時*：黑王（STM）寫入垂直翻轉座標：`king_square = blackKing.flippedVertically()`，白王（NSTM）直接寫入原始座標：`opp_king_square = whiteKing`。
> - **轉換器設計**：因此 `convert_bullet_bin.py` 在提取 NSTM 特徵時，直接讀取 `bk_sq_sym = bk_sq` 即可（無須在轉換器中重複進行 `^ 56`）。
> - **測試驗證**：若要進行特徵提取的一致性比對測試，模擬輸入必須重現此歸一化邏輯，否則會使 NSTM King 的分桶索引偏移 28 個桶（差值 `19712`）。


### 階段 2：從零訓練 NNUE

改過 feature order 或 target scale 後，不要沿用舊 `.npz` 或舊 `best_model.pth`。測試 VRAM 是否足夠時，先保留目前 `BATCH_SIZE = 8192`；若 CUDA OOM，再降到 `4096`。

```bash
python train.py
```

訓練會：

- 讀取 `tuner/ultimate_halfka_farseerT75.npz`。
- 使用 `BCEWithLogitsLoss`、AMP、gradient clipping。
- 對 FC1 使用 `weight_decay=0`，dense/factorizer/delta 使用 `weight_decay=0.001`。
- 儲存 `weights/latest_model.pth`、`weights/best_model.pth`。
- 驗證集創新高時自動呼叫 `export_weights(model)`，輸出推理用 `.npy`。

### 階段 3：必要時手動重新量化

若已有 `weights/best_model.pth`，但想重新輸出 `.npy` 權重：

```bash
python quantize_weights.py
```

### 階段 4：測試 Numba 推理

```bash
python inference.py
```

推理核心目前關閉 Numba `cache=True`，避免全域 NumPy 權重被 stale cache 固定；`warmup_numba_kernels()` 會在權重載入後集中完成 JIT 編譯。最終 cp divisor 使用與 `sigma(0.0025 * cp)` 數學對齊的 `41`（`127 × 128 × 0.0025 ≈ 40.64`）。

---

## 三、 展望與調優

1. **數據庫擴充**：
    可透過 `fetch_official_data.py`、`convert_bullet_bin.py` 與後續合併工具，將官方 binpack、大師棋譜與戰術題進行聯集，擴增模型的泛化能力。
2. **參數微調**：
    網路各層的量化縮放係數 $Q_1=127$, $Q_{hidden}=128$ 已經過優化，既可防止溢出，又可完全利用二進位位移 (Bit-Shift) 加速乘除法運算；最終 cp divisor 採用與 $\sigma(0.0025 \times \text{cp})$ 數學對齊的 `41`。
