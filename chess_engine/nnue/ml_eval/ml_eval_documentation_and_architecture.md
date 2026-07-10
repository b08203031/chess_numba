# 神經網路評估 (NNUE) 系統：使用說明書

本文件說明 `chess_engine/nnue/ml_eval/` 的**離線訓練、量化與推理操作**。網路拓撲與數學細節見 **[ARCHITECTURE.md](../ARCHITECTURE.md)**；資料管線與 SPSA 總覽見 **[tuner/README.md](../../../../tuner/README.md)**。

---

## 一、目錄分工

| 位置 | 角色 |
| :--- | :--- |
| `chess_engine/nnue/ml_eval/train.py` | PyTorch 訓練 LayerStackNNUE |
| `chess_engine/nnue/ml_eval/quantize_weights.py` | `.pth` → 推理用 `.npy` |
| `chess_engine/nnue/ml_eval/inference.py` | Numba 整數前向與暖機 |
| `chess_engine/nnue/ml_eval/weights/` | `*.pth` / `*.npy` / `metadata.json` |
| `tuner/nnue_pipeline/` | 下載 binpack、Primer、轉 HalfKA NPZ、合併資料集 |
| `tuner/ultimate_halfka_farseerT75.npz` | 預設訓練資料集（由管線產生） |

**所有命令皆在專案根目錄執行。**

---

## 二、流程：資料 → 訓練 → 量化 → 推理

### 階段 0：外部工具

`tuner/nnue_pipeline/fetch_official_data.py` 會呼叫 `external/primer.exe`。若只有 Primer 原始碼，請先編譯或將執行檔放到 `external/primer.exe`；也可手動準備 Bullet `.bin`，再只跑轉換腳本。

### 階段 1：生成 HalfKAv2_hm 訓練資料

推薦官方 Stockfish binpack。預設面向約 16GB RAM / 6GB VRAM：下載 1024MB binpack、Primer 上限約 32M 局面。

```bash
python tuner/nnue_pipeline/fetch_official_data.py --target-mb 1024 --limit 32000000
```

典型產物：

- `tuner/official_data_farseerT75.binpack`
- `tuner/official_data_farseerT75.bin`（Primer Bullet records）
- `tuner/ultimate_halfka_farseerT75.npz`（`train.py` 讀取）

轉換細節（特徵順序、WDL target）由 `tuner/nnue_pipeline/convert_bullet_bin.py` 處理：Primer raw `Value` → UCI cp（`cp = value * 100 / 208`）→ `sigma(0.0025 * cp)`。

> [!IMPORTANT]
> **國王格子視角歸一化**：Primer 寫入 `.bin` 時，非行棋方國王座標已映到 NSTM 自身視角。轉換器讀 NSTM 時不必再對國王做 `^ 56`。一致性測試必須重現此歸一化，否則特徵桶會偏移。

其他管線工具（見 `tuner/README.md`）：

```bash
python tuner/nnue_pipeline/convert_bullet_bin.py
python tuner/nnue_pipeline/merge_datasets.py
python tuner/nnue_pipeline/merge_tactical.py
```

### 階段 2：訓練

改過 feature order 或 target scale 後，不要沿用舊 `.npz` / 舊 `best_model.pth`。VRAM 不足時可將 `BATCH_SIZE` 自 8192 降到 4096。

```bash
python chess_engine/nnue/ml_eval/train.py
```

訓練會：

- 讀取 `tuner/ultimate_halfka_farseerT75.npz`（或腳本內相對路徑回退）
- 使用 `BCEWithLogitsLoss`、AMP、gradient clipping
- FC1：`weight_decay=0`；dense / factorizer / delta：`weight_decay=0.001`
- 寫入 `chess_engine/nnue/ml_eval/weights/latest_model.pth`、`best_model.pth`
- 驗證集創新高時自動導出推理用 `.npy`

### 階段 3：手動重新量化（可選）

```bash
python chess_engine/nnue/ml_eval/quantize_weights.py
```

### 階段 4：Numba 推理自檢

```bash
python chess_engine/nnue/ml_eval/inference.py
```

推理路徑對全域 NumPy 權重通常關閉長期 `cache=True` 固定，改以 `warmup_numba_kernels()` 在載入後集中 JIT。輸出 cp 除數與 `sigma(0.0025 * cp)` 對齊為 **41**（`127 × 128 × 0.0025 ≈ 40.64`）。

單元測試：

```bash
python -m unittest tests.test_nnue_validation
```

---

## 三、量化與調優備註

1. **資料擴充**：可合併官方 binpack、HF 資料與戰術題（`tuner/nnue_pipeline/`、`tuner/diagnostics/`）。
2. **量化係數**：$Q_1=127$、$Q_{\text{hidden}}=128$；最終 cp divisor = 41。
3. **架構不變性**：訓練期 Factorizer + 分桶殘差，導出期融合為 8 組獨立權重；細節見 [ARCHITECTURE.md](../ARCHITECTURE.md)。
