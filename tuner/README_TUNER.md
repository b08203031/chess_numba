# Tuner 使用說明書

這份文件詳細說明了如何使用此 SPSA 調參工具來最佳化西洋棋引擎的評估參數。本系統已針對效能進行深度優化，支援多核心並行數據生成與 Mini-batch 快速訓練，並具備嚴謹的約束條件管理系統。

## 1. 系統特色

*   **高效能訓練**：支援 **Mini-batch SPSA**，在大數據集上訓練速度極快。
*   **多進程數據生成**：自動利用所有 CPU 核心並行分析 PGN，生成速度提升數倍。
*   **嚴謹的約束系統**：防止參數出現不合理的數值（如負數獎勵、數值倒掛）。
*   **精確的評估同步**：調參器的評估函數完全鏡像主引擎邏輯，包含前哨 (Outposts)、安全機動性 (Safe Mobility) 與不可阻擋通路兵 (Unstoppable Pawns)。
*   **靈活的參數鎖定**：可選擇性調整或鎖定特定參數。

---

## 2. 完整工作流程 (Workflow)

要訓練出一個強大的引擎，請遵循以下步驟：

### 步驟 1: 生成訓練數據 (Generation)

使用 `tuner/generate_training_data.py` 從大師對局 (PGN) 中提取局面。

*   **自動分流**：程式會自動將局面分為 **中局 (Middlegame)** 與 **殘局 (Endgame)** 兩個檔案。
*   **嚴格清洗**：使用 Stockfish 作為裁判，過濾掉評估錯誤或靜態評估不穩定的局面（如剛吃子後）。
*   **並行加速**：程式會自動偵測核心數並啟動多個 Worker。

**建議指令：**

```bash
# 生成所有對局 (--games 0)，使用預設 Stockfish 路徑
python tuner/generate_training_data.py --games 0
```

**常用參數：**
*   `--pgn <file>`: 指定 PGN 檔案（預設會自動尋找 `twic1613.pgn`）。
*   `--stockfish-path <path>`: 指定 Stockfish 執行檔路徑。
*   `--workers <N>`: 指定使用的 CPU 核心數（預設為 CPU 總數 - 1）。
*   `--consistency-threshold`: 設定和棋的允許誤差範圍 (cp)。

### 步驟 2: 轉換數據格式 (Preprocess)

將生成的 `.jsonl` 文字檔轉換為調參器專用的高效能 `.npz` 二進位格式。

```bash
python tuner/preprocess_data.py
```

這會自動合併目錄下的中局與殘局檔案。如果您只想針對特定階段調參，可以指定檔案：
```bash
python tuner/preprocess_data.py --files tuner/training_data_endgame_cleaned.jsonl --output tuner/dataset_endgame.npz
```

### 步驟 3: 執行調參 (Tuning)

使用 `tuner/tuner.py` 開始訓練。

**建議指令（快速模式）：**

```bash
# 使用 Mini-batch (每次隨機取 16384 筆)，迭代 5000 次
python -m tuner.tuner --iter 5000 --batch-size 16384
```

**參數說明：**
*   `--iter <N>`: 總迭代次數。
*   `--batch-size <N>`: **(效能關鍵)** 每次迭代使用的樣本數。
    *   設為 `16384` 或 `32768` 可以大幅加快訓練速度，且通常能獲得不錯的收斂效果。
    *   若不設定，則每次都計算整個數據集的誤差（最準確但最慢）。
*   `--alpha`: 學習率（預設 20000）。數值越大變動越劇烈。
*   `--c`: 擾動幅度（預設 5.0）。

---

## 3. 進階功能：參數選擇與約束

### 參數鎖定 (Freezing & Selecting)

您可以只專注調整引擎的某個部分，例如只調「王的安全」。

*   **只調整指定參數 (`--tune`)**：
    ```bash
    python -m tuner.tuner --tune KING_SAFETY_TABLE KING_SAFETY_ATTACK_UNITS
    ```
*   **排除特定參數 (`--exclude`)**：
    ```bash
    python -m tuner.tuner --exclude MG_MATERIAL_VALUES
    ```

### 約束條件 (Constraint System)

調參器內建 `ConstraintManager`，確保參數符合西洋棋邏輯。您可以在 `tuner/tuner.py` 的 `_setup_constraints` 方法中查看或修改。

**目前的約束規則：**
1.  **範圍限制 (Range)**：
    *   獎勵項 (Bonus) 必須 >= 0 (如 `PASSED_PAWN_BONUS`, `OUTPOST_BONUS`).
    *   懲罰項 (Penalty) 必須 >= 0 (如 `BACKWARD_PAWN_PENALTY`, 因為程式邏輯是 `score -= penalty`)。
    *   例外：`ISOLATED_PAWN_PENALTY` 必須 <= 0 (因為程式邏輯是 `score += penalty`)。
2.  **單調性限制 (Monotonicity)**：
    *   **通路兵**：Rank 越高，獎勵必須越高 (`PASSED_PAWN_BONUS` 隨 Rank 遞增)。
    *   **王的安全**：攻擊單位越多，懲罰必須越重 (`KING_SAFETY_TABLE` 遞增)。
    *   **王的向性**：攻擊棋子越強 (Q>R>B>N>P)，權重越高。

---

## 4. 輸出結果

當調參器發現更低的 MSE (Mean Squared Error) 時，會自動更新輸出檔案：

*   **檔案路徑**: `tuner/tuned_constants.py`

您可以直接將此檔案的內容複製回引擎的 `constants.py`，或修改引擎程式碼以直接 import 此檔案。

## 5. 常見問題 (FAQ)與超參數調整指南

### 如何調整 Alpha (學習率) 與 C (擾動幅度)？

SPSA 演算法對這兩個參數相當敏感。如果您發現訓練效果不佳，請觀察 MSE 的變化並依據下表進行調整：

| 症狀 (Symptom) | 可能原因 | 建議調整 |
| :--- | :--- | :--- |
| **MSE 劇烈波動**，忽高忽低，無法穩定下降。 | 學習率過高，步子邁太大，跳過了最佳解。 | **降低 `--alpha`** (例如除以 2)。 |
| **MSE 下降極慢**，幾百次迭代才變動 0.000001。 | 學習率過低，收斂太慢。 | **增加 `--alpha`** (例如乘以 2)。 |
| **參數完全不動**，或者 "New Best MSE" 幾乎不出現。 | 擾動幅度過小，無法探測到更好的方向；或學習率過小。 | 優先 **增加 `--c`** (尤其是整數參數，微小的擾動可能被四捨五入為 0)。 |
| **MSE 很不穩定且很高**。 | 擾動過大，導致梯度估計充滿雜訊。 | **降低 `--c`**。 |

**經驗法則：**
*   西洋棋引擎的參數通常是整數（分），且數值較大（如 300, 500）。因此 `alpha` 通常需要設得很大（如 20000 或 50000），因為梯度往往很小（小數點後好幾位）。
*   `c` 不宜過小，建議在 2.0 到 5.0 之間，確保擾動能讓參數發生整數變化。

### 其他問題

**Q: 為什麼 Backward Pawn Penalty 是正數？**
A: 引擎內部的邏輯是 `score -= BACKWARD_PAWN_PENALTY`，所以參數本身必須是正數才能產生懲罰效果。調參器已經正確設定了這個約束。

**Q: Mini-batch 會影響準確度嗎？**
A: 會有些許雜訊，但 SPSA 本身就是隨機算法。使用 Mini-batch 能讓你在相同的時間內跑更多次迭代，通常能更快找到全域最佳解。建議最後可以用全量數據 (`--batch-size` 不設) 微調一下。
