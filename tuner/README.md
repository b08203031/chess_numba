# ♟️ 西洋棋引擎調參與 NNUE 訓練手冊

這個資料夾包含西洋棋引擎的兩套優化與數據處理系統：

1. **Classical SPSA Tuner**：微調傳統手寫評估函數的參數。
2. **NNUE 訓練管線**：下載、清洗並格式化深度學習 NNUE 模型的訓練數據。

---

## 📂 目錄結構與模組說明

經過整理後的 `tuner/` 目錄結構如下，各子目錄分工明確：

```
tuner/
├── README.md                  # 本說明文件
├── tuner.py                   # SPSA 主程式
├── parameters.py              # 參數映射與 theta 向量管理
├── constraint_manager.py      # 物理約束管理（確保參數符合棋理）
├── constants_to_be_tuned.py   # 待調參的評估常數（constants.py 副本）
├── tunable_eval.py            # Numba 優化、接受 theta 向量的評估函數
├── tuned_constants.py         # 調參器自動輸出的最佳參數（成果輸出）
├── generate_training_data.py  # 數據生成：大師棋譜 PGN ＋ Stockfish 評分
├── preprocess_data.py         # 數據預處理：JSONL 轉換為 NPZ 格式
├── download_twic.py           # 輔助：自動下載 TWIC PGN 棋譜
├── analyze_pgn.py             # 輔助：分析 PGN 對局 ELO 分佈與結果
│
├── nnue_pipeline/             # 🔵 NNUE 訓練數據管線
│   ├── fetch_official_data.py # 下載官方 binpack ➔ primer ➔ 轉 NPZ
│   ├── convert_bullet_bin.py  # 將 Primer 的 .bin 轉為 HalfKA NPZ
│   ├── download_hf_dataset.py # 從 HuggingFace 下載 Lichess-Stockfish 數據集
│   ├── merge_datasets.py      # 合併多個 NPZ 數據庫
│   ├── merge_tactical.py      # 將戰術題 JSONL 注入 HalfKA NPZ 母庫
│   └── add_metadata_to_dataset.py # 為舊版 .npz 補寫 metadata_json (免去重新轉檔)
│
├── diagnostics/               # 🟢 NNUE 診斷與一次性分析工具
│   ├── analyze_blunders.py    # 分析 failed_puzzles.json 中 NNUE 盲點
│   ├── analyze_dispersion.py  # 分析 NNUE 預測與 Stockfish 評分的離散度
│   ├── debug_material.py      # NNUE 量化健康度、權重分佈分析
│   ├── test_material_values.py# 測試 NNUE 對各棋子價值的認知
│   ├── inspect_npz.py         # 驗證 HalfKA NPZ 的數據並以 ASCII 還原棋盤
│   └── preview_bullet.py      # 預覽 Bullet .bin 中的原始棋盤數據
│
└── codegen/                   # 🟡 程式碼生成工具
    ├── generate_tunable_eval.py # 自動從 evaluation.py 轉譯出 tunable_eval.py
    ├── generate_indices.py    # 自動生成參數名稱與 theta 向量的索引映射
    └── new_indices.txt        # 生成的索引映射常量暫存
```

> [!NOTE]
> 外部工具如 `Primer` 原始碼、`stockfish_dir` (Stockfish 執行檔) 以及解壓工具 `primer.exe` 已被移至專案根目錄的 `external/` 資料夾，以保持 `tuner/` 目錄的乾淨。

---

## ⚙️ 系統一：Classical SPSA 調參工作流

### 步驟 1: 生成數據

執行 `generate_training_data.py`。程式會調用 Stockfish 計算 PGN 中局面的評分，轉換為連續的 Sigmoid 標籤，並過濾掉戰術不穩定與大比分傾斜的局面，最後輸出中局和殘局的 JSONL。

```bash
python tuner/generate_training_data.py --games 0 --min-elo 2400 --max-cp 200
```

### 步驟 2: 預處理轉檔

執行 `preprocess_data.py`。將生成的 JSONL 轉換為高速載入的 `.npz` 二進位格式，使後續訓練全部在 RAM 中進行。

```bash
python tuner/preprocess_data.py
```

### 步驟 3: 執行 SPSA 最佳化

執行 `tuner.py`。利用 SPSA 演算法以 Mini-batch 方式隨機擾動並優化參數，目標是將引擎的靜態評估與 Stockfish 標籤之間的 MSE (均方誤差) 降到最低。

```bash
python -m tuner.tuner --iter 5000 --batch-size 16384 --alpha 30000 --c 2.0
```

* **只調特定參數**：`--tune KING_SAFETY_TABLE`
* **排除特定參數**：`--exclude MG_MATERIAL_VALUES`

### 步驟 4: 整合回引擎

調參有新進展時，最佳參數會自動寫入 `tuner/tuned_constants.py`。請手動複製該檔案中的字典內容，覆蓋 `chess_engine/constants.py` 的對應數值即可。

---

## 🧠 系統二：NNUE 訓練數據管線

此管線用於下載和生成 HalfKA 特徵工程所需的訓練數據。

### A. 使用官方 Binpack 資料（推薦，速度最快、質量最高）

1. 執行 `fetch_official_data.py`。它會自動下載官方大師級 binpack 資料庫。
2. 調用 `external/primer.exe` 執行戰術將軍過濾與解壓。
3. 調用 `tuner/nnue_pipeline/convert_bullet_bin.py` 將其提煉為 `tuner/ultimate_halfka_farseerT75.npz`。

> [!NOTE]
> **數據格式提示**：`primer.exe` 輸出的 Bullet 二進位檔中，`opp_king_square`（非行棋方國王位置）已被預先轉換為其自身的對稱視角（例如白方行棋時，`opp_king_square` 儲存的是黑王垂直翻轉後的座標 `blackKing ^ 56`）。因此在 `convert_bullet_bin.py` 轉換特徵時不需要額外對國王做垂直翻轉。


```bash
python tuner/nnue_pipeline/fetch_official_data.py
```

### B. 使用 HuggingFace 資料庫 (已封存舊轉換腳本)

1. 執行 `download_hf_dataset.py` 從 HuggingFace 下載 Lichess-Stockfish 數據集。
   * **安全提示**：此腳本需要 HuggingFace Token。請在終端中設定環境變數：
     * Windows (PowerShell): `$env:HF_TOKEN="你的Token"`
     * Windows (CMD): `set HF_TOKEN="你的Token"`
2. （舊有的 `convert_dataset.py` 轉換腳本已被封存，目前統一使用 `convert_bullet_bin.py` 作為 HalfKAv2_hm 特徵轉換入口。）
3. 如果需要，可以使用 `merge_datasets.py` 與 `merge_tactical.py` 合併其他數據或戰術題局面。

---

## 📊 系統三：診斷與分析工具 (`tuner/diagnostics/`)

當你訓練完 NNUE 模型後，可以使用診斷工具分析模型的表現：

* **`analyze_blunders.py`**：讀取 `tests/` 目錄下的 `failed_puzzles.json`，對比正確走步與引擎走步的 NNUE 靜態評分，判斷是**模型靜態評估盲區**還是**搜尋剪枝過度**。
* **`analyze_dispersion.py`**：繪製 NNUE 預測分數與 Stockfish 真實分數的離散度圖表，檢視收斂程度。
* **`debug_material.py`**：檢查模型權重中對各子力（馬、象、車、后）在不同分桶 (Buckets) 下的估值，確保模型沒有因為量化而崩潰。
* **`inspect_npz.py`**：想肉眼看 `.npz` 訓練資料的盤面是否正確時，可執行此腳本在終端印出 ASCII 棋盤。

---

## 🛠️ 系統四：程式碼生成 (`tuner/codegen/`)

當你的評估函數 `chess_engine/classical/evaluation.py` 新增或修改了常數與算法時：

1. 執行 `generate_indices.py` 重新生成 `new_indices.txt` 的 theta 向量索引映射。
2. 執行 `generate_tunable_eval.py` 自動轉譯產生支援調參的 `tuner/tunable_eval.py` 評估程式碼。
