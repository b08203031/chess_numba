# Classical HCE 調參手冊（`tuner/`）

本目錄負責 **Classical HCE 評估參數的 Texel SPSA / Grid 調優**，以及獨立的 **NNUE 訓練資料管線**。  
搜尋參數請用 [`tune_search/`](../tune_search/README.md)；NNUE 訓練請看 [`chess_engine/nnue/ml_eval/`](../chess_engine/nnue/ml_eval/ml_eval_documentation_and_architecture.md)。

> **核心原則**：對齊 SF11 結構後，目標是在本引擎 ~100 cp 尺度找最優權重。**MSE 下降 ≠ Elo 一定上升**——靜態閘門過關後，仍須經使用者同意以 `tools/tournament.py` 實戰驗收。

---

## 1. 資料來源與 SF19 混合標籤

| 用途 | 來源 / 說明 | 本機路徑 |
| :--- | :--- | :--- |
| **主訓練 + Dev + Holdout（推薦）** | E12.33/41/52 + Lichess-big3 平靜 EPD（~19.63M，含 6.07% 易位權） | `tuner/data/hce/dataset_epd_wdl_20m.npz` |
| **SF19 搜尋側車標籤（選用）** | `annotate_sf_labels.py` 產生，供 `USE_SF_LABELS = True` 動態混合（支援續傳） | `tuner/data/hce/dataset_epd_wdl_20m.sf19.npz` |
| **獨立實戰驗證集** | 本引擎高節點（≥150k）對戰 PGN（~1.59M，含 `castling!=0` 易位權桶） | `tuner/data/hce/val_pgn.npz` |
| **跨來源平靜仲裁集（選用）** | Zurichess quiet-labeled EPD（~1.08M，含 13.53% 易位權） | `tuner/data/hce/val_zurichess.npz` |
| **NNUE 專用資料** | HalfKA `.npz` / Bullet `.bin`（**勿直接用於 HCE**；勿刪既有數 GB 檔） | `tuner/data/nnue/` |

查看目前資料與狀態：`python -m tuner.run status`

---

## 2. 推薦工作流：互動式 Notebook（`texel_tuner_runner.ipynb`）

[`texel_tuner_runner.ipynb`](texel_tuner_runner.ipynb) 透過 [`notebook_session.py`](notebook_session.py) 在單一 Kernel 載入一次 20M 資料集 + 獨立驗證集（`MAX_TRAIN_ROWS="auto"` 在 RAM 足夠時載滿、約 3.4–3.7 GB RAM，免重載與免重編譯 Numba）。重跑 `# 0` 會呼叫 `session.reconfigure(...)` 在 ~3 秒內切換模式：

* **`USE_PGN_VAL` / `USE_ZURICHESS_VAL`**：分別啟用 `val_pgn.npz` 與 `val_zurichess.npz` 獨立純 WDL 驗證閘門。
* **`TARGET_ENGINE`**：`"classical"` 或 `"classical_old"`；輸出與備份自動隔離至 `tuner/runs/<engine>_<mode>[_sfXX]/`。
* **`RANDOM_SEED`**（Notebook 預設 `True` / `TexelTuningSession` 預設 `False`）：為每個 block 產生動態時間種子並寫入紀錄，避免反覆調參重複相同 batch 序列。
* **`USE_SF_LABELS` & `SF_LAMBDA`**（預設 `0.25`）：在 `train_idx` 的平靜競爭局面（`quiet_annotated_mask` 且 `|sf_cp| <= 1000 cp`，避免殘局庫極端大比分壓低 $K_{\text{sf}}$）擬合 $K_{\text{sf}}$ 並合成 $y_{\text{mix}} = (1-\lambda)\,\text{WDL} + \lambda\,\sigma(K_{\text{sf}}\cdot \text{sf\_cp})$。
  * **`train_idx` vs `dev_idx` 分布一致**：`train_idx` 與 `dev_idx` 皆使用 `annotated_mask`（**不再從 `train_idx` 剔除 `|sf_cp - sf_d1_cp| > 180` 局面**，而是讓這類戰術跳分局面保留並退回純 `WDL`），避免刪除動態突破成功樣本導致的存活者偏差。
* **Google Colab + Drive 支援（`# 00`、`# 9d`）**：本機自動略過 `# 00`；Colab 會掛載 Drive `MyDrive/chess_numba_data/{hce,texel_runs}/` 並支援 `RESUME_FROM_DRIVE = True` 斷線續跑；[`sf19_annotate_colab_runner.ipynb`](sf19_annotate_colab_runner.ipynb) 可於雲端產生並同步 `.sf19.npz`。

---

## 3. 尺度校準（`K*`）、Gate 守衛與超參數實戰經驗

### (1) Per-dataset `K*` 分離與徹底移除 `K=1.000`
* 本引擎將兵錨定在 ~100 cp，各資料集最佳斜率 $K^*$ 明顯大於 1（例：`USE_SF_LABELS=True` 時 `dev_K* ≈ 1.135`，純 WDL `holdout_K* ≈ 1.264`、`val_pgn_K* ≈ 1.168`）。
* 在未校準的 `K=1.000`（陡坡位置）檢驗會產生假性高 Z 值退步，且把混合標籤 `dev_K*` 拿去評純 WDL `holdout` 會錯殺 `Block 5a/5b/5c` 這類振幅收斂修正。
* **現行機制**：訓練期 SPSA / Grid 與 early-stop 使用 `_step_dev_k_star`（零 holdout 洩漏）；驗證與 `_gate_summary` 則對 `holdout`、`val_pgn`、`val_zurichess` **各自以該資料集純 WDL baseline 獨立擬合 $K^*$**（`fit_k=False`，CLI `validate_tuned.py` 預設 `--k-factor auto`），已全面移除 `K=1.000` 報表與閘門。所有最終驗收閘門皆**嚴格維持純 WDL 目標**，不混入 SF19 標籤，以避免教師偏好自我強化並保持跨實驗基準一致。

### (2) `_gate_summary` 子桶防崩壞守衛與雙外部仲裁機制
* 驗收除整體 `@ K*` 與 `broad-EG (phase<=64)` 外，對局部極端退步設有雜訊下限防護：
  * Holdout：`pure king+pawns`（`z > 3.5` & `delta > 2e-5` 判 `REJECT`）、`phase=0`（`z > 3.5` & `delta > 1e-5` 判 `REJECT`）。
  * 獨立驗證集（`val_pgn` / `val_zurichess`）：`broad-EG (phase<=64)`、`pure king+pawns` 與極端 `castling!=0` 退步，保留較輕微 `CAUTION` 提示。
* **`val_pgn castling!=0` 相關性校正與 `val_zurichess` 仲裁**：
  * `val_pgn` 同局開局有易位權局面約連續 ~10 plies 共用結果，會膨脹 i.i.d. Z 值。當 `holdout` 與 `val_pgn` 全體及 `broad-EG` 皆顯著改善（`z < -3.0`），且 **僅** `val_pgn castling!=0` 出現中度退步（不同時超過極端門檻 `z > 8.0` 且 `delta > 3e-5`）時：若有 `val_zurichess` 交叉檢查且未同步觸發 severe，或未啟用 `val_zurichess` 時，將降級為 `CAUTION`（保留警告但不一刀 `REJECT`）。
  * 啟用 `USE_PGN_VAL=True` 與 `USE_ZURICHESS_VAL=True` 時，`val_zurichess` 扮演**診斷／仲裁角色**（不單方面對雙重 `PASS` 區塊投否決票，僅於 `castling!=0` 與 `val_pgn` 共同觸發 severe 時強化為 `REJECT`，或觸發單邊警告為 `CAUTION`；輕微正 Z 僅印出 `Info:`）。只有停用 `USE_PGN_VAL` 時，`val_zurichess` 才作為唯一外部閘門。

### (3) `SPSAOptimizer` 的 `scales^2` 座標步幅效應
* `SPSAOptimizer` 透過 `u = theta / scales` 進行 `auto_scale=True` 縮放：`pert = ck * scales * delta` 且更新步幅再乘 `scales`，使等效座標學習率正比於 **`scales^2`**；同時啟用整數擾動保底 `c_floor = 1.0~1.25` 與 EMA 平滑候選追蹤。
* 搭配 `USE_SF_LABELS=True`（`dev` MSE 較純 WDL 低 ~35.7%）：
  * **有較大 `scales` 的非 PST 區塊**（`1a`, `1d`, `2a`, `3`, `4a`, `5a`, `5c`, `7a`, `7b`）：`alpha` 過大易在前 200 步過衝或停滯；實戰宜調低 `alpha`（`15000~45000`）並設衰退視野 `A=150`。
  * **單棋種 PST (`6a–6d`, `P=128`, `scales=1.0`) 與 `Block 8` (`P=520`)**：格點出現率低且受固定列平均約束，需 `batch_size=131072`、`A=250`、`iters=4500~5000` 與較高 `alpha`（`6a–6d`: `135000~155000`；`Block 8`: `65000`）。
  * **Grid 掃描與跨平台注意事項**：`scan_grid` 預設以 `SPSAOptimizer` 約束裁切範圍（如 `CONNECTED_SUPPORT_WEIGHT*` 與 `KING_FLANK_*` 限制於 `θ₀ ± 1`）；後段執行與分析格需自動切換至專案根目錄（勿寫死 Colab 專用 `%cd /content/chess_numba`）；執行長跑時勿從硬碟覆寫開啟中的 `.ipynb`。

---

## 4. 參數不變量、約束與常規指令

### 參數約束與耦合
* **永遠凍結**（113 組共 1,276 維之外）：`MG Pawn = 100`、`MG/EG King = 0`、共線分母（`KING_DANGER_OUT_*_DEN`、`KING_FLANK_ATTACK_DEN`、`KING_SHELTER_FEEDBACK_DEN`）、結構常數（`LAZY_EVAL_THRESHOLD`、`PHASE_*`、`SCALE_FACTOR_*`、`CANDIDATE_PASSER_DIVISOR=2`、`UNSTOPPABLE_PAWN_BONUS=800`）。
* **結構守恆**：`MIDGAME_LIMIT` / `ENDGAME_LIMIT`（維持 `3915/15258` 比例）由候選 MG `N/B/R/Q` 動態推導；`PST_MG/EG` 保持各棋種列平均值（兵僅計 rank 2–7）；`Mobility` 0-bin 固定於 `θ₀`；`validation bucket` 成員依初始 `θ₀` 固定。
* **Soft band 與離散項**：`EG Pawn` ±10%（`--pst-pieces P` 且含 `EG_MATERIAL_VALUES` 時解鎖）、`KING_PAWN_PROXIMITY_FACTOR∈[2,8]`、`BISHOP_PAWNS_CENTER_BLOCKED_FACTOR∈[0,3]`、`KING_PROX_FRIENDLY_MULT∈{1,2}`、`KING_DANGER_OUT_*_NUM` / `EG_SAFETY_SCALE` ±35%、`KING_FLANK_*` / `KING_SHELTER_FEEDBACK_NUM` 與 `CONNECTED_SUPPORT_WEIGHT*` 限制於 `θ₀ ± 1`（細節見 [`tuner.py`](tuner.py) `_setup_constraints`）。

### 常用 CLI 指令（專案根目錄執行）

```bash
# 0) 檢查資料 / 產生 SF19 側車標籤（可 --limit N 先測試；Colab 版 sf19_annotate_colab_runner.ipynb 支援 "val_pgn" / "all_val" / "both" / "zurichess" / "holdout_20m"）/ 重建 val_pgn
python -m tuner.run status
python -m tuner.data_pipeline.annotate_sf_labels --nodes 4000 --skip-holdout
python -m tuner.data_pipeline.build_pgn_validation --min-median-nodes 150000 --workers 10

# 1) 改過 classical eval/constants 後：同步快照並通過 parity 測試（失敗勿調參）
python -m tuner.sync_constants_snapshot
python -m tuner.codegen.generate_tunable_eval
python -m unittest tests.test_tunable_eval_parity tests.test_texel_parameterization -v

# 2) 獨立驗證 / 暫時覆寫驗證 / 離散窮舉（各資料集自動以 baseline K* 校準；若同目錄存在 *.sf19.npz 會自動附帶純 WDL sf19-quiet / sf19-tactical 診斷桶）
python -m tuner.validate_tuned --dataset tuner/data/hce/dataset_epd_wdl_20m.npz \
  --pgn-val tuner/data/hce/val_pgn.npz --zurichess-val tuner/data/hce/val_zurichess.npz \
  --tuned tuner/tuned_constants.py --split holdout
python -m tuner.validate_tuned --dataset tuner/data/hce/dataset_epd_wdl_20m.npz \
  --tuned tuner/constants_snapshot.py --override KING_DANGER_THRESHOLD=69 --split holdout
python -m tuner.scan_scalar_grid --dataset tuner/data/hce/dataset_epd_wdl_20m.npz \
  --grid KING_DANGER_OFFSET=0:4 --grid KING_DANGER_THRESHOLD=66:72 --split dev --eval-sample 1000000 --top 10

# 3) CLI SPSA 單組執行與合併（推薦優先使用 texel_tuner_runner.ipynb）
python -m tuner.tuner --list-params
python -m tuner.tuner --dataset tuner/data/hce/dataset_epd_wdl_20m.npz \
  --tune KNIGHT_MOBILITY_BONUS BISHOP_MOBILITY_BONUS ROOK_MOBILITY_BONUS QUEEN_MOBILITY_BONUS \
  --iter 4000 --batch-size 65536 --alpha 45000 --A 150 --c 2.5 --seed 0 --output tuner/tuned_constants.py
python -m tuner.apply_tuned --dry-run
python -m tuner.apply_tuned
```

### 對局驗收（需取得同意）
```bash
# 套用 candidate「之前」若有需要才同步 classical_old；套用後切勿再同步否則 New == Old
python tools/tournament.py -c 4 --games 500 --nodes 300000 --name1 New --name2 Old
python tournament_analysis/統計數據.py --target New
```

---

## 5. 目錄結構一覽

| 檔案 / 目錄 | 職責 |
| :--- | :--- |
| `texel_tuner_runner.ipynb`、`notebook_session.py` | 單 Kernel 分區調參工作台與雙重/三驗證閘門控制器 |
| `tuner.py`、`parameters.py`、`constraint_manager.py` | `SPSAOptimizer`、1,276 維註冊對映與棋理投影約束 |
| `validate_tuned.py`、`scan_scalar_grid.py` | Paired WDL / `K*` 驗證器與小範圍離散整數窮舉 |
| `constants_snapshot.py`、`sync_constants_snapshot.py` | `classical/constants.py` 側車快照（改常數後必重跑同步） |
| `tunable_eval.py`、`codegen/` | 可接受 `theta` 之 Numba 評估碼生成器 |
| `apply_tuned.py`、`run.py` | AST 行段安全合併回 `constants.py` 與 CLI 總入口 |
| `data_pipeline/` | EPD/PGN/Bullet 轉換、`val_pgn` 建構與 SF19 搜尋標註 |
| `nnue_pipeline/` | NNUE 資料建構（`fetch_official_data.py` 等），與 HCE SPSA 分離 |
| `archive/`、`runs/`、`diagnostics/` | 歷史 dump、各輪調參輸出備份與 NNUE/舊版診斷腳本 |
