# Classical HCE 調參手冊（`tuner/`）

本目錄負責 **Classical 手寫評估（HCE）參數的 SPSA 調優**，以及（獨立）**NNUE 訓練資料管線**。  
搜尋參數請用 [`tune_search/`](../tune_search/README.md)；NNUE 訓練邏輯在 [`chess_engine/nnue/ml_eval/`](../chess_engine/nnue/ml_eval/)。

> **重要：** 結構已對齊 SF11 後，調參目標是在本引擎尺度上找更好的權重，而不是再硬抄 SF 係數。  
> **MSE 下降 ≠ Elo 上升** — paired MSE 可作為靜態接受閘門，但正式宣稱棋力提升前仍需另行取得同意並用對局驗證（`tools/tournament.py`）。

---

## 你現在的資料來源是什麼？

| 用途 | 實際來源 | 本機路徑 | 現況 |
| :--- | :--- | :--- | :--- |
| **HCE SPSA（純 Texel）** | Bullet 對局勝負 | `tuner/data/hce/dataset.npz` | 已驗證：10M、3 種 WDL labels |
| **HCE SPSA（目前廣義殘局 run）** | 70% 勝負 + 30% SF sigmoid | `tuner/data/hce/dataset_mix03_20m_s2.npz` | 已驗證：20M、mix labels |
| **來源 A（推薦、你已有）** | NNUE 的 **Bullet `.bin`**（棋盤 + SF 分 + **對局勝負**） | `tuner/data/nnue/official_data_farseerT75.bin` | 用 `--from-bullet` 轉換 |
| 來源 B | 自備 PGN（TWIC）→ JSONL → NPZ | `tuner/data/pgn/` | `data --pgn ...` |
| HCE 標籤 | `game` = 標準 Texel WDL；`sf` = SF 分蒸餾；`mix` = `(1-λ)·game+λ·sf`（預設 λ=0.3） | 寫進 `results` | `game` 與 `mix` 目標不同，不能只以 label 是否連續判優劣 |
| **不能直接用** | HalfKA `.npz`（只有特徵索引，無完整棋盤） | `ultimate_halfka_*.npz` | 僅 NNUE |
| **NNUE 訓練** | 同上 halfka / bullet | `tuner/data/nnue/*` | 與 HCE 管線分開 |

一句話：**可以借 NNUE 的 Bullet bin（含勝負）來調 HCE**；**不要**直接拿 halfka npz。  
先看狀態：

```bash
python -m tuner.run status
```

---

## 濃縮指令（推薦日常用法）

```bash
# 0) 看資料是否齊
python -m tuner.run status

# 1) classical 改過評估/常數後：同步 tunable
python -m tuner.run sync

# 2a) 最快：從已有 NNUE Bullet 轉 HCE 資料（含勝負標籤）
python -m tuner.run data --from-bullet --label-mode game --limit 2000000

# 2a') 混合標籤：70% 勝負 + 30% SF sigmoid（另存，勿覆蓋 game 庫）
python -m tuner.run data --from-bullet --label-mode mix --sf-weight 0.3 \
  --limit 10000000 --output tuner/data/hce/dataset_mix03.npz

# 2b) 或從 PGN
python -m tuner.run data --pgn tuner/data/pgn/twic_merged.pgn --games 0

# 3) SPSA（整數 HCE 先用 c=3；不要用 c<1 作主調參）
python -m tuner.run tune --tune KNIGHT_MOBILITY_BONUS BISHOP_MOBILITY_BONUS --iter 5000 --c 3.0
# --patience 30（預設：連續 30 次報告無新 best 則停）；--patience 0 關閉
# --eval-sample 1000000（預設 hold-out 驗證集，不進訓練 batch）
# --metrics-sample 500000（固定 validation 子集，輸出 phase／廣義殘局 bucket MSE）
# --seed 0（SPSA batch/delta；validation split 固定，方便跨 seed 比較）

# 直接呼叫 tuner（完整 CLI）：
#   --pst-pieces NBRQ | P | K     只動 PST 指定子（可與其他 --tune 聯調）
#   --king-pst-only / --non-king-pst-only
#   --output path/to/dump.py
python -m tuner.tuner --dataset tuner/data/hce/dataset_mix03_20m_s2.npz \
  --tune PST_MG PST_EG --pst-pieces NBRQ --c 3.0 --iter 4000 \
  --batch-size 65536 --seed 0 \
  --output tuner/tuned_constants.py

# 4) 合入 classical（先 dry-run；可 --only 限制名稱）
python -m tuner.apply_tuned --dry-run
python -m tuner.apply_tuned --only ROOK_ON_OPEN_FILE_BONUS WEAK_QUEEN
# 若之後要對戰：在套用 candidate「之前」同步 classical_old；套用後勿再同步，否則 New=Old

# 或一條龍（sync + data + tune）
python -m tuner.run pipeline --pgn path/to.pgn --tune KNIGHT_MOBILITY_BONUS --iter 500
```

下載 TWIC（若還沒有 PGN）：

```bash
python -m tuner.data_pipeline.download_twic --start 1600 --end 1630 --output tuner/data/pgn/twic_merged.pgn
```

---

## 過時 / 用不到的檔案

| 路徑 | 狀態 | 建議 |
| :--- | :--- | :--- |
| `archive/` | 舊腳本副本 | **可忽略**；只考古 |
| `archive/tuned_dumps/` | 歷史 SPSA dump（`tuned_constants_*`） | 考古 / 回滾用；**勿當 θ₀** |
| `constants_to_be_tuned.py` | 相容 shim → `constants_snapshot` | 可留；勿手改 |
| `tuned_constants.py` | 最近一次 SPSA 預設輸出 | 合入前先 `--dry-run` |
| `diagnostics/*` | **NNUE** 診斷、離散圖、一次性 txt | HCE 日常**不用** |
| `nnue_pipeline/*` + `data/nnue/*` | **NNUE** 資料管線與大檔 | HCE **不用**；勿刪大檔 |
| `codegen/new_indices.txt` | 生成中間檔 | 自動重生 |
| `dispersion_plot.png`, `train_diagnosis.txt` | 歷史產物 | 可刪或移 archive |

**HCE 日常真正會動到的：**  
`run.py`、`tuner.py`、`parameters.py`、`constraint_manager.py`、`constants_snapshot.py`、`tunable_eval.py`、`data_pipeline/`、`codegen/`、`apply_tuned.py`、`sync_constants_snapshot.py`。

---

## 目錄結構

```
tuner/
├── README.md                 # 本文件
├── tuner.py                  # SPSA 主程式
├── parameters.py             # theta 向量 ↔ 常數名稱
├── constraint_manager.py     # 棋理約束
├── constants_snapshot.py     # classical/constants.py 快照（可重生）
├── sync_constants_snapshot.py
├── tunable_eval.py           # 接受 theta 的評估（codegen 產物）
├── tuned_constants.py        # SPSA 輸出（產物）
├── apply_tuned.py            # 把輸出 merge 回 classical/constants.py
├── validate_tuned.py         # 排除固定 validation 的 paired MSE / bucket z
├── scan_scalar_grid.py       # 少量整數 scalar 的固定樣本離散窮舉
├── data_pipeline/            # HCE 標籤資料
│   ├── download_twic.py
│   ├── generate_training_data.py
│   ├── preprocess_data.py
│   └── analyze_pgn.py
├── codegen/                  # 重生 tunable_eval
├── nnue_pipeline/            # NNUE 專用資料（與 HCE 調參無關）
├── diagnostics/              # NNUE / 一次性診斷
├── archive/                  # 舊腳本
└── data/                     # 大檔與中間產物（本機保留，通常 gitignore）
    ├── pgn/
    ├── jsonl/
    ├── hce/                  # dataset.npz
    └── nnue/                 # halfka npz、bullet bin 等（**既有多 GB 資料可用**）
```

**大檔說明：** 既有 `ultimate_halfka_farseerT75.npz`、`official_data_farseerT75.bin` 等只是搬到 `tuner/data/nnue/`，**不刪除、不重算**。管線預設路徑已更新。

---

## 調參前：與 classical 同步（必做）

評估或常數改過後：

```bash
python -m tuner.sync_constants_snapshot
python -m tuner.codegen.generate_tunable_eval
python -m unittest tests.test_tunable_eval_parity -v
```

Parity 失敗時**不要**跑 SPSA。

---

## HCE 調參流程

### 1. 準備 PGN

```bash
python -m tuner.data_pipeline.download_twic --start 1600 --end 1630 --output tuner/data/pgn/twic_merged.pgn
```

### 2. 產生標籤

| 模式 | 旗標 | 用途 |
| :--- | :--- | :--- |
| **game（預設，推薦）** | `--label-mode game` | Texel：對局結果 {0, 0.5, 1}，導向真實棋力 |
| **sf** | `--label-mode sf` | SF 深搜 → sigmoid，易擬合 SF，棋力未必升 |

```bash
# 推薦：game labels（不需 Stockfish）
python -m tuner.data_pipeline.generate_training_data --pgn tuner/data/pgn/twic_merged.pgn --label-mode game --min-elo 2400

# 可選：SF 蒸餾
python -m tuner.data_pipeline.generate_training_data --label-mode sf --stockfish-depth 10
```

### 3. 轉 NPZ

```bash
python -m tuner.data_pipeline.preprocess_data --output tuner/data/hce/dataset.npz
```

### 4. SPSA（先分組 warm start，最後聯合 polish）

目前要改善廣義殘局時，建議分組（小 `c`、每組對局閘門；**MSE ≠ Elo**）：

1. **Material + 同棋種 PST 聯調**：依序 `N / B / R / Q` 各跑一組；`--pst-pieces` 會同時只開對應的 MG/EG material entry。這是 block-coordinate warm start，不是最終解。
2. Mobility + Threats；mobility 的 0-bin 已固定，不再和 material 搶常數偏移。
3. Pawn structure / passer，再與 EG Pawn material + `--pst-pieces P` 做同一個 block-coordinate 接受週期（MG Pawn 維持 100cp anchor）。
4. King danger **或** shelter；王 PST 宜窄帶且優先跟 shelter，勿與 danger 大開聯調。若單一 `NO_QUEEN` 對 rook ending 與 minor-only ending 呈穩定反向梯度，可使用現有的 `KING_DANGER_NO_QUEEN`（攻方仍有車）與 `KING_DANGER_NO_QUEEN_ROOKLESS`（攻方也無車）兩值分流；勿再細拆所有材質組合。
5. Coordination（車線、弱后、壞象、`KING_PROTECTOR`…）。
6. Imbalance / Space / Initiative — imbalance 二次仍緊；initiative / passer dynamics / king-prox 已 soft-band（±1 或 ±10%），勿當唯一主力。
7. **聯合 polish**：主要 block 接受後，再聯調 `MG_MATERIAL_VALUES EG_MATERIAL_VALUES PST_MG PST_EG --pst-pieces NBRQ`。使用已接受結果作 warm start 與較小 `alpha`，用來消除 block 間交互作用；Pawn/King 因尺度 anchor 與合法格約束仍分開。

不要把 cp tables 與 `*_MULT`、`*_DIV`、support weights、θ₀±1 soft-band scalars 用同一組 SPSA 尺度混調。新 block 的 cp table 可從 `c=3, alpha=30000` 起步；已完成分組 warm start 的 joint polish 宜降到 `alpha=8000~15000`。小範圍倍率建議獨立使用 `c=1, alpha=1000~3000`，只有少數整數候選時優先用離散枚舉。單一 `c/alpha` 在異尺度參數上會造成 constraint clipping 與有偏梯度。


```bash
# 列出可調參數
python -m tuner.tuner --list-params

# 範例：只調 mobility
python -m tuner.tuner --dataset tuner/data/hce/dataset_mix03_20m_s2.npz \
  --tune KNIGHT_MOBILITY_BONUS BISHOP_MOBILITY_BONUS ROOK_MOBILITY_BONUS QUEEN_MOBILITY_BONUS \
  --iter 4000 --batch-size 65536 --alpha 30000 --c 3.0

# 範例：NBRQ PST only
python -m tuner.tuner --dataset tuner/data/hce/dataset_mix03_20m_s2.npz \
  --tune PST_MG PST_EG --pst-pieces NBRQ --c 3.0 --iter 4000 --batch-size 65536

# 廣義殘局第一輪：Knight material + Knight PST（P/B/R/Q 依序各跑一次）
python -m tuner.tuner --dataset tuner/data/hce/dataset_mix03_20m_s2.npz \
  --tune MG_MATERIAL_VALUES EG_MATERIAL_VALUES PST_MG PST_EG \
  --pst-pieces N --c 3.0 --iter 4000 --batch-size 65536 \
  --eval-sample 1000000 --metrics-sample 500000 --report-every 50 --patience 30 --seed 0

# 範例：coordination
python -m tuner.tuner --dataset tuner/data/hce/dataset_mix03_20m_s2.npz \
  --tune ROOK_ON_SEMI_OPEN_FILE_BONUS ROOK_ON_OPEN_FILE_BONUS ROOK_ON_QUEEN_FILE \
         WEAK_QUEEN LONG_DIAGONAL_BISHOP BISHOP_PAWNS_PENALTY KING_PROTECTOR \
  --c 3.0 --iter 4000 --batch-size 65536
```

**永遠凍結 / 硬 pin：** MG Pawn 材質 100、MG/EG King 材質 0、unused table 洞、部分公式骨架（`UNSTOPPABLE_PAWN_BONUS` 等）。

**Soft band（可小動）：** EG Pawn 材質 θ₀±10%（≈108–132，選 `--pst-pieces P` 且包含 `EG_MATERIAL_VALUES` 時會開啟）；`PASSED_DYNAMICS_*` / 多數 `KING_PROX_*` 為 θ₀±1；`KING_PROX_FRIENDLY_MULT∈{1,2}`；prox clamp 與 `INITIATIVE_*` ±10%；imbalance 二次非零 ±10% 且至少可動一個整數（0 釘死）。見 `tuner.py` `_setup_constraints`。

`TRAPPED_ROOK` / `FLANK_ATTACKS` / `THREAT_RESTRICTED_PIECE` 為 soft band（非死鎖）。

### Material / phase / PST 的尺度不變量

- `MIDGAME_LIMIT` 由候選 MG `N/B/R/Q` 的完整開局非兵材質即時計算；`ENDGAME_LIMIT` 保持 SF11 的 `3915/15258` 比例。正式引擎、runtime match weights、Texel 的 `theta+ / theta-` 都走同一公式，因此 start position 永遠是 phase 128。
- `PST_MG/PST_EG` 每個棋種列保留 θ₀ 平均值（兵只計 rank 2–7），只調格子間形狀；現有表不會被重新置中。
- 每張 mobility table 的 0-mobility MG/EG bin 固定在 θ₀，material 負責棋種基準值。
- tuner 候選與輸出檔都使用 `np.rint/np.round` 的同一整數量化；因此主調參 `c` 應至少為 1，預設／建議起點為 3。
- validation bucket 以初始 θ₀ 固定成員，候選 material 改變時不重分桶；這樣各次報告可直接比較。
- 跨輪比較必須固定 dataset、`--eval-sample` 與 validation split。輸出 `tuned_*.py` 不會自動套回 production；接受後須先套用、執行 `python -m tuner.run sync` 清除 stale Numba cache，再啟動下一輪，否則下一輪不會承接前一輪 best。
- 20M dataset 的目前預設為 batch 65,536、validation 1,000,000（5%）、bucket metrics 500,000、report every 50、patience 30。`validate_tuned.py` 預設排除這固定 1M，從 SPSA train pool 另抽 10M 做 secondary paired check；它能檢查 robustness，但因仍來自同一 dataset，不能冒充真正獨立資料源。

接受一組參數時，除了總 `val MSE`，至少同時查看 `phase=0`、`phase=1..31`、`phase=32..63` 與 `broad-EG phase<=64`。若總 MSE 下降但 broad-EG 持續變差，該組不應進入對局閘門。

### 5. Secondary paired validation 與離散 scalar

```bash
# 候選檔相對目前 constants snapshot 的 10M paired MSE、SE、z；預設排除固定 1M validation
python -m tuner.validate_tuned \
  --dataset tuner/data/hce/dataset_mix03_20m_s2.npz \
  --tuned tuner/tuned_constants.py --sample 10000000 \
  --exclude-fixed-val 1000000 --chunk-size 1000000

# 可暫時覆寫 scalar 或陣列單格，不修改檔案
python -m tuner.validate_tuned --dataset tuner/data/hce/dataset_mix03_20m_s2.npz \
  --tuned tuner/constants_snapshot.py --override KING_DANGER_THRESHOLD=69 \
  --override "EG_MATERIAL_VALUES[0]=119" --sample 2000000

# 少數整數候選優先窮舉；範圍可用 LO:HI 或 LO:HI:STEP，--top 只列最佳 N 行
python -m tuner.scan_scalar_grid \
  --dataset tuner/data/hce/dataset_mix03_20m_s2.npz \
  --grid KING_DANGER_OFFSET=0:2 --grid KING_DANGER_THRESHOLD=68:74 \
  --eval-sample 1000000 --top 10
```

scalar grid 的 bucket 以 baseline θ₀ 固定，負 delta 較好。先用固定 1M 探索，再用 `--exclude-fixed-val 1000000 --eval-sample 10000000` 或 `validate_tuned.py` 複驗，避免只接受固定 validation 的局部最佳。

### 6. 合入引擎

```bash
python -m tuner.apply_tuned --dry-run   # 先看會改哪些
python -m tuner.apply_tuned
python -m unittest tests.test_texel_parameterization tests.test_endgame_conformance tests.test_tunable_eval_parity -v
```

### 7. 對局驗證（需明確同意再跑長測）

```bash
# 固定 nodes，New vs Old（classical_old 必須在套用 candidate 之前 --sync）
python tools/tournament.py -c 4 --games 500 --nodes 300000 --name1 New --name2 Old
# 局級統計
python tournament_analysis/統計數據.py --target New
```

MSE 變好但 Elo 變差 → 回滾該組參數（可用 `archive/tuned_dumps/` 或 `git checkout`）。

---

## 與其他管線的關係

| 目標 | 工具 | 訊號 |
| :--- | :--- | :--- |
| HCE 靜態參數 | 本目錄 SPSA | MSE(sigmoid(eval), label) |
| 搜尋參數 | `tune_search/` | 對局勝率 |
| NNUE | `nnue/ml_eval` + `tuner/nnue_pipeline` | 訓練 loss |

**不要**同時 SPSA eval 與 search。順序：先穩定 eval，再調 search。

### NNUE 資料（參考）

```bash
python tuner/nnue_pipeline/fetch_official_data.py
# 輸出預設：tuner/data/nnue/ultimate_halfka_farseerT75.npz
```

細節見 `nnue_pipeline/` 與 [ml_eval 文件](../chess_engine/nnue/ml_eval/ml_eval_documentation_and_architecture.md)。

---

## 常見問題

1. **tunable 與 classical 分數不一致**  
   重跑 `sync_constants_snapshot` + `generate_tunable_eval` + parity 測試。

2. **參數表過時**  
   只應註冊 `evaluation` / `pawns` / `material` 實際使用的常數；見 `parameters.py`。

3. **舊 `tuned_constants.py` / `archive/tuned_dumps/*` 材質漂掉**  
   歷史 dump 只作對照；θ₀ 以 `sync_constants_snapshot` 後的 snapshot 為準重跑。

4. **資料在哪**  
   大檔：`tuner/data/nnue/`、`tuner/data/hce/`。程式預設路徑已指向此處。

5. **王 PST 與 danger / shelter**  
   王 PST 宜與 **shelter** 聯調（窄帶）；**danger 建議單獨**。三者一次全開易搶訊號（e1/g1 易被抬高）。
