# 對局式 HCE 評估權重 SPSA (`tune_eval_match/`)

本目錄負責 **Classical HCE 非靜態／搜尋耦合評估權重的單進程共享 JIT 對局式 SPSA**。
靜態 Texel 調參難以避免 qsearch 視界偏差的項目（威脅 `THREAT_*`、王安全 `KING_DANGER_*`/`SAFE_CHECK_*`/`KING_ATTACK_WEIGHTS`、空間 `SPACE_*`、不平衡 `IMBALANCE_*`、通路兵動態項等）統一透過 `SearchContext.eval_weights` 進行實戰對局調優。

---

## 1. 為什麼使用單進程共享 JIT？

* Numba `@njit` 會凍結模組層級的全域常數，若每回合重開 UCI 子進程將重新編譯或產生跨進程負擔。
* `SearchContext.eval_weights`（長度 `128` 的 `np.int32[::1]` StructRef 欄位）在 `nopython` 下為**即時可變陣列**；整個 SPSA 僅編譯一次機器碼，透過 `ThreadPoolExecutor` 平行進行多對換先對局（Numba 釋放 GIL），回合間僅更新 `eval_weights` 並重置 TT/歷史表。

---

## 2. 可調參數區塊 (`--block`) 與 `EVAL_PARAM_INDEX`（0..98）

所有參數預設值一律從現行 `chess_engine/classical/constants.py` **即時讀取**（不再寫死過時快照），並支援 `--params P1,P2,...` 自訂任意子集。

| `--block` | 維度 | 內容說明 |
| :--- | ---: | :--- |
| `threats_mg` *(預設)* | 17 | 全中局威脅權重（`THREAT_SAFE_PAWN_MG`、`BY_MINOR_*_MG`、`BY_ROOK_*_MG`、`BY_KING_MG`、`HANGING_MG`、`PAWN_PUSH_MG`、`*_ON_QUEEN_MG`、`RESTRICTED_PIECE_MG`） |
| `threats_eg` | 17 | 全殘局威脅權重 |
| `threats` | 34 | 完整威脅表（`threats_mg` + `threats_eg`） |
| `king_weights` | 11 | 高槓桿王安全權重與輸出尺度（`KING_ATTACK_WEIGHT_{N,B,R,Q}`、`NO_QUEEN`、`NO_QUEEN_ROOKLESS`、`OUT_MG/EG_NUM`、`THRESHOLD`、`EG_SAFETY_SCALE`、`FLANK_ATTACK_NUM`） |
| `g4` | 14 | `TEMPO_BONUS`、`SAFE_CHECK_*`、`KING_DANGER_*`（含 `NO_QUEEN_ROOKLESS`） |
| `a` | 14 | 王翼攻防、盾回授、`PAWNLESS_FLANK`、`FLANK_ATTACKS`、`KING_DANGER_OUT_*`、`SHELTER_BASE_*`、`BLOCKED_STORM*` |
| `space` | 3 | `SPACE_THRESHOLD`、`SPACE_BONUS_DIVISOR`、`SPACE_SCALE_DIVISOR` |
| `passed` | 10 | 通路兵路徑安全、動態乘數/偏置、王距與邊線加成 |
| `passed_rank` | 8 | 通路兵第 3–6 線 MG/EG 獎勵 |
| `material` | 10 | `MAT_{N,B,R,Q}_{MG,EG}` + `IMBALANCE_SCALE_{MG,EG}` |
| `imbalance` | 3 | `IMBALANCE_DIVISOR`、`IMBALANCE_SCALE_MG`、`IMBALANCE_SCALE_EG` |

執行 `python tune_eval_match/spsa_match.py --list-blocks` 可列出全部區塊與現行即時預設值。

---

## 3. 預設三階段協同協議（L1 → L2 → 400k 最終確認）

| 階段 | 預設設定 | 關鍵設計 |
| :--- | :--- | :--- |
| **L1 擾動搜尋** | `500` games @ `80,000` nodes (`-c 8`, `--adjudicate`) | 固定獨立 `opening_seed` 洗牌 `data/openings.epd`（34,700 局），每輪依序推進 `opening_offset` 不重複開局 |
| **L2 階段驗收** | 每 `10` iter，對上 `best` 至多 `500` games @ `200,000` nodes | 使用獨立 **25% 保留開局集 (`l2_holdout`)** 與窗口後半平均候選 (`--l2-candidate tail`)；支援分批輸出 Elo±95% CI、`--l2-early-trash`（150 局 `<0.45` 早退）、可選 `--l2-sprt`，預設門檻 `score >= 0.510` |
| **最終確認** | `400,000` nodes 雙線 SPRT | 使用 [`tools/run_eval_param_match.py`](../tools/run_eval_param_match.py)（或將候選寫入 `constants.py` 後跑 [`tools/tournament.py`](../tools/tournament.py)）對戰基準 `classical_old`，檢驗 `SPRT[0, 10]` 與 `SPRT[-3, 3]` |

---

## 4. 常用指令

```bash
# 0. 查看區塊與即時預設值（無需等待 JIT）
python tune_eval_match/spsa_match.py --list-blocks

# 1. 啟動 L1 + L2 Match-SPSA（範例：threats_mg）
python -u tune_eval_match/spsa_match.py --block threats_mg \
  --nodes 80000 --games 500 --l2-every 10 --l2-nodes 200000 --l2-games 500 \
  --out-dir tournament_analysis/spsa_eval_threats_mg

# 中斷接續
python -u tune_eval_match/spsa_match.py --block threats_mg \
  --resume tournament_analysis/spsa_eval_threats_mg/checkpoint.json \
  --out-dir tournament_analysis/spsa_eval_threats_mg

# 2. 最終 400k nodes 獨立 SPRT 確認（對戰現行或 classical_old 基準）
python -u tools/run_eval_param_match.py \
  --params-json tournament_analysis/spsa_eval_threats_mg/best_params.json \
  --baseline-old --nodes 400000 --games 2000 --sprt 0,10 --batch-games 50 \
  --out tournament_analysis/verify_threats_mg_400k.json
```

---

## 5. 輸出產物與回寫規範

`--out-dir` 產出：
* `spsa_log.jsonl`：每輪 L1（含 `drift_z`、`penta_plus`）與每次 L2 驗收紀錄。
* `checkpoint.json` / `latest_params.json`：含浮點 `theta`、開局偏移量與隨機數狀態以便無縫 `--resume`。
* `best_params.json` / `best_constants_diff.txt`：L2 驗收通過之整數配置與與現行常數差異摘要。

400k SPRT 通過後方可將變動寫入 `chess_engine/classical/constants.py`，依序執行 `python -m tuner.sync_constants_snapshot`、`python -m tuner.codegen.generate_tunable_eval`，並在明確確認後執行 `python tools/sync_classical_old.py --sync`。
