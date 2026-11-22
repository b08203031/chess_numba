# Classical 引擎除錯、優化與棋力實驗紀錄 (Optimization Experiment Log)

> **用途**：承接 [`CLASSICAL_ENGINE_AUDIT_20261122.md`](CLASSICAL_ENGINE_AUDIT_20261122.md) 之審計發現，完整記錄後續每一波程式碼修復、零變動純 NPS 優化、測試與對戰驗證（SPRT / Depth-10 Benchmark）結果。  
> **起始基準**：Commit `d015345`（Phase 14 Match-SPSA `material` + Lazy SMP E148；`classical/` 與 `classical_old/` 一致）  
> **建立日期**：2026-11-22

---

## 總覽進度表

| 波次 / 編號 | 日期 | 主題與改動摘要 | 節點／評分恆等 | 測試與實測結論 | 狀態 |
| :--- | :---: | :--- | :---: | :--- | :---: |
| **Audit-0** | 2026-11-22 | 全模組純只讀深度審計（31 BUGs / 11 純加速 / 16+3 棋力提案） | — | 產出 `CLASSICAL_ENGINE_AUDIT_20261122.md` | ✅ 完成 |
| **Wave 2A** | 2026-11-22 | 零樹變動純 NPS 優化與死碼清理（搜尋／啟發式／TT／Movegen／SEE／HCE） | **0 差異**（20/20 FENs `1,056,156` nodes 100% 一致） | 96 項測試全過；20 盤面 Depth 10 × 5 次加權 NPS `485,290 → 602,738`（**+24.20%**），算術平均 NPS **+30.35%**，中位數均值 **+28.67%** | ✅ 完成 |
| **Wave 2B** | 2026-11-22 | 補齊剩餘 Wave 2 架構優化（64B AoS `pawn_table`、`SearchContext` 80→65 欄、TT 16B 對齊、`.nbi` 多重特化收斂、`quiets` 展開） | **0 差異**（20/20 FENs `1,056,156` nodes 100% 一致） | 96 項測試熱跑 `21.39s` 全過；`.nbi` 過載大幅收斂（`shelter` 24→2、`attacked` 13→1、`pinned` 4→1、`cont_hist` 5→1）；中位數均值 NPS `458,684 → 601,128`（**+31.05%**），算術平均 NPS **+30.89%**，加權 NPS **+23.10%** | ✅ 完成待對戰／後續波次 |

---

## Wave 2：零搜尋樹／零評估變動純 NPS 優化與死碼清理（2026-11-22）

### 1. 實施範圍與設計原則
* **嚴格不變量**：所有改動必須維持 **100% 搜尋樹與評估分數恆等**（固定深度下 `nodes_searched`、`quiescence_nodes`、`best_eval`、`best_move` 均 `0` 差異），因此**不改動** `SearchContext` StructRef 的 `pawn_table_*` 欄位介面（留待測試契約同步時再處理 AoS 打包），亦**暫不引入**會改變葉節點計數的 `depth <= 0` 直接分派 QS。
* **納入改動明細**：
  1. `search.py`：
     * 延遲計算敵方 `pinned_*`、`check_sq_*` 與 `blockers_for_their_king_s`（僅於離開 `STAGE_TT_MOVE` / 進入 Step 14 / 吃子 LMR 時才準備），跳過大量 Move-1 TT 截斷與 Multi-Cut 的魔術位元板查表。
     * 將 `is_giving_check_after_move` 從 `make_move` 後無條件計算移至 LMR 的 `if lmr > 0:` 分支內延遲計算。
     * Step 14d Quiet SEE 複用 Step 14c 已計算的 `fp_lmr_d`。
     * `pv_table` 子列複製改為僅在 `if is_pv:` 執行。
     * 新增 `@njit` 整數老化函式 `_age_history_tables_jit(search_context)`，取代 `iterative_deepening_search` 中 6 條會配置 `float64` 暫存陣列的 `np.divide(..., 2)`。
     * 移除已確認不可達分支（`L1553-1558`、`L2841-2846`）與無用區域變數。
  2. `search_heuristics.py`：
     * `score_quiets` 將 5 組 Continuation History 堆疊查詢（`ply-1, 2, 3, 4, 6`）外提至著法迴圈之外。
     * `score_captures_lazy` 與 `score_captures_with_tt_lazy` 先檢查 `(opp_occ & BB_SQUARES[to_sq]) != 0` 再呼叫 `find_piece_type_on_square_side`。
  3. `transposition_table.py`：
     * `store_tt` `else:` 分支追蹤是否有實際變更，無變更時不再重算並回寫 `padding` checksum；移除死函式 `estimate_tt_occupancy_percentage`。
  4. `move_generator.py` / `see.py` / `bitboard_utils.py` / `zobrist.py` / `core.py`：
     * `generate_pseudo_legal_captures_buffer` 的滑動子預先用 `BISHOP_RAYS` / `ROOK_RAYS & opponent_pieces_bb` 過濾再查 Magic 表；移除經全倉 grep 驗證為 0 呼叫者的死函式與死表（約 450+ 行）。
     * `_see_ge_jit` 以 `has_promo_pawn` 取代單純 `is_promo_rank`，恢復約 99% 底線吃子的 O(1) Pre-Magic 早退；移除 `get_attackers_for_see` 等死碼。
     * `bitboard_utils.py`、`zobrist.py`、`core.py` 移除 0 呼叫者死函式與重複定義。
  5. `evaluation.py` / `pawns.py` / `endgame.py` / `constants.py`：
     * `SliderOnQueen` 與 `_evaluate_king_attackers` 加入 `ROOK_RAYS` / `BISHOP_RAYS` 前置過濾。
     * 門控 `THREAT_RESTRICTED_PIECE`（僅非零權重時計算）、複用兵攻擊左/右位移、單次載入 `mat_1..4`、別名 `BLACK_KING_ZONES = WHITE_KING_ZONES`。
     * `evaluate_piece_coordination` 改為僅遍歷有車的直線，避免空轉 16 次 `count_bits`；`endgame.py` 複用預計算遮罩與模組級 `_KRPPKRP_SCALE`；移除 `constants.py` 中 `SCALE_FACTOR_KRPKR_FORTRESS` 與 `CAPTURE_HISTORY_VICTIM_TYPES`，並同步 `tuner/constants_snapshot.py` 與 `tuner/tunable_eval.py`。

### 2. 實驗設計（Depth-10 × 5 Runs Baseline vs Wave 2）
* **盤面組（共 20 個多樣化盤面）**：16 個等距抽樣自 `data/openings.epd`（行號 `1, 2001, ..., 30001`）＋ 4 個經典標準盤面（`Startpos`、`Kiwipete`、複雜戰術中局、車兵殘局）。
* **量測協定**：完成 JIT 預熱後，每個盤面於確定性重置狀態（清除 TT、`reset_search_context`、重置 Pawn Table）下執行 **固定深度 `depth = 10` × 5 次獨立重複**，驗證 `nodes_searched`、`quiescence_nodes`、`best_eval`、`best_move` 100% 一致並統計平均與中位數 NPS。
* **旁路工具同步**：同步執行 `python -m tuner.sync_constants_snapshot` 與 `python -m tuner.codegen.generate_tunable_eval`。

### 3. 驗證結果
1. **單元與契約測試**（`python -m unittest tests.test_tunable_eval_parity tests.test_endgame_conformance tests.test_pinned_eval_geometry tests.test_see tests.test_search_core_contracts tests.test_lazy_smp -v`）：**96 / 96 項全數通過 (`OK`)**。
2. **跨進程熱快取重載**：`warm_import = 7.43s`，`warmup_search_jit() = 0.35s`，`0` `.numba.unresolved$` 符號錯誤。
3. **行尾衛生檢查**：所有修改檔 `0` 筆 `\r\r\n`（CRCRLF）。

### 4. Depth-10 × 5 Runs 修改前後基準對照表（20 盤面）

* **總節點一致性**：修改前總計 **`1,056,156`** nodes vs 修改後總計 **`1,056,156`** nodes（**`0 / 20`** 差異，包含 `nodes_searched`、`quiescence_nodes`、`best_eval`、`best_move` 全部 100% 吻合）。
* **總平均耗時**：`2.176s → 1.752s`（預熱 `1.97s → 0.40s`）。
* **加權總 NPS**（`Σnodes / Σmean_dt`）：`485,290 → 602,738`（**`+24.20%`**）。
* **20 盤面算術平均 NPS**：`452,295 → 589,545`（**`+30.35%`**）；**各盤面中位數均值**：`458,684 → 590,195`（**`+28.67%`**）。

| 盤面 | 著法 (`eval`) | 總節點 (`QS` 節點) | 修改前均值耗時 | 修改後均值耗時 | 修改前 Mean NPS | 修改後 Mean NPS | NPS 增幅 |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `EPD_1` | `b1d2` (`105`) | 33,355 (16,315) | 82.5 ms | 51.8 ms | 407,339 | 646,277 | **+58.66%** |
| `EPD_2001` | `c1d2` (`18`) | 44,564 (19,427) | 87.8 ms | 81.7 ms | 511,995 | 550,105 | **+7.44%** |
| `EPD_4001` | `f3e5` (`-15`) | 58,769 (25,336) | 118.8 ms | 96.9 ms | 498,874 | 611,148 | **+22.51%** |
| `EPD_6001` | `f1c4` (`-7`) | 85,703 (43,571) | 173.9 ms | 162.8 ms | 496,269 | 530,169 | **+6.83%** |
| `EPD_8001` | `d1d4` (`42`) | 60,435 (24,494) | 127.3 ms | 110.2 ms | 476,166 | 560,270 | **+17.66%** |
| `EPD_10001` | `d2c4` (`15`) | 62,313 (31,621) | 116.6 ms | 108.9 ms | 545,548 | 586,100 | **+7.43%** |
| `EPD_12001` | `a2a3` (`-6`) | 81,219 (40,220) | 153.9 ms | 128.6 ms | 532,129 | 633,871 | **+19.12%** |
| `EPD_14001` | `e2e4` (`60`) | 106,482 (59,896) | 190.8 ms | 164.8 ms | 563,161 | 648,201 | **+15.10%** |
| `EPD_16001` | `e2d4` (`13`) | 77,928 (40,325) | 153.9 ms | 136.7 ms | 508,985 | 574,951 | **+12.96%** |
| `EPD_18001` | `e1g1` (`44`) | 85,657 (59,054) | 154.7 ms | 114.1 ms | 557,603 | 752,373 | **+34.93%** |
| `EPD_20001` | `a4b3` (`32`) | 44,746 (22,658) | 95.8 ms | 74.9 ms | 478,752 | 605,422 | **+26.46%** |
| `EPD_22001` | `c1e3` (`32`) | 63,387 (30,321) | 119.0 ms | 90.7 ms | 535,425 | 700,835 | **+30.89%** |
| `EPD_24001` | `e2d2` (`55`) | 18,750 (6,578) | 57.4 ms | 40.9 ms | 333,096 | 506,231 | **+51.98%** |
| `EPD_26001` | `f1d3` (`-16`) | 37,116 (20,678) | 75.6 ms | 66.4 ms | 491,945 | 569,352 | **+15.73%** |
| `EPD_28001` | `f1d3` (`65`) | 12,276 (4,145) | 38.3 ms | 25.2 ms | 321,159 | 496,824 | **+54.70%** |
| `EPD_30001` | `f3d2` (`-13`) | 57,723 (29,652) | 127.6 ms | 101.9 ms | 458,566 | 577,586 | **+25.95%** |
| `Startpos` | `d2d4` (`30`) | 41,360 (16,186) | 97.0 ms | 66.8 ms | 437,527 | 627,553 | **+43.43%** |
| `Kiwipete` | `e2a6` (`-117`) | 66,878 (41,887) | 140.2 ms | 103.9 ms | 482,075 | 648,384 | **+34.50%** |
| `Tactical_MG` | `f4f8` (`29999`) | 587 (65) | 20.6 ms | 4.5 ms | 28,932 | 132,171 | **+356.83%** |
| `Rook_EG` | `b4f4` (`21`) | 16,908 (6,194) | 44.8 ms | 20.5 ms | 380,359 | 833,077 | **+119.02%** |

> **註**：`Tactical_MG` 僅 587 nodes（根節點將死），先前因 `np.divide` 老化每次呼叫額外耗費 ~17 ms 導致實測僅 ~29k NPS；改用 `_age_history_tables_jit` 後短步耗時從 20.6 ms 降至 4.5 ms。其餘標準中長盤與殘局盤面（如 `Rook_EG` +119%、`Startpos` +43%、`Kiwipete` +34.5%）則體現延遲計算敵方幾何、`is_giving_check_after_move`、外提 `score_quiets` 查詢與射線前置過濾的綜合加速效果。

---

## Wave 2B：補齊剩餘 Wave 2 架構優化（64B AoS `pawn_table`、TT 對齊、Numba `.nbi` 特化收斂、`quiets` 展開）（2026-11-22）

### 1. 實施範圍與架構變更明細
1. **`engine_types.py` & `pawns.py` — 64B 單 Cache-Line AoS `pawn_table` 打包與緩衝區縮容**：
   * 定義 64-byte 結構化型別 `pawn_table_entry_dtype`（5×`uint64` 位元板 + 2×`int32` `mg`/`eg` + 4×`int16` 兵盾分數 + 2×`int8` 王格 + `uint8` 易位權 + 5B padding = **64 Bytes**，精確對齊單一 CPU Cache Line），將原本分散於 14 條獨立陣列（每次查詢跨 14 條 cache line、總計 17.56 MiB/thread）的 `pawn_table_*` 合併為單一欄位 `('pawn_table', numba_pawn_table_entry_type[::1])`（16.00 MiB/thread）。
   * 於 `SearchContextProxy` 自動綁定 14 個向後相容的 Python `@property` 欄切片視圖（`pawn_table_keys`、`pawn_table_w_king_sq` 等），使 `main.py`、`tools/match_core.py`、`tune_eval_match/inprocess_match.py` 與既有測試契約零摩擦相容。
   * 移除 0 呼叫者死欄位 `aggressor_cache` 與 `victim_cache`（省 64 KiB/thread），將 `SearchContext` StructRef 欄位數由 **80 欄縮減至 65 欄**；並將 `quiet_moves_tried` 與 `quiet_pieces_tried` 由 `(MAX_PLY, 256)` 縮減至 `(MAX_PLY, 64)`（對齊搜尋迴圈 64 步上限，再省 72 KiB/thread）。
2. **`transposition_table.py` — 16B TT 條目自然對齊修復**：
   * 對調 `tt_entry_dtype` 中 `best_move` (`uint16`) 與 `is_pv` (`bool_`) 的欄位順序，使 `is_pv` 位於 offset `11`、`best_move` 對齊至偶數 offset `12..13`，消除每次 TT 讀寫的跨界未對齊 16-bit 存取。
3. **Numba `.nbi` / `.nbc` 多重特化風暴收斂**：
   * `pawns.py`：新增 `inline="never"` 字面量消解輔助 `_p_i64` 與 `_p_bool` 包裝 `evaluate_shelter_aligned` 與 `evaluate_pawn_structure` 呼叫點，將 `evaluate_shelter_aligned` 由 **24 過載收斂至 2 過載**（含 SearchContext 與 standalone 兩路）。
   * `move_generator.py`：為 `is_square_attacked` 與 `get_pinned_pieces` 補上顯式 Numba 簽名並正規化呼叫端型別，使 `is_square_attacked` 由 **13 過載收斂至 1 過載**、`get_pinned_pieces` 由 **4 過載收斂至 1 過載**。
   * `search_heuristics.py`：為 `update_continuation_history` 補上顯式簽名，由 **5 過載收斂至 1 過載**。
   * `search.py`：正規化全部 5 處 `store_tt` 與 7 處 `_see_ge_jit` 熱路徑呼叫端型別，使 `store_tt` 由 **6 過載降至 3 過載**（熱路徑僅 1 過載）、`_see_ge_jit` 由 **12 過載降至 7 過載**（搜尋熱路徑僅 1 過載）。
   * `tuner/codegen/generate_tunable_eval.py`：修正 `_fix_numba_signatures` 僅對實際注入 `theta` 的函式附加 `numba.float64[:]`，並重新生成 `tuner/tunable_eval.py`。
4. **`move_generator.py` & `evaluation.py` 熱路徑微優化**：
   * `generate_pseudo_legal_quiets_buffer`：當單側同時具備短易位與長易位路徑淨空時共享 `e1`/`e8` 未受攻檢查（避免重複呼叫 `is_square_attacked`），並將 `for piece_type in range(5):` 展開為 `N → B → R → Q → K` 5 段直接位元迴圈（維持 100% 相同著法生成順序）。
   * `evaluation.py`：`KnightOnQueen` / `SliderOnQueen` 以 `bb != 0 and (bb & (bb - 1)) == 0` 取代 `count_bits(bb) == 1`。

### 2. 驗證結果
1. **Numba 過載與 `.numba.unresolved$` 檢查**：
   * `unresolved: []`（**`0`** 個 `.numba.unresolved$` 符號）。
   * 磁碟 `.nbi` 過載數：`_search = 1`、`quiescence_search = 1`、`_lazy_smp_helper_loop_jit = 1`、`is_square_attacked = 1`、`get_pinned_pieces = 1`、`update_continuation_history = 1`、`evaluate_shelter_aligned = 2`、`store_tt = 3`、`_see_ge_jit = 7`。
2. **單元與契約測試**（`tests.test_tunable_eval_parity`、`tests.test_endgame_conformance`、`tests.test_pinned_eval_geometry`、`tests.test_see`、`tests.test_search_core_contracts`、`tests.test_lazy_smp`）：
   * 冷編譯後熱快取重跑 **96 / 96 項全數通過（`Ran 96 tests in 21.389s, OK`）**（僅同步更新 `tests/test_search_core_contracts.py:333` 對 Step 14d `_see_ge_jit` 參數正規化字串之斷言）。

### 3. Wave 2B Depth-10 × 5 Runs 完整基準對照表（20 盤面：Baseline vs Wave 2A vs Wave 2B）

* **總節點與評分恆等**：**`mismatches = 0 / 20`**（Baseline / Wave 2A / Wave 2B 總節點數皆為 **`1,056,156`**，20 盤面 `nodes_searched`、`quiescence_nodes`、`best_eval`、`best_move` 100% 一致）。
* **各盤面中位數均值 NPS（Mean-of-Medians，抗背景抖動指標）**：`458,684`（Baseline）→ `590,195`（Wave 2A, `+28.67%`）→ **`601,128`（Wave 2B, `+31.05%`）**。
* **20 盤面算術平均 NPS**：`452,295`（Baseline）→ `589,545`（Wave 2A, `+30.35%`）→ **`592,026`（Wave 2B, `+30.89%`）**。
* **加權總 NPS**（`Σnodes / Σmean_dt`）：`485,290`（Baseline）→ **`597,371`（Wave 2B, `+23.10%`）**（總均值耗時 `2.176s → 1.768s`，預熱 `1.97s → 0.38s`）。

| 盤面 | 著法 (`eval`) | 總節點 (`QS` 節點) | Baseline Mean NPS | Wave 2A Mean NPS | Wave 2B Mean NPS | Wave 2B vs Baseline |
| :--- | :---: | ---: | ---: | ---: | ---: | :---: |
| `EPD_1` | `b1d2` (`105`) | 33,355 (16,315) | 407,339 | 646,277 | **658,545** | **+61.67%** |
| `EPD_2001` | `c1d2` (`18`) | 44,564 (19,427) | 511,995 | 550,105 | **527,638** | **+3.06%** |
| `EPD_4001` | `f3e5` (`-15`) | 58,769 (25,336) | 498,874 | 611,148 | **640,687** | **+28.43%** |
| `EPD_6001` | `f1c4` (`-7`) | 85,703 (43,571) | 496,269 | 530,169 | **590,595** | **+19.01%** |
| `EPD_8001` | `d1d4` (`42`) | 60,435 (24,494) | 476,166 | 560,270 | **601,716** | **+26.37%** |
| `EPD_10001` | `d2c4` (`15`) | 62,313 (31,621) | 545,548 | 586,100 | **608,055** | **+11.46%** |
| `EPD_12001` | `a2a3` (`-6`) | 81,219 (40,220) | 532,129 | 633,871 | **646,296** | **+21.45%** |
| `EPD_14001` | `e2e4` (`60`) | 106,482 (59,896) | 563,161 | 648,201 | **652,129** | **+15.80%** |
| `EPD_16001` | `e2d4` (`13`) | 77,928 (40,325) | 508,985 | 574,951 | **508,442** | **-0.11%** |
| `EPD_18001` | `e1g1` (`44`) | 85,657 (59,054) | 557,603 | 752,373 | **703,235** | **+26.12%** |
| `EPD_20001` | `a4b3` (`32`) | 44,746 (22,658) | 478,752 | 605,422 | **567,603** | **+18.56%** |
| `EPD_22001` | `c1e3` (`32`) | 63,387 (30,321) | 535,425 | 700,835 | **599,447** | **+11.96%** |
| `EPD_24001` | `e2d2` (`55`) | 18,750 (6,578) | 333,096 | 506,231 | **567,439** | **+70.35%** |
| `EPD_26001` | `f1d3` (`-16`) | 37,116 (20,678) | 491,945 | 569,352 | **569,980** | **+15.86%** |
| `EPD_28001` | `f1d3` (`65`) | 12,276 (4,145) | 321,159 | 496,824 | **583,259** | **+81.61%** |
| `EPD_30001` | `f3d2` (`-13`) | 57,723 (29,652) | 458,566 | 577,586 | **577,654** | **+25.97%** |
| `Startpos` | `d2d4` (`30`) | 41,360 (16,186) | 437,527 | 627,553 | **574,734** | **+31.36%** |
| `Kiwipete` | `e2a6` (`-117`) | 66,878 (41,887) | 482,075 | 648,384 | **615,140** | **+27.60%** |
| `Tactical_MG` | `f4f8` (`29999`) | 587 (65) | 28,932 | 132,171 | **180,475** | **+523.78%** |
| `Rook_EG` | `b4f4` (`21`) | 16,908 (6,194) | 380,359 | 833,077 | **867,460** | **+128.06%** |

*(註：Wave 2 所有嚴格 0 搜尋樹變動的加速與架構優化已全數到位；`depth <= 0` 直接分派 QS 因依賴 Bug #12 的 QS 重複偵測且會微幅改變葉節點計數，留待 Wave 1B 修妥 Bug #12 後一併處理。未經使用者同意不執行 `tools/sync_classical_old.py --sync`。)*
