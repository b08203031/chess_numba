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
| **Wave 2B** | 2026-11-22 | 補齊剩餘 Wave 2 架構優化（64B AoS `pawn_table`、`SearchContext` 80→65 欄、TT 16B 對齊、`.nbi` 多重特化收斂、`quiets` 展開） | **0 差異**（20/20 FENs `1,056,156` nodes 100% 一致） | 96 項測試熱跑 `21.39s` 全過；中位數均值 NPS `458,684 → 601,128`（**+31.05%**）；**Colab 500ms/move 實戰 780 局 `417.5 - 362.5`（53.53%，`194W 447D 139L`，`+24.54 ± 15.92 Elo`，`LLR=+2.95` H1 Accepted）** | ✅ **SPRT H1 通過** |
| **Wave 1A** | 2026-11-22 | 安全／介面／殘局純正確性 BUG 修復（#4 #5 #6 #14 #15 #16 #17 #18 #25 #29 #30，共 11 項） | **20/20 FENs 節點 `1,056,156` 100% 一致**（一般局面零變動；僅特定殘局／畸形 FEN／開局庫輸出被修正） | 130 項測試全過；20 盤面 Depth 10 × 5 次加權 NPS `607,033`（+1.62%）；**Colab 500ms/move 實戰 2,600 局 `545W 1520D 535L`（`50.19%`，`+1.34 ± 8.24 Elo`，LOS `62.5%`，`SPRT[-10, 0] LLR=+3.58` H1 Accepted）；已同步至 `classical_old/`** | ✅ **無退化驗收 & 已同步基準** |
| **Wave 1B-1** | 2026-11-23 | Bug #1 Part A：修正 `has_best_move` 判定（排除 `TT_FLAG_ALPHA` Fail-Low），啟用負向 Correction History | 20 FENs Depth-10 總節點 `1,056,156 → 1,059,970`（**`+0.36%`**）；`9/20` 變著、`19/20` 變分 | 72 項測試全過；熱匯入 `5.84s` / `0` `.numba.unresolved$`；**Colab `300k nodes` 1,628 局 `303W 1055D 270L`（`51.01%`，`+7.04 ± 9.72 Elo`，LOS `92.2%`，`SPRT[-10, 0] LLR=+4.90` H1 Accepted，`[-5, 0] +1.94`）；已同步至 `classical_old/`** | ✅ **接受並已同步基準** |
| **Wave 1B-2** | 2026-11-24 | Bug #2：禁止 `lazy=True` 粗評寫入 `TT.static_eval` 與校正歷史，並複用 TT 完整靜態分 | 20 FENs Depth-10 總節點 `1,059,970 → 900,152`（**`-15.08%`**）；`7/20` 變著、`18/20` 變分 | 74 項測試全過；Weighted NPS `644,881 → 643,029`（`-0.29%` 近零損失）；熱匯入 `6.51s` / `0` `.numba.unresolved$`；使用者確認無退化並已同步 `classical_old/` | ✅ **已同步基準** |
| **Wave 1B-3** | 2026-11-24 | Bug #3 + #8：禁止 QS 寫入 `TT_FLAG_EXACT` 覆寫深層 TT；修復無鎖 TT RMW 競態、`int16(-32768)` 溢位與 `probe_tt_best_move` TOCTOU | Depth-10 (18 FENs) `+20.44%` nodes；**Depth-14 (40 FENs)** `12.04M → 12.73M`（`+5.75%`），`12/40` 變著、`35/40` 變分（`mean\|dEval\| = 30.1 cp`）；**`d0_EXACT: 242,035 → 0`**、**`TT d≥6: +9.62%`** | **82 / 82** 測試全過（`6.382s`）；熱匯入 `4.30s` / `0` `.numba.unresolved$`；**Colab `300k nodes` 882 局達 `SPRT[-10, 0] +3.13` 並同步（`c464f50`）；2,000 局跑完終局 `350W 1296D 354L`（`49.90%`，`-0.69 ± 8.77 Elo`，`[-12, 0] +3.17` H1、`[-10, 0] +2.15`，無退化持平）** | ✅ **無退化驗證 & 已同步基準** |
| **Wave 1B-4** | 2026-11-24 | Bug #7：於 `score_captures_lazy` 與 `score_captures_with_tt_lazy` 補齊升變價值增量，並將無吃子升變納入 `QS_EVASION_CAPTURE_BUCKET` | 20 標準 FENs Depth-10 `1,080,177 → 1,079,934`（**`-0.02%`**）；5 升變專題 FENs `216,505 → 201,459`（**`-6.95%`**，`Promo_QSEvasion -20.17%`、`eval 1258 → 11385`） | **84 / 84** 測試全過（`16.504s`）；Weighted NPS `643,139 → 654,694`；熱匯入 `7.11s` / `0` `.numba.unresolved$`；**Colab `400k nodes` 3,090 局 `548W 2004D 538L`（`50.16%`，`+1.12 ± 7.26 Elo`，LOS `62.3%`，`SPRT[-10, 0] LLR=+4.76` H1 Accepted，`[-5, 0] +1.41`）；已正式驗收並同步 `classical_old/`（`f7d271a`）** | ✅ **無退化驗收 & 已同步基準** |
| **Wave 1B-5** | 2026-11-24 | Bug #9：移除 PV 節點零窗口 Fail-High 兩處 `and evaluation < beta` 守衛，強制以 `is_pv=True` 全窗口重搜驗證並補全 PV | 20 標準 FENs Depth-10 `1,080,177 → 990,157`（**`-8.33%`**）；`7/20` 變著、`17/20` 變分 | **86 / 86** 測試全過（`5.278s`）；熱匯入 `8.41s` / `1` 過載 / `0` `.numba.unresolved$`；**Colab `400k nodes` 1,178 局 `200W 757D 221L`（`49.11%`，`-6.19 ± 11.86 Elo`，LOS `15.5%`，`SPRT[0, +10] LLR=-3.00` H0 Rejected，`[-5, 0] -0.50`，`[-3, +3] -1.00`）；`classical_old/` 未受影響** | ❌ **SPRT H0 拒絕（待回退或限縮重測）** |

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

---

### 4. Wave 2 固定時間（`500 ms/move`）Colab SPRT 實戰對戰驗證（Commit `0f4c2eb` vs `d015345`）

* **實驗目的**：由於 Wave 2（含 2A + 2B）為嚴格「零搜尋樹／零評估變動」的純加速與記憶體佈局優化（固定節點下為 0 差異），因此以**固定每步時間 `--time 500`（`500 ms/move`）**進行成對換先 SPRT 錦標賽，直接檢驗 **NPS 提升轉化為搜尋深度與實戰棋力（Elo）** 的幅度。
* **硬體與對戰配置**（記錄於 [`tournament_colab_runner_se_multicut.ipynb`](../../tournament_colab_runner_se_multicut.ipynb)）：
  * **環境**：Google Colab 8-vCPU `AMD EPYC 7B12`（50 GiB RAM，Python 3.13.18）
  * **引擎對照**：`New` = `chess_engine.classical`（Commit `0f4c2eb`，Wave 2 完整版）vs `Old` = `chess_engine.classical_old`（Commit `d015345` 基準）
  * **對局參數**：`--time 500`（每步固定 0.5 秒）、`--concurrency 7`（7 組平行對局 Worker）、`--tt-mb 64`、開局庫 `data/openings.epd`（成對換先）、預設裁決開啟、SPRT `elo0 = 0.0, elo1 = 10.0`（$\alpha = \beta = 0.05$，邊界 `[-2.94, +2.94]`）。
* **對局結果與統計檢定（於第 780 局 / 390 對正式觸發 `H1 Accepted` 早停）**：
  * **總戰績**：**`New 417.5 - 362.5 Old`**（**780 局：194 勝 447 和 139 負**，得分率 **`53.53%`**，淨勝 **`+55` 局**，和局率 `57.31%`）。
  * **Elo 增益**：**`+24.54 ± 15.92 Elo`**（成對換先群聚 95% CI：**`[+8.54, +40.64]`**，下界顯著 $> 0$）。
  * **SPRT `[0, 10]`**：**`LLR = +2.95`**（突破上界 `+2.94`）→ **`H1 Accepted (Engine improved)` 🏆**。
  * **先後手對稱表現**：
    * `New` 持白：**`55.00%`**（`103W - 223D - 64L`，淨勝 `+39` 局）
    * `New` 持黑：**`52.05%`**（`91W - 224D - 75L`，淨勝 `+16` 局）
  * **成對換先九宮格（共 390 對唯一開局 FEN，0 重複對局）**：
    * `DD: 135 (34.6%)` | `WD: 54 (13.8%)` | `DW: 51 (13.1%)` | `DL: 37 (9.5%)` | `LD: 35 (9.0%)` | `WW: 26 (6.7%)` | `WL: 23 (5.9%)` | `LL: 15 (3.8%)` | `LW: 14 (3.6%)`
    * **雙勝對比**：`WW : LL = 26 : 15`（淨勝 **`+11` 對雙勝**，比值 `1.73x`）
    * **單勝對比**：`(WD + DW) : (DL + LD) = 105 : 72`（淨勝 **`+33` 對單勝**）
    * **成對總淨勝**：`131` 勝對 vs `87` 負對（**淨勝 `+44` 對**）。
  * **對局長度分桶（平均 `51.2` 回合，$\sigma = 20.7$）**：
    * `short (<=25m)`：`n=68`，`3W-59D-6L`（`score = 0.478`）
    * `medium (26-50m)`：`n=354`，**`97W-203D-54L`（`score = 0.561`，淨勝 `+43` 局）**
    * `long (51-80m)`：`n=297`，**`82W-142D-73L`（`score = 0.515`，淨勝 `+9` 局）**
    * `very_long (>80m)`：`n=61`，**`12W-43D-6L`（`score = 0.549`，淨勝 `+6` 局）**
  * **序列穩定性**：前半場 195 對（390 局）`score = 0.532`（`100W-215D-75L`），後半場 195 對（390 局）`score = 0.538`（`94W-232D-64L`）；Lag-1 自相關 `0.0161`、游程檢定 `p = 0.4265`（全程穩定無漂移）。
* **搜尋遙測對比（`parse_search_stats.py`，`New` 39,689 步 vs `Old` 39,707 步，`draw_depth_cap=40`）**：

| 遙測指標 | `Old` (`classical_old`) | `New` (`classical` Wave 2) | 差異 / 增幅 | 統計顯著性 (Welch $t$-test) |
| :--- | :---: | :---: | :---: | :---: |
| **每步平均耗時 (ms)** | `355.8 ms` ($\sigma=81.6$) | `356.3 ms` ($\sigma=82.1$) | `+0.5 ms` (+0.1%) | $t = +0.861, p = 0.3895$（**等時一致**） |
| **7 並發實戰平均 NPS** | `260,640` (med `216,966`) | **`290,074`** (med **`242,205`**) | **`+11.3%`** (med `+11.6%`) | $t = +32.060, p = 4.47 \times 10^{-224}$ (**極顯著**) |
| **每步平均節點數 (Nodes)** | `91,829` (med `79,081`) | **`102,294`** (med **`88,047`**) | **`+11.4%`** (med `+11.3%`) | $t = +30.097, p = 7.03 \times 10^{-198}$ (**極顯著**) |
| **每步平均搜尋深度 (Depth)** | `14.22` ($\sigma=3.14$, med `14`) | **`14.53`** ($\sigma=3.08$, med `14`) | **`+0.31 ply`** | $t = +13.985, p = 2.17 \times 10^{-44}$ (**極顯著**) |
| **深層搜尋比例 (`depth >= 14`)** | `50.5%` | **`56.0%`** | **`+5.5%`** | `16-20 ply`: `20.9% → 23.9%` |

> **結論**：在 8 邏輯核心跑滿 7 對局並發（14 個引擎 Context 共享 L3 Cache 與記憶體頻寬）的嚴苛環境下，Wave 2 的熱路徑延遲計算、64B 單 Cache-Line AoS `pawn_table`、`.nbi` 特化收斂與整數歷史老化仍穩定提供 **`+11.3%` 實戰並發吞吐量**，推升搜尋深度 **`+0.31 ply`**，並在 780 局直接以 **`+24.54 ± 15.92 Elo`（`LLR = +2.95`）擊穿 SPRT H1 上界通過驗收**。待使用者確認後，即可執行 `python tools/sync_classical_old.py --sync` 將 Wave 2 正式固化至 `classical_old/` 生產基準，作為後續 Wave 1A / 1B 除錯的全新對照基線。

---

## Wave 1A：安全／介面／殘局純正確性 BUG 修復（2026-11-22）

> **性質**：搜尋與評估對「一般中局／開局局面」零變動（節點恆等已驗證）；只修正 (a) `boundscheck=False` 下的越界風險、(b) UCI／開局庫／FEN 介面輸出非法資料、(c) 稀有殘局（KBNK、KXK、KBPsK b/g 線、KRPKN、KRPPKRP）的錯誤評分／縮放。對應審計報告 [`CLASSICAL_ENGINE_AUDIT_20261122.md`](CLASSICAL_ENGINE_AUDIT_20261122.md) 的 Wave 1A。

### 1. 修改明細

| Bug | 檔案 | 修復內容 | SF 對照 |
| :---: | :--- | :--- | :--- |
| **#4** | [`search.py`](search.py) `_search` | 將 `ply >= MAX_PLY` 守衛改為 `ply >= MAX_PLY - 1` 並**前移至任何 per-ply 堆疊存取之前**（原本在 `cutoff_cnt[ply+2]`、`follow_pv_stack[ply]` 之後才檢查，且 `ply == 127` 時 6 處 `[ply+1]` 越界）。QSearch 在 `ply >= MAX_PLY` 守衛前無堆疊存取，維持不動 | — |
| **#5** | [`opening_book.py`](opening_book.py) | Polyglot 以「王吃己方車」編碼易位：依 `board_state` 映射 `e1h1/e1a1/e8h8/e8a8 → e1g1/e1c1/e8g8/e8c8` 並設 `SPECIAL_MOVE_FLAG_CASTLING`；兵斜走至空格設 `SPECIAL_MOVE_FLAG_EN_PASSANT`。實測西班牙／義大利開局原回傳非法 `e1h1`，現為合法 `e1g1` | Polyglot 規格 |
| **#6** | [`endgame.py`](endgame.py) `mate_kbnk` | (1) 逃生格不再以 `~strong_pieces` 排除（**無保護**的象可被弱王吃掉 → 非將死；有保護的子本來就落在 `strong_attacks`）；(2) 新參數 `weak_to_move`，逼和／將死捷徑僅在弱方行棋時成立（原強方行棋亦回傳 `0` 幻覺逼和）；(3) 象攻擊以 `occ ^ weak_king` 計算 X-Ray | SF11 `endgame.cpp` KBNK |
| **#14** | `endgame.py` `KBPsK` | b/g 線堡壘改為檢查**雙方所有兵**皆在該線（`(piece_bbs[0] \| piece_bbs[6]) & ~FILE_MASKS[f] == 0`） | SF11 `endgame.cpp:363` |
| **#15** | `endgame.py` | 移除非 SF 的 `KRPKN` 縮放歸零規則（白方／黑方兩段） | SF11 無此規則 |
| **#16** | `endgame.py` `_eval_kxk` | 弱方行棋時：孤王所有鄰格被攻擊（含穿透自身 X-Ray）且未被將軍 → 回傳 `0`（逼和）。利用既有 `is_square_attacked_with_occ` | SF11 `endgame.cpp:124` |
| **#25** | `endgame.py` `KRPPKRP` | `if 1 < r < 6` → `if 0 < r < 6`（第 2 橫排 `sf = 9` 現可達；白／黑兩段） | SF11 `r > RANK_1 && r < RANK_7` |
| **#17** | [`engine_types.py`](engine_types.py) | `eval_weights` 取得失敗不再靜默降級為 `np.zeros(64)`（`evaluation.py` 會在 `boundscheck=False` 下讀到索引 98）；改為明確 `RuntimeError`，並檢查長度 `>= 128` | — |
| **#18** | [`fen_parser.py`](fen_parser.py) | (1) 易位權需王／車位於原位才保留；(2) EP 格需符合行棋方對應橫排、後方有剛雙步之兵、且兩格皆空，否則忽略（**保留標準 FEN 的 EP 設定行為，不改 Zobrist 策略**；EP 雜湊策略屬 Wave 1B Bug #13）；(3) 半步鐘夾限 `[0, 255]`（負值原觸發 `OverflowError`） | SF19 `position.cpp` 驗證 |
| **#29** | `engine_types.py` `reset_search_context` | 補清 `game_history`、`game_history_count`、`tt_generation`、`nmp_min_ply`、`diag_stats`，另含 `cutoff_cnt`、`move_count_stack`、`tt_hit_stack`、`move_is_capture_stack` | — |
| **#30** | [`move.py`](move.py) | `move_to_uci(NO_MOVE)` 回傳 `"0000"`（原為不可能的 `"a1a1"`，將死／逼和盤面 UCI 輸出非法 `bestmove`） | UCI null move |

### 2. 驗證

* **新增回歸測試** [`tests/test_wave1a_fixes.py`](../../tests/test_wave1a_fixes.py)（27 項）：Polyglot 易位／EP／真實開局庫著法皆為合法著法、`NO_MOVE → "0000"`、FEN 畸形輸入消毒、KBNK 三個審計反例與真將死／真逼和、KXK 雙方逼和與有逃生格仍判勝、KBPsK／KRPKN／KRPPKRP 縮放、`eval_weights` 完整表與 `reset_search_context`、`MAX_PLY` 守衛順序契約。
* **審計反例實測（修復前 → 修復後，白方視角）**：
  * `k7/B1K5/1N6/8/8/8/8/8 b - - 0 1`：`+29871`（幻覺將死）→ `+14890`（正常大優勢）
  * `k7/2K5/2N5/8/8/8/8/6B1 w - - 0 1`：`0`（幻覺逼和）→ `+14855`
  * `k7/2Q5/1K6/8/8/8/8/8 b - - 0 1`（逼和）：`-11515` → `0`
  * `1k6/1p6/1P6/4p3/8/3B4/8/4K3 w - - 0 1`：`sf = 0` → `sf = 43`；`8/4k3/4P3/3nK3/8/8/8/4R3 w - - 0 1`（KRPKN）：`sf = 0` → `sf = 43`
  * KRPPKRP `r = 1`：不可達 → `sf = 9`
* **全批次單元測試**（一個 process，`tests.test_wave1a_fixes tests.test_tunable_eval_parity tests.test_endgame_conformance tests.test_pinned_eval_geometry tests.test_see tests.test_search_core_contracts tests.test_lazy_smp tests.test_history`）：**130 項全過**。因 `endgame.py`／`search.py`／`engine_types.py` 變動（Numba 磁碟快取僅以「定義檔」mtime 判定失效，不會偵測被呼叫端變更），已先清除 `chess_engine/classical/__pycache__` 與 `tuner/__pycache__` 全部 `.nbi/.nbc` 後**冷編譯**重跑，耗時 `1516 s`（約 25 分鐘；受背景 CPU 競爭影響偏慢）。
* **節點恆等（20 盤面 Depth 10，與 Wave 2 後基準比對）**：`mismatches = 0 / 20`，總節點 `1,056,156`，`best_move` / `best_eval` / `quiescence_nodes` 逐盤面 100% 一致 — 證明修復對一般局面（開局庫 EPD、Startpos、Kiwipete、戰術中局、Rook 殘局）完全無搜尋樹影響。
* 未執行 Perft：本波未更動 `board_operations` / `move_generator` / `zobrist`。

### 3. 風險與後續

* **#17 行為改變**：`tune_eval_match.eval_weights` 不可匯入時 `SearchContext` 現在會直接失敗（原本靜默產出無意義評估）。部署／打包時務必隨附 `tune_eval_match/eval_weights.py`（或日後改為內建於 `classical/`）。
* **#16 / #6 / #14 / #15 / #25 為稀有殘局修正**：預期對 Elo 影響極小（< 數 Elo、需極多局才可量測）；建議不單獨跑 SPRT，併入 Wave 1B 之後的整合對戰（以 `[-5, 0]` 非退化檢定）一併驗證。
* **#5** 影響 `OwnBook=True` 的 UCI 對局（`tools/tournament.py` 預設關閉 OwnBook，故過去 SPRT 未受影響）。

### 4. 重測效能基準與 Colab 實戰對戰驗證（Wave 1A vs Wave 2）

* **重測效能基準（20 盤面 Depth 10 × 5 次，系統回復正常負載後重跑）**：
  * 總節點數：**`1,056,156`**，`mismatches = 0 / 20`（與 Wave 2 完全一致），總耗時 **`1.740 s`**（Wave 2 為 `1.768 s`）。
  * 加權平均 NPS：`597,371 → 607,033`（**`+1.62%`**）；算術平均 NPS：`592,026 → 594,421`（**`+0.40%`**）；中位數均值 NPS：`601,128 → 598,558`（`-0.43%`）。（與熱路徑無變動預期完全相符；該輪稍早因 25 分鐘冷編譯後筆電降頻 `% Processor Performance = 55.6%` 所測之 `307,009` 已確認屬降頻暫態，本次回復至 `% Processor Performance = 83.56%`、`2187.00 MHz`。）
* **Colab 固定時間（`500 ms/move`，7 並發 Worker，64 MB TT）實戰驗證（Commit `b8b8717` vs `0f4c2eb`，2026-11-22 ~ 11-23）**：
  * **對局規模與總成績**：**2,600 局（1,300 組嚴格成對換先，0 重複對局）**，得分 **`1,305.0 / 2,600`**（**`50.19%`**，**`545W 1520D 535L`**，淨勝 **`+10` 局**，和局率 `58.46%`）。
  * **Pentanomial 五項分佈 `[LL, LD/DL, DD/WL, WD/DW, WW]`**：`[48, 298, 598, 308, 48]`（`WW - LL = 0` 對，`WD - LD = +10` 對；詳細九宮格 `DD:457`, `DL:180`, `WD:166`, `DW:142`, `LD:118`, `WL:88`, `LW:53`, `WW:48`, `LL:48`）。
  * **相對 Elo 與 LOS**：Pentanomial **`+1.34 ± 8.24 Elo`**（成對換先群聚 95% CI **`[-6.91, +9.58]`**，**LOS = `62.5%`**）。
  * **多區間非退化 SPRT（[tools/sprt_calc.py](../../tools/sprt_calc.py)）**：
    * `SPRT[-10.00, +0.00]`：**`LLR = +3.5828`**（`H1 Accepted`）
    * `SPRT[-5.00, +0.00]`：**`LLR = +1.0850`**（正向推進）
    * `SPRT[-3.00, +0.00]`：**`LLR = +0.4813`** / `SPRT[-2.00, +0.00]`：**`+0.2643`** / `SPRT[-1.75, +0.25]`：**`+0.2360`**
  * **搜尋遙測（`New` 131,740 步 vs `Old` 131,718 步）**：平均耗時 `356.9 ms` vs `357.2 ms`（`p = 0.4196`）、平均深度 `14.48` vs `14.46`（`p = 0.3769`）、每步節點 `100,597` vs `100,250`（`p = 0.0751`）、NPS `284,946` vs `283,780`（`+0.4%`, `p = 0.0241`）— 兩版搜尋速度與深度完全持平。
  * **結論與基準同步**：由於 Wave 1A 11 項修復中有 6 項為介面／防崩潰安全守衛（`OwnBook` 預設關閉且未觸發極深越界／畸形 FEN），另外 `KBNK`/`KXK` 亦多數被 `±500 cp` Adjudication 提早裁定，因此主要剩 `KBPsK` / `KRPKN` / `KRPPKRP` 貢獻微幅正增益；2,600 局 `+1.34 Elo` / `+10` 局淨勝已充分完成無退化驗收，並已同步固化至 [`classical_old/`](../classical_old/) 作為後續 **Wave 1B** 的全新對照基線。

---

## Wave 1B-1：修復 Bug #1 Part A — 啟用 Fail-Low 負向 Correction History（2026-11-23）

> **對照基準**：Commit `b212150`（`classical_old/` 已同步至 Wave 1A；本項為嚴格單軸修復，**未改動** `ply > 0` `best_move` 更新邏輯，留待 Part B 獨立驗證）。

### 1. 修復內容（[`search.py`](search.py) `# --- Update Correction History ---`）

* **問題根因**：在 `ply > 0` 時，Step 14 受 `not is_loss(max_eval)` 保護，第 1 個合法著法在 `evaluation > -INFINITY` 下必將 `best_move` 設為非 `NO_MOVE`。原本寫 `has_best_move = (best_move != NO_MOVE)` 導致在全部 `TT_FLAG_ALPHA`（Fail-Low）節點中 `has_best_move` 仍為 `True`，不僅使 `(max_eval > static_score) == has_best_move` 在 `max_eval < static_score` 時 100% 拒絕更新負向校正（`CORRECTION_BONUS_NO_MOVE_DIV = 8` 完全死碼），還會在未證實上界 `static_score < max_eval <= original_alpha` 時錯誤寫入正向校正。
* **單軸修法**：
  * 改為 `has_best_move = (final_flag != TT_FLAG_ALPHA) and (best_move != NO_MOVE)`（對齊 SF19 `search.cpp:1631-1637`：僅於提高 `alpha` 時才認定存在已證實最佳著法）。
  * 僅在 `if has_best_move:` 下檢查 `is_capture_or_promo`（Fail-Low 無 `bestMove`，即使第 1 步碰巧試了吃子也不阻斷全域結構性負向校正）。
  * 加入 `if diff != 0:` 避免無意義寫入（不改動 `test_search_core_contracts.py` 所驗證之核心契約行）。

### 2. 驗證與診斷結果

* **單元測試**：`python -m unittest tests.test_wave1b_fixes tests.test_search_core_contracts tests.test_wave1a_fixes tests.test_lazy_smp tests.test_history -v` — **72 / 72 項全過**（`2.343s` 熱跑）。
  * 新增 [`tests/test_wave1b_fixes.py`](../../tests/test_wave1b_fixes.py)（2 項）：契約檢查 `has_best_move` 條件，並驗證實搜 `Startpos depth=9` 下全部 5 張校正表均同時存在正項（`pos > 0`）與負項（`neg > 0`），且絕對值 $\le 1024$。
* **Numba 快取與符號檢查**：跨進程 `warm_import = 5.84s`、`warmup_search_jit = 0.22s`；`_search` / `quiescence_search` / `_lazy_smp_helper_loop_jit` 均維持 **1** 過載、`unresolved: []`（**`0`** 個 `.numba.unresolved$`）。
* **基準差異與衛生**：`python tools/sync_classical_old.py --diff` 顯示僅 `search.py` 上述單處差異；修改與新增檔案 `CRCRLF = 0`。
* **20 盤面 Depth 10 搜尋樹與校正符號對比（`classical_old` Wave 1A vs `classical` Wave 1B-1）**：
  * **總節點數**：`1,056,156 → 1,059,970`（**`+0.36%`**，樹規模極為穩定）；Weighted NPS `568,866 → 606,272`；著法變動 `9 / 20`、評分變動 `19 / 20`（唯一無變動為 `Tactical_MG` 步殺 `587` 節點）。
  * **Kiwipete Depth-10 校正表分佈 `(pos, neg, min, max, nonzero_mean)`**：

| Correction Table | Wave 1A (`classical_old`) | Wave 1B-1 (`classical`) | 變化判讀 |
| :--- | :--- | :--- | :--- |
| `pawn_correction_history` (W-abs) | `(10, 6, -245, +500, +37.3)` | `(1, 8, -366, +9, -173.0)` | 不再誤寫假正向，反映白方靜態微高估 |
| `minor_correction_history` (W-abs) | `(17, 6, -479, +460, +40.9)` | `(1, 7, -608, +11, -230.2)` | 同上，恢復下修方向 |
| `non_pawn_correction_history_white` | `(18, 0, 0, +513, +155.6)` | **`(1, 6, -551, +13, -275.9)`** | **由 0 負項解除死碼** |
| `non_pawn_correction_history_black` | `(5, 0, 0, +538, +326.6)` | `(2, 0, 0, +148, +109.5)` | Startpos d9 亦確認正負兼具 `(4, 2)` |
| `continuation_correction_history` | `(36, 0, 0, +489, +111.8)` | **`(6, 10, -407, +119, -88.2)`** | **由 0 負項解除死碼** |

### 3. Colab 固定節點（`300k nodes`）SPRT 實戰驗證（Commit `d896770` vs `b212150`）

* **實驗配置**：`--nodes 300000`、`--concurrency 7`、`--tt-mb 64`、成對換先開局（`data/openings.epd`）。
* **總戰績（1,628 局 / 814 對）**：**`New 830.5 - 797.5 Old`**（**`303W 1055D 270L`**，得分率 **`51.01%`**，淨勝 **`+33` 局**）。
* **Pentanomial 五項分佈 `[LL, LD/DL, DD/WL, WD/DW, WW]`**：`[17, 177, 402, 192, 26]`（`WW - LL = +9` 對、`WD - LD = +15` 對，淨勝 **`+24` 對**；九宮格：`WW=26, WD=111, DW=81, WL=40, DD=343, LW=19, DL=93, LD=84, LL=17`）。
* **相對 Elo 與 LOS**：Pentanomial **`+7.04 ± 9.72 Elo`**（95% CI **`[-2.68, +16.76]`**，**LOS = `92.2%`**）；前半場（814 局）`51.66%`、後半場（814 局）`50.37%` 全程穩定為正。
* **多區間 SPRT（[tools/sprt_calc.py](../../tools/sprt_calc.py)）**：
  * `SPRT[-10.00, +0.00]`：**`LLR = +4.8977`**（`H1 Accepted`）
  * `SPRT[-5.00, +0.00]`：**`LLR = +1.9411`** / `SPRT[-3.00, +3.00]`：**`+1.7191`** / `SPRT[0.00, +5.00]`：**`+0.9241`** / `SPRT[0.00, +10.00]`：**`+0.8314`**
* **結論與同步決策**：已以 `92.2%` LOS 確認無退步並取得實質正增益（預期在修復 Bug #2 移除 `lazy=True` 粗評污染後校正訊號會進一步純化）；已依使用者確認執行 `python tools/sync_classical_old.py --sync` 將 [`classical_old/`](../classical_old/) 推進至 Wave 1B-1 基準，準備進入 **Wave 1B-2（Bug #2）**。

---

## Wave 1B-2：修復 Bug #2 — 禁止 `lazy=True` 粗評寫入 `TT.static_eval` 與校正歷史（2026-11-24）

> **對照基準**：Commit `648d8ac`（`classical_old/` 已同步至 Wave 1B-1；嚴格單軸修復，未混入 Bug #3 QSearch `TT_FLAG_EXACT` 修正）。

### 1. 修改明細（[`search.py`](search.py) 三處）

1. **TT 完整靜態分複用**（TT Probe 後）：當 `tt_hit and tt_static_eval_raw != 32767` 時，設定 `static_eval_is_full = True`（因日後寫入 TT 的靜態分保證為 `lazy=False` 完整 HCE，可安全避免 RFP/Razoring/NMP/FP 重複計算完整評估；所有 `cached_pinned_*` 下游均已具備 `PINNED_UNCOMPUTED_SENTINEL` 檢查）。
2. **Correction History 更新前升級完整評估**（`# --- Update Correction History ---`）：於安靜深度 $\ge 4$ 節點且 `not static_eval_is_full` 時，在執行校正更新前呼叫 `_evaluate_position_jit(..., False, search_context)` 與 `apply_correction_history_score(...)` 升級至完整 HCE 並設 `static_eval_is_full = True`（使校正歷史不再針對缺少 Tempo／兵型／王安全／機動性／威脅的粗評訓練，並讓 `store_tt` 一併寫入精確完整靜態分）。
3. **阻斷 `lazy=True` 寫入 `TT.static_eval`**：`store_tt` 前僅當 `static_eval_is_full and raw_static_eval != -INFINITY and abs(raw_static_eval) < MATE_SCORE - MAX_PLY` 才寫入 `raw_static_eval`，否則存哨兵 `32767`，防止 QSearch 葉節點把 `lazy=True` 粗評當作 `stand_pat`。

### 2. 驗證與診斷結果

* **單元測試**：`python -m unittest tests.test_wave1b_fixes tests.test_search_core_contracts tests.test_wave1a_fixes tests.test_lazy_smp tests.test_history -v` — **74 / 74 項全過**。
  * 新增 [`TestBug2LazyEvalTTIsolation`](../../tests/test_wave1b_fixes.py)：驗證結構契約，並於審計反例 `r4rk1/ppp3p1/2n4p/4p1Nq/2B2Q2/8/PPP3PP/R4R1K w - - 0 1` 與 `Kiwipete` 以 `depth = 2 / 6` 實搜確認 `probe_tt_snapshot` 讀回之 `tt_static_eval` 絕不為 `lazy_eval`（僅為 `32767` 或 `full_eval`）。
* **Numba 快取與符號檢查**：`warm_import = 6.51s`、`warmup_search_jit = 0.23s`；`_search` / `quiescence_search` / `_lazy_smp_helper_loop_jit` 各 **1** 過載、`unresolved: []`。
* **20 盤面 Depth 10 對照（`classical_old` Wave 1B-1 vs `classical` Wave 1B-2）**：
  * **總節點數**：`1,059,970 → 900,152`（**`-15.08%`**，搜尋樹顯著收斂！）
  * **Weighted NPS**：`644,881 → 643,029`（`-0.29%`，靠 TT 完整靜態分複用幾乎完全抵銷 `Correction History` 升級完整評估的開銷）
  * **著法／評分變化**：`7 / 20` 盤面改變最佳著法、`18 / 20` 盤面改變評分（`EPD_6001 f1c4→f1b5`, `EPD_8001 e3d4→d1d4`, `EPD_10001 h2h3→d2c4`, `EPD_14001 d1a4→f4e5`, `EPD_16001 e2f4→e1g1`, `EPD_18001 c3b5→e1g1`, `EPD_22001 a2a3→c1e3`）。
  * **Kiwipete Depth-10 校正表分佈**：5 張校正表均維持穩定負向修正（`pawn (1,10)`, `minor (1,9)`, `np_white (1,9)`, `np_black (2,0)`, `cntcv (6,14)`），且不致因 `lazy=True` 缺特徵而過度外推（`non_pawn_correction_history_white` nonzero mean 由 `-275.9` 收斂至 `-184.9`）。

> **後續紀錄**：使用者於 2026-11-24 確認 Wave 1B-2 測試結果無退步，已執行 `python tools/sync_classical_old.py --sync` 將 [`classical_old/`](../classical_old/) 同步至 Wave 1B-2，作為 Wave 1B-3（Bug #3 + #8）的對照基準。

---

## Wave 1B-3：修復 Bug #3（QS `TT_FLAG_EXACT` 覆寫深層 TT）與 Bug #8（無鎖 TT 競態與欄位完整性）（2026-11-24）

> **對照基準**：`classical_old/` 已同步至 Wave 1B-2；本波維持 `classical_old/` 於 Wave 1B-2 不動，以便單軸 SPRT 驗證 Bug #3 + #8。

### 1. 修改明細

1. **[`transposition_table.py`](transposition_table.py)（Bug #8 + Bug #3 防禦守衛）**：
   * **`penalize_tt`**：進入槽位先以區域純量快照全欄位；遇 `TT_FLAG_NONE` 跳過（`continue`）、`> TT_FLAG_BETA` 終止；先驗算 16-bit XOR checksum 才允許修改（拒絕把 SMP 撕裂條目「封成合法 checksum」）；深度未變時不寫回；有變動時以已驗證區域純量重算 checksum 並完整寫回。
   * **`probe_tt_snapshot`**：於 `e_key == key32` 且 `e_flag == TT_FLAG_NONE` 時改為 `continue`（避免 `key32 == 0` 空格提前中斷 4-way 掃描）。
   * **`probe_tt_best_move`**：移除經 `probe_tt(...)` 回傳可變結構指標的 TOCTOU 讀取，改為直接於區域純量快照並驗算 checksum（保留 `probe_tt` / `_EMPTY_TT_ENTRY` 匯出及 `flag == TT_FLAG_NONE` 測試契約）。
   * **`store_tt`**：
     * 精確匹配條件改為 `e_key == key32 and e_flag != TT_FLAG_NONE`，快照並驗證舊 checksum（`e_valid`）；若舊條目損毀則強制乾淨覆寫，不繼承撕裂的 `best_move` 或 `static_eval`。
     * **Bug #3 守衛**：`flag == TT_FLAG_EXACT and (depth > 0 or e_depth == 0)` 禁止 `depth=0` 的 `EXACT` 條目覆寫 `depth > 0` 主搜尋條目。
     * `replace=True` 無條件寫回 `static_eval`（來值為 `32767` 且舊條目合法時沿用 `e_static_eval`），消除算進 checksum 卻漏寫欄位的不一致。
     * `else:`（不覆寫）路徑僅在 `updated == True` 時才以已驗證區域純量重算 checksum 並寫回；允許 `bm_u16 != 0 and flag != TT_FLAG_ALPHA` 在 `(e_best_move == 0 or (depth > 0 and e_flag == TT_FLAG_ALPHA))` 時補寫較可靠 `best_move`；次級將死老化改用 `abs(np.int32(e_score))` 修正 Numba 中 `int16(-32768)` 溢位問題。
     * 碰撞替換迴圈改用無符號區域純量快照 `c_flag` / `c_gen` / `is_pv`。
2. **[`search.py`](search.py) `quiescence_search`（Bug #3）**：
   * 移除未用變數 `original_alpha`，並自 `probe_tt_snapshot` 解包 `qs_tt_is_pv`，於全部 3 處 `store_tt` 呼叫保留 `bool(qs_tt_is_pv)` 並統一傳入 `int(_i32(...))` 型別。
   * Stand-pat beta 截斷之 `store_tt` 改為僅於 `qs_tt_flag == TT_FLAG_NONE` 時寫入（對齊 SF19 `search.cpp:1750`）。
   * 無吃子葉節點（`move_count == 0`）改存 `TT_FLAG_ALPHA`（原誤存 `TT_FLAG_EXACT`）。
   * 著法迴圈結尾改存 `TT_FLAG_BETA if best_eval >= beta else TT_FLAG_ALPHA`（永不於 QS 寫入 `TT_FLAG_EXACT`，對齊 SF19 `search.cpp:1877`）。

### 2. 驗證與診斷結果

* **單元測試**：`python -m unittest tests.test_wave1b_fixes tests.test_search_core_contracts tests.test_lazy_smp tests.test_wave1a_fixes tests.test_history -v` — **82 / 82 項全過**（`6.382s`）。
  * 於 [`tests/test_wave1b_fixes.py`](../../tests/test_wave1b_fixes.py) 新增 `TestBug3QSearchTTNoExactOverwrite`（3 項）與 `TestBug8LocklessTTSafety`（5 項），涵蓋 QS 無 `TT_FLAG_EXACT` 契約、`depth=0` 守衛、實搜保留 `depth=12` TT 與零 `d0_EXACT`、`penalize_tt` 拒封撕裂條目、`store_tt` 撕裂清理、`static_eval` 寫回一致、`int16(-32768)` 次級老化及 `probe_tt_best_move` 純量快照；另於 `TestBug1PartANegativeCorrectionHistory` 補入黑先中局盤面（`r2q1rk1/ppp2ppp/2n1bn2/2bpp3/4P3/2NP1NP1/PPP1QPBP/R1B2RK1 b - - 0 8`）以穩定覆蓋 `non_pawn_correction_history_black` 負向樣本（Bug #3 清除 QS 假截斷後，原兩盤白先 depth-9 測試已不足以固定產生黑方非兵負向節點）。
* **Numba 快取與符號檢查**：跨進程 `warm_import = 4.30s`、`warmup_search_jit = 0.12s`；`.nbi` 磁碟過載數 `_search = 1`、`quiescence_search = 1`、`_lazy_smp_helper_loop_jit = 1`、`probe_tt_best_move = 1`、`probe_tt_snapshot = 1`、`penalize_tt = 2`、`store_tt = 3`；LLVM IR 檢查 `unresolved_ir_lines: []`（`0` `.numba.unresolved$`）。
* **基準差異與衛生**：`python tools/sync_classical_old.py --diff-code` 僅顯示 `search.py` 與 `transposition_table.py` 上述修復；`CRCRLF = 0`。

### 3. Depth-10 連續性與 Depth-14 多資料集對照（Wave 1B-2 `classical_old` vs Wave 1B-3 `classical`，64 MB TT）

* **Depth-10 連續性基準（18 盤面：16 抽樣 `data/openings.epd` + `Startpos` + `Kiwipete`）**：
  * Total nodes：`882,030 → 1,062,334`（**`+20.44%`**，幾何平均 `1.2280x`）；QS nodes：`439,402 → 533,892`（`+21.50%`）；Weighted NPS：`785,241 → 802,418`（**`+2.19%`**）；`mv_chg = 4/18`、`eval_chg = 14/18`、`mean|dEval| = 10.6 cp`；**`d0_EXACT: 16,243 → 0`**；**`TT d≥6: 1,101 → 1,534`（`+39.33%`）**；`TT_hits: 41,225 → 46,879`。
* **Depth-14 多資料集測試（共 40 盤面：`.epd` 與 `tuner/data/hce/val_pgn.npz`）**：

| 資料集來源 | 盤面數 | 總節點 (`Old → New`) | 變動 (Geomean) | Weighted NPS | 變著 / 變分 (`mean\|dEval\|`) | `d0_EXACT` (`Old → New`) | `TT d≥6` 保留量 |
| :--- | :---: | ---: | :---: | :---: | :---: | :---: | :---: |
| `data/openings.epd` | 10 | `4,181,301 → 4,044,130` | `-3.28%` (`0.9898x`) | `766,596 → 756,303` (`-1.34%`) | `4/10` / `10/10` (`13.1 cp`) | `59,682 → 0` | `14,034 → 13,108` (`-6.60%`) |
| `data/sts/STS1-STS15_LAN_v6.epd` | 10 | `2,535,525 → 3,390,789` | `+33.73%` (`1.2971x`) | `794,046 → 780,147` (`-1.75%`) | `4/10` / `8/10` (`30.2 cp`) | `43,773 → 0` | `10,840 → 15,405` (`+42.11%`) |
| `data/hist_diag_puzzles_500.epd` | 10 | `1,484,445 → 1,667,647` | `+12.34%` (`0.9584x`) | `774,645 → 806,115` (`+4.06%`) | `2/10` / `8/10` (`69.7 cp`) | `31,532 → 0` | `6,230 → 5,872` (`-5.75%`) |
| `val_pgn.npz` MG (`18..26` 子) | 5 | `1,758,508 → 1,940,656` | `+10.36%` (`1.3153x`) | `789,700 → 801,659` (`+1.51%`) | `2/5` / `5/5` (`14.8 cp`) | `21,596 → 0` | `7,209 → 8,352` (`+15.86%`) |
| `val_pgn.npz` EG (`7..13` 子) | 5 | `2,078,235 → 1,687,408` | `-18.81%` (`0.9958x`) | `1,550,485 → 1,465,878` (`-5.46%`) | `0/5` / `4/5` (`8.0 cp`) | `85,452 → 0` | `4,703 → 4,417` (`-6.08%`) |
| **總計 (`D14_ALL`)** | **40** | **`12,038,014 → 12,730,630`** | **`+5.75%`** (`1.0893x`) | **`851,886 → 830,209`** (`-2.54%`) | **`12/40` / `35/40` (`30.1 cp`)** | **`242,035 → 0`** | **`43,016 → 47,154` (`+9.62%`)** |

* **關鍵判讀**：
  * 舊版在 40 個 Depth-14 盤面合計產生 **`242,035` 筆 `depth=0, TT_FLAG_EXACT` 條目**（尤以殘局 `val_pgn_EG` 單組高達 `85,452` 筆、`VAL_EG_5000` 單盤 `51,764` 筆最甚）；修復後完全清零為 **`0`**，全庫 `depth >= 6` 深層 TT 保留量提升 **`+9.62%`**（STS 更達 `+42.11%`），`TT_hits` 由 `633,939` 增至 `651,798`。
  * 消除 QS 誤標 `EXACT`（把帶 Delta/SEE 剪枝的界限錯當精確值）引發的假截斷後，Depth-14 總節點僅微增 `+5.75%`（殘局反降 `-18.81%`、開局 `-3.28%`），並修正多盤明顯估值失真（如 `PUZ_200 +522 → +55`、`STS_360 +38 → -35`、`OPEN_6000 +30 → -6`、`PUZ_250 e8b8(+68) → h6e6(-14)`）。

### 4. 固定節點（`300k nodes`）實戰對戰驗證（`tournament_analysis/wave1b3_300k.pgn`，Wave 1B-3 vs Wave 1B-2）

* **882 局中段紀錄（同步決策點，`c464f50`）**：`452.5 - 429.5`（`51.30%`，`164W 577D 141L`，`+9.06 ± 13.15 Elo`，LOS `91.2%`，`SPRT[-10, 0] LLR = +3.1255` 觸發 `H1 Accepted`），當場歸檔 `chess_numba_20261124.bundle` 並同步 `classical_old/`。
* **2,000 局完整收斂結果（`tools/sprt_calc.py`，1,000 對成對換先）**：
  * **總戰績與得分**：**`New 998.0 - 1002.0 Old`**（**`49.90%`**，**`350W 1296D 354L`**，淨差 `-4` 局）。
  * **Pentanomial 五項分佈 `[LL, LD/DL, DD/WL, WD/DW, WW]`**：`[28, 229, 483, 239, 21]`（`WW - LL = -7` 對、`WD - LD = +10` 對）。
  * **相對 Elo 與 LOS**：Pentanomial **`-0.69 ± 8.77 Elo`**（95% CI **`[-9.47, +8.08]`**，**LOS = `43.8%`**；WDL `-0.69 ± 9.03`，LOS `44.0%`）。
  * **多區間 SPRT（Pentanomial）**：`[-12.00, +0.00] LLR = +3.1744`（**`H1 Accepted`**）、`[-10.00, +0.00] +2.1472`、`[-5.00, +0.00] +0.4504`、`[-3.00, +0.00] +0.1205`、`[-3.00, +3.00] -0.2081`、`[+0.00, +5.00] -0.7971`、`[+0.00, +10.00] -2.8406`。
* **對局結構與序列穩定性（`game_stats_report.txt`，2,000 局完整報告）**：
  * 和局率 `64.80%`、平均局數 `51.2` 回合（$\sigma = 21.5$）；持白 `53.45%`（`213W 643D 144L`）、持黑 `46.35%`（`137W 653D 210L`）。
  * 序列穩定性（1,000 對）：前半 500 對 `0.518`、後半 500 對 `0.480`；Lag-1 自相關 `0.0241`、游程檢定 `p = 0.2925`。
* **搜尋遙測對比（`search_stats_report.txt`，2,000 局樣本：`New` 101,910 步 vs `Old` 101,864 步）**：
  * 平均搜尋深度 `16.25`（$\sigma=3.40$）vs `16.23`（$\sigma=3.37$）（`+0.02`, Welch $t=+1.101, p=0.2707$；`depth >= 14`：`86.2%` vs `86.1%`）。
  * 每步平均節點 `183,383` vs `183,428`（`-0.0%`, $p=0.7192$）、每步耗時 `336.2 ms` vs `335.9 ms`（$p=0.3481$）、平均 NPS `606,495` vs `608,530`（`-0.3%`, $p=0.1041$）；同深度（`d = 12..24`）節點比穩定落於 **`0.995–1.002`**。
* **結論**：Bug #3（禁止 QS 寫 `TT_FLAG_EXACT`）與 Bug #8（無鎖 TT 競態與 `int16(-32768)` 溢位）在單執行緒 `300k nodes / 64 MB TT` 實戰下與 Wave 1B-2 持平無退化（`-0.69 ± 8.77 Elo`、`[-12, 0]` H1 通過），其正確性與 SMP 並發防護已正式納入 [`classical_old/`](../classical_old/) 基線，隨後推進 **Wave 1B-4（Bug #7）**。

---

## Wave 1B-4：修復 Bug #7 — 升變步於吃子與 QSearch 排序評分修正（2026-11-24）

> **對照基準**：Commit `c464f50`（`classical_old/` 已同步至 Wave 1B-3；本波維持 `classical_old/` 於 Wave 1B-3 不動，以便單軸驗證 Bug #7）。

### 1. 修改明細（[`search_heuristics.py`](search_heuristics.py)）

1. **`score_captures_lazy`（[#L329-L360](search_heuristics.py#L329-L360)）**：
   * 單次讀取 `special_flag = get_special_move_flag(move)` 並供吃過路兵與升變共用。
   * 當 `special_flag == SPECIAL_MOVE_FLAG_PROMOTION` 時，於 `mvv_lva` 加上 **`MG_MATERIAL_VALUES[get_promotion_piece(move) + 1] - MG_MATERIAL_VALUES[PAWN]`**（使無吃子升變依 `Q > R > B > N` 排序而非全部並列 `0` 分，吃子升變亦不再 `Q/R/B/N` 同分）。
2. **`score_captures_with_tt_lazy`（[#L362-L416](search_heuristics.py#L362-L416)）**：
   * 同樣為 `is_promotion = (special_flag == SPECIAL_MOVE_FLAG_PROMOTION)` 加上升變價值增量（`MG_MATERIAL_VALUES[promo + 1] - MG_MATERIAL_VALUES[PAWN]`）。
   * 新增 **`elif is_promotion: base_score += QS_EVASION_CAPTURE_BUCKET`**（位於 `if victim_rel != -1:` 與 `elif aggressor_type >= 0:` 之間），使無吃子升變不再落入安靜逃將分支（原本會誤吃 `pawn_history` 負分而排在虧損吃子 `QxP` 之後），同時完整保留既有安靜逃將（`get_quiet_stat_score` + `QS_EVASION_QUIET_CLAMP`）契約。

### 2. 驗證與診斷結果

* **單元與契約測試**（清除 `*search*` / `*transposition_table*` 過期 `.nbi/.nbc` 後冷編譯重跑 `tests.test_wave1b_fixes tests.test_search_core_contracts tests.test_lazy_smp tests.test_wave1a_fixes tests.test_history`）：**84 / 84 項全過**（`16.504s`）。
  * 於 [`tests/test_wave1b_fixes.py`](../../tests/test_wave1b_fixes.py) 新增 `TestBug7PromotionScoring`（2 項）：驗證 `score_captures_lazy` 的無吃子／吃子升變嚴格滿足 `Q > R > B > N > 0`，以及 `score_captures_with_tt_lazy` 中無吃子變后嚴格高於 `QS_EVASION_CAPTURE_BUCKET`、虧損吃子 `QxP` 與安靜逃將，且升變子類滿足 `Q > R > B > N`。
* **Numba 快取與符號檢查**：`warm_import = 7.11s`、`warmup_search_jit = 0.33s`；`_search` / `quiescence_search` / `_lazy_smp_helper_loop_jit` 各 **1** 過載、`unresolved: []`（`0` 個 `.numba.unresolved$`）。
* **基準差異與衛生**：`python tools/sync_classical_old.py --diff` 確認僅 `search_heuristics.py` 上述修復；`CRCRLF = 0`。
* **Depth-10 對照基準（`classical_old` Wave 1B-3 vs `classical` Wave 1B-4）**：
  * **20 個標準盤面**：總節點 `1,080,177 → 1,079,934`（**`-0.02%`**），`0 / 20` 變著、`0 / 20` 變分；Weighted NPS `643,139 → 654,694`（無額外熱路徑開銷）。
  * **5 個升變專題盤面**：總節點 `216,505 → 201,459`（**`-6.95%`**），其中將軍下含無吃子升變的 `Promo_QSEvasion`（`3rk3/P7/4p3/3p4/4Q3/8/8/3K4 w - - 0 1`）節點數從 `80,920 → 64,599`（**`-20.17%`**），Depth-10 評估從 `1258 → 11385`（更早搜出 `VALUE_KNOWN_WIN` 升變線路）。

### 3. Colab 固定節點（`400k nodes`）實戰對戰最終紀錄（[`tournament_colab_runner_se_multicut copy.ipynb`](../../tournament_colab_runner_se_multicut%20copy.ipynb)，Commit `754255c` vs `c464f50`）

* **實驗配置**：`--nodes 400000`、`--concurrency 7`、`--tt-mb 64`、嚴格成對換先開局（`data/openings.epd`），跑至第 **3,090 局 / 1,545 對**終止。
* **總戰績與統計檢定**：
  * **得分**：**`New 1550.0 - 1540.0 Old`**（**`548W 2004D 538L`**，勝率 **`50.16%`**，淨勝 **`+10` 局**，和局率 `64.85%`）。
  * **成對九宮格與 Pentanomial `[LL, LD/DL, DD/WL, WD/DW, WW]`**：`[39, 348, 761, 358, 39]`（九宮格：`DD:649, WD:205, DL:202, DW:153, LD:146, WL:74, LL:39, WW:39, LW:38`；`WW - LL = 0` 對、`WD - LD = +10` 對）。
  * **相對 Elo 與 LOS**：Pentanomial **`+1.12 ± 7.26 Elo`**（95% CI **`[-5.91, +8.16]`**，**LOS = `62.3%`**）。
  * **多區間 SPRT（`tools.match_core.SPRTTest`）**：
    * **`SPRT[-10, 0]`**：**`LLR = +4.76`**（**`H1 Accepted` — 確認無退化**）
    * `SPRT[-5, 0]`：**`+1.41`** | `SPRT[-3, 0]`：**`+0.61`** | `SPRT[-1.75, +0.25]`：**`+0.29`** | `SPRT[-3, +3]`：**`+0.52`**
    * `SPRT[0, +5]`：`-0.53` | `SPRT[0, +10]`：**`-3.01`**（`H0 Accepted` — 因升變局佔比有限：整體對局和局平均升變僅 `0.05` 次、決勝局 `0.14–0.17` 次）
* **序列穩定性與執色分析**：
  * 前半 772 對 `0.504`、後半 773 對 `0.499`（Lag-1 自相關 `0.0196`、游程檢定 `p = 0.5950`）；持白 `53.07%`（`318W 1004D 223L`）、持黑 `47.25%`（`230W 1000D 315L`）。
* **搜尋遙測對比（`New` 155,101 步 vs `Old` 155,128 步）**：
  * 平均深度 `16.99` vs `16.98`（`p = 0.3591`）、每步節點 `243,571` vs `243,373`（`+0.1%`, `p = 0.1396`）、耗時 `870.9 ms` vs `887.5 ms`、平均 NPS `322,514` vs `315,949`（**`+2.1%`**）；同深度（`d=12..24`）節點比 `0.998–1.005`。
* **結論**：正式通過 **`SPRT[-10, 0]`（`LLR = +4.76`）無退化驗收**，先前同步 [`classical_old/`](../classical_old/) 至 Commit `f7d271a` 的基準決策完全成立。

---

## Wave 1B-5：修復 Bug #9 — PV 節點零窗口 Fail-High 強制全窗口驗證（2026-11-24）— ❌ 已回退（REVERTED）

> **對照基準**：Commit `f7d271a`（`classical_old/` 已同步至 Wave 1B-4；本波維持 `classical_old/` 於 Wave 1B-4 不動，以便單軸驗證 Bug #9）。

### 1. 修改明細（[`search.py`](search.py) `_search` 兩處）

1. **非 LMR / 首段零窗口 Fail-High 分支（[#L2979-L2983](search.py#L2979-L2983)）**：
   * 移除 `and evaluation < beta`，將 `do_full_pv_search = is_pv and evaluation < beta` 改為 **`do_full_pv_search = is_pv`**（對齊 SF19 `search.cpp:1409` `if (PvNode && (moveCount == 1 || value > alpha))`）。
2. **LMR 重搜後全窗口判斷（[#L3044](search.py#L3044)）**：
   * 移除 `and evaluation < beta`，將原本的 `if is_pv and evaluation > alpha and evaluation < beta: ...` 收斂為 **`do_full_pv_search = is_pv and evaluation > alpha`**。
* **修復原理**：第 2 步以後的非 PV 零窗口搜尋允許 RFP / NMP / ProbCut / LMP / FP / SEE / TT-cut 等激進剪枝，其 fail-soft 回傳的 `evaluation >= beta` 可能帶有不精確界限或缺乏真實 `pv_table[ply+1]` 主變；過去因 `evaluation < beta` 跳過 `is_pv=True` 全窗口驗證，會直接讓未驗證的 fail-soft 高分決定 PV 截斷與根著法排序。

### 2. 驗證與診斷結果

* **單元與契約測試**（清除 `*search*` 舊 `.nbi/.nbc` 後冷編譯並執行 `tests.test_wave1b_fixes tests.test_search_core_contracts tests.test_lazy_smp tests.test_wave1a_fixes tests.test_history`）：**86 / 86 項全過**（熱跑 `5.278s`）。
  * 於 [`tests/test_wave1b_fixes.py`](../../tests/test_wave1b_fixes.py) 新增 `TestBug9PVFailHighFullWindowVerification`（2 項）：契約驗證兩處不再出現 `evaluation < beta`；於對稱兵殘局（`4k3/4p3/8/8/8/8/4P3/4K3 w - - 0 1`）對非首步子節點植入 `TT_FLAG_ALPHA (-5000, depth=10)` 誘發零窗口 fail-soft `+5000 >= beta (50)`，實測 `depth=3`（無 LMR）與 `depth=5`（含 LMR）皆確實觸發 `is_pv=True` 全窗口重搜驗證，不再回傳未驗證的 `5000`。
* **Numba 快取與符號檢查**：`warm_import = 8.41s`、`warmup_search_jit = 0.27s`；`_search` / `quiescence_search` / `_lazy_smp_helper_loop_jit` 各維持 **`1`** 過載、`unresolved: []`、`unresolved_llvm: []`。
* **基準差異與衛生**：`python tools/sync_classical_old.py --diff-code` 僅顯示 `search.py` 上述 Bug #9 修復；`CRCRLF = 0`。
* **20 盤面 Depth-10 對照（`classical_old` Wave 1B-4 vs `classical` Wave 1B-5，64 MB TT）**：
  * **總節點數**：`1,080,177 → 990,157`（**`-8.33%`**，顯著收斂且無節點膨脹）。
  * **Weighted NPS**：`362,590 → 451,323`（無異常額外負擔）。
  * **著法／評分變化**：`7 / 20` 盤面改變最佳著法、`17 / 20` 盤面改變評分；`Tactical_MG`（`588` nodes）與 `Startpos` 著法維持不變，中盤多處高節點盤面大幅收斂（如 `EPD_14001 -73.94%`, `EPD_28001 -71.59%`, `EPD_1 -51.12%`）。

### 3. Colab 固定節點（`400k nodes`）實戰對戰驗證（[`tournament_colab_runner_se_multicut.ipynb`](../../tournament_colab_runner_se_multicut.ipynb)，Commit `62473d5` vs `f7d271a`）

* **實驗配置**：`--nodes 400000`、`--concurrency 7`、`--tt-mb 64`、嚴格成對換先開局（`data/openings.epd`），於第 **1,178 局 / 589 對**觸發 `SPRT[0, +10]` 下界早停。
* **總戰績與統計檢定**：
  * **得分**：**`New 578.5 - 599.5 Old`**（**`200W 757D 221L`**，勝率 **`49.11%`**，淨負 **`-21` 局**，和局率 `64.26%`）。
  * **成對九宮格與 Pentanomial `[LL, LD/DL, DD/WL, WD/DW, WW]`**：`[24, 132, 295, 117, 21]`（九宮格：`DD:254, DL:71, WD:70, LD:61, DW:47, WL:25, LL:24, WW:21, LW:16`；`WW - LL = -3` 對、`WD - LD = -15` 對）。
  * **相對 Elo 與 LOS**：Pentanomial **`-6.19 ± 11.86 Elo`**（95% CI **`[-18.18, +5.78]`**，**LOS = `15.5%`**）。
  * **多區間 SPRT（`tools.match_core.SPRTTest`）**：`[0, +10] LLR = -3.00`（**`H0 Accepted` 拒絕**）、`[0, +5] -1.17`、`[-3, +3] -1.00`、`[-5, 0] -0.50`、`[-3, 0] -0.38`、`[-10, 0] -0.32`。
* **序列穩定性與遙測（`New` 59,890 步 vs `Old` 59,872 步）**：
  * 前半 294 對 `0.494`、後半 295 對 `0.488`（前後雙半皆負，Lag-1 `-0.0274`、游程檢定 `p = 0.4433`）。
  * 平均深度 `16.94` vs `16.94`（`p = 0.8376`）、每步節點 `242,881` vs `243,531`（`-0.3%`）、NPS `313,137` vs `312,002`（`+0.4%`）；同深度（`d=13..21`）節點比 `0.994–0.999`。
* **初步技術診斷與處置建議**：
  * 與 Wave 1B-4 不同，Wave 1B-5 屬於全局每次搜尋都會觸發的 PV 重搜機制，卻在固定節點對戰呈現前後雙半均劣於基線（`LOS = 15.5%`, `-6.19 ± 11.86 Elo`, `SPRT[-3, +3] = -1.00`）。
  * 主因研判：(1) 在根節點 `ply == 0` 外層已有 Aspiration Window 迴圈，非首步在窄視窗 Fail-High（`>= beta`）時強制先用原窄視窗跑一次 `is_pv=True` 全窗口重搜（依 H9 關閉 TT 分數截斷＋全套前向剪枝）再於外層放寬視窗重搜，會形成雙重重複搜尋消耗固定節點預算；(2) 本機在 LMR 已先經 `adjusted_depth` 非 PV 驗證達 `>= beta` 後再啟重型 `is_pv=True` 全窗口驗證，於固定節點預算下投資報酬率為負。
  * **處置結果（❌ 已回退／REVERTED，方案 A）**：依用戶決議採取方案 A，已將 [`search.py`](search.py) 與 [`tests/test_wave1b_fixes.py`](../../tests/test_wave1b_fixes.py) 中的 Bug #9 修改全數回退並還原至已驗收之 Wave 1B-4 基線（[`classical_old/`](../classical_old/) Commit `f7d271a`），同時完整保留本節 1,178 局 `-6.19 ± 11.86 Elo` 實測與診斷紀錄以免重蹈退化覆轍，隨後直接推進 **Wave 1B-6（Bug #10）**。
