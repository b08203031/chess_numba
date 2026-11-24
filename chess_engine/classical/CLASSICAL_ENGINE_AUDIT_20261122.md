# Classical HCE 引擎深度工程審計報告 (2026-11-22)

> **文檔屬性**：`chess_engine/classical/` 全模組靜態審計、交叉反證與架構演進藍圖（Read-Only Audit + Adversarial Cross-Check）  
> **基準狀態**：Commit `d015345`（Phase 14 Match-SPSA `material` + Lazy SMP E148；與 `classical_old/` 同步）  
> **審計範圍**：`chess_engine/classical/` 全 **20** 支 `.py` 原始碼（不含 `nnue/`、`classical_old/`；本次暫略過 `time_manager.py`）  
> **對照參考**：搜尋架構對照 `stockfish_repo/src/`（SF19）；手工評估與殘局對照 `stockfish_11/src/`（SF11）  
> **既有保留約束**：保留全部 SPSA `tune[]`（44 軸）、對局 HCE `eval_weights`（99 軸）、Texel 調參鉤子與 `DIAG_*` 診斷計數器；僅檢舉其中真實 BUG 或非調參用途之死碼／熱路徑冗餘。  
> **相關紀錄**：後續修復與實測數據見 [`CLASSICAL_OPTIMIZATION_EXP_LOG.md`](CLASSICAL_OPTIMIZATION_EXP_LOG.md)；歷史評估審計見 [`EVAL_AUDIT_AND_FIX_PLAN_20261031.md`](EVAL_AUDIT_AND_FIX_PLAN_20261031.md)；搜尋單項實驗史見 [`SEARCH_EXP_LOG.md`](../../tournament_analysis/SEARCH_EXP_LOG.md)。

---

## 執行摘要 (Executive Summary)

本次審計將 `chess_engine/classical/` 拆為四大領域（搜尋核心 `search.py`、搜尋基礎設施 `transposition_table.py` / `search_heuristics.py` / `engine_types.py`、棋盤與著法產生器 `move_generator.py` / `board_operations.py` / `see.py` / `zobrist.py` / `bitboard_utils.py` / `move.py` / `fen_parser.py` / `opening_book.py` / `core.py` / `debug_utils.py`、手工評估與殘局 `evaluation.py` / `pawns.py` / `endgame.py` / `material.py` / `constants.py`）平行清查，並逐行複核與全倉 grep 驗證，最終確認：

1. **已確認 BUG 共 31 項**（**P0 級 6 項、P1 級 12 項、P2 級 13 項**）：
   * 最嚴重的系統性缺陷包括：**負向 Correction History 100% 死碼**（導致對行棋方評分單調正向偏誤）、**`lazy=True` 粗評（僅 Material+PST+Imbalance）寫入 `TT.static_eval` 並被 QSearch 直接當作 `stand_pat` 使用**（實測誤差可達 `370 cp`）、**QSearch 以 `depth=0, TT_FLAG_EXACT` 無條件洗掉深層主搜尋 TT 條目**、`MAX_PLY (128)` 邊界堆積越界寫入、Polyglot 開局庫易位輸出非法 UCI 著法 `e1h1`、以及 `mate_kbnk` 送子被吃幻覺為將死（`+29871`）。
2. **純加速（零搜尋樹／零評估變動）與死碼清理**：
   * 識別出 **11 項熱路徑純 NPS 優化**（含延遲計算 `check_sq_*` / `get_blockers_for_king` / `pinned_them` 與僅於 `lmr > 0` 才算 `is_giving_check_after_move`、`pawn_table_*` 14 條 SoA 合併為 64B 單 Cache-Line AoS、以 JIT 整數除法取代每步 `17.1 ms` `np.divide` float64 暫存配置、Magic Bitboard 射線前置過濾、恢復 `_see_ge_jit` 底線 O(1) 早退、外提 `score_quiets` 迴圈不變量、消除無鎖 TT `else` 分支空寫、收斂 Numba `.nbc` 多重特化風暴等）及 **500+ 行已驗證 0 呼叫者死碼／冗餘常數**。
3. **高 ROI 棋力提升提案（16 項）**：
   * 嚴格剔除 NNUE 專屬機制、時間管理以及歷史上已實戰否決項目（如 P4 post-LMR continuation history、E82 QSearch Correction History、E41 FP `max_eval=futility` 等），保留 **12 項搜尋／排序強化（`S-1`～`S-12`）**、**4 項 HCE／評估↔搜尋介面強化（`E-1`～`E-4`）**，以及 **3 項純限時 NPS 架構升級（`T-1`～`T-3`）**。

---

## 一、已確認 BUG 清單（共 31 項）

### 🔴 P0 — 關鍵正確性／記憶體安全／嚴重搜尋汙染（6 項）

#### Bug #1：負向 Correction History 100% 死碼，導致評分單調正向偏誤（以及 Fail-Low TT `best_move` 寫入）
* **位置**：[`search.py:3036-3041, 3389-3407`](search.py#L3389-L3407)、[`search.py:3489`](search.py#L3489) → [`transposition_table.py:280-282`](transposition_table.py#L280-L282)
* **根因與數學推導**：
  * 著法迴圈起始時 `max_eval = -INFINITY (-32000)`（`L2094`）。Step 14 全部 7 個剪枝點都受 `not is_loss(max_eval)`（`L2188`）保護，故第 1 個合法著法永不被剪枝。
  * 在 `ply > 0` 時，只要合法著法數 $\ge 1$，第 1 步必有 `evaluation > -32000`，且 `L3037` 無條件執行 `max_eval, best_move = np.int32(evaluation), move`（不像 `ply == 0` 的 `L3030` 或 SF19 `search.cpp:1518-1520` 要求 `evaluation > alpha`）。因此就算整個節點 Fail-Low，`best_move` 仍必為非 `NO_MOVE`。
  * 結果導致 `L3400` 的 `has_best_move = (best_move != NO_MOVE)` **在 100% 節點恆為 `True`**，產生四重連鎖破壞：
    1. Fail-Low 負向校正（`max_eval < static_score`）從未觸發，`CORRECTION_BONUS_NO_MOVE_DIV = 8`（`L3406`）完全死碼；
    2. Fail-Low 時若 `static_score < max_eval <= original_alpha`，反而在未證實的上界節點錯誤寫入正向校正；
    3. `non_pawn_correction_history_white`、`non_pawn_correction_history_black` 與 `continuation_correction_history` 永遠只能寫入正數，形成對行棋方（STM）的單調正向評分膨脹；
    4. Fail-Low 將非零 `best_move` 傳入 `store_tt`（`L3489`），削弱 E105 保護既有 TT 著法的機制（`transposition_table.py:281`）。
* **建議修法（拆為兩次獨立 SPRT）**：
  * **Part A（必先獨立修復）**：在 `L3390` 前計算 `has_best_move = (final_flag != TT_FLAG_ALPHA and best_move != NO_MOVE)` 並將 `L3390` 的 `is_capture_or_promo` 檢查包入 `if has_best_move:`（注意首次啟用負向校正後，可再檢驗校正除數，參考 E79）。
  * **Part B（獨立開 SPRT，勿與 Part A 混包）**：`L3037` 僅在 `evaluation > alpha` 時更新 `best_move` 與 `pv_table`（SF19 `search.cpp:1518-1524`）。因 E73/E80 曾顯示 Fail-Low `best_move` 會與 `!tt_hit` IIR 及迭代快取連動，故必須單軸驗證。

#### Bug #2：`lazy=True` 粗評（僅 Material+PST+Imbalance）寫入 `TT.static_eval`，被 QSearch 當作葉節點 `stand_pat`
* **位置**：[`search.py:1487, 1496-1499, 3479-3481, 608-610`](search.py#L1487-L1501)、[`evaluation.py:1699-1711`](evaluation.py#L1699-L1711)
* **根因與反例**：
  * `_search` 在 TT 未命中時呼叫 `_evaluate_position_jit(..., lazy=True)`（`L1499`），僅計算 Material + PST + Imbalance（不含兵型、通路兵、王安全、機動性、威脅、空間、主動權及 `+19` Tempo）。
  * 在 PV 節點及 `depth > 8` 節點（RFP/Razoring/NMP/FP 未觸發完整評估升級），`static_eval_is_full` 維持 `False`，但 `L3480-3481` 仍把該 `raw_static_eval` 寫入 `TT.static_eval`（違背 `SEARCH_EVAL_INTERFACE.md §5 P4`），`L3382` 也拿它更新 Correction History。
  * 當 `quiescence_search` 隨後命中該 TT 條目並進入葉節點（`L608-609`），便直接採用 `qs_tt_static_eval` 作為 `stand_pat`（甚至在無吃子時存為 `TT_FLAG_EXACT`）。
  * **實測反例**：局面 `r4rk1/ppp3p1/2n4p/4p1Nq/2B2Q2/8/PPP3PP/R4R1K w - - 0 1` 下，完整 HCE 為 **`+702`**，而 `lazy=True` 為 **`+332`**，單步靜態誤差高達 **`370 cp`**！同時也污染 `static_eval_stack`（混用有無 Tempo 與完整特徵的分數），扭曲 `improving`、`opponent_worsening`、hindsight `depth±1`、ProbCut SEE 與 Capture FP。
* **建議修法**：`search.py:3480` 僅在 `static_eval_is_full` 為真時寫入 `TT.static_eval`（否則存 `32767`）；同步視情況將 `L3382` 校正歷史更新限制於 `static_eval_is_full`。一旦保證 TT 僅存完整評估，`L1496` 命中有效 `tt_static_eval_raw != 32767` 時即可安全設 `static_eval_is_full = True`，避免 RFP/NMP/Razoring/FP 重算完整 HCE。

#### Bug #3：`quiescence_search` 寫入 `depth=0, TT_FLAG_EXACT`，無條件覆寫深層主搜尋 TT 條目
* **位置**：[`search.py:635-640, 1008, 1013`](search.py#L635-L640) → [`transposition_table.py:286-287`](transposition_table.py#L286-L287)
* **根因與反例**：
  * `store_tt:286-287` 設定 `if flag == TT_FLAG_EXACT: replace = True`。
  * 當 QSearch 碰到無吃子葉節點（`move_count == 0`, `L639`）或因 `stand_pat` 抬升過 `alpha` 後所有吃子 Fail-Low（`L632` → `L1008`）時，直接以 `depth=0, TT_FLAG_EXACT, is_pv=False` 呼叫 `store_tt`，**無條件覆寫主搜尋留下的 `depth=12+` 深層 TT 條目**（且會把已套用 Delta / SEE 剪枝的 QS 分數錯標成 `EXACT`）。對照 SF19 `search.cpp:1876-1878`，QS 永遠只寫 `BOUND_LOWER` 或 `BOUND_UPPER`，永不寫 `BOUND_EXACT`。
* **建議修法**：在 `quiescence_search` 中絕不存入 `TT_FLAG_EXACT`（`L639` 改存 `TT_FLAG_ALPHA`；`L1008` 改依 `best_eval >= beta` 存 `TT_FLAG_BETA` 或 `TT_FLAG_ALPHA`；保留 `qs_tt_is_pv`；`L630` 僅於 `qs_tt_flag == TT_FLAG_NONE` 時寫入）；並在 `transposition_table.py:286` 加守衛 `depth > 0 or existing_entry['depth'] == 0`。

#### Bug #4：`boundscheck=False` 下 `ply == MAX_PLY (128)` 與 `ply == 127` 堆積緩衝區越界讀寫
* **位置**：[`search.py:1234`](search.py#L1234)（在 `L1247` 守衛之前）及 7 處未防護的 `[ply + 1]` 存取（[`L1748, 1969, 2758, 2833, 2886, 2941, 2991`](search.py#L2833)）
* **根因**：`follow_pv_stack`、`reduction_stack` 與 `cutoff_cnt` 長度皆為 `MAX_PLY = 128`。當 `_search` 在 `ply == 128` 進入時，`L1234` 先執行 `follow_pv_stack[128] = False` 才到 `L1247` 的 `if ply >= MAX_PLY:`；而在 `ply == 127` 時，6 處 `reduction_stack[128]` 寫入與 `L2833` 的 `cutoff_cnt[128]` 讀取全數越界（因 `@njit(boundscheck=False)` 直接覆寫相鄰記憶體）。
* **建議修法**：將 `if ply >= MAX_PLY - 1:` 移至 `L1215` 入口之前，並可將堆疊陣列配置加寬至 `MAX_PLY + 2`。

#### Bug #5：Polyglot 開局庫易位輸出非法 UCI 著法（`e1h1`/`e1a1`/`e8h8`/`e8a8`）且漏標易位／吃過路兵旗標
* **位置**：[`opening_book.py:74-114`](opening_book.py#L74-L114) → [`main.py:321`](../../main.py#L321)
* **實測重現（已於 `chess_engine/polyglot.bin` 驗證）**：
  * 西班牙開局（`1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6`）：`book.lookup()` 僅回傳 `[('e1h1', 2)]`，導致 UCI（`OwnBook=True`）直接輸出 `bestmove e1h1` 遭判非法著敗北；義大利開局（`1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5`）亦回傳 `('e1h1', 46)`。（註：`tools/tournament.py` 預設關閉 OwnBook，故未反映在過去 SPRT。）
* **根因**：Polyglot 格式以王吃車（`e1h1`/`e1a1`/`e8h8`/`e8a8`）編碼易位，而 `_convert_polyglot_move_to_engine_move(poly_move, board_state)` 忽略已傳入的 `board_state`，直接以 `SPECIAL_MOVE_FLAG_NORMAL` 與車格編碼，也未對斜進至 `ep_sq` 的兵標記 `SPECIAL_MOVE_FLAG_EN_PASSANT`。
* **建議修法**：利用 `board_state` 判斷王移動 `e1→h1/a1` / `e8→h8/a8` 時映射為 `g1/c1` / `g8/c8` 並設 `SPECIAL_MOVE_FLAG_CASTLING`；兵斜進至 `game_state[2]` 時設 `SPECIAL_MOVE_FLAG_EN_PASSANT`。

#### Bug #6：`mate_kbnk` 三重邏輯錯誤：無保護象被吃和棋幻覺為將死（`+29871`）、己方行棋幻覺逼和（`0`）、漏算主教 X-Ray
* **位置**：[`endgame.py:202-225, 530, 534`](endgame.py#L202-L225)
* **實測反例**：
  1. `k7/B1K5/1N6/8/8/8/8/8 b - - 0 1`（`Nb6` 將軍 `Ka8`，但 `Ba7` 無保護，`1... Kxa7` 立刻變 KN vs K 理論和棋）：`L215` 使用 `~strong_pieces` 把 `a7` 從逃生格排除，導致 `safe_sq_count == 0 and is_check` 誤判為將死並回傳 **`+29871`**！
  2. `k7/2K5/2N5/8/8/8/8/6B1 w - - 0 1`（白棋走且必勝）：未傳入 `side_to_move`，`L220` 見弱王 0 安全格且未被將軍即回傳 **`0`**（逼和幻覺）。
  3. `8/1k6/2BN4/1K6/8/8/8/8 b - - 0 1`：`L209` 計算 `get_bishop_attacks(bishop_sq, occ_all)` 未扣除 `BB_SQUARES[weak_king_sq]`，使主教背後非法格 `a8` 未被標記為受攻。
* **建議修法**：傳入 `side_to_move` 與 `strong_side`；僅在 `side_to_move != strong_side` 時才執行逼和／將死捷徑；`L215` 改用 `~BB_SQUARES[strong_king_sq]`；`L209` 改以 `occ_all ^ BB_SQUARES[weak_king_sq]` 算象攻擊。

---

### 🟠 P1 — 高影響搜尋／並發／排序／雜湊／殘局 BUG（12 項）

7. **升變步於 `score_captures_lazy` 與 `score_captures_with_tt_lazy` 嚴重誤評（[`search_heuristics.py:326-399`](search_heuristics.py#L326-L399)）**
   * 無吃子變后（`a7a8=Q`）目標格無敵子（`victim_rel == -1`），在 `score_captures_with_tt_lazy` 跳過 `QS_EVASION_CAPTURE_BUCKET (+20000)` 掉入安靜逃將分支，吃到尚未更新的 `pawn_history = -1238`（約 `-2476/-1238`），導致免費升變后在 QSearch 排在所有虧損吃子（`QxP ≈ +13921`）之後！同時在 `score_captures_lazy` 中，無吃子升變為 `0` 分，且吃子升變 `Q/R/B/N` 全數同分，在不穩定選擇排序下（`search.py:798-805, 1916-1923`）經常把升變車／象／馬排在皇后前面。
   * **修法**：檢查 `SPECIAL_MOVE_FLAG_PROMOTION` 並加上升變價值增量（`MG[promo] - MG[PAWN]`），且在 `score_captures_with_tt_lazy` 納入吃子 bucket（建議與提案 `S-2` 協同）。

8. **無鎖 Lazy SMP 的 TT Read-Modify-Write 競態與 `probe_tt_best_move` TOCTOU（[`transposition_table.py:133-149, 245-248, 300-335`](transposition_table.py#L245-L335)）**
   * (a) `store_tt` 的 `else:`（`L324-334`）與 `penalize_tt`（`L138-148`）未先驗證舊 checksum 即重讀現場記憶體欄位重算 checksum 回寫，在 4T Lazy SMP 下可把平行執行緒寫了一半的撕裂條目「封成合法 checksum」，或將碰撞替換的新 `keyB` 條目用舊 `keyA` 覆寫 `padding` 毀掉；(b) `replace=True` 在 `static_eval == 32767` 時用 `existing_entry['static_eval']` 算進 `chk` 卻不回寫 `tt[idx]['static_eval']`；(c) `probe_tt_best_move` 在 Helper 尚未 join 前（`search.py:5133`）仍經由回傳可變記錄指標的 `probe_tt` 讀取欄位（TOCTOU）；(d) `store_tt` 的 `else:` 算完 `final_best_move`（`L280-282`）卻直接丟棄未存。（另 `L322` `abs(int16(-32768))` 在 Numba 會溢位。）
   * **修法**：在 `store_tt` `else:` 與 `penalize_tt` 先完整快照並驗算舊 checksum 再修改，無變動時完全不寫；`replace=True` 無條件寫 `static_eval`；`else:` 於合理條件保存非零 `best_move`；`probe_tt_best_move` 直接做純量快照（同時維持 `test_search_core_contracts.py:583-589` 測試契約）。

9. **PV 節點零窗口 Fail-High 跳過 `is_pv=True` 全窗口驗證（[`search.py:2922, 2984`](search.py#L2922-L2984)）**
   * 兩處皆多寫 `and evaluation < beta`。當第 2 步以後在非 PV 零窗口（啟用 RFP/NMP/ProbCut/LMP/FP/SEE/TT-cut）出現 fail-soft `evaluation >= beta` 時，會跳過 `is_pv=True` 全窗口重搜驗證並截斷 `pv_table[ply+1]`。對齊 SF19 `search.cpp:1409`，應移除 `and evaluation < beta`。

10. **ProbCut 遺留舊 `move_is_capture_stack[ply]`（以吃子污染安靜歷史）且先呼 QS 再寫堆疊（[`search.py:1947-1976`](search.py#L1947-L1976)）**
    * `L1967-1968` 未重設 `move_is_capture_stack[ply] = True` 與 `move_count_stack[ply] = 0`，殘留 `False` 會導致子節點 fail-low（`L3330`）與 fail-high（`L3151`）把 ProbCut 吃子寫進全部 quiet history 表；且 `L1947` 在更新 `move_stack[ply]`/`piece_stack[ply]` 前即先呼叫 QS。
    * **修法**：於 `L1947` 與 `L1970` 前後完整備份、設定並還原上述四個堆疊欄位。

11. **Lazy SMP 根節點首著遭 Helper 劫持 + 根層完成時 Fail-Low `(score, move)` 錯配（[`search.py:1322-1324, 3007-3034, 4699-4705`](search.py#L1322-L1324)）**
    * SMP 下 `L1323` 僅在 `tt_move == NO_MOVE` 時取用 `last_iteration_pv[0]`。若 Helper 將另一著法 `B` 寫入共享根 TT，Thread 0 會先搜 `B`；若 `B` 虧損 `-400 <= alpha` 後於第 2 步超時，因 `-400 > -10000` 不觸發 `aborted_loss_search` 回滾，直接送出漏著 `B`。此外在完整結束的根層 Aspiration Fail-Low 若 Move 1=`A (-500)`、Move 2=`B (+55 <= alpha)`，`L3028-3030` 會更新 `max_eval=+55` 卻不更新 `best_move`，回傳錯配 `(+55, A)`。
    * **修法**：對齊 SF19 `search.cpp:820`，`ply == 0 and last_iteration_pv_len > 0` 時無條件優先使用 `last_iteration_pv[0]`；並防護中斷 fail-low 候選覆寫。

12. **QSearch 缺少重複局面偵測（`_is_repetition_jit`）、`ply_path_stack[ply]` 與 `move_stack`/`piece_stack` 更新（[`search.py:501-561, 961-978`](search.py#L501-L561)）**
    * 被將軍進入 QS 可在重複逃將循環下撐滿 `q_ply == 10` 後回傳靜態分而非和棋 `0`（違背 `search-constraints.md §3`）；且 `q_ply >= 1` 逃將排序（`search_heuristics.py:389`）讀到跨分支殘留的 `move_stack`/`piece_stack`。
    * **修法**：`ply > 0 and halfmove_clock_qs >= 2` 時更新 `ply_path_stack[ply]` 並呼叫 `_is_repetition_jit`；`L974` 前後保存／更新／還原 `move_stack[ply]` 與 `piece_stack[ply]`。（**嚴禁**於 QS 啟用 Correction History，E82 已確認 `-204.75 Elo`。）

13. **無條件設定 En Passant 格與雜湊鍵，破壞 `polyglot.bin`、TT 轉置與 FIDE 三次重複（[`board_operations.py:240-251`](board_operations.py#L240-L251)、[`zobrist.py:139-141`](zobrist.py#L139-L141)、[`fen_parser.py:64-66`](fen_parser.py#L64-L66)）**
    * 實測開局庫重現：`1. e4` 後無黑兵在 `d4`/`f4` 卻仍設 `ep=20` 並 XOR `EN_PASSANT_FILE_KEYS[4]`，造成 `1. e4`、`1. e4 e5`、`1. e4 c5` 100% 錯失 `polyglot.bin`；亦造成 `1. e4 e5 2. Nf3 Nf6 3. Ng1 Ng8` 與 `1. e4 e5` 雜湊不同，漏判內部二次與三次重複以及 TT 轉置。
    * **修法**：僅當相鄰格確實存在敵兵可吃過路兵時才設定 `ep_sq` 與 XOR `EN_PASSANT_FILE_KEYS`（同步對齊 `make_move`、`compute_initial_hash` 與 `parse_fen`；注意須一併核對 Zobrist/FEN 單元測試）。

14. **`KBPsK` b/g 直線堡壘未檢查弱方兵（[`endgame.py:677-701, 750-773`](endgame.py#L677-L701)）**
    * 漏檢查 SF11 `endgame.cpp:363` 的 `!(pos.pieces(PAWN) & ~file_bb(pawnsFile))`。反例 `1k6/1p6/1P6/4p3/8/3B4/8/4K3 w - - 0 1` 誤判 `sf = 0`（`eval = 19`）。應檢查 `(piece_bbs[0] | piece_bbs[6]) & ~FILE_MASKS[1/6] == 0`。

15. **非 SF 的 `KRPKN` 縮放規則誤將必勝車兵對馬殘局歸零（[`endgame.py:1058-1079`](endgame.py#L1058-L1079)）**
    * 只要弱王與弱馬靠近 `p+8` 即回傳 `sf = 0`，連黑馬白送（`8/4k3/4P3/3nK3/8/8/8/4R3 w - - 0 1`）或被牽制（`8/4k3/R3n3/4P3/4K3/8/8/8 w - - 0 1`）都被判和棋。SF11 無此規則且 0 測試依賴，應移除。

16. **`_eval_kxk` 缺孤王逼和守衛，QSearch 對逼和局面回傳 `±11515`（[`endgame.py:269-314`](endgame.py#L269-L314) ↔ [`search.py:611-640`](search.py#L611-L640)）**
    * `_eval_kxk` 忽略 `side_to_move`（對照 SF11 `endgame.cpp:124-125`）。逼和 `k7/2Q5/1K6/8/8/8/8/8 b - - 0 1` 回傳 `-11515`，非將軍 QSearch 0 吃子後直接回傳 `-11515`，誘使 depth-1 走逼和 `Qc7??`。輪弱方走時須檢查孤王是否至少有一未受攻鄰格（扣除 `wk`），否則回 `0`。

17. **`SearchContext` 靜默降級 `eval_weights = np.zeros(64)` 致 `boundscheck=False` 越界讀取 `64..98`（[`engine_types.py:789-794`](engine_types.py#L789-L794)）**
    * `evaluation.py:1118-1359` 在 `boundscheck=False` 下讀至索引 `98`；應將長度 128 的 `default_eval_weights` 內建至 `chess_engine/classical/` 或移除錯誤的 64 維零填充降級。

18. **`fen_parser.py` 畸形 FEN 破壞棋盤不變量與 `OverflowError` 崩潰（[`fen_parser.py:57-74`](fen_parser.py#L57-L74)）**
    * 殘留無車易位權在 `board_operations.py:218` XOR 憑空生出 2 隻白車；無目標兵的假 EP 在 `L210` 破壞 `occ[0]|occ[1] == occ[2]`；負 `halfmove`（`-1`）於 NumPy 2.x 觸發未攔截的 `OverflowError`。對齊 SF19 `position.cpp:336-418` 補齊驗證與 `max(0, int(parts[4]))`。

---

### 🟡 P2 — 中／低嚴重度與邊界 BUG（13 項）

19. **`pawn_history` 初始先驗（`-1238`）與 `GOOD_QUIET_THRESHOLD = -1200` 衝突**（[`search_heuristics.py:217, 431`](search_heuristics.py#L217)、[`constants.py:690`](constants.py#L690)）：開局或重設後未見安靜步（即經一次減半 `-619 * 2 = -1238 < -1200`）全被降級至 `STAGE_BAD_CAPTURES` 之後且不排序，直至第 11 步隨每次減半漂回 `0`。（修法須單軸 SPRT 驗證。）
20. **`get_blockers_for_king` 永不填充 `pinners_bb`（恆回 `0`）且漏 `^ snipers`**（[`move_generator.py:437-497`](move_generator.py#L437-L497)）：違反 `(blockers, pinners)` 契約並異於 SF19 `position.cpp:624`（目前因呼叫端僅查敵方並丟棄 `pinners` 而被掩蓋）。
21. **`is_move_pseudo_legal` 未拒絕非升變步帶非零 `promotion_piece` 且 EP 未驗目標格為空**（[`move_generator.py:500-650`](move_generator.py#L500-L650)）：TT 碰撞或撕裂步可破壞 `candidate == tt_move` 去重與 `low_ply_history` 索引。
22. **`make_null_move` 誤遞增 `halfmove_clock`（`game_state[3] += 1`）**（[`board_operations.py:388-407`](board_operations.py#L388-L407)）：SF19 `position.cpp:1344-1373` 不更動 `rule50`；`hmc == 99` 且 `beta <= 0` 時 NMP 子節點會因 `hmc == 100` 誤回 `0` 形成 zugzwang 假截斷，且偏移 `get_tt_key` 桶位。
23. **Fail-Low 對手著法獎勵過度將非兵歷史表包入 `if (prev_piece % 6) != 0:`**（[`search.py:3332-3377`](search.py#L3332-L3377)）：剝奪安靜推兵步在 `history_table`、`butterfly_history` 與 5 張 `continuation_history` 的獎勵（SF19 `search.cpp:1594-1600` 僅限 `update_pawn_history`）。
24. **`select_best_thread` 在 HCE `(-29872, -10000]` 存在投票死區**（[`search.py:4648-4658, 4785`](search.py#L4648-L4658)）：落敗特化殘局同時滿足 `is_exact_decisive() == False` 與 `is_loss() == True`，導致拒斥全部 Helper 票。
25. **`KRPPKRP` 差一錯誤 `if 1 < r < 6:` 使第 2 橫排（`r=1, sf=9`）不可達**（[`endgame.py:956-971`](endgame.py#L956-L971)）：應改為 `if 0 < r < 6:`。
26. **`DIAG_HP_MARGIN_SUM` 在 Continuation History 剪枝觸發時發生 `uint64` 下溢**（[`search.py:2333, 2410, 2417`](search.py#L2410-L2417)）：負 `hp_margin` 轉 `uint64` 變 `~1.84e19` 並誤歸類至 `DIAG_HP_MARGIN_NEAR`。
27. **`TUNE_LMR_TTMOVE_RED`（Slot 19）在現行 SPSA 中為 100% 不可達空轉軸**（[`search.py:2841-2846`](search.py#L2841-L2846)、[`tune_search/search_param_registry.py:135`](../../tune_search/search_param_registry.py#L135)）：`tt_move` 永遠只會在 `searched_legal_moves == 1` 搜尋；應維持 ABI 槽位但從活躍 SPSA 擾動凍結或改接有用訊號（亦須注意 `LMR_R_FLOOR = -10` 在 1024 尺度下的意義）。
28. **多執行緒固定節點超標（~2.5–4x）與 `_SMP_HELPER_CONTEXT_CACHE` 記憶體洩漏**（[`search.py:4503-4552, 4994-5006`](search.py#L4503-L4552)）：Helper 節點無上限且以 `id(ctx)` 快取每條 ~36 MB，短生命週期 context 會洩漏且 `reset_search_context` 不會重置快取內 Helper。
29. **`reset_search_context` 漏清 `game_history`、`game_history_count`、`tt_generation`、`nmp_min_ply`、`diag_stats`**（[`engine_types.py:992-1024`](engine_types.py#L992-L1024)）。
30. **`move_to_uci(NO_MOVE)` 回傳 `"a1a1"` 而非 `"0000"`**（[`move.py:135-163`](move.py#L135-L163) → `main.py:337` 在將死／逼和盤面輸出非法 `bestmove a1a1`）。
31. **其餘細微邊界與契約守衛**：
    * [`search.py:624, 1001`](search.py#L624)（QSearch 平滑缺 `abs(beta) < MATE_IN_MAX_PLY`，使逼和／均勢反駁將死時把 TT 下界壓低跨過 `-VALUE_KNOWN_WIN`）；
    * [`evaluation.py:1705-1832`](evaluation.py#L1819-L1832)（多后極端局面非特化 HCE 未夾限於 `±(VALUE_KNOWN_WIN - 1)`）；
    * [`board_operations.py:271`](board_operations.py#L271)（`old_halfmove_clock` `uint8` 於 `hmc >= 256` 合成局面溢位回捲）；
    * [`search.py:325-339`](search.py#L325-L339)（`compute_full_corrected_static_eval` 未對哨兵回填 `get_pinned_pieces`）、[`search.py:2188`](search.py#L2188)（`enable_see_pruning` 誤鎖全部 Step 14 剪枝）、[`search.py:5343`](search.py#L5343)（`format_score_for_uci` 應與 `is_decisive` 一致用 `>=`）、[`core.py:17`](core.py#L17)（缺 `king_bb == 0` 守衛）、[`debug_utils.py:25`](debug_utils.py#L25)（`np.uint64` 一元負號警告）。

---

## 二、已確認冗餘、死碼與「零搜尋樹／零評估變動」純加速機會

### A. 高效益熱路徑純加速（11 項）

| # | 位置 | 瓶頸描述與優化做法 | 樹／分變動 |
| :-: | :--- | :--- | :---: |
| 1 | [`search.py:2024-2093, 2644-2711, 2881`](search.py#L2024-L2093) | 延遲敵方 `pinned_*`、`check_sq_*`、`blockers_for_their_king_s` 至離開 `STAGE_TT_MOVE` / Step 14 後；將 `is_giving_check_after_move` 延遲到 `if lmr > 0:` 時才算 | **0 變動** |
| 2 | [`engine_types.py:139-152`](engine_types.py#L139-L152)、[`pawns.py:435-530`](pawns.py#L435-L530) | 將 14 條獨立 SoA `pawn_table_*`（16.75 MiB/thread）打包為單一 64B Cache-Line AoS 結構陣列（14 條 cache line → 1 條；需注意現有測試契約需同步更新） | **0 變動** |
| 3 | [`search.py:4819-4824`](search.py#L4819-L4824)、[`engine_types.py:721, 744-745`](engine_types.py#L721) | 以 `@njit` 整數 `trunc_div(..., 2)` 取代 Python 層每步 `np.divide(..., 2)`（省 ~17.1 ms/move float64 配置）；縮小過度配置的 `low_ply_history`（93.75% 為 0）與 `quiet_*_tried`（`256 → 64`） | **0 變動** |
| 4 | [`move_generator.py:1418-1424`](move_generator.py#L1418-L1424)、[`evaluation.py:382-383, 1244-1268`](evaluation.py#L1244-L1268) | Magic Bitboard 射線前置過濾（`ROOK_RAYS`/`BISHOP_RAYS`）；拆開 `for piece_type in range(5)` 迴圈與 `pawnsOn7`；跳過 ~85–95% `SliderOnQueen` 與 `kingAttackers` Magic 查表 | **0 變動** |
| 5 | [`see.py:318, 351-354, 396-415`](see.py#L318-L354) | 以 `has_promo_pawn`（檢查目標格實際是否有可升變兵攻擊）取代寬鬆 `is_promo_rank`，恢復約 99% 底線吃子的 O(1) Pre-Magic 早退 | **0 變動** |
| 6 | [`search_heuristics.py:209-254, 334, 367, 424-432`](search_heuristics.py#L402-L432) | 將 `score_quiets` 5 組 `(prev_piece, prev_to)` 查詢外提至迴圈外；`score_captures_*` 先測 `(opp_occ & BB_SQUARES[to_sq]) != 0` 再掃 6 個棋子位元板 | **0 變動** |
| 7 | [`transposition_table.py:22-24, 316-335`](transposition_table.py#L316-L335) | `store_tt` `else:` 僅在資料確實更新時才重算與回寫 `padding` checksum（消弭 SMP MESI 快取行抖動）；`is_pv` 與 `best_move` 交換順位修復 offset-11 未對齊 | **0 變動** |
| 8 | [`search.py:3041-3050`](search.py#L3041-L3050) | 僅在 `if is_pv:` 時才複製 `pv_table[ply+1]` 列（省去 >95% 非 PV 節點複製） | **0 變動** |
| 9 | [`search.py:1289-1476` → `533-560`](search.py#L533-L560) | 待 Bug #12 為 QS 補上重複偵測後，於 `_search` 頂部直接 `if depth <= 0: return quiescence_search(...)`，消除 `depth <= 0` 的雙重 TT 探測／將軍檢查／子力檢查（會改變 leaf 節點數與 depth-0 cutoff，故列為 Wave 1B 後續項目） | *微幅節點變動* |
| 10 | [`__pycache__/*.nbc`](.) 實查 | 收斂 Numba `.nbc` 多重特化風暴：`pawns.evaluate_shelter_aligned`（**18 過載**）、`is_square_attacked`（**13 過載**）、`_see_ge_jit`（**9 過載**）、`update_continuation_history` / `store_tt`（各 **5 過載**）、`get_pinned_pieces`（**4 過載**） | **0 變動** |
| 11 | 多模組微優化 | `search.py:2429/2464` 複用 `fp_lmr_d`、合併重複 26 行合法性檢查與 5× `trusted_tt_score`；`pawns.py:567-590` 改遍歷車位元；`evaluation.py:1603-1673` 單次載入 `mat_1..4`、門控 `THREAT_RESTRICTED_PIECE`、Pawn Hash 快取 `Endgame King Proximity`、去重雙位移；`endgame.py` 複用預計算通路遮罩與模組級 `_KRPPKRP_SCALE`；`bitboard_utils.py` 增設 `lsb_nonzero` | **0 變動** |

### B. 已驗證 0 呼叫者的純死碼與冗餘常數（全倉 grep 確認）
* `engine_types.py`：`aggressor_cache`、`victim_cache`（共 64 KiB/thread，全倉 0 讀寫）。
* `transposition_table.py:387-404`：`estimate_tt_occupancy_percentage`。
* `move_generator.py`（約 370 行）：`GLOBAL_ATTACK_TABLES`、`check_legality_and_gives_check`、`_generate_tactical_moves_jit` / `generate_tactical_moves`、`generate_captures_buffer` / `generate_captures`、`get_all_{pawn,knight,bishop,rook}_attacks`。
* `zobrist.py`（約 60 行）：import 即編譯卻 0 呼叫的 `compute_initial_{pawn,minor,non_pawn}_hash`、與 `bitboard_utils.py` 重複的 `count_trailing_zeros` / `get_lsb_index`。
* `see.py`（約 45 行）：`get_attackers_for_see`、`DIAGONAL_DIRECTIONS`、`ORTHOGONAL_DIRECTIONS`、未用 import（`find_piece_type_on_square`、`SQUARES_BETWEEN`、`count_bits`）。
* `bitboard_utils.py:212-228` & `core.py:10-25`：`find_piece_type_on_square`、`_is_king_in_check_jit` / `is_king_in_check`。
* `constants.py`：`SCALE_FACTOR_KRPKR_FORTRESS`（`L441`）、`CAPTURE_HISTORY_VICTIM_TYPES`（`L763`）、`_sf_danger_to_ours`（`L321`）；以及不可達的 `OUTPOST_BONUS_{KNIGHT,BISHOP}[2],[6]`（建議在 `tuner/tuner.py:571` 一併鎖死 `(0,1,2,6,7)` 消除 8 個無效 SPSA 維度）。
* `evaluation.py` & `endgame.py`：死包裝 `evaluate_space`（需同步從 `tuner/codegen/generate_tunable_eval.py:300` 移除）、重複的 `DARK_SQUARES`/`LIGHT_SQUARES`、6 個未使用局部 `_EW_*` 別名（`evaluation.py:59-62, 84-85`；註：這些名稱在 `pawns.py`/`material.py` 中正常使用）、18 個未用 import。
* `search.py`：不可達的 `L1553-1558`（`depth >= 2` 內的 `if depth <= 0:`）、`L2841-2846`（`searched_legal_moves >= 2` 內的 `elif move == tt_move:`）、被 `L2884` 抵銷的 `+ LMR_D_MAX_EXTRA`（`L2877`）、以及未用區域變數（`moves_generated`、`can_prune_quiet`、`score_val`、`num_sorted`、`pinners_for_their_king_qs`）。

---

## 三、高 ROI 棋力提升提案（Classical HCE 適用，對照 SF19 / SF11）

### A. 搜尋與排序演算法（固定 Nodes 可直接單軸 SPRT 驗證）

| 優先序 | 提案內容 | SF19 參考位置 | 預估 Elo / 難度 | Numba JIT 與實驗要點 |
| :-: | :--- | :--- | :---: | :--- |
| **S-1** | **雙側 Non-Pawn Correction History**：對任意行棋方同時讀取並更新 `np_white_key` 與 `np_black_key`（搭配 Bug #1 Part A） | `search.cpp:91-100, 121-122` | **+8–15 Elo** / 極低 | 兩個鍵已於 `make_move` 增量維護；`SearchContext` 增加 0–64 KiB。 |
| **S-2** | **SF19 吃子／升變排序尺度（`K*MG[victim] + cap_hist`）、將無吃子欠升變移出吃子／QS、歷史導引 Good-Capture SEE 門檻（`-score // D`）** | `movegen.cpp:86-103`, `movepick.cpp:224-226, 306-308` | **+8–18 Elo** / 低 | 拆成 2–3 個單軸 SPRT（配 Bug #7 修復，維持 `full == staged` 契約）。 |
| **S-3** | **Cuckoo 即將重複偵測（`upcoming_repetition`）**（與 Bug #13 條件式 EP 雜湊協同） | `position.cpp:106-163, 1527-1568`, `search.cpp:737-742` | **+8–15 Elo** / 中 | `CUCKOO_KEYS` + `CUCKOO_MOVES` 共 **80 KiB**（`<< 1 MB` 快取限制）。 |
| **S-4** | **靜態評估差值 `evalDiff` 安靜歷史更新（`-(parent_eval + child_eval)`）**（搭配 Bug #2 完整評分一致化） | `search.cpp:979-986` | **+8–15 Elo** / 低 | 夾限於 HCE cp 尺度；限定非將軍／非吃子／非升變／非 SE。 |
| **S-5** | **QSearch 戰術剪枝與精化組**：`moveCount > 2` 剪非將軍非回吃、SEE-vs-Alpha delta 門檻 `!see_ge(m, alpha - futilityBase)`、TT 界限精化 `stand_pat` + `is_pv`、吃子後逼和檢查（**維持 E82 禁令不接 Correction History**） | `search.cpp:1709, 1733-1735, 1790-1811, 1864-1868` | **+8–15 Elo** / 低 | 為 `quiescence_search` 增加 `is_pv` 時須更新簽名並重驗 `0` `.numba.unresolved$`。 |
| **S-6** | **SF19 `DEPTH_UNSEARCHED` 純評估 TT 快取 + 非覆寫時保留 `best_move`**（搭配 Bug #2、#3、#8） | `types.h:234-241`, `tt.cpp:51-52, 104-121`, `search.cpp:853, 1750` | **+5–12 Elo + NPS** / 低–中 | 利用既有 16B `tt_entry_dtype` 的 `uint8 depth` 編碼，零 schema 破壞。 |
| **S-7** | **歷史回饋升級（已排除 P4 post-LMR）**：(a) 動態 `bonusScale` fail-low 安靜步 + 先前吃子 fail-low 獎勵；(b) 深度加權 Continuation History + `CMHCMultipliers` + `inCheck` 2-ply 上限；(c) TT 截斷時懲罰對手前手早期安靜步 | `search.cpp:886-888, 1579-1609, 2017-2041` | **+6–15 Elo** / 低 | 嚴格單軸 SPRT + warm-sequence 檢驗；新增 128B `captured_piece_stack`。 |
| **S-8** | **LMR 負減步延伸（`d = search_depth + 1` + `PvNode` 寬免）與 SE 精化**（奇異 `depth++`、`cutoffCnt += (ext < 2) || PvNode`、Multi-Cut 校正歷史、`is_shuffling`、`MIN_SINGULAR_DEPTH + tt_pv`） | `search.cpp:153-160, 1246-1287, 1369-1373, 1529` | **+6–12 Elo** / 低–中 | 純整數邏輯；避開已否決的 E52 `double_ext_min_depth=7`。 |
| **S-9** | **分色 `butterfly_history[2, 64, 64] int16`、隨深度縮放 `partial_insertion_sort(-K*depth)`，及協同校正 `GOOD_QUIET_THRESHOLD`／歷史老化策略** | `history.h:128, 132`, `movepick.cpp:280, 324`, `search.cpp:326-330` | **+5–12 Elo** / 低–中 | 維持 16 KiB；拆為 3 次單軸 SPRT。 |
| **S-10** | **Lazy SMP `SharedHistories`（以 `np.frombuffer` 共享 `pawn_history` + 5 張 `correction_history`）+ SF19 `averageScore` 根層抱負視窗** | `search.cpp:88, 117, 171, 376, 2056` | **+5–15 Elo（4T+ SMP）** / 低 | 零 StructRef 欄位變動，並為每條 Helper 省下 ~13–19 MiB RAM。 |
| **S-11** | **精簡 TT 條目打包（12B 含校驗 5-way 或 10B 3-way）+ `mul_hi64` 支援非 2 的冪次 Hash 大小** | `tt.cpp:37-95, 168-175, 294-296` | **+4–10 Elo（固定 Hash）** / 中 | 每 MB 多存 25–50% 條目；5 行 LLVM `@intrinsic` 實作 `mul_hi64`。 |
| **S-12** | **NMP 後 `improving |= static >= beta`、RFP 納入 `opponentWorsening` 與 `abs(corr)` 裕度、50 步高半步時降級 TT 將死分** | `search.cpp:1002-1004, 1046, 1928-1949` | **+4–8 Elo** / 極低 | 各 1–10 行，防止 8-ply `RULE50_KEYS` 分桶引發將死幻覺。 |

### B. Classical HCE 與評估↔搜尋介面（固定 Nodes 可直接單軸 SPRT 驗證）

| 優先序 | 提案內容 | SF11 / SF19 參考 | 預估 Elo / 難度 | 備註 |
| :-: | :--- | :--- | :---: | :--- |
| **E-1** | **動態 Optimism / Contempt**：每層根搜尋更新 `optimism[us] = K * avg / (abs(avg) + C)`，評估端依 SF11 材質／階段公式縮放 | SF19 `search.cpp:381-383, 1901-1904`; SF11 `evaluate.cpp:784` | **+8–15 Elo** / 低 | 加 2 個 `int32` StructRef 欄位，`search_context is None` 時為 0（Texel 零影響），鎖定現行 66% 高和棋率。 |
| **E-2** | **統一評估層線性 50 步衰減（`v -= v * rule50 // 200`）並移除 `search.py` `hmc >= 80` 斷崖** | SF19 `evaluate.cpp:63` | **+5–10 Elo** / 低 | 解決 `SEARCH_EVAL_INTERFACE.md` P7 雙重扣減問題；保留 `TT_CUTOFF_HALFMOVE_MAX=96` 與 `>=100` 和棋規則。 |
| **E-3** | **將敵方閃將阻擋子（`blockers_for_king`）納入 `kingDanger` 與 `mobility_area`**（與 Bug #20 協同） | SF11 `evaluate.cpp:230, 449`, `position.cpp:447-478` | **+4–8 Elo** / 低–中 | 單次射線回傳 `(pinned_friendly, all_blockers)`；對 `search.py` 維持僅回傳友方牽制子。 |
| **E-4** | **通路兵感知的異色格象縮放因子**：純 OCB `18 + 4 * strong_passed_cnt`、OCB+他子 `22 + 3 * strong_non_pawn_cnt` | SF11/SF12 `evaluate.cpp` | **+2–5 Elo**（純象殘局） / 低 | 複用 `evaluation.py:1814` 已算好的 `w_passed`/`b_passed`。 |

### C. 純限時／NPS 結構升級（固定 Nodes 為 0 Elo，須以節點恆等＋限時賽驗證）
* **T-1**：增量 PST + 每執行緒 8,192 格 Material Hash Table（SF11 `material.cpp:118-217`，~192 KiB StructRef，僅變動 `eval_weights` 時清空）→ 預估 **評估提速 +12–20% / 限時 +8–15 Elo**。
* **T-2**：專屬 `generate_evasions_buffer`（SF19 `movegen.cpp:207-240`，直接在生成端排除 84.27% 被將軍非法候選，避開 P3 已回退的生成後壓縮）→ 預估 **+4–8% NPS / 限時 +5–10 Elo**。
* **T-3**：將易位路徑受攻檢查自 MoveGen 延遲至 pre-make 合法性驗證（SF19 `movegen.cpp:234-237`）→ 預估 **開中局 +1–2% NPS**。

---

## 四、交叉反證過濾與注意事項（Audit Transparency）

1. **已剔除誤報**：Post-LMR Continuation History 更新（`search.cpp:1390`）已在 **P4** 實測 3 種變體並回退（見 [`HISTORY_HEURISTICS_CN.md`](HISTORY_HEURISTICS_CN.md#L9)），`DIAG_POST_LMR_*` 為刻意保留的影子診斷。
2. **拆軸警告（Bug #1 Part B）**：Bug #1 **Part A**（恢復 `search.py:3390-3407` 負向 Correction History）可直接修復；**Part B**（fail-low 不將 `best_move` 寫入 TT）因與 ALL-node IIR 及迭代快取互動（E73 `-7.99 Elo`、E80 `-4.63 Elo` 曾因綑綁回退），**必須另開單軸 SPRT**。
3. **測試契約鎖定提醒**：`transposition_table.py` 的 `probe_tt` / `_EMPTY_TT_ENTRY` 被 `tests/test_lazy_smp.py:33` 匯入；`probe_tt_best_move` 的 `TT_FLAG_NONE` 契約鎖定於 `tests/test_search_core_contracts.py:583-589`；`pawn_table_*` 欄位名稱鎖定於 `test_search_core_contracts.py`、`test_endgame_conformance.py`、`test_lazy_smp.py`。任何重構須同步維護測試契約。
4. **條件式 EP 雜湊（Bug #13）提醒**：修正後雙步推兵無鄰近敵兵的盤面 Zobrist key 會改變，施行時須同步核對既有 Zobrist / FEN 單元測試。

---

## 五、分階段工程推進路線圖

1. **Wave 2（零樹／零分變動純 NPS 優化 + 死碼清理）— ✅ 已完成並通過 SPRT H1 驗收（Commit `0f4c2eb`）**：
   * 20 盤面 Depth 10 × 5 次基準測試：`mismatches = 0 / 20`（總節點數 `1,056,156` 100% 恆等），中位數均值 NPS `458,684 → 601,128`（**`+31.05%`**），算術平均 NPS `452,295 → 592,026`（**`+30.89%`**）。
   * Colab 固定時間（`500 ms/move`，7 並發 Worker，64 MB TT）新舊引擎實戰對抗：**780 局（390 對換先）** 打出 **`417.5 - 362.5`（53.53%，`194W 447D 139L`，淨勝 `+55` 局 / `+44` 對）**，**`+24.54 ± 15.92 Elo`**（95% CI `[+8.54, +40.64]`），**`SPRT[0, 10] LLR = +2.95` 觸發 `H1 Accepted`**；高並發下實戰平均深度推進 **`+0.31 ply`**（`14.22 → 14.53`, `p = 2.17e-44`），每步節點數與 NPS 雙雙提升 **`+11.4% / +11.3%`**（`p < 1e-197`）。詳見 [`CLASSICAL_OPTIMIZATION_EXP_LOG.md`](CLASSICAL_OPTIMIZATION_EXP_LOG.md)。
2. **Wave 1A（安全／介面／殘局純正確性 BUG 修復）— ✅ 已完成、通過 2,600 局無退化驗證並同步至 `classical_old/`**：Bug #4（`MAX_PLY` 越界）、#5（Polyglot 易位／EP）、#6（`mate_kbnk`）、#14–#16（`KBPsK` / `KRPKN` / `KXK`）、#17–#18（`eval_weights` fallback / `fen_parser`）、#25（`KRPPKRP`）、#29–#30（`reset_search_context` / `NO_MOVE`）共 11 項全數修復；20 盤面 Depth 10 節點 `1,056,156` 100% 恆等、回歸測試 [`tests/test_wave1a_fixes.py`](../../tests/test_wave1a_fixes.py)（27 項）與 130 項批次測試全過；Colab `500 ms/move` 實戰 2,600 局打出 `545W 1520D 535L`（**`+1.34 ± 8.24 Elo`**，LOS `62.5%`，`SPRT[-10, 0] LLR=+3.58` H1 Accepted）。詳見 [`CLASSICAL_OPTIMIZATION_EXP_LOG.md`](CLASSICAL_OPTIMIZATION_EXP_LOG.md) Wave 1A。
3. **Wave 1B（高影響搜尋 BUG — 逐項固定節點 SPRT 驗證）**：
   * **Wave 1B-1（Bug #1 Part A：啟用 Fail-Low 負向 Correction History）— ✅ 已通過固定節點驗證並同步 `classical_old/`**：`300k nodes` 1,628 局打出 `303W 1055D 270L`（`51.01%`，`+7.04 ± 9.72 Elo`，LOS `92.2%`，`SPRT[-10, 0] LLR=+4.90` H1 Accepted、`[-5, 0] +1.94`）。詳見 [`CLASSICAL_OPTIMIZATION_EXP_LOG.md`](CLASSICAL_OPTIMIZATION_EXP_LOG.md) Wave 1B-1。
   * **Wave 1B-2（Bug #2：禁止 `lazy=True` 粗評寫入 `TT.static_eval` 與校正歷史）— ✅ 已確認無退化並同步 `classical_old/`**：20 FENs Depth-10 總節點降 **`-15.08%`**、Weighted NPS `-0.29%` 近零開銷，已同步作為 Wave 1B-3 對照基線，見 [`CLASSICAL_OPTIMIZATION_EXP_LOG.md`](CLASSICAL_OPTIMIZATION_EXP_LOG.md) Wave 1B-2。
   * **Wave 1B-3（Bug #3 + #8：禁止 QS 寫入 `TT_FLAG_EXACT` + 無鎖 TT 競態／TOCTOU／溢出修復）— ✅ `[-10, 0]` H1 通過（882 局 `+9.06 ± 13.15 Elo`，已接受並同步 `classical_old/`）**：40 盤面 Depth-14 `d0_EXACT` 由 `242,035 → 0`、`TT d>=6` 保留量 `+9.62%`、總節點 `+5.75%`；`300k nodes` 實戰 882 局打出 `164W 577D 141L`（`51.30%`，`+9.06 ± 13.15 Elo`，LOS `91.2%`，`SPRT[-10, 0] LLR=+3.13` H1 Accepted、`[-5, 0] +1.29`）。見 [`CLASSICAL_OPTIMIZATION_EXP_LOG.md`](CLASSICAL_OPTIMIZATION_EXP_LOG.md) Wave 1B-3。
   * **Wave 1B-4（Bug #7：升變步於 `score_captures_lazy` 與 `score_captures_with_tt_lazy` 排序評分修正）— 🧪 已實作並通過單元／快取／升變專題對照（待固定節點 SPRT 驗證）**：84 項測試全過；20 標準 FENs `-0.02%` 節點、5 升變專題 FENs `-6.95%` 節點（`Promo_QSEvasion -20.17%`），見 [`CLASSICAL_OPTIMIZATION_EXP_LOG.md`](CLASSICAL_OPTIMIZATION_EXP_LOG.md) Wave 1B-4。
   * **後續待辦依序**：Bug #1 Part B（視 SPRT 結果）→ Bug #9/#10/#11/#12 → Bug #13 + #22。
4. **Wave 3（高 ROI 搜尋／HCE 強化 — 嚴格單軸 SPRT）**：依優先序挑選 `S-1`～`S-5`、`E-1`、`E-2` 逐項驗證後決定是否同步 `classical_old/`。
