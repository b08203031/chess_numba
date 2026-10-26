
---

# 西洋棋引擎前向剪枝（Forward Pruning）修復計畫書 (v2 Final)

> [!IMPORTANT]
> **給執行 AI Agent 的定位守則**：
> 1. **請勿依賴絕對行號**：請使用本計畫書提供的**函式名稱**與**唯一程式碼片段（Code Snippet Anchors）**進行搜尋與替換。
> 2. **分階段提交與測試**：請先執行 **Phase A（純邏輯與結構修復）** 並通過驗證，再個別測試 **Phase B（剪枝範圍調整）** 與 **Phase C（獨立清理項目）**，切勿將所有調參變更混在同一個未測試的 commit 中。
> 3. **禁止修改項目**：
>    * `see.py` 與 `transposition_table.py` 邏輯正確，**請勿修改**。
>    * `search.py` Step 14 內的 `quiet_move_counter += 1` 位置與 Stockfish 19 的 `++moveCount` 語意一致（被剪枝的著法本來就會計入），**請勿移動 `quiet_move_counter` 的遞增位置**。

---

## Phase A：核心邏輯與結構性 Bug 修復（優先執行）

### 任務 A1：[P0] 移除 `_search` 在 Fail-Low 時偽造 `best_move` 的 Fallback

#### 1. 問題原理
* `transposition_table.py` 的 `store_tt` 在收到 `best_move == 0 (NO_MOVE)` 時，本來就會自動保留 TT 內既有的舊 `best_move`。
* 目前 `_search` 在迴圈結束後，若 `best_move == NO_MOVE`（即 Fail-Low / ALL-Node），會強行從 `moves` 緩衝區挑出第一個未排除的 pseudo-legal 著法 `moves[i]` 賦值給 `best_move` 並寫入 TT。
* 這不僅可能選到**不合法的 pseudo-legal 著法**，還會覆蓋掉 TT 原有的真正好棋，並使後續探測該局面時的 **Step 9 RFP** (`tt_move == NO_MOVE or tt_capture`)、**Step 11 IIR** (`tt_move == NO_MOVE`)、**Step 14 剪枝豁免** (`move != tt_move`) 與 **Step 17 LMR** (`if tt_move == NO_MOVE:`) 全部失真。
* 根節點 (`ply == 0`) 在 `iterative_deepening_search` 中已有完整的 `res[1] -> TT -> pv_table` 保底鏈，因此 `_search` 內部完全不需要這段 fallback。

#### 2. 修改位置與方法 (`chess_engine/classical/search.py` -> `_search`)
* **定位錨點**：在 `_search` 函式接近尾聲、Step 21 歷史更新與 Fail-Low Opponent Move Bonus 結束之後，搜尋 `original_best_move = best_move`。
* **修改內容**：將挑選 `moves[i]` 的 fallback 整段刪除，使 `best_move` 在 Fail-Low 時維持為 `NO_MOVE`。

```python
    # ======= 修改前 =======
    original_best_move = best_move

    generated_end = max(search_context.mp_captures_end[ply], search_context.mp_quiets_end[ply])
    if best_move == NO_MOVE and generated_end > 0:
        # Fallback to first move that is not excluded
        for i in range(generated_end):
            if moves[i] != excluded_move:
                best_move = moves[i]
                break

    # ======= 修改後 =======
    original_best_move = best_move
```

---

### 任務 A2：[P0] 修復 `quiescence_search` 與 ProbCut 的 `move_stack` 維護及 Correction History 斷層

#### 1. 問題原理與執行順序
* **`apply_correction_history_score` 依賴 `move_stack[ply - 1]` 與 `piece_stack[ply - 1]`**：因此必須先確保所有進入 `quiescence_search` 的路徑都有正確寫入 stack（步驟 1 & 2），接著才在 `quiescence_search` 中套用 `apply_correction_history_score`（步驟 3）。
* **三個修復點**：
  1. `quiescence_search` 在 `make_move` 後未寫入 `move_stack[ply]` 與 `piece_stack[ply]`，導致深層 QS 排序 (`get_quiet_stat_score`) 與 `cntcv` 讀到過期殘留值。
  2. `_search` 的 **Step 12 ProbCut** 在呼叫第一階段 `quiescence_search(..., ply + 1)` **之前**沒有設定 `move_stack[ply]` 與 `piece_stack[ply]`（原本寫在 QS 結束之後），導致 ProbCut 呼叫的 QS 讀到過期的 `move_stack[ply]`。
  3. `quiescence_search` 的 `stand_pat` 直接使用未修正的 `raw_eval`，與主搜尋 `_search`（如 Razoring / FP）經過 `apply_correction_history_score` 修正後的評估產生斷層（對照 SF19 `qsearch` 會呼叫 `to_corrected_static_eval`）。

#### 2. 修改位置與方法 (`chess_engine/classical/search.py`)

##### 步驟 A2-1：在 `quiescence_search` 的 `make_move` 後寫入 stack
* **定位錨點**：在 `quiescence_search` 的著法迴圈內，搜尋 `legal_moves_tried += 1`。
```python
        # ======= 修改前 =======
        legal_moves_tried += 1

        score, child_q_nodes = quiescence_search(
            piece_bbs, occupancy_bbs, game_state,
            _i32(-beta), _i32(-alpha), _i32(ply + 1), search_context, _i32(q_ply + 1),
        )

        # ======= 修改後 =======
        legal_moves_tried += 1

        qs_moved_piece = unmake_info[0]
        if qs_moved_piece != -1 and qs_side == BLACK:
            qs_moved_piece += 6
        search_context.move_stack[ply] = move
        search_context.piece_stack[ply] = qs_moved_piece

        score, child_q_nodes = quiescence_search(
            piece_bbs, occupancy_bbs, game_state,
            _i32(-beta), _i32(-alpha), _i32(ply + 1), search_context, _i32(q_ply + 1),
        )
```

##### 步驟 A2-2：在 `_search` Step 12 ProbCut 呼叫 `quiescence_search` **之前**設定 stack
* **定位錨點**：在 `_search` 的 Step 12 ProbCut 迴圈內，搜尋 `pc_q_score, pc_q_nodes = quiescence_search`。
* **注意**：提前設定 stack 後，若在 `stop_flag[0]` 或 `pc_q_score < probcut_beta` 提前退出，也必須還原 `saved_pc_stack_move` 與 `saved_pc_stack_piece`。
```python
                # ======= 修改前 =======
                pc_q_score, pc_q_nodes = quiescence_search(
                    piece_bbs, occupancy_bbs, game_state,
                    _i32(-probcut_beta), _i32(-probcut_beta + 1), _i32(ply + 1), search_context, _i32(0),
                )
                quiescence_nodes += pc_q_nodes
                pc_q_score = -pc_q_score

                if search_context.stop_flag[0]:
                    unmake_move(piece_bbs, occupancy_bbs, game_state, pc_move, pc_unmake)
                    return (np.int32(0), NO_MOVE, nodes_searched, quiescence_nodes, tt_hits)

                if pc_q_score < probcut_beta:
                    unmake_move(piece_bbs, occupancy_bbs, game_state, pc_move, pc_unmake)
                    continue

                saved_pc_stack_move = search_context.move_stack[ply]
                saved_pc_stack_piece = search_context.piece_stack[ply]
                pc_moved_piece_type = pc_unmake[0]
                if pc_moved_piece_type != -1 and pc_original_side == BLACK:
                    pc_moved_piece_type += 6
                search_context.move_stack[ply] = pc_move
                search_context.piece_stack[ply] = pc_moved_piece_type

                # ======= 修改後 =======
                saved_pc_stack_move = search_context.move_stack[ply]
                saved_pc_stack_piece = search_context.piece_stack[ply]
                pc_moved_piece_type = pc_unmake[0]
                if pc_moved_piece_type != -1 and pc_original_side == BLACK:
                    pc_moved_piece_type += 6
                search_context.move_stack[ply] = pc_move
                search_context.piece_stack[ply] = pc_moved_piece_type

                pc_q_score, pc_q_nodes = quiescence_search(
                    piece_bbs, occupancy_bbs, game_state,
                    _i32(-probcut_beta), _i32(-probcut_beta + 1), _i32(ply + 1), search_context, _i32(0),
                )
                quiescence_nodes += pc_q_nodes
                pc_q_score = -pc_q_score

                if search_context.stop_flag[0]:
                    search_context.move_stack[ply] = saved_pc_stack_move
                    search_context.piece_stack[ply] = saved_pc_stack_piece
                    unmake_move(piece_bbs, occupancy_bbs, game_state, pc_move, pc_unmake)
                    return (np.int32(0), NO_MOVE, nodes_searched, quiescence_nodes, tt_hits)

                if pc_q_score < probcut_beta:
                    search_context.move_stack[ply] = saved_pc_stack_move
                    search_context.piece_stack[ply] = saved_pc_stack_piece
                    unmake_move(piece_bbs, occupancy_bbs, game_state, pc_move, pc_unmake)
                    continue
```

##### 步驟 A2-3：在 `quiescence_search` 中對 `stand_pat` 套用 `apply_correction_history_score`
* **定位錨點**：在 `quiescence_search` 開頭初始化變數處與 `--- QSearch TT static_eval probe ---` 註解處。
* **修改原則**：以 `raw_stand_pat` 保存未修正的靜態評估（供 `store_tt` 的 `static_eval` 欄位儲存），並以 `stand_pat = apply_correction_history_score(game_state, search_context, raw_stand_pat, ply)` 作為 `best_eval`、Stand-pat Beta Cutoff、Alpha 更新與 Delta Pruning 的基準。
```python
    # 1. 初始化處新增 raw_stand_pat:
    stand_pat = np.int32(32767)
    raw_stand_pat = np.int32(32767)
    best_eval = np.int32(-32767)

    # 2. QSearch TT static_eval probe 區塊修改:
        # --- QSearch TT static_eval probe ---
        if tt_entry['flag'] != TT_FLAG_NONE and tt_entry['static_eval'] != 32767:
            raw_stand_pat = np.int32(tt_entry['static_eval'])
        else:
            stand_pat_val, p_w, p_b = _evaluate_position_jit(piece_bbs, occupancy_bbs, game_state, False, search_context)
            raw_stand_pat = np.int32(stand_pat_val)
            if p_w != PINNED_UNCOMPUTED_SENTINEL:
                pinned_white = p_w
                pinned_black = p_b
                pinned_computed = True

        stand_pat = apply_correction_history_score(game_state, search_context, raw_stand_pat, ply)
        best_eval = stand_pat

        if stand_pat >= beta:
            _diag_add(search_context, DIAG_Q_STANDPAT_CUT)
            # B1: QSearch Stand-pat smoothing
            val = stand_pat
            if abs(val) < MATE_IN_MAX_PLY:
                val = np.int32(trunc_div(QS_STANDPAT_SMOOTH_A * val + QS_STANDPAT_SMOOTH_B * beta, QS_SMOOTH_DIV))
            
            tt_score = val
            if tt_score > MATE_IN_MAX_PLY: tt_score += ply
            elif tt_score < -MATE_IN_MAX_PLY: tt_score -= ply
            store_tt(search_context.transposition_table, tt_key, 0, tt_score, np.int16(raw_stand_pat), TT_FLAG_BETA, NO_MOVE, search_context.tt_generation, False)
            return np.int32(val), q_nodes
        alpha = max(alpha, stand_pat)

        move_count = generate_pseudo_legal_captures_buffer(piece_bbs, occupancy_bbs, game_state, search_context.moves_buffer, ply)
        if move_count == 0:
            tt_score = np.int32(stand_pat)
            if tt_score > MATE_IN_MAX_PLY: tt_score += ply
            elif tt_score < -MATE_IN_MAX_PLY: tt_score -= ply
            store_tt(search_context.transposition_table, tt_key, 0, tt_score, np.int16(raw_stand_pat), TT_FLAG_EXACT, NO_MOVE, search_context.tt_generation, False)
            return np.int32(stand_pat), q_nodes

    # 3. 函式結尾 store_tt 前的 tt_static_eval_to_store 修改:
    tt_static_eval_to_store = np.int16(32767)
    if not is_currently_in_check and raw_stand_pat != 32767 and abs(raw_stand_pat) < MATE_SCORE - MAX_PLY:
        tt_static_eval_to_store = np.int16(raw_stand_pat)
    store_tt(search_context.transposition_table, tt_key, 0, tt_score, tt_static_eval_to_store, tt_flag, best_move, search_context.tt_generation, False)
```

---

### 任務 A3：[P0] 對齊 Stockfish 19 的非吃子升變語意（升后 vs 欠升變）

#### 1. 問題原理
* 在 Stockfish 19 中，`capture_stage(m)` 的定義是**「吃子著法 或 升后（Queen Promotion）」**；而非吃子的**欠升變（`=R, =B, =N`）**則被視為後段著法並接受剪枝與減深。
* 目前程式碼將 `is_pseudo_quiet` 定義為 `not is_capture and not is_promotion`，導致非吃子升變（含欠升變）既不進 Step 14 的 `if is_capture:`（漏掉 SEE 剪枝），也不進 `elif is_pseudo_quiet:`，且在 Step 17 LMR 中 `is_eligible` 永遠為 `False`（所有欠升變都以 `lmr = 0` 全深度搜尋）。
* **注意**：在 `score_captures_lazy` 與 `score_captures_with_tt_lazy` 中，**只有升后（`PROMO_QUEEN`）**才應獲得正向子力加分；欠升變不能給高額正分，否則會排在一般好吃子前面。此外，在 Step 14 的 Capture FP 中，因 `pre_move_gives_check` 已包含 `is_promotion`，升變本來就會跳過 Capture FP，故無須修改 Capture FP 的 `victim_val`，只需讓升變進入 SEE 剪枝與 LMR。

#### 2. 修改位置與方法

##### 步驟 A3-1：修改 `chess_engine/classical/search_heuristics.py` 的 `score_captures_lazy` 與 `score_captures_with_tt_lazy`
* **定位錨點**：在 `search_heuristics.py` 中搜尋 `def score_captures_lazy(` 與 `def score_captures_with_tt_lazy(`（並確認已從 `chess_engine.classical.move` 匯入 `PROMO_QUEEN`）。
* **修改方式**：僅對 `PROMO_QUEEN` 給予 `MG_MATERIAL_VALUES[QUEEN] - MG_MATERIAL_VALUES[PAWN]` 加分，欠升變給予 `-500` 懲罰使其排在正常吃子之後。
```python
# 在 score_captures_lazy 迴圈內:
        promo_bonus = 0
        if get_special_move_flag(move) == SPECIAL_MOVE_FLAG_PROMOTION:
            if get_promotion_piece(move) == PROMO_QUEEN:
                promo_bonus = MG_MATERIAL_VALUES[QUEEN] - MG_MATERIAL_VALUES[PAWN]
            else:
                promo_bonus = -500

        base_score = mvv_lva + promo_bonus
        if victim_rel != -1:
            base_score += search_context.capture_history[aggressor_type, to_square, victim_rel]
```
在 `score_captures_with_tt_lazy` 中，將 `if victim_rel != -1:` 改為：
```python
            is_promo = (get_special_move_flag(move) == SPECIAL_MOVE_FLAG_PROMOTION)
            promo_bonus = 0
            if is_promo:
                if get_promotion_piece(move) == PROMO_QUEEN:
                    promo_bonus = MG_MATERIAL_VALUES[QUEEN] - MG_MATERIAL_VALUES[PAWN]
                else:
                    promo_bonus = -500

            if victim_rel != -1 or is_promo:
                base_score = mvv_lva + promo_bonus + QS_EVASION_CAPTURE_BUCKET
                if victim_rel != -1:
                    base_score += search_context.capture_history[aggressor_type, to_square, victim_rel]
            elif aggressor_type >= 0:
                ...
```

##### 步驟 A3-2：修改 `chess_engine/classical/search.py` 的 Step 14 與 Step 17 LMR
* **定位錨點 1（Step 14 吃子/升變分支）**：在 `_search` 的 Step 14 內搜尋 `if is_capture:`。
  * 將 `if is_capture:` 改為 `if is_capture or is_promotion:`。
  * 在計算 `cap_victim_see` 時（搜尋 `cap_victim_see = -2 if flag == SPECIAL_MOVE_FLAG_EN_PASSANT else cap_victim`），當 `not is_capture`（即非吃子升變）時，`cap_victim` 為 `-1`，`_see_ge_jit` 會自動以空目標格加上升變價值 (`get_promotion_piece(move) + 1`) 計算 SEE，直接淘汰推入敵方控制格白送子的升變與欠升變。
* **定位錨點 2（Step 17 LMR 資格與 `is_bad_capture`）**：在 `_search` 中搜尋 `DIAG_LMR_CAP_SEE_TRY` 以及 `if search_context.enable_lmr and depth >= LMR_MIN_DEPTH:`。
  1. 計算 `is_bad_capture` 處：
     ```python
     # 修改前:
     if (is_capture and search_context.enable_lmr and depth >= LMR_MIN_DEPTH
             and searched_legal_moves >= 1):
     # 修改後:
     if ((is_capture or is_promotion) and search_context.enable_lmr and depth >= LMR_MIN_DEPTH
             and searched_legal_moves >= 1):
     ```
  2. LMR `is_eligible` 判定處：
     ```python
     # 修改前:
     is_eligible = False
     if is_quiet_move and quiet_move_counter >= LMR_MIN_QUIET_MOVE_INDEX and moved_piece_type != -1:
         is_eligible = True
     elif is_capture:
         is_eligible = True

     # 修改後:
     is_eligible = False
     if is_quiet_move and quiet_move_counter >= LMR_MIN_QUIET_MOVE_INDEX and moved_piece_type != -1:
         is_eligible = True
     elif is_capture or is_promotion:
         is_eligible = True
     ```
  3. LMR 內部的 `if is_capture:` 好壞吃子調整處（搜尋 `r -= search_context.tune[TUNE_GOOD_CAP_RELIEF]`）：
     ```python
     # 修改前:
     if is_capture:
         if not is_bad_capture:
     # 修改後:
     if is_capture or is_promotion:
         if not is_bad_capture and (is_capture or get_promotion_piece(move) + 1 == QUEEN):
             r -= search_context.tune[TUNE_GOOD_CAP_RELIEF]
         else:
             r += search_context.tune[TUNE_BAD_CAP_BONUS]
     ```
     （如此一來，非吃子的欠升變 `=R, =B, =N` 在 LMR 中會獲得 `TUNE_BAD_CAP_BONUS` 加重減深，不會搶佔搜尋深度！）

---

### 任務 A4：[P0] 修復 Step 14 少子殘局守衛 (`side_non_pawn_count > 0`) 與 Capture FP Lazy Eval 升級

#### 1. 問題原理
1. **王兵殘局守衛缺失**：Stockfish 19 在 Step 14 最外層有 `pos.non_pawn_material(us)` 守衛。本專案目前僅在部分分支檢查 `low_material_pruning_guard` (`total_piece_count <= 7`)，若遇到雙方多兵的純王兵殘局（總子數 > 7 但 `side_non_pawn_count == 0`），Step 14 會誤砍關鍵王步與推兵。
2. **Capture FP 漏掉 Lazy Eval 升級驗證**：當 `futility_value <= alpha` 觸發時，若 `not static_eval_is_full`，必須先呼叫 `compute_full_corrected_static_eval` 確認完整評估值，避免被誤差較大的 Lazy Eval 誤砍吃子。

#### 2. 修改位置與方法 (`chess_engine/classical/search.py`)
* **定位錨點 1**：搜尋 `# --- NEW: Shallow Depth Pruning (Stockfish Step 14) ---`。
  ```python
  # 修改前:
  if (ply > 0 and move != tt_move and search_context.enable_see_pruning and not is_loss(max_eval)
          and depth <= PRUNING_SHALLOW_DEPTH and not is_currently_in_check and not is_pv):

  # 修改後 (加入 side_non_pawn_count > 0):
  if (ply > 0 and move != tt_move and side_non_pawn_count > 0 and search_context.enable_see_pruning and not is_loss(max_eval)
          and depth <= PRUNING_SHALLOW_DEPTH and not is_currently_in_check and not is_pv):
  ```
* **定位錨點 2**：在 Step 14 Capture FP 內搜尋 `if futility_value <= alpha:`。
  ```python
                        # 修改前:
                        if cap_lmr_d < CAP_FP_MAX_LMR_DEPTH:
                            victim_val = int(MG_MATERIAL_VALUES[cap_victim])
                            futility_value = (
                                tt_refined_eval
                                + CAP_FP_BASE
                                + CAP_FP_LMR_MULT * cap_lmr_d
                                + victim_val
                                + (cap_hist * CAP_FP_CAPTHIST_NUM // CAP_FP_CAPTHIST_DEN)
                            )
                            if futility_value <= alpha:
                                pruned_moves += 1
                                _diag_add(search_context, DIAG_SEE_CAP_SKIP)
                                continue

                        # 修改後 (補上 Lazy Eval 升級驗證，並維持 improving |= static_score >= beta):
                        if cap_lmr_d < CAP_FP_MAX_LMR_DEPTH:
                            victim_val = int(MG_MATERIAL_VALUES[cap_victim])
                            futility_value = (
                                tt_refined_eval
                                + CAP_FP_BASE
                                + CAP_FP_LMR_MULT * cap_lmr_d
                                + victim_val
                                + (cap_hist * CAP_FP_CAPTHIST_NUM // CAP_FP_CAPTHIST_DEN)
                            )
                            if futility_value <= alpha:
                                if not static_eval_is_full:
                                    raw_static_eval, static_score, improving, cached_pinned_white, cached_pinned_black = compute_full_corrected_static_eval(
                                        piece_bbs, occupancy_bbs, game_state, search_context, ply
                                    )
                                    static_eval_is_full = True
                                    if static_score >= beta:
                                        improving = True
                                    tt_refined_eval = static_score
                                    if trusted_tt_score:
                                        if tt_flag == TT_FLAG_BETA and tt_score_adj > tt_refined_eval:
                                            tt_refined_eval = tt_score_adj
                                        elif tt_flag == TT_FLAG_ALPHA and tt_score_adj < tt_refined_eval:
                                            tt_refined_eval = tt_score_adj
                                        elif tt_flag == TT_FLAG_EXACT and trusted_exact_depth:
                                            tt_refined_eval = tt_score_adj
                                    lmr_r = compute_lmr_reduction_1024(
                                        depth,
                                        searched_legal_moves + 1,
                                        improving,
                                        search_context.tune[TUNE_LMR_TABLE_SCALE],
                                        search_context.tune[TUNE_LMR_NOT_IMP],
                                    )
                                    cap_lmr_d = max(0, (depth - 1) - (lmr_r // 1024))
                                    futility_value = (
                                        tt_refined_eval
                                        + CAP_FP_BASE
                                        + CAP_FP_LMR_MULT * cap_lmr_d
                                        + victim_val
                                        + (cap_hist * CAP_FP_CAPTHIST_NUM // CAP_FP_CAPTHIST_DEN)
                                    )
                                if cap_lmr_d < CAP_FP_MAX_LMR_DEPTH and futility_value <= alpha:
                                    pruned_moves += 1
                                    _diag_add(search_context, DIAG_SEE_CAP_SKIP)
                                    continue
  ```

---

### 任務 A5：[P1] 對齊 Stockfish 19 的 Step 10/11/12/13 (NMP 後更新 `improving`、IIR 排除 `allNode`、ProbCut 深度與 TT 存值)

#### 1. 問題原理
1. **Step 10 NMP 結束後更新 `improving`（含後續 Lazy Eval 升級保護）**：
   * 對照 SF19 `improving |= ss->staticEval >= beta;`。
   * **關鍵細節**：在 Step 10 之後，Step 14 的 Capture FP 與 Quiet FP 可能會觸發 Lazy Eval 升級並呼叫 `compute_full_corrected_static_eval`，該函式會回傳新的 `improving` 並覆蓋掉變數。因此在每次 Step 10 之後的 `compute_full_corrected_static_eval` 後方，都必須補上 `if static_score >= beta: improving = True`。
2. **Step 11 IIR 排除 `allNode`**：
   * 對照 SF19 `if (!ss->followPV && !allNode && ...)`，加入 `and (is_pv or cut_node)`。
3. **Step 12 ProbCut 深度倒置與 TT 存值修正**：
   * 移除 `probcut_min_depth = probcut_r + 1`（避免 `improving == True` 時在 `depth = 4, 5` 反而不能進 ProbCut）。改用固定 `depth >= 3`（或 `PROBCUT_MIN_DEPTH`），內部計算 `probcut_depth = max(0, depth - probcut_r)`；當 `probcut_depth > 0` 時才跑第 2 階段 `_search(probcut_depth)`，若 `probcut_depth == 0` 則僅由第 1 階段 `quiescence_search` 決定。
   * ProbCut 成功時，寫入 `store_tt` 的分數應為**未扣減的 `pc_score`**（因為搜尋證明了 `>= probcut_beta`）與已知的 `raw_static_eval`，寫入 TT 後才回傳 `pc_score - (probcut_beta - beta)`。
4. **Step 13 TT ProbCut Shortcut 獨立於 `in_check` 與 `probcut_min_depth` 之外**：
   * 將 `ENABLE_PROBCUT_TT_SHORTCUT` 移出 `not is_currently_in_check` 與 `depth >= probcut_min_depth` 的外層限制，明確檢查 `tt_flag == TT_FLAG_BETA or tt_flag == TT_FLAG_EXACT` 且 `tt_depth >= depth - 4`。（註：保留 `not is_pv` 為本專案刻意採取的保守設計）。

#### 2. 修改位置與方法 (`chess_engine/classical/search.py`)
* **定位錨點 1**：搜尋 `# Step 11. Internal Iterative Reduction (IIR) (SF19 Step 11)`。
  ```python
    # 在 Step 11 之前補上:
    if not is_currently_in_check and static_score >= beta:
        improving = True

    # Step 11 條件加入 (is_pv or cut_node):
    if (tt_move == NO_MOVE and ENABLE_IIR and not is_exclusion_search
            and not is_currently_in_check and not follow_pv and (is_pv or cut_node)):
        if depth >= IIR_MIN_DEPTH:
            depth -= 1
  ```
* **定位錨點 2**：在 Step 14c Quiet FP 的 Lazy Eval 升級處（搜尋 `if margin > 0 and tt_refined_eval + margin < alpha:` 內的 `compute_full_corrected_static_eval`），在 `static_eval_is_full = True` 後方補上：
  ```python
                                if static_score >= beta:
                                    improving = True
  ```
* **定位錨點 3**：搜尋 `# Step 12. ProbCut (aligned to SF19 Step 12: after NMP and IIR)`。
  1. 先獨立執行 TT ProbCut Shortcut（固定 `tt_depth >= depth - 4`，且明確檢查 `tt_flag == TT_FLAG_BETA or tt_flag == TT_FLAG_EXACT`）：
     ```python
     _pc_style = search_context.tune[TUNE_PROBCUT_STYLE]
     if (search_context.enable_probcut and _pc_style != 2 and ENABLE_PROBCUT_TT_SHORTCUT
             and not is_pv and ply > 0 and not is_exclusion_search and tt_hit
             and abs(beta) < MATE_IN_MAX_PLY and abs(beta) < VALUE_KNOWN_WIN
             and not is_loss(beta)):
         probcut_beta_s12 = beta + PROBCUT_TT_SHORTCUT_MARGIN
         if (tt_flag == TT_FLAG_BETA or tt_flag == TT_FLAG_EXACT) and tt_depth >= depth - 4:
             tt_score_s12 = tt_score_raw
             if tt_score_s12 > MATE_IN_MAX_PLY:
                 tt_score_s12 -= ply
             elif tt_score_s12 < -MATE_IN_MAX_PLY:
                 tt_score_s12 += ply
             if fifty_move_scale < FIFTY_MOVE_MAX_SCALE and abs(tt_score_s12) < MATE_IN_MAX_PLY:
                 tt_score_s12 = tt_score_s12 * fifty_move_scale // FIFTY_MOVE_MAX_SCALE
             if (tt_score_s12 >= probcut_beta_s12
                     and not is_decisive(tt_score_s12) and not is_win(tt_score_s12)):
                 search_context.pv_table[ply, ply] = NO_MOVE
                 _diag_add(search_context, DIAG_PROBCUT)
                 tt_hits += np.uint64(1)
                 return (np.int32(probcut_beta_s12), tt_move, nodes_searched, quiescence_nodes, tt_hits)
     ```
  2. 接著進入完整 ProbCut（使用 `depth >= 3` 與 `probcut_depth = max(0, depth - probcut_r)`）：
     ```python
     probcut_r = PROBCUT_R_IMPROVING if improving else PROBCUT_R_NOT_IMPROVING
     probcut_depth = max(0, depth - probcut_r)
     if (search_context.enable_probcut and _pc_style != 2 and not is_pv and ply > 0
             and not is_exclusion_search and depth >= 3
             and abs(beta) < MATE_IN_MAX_PLY and abs(beta) < VALUE_KNOWN_WIN
             and not is_currently_in_check and not is_win(static_score) and not is_loss(beta)):
     ```
  3. 在 ProbCut 迴圈內，當 `pc_q_score >= probcut_beta` 通過後：
     ```python
                pc_score = pc_q_score
                if probcut_depth > 0:
                    search_context.reduction_stack[ply + 1] = 0
                    res_pc = _search(
                        piece_bbs, occupancy_bbs, game_state, _i32(probcut_depth),
                        _i32(-probcut_beta), _i32(-probcut_beta + 1), search_context, _i32(ply + 1),
                        _u16(NO_MOVE), _b_false(), not cut_node,
                    )
                    pc_score = -res_pc[0]
                    nodes_searched += res_pc[2]
                    quiescence_nodes += res_pc[3]
                    tt_hits += res_pc[4]

                search_context.move_stack[ply] = saved_pc_stack_move
                search_context.piece_stack[ply] = saved_pc_stack_piece
                unmake_move(piece_bbs, occupancy_bbs, game_state, pc_move, pc_unmake)

                if search_context.stop_flag[0]:
                    return (np.int32(0), NO_MOVE, nodes_searched, quiescence_nodes, tt_hits)

                if pc_score >= probcut_beta:
                    tt_store_score_pc = pc_score
                    if tt_store_score_pc > MATE_IN_MAX_PLY: tt_store_score_pc += ply
                    elif tt_store_score_pc < -MATE_IN_MAX_PLY: tt_store_score_pc -= ply
                    pc_tt_static = np.int16(32767)
                    if raw_static_eval != -INFINITY and abs(raw_static_eval) < MATE_SCORE - MAX_PLY:
                        pc_tt_static = np.int16(raw_static_eval)
                    store_tt(
                        search_context.transposition_table, tt_key, probcut_depth + 1,
                        tt_store_score_pc, pc_tt_static, TT_FLAG_BETA, pc_move,
                        search_context.tt_generation, tt_pv
                    )
                    if not is_win(pc_score) and not is_decisive(pc_score):
                        probcut_return_score = pc_score - (probcut_beta - beta)
                        _diag_add(search_context, DIAG_PROBCUT)
                        return (np.int32(probcut_return_score), pc_move, nodes_searched, quiescence_nodes, tt_hits)
     ```

---

## Phase B：行為與調參變更（需獨立跑 SPRT 驗證，勿與 Phase A 混併）

> [!WARNING]
> 以下兩項修改雖然對齊 Stockfish 19，但會**改變 Step 14 的剪枝基準與觸發深度範圍**。請在 Phase A 穩定後，以獨立分支進行 SPRT 對局測試。

### 任務 B1：Step 14 (Capture FP & Quiet FP) 改用 `static_score` 取代 `tt_refined_eval`
* **原理**：SF19 在 Step 14 使用的是 `ss->staticEval` (`static_score`)，而非經 TT bound 修正後的 `eval` (`tt_refined_eval`)。注意此修改為雙向影響（避免 `TT_FLAG_ALPHA` 壓低分數造成過度剪枝，也避免 `TT_FLAG_BETA` 拉高分數造成剪枝不足）。
* **修改位置**：在 `_search` 的 Step 14 Capture FP 與 Step 14c Quiet FP 中，將 `tt_refined_eval` 改為 `static_score`。

### 任務 B2：Step 14c Quiet FP 門檻改用 `quiet_lmr_d <= FP_MAX_DEPTH` 並共用 `quiet_lmr_d`
* **原理**：
  * 目前 Step 14c 檢查原始 `depth <= FP_MAX_DEPTH` (`6`)，導致 `depth = 7 ~ 12` 時即便 `fp_lmr_d` 很小也完全不執行 FP。改為檢查 `quiet_lmr_d <= FP_MAX_DEPTH` 會擴大深度 7~12 後段安靜著法的剪枝範圍，必須經 SPRT 驗證。
  * 同時可消除同迴圈內重複計算：在 Step 14c 前先算一次 `quiet_lmr_d`，**若觸發 Lazy Eval 升級（因 `improving` 可能改變）則在升級區塊內重算一次 `quiet_lmr_d`**，最後在 Step 14d (Quiet SEE) 直接沿用 `quiet_lmr_d`。

---

## Phase C：P2 工具開關解耦與獨立 Bug 清理

### 任務 C1：[P2] 解耦 Step 14 外層的 `search_context.enable_see_pruning` 開關
* **定位錨點**：`_search` 的 `# --- NEW: Shallow Depth Pruning (Stockfish Step 14) ---`。
* **修改方式**：將 `search_context.enable_see_pruning` 從 Step 14 最外層 `if` 移除，改為分別加在 **Capture SEE**、**Quiet Check SEE** 與 **Step 14d Quiet SEE** 各自的 `if` 條件上，使關閉 SEE pruning 做消融實驗時不會連帶關閉 LMP、History Pruning 與 FP。

### 任務 C2：[獨立任務] 移除 `_search` 末端重複複製的 Continuation Correction History 更新區塊
* **定位錨點**：在 `_search` 末端的 `# --- Update Correction History ---` 區塊內，搜尋註解 `# Update Continuation Correction History (duplicate block kept for parity)`。
* **問題說明**：第 1 份 `# Update Continuation Correction History` 與緊接在後的第 2 份 `# Update Continuation Correction History (duplicate block kept for parity)` 程式碼完全相同，導致每次更新 Continuation Correction History 時都重複套用了 2 倍的 bonus。建議作為獨立任務刪除第 2 份重複區塊並進行 A/B 測試。