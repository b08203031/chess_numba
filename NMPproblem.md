# HCE 搜尋引擎 NMP 與 Phase 3 剪枝流程修復計畫書（AI Agent 執行規格書）

## 0. 專案背景與目標
本計畫書旨在修復 `chess_engine/classical/search.py` 中關於 **Null Move Pruning (NMP)** 與 **Phase 3 節點級剪枝（Node-Level Pruning）** 的邏輯缺陷、執行順序錯置與算力浪費問題，並對齊 **Stockfish 19 (`search.cpp` Step 6 ~ Step 12)** 標準架構，同時保留本引擎特有的 **HCE（手工特徵評估）兩階段懶惰評估（`lazy=True` / `lazy=False`）** 保護機制。

* **唯一需修改檔案**：`chess_engine/classical/search.py`
* **不需修改檔案**：`board_operations.py`、`constants.py`、`engine_types.py`、`evaluation.py`

---

## 1. 嚴格保護規範（Guardrails — 絕對禁止修改項目）

執行修復的 AI Agent **必須遵守以下限制**，切勿誤刪 HCE 專用保護邏輯：

1. **禁止移除 NMP 中的 `not is_win(static_score)` 與 `not is_win(null_move_score)`**：
   * 在本引擎中，`VALUE_KNOWN_WIN = 10000`（定義於 `constants.py`）**僅**由 `endgame.py` 的特殊必勝殘局評估函數（如 KXK、KBNK、KPK、KRKP）回傳，一般中殘局 HCE 分數絕不會達到 `10000`。
   * 若在 `>= 10000` 的特殊必勝殘局中允許 NMP 截斷，引擎會在逼王入角的過程中直接依賴 `+10000` 的人造靜態分數進行空著截斷，導致無法搜尋出實質將死（`MATE_SCORE = 30000`）。因此這兩個 `not is_win(...)` 條件必須保留。
2. **禁止修改 `board_operations.py` 的 `make_null_move`**：
   * `make_null_move` 內部已執行 `game_state[3] += 1`（半步鐘遞增），這確保了 `_search` 重複局面偵測中 `hist_to_check = halfmove_clock - ply` 的奇偶校驗（Parity）不會錯位，同時維持了 GHI 保護 `get_tt_key(zobrist_key, halfmove_clock)` 的雜湊連續性。
3. **遵守 Numba `@njit` 型別一致性**：
   * `search.py` 為 Numba AOT/JIT 編譯模組，所有新增或修改的變數必須保持固定型別（例如位元棋盤使用 `np.uint64`、分數使用 `np.int32`），且不可改變 `_search` 的函數簽名。

---

## 2. 修改任務清單與實作規格（共 5 項任務）

---

### 任務 1：移除 `compute_full_corrected_static_eval` 中過早強制計算 Pin 的邏輯

* **修改位置**：`chess_engine/classical/search.py` 第 175–194 行（函數 `compute_full_corrected_static_eval`）
* **問題原理**：
  在 `evaluation.py` 第 924–934 行中，當 `_evaluate_position_jit(..., lazy=False, ...)` 發現雙方分數差距極大（`abs(v) > LAZY_EVAL_THRESHOLD + npm // 64`）時，會觸發動態提早退出（Dynamic Lazy Exit）並回傳 `PINNED_UNCOMPUTED_SENTINEL` 以節省算力。
  然而目前 `compute_full_corrected_static_eval` 卻立刻檢查 `if pinned_white == PINNED_UNCOMPUTED_SENTINEL:` 並強制呼叫 `get_pinned_pieces` 兩次。由於這類大幅領先的節點在 RFP 或 NMP 中幾乎都會直接截斷返回，提前計算 Pin 屬於 100% 的算力浪費。
* **修復方法**：
  刪除 `compute_full_corrected_static_eval` 內部的 `if pinned_white == PINNED_UNCOMPUTED_SENTINEL:` 區塊，保留 `PINNED_UNCOMPUTED_SENTINEL` 原樣回傳，交由後續真正需要使用 Pin 的 ProbCut 或 Move Ordering 階段再延遲計算並快取。

#### 【程式碼替換規格 — 任務 1】
尋找以下原始程式碼（約第 175–194 行）：
```python
@numba.njit(cache=True, nogil=True, boundscheck=False, fastmath=True)
def compute_full_corrected_static_eval(piece_bbs, occupancy_bbs, game_state, search_context, ply):
    raw_eval, pinned_white, pinned_black = _evaluate_position_jit(piece_bbs, occupancy_bbs, game_state, False, search_context)
    if pinned_white == PINNED_UNCOMPUTED_SENTINEL:
        pinned_white = get_pinned_pieces(piece_bbs, occupancy_bbs, WHITE)
        pinned_black = get_pinned_pieces(piece_bbs, occupancy_bbs, BLACK)
    corrected_eval = apply_correction_history_score(game_state, search_context, raw_eval, ply)
    search_context.static_eval_stack[ply] = corrected_eval

    improving = False
    if ply >= 2:
        ref_eval = search_context.static_eval_stack[ply - 2]
        # V3 3.5: Fallback to ply-4 when ply-2 was null move (static_eval = -INFINITY)
        if ref_eval == -INFINITY and ply >= 4:
            ref_eval = search_context.static_eval_stack[ply - 4]
        if ref_eval != -INFINITY and corrected_eval > ref_eval:
            improving = True

    return raw_eval, corrected_eval, improving, pinned_white, pinned_black
```

替換為：
```python
@numba.njit(cache=True, nogil=True, boundscheck=False, fastmath=True)
def compute_full_corrected_static_eval(piece_bbs, occupancy_bbs, game_state, search_context, ply):
    raw_eval, pinned_white, pinned_black = _evaluate_position_jit(piece_bbs, occupancy_bbs, game_state, False, search_context)
    corrected_eval = apply_correction_history_score(game_state, search_context, raw_eval, ply)
    search_context.static_eval_stack[ply] = corrected_eval

    improving = False
    if ply >= 2:
        ref_eval = search_context.static_eval_stack[ply - 2]
        # V3 3.5: Fallback to ply-4 when ply-2 was null move (static_eval = -INFINITY)
        if ref_eval == -INFINITY and ply >= 4:
            ref_eval = search_context.static_eval_stack[ply - 4]
        if ref_eval != -INFINITY and corrected_eval > ref_eval:
            improving = True

    return raw_eval, corrected_eval, improving, pinned_white, pinned_black
```

---

### 任務 2：分離純靜態評估 `static_score` (`ss->staticEval`) 與 TT 修正評估 `tt_refined_eval` (`eval`)

* **修改位置**：`chess_engine/classical/search.py` 第 698–840 行（Phase 3 靜態評估、Dissonance、Razoring、RFP）
* **問題原理**：
  在 Stockfish 19 中，`ss->staticEval`（僅含 Correction History 修正的靜態評估）與 `eval`（被 TT bound 修正後的評估值）是兩個獨立變數。
  目前程式碼在 `trusted_tt_score` 成立時，直接用 `tt_score` 覆蓋了 `static_score` 並寫入 `static_eval_stack[ply]`，導致以下連鎖錯誤：
  1. **Dissonance 失效**：第 764 行計算 `dissonance = abs(tt_score - static_score)` 時，因為 `static_score` 已被覆蓋為 `tt_score`，`dissonance` 恆為 `0`。
  2. **RFP 內層丟失 TT 修正**：RFP 外層判斷使用了含 TT 修正的 `static_score`，但內層呼叫 `compute_full_corrected_static_eval` 升級為完整 HCE 評估後，`static_score` 被重置為不含 TT 修正的分數，導致內外層判斷基準不一。
  3. **NMP 外層誤觸昂貴的完整 HCE 評估**：NMP 內層與 SF19 相同，只看純靜態評估；但外層若使用被 `tt_score` 墊高的值，會在 `tt_score >= nmp_threshold > static_score` 時白白觸發一次完整的 `_evaluate_position_jit(..., False, ...)`，然後在內層失敗。
  4. **污染歷史修正與 LMR**：`improving`、`opponent_worsening`、LMR 的 `correction_mag = abs(static_score - raw_static_eval)`（第 1421 行），以及搜尋結尾的 Correction History 更新 `diff = max_eval - static_score`（第 1732 行），若未觸發完整重算，都會誤把 `tt_score` 當作靜態評估。
* **修復方法**：
  1. 將 `static_score` 嚴格固定為「純靜態評估 + Correction History」，並存入 `static_eval_stack[ply]`。
  2. 新增 `tt_refined_eval` 變數（對應 SF19 的 `eval`），初始等於 `static_score`，若 `trusted_tt_score` 成立則以 `tt_score_adj` 修正。
  3. 每當呼叫 `compute_full_corrected_static_eval` 將 `static_score` 升級為完整 HCE 評估時，同步以 `trusted_tt_score` 規則更新 `tt_refined_eval`。
  4. **Step 8 (Razoring)** 與 **Step 9 (RFP)** 使用 `tt_refined_eval`；其餘模組（`improving`、`opponent_worsening`、`dissonance`、**Step 10 NMP**、**Step 12 ProbCut**、**Step 14c FP**、**LMR**、**Correction History**）一律使用 `static_score`。

---

### 任務 3：對齊 SF19 剪枝順序（Step 10 NMP $\rightarrow$ `improving` 更新 $\rightarrow$ Step 11 IIR $\rightarrow$ Step 12 ProbCut）與共用 Pin 快取

* **修改位置**：`chess_engine/classical/search.py` 第 840–1140 行
* **問題原理**：
  1. **剪枝順序顛倒**：目前順序為 `ProbCut -> NMP -> IIR`，而 SF19 標準順序為 **`Step 10 (NMP) -> improving |= (staticEval >= beta) -> Step 11 (IIR) -> Step 12 (ProbCut)`**。
     NMP 只需極低成本的空著搜尋（無需生成著法、無需算 Pin 或 SEE）；ProbCut 卻需要生成所有吃子、計算 Pin、跑 SEE、執行 `make_move`/`unmake_move`、跑 `qsearch` 及 `depth - probcut_r` 搜尋。把 ProbCut 放在 NMP 之前，一方面浪費算力，另一方面在 `depth >= 9`（超過 `RFP_MAX_DEPTH = 8`）時，ProbCut 拿到的 `static_score` 只是粗糙的 `lazy=True`（僅材質+PST）分數，導致 `see_threshold_pc = probcut_beta - static_score` 嚴重失準！
  2. **缺少 NMP 後的 `improving` 更新**：SF19 在 Step 10 結束後會執行 `improving |= (ss->staticEval >= beta)`，讓後續的 ProbCut（決定 `probcut_r` 減深幅度與 margin）、LMP、FP、LMR 受惠。
  3. **Pin 重複計算**：在舊順序中，ProbCut 計算完 `pc_pinned_w` / `pc_pinned_b` 後未存入 `cached_pinned_white` / `cached_pinned_black`，導致後續 NMP 或 Move Ordering 再算第二次 `get_pinned_pieces`。將 NMP 移到 ProbCut 之前，並讓 ProbCut 與 Move Ordering 透過 `cached_pinned_white != PINNED_UNCOMPUTED_SENTINEL` 共用快取，即可保證每個節點最多只計算一次 Pin。

---

### 任務 4：修復 NMP Verification Search 的 `nmp_min_ply` 邊界失效 Bug 與狀態還原

* **修改位置**：`chess_engine/classical/search.py` NMP 驗證搜尋區塊（原第 1050–1095 行）
* **問題原理**：
  1. **`nmp_min_ply` 整數除法截斷為 0 的 Bug**：
     根據 `constants.py` 的參數（`NMP_VERIFICATION_DEPTH = 16`, `NMP_R_BASE = 7`, `NMP_R_DIV = 3`, `NMP_EVAL_MARGIN_MAX = 4`），當 `depth = 16` 且 `eval_margin` 為 `3` 或 `4` 時，`nmp_reduction` 為 `15` 或 `16`，使得 `max(1, depth - nmp_reduction)` 等於 `1`。
     此時原程式碼 `trunc_div(3 * max(1, depth - nmp_reduction), 4)` 算出來是 `3 // 4 = 0`，導致 `search_context.nmp_min_ply = max(1, ply + 0) = ply`！
     當 `res_verify`（在 `ply` 執行）展開其子節點（位於 `ply + 1`）時，檢查 `ply >= search_context.nmp_min_ply` 變成 `ply + 1 >= ply`（恆為 `True`），導致**驗證搜尋子樹完全沒有禁用 NMP**！
     同理，當 `depth - nmp_reduction == 2` 時，`trunc_div(3 * 2, 4) = 1`，`nmp_min_ply = ply + 1`，而 `res_verify` 的第一層子節點（位於 `ply + 1`）恰好是 `cut_node = True`，`(ply + 1) >= (ply + 1)` 依然成立，第一層子節點會立刻觸發 NMP。
  2. **同 `ply` 呼叫 `res_verify` 的堆疊污染**：
     因為 `res_verify = _search(..., _i32(ply), ...)` 使用同一個 `ply`，它會改寫 `move_stack[ply]`、`piece_stack[ply]`、`move_count_stack[ply]` 與 `move_is_capture_stack[ply]`。若驗證失敗（`verify_score < beta`）繼續執行當前節點，必須將這些陣列在 `[ply]` 的狀態還原。
* **修復方法**：
  * 將禁用跨度設為 `verify_disabled_span = max(2, trunc_div(3 * max(1, depth - nmp_reduction), 4))`，並設定 `search_context.nmp_min_ply = ply + verify_disabled_span`（確保至少禁用至 `ply + 2`，擋住驗證搜尋的第一層 `cut_node` 子節點）。
  * 在 `res_verify` 前後保存並還原 `move_count_stack[ply]`、`move_is_capture_stack[ply]`，並將 `move_stack[ply]` 重置為 `NO_MOVE`、`piece_stack[ply]` 重置為 `-1`。

---

#### 【程式碼替換規格 — 合併執行任務 2、3、4】
尋找 `chess_engine/classical/search.py` 第 698–1140 行（從 `# --- H2: Static Evaluation & Improving Flag` 到 Move Ordering 的 `pinned_black = get_pinned_pieces(...)`）：

```python
    # --- H2: Static Evaluation & Improving Flag (computed BEFORE any pruning) ---
    static_score = -INFINITY
    improving = False
    
    # Correction History Adjustment
    raw_static_eval = -INFINITY
    static_eval_is_full = False
    cached_pinned_white = np.uint64(0)
    cached_pinned_black = np.uint64(0)

    if not is_currently_in_check:
        if tt_hit and tt_static_eval_raw != 32767:
            raw_static_eval = np.int32(tt_static_eval_raw)
        else:
            raw_static_eval, _, _ = _evaluate_position_jit(piece_bbs, occupancy_bbs, game_state, True, search_context)

        static_score = apply_correction_history_score(game_state, search_context, raw_static_eval, ply)
        
        # --- H2 Enhancement: Refine static_score using trusted TT Score Bounds ---
        if tt_hit and not is_exclusion_search:
            tt_score = tt_score_raw
            
            # Adjust mate scores relative to current ply
            if tt_score > MATE_IN_MAX_PLY: tt_score -= ply
            elif tt_score < -MATE_IN_MAX_PLY: tt_score += ply

            trusted_bound_depth = tt_depth >= max(1, depth - 2)
            trusted_exact_depth = tt_depth >= depth
            trusted_tt_score = trusted_bound_depth and abs(tt_score) < MATE_IN_MAX_PLY

            if trusted_tt_score:
                # If TT says score is at least tt_score (LOWER), and tt_score > static_score
                if tt_flag == TT_FLAG_BETA and tt_score > static_score:
                    static_score = tt_score
                # If TT says score is at most tt_score (UPPER), and tt_score < static_score
                elif tt_flag == TT_FLAG_ALPHA and tt_score < static_score:
                    static_score = tt_score
                # Exact scores may replace static eval only when they are deep enough for this node.
                elif tt_flag == TT_FLAG_EXACT and trusted_exact_depth:
                    static_score = tt_score
        
        search_context.static_eval_stack[ply] = static_score

        # V3 3.5: Improved improving calculation — fallback to ply-4 when ply-2 was null move
        if ply >= 2:
            ref_eval = search_context.static_eval_stack[ply - 2]
            if ref_eval == -INFINITY and ply >= 4:
                ref_eval = search_context.static_eval_stack[ply - 4]
            if ref_eval != -INFINITY and static_score > ref_eval:
                improving = True
    else:
        search_context.static_eval_stack[ply] = -INFINITY

    # --- opponentWorsening & Hindsight Depth Adjustment ---
    opponent_worsening = False
    if not is_currently_in_check and ply > 0:
        parent_eval = search_context.static_eval_stack[ply - 1]
        if static_score != -INFINITY and parent_eval != -INFINITY:
            opponent_worsening = static_score > -parent_eval
            
            prior_reduction = search_context.reduction_stack[ply]
            if prior_reduction >= HINDSIGHT_REDUCE_MIN_PRIOR and not opponent_worsening:
                depth += 1
            if prior_reduction >= HINDSIGHT_INCREASE_MIN_PRIOR and depth >= 2:
                if static_score + parent_eval > HINDSIGHT_EVAL_SUM_MARGIN:
                    depth -= 1

    if depth <= 0:
        search_context.pv_table[ply, ply] = NO_MOVE
        eval_score, q_nodes = quiescence_search(
            piece_bbs, occupancy_bbs, game_state, alpha, beta, ply, search_context, _i32(0)
        )
        return (np.int32(eval_score), NO_MOVE, nodes_searched, q_nodes, tt_hits)

    # --- Dissonance Calculation (DGP Algorithm) ---
    dissonance = 0
    if tt_hit and not is_currently_in_check and static_score != -INFINITY:
        # Check if depth is reasonably close to current depth to trust the TT score
        if tt_depth >= depth - DISSONANCE_TT_DEPTH_SLACK:
            tt_score = tt_score_raw
            # Avoid using mate scores for dissonance calculation
            if abs(tt_score) < MATE_IN_MAX_PLY and abs(static_score) < MATE_IN_MAX_PLY:
                dissonance = abs(tt_score - static_score)

    total_piece_count = np.int32(0)
    side_non_pawn_count = np.int32(0)
    if not is_currently_in_check:
        total_piece_count = count_bits(occupancy_bbs[WHITE] | occupancy_bbs[BLACK])

        stm_piece_offset = 0 if game_state[0] == WHITE else 6
        side_non_pawn_count = count_bits(
            piece_bbs[stm_piece_offset + KNIGHT] |
            piece_bbs[stm_piece_offset + BISHOP] |
            piece_bbs[stm_piece_offset + ROOK] |
            piece_bbs[stm_piece_offset + QUEEN]
        )

    low_material_pruning_guard = total_piece_count <= LOW_MATERIAL_PRUNING_PIECE_COUNT

    # PR-C: if lazy/TT static is already near known-win band, force full HCE before pruning
    if (not is_currently_in_check and not static_eval_is_full
            and (is_win(static_score) or is_loss(static_score)
                 or abs(static_score) >= (VALUE_KNOWN_WIN - FULL_EVAL_KNOWN_WIN_MARGIN))):
        raw_static_eval, static_score, improving, cached_pinned_white, cached_pinned_black = compute_full_corrected_static_eval(
            piece_bbs, occupancy_bbs, game_state, search_context, ply
        )
        static_eval_is_full = True

    # =====================================================================
    # --- Node-Level Pruning (H3: ALL guarded by not is_currently_in_check) ---
    # =====================================================================
    if not is_currently_in_check and not is_pv and not is_exclusion_search:
        # --- Step 8: Razoring (SF19 Step 8: eval < alpha - 482 * depth * depth) ---
        if (search_context.enable_razoring and depth <= RAZORING_MAX_DEPTH and not low_material_pruning_guard
                and not is_win(static_score) and not is_loss(alpha)
                and abs(alpha) < VALUE_KNOWN_WIN):
            razor_margin = search_context.tune[TUNE_RAZOR_COEFF] * depth * depth
            if static_score < alpha - razor_margin:
                if not static_eval_is_full:
                    raw_static_eval, static_score, improving, cached_pinned_white, cached_pinned_black = compute_full_corrected_static_eval(
                        piece_bbs, occupancy_bbs, game_state, search_context, ply
                    )
                    static_eval_is_full = True
                if (static_score < alpha - razor_margin
                        and not is_win(static_score) and not is_loss(alpha)):
                    search_context.pv_table[ply, ply] = NO_MOVE
                    razor_score, child_q_nodes = quiescence_search(
                        piece_bbs, occupancy_bbs, game_state, alpha, beta, ply, search_context, _i32(0)
                    )
                    quiescence_nodes += child_q_nodes
                    _diag_add(search_context, DIAG_RAZOR_CUT)
                    return (np.int32(razor_score), NO_MOVE, nodes_searched, quiescence_nodes, tt_hits)

        # --- M5: Reverse Futility Pruning (SF19 Step 9: !tt_pv, !tt_move or tt_capture) ---
        # P2: never return known-win / loss-band static as a proven cutoff (SF11/18/19).
        if (search_context.enable_rfp and depth <= RFP_MAX_DEPTH and not low_material_pruning_guard
                and not tt_pv and (tt_move == NO_MOVE or tt_capture)
                and not is_win(static_score) and not is_loss(beta)):
            # SF11: futility_margin = 217 * (depth - improving)
            # Larger margin when NOT improving → harder to RFP (safer). Do not invert this.
            rfp_multiplier = search_context.tune[TUNE_RFP_MULT]
            if not tt_hit:
                rfp_multiplier += RFP_NO_TT_PENALTY  # small safety when no TT
            rfp_depth_factor = depth - (1 if improving else 0)
            if rfp_depth_factor < 1:
                rfp_depth_factor = 1
            rfp_margin = rfp_multiplier * rfp_depth_factor
                
            if static_score - rfp_margin >= beta:
                if not static_eval_is_full:
                    raw_static_eval, static_score, improving, cached_pinned_white, cached_pinned_black = compute_full_corrected_static_eval(piece_bbs, occupancy_bbs, game_state, search_context, ply)
                    static_eval_is_full = True
                    rfp_depth_factor = depth - (1 if improving else 0)
                    if rfp_depth_factor < 1:
                        rfp_depth_factor = 1
                    rfp_margin = rfp_multiplier * rfp_depth_factor
                # Re-check known-win after full eval (specialized EG may surface)
                if (static_score - rfp_margin >= beta
                        and not is_win(static_score) and not is_loss(beta)):
                    _diag_add(search_context, DIAG_RFP_CUT)
                    # SF19 Step 9: Fail-soft score blending via TUNE_RFP_BLEND_BETA
                    w_beta = np.int64(search_context.tune[TUNE_RFP_BLEND_BETA])
                    w_stat = np.int64(1024) - w_beta
                    rfp_score = np.int32(trunc_div(
                        w_beta * np.int64(beta) + w_stat * np.int64(static_score),
                        np.int64(1024)
                    ))
                    return (rfp_score, NO_MOVE, nodes_searched, quiescence_nodes, tt_hits)

    # --- ProbCut (must be before NMP, guarded by H3) ---
... (中略：原 ProbCut、NMP、IIR、Move Ordering 直到第 1136 行) ...
    # Optimization: Reuse pinned pieces from full evaluation if available
    if static_eval_is_full and cached_pinned_white != PINNED_UNCOMPUTED_SENTINEL:
        pinned_white = cached_pinned_white
        pinned_black = cached_pinned_black
    else:
        pinned_white = get_pinned_pieces(piece_bbs, occupancy_bbs, WHITE)
        pinned_black = get_pinned_pieces(piece_bbs, occupancy_bbs, BLACK)
```

將上述整個區段替換為以下程式碼：
```python
    # --- H2: Static Evaluation & Improving Flag (computed BEFORE any pruning) ---
    # static_score: 對應 SF19 ss->staticEval (純靜態評估 + Correction History，絕不被 tt_score 覆蓋)
    # tt_refined_eval: 對應 SF19 eval (套用可信 TT Bound 修正後的評估，專供 Razoring 與 RFP 使用)
    static_score = np.int32(-INFINITY)
    tt_refined_eval = np.int32(-INFINITY)
    improving = False
    
    # Correction History Adjustment
    raw_static_eval = np.int32(-INFINITY)
    static_eval_is_full = False
    cached_pinned_white = PINNED_UNCOMPUTED_SENTINEL
    cached_pinned_black = PINNED_UNCOMPUTED_SENTINEL

    trusted_tt_score = False
    trusted_exact_depth = False
    tt_score_adj = np.int32(0)

    if not is_currently_in_check:
        if tt_hit and tt_static_eval_raw != 32767:
            raw_static_eval = np.int32(tt_static_eval_raw)
        else:
            raw_static_eval, _, _ = _evaluate_position_jit(piece_bbs, occupancy_bbs, game_state, True, search_context)

        static_score = apply_correction_history_score(game_state, search_context, raw_static_eval, ply)
        tt_refined_eval = static_score
        
        # --- H2 Enhancement: Refine tt_refined_eval using trusted TT Score Bounds ---
        if tt_hit and not is_exclusion_search:
            tt_score_adj = tt_score_raw
            
            # Adjust mate scores relative to current ply
            if tt_score_adj > MATE_IN_MAX_PLY: tt_score_adj -= ply
            elif tt_score_adj < -MATE_IN_MAX_PLY: tt_score_adj += ply

            trusted_bound_depth = tt_depth >= max(1, depth - 2)
            trusted_exact_depth = tt_depth >= depth
            trusted_tt_score = trusted_bound_depth and abs(tt_score_adj) < MATE_IN_MAX_PLY

            if trusted_tt_score:
                # If TT says score is at least tt_score (LOWER), and tt_score > tt_refined_eval
                if tt_flag == TT_FLAG_BETA and tt_score_adj > tt_refined_eval:
                    tt_refined_eval = tt_score_adj
                # If TT says score is at most tt_score (UPPER), and tt_score < tt_refined_eval
                elif tt_flag == TT_FLAG_ALPHA and tt_score_adj < tt_refined_eval:
                    tt_refined_eval = tt_score_adj
                # Exact scores may replace static eval only when they are deep enough for this node.
                elif tt_flag == TT_FLAG_EXACT and trusted_exact_depth:
                    tt_refined_eval = tt_score_adj
        
        search_context.static_eval_stack[ply] = static_score

        # V3 3.5: Improved improving calculation — fallback to ply-4 when ply-2 was null move
        if ply >= 2:
            ref_eval = search_context.static_eval_stack[ply - 2]
            if ref_eval == -INFINITY and ply >= 4:
                ref_eval = search_context.static_eval_stack[ply - 4]
            if ref_eval != -INFINITY and static_score > ref_eval:
                improving = True
    else:
        search_context.static_eval_stack[ply] = -INFINITY

    # --- opponentWorsening & Hindsight Depth Adjustment ---
    opponent_worsening = False
    if not is_currently_in_check and ply > 0:
        parent_eval = search_context.static_eval_stack[ply - 1]
        if static_score != -INFINITY and parent_eval != -INFINITY:
            opponent_worsening = static_score > -parent_eval
            
            prior_reduction = search_context.reduction_stack[ply]
            if prior_reduction >= HINDSIGHT_REDUCE_MIN_PRIOR and not opponent_worsening:
                depth += 1
            if prior_reduction >= HINDSIGHT_INCREASE_MIN_PRIOR and depth >= 2:
                if static_score + parent_eval > HINDSIGHT_EVAL_SUM_MARGIN:
                    depth -= 1

    if depth <= 0:
        search_context.pv_table[ply, ply] = NO_MOVE
        eval_score, q_nodes = quiescence_search(
            piece_bbs, occupancy_bbs, game_state, alpha, beta, ply, search_context, _i32(0)
        )
        return (np.int32(eval_score), NO_MOVE, nodes_searched, q_nodes, tt_hits)

    # --- Dissonance Calculation (DGP Algorithm) ---
    dissonance = 0
    if tt_hit and not is_currently_in_check and static_score != -INFINITY:
        # Check if depth is reasonably close to current depth to trust the TT score
        if tt_depth >= depth - DISSONANCE_TT_DEPTH_SLACK:
            tt_score_d = tt_score_raw
            if tt_score_d > MATE_IN_MAX_PLY: tt_score_d -= ply
            elif tt_score_d < -MATE_IN_MAX_PLY: tt_score_d += ply
            # Avoid using mate scores for dissonance calculation
            if abs(tt_score_d) < MATE_IN_MAX_PLY and abs(static_score) < MATE_IN_MAX_PLY:
                dissonance = abs(tt_score_d - static_score)

    total_piece_count = np.int32(0)
    side_non_pawn_count = np.int32(0)
    if not is_currently_in_check:
        total_piece_count = count_bits(occupancy_bbs[WHITE] | occupancy_bbs[BLACK])

        stm_piece_offset = 0 if game_state[0] == WHITE else 6
        side_non_pawn_count = count_bits(
            piece_bbs[stm_piece_offset + KNIGHT] |
            piece_bbs[stm_piece_offset + BISHOP] |
            piece_bbs[stm_piece_offset + ROOK] |
            piece_bbs[stm_piece_offset + QUEEN]
        )

    low_material_pruning_guard = total_piece_count <= LOW_MATERIAL_PRUNING_PIECE_COUNT

    # PR-C: if lazy/TT static is already near known-win band, force full HCE before pruning
    if (not is_currently_in_check and not static_eval_is_full
            and (is_win(static_score) or is_loss(static_score) or is_win(tt_refined_eval) or is_loss(tt_refined_eval)
                 or abs(static_score) >= (VALUE_KNOWN_WIN - FULL_EVAL_KNOWN_WIN_MARGIN))):
        raw_static_eval, static_score, improving, cached_pinned_white, cached_pinned_black = compute_full_corrected_static_eval(
            piece_bbs, occupancy_bbs, game_state, search_context, ply
        )
        static_eval_is_full = True
        tt_refined_eval = static_score
        if trusted_tt_score:
            if tt_flag == TT_FLAG_BETA and tt_score_adj > tt_refined_eval:
                tt_refined_eval = tt_score_adj
            elif tt_flag == TT_FLAG_ALPHA and tt_score_adj < tt_refined_eval:
                tt_refined_eval = tt_score_adj
            elif tt_flag == TT_FLAG_EXACT and trusted_exact_depth:
                tt_refined_eval = tt_score_adj

    # =====================================================================
    # --- Node-Level Pruning (H3: ALL guarded by not is_currently_in_check) ---
    # =====================================================================
    if not is_currently_in_check and not is_pv and not is_exclusion_search:
        # --- Step 8: Razoring (SF19 Step 8: eval < alpha - 482 * depth * depth) ---
        if (search_context.enable_razoring and depth <= RAZORING_MAX_DEPTH and not low_material_pruning_guard
                and not is_win(tt_refined_eval) and not is_loss(alpha)
                and abs(alpha) < VALUE_KNOWN_WIN):
            razor_margin = search_context.tune[TUNE_RAZOR_COEFF] * depth * depth
            if tt_refined_eval < alpha - razor_margin:
                if not static_eval_is_full:
                    raw_static_eval, static_score, improving, cached_pinned_white, cached_pinned_black = compute_full_corrected_static_eval(
                        piece_bbs, occupancy_bbs, game_state, search_context, ply
                    )
                    static_eval_is_full = True
                    tt_refined_eval = static_score
                    if trusted_tt_score:
                        if tt_flag == TT_FLAG_BETA and tt_score_adj > tt_refined_eval:
                            tt_refined_eval = tt_score_adj
                        elif tt_flag == TT_FLAG_ALPHA and tt_score_adj < tt_refined_eval:
                            tt_refined_eval = tt_score_adj
                        elif tt_flag == TT_FLAG_EXACT and trusted_exact_depth:
                            tt_refined_eval = tt_score_adj
                if (tt_refined_eval < alpha - razor_margin
                        and not is_win(tt_refined_eval) and not is_loss(alpha)):
                    search_context.pv_table[ply, ply] = NO_MOVE
                    razor_score, child_q_nodes = quiescence_search(
                        piece_bbs, occupancy_bbs, game_state, alpha, beta, ply, search_context, _i32(0)
                    )
                    quiescence_nodes += child_q_nodes
                    _diag_add(search_context, DIAG_RAZOR_CUT)
                    return (np.int32(razor_score), NO_MOVE, nodes_searched, quiescence_nodes, tt_hits)

        # --- Step 9: Reverse Futility Pruning (SF19 Step 9: !tt_pv, !tt_move or tt_capture) ---
        # P2: never return known-win / loss-band static as a proven cutoff (SF11/18/19).
        if (search_context.enable_rfp and depth <= RFP_MAX_DEPTH and not low_material_pruning_guard
                and not tt_pv and (tt_move == NO_MOVE or tt_capture)
                and not is_win(tt_refined_eval) and not is_loss(beta)):
            # SF11: futility_margin = 217 * (depth - improving)
            # Larger margin when NOT improving → harder to RFP (safer). Do not invert this.
            rfp_multiplier = search_context.tune[TUNE_RFP_MULT]
            if not tt_hit:
                rfp_multiplier += RFP_NO_TT_PENALTY  # small safety when no TT
            rfp_depth_factor = depth - (1 if improving else 0)
            if rfp_depth_factor < 1:
                rfp_depth_factor = 1
            rfp_margin = rfp_multiplier * rfp_depth_factor
                
            if tt_refined_eval - rfp_margin >= beta:
                if not static_eval_is_full:
                    raw_static_eval, static_score, improving, cached_pinned_white, cached_pinned_black = compute_full_corrected_static_eval(
                        piece_bbs, occupancy_bbs, game_state, search_context, ply
                    )
                    static_eval_is_full = True
                    tt_refined_eval = static_score
                    if trusted_tt_score:
                        if tt_flag == TT_FLAG_BETA and tt_score_adj > tt_refined_eval:
                            tt_refined_eval = tt_score_adj
                        elif tt_flag == TT_FLAG_ALPHA and tt_score_adj < tt_refined_eval:
                            tt_refined_eval = tt_score_adj
                        elif tt_flag == TT_FLAG_EXACT and trusted_exact_depth:
                            tt_refined_eval = tt_score_adj
                    rfp_depth_factor = depth - (1 if improving else 0)
                    if rfp_depth_factor < 1:
                        rfp_depth_factor = 1
                    rfp_margin = rfp_multiplier * rfp_depth_factor
                # Re-check known-win after full eval (specialized EG may surface)
                if (tt_refined_eval - rfp_margin >= beta
                        and not is_win(tt_refined_eval) and not is_loss(beta)):
                    _diag_add(search_context, DIAG_RFP_CUT)
                    # SF19 Step 9: Fail-soft score blending via TUNE_RFP_BLEND_BETA
                    w_beta = np.int64(search_context.tune[TUNE_RFP_BLEND_BETA])
                    w_stat = np.int64(1024) - w_beta
                    rfp_score = np.int32(trunc_div(
                        w_beta * np.int64(beta) + w_stat * np.int64(tt_refined_eval),
                        np.int64(1024)
                    ))
                    return (rfp_score, NO_MOVE, nodes_searched, quiescence_nodes, tt_hits)

    # =========================================================================
    # Step 10. Null move search with verification search (aligned to SF19 Step 10)
    # SF19: cutNode && staticEval >= beta - 13*d - 47*imp + 365 && !excludedMove
    #       && non_pawn_material(us) && ply >= nmpMinPly && beta >= -2000
    # =========================================================================
    if (search_context.enable_nmp and cut_node and ply > 0 and not is_exclusion_search
            and depth >= NMP_MIN_DEPTH and not is_currently_in_check
            and search_context.move_stack[ply - 1] != NO_MOVE
            and side_non_pawn_count >= NMP_MIN_SIDE_NON_PAWNS
            and ply >= search_context.nmp_min_ply
            and beta >= NMP_BETA_FLOOR):

        _diag_add(search_context, DIAG_NMP_ELIGIBLE)

        # 1. Linear gate with Lazy Evaluation (uses pure static_score, aligned with SF19 ss->staticEval)
        nmp_threshold = (beta - search_context.tune[TUNE_NMP_DEPTH_COEF] * depth
                              - search_context.tune[TUNE_NMP_IMP_COEF] * (1 if improving else 0)
                              + search_context.tune[TUNE_NMP_BASE])

        if static_score >= nmp_threshold:
            if not static_eval_is_full:
                raw_static_eval, static_score, improving, cached_pinned_white, cached_pinned_black = compute_full_corrected_static_eval(
                    piece_bbs, occupancy_bbs, game_state, search_context, ply
                )
                static_eval_is_full = True
                tt_refined_eval = static_score
                if trusted_tt_score:
                    if tt_flag == TT_FLAG_BETA and tt_score_adj > tt_refined_eval:
                        tt_refined_eval = tt_score_adj
                    elif tt_flag == TT_FLAG_ALPHA and tt_score_adj < tt_refined_eval:
                        tt_refined_eval = tt_score_adj
                    elif tt_flag == TT_FLAG_EXACT and trusted_exact_depth:
                        tt_refined_eval = tt_score_adj
                nmp_threshold = (beta - search_context.tune[TUNE_NMP_DEPTH_COEF] * depth
                                      - search_context.tune[TUNE_NMP_IMP_COEF] * (1 if improving else 0)
                                      + search_context.tune[TUNE_NMP_BASE])

            if static_score >= nmp_threshold and not is_win(static_score):
                _diag_add(search_context, DIAG_NMP_GATE_PASS)

                nmp_saved_side = game_state[0]
                nmp_saved_ep   = game_state[2]
                nmp_saved_hmc  = game_state[3]
                nmp_saved_key  = game_state[4]

                search_context.move_stack[ply] = NO_MOVE
                search_context.piece_stack[ply] = -1

                make_null_move(game_state)
                _diag_add(search_context, DIAG_NMP_NULL_TRY)

                # 2. 動態減深 R (SF19: R_BASE + d//R_DIV + max(0, (static - beta)//EVAL_DIV))
                _eval_div = max(1, search_context.tune[TUNE_NMP_EVAL_DIV])
                _r_div = max(1, search_context.tune[TUNE_NMP_R_DIV])
                eval_margin = min(NMP_EVAL_MARGIN_MAX, max(0, (static_score - beta) // _eval_div))

                nmp_reduction = (search_context.tune[TUNE_NMP_R_BASE]
                                 + depth // _r_div
                                 + eval_margin)
                search_depth = max(0, depth - nmp_reduction)
                search_context.reduction_stack[ply + 1] = 0

                res_nm = _search(
                    piece_bbs, occupancy_bbs, game_state, _i32(search_depth),
                    _i32(-beta), _i32(-beta + 1), search_context, _i32(ply + 1),
                    _u16(NO_MOVE), _b_false(), not cut_node,
                )
                null_move_score = -res_nm[0]
                nodes_searched += res_nm[2]; quiescence_nodes += res_nm[3]; tt_hits += res_nm[4]

                game_state[0] = nmp_saved_side
                game_state[2] = nmp_saved_ep
                game_state[3] = nmp_saved_hmc
                game_state[4] = nmp_saved_key

                if search_context.stop_flag[0]:
                    return (np.int32(0), NO_MOVE, nodes_searched, quiescence_nodes, tt_hits)

                # 3. SF19: Do not return unproven mate or TB scores (!is_win).
                #    HCE is_win also covers the KNOWN_WIN band (superset of mate).
                if null_move_score >= beta and not is_win(null_move_score):
                    _diag_add(search_context, DIAG_NMP_NULL_FH)

                    # 4. SF19: Verification search at high depths
                    _verify_threshold = search_context.tune[TUNE_NMP_VERIFY_DEPTH]
                    if search_context.nmp_min_ply == 0 and depth >= _verify_threshold:
                        _diag_add(search_context, DIAG_NMP_VERIFY_TRY)

                        # Save ply state to avoid polluting search context on verify fail
                        saved_v_reduction = search_context.reduction_stack[ply]
                        saved_v_cutoff_cnt = search_context.cutoff_cnt[ply]
                        saved_v_child_cutoff_cnt = search_context.cutoff_cnt[ply + 1] if (ply + 1 < MAX_PLY) else np.int32(0)
                        saved_v_grandchild_cutoff_cnt = search_context.cutoff_cnt[ply + 2] if (ply + 2 < MAX_PLY) else np.int32(0)
                        saved_v_static_eval = search_context.static_eval_stack[ply]
                        saved_v_tt_hit = search_context.tt_hit_stack[ply]
                        saved_v_tt_pv = search_context.tt_pv_stack[ply]
                        saved_v_follow_pv = search_context.follow_pv_stack[ply]
                        saved_v_move_cnt = search_context.move_count_stack[ply]
                        saved_v_move_cap = search_context.move_is_capture_stack[ply]

                        # Ensure NMP is disabled at least through ply + 1 (the first cut_node child of verify search)
                        verify_disabled_span = max(2, trunc_div(3 * max(1, depth - nmp_reduction), 4))
                        search_context.nmp_min_ply = ply + verify_disabled_span
                        verification_depth = max(1, depth - nmp_reduction)
                        search_context.reduction_stack[ply] = 0

                        res_verify = _search(
                            piece_bbs, occupancy_bbs, game_state, _i32(verification_depth),
                            _i32(beta - 1), _i32(beta), search_context, _i32(ply),
                            _u16(NO_MOVE), _b_false(), _b_false(),
                        )
                        search_context.nmp_min_ply = 0

                        # Restore ply state
                        search_context.reduction_stack[ply] = saved_v_reduction
                        search_context.cutoff_cnt[ply] = saved_v_cutoff_cnt
                        if ply + 1 < MAX_PLY:
                            search_context.cutoff_cnt[ply + 1] = saved_v_child_cutoff_cnt
                        if ply + 2 < MAX_PLY:
                            search_context.cutoff_cnt[ply + 2] = saved_v_grandchild_cutoff_cnt
                        search_context.static_eval_stack[ply] = saved_v_static_eval
                        search_context.tt_hit_stack[ply] = saved_v_tt_hit
                        search_context.tt_pv_stack[ply] = saved_v_tt_pv
                        search_context.follow_pv_stack[ply] = saved_v_follow_pv
                        search_context.move_count_stack[ply] = saved_v_move_cnt
                        search_context.move_is_capture_stack[ply] = saved_v_move_cap
                        search_context.move_stack[ply] = NO_MOVE
                        search_context.piece_stack[ply] = -1
                        search_context.pv_table[ply, ply] = NO_MOVE

                        verify_score = res_verify[0]
                        nodes_searched += res_verify[2]; quiescence_nodes += res_verify[3]; tt_hits += res_verify[4]

                        if search_context.stop_flag[0]:
                            return (np.int32(0), NO_MOVE, nodes_searched, quiescence_nodes, tt_hits)

                        if verify_score < beta:
                            _diag_add(search_context, DIAG_NMP_VERIFY_FAIL)
                            null_move_score = verify_score

                    if null_move_score >= beta:
                        search_context.pv_table[ply, ply] = NO_MOVE
                        _diag_add(search_context, DIAG_NMP_CUT)
                        return (np.int32(null_move_score), NO_MOVE, nodes_searched, quiescence_nodes, tt_hits)

    # SF19 post-NMP improving update: improving |= ss->staticEval >= beta
    if not is_currently_in_check and static_score >= beta:
        improving = True

    # =========================================================================
    # Step 11. Internal Iterative Reduction (IIR) (SF19 Step 11)
    # =========================================================================
    if (tt_move == NO_MOVE and ENABLE_IIR and not is_exclusion_search
            and not is_currently_in_check and not follow_pv):
        if depth >= IIR_MIN_DEPTH:
            depth -= 1

    # =========================================================================
    # Step 12. ProbCut (aligned to SF19 Step 12: after NMP and IIR)
    # =========================================================================
    _pc_style = search_context.tune[TUNE_PROBCUT_STYLE]
    probcut_r = PROBCUT_R_IMPROVING if improving else PROBCUT_R_NOT_IMPROVING
    probcut_min_depth = probcut_r + 1
    if (search_context.enable_probcut and _pc_style != 2 and not is_pv and ply > 0
            and not is_exclusion_search and depth >= probcut_min_depth
            and abs(beta) < MATE_IN_MAX_PLY and abs(beta) < VALUE_KNOWN_WIN
            and not is_currently_in_check and not is_win(static_score) and not is_loss(beta)):
        # --- TT-based ProbCut Shortcut (SF 18/19 Step 12) ---
        if ENABLE_PROBCUT_TT_SHORTCUT and ply > 0:
            probcut_beta_s12 = beta + PROBCUT_TT_SHORTCUT_MARGIN
            if tt_flag == TT_FLAG_BETA or tt_flag == TT_FLAG_EXACT:
                if tt_depth >= depth - probcut_r:
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

        if _pc_style == 1:
            _pc_add = (PROBCUT_SF_BASE - PROBCUT_SF_IMPROVING * (1 if improving else 0)) * HCE_SCALE_NUM // HCE_SCALE_DEN
            if _pc_add < 1:
                _pc_add = 1
            probcut_beta = beta + _pc_add
        else:
            probcut_beta = beta + search_context.tune[TUNE_PROBCUT_MARGIN]
        
        # --- ProbCut TT Defense ---
        skip_probcut = False
        if tt_hit:
            tt_score_pc = tt_score_raw
            if tt_score_pc > MATE_IN_MAX_PLY: tt_score_pc -= ply
            elif tt_score_pc < -MATE_IN_MAX_PLY: tt_score_pc += ply
            
            if ((tt_flag == TT_FLAG_ALPHA or tt_flag == TT_FLAG_EXACT)
                    and tt_depth >= depth - probcut_r
                    and tt_score_pc < probcut_beta):
                skip_probcut = True
                
        if not skip_probcut:
            pc_move_count = generate_pseudo_legal_captures_buffer(piece_bbs, occupancy_bbs, game_state, search_context.moves_buffer, ply)
            
            if cached_pinned_white != PINNED_UNCOMPUTED_SENTINEL:
                pc_pinned_w = cached_pinned_white
                pc_pinned_b = cached_pinned_black
            else:
                pc_pinned_w = get_pinned_pieces(piece_bbs, occupancy_bbs, WHITE)
                pc_pinned_b = get_pinned_pieces(piece_bbs, occupancy_bbs, BLACK)
                cached_pinned_white = pc_pinned_w
                cached_pinned_black = pc_pinned_b

            score_captures_lazy(
                piece_bbs, occupancy_bbs, game_state,
                search_context.moves_buffer[ply],
                search_context.move_scores[ply],
                0, pc_move_count, search_context,
                pc_pinned_w, pc_pinned_b,
                ply
            )

            pc_moves = search_context.moves_buffer[ply]
            pc_scores = search_context.move_scores[ply]
    
            for i in range(pc_move_count):
                best_idx = i
                for j in range(i + 1, pc_move_count):
                    if pc_scores[j] > pc_scores[best_idx]:
                        best_idx = j
                pc_moves[i], pc_moves[best_idx] = pc_moves[best_idx], pc_moves[i]
                pc_scores[i], pc_scores[best_idx] = pc_scores[best_idx], pc_scores[i]
    
                pc_move = pc_moves[i]
                if pc_move == excluded_move:
                    continue
                pc_from = get_from_square(pc_move)
                pc_to = get_to_square(pc_move)
    
                see_threshold_pc = probcut_beta - static_score if static_score != -INFINITY else 0
                _diag_add(search_context, DIAG_PROBCUT_SEE_TRY)
                if not _see_ge_jit(piece_bbs, occupancy_bbs, game_state[0], pc_from, pc_to, see_threshold_pc, pc_pinned_w, pc_pinned_b, -1, -1, get_promotion_piece(pc_move) + 1):
                    continue
    
                pc_original_side = game_state[0]
                pc_unmake = make_move(piece_bbs, occupancy_bbs, game_state, pc_move)
    
                pc_king_bb = piece_bbs[5] if (1 - game_state[0]) == 0 else piece_bbs[11]
                pc_king_sq = get_lsb_index(pc_king_bb) if pc_king_bb else 0
                if is_square_attacked(piece_bbs, occupancy_bbs, pc_king_sq, game_state[0]):
                    unmake_move(piece_bbs, occupancy_bbs, game_state, pc_move, pc_unmake)
                    continue
    
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
                search_context.reduction_stack[ply + 1] = 0
                res_pc = _search(
                    piece_bbs, occupancy_bbs, game_state, _i32(depth - probcut_r),
                    _i32(-probcut_beta), _i32(-probcut_beta + 1), search_context, _i32(ply + 1),
                    _u16(NO_MOVE), _b_false(), not cut_node,
                )
                search_context.move_stack[ply] = saved_pc_stack_move
                search_context.piece_stack[ply] = saved_pc_stack_piece
                pc_score = -res_pc[0]
                
                child_pc_nodes = res_pc[2]; child_pc_q_nodes = res_pc[3]
                child_pc_tth = res_pc[4]
                
                nodes_searched += child_pc_nodes; quiescence_nodes += child_pc_q_nodes
                tt_hits += child_pc_tth

                unmake_move(piece_bbs, occupancy_bbs, game_state, pc_move, pc_unmake)

                if search_context.stop_flag[0]:
                    return (np.int32(0), NO_MOVE, nodes_searched, quiescence_nodes, tt_hits)
    
                if pc_score >= probcut_beta:
                    probcut_return_score = pc_score - (probcut_beta - beta)
                    if is_win(probcut_return_score) or is_decisive(probcut_return_score):
                        probcut_return_score = beta
                    tt_store_score_pc = probcut_return_score
                    if tt_store_score_pc > MATE_IN_MAX_PLY: tt_store_score_pc += ply
                    elif tt_store_score_pc < -MATE_IN_MAX_PLY: tt_store_score_pc -= ply
                    pc_tt_depth = max(1, depth - probcut_r + 1)
                    store_tt(search_context.transposition_table, tt_key, pc_tt_depth, tt_store_score_pc, np.int16(32767), TT_FLAG_BETA, pc_move, search_context.tt_generation, tt_pv)
                    _diag_add(search_context, DIAG_PROBCUT)
                    return (np.int32(probcut_return_score), pc_move, nodes_searched, quiescence_nodes, tt_hits)

    # --- Move Ordering ---
    # Retrieve Counter Move if available (SF: [prev_piece][prev_to])
    counter_move = NO_MOVE
    if ply > 0:
        prev_move = search_context.move_stack[ply - 1]
        prev_piece = search_context.piece_stack[ply - 1]
        if prev_move != NO_MOVE and prev_piece != -1:
            prev_to = get_to_square(prev_move)
            counter_move = search_context.counter_moves[prev_piece, prev_to]

    moves = search_context.moves_buffer[ply]
    scores = search_context.move_scores[ply]

    # Safe access to killers:
    safe_ply = min(ply, MAX_PLY - 1)
    killer_1 = search_context.killer_moves[safe_ply*2]
    killer_2 = search_context.killer_moves[safe_ply*2+1]

    # Optimization: Reuse pinned pieces from full evaluation or ProbCut if available
    if cached_pinned_white != PINNED_UNCOMPUTED_SENTINEL:
        pinned_white = cached_pinned_white
        pinned_black = cached_pinned_black
    else:
        pinned_white = get_pinned_pieces(piece_bbs, occupancy_bbs, WHITE)
        pinned_black = get_pinned_pieces(piece_bbs, occupancy_bbs, BLACK)
```

---

### 任務 5：在 `iterative_deepening_search` 初始化 `nmp_min_ply`

* **修改位置**：`chess_engine/classical/search.py` 約第 2208 行（函數 `iterative_deepening_search` 進入 `for current_depth` 迴圈前）
* **問題原理**：防止跨搜尋殘留 `nmp_min_ply` 狀態。

#### 【程式碼替換規格 — 任務 5】
尋找以下原始程式碼（約第 2208 行）：
```python
    search_context.move_stack.fill(NO_MOVE)
    search_context.piece_stack.fill(-1)

    for current_depth in range(1, max_depth + 1):
```

替換為：
```python
    search_context.move_stack.fill(NO_MOVE)
    search_context.piece_stack.fill(-1)
    search_context.nmp_min_ply = np.int32(0)

    for current_depth in range(1, max_depth + 1):
```

---

## 3. 修改後驗證清單（Verification Checklist）

請執行修復的 AI Agent 在完成上述修改後，依序執行以下檢查：

1. **JIT 編譯與型別檢查**：
   * 執行 `warmup_search_jit()`（或相關單元測試），確認 Numba `@njit` 編譯 `_search` 與 `compute_full_corrected_static_eval` 無任何 `TypingError` 或未定義變數錯誤。
2. **NMP 漏斗診斷計數器檢查（Diagnostics Check）**：
   * 開啟 `enable_diagnostics=True` 對基準局面（如起始局面與 Kiwipete，深度 12~16）執行搜尋，呼叫 `format_search_diagnostics(search_context)`：
     * 確認 `nmp_funnel` 的 `elig`、`gate`、`try`、`fh`、`verify`、`cut` 正常運作。
     * 確認在 `depth >= 16` 觸發 `verify` 時，`nmp_min_ply` 能正確限制子節點重複空著。
3. **特殊殘局（`VALUE_KNOWN_WIN`）保護測試**：
   * 測試 KQK 或 KRK 殘局 FEN（例如 `8/8/8/4k3/8/8/8/4K2R w - - 0 1`），確認引擎能正常找到實質將死（`mate` 分數），不會因為 NMP 截斷而在 `+10000` 分數區間停滯。