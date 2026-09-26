# 搜尋算法分析：理念、意義、細節與方法論

本文件對分拆包中實現的搜尋算法（[search.py](search.py) 與 [nnue/search.py](../nnue/search.py)）及相關模組（如 [constants.py](constants.py)、[transposition_table.py](transposition_table.py)）進行了全面的分析。它闡明了底層邏輯（理念）、參數的意義、具體的實現選擇以及整體的算法方法論。

---

## 1. 總覽

引擎在**迭代加深 (Iterative Deepening)** 框架內採用了**主要變例搜尋 (Principal Variation Search, PVS)**。這是現代西洋棋引擎的標準做法，通過優先嘗試最優可能的移動（主要變例，PV），並以極低的開銷證明其他移動皆劣於當前最優解（零窗口搜尋，Zero Window Search），從而高效地探索博弈樹。

**核心檔案：**
*   [search.py](search.py) / [nnue/search.py](../nnue/search.py)：核心搜尋邏輯（迭代加深、PVS、靜態搜尋）。
*   [constants.py](constants.py) / [nnue/constants.py](../nnue/constants.py)：調優參數、剪枝邊界與權重。
*   [transposition_table.py](transposition_table.py) / [nnue/transposition_table.py](../nnue/transposition_table.py)：置換表（哈希表）的實現。

---

## 2. 核心算法

### 2.1 迭代加深 (Iterative Deepening)
*   **理念**：以遞增的深度（深度 1, 2, 3...）逐步進行搜尋，而非直接搜尋到固定深度。
*   **意義**：使引擎在時間耗盡時能隨時返回一個「最佳移動」。更重要的是，它通過淺層搜尋填充了置換表 (TT) 和歷史啟發 (History Heuristic)，這極大地改善了深層搜尋時的移動排序（Move Ordering）。
*   **細節**：
    *   實現於 `iterative_deepening_search` 函數。
    *   使用以先前迭代分數為中心的**抱負窗口 (Aspiration Windows)** (`alpha`, `beta`) 來收窄搜尋窗口並增加截斷。
    *   在迭代之間進行時間管理檢查。

### 2.2 主要變例搜尋 (Principal Variation Search, PVS) / Negamax
*   **理念**：假設第一步棋（排序最好的一步）是主要變例 (PV)，使用全窗口 `(alpha, beta)` 對其進行搜尋。隨後假設所有其他移動皆較差，使用「零窗口」或「空窗口」`(alpha, alpha+1)` 進行搜尋，以低成本證明它們會低於預期（Fail Low）。
*   **意義**：如果移動排序品質高，這將大幅降低博弈樹的有效分支因子。
*   **細節**：
    *   實現於遞迴搜尋函數 `_search` 中。
    *   如果 `searched_move_count > 1`，則執行零窗口搜尋 (`-alpha - 1, -alpha`)。
    *   如果零窗口搜尋意外產生高出邊界（Fail High，返回分數 > alpha），說明最初的假設錯誤，必須使用全窗口重新搜尋 (`-beta, -alpha`)。

### 2.3 靜態搜尋 (Quiescence Search, QS)
*   **理念**：在葉子節點（深度 0）擴展搜尋以抵達「靜態局面」（即不存在吃子或重大威脅的局面），以緩解「地平線效應 (Horizon Effect)」。
*   **意義**：防止引擎在吃子交換序列的中途停止搜尋（例如在 QxP 之後但 recapturing PxQ 之前停下，導致對局勢產生嚴重誤判）。
*   **細節**：
    *   實現於 `quiescence_search`。
    *   僅考慮吃子（若被將軍，則額外考慮逃避將軍的偽合法移動）。
    *   被將軍時，吃子逃逸與 quiet 逃逸分屬不同排序 bucket；quiet 逃逸以既有 piece-to / pawn / butterfly / continuation history 排序並限幅，避免產生器順序決定整棵強迫樹。
    *   引入**增益剪枝 (Delta Pruning)**：如果當前評估極差，即使吃掉對手最大的子（如后）也無法挽回，則直接剪枝。
    *   引入**SEE 剪枝 (SEE Pruning)**：在靜態搜尋中直接剪除不划算的吃子移動 (SEE < 0) 以節省時間。
    *   對齊 SF11/SF18：checking capture 不受 delta/futility gate；但仍必須通過最後的 SEE。普通著以 pre-make check geometry 判斷，升變與 en passant 以暫時 make/unmake 精確判斷。

---

## 3. 移動排序 (Move Ordering)

移動排序是決定 Alpha-Beta 搜尋效率的單一最關鍵因素。引擎採用以下排序結構：

1.  **置換表移動 (TT Move)**：先前搜尋或迭代中記錄的最佳移動。此移動一律最先嘗試。
2.  **吃子移動 (Captures)**：
    *   根據 **MVV-LVA**（最有價值受害者 - 最無價值攻擊者）進行排序。
    *   **划算吃子 (Good Captures)**：優先考慮 SEE >= 0 的吃子移動。
    *   **不划算吃子 (Bad Captures)**：SEE < 0 的吃子移動會被降級，稍後再搜尋。
3.  **殺手步 (Killer Moves)**：在兄弟節點中，在同一深度下產生 Beta 截斷的兩個安靜移動。
4.  **反擊步 (Counter Move)**：記錄對手前一步棋在歷史上最常引起截斷的回應步。
5.  **歷史啟發 (History Heuristic) 與進階安靜步排序**（現用 **Phase 1a**）：
    *   **Piece-to**：`history_table[piece][to]`；與 `butterfly[from][to]`、pawn history 共同排序 quiet。
    *   **延續歷史**：`[ply_idx][prev_piece][prev_to][curr_piece][curr_to]`，offsets 1/2/3/4/6。
    *   **低層歷史**：`low_ply_history[5][65536]`。
    *   **將軍獎勵 / 威脅重排**：仍 `CHECK_BONUS=20000`、`ENABLE_THREAT_REORDERING=False`。
    *   **反制步 / 吃子歷史**：`[prev_piece][prev_to]`、`capture[agg][to][victim0..5]`。
    *   **寫入**：線性 depth bonus/malus + gravity；reverse quiet 關閉。
    *   **衰減**：每次 root search 對各 history 表以向零截斷方式除以 2；full-history/soft-age 已回退。
    *   **診斷**：`DIAG_HIST_*`、`format_history_table_stats`。細節見 [`HISTORY_HEURISTICS_CN.md`](HISTORY_HEURISTICS_CN.md)。

### 3.1 Main quiet SEE 與 check geometry（P1）

P1 曾把 `_search` 的五組 check bitboard 傳入 `score_quiets()`，並把 ordinary quiet 的 pin／king
legality 移到 shallow SEE 前；固定節點對戰沒有接納 P0/P1/P3 組合，因此兩項 production 優化
均已回退。現行 scorer 恢復 lazy local bishop/rook geometry，quiet legality 恢復在 shallow SEE
之後；EP、castling、capture 與 in-check 路徑仍保留 post-make safety fallback。P1 的
`DIAG_MAIN_QUIET_*`、`DIAG_MAIN_GEOMETRY_NODE` 與 `DIAG_MAIN_SCORE_GEOMETRY_REBUILD` 仍只在
diagnostics context 啟用，以 shadow 分類量測可省成本，不拒絕或重排 production 候選。

### 3.2 Aspiration diagnostics（P2）

`iterative_deepening_search()` 仍以先前迭代的 root score 建立 HCE 窗口：
`delta = max(ASPIRATION_WINDOW_MIN, ASPIRATION_WINDOW_BASE + abs(score) // SCALE_DIV)`，
目前 production `ASPIRATION_WINDOW_BASE=20`（BASE 30 曾在 P2 A/B 後短暫採用，現因固定節點對戰回退）。P2 在不改窗口語義的前提下，於 diagnostics context
按深度記錄 fail-low、fail-high、額外 re-search、最大 `delta` 及失敗窗口消耗的 nodes，並由
`collect_hist_diag_dict()` 輸出 aggregate 與 `aspiration_by_depth`。

30k 混合診斷顯示 aspiration 重搜在戰術／開局都具事件量；BASE 30 曾在重搜與單執行緒 elapsed
間取得折衷，但固定節點對戰未顯示棋力收益。將 BASE 20 改成 40 雖降低更多重搜，
單執行緒 elapsed 略慢且固定 nodes 平均完成深度下降；SF18-inspired score-square selective
widening 也沒有速度收益。因此目前保留 BASE 20，任何後續窗口變更都必須通過同執行緒速度、固定
 depth 與實戰對戰三重驗收。

#### 3.2.1 SF17/18 視窗收斂與幾何擴展（E37 測試）
針對迭代加深中 Aspiration Window 在 Fail-Low 與 Fail-High 時的重搜邊界行為，全面對齊 Stockfish 17/18（`Thread::search()`）：
1. **Fail-Low 邊界鎖定**：舊版在 `score <= alpha` 時將 `beta = (alpha + beta) // 2`；但數學上既已證明得分小於等於 `alpha`，則大於 `alpha` 之區間皆為死樹。SF17 直接將 `beta = alpha` 鎖死上方邊界，並僅以最新分數向下方擴張 `alpha = max(-INFINITY, score - delta)`。
2. **Fail-High 邊界提升**：舊版在 `score >= beta` 時僅擴張 `beta` 而讓 `alpha` 保持不動；SF17 則同步將下界拉升至 `alpha = max(beta - delta, alpha)`，大幅收窄無效搜尋區間。
3. **幾何擴張**：將 SF11 的線性 `delta += delta // 3 + 3` 升級為 SF17/18 整數縮放 `delta += 44 * delta // 128`（約 1.344 倍），加快在極端波動盤面中的視窗脫困速度。
* **實戰驗證（E37 採納，2026-09-23）**：400 盤固定 300,000 nodes 對戰，戰績 94W 212D 94L（得分率 50.00%，Elo **0.00 ± 22.80**，NPS +0.6%）。在深層固定節點下呈現完全對稱的數學中性與健全性，已正式採納進生產基線（同步至 `classical_old`）。

### 3.3 Main in-check picker 與深層成本診斷（P3）

P3 的診斷分成三條 opt-in funnel：

* qsearch non-check/check cap hit，以及 cap 當下的 pseudo capture/evasion 候選數；這是
  forcing 上界，不宣稱候選一定合法；
* main in-check picker 的 checker count、single-check capture/interposition targets、pin
  ray 與 king destination。`get_next_move()` 只在 diagnostics context 評分前做 shadow
  classification；production 不壓縮 capture/quiet segment，也不拒絕 TT move。EP 與無法靜態
  證明的情形仍記為 fallback；
* TT deep move verification 的 `try/pass/reject/saved-cut/skip`，用來確認 SF18 child-TT
  direction guard 的成本與保護價值。

正式 500+500 n30k（每盤 30k nodes）平均顯示：main in-check 候選 84.06% 可在評分前證明
非法，qsearch check-cap 沒有命中，TT verify reject 約 0.105% 但 saved-cut 約 32.40%。因此
歷史上曾暫時接納 main in-check pre-score compaction；後續固定節點對戰判定組合版本較弱，現已
回退為 shadow-only。qsearch cap 仍不放寬，也不移除或弱化 TT verify。這再次證明固定 nodes 下
的微小 depth/nodes 差異不等同 Elo，需由新舊引擎對戰判定。

---

## 4. 剪枝與歸約 (Pruning and Reductions)

引擎實現了一套激進的剪枝技術來減少搜尋空間。

### 4.1 空著剪枝 (Null Move Pruning, NMP)
*   **理念**：如果我們跳過自己的一步棋（下空著），而對手**仍然**無法將局面評分提升到 Beta 以上，說明我們佔據了巨大的優勢，可以直接進行截斷。
*   **邏輯**：
    *   條件：`static_eval >= beta` 且 `depth >= 3`。
    *   操作：以降低的深度進行搜尋。
    *   **動態歸約**：根據靜態評估超出 Beta 的幅度 (`(static_eval - beta) // 200`) 動態增加額外的歸約深度 (最大額外歸約 3 步)。
    *   驗證：如果返回分數 `>= beta`，則直接返回 Beta（截斷）。
    *   **安全性**：被將軍時或殘局中不套用（以防雙向無子限制 zugzwang 危害）。

### 4.2 反向徒勞剪枝 (Reverse Futility Pruning, RFP) / 靜態空著剪枝
*   **理念**：如果靜態評估值非常高，以至於減去一個很大的安全邊界後依然高於 Beta，我們可以直接進行截斷。
*   **邏輯**：
    *   適用於 `depth <= 2`。
    *   條件：`static_eval - margin >= beta`。

### 4.3 剃刀剪枝 (Razoring)
*   **理念**：如果在極淺的深度下靜態估值顯著低於 Alpha，我們假設該分支無法挽回，提前進入靜態搜尋以節省時間。
*   **邏輯**：
    *   適用於淺深度。
    *   條件：`static_eval + margin < alpha`。
    *   操作：調用靜態搜尋驗證，若仍低於 Alpha 則直接返回 Alpha。

### 4.4 徒勞剪枝 (Futility Pruning, FP)
*   **理念**：在淺層深度搜尋時，如果靜態評估加上與深度正相關的邊界後仍然低於 Alpha，則直接跳過剩餘的安靜移動。
*   **邏輯**：
    *   條件：`static_eval + margin < alpha`。
    *   操作：跳過當前的安靜移動。

### 4.5 晚期移動剪枝 (Late Move Pruning, LMP)
*   **理念**：如果在一個節點中已經搜尋了大量的安靜移動卻仍未找到更好的移動，那麼剩餘排序極後的安靜移動不太可能有所幫助。
*   **邏輯**：
    *   條件：已嘗試的安靜移動數超過特定深度的閾值。
    *   操作：直接忽略該節點剩餘的所有安靜移動。

### 4.6 概率剪枝 (ProbCut)
*   **理念**：空著剪枝的概率版。使用一個更寬的窗口和較淺的深度進行搜尋，以預測當前移動是否極有可能引發強烈的截斷。
*   **Step 11: 吃子 ProbCut (Capture-based ProbCut)**：
    *   在非排除搜尋、非將軍、深度足夠（`depth >= 5`）時，以 `probCutBeta = beta + margin` 且搜尋深度 `depth - 4` 嘗試吃子走步。若 QSearch 與子搜尋皆 hold 且 `>= probCutBeta`，則存入 TT 並回傳截斷。
*   **Step 12: TT-based ProbCut Shortcut (純 TT 快速捷徑，對齊 SF 18)**：
    *   **位置**：放置於 ProbCut 入口處（在吃子生成、SEE 計算與 QSearch 之前）。
    *   **邏輯**：若 TT 條目已存在且下界有效（`BOUND_LOWER` 或 `BOUND_EXACT`），深度 `tt_depth >= depth - 4`，且其分數 `tt_score >= beta + PROBCUT_TT_SHORTCUT_MARGIN`（334 cp，對齊 SF18 的 428 cp 乘以 HCE 0.78 縮放）：
        *   此節點在 `depth - 4` 已經被證明能獲得壓倒性高分，符合 ProbCut 截斷本意。
        *   直接判定 Fail-High 回傳 `probcut_beta`，達成 $O(1)$ 零搜尋開銷截斷，徹底跳過整段昂貴的吃子生成、SEE 評估、QSearch、子搜尋，以及後續的 NMP 與 Razoring。
        *   守護條件：嚴格排除 `ply == 0`（根節點禁止啟發式截斷）、排除奇異延伸搜尋、排除被將軍盤面、排除將死與確知勝負分數。
    *   **實戰驗證（E34 採納，2026-09-21）**：
        *   對戰條件：400 盤，每步 300,000 nodes，TT 64MB（對戰剛獲 +30.48 Elo 的 E33 強基準）。
        *   最終比分：206.5 - 193.5（93W 227D 80L，勝率 51.63%，Elo: **+11.30 ± 22.39**，SPRT LLR: 0.51）。
        *   分析：在大幅提升的基線上再次實現正向淨收益，證明將 TT 捷徑前置能大幅抑制已證偽節點中的走步生成與子搜尋算力消耗。

### 4.7 晚期移動歸約 (Late Move Reduction, LMR)
*   **理念**：對於排序靠後（可能較差）的移動，以降低的深度進行搜尋。
*   **邏輯**：
    *   條件：`depth >= 3`，移動是安靜移動或吃子移動。
    *   歸約：基於 precomputed LMR 表格及 1024-scale 的細緻調整公式。
    *   **吃子移動歸約**：吃子移動也參與 LMR 歸約（解決節點爆炸問題），根據 SEE 結果進行偏置：好吃子減少 1 ply 歸約 (`r -= 1024`)，壞吃子增加 1 ply 歸約 (`r += 1024`)。
    *   **通路兵安全網**：如果是兵的前進且已進入危險橫排 (White >= 5, Black <= 2)，減少 1.5 ply 歸約 (`r -= 1536`) 以免低估通路兵威脅。
    *   **殺手/反擊步安全網**：殺手步與反擊步給予 1 ply 的歸約豁免 (`r -= 1024`)。
    *   調整：受歷史分數調節（歷史表現好的移動減少歸約，表現差的增加歸約）。
    *   例外：給予將軍、升變等強迫性移動通常不進行歸約。

### 4.8 奇異延展 (Singular Extensions, SE) 與 Multi-Cut（SF18 Step 15 對齊）
*   **理念**：如果 TT 移動顯著好於其他所有候選移動（單一優勢），則加深該著法的搜尋，以確保強迫線計算精準。反之若排除 TT 移動後其他移動仍能 Fail-High（`>= beta`），則觸發 Multi-Cut 立即截斷。
*   **Pre-make 架構（SF18 Step 15 完整對齊）**：
    *   **零走棋開銷**：將 SE 與 Multi-Cut 判定前置於 `make_move` 之前，直接在當前 parent 盤面以 `excluded_move = tt_move` 啟動排除搜尋，徹底消除了舊架構 `make -> unmake -> search -> make` 的 4 次走棋往返浪費。
    *   **合法性守護（SF `pos.legal(move)`）**：要求 `skip_post_legality_s` 為真，確保該 TT 著法為嚴格合法著法，絕不在非法/牽制著法上浪費排除搜尋。
    *   **狀態保存與隔離（Invariant Protection）**：
        *   排除搜尋在相同 `ply` 下呼叫，結束後必須完整還原 `follow_pv_stack[ply]`、`static_eval_stack[ply]`、`tt_hit_stack[ply]`、`tt_pv_stack[ply]`、`reduction_stack[ply]`、`cutoff_cnt[ply]`，並重置 `pv_table[ply, ply] = NO_MOVE`。
        *   **TT 寫入隔離**：排除搜尋（`excluded_move != NO_MOVE`）末端禁止寫入置換表（對齊 SF18 `if (!excludedMove) ttWriter.write(...)`），避免殘缺的走步與評估污染該局面的全局 TT 快取。
    *   **門檻**：`ply > 0`、`excluded_move == NO_MOVE`、`move == tt_move`、`depth >= MIN_SINGULAR_DEPTH (+1 if ttPv)`（預設 7）；TT bound 為 Lower/Exact 且 TT depth 足夠（`depth - 3`）。
    *   **Multi-Cut 剪枝（SF18 Step 15）**：若 `exclusion_score >= beta` 且非勝負分，直接回傳 `exclusion_score` 截斷，無需任何 `unmake_move`（**E31 採納**）。
    *   **單 / 雙延展**：若 `exclusion_score < exclusion_beta`，賦予 `singular_extension = 1`；若差距達到 `double_margin`（`2 * depth`）且 `depth >= 10`，賦予雙延展 `singular_extension = 2`。
    *   **負延展（SF17 Step 15 對齊，E39 測試中）**：若非 PV 節點且非奇異（其他著法亦能達到 `exclusion_beta`）：若置換表自身評估預期高於邊界（`se_tt_score >= beta`）則賦予 `-3`；若在 `cut_node` 則賦予 `-2`。修復了舊版因 `exclusion_score >= beta` 提前被 Multi-Cut 截斷導致大負延伸成為死碼的問題。
    *   **延展合併與將軍延伸 Pre-make（E36 採納）**：走棋後與將軍延展（`check_extension`）合併：正向延展取較大值（`max`），負向延展在無將軍時覆蓋。對於將軍延伸的 `_see_ge_jit` 檢驗，全面改為在 `make_move` 前直接利用父盤面評估（`check_gate_see_ok`），徹底根除了舊代碼在走棋後 `unmake -> SEE -> remake` 的兩次冗餘棋盤突變開銷。
    *   **實戰驗證（Step 15 Pre-make SE 採納，2026-09-21）**：
        *   對戰條件：400 盤，每步 300,000 nodes，TT 64MB。
        *   最終比分：104W 193D 103L（勝率 50.12%，Elo: **+0.87 ± 24.49**）。
        *   分析：Pre-make 結合合法性過濾、狀態 stack 還原及 TT 隔離後展現完全健全的架構穩定性（0 崩潰、深度 $\ge 14$ 佔比提升 4.8%），確認結構中性偏正，正式採納進生產基線（同步至 `classical_old`）。

### 4.9 內部迭代加深 (Internal Iterative Deepening, IID) / IIR
*   **理念**：如果當前節點沒有 TT 移動（例如是新節點），則先以較淺的深度搜尋該節點，以獲得一個合適的移動進行排序，再進行完整搜尋。現行實作為 **IIR**（直接減深），而非昂貴的子搜尋 IID。
*   **followPV 豁免（2026-07-20）**：上一輪迭代 PV 沿線節點不做 IIR（`ENABLE_FOLLOW_PV`），避免在關鍵變例上減深漏戰術。

### 4.10 Alpha 提升後減深 (Alpha-Raise Depth Reduction)
*   **理念（SF18）**：在 **non-PV** 節點，若某步已把 `alpha` 抬高但仍未 fail-high，則對**後續兄弟著**把本節點剩餘 `depth` 減去固定 plies（SF：`2 < depth < 13` 且 `depth -= 2`，非 decisive）。
*   **意義**：已找到可接受著後，少花節點證明其餘著更差。
*   **HCE 注意（2026-07-20）**：
    *   SF 的 −2 是在 **NNUE + 極強排序** 上調的；HCE 排序較噪時，第一次抬 alpha 較常是「暫時最好」而非真 PV，連續抬 alpha 還會**疊加減深**，在 fixed-nodes 下容易少算戰術反制。
    *   頻段預設 **5–12 ply**（`LO=4, HI=13`）：略過 3–4（相對減深過猛，且已有 LMP/FP）；SF 原為 3–12。
    *   減深量：**`ALPHA_RAISE_DEPTH_DELTA`**（SF=2；HCE 曾試 1≈0 Elo，現測 2）。
    *   **含 PV**（與 SF 一致）。曾誤加 `not is_pv`：null-window 下 `α < score < β` 幾乎不成立 → 死碼。
    *   **HCE：`ply > 0`**（root 不減深，保留 bestmove 全寬排序；SF 連 root 也減）。
    *   `ALPHA_RAISE_MIN_IMPROVEMENT` 可要求最小增益（cp）才減深。
*   **開關／旋鈕**：`ENABLE_ALPHA_RAISE_DEPTH_REDUCTION`、`ALPHA_RAISE_DEPTH_LO/HI`、`ALPHA_RAISE_DEPTH_DELTA`、`ALPHA_RAISE_MIN_IMPROVEMENT`。

### 4.11 followPV（上一輪迭代 PV 追蹤）
*   **理念（SF18 `ss->followPV`）**：迭代加深每層完成後把 root PV 存入 `last_iteration_pv`；下一層搜尋時，若路徑上每步都對齊該 PV，則標記 `follow_pv`。
*   **用途**：
    *   **IIR**：`!follow_pv` 才減深。
    *   **安靜著剪枝**（LMP / FP / history / quiet SEE）：在既有 `!is_pv` 之外，再要求 `!follow_pv`（對齊 SF `!followPV || !PvNode` 的安全側）。
*   **HCE 注意**：不依賴 eval 尺度；mate 判定沿用既有 `is_decisive` / `MATE_IN_MAX_PLY`。
*   **開關**：`ENABLE_FOLLOW_PV`（E27 測試中；可與下項疊加）。

### 4.12 History → lmrDepth → quiet FP / quiet SEE
*   **理念（SF11/18 Step 14）**：剪枝用的深度不是 raw `depth`，而是近似 **LMR 後的 lmrDepth**，並用 **quiet history** 調整（好 history → lmrDepth 較高 → FP margin / SEE 門檻較鬆，較難被剪）。
*   **實作**：`compute_quiet_pruning_lmr_depth`（base LMR 表 + `TUNE_LMR_BASE_OFFSET` + `get_lmr_stat_score`）；套用在 **quiet SEE** 與 **parent FP**。
*   **開關**：`ENABLE_HIST_LMR_DEPTH_PRUNING`（**E28 採納** True）。

### 4.13 Capture Futility + captHist SEE
*   **Capture futility（SF18 Step 14）**：非將軍吃子、`lmrDepth < CAP_FP_MAX_LMR_DEPTH`：  
    `fut = static + BASE + MULT*lmrDepth + MG(victim) + captHist*131/1024`；`fut <= alpha` 則 skip。  
*   **Capture SEE margin**：`threshold = SEE_CAP_MARGIN * depth - captHist * 34/1024`。  
*   **歷史演進**：
    *   E29（2026-07-20）：曾置於 `make_move` 之後，存在 `unmake -> eval -> make` 走棋浪費且裕度偏大（-16 Elo 否決）。
    *   E33（2026-09-20）：重構為 SF18 Step 14 前置零走棋開銷，並縮放至 HCE 體系（`BASE=110`, `MULT=100`）。300@n300k 對戰獲得 **+8.11 Elo**，節點數顯著減少 0.5% ($p=0.0079$)。
*   **開關**：`ENABLE_CAPTURE_FUTILITY = True`（**E33 採納進生產基線**）。

### 4.14 延續歷史淺層剪枝 (Continuation History Pruning, SF17 Step 14)
* **理念（SF17 Step 14）**：在淺層非 PV 節點，若安靜著法在局部延續歷史（1-ply、2-ply 應手）與兵結構歷史的未加權和極度為負（多次被證偽為敗著），直接在走棋前跳過。
* **解決缺陷**：舊版歷史剪枝依賴全局 `main_history_score < 0` 守護，導致熱門格位的戰術昏著無法被剪枝。延續歷史剪枝直接針對「特定戰術上下文」進行精準截斷。
* **門檻縮放**：SF17 原生為 `-4313 * depth`（總表容量 68,192）；本引擎等比縮放（總表容量 28,672）為 `PRUNING_CONTINUATION_THRESHOLD = -1800`。
* **開關與實戰驗證**：`ENABLE_CONTINUATION_HISTORY_PRUNING = True`。400 盤固定 300,000 nodes 對戰，戰績 105W 210D 85L（得分率 52.50%，Elo **+17.39 ± 23.46**，超長局勝率 57.90%）。確認帶來顯著正向收益，正式採納進生產基線（**E38 採納**，同步至 `classical_old`）。

### 4.15 Tactical anti-misprune（已移除）
*   E30 試過 LMR check/promo/recapture 減免 + TT 吃子免 shallow SEE。  
*   **~334@n10k: 45.1% · Elo≈−34 · CI 上界&lt;0** → **整段刪除**（含旗標與常數）。  
*   將軍 LMR 僅保留原有 `lmr = max(0, lmr - 1)` 收尾減免。

### 4.15 Pre-make Quiet Pruning Cascade (Step 14: LMP → History Pruning → FP → SEE)
*   **架構重構（SF18 Step 14 完整對齊）**：
    *   將原本位於 `make_move` 之後、需調用 `unmake_move` 復原的安靜著剪枝（LMP、History Pruning、Futility Pruning），全部前移至走步迴圈開頭的 **Step 14 前置淺層剪枝區**（在 `make_move` 之前直接 `continue`）。
    *   **代價由低至高階梯式評估**：
        1.  **Step 14a: LMP**：純計數器 $O(1)$ 判定（`quiet_move_counter >= limit`）。
        2.  **Step 14b: History Pruning**：查閱綜合曆史表與主曆史表。
        3.  **Step 14c: Futility Pruning (FP)**：`static_score + margin < alpha`。若需要 full static eval，可直接在當前 parent state 執行，不再需要舊架構中 `unmake -> full eval -> remake` 的沉重往返。
        4.  **Step 14d: Quiet SEE Pruning**：僅在通過上述所有輕量剪枝後才調用射線攻防計算。
    *   **守護條件**：
        *   `ply > 0`（對齊 SF18 `!rootNode`，根節點永遠不執行淺層啟發式剪枝）。
        *   `move != tt_move`（**核心保護**：置換表最佳走步永遠穿透 Step 14，直達 Step 15 奇異擴展與 Multi-Cut 檢驗，根絕 TT Move Drop 引發的搜尋不穩定性）。
        *   `depth <= PRUNING_SHALLOW_DEPTH` (12)、`!is_pv`、`!follow_pv`、`!is_exclusion_search`、`!is_currently_in_check`、`!low_material_pruning_guard`。
        *   利用 `_qsearch_ordinary_move_gives_check` 前置探測將軍，將軍著走 Checking SEE 專屬通道，絕不被安靜剪枝過濾。
        *   使用 `quiet_counted` 旗標精確追蹤計數，確保跳過 Step 14 的深層節點、PV 節點與 `tt_move` 在 `make_move` 後仍準確累計 `quiet_move_counter` 供後續 LMR 減深使用。
    *   **實戰成果（E33, 2026-09-21）**：
        *   400 局 @ 300k nodes 對戰：**+30.48 ± 22.63 Elo (95% CI)** (217.5 - 182.5, 54.37%)，顯著消除深層戰術盲區與盲目漏著。

---

## 5. 置換表 (Transposition Table, TT)

*   **理念**：一個哈希表，用於存儲搜尋過的節點結果，以便在不同路徑到達相同局面時直接復用。
*   **實現細節**：
    *   **鍵值**：Zobrist Hash (64位)。
    *   **結構**：`key`, `score`, `depth`, `flag` (Exact/Alpha/Beta), `generation`, `best_move`, `static_eval`。
    *   **替換策略**：
        1.  **世代 (Generation)**：優先替換來自舊搜尋世代的條目。
        2.  **深度 (Depth)**：在相同世代中，優先保留深度較深的條目。
    *   **優化**：在 Numba 內以一維結構體數組的形式高效訪問。
    *   **分數調整**：將軍殺死的分數在寫入表前會轉換為相對於根節點的絕對值，在讀出時再轉換回相對於當前層數 (ply) 的分數。
    *   **TT Cutoff 安全防護與深度調整**：
        *   **自適應深度門禁**：當 TT 記錄即將觸發 Fail-High（分數 $\ge \beta$）時，對 TT 深度的要求比 fail-low 時額外多 1 層（即 `required_depth = depth + 1`），以確保 Beta 剪枝的安全性。
        *   **cutNode 一致性守護**：唯有當 TT 分數大於等於 Beta 的布林狀態與目前節點的預期 `cut_node` 屬性一致時（或者在 `depth > 4` 的深層搜尋中），才允許直接進行置換表 Cutoff，有效防範圖歷史交互 (GHI) 污染所引起的錯誤剪枝。
        *   **深度移動驗證 (Deep Move Verification)**：當搜尋深度較深（`depth >= 7`）且即將觸發置換表剪枝時，透過暫時模擬並執行 `tt_move`，實地探測子節點的 TT 值。如果子節點 TT 記錄在 `beta` 邊界上的布林方向與當前節點不一致（表明此處可能發生 GHI 污染），則拒絕本次 TT Cutoff，強迫執行完整搜尋以確保安全性。

---

## 6. 時間管理

*   **理念**：根據剩餘時間、增量時間以及局面的複雜程度分配搜尋時間。
*   **邏輯**：
    *   **軟限制 (Soft Limit)**：`optimum_time`。若超出，在下一個搜尋層級結束時退出。
    *   **硬限制 (Hard Limit)**：`maximum_time`。一旦超出，立即中斷搜尋並設置 `stop_flag`。
    *   **預測性退出**：如果在當前深度迭代中已經耗費了大量時間（例如已達極限的 40% 以上），預測下一層迭代無法在硬限制前完成，則提前中斷，避免浪費時間。

### 6.1 固定節點計時器的不變量（2026-07-30）

固定節點模式的 Python timer 透過 `SearchContext.nodes_searched_array[0]` 觀察目前 iterative-deepening iteration 的節點數。這個共享槽位必須在每次 root search，以及每個新的 depth iteration 開始時清零；`total_nodes` 才負責累計已完成 iteration。若沿用上一手的值，timer 可能在 depth 1 完成前就設置 `stop_flag`，使搜尋返回 `NO_MOVE`，對戰裁判會誤判成 `engine failure`。現行 `search.py` 已在兩個邊界重置該槽位，`classical_old` 亦由同步工具保持一致。

搜尋熱路徑另以 `STOP_CHECK_MASK=4095` 每 4,096 nodes 發佈進度並觀察
`stop_flag`。舊的 32,768-node cadence 在 100k fixed-node baseline 實測平均超搜
17.8%（最差 32.6%），不適合當小預算 A/B 的節點界限。
固定 nodes 不再依賴 Python 2ms polling：`nodes_searched_array[1]` 存當前 iteration
剩餘預算，JIT 熱路徑到 cadence 時直接置 `stop_flag`；Python thread 只處理時間限制。
同套 10 局面 100k 實測改為平均 **102,626**（+2.6%）、最差 **103,892**
（+3.9%）；13 局面 300k 為平均 **302,415**、最差 **303,872**，無 `NO_MOVE`。

---

## 7. 重複偵測 (Repetition Detection)

*   **理念**：檢測三手重複或 50 步規則以正確將和棋局面評分為 0.00。
*   **邏輯**：
    *   檢查搜尋路徑棧 `ply_path_stack` 以及搜尋前的歷史對局記錄 `game_history`。
    *   **優化**：僅檢查行棋方相同的局面（每隔兩步檢查一次：`ply-2, ply-4` 等）。
    *   **限制**：檢查範圍受半步計數器 (halfmove clock) 限制，任何兵的推進或吃子操作都會重置重複檢查起點。

---

## 8. 分析與結論

Antares 引擎的搜尋算法架構現代且完備，包含了當今最先進的剪枝和歸約啟發式方法。通過 Numba (JIT) 進行底層編譯後，搜尋邏輯在 Python 環境中能展現出極佳的每秒搜尋節點數 (NPS)。

---

## 9. 凍結基準（2026-07-15）— 搜尋調參（政策已更新）

**狀態：** 2026-07-15 凍結曾寫入 `classical_old`。  
**政策變更（2026-07-19）：** **取消**「禁止比 SF11 更兇」的硬上限。SF11 仍是**參考與對照基準**，是否更兇只以 **pass / Elo / depth** 閘道決定，不再用「不得超過 SF11」否決實驗。

| 項目 | 內容 |
| :--- | :--- |
| 方向（舊） | LMR / 剪枝曾以 SF11 為侵略性上限（HCE ×0.78） |
| 方向（新） | 允許更兇或更鬆；**單軸 A/B + 對戰**；已知失敗點（S1 full SF NMP、E2 LMR scale 70 等）仍勿盲抄 |
| 主收益（當時） | 固定 nodes（20 萬）下 mean depth **+1.5 ply** vs 調參前 Old |
| NMP 結構（現行） | 生產為 **scope=0**（只限 `cut_node`）；scope 3 對戰不足以保留而回退，gate/R 仍 legacy（見 `NMP_EXPERIMENT_PLAN`） |

### 9.1 樹效率診斷（2026-07-15+）

`SearchContext.diag_stats`（`DIAG_*` 於 `constants.py`）在搜尋中累計：

- beta cut 次數 / **第一著 cut 比例** / cut 時平均第幾手  
- LMR 嘗試與 **re-search 比例**  
- TT / NMP / RFP / razor / ProbCut / LMP / FP 觸發次數  

Python 讀取：`format_search_diagnostics(ctx, total_nodes, q_nodes)`；  
套件量測：`python scratch/node_bench.py --depth 10 --diag`。

`iterative_deepening_search` 的回傳 tuple 將 **main-search nodes** 與 **qnodes** 分開；
診斷工具的 total nodes 應只計算一次 `nodes + qnodes`。內部另有包含 QS 的
all-node counter，僅用於 UCI `nodes` 與 fixed-node stop，不可再與 `qnodes` 相加。

小修：LMR 吃子 history 的 victim 價值改為 **`MG_MATERIAL_VALUES`**（不再用硬編碼 320/330）。

**P0 QS 實驗（已回退）：** `DELTA 350→300`、`QS_SEE -80→-60`。  
半套 puzzle 仍 243/250；但 `node_bench` d10 套件 SUM_total **上升**（~0.83M→0.99M），qnode% 幾乎不變 → **不採納**。

**P1 LMR bad-capture（採納）：** `LMR_BAD_CAPTURE_BONUS` **1024→1536**（+1.5 ply 對 SEE&lt;0 吃子）。  
`puzzle_half --stride 2 --diag`：基準 243/250 → **245/250**（rate 98.0%）；`sum_nodes` ~540M→542M、`avg_depth` 21.56→21.59（持平）。  
日誌：`scratch/puzzle_half_p1_badcap.txt`。常數：`LMR_BAD_CAPTURE_BONUS` / `LMR_GOOD_CAPTURE_RELIEF`。

**P2 capture LMR 資格（已回退）：** 對齊 SF11 — 吃子僅在 `cutNode` 或 `staticEval+EG(victim)≤alpha` 時 LMR，並加 `depth&lt;8 && mc&gt;2 → r+=1024`。  
puzzle：**244/250**（&lt; P1 的 245）、`avg_depth` 21.36、`sum_nodes` ~551M → 戰術略退、節點未降 → **不採納**。日誌：`scratch/puzzle_half_p2_cap_lmr.txt`。

**P3–P6（連續 A/B，均未採納；基準仍為 P1 的 245/250）：**

| 實驗 | 變更 | 結果 | 判定 |
| :--- | :--- | :--- | :--- |
| P3 | `LMR_HISTORY_SCALE` 150→180 | 244/250 | 回退 |
| P5 | `LMR_GOOD_CAPTURE_RELIEF` 1024→768 | 244/250 | 回退 |
| P6 | `LMR_KILLER_COUNTER_RELIEF` 1024→512 | 125/250 時已 fail=6（P1 終局 5）→ **中止** | 回退 |
| P4 | `LMR_NO_TTMOVE` 1024→768 | 未跑 | — |

**現況（採納棧）：** SF11-cap + diag + LMR victim MG + **P1 bad-capture +1536**。其餘 LMR 邊角常數維持原值。

**Soft pack P2′+P3′+P5′（已回退）：** late-cap `+512` + history `160` + good-relief `896`（無 SF11 capture gate）。  
puzzle **244/250**（&lt; P1 245）、avg_depth 21.49、sum_nodes ~544M → **不採納**。`scratch/puzzle_half_soft_p235.txt`。

### 9.1b 消融 + 係數掃描（樹肥主線）

**消融 d10（關閉單項，看節點膨脹）** — `scratch/node_ablation_d10.txt`：

| 關閉 | ×baseline | 解讀 |
| :--- | ---: | :--- |
| LMR | **2.89** | 最大省節點來源（仍不夠追上 SF11） |
| LMP | **2.45** | 第二 |
| SEE shallow | 1.54 | 重要 |
| NMP / RFP | ~1.17–1.18 | 中等 |
| Razor / FP | ~1.03–1.05 | **省很少**（razor 僅 depth&lt;2） |
| ProbCut | **0.95** | 關了反而略瘦（實作/尺度問題） |

vs SF11 同套 d10：**~2.7×**；最肥局面 start/pos4/mid1/quiet1（≥3×）。

**Razor / FP 與 SF11：**  
SF11 razor ~1 Elo、僅 d&lt;2；FP/RFP 才是 ~50 Elo 級。我們消融與 SF 一致——**不是「razor/FP 剪大量分支我們卻不敢」**，而是 **LMR/LMP/SEE 與 eval 尺度** 決定主體積。收緊 razor 邊際有限；RFP 掃描反而顯示 **更安全的 mult 更好**（少剪錯）。

**一次 JIT 掃描** — `scratch/pruning_sweep.py` / `pruning_sweep_full_half_d10.txt`：  
250 題 fixed **depth=10**、58 configs、`SearchContext.tune[]` 運行時改係數。

Top（pass↑ 且 nodes↓ vs baseline 226/250 · 10.36M）：

| config | pass | nodes× |
| :--- | ---: | ---: |
| fpm_160 | **234** | 0.92 |
| fpb_140 | 232 | 0.90 |
| razor_400 | 232 | 0.97 |
| rfp_200 / lmrhist_180 / badcap_1024 | 231 | ~0.91–0.95 |
| lmrbase_768 | 226 | **0.85** |

**Dual（曾採納）：** `LMR_BASE=768` + `FP_MULT=160` → 3s half **244/250**、節點↓；user 接受 −1。

**座標下降 full@d10（採納，2026-07-15）：**  
腳本 `scratch/coord_descent_full_d10.py`（一次 JIT、14 參數 ×~5 值、full 500 @ depth 10）。  
選法：先 pass；pass−1 僅當 nodes≤0.95×；否則 min nodes。鎖死後續參數。

| 相對 dual baseline | full@d10 | 3s half |
| :--- | ---: | ---: |
| dual start | 448/500 · 19.8M | 244/250 |
| **descent end** | **458/500 · 20.5M** | **246/250** |

鎖定值：`LMR_BASE=512`、`LMR_HISTORY=210`、`LMP_SCALE=110%`、`FP_BASE=160`、`FP_MULT=190`、bad-cap 1536；其餘維持 SF11-cap 預設。  
日誌：`scratch/coord_descent_full_d10.log`、`coord_descent_full_d10_results.json`。

**Round-2（反向細格，一次 JIT，2026-07-15）：**  
`scratch/coord_descent_round2.py` — reverse order + finer grids + `QS_SEE`；full 500 @d10。  
結果：pass **458→458**；nodes **20.55M→20.43M（0.994×）**；僅 **`FP_MULT 190→210`** 勝出，其餘鎖在 R1。  
3s half：**246/250**（與 R1 持平）。→ 1D 細掃已平台；生產常數採 `FP_MULT=210`。

**Round-3（結構 LMR，一次 JIT，2026-07-15）：**  
接 `tune[]`：`CUTNODE` / `NO_TTMOVE` / `TTCAP` / `MC_FACTOR` / `TTMOVE_RED`，再 re-lock volume（base/hist/LMP/FP）。  
腳本：`scratch/coord_descent_round3.py`；日誌 `coord_descent_round3.log` / `coord_descent_round3_results.json`。

| 結果 | 數值 |
| :--- | ---: |
| full@d10 | **458/500 · 20.43M**（= R2，**10/10 參數全鎖原值**） |
| 3s half | **242/250**（R1/R2 為 246；timed 有噪，fixed-d10 不變） |

解讀：結構 LMR 與 volume re-lock 在 d10 座標下降下**無任何移動**——現有 R2 鎖已是局部最優。  
`LMR_TTMOVE_REDUCTION` 五個候選 node 數完全相同（此路徑在 d10 題庫影響極小或整 ply 量化後無差）。  
**常數不改**（維持 R2：`FP_MULT=210` 等）。`tune[]` 結構槽位保留供後續實驗。  
**剪枝係數座標下降已平台**；下一刀應改結構/演算法（NMP 公式、LMR 表形狀、延伸、eval 尺度），而非再掃 ±margin。

### 9.2 vs SF11 診斷（fixed-depth + 部分 puzzle）

| 測試 | SF11 | 我們 (P1) | 比值 / 差 |
| :--- | ---: | ---: | :--- |
| node_bench d10 **SUM_total**（10 FEN, Hash 256） | **336,173** | **827,572** | **~2.46×** |
| d10 典型 NPS | ~1.8–2.5M | ~0.85–1.0M | **~2× 慢** |
| puzzle stride2 @3s（我們完整 / SF11 前 75 題） | **75/75** 完美；mean NPS ~5M；75 題已 ~760M nodes | **245/250**；avg_depth **21.6**；250 題 ~542M nodes | 同時間我們 **深度與吞吐** 都落後 |

**問題分層（非排序崩壞：cut_first~91%）**

1. **Same-depth 樹肥 ~2.5×** — LMR/剪枝在 HCE×0.78 cap 下偏軟；eval 驅動的 RFP/NMP/ProbCut 觸發與 SF 不同。  
2. **NPS ~2×** — Numba/Python vs C++；同 3s 可挖節點少一半量級。  
3. **深度 = f(樹瘦 × NPS)** — 兩者疊加 → timed 深度遠低於 SF11；戰術題仍可高過 97% 因多數短殺有 TT/延伸，但高難題與對局 Elo 仍吃虧。  
4. **LMR 邊角常數已掃完** — P0/P2–P6/soft 皆無正邊際；**僅 P1** 有效。下一刀應在結構或 eval，不是再 ±256 LMR。

### 9.3 結構差距分析（R3 平台後，2026-07-15）

座標下降 R1–R3 證明：**調 margin 無法再擠出 pass/nodes**。以下對照 SF11 `search.cpp` 與現行 `search.py`，找「非係數」肥源。

#### A. 診斷重述（node_bench d10 + ablation）

| 現象 | 證據 | 含義 |
| :--- | :--- | :--- |
| 同深度 ~2.5–4× SF11 | start 3.9×、pos4 4.6×、quiet 3.4× | 樹形/剪枝觸發，非純 NPS |
| 排序正常 | cut_first ~91%、avg_move_at_cut ~1.2 | **不是** move ordering 崩壞 |
| LMR 有效但已到頂 | 關 LMR → 2.89×；R3 調 cutnode/base/hist 全不動 | 再硬 LMR 邊角會掉 pass |
| LMP 第二 | 關 LMP → 2.45× | 與 LMR 同屬「已在用」的大槓桿 |
| NMP 中等 | 關 NMP → 1.18×；部分 FEN `nmp_cut=0`（如 pos3） | **觸發率**可能偏低 |
| ProbCut 反常 | 關 ProbCut → 0.95×（略瘦） | 實作/尺度可能有害 |
| QS 佔比 | q ≈ 26–35% total | QS 有體積，但主肥在 search 本體 |
| LMR research | 多局面 2–7% | 未見 re-search 爆炸 |

#### B. NMP：最大結構嫌疑

| 項目 | SF11 | 我們 | 影響 |
| :--- | :--- | :--- | :--- |
| 節點類型 | **`!PvNode`**（所有非 PV） | **`cut_node` only** | ALL-node 完全不做 NMP → 少剪 |
| static 門檻 d=10 !imp | `eval≥β` 為主（公式約 β−28） | **static ≥ β+60** | **明顯更難觸發** |
| static 門檻 d=10 imp | `eval≥β` | static ≥ β+15 | 仍略嚴 |
| 削減 R (d=10) | **(854+68d)/258 ≈ 5** + margin/192 | **7+d//3 = 10** + margin/150 | 觸發後反而減更深 |
| 其他閘 | `eval≥staticEval`、`(ss-1)->statScore`、npm | `side_non_pawn≥2`、前步非 null | 細節不同 |

**結論：** 我們是「**難觸發 + 觸發後猛砍深度**」。消融只顯示 +18% 節點（關 NMP），是因為**現行觸發已經偏少**；若對齊 SF 的 NonPV 門檻，理論空間 > 消融數字。

建議實驗 **S1（優先）**：  
1. NMP 改 `not is_pv`（對齊 NonPV，不再要求 cut_node）；  
2. 門檻改 SF11：`static >= beta` 且 `static >= beta - 32*d + 292 - 30*improving`（cp 可先 0.78 縮放試一版）；  
3. R 改 `(854+68*d)//258 + min((static-beta)//150, 3)`。  
閘：full@d10 pass ≥ 458−2 且 nodes ≤ 0.95×；再 3s half。

#### C. LMR：係數平台的結構含義

數值對照（improving，1024-scale → plies）：

| d, mc | SF 整 ply | 我們 base(+1027) | +LMR_BASE 512 | +cutnode 2048 |
| ---: | ---: | ---: | ---: | ---: |
| 10, 5 | 2 | 3.2 | 3.7 | 5.7 |
| 10, 8 | 3 | 3.8 | 4.3 | 6.3 |

→ **base LMR 已不比 SF 軟**；R3 調 cutnode/no_tt/ttcap/mc 全鎖原值。  
樹仍肥 ⇒ 肥源更可能是：

1. **NMP/ProbCut/eval 觸發**（見 B、D），不是 r 再 +256；  
2. **LMR 資格**：我們對幾乎所有 capture 做 LMR；SF11 較嚴（cutNode / static+victim≤α / late 等）。P2 曾試嚴化 → pass 掉、nodes 未降 → 單純抄 gate 不夠；  
3. **1024 路徑與 SF 整 ply 混搭**（history 除數、mc_factor 40 vs SF 62）已掃過，邊際耗盡。

#### D. ProbCut / RFP / SE

| 技術 | SF11 | 我們 | 備註 |
| :--- | :--- | :--- | :--- |
| ProbCut raisedβ | `β + 189 − 45·imp` | 固定 **+150** | 消融顯示關了更瘦 → **S2** 對齊公式或暫時關 |
| RFP margin | `217·(d−imp)`，d&lt;6 | `170·(d−imp')` | 係數已掃；結構接近 |
| SE 深度 | d≥6 | d≥7（+tt_pv） | 略晚；影響 Elo 多於 fixed-d10 nodes |
| SE margin | `2·depth` | `(60+…)*d//59` | 近似 SF18 系；非主肥 |
| Multicut | SE 失敗且 singularβ≥β 可剪 | `ENABLE_MULTICUT=False` | 曾回歸；需重設計才開 |

#### E. Eval 尺度（跨層）

HCE 相對 SF11 internal 約 **×0.78**（pawn MG 100/128）。  
所有「eval 與 β 比大小」的剪枝（NMP/RFP/ProbCut/razor）在**未統一縮放門檻**時會系統性偏觸發。  
R1–R3 掃的是**單一 margin 常數**，無法修「公式形狀 × 尺度」交互。

#### F. 優先級（實驗路線，仍一次 JIT / 結構 A/B）

| ID | 改動 | 期望 | 風險 |
| :--- | :--- | :--- | :--- |
| **S1** | NMP：NonPV + SF11 門檻 + SF11 R | nodes↓ 明顯；pos3 等 nmp_cut 上升 | 過度剪 → pass↓；需 full@d10 |
| **S2** | ProbCut：對齊 `189−45·imp` 或 flag 關 | 修 0.95× 反常 | 小 |
| **S3** | NMP/RFP/ProbCut 門檻統一 ×0.78 尺度表 | 觸發率更接近 SF 比例 | 需 diag 對照 |
| **S4** | QS/delta 結構（非再掃 −20） | 降 q% | P0 已失敗一次，低優先 |
| **S5** | SE/multicut 重設 | Elo；d10 nodes 次要 | 高風險 |

**不做：** 再一輪 margin 座標下降（R3 已證平台）。

**一次編譯結構 A/B（完成，2026-07-15）：**  
`tune[]`：`TUNE_NMP_SCOPE/GATE/R`、`TUNE_PROBCUT_STYLE`（生產預設全 0）。  
腳本 `scratch/struct_ab_once.py` — 12 packs、一次 JIT、full@d10 + 3s。  
日誌：`struct_ab_once.log` / `struct_ab_once_results.json`。

| pack | pass | nodes×base | 解讀 |
| :--- | ---: | ---: | :--- |
| **baseline** | **458** | **1.000** | **最優（採納）** |
| S1a_nonpv | 451 | 0.986 | 唯一下 nodes；pass −7 → 不採 |
| S1_full_raw / sf78 | 455/454 | 1.06–1.07 | 結構全開 **更肥且掉 pass** |
| S1b_gate / S1c_r | 446–450 | 1.03–1.05 | 單改門檻或 R 皆差 |
| S2_pc_sf78 / off | 450/445 | 1.07 | ProbCut 改/關皆差（舊消融 0.95× 不復現於 R2 棧） |
| 組合 | ≤452 | ≥1.02 | 無協同增益 |

3s half baseline：**242/250**（timed 噪；fixed d10 仍 458）。

**結論（結構 S1/S2）：**  
直接「對齊 SF11 NMP/ProbCut 公式」在 **現行 HCE×0.78 + R2 剪枝鎖** 下 **全部失敗**。  
S1a 證實 non-PV NMP 可略瘦樹，但戰術 pass 代價過大。  
→ 樹肥主因**不是**「NMP 沒抄 SF 公式」 alone；更可能是 **eval 形狀/尺度使同一公式最優點不同**，或 **LMR/LMP 已在 pass 邊界上最優**（R3）。  
生產：結構 mode 保持 0；保留 `tune[]` 供後續。

#### G. 成功指標（沿用）

- 主閘：full 500 @ **depth 10** pass ≥ **456**，nodes 目標 **≤ 0.90×** R2（20.43M → ≤18.4M）  
- 副閘：3s half ≥ **244/250**（允許 ±2 timed 噪）  
- diag：nmp_cut 顯著↑且 lmr_research 不爆炸（&lt;12%）

#### H. E2 LMR 表係數（一次 JIT；對戰後**撤回** scale=70）

腳本 `scratch/lmr_table_sweep.py`：`TUNE_LMR_TABLE_SCALE` / `TUNE_LMR_NOT_IMP`。

| scale% | full@d10 | 固定 200k 對戰 vs Old |
| ---: | :--- | :--- |
| 70 | pass **464**、nodes **1.25×** | depth 明顯↓；無 Elo 證 |
| 85 | — | 66 局 score 45.5%、Elo −32、depth **−0.66** |
| **100** | pass 458、nodes 基準 | 66 局 score 47%、Elo −21、depth **−0.22**（最接近 Old） |

- 3s half（scale=70）：244/250  
- **2026-07-16 決策：生產 `LMR_TABLE_SCALE_PERCENT = 100`（撤回 70）**  
- 完整戰役報告：[`tournament_analysis/REPORT_2026-07-15_16_search_campaign.md`](../../tournament_analysis/REPORT_2026-07-15_16_search_campaign.md)

**2026-07-16 搜尋參數對齊 classical_old：**  
`FP` 180/130、`LMP_SCALE` 100、`LMR_HISTORY` 150、`LMR_TABLE_SCALE` 100、  
`LMR_BAD/GOOD/KILLER` **1024**（撤回 P1 1536）。  
保留 `tune[]`/diag 基礎設施（預設值=舊行為）。

**LMR capture victim（正確值）：** `LMR_CAPTURE_VICTIM_SCALE * MG_MATERIAL_VALUES[victim] // LMR_CAPTURE_VICTIM_DIV + cap_hist`  
（對齊 SF18 `PieceValue[captured]`；不再用舊硬編碼 100/320/330/500/900）。  
對應 MG：P100 / N345 / B376 / R578 / Q1178。

**2026-07-16 參數集中：** search 內重要可調係數（NMP/LMR/QS smooth/history bonus/singular/…）  
已移入 `constants.py`「Search internals」區塊；`search.py` / `search_heuristics.py` 只引用命名常數。

#### J. NMP full@d11 戰役（2026-07-16，一次 JIT）— **對戰後撤回 BASE=80**

腳本：`scratch/nmp_d11_campaign.py`；結果：`scratch/nmp_d11_campaign_results.json`。  
對戰：`tournament_results_nmp_base80.pgn` / `stats_nmp_base80/`。

| 指標 | baseline (BASE=200) | gate_base_80（曾試） |
| :--- | ---: | ---: |
| full@d11 pass | 455/500 | 472/500（+17） |
| sum_nodes | 34.87M | 34.40M（0.986×） |
| NMP_SAVE | 11.3% | 12.5% |
| **@300k nodes 對戰 300 局** | — | **142.5–157.5（47.5%），Elo −17.4 [−45, +10]，64W–157D–79L** |

- **決策：不採納（已回退）。** 生產維持 `NMP_LEGACY_BASE = 200`。  
- SPRT LLR −1.21（Continue，方向穩定偏負；使用者判定落後結案）。  
- 基建保留：`tune` gate=3 / scope=2 / R=2 / funnel diag（預設仍 legacy 行為）。  
- 教訓：puzzle@fixed-d 升 pass ≠ 固定節點對局 Elo；略放寬 NMP gate 可傷決勝局。

#### I. E1 詳細 eval 對照（2026-07-15，solo）

腳本：`scratch/eval_compare_sf11_detail.py`  
輸出：`eval_compare_sf11_detail.txt` / `.json`  
（曾卡在 Windows 上 SF `readline` 無超時；已改 thread+timeout。）

**Root static totals（white cp，`evaluate` + tempo 路徑）：**

| FEN | our_W | sf_W | diff | ratio |
| :--- | ---: | ---: | ---: | ---: |
| start | 22 | 13 | +9 | 1.69 |
| kiwipete | 26 | 36 | −10 | 0.72 |
| pos3 | 63 | 69 | −6 | 0.91 |
| mid1 | 64 | 23 | +41 | 2.78 |
| mid2 | 153 | 75 | +78 | 2.04 |
| mid3 | −262 | −270 | +8 | 0.97 |
| pos4 | 94 | n/a | — | SF `none (in check)` |
| tact1 / quiet1 / end1 | 見報告 | 見報告 | — | 部分 term 淨和≈0 |

- **mean ratio ≈ 1.55**（n=7，非 0.78）  
- **ratio 散布 0.72–2.78** → **不是純尺度**，是 **landscape 失配**  
- mean |diff| ≈ **23 cp**（root 已如此，搜尋樹內會放大）

**Mapped term MAD（優先修的 bundle）：**

1. **Material+pieces_PST** MAD 36.6（worst kiwipete）  
2. **Mobility+outpost** MAD 31.1（worst pos4）  
3. **King safety** MAD 18.2  
4. **Threats** MAD 16.9  
5. Pawns / Imbalance 次之  

**結論：** HCE「結構像 SF」≠ root static 對齊；剪枝常數掃不動樹肥與此一致。下一刀應針對 **Material/PST 與 Mobility** 做 term 級對照修，而非再掃 LMR margin。

#### J. 下一刀

1. Term 級修：**Material+PST**、**Mobility**（E1 MAD 最大）  
2. **NPS / 熱路徑**（長期）  
3. 不再掃「更硬 LMR 減樹」（E2 已證與 pass 對立）

#### K. 搜尋核心稽核與 opt-in 診斷（2026-07-30）

本輪凍結 HCE，未修改評估參數，也未執行 tournament。改動集中在搜尋正確性、
NPS 熱路徑與可觀測性：

1. **修正 singular exclusion terminal：** exclusion search 沒有替代合法著時，
   原本錯回真實將死／和棋；現依 SF11/SF18 Step 20 回傳 `original_alpha`（fail-low）。
2. **capture LMR 的 SEE 延遲：** 舊路徑在每個 capture 進入 move loop 時重算 SEE(0)，
   但結果只供 capture LMR 使用。現僅在 `depth >= LMR_MIN_DEPTH`、不是第一個已搜尋著、
   且 capture 可能進入 LMR 時，在父局面計算。Kiwipete d14 的 LMR SEE 呼叫為
   **27,258**，相對 MovePicker capture SEE **374,241**，避免約 **347k** 次重複 SEE。
3. **診斷改為 opt-in：** `SearchContext(..., enable_diagnostics=False)` 為正式預設；
   benchmark/diagnostic 工具明確啟用。新增 qsearch、SEE 與 extension funnel。
4. **清理：** 移除未被引用的 `search.py` imports、重複 terminal 分支，以及 package
   內兩份 `search_heuristics_fullhist_backup.py` 暫存備份。

Kiwipete d14（同一棵樹）驗證：`901,038` total nodes、score `-87`、best `e2a6`、
PV 完全一致。三組同進程中位數：diag on 約 **529k NPS**、off 約 **549k NPS**
（診斷成本約 3.7%；效能數字受 Windows 排程影響，結論以移除的重複 SEE 次數為主）。

d10 13-position diagnostic：650,943 nodes、qnodes 45.4%、cut-first 90.3%、
LMR re-search 4.9%、失敗 0；戰術 gate d12 **11/11**。100k fixed-node 13-position：
平均 102,851（+2.85%）、最大 103,913（+3.91%）、失敗 0。

新診斷指出下一個主要成本候選不是 LMR/NMP：Kiwipete d14 有約 **2.07M** 次
main quiet-SEE 呼叫（其中 543,704 次 prune），qsearch 產生 1.36M pseudo moves，
其中約 394k（29%）最後因不合法被丟棄。後續應先做單一結構 A/B（合法著過濾／
quiet SEE reuse），不可直接放寬剪枝門檻；另需修正 qsearch 對 checking capture 的
delta/SEE gate 與 SF11/SF18 `!givesCheck` 條件差異，再過戰術 gate。

#### L. QSearch 逃逸排序、SF check 保護與暖機污染（2026-07-30）

本輪由使用者先將 `classical` 與 `classical_old` 同步；此後只修改新引擎，
`classical_old` 保持凍結供對戰。實作分成兩項：

1. **被將軍逃逸排序：** `score_captures_with_tt_lazy` 原本讓所有非吃子逃逸同為 0 分，
   等同依 move-generator 順序搜尋。現改用既有 quiet history 組合；吃子加
   `QS_EVASION_CAPTURE_BUCKET=20000`，quiet history 限幅於 ±5000，確保 history
   只排列 quiet 彼此，不跨越 capture bucket。
2. **SF11/SF18 checking-delta 保護：** qsearch move 若會將軍，不因 delta/futility
   被跳過；最終 `QS_SEE_THRESHOLD=-80` 仍生效。新增 opt-in
   `DIAG_Q_CHECK_PROTECT`。普通著以前置 check geometry 判斷，promotion / EP 用精確
   暫時 make/unmake。

固定深度 d10（13 positions）與 tactical d12（11 題）A/B：

| 版本 | d10 nodes | 相對凍結舊版 | tactical | tactical nodes |
| :--- | ---: | ---: | :---: | ---: |
| 凍結舊版 / 本輪起點 | 650,943 | — | 11/11 | 85,483 |
| 僅逃逸 history bucket | 620,652 | −4.65% | 11/11 | 80,467 |
| **最終：bucket + SF check 保護** | **822,304** | **+26.32%** | **11/11** | **78,128** |

SF 保護讓此小型戰術集再少 2.9% nodes，但一般固定深度樹明顯增大；這是刻意保留的
正確性／戰術安全取捨，不應描述為節點效率提升。固定 100k nodes（13 positions）最終
結果：sum 1,331,860、mean 102,450、max 104,006、failures 0。完整
`tests/test_search.py` 162 positions 通過：mean NPS 861,996、median 906,768、mean depth
13.14（時間制數字受排程影響，只作穩定性回歸）。

另修正 `assistant.py` 暖機污染：Kiwipete depth-15 暖機結束後原本保留 TT 與所有
history/correction tables。現在收到暖機 `bestmove` 後立刻送 `ucinewgame`，保留 JIT
machine code、清除棋局搜尋狀態。`scratch/search_state_pollution_probe.py` 的 d10 四象限：

| 初始狀態 | move / score | nodes |
| :--- | :--- | ---: |
| clean | `c8d7` / −76 | 71,410 |
| history only | `c8d7` / −70 | 92,238 |
| TT only | `c8d7` / −76 | 71,410 |
| TT + history | `c8d7` / −70 | 92,238 |

此例確認差異來自不相關的 history/correction 狀態（+29.2% nodes），不是 TT；走法仍
一致，因此是選擇性搜尋對排序的敏感性／暖機污染證據，尚不足以證明 history 索引損壞。

#### M. QSearch evasion prefilter、主搜尋 check 保護與 UCI history 生命週期（2026-07-30）

1. **被將逃逸合法性前置過濾：** qsearch 先計算 checker bitboard。雙將只允許王移動；
   單將的普通非王著必須吃 checker 或走到 `SQUARES_BETWEEN`，並通過 pin ray。EP 與無法
   靜態證明的邊角仍保留 post-make 驗證。d10 13-position 診斷中，204,599 個非法 qsearch
   候選有 **202,727（99.1%）** 在 make/unmake 前排除；Kiwipete 為 96,317/97,023。
2. **主搜尋 checking-SEE 路由：** 原本 ordinary quiet check 在 gives-check 判定前可能被
   quiet `lmrDepth²` SEE gate 剪掉。現用既有 direct/discovered-check geometry 前置分類，
   將 checking quiet 對齊 SF11/SF18 的 capture/check SEE margin；仍可剪明顯不成立的犧牲，
   但不再套 quiet-history gate。d10 suite 觸發 26,946 次。
3. **History/UCI 生命週期：** `main.py` 原本每次 `go` 重建 `SearchContext`，使 context-owned
   counter move、low-ply history 與大型 pawn/search buffers 每步重置／重配，和 tournament
   in-process backend 不一致。現改為每個引擎 process 配置一次，`ucinewgame` 明確清除棋局
   heuristic/stack 狀態。表能量診斷未見普遍飽和，因此不重開失敗的 full-history/soft-age；
   同時移除宣稱 SOFT800 生效但實際未讀取的暫存常數，文件鎖定現行 Phase 1a DIV2。
4. **其他護欄稽核：** NMP/RFP/razor/ProbCut 既有 in-check／known-win 防護保留；quiet
   history/LMP/FP 仍要求 `!givesCheck`；IIR 保留 `!inCheck && !followPV`；checking LMR 至少
   relief 1 ply；alpha-raise depth reduction 仍禁止 root 與 decisive score。本輪未再加重
   LMR 或改 pruning margin。

驗證結果：

| 測試 | 結果 |
| :--- | :--- |
| d10 13 positions | 870,909 nodes；相對上一版 822,304 為 +5.91%（checking 安全路由的樹成本） |
| tactical d12 | 11/11；44,532 nodes（上一版 78,128） |
| fixed 100k ×13 | sum 1,335,401；failures 0（上一版 1,331,860，+0.27%） |
| `tests/test_search.py` | 162/162 完成；mean/median NPS 941,090/1,016,800；mean depth 13.30 |
| 同回合修改前 `tests/test_search.py` | mean/median NPS 816,966/905,072；mean depth 12.93 |
| 輕量契約 + history | 15/15 通過 |

時間制 NPS 受 Windows 排程影響，但同回合 mean/median 分別約 +15.2%/+12.3%，且固定節點、
戰術與合法性 gate 無失敗。`classical_old` 保持凍結，未同步。

## 14. 2026-08-04 同步後搜尋診斷

使用者在固定節點對戰確認新版較強後，已自行同步新舊引擎。同步後的 13-position d10 基線仍為
870,909 nodes，tactical d12 為 11/11、44,532 nodes；另完成 500 戰術 + 500 開局／安靜盤面的
n30k baseline 與 P0 複驗。P0 v2 曾將 qsearch in-check evasion prefilter 移到 scorer 前，以 stable
in-place compaction 排除明確非法候選；P0 v1/v2 的 1000 個 FEN 完全 deterministic，但在
4-worker wall time 未證明 NPS 提升，後續固定節點對戰也未接納 P0/P1/P3 組合。現行 production
已回到 scorer／排序後 legality，三者只保留 shadow diagnostics。後續仍以 n30k 為正式搜尋診斷，
並由新舊引擎對戰決定接納。回退後 500+500 n30k 與同步前 baseline 逐盤的 total nodes、qsearch
nodes、完成深度均 1000/1000 完全一致，確認 production 搜尋樹已精確還原。完整數據、
SF11/SF18 對照、風險與測試流程見 [`SEARCH_OPTIMIZATION_DIAGNOSTIC_20260804.md`](SEARCH_OPTIMIZATION_DIAGNOSTIC_20260804.md)。

### 14.1 P4 每節點品質診斷（2026-08-04）

在 production 回退且使用者重新同步新舊引擎後，新增 opt-in shadow counters，直接量測 move
source cutoff precision、killer/counter precision、quiet cutoff rank、LMR reduction 分桶、re-search
keep/reject，以及 SF11/SF18 類 post-LMR continuation-history 的反事實樣本。沒有改排序、縮減或
history update。正式 500+500 n30k 與回退基準逐盤的 total/search/qsearch nodes、depth 全部
1000/1000 一致。

raw-weighted 合計：TT／good capture／good quiet cutoff precision 71.16%／56.16%／24.09%，
bad capture／bad quiet 2.72%／0.70%；quiet cutoff 在實際搜尋 quiet 中平均 rank 1.304。LMR R1／R2／R3+ reduced
fail-high 為 6.16%／4.46%／1.45%，較深縮減沒有顯示全域漏搜。值得研究的是 verification：
37.36% 的已驗證 reduced fail-high 被推翻，而 butterfly 與現有 LMR statScore 在 keep/reject 樣本
仍大多為正。因此當時將有界 post-LMR continuation-history bonus/malus 列為單軸候選，
後續 production A/B 結果見 §14.2；完整定義與拆分數據見診斷報告 §10。

### 14.2 P4 production A/B 結果（全部回退）

依 P4 診斷實作並完整測試四版：post-LMR keep/reject 對稱 cap 900、非對稱 keep 600/reject 300、
reject-only 300，以及在 LMR statScore 加入 butterfly from-to。合計 mean depth 依序為
10.390、10.386、10.396、10.357，全部低於純診斷基準 10.416；配對提升／退化依序為
141/148、99/125、85/103、163/190。強雙向版使一個戰術盤 depth 30→7，完整 butterfly 版則
30→14。所有 production 寫入與 statScore 變更均已回退，P4 counters 保留。
回退後的最終 n30k 複驗與純診斷基準在 1000/1000 盤的全部搜尋語意欄位完全
一致，僅 `elapsed_s` 不同；輸出為 `hist_diag_p4_production_ab_rollback_final_20260804_n30k.json`。

這次確認 continuation history、history pruning 與整 ply LMR 量化的耦合比 SF 原始碼表面公式更
強；不能因 SF11/SF18 都有某個更新就直接移植。下一輪應先做連續走步／暖 history attribution，
特別量測哪一個 continuation layer 與哪一次 history prune 造成路徑翻轉。正式矩陣見診斷報告 §10.4。

### 14.3 P5 暖 history 與 history-prune attribution

新增純 shadow 的 8 層 prune attribution，並讓 `hist_diag_bench.py` 能以同一 context
沿最佳著前進，在每個到達局面配對 fresh context。每步預設清 TT，避免把上一輪
子樹 TT 命中誤當 history 收益。100+100 seeds、3 ply、warm/fresh 各 n30k 的 ply-0
控制 200/200 完全一致。

opening ply 1/2 的 warm 平均深度 8.06/8.00，fresh 為 7.96/7.86，但 best-move
一致率只有 56%/46%，平均評分差約 20.4/18.9 cp。history prune 由 fresh 的
42,730/45,020 降為 warm 6,706/2,082，證實 cold-per-FEN benchmark 不能單獨用來
決定 history 改動。warm opening ply1+ 中 cont1 是單層 decisive 56.1%、dominant 40.9%；
puzzle 為 74.6%/77.5%。下一個最小候選是只對 history-prune 路徑的 weighted cont1
負尾限幅，不動排序、LMR 或寫入。完整定義與數據見診斷報告 §11。
另以 cold 500+500 n30k 完整複驗 P5 shadow；與 P4 基準的所有既有逐盤語意欄位
1000/1000 精確一致，mean nodes/depth 維持 31,612.003/10.416。

### 14.4 P6 history-prune cont1 負尾保護

依 P5 attribution，只在 history-prune 的 composite read 將 weighted cont1 設下限 −4096；
`get_quiet_stat_score()` 的預設值、quiet ordering、LMR statScore、history 寫入與其餘 continuation
層完全不變。另加 cap hit、relief 與 raw-would-prune/capped-would-not 的 rescue counters。

cold 500+500 n30k 共觸發 653,259 次 cap，救回 11,489 次（1.76%）；history prune 由
293,646 降為 280,398（−4.51%）。mean nodes/depth 為 31,600.384/10.411，基準為
31,612.003/10.416。puzzle 最佳著 500/500 不變；opening 496/500 不變，4 個變更的
score 差均不超過 41 cp。深度配對為提升 5、相同 988、降低 7。

warm 100+100 seeds、3 ply、每步清 TT 的正式複驗中，P6 與 P5 的可比 puzzle 位置
最佳著 300/300 相同；opening 為 292/298（97.99%）。P6 warm/fresh opening 最佳著
一致率 ply1/2 為 56%/47%，P5 為 56%/46%；平均 score 差由 20.43/18.86 cp
降為 19.38/17.98 cp，沒有放大 warm-history 不穩定。此候選保留進固定節點新舊引擎對戰。

### 14.5 搜尋參數調參基建（2026-08-04）

P6 對戰結果為平手，因此不宣稱 P6 有 Elo 收益，暫作目前搜尋基線。調參前先建立可恢復備份
`archive/search_tuning_p6_backup_20260804.zip`（SHA-256
`6AF9241F50E04D9B4F700A94AE4EDA50E77987F56DB552D88D6EF6110B693DF0`）。

`tune_search/search_param_registry.py` 目前盤點 114 個真正存在且被搜尋路徑引用的 scalar：34 個
映射到 `SearchContext.tune[]`，其餘是需要新進程/JIT 的 compile-time 候選。原先配置中的
`FP_MARGIN_D1`、`FP_MARGIN_D2`、`RFP_MARGIN_D1`、`SINGULAR_EXTENSION_MARGIN` 已從可調清單移除，
避免調參器表面更新、引擎實際忽略的假改善。調參器預設以 runtime 34 軸為安全 profile，並提供
`pruning`、`nmp`、`lmr`、`qsearch`、`history`、`ordering`、`safeguards`、`aspiration` 等分組；
不建議把 114 軸一次交給 SPSA。每軸使用自己的 step/c/a 尺度，離散 mode 需獨立 A/B 或小 block。

NMP gate/R、ProbCut style、NMP beta requirement 的預設值也由常量明確保存（仍全為 0，生產行為不變），
因此 adapter 可以在首次 JIT 前選擇它們，未來 runtime fixed-node matcher 則可直接覆寫 tune slot。
本節只涉及調參入口與可追溯性，沒有新增搜尋剪枝或改變 P6 搜尋樹。
