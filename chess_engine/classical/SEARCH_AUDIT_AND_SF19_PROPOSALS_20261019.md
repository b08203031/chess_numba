# 經典搜尋演算法深度審計與 Stockfish 19 對齊優化報告 (2026-10-19)

本報告針對經典搜尋模組（[`chess_engine/classical/search.py`](search.py)、[`search_heuristics.py`](search_heuristics.py)、[`constants.py`](constants.py)、[`transposition_table.py`](transposition_table.py)）進行全檔逐行審計，梳理 SPSA 參數調優現狀、記錄程式碼中發現的潛在邏輯缺陷與重大 Bug，並對照 **Stockfish 19**（[`stockfish_repo/src/search.cpp`](../../stockfish_repo/src/search.cpp)、[`movepick.cpp`](../../stockfish_repo/src/movepick.cpp)）提出下一階段旨在提升 Elo 分數的架構與演算法優化方案。

---

## 1. SPSA 調優現況與近期演進總結

### 1.1 SPSA 架構與參數調優協定
* **單次編譯 In-Process 架構**：[`search_param_registry.py`](../../tune_search/search_param_registry.py) 管理 114+ 個搜尋參數，其中 **42 個熱路徑參數** 映射至 [`SearchContext.tune[0..41]`](engine_types.py)，支援 `--backend inprocess --scale-mode integer`，完全避免多進程重複 JIT 編譯開銷。
* **兩階段驗證流程**：
  1. **L1 SPSA**：Colab GPU/CPU 叢集 50k–200k nodes，採 Polyak-Ruppert 尾部平均（`--tail 0.30`）。
  2. **一維流形阻尼（1D Manifold Damping）**：若全量更新向量 $\vec{v} = \vec{\theta}_{\text{tail30}} - \vec{\theta}_{\text{start}}$ 在長局數出現過度剪枝，透過 $\vec{\theta}(\alpha) = \text{round}(\vec{\theta}_{\text{start}} + \alpha \vec{v})$ 沿梯度主軸進行線搜尋。
  3. **L2 LTC 驗收**：本機單進程共享 JIT，以 **300,000 nodes/move**（400–1,000 局，64MB TT，SPRT 門檻 +3.00 Elo）作為最終落地準則。

### 1.2 近期重大演進戰果（E46–E71.1）
* **E46–E53**：
  * Razoring 對齊 SF19 Step 8（`482 * d^2`，+6.08 Elo）
  * 動態 ProbCut（`improving ? 5 : 3`，+12.17 Elo）
  * LMR ALL-Node 深度自適應縮放（+6.95 Elo）
  * 徹底廢除 Check Extension（E50，+22.62 Elo，平均搜尋深度躍升 +1.20 ply）
  * Continuation History 剪枝門檻下修至 `-1500`（E53，+18.26 Elo）
* **Phase 5 (E64)**：
  * 前向剪枝參數群 SPSA（`LMP_SCALE_PERCENT=137`, `FP_BASE=174`, `FP_MULTIPLIER=96`, `RFP_BASE_MULT=168`，+8.70 Elo）
* **Phase 6 (Plan A)**：
  * 6 軸 History/LMR SPSA + SF19 深度自適應 LMR 除數 (`dyn_div = 4096 + max(0, depth - 10) * 128`) + `HISTORY_MALUS_CAP=1600`（+19.13 Elo）
* **Phase 8 (NMP SF19 對齊)**：
  * 對齊 SF19 Step 10 極簡無狀態架構（移除離散模式、引進 `cut_node` 門檻與 `nmp_min_ply` 驗證）
  * 修復 Full Eval 縮排 Bug 與 Numba 結構化陣列 `tt_entry` 記憶體指標別名跨搜尋重用污染 Bug（+12.17 Elo, +0.47 ply）
* **E65–E66 (奇異延伸 SE 對齊)**：
  * Pre-make SE + Multi-Cut、`cutoff_cnt` 跨層隔離、解鎖排除搜尋 Step 14 剪枝、HCE 100cp 縮放（+17.39 Elo）
* **E67 / E69**：
  * 零樹變動熱路徑加速（重用 `STAGE_BAD_CAPTURES` 消除重複 SEE、傳遞已知棋子型別，+10.21% NPS）
  * ProbCut TT 寫入深度對齊 SF19 `max(1, depth - probcut_r + 1)`（+2.90 Elo）
* **Phase 12 (E71 / E71.1)**：
  * 修復 Capture FP 槽位、重複安靜升變生成、`_see_ge_jit` 空位標記、ProbCut `!is_pv` 守衛。
  * 發現 120k $\to$ 300k 的「尺度反轉」現象（短節點過度削減），採用 1D 流形阻尼線搜尋，$\alpha=0.40$（E71.1）以 **102W 303D 95L (+4.86 Elo)** 擊敗 $\alpha=0.50$，累計較 E69 **+11.8 Elo**，已合入當前 HEAD（`8b2c579`）。

---

## 2. 搜尋演算法審計：發現的 4 個關鍵邏輯缺陷與 Bug

### 🔴 Bug 1：Fail-Low (ALL-Node) 強制回退 `best_move = moves[0]`，覆寫並嚴重污染置換表
* **檔案與行號**：[`chess_engine/classical/search.py:3275-3280`](search.py#L3275-L3280) 與 [`chess_engine/classical/transposition_table.py:180-184`](transposition_table.py#L180-L184)
* **原始程式碼**：
  ```python
  if best_move == NO_MOVE and search_context.mp_captures_end[ply] + search_context.mp_quiets_end[ply] > 0:
      # Fallback to first move that is not excluded
      for i in range(search_context.mp_captures_end[ply] + search_context.mp_quiets_end[ply]):
          if moves[i] != excluded_move:
              best_move = moves[i]
              break
  ```
* **問題機制分析**：
  1. **破壞置換表中已有的合法 `best_move`**：
     在 [`store_tt()`](transposition_table.py#L181-L183) 中，原設計包含標準 Stockfish 保護邏輯：
     ```python
     final_best_move = np.uint16(best_move)
     if final_best_move == 0:
         final_best_move = existing_entry['best_move']
     ```
     意即當節點 Fail-Low 且沒有找到提升 Alpha 的最佳步時，應保留先前記住的 `best_move`。
     然而在 `_search` 的 MovePicker 設計中，`STAGE_TT_MOVE` 在處理既有 `tt_move` 後，會在生成吃子與安靜步階段**將 `tt_move` 從 `moves` 緩衝區徹底剔除**。因此，`moves[0]` 絕對不可能是 `tt_move`！
     當一個曾有合法 `tt_move` 的局面在更深層搜尋或不同視窗下 Fail-Low 時，第 3278 行把 `best_move` 設為 `moves[0]`（非 0），使得 `store_tt` 認為找到了新最佳步，**直接將原本的正確 `best_move` 覆寫摧毀，換成一個隨機失敗的 `moves[0]`**！
  2. **污染全新 ALL-Node，導致多項啟發與剪枝機制失常**：
     在 Stockfish 19（[`search.cpp:1505`](../../stockfish_repo/src/search.cpp#L1505)）中，ALL-Node 存入 TT 的最佳步一律為 `Move::none()`（`0`）。而本專案中所有 Fail-Low 節點都被強行塞入了 `moves[0]`，在後續造訪該局面時引發嚴重連鎖反應：
     * **IIR 完全失效**：[`search.py:1948`](search.py#L1948) 的 `if tt_move == NO_MOVE:` 在該節點永遠為 `False`，無法發揮 IIR 的自適應深度調節功能。
     * **RFP 被異常阻擋**：[`search.py:1619`](search.py#L1619) 的 `and (tt_move == NO_MOVE or tt_capture)`，因被塞入偽造的安靜步 `moves[0]`，導致 RFP 遭到錯誤攔截無法剪枝。
     * **LMR 獎懲失真**：[`search.py:2735`](search.py#L2735) `TUNE_LMR_NO_TTMOVE` 懲罰無法觸發；若 `moves[0]` 剛好是吃子步，還會誤判為 `tt_capture = True`，對所有安靜步錯誤加罰。
     * **緩衝區索引越界風險**：[`search.py:1105`](search.py#L1105) 中 `mp_quiets_end[ply]` 已是 `captures_end + num_quiets`，第 3275 行計算上限寫成 `mp_captures_end + mp_quiets_end` 導致雙重計算，可能讀取到未初始化的記憶體區間。
* **修復方案**：
  移除 `ply > 0` 的 fallback 邏輯；`ply == 0`（Root）已有 [`iterative_deepening_search:4365-4375`](search.py#L4365-L4375) 的安全保底機制。

---

### 🔴 Bug 2：未執行 LMR (`lmr == 0`) 的全深度零視窗搜尋硬編碼 `cut_node = True` 且無重搜
* **檔案與行號**：[`chess_engine/classical/search.py:2827-2854`](search.py#L2827-L2854) 對照 SF19 [`stockfish_repo/src/search.cpp:1372-1404`](../../stockfish_repo/src/search.cpp#L1372-L1404)
* **原始程式碼**：
  ```python
  # 2.1 Zero-window search (with optional LMR)
  res = _search(
      piece_bbs, occupancy_bbs, game_state, _i32(search_depth - lmr),
      _i32(-alpha - 1), _i32(-alpha), search_context, _i32(ply + 1),
      _u16(NO_MOVE), _b_false(), _b_true(),  # <-- 無條件傳入 cut_node = True
  )
  ```
* **問題機制分析**：
  * 在 SF19 中，只有被 LMR 削減深度的著法（`newDepth - r`）才會以樂觀的 `cutNode = true` 進行低成本探測；若該著法突破 Alpha，則會在全深度以 `!cutNode` 重搜。
  * 若某著法**跳過 LMR**（如 `depth <= 2`，或安靜步緊接在吃子步之後導致 `quiet_move_counter == 1`），SF19（Step 19，[`search.cpp:1403`](../../stockfish_repo/src/search.cpp#L1403)）是以全深度搭配 **`!cutNode`** 搜尋。
  * 本專案在 `lmr == 0` 時，直接以全深度搜尋子節點，卻硬編碼傳入 `cut_node = True`。當該節點突破 Alpha 時，由於 `lmr == 0` 且在 Non-PV 節點 `do_full_pv_search == False`，**引擎完全不進行重搜**，直接採納該結果。這使得原本屬於 `all_node` 的子樹被誤當成 `cut_node`（在子樹內錯誤觸發 NMP 與 CutNode LMR 懲罰）。
* **修復方案**：
  將零視窗呼叫改為：
  ```python
  child_cut_node = _b_true() if lmr > 0 else _b_not(cut_node)
  ```

---

### 🟠 Bug 3：Continuation Correction History 更新區塊完全重複執行兩次
* **檔案與行號**：[`chess_engine/classical/search.py:3344-3374`](search.py#L3344-L3374) 與 [`search.py:3375-3403`](search.py#L3375-L3403)
* **問題機制分析**：
  第 3375–3403 行重複執行了第 3344–3374 行對 `continuation_correction_history`（`ply - 2` 與 `ply - 4`）的更新。這使得每個節點的延續修正值飽和速度異常倍增，且浪費熱路徑 CPU 週期。
* **修復方案**：刪除重複的 3375–3403 行區塊。

---

### 🟠 Bug 4：ProbCut TT Defense 未校驗置換表邊界標誌與深度
* **檔案與行號**：[`chess_engine/classical/search.py:1687-1698`](search.py#L1687-L1698)
* **問題機制分析**：
  程式碼註解載明「若 TT 紀錄為 Upper Bound (`TT_FLAG_ALPHA`) 且分數低於 `probcut_beta` 則跳過 ProbCut」，但實際程式碼僅判斷 `if tt_score_pc < probcut_beta:`，未檢查 `tt_flag` 與 `tt_depth`。如果 TT 存放的是一個低深度的下界分數（`TT_FLAG_BETA`），其真實分數很可能大於等於 `probcut_beta`，卻會被誤判跳過 ProbCut。
* **修復方案**：補齊 `tt_flag == TT_FLAG_ALPHA` 與深度有效性檢查。

---

## 3. 對照 Stockfish 19 的演算法與結構優化提案 (Elo 增長方向)

### 提案 A：前向剪枝執行順序調整（NMP 與 IIR 移至 ProbCut 之前）並補齊 `improving |= static_score >= beta`
* **SF19 對齊依據**：[`stockfish_repo/src/search.cpp:1010-1064`](../../stockfish_repo/src/search.cpp#L1010-L1064)
* **優化價值**：
  1. **消除昂貴的無謂計算**：NMP 僅需一次極淺的空步搜尋（$R \ge 7 + d/3$），不生成著法、不計算 Pin、不執行 SEE；而 ProbCut 需生成全部吃子步、計算 Pin 遮罩、排序、執行多次 SEE、QSearch 與較深的減深搜尋（`depth - 3` 或 `depth - 5`）。將 ProbCut 誤置於 NMP 之前，使許多本可被 NMP 快速截斷的節點先行承受昂貴的 ProbCut 開銷。
  2. **補齊 `improving` 狀態轉移**：SF19 在 NMP 後執行 `improving |= ss->staticEval >= beta;`。當靜態評估已高於 Beta 時將 `improving` 升級為 True，直接為隨後的 ProbCut（深度減少從 3 擴大到 5、門檻放寬）以及主搜尋 LMP 與 LMR 提供更精準的資訊。

### 提案 B：IIR（內部迭代降深）限定於非 ALL-Node（`is_pv or cut_node`）
* **SF19 對齊依據**：[`stockfish_repo/src/search.cpp:1051`](../../stockfish_repo/src/search.cpp#L1051)
* **優化價值**：
  在修復 **Bug 1** 之後，ALL-Node 不再被強塞假 `best_move`，`tt_move == NO_MOVE` 在 ALL-Node 上恢復為常態。若不加上 `(is_pv or cut_node)` 守衛，所有 ALL-Node 都會被無條件降深。

### 提案 C：Step 19 未執行 LMR 時的過渡降深（Fallback Reduction）
* **SF19 對齊依據**：[`stockfish_repo/src/search.cpp:1395-1404`](../../stockfish_repo/src/search.cpp#L1395-L1404)
* **優化價值**：
  目前引擎中，緊接在吃子步之後的第一個安靜步（`quiet_move_counter == 1` 但 `searched_legal_moves >= 2`）完全跳過 LMR。此前 E70 嘗試將其直接納入對數 LMR 導致 -11 Elo；採用 SF19 Step 19，僅在預估削減量 $r$ 極高（如嚴重負歷史）時給予溫和的 1 ply 降深，能在確保戰術安全的前提下節省大量節點。

### 提案 D：RFP 靜態分數阻尼混合（Fail-Soft Score Blending）
* **SF19 對齊依據**：[`stockfish_repo/src/search.cpp:1007`](../../stockfish_repo/src/search.cpp#L1007) `return (661 * beta + 363 * eval) / 1024;`
* **優化價值**：
  目前 RFP 在觸發截斷時直接回傳原始 `static_score`。在 Classical HCE 評估下，靜態評估偶爾會因未深究的戰術假象而虛高，回傳未經阻尼的分數會污染父節點的 Alpha/Beta 視窗。採用 SF19 權重混合可平滑極端值。

### 提案 E：MovePicker `STAGE_GOOD_CAPTURES` 零樹變動 SEE 快速短路（Zero-Tree-Delta NPS 提升）
* **優化依據**：
  在 `STAGE_GOOD_CAPTURES`（門檻為 0）以及 QSearch（門檻 $\le 0$）中，當吃子步為常規吃子（非升變、非吃過路兵）且被吃子價值大於等於吃子方價值（以小吃大，例如兵吃馬、馬吃車、車吃后）時，即使吃子棋子隨後被無償吃回，第一輪交換淨利已 $\ge 0$。其 `SEE >= 0` 在數學上**恆成立（100% 必過）**。
  在此情境下直接判定通過，完全省去昂貴的 `_see_ge_jit` 位元棋盤模擬，保證**搜尋樹節點數 100% 不變**，可顯著提升引擎 NPS。

---

## 4. 執行路線圖與驗證規範

1. **Step 1（已完成）**：實作提案 E 之零樹變動 SEE Fast-Path，於 6 個標準 Perft 盤面執行 Depth 15（各 5 次取平均），量測基準 NPS 提升並確保節點數 100% 一致。
2. **Step 2**：修復 Bug 2（Non-LMR 全深度搜尋 `cut_node` 傳遞錯誤）與 Bug 3（Continuation Correction 重複區塊），進行 A/B 對局驗證。
3. **Step 3**：修復 Bug 1（Fail-Low 污染置換表）並同步啟用提案 B（IIR `!all_node` 守衛）。
4. **Step 4**：重構前向剪枝順序（提案 A：NMP/IIR 前置於 ProbCut，補齊 `improving` 轉移），並修正 Bug 4。

---

## 5. Step 1 實測驗證成果（6 個 Perft 盤面 Depth 15 基準測試）

### 5.1 測試環境與協定
* **測試腳本**：[`tools/bench_nps_perft.py`](../../tools/bench_nps_perft.py)
* **盤面集**：標準 Perft 6 盤面（Startpos、Kiwipete、Position 3、Position 4、Position 5、Position 6）
* **測試參數**：固定搜尋深度 **Depth 15**，各盤面執行 **5 次獨立搜尋** 取平均值；每次搜尋前重置 TT（64MB）與上下文歷史狀態以確保完全確定性。

### 5.2 實測數據對照表

| 盤面名稱 | Baseline 節點數 | Candidate 節點數 | 節點差異 (Tree Delta) | Baseline 時間 (s) | Candidate 時間 (s) | Baseline NPS | Candidate NPS | NPS 變化率 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **startpos** | 599,008 | 599,008 | **0 (100.0%)** | 1.4607s | 1.3173s | 410,082 | 454,736 | **+10.89%** |
| **kiwipete** | 397,536 | 397,536 | **0 (100.0%)** | 1.0774s | 0.8479s | 368,974 | 468,831 | **+27.06%** 🚀 |
| **Position 3** | 337,504 | 337,504 | **0 (100.0%)** | 0.4195s | 0.3908s | 804,551 | 863,715 | **+7.35%** |
| **Position 4** | 262,564 | 262,564 | **0 (100.0%)** | 0.6910s | 0.6445s | 380,003 | 407,371 | **+7.20%** |
| **Position 5** | 1,042,640 | 1,042,640 | **0 (100.0%)** | 2.8514s | 2.5735s | 365,660 | 405,151 | **+10.80%** |
| **Position 6** | 357,960 | 357,960 | **0 (100.0%)** | 0.8864s | 0.8071s | 403,833 | 443,537 | **+9.83%** |
| **總計 / 加權** | **2,997,212** | **2,997,212** | **0 (0.00% 樹變動)** | **7.3863s** | **6.5810s** | **405,777** | **455,434** | **+12.24%** 🚀 |

### 5.3 成果分析
1. **100% 零搜尋樹變動（Zero-Tree-Delta）**：全量 2,997,212 個節點完全吻合，無任何搜尋邏輯失真或戰術剪枝偏差。
2. **NPS 大幅躍升**：在全域 6 盤面平均 NPS 從 405,777 提升至 455,434（**+12.24%**），而在吃子戰術最密集的 **Kiwipete** 盤面更達到驚人的 **+27.06% NPS 提升**，總搜尋耗時縮減 10.90%。


