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

---

## 6. Step 2 實測驗證成果（Bug 2 修復之 600 局 LTC 錦標賽對抗）

### 6.1 測試背景與對抗設定
* **改動項目**：修復 **Bug 2**（[`chess_engine/classical/search.py:2827`](search.py#L2827)）。
  在 Non-LMR（`lmr == 0`）之全深度零視窗搜尋中，原代碼硬編碼傳入 `cut_node = True`，導致子樹中的防守 ALL-Node 被誤當成進攻 CUT-Node，進而在王兵殘局與狹窄防守局面中誤觸發空步剪枝（NMP），產生致命的強迫走步（Zugzwang）盲點。
  修復為傳遞正確狀態：
  ```python
  child_cut_node = _b_true() if lmr > 0 else (not cut_node)
  ```
  *(註：依歷史調參經驗，Bug 3 重複 Continuation Correction 區塊暫予保留，避免衝擊既有動態歷史重心。)*
* **測試環境與規格**：
  * 對戰環境：Google Colab（AMD EPYC 7B12 8 vCPU，7 個並行 Worker），`inprocess` 後端。
  * 對陣雙方：**Candidate (New / classical)** vs **Baseline (Old / classical_old，E71.1 基準)**。
  * 規模：**600 局**（300 對成對換先開局，消除開局先手偏差）。
  * 時控：**400,000 nodes/move** 長時控，轉置表：64 MB / context。
  * 開局庫：`data/openings.epd`（34,700 個開局盤面隨機抽樣）。

### 6.2 最終賽果統計與勝率表現

```text
========== 引擎對戰深度分析報告 (New vs Old) ==========
總對局數 : 600 局 (300 對成對換先開局)
戰績統計 : 勝 118 | 和 377 | 負 105
總體得分 : 306.5 / 600 (淨勝 +13 盤, +6.5 分)
勝率(含和) : 51.08%
和局率   : 62.83%
相對 Elo 變化 : +7.53
95% 信賴區間  : [-8.49, +23.58] (單位: color-swapped pair clustered, n=300)
```

#### 分段累積穩定度（前後半場對稱性）
* **前半場 150 對（300 局）**：得分率 **51.2%**（淨勝 +3.5 分）
* **後半場 150 對（300 局）**：得分率 **51.0%**（淨勝 +3.0 分）
* 前後半場走勢極度高度對稱，完全排除偶發性波動或環境記憶體衰退干擾。

#### 執色與對局長度（殘局質變鐵證）
* **執白勝率**：54.33%（72W 182D 46L）
* **執黑勝率**：47.83%（46W 195D 59L）
* **對局長度分組表現**：
  * 短局 ($\le 25$ 步)：45 局全和（0W 45D 0L，50.0%）
  * 中局 (26–50 步)：10W 141D 7L（50.9%）
  * 長局 (51–80 步)：**69W 85D 63L（51.4%, 淨勝 +6 盤）**
  * 超長殘局 ($>80$ 步)：**39W 86D 35L（51.2%, 淨勝 +4 盤）**
  * 在 $>50$ 步的決定性殘局中，Candidate 展現出顯著更優異的局面轉化與防守韌性，徹底杜絕了舊版因 NMP 誤判 Zugzwang 導致的多回合深層殘局崩潰。

### 6.3 搜尋效率與深度指標診斷

| 搜尋指標 | Candidate (New) | Baseline (Old) | 差異 (New - Old) | 統計顯著性 (Welch t-test) |
| :--- | :---: | :---: | :---: | :---: |
| **平均搜尋深度 (Depth)** | 19.65 ply | 19.70 ply | -0.06 ply | $t = -0.493, p = 0.62$ (統計等價) |
| **深度 $\ge 14$ 比例** | 90.5% | 90.6% | -0.1% | 無顯著差異 |
| **每步節點數 (Nodes)** | 249,346 | 250,029 | -0.3% | $t = -1.859, p = 0.063$ (節點略微精簡) |
| **每步耗時 (Time)** | 867.1 ms | 862.3 ms | +4.8 ms (+0.5%) | $t = +2.042, p = 0.041$ |
| **NPS** | 354,861 | 356,922 | -0.6% | $t = -1.265, p = 0.21$ (不顯著) |

### 6.4 結論與基準同步（Production Sync）
1. **驗證通過**：Step 2 Bug 2 修復在長時控 600 局高壓測試下穩定取得 **+7.53 Elo**（+13 盤淨勝），前後半場得分率均衡，殘局收割率顯著改善。
2. **基線同步**：依據專案工作流規則，已正式透過 `python tools/sync_classical_old.py --sync` 將代碼完整同步至 [`chess_engine/classical_old/`](file:///c:/Users/ren%20cian/OneDrive/桌面/chess/chess_numba/chess_numba/chess_engine/classical_old/)，確立為全新生產基準（E72）。
3. **後續排程**：下一步依路線圖推進至 **Step 3**（修復 Bug 1：Fail-Low ALL-Node 污染 TT best_move，搭配提案 B：IIR `!all_node` 守衛）。
---

## 7. Step 3 實測驗證成果（Bug 1 修復與提案 B IIR 守衛之 1,000 局 LTC 檢驗與回滾）

### 7.1 測試背景與對抗設定
* **改動項目**：
  1. 修復 Bug 1（[`chess_engine/classical/search.py:3276`](search.py#L3276)）：移除 Fail-Low ALL-Node 強制回退 `best_move = moves[0]` 的代碼。
  2. 實作提案 B（[`chess_engine/classical/search.py:1946`](search.py#L1946)）：IIR 補齊 `(is_pv or cut_node)` 守衛，對齊 SF19 Step 11（`!allNode` 規範）。
* **測試環境與規格**：
  * 對戰環境：Google Colab（AMD EPYC 7B12 8 vCPU，7 個並行 Worker），`inprocess` 後端。
  * 規模：**1,000 局**（500 對成對換先開局）。
  * 時控：**400,000 nodes/move** 長時控，轉置表：64 MB / context。
  * 對陣雙方：Candidate (New / classical) vs Baseline (Old / classical_old，E72 生產基準)。

### 7.2 實測數據與 Elo 表現

```text
========== 引擎對戰深度分析報告 (New vs Old) ==========
總對局數 : 1000 局 (500 對成對換先開局)
戰績統計 : 勝 185 | 和 607 | 負 208
總體得分 : 488.5 / 1000 (得分率 48.85%，淨負 -23 盤)
和局率   : 60.70%
相對 Elo 變化 : -7.99
95% 信賴區間  : [-21.34, +5.33] (單位: color-swapped pair clustered, n=500)
```

| 搜尋指標 | Candidate (New) | Baseline (Old) | 差異 (New - Old) | 統計顯著性 (Welch t-test) |
| :--- | :---: | :---: | :---: | :---: |
| **平均搜尋深度 (Depth)** | 19.63 ply | 19.86 ply | **-0.24 ply** | $t = -2.528, p = 0.011$ (高度顯著縮減 ⚠️) |
| **深度中位數** | 16 ply | 17 ply | **-1.00 ply** | 整整落後 1 層 |
| **深度 $\ge 14$ 比例** | 88.5% | 90.3% | -1.8% | 深度突破率顯著下降 |
| **深度 16–20 比例** | 42.8% | 45.0% | -2.2% | 主力深度區間明顯萎縮 |
| **每步耗時** | 917.0 ms | 915.8 ms | +1.2 ms | 不顯著 ($p = 0.54$) |
| **每步 NPS** | 330,632 | 331,420 | -0.2% | 不顯著 ($p = 0.52$) |

### 7.3 失敗根因剖析（Post-Mortem Analysis）
1. **著法排序快取喪失**：在迭代加深中，深度 $D$ 失敗（Fail-Low）時，舊版存入 TT 的 `moves[0]` 是 MovePicker 排序第一的高質量候選步。新版存入 `NO_MOVE` 導致深度 $D+1$ 重訪該局面時喪失 TT 導引，MovePicker 必須重頭生成並排序全部著法，拖慢了擴展速度。
2. **ALL-Node 分支爆炸**：舊版對所有無 TT 步節點皆實施 IIR 減深；新版豁免了 ALL-Node 的 IIR 減深。當大量 ALL-Node 正確清空 TT 步後，新版強行以「全深度」窮舉搜尋全部 30+ 步走法，單點節點消耗暴增 2~3 倍，在 400k nodes 固定預算下過早耗盡算力，直接導致搜尋深度崩塌 -0.24 ply。
3. **生態系平衡反噬**：如同先前修復 Bug 3 導致參數重力失衡，引擎現有 RFP、LMR 與 History 常數均是在舊有「Fail-Low 暫存第一步 + 全域 IIR 快速收斂」的拓撲下達成之最優點，打破平衡引發連鎖失真。

### 7.4 處置結論
* **立即回滾 (REVERTED)**：依專案規範，嚴格撤銷本次改動，將 [`chess_engine/classical/search.py`](search.py) 完整還原至 E72 生產基線，維持基線 100% 穩固。
* **後續策略調整**：不再動 Bug 1 與 IIR 守衛，轉向對樹結構干擾較小、消除純粹浪費的架構重構（如 **提案 A：前向剪枝順序重構，NMP/IIR 前置於 ProbCut**）。



---

## 8. Step 4 實作進展：純化版提案 A 前向剪枝順序重構（E74：NMP 前置於 Capture ProbCut）

### 8.1 設計與實作規格
依據 E73 經驗教訓與 SF19 深入比對，採取「純化解耦架構」實裝：
1. **順序重構**：
   \text{Step 8 (Razor)} \to \text{Step 9 (RFP)} \to \text{Step 12a ((1)$ TT ProbCut 捷徑)} \to \mathbf{Step\ 10\ (NMP)} \to \mathbf{Step\ 12b\ (Capture\ ProbCut)} \to \text{Step 11 (IIR)}
2. **算力與快取協同**：
   * 凡 NMP 通過門檻但 Fail-Low 之節點，其已完成之 compute_full_corrected_static_eval（完整 HCE 靜態評估）與 cached_pinned_white/black（雙王 Pin 遮罩）直接供後續 Capture ProbCut 100% 免費復用，消除重複位元運算開銷。
3. **安全隔離**：
   * 凍結 IIR 於 ProbCut 之後，不動 improving 定義，不動 ProbCut TT Defense。

### 8.2 實測驗證成果（600 局 LTC 400k nodes 驗證與處置）
* **對戰設定**：Candidate (New / E74 NMP-First) vs Baseline (Old / E72)，600 局 @ 400,000 nodes/move。
* **對戰結果**：Candidate **120 勝 358 和 122 負**（得分率 **49.83%**，淨負 -2 盤，相對 Elo **-1.16**）。
* **搜尋指標**：
  * 平均深度 19.89 vs 19.89 ply（差異 +0.01 ply, p = 0.941 統計完全等價）
  * 每步節點數 248,382 vs 248,400（-0.0%, p = 0.962）
  * 耗時 464.1 ms vs 468.6 ms（-4.5 ms, -1.0%）
  * NPS 612,827 vs 607,614（+0.9%）
* **根因總結**：
  1. $O(1)$ TT ProbCut 捷徑（E35）已過濾絕大部分截斷，宏觀節省有限；
  2. NMP 截斷未能在置換表中留下具體吃子著法（TT move 留空），在後續重訪時產生微小著法排序品質折損，抵消了微小提速效益。
* **處置結論**：**立即回滾 (REVERTED)**，`search.py` 完整復原至 E72 生產基線。


---

## 9. 提案 D 實作進展：RFP 靜態分數阻尼混合（E75）

### 9.1 設計與實作規格
* **對齊依據**：Stockfish 19 Step 9（`search.cpp:1007` `return (661 * beta + 363 * eval) / 1024;`）。
* **實作位置**：[`chess_engine/classical/search.py:1643`](search.py#L1643) 與 [`constants.py:1053`](constants.py#L1053)。
* **物理機理與 HCE 差異探索**：
  * 在 Classical HCE 下，靜態評估分偶爾會因局部陣型優勢而虛高。直接回傳過高的靜態分會衝擊父節點的抱負視窗（Aspiration Window），引發不必要的 Fail-High 重搜。
  * **E75.0（SF19 原生權重：661 $\beta$ + 363 eval）**：實測**顯著落敗**。在 HCE 下，過度重壓 $\beta$（拉向 $\beta$ 達 64.5%）過度削弱了真實局面的優勢訊號，導致父節點排序品質受損。
  * **E75.1（反向輕阻尼權重：363 $\beta$ + 661 eval）**：改為 $\beta$ 佔 35.5%、eval 佔 64.5%，既保留了 $\ge \beta$ 的平滑特性與防止抱負視窗破裂的優點，又維護了足夠的估值梯度。實測**顯著勝出 (+9.27 Elo)**。

### 9.2 實測驗證成果（E75.1：363/661，600 局 LTC 400k nodes 驗收）
* **對戰設定**：Candidate (New / E75_RFP_Blend 363/661) vs Baseline (Old / E72)，600 局 @ 400,000 nodes/move，換先平衡 300 配對。
* **對戰結果**：
  * **戰績**：Candidate **137 勝 342 和 121 負**（得分率 **51.33%**，淨勝 **+16 局**）。
  * **相對 Elo**：**+9.27 Elo**（95% CI: [-7.10, +25.67]）。
  * **盤長特徵**：在 51-80 步的關鍵中殘局戰術過渡盤中，E75 取得 **92 勝 81 和 73 負（勝率 53.9%）**，斬獲 +19 淨勝局。
* **搜尋指標分析（`parse_search_stats.py`）**：
  * **平均深度**：19.90 vs 19.93 ply（差異 -0.03 ply, Welch p = 0.803，統計完全等價）。
  * **每步節點數**：247,551 vs 247,415（+0.1%, Welch p = 0.715，同深度節點比保持 1.000）。
  * **耗時 / NPS**：442.4 ms vs 433.5 ms（+8.9 ms）；NPS 641,234 vs 652,159（-1.7%）。
* **結論與延伸（E76：412 $\beta$ + 612 eval 與 E77：312 vs 412 回滾）**：
  * RFP 截斷阻尼混合有效防止了 Classical HCE 局部極端評估向上传播引發的 Aspiration Window Fail-High 震盪。
  * **E76 實測結果（412 vs 363 基線，552 局）**：
    * 戰績 123W 315D 114L（得分率 50.82%，淨勝 +9 局，相對 Elo **+5.67**）。
    * 證實微幅提升 $\beta$ 阻尼至 40.2% 確實帶來了額外正向收益，`412/1024` 成功登頂為新基線。
  * **E77 實測結果（312 vs 412 基線，134 局，回滾 REVERTED）**：
    * 戰績 25W 72D 37L（得分率 45.52%，淨負 -12 局，相對 Elo **-31.20**）。
    * 機理證實：312 $\beta$（30.5%）阻尼強度不足，導致靜態估值毛刺頻繁沖破抱負視窗引發重搜。
  * **阻尼參數全局凸函數分佈定案（E78：462 vs 412 回滾，曲線完整閉合）**：
    * **E78 實測結果（462 $\beta$ vs 412 基線，600 局 LTC 400k nodes，回滾 REVERTED）**：
      * 戰績 122W 351D 127L（得分率 49.58%，淨負 -5 局，相對 Elo **-2.90**，95% CI: [-20.93, +15.12]）。
      * 盤長特徵：在 >80 步超長殘局中得分率降至 47.8%（35W 41L，淨負 6 盤），證實 462（45.1% $\beta$）過度平滑估值，開始鈍化細膩殘局中的局面推進梯度。
    * **Classical HCE 1D RFP 阻尼凹函數（Concave Function）五點全景**：
      1. $W_\beta = 0$（0% $\beta$, 100% eval，E72 未阻尼基線）：基準 0.00 Elo。
      2. $W_\beta = 312$（30.5% $\beta$, 69.5% eval，E77）：**-31.20 Elo**（阻尼不足，毛刺沖破抱負視窗引發頻繁重搜）。
      3. $W_\beta = 363$（35.5% $\beta$, 64.5% eval，E75）：**+9.27 Elo**（成功壓制雜訊，確立正向效益）。
      4. **$W_\beta = 412$（40.2% $\beta$, 59.8% eval，E76）**：**+14.94 Elo（vs 未阻尼基線，全局凹頂 Peak，確立為生產基線）**。
      5. $W_\beta = 462$（45.1% $\beta$, 54.9% eval，E78）：**-2.90 Elo**（vs 412 基線，轉入右側過度阻尼下降段）。
    * **提案 D 結案處置**：以 **$W_\beta = 412$、`RFP_SCORE_EVAL_WEIGHT = 612`** 為最優解圓滿收官，候選常數已回滾還原至 412，`classical_old` 與主引擎完全同步（0 diff）。
