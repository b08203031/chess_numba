# Static Exchange Evaluation (SEE) 分析報告

## 1. 精神與意義 (Spirit and Meaning)

靜態交換評估 (SEE) 的核心精神是在不進行完整搜尋的情況下，評估一個特定方格上的「交換序列」是否有利。它主要用於：
- **移動排序 (Move Ordering)**：區分「好的吃子」和「壞的吃子」，確保好的移動優先被搜尋。
- **靜態搜尋 (Quiescence Search)**：剪掉那些明顯不利的吃子動作。
- **搜尋剪枝 (Pruning)**：在某些深度較淺的節點，剪掉 SEE 分數過低的移動。

SEE 的基本假設是：雙方都會以「最小價值」的棋子依次吃掉目標方格上的棋子（LVA - Least Valuable Attacker）。

---

## 2. 演算法方法 (Methodology)

目前的實現採用了類似 Stockfish 的 **Swap 演算法**：

1. **初始值**：計算初始吃子的價值（被吃掉的棋子價值 + 升變增益）。
2. **攻擊者集合**：找出所有攻擊目標方格的棋子。
3. **疊代模擬**：
   - 雙方輪流從攻擊者集合中選出價值最低的棋子進行「吃子」。
   - 每一步計算該方的「累積增益」。
   - **X-Ray 處理**：當一個滑動棋子（車、象、后）或兵被移走時，檢查其後方是否露出了新的 X-Ray 攻擊者。
   - **動態牽制防護**：被牽制棋子若沿牽制線（Pin Ray）反吃或直接反吃牽制源，視為合法走法；若牽制線上曾有棋子於交換過程中離開（初始走法為牽制者離線，或中途牽制者先吃進 `to_sq`），則動態驗證牽制是否已解除。
   - **國王安全**：國王不能移動到受攻擊的方格進行吃子（這在模擬中代表吃子後會被立即反殺）。
4. **Minimax 傳導**：從序列最後端向前傳導分數，模擬雙方在發現後續不利時會選擇停止吃子的情況（`_see_ge_jit` 則採用代數等價的快速布林剪枝，並在底線格考慮後續兵重吃升變的潛在增益）。

---

## 3. 細節檢查與邏輯驗證 (Details and Verification)

### 3.1 棋子價值與評估材質尺度對齊 (`SEE_PIECE_VALUES` ↔ `MG_MATERIAL_VALUES` & SPSA)
- **現狀**：
  - [`SEE_PIECE_VALUES`](see.py#L26) 已直接引用 [`MG_MATERIAL_VALUES`](constants.py#L43) 的前 5 項（`PAWN=100, KNIGHT=347, BISHOP=364, ROOK=572, QUEEN=1170`），並保留 `KING=20000` 以確保國王在 Swap 演算法中不可被犧牲（因 `MG_MATERIAL_VALUES[KING] = 0` 若直接用於 `_see_ge_jit` 會導致國王吃子立即觸發 `swap <= 0` 提前返回 `True`）。
  - 此舉消除了過去獨立 `[100, 320, 330, 500, 900]` 與評估尺度之間的不一致，使 **ProbCut SEE 過濾器**（`see_threshold_pc = probcut_beta - static_score`）、**Capture LMR 吃子折減** 與 **HCE 靜態評估** 完全共用同一套子力尺度。
- **SEE 相關參數等比例縮放（$\times 1.134$）與 Runtime SPSA 整合**：
  - 依雙方開局總子力比值 $4536 / 4000 = 1.134$（非兵子力比值 $3736 / 3200 = 1.1675$），將所有與 SEE 相關的門檻與歷史斜率常數等比例縮放，完美保持原有的子力邊界不變量（例如 `QS_SEE_THRESHOLD = -91 > -100` 仍剪除淨損一整兵的吃子；`PRUNING_QUIET_SEE_MARGIN = -36` 在 `d=3` 時 $-36 \times 9 = -324 > -347$ 仍精確剪除 `d <= 3` 白送騎士的安靜步）：
    1. `PRUNING_CAPTURE_SEE_MARGIN`: `-102` $\to$ **`-116`**（Runtime Slot `TUNE_SEE_CAP_MARGIN = 4`）
    2. `PRUNING_QUIET_SEE_MARGIN`: `-32` $\to$ **`-36`**（Runtime Slot `TUNE_SEE_QUIET_MARGIN = 5`）
    3. `QS_SEE_THRESHOLD`: `-80` $\to$ **`-91`**（Runtime Slot `TUNE_QS_SEE = 14`，同時綁定 `quiescence_search` 與 `score_captures_with_tt_lazy`）
    4. `CHECK_SEE_THRESHOLD`: `-75` $\to$ **`-85`**（新增 Runtime Slot `TUNE_CHECK_SEE_THR = 42`，綁定 `score_quiets` 與 `score_moves`）
    5. `CAP_SEE_CAPTHIST_NUM`: `34` $\to$ **`39`**（新增 Runtime Slot `TUNE_CAP_SEE_HIST_NUM = 43`，綁定主搜尋吃子 SEE 剪枝之 Capture History 調整項）
  - 上述 5 個參數已全數納入 `tune_search/search_param_registry.py` 的 **Runtime SPSA**（`TUNE_SIZE = 44`），並提供專屬 `--profile see`，可在單次 JIT 編譯下直接透過 `--backend inprocess` 進行實戰對局 SPSA 微調。

### 3.2 攻擊者尋找 (Attacker Identification)
- **現狀**：Magic Bitboards + 預計算攻擊表；在 `get_all_attackers_for_see` 與 X-Ray 路徑上，先以 `BISHOP_RAYS` / `ROOK_RAYS` 預過濾，再必要時做 Magic 查表。

### 3.3 X-Ray 處理 (X-Ray Handling)
- **現狀**：滑動棋子被移開後，沿射線補上後方攻擊者；邏輯已以 0–5 棋子類型索引處理，不依賴側翼偏移的冗餘判斷。

### 3.4 國王捕捉模擬 (King Capturing)
- **現狀**：正確處理了國王不能捕獲受保護棋子的情況（避免送王）。

### 3.5 過路兵 (En Passant)
- **現狀**：直接透過 `(initial_occ & BB_SQUARES[to_sq]) == 0` 配合 `attacker_type == PAWN and abs((from_sq % 8) - (to_sq % 8)) == 1` 自包含判定過路兵（同時相容 `victim_type = -1`、`-2` 或 `PAWN`），正確計算初始捕獲價值並從 `occupied` 位元棋盤移除 `ep_sq` 上的被吃兵，確保 X-Ray 掃描結果精確。

### 3.6 兵升變與欠升變 (Pawn Promotion & Underpromotion)
- **現狀**：
  - **初始走法升變與欠升變**：`_see_jit` 與 `_see_ge_jit` 支援 `promo_type` 參數（預設 `QUEEN = 4`，由呼叫端透過 `get_promotion_piece(move) + 1` 傳入 `KNIGHT..QUEEN`），初始增益 `value` 與駐留子價值 `attacker_val` 均使用實際升變子力之 `SEE_PIECE_VALUES[promo_type]`，避免欠升變（升馬/象/車）在 QSearch 與主搜尋 SEE 剪枝中被高估為皇后（1170 cp）。
  - **後續迴圈底線兵重吃升變**：在 `_see_jit` 與 `_see_ge_jit` 的交換迴圈內，當 `to_sq` 位於底線（`to_sq >= 56 or to_sq < 8`）且輪到 `PAWN` 重吃時，計入升變差額 `promo_gain = SEE_PIECE_VALUES[QUEEN] - SEE_PIECE_VALUES[PAWN]`（+1070 cp）並將駐留子價值更新為 `SEE_PIECE_VALUES[QUEEN]`；同時在 `_see_ge_jit` 的提前截斷條件中考慮底線潛在升變增益（`if not is_promo_rank or (swap + promo_gain <= 0)` 與 `if not is_promo_rank or (swap + promo_gain < res)`），防止因上一輪未計入下一輪兵升變收益而提早誤判跳出。

### 3.7 動態牽制解開與牽制線 (Dynamic Unpinning & Pin Ray)
- **現狀**：透過 `filter_pinned_attackers` 處理被牽制子：
  - 若目標格 `to_sq` 位於 `LINE_BB[king_sq, to_sq]` 上，被牽制子可沿牽制線合法反吃。
  - 若被牽制子不在 `LINE_BB[king_sq, to_sq]` 上，先以 $O(1)$ 檢查 `(removed_occ & LINE_BB[king_sq, p_sq]) == 0`（其中 `removed_occ = initial_occ ^ occupied`）；若牽制線上曾有棋子在交換過程中離開（例如初始走法本身就是敵方牽制者離開牽制線，或交換中途敵方牽制者先一步吃進 `to_sq`），則動態驗證 `occupied` 上是否仍有敵方滑動子沿 `pin_line` 牽制 `p_sq`，解除牽制後即恢復該棋子的重吃能力。

### 3.8 哨兵值 -2 與呼叫端互動 (Sentinel -2, Castling & Pre-make Legality)
- **現狀**：
  - `_see_jit` 與 `_see_ge_jit` 相容 `victim_type = -2` 作為已知空格哨兵值，同時以 `(initial_occ & BB_SQUARES[to_sq]) == 0` 優先判定空格與過路兵。
  - **王車易位（Castling）與將軍/SEE**：在 `search.py` 中，王車易位的將軍預判改以易位後城堡落點 `rook_to_sq = (from_sq + to_sq) >> 1` 及更新後佔用位元棋盤檢查是否對敵王將軍，且易位走法不送入 `_see_ge_jit`（比照 Stockfish 視為 `SEE = 0`）。
  - **淺層剪枝前之 $O(1)$ 合法性預檢**：在 `search.py` 的 Step 14 淺層剪枝區塊入口先執行 $O(1)$ 預先合法性檢查（涵蓋牽制子、國王走法、過路兵揭露將軍），確保非法偽合法走法不會因 SEE/LMP/History/Futility 剪枝而遞增 `pruned_moves`，保護無合法走法時的逼和（Stalemate）終局判定。

---

## 4. 相關檔案說明 (Related Files)

- [see.py](see.py) 與 [nnue/see.py](../nnue/see.py): SEE 的核心實現（兩版本完全獨立）。
- [constants.py](constants.py) 與 [nnue/constants.py](../nnue/constants.py): 定義了棋子價值和 SEE 閾值。
- [search.py](search.py) 與 [nnue/search.py](../nnue/search.py): 在移動排序、靜態搜尋和淺深度剪枝中調用 `see_ge`。
- [search_heuristics.py](search_heuristics.py): 在吃子與安靜步排序中調用 `_see_ge_jit`。
- [move_generator.py](move_generator.py) 與 [nnue/move_generator.py](../nnue/move_generator.py): 提供基礎的攻擊生成邏輯。
- [bitboard_utils.py](bitboard_utils.py) 與 [nnue/bitboard_utils.py](../nnue/bitboard_utils.py): 提供硬體加速的位元掃描和預計算的射線表 (`LINE_BB`, `ROOK_RAYS`, `BISHOP_RAYS`)。

---

## 5. 優化與修正紀錄 (Optimization and Fix Records)

本分析過程中，已針對以下項目進行優化與修正：
1. **修正國王價值**：設置專門的 `SEE_PIECE_VALUES` (`King = 20000`)，確保國王在交換評估中不會被視為普通棋子犧牲。
2. **性能優化**：在 `get_all_attackers_for_see` 與 X-Ray 邏輯中加入射線預過濾 (`BISHOP_RAYS`, `ROOK_RAYS`)，顯著減少昂貴的 Magic Bitboard 查表次數。
3. **優化 X-Ray 邏輯**：移除冗餘的側翼索引判斷，簡化代碼邏輯。
4. **修正過路兵細節**：直接由 `(initial_occ & BB_SQUARES[to_sq]) == 0` 判定空格與過路兵，在模擬過程中正確從 `occupied` 移除被吃掉的過路兵，不再脆弱依賴呼叫端傳入特定 `victim_type`。
5. **處理兵升變與欠升變**：支援 `promo_type` 傳入實際升變子力（馬/象/車/后），並在 `_see_jit` 與 `_see_ge_jit` 交換迴圈內完整處理底線後續兵重吃升變與提前截斷防護。
6. **支援牽制線反吃與動態解牽制**：利用 `filter_pinned_attackers` 允許被牽制棋子沿牽制線反吃，並在牽制者於交換初始或中途離開牽制線時動態解除牽制。
7. **修正哨兵值 -2 取模錯誤**：在 `_see_jit` 加入 `victim_type == -2` 相容處理，並在 `search.py` Fast Path 加上 `vic_type >= 0` 條件，杜絕 `-2 % 6 == 4` 誤判。
8. **修復 Move Ordering 過路兵與安靜將軍傳參**：在 `search_heuristics.py` 與 `search.py` 中傳遞 `get_promotion_piece(move) + 1`，並修正王車易位誤送入 SEE 以及淺層剪枝前未先驗合法性導致 `pruned_moves` 汙染逼和判定的問題。

這些改進提高了搜尋效率，並使引擎在處理複雜吃子序列時更加精準。

---

## 6. 實戰驗收與對戰統計 (E84 — 600 局 LTC 400k nodes 正式採納)

上述 SEE 核心漏洞修補與呼叫端契約加固（動態解牽制、升變/欠升變完整計分、過路兵自包含判定、王車易位將軍/SEE 分流、Step 14 前置 $O(1)$ 合法性預檢）已完成 600 局 LTC 實戰對抗驗收，並正式同步至 `chess_engine/classical_old/` 生產基線：

- **對戰設定**：600 局（300 對成對換先開局），時控 **400,000 nodes/move**，TT 64 MB。
- **總體戰績**：**123 勝 372 和 105 負**（總分 309.0 / 600，勝率 **51.50%**，淨勝 **+18 局 / +9.0 分**）。
- **相對 Elo**：**+10.43 Elo**（95% CI: `[-5.69, +26.59]`，成對換先配對聚類 $n=300$）。
- **執色與分段表現**：
  - 持白勝率：**52.83%**（68W 181D 51L，淨勝 +17 局）
  - 持黑勝率：**50.17%**（55W 191D 54L，淨勝 +1 局，持黑突破 50%）
  - 成對換先九宮格：WD 46 vs LD 34（淨勝 +12 對）、DW 39 vs DL 31（淨勝 +8 對）。
  - 盤長分佈：短局 ($\le 25$ 步) `50.7%`（1W 75D 0L）、中局 (26–50 步) `52.0%`（11W 135D 5L，淨勝 +6 局）、長局 (51–80 步) **`52.1%`**（83W 82D 73L，淨勝 +10 局）、超長局 ($>80$ 步) `50.4%`（28W 80D 27L，淨勝 +1 局）。
- **搜尋效能與深度指標（`parse_search_stats.py`，樣本數 36,777 步 vs 36,735 步）**：
  - **平均深度**：19.64 vs 19.76 ply（差異 `-0.12 ply`, Welch $t = -1.045, p = 0.296$ 統計不顯著；$\text{depth} \ge 14$ 比例 `89.7%` vs `90.1%`）。
  - **每步節點數與同深度節點比**：245,396 vs 248,189（**`-1.1%`**, Welch $t = -7.505, p = 6.20 \times 10^{-14}$ 極度顯著）；在 `d = 11 .. 24` 全數 14 個主力深度層上，New/Old 同深度節點比一致落在 **`0.981 ~ 0.997`**，證實精準的 SEE 交換評估與前置合法性過濾顯著提升了剪枝與排序效率。
  - **耗時與 NPS**：每步耗時 790.7 ms vs 782.2 ms（`+1.1%`），NPS 365,350 vs 372,166（`-1.8%`）。
