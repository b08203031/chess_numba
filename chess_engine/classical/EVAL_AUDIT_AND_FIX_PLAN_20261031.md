# 經典評估函數 (`evaluation.py`) 深度審計與修復計畫書 (2026-10-31)

> **文檔屬性**：Classical HCE 程式碼全面審計報告與分階段工程實施計畫書（可供團隊傳閱、Code Review 與接續 AI Agent 按步執行）  
> **審計對象**：[`chess_engine/classical/evaluation.py`](evaluation.py)（全檔 1,814 行）及其強耦合模組 [`pawns.py`](pawns.py)、[`endgame.py`](endgame.py)、[`constants.py`](constants.py)、[`tune_eval_match/eval_weights.py`](../../tune_eval_match/eval_weights.py)、[`tuner/codegen/generate_tunable_eval.py`](../../tuner/codegen/generate_tunable_eval.py)  
> **基準 Commit**：`ca0198d`  
> **對照基準**：官方 Stockfish 11（[`stockfish_11/src/evaluate.cpp`](../../stockfish_11/src/evaluate.cpp)、[`pawns.cpp`](../../stockfish_11/src/pawns.cpp)、[`position.cpp`](../../stockfish_11/src/position.cpp)）  
> **關聯主文件**：[`HCE_SF11_GAP_AUDIT.md`](HCE_SF11_GAP_AUDIT.md)、[`SEARCH_EVAL_INTERFACE.md`](SEARCH_EVAL_INTERFACE.md)、[`JIT_COMPILE_CACHE_REPORT.md`](JIT_COMPILE_CACHE_REPORT.md)

---

## 執行摘要 (Executive Summary)

本報告對經典手工評估核心 [`evaluation.py`](evaluation.py) 進行逐行靜態審計與數學反例推導，共識別出 **8 項邏輯／幾何／對齊 BUG**、**3 項計算順序錯誤**（集中於動態通路兵評估），以及 **7 組 Numba `@njit` 熱路徑冗餘計算與記憶體配置瓶頸**。

其中最關鍵的三大高優先級缺陷為：
1. **被牽制皇后（Pinned Queen）攻擊遮罩幾何錯誤（`BUG-1`）**：誤用包含雙軸的 `ROOK_RAYS` / `BISHOP_RAYS` 過濾雙向滑動的皇后，導致非牽制軸上的非法攻擊格洩漏至全盤攻擊表、機動性、威脅與王安全。
2. **純王兵殘局方形法則漏判自阻擋與測試假陽性（`BUG-2`）**：同線疊通路兵會重複領取兩次 `UNSTOPPABLE_PAWN_BONUS`（合計 `+1600 cp`）並無視前兵穿牆雙步；既有單元測試因單兵對照組走進 `_eval_kpk`（`+10119 cp`）而產生假通過。
3. **動態通路兵折半順序顛倒（`ORD-1`）**：先扣除中央直線懲罰 `PassedFile` 使低橫排通路兵變為負分（如 `-48`），隨後對候選／受阻兵執行 `//= 2` 將負分減半為 `-24`，造成**「受阻／候選通路兵在中局反比暢通真通路兵高出 +21 ～ +24 cp」**的反常倒掛。

為避免效能重構與棋力分數變動互相干擾，本計畫書將修復工作拆解為 **4 個獨立驗收階段（Phase 1 ～ Phase 4）**，明確定義每階段的程式碼改動原型、工具鏈同步步驟與量化驗收門檻（0-Node Tree Delta / 單元測試 / SPRT 對局）。

---

## 總覽問題矩陣 (Audit Summary Matrix)

| 編號 | 類別 | 嚴重度 | 模組與行號 | 問題摘要 | 分數變動 | 排入階段 |
| :--- | :---: | :---: | :--- | :--- | :---: | :---: |
| **BUG-1** | 幾何邏輯 | 🔴 高 | [`evaluation.py:846-852, 1056-1062`](evaluation.py#L846-L852) | 被牽制皇后誤用 `ROOK_RAYS`/`BISHOP_RAYS`，保留非牽制方向的非法攻擊格 | 是（修錯） | Phase 2 |
| **BUG-2** | 殘局邏輯 | 🔴 高 | [`evaluation.py:1319-1351`](evaluation.py#L1319-L1351) | 王兵殘局方形法則漏判疊兵／己方王自阻擋（疊兵領 `+1600 cp`）；單元測試假陽性 | 是（修錯） | Phase 2 |
| **BUG-3** | 對稱性 | 🟠 中 | [`evaluation.py:562, 1690-1798`](evaluation.py#L1795-L1798) | 有號分數使用向下取整 `//` 而非向零截斷，造成全盤黑白鏡像系統性偏袒黑方 1～2 cp | 是（±1~2 cp） | Phase 4 |
| **BUG-4** | 權重接線 | 🟠 中 | [`evaluation.py:501, 1316, 1454, 1608`](evaluation.py#L501) | `EG_SAFETY_SCALE=0.96` 實質死碼；特化殘局、王兵殘局、空間評估與 `eval_weights` 脫鉤 | 視模式 | Phase 2 |
| **BUG-5** | SF11 對齊 | 🟡 低–中 | [`evaluation.py:1183-1193`](evaluation.py#L1183-L1193) | `ThreatByKing` 誤乘 `popcount`（王一回合僅能吃一子，SF11 為布林觸發一次） | 是 | Phase 4 |
| **BUG-6** | SF11 對齊 | 🟡 低–中 | [`evaluation.py:710, 759, 918, 968`](evaluation.py#L710) | `MinorBehindPawn` 僅檢查己方兵，漏計敵兵正後方的掩體／封鎖哨站格 | 是 | Phase 4 |
| **BUG-7** | SF11 對齊 | 🟡 低–中 | [`evaluation.py:870-890, 1080-1100`](evaluation.py#L870-L890) | `WeakQueen` 未排除同線重疊的敵方狙擊子（Snipers），雙車／雙象疊線誤觸懲罰 | 是 | Phase 4 |
| **BUG-8** | SF11 對齊 | 🟡 低–中 | [`evaluation.py:719, 779, 837, 867...`](evaluation.py#L716-L719) | `kingAttacksCount` 置於 `if att & enemy_king_zone:` 外，誤計雙兵保護剔除格並浪費 `POPCNT` | 微幅 | Phase 4 |
| **ORD-1** | 計算順序 | 🔴 高 | [`evaluation.py:184-233, 266-315`](evaluation.py#L184-L233) | `evaluate_passed_pawns` 先扣 `PassedFile` 再 `//= 2`，負分減半使受阻兵高出真通路兵 +24 cp | 是（關鍵） | Phase 3 |
| **ORD-2** | 計算順序 | 🟠 中 | [`evaluation.py:237-253, 319-335`](evaluation.py#L237-L253) | `evaluate_passed_pawns` 國王距離獎懲在 `//= 2` 之後才累加，候選／受阻兵未折半 | 是 | Phase 3 |
| **ORD-3** | 門檻順序 | 🟠 中 | [`evaluation.py:192, 238, 274, 320`](evaluation.py#L192) | 通路兵自由推進用 `rank > 2`，國王距離卻用 `rank > 3`，漏計第 4 橫排王距離（`w = +2`） | 是 | Phase 3 |
| **PERF-1** | 記憶體/快取 | 🔴 高 | [`evaluation.py:1358-1375`](evaluation.py#L1358-L1375) | 王兵殘局熱路徑呼叫 `np.zeros(3)` 觸發堆分配，且繞過 Pawn Hash 白算 6 次王翼兵盾 | **0（純加速）** | Phase 1 |
| **PERF-2** | 重複計算 | 🟠 中 | [`evaluation.py:1608, 1638, 1793`](evaluation.py#L1638-L1650) | 單次完整評估跨 `endgame.py` 與 `evaluation.py` 重複 3 次全盤 `count_bits`（30 次 `POPCNT`） | **0（純加速）** | Phase 1 |
| **PERF-3** | 迴圈不變量 | 🟠 中 | [`evaluation.py:768-795, 977-1005`](evaluation.py#L768-L795) | 主教迴圈內重算盤面不變量，且對全 64 格無條件跑第 2 次 `get_bishop_attacks` | **0（純加速）** | Phase 1 |
| **PERF-4** | 重複計算 | 🟠 中 | [`evaluation.py:1449-1510`](evaluation.py#L1449-L1510) | `evaluate_space` 重算 `npm`、每次執行期查 10 次陣列組裝固定常數遮罩 | **0（純加速）** | Phase 1 |
| **PERF-5** | 空轉迴圈 | 🟡 低–中 | [`evaluation.py:1150-1180`](evaluation.py#L1150-L1180) | `ThreatByMinor`/`ThreatByRook` 無威脅時仍空轉 24 次迴圈，且迭代恆為 0 的 `pt=5`（王） | **0（純加速）** | Phase 1 |
| **PERF-6** | 死碼/重算 | 🟡 低 | [`evaluation.py:656-1236`](evaluation.py#L656) | 死變數 `white/black_minor_attacks`；`ThreatByPawnPush` 重算已存在的 `white_safe`/`black_safe` | **0（純加速）** | Phase 1 |
| **PERF-7** | 冗餘分支 | 🟡 低 | [`evaluation.py:213, 453, 1716, 1782`](evaluation.py#L1716) | 恆真 `if not lazy:`、`if block_sq < 64:`、重複計算 `w`、跨 `@njit` 傳遞未使用參數 | **0（純加速）** | Phase 1 |

---

## 第一部分：確認的 BUG 深度診斷與修復規格

### 🔴 BUG-1：被牽制皇后（Pinned Queen）攻擊遮罩幾何錯誤【✅ 已修復驗證】

* **位置**：[`evaluation.py:737, 791, 829, 938, 991, 1029`](evaluation.py#L829)（象、車、后 6 個迴圈）
* **現行程式碼（修復後）**：
  ```python
  if pinned_white & sq_bb:
      att &= LINE_BB[white_king_sq, sq]
  ```
* **驗證結果**：
  * 新增專屬幾何單元測試 [`tests/test_pinned_eval_geometry.py`](../../tests/test_pinned_eval_geometry.py) 4 項測試全數通過（覆蓋白/黑方正交直線牽制不漏洩斜向攻擊格、斜線牽制不漏洩正交攻擊格）。
  * 重新生成 [`tuner/tunable_eval.py`](../../tuner/tunable_eval.py) 並以 [`tests/test_tunable_eval_parity.py`](../../tests/test_tunable_eval_parity.py) 驗證 classical 與 tunable eval 100% 保持完全一致。
* **問題根因與具體反例**：
  [`ROOK_RAYS[king_sq]`](bitboard_utils.py#L178-L185) 同時包含王所在的**橫排與直線**；[`BISHOP_RAYS[king_sq]`](bitboard_utils.py#L186-L195) 同時包含王所在的**兩條正反對角線**。主教與車僅單一類型移動，與王共軸時不會跨到垂直軸；但**皇后兼具直線與斜線攻擊**，其非牽制方向的攻擊射線必然與王的另一條軸相交：
  1. **直線牽制反例**：白 `Ke1`、`Qe3`，黑 `Re8`（`Qe3` 被嚴格釘死在 e 直線 `e2-e8`）。`get_queen_attacks(e3)` 產生的斜線攻擊包含 `c1` 與 `g1`。由於 `c1`、`g1` 位於第 1 橫排（屬於 `ROOK_RAYS[e1]`），`att &= ROOK_RAYS[e1]` **錯誤保留了 `c1` 與 `g1` 兩個完全非法的攻擊格**！
  2. **斜線牽制反例**：白 `Ke1`、`Qg3`，黑 `Bh4`（`Qg3` 被嚴格釘死在 `e1-h4` 對角線）。`get_queen_attacks(g3)` 的第 3 橫排攻擊包含 `c3`。由於 `c3` 剛好落在通過 `e1` 的另一條對角線 `e1-a5`（屬於 `BISHOP_RAYS[e1]`），`att &= BISHOP_RAYS[e1]` **錯誤保留了非法格 `c3`**！
* **連鎖影響**：非法攻擊格直接混入 `white_attacks`、`white_attacks2`、`white_queen_attacks`，同時污染機動性（Mobility）、雙重攻擊判定、威脅（Threats）與敵王安全（King Danger `queen_check` / `weak` 防守判定）。
* **SF11 對照與標準修法**：
  SF11（[`evaluate.cpp:273-274`](../../stockfish_11/src/evaluate.cpp#L273-L274)）統一使用通過兩點的唯一共線遮罩 `b &= LineBB[ksq][s]`。本專案 [`bitboard_utils.py:210`](bitboard_utils.py#L210) 已預先建好 `LINE_BB`（`64×64 uint64` = 32 KiB，遠低於 Numba 1 MB 快取上限）。可將象／車／后共 **6 個迴圈**的多分支判斷統一替換為單行查表：
  ```python
  if pinned_white & sq_bb:
      att &= LINE_BB[white_king_sq, sq]
  ```
  > [!IMPORTANT]
  > 修改 `evaluation.py` 匯入 `LINE_BB` 時，**必須同步**將 `LINE_BB` 加入 [`tuner/codegen/generate_tunable_eval.py:250-253`](../../tuner/codegen/generate_tunable_eval.py#L250-L253) 的固定 `header` 匯入清單，否則重新生成 `tunable_eval.py` 時會發生 `NameError`。

---

### 🔴 BUG-2：純王兵殘局方形法則漏判「疊兵／己方王自阻擋」與測試假陽性【✅ 已修復驗證】

* **位置**：[`evaluation.py:1294-1335`](evaluation.py#L1294-L1335)（`_evaluate_king_pawn_endgame` 白兵與黑兵迴圈）
* **現行程式碼（修復後）**：
  ```python
  # 白方方形法則（黑方鏡像同理）：
  other_occ_w = all_occ ^ piece_bbs[5]  # 排除己方王，保留起點雙步潛力（避免王在第3/4橫排時產生「剝奪雙步+王讓路」雙重懲罰）
  forward_mask = WHITE_FORWARD_FILE_MASKS[sq]
  if not (WHITE_PASSED_PAWN_MASKS[sq] & black_pawns) and not (forward_mask & white_pawns):
      rank = sq >> 3
      steps_to_promote = 7 - rank
      if rank == 1 and not (other_occ_w & (BB_SQUARES[sq + 8] | BB_SQUARES[sq + 16])):
          steps_to_promote = 5  # double-push correction
      if forward_mask & piece_bbs[5]:
          steps_to_promote += 1  # own king blocks advance on promotion path
      promotion_sq = (sq & 7) + 56
      king_dist = CHEBYSHEV_DISTANCE[black_king_sq, promotion_sq]
      adjusted_pawn_steps = steps_to_promote
      if side_to_move == 1:
          adjusted_pawn_steps += 1
      if king_dist > adjusted_pawn_steps:
          score += UNSTOPPABLE_PAWN_BONUS
  ```
* **驗證結果**：
  * **消除疊兵重疊領賞**：修復前 `7k/7p/8/8/8/P7/P7/7K w` 得分為 `1699`（比單兵多 `+902 cp`，重疊領取了兩次 800 cp 獎勵）；修復後得分為 `899`（比單兵多 `+102 cp`，正確反映單兵物質與 PST 差額，`diff` 由 `902 cp` 縮減為合理的 `102 cp`）。
  * **修正測試假陽性與擴充覆蓋**：修正 [`tests/test_endgame_conformance.py`](../../tests/test_endgame_conformance.py) 假陽性（原單兵 FEN 觸發 10,119 KPK 位元庫），並新增黑方疊兵鏡像測試與王擋路測試，全部 35 項單元測試 100% 通過。
  * **Parity 100% 吻合**：重新生成 `tuner/tunable_eval.py` 後通過 7 項 Tunable Parity 測試。
* **問題根因與具體反例**：
  1. **疊通路兵雙重領賞 + 穿牆雙步**：在局面 `7k/7p/8/8/8/P7/P7/7K w - - 0 1`（白 `Pa3` + `Pa2` 疊兵）中，`a3` 與 `a2` 的 `WHITE_PASSED_PAWN_MASKS[sq] & black_pawns` 皆為 `0`。結果兩兵**同時領取 `+800 cp`（合計 `+1600 cp`）**，且 `a2`（`rank == 1`）無視正前方 `a3` 的阻擋，直接套用雙步推進 `steps_to_promote = 5`！
  2. **己方王位於升變路徑上的自阻擋**：例如白 `Pa4`、白王在 `a5`、黑王在 `e5`（輪黑走），白方必須先花 1 步移開 `Ka5` 才能推兵，真實升變需 5 步而非 4 步，黑王完全來得及進入方形區攔截，但現行公式未對路徑上的己方王加計繞行步數（亦未檢查 `rank == 1` 時 `sq + 8` 與 `sq + 16` 是否被己方王佔據）。
  3. **既有單元測試假陽性剖析**：[`tests/test_endgame_conformance.py:233-244`](../../tests/test_endgame_conformance.py#L233-L244)（`test_doubled_pawns_front_blocks_rear_unstoppable`）之所以通過，是因為其對照組 `fen_single = "7k/8/8/8/8/P7/8/7K w - - 0 1"` 只有單兵，在 `_evaluate_position_jit:1608` 直接命中特化 KPK 位元庫並回傳 **`+10119 cp`**，導致 `diff = ~1834 - 10119 = -8285 < 500` 而誤判通過！對照組必須在另一翼放一顆無關敵兵（如 `7k/7p/...`）才能真正測試 `_evaluate_king_pawn_endgame`。
* **標準修法（遵守 `HCE_SF11_GAP_AUDIT.md` §12 禁令）**：
  * **嚴禁**恢復已回滾的 P3b「全盤僅競速最快單兵可領賞」邏輯。
  * 僅補齊同直線前方的自阻擋幾何條件：
    ```python
    forward_mask = WHITE_FORWARD_FILE_MASKS[sq]
    # 1. 前方無敵兵/敵防線（passed）且同線前方無己方兵（排除疊兵後兵，與 pawns.py 定義一致）
    if (WHITE_PASSED_PAWN_MASKS[sq] & black_pawns) == 0 and (forward_mask & white_pawns) == 0:
        rank = sq >> 3
        king_on_path = (forward_mask & piece_bbs[5]) != 0
        steps_to_promote = 7 - rank
        # 2. 起點雙步排除己方王（因王移開僅花 1 步，兵仍可雙步；使用 other_occ_w 避免雙重加罰）
        if rank == 1 and ((other_occ_w & (BB_SQUARES[sq + 8] | BB_SQUARES[sq + 16])) == 0):
            steps_to_promote = 5
        # 3. 己方王擋在升變直線上，至少需多花 1 步讓路
        if king_on_path:
            steps_to_promote += 1
        promotion_sq = (sq & 7) + 56
        king_dist = CHEBYSHEV_DISTANCE[black_king_sq, promotion_sq]
        adjusted_pawn_steps = steps_to_promote + (1 if side_to_move == 1 else 0)
        if king_dist > adjusted_pawn_steps:
            score += UNSTOPPABLE_PAWN_BONUS
    ```

---

### 🟠 BUG-3：有號分數使用向下取整 `//` 造成全盤黑白鏡像不對稱【✅ 已修復驗證】

* **位置**：
  * 材質不平衡二次多項式與縮放：[`material.py:69-72`](material.py#L69-L72)
  * Tapered 階段混合與縮放：[`evaluation.py:1639-1642`](evaluation.py#L1639-L1642)、[`1673-1676`](evaluation.py#L1673-L1676)、[`1752-1755`](evaluation.py#L1752-L1755)
  * 動態 Lazy 門檻均值：[`evaluation.py:1665`](evaluation.py#L1665) `v = _trunc_div(mg_score + eg_score, 2)`
  * 王安全殘局縮放：[`evaluation.py:556`](evaluation.py#L556) `eg_safety_score = _trunc_div((white_eg_score - black_eg_score) * eg_safe_scale, np.int32(100))`
* **驗證結果**：
  * 新增專屬單元測試 [`tests/test_phase4_sf11_alignment.py`](../../tests/test_phase4_sf11_alignment.py) `test_bug3_color_flip_mirror_symmetry_500_fen`，在 500 個涵蓋開局、中局、殘局的真實對局開局 FEN（`data/hist_diag_openings_500.epd`）中進行黑白翻轉對稱評估，**500/500 盤面 100% Bit-Exact 零誤差（0 mismatches）**。
* **問題根因與數學反例**：
  在 C++（[`evaluate.cpp:817-820`](../../stockfish_11/src/evaluate.cpp#L817-L820)）中，有號整數除法 `/` **向零截斷（Truncate toward zero）**，滿足奇函數性質 $f(-x) = -f(x)$；但在 Python 與 Numba 中，`//` 為 **向負無窮向下取整（Floor division）**：
  * **反例 1（Tapered Blend 偏袒黑方）**：設白方微小優勢 `mg_score = +1, eg_score = 0, phase = 64`，白視角 `final_score = 64 // 128 = 0`。將棋盤完全黑白上下翻轉（Color Flip）後，`mg_score = -1, eg_score = 0, phase = 64`，卻得到 `final_score = -64 // 128 = -1`（黑方憑空多得 `+1 cp`）。若同時觸發 `sf != 64`，`eg_score * sf // 64` 先向下取整一次，`// 128` 再向下取整第二次，黑白鏡像落差可達 **`2 cp`**。
  * **反例 2（Lazy Early-Exit 分支不對稱）**：當 `mg_score + eg_score = +2201`（`npm = 0`）時，`+2201 // 2 = 1100`（未超過門檻 `1100`，繼續執行完整評估）；而完全對稱的黑優局面 `-2201 // 2 = -1101`，`abs(-1101) > 1100` 卻直接觸發 Lazy 提早退出！
  * **反例 3（材質不平衡 1 cp 偏差）**：當 `w_bonus - b_bonus = 1445` 時，`1445 // 16 = 90`；黑白翻轉後 `-1445 // 16 = -91`，造成即便未進入 Tapered 混合，單純材質不平衡即產生 1 cp 的不可逆偏差。
* **標準修法**：
  定義 `@numba.njit(cache=True, inline='always')` 向零截斷輔助函式：
  ```python
  @numba.njit(cache=True, inline="always")
  def _trunc_div(n, d):
      return n // d if n >= 0 else -((-n) // d)
  ```
  （註：在 Numba `nopython` 下，`n // d if n >= 0 else -((-n) // d)` 會編譯為無分支 `cmov` 整數指令，無浮點轉型誤差且完全符合 C++ `/` 語義。）

---

### 🟠 BUG-4：`EG_SAFETY_SCALE` 實質死碼與 `SearchContext.eval_weights` 脫鉤【✅ 已修復驗證】

* **位置與四處脫鉤明細與修復結果**：
  1. **`EG_SAFETY_SCALE` 統一**：[`constants.py:375`](constants.py#L375) 的死碼浮點值 `0.96` 已統一為整數定點值 `EG_SAFETY_SCALE = 100`；[`evaluation.py:495`](evaluation.py#L495) 與 [`tune_eval_match/eval_weights.py:208`](../../tune_eval_match/eval_weights.py#L208) 皆統一引用 `constants.EG_SAFETY_SCALE`，消除魔術數字硬編碼。
  2. **特化殘局接入 Runtime 材質權重**：[`_evaluate_position_jit:1563-1570`](evaluation.py#L1563-L1570) 呼叫 `evaluate_special_endgame_with_material` 時改為傳入 `_material_mg_eg(..., ew, use_ew)`，保證 KXK/KRKP/KQKP/KQKR 殘局在 `tune_eval_match` SPSA 調優時能動態反映材質變動。
  3. **純王兵殘局接入 Runtime 權重**：[`_evaluate_king_pawn_endgame`](evaluation.py#L1271) 新增 `ew, use_ew` 參數，將其傳遞至 `evaluate_pawn_structure` 與 `evaluate_passed_pawns`，消除原本寫死 `_DUMMY_EW, False` 的脫鉤。
  4. **空間評估接入 Runtime 材質權重**：已於 Phase 1 重構為直接接收外部依 `ew` 算妥之 `npm`。
* **驗證結果**：新增專屬單元測試 [`tests/test_runtime_eval_weights.py`](../../tests/test_runtime_eval_weights.py)，驗證常數一致性、KQKR 殘局在 Queen EG 材質加權時評估即時連動，以及王兵殘局參數傳遞，3 項測試全數通過；全量 38 項單元測試與 Tunable Parity 測試 100% 通過。

---

### 🟡 BUG-5 ～ BUG-8：Stockfish 11 細節語義對齊差異【✅ 已修復驗證】

1. **BUG-5：`ThreatByKing` 重複乘上 `popcount`**（[`evaluation.py:1134-1142`](evaluation.py#L1134-L1142)）
   * **問題**：原代碼 `mg_threats += THREAT_BY_KING[0] * count`。國王一回合只能吃一顆棋子，且多顆無根子同時受攻的懲罰已由下方的 `THREAT_HANGING` 按 `count_bits` 線性累加。SF11（[`evaluate.cpp:514-515`](../../stockfish_11/src/evaluate.cpp#L514-L515)）為 `if (weak & attackedBy[Us][KING]) score += ThreatByKing;`（布林觸發一次，不乘 `count`）。
   * **修復**：改為 `if (KING_ATTACKS[king_sq] & enemy_weak) != np.uint64(0): mg_threats += THREAT_BY_KING[0]`。
   * **驗證**：通過單元測試 `test_bug5_threat_by_king_boolean_not_popcount`，確認攻擊 2 顆弱子時僅增加 1 次 ThreatByKing 獎勵（差額僅為 Hanging piece 獎勵）。
2. **BUG-6：`MinorBehindPawn` 漏計敵兵正後方掩體格**（[`evaluation.py:709-713, 757-761`](evaluation.py#L709-L713)）
   * **問題**：原代碼僅檢查 `wp_bb & BB_SQUARES[sq + 8]`（己方兵）。SF11（[`evaluate.cpp:302`](../../stockfish_11/src/evaluate.cpp#L302) `shift<Down>(pos.pieces(PAWN))`）檢查的是**雙方所有兵** `(wp_bb | bp_bb)`——因為騎士或主教躲在**敵兵正後方**（如白馬在 `d5`、黑兵在 `d6`）同樣享有免受敵方正面車重子直擊的天然掩體（Nimzowitsch 封鎖格）。
   * **修復**：定義 `all_pawns = wp_bb | bp_bb`，於白黑騎士與主教迴圈中使用 `all_pawns`。
   * **驗證**：通過單元測試 `test_bug6_minor_behind_enemy_pawn`。
3. **BUG-7：`WeakQueen` 未排除同線重疊狙擊子（Snipers）**（[`evaluation.py:849-866, 1042-1059`](evaluation.py#L849-L866)）
   * **問題**：計算 `blockers_between = between & all_occupancy` 時未如 SF11 `slider_blockers`（[`position.cpp:458`](../../stockfish_11/src/position.cpp#L458) `occupied ^ snipers`）先剔除同線上的其他敵方狙擊子。
   * **修復**：在皇后迴圈前預先計算 `snipers = (ROOK_RAYS[sq] & enemy_rooks) | (BISHOP_RAYS[sq] & enemy_bishops)`，並以 `occ = all_occupancy ^ snipers` 排除狙擊子干擾。
   * **驗證**：通過單元測試 `test_bug7_weak_queen_multiple_snipers_file`。
4. **BUG-8：`kingAttacksCount` 置於 `if att & enemy_king_zone:` 之外**（[`evaluation.py:718, 772, 825, 850, 910, 964, 1017, 1042`](evaluation.py#L718)）
   * **問題**：SF11（[`evaluate.cpp:280-285`](../../stockfish_11/src/evaluate.cpp#L280-L285)）僅在棋子觸發 `if (b & kingRing[Them])` 時才累加 `kingAttacksCount += popcount(b & attackedBy[Them][KING])`。原放在 `if` 之外，對每顆未參與攻王的棋子多做一次無謂的 `POPCNT`，且當棋子僅擦過被剔除格時也會誤加。
   * **修復**：將白黑雙方 N, B, R, Q 共 8 個迴圈中的 `king_attacks_count` 累加語句嚴格移入 `if att & king_zone:` 區塊內。
   * **驗證**：通過單元測試 `test_bug8_king_attacks_count_in_zone`。

---

## 第二部分：計算順序錯誤深度剖析 (Order-of-Evaluation Bugs)

### 🔴 ORD-1 ～ ORD-3：`evaluate_passed_pawns` 候選／受阻兵折半順序顛倒

* **位置**：[`evaluation.py:355-615`](evaluation.py#L355-L615)（白黑雙迴圈），對照 SF11 [`evaluate.cpp:596-644`](../../stockfish_11/src/evaluate.cpp#L596-L644)
* **進度狀態**：
  * **`ORD-1`（直線懲罰後移）**：【✅ 已修復驗證 (2026-10-31)】
  * **`ORD-2`（國王距離門檻對齊與提前累加）**：【✅ 已修復驗證 (2026-10-31)】
  * **`ORD-3`（候選／受阻向零截斷對齊）**：【✅ 已修復驗證 (2026-10-31)】
* **三處順序與門檻錯位對照**：

| 步驟 | 原先 `evaluation.py` 順序 | Stockfish 11 (`evaluate.cpp:596-644`) 標準順序 | 修復狀態與結果 |
| :--- | :--- | :--- | :--- |
| **1. 基礎分** | `p_bonus = PASSED_PAWN_BONUS[r]` | `bonus = PassedRank[r]` | 一致 |
| **2. 直線懲罰 (ORD-1)** | ❌ **先執行** `p_bonus -= PassedFile * edge_dist` | ✅ **留到最後第 6 步執行，不參與折半** | **【✅ 已修復】** 移至折半後扣除，消除倒貼 |
| **3. 自由推進** | `if not is_blocked and r > 2: p_bonus += k * w` | `if (r > RANK_3 && !blocked) bonus += k * w` | 一致 |
| **4. 國王距離 (ORD-2)** | ❌ 延後到第 6 步，且門檻寫成 **`r > 3`** | ✅ **在此處累加進 `bonus`**，門檻同為 **`r > RANK_3` (`r > 2`)** | **【✅ 已修復】** 第 4 橫排計入王距離且受折半約束 |
| **5. 候選／受阻折半 (ORD-3)** | `if not next_passed or any_pawn_ahead: p_bonus //= 2` | `if (!next_passed || pawn_ahead) bonus = bonus / 2` | **【✅ 已修復】** 引入 `_trunc_div` 向零截斷對齊 C++ 語義 |
| **6. 收尾 (ORD-1 對齊)** | 累加 `proximity_bonus`（未折半） | ✅ **最後才扣** `bonus - PassedFile * edge_dist` | **【✅ 已修復】** 與 SF11 一致最後扣除 |

* **`ORD-1` 實測數值驗算對比（中局 d3 / d4 兵，`edge_dist = 3`，`file_mg = 16`, `file_eg = 4`）**：

| 局面與兵種 | 修復前實測分數 `(MG, EG)` | 修復後實測分數 `(MG, EG)` | 結論 |
| :--- | :---: | :---: | :--- |
| 白方 `d3` 真通路兵 | `(-48, -12)` | **`(-48, -12)`** | 基礎分 0，扣除邊緣懲罰 48 |
| 白方 `d3` 候選兵（黑兵 `e5` 防守 `d4`） | `(-24, -6)` ❌ | **`(-48, -12)`** ✅ | 舊版倒貼 `+24 cp` BUG 徹底消除 |
| 白方 `d4` 真通路兵 | `(-32, +21)` | **`(-32, +21)`** | 自由推進獎勵正常生效 |
| 白方 `d4` 候選兵（黑兵 `e6` 防守 `d5`） | `(-40, +4)` | **`(-40, +4)`** | **真通路兵全域嚴格領先 (+8 MG, +17 EG)** ✅ |
| 白方 `d4` 受阻兵（黑兵 `d5` 直線阻擋） | `(-40, +4)` | **`(-40, +4)`** | **真通路兵全域嚴格領先 (+8 MG, +17 EG)** ✅ |
| 黑方 `d5` 真通路兵 vs 候選兵 | — | **`(-32, +21)` vs `(-40, +4)`** | **黑白鏡像完全對稱一致** ✅ |

* **`ORD-2` 實測數值驗算對比（國王距離門檻對齊與折半約束）**：

| 測試情境 | 舊版評估（Buggy） | 新版評估（`ORD-2` 修復後） | 判定依據 |
| :--- | :---: | :---: | :--- |
| 白方 `d4` 兵（`rank = 3`）：己方王在 `e4` vs 己方王在 `a1` | `(-32, 19)` vs `(-32, 19)`<br>（Delta = 0 cp，漏算王距離）❌ | **`(-32, 25)` vs `(-32, 19)`<br>（Delta = +6 cp）** ✅ | 第 4 橫排（`rank > 2`）正確計入王接近度支援 |
| 白方 `d5` 兵（`rank = 4`）：真通路兵之己王接近度加分 | Delta EG = **+40 cp** | Delta EG = **+40 cp** | 暢通兵獲得全額國王保護支援 |
| 白方 `d5` 兵（`rank = 4`）：候選兵（黑兵 `e7`）之己王接近度加分 | Delta EG = +40 cp ❌<br>（未受候選折半約束） | **Delta EG = +20 cp ✅<br>（`40 // 2 = 20` 嚴格折半）** | 候選兵王支援正確折半，無根受阻兵不再高估 |

* **驗證依據**：
  * 專屬測試：[`tests/test_passed_pawn_order.py`](../../tests/test_passed_pawn_order.py)（6/6 項通過）。
  * 全量測試：44 項單元測試全數通過，`test_tunable_eval_parity` 7/7 項全數通過（共 51 項測試 100% 通過）。

* **標準修復結構（白方示意，黑方完全對稱）**：
  ```python
  p_bonus_mg, p_bonus_eg = _passed_rank_bonus(rank, ew, use_ew)

  if rank > 2:
      w = np.int32(dyn_mult * rank + dyn_off)

      # 1. King proximity (accumulated into p_bonus_eg BEFORE candidate halving)
      dist_friendly = min(CHEBYSHEV_DISTANCE[white_king_sq, block_sq], KING_PROXIMITY_MAX_DIST)
      dist_enemy = min(CHEBYSHEV_DISTANCE[black_king_sq, block_sq], KING_PROXIMITY_MAX_DIST)
      proximity_bonus = ((dist_enemy * prox_en) // prox_en_div - dist_friendly * prox_fr) * w
      if (block_sq >> 3) != 7:
          dist_friendly_2nd = min(CHEBYSHEV_DISTANCE[white_king_sq, block_sq + 8], KING_PROXIMITY_MAX_DIST)
          proximity_bonus -= dist_friendly_2nd * w
      p_bonus_eg += max(KING_PROX_MIN_BONUS, min(KING_PROX_MAX_BONUS, proximity_bonus))

      # 2. Free-path push dynamics
      if not is_blocked:
          # ... (compute k as before) ...
          dynamic_bonus = k * w
          p_bonus_mg += dynamic_bonus
          p_bonus_eg += dynamic_bonus

  # 3. Phalanx / Connected passed pawn bonuses
  if (white_pawns & PHALANX_MASK[sq]) != np.uint64(0):
      p_bonus_mg += PASSED_PHALANX_BONUS[rank]
      p_bonus_eg += PASSED_PHALANX_BONUS[rank]
  elif (white_pawns & WHITE_SUPPORT_MASK[sq]) != np.uint64(0):
      p_bonus_mg += PASSED_CONNECTED_BONUS[rank]
      p_bonus_eg += PASSED_CONNECTED_BONUS[rank]

  # 4. Candidate / pawn-blocked halving BEFORE subtracting PassedFile
  if rank < 7:
      next_sq = sq + 8
      next_passed = (black_pawns & WHITE_PASSED_PAWN_MASKS[next_sq]) == np.uint64(0)
      any_pawn_ahead = (all_pawns & BB_SQUARES[next_sq]) != np.uint64(0)
      if not next_passed or any_pawn_ahead:
          p_bonus_mg = _trunc_div(p_bonus_mg, cand_div)
          p_bonus_eg = _trunc_div(p_bonus_eg, cand_div)

  # 5. Subtract PassedFile LAST (matches SF11 evaluate.cpp:644)
  edge_dist = np.int32(min(file_idx, 7 - file_idx))
  p_bonus_mg -= file_mg * edge_dist
  p_bonus_eg -= file_eg * edge_dist
  ```

---

### ✅ 主幹管道順序驗證（`_evaluate_position_jit` 確認無誤）

對 [`_evaluate_position_jit:1603-1808`](evaluation.py#L1603-L1808) 的跨函數資料流審計結果為**依賴順序完全正確**：

```mermaid
flowchart LR
    A["特化殘局 / 純王兵殘局"] --> B["材質 + PSQT + Phase + Imbalance"]
    B --> C["顯式 Lazy 出口 (含 Scale Factor)"]
    C --> D["Pawn Structure (Pawn Hash)"]
    D --> E["動態 Lazy 出口 (含 Scale Factor)"]
    E --> F["Attacks + Mobility + Outpost + Threats"]
    F --> G["Passed Pawns + King Safety + Coord + Space"]
    G --> H["Initiative (匯總全部 MG/EG 後)"]
    H --> I["Endgame Scale Factor + Tapered Blend + Tempo"]
```

* 所有攻擊位元板、`pinned_white/black`、`mg_mobility_pure` 均在 `evaluate_attacks_mobility_threats` 完整產出後才交由 `evaluate_passed_pawns`、`evaluate_king_safety` 與 `evaluate_space` 使用。
* `_compute_initiative` 嚴格位於所有子項目累加完畢之後、`get_endgame_scale_factor` 之前。
* 兩處 Lazy 提前返回皆呼叫 `get_endgame_scale_factor`，此為 2026-10-16 修復要塞殘局 TT 污染的關鍵護欄（見 [`HCE_SF11_GAP_AUDIT.md:425-434`](HCE_SF11_GAP_AUDIT.md#L425-L434)），**必須完整保留**。

---

## 第三部分：冗餘計算與 Numba 熱路徑效能瓶頸 (Zero-Eval-Delta Optimizations)

以下 7 項優化**完全不改變任何局面的評估分數（Bit-Exact 0 分差）**，純粹消除熱路徑記憶體分配與重複位元運算：

1. **PERF-1：消除 `_evaluate_king_pawn_endgame` 的 NRT 堆分配並接入 Pawn Hash**（[`evaluation.py:1358-1375, 1626`](evaluation.py#L1358-L1375)）
   * **問題**：`occ_bbs = np.zeros(3, dtype=np.uint64)`（L1371）在每次評估王兵殘局時向 Numba NRT 申請並釋放堆記憶體，而呼叫端 `_evaluate_position_jit` 本就持有最新的 `occupancy_bbs`！此外 L1358 呼叫未快取的 `evaluate_pawn_structure`，放棄了深層殘局搜尋 >99% 命中率的 Pawn Hash，且每次白算最多 6 次王翼兵盾 `evaluate_shelter_aligned` 後直接丟棄 `res[2..5]`。
   * **修法**：直接傳入 `occupancy_bbs` 與 `search_context`；當 `search_context is not None` 時改呼叫 `evaluate_pawn_structure_cached`（回傳元組之 `eg_pawn_structure`、`w_passed`、`b_passed` 與未快取版 100% 恆等）。

2. **PERF-2：消除單次評估跨模組重複 3 次的全盤 `count_bits` 與王格查找**
   * **問題**：在一次常規評估中，[`evaluate_special_endgame_with_material`](endgame.py#L522-L532)、[`_process_piece_score_and_count`](evaluation.py#L1638-L1650) 與 [`get_endgame_scale_factor`](endgame.py#L645-L655) 分別對 10 個棋種位元棋盤各跑一次 `count_bits`（合計 30 次 `POPCNT`），且雙王 `get_lsb_index` 亦重複執行 3～4 次。
   * **修法**：先在 `_evaluate_position_jit` 頂部完成一次 `piece_counts` 與 `white_king_sq` / `black_king_sq` 提取（或為 `endgame.py` 提供接受預先算好 `piece_counts` 的內部入口），將 20 次冗餘 `POPCNT` 降為 0。

3. **PERF-3：外提主教迴圈不變量 + 大對角線遮罩過濾二次 Magic 查表**（[`evaluation.py:768-795, 977-1005`](evaluation.py#L768-L795)）
   * **問題**：
     * `blocked_w = wp_bb & (all_occupancy >> 8)`、`center_blocked`、`multiplier`、`pawn_only_occ = wp_bb | bp_bb` 與單顆主教位置 `sq` 完全無關，卻在 `while temp_bb:` 迴圈內每隻象重算一次。
     * `LongDiagonalBishop` 對站在任何格子上的主教都無條件執行第二次 `get_bishop_attacks(sq, pawn_only_occ) & CENTER_SQUARES`；但幾何上，只有站在**兩條大對角線非中心 12 格**（`a1, b2, c3, f6, g7, h8, a8, b7, c6, f3, g2, h1`，遮罩 `0x8142240000244281`）的主教才可能同時攻擊 2 個中心格！
   * **修法**：將不變量外提至 `if wb_bb:` / `if bb_bb:` 區塊一次算完，並加上 `if sq_bb & np.uint64(0x8142240000244281):` 守衛，省下 ~81% 的二次 Magic Bitboard 查表。

4. **PERF-4：簡化 `evaluate_space` 傳參與編譯期常數空間遮罩**（[`evaluation.py:1449-1510`](evaluation.py#L1449-L1510)）
   * **修法**：直接傳入外部已算好的 `npm`（省去 8 次乘法與陣列讀取），移除未使用的 `occupancy_bbs` 參數，並將每次執行 10 次 `FILE_MASKS`/`RANK_MASKS` 查表的 `white_space_mask` / `black_space_mask` 替換為常數 `np.uint64(0x000000003C3C3C00)` 與 `np.uint64(0x003C3C3C00000000)`。

5. **PERF-5：`ThreatByMinor` / `ThreatByRook` 防空轉守衛與移除 `pt = 5` 恆零迭代**（[`evaluation.py:1150-1180`](evaluation.py#L1150-L1180)）
   * **修法**：外層加上 `if w_minor_threatens:` / `if b_minor_threatens:` / `if w_rook_threatens:` / `if b_rook_threatens:`（安靜局面直接跳過全部 4 個迴圈），並將 `range(6)` 縮減為 `range(5)`（`black_weak` 與 `black_defended` 已排除國王，`pt = 5` 數學上恆為 `0`）。

6. **PERF-6：移除死變數 `white/black_minor_attacks` 並重用 `white_safe` / `black_safe`**
   * 移除 [`evaluation.py:656, 666, 698, 747, 906, 956`](evaluation.py#L656) 從未讀取的 `white_minor_attacks` 與 `black_minor_attacks`。
   * 將 [`evaluation.py:1226, 1236`](evaluation.py#L1226) `ThreatByPawnPush` 中的 `(~black_attacks | white_attacks)` 與 `(~white_attacks | black_attacks)` 直接替換為 L1127、L1136 已算好的 `white_safe` 與 `black_safe`。

7. **PERF-7：清除恆真條件、重複純量與無用跨函數參數**
   * 移除 [`evaluation.py:1716, 1782`](evaluation.py#L1716) 恆為 `True` 的 `if not lazy:`（L1687 已 `if lazy: return`）。
   * 移除 [`evaluation.py:237, 319`](evaluation.py#L237) 恆為 `True` 的 `if block_sq < 64:` / `if block_sq >= 0:`，並合併同一迴圈內重算兩次的 `w = np.int32(dyn_mult * rank + dyn_off)`。
   * 移除 [`evaluate_king_safety:453-454`](evaluation.py#L453-L454) 從未使用的 `game_state` 與 `piece_counts` 參數，縮減 `@njit` 呼叫簽章。
   * 註：`KING_PROTECTOR` 在 `constants.py:453` 雖為 `[0, 0]`，但因 [`tuner/parameters.py`](../../tuner/parameters.py) 與 [`generate_tunable_eval.py`](../../tuner/codegen/generate_tunable_eval.py) 會透過正則將 `KING_PROTECTOR[0]` 改寫為 `get_array_val(theta, IDX_KING_PROTECTOR, 0)`，**必須保留該運算式名稱**或加 `if KING_PROTECTOR[0] != 0 or KING_PROTECTOR[1] != 0:` 以維持 Texel Tuner 100% 相容。

---

## 第四部分：工程護欄與工具鏈耦合約束 (Mandatory Guardrails)

任何執行本計畫的開發者或 AI Agent **必須嚴格遵守**以下四條專案不變量：

> [!CAUTION]
> **護欄 1：`search.py` 牽制遮罩回傳契約不可更改**  
> `_evaluate_position_jit` 回傳的 `(val, pinned_white, pinned_black)` 直接供 [`search.py`](search.py) 進行著法合法性快速過濾：
> 1. `pinned_white` / `pinned_black` **必須僅包含己方被絕對牽制的棋子**（不可混入敵方閃將阻擋子）。
> 2. 所有提前返回分支（特化殘局、純王兵殘局、顯式／動態 Lazy 出口）**必須回傳 `PINNED_UNCOMPUTED_SENTINEL`（`0xFFFFFFFFFFFFFFFF`）**，絕不可回傳 `0`。

> [!WARNING]
> **護欄 2：`tuner/tunable_eval.py` 自動生成管線同步**  
> 每次修改 [`evaluation.py`](evaluation.py)、[`pawns.py`](pawns.py) 或 [`material.py`](material.py) 後，**必須執行**：
> ```bash
> python -m tuner.codegen.generate_tunable_eval
> python -m unittest tests.test_tunable_eval_parity
> ```
> 注意 [`generate_tunable_eval.py`](../../tuner/codegen/generate_tunable_eval.py) 的三項耦合機制：
> 1. **固定 Header 匯入（L248–260）**：原始 `evaluation.py` 的 `import` 區塊會被剝除，若新增匯入符號（如 `LINE_BB`），必須同步寫入 `generate_tunable_eval.py` 的 `header`。
> 2. **`theta` 參數自動注入清單（L284–302）**：若新增或改名內部讀取評估常數的 `@njit` 輔助函式，必須將函式名稱加入 `funcs` 清單；反之，純數學輔助函式（如 `_trunc_div`）**不可**加入 `funcs`。
> 3. **明確簽章注入規則（L172–212）**：帶有明確 Numba 型別簽章的裝飾器若在 `funcs` 內，格式須保持 `@numba.njit(ret_type(...), cache=True, ...)` 以便 `_fix_numba_signatures` 正確插入 `numba.float64[:]`。

> [!IMPORTANT]
> **護欄 3：歷史已驗證結論與 Numba Disk Cache 規範**  
> * **禁止恢復**已回滾或移除之非 SF 項目：通路兵材質縮放（Phase B4 移除）、王兵殘局 P3b「僅競速唯一贏家領取不可阻擋獎賞」（Phase P3 回滾）。
> * **禁止移除**兩處 Lazy 出口與純王兵殘局的 `get_endgame_scale_factor` 護欄（2026-10-16 防要塞 TT 污染修復）。
> * 新增任何模組級 NumPy 查表須保持連續記憶體且 `nbytes <= 1_000_000`，確保 `@numba.njit(cache=True, boundscheck=False, fastmath=True)` 跨進程磁碟快取有效。

---

## 第五部分：分階段實施路線圖與驗收標準 (Phased Execution Plan)

為確保每一步改動皆可獨立歸因（效能 vs 準確度 vs 棋力 Elo），採 **4 階段推進**：

```mermaid
flowchart TD
    P1["Phase 1: 零分數變動熱路徑加速 (PERF-1 ~ PERF-7)"] --> G1{"Gate 1: 1000 FEN 0 分差\n& Depth 15 0-Node Delta"}
    G1 -->|通過| P2["Phase 2: 幾何與殘局契約修復 (BUG-1, BUG-2, BUG-4)"]
    P2 --> G2{"Gate 2: 單元測試全綠\n& 非退化對局驗證"}
    G2 -->|通過| P3["Phase 3: 通路兵計算順序對齊 (ORD-1 ~ ORD-3)"]
    P3 --> G3{"Gate 3: 單進程 A/B 對局\n(400-600 局 @ 300k SPRT)"}
    G3 -->|通過| P4["Phase 4: 黑白鏡像對稱與 SF11 細節對齊 (BUG-3, BUG-5 ~ BUG-8)"]
    P4 --> G4{"Gate 4: 500 FEN 鏡像 0 誤差\n& LTC 對局定錨 + 同步文件"}
```

### Phase 1：零分數變動效能重構（Zero-Eval-Delta NPS Optimization） — ✅ 已完成 (2026-10-31)
* **實施範圍**：`PERF-1`、`PERF-2`、`PERF-3`、`PERF-4`、`PERF-5`、`PERF-6`、`PERF-7`。
* **已落地改動**：
  1. [`endgame.py`](endgame.py)：`evaluate_special_endgame_with_material` 頂部新增 `<= 1` 非王棋子快速位元短路（免除 >99% 常規節點的 10 次 `POPCNT` 與 22 道分支）；新增 `get_endgame_scale_factor_with_counts` 直接接收預先算妥的 `cnt_0..cnt_10` 與雙王格，並以 `w_minor_major <= 1 and b_minor_major <= 1` 守衛 560 行特化殘局縮放分支。
  2. [`evaluation.py`](evaluation.py)：
     * `_evaluate_king_pawn_endgame` 徹底移除 `np.zeros(3)` NRT 堆分配，重用 `occupancy_bbs` 並於 `search_context is not None` 時接入 `evaluate_pawn_structure_cached`。
     * 雙王 PST 改為直接索引（省去 2 個 `while` 迴圈與 2 次 `get_lsb_index`）。
     * 主教迴圈外提 `w_bp_mult` / `b_bp_mult` / `pawn_only_occ` 不變量，以位元遮罩取代 `(sq // 8 + sq % 8) % 2`，並加上 `0x8142240000244281` 大對角線遮罩過濾二次 Magic 查表；`KING_PROTECTOR == 0` 時跳過距離查表。
     * `evaluate_space` 直接接收 `npm`、改用常數空間遮罩 `0x000000003C3C3C00` / `0x003C3C3C00000000` 並合併正除數。
     * `ThreatByMinor` / `ThreatByRook` 加入非零守衛與 `range(5)`；`ThreatByPawnPush` 重用 `white_safe` / `black_safe`；清除死變數 `white/black_minor_attacks`、恆真 `if not lazy:`、`if block_sq < 64:`、重複 `w` 計算與 `evaluate_king_safety` 未使用參數。
  3. 同步更新 [`generate_tunable_eval.py`](../../tuner/codegen/generate_tunable_eval.py) 並重新生成 [`tunable_eval.py`](../../tuner/tunable_eval.py)。
* **實測驗收結果（Gate 1 通過）**：
  * **4,000 組評估逐位元零分差**：1,000 個涵蓋開中殘局、特化殘局、純王兵殘局、要塞縮放之 FEN × 4 種呼叫模式（`lazy=False/True` × `search_context=None/SearchContext`），`(score, pinned_w, pinned_b)` **`mismatches = 0`**（因純函數輸出 100% 恆等，保證搜尋樹節點數 0-Node Tree Delta）。
  * **同進程交替 `@njit` 吞吐量基準（5 輪中位數）**：
    * 一般 1,000-FEN 完整評估：`386,568` → **`671,585` eval/s（+73.73%）**
    * 純王兵殘局（KP Endgame）評估：`2,415,412` → **`3,006,343` eval/s（+24.47%，零堆記憶體分配）**
  * **單元測試與磁碟快取**：`tests.test_endgame_conformance` 與 `tests.test_tunable_eval_parity` 共 **36 項測試 100% 通過**；`NUMBA_DEBUG_CACHE=1` 驗證 `.nbc` 跨進程磁碟快取 100% 命中，無 dynamic globals 警告。

---

### Phase 2：高確定性幾何錯誤、殘局自阻擋與 `eval_weights` 接線修復【✅ 100% 全數完成】
* **實施範圍**：
  * [x] **`BUG-1`（Pinned Queen `LINE_BB`）**：✅ 已完成。象／車／后 6 個迴圈牽制過濾統一改為 `att &= LINE_BB[king_sq, sq]`，同步更新 `generate_tunable_eval.py` Header，重新生成 `tunable_eval.py`，通過專屬幾何單元測試與 Parity 測試。
  * [x] **`BUG-2`（王兵方形法則疊兵／王自阻擋）**：✅ 已完成。`_evaluate_king_pawn_endgame` 排除同線前兵、起點雙步淨空檢驗、己方王擋路 `+1` 修正；修正既有測試假陽性並新增黑方疊兵與王阻擋測試；通過所有殘局規範與 Parity 測試。
  * [x] **`BUG-4`（補齊特化殘局、王兵殘局、空間評估之 `ew` 傳遞；`EG_SAFETY_SCALE` 確認對齊策略）**：✅ 已完成。統一 `EG_SAFETY_SCALE = 100`；特化殘局與王兵殘局全數接入 `_material_mg_eg` 與 `ew, use_ew`；新增專屬測試並通過全量測試。
* **實作重點**：
  1. 象／車／后 6 個迴圈牽制過濾統一改為 `att &= LINE_BB[king_sq, sq]`，同步更新 `generate_tunable_eval.py` Header。
  2. `_evaluate_king_pawn_endgame` 補上同線前兵排除、起點雙步淨空檢查與己方王擋路 `+1` 步修正。
  3. 修正 [`tests/test_endgame_conformance.py:233-244`](../../tests/test_endgame_conformance.py#L233-L244) 的單兵 KPK 假陽性對照組，並新增：
     * [`tests/test_pinned_eval_geometry.py`](../../tests/test_pinned_eval_geometry.py)（驗證 `Ke1+Qe3+Re8` 不含 `c1/g1`，`Ke1+Qg3+Bh4` 不含 `c3`）。
     * `test_doubled_pawns_front_blocks_rear_unstoppable`、`test_black_doubled_pawns_front_blocks_rear_unstoppable`、`test_king_on_path_blocks_unstoppable_pawn`。
     * [`tests/test_runtime_eval_weights.py`](../../tests/test_runtime_eval_weights.py)（驗證常數一致性與 KQKR 殘局受 runtime Queen EG 材質即時連動）。
  4. **關於 `EG_SAFETY_SCALE` 的決策點**：已將 `eval_weights.py`、`constants.py` 與 `evaluation.py` 接線統一為讀取 `constants.EG_SAFETY_SCALE = 100`（維持現行實戰基準 100）。
* **驗收標準（Gate 2 通過）**：
  * 新增與既有輕量單元測試共 38 項 100% 通過（`test_endgame_conformance`, `test_pinned_eval_geometry`, `test_runtime_eval_weights`）。
  * Tunable Parity 測試 7 項 100% 通過（`test_tunable_eval_parity`）。
  * **實戰對局驗收（Colab 長時控 400,000 nodes/move，成對換先開局對陣 `classical_old`）**：
    * **`copy 8.ipynb`（Phase 1 + BUG-1，Commit `78e9286`，1,000 局完整跑滿）**：
      * 總戰績：**513.5 - 486.5**（217 勝 593 和 190 負），勝率 **51.35%**，淨勝 **+27 場**。
      * Elo 表現：**+9.38 ± 13.74 Elo**（SPRT LLR = 1.03）。
      * 結論：千局大樣本確認 BUG-1 幾何修復具備實質正向收益（~+9.4 Elo）。
    * **`copy.ipynb`（Phase 2 累積至 BUG-4，Commit `34a7ef0`，686 局提前達成）**：
      * 總戰績：**369.0 - 317.0**（165 勝 408 和 113 負），勝率 **53.79%**，淨勝 **+52 場**。
      * Elo 表現：**+26.39 ± 16.53 Elo**（95% CI：`[+9.64, +43.26]`，下界嚴格大於 0）。
      * SPRT 狀態：**`LLR = 2.95 >= 2.94`**，**`H1 Accepted (Engine improved)` 正式通過晉級！**
      * 結論：殘局安全性統一與 `eval_weights` 接通後，深層殘局推演品質飛躍，純決策 Elo 提升 +26.39！

---

### Phase 3：動態通路兵計算順序與橫排門檻對齊【進行中】
* **實施範圍**：
  * [x] **`ORD-1`（邊緣直線懲罰 `PassedFile` 移至候選／受阻折半之後最後扣除）**：✅ 已完成 (2026-10-31)。
    * 白黑雙迴圈移除頂部提前扣除 `PassedFile * edge_dist`，改為在 `if not next_passed or any_pawn_ahead: p_bonus //= cand_div` 折半完成後才扣除，徹底消除低橫排中央兵負分折半反而倒貼 `+24 cp` 導致候選／受阻劣勢兵估值反超暢通真通路兵的長期重大 BUG。
    * 新增專屬單元測試 [`tests/test_passed_pawn_order.py`](../../tests/test_passed_pawn_order.py)（4/4 項通過），驗證 `d4` / `d5` 暢通通路兵評估分數在 MG 與 EG 均嚴格高於受阻兵與候選兵，且 `d3` 邊緣懲罰不被折半。
    * 同步更新生成 [`tuner/tunable_eval.py`](../../tuner/tunable_eval.py)，通過全量 42 項單元測試與 7 項 Tunable Parity 測試。
  * [x] **`ORD-2`（國王距離門檻對齊 `rank > 2` / `relative_rank > 2` 並納入候選／受阻折半前累加）**：✅ 已完成 (2026-10-31)。
    * 統一白黑雙迴圈國王距離門檻為 `rank > 2`（原 `rank > 3` 漏算第 4 橫排所有王接近度支援，現 `d4` 兵己王接近正確獲得 +6 EG 支持）。
    * 將 `proximity_bonus` 提前累加至 `p_bonus_eg`，使其受後續候選兵／受阻兵 `// cand_div` 縮放約束（實測 `d5` 候選兵王支持由 40 嚴格減半為 20 EG），消除了受阻無根兵享受 100% 國王支持的邏輯漏洞。
    * 擴充 [`tests/test_passed_pawn_order.py`](../../tests/test_passed_pawn_order.py) 覆蓋第 4 橫排王支援與候選兵王支援折半（6/6 項通過），全量 51 項單元測試與 Parity 測試 100% 通過。
  * [x] **`ORD-3`（候選／受阻兵向零截斷 `_trunc_div` 語義對齊）**：✅ 已完成 (2026-10-31)。
    * 引入 `@numba.njit(cache=True, inline="always") def _trunc_div(n, d): return n // d if n >= 0 else -((-n) // d)`。
    * 將白黑雙迴圈中之候選兵／受阻兵折半運算由 `p_bonus //= cand_div` 改為 `_trunc_div(p_bonus, cand_div)`，嚴格對齊 C++ 語義 `bonus = bonus / 2`，消除 Python `//` 對負數向下取整的不對稱偏差。
    * 擴充 [`tests/test_passed_pawn_order.py`](../../tests/test_passed_pawn_order.py) 覆蓋負值向零截斷對齊測試（7/7 項通過），全量單元測試與 Parity 測試 100% 通過。
* **實作重點**：
  1. 重排 `evaluate_passed_pawns` 白黑雙迴圈：將國王距離門檻統一為 `rank > 2`（`relative_rank > 2`）並併入 `p_bonus_eg` → 執行候選／受阻兵向零截斷折半（`_trunc_div`）→ **最後才扣除 `PassedFile * edge_dist`**。
  2. 新增單元測試覆蓋真通路兵高於候選兵、第 4 橫排王支持、候選兵折半約束、向零截斷對稱性。
  3. 重新生成 `tuner/tunable_eval.py` 並驗證 parity。
* **驗收標準（Gate 3）**：
  * 單元測試 100% 通過（7/7 項 Passed Pawn 測試 + 7/7 項 Tuner Parity 測試）。
  * **實戰對局驗收**：已準備 Colab 對戰驗證（建議 400～600 局 @ 400k nodes/move，SPRT 驗收），驗證 Phase 3 (`ORD-1` ~ `ORD-3`) 相較基線的棋力提升。

---

### Phase 4：黑白鏡像對稱化（`_trunc_div`）與 SF11 威脅／棋子細節對齊【✅ 100% 全數完成】
* **實施範圍**：
  * [x] **`BUG-3`（全盤向下取整替換為向零截斷 `_trunc_div`）**：✅ 已完成 (2026-10-31)。
    * 於 `material.py` 與 `evaluation.py` 引入向零截斷除法 `_trunc_div`，替換所有有號分數除法：
      - 材質不平衡除法 `(w_bonus - b_bonus) // 16` 與 scale 乘積（消除了 1445 時 $1445//16=90$ vs $-1445//16=-91$ 的 1 cp 鏡像偏差）。
      - Tapered 評估混合：`(mg_score * phase + eg_score * (128 - phase)) // 128`。
      - Scale Factor 縮放：`eg_score * sf // 64`。
      - 動態 Lazy 評估均值門檻：`(mg_score + eg_score) // 2`。
      - 王安全殘局縮放：`((white_eg_score - black_eg_score) * eg_safe_scale) // 100`。
    * **實測 500-FEN 黑白鏡像翻轉測試**（`hist_diag_openings_500.epd`）：**500/500 盤面 100% Bit-Exact 零誤差**，徹底消除系統性偏袒黑方的 1~2 cp 鏡像偏差！
  * [x] **`BUG-5`（`ThreatByKing` 布林單次觸發）**：✅ 已完成 (2026-10-31)。
    * 對齊 SF11 `evaluate.cpp:514-515`：`if (KING_ATTACKS[sq] & enemy_weak) != 0`，觸發時直接累加單次 `THREAT_BY_KING`，不再乘以 `popcount`。王單回合僅能吃一子，多顆懸子威脅由下方的 `THREAT_HANGING` 線性承擔。
  * [x] **`BUG-6`（`MinorBehindPawn` 納入敵方兵掩體／封鎖格）**：✅ 已完成 (2026-10-31)。
    * 對齊 SF11 `evaluate.cpp:302` `shift<Down>(pos.pieces(PAWN))`：將原先僅檢查己方兵 `wp_bb & BB_SQUARES[sq + 8]` 擴展為雙方兵 `all_pawns = wp_bb | bp_bb`。騎士與主教位於敵兵正後方（Nimzowitsch 封鎖格）同樣享有免受敵方正面車重子直擊的天然掩體支持。
  * [x] **`BUG-7`（`WeakQueen` 排除長程狙擊子疊線假阻擋）**：✅ 已完成 (2026-10-31)。
    * 對齊 SF11 `position.cpp:458` `Position::slider_blockers`：在計算雙方皇后被車／象串擊（pin / discovered attack）時，以 `snipers = (ROOK_RAYS & rooks) | (BISHOP_RAYS & bishops)` 預先從佔據位元盤剔除 `occ = all_occupancy ^ snipers`，消除了同一直線雙車或同斜線雙象重疊時被誤算為單一阻擋子而扣罰 `WEAK_QUEEN` 的誤判。
  * [x] **`BUG-8`（`kingAttacksCount` 嚴格移入 `king_zone` 內累加）**：✅ 已完成 (2026-10-31)。
    * 對齊 SF11 `evaluate.cpp:280-285`：白黑雙方共 8 個棋子迴圈（Knight, Bishop, Rook, Queen）將 `king_attacks_count += count_bits(att & king_adjacent)` 嚴格移入 `if att & king_zone:` 區塊內。當棋子未進犯敵王警戒區時，完全略過位元與 popcount 運算，兼具效能與語義精確對齊。
  * [x] **`KP-DISCONT`（純王兵殘局分數連續性修復）**：✅ 已完成 (2026-11-01)。
    * 修復 `all_pieces_except_pawns_and_kings == 0` 分支漏加 `tempo`、漏套用 `sf != 64` 縮放與 50 步衰減的重大斷層缺陷。
    * 若 `sf == SCALE_FACTOR_DRAW` 直接回傳精確 0 分；否則套用 `_trunc_div(score * sf, 64)` 與 `tempo` 加成，徹底消除吃掉最後一顆輕子進入王兵殘局時的分數斷層與 50 步規則失效問題。
* **工具鏈與 Tuner 同步**：
  * 更新 `tuner/codegen/generate_tunable_eval.py` 正則表達式，自動剝離 material 匯入並內嵌 `_trunc_div`。
  * 重新生成 `tuner/tunable_eval.py`（114,127 bytes）。
* **驗收結果（Gate 4 通過）**：
  * 新增專屬單元測試 [`tests/test_phase4_sf11_alignment.py`](../../tests/test_phase4_sf11_alignment.py)（5/5 項 100% 通過，涵蓋 500-FEN 鏡像對稱、ThreatByKing 布林次數、敵兵後方哨站、WeakQueen 狙擊子、kingAttacksCount 範圍）。
  * 全量 Tuner Parity 測試 [`tests/test_tunable_eval_parity.py`](../../tests/test_tunable_eval_parity.py) 驗證 100% 一致。

