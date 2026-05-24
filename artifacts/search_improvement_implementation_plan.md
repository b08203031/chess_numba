# NNUE Search 改進計畫書（結合 Calibration 決策與 Stockfish 官方參考）

產出日期：2026-05-24  
用途：提交給 AI coding assistant，作為修改 `search.py` / `constants.py` 的實作計畫。  
主策略來源：`artifacts/calibration_combined_decision_report.md` + `artifacts/search_py_stockfish18_review.md`

## 0. 執行總原則

本輪改進分成兩類：

1. **必須修改的搜尋邏輯 bug / safety guard**：這些會直接影響搜尋正確性、TT bound、LMR/history、quiet check、singular exclusion search。應優先修。
2. **暫不修改的 calibration 參數**：`ASPIRATION_WINDOW_SIZE`、`FP_BASE`、`FP_MULTIPLIER`、`RFP_BASE_MULT`、`RAZORING_MARGIN` 先保持目前值，不採用外部 plan 的 margin 數字。

原因：`artifacts/calibration_combined_decision_report.md` 已判斷 P1 異常主要集中在低子力與 pawn/endgame，而目前 margin 已比外部 plan 建議更保守；現在更重要的是先修搜尋邏輯，避免 benchmark 繼續被 bug noise 污染。

## 1. 官方參考資料

請實作 AI 助手優先參考下列官方來源：

| 用途 | 官方來源 |
|---|---|
| Stockfish 18 官方搜尋碼 | `https://github.com/official-stockfish/Stockfish/blob/sf_18/src/search.cpp` |
| Stockfish 18 release/tag | `https://github.com/official-stockfish/Stockfish/releases/tag/sf_18` |
| Stockfish 官方主線搜尋碼，可作行號交叉核對 | `https://github.com/official-stockfish/Stockfish/blob/master/src/search.cpp` |
| Stockfish 官方主線 move picker | `https://github.com/official-stockfish/Stockfish/blob/master/src/movepick.cpp` |

本計畫中的 `SF search.cpp:Lx-Ly` 是指 Stockfish 官方 `src/search.cpp` 中相同 Step / 程式片段的行號範圍；若 GitHub tag 頁面顯示行號略有位移，請用 Step 註解與關鍵字搜尋核對。

## 2. 不要改的項目（Calibration 決策）

### 2.1 保留 search margins

**檔案 / 位置**：`constants.py:626-L711`

不要改：

```python
ASPIRATION_WINDOW_SIZE = 32
RAZORING_MARGIN = 760
FP_BASE = 254
FP_MULTIPLIER = 254
RFP_BASE_MULT = 254
PROBCUT_MARGIN = 190
```

**原因**：

- `calibration_combined_decision_report.md` 顯示外部 plan 的 margin 數字多數會讓 pruning 更激進，而不是更保守。
- P1 full-search bucket 顯示問題集中在 `pawn_endgame_rfp` / `low_material_nmp`，不是 quiet/tactical 全域 scale 或 margin 全面失衡。

**官方參考**：

- Stockfish 的 pruning margin / scaler 類參數高度依賴測試與非線性調校，不應未經對局/benchmark 直接搬數字。（參考 `SF search.cpp:L66-L70` 關於 time-management scaler 的調校警告；同一原則適用 search margin calibration。）

### 2.2 保留低子力 guard 常數

**檔案 / 位置**：`constants.py:681-L684`

保留：

```python
NMP_STATIC_MARGIN = 190
NMP_MIN_SIDE_NON_PAWNS = 2
LOW_MATERIAL_PRUNING_PIECE_COUNT = 7
```

**原因**：

- `calibration_combined_decision_report.md` 的 P1 數據顯示 `P06`、`N01-N04` 是主要異常來源。
- 目前 guard 已比外部 plan 的「major piece 或 two minors」形式更保守：同時要求低子力區外、行棋方至少兩個非兵子、static 高於 beta margin。

**官方參考**：

- Stockfish NMP 會用 non-pawn material / eval margin / depth guard 避免 zugzwang 與低子力誤剪。（參考 `SF search.cpp:L867-L895` Null Move Pruning step。）

## 3. 必須修改 P0：搜尋語意與安全性

### P0-1. 修正 LMR/history 使用 make move 後錯誤 side/from-square

**檔案 / 位置**：`search.py:932-L958`、`search.py:1108-L1148`、`search.py:1208-L1282`

**目前問題**：

`search.py` 在 `make_move()` 後才於 LMR 區塊使用：

```python
aggressor_type = find_piece_type_on_square_side(piece_bbs, from_sq, game_state[0])
```

但 make move 後：

- `game_state[0]` 已切到對手。
- `from_sq` 已空。

因此 `aggressor_type` 可能變成 `-1`，污染 `history_table`、`continuation_history`、LMR reduction。

**修改方式**：

1. 在 move loop 中，`make_move()` 前保存：

```python
original_side = game_state[0]
from_sq = get_from_square(move)
to_sq = get_to_square(move)
flag = get_special_move_flag(move)
```

2. `make_move()` 後，立刻使用 `unmake_info[0]` 保存移動棋子：

```python
moved_piece_type = unmake_info[0]
```

3. LMR、history bonus/malus、pawn push protection 全部改用 `moved_piece_type` / `original_side`，不要再從 `from_sq` + post-move `game_state[0]` 查棋子。

4. 對所有仍需查棋子的地方加入 guard：

```python
if aggressor_type == -1:
    # 不更新 history / 不使用該值進 LMR
```

**實作注意**：

- `moved_piece_type` 目前已在 `search.py:937` 設定，但後續 LMR 區塊沒有用它。
- `is_pawn` 應改成：

```python
is_pawn = (moved_piece_type % 6) == PAWN
```

- 兵推方向應用 `original_side` 判斷：

```python
if (original_side == WHITE and to_rank >= 5) or (original_side == BLACK and to_rank <= 2):
    lmr = max(0, lmr - 2)
```

**修改原因**：

LMR 是強剪枝/降深邏輯，若 history 索引錯誤，會系統性錯降好棋或錯保護壞棋。這是比參數更優先的棋力問題。

**官方參考**：

- Stockfish 在 move loop 中先取 `movedPiece = pos.moved_piece(move)` 與 `givesCheck = pos.gives_check(move)`，後續 history / statScore / LMR 都使用保存好的 `movedPiece`。（`SF search.cpp:L1030-L1034`、`L1194-L1248`）
- Stockfish 先檢查合法性，再進 history / pruning。（`SF search.cpp:L1005-L1010`）

### P0-2. 防止 shallow pruning 剪掉 quiet check

**檔案 / 位置**：`search.py:900-L928`、`search.py:955-L961`、`search.py:1009-L1033`

**目前問題**：

`search.py:900-L928` 在 make move 前做 shallow SEE/history pruning。此時還不知道走法是否 `is_giving_check_after_move`，因此非吃子、非升變的 quiet check 可能被當成 quiet move 剪掉。

**修改方式**：

建議採用保守方案：

1. 保留 capture SEE pruning 在 make move 前，但新增註解與 guard：不要剪掉可能 checking 的 capture；如果沒有便宜 `gives_check(move)`，capture pruning 可以先維持，但 quiet pruning 必須移後。
2. 將 quiet history pruning / quiet SEE pruning 從 `search.py:912-L928` 移到 `make_move()`、legality check、`is_giving_check_after_move` 計算之後。
3. 新位置應在 `search.py:960-L1033` 附近，也就是已經有：

```python
is_giving_check_after_move = is_square_attacked(...)
is_quiet_move = is_pseudo_quiet and not is_giving_check_after_move
```

之後再做：

```python
if ENABLE_SHALLOW_SEE_PRUNING and depth <= PRUNING_SHALLOW_DEPTH and not is_currently_in_check and not is_pv:
    if is_quiet_move:
        # history pruning + quiet SEE pruning
```

4. 所有 quiet pruning 都必須以 `is_quiet_move` 為條件，不得用 `is_pseudo_quiet`。

**修改原因**：

quiet check 是戰術入口，錯剪會造成 missed mate / missed tactic。這類錯誤比 nodes 增減更嚴重。

**官方參考**：

- Stockfish 在 pruning 前計算 `givesCheck = pos.gives_check(move)`。（`SF search.cpp:L1030-L1034`）
- Stockfish Step 14 的 capture futility 有 `!givesCheck` guard；quiet pruning 分支只在非 capture / 非 check 情境進行。（`SF search.cpp:L1050-L1114`）

### P0-3. 修正 excluded-move search：禁止被排除走法透過 pruning/ProbCut 回來

**檔案 / 位置**：`search.py:366`、`search.py:501`、`search.py:653-L830`、`search.py:1043-L1095`

**目前問題**：

singular extension 的 exclusion search 應排除 TT move，檢查其他走法是否都低於 singular beta。但目前：

- RFP / NMP / Razoring / ProbCut 仍可在 exclusion search 中提前 return。
- ProbCut capture loop 沒有跳過 `pc_move == excluded_move`。
- singular extension 條件沒有 `excluded_move == NO_MOVE`，可能 recursive singular。

**修改方式**：

1. 在 `_search()` 入口加入：

```python
is_exclusion_search = excluded_move != NO_MOVE
```

2. 修改 TT cutoff guard：現有 guard 已部分處理，保留並確認語意是：

```python
if ply > 0 and not is_pv and not is_exclusion_search and tt_entry['depth'] >= depth ...:
```

或至少確保 excluded search 不被 TT bound 直接 cutoff。

3. RFP / NMP / Razoring / ProbCut / IIR 條件都加入 `not is_exclusion_search`，至少第一版保守處理：

```python
if not is_exclusion_search and ENABLE_RFP ...
if not is_exclusion_search and ENABLE_NMP ...
if not is_exclusion_search and ENABLE_PROBCUT ...
if not is_exclusion_search and ENABLE_RAZORING ...
if not is_exclusion_search and ENABLE_IIR ...
```

4. ProbCut loop 中加入：

```python
if pc_move == excluded_move:
    continue
```

5. Singular extension 條件改成：

```python
if ENABLE_SINGULAR_EXTENSIONS and excluded_move == NO_MOVE and move == tt_move ...:
```

**修改原因**：

若 exclusion search 被 RFP/NMP/ProbCut 直接 cutoff，它可能沒有真正搜尋「除 TT move 以外的走法」，singular extension 判斷會失真，造成錯誤 extension 或錯誤 multi-cut。

**官方參考**：

- Stockfish TT cutoff 要避開 `excludedMove`。（`SF search.cpp:L711-L715`）
- Stockfish NMP 條件含 `!excludedMove`。（`SF search.cpp:L867-L870`）
- Stockfish ProbCut loop 跳過 `move == excludedMove`。（`SF search.cpp:L933-L938`）
- Stockfish singular extension 條件含 `!excludedMove`，並用 `ss->excludedMove = move` 做 reduced exclusion search。（`SF search.cpp:L1122-L1143`）

### P0-4. 讓 ProbCut 與 MultiCut 開關真正生效

**檔案 / 位置**：`constants.py:632`、`constants.py:637`、`search.py:724-L822`、`search.py:1087-L1095`

**目前問題**：

- `ENABLE_PROBCUT` 有 import，但 ProbCut 條件沒有使用它。
- `ENABLE_MULTICUT = False`，但 singular extension 內仍有類 multi-cut return。

**修改方式**：

1. ProbCut 條件改為：

```python
if search_context.enable_probcut and not is_exclusion_search and depth >= 5 ...:
```

若目前 `SearchContext` 還沒有 dynamic flag，先用：

```python
if ENABLE_PROBCUT and not is_exclusion_search and depth >= 5 ...:
```

但依 `artifacts/task.md`，應優先使用 `search_context` flags。

2. MultiCut branch 加開關：

```python
elif ENABLE_MULTICUT and exclusion_beta >= beta:
    ...
```

若有 dynamic flag，使用 `search_context.enable_multicut`。

3. 若沒有把握，第一版可直接停用 singular 內的 multi-cut return，保留 singular extension 本身。

**修改原因**：

開關不生效會讓 UCI `setoption` 與 calibration 實驗失真。`ENABLE_MULTICUT=False` 但仍執行 multi-cut 類 return，會讓報告與實際行為不一致。

**官方參考**：

- Stockfish ProbCut 是獨立 Step 11，有明確 guard 和 verification。（`SF search.cpp:L916-L974`）
- Stockfish singular/multicut 類邏輯位於嚴格 singular guard 內。（`SF search.cpp:L1122-L1143`）

### P0-5. 修正 50-move scale-down 與 TT flag 不一致

**檔案 / 位置**：`search.py:1297`、`search.py:1326-L1328`、`search.py:1391`

**目前問題**：

目前先決定：

```python
final_flag = TT_FLAG_ALPHA if max_eval <= original_alpha else (TT_FLAG_BETA if max_eval >= beta else TT_FLAG_EXACT)
```

之後才 scale：

```python
if fifty_move_scale < FIFTY_MOVE_MAX_SCALE:
    max_eval = max_eval * fifty_move_scale // FIFTY_MOVE_MAX_SCALE
```

這可能讓「已低於 beta 的縮放分數」仍以 `TT_FLAG_BETA` 寫入 TT。

**修改方式**：

1. 將 50-move scale-down 移到 `final_flag` 計算前。
2. 或在 scale 後重新計算 `final_flag`。
3. `store_tt()` 使用的 `tt_score` 必須和 `final_flag` 基於同一個 `max_eval`。
4. 如果 `max_eval == -INFINITY`，先做 fallback，再做 scale，再決定 flag。

建議順序：

```python
if max_eval == -INFINITY:
    max_eval = original_alpha

if fifty_move_scale < FIFTY_MOVE_MAX_SCALE and abs(max_eval) < MATE_IN_MAX_PLY:
    max_eval = max_eval * fifty_move_scale // FIFTY_MOVE_MAX_SCALE

final_flag = TT_FLAG_ALPHA if max_eval <= original_alpha else (TT_FLAG_BETA if max_eval >= beta else TT_FLAG_EXACT)
```

**修改原因**：

TT bound flag 必須和 stored score 的 alpha/beta 關係一致，否則後續 TT cutoff 會錯誤。

**官方參考**：

- Stockfish TT cutoff 以 bound flag 與 value 對 alpha/beta 的關係決定是否 cutoff。（`SF search.cpp:L711-L715`）
- Stockfish 在 TT write 時，value 與 bound 是同一個 bestValue 語意。（`SF search.cpp:L1485-L1488`）
- Stockfish 對 high rule50 / GHI 有特別防護，不讓不安全的 TT cutoff 污染搜尋。（`SF search.cpp:L731-L734`）

## 4. 應修改 P1：穩定性與效果提升

### P1-1. 補 draw detection / insufficient material

**檔案 / 位置**：`search.py:11`、`search.py:452-L456`、`search.py:126-L154`

**修改方式**：

1. 在 `_search()` early draw detection 位置，加入不足子力判斷：

```python
if ply > 0 and not has_sufficient_material(piece_bbs):
    search_context.pv_table[ply, ply] = NO_MOVE
    return (np.int32(0), NO_MOVE, nodes_searched, quiescence_nodes, tt_hits)
```

實際呼叫簽名請以 `move_generator.has_sufficient_material` 為準。

2. 在 `quiescence_search()` early section 也加入一致 draw handling。

**修改原因**：

低子力 calibration 已指出 endgame 是主要風險；不足子力若不回 draw，search/qsearch leaf 可能保留不該有的 eval。

**官方參考**：

- Stockfish main search early check：`pos.is_draw(ss->ply)`。（`SF search.cpp:L668-L674`）
- Stockfish qsearch 也有 immediate draw check。（`SF search.cpp:L1558-L1560`）

### P1-2. qsearch 寫入 TT，改善 tactical leaf 重複效率

**檔案 / 位置**：`search.py:107-L124`、`search.py:147-L154`、`search.py:241-L248`

**修改方式**：

1. 在 qsearch 開頭保存：

```python
original_alpha = alpha
best_move = NO_MOVE
```

2. 在 move loop 中如果某個 capture/evasion 提升 alpha，更新 `best_move`。

3. 在 qsearch 結尾依結果寫 TT：

```python
flag = TT_FLAG_ALPHA if alpha <= original_alpha else TT_FLAG_EXACT
# beta cutoff 時用 TT_FLAG_BETA
store_tt(... depth=0 或 qsearch depth marker ..., score=alpha, static_eval=stand_pat 或 32767, flag=flag, best_move=best_move, ...)
```

4. mate score 存取仍需依現有 `_search()` 的 TT mate score 轉換規則處理。

**修改原因**：

qsearch 重複 tactical leaf 很常見，只 probe 不 store 會降低 TT 效益。

**官方參考**：

- Stockfish qsearch probe TT。（`SF search.cpp:L1516-L1566`）
- Stockfish qsearch move loop 管理 `bestMove`。（`SF search.cpp:L1631-L1708`）
- Stockfish qsearch 結尾寫 TT，depth 使用 qsearch depth。（`SF search.cpp:L1744-L1748`）

### P1-3. en passant delta pruning 修正

**檔案 / 位置**：`search.py:197-L207`

**修改方式**：

在 qsearch delta pruning 中，若是 en passant，victim value 應至少是 pawn value：

```python
if get_special_move_flag(move) == SPECIAL_MOVE_FLAG_EN_PASSANT:
    victim_value = MG_MATERIAL_VALUES[PAWN]
else:
    victim_type = find_piece_type_on_square_side(...)
    victim_value = MG_MATERIAL_VALUES[victim_type % 6] if victim_type != -1 else 0
```

**修改原因**：

en passant 的被吃兵不在 `to_square`，用 `to_square` 查 victim 會低估 capture gain，導致錯誤 delta pruning。

**官方參考**：

- Stockfish qsearch 先做 legal check，並以 position-level capture semantics 處理 move，而不是只靠 `to_sq` victim 查詢。（`SF search.cpp:L1640-L1648`）

### P1-4. root best move 優先使用 `_search()` 回傳值

**檔案 / 位置**：`search.py:1467-L1512`、`search.py:1541-L1548`

**修改方式**：

1. 在 iterative deepening 中，完成每層後優先：

```python
if res[1] != NO_MOVE:
    best_move_from_last_depth = res[1]
```

2. TT probe 僅作備援：

```python
elif tt_entry['flag'] != TT_FLAG_NONE and tt_entry['best_move'] != NO_MOVE:
    best_move_from_last_depth = tt_entry['best_move']
```

3. 若仍無 move，可用 `pv_table[0,0]` 作最後備援。

**修改原因**：

root best move 不應依賴 TT entry 一定存在；TT 可能被替換或因 key/GHI 條件找不到。

**官方參考**：

- Stockfish root search 在 root move loop 中直接維護 root move / PV / score，而不是搜尋完成後靠 TT probe 取得 best move。（`SF search.cpp:L1326-L1338`、`L2189-L2217`）

## 5. 可選 P2：之後 benchmark 再決定

### P2-1. NMP major/minor refinement

**檔案 / 位置**：`search.py:635-L648`、`search.py:672-L723`

**目前狀態**：保留 `side_non_pawn_count >= NMP_MIN_SIDE_NON_PAWNS`。

**可選修改**：若 depth 8/10 顯示低子力 guard 過度保守、nodes 大幅上升，再改成：

```python
has_major_piece = side_rooks_or_queens != 0
minor_count = count_bits(side_knights_or_bishops)
nmp_material_ok = has_major_piece or minor_count >= 2
```

但仍必須保留：

```python
not low_material_pruning_guard
static_score >= beta + NMP_STATIC_MARGIN
```

**原因**：

calibration report 判斷目前版本比外部 plan 更保守；先不要放寬，等 benchmark 證明必要再做。

**官方參考**：

- Stockfish NMP 使用 non-pawn material / eval margin / depth 共同 guard。（`SF search.cpp:L867-L895`）

### P2-2. Aspiration window 單獨測試

**檔案 / 位置**：`constants.py:626-L627`

**目前狀態**：保留 `ASPIRATION_WINDOW_SIZE = 32`。

**可選修改**：只有在 benchmark 顯示大量 aspiration fail-low/fail-high re-search 時，單獨測試 48 / 64。

**原因**：

calibration report 顯示目前缺少 re-search 統計支持，不應和 pruning safety 同輪混改。

**官方參考**：

- Stockfish aspiration window 是 iterative deepening/root search 的獨立穩定性策略，調整時應觀察 fail-high/fail-low 和 nodes，而不是用低子力 pruning 異常直接推導。（對照 Stockfish root iterative search / aspiration flow：`SF search.cpp:L271-L286`、root search loop 附近。）

## 6. 實作順序建議

請 AI coding assistant 依照以下順序實作：

1. **P0-1 LMR/history side bug**：最優先，避免 history/LMR 繼續污染。
2. **P0-2 quiet check pruning**：避免戰術性 quiet check 被剪。
3. **P0-3 excluded search guard**：修 singular extension 語意。
4. **P0-4 ProbCut/MultiCut switch**：讓 calibration / UCI option 真正可控。
5. **P0-5 TT flag vs 50-move scale**：避免 TT bound 污染。
6. **P1-1 draw detection**：強化低子力 safety。
7. **P1-3 en passant delta pruning**：小範圍正確性修正。
8. **P1-4 root best move**：改善 UCI bestmove 穩定性。
9. **P1-2 qsearch TT store**：最後做，因為牽涉 TT 寫入策略與測試較多。

## 7. 驗證計畫

### 7.1 靜態檢查

```bash
python -m py_compile search.py constants.py
```

若專案實際以 package 匯入，改用現有 engine compile command。

### 7.2 基本功能測試

至少跑：

1. startpos depth 4 / 6。
2. P1 calibration positions depth 6 / 8。
3. 一組 quiet-check tactical position，確認 shallow pruning on/off 不會改掉唯一 quiet check。
4. K vs K、K+B vs K、K+N vs K，確認 score 回 draw。
5. en passant + pin position，確認不 illegal move / 不錯剪。

### 7.3 Calibration benchmark

沿用 `calibration_combined_decision_report.md`：

```bash
python nnue_scale_benchmark.py \
  --engine ./your_engine \
  --priorities P1 \
  --depths 4,6,8,10
```

有 Stockfish 時：

```bash
python nnue_scale_benchmark.py \
  --engine ./your_engine \
  --stockfish auto \
  --priorities P1 \
  --depths 4,6,8,10
```

重點觀察：

| 指標 | 期待 |
|---|---|
| `N01-N04` full vs no-pruning 差距 | 不擴大，最好縮小 |
| `P06` volatility | 不應更糟 |
| tactical bucket | bestmove / d10 不退化 |
| quiet bucket | nodes 不爆炸 |
| low-material nodes | 可接受增加，這是 safety tradeoff |
| aspiration fail-high/fail-low | 若過多，下一輪才調 window |

## 8. 不應做的事

- 不要直接套用外部 plan 的 `FP_BASE=200`、`FP_MULTIPLIER=200`、`RAZORING_MARGIN=650`。
- 不要在同一輪同時改 search margins、LMR table、NMP guard、aspiration window，否則 benchmark attribution 會混亂。
- 不要只修參數而不修 P0 邏輯 bug。
- 不要讓 `ENABLE_MULTICUT=False` 時仍有 multi-cut 類 return。
- 不要在 exclusion search 中允許被排除的 TT move 透過 ProbCut 或 TT cutoff 回來。

## 9. 一句話結論

本輪正式改進計畫是：**保留 calibration report 已決定的低子力 guard 與目前 margins，不採用外部 plan 的 aggressive margin 數字；優先修正 `search.py` 的 LMR/history、quiet-check pruning、excluded-move search、ProbCut/MultiCut switch、TT bound flag 等搜尋語意問題，修完後再用 P1 depth 8/10 benchmark 驗證。**