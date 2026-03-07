# 走法選擇器 (Move Picker) 封裝與重構計畫書

此重構計畫的目標是針對 [search.py](file:///c:/Users/ren%20cian/OneDrive/%E6%A1%8C%E9%9D%A2/chess/chess_numba/chess_numba/chess_engine_v2/chess_engine/search.py) 中高度複雜的走法生成狀態機 (Move Picker) 進行完整封裝，讓主搜尋演算法 [_search](file:///c:/Users/ren%20cian/OneDrive/%E6%A1%8C%E9%9D%A2/chess/chess_numba/chess_numba/chess_engine_v2/chess_engine/search.py#244-1189) 只需保留極簡的讀取邏輯。

## User Review Required

> [!WARNING]
> Numba 的物件分配機制特殊。與傳統 C++ 引擎 (如 Stockfish 把 Move Picker 實例化在 Stack 上) 不同，Numba 的 `@jitclass` 物件是分配在 Heap 中的。在每秒數百萬節點的搜尋內部頻繁實例化類別會導致**極端嚴重的記憶體暴增與效能雪崩 (NPS 暴跌)**。
> 因此，我們的封裝必須採取 **「扁平化狀態陣列 + 輔助純函數 (Pure Function)」** 的方式，避免在熱迴圈內部創造任何新物件。

## Proposed Changes

---

### chess_engine/engine_types.py

修改 [SearchContext](file:///c:/Users/ren%20cian/OneDrive/%E6%A1%8C%E9%9D%A2/chess/chess_numba/chess_numba/chess_engine_v2/chess_engine/engine_types.py#58-125) 的佈局，預先為每個搜尋深度 (Ply) 分配好 Move Picker 需要的狀態變數，達成零記憶體分配 (Zero-Allocation)：

#### [MODIFY] engine_types.py
- 為 [SearchContext](file:///c:/Users/ren%20cian/OneDrive/%E6%A1%8C%E9%9D%A2/chess/chess_numba/chess_numba/chess_engine_v2/chess_engine/engine_types.py#58-125) 增加以下整數陣列成員 (大小皆為 `MAX_PLY`)，用以取代目前的區域變數：
  - `mp_stage`: 追蹤當前的生成階段 (如 `STAGE_TT_MOVE`, `STAGE_GEN_CAPTURES` 等)。
  - `mp_current_idx`: 追蹤陣列目前迭代的指標。
  - `mp_captures_end`: 好捕捉步的數量邊界。
  - `mp_quiets_end`: 靜止步的數量邊界。
  - `mp_bad_captures_count`: 壞捕捉步的總數。
  - `mp_bad_captures_idx`: 壞捕捉步當前的指標。

---

### chess_engine/search.py

將現有主迴圈中的走法生成與排序邏輯，抽出成為獨立的輔助函數：

#### [MODIFY] search.py
**1. 新增函數 `get_next_move(...)`**:
  - 定義: 
    ```python
    @numba.njit(...) 
    def get_next_move(piece_bbs, occupancy_bbs, game_state, search_context, ply, tt_move, excluded_move, killer_1, killer_2, counter_move, pinned_white, pinned_black)
    ```
  - 將原本 [_search](file:///c:/Users/ren%20cian/OneDrive/%E6%A1%8C%E9%9D%A2/chess/chess_numba/chess_numba/chess_engine_v2/chess_engine/search.py#244-1189) 內從 `while mp_stage < STAGE_DONE:` 開始的 100 多行狀態機程式碼整段移入。
  - 遇到合理的新 `move` 時便 `return move`，若已經跑完所有階段則回傳 `NO_MOVE`。內部透過更新 `search_context.mp_stage[ply]` 等狀態來維持下一次被呼叫時能精準接續上一次的階段。

**2. 化簡 [_search](file:///c:/Users/ren%20cian/OneDrive/%E6%A1%8C%E9%9D%A2/chess/chess_numba/chess_numba/chess_engine_v2/chess_engine/search.py#244-1189) 迴圈**:
  - 初始化每層的 MovePicker 狀態 (取代區域變數):
    ```python
    search_context.mp_stage[ply] = STAGE_TT_MOVE
    search_context.mp_current_idx[ply] = 0
    search_context.mp_captures_end[ply] = 0
    search_context.mp_quiets_end[ply] = 0
    search_context.mp_bad_captures_count[ply] = 0
    search_context.mp_bad_captures_idx[ply] = 0
    ```
  - 取代原本長達百行的生成區塊，改為簡潔的純函數呼叫：
    ```python
    while True:
        move = get_next_move(
            piece_bbs, occupancy_bbs, game_state, search_context, ply,
            tt_move, excluded_move, killer_1, killer_2, counter_move,
            pinned_white, pinned_black
        )

        if move == NO_MOVE:
            break
            
        # ... 接續在此之下的淺層剪枝、移動、子搜尋等核心演算法邏輯 ...
    ```

## Verification Plan

### Automated Tests
- **邏輯正確性**: 執行 `python test_puzzle.py` 確保 Numba 編譯過程沒有型別推論 (Type Inference) 錯誤，且測試謎題解答結果與原先一致。
- **效能測量**: 執行 `python benchmark.py` 測量 NPS (Nodes Per Second)。移除了 [_search](file:///c:/Users/ren%20cian/OneDrive/%E6%A1%8C%E9%9D%A2/chess/chess_numba/chess_numba/chess_engine_v2/chess_engine/search.py#244-1189) 體積龐大的樹狀分支結構並避免物件分配，預期重構後的編譯產出效能將能 **維持原本水準，甚至稍微改善緩存一致性 (Cache-coherence)**。
