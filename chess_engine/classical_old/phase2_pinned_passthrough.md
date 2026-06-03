# Phase 2：消除 `_evaluate_position_jit` 與 `_search` 之間重複的 `get_pinned_pieces` 計算

## 目標

在不改變搜尋樹的情況下，消除 `evaluation.py` 與 `search.py` 之間對 `get_pinned_pieces(WHITE)` 和 `get_pinned_pieces(BLACK)` 的重複呼叫。這是 Phase 1 之後剩餘的最大 NPS 浪費源。

> [!IMPORTANT]
> 此優化的搜尋樹影響為**零**——pinned pieces 的值完全一致，只是避免重複計算同一個結果。

---

## 背景分析

### 問題描述

目前在 `_search` 的每個非剪枝、完整評估節點中，`get_pinned_pieces` 被呼叫了**兩次**：

1. **第一次**：在 `evaluate_attacks_mobility_threats()` 內部（evaluation.py:801-802）
   ```python
   pinned_white = get_pinned_pieces(piece_bbs, occupancy_bbs, np.uint64(0))  # WHITE
   pinned_black = get_pinned_pieces(piece_bbs, occupancy_bbs, np.uint64(1))  # BLACK
   ```
   這些值用於計算 mobility area 和 king safety，然後被包含在 `evaluate_attacks_mobility_threats` 的回傳值中（evaluation.py:1216-1218），但 `_evaluate_position_jit` 最終只回傳 `np.int32(final_score)`，**丟棄了 pinned pieces 資訊**。

2. **第二次**：在 `_search` 的主迴圈初始化時（search.py:934-935）
   ```python
   pinned_white = get_pinned_pieces(piece_bbs, occupancy_bbs, WHITE)
   pinned_black = get_pinned_pieces(piece_bbs, occupancy_bbs, BLACK)
   ```
   用於 move ordering 中的 `score_captures`、`score_quiets` 和 SEE pruning。

### `get_pinned_pieces` 的成本

此函數需要遍歷敵方的所有 sliding pieces（bishop/rook/queen），對每個進行 magic bitboard lookup 來判斷是否穿過己方棋子 pin 向己方國王。這是一個 O(n_sliders) 的計算，涉及大量的 `SQUARES_BETWEEN` 查表和 bitwise 操作。

### 延遲評估的複雜性

`_search` 中存在延遲評估機制：初始評估可能只是 `lazy=True`（僅材質+PST，不計算 pinned），只有在剪枝條件觸發時才升級到完整評估。具體流程如下：

| 步驟 | 程式碼位置 | 說明 |
|------|-----------|------|
| 初始 lazy eval | search.py:612 | `_evaluate_position_jit(..., True)` — TT miss 時用 lazy |
| TT static_eval | search.py:609-610 | TT hit 時直接用 TT 中存儲的 static_eval |
| RFP 升級 | search.py:691-693 | `compute_full_corrected_static_eval()` |
| NMP 升級 | search.py:706-708 | `compute_full_corrected_static_eval()` |
| Razoring 升級 | search.py:898-900 | `compute_full_corrected_static_eval()` |
| Futility 升級 | search.py:1079-1082 | `compute_full_corrected_static_eval()` |

**核心洞察**：只有當 `static_eval_is_full == True`（即 `compute_full_corrected_static_eval` 被呼叫過）時，pinned pieces 才已經在 evaluation 內部被計算過。在這種情況下，我們可以直接復用，而非在 line 934-935 重新計算。

---

## 修改方案

### Step 1：修改 `_evaluate_position_jit` 回傳類型

**檔案**：`evaluation.py`

#### 1a. 修改函數簽名和回傳值（line 1339-1340）

將回傳類型從 `np.int32` 改為 `(np.int32, np.uint64, np.uint64)`，分別是 `(score, pinned_white, pinned_black)`。

```python
# 修改前
@numba.njit(cache=True, boundscheck=False, fastmath=True)
def _evaluate_position_jit(piece_bbs, occupancy_bbs, game_state, lazy: bool):
    ...
    return np.int32(final_score)

# 修改後
@numba.njit(cache=True, boundscheck=False, fastmath=True)
def _evaluate_position_jit(piece_bbs, occupancy_bbs, game_state, lazy: bool):
    ...
    return (np.int32(final_score), pinned_white, pinned_black)
```

#### 1b. 所有 return 路徑都必須修改

需要修改以下 return 路徑：

| 路徑 | 位置 | pinned 值 |
|------|------|-----------|
| King-pawn endgame | line 1362 | `(score, np.uint64(0), np.uint64(0))` — 王兵殘局無 sliding pieces |
| Lazy eval | line 1396-1397 | `(score, np.uint64(0), np.uint64(0))` — lazy 不計算 pinned |
| Full eval (白方) | line 1478-1479 | `(score, pinned_white, pinned_black)` |
| Full eval (黑方) | line 1480-1481 | `(score, pinned_white, pinned_black)` |

> [!WARNING]
> `pinned_white` 和 `pinned_black` 來自 `evaluate_attacks_mobility_threats` 的回傳值（line 1400-1403），它們已經存在於 full eval 路徑的局部變數中，只需穿透回傳即可。

#### 1c. 修改 `evaluate_position` wrapper（line 1483-1488）

同步修改回傳類型。

---

### Step 2：修改 `compute_full_corrected_static_eval` 回傳類型

**檔案**：`search.py`

#### 2a. 修改回傳類型定義（line 101）

```python
# 修改前
full_static_eval_return_type = numba.types.Tuple((numba.int32, numba.int32, numba.boolean))

# 修改後
full_static_eval_return_type = numba.types.Tuple((numba.int32, numba.int32, numba.boolean, numba.uint64, numba.uint64))
```

#### 2b. 修改 `compute_full_corrected_static_eval` 函數體（line 103-113）

```python
# 修改前
def compute_full_corrected_static_eval(piece_bbs, occupancy_bbs, game_state, search_context, ply):
    raw_eval = _evaluate_position_jit(piece_bbs, occupancy_bbs, game_state, False)
    corrected_eval = apply_correction_history_score(game_state, search_context, raw_eval)
    search_context.static_eval_stack[ply] = corrected_eval
    improving = False
    if ply >= 2 and corrected_eval > search_context.static_eval_stack[ply - 2]:
        improving = True
    return raw_eval, corrected_eval, improving

# 修改後
def compute_full_corrected_static_eval(piece_bbs, occupancy_bbs, game_state, search_context, ply):
    raw_eval, pinned_white, pinned_black = _evaluate_position_jit(piece_bbs, occupancy_bbs, game_state, False)
    corrected_eval = apply_correction_history_score(game_state, search_context, raw_eval)
    search_context.static_eval_stack[ply] = corrected_eval
    improving = False
    if ply >= 2 and corrected_eval > search_context.static_eval_stack[ply - 2]:
        improving = True
    return raw_eval, corrected_eval, improving, pinned_white, pinned_black
```

---

### Step 3：修改 `_search` 中所有呼叫點

**檔案**：`search.py`

#### 3a. 增加 cached pinned 變數

在 `static_eval_is_full` 附近（line 606）增加：

```python
static_eval_is_full = False
cached_pinned_white = np.uint64(0)
cached_pinned_black = np.uint64(0)
```

#### 3b. 修改所有 `compute_full_corrected_static_eval` 呼叫點（共 4 處）

每處都需要接收 pinned pieces：

| 位置 | 用途 |
|------|------|
| line 692 | RFP 升級 |
| line 707 | NMP 升級 |
| line 899 | Razoring 升級 |
| line 1081 | Futility 升級 |

每處修改方式一致：

```python
# 修改前
raw_static_eval, static_score, improving = compute_full_corrected_static_eval(...)
static_eval_is_full = True

# 修改後
raw_static_eval, static_score, improving, cached_pinned_white, cached_pinned_black = compute_full_corrected_static_eval(...)
static_eval_is_full = True
```

#### 3c. 核心：修改 `_search` 中 pinned pieces 的使用（line 933-935）

```python
# 修改前
# Optimization: Calculate pinned pieces once for shallow pruning logic and score_moves
pinned_white = get_pinned_pieces(piece_bbs, occupancy_bbs, WHITE)
pinned_black = get_pinned_pieces(piece_bbs, occupancy_bbs, BLACK)

# 修改後
# Optimization: Reuse pinned pieces from full evaluation if available
if static_eval_is_full:
    pinned_white = cached_pinned_white
    pinned_black = cached_pinned_black
else:
    pinned_white = get_pinned_pieces(piece_bbs, occupancy_bbs, WHITE)
    pinned_black = get_pinned_pieces(piece_bbs, occupancy_bbs, BLACK)
```

---

### Step 4：修改 `quiescence_search` 和其他直接呼叫 `_evaluate_position_jit` 的地方

在 `quiescence_search` 中有多處直接呼叫 `_evaluate_position_jit`，需要調整為接收 tuple 回傳值：

| 位置 | 用途 | 修改方式 |
|------|------|----------|
| line 105 | QSearch 內的 raw_eval | `raw_eval, _, _ = _evaluate_position_jit(...)` |
| line 135 | QSearch TT cutoff return | 解構取第一個元素 |
| line 168 | QSearch in-check eval | 同上 |
| line 178 | QSearch stand-pat return | 同上 |
| line 180 | QSearch stand-pat eval | 同上 |
| line 438 | _search singular exclusion | 同上 |
| line 612 | _search lazy eval | 同上（lazy 回傳 `(score, 0, 0)`） |

> [!IMPORTANT]
> 所有呼叫 `_evaluate_position_jit` 的地方，如果不需要 pinned pieces，使用 `score, _, _ = _evaluate_position_jit(...)` 解構即可。Numba 支持 tuple unpacking。

---

### Step 5：ProbCut 中的 pinned pieces（可選優化）

ProbCut（search.py:799-800）中也有獨立的 `get_pinned_pieces` 呼叫。如果 `static_eval_is_full == True`，也可以復用 `cached_pinned_white` 和 `cached_pinned_black`：

```python
# 修改前
pc_pinned_w = get_pinned_pieces(piece_bbs, occupancy_bbs, WHITE)
pc_pinned_b = get_pinned_pieces(piece_bbs, occupancy_bbs, BLACK)

# 修改後
if static_eval_is_full:
    pc_pinned_w = cached_pinned_white
    pc_pinned_b = cached_pinned_black
else:
    pc_pinned_w = get_pinned_pieces(piece_bbs, occupancy_bbs, WHITE)
    pc_pinned_b = get_pinned_pieces(piece_bbs, occupancy_bbs, BLACK)
```

---

## 修改檔案總覽

| 檔案 | 操作 | 影響範圍 |
|------|------|----------|
| `evaluation.py` | 修改 `_evaluate_position_jit` 回傳類型 | ~6 處 return 語句 |
| `search.py` | 修改 `compute_full_corrected_static_eval` + 所有呼叫點 | ~12 處 |

---

## Numba JIT 注意事項

> [!CAUTION]
> 1. **清除 `__pycache__`**：回傳類型改變會導致 JIT cache 失效，首次編譯前必須手動刪除所有 `__pycache__` 目錄。
> 2. **Tuple 回傳類型一致性**：Numba 要求所有 return 路徑回傳完全相同的 tuple 類型。`lazy=True` 和 king-pawn endgame 路徑必須回傳 `(np.int32(score), np.uint64(0), np.uint64(0))`，不可省略。
> 3. **顯式簽名**：`_evaluate_position_jit` 使用 `@numba.njit(cache=True, ...)` 無顯式簽名，**不需要額外修改裝飾器**。但 `compute_full_corrected_static_eval` 的裝飾器（line 103）有顯式 return type `full_static_eval_return_type`，**必須同步修改**。

---

## 預期效果

- **NPS 提升**：在所有 `static_eval_is_full == True` 的搜尋節點中，完全消除 2 次 `get_pinned_pieces` 呼叫。根據搜尋中 RFP/NMP/Razoring 的觸發比例，預估有 **50-70% 的主搜尋節點**可以受益。
- **搜尋樹影響**：零。pinned pieces 的計算結果完全等價。

---

## 驗證計畫

1. **編譯驗證**：清除 `__pycache__` 後執行 `python -m tests.test_search`，確認 JIT 編譯成功。
2. **正確性驗證**：執行 `python -m tests.test_zobrist`，確認 10/10 通過。
3. **答對率驗證**：由使用者執行 `python -m tests.test_puzzle`，比對失敗題數與 NPS 數據。
4. **NPS 基準對比**：與 Phase 1 的數據（平均 NPS 約 784K）進行比較。
