# 搜尋演算 / 剪枝 / 歷史啟發 / 置換表 優化計畫

**目標**：聚焦在搜尋核心的四大支柱，不涉及評估函數改進。

---

## 一、歷史啟發（History Heuristics）

### ✅ 1.1 History-Based LMR 強化（已完成）
- **改動**：[get_lmr_reduction()](file:///c:/Users/ren%20cian/OneDrive/%E6%A1%8C%E9%9D%A2/chess/chess_numba/chess_numba/chess_engine_v2/chess_engine/search_heuristics.py#206-243) 中的歷史縮放從 ±1.5 提升至 ±2.5
- **新增**：深度負分走法（`< -MAX_HISTORY // 2`）額外 +1 縮減層
- **位置**：`search_heuristics.py:206`

### 🔲 1.2 History-Based 剪枝強化
**預估 Elo 增益**：+5 到 +10

當走法歷史分數極低時，直接在 move loop 中剪枝（不只縮減深度）。

```python
# 在 Shallow Pruning 區塊後，進入 LMR 前新增
if is_quiet_move and depth <= 6 and history_score < HIST_PRUNE_THRESHOLD * depth:
    pruned_moves += 1
    continue
```

**新增參數**（[constants.py](file:///c:/Users/ren%20cian/OneDrive/%E6%A1%8C%E9%9D%A2/chess/chess_numba/chess_numba/chess_engine_v2/chess_engine/constants.py)）：
```python
HIST_PRUNE_THRESHOLD = -2000  # 每深度的歷史分門檻
```

---

## 二、剪枝邏輯（Pruning Logic）

### 🔲 2.1 Multi-Cut Pruning
**預估 Elo 增益**：+5 到 +12

在縮減搜尋中，若有 M 個走法超過 beta 節點，可判斷此節點為 Cut-node 並提早返回。

```python
# 在 NMP 之後，Move Loop 之前插入
if not is_pv and depth >= MC_DEPTH and not is_currently_in_check:
    mc_count = 0
    for mc_move in probe_first_N_moves(piece_bbs, ...):
        mc_res = _search(... , depth - MC_REDUCTION, ...)
        if mc_res > beta:
            mc_count += 1
        if mc_count >= MC_CUTOFF:
            return beta
```

**新增參數**（[constants.py](file:///c:/Users/ren%20cian/OneDrive/%E6%A1%8C%E9%9D%A2/chess/chess_numba/chess_numba/chess_engine_v2/chess_engine/constants.py)）：
```python
MC_DEPTH = 6       # 啟用多重剪枝的最小深度
MC_REDUCTION = 3   # 縮減多少層做驗證
MC_CUTOFF = 2      # 需要幾個 fail-high 才能剪枝
```

### 🔲 2.2 Futility Pruning 在 -improving 時加倍
**預估 Elo 增益**：+2 到 +5

目前 Futility 在非 improving 節點已縮減 25%。可直接排除 `is_quiet_move` 且極低靜態分的位置。

---

## 三、置換表（Transposition Table）

### 🔲 3.1 TT 年齡替換策略改進
**預估 Elo 增益**：+2 到+5

目前 TT 替換沒有考慮 `generation`（搜尋代次）的老化衰減。新策略：

```python
# 替換條件改為：寧願替換舊代次的淺深度，也不替換新代次的深度
is_old = (stored_gen < current_gen - 1)
if is_old or stored_depth <= new_depth:
    # 允許替換
```

### 🔲 3.2 TT 命中率動態反映在 LMR
**預估 Elo 增益**：+2 到 +3

如果 TT 有完整的精確分（`EXACT bound`），此位置搜尋品質高，可以稍微收緊 LMR。

---

## 四、搜尋演算（Search Algorithm）

### 🔲 4.1 Aspiration Window 重啟策略
**預估 Elo 增益**：+2 到 +5

當 Aspiration 失敗（fail-low）時，重啟策略可以更積極地縮小 delta。目前使用固定 delta。

### 🔲 4.2 IID 深度選擇優化
目前 IID 在沒有 TT 走法時直接 -1 深度。可以根據深度動態縮減（深度越大縮得越少）：

```python
if tt_move == NO_MOVE:
    depth -= 1 if depth < 10 else 0
```

---

## 建議優先順序

| # | 任務 | 預估 ΔElo | 難度 |
|:--:|:--|:--:|:--:|
| 1 | ✅ History LMR 強化 | +8–15 | 完成 |
| 2 | 🔲 History-Based 剪枝 | +5–10 | 低 |
| 3 | 🔲 Multi-Cut Pruning | +5–12 | medium |
| 4 | 🔲 TT 年齡替換策略 | +2–5 | 低 |
| 5 | 🔲 Aspiration Window 優化 | +2–5 | 低 |
