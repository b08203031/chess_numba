# [evaluate_attacks_mobility_threats](file:///c:/Users/ren%20cian/OneDrive/%E6%A1%8C%E9%9D%A2/chess/chess_numba/chess_numba/chess_engine_v2/chess_engine/evaluation.py#725-1005) vs SF11 Analysis

## 概述

v2 將 mobility、tropism、threats、outposts 全部放在一個大函數 [evaluate_attacks_mobility_threats](file:///c:/Users/ren%20cian/OneDrive/%E6%A1%8C%E9%9D%A2/chess/chess_numba/chess_numba/chess_engine_v2/chess_engine/evaluation.py#725-1005) 中迴圈處理。SF11 使用物件導向的 `Evaluation<T>` class，透過 `initialize()`、`pieces()`、[king()](file:///c:/Users/ren%20cian/OneDrive/%E6%A1%8C%E9%9D%A2/chess/chess_numba/chess_numba/chess_engine_v2/chess_engine/evaluation.py#675-723)、[threats()](file:///c:/Users/ren%20cian/OneDrive/%E6%A1%8C%E9%9D%A2/chess/chess_numba/chess_numba/chess_engine_v2/chess_engine/evaluation.py#725-1005) 等模板函數分開計算，各步驟共享 `attackedBy[][]` 和 `attackedBy2[]` 的持久化狀態。

---

## 差異總表

| 面向 | v2 | SF11 |
|---|---|---|
| 結構 | 單一大函數 | Class + 多模板函數 |
| Attack 追蹤 | `white_attacks` / `black_attacks` (所有棋子) | `attackedBy[color][pieceType]` + `attackedBy2` |
| 雙重攻擊 | ❌ **未追蹤** | ✅ `attackedBy2` 用於 threats / king safety |
| Mobility Area | 簡單：`~enemy_pawn_attacks` | 複雜：排除被卡兵、王、后、釘子棋子 |
| Mobility 評分 | 線性：[(moves - base) * weight](file:///c:/Users/ren%20cian/OneDrive/%E6%A1%8C%E9%9D%A2/chess/chess_numba/chess_numba/tournament.py#18-37) | 非線性查表：`MobilityBonus[piece][count]` |
| Slider X-Ray | ❌ 無 | ✅ 象繞過己方后、車繞過同色車和后 |
| 釘子限制移動 | ❌ 無 | ✅ 被釘子棋子只能沿 LineBB 移動 |
| 威脅評估 | 4 種（Safe Pawn / Hanging / Minor on Major / Rook on Queen） | 8+ 種 (ThreatByMinor/Rook 按**被攻擊棋子類型**給分、KingThreats、PawnPush、KnightOnQueen、SliderOnQueen、RestrictedPiece、Hanging) |
| 威脅評分細粒 | 每種一個固定 bonus | 每種依對象類型不同（ThreatByMinor[PAWN/KNIGHT/ROOK/QUEEN]） |
| Tropism | 曼哈頓距離加成 | ❌ SF11 無獨立 tropism；由 king safety 的 `kingAttackersWeight` 涵蓋 |
| King Protector | ❌ 無 | ✅ `-KingProtector * distance(s, king_sq)` 對 Knight / Bishop |
| MinorBehindPawn | ❌ 無 | ✅ 輕子在己方兵後方 bonus |
| BishopPawns | ❌ 無 | ✅ 象顏色上的己方兵懲罰（含封閉中心加重） |
| LongDiagonalBishop | ❌ 無 | ✅ 象在長斜線且能看到兩個中心格 |
| RookOnQueenFile | ❌ 無 | ✅ 車和己方后在同一列 |
| TrappedRook | ❌ 無 | ✅ 車 mob ≤ 3 且在王同側懲罰 |
| WeakQueen | ❌ 無 | ✅ 后受到相對釘子或閃現攻擊懲罰 |
| ReachableOutpost | ❌ 無 | ✅ 騎士「可以到達」前哨的 bonus |
| PawnPush Threat | ❌ 無 | ✅ 兵推進後能威脅的 bonus |

---

## 詳細說明

### 1. Mobility Area（最核心差距）

**v2**:
```python
white_safe_mask = ~black_pawn_attacks
# 過濾邏輯：att & ~white_occupancy & white_safe_mask
```

**SF11**:
```cpp
// Squares occupied by blocked pawns, king, queen, pinned pieces
// or controlled by enemy pawns are excluded
mobilityArea[Us] = ~(blocked_pawns | Pieces(Us, KING, QUEEN) | pinned | enemy_pawn_attacks);
```

> [!IMPORTANT]
> SF11 的 `mobilityArea` 額外排除了**己方后、王、受釘子的棋子、被卡住的低兵**。這使得 mobility 分數能更準確衡量真實的出動自由度，而非攻擊範圍。v2 的 `safe_mask` 只排除了對方兵的控制格，過於寬鬆。

### 2. Mobility 評分方式

**v2（線性）**:
```python
mg_mobility += (moves - KNIGHT_MOBILITY_BASE_MOVES) * KNIGHT_MOBILITY_WEIGHT[0]
```

**SF11（非線性查表）**:
```cpp
mobility[Us] += MobilityBonus[Pt - 2][mob];
// e.g. Knight: {S(-62,-81), S(-53,-56), ..., S(33,33)} for 0~8 squares
```

SF11 的查表讓前幾格移動的邊際效益遠高於後面的格子（遞減報酬），更接近真實棋理。

### 3. attackedBy2（雙重攻擊追蹤）

SF11 追蹤每個格子是否被己方兩個以上的棋子攻擊：
- **威脅計算**：只有 weakly protected 的棋子（未被 attackedBy2 保護）才算受到真正威脅。
- **王安全**：king 函數中的 `weak` squares 依賴 attackedBy2。
- **SliderOnQueen**：要求攻擊者是 attackedBy2 的一部分才給分。

v2 未追蹤 `attackedBy2`，導致許多威脅判斷缺乏「是否有足夠保護」的判斷。

### 4. X-Ray 攻擊

SF11:
```cpp
b = Pt == BISHOP ? attacks_bb<BISHOP>(s, pieces() ^ pieces(QUEEN))
  : Pt ==   ROOK ? attacks_bb<ROOK>(s, pieces() ^ pieces(QUEEN) ^ pieces(Us, ROOK))
                 : attacks_from<Pt>(s);
```
象繞過己方后計算 mobility，車繞過己方其他車和后，使得 mobility / attack 的計算更精準。

### 5. 威脅評估品質

SF11 [threats()](file:///c:/Users/ren%20cian/OneDrive/%E6%A1%8C%E9%9D%A2/chess/chess_numba/chess_numba/chess_engine_v2/chess_engine/evaluation.py#725-1005) 的評分是**按被攻擊棋子類型**細化的：

```cpp
b = (defended | weak) & (attackedBy[Us][KNIGHT] | attackedBy[Us][BISHOP]);
while (b)
    score += ThreatByMinor[type_of(pos.piece_on(pop_lsb(&b)))];
// ThreatByMinor: Pawn=S(6,32), Knight=S(59,41), Rook=S(79,56), Queen=S(90,119)
```

v2 的 `THREAT_MINOR_ON_MAJOR` 只區分「輕子攻大子」，無法反映「車攻后」比「車攻車」更重要這種細微差異。

---

## 是否需要大幅重構？

> [!NOTE]
> **結論：不需要結構性大重構，但需要針對性強化**

### 建議不動的部分
- 函數整合進一個 pass 的策略在 Numba JIT 下性能較優（減少 Python/Numba 邊界呼叫次數）。
- Tropism 作為 king safety 的額外信號是合理的補充（SF11 由 kingAttackersWeight 涵蓋，但我們的 king safety 模型已有所不同）。
- 基本的 outpost、pawn shield、pawn structure 邏輯已足夠。

### 高優先級改進（高效能影響）

1. **改進 Mobility Area**：排除己方被釘子的棋子和后。這不需要重構，只需傳遞額外資訊。
2. **非線性 MobilityBonus 查表**：將線性公式替換為 per-move-count 的查表。可以在 constants.py 中定義，評估函數改成 `KNIGHT_MOBILITY_TABLE[moves]`。
3. **加入 attackedBy2 追蹤**：在現有迴圈中添加 `attacked_twice_bb |= (existing_attacks & new_attacks)`，可以改善 threats 和 king safety 的品質。

### 中優先級改進（有增益但較小）

4. **細化威脅評分**：依被攻擊棋子類型拆分 bonus（`ThreatByMinor[piece_type]`）。
5. **加入 `KingProtector`**：輕子距己方王越遠懲罰越重。
6. **加入 `MinorBehindPawn`**：輕子在兵後方 bonus。
7. **加入 `BishopPawns` 懲罰**：象顏色上的己方兵數量。

### 低優先級（可選）

8. X-Ray mobility（計算成本較高）。
9. `TrappedRook`、`WeakQueen`、`ReachableOutpost` 等精細 bonus。
