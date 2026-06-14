# Stockfish 11 與 Chess Numba 兵相關評估剩餘差異分析報告 (已修訂)

本報告對比了 **Chess Numba** 經典手工評估模組（`evaluation.py`）與 **Stockfish 11** 在「兵相關評估項目」上的剩餘演算法與架構差異。

這些差異存在於之前已討論並修復的項目（如前哨排數遮罩不對稱、通路兵直線得分符號反轉、王鄰近度 Clamp 與 MinorBehindPawn 等）之外。

---

## 差異總覽 (Summary of Differences)

| 評估項目 (Component) | Stockfish 11 (SF11) 行為 | Chess Numba 行為 | 影響與分析 |
| :--- | :--- | :--- | :--- |
| **ThreatBySafePawn**<br>(安全兵威脅) | 僅有位於**安全格**（未被敵攻擊，或我方雙重守護）的己方兵，其產生的攻擊才能獲得威脅獎勵。 | 任何己方兵攻擊敵方弱棋（非兵棋）時均獲得該獎勵，不檢查攻擊兵本身是否安全。 | **過於寬鬆的威脅判定**：使兵威脅獎勵更容易被觸發，相較於 SF11 略顯粗糙。 |
| **ThreatByPawnPush**<br>(兵推威脅) | 1. 篩選落地格：要求前推落地格必須不被敵兵攻擊且處於安全狀態。<br>2. 攻擊目標格：無雙重防守限制（僅要求攻擊非兵敵子）。 | 1. 篩選落地格：未檢查前推落地格的安全性。<br>2. 攻擊目標格：限制較嚴，要求目標格不能被敵方雙重防守（`~black_attacks2`）。 | **雙向差異**：落地格篩選較寬鬆（可能產生戰術盲目兵推），但目標格篩選較嚴格（形成部分補償）。 |
| **Restricted Piece Threat**<br>(限制棋子威脅) | 當我方棋子限制住敵方棋子的移動時給予獎勵（此項目極度依賴兵型佔用與兵攻擊）。 | 在 `evaluate_attacks_mobility_threats` 中被完整註解掉（L1605–1617）。 | **缺失的評估項**：遺漏了 SF11 中用於衡量陣地壓迫與限制空間的啟發式分數。 |
| **King-Pawn Endgames**<br>(純王兵殘局) | 1. 專用評估分支：若材質匹配專用殘局（如 KPK），會完全繞過標準評估。<br>2. KPK 殘局使用精確的位元資料庫（KPK Bitbase）。對於多兵殘局則依賴一般評估。 | 1. 專用評估分支：若無非兵棋子，直接短路繞過標準評估。<br>2. 所有王兵殘局（含多兵）皆使用自定義 `_evaluate_king_pawn_endgame` 函數，配合幾何方形法則（Chebyshev 距離）進行不可阻擋通路兵的近似評估（加 800 分）。 | **架構上對等，但品質與範圍不同**：繞過標準評估的短路架構與 SF11 一致；但 Chess Numba 擴大了專用評估的適用範圍（涵蓋多兵王兵殘局），且精準度低於 SF11 的 1P KPK 位元資料庫。 |

---

## 詳細代碼與出處對比

### 1. ThreatBySafePawn (安全兵威脅)

#### 🔴 Stockfish 11 原始碼出處
SF11 首先定義了安全格 `safe`，接著過濾出站在安全格上的己方兵 `pos.pieces(Us, PAWN) & safe`，最後計算這些安全兵對敵方非兵棋子的攻擊。

* **出處**：[stockfish_11/src/evaluate.cpp:L529-535](file:///c:/Users/ren%20cian/OneDrive/桌面/chess/chess_numba/chess_numba/stockfish_11/src/evaluate.cpp#L529-L535)
```cpp
    // Protected or unattacked squares
    safe = ~attackedBy[Them][ALL_PIECES] | attackedBy[Us][ALL_PIECES];

    // Bonus for attacking enemy pieces with our relatively safe pawns
    b = pos.pieces(Us, PAWN) & safe;
    b = pawn_attacks_bb<Us>(b) & nonPawnEnemies;
    score += ThreatBySafePawn * popcount(b);
```

#### 🟡 Chess Numba 實作
Chess Numba 直接使用全體兵的攻擊 `white_pawn_attacks`，並僅檢查目標格是否「未被敵方強保護」，而沒有限制攻擊兵本身的安全性。

* **出處**：[evaluation.py:L1531-1544](file:///c:/Users/ren%20cian/OneDrive/桌面/chess/chess_numba/chess_numba/chess_engine/classical/evaluation.py#L1531-L1544)
```python
    # --- 1. ThreatBySafePawn: safe pawn attacks enemy non-pawn ---
    # White safe pawns: pawn attacks squares not strongly defended by enemy
    white_safe_pawn_sq = white_pawn_attacks & black_non_pawns & ~black_strongly_protected
    if white_safe_pawn_sq:
        count = count_bits(white_safe_pawn_sq)
        mg_threats += THREAT_SAFE_PAWN[0] * count
        eg_threats += THREAT_SAFE_PAWN[1] * count
```

---

### 2. ThreatByPawnPush (兵推威脅)

#### 🔴 Stockfish 11 原始碼出處
SF11 篩選出兵前推後的落地格 `b`，並強制要求該落地格必須不被敵兵攻擊且處於安全狀態：`b &= ~attackedBy[Them][PAWN] & safe`。接著才從該落地格計算對敵方非兵棋子的攻擊。最終攻擊目標僅使用 `& nonPawnEnemies`。

* **出處**：[stockfish_11/src/evaluate.cpp:L537-546](file:///c:/Users/ren%20cian/OneDrive/桌面/chess/chess_numba/chess_numba/stockfish_11/src/evaluate.cpp#L537-L546)
```cpp
    // Find squares where our pawns can push on the next move
    b  = shift<Up>(pos.pieces(Us, PAWN)) & ~pos.pieces();
    b |= shift<Up>(b & TRank3BB) & ~pos.pieces();

    // Keep only the squares which are relatively safe
    b &= ~attackedBy[Them][PAWN] & safe;

    // Bonus for safe pawn threats on the next move
    b = pawn_attacks_bb<Us>(b) & nonPawnEnemies;
    score += ThreatByPawnPush * popcount(b);
```

#### 🟡 Chess Numba 實作
Chess Numba 直接計算潛在前推兵的攻擊網 `w_push_attacks`。它**沒有**對前推落地格做安全性過濾，但在最終攻擊目標格上，它多加上了 `~black_attacks2` 的雙重防守限制。

* **出處**：[evaluation.py:L1618-1637](file:///c:/Users/ren%20cian/OneDrive/桌面/chess/chess_numba/chess_numba/chess_engine/classical/evaluation.py#L1618-L1637)
```python
    # --- 7. ThreatByPawnPush ---
    w_push1 = (wp_bb << np.uint64(8)) & ~all_occupancy & ~np.uint64(0xFF00000000000000)
    w_push2 = ((w_push1 & np.uint64(0xFF0000)) << np.uint64(8)) & ~all_occupancy  # double push from rank 2
    w_push_attacks = (((w_push1 | w_push2) & NOT_A_FILE) << np.uint64(7)) | (((w_push1 | w_push2) & NOT_H_FILE) << np.uint64(9))
    w_push_threat = w_push_attacks & black_non_pawns & ~black_attacks2
```

---

### 3. Restricted Piece Threat (限制棋子威脅)

#### 🔴 Stockfish 11 原始碼出處
SF11 會計算被我方多重攻擊或兵封鎖、導致移動嚴重受限的敵方棋子數量，給予額外的壓迫分數。

* **出處**：[stockfish_11/src/evaluate.cpp:L523-527](file:///c:/Users/ren%20cian/OneDrive/桌面/chess/chess_numba/chess_numba/stockfish_11/src/evaluate.cpp#L523-L527)
```cpp
    // Bonus for restricting their piece moves
    b =   attackedBy[Them][ALL_PIECES]
       & ~stronglyProtected
       &  attackedBy[Us][ALL_PIECES];

    score += RestrictedPiece * popcount(b);
```

#### 🟡 Chess Numba 實作
該項評估在 Chess Numba 中被完全註解，導致引擎缺乏對敵方受困/受限棋子的空間壓制分數評估。

* **出處**：[evaluation.py:L1605-1617](file:///c:/Users/ren%20cian/OneDrive/桌面/chess/chess_numba/chess_numba/chess_engine/classical/evaluation.py#L1605-L1617)
```python
    # Restricted piece threat (RestrictedPiece * popcount(b))
    # w_restricted = black_attacks & ~black_strongly_protected & white_attacks
    # if w_restricted:
    ...
```

---

### 4. King-Pawn Endgames (純王兵殘局)

#### 🔴 Stockfish 11 原始碼出處
當材質組合匹配為專用殘局時，SF11 會直接調用其專用評估函數，完全繞過標準評估（與 Chess Numba 的短路概念在架構上完全對等）。針對純王單兵殘局（KPK），它在 `endgame.cpp` 中提供了數學上精確的 KPK Bitbase（位元資料庫）查詢器；若是多兵王兵殘局，則仍退回常規手工評估。

* **出處**：[stockfish_11/src/evaluate.cpp:L778-779](file:///c:/Users/ren%20cian/OneDrive/桌面/chess/chess_numba/chess_numba/stockfish_11/src/evaluate.cpp#L778-L779)
```cpp
    if (me->specialized_eval_exists())
        return me->evaluate(pos);  // ← 直接返回，繞過標準評估
```
[stockfish_11/src/endgame.cpp:L172](file:///c:/Users/ren%20cian/OneDrive/桌面/chess/chess_numba/chess_numba/stockfish_11/src/endgame.cpp#L172)
```cpp
Value Endgame<KPK>::operator()(const Position& pos) const {
  // Probe KPK bitbase...
}
```

#### 🟡 Chess Numba 實作
Chess Numba 在 `_evaluate_position_jit` 的開頭，若偵測到除了雙王和雙方兵外無其他棋子，會直接短路並調用專門的 `_evaluate_king_pawn_endgame` 函數。該函數涵蓋了**所有**王兵殘局（包含多兵），並使用幾何方形法則（Chebyshev 距離）計算不可阻擋通路兵並給予大額加分。

* **出處**：[evaluation.py:L1647-1700](file:///c:/Users/ren%20cian/OneDrive/桌面/chess/chess_numba/chess_numba/chess_engine/classical/evaluation.py#L1647-L1700)
```python
def _evaluate_king_pawn_endgame(piece_bbs, side_to_move, castling_rights):
    # Unstoppable Pawn Logic (White)
    if not (WHITE_PASSED_PAWN_MASKS[sq] & black_pawns):
        ...
        king_dist = CHEBYSHEV_DISTANCE[black_king_sq, promotion_sq]
        ...
        if king_dist > adjusted_pawn_steps:
             score += 800  # Queen value approx
```
