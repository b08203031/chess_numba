# 王的安全評估分析報告：Stockfish 11 vs v2 引擎

## 摘要

本報告比較 Stockfish 11（SF11）與 v2 引擎在「王的安全」（King Safety）評估函數部分的結構差異與邏輯差異，並說明是否需要大幅重構。

---

## 一、整體架構對比

| 維度 | Stockfish 11 | v2 引擎 |
|------|-------------|---------|
| 兵盾計算位置 | `pawns.cpp`，獨立的 Pawn Hash Table（緩存！） | [evaluation.py](file:///c:/Users/ren%20cian/OneDrive/%E6%A1%8C%E9%9D%A2/chess/chess_numba/chess_numba/chess_engine_v2/chess_engine/evaluation.py) 中直接計算，無緩存 |
| 攻擊者追蹤 | 在 `pieces()` 迭代棋子時即時累積到 `kingAttackersCount/Weight/Count` | 在 [_evaluate_king_attackers()](file:///c:/Users/ren%20cian/OneDrive/%E6%A1%8C%E9%9D%A2/chess/chess_numba/chess_numba/chess_engine_v2/chess_engine/evaluation.py#508-624) 中重新迭代所有棋子 |
| 危險分數公式 | quadratic: `kingDanger²/4096`（中局），`kingDanger/16`（殘局） | lookup table: `KING_SAFETY_TABLE[i] = int(i²)/2` |
| 安全格檢測（Safe Checks） | ✅ 完整：Q/R/B/N 各自計算 safe check，有獨立高權重懲罰 | ❌ 不存在 |
| 弱格（Weak Squares） | ✅ 只被己方后/王防禦，用 `attackedBy2[Us]` 雙重防禦區分 | ⚠️ 部分：計算 `undefended_in_zone`，但沒有使用 `attackedBy2` |
| 國王翼攻擊（Flank Attack） | ✅ `kingFlankAttack²/8 + kingFlankDefense` | ❌ 不存在 |
| 后缺席懲罰 | ✅ `- 873 * !pos.count<QUEEN>(Them)`（敵方沒有王后時大幅降低危險）| ❌ 不存在 |
| 機動性差值修正 | ✅ `+ mg_value(mobility[Them] - mobility[Us])` 加入 kingDanger | ❌ 不存在 |
| 釘子棋子懲罰 | ✅ `+ 98 * popcount(pos.blockers_for_king(Us))` | ❌ 不存在 |
| 己方騎士防守獎勵 | ✅ `- 100 * bool(attackedBy[Us][KNIGHT] & attackedBy[Us][KING])` | ❌ 不存在 |
| 殘局縮放 | ✅ 殘局影響 `kingDanger/16`，自然衰減 | ⚠️ 固定比例 `EG_SAFETY_SCALE = 0.5`，非自適應 |
| 無兵翼懲罰（PawnlessFlank） | ✅ 存在 | ❌ 不存在 |
| Tropism（向心性） | ❌ 不存在（SF11 用機動性和攻擊範圍隱式處理） | ✅ 存在（Manhattan distance 計算） |
| 材質縮放 | ❌ 不直接做，由王后缺席懲罰、材質不平衡隱式處理 | ✅ 顯式縮放（`SCALING_WEIGHTS`） |

---

## 二、兵盾與兵風暴（Pawn Shield & Storm）差異

### Stockfish 11 的做法
```cpp
// pawns.cpp - evaluate_shelter()
constexpr Value ShelterStrength[FILE_NB/2][RANK_NB] = { ... };  // 按 file distance + rank
constexpr Value UnblockedStorm[FILE_NB/2][RANK_NB]  = { ... };  // 按 file distance + rank

// 計算 3 個 file，對每個 file：
int ourRank   = 最前方友方兵的相對 rank（0 = 無兵）
int theirRank = 最前方敵方兵的相對 rank（0 = 無兵）
bonus += ShelterStrength[d][ourRank]            // d = file 到邊緣距離（0-3）
bonus -= UnblockedStorm[d][theirRank]           // 敵兵風暴懲罰
if (ourRank && ourRank == theirRank - 1)        // 兵盾被阻擋（被敵兵頂住）
    bonus -= BlockedStorm                        // 僅在 rank 3 時懲罰

// 考慮易位後情況（max over current + KS castling + QS castling）
shelter = max(current, after_KS, after_QS)

// 殘局：王接近自己兵的距離
return shelter - make_score(0, 16 * minPawnDist)
```

**關鍵點：**
- 使用 **4×8 的 2D 查找表**，而非幾個固定常數
- 兵盾強度依 file 到邊緣的距離與兵的 rank 而定（更精細）
- 兵風暴懲罰也用 2D 表，且會區分「被阻擋的風暴」（減輕懲罰）
- **考慮易位後的最大值**（非常重要，避免懲罰即將易位的局面）
- 兵風暴懲罰只計入中局（殘局 = 0）

### v2 引擎的做法
```python
# _evaluate_pawn_shield_for_color()
for f in range(king_file-1, king_file+2):
    if not pawns_on_file:
        score -= PAWN_SHIELD_MISSING_PENALTY  # = 15
    else:
        if pawn_rank == original_rank:  score += 10  # intact
        elif pawn_rank == one_step_rank: score += 5   # advanced
        else:                            score -= 10  # pushed
    if no_friendly_pawn:
        if no_enemy_pawn: score -= 10  # open file
        else:             score -= 5   # semi-open

# evaluate_pawn_structure()（分開的函式）
# 兵風暴：
if abs(file_idx - king_file) <= 1:
    pawn_storm -= PAWN_STORM_PENALTY_BY_RANK[rank]  # rank-based array
```

**問題：**
1. 兵盾強度用固定常數，不考慮 file 距離邊緣（A/H file 比 E/F file 危險）
2. 兵盾在 [evaluate_pawn_structure](file:///c:/Users/ren%20cian/OneDrive/%E6%A1%8C%E9%9D%A2/chess/chess_numba/chess_numba/chess_engine_v2/chess_engine/evaluation.py#122-379) 與 [_evaluate_pawn_shield_for_color](file:///c:/Users/ren%20cian/OneDrive/%E6%A1%8C%E9%9D%A2/chess/chess_numba/chess_numba/chess_engine_v2/chess_engine/evaluation.py#467-507) 中分散，邏輯重複（open/semi-open file 在兵盾函式裡算了一次，tropism 影響又分開算）
3. 沒有考慮易位後的最大值
4. 兵風暴懲罰的方向性計算不完整：SF11 區分「阻擋風暴」和「未阻擋風暴」，v2 只靠 rank 懲罰

---

## 三、攻擊者計算（King Attacker）差異

### Stockfish 11 的做法

在 `pieces()` 迭代每個棋子時即時更新：
```cpp
if (b & kingRing[Them])
{
    kingAttackersCount[Us]++;
    kingAttackersWeight[Us] += KingAttackWeights[Pt];  // N=81, B=52, R=44, Q=10
    kingAttacksCount[Us] += popcount(b & attackedBy[Them][KING]);  // 直接攻擊王格數
}
```
然後在 [king()](file:///c:/Users/ren%20cian/OneDrive/%E6%A1%8C%E9%9D%A2/chess/chess_numba/chess_numba/chess_engine_v2/chess_engine/evaluation.py#625-675) 中組合：
```cpp
kingDanger += kingAttackersCount[Them] * kingAttackersWeight[Them]
            + 185 * popcount(kingRing[Us] & weak)  // 王圈內的弱格數
            + 148 * popcount(unsafeChecks)          // 不安全的格子（潛在將軍）
            + 98  * popcount(blockers_for_king)      // 釘子棋子數
            + 69  * kingAttacksCount[Them]           // 直接攻擊王周圍格數
            + 3   * kingFlankAttack² / 8             // 翼攻擊
            + mg_value(mobility[Them] - mobility[Us]) // 機動性
            - 873 * !enemy_queen                      // 無后大幅減少危險
            - 100 * friendly_knight_near_king         // 己方騎士護王
            - 6   * mg_value(pawn_score_contribution)
            - 4   * kingFlankDefense + 37
```

> [!IMPORTANT]
> SF11 的 `KingAttackWeights` 順序是 **N=81 > K=0 > B=52 > R=44 > Q=10**！
> 騎士的重量最高，因為騎士可以跳過阻擋！這與直覺相反，是精心調校的結果。
> v2 的 `KING_SAFETY_ATTACK_UNITS = [P=1, N=2, B=2, R=5, Q=8]`——后和車更重，騎士更輕，這與 SF11 相反。

### v2 引擎的做法

```python
# _evaluate_king_attackers()
# 先做快速跳過：
if not (enemy_attacks_bb & king_zone): return 0

# 然後遍歷所有敵方棋子：
for each N/B/R/Q:
    attacks_in_zone = piece_attacks & king_zone
    if attacks_in_zone:
        total_attack_units += KING_SAFETY_ATTACK_UNITS[piece_type]
        attacker_count += 1
        # 弱格（undefended）額外加 KING_SAFETY_WEAK_UNITS
        weak_count = count_bits(attacks_in_zone & ~friendly_attacks_without_king)
        total_attack_units += weak_count * KING_SAFETY_WEAK_UNITS[piece_type]

if attacker_count < 2: return 0  # 少於2個攻擊者無懲罰

return -KING_SAFETY_TABLE[min(total_attack_units, 99)]
```

**問題：**
1. 攻擊者數量的閾值要求（`< 2`）比 SF11 更嚴格——SF11 沒有硬性的 2人閾值，而是線性加乘讓少量攻擊者自然得分低
2. `KING_SAFETY_TABLE` 的尺度：`i²/2`，v2 索引 20 → 200 分；但 SF11 的 `kingDanger²/4096`，危險值 100+ 才開始扣分，危險值 200 → 約 10 中局分，危險值 400 → 約 39 分。兩者尺度完全不同
3. v2 缺少 Safe Check 機制——SF11 的 `QueenSafeCheck=780, RookSafeCheck=1080` 是巨大的固定懲罰，當敵方有安全將軍時直接加到 kingDanger

---

## 四、Safe Checks（安全格將軍）——v2 完全缺失的功能

這是 SF11 王安全最重要的組成部分之一，v2 完全沒有實作：

```cpp
// SF11 evaluate.cpp - king()
// 計算「可以給將的安全格子」：
safe  = ~pos.pieces(Them);
safe &= ~attackedBy[Us][ALL_PIECES] | (weak & attackedBy2[Them]);

// 從王位置射出射線，找到可被攻擊的格子
b1 = attacks_bb<ROOK>(ksq, pieces ^ ourQueen);
b2 = attacks_bb<BISHOP>(ksq, pieces ^ ourQueen);

rookChecks   = b1 & safe & attackedBy[Them][ROOK];
queenChecks  = (b1|b2) & attackedBy[Them][QUEEN] & safe & ~rookChecks;
bishopChecks = b2 & safe & attackedBy[Them][BISHOP] & ~queenChecks;
knightChecks = king_attacks(ksq) & attackedBy[Them][KNIGHT];

if (rookChecks)   kingDanger += 1080;
if (queenChecks)  kingDanger += 780;
if (bishopChecks) kingDanger += 635;
if (knightChecks & safe) kingDanger += 790;
```

Safe check 的邏輯是：從王的位置射出射線，如果敵方棋子可以到達那個格子且那個格子是「安全的」（不被我方棋子防守），那就是一個真實的將軍威脅。這個機制能正確識別短期戰術威脅。

---

## 五、其他缺失的 SF11 特性

### 5.1 King Flank Attack
```cpp
b1 = attackedBy[Them][ALL_PIECES] & KingFlank[file_of(ksq)] & Camp;
b2 = b1 & attackedBy2[Them]; // 被攻擊兩次的格子
b3 = attackedBy[Us][ALL_PIECES] & KingFlank[file_of(ksq)] & Camp;
int kingFlankAttack  = popcount(b1) + popcount(b2);
int kingFlankDefense = popcount(b3);
kingDanger += 3 * kingFlankAttack * kingFlankAttack / 8;
score -= FlankAttacks * kingFlankAttack; // = S(8, 0)
```
這個指標衡量王側翼整體被攻擊的程度，而不僅僅是王圈。v2 沒有實作。

### 5.2 Blockers for King（釘子棋子）
SF11 額外懲罰攻擊路徑上被固定的子（`+ 98 * popcount(pos.blockers_for_king(Us))`），v2 未實作。

### 5.3 Pawnless Flank 懲罰
如果王所在的側翼沒有任何兵，SF11 使用 `PawnlessFlank = S(17, 95)` 懲罰（殘局更重）。v2 未實作。

---

## 六、v2 引擎有但 SF11 沒有的特性

| 特性 | v2 實作 | 分析 |
|------|---------|------|
| 顯式 Tropism | Manhattan distance 向心性 | SF11 用機動性隱式包含，但顯式 tropism 是合理的補充 |
| 材質縮放 | `white_safety *= black_material / max_material` | SF11 用后缺席懲罰（-873）達到相似效果，但更精確 |
| open/semi-open file on king | 在兵盾評估中計算 | SF11 在 `ShelterStrength` 表格中隱含（rank=0 對應無兵） |

---

## 七、是否需要大幅重構？

### 結論：**需要中等程度的重構，而非推倒重來**

v2 的整體框架（bitboard 架構、tapered eval、分函式計算）是正確的，核心問題是幾個**關鍵邏輯缺失**和**參數方向錯誤**，而非整體設計錯誤。

### 優先修復項目（高影響）

> [!CAUTION]
> **P0：修正 KING_SAFETY_ATTACK_UNITS 的比例關係**
> 目前 v2 的 `Q=8, R=5, N=2, B=2`，但 SF11 是 `N=81, B=52, R=44, Q=10`。
> 騎士攻擊王應該比后攻擊更危險（騎士不能被棋子阻擋）。建議調整為 `N > B > R > Q` 的相對順序。

> [!IMPORTANT]
> **P1：加入 Safe Check 懲罰**
> 這是最顯著的功能缺口。缺少 Safe Check 會導致引擎無法感知到「下一步就可以將軍」的局面。建議在 [_evaluate_king_attackers](file:///c:/Users/ren%20cian/OneDrive/%E6%A1%8C%E9%9D%A2/chess/chess_numba/chess_numba/chess_engine_v2/chess_engine/evaluation.py#508-624) 中加入 safe check 計算，參考 SF11 邏輯。

> [!IMPORTANT]
> **P2：兵盾改用 2D 查找表**
> 將 `ShelterStrength[file_distance][rank]` 方式引入，替換目前的固定常數。同時加入考慮易位後情況的最大值選擇。

### 次優先修復項目（中影響）

- **P3：加入 King Flank Attack 指標** — 翼攻擊是 SF11 王安全的重要組成部分，對評估長期壓力很有價值
- **P4：后缺席時大幅降低危險分數** — 目前 v2 靠材質縮放漸變，但 SF11 的硬切效果更強（敵方沒有后，王安全威脅降 873 單位）
- **P5：攻擊者計數閾值改為線性** — 移除 `if attacker_count < 2: return 0` 硬截斷，改為讓公式自然處理

### 可暫緩項目（低影響）

- Blockers for king 懲罰
- Pawnless flank 懲罰
- 兵盾計算移入獨立 Hash Table（性能優化，需重構更大）

---

## 八、修改建議（不需推倒重來）

```python
# 1. 修正攻擊單位比例（在 constants.py）
KING_SAFETY_ATTACK_UNITS = np.array([1, 8, 5, 4, 2], dtype=np.int32)  # P, N, B, R, Q
# 騎士最危險，后最不危險（相對而言）

# 2. 在 _evaluate_king_attackers 中加入 Safe Check：
# 從王位置射出射線，看哪些格子是 safe 的
rook_rays   = get_rook_attacks(king_sq, occupancy_without_own_queen)
bishop_rays = get_bishop_attacks(king_sq, occupancy_without_own_queen)
safe_squares = ~friendly_attacks_bb | weak_squares
if rook_rays & en_r & safe_squares:   total_attack_units += 1080 // scale
if (rook_rays | bishop_rays) & en_q & safe_squares: total_attack_units += 780 // scale
if bishop_rays & en_b & safe_squares: total_attack_units += 635 // scale

# 3. 后缺席大幅降低危險（在 _evaluate_king_attackers）：
enemy_queens = en_q
if not enemy_queens:
    total_attack_units = total_attack_units // 4  # 大幅削減

# 4. 兵盾改為 2D 表（在 constants.py）：
SHELTER_STRENGTH = np.array([
    [-6,  81,  93,  58,  39,  18,  25, 0],  # file dist=0 (center)
    [-43, 61,  35, -49, -29, -11, -63, 0],  # file dist=1
    [-10, 75,  23,  -2,  32,   3, -45, 0],  # file dist=2
    [-39,-13, -29, -52, -48, -67,-166, 0],  # file dist=3 (edge)
], dtype=np.int32)  # SHELTER_STRENGTH[file_dist_to_edge][relative_rank]
```

---

## 九、總結

v2 引擎的王安全框架**結構合理**，但在以下幾點上與 SF11 存在顯著差距：

1. **缺少 Safe Check 計算**——這是最大的功能漏洞
2. **攻擊單位比例錯誤**——騎士應比后更重
3. **兵盾精細度不足**——固定常數 vs 2D 查找表
4. **缺少 King Flank Attack**——無法評估翼攻擊壓力
5. **后缺席無硬切懲罰**——導致在殘局轉換時高估王的危險

這些問題**不需要大幅重構**，可以在現有架構內逐步修補。建議按 P0→P1→P2→P3 的優先順序進行，預計可以帶來顯著的評估準確性提升。
