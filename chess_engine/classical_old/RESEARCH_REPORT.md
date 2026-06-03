# 經典國際象棋引擎評估函數研究報告

## 一、前言

本報告深入研究了 **Stockfish 11**（2020 年，最後一代完全依靠手工評估函數的頂級引擎）的評估架構，並與您現有的 `chess_engine` 評估函數進行逐項比較分析。Stockfish 11 的經典評估函數經過數十年的人類專家經驗 + 自動化參數調優（Texel Tuning / SPSA），代表了手工評估的巔峰。

---

## 二、Stockfish 11 評估架構概覽

Stockfish 11 的 `evaluate.cpp` 將最終評估分解為以下 **10 大模組**：

| 模組 | 說明 | 您的引擎是否實現 |
|---|---|---|
| **Material + PSQT** | 增量更新的材質 + 位置分數 | ✅ |
| **Imbalance（材質不平衡）** | 針對特定棋子組合的加減分 | ❌ |
| **Pawns（兵型結構）** | 通路兵、孤兵、雙兵、後兵等 | ✅ |
| **Pieces（棋子評估）** | 機動性、前哨、輕子在兵後、受困車等 | ⚠️ 部分 |
| **Mobility（機動性）** | 基於可到達格數的非線性查表 | ⚠️ 線性近似 |
| **King Safety（王的安全）** | 兵盾、攻擊者權重、安全將軍、kingDanger 公式 | ⚠️ 簡化版 |
| **Threats（威脅）** | 分棋子類型的細粒度威脅、懸掛子、限制子 | ⚠️ 簡化版 |
| **Passed Pawns（通路兵）** | 進階邏輯含路徑安全性、阻擋格、候選通路兵 | ⚠️ 部分 |
| **Space（空間）** | 中央 4 線控制，與棋子數量平方成正比 | ❌ |
| **Initiative（主動權）** | 基於 complexity 的動態修正，防止和棋漂移 | ⚠️ 過於簡單 |

---

## 三、逐項深入對比分析

### 3.1 材質值（Material Values）

#### Stockfish 11
Stockfish 使用增量更新的 PST 值（包含材質），並通過 `material.cpp` 專門處理材質不平衡。具體材質值嵌入 PST 中，典型估算：

| 棋子 | MG | EG |
|---|---|---|
| Pawn | ~128 | ~213 |
| Knight | ~781 | ~854 |
| Bishop | ~825 | ~915 |
| Rook | ~1276 | ~1380 |
| Queen | ~2538 | ~2682 |

#### 您的引擎
定義於 [constants.py](constants.py) 中：
```
MG: [100, 320, 330, 500, 900]
EG: [120, 310, 340, 530, 950]
```

> [!WARNING]
> **發現**：您的材質值比例基本合理（接近傳統 1-3-3-5-9 比例），但與 Stockfish 的比例存在差異。Stockfish 中象比馬約值多 5%（MG: 825 vs 781），而您的引擎中象僅多 3%（330 vs 320）。**象的殘局價值應更高**。

---

### 3.2 機動性（Mobility）

#### Stockfish 11 — 非線性查表
Stockfish 使用 **每個格數對應獨立分數** 的查表方式，分數呈非線性增長：

**騎士（9 個檔位）：**
```
MG: [-62, -53, -12,  -4,   3,  13,  22,  28,  33]
EG: [-81, -56, -30, -14,   8,  15,  23,  27,  33]
```

**象（14 個檔位）：**
```
MG: [-48, -20,  16,  26,  38,  51,  55,  63,  63,  68,  81,  81,  91,  98]
EG: [-59, -23,  -3,  13,  24,  42,  54,  57,  65,  73,  78,  86,  88,  97]
```

**車（15 個檔位）：**
```
MG: [-58, -27, -15, -10,  -5,  -2,   9,  16,  30,  29,  32,  38,  46,  48,  58]
EG: [-76, -18,  28,  55,  69,  82, 112, 118, 132, 142, 155, 165, 166, 169, 171]
```

**后（28 個檔位）：**
```
MG: [-39, -21, 3, 3, 14, 22, 28, 41, 43, 48, 56, 60, 60, 66, 67, 70, 71, 73, 79, 88, 88, 99, 102, 102, 106, 109, 113, 116]
EG: [-36, -15, 8, 18, 34, 54, 61, 73, 79, 92, 94, 104, 113, 120, 123, 126, 133, 136, 140, 143, 148, 166, 170, 175, 184, 191, 206, 212]
```

**設計理念**：
- 機動性 0-2 的棋子受到**嚴厲懲罰**（被困、受限）
- 機動性增長呈**遞減邊際效益**（前幾步有用的格最重要）
- **車和后在殘局的機動性價值遠高於中局**

#### 您的引擎 — 線性公式
實作於 [evaluation.py](evaluation.py)：
```python
mobility = (moves - BASE_MOVES) * WEIGHT
# Knight: base=4, weight=[4,2]
# Bishop: base=5, weight=[4,2]
# Rook:   base=6, weight=[3,1]
# Queen:  base=8, weight=[2,1]
```

> [!CAUTION]
> **嚴重差距**：線性模型無法捕捉機動性的非線性特徵。在 Stockfish 中：
> 1. 一個被完全封鎖的車（0 格）受到 -58/-76 的懲罰，而您的引擎只有 -18/-6
> 2. 車的殘局機動性權重巨大（EG 方向從 -76 到 +171，跨度 247），您的引擎僅 1 點/格
> 3. 沒有遞減邊際效益 — 第 14 格 and 第 2 格的增益相同

**建議**：改用非線性查表（即使簡化版也好過線性）。

---

### 3.3 國王安全（King Safety）

#### Stockfish 11 — kingDanger 公式

Stockfish 的國王安全是最複雜的模組，核心是一個 `kingDanger` 累加器：

```cpp
kingDanger = kingAttackersCount * kingAttackersWeight   // 攻擊者數 × 權重
           + 185 * popcount(kingRing & weak)            // 弱格數
           + 148 * popcount(unsafeChecks)               // 不安全將軍
           +  98 * popcount(blockers_for_king)          // 牽制子
           +  69 * kingAttacksCount                     // 攻擊格數
           +   3 * kingFlankAttack² / 8                 // 翼側攻擊
           +       mg(mobility[Them] - mobility[Us])    // 機動性差
           - 873 * !enemyQueen                          // 無后大幅減弱
           - 100 * knightDefendsKing                    // 馬防守加分
           -   6 * mg(pawnShelter) / 8                  // 兵盾計入
           -   4 * kingFlankDefense                     // 翼側防守
           +  37;                                       // 常數偏移

// 非線性轉換：
if (kingDanger > 100)
    score -= S(kingDanger² / 4096, kingDanger / 16);
```

**攻擊者權重**（KingAttackWeights）：
```
[0, 0, 81, 52, 44, 10]  // None, Pawn, Knight, Bishop, Rook, Queen
```

**安全將軍懲罰**：
```
QueenSafeCheck  = 780
RookSafeCheck   = 1080
BishopSafeCheck = 635
KnightSafeCheck = 790
```

#### 您的引擎
實作於 [evaluation.py](evaluation.py) 與 [constants.py](constants.py)：
```python
KING_SAFETY_ATTACK_UNITS = [1, 2, 2, 5, 8]  # P, N, B, R, Q
KING_SAFETY_TABLE = [min(i²/2, 1000) for i in range(100)]  # 非線性表格
# 需要 attacker_count >= 2 才開始計分
```

> [!IMPORTANT]
> **關鍵差距**：
> 1. **缺少安全將軍檢測**：Stockfish 單獨計算每種棋子的安全將軍威脅，這是王安的核心。一個車的安全將軍 = 1080 單位，幾乎等於觸發整個非線性懲罰。
> 2. **缺少弱格計算**：Stockfish 特別計算「被攻擊且不被雙重防守」的弱格（185/格）。
> 3. **缺少牽制子/針對子**：被牽制子暴露國王（98/子）。
> 4. **no-queen 折扣太小**：Stockfish 在沒有后時直接減 873 單位，幾乎取消國王危險。您的引擎通過 scaling，但效果不夠明顯。
> 5. **kingDanger 公式更精細**：包含機動性差、翼側攻擊平方、兵盾、馬防守等。

---

### 3.4 威脅評估（Threats）

#### Stockfish 11

Stockfish 的威脅系統按 **攻擊者×被攻擊者** 類型細分：

**ThreatByMinor（輕子攻擊）按被攻擊棋子類型**：
```
           None  Pawn  Knight Bishop  Rook  Queen
MG:  [  0,    6,    59,    79,    90,    79 ]
EG:  [  0,   32,    41,    56,   119,   161 ]
```

**ThreatByRook（車攻擊）按被攻擊棋子類型**：
```
           None  Pawn  Knight Bishop  Rook  Queen
MG:  [  0,    3,    38,    38,     0,    51 ]
EG:  [  0,   44,    71,    61,    38,    38 ]
```

#### 您的引擎
實作於 [constants.py](constants.py)：
```python
THREAT_SAFE_PAWN = [45, 45]         # vs SF: S(173, 94)
THREAT_MINOR_ON_MAJOR = [25, 15]    # vs SF: S(79-90, 56-161) 分棋子類型
THREAT_ROOK_ON_QUEEN = [20, 10]     # vs SF: S(51, 38)
THREAT_HANGING = [25, 15]           # vs SF: S(69, 36)
```

> [!WARNING]
> **差距分析**：
> 1. **威脅值嚴重低估**：您的 `ThreatBySafePawn` = 45，Stockfish 是 173（MG），差了近 4 倍！安全兵威脅在 Stockfish 中是最大的單項威脅。
> 2. **缺少按被攻擊棋子類型細分**：Stockfish 的 ThreatByMinor 對車是 90/119，對后是 79/161。您一律用 25/15。
> 3. **缺少重要威脅項**：ThreatByPawnPush、RestrictedPiece、ThreatByKing、KnightOnQueen、SliderOnQueen 全部缺失。
> 4. **Hanging 值偏低**：Stockfish 69/36 vs 您的 25/15。

---

### 3.5 通路兵（Passed Pawns）

#### Stockfish 11
```cpp
// PassedRank 基礎獎勵：
PassedRank[RANK_NB] = {
    S(0,0), S(10,28), S(17,33), S(15,41), S(62,72), S(168,177), S(276,260)
};

// 高階通路兵邏輯（rank > 3）：
w = 5 * rank - 13;  // 例如 rank5: w=12, rank6: w=17, rank7: w=22
bonus += S(0, (enemy_king_dist * 19/4 - friendly_king_dist * 2) * w);
// 根據阻擋格安全性給予額外獎勵 (k = 0~35)
bonus += S(k*w, k*w);
```

#### 您的引擎
實作於 [constants.py](constants.py)：
```python
PASSED_PAWN_BONUS = [
    [0,0], [0,0], [10,20], [30,50], [50,80], [80,150], [150,250], [0,0]
]
```

> [!NOTE]
> **分析**：
> 1. 您的通路兵獎勵值整體合理，但 Rank 6-7 的值與 Stockfish 差異明顯（SF: 168-276 MG vs 您: 80-150 MG）。
> 2. **缺少通路兵路徑安全性評估**：Stockfish 會檢查升變路徑是否被攻擊，阻擋格是否安全，據此給予 0-35 倍加權獎勵。
> 3. **缺少 PassedFile 修正**：Stockfish 對邊線通路兵減分（離中心越遠價值越低）。
> 4. **缺少候選通路兵識別**：需要多次推進才能成為通路兵的候選者獎勵減半。

---

### 3.6 空間評估（Space）

#### Stockfish 11
```cpp
// 只在材質充足時計算 (non_pawn_material > SpaceThreshold = 12222)
SpaceMask = CenterFiles & (Rank2-4 for White)
safe = SpaceMask & ~ownPawns & ~enemyPawnAttacks
behind = ownPawns | shift(behind) | shift²(behind)  // 己方兵後方 3 排
bonus = popcount(safe) + popcount(behind & safe & ~enemyAttacks)
weight = pieceCount - 1
score = S(bonus * weight² / 16, 0)  // 僅影響中局
```

**設計理念**：空間控制在開局/中局非常重要（鼓勵佔據中心），但在殘局無意義。bonus 與棋子數量的平方成正比——棋子越多，空間越重要。

#### 您的引擎
**完全缺失。**

> [!CAUTION]
> 空間評估是 Stockfish 的核心模組之一。在中局中，一方控制更多空間通常意著更靈活的棋子部署。這直接影響了引擎在開局和中局的棋力。

---

### 3.7 棋子特殊位置評估

#### Stockfish 11 獨有的項目

| 項目 | 值 (MG, EG) | 說明 |
|---|---|---|
| `MinorBehindPawn` | S(18, 3) | 輕子在兵後方（受保護） |
| `KingProtector` | S(7, 8) × dist | 輕子離王越遠越扣分 |
| `BishopPawns` | S(3, 7) × count | 與象同色的兵越多越扣分 |
| `LongDiagonalBishop` | S(45, 0) | 象在長斜線上能看到兩個中心格 |
| `TrappedRook` | S(52, 10) × (1 + !castling) | 被困車（國王未易位時更嚴重） |
| `WeakQueen` | S(49, 15) | 后被釘/被發現攻擊 |
| `RookOnQueenFile` | S(7, 6) | 車在后所在直線 |
| `CorneredBishop` | S(50, 50) | 角上的象被兵封（Chess960） |

#### 您的引擎中缺失的
- ❌ MinorBehindPawn
- ❌ KingProtector（輕子距離國王的懲罰）
- ❌ BishopPawns（壞象懲罰）
- ❌ LongDiagonalBishop
- ❌ TrappedRook（受困車）
- ❌ WeakQueen
- ❌ RookOnQueenFile

---

### 3.8 主動權 / 複雜度修正 (Initiative)

#### Stockfish 11
```cpp
complexity = 9 * passedCount + 11 * pawnCount + 9 * outflanking
           + 12 * infiltration + 21 * pawnsOnBothFlanks
           + 51 * noPieces - 43 * almostUnwinnable - 100;

// 將 complexity 應用為修正（防止和棋漂移）
mg_correction = sign(mg) * max(min(complexity + 50, 0), -abs(mg));
eg_correction = sign(eg) * max(complexity, -abs(eg));
```

**設計理念**：這個修正防止引擎在評估值接近 0 時產生「和棋漂移」—— 即認為稍有優勢但實際上無法取勝的局面。它考慮：通路兵數量、總兵數、國王是否突前、兵是否分佈在兩翼等因素。

#### 您的引擎
實作於 [evaluation.py](evaluation.py)：
```python
INITIATIVE_BONUS = 10  # 固定 10 cp 先手獎勵
```

> [!WARNING]
> 這個固定的 10cp 獎勵過於簡單。Stockfish 的 initiative 是一個複雜的動態修正系統，它會在殘局中根據局面的「可贏性」動態調整評估值，防止引擎在不可能取勝的情況下追求微小的優勢。

---

### 3.9 殘局比例因子 (Scale Factor)

#### Stockfish 11
```cpp
if (opposite_bishops && non_pawn_material == 2 * BishopValueMg)
    sf = 22;  // 異色象殘局大幅縮放（和棋傾向）
else
    sf = min(sf, 36 + (opposite_bishops ? 2 : 7) * pawnCount);
sf = max(0, sf - (rule50 - 12) / 4);  // 50 步規則接近時縮放
```

#### 您的引擎
**沒有殘局比例因子系統。** 雖然有 `EG_SAFETY_SCALE = 0.5`，但這只是國王安全的縮放，不是整體殘局評估的縮放。

> [!IMPORTANT]
> 異色象殘局是最著名的和棋傾向局面。Stockfish 將評估值縮放到 22/128 ≈ 17%。缺少這個功能會導致引擎過高估計很多殘局的勝率。

---

### 3.10 材質不平衡 (Imbalance)

Stockfish 有獨立的 `material.cpp` 處理材質不平衡——例如「雙象 vs 雙馬」、「車+象 vs 車+馬」等特定組合的修正。這是一個矩陣查找系統。

您的引擎僅有 `BISHOP_PAIR_BONUS = [20, 30]`。

---

## 四、優化優先級建議

根據對棋力提升的預期影響，將改進建議按優先級排列：

### 🔴 高優先級（預計 Elo 提升最大）

| # | 優化項 | 預計影響 |
|---|---|---|
| H1 | **機動性改為非線性查表** | +30~50 Elo |
| H2 | **威脅值大幅提升** | +20~30 Elo |
| H3 | **國王安全增加安全將軍檢測** | +15~25 Elo |

### 🟡 中優先級

| # | 優化項 | 預計影響 |
|---|---|---|
| M1 | **新增空間評估** | +10~20 Elo |
| M2 | **改進主動權/複雜度修正** | +10~15 Elo |
| M3 | **壞象懲罰 (BishopPawns)** | +5~10 Elo |
| M4 | **受困車懲罰 (TrappedRook)** | +5~10 Elo |
| M5 | **通路兵路徑安全性評估** | +5~10 Elo |

### 🟢 低優先級

| # | 優化項 | 預計影響 |
|---|---|---|
| L1 | 殘局比例因子（異色象等） | +3~5 Elo |
| L2 | MinorBehindPawn 獎勵 | +2~3 Elo |
| L3 | KingProtector 距離懲罰 | +2~3 Elo |
| L4 | LongDiagonalBishop | +1~2 Elo |
| L5 | 材質不平衡矩陣 | +5~10 Elo（但實現複雜度高）|
