# Classical HCE 與 Stockfish 11 — 對齊審核報告

**日期：** 2026-07-11（階段 A+B+B3+B4；B4 對戰 **200 局** 定稿）  
**範圍：** `chess_engine/classical` 手工評估（HCE） vs `stockfish_11`  
**基準分支狀態：**

| 目錄 | 內容 |
| :--- | :--- |
| `chess_engine/classical` | **A+B+B3+B4**（含移除通路兵材質 scale；現行 New） |
| `chess_engine/classical_old` | **A+B+B3**（B4 前 sync；仍含通路兵材質 scale） |

**結論摘要：** 核心結構與 SF11 **高度對齊**。本文為 HCE 對齊**唯一主文件**（舊 `EVALUATION_ANALYSIS` / `PAWN_EVALUATION_DIFFERENCES` / `RESEARCH_REPORT` 已刪除；`classical_old/` 亦不維護 markdown）。

**對戰摘要：**

| 測試 | 結果 | 文件 |
| :--- | :--- | :--- |
| 階段 A 整包 vs 更舊 Old | ~**+61 Elo** / 80 局（顯著） | `PHASE_A_TOURNAMENT_ANALYSIS.md` |
| B（尺度混用）vs A | ~**−9 Elo** / 120 局 | — |
| **B（尺度一致）vs A** | ~**+30 Elo** / **200 局**（CI 含 0） | `PHASE_B_TOURNAMENT_ANALYSIS.md` |
| **B4（去通路兵材質 scale）vs B3** | ~**+9 Elo** / **200 局**（51.25%，CI 含 0，**打平**） | `PHASE_B4_TOURNAMENT_ANALYSIS.md` |

---

## 1. 一句話總結

Classical HCE 是 **以 Stockfish 11 為藍本、維持 ~100cp 人類可讀尺度的 tapered 評估**。  
階段 A 修正確性與專用殘局；階段 B 補上 **npm phase** 與 **SF11 kingDanger 乘積模型**。  
**不是**與 SF11 內部單位（128/781/…）的數值位元對等移植。

---

## 2. 整體對齊總表（現況）

| 組件 | 狀態 | 說明 |
| :--- | :---: | :--- |
| 材質值 + PSQT | 部分 | 刻意 `100/320/330/500/900`；PST 非 SF11 原表 |
| 材質不平衡多項式 | ✅ 對齊（縮放） | `material.py` QuadraticOurs/Theirs + `//16` |
| 局面階段 (phase) | ✅ 對齊（尺度適配） | npm taper：`MIDGAME_LIMIT=6400`, `ENDGAME_LIMIT=1641`, phase∈[0,128] |
| 機動性區域 | ✅ | K/Q、牽制、低位/受阻兵、敵兵攻擊 |
| 機動性表 | ✅ 縮放 | 非線性查表 |
| 棋子特徵 (N/B/R/Q) | ✅ 大多 | 見 §3；CorneredBishop 僅 960 可忽略 |
| 威脅 | ✅ | ThreatByMinor=`defended\|weak`；SafePawn/PawnPush/Restricted 等 |
| 國王安全 | ✅ 結構對齊 | SF11 `count×weight` + 安全將軍原係數 + 一次 100cp 縮放 |
| 兵型結構 | ✅ 強 | stopper/lever/phalanx/connected 等 |
| 通路兵 | ✅ | 路徑/王鄰近/候選半價；**已移除**非 SF 的材質 `PASSED_SCALE_*` |
| 空間 | ✅ 啟用 | SF 公式 + 刻意 `// SPACE_SCALE_DIVISOR(4)` |
| 主動權 (initiative) | ✅ | 雙套 MG/EG 權重（SF 單一 complexity） |
| 殘局縮放因子 | ✅ 強 | `get_endgame_scale_factor` 多種堡壘 |
| 專用殘局**評估** | ✅ 目標五種 | **KXK / KPK bitbase / KRKP / KQKP / KBNK** |
| Tempo / Lazy | ✅ | 門檻已縮放 |

---

## 3. 已對齊 / 已完成項目清單

### 3.1 評估結構（與 SF11 同骨架）

| # | 項目 | 實作位置 |
| :-: | :--- | :--- |
| 1 | Imbalance 二次多項式 | `material.py` |
| 2 | 機動性區域 | `evaluation.py` → `evaluate_attacks_mobility_threats` |
| 3 | 威脅全套（含 SafePawn、PawnPush 安全落地、Restricted、Hanging、Rook/King、N/SliderOnQueen） | `evaluation.py` |
| 4 | ThreatByMinor = `(defended \| weak)` | 階段 A |
| 5 | 兵型 A/B/C 通路判定 + connected/isolated/backward/weak lever | `pawns.py` |
| 6 | 通路兵動態路徑 / 王鄰近 / 候選半價 | `evaluation.py` |
| 7 | 兵盾+兵風暴（含易位虛擬化）快取 | `pawns.py` |
| 8 | npm game phase | 階段 B：`constants.py` + `evaluation.py` |
| 9 | kingDanger 乘積模型 | 階段 B：見 §11 |
| 10 | 空間評估（有呼叫） | `evaluate_space` |
| 11 | Initiative 複雜度修正 | `_compute_initiative` |
| 12 | 殘局 scale + rule50 衰減 | `endgame.py` |
| 13 | 專用殘局五種短路 | `evaluate_special_endgame` |
| 14 | Tempo | `TEMPO_BONUS` |

### 3.2 棋子特徵

| 特徵 | 狀態 |
| :--- | :---: |
| MinorBehindPawn | ✅ |
| KingProtector | ✅（係數略輕） |
| BishopPawns + 中央阻擋 | ✅ |
| LongDiagonalBishop | ✅ |
| RookOnQueenFile | ✅ |
| RookOnFile（按車數） | ✅ 階段 A |
| TrappedRook（非半開放 + 易位縮放） | ✅ 階段 A |
| WeakQueen | ✅ |
| Outpost ranks 4–6 + ReachableOutpost（馬） | ✅ 階段 A |
| Outpost hole 加分 | ⚠️ 非 SF 額外項 |
| Rook on 7th | ⚠️ 非 SF 額外項 |
| King tropism（Manhattan） | ✅ 已移除（B3；原為死代碼） |
| CorneredBishop | ⏭ 僅 Chess960 |

---

## 4. 剩餘落差（更新後）

### 4.1 刻意保留（非 bug）

| 項目 | 說明 |
| :--- | :--- |
| 材質 / PST 尺度 | 維持 ~100cp，不導入 SF 781 馬 |
| 空間 `// 4` | 適配本引擎材質尺度 |
| Initiative 雙套權重 | MG/EG 分別縮放 0.78 / 0.56 |
| 通路兵 path 係數 | SF raw 35/20/9/+5 → 本引擎 21/12/5/+3（約 ×3/5） |
| Tempo / Lazy 門檻數值 | 已按尺度縮放，非逐位元相等 |

### 4.2 可選優化（中低優先）

| # | 項目 | 現況 | 建議 |
| :-: | :--- | :--- | :--- |
| 1 | Outpost hole | 額外 `OUTPOST_HOLE_BONUS` | 可併入 Outpost 或刪 |
| 2 | Rook on 7th | `evaluate_piece_coordination` | 可保留作 classical 特徵或刪 |
| 3 | KingProtector 係數 | 相對 SF 偏輕 | 可微調 |
| 4 | 更多專用殘局 | 無 KRKB/KRKN/KQKR/KNNK… | 僅在殘局轉換仍弱時再加 |
| 5 | PSQT 形狀 | 簡化表 | 需大規模調參才值得動 |
| 6 | KPK bitbase 與 SF 位元完全一致 | 自建 retrograde，已修 rank 編碼 | 可再對 golden FEN 抽樣 |

### 4.3 已關閉的舊「高嚴重度」項

| 原問題 | 狀態 |
| :--- | :---: |
| phase 計數式 | ✅ 階段 B → npm |
| kingDanger 非乘積 / 無兵 seed | ✅ 階段 B |
| ThreatByMinor 漏強保護子 | ✅ 階段 A |
| TrappedRook 無半開放門檻 | ✅ 階段 A |
| Outpost rank 過寬 | ✅ 階段 A |
| RookOnFile 每線一次 | ✅ 階段 A |
| 專用殘局未接入 | ✅ 階段 A（五種） |
| `evaluate_special_endgame` 死碼 | ✅ 已接主路徑 |
| `attacker_count` 未用於 danger | ✅ 已為 `count×weight` |

---

## 5. 實作路線圖狀態

### 階段 A — 正確性 + 五專用殘局（✅ 完成）

| 項目 | 檔案 | 狀態 |
| :--- | :--- | :---: |
| A1 ThreatByMinor | `evaluation.py` | ✅ |
| A2 TrappedRook 半開放門檻 | `evaluation.py` | ✅ |
| A3 Outpost ranks 4–6 | `evaluation.py` | ✅ |
| A4 RookOnFile per-rook | `pawns.py` | ✅ |
| A5 KXK / KPK / KRKP / KQKP / KBNK | `endgame.py`、`evaluation.py` | ✅ |

### 階段 B — 核心公式（✅ 完成）

| 項目 | 檔案 | 狀態 |
| :--- | :--- | :---: |
| B1 npm phase | `constants.py`、`evaluation.py` | ✅ |
| B2 kingDanger SF 結構 + **全式 100/128 尺度一致** | `constants.py`、`evaluation.py` | ✅ |
| B3 移除 eval tropism（原為死代碼） | `evaluation.py`、`constants.py` | ✅ |
| B4 移除通路兵材質 scale（對齊 SF11） | `evaluation.py`、`constants.py` | ✅ |

### 階段 C — 殘局擴充（可選）

| 項目 | 說明 | 狀態 |
| :--- | :--- | :---: |
| C1 更多 specialized EG | KRKB、KQKR… | 待定 |
| C2 KPK golden 對照 | 與 SF11 bitbase 抽樣 | 待定 |

### 階段 D — 整潔 / 驗證

| 項目 | 狀態 |
| :--- | :---: |
| 本審核報告為 HCE 唯一主文件 | ✅ |
| 刪除過時評估文檔 | ✅ `EVALUATION_ANALYSIS.md`、`PAWN_EVALUATION_DIFFERENCES.md`、`RESEARCH_REPORT.md` |
| 單元測試 | ✅ `tests/test_phase_a_eval.py`、`tests/test_endgame_conformance.py` |
| SF 殘局 oracle | ✅ `ENDGAME_SF_VERIFICATION.md` |
| 階段 A 對戰 | ✅ `PHASE_A_TOURNAMENT_ANALYSIS.md` |
| 階段 B 對戰（200 局，~+30 Elo） | ✅ `PHASE_B_TOURNAMENT_ANALYSIS.md` |
| 階段 B4 對戰（200 局，~+9 Elo 打平） | ✅ `PHASE_B4_TOURNAMENT_ANALYSIS.md` |
| 再跑至 95% 顯著（可選） | 待定（B4 已視為棋力中性） |

---

## 6. 關鍵檔案

| 路徑 | 角色 |
| :--- | :--- |
| `chess_engine/classical/evaluation.py` | 主評估、王安全、威脅、機動性、通路兵、initiative、空間 |
| `chess_engine/classical/pawns.py` | 兵型、兵盾、RookOnFile/7th |
| `chess_engine/classical/material.py` | Imbalance |
| `chess_engine/classical/endgame.py` | 專用殘局 + scale + KPK bitbase |
| `chess_engine/classical/constants.py` | 權重、phase 上下限、kingDanger 係數 |
| `chess_engine/classical/bitboard_utils.py` | 王區域預計算 |
| `tools/sync_classical_old.py` | New→Old 同步 |
| `stockfish_11/src/evaluate.cpp` | 公式參考 |
| `stockfish_11/src/material.cpp` | phase + imbalance |
| `stockfish_11/src/endgame.cpp` / `bitbase.cpp` | 專用殘局 / KPK |

---

## 7. 驗證方式（已具備 / 建議）

### 已具備

```bash
python -m unittest tests.test_phase_a_eval tests.test_endgame_conformance -v
python tools/verify_endgame_vs_sf.py           # 可選 SF 搜尋 oracle（勿當 CI）
python tools/sync_classical_old.py --diff       # 確認 old 基準
python tournament_analysis/統計數據.py          # 對戰 PGN 分析
```

### 建議（階段 B 效果）

```bash
# Old = 階段 A（classical_old），New = 階段 A+B（classical）
# 依專案 tournament 入口跑 SPRT / 固定局數後再統計
python tools/tournament.py ...
python tournament_analysis/統計數據.py
```

**預期觀察點：** 中盤攻王、多子換后後的 phase 過渡、殺王攻勢是否更果斷；搜尋 depth/nodes 應與 Old 接近（評估改動非 NPS 改動）。

---

## 8. 決策指引（更新）

| 目標 | 建議 |
| :--- | :--- |
| 量測階段 B 淨增益 | 見 `PHASE_B_TOURNAMENT_ANALYSIS.md`（A vs A+B，~+30 Elo） |
| 量測 B4 通路兵 scale 移除 | ✅ **200 局** ≈ **+9 Elo / 51.25%**（CI 含 0，視為打平） |
| 再挖 Elo | 優先更多 EG；其次 outpost hole / rook 7th |
| 維持 100cp | 繼續禁止 raw SF 材質；新係數一律「SF 結構 + 一次縮放」 |
| 文件 | 以本報告為 HCE 對齊唯一真相來源 |

---

## 9. 不在範圍

- 搜尋（以 SF18 / `stockfish_repo` 為準）
- NNUE 與 classical 分數對齊
- SF11 完整 PSQT 數值複製

---

## 10. SF11 原始碼錨點

| 主題 | 位置 |
| :--- | :--- |
| KingAttackWeights / SafeCheck / Mobility / Threats | `evaluate.cpp` ~L80–150 |
| mobilityArea / kingRing / 雙兵 mask | `evaluate.cpp` ~L211–248 |
| pieces() 前哨、困車、弱后 | `evaluate.cpp` ~L252–367 |
| king() danger | `evaluate.cpp` ~L370–474 |
| threats() | `evaluate.cpp` ~L477–560 |
| initiative / scale_factor / value() | `evaluate.cpp` ~L699–834 |
| phase + imbalance | `material.cpp` |
| 專用殘局 / KPK bitbase | `endgame.cpp`、`bitbase.cpp` |

---

## 11. 階段 A 變更紀錄（2026-07-10）

### 正確性

1. **ThreatByMinor** → `(defended | weak)`  
2. **TrappedRook** → 僅非半開放線  
3. **Outpost** → 相對 rank 4–6  
4. **RookOnFile** → 按車 `popcount`

### 專用殘局（主路徑短路，白方視角分數）

| 殘局 | 概要 |
| :--- | :--- |
| KBNK | `mate_kbnk` + VALUE_KNOWN_WIN |
| KPK | 自建 bitbase（已修 rank 編碼 `6 - rank`） |
| KRKP | SF 相對座標幾何 |
| KQKP | A/C/F/H 7 橫排例外 |
| KXK | 弱方孤王且強方 npm ≥ 車 |

多兵純王兵 → `_evaluate_king_pawn_endgame`。

### 測試

- `tests/test_endgame_conformance.py`  
- `tests/test_phase_a_eval.py`（A1–A4 + 擴充 KPK 含黑方/鏡射）  
- `ENDGAME_SF_VERIFICATION.md`（SF 搜尋 oracle）

### 對戰（階段 A 整包 vs 更舊 Old）

- 約 **+61 Elo**，95% CI 不含 0；優勢集中在 **>80 步長局**（見 `PHASE_A_TOURNAMENT_ANALYSIS.md`）。

---

## 12. 階段 B 變更紀錄（2026-07-10）

### B1 — npm game phase

```text
npm = sum of MG non-pawn material (both sides)
npm' = clamp(npm, ENDGAME_LIMIT=1641, MIDGAME_LIMIT=6400)
phase = ((npm' - 1641) * 128) / (6400 - 1641)   # ∈ [0, 128]
final = (mg * phase + eg * (128 - phase) * sf/64) / 128   # sf 在 scale≠NORMAL 時作用於 eg
```

- 檔案：`constants.py`（`PHASE_MIDGAME`、`MIDGAME_LIMIT`、`ENDGAME_LIMIT`）、`evaluation.py` 主路徑。

### B2 — kingDanger（SF11 結構 + **本引擎尺度一致**）

**尺度原則（2026-07-10 修訂）：**  
SF 係數在「pawn MG=128」空間與 SF mobility/shelter 共調。我們材質/機動性/兵盾在 **~100cp**。  
因此：

1. **所有純 SF danger 係數**先 `×100/128` 再累加（含 attack weights、safe check、weak/unsafe/blockers、no-queen、offset、knight-def、flank popcount 項）。  
2. **進 `kd` 的機動性**只用 `mg_mobility_pure`（僅 MobilityBonus 表，W−B），**不含** tropism / KingProtector / TrappedRook 等。  
3. **兵盾**用本引擎已縮放的 `mg_shield`，與上列同一空間：`−6*shield/8`。  
4. 輸出（`kd` 已在 our-scaled 空間，等價於 `kd_us ≈ (100/128)·kd_sf`）：  
   - `pen_mg = kd² · 128 / (4096 · 100)`  
   - `pen_eg = kd · 128 · 120 / (100 · 16 · 213)`  
   - `threshold ≈ 78`（`100×100/128`）

| 要素 | 實作 |
| :--- | :--- |
| 攻擊者 | `count × weight`；權重已縮放（約 63/40/34/7） |
| 兵 | 攻擊 ring 只 **+count** |
| kingRing | zone **去掉己方雙兵防守格** |
| 安全將軍 | SF 原值 ×100/128 |
| 線性項 | weak/unsafe/blockers/adj 等同縮放 |
| 機動性差 | **純** MobilityBonus（Them−Us） |
| 兵盾反饋 | 本引擎 shield 分 |
| 移除 | 材質二次縮放、牽制雙算、混用未縮放 SF 係數 + 100cp mobility |

- 檔案：`evaluation.py`、`constants.py`（見 `_sf_danger_to_ours` 與 `KING_DANGER_OUT_*`）。

### B3 — 移除 King tropism（2026-07-10）

- 發現 `white/black_piece_tropism` **只累加、從未 `+=` 進 mg/eg 總分**（死代碼）。
- 已刪除：機動性迴圈中的 Manhattan tropism 計算、回傳欄位、`KING_TROPISM_*` 常數。
- 同步：先 `sync_classical_old`（Old = A+B 含 tropism 死代碼）→ 再只改 `classical` 刪 tropism。
- **預期 Elo ≈ 0**（評估分數不變）；可略提升 NPS（少 8 次/子距離查表）。
- outpost hole 仍保留。

### B4 — 通路兵材質 scale 對齊 SF11（2026-07-10）

- SF11 `Evaluation::passed()` **沒有**依非兵子力落後把 passer bonus 打折。
- 本引擎原有 `PASSED_SCALE_FULL/HALF/MIN`（4/2/1，再 `//4`）在「非兵數落後 / 無輕子」時把通路兵與王距離 bonus 打到 50% 或 25%。
- **已移除**該縮放；候選通路兵 `//2`、路徑安全、王鄰近公式保留。
- **尺度差異保留**：path 係數仍用 21/12/5/+3（SF 35/20/9/+5 的 ~3/5，適配 100cp 材質），**不**改回 raw SF 係數。
- 同步：先 `sync_classical_old`（Old = B3）→ New 只改通路兵 scale。
- **對戰（200 局，nodes≈200k）**：New **51.25%**（56–93–51，~**+8.7 Elo**，CI **[−26.7, +44.2]** 含 0）— 見 `tournament_analysis/PHASE_B4_TOURNAMENT_ANALYSIS.md`。  
  - 中途 76 局曾見 44.7%（~−37）→ **小樣本噪音**；滿 200 局收斂至 ~51%。  
  - 影響面窄（僅「非兵落後 + 通路兵」）；棋力**中性**（略正、未顯著）；非 nodes 設定錯誤。

### 同步與對戰

1. A 結束：`sync` → Old = A。  
2. B 在 `classical` → 200 局：**+29.6 Elo** vs A（見 `PHASE_B_TOURNAMENT_ANALYSIS.md`）。  
3. B3 tropism 死代碼清理（分數不變）。  
4. B4 前 `sync` → Old = A+B+B3；New 移除通路兵材質 scale（可對打）。

---

## 13. 版本對照速查

| 版本標籤 | 內容 |
| :--- | :--- |
| `classical_old`（目前） | 階段 A+B+B3（仍含通路兵 **材質** scale） |
| `classical`（目前） | 階段 A+B+B3+B4（通路兵 **無** 材質 scale） |
| B4 對戰（**200 局**） | New **51.25%**（~**+9 Elo**，CI 含 0）→ 見 `PHASE_B4_TOURNAMENT_ANALYSIS.md` |
| 建議下一動 | **保留 B4**（結構對齊、棋力中性）；階段 C 更多 EG；可選 ≥1000 局精測 |

---

## 14. 相關文件索引（現行）

| 文件 | 用途 |
| :--- | :--- |
| **本文** `HCE_SF11_GAP_AUDIT.md` | HCE ↔ SF11 對齊主文件 |
| `ENDGAME_SF_VERIFICATION.md` | 專用殘局 vs SF 搜尋 oracle |
| `tournament_analysis/PHASE_A_TOURNAMENT_ANALYSIS.md` | 階段 A 對戰（歷史，+61 Elo） |
| `tournament_analysis/PHASE_B_TOURNAMENT_ANALYSIS.md` | 階段 B 對戰（200 局，+30 Elo） |
| `tournament_analysis/PHASE_B4_TOURNAMENT_ANALYSIS.md` | 階段 B4 通路兵 scale 移除（200 局，~+9 Elo，打平） |
| `tests/test_phase_a_eval.py` | A1–A4、KPK、Phase B smoke |
| `tests/test_endgame_conformance.py` | 五專用殘局 |

**已刪除（過時）：** `EVALUATION_ANALYSIS.md`、`PAWN_EVALUATION_DIFFERENCES.md`、`RESEARCH_REPORT.md`。

---

*本報告為 HCE 與 SF11 對齊之主要技術文件。實作與數值細節以 `stockfish_11` 原始碼與本倉庫 `classical/` 現況為準。*
