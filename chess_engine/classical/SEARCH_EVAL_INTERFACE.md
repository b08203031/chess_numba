# 搜尋 ↔ 評估介面分析與修繕優先序

**日期：** 2026-07-11  
**範圍：** `chess_engine/classical` 搜尋如何使用 HCE 靜態分（不含 Outpost / 調參 / 再加殘局）  
**參考：** `stockfish_11/src/search.cpp`、`stockfish_repo/src/search.cpp`（SF18 系）、本倉庫 `search.py` / `evaluation.py` / `constants.py`  
**評估定稿前提：** HCE 已凍結於 **B5 + P2 + C2**（`VALUE_KNOWN_WIN = 10000` 專用殘局短路）  
**實作狀態：**  
- **PR-A（P0+P1+P2）已定稿** — RFP/NMP known-win 護欄；64 局 50%。  
- **PR-B + PR-C 已定稿** — ProbCut/Razoring 護欄；correction clamp；敏感區 forced full eval；**122 局 50%、Elo 0**。  
**基準：** HCE B5+P2+C2 + 搜尋 **PR-A+B+C**（`classical` ≡ `classical_old`，已 `sync`）。

**相關文件：**

| 文件 | 角色 |
| :--- | :--- |
| [HCE_SF11_GAP_AUDIT.md](HCE_SF11_GAP_AUDIT.md) | 評估對齊主文件 |
| [SEARCH_ANALYSIS.md](SEARCH_ANALYSIS.md) | 搜尋算法總覽（PVS / 剪枝目錄） |
| **本文** | 搜尋–評估介面專題與實作優先序 |

---

## 1. 結論摘要

### 1.1 一句話

評估在 specialized 殘局會給出 **±KNOWN_WIN（~10000）** 級靜態分；搜尋若把這些分當成「已證明的勝勢」去做 RFP / NMP / ProbCut / Razoring，會 **錯誤剪枝**。  
**PR-A～C 已實作並對戰定稿**（A：64 局 50%；B+C：122 局 50%，搜尋中性）。

### 1.2 設計取向（本系統最優解）

| 來源 | 採納 |
| :--- | :--- |
| **SF18** | 剪枝主結構：NMP / RFP / improving / verification、`is_win` / `is_loss` 語意 |
| **SF11** | 與 HCE 配套：`VALUE_KNOWN_WIN`「未證實大分不當真」 |
| **本引擎** | 保留 correction history、cut_node NMP、non-pawn / low-material guard、已調 margin |

```text
最優解 = SF18 剪枝骨架 + SF11/HCE 的 KNOWN_WIN 防護 + 不破壞現有 margin 調參
```

### 1.3 PR-A / B / C 內容與定稿結果

| 代號 | 內容 | 狀態 |
| :--- | :--- | :---: |
| **P0** | `is_win` / `is_loss` / `is_decisive` | ✅ |
| **P1** | NMP：static / null 回傳 KNOWN_WIN 護欄 | ✅ |
| **P2** | RFP：`return static` 前 KNOWN_WIN 護欄 | ✅ |
| **P3（PR-B）** | ProbCut / Razoring KNOWN_WIN 護欄 | ✅ |
| **P4（PR-C）** | correction clamp + forced full near known-win | ✅ |

| 對戰 | 結果 |
| :--- | :--- |
| **PR-A** 64 局 · New vs Old | **20–24–20，50.0%，Elo 0** |
| **PR-B+C** **122 局** · New（A+B+C）vs Old（A） | **34–54–34，50.0%，Elo 0**；CI **[−46.5, +46.5]** |
| very_long（>80） | score **0.480**（n=74；中性） |
| long（51–80） | score **0.580**（n=25） |
| 前後半場 | first 0.533 / second 0.467（無單邊崩） |
| 深度 / nodes / 耗時 | **不顯著**（depth Δ+0.13；nodes ≈0%） |
| NPS | mean **−1.4%**（Welch p≈0.027；實務可忽略） |
| 結論 | **棋力中性；防護保留；已 `sync` 凍結** |

**下一步（可選，低優先）：** P5–P7（低子力 guard / fifty-move 雙通道）；**勿**再開 P3a/b 式評估係數抬升。

---

## 2. 背景：資料流

```text
葉節點 / 靜態點
  → _evaluate_position_jit
       ├─ specialized EG（KPK / KNNK / KQKR …）→ ±KNOWN_WIN 或 0
       ├─ 多兵王兵短路
       └─ 完整 HCE + endgame scale + tempo
  → （可選）correction history
  → static_eval_stack[ply]
  → improving / RFP / NMP / razoring / ProbCut / QSearch stand-pat / TT.static_eval
```

**介面** =：何時呼叫完整 eval、分數屬於哪一「帶」、何時禁止用 eval 做 fail-high 剪枝。

### 2.1 分數帶（本引擎尺度）

| 帶 | 約略範圍 | 意義 |
| :--- | :--- | :--- |
| 普通 HCE | \|v\| ≪ 10000 | 可參與多數剪枝 |
| **KNOWN_WIN** | \|v\| ≥ 10000 | 靜態「已知優勢」（specialized EG）；**未搜尋證明** |
| **Mate 距離** | \|v\| ≥ MATE_IN_MAX_PLY | 搜尋樹上的殺棋分；可當 decisive |

```text
建議語意（對齊 SF18，映射到本引擎）:

  is_decisive(v) := abs(v) >= MATE_IN_MAX_PLY
  is_win(v)      := v >= VALUE_KNOWN_WIN      # 含 specialized
  is_loss(v)     := v <= -VALUE_KNOWN_WIN
```

---

## 3. SF11 / SF18 / 本引擎對照

### 3.1 Futility / RFP（用 static 直接 return）

| | SF11 | SF18 | 本引擎（現況） |
| :--- | :--- | :--- | :--- |
| 機制 | `eval - margin >= beta` → return eval | 同精神，margin 更動態 | Reverse Futility（RFP），`return static_score` |
| 大分防護 | **`eval < VALUE_KNOWN_WIN`** | **`!is_win(eval)`** 且 **`!is_loss(beta)`** | ✅ `not is_win(static)` + `not is_loss(beta)`（PR-A） |
| improving | 有 | 有 + opponentWorsening | 有 |

### 3.2 Null Move Pruning

| | SF11 | SF18 | 本引擎（現況） |
| :--- | :--- | :--- | :--- |
| 需 non-pawn | ✅ | ✅ `non_pawn_material(us)` | ✅ `side_non_pawn_count >= 2` |
| 節點類型 | 非 PV | **cutNode** | **cut_node**（偏 SF18） |
| beta 危險 | — | **`!is_loss(beta)`** | `abs(beta) < VALUE_KNOWN_WIN` + `not is_loss(beta)` ✅ |
| static 已是 win | — | 與 eval 門檻連動 | ✅ `not is_win(static)`（PR-A） |
| null 回傳 mate / known-win | 壓成 beta | **`!is_win(nullValue)`** | ✅ known-win / decisive 壓成 beta（PR-A） |
| Verification | nmpMinPly 窗 | 同概念 | 關 NMP 重搜 ✅ |

### 3.3 ProbCut / Razoring

| | SF18 | 本引擎（PR-B 後） |
| :--- | :--- | :--- |
| ProbCut | `!is_decisive(beta)` | mate 擋 + `abs(beta) < KNOWN_WIN` + `not is_win(static)` + `not is_loss(beta)`；回傳壓 known-win |
| Razoring | 與 static/alpha 相關 | dissonance / low_material + `not is_win(static)` + `not is_loss(alpha)` + `abs(alpha) < KNOWN_WIN` |

### 3.4 Static eval 管道

| | SF11 | SF18 | 本引擎 |
| :--- | :--- | :--- | :--- |
| Lazy eval | 評估內 early exit | NNUE 為主 | HCE lazy + 剪枝後可能 full |
| Correction | 無（舊） | `to_corrected_static_eval` + clamp | correction history ✅ |
| improving | 2/4 ply | 現代 | ply-2 + null 時 ply-4 ✅ |

### 3.5 代差說明

| 模組 | 對齊對象 |
| :--- | :--- |
| HCE 評估 | **SF11** |
| 搜尋剪枝節奏 | **SF18 系**（本倉庫註解與結構） |

因此搜尋「不像 SF11」是預期的；介面修繕應服務 **SF18 剪枝 + HCE KNOWN_WIN**，不要退回整份 SF11 `search.cpp`。

---

## 4. 本引擎現況要點（程式錨點）

### 4.1 已定稿的正確防護（PR-A+B+C）

- RFP / NMP：`is_win` / `is_loss` 護欄；null 回傳 known-win / decisive → beta（PR-A）
- NMP：`abs(beta) < VALUE_KNOWN_WIN`、`side_non_pawn_count >= NMP_MIN_SIDE_NON_PAWNS`、verification
- ProbCut / Razoring：KNOWN_WIN 入口與回傳護欄（PR-B）
- Correction：known 帶不修正 + clamp 勿跨入 ±KNOWN_WIN；敏感區 forced full（PR-C）
- low material：`total_piece_count <= LOW_MATERIAL_PRUNING_PIECE_COUNT` 影響 RFP/razor
- specialized EG 在 eval 入口短路，不經 lazy 細節

### 4.2 殘餘可選項（非阻塞）

| 項目 | 位置概念 | 優先 |
| :--- | :--- | :--- |
| fifty-move：eval scale + TT 縮放雙通道 | 長戰「想和」 | 可選（P7） |
| 低子力 guard 微調 | RFP/razor 邊界 | 可選（P5） |
| Contempt / Lazy 門檻 | 風格 / 效能 | 低（P8–P9） |

---

## 5. 修繕優先序（完整表）

### P0 — 大正分禁止「當證明」剪枝 ⭐ 最高

**目標：** 對齊 SF11 的 `VALUE_KNOWN_WIN` 與 SF18 的 `is_win` / `is_loss`。

**建議 API（`search.py` 或小 helper，Numba 友善）：**

```text
is_decisive(v) := abs(v) >= MATE_IN_MAX_PLY
is_win(v)      := v >= VALUE_KNOWN_WIN
is_loss(v)     := v <= -VALUE_KNOWN_WIN
```

**應用面：**

| 剪枝 | 條件增量 |
| :--- | :--- |
| RFP | 僅當 `not is_win(static_score) and not is_loss(beta)` |
| NMP | 另加 `not is_win(static_score)`（beta 側已有 KNOWN 近似） |
| null 回傳 | 若 `is_win(null_score) or is_decisive(null_score)` → **壓成 beta** |

**驗證：**

- KBNK / KQK：仍能找到殺（不因護欄搜不到）
- KNNK：不因舊路徑假 win 被 NMP 裁光
- 單元：靜態 eval 大分局面下 RFP/NMP 不 return 該大分當 cutoff

**風險：** 極低；殘局 nodes 可能略增。

---

### P1 — NMP 回傳語意與 SF18 對齊

**在 P0 之上明確：**

```text
if null_move_score >= beta:
    if is_win(null_move_score) or null_move_score >= MATE_IN_MAX_PLY:
        null_move_score = beta
    # verification 維持現有「關 NMP 再搜」
```

**維持不動（已優）：** cut_node、non-pawn 下限、verification depth。

**風險：** 低。

---

### P2 — RFP 護欄（與 P0 同 PR）

**現況：** `static_score - rfp_margin >= beta` → `return static_score`，無 KNOWN 檢查。

**最優解：**

```text
if RFP 觸發 and not is_win(static_score) and not is_loss(beta):
    return static_score
```

**風險：** 低。

---

### P3 — ProbCut / Razoring 護欄

| 機制 | 建議 |
| :--- | :--- |
| ProbCut | SF18：`!is_decisive(beta)`；再加 `abs(beta) < VALUE_KNOWN_WIN` 或 `not is_loss(beta)` |
| Razoring | `abs(static/alpha) < VALUE_KNOWN_WIN` 且維持 low_material / dissonance |

**建議 PR：** 第二包，接在 P0–P2 之後。

---

### P4 — Static 管道一致性

1. 剪枝前若 `|static|` 接近 KNOWN_WIN 或與 beta 極近 → **forced full eval**（NMP 已部分做；擴到 RFP）。  
2. Correction 後 **clamp** 到 `(-VALUE_KNOWN_WIN+1, VALUE_KNOWN_WIN-1)`（真 mate 不走 correction）。  
3. TT.static_eval 優先存 **full** 結果。

對齊 SF18 `to_corrected_static_eval` clamp 精神。

---

### P5 — 低子力 guard 微調

**已有：** `LOW_MATERIAL_PRUNING_PIECE_COUNT`、NMP non-pawn 數。

**可選：** 當前方 `non_pawn_count == 0` 時禁 RFP（延伸 SF「無非兵不做 NMP」）。  
非必須；P0–P2 後再視 PGN。

---

### P6 — Improving / opponentWorsening

**結論：維持。** 已達或超過 SF11，接近 SF18。勿為「像 SF11」而刪除。

---

### P7 — Fifty-move 雙通道

Eval scale 與 TT 分數縮放可能疊加。  
**長期：** 主通道放 eval + 搜尋和棋檢測；弱化 TT 二次 scale。  
**短期：** 僅當長戰「過度作和」再動。

---

### P8 — Contempt

SF 併入 eval 偏好；本引擎幾乎未用。屬風格，非正確性。預設維持。

---

### P9 — Lazy eval 門檻

`LAZY_EVAL_THRESHOLD = 1100`。  
P0–P4 的 forced full 已覆蓋危險區；**不要先調門檻**。

---

## 6. 建議實作路線圖

| 階段 | 內容 | 驗證 |
| :---: | :--- | :--- |
| **PR-A** | **P0 + P1 + P2** | ✅ 定稿；64 局 50%；曾 `sync` |
| **PR-B** | ProbCut / Razoring 護欄 | ✅ **定稿**；與 C 合併對戰 |
| **PR-C** | correction clamp + forced full near known-win | ✅ **定稿**；122 局 50% / Elo 0 |
| **可選** | P5–P7 | 有 PGN 證據再做 |

### PR-B / PR-C 實作摘要與對戰定稿（2026-07-11）

| 項 | 行為 |
| :--- | :--- |
| ProbCut 入口 | `abs(beta) < VALUE_KNOWN_WIN`、`not is_win(static)`、`not is_loss(beta)`（另保留 mate 擋） |
| ProbCut 回傳 | known-win / decisive 壓成 `beta` |
| Razoring | `not is_win(static)`、`not is_loss(alpha)`、`abs(alpha) < KNOWN_WIN`；full 後再檢 |
| Correction | raw 已 known-win/loss/decisive → 不修正；否則 clamp 勿進入 ±KNOWN_WIN 帶 |
| Forced full | 剪枝前若 lazy static 已 near known-win（\|s\| ≥ KNOWN−500 或 is_win/loss）→ full HCE |

| 對戰指標（122 局 · nodes=200k） | 數值 |
| :--- | :--- |
| W–D–L / 勝率 | **34–54–34 / 50.00%** |
| Elo / 95% CI | **0.00 / [−46.5, +46.5]**（含 0） |
| 和局率 | 44.3% |
| 深度 / nodes | 不顯著 |
| NPS | −1.4%（統計顯著、實務可忽略） |
| 結論 | **棋力中性；定稿保留；已 sync Old** |

### 6.1 為什麼第一步必須是 P0+P1+P2？

| 理由 | 說明 |
| :--- | :--- |
| 因果 | P2 殘局讓 static 更常進 KNOWN_WIN 帶；搜尋護欄是配套 |
| 共識 | SF11 與 SF18 都禁止「未證實大分當 cutoff」 |
| 範圍 | 改動集中在 `search.py` 剪枝條件，不碰 HCE 定稿 |
| 可測 | 殘局 FEN + 對戰即可驗證，無需調參 |
| 風險 | 低於再動評估係數（P3a/b 已證明係數風險） |

**不要**把 P3–P9 與 P0–P2 綁同一 PR。

---

## 7. 建議程式變更草圖（PR-A）

### 7.1 輔助謂詞

```python
# 概念碼（需符合 numba nopython）
def is_win(v):
    return v >= VALUE_KNOWN_WIN

def is_loss(v):
    return v <= -VALUE_KNOWN_WIN

def is_decisive(v):
    return abs(v) >= MATE_IN_MAX_PLY
```

### 7.2 RFP

```text
既有 RFP 條件
AND not is_win(static_score)
AND not is_loss(beta)
→ return static_score
```

### 7.3 NMP 入口

```text
既有條件
AND not is_win(static_score)   # 新增
# abs(beta) < VALUE_KNOWN_WIN 可保留或改為 not is_loss(beta)
```

### 7.4 NMP 成功回傳

```text
if null_move_score >= beta:
    if is_win(null_move_score) or is_decisive(null_move_score):
        null_move_score = beta
    … verification …
```

### 7.5 明確不改（PR-A）

- HCE 公式與 `VALUE_KNOWN_WIN` 數值  
- NMP reduction 公式、RFP_BASE_MULT  
- correction history 權重  
- classical_old（對戰前不要 sync；合併後再 sync）

---

## 8. 驗證計畫

### 8.1 輕量（必跑）

```bash
python -m unittest tests.test_endgame_conformance tests.test_phase_a_eval -v
```

可選新增：`tests/test_search_known_win_guards.py`

- 構造 static 級 KNOWN_WIN 局面，斷言 RFP 不直接 return 該分當 fail-high（若可注入）  
- 或整合：`go nodes` 於 KBNK / KNNK，方向正確

### 8.2 對戰（定稿前）

```bash
python tools/tournament.py --nodes 200000 --games 200 --name1 New --name2 Old -c 6
python tournament_analysis/統計數據.py
python tournament_analysis/parse_search_stats.py
```

| 期望 | 說明 |
| :--- | :--- |
| Elo | 中性或小正（防護性改動） |
| 搜尋 | depth/NPS 不應劇變；殘局 nodes 可略升 |
| 長局 | 不應再現 P3b 式 very_long 崩盤 |

### 8.3 文件

- 實作後更新本文「現況」欄與 [SEARCH_ANALYSIS.md](SEARCH_ANALYSIS.md) 相關小節  
- 重大結論可在 [HCE_SF11_GAP_AUDIT.md](HCE_SF11_GAP_AUDIT.md) 加一行交叉引用  

---

## 9. 與評估工作的邊界

| 做 | 不做（本專題） |
| :--- | :--- |
| 搜尋剪枝對 KNOWN_WIN / mate 的語意 | 改 Threat/Outpost/SafePawn 係數 |
| NMP/RFP/ProbCut 護欄 | 再加 specialized 殘局種類 |
| static 管道 clamp | 整包改 SF 材質尺度 |

評估定稿：**B5+P2+C2**。  
搜尋定稿：**PR-A + PR-B + PR-C**（KNOWN_WIN 全剪枝護欄 + correction clamp）。  
可選下一步：P5–P7；或棋力向 PST/搜尋強度（非再開評估係數實驗）。

---

## 10. 優先序一覽（速查）

| 順位 | 代號 | 項目 | 建議 PR | 狀態 |
| :---: | :---: | :--- | :---: | :---: |
| 1 | **P0** | is_win / is_loss / is_decisive 語意 | **A** | ✅ |
| 2 | **P1** | NMP static + null 回傳護欄 | **A** | ✅ |
| 3 | **P2** | RFP 護欄 | **A** | ✅ |
| 4 | P3 | ProbCut / Razoring | **B** | ✅ |
| 5 | P4 | Full eval / correction clamp | **C** | ✅ |
| 6 | P5 | 低子力 guard 微調 | 可選 | — |
| 7 | P6 | improving（維持） | — | — |
| 8 | P7 | Fifty-move 雙通道 | 可選 | — |
| 9 | P8–P9 | Contempt / Lazy 門檻 | 低 | — |

---

## 11. 歷史決策：「第一步是否 P0 P1 P2？」

**是；且 A→B→C 已完成並對戰定稿。**

1. **P0** 定義分數帶語意。  
2. **P1** 接到 NMP；**P2** 接到 RFP → PR-A，64 局 50%。  
3. **P3** ProbCut/Razoring + **P4** correction/forced full → PR-B+C，**122 局 50%**。  

定稿後已 `python tools/sync_classical_old.py --sync`；`--diff` 確認邏輯一致。

---

## 12. 釘子位元棋盤 Sentinel 安全性修復 (2026-10-02)

### 12.1 根因與危害
在動態懶惰評估（Dynamic Lazy Eval）中，當局面評估超過閥值（`abs(v) > LAZY_EVAL_THRESHOLD + npm // 64`）時，評估函數為節省運算提早退出，並將釘子位元棋盤回傳為 Sentinel `PINNED_UNCOMPUTED_SENTINEL = np.uint64(18446744073709551615)`（全 1）。

在 `search.py` 中，`compute_full_corrected_static_eval` 原先未檢驗釘子位元棋盤是否為 Sentinel，直接快取為 `cached_pinned_white / cached_pinned_black` 並標記 `static_eval_is_full = True`。導致：
1. 走步預走合法性判定（Pre-make Legality，`search.py:L2402-L2413`）中，`pinned_us_s` 全是 1，所有不在國王射線上的常規合法走步均被誤判為脫離釘子射線走子，直接被 `continue` 剔除，在勝勢盤面引發合法走步丟棄。
2. ProbCut 與走步排序中的 SEE 檢驗將所有棋子誤判為釘子。

### 12.2 修復方案
1. **常數統一**：在 `constants.py` 定義 `PINNED_UNCOMPUTED_SENTINEL = np.uint64(18446744073709551615)`，消除魔法數字。
2. **評估介面保證**：在 `compute_full_corrected_static_eval` 中，若評估函數因動態懶惰評估回傳 Sentinel，即時呼叫 `get_pinned_pieces(piece_bbs, occupancy_bbs, side)` 補全釘子棋盤，確保快取的釘子棋盤永遠合法且真實。
3. **搜尋端多層防護**：在 `quiescence_search`、ProbCut 及 `_search` 的走步排序前，加入 `cached_pinned_white != PINNED_UNCOMPUTED_SENTINEL` 防禦檢查，杜絕全 1 遮罩污染合法性檢驗。

---

*本文為 classical 搜尋–評估介面之技術報告。實作以 `search.py` 現況與 `stockfish_11` / `stockfish_repo` 為準；評估對齊仍以 `HCE_SF11_GAP_AUDIT.md` 為準。*

