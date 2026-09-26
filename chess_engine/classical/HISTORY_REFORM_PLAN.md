# 搜尋歷史啟發重構工程計畫書（HCE）

| 項目 | 內容 |
| :--- | :--- |
| 狀態 | **2026-07-16：Phase 2–5 結構一次落地（full history）· 待對戰校準係數** |
| 範圍 | `chess_engine/classical` 歷史表本體、更新公式、迭代衰減、fail-low 回饋、history 相依剪枝、文件 |
| 參考 | `stockfish_repo/`（結構與現代更新邏輯）、`stockfish_11/`（HCE 時代 bonus 曲線與 reverse-move） |
| 評估約束 | **評估為已調參 HCE**；搜尋係數必須按本引擎分數尺度調整，禁止直接貼 SF18 SPSA 原值當真理 |
| 基線策略 | 每階段結束後 vs `classical_old`（`main_old.py`）對戰；**顯著優勢才同步基線並進下一階段** |
| 目前程式基線 | **Phase 1a 已在 `classical/`**；**未**因 history 改革 sync `classical_old`（Old 仍為對戰對照） |
| 相關現況文件 | [`HISTORY_HEURISTICS_CN.md`](HISTORY_HEURISTICS_CN.md)、[`SEARCH_ANALYSIS.md`](SEARCH_ANALYSIS.md) |

### 0.0 決策凍結（2026-07-13）

| 決策 | 裁定 |
| :--- | :--- |
| Main History | **語意對齊 SF**（color × from-to），**保留 piece-to 為輔助**；不字面刪 piece-to、不採用 `move.raw()` main（§1.3） |
| Cont 完整版 | Phase 3 **一次做完**：`inCheck × isCapture × 5 offsets` + 分層權重 + CMHC |
| Cap / Counter | Phase 1 與 SF 索引一致 |
| Bonus/Malus | Phase 2 結構完整、係數保守 |
| Fail-low | Phase 5 結構完整、係數保守 |
| 對戰協定（Phase 1） | `python tools/tournament.py -c 4 --games 500 --nodes 300000`（見 §0.4） |

---

## 0. 目標與非目標

### 0.1 目標

1. 讓歷史表的**索引語意**與 SF 一致（可比較、可移植係數、可減少桶污染）。
2. 讓 **bonus / malus / 分表縮放 / cont 分層 / 迭代衰減 / fail-low 回饋** 成為完整、可測的閉環。
3. 在 HCE 下用**保守但完整**的係數起步，避免一次抄 NNUE 尺度造成 prune 過兇或 ordering 抖動。
4. 每階段可獨立對戰、可回滾；文件與常數表同步，杜絕「文件 956 / 程式 990」類漂移。

### 0.2 非目標（本計畫不預設做）

* 改 HCE 評估項或重新大調 material / PSQT。
* 引入多執行緒 shared history / NUMA 分片（SF 的 DynStats）。
* 把 main history 強制改成 `move.raw()` 全 16-bit 稀疏表（除非 Phase 4 實驗證明必要）。
* 在本計畫內跑完整 SPSA（僅預留「對戰通過後可進 `tune_search`」的接口常數）。

### 0.3 成功判準（每階段通用）

| 等級 | 建議判讀 | 動作 |
| :--- | :--- | :--- |
| **通過** | 固定開局庫、對等時間控制下，相對基線 **≥ +10 Elo** 且 95% 區間不嚴重跨 0；或 SPRT 接受 H1 | 同步 `classical_old`，進下一階段 |
| **中性** | 約 −5 ~ +10 Elo，區間跨 0 | 允許小修係數 1 次後重測；仍中性則 **保留結構、凍結係數**，結構債已還清可進下一階段（需人工標記） |
| **失敗** | ≤ −10 Elo 或明顯變笨 / 超時異常 | 回滾該階段；分析是結構還是係數後再拆 PR |

### 0.4 對戰協定

**Phase 1 主測（已選定）：**

```bash
python tools/tournament.py -c 4 --games 500 --nodes 300000
# NEW = main.py，OLD = main_old.py（classical_old）
```

**為何 Phase 1–4 適合 `--nodes`（而非純時間）：**

| 優點 | 說明 |
| :--- | :--- |
| 等化搜尋量 | 每步固定 30 萬節點，比較的是 **同節點下的著法品質 / ordering**，不是 NPS 或時間管理 |
| 可重現 | 歷史表改動若略微改變 NPS，時間控制會混淆 Elo |
| 排除 timeman | 本計畫不改時間管理；用 nodes 可避免把 5s≈1s 假說混進 Phase 1 |
| 與 SPRT 相容 | `tournament.py` 在 `--nodes` 下仍跑 SPRT |

**何時改用時間控制：**

* Phase 2 的 **1s vs 5s 衰減實驗**必須用 `--time`（nodes 無法驗證長 TC）。
* 最終上線前建議加一組時間控制 smoke（例如 `--time 1000` 或 `5000`）確認實戰體感。

**其他階段預設**（除非該階段另註）：與 Phase 1 相同 `-c 4 --games 500 --nodes 300000`，保持跨階段可比。

---

## 1. 設計決策：Main History 要不要「照抄」SF？

### 1.1 SF 實際在學什麼

| 引擎 | Main 表語意 | Ceiling |
| :--- | :--- | ---: |
| SF11 | `[COLOR][from_to]` Butterfly | 10692 |
| SF modern | `[COLOR][move.raw()]`（仍是 from-to 系） | 7183 |
| 本引擎現況 | `history_table[piece][to]` **+** `butterfly[from][to]`（無 color） | 16384 / 8192 |

SF 的「main history」**不是** piece-to；piece-to 主要活在 **continuation** 裡。  
你現在的 `history_table[12][64]` 更接近 SF 的 **PieceToHistory**，卻承擔了 main 的 LMR / prune 角色。

### 1.2 照抄的好處與壞處

**好處（語意對齊）：**

1. 與 SF 的 LMR `statScore ≈ 2*main + cont0 + cont1`、history prune 門檻文獻可直接類比。
2. Color 分離後，雙方 from→to 不再互相污染。
3. 去掉「同一 cutoff 全額寫 piece-to + butterfly」的雙重計算，排序尺度更乾淨。
4. 後續 fail-low / reverse-move / low-ply 都自然掛在同一 move 索引上。

**壞處 / 風險（HCE + Numba）：**

1. Piece-to 對「馬到 f5」這類跨 from 的模式樣本更集中；HCE 節點少時 piece-to 有時比 pure butterfly **更穩**。
2. 一次性刪 piece-to 可能讓 move ordering 暫時變差，Elo 波動大，難判是公式問題還是索引問題。
3. `move.raw()` 65536 槽對你的 low-ply 已在用；main 再抄一份收益有限，記憶體與 cache 壓力上升。

### 1.3 本計畫裁定（推薦，非整份照抄）

採用 **「SF 語意對齊，而非 SF 資料結構字面複製」**：

| 角色 | 本計畫目標形狀 | 說明 |
| :--- | :--- | :--- |
| **Main History** | `main_history[2][64][64]` 或 `main_history[2][4096]`（color × from_to） | 承接 SF main / butterfly 的排序與 LMR 主通道 |
| **Piece-To History** | **保留** `piece_to_history[12][64]` | 降權為輔助通道（更新 scale &lt; main），服務 HCE 稀樣本 |
| **Low-Ply** | 維持 `[5][65536]` 以 move 編碼 | 已與 SF 對齊，不改索引 |
| **禁止** | 同一 quiet 對 main 與 piece-to 寫**相同全額** bonus | 必須分 scale，避免雙計 |

**為什麼這比「整份照抄只留 butterfly」好：**

* 你仍得到 color-aware from-to 的正確語意（回答「要修改 Main」）。
* 不扔掉 HCE 下可能有用的 piece-to 信號。
* 對戰可在 Phase 4 用開關做 ablation：`piece_to_scale=0` 測試「純 SF main」是否更強；有顯著優勢再刪表。

**不建議** Phase 1 就刪 piece-to 或改 `move.raw()` main——那會把索引重構與 bonus 重標混成一團，失敗時無法歸因。

---

## 2. 記憶體與 NPS 影響（完整版可行性）

以 `int16` / `int32` 估算（單執行緒 SearchContext）：

| 表 | 現況約 | 完整目標約 | 備註 |
| :--- | ---: | ---: | :--- |
| Cont `[ply5][12][64][12][64]` | **5.6 MB** | — | 現況 |
| Cont **+ inCheck** | — | **11.2 MB** | ×2 |
| Cont **+ inCheck + isCapture** | — | **22.4 MB** | ×4；**本計畫採此完整形** |
| Capture victim 12→6 | 36 KB | 18 KB | 略減 |
| Main color×from×to | 16 KB butterfly | 32 KB | 可忽略 |
| Counter piece×to | 8 KB | 1.5 KB | 可忽略 |

**結論：** Continuation 完整版（inCheck × capture × 5 offsets）約 **+17 MB**。相對 TT（通常數百 MB 級可配）可接受；NPS 主要敏感點是 **更新時更多次 gravity 寫入與多一次維度索引**，不是表大小本身。  
Numba 下只要索引是編譯期常量維度、熱路徑無 Python 物件，**不應**因「完整版」而改簡化公式；若實測 NPS 掉 &gt;3–5%，再考慮把 cont 更新向量化或減少 malus 迴圈內的重複 prev 查找（快取 prev 五元組），**而不是**砍 inCheck/capture 維度。

---

## 3. 係數哲學（HCE 保守完整版）

### 3.1 原則

1. **結構完整對齊 SF 現代邏輯**（有哪些乘子、哪些表、哪些 ply）。
2. **數值保守於 SF18 原值**：通常取 SF 尺度的 **70–90%**，或改用 SF11 二次式再掛現代非對稱乘子。
3. 所有寫入經 **gravity**；單次 `|bonus|` clamp 到各表 `D`。
4. 改 ceiling 或 bonus 後，**同 PR 必須改** history prune / LMR history 除數，使「典型深度 8–12 的 quiet hist 分佈」落在可剪枝但不誤殺的區間。
5. 常數集中於 `constants.py`，禁止 magic number 散落（除短暫 A/B 實驗旗標）。

### 3.2 目標 Ceiling（完整版起點，可 SPSA）

| 常數 | 現況 | 目標起點 | 對齊理由 |
| :--- | ---: | ---: | :--- |
| `HISTORY_MAX_MAIN`（from-to main） | 16384（舊 piece-to） | **8192** | 近 SF main 7183，略寬 |
| `HISTORY_MAX_PIECE_TO`（輔助） | — | **12288** | 低於舊 main，避免蓋過 from-to |
| `HISTORY_MAX_BUTTERFLY` | 8192 | **併入 main 後刪或 alias** | 避免雙表同義 |
| `HISTORY_MAX_CAPTURE` | 8192 | **10692** | 對齊 SF |
| `HISTORY_MAX_CONTINUATION` | 12288 | **24576** | 近 SF 30000，HCE 略保守 |
| `HISTORY_MAX_PAWN` | 4096 | **8192** | 對齊 SF pawn D |
| `LOW_PLY_HISTORY_MAX` | 7183 | **7183** | 已對齊 |

### 3.3 Bonus / Malus 曲線（保守完整）

採用 **SF11 二次骨架 + 現代附加項**（比純 `120*depth` 更適合 HCE；比 SF18 線性 SPSA 更可解釋）：

```text
stat_bonus(d) = min(16*d*d + 96*d, 1800)          # d clamp 1..14
stat_malus(d) = min(18*d*d + 88*d, 2000)          # 略大於 bonus

# 附加（完整，係數保守）
bonus += 280 if best_move == tt_move else 0       # SF ~382 → 280
bonus += parent_stat_score // 40                  # SF /30 → 更弱
if not is_pv:
    bonus += bonus * (quiets_tried + caps_tried) // 256   # 保留 SF node-width

# 分表縮放（寫入前）
quiet_main_bonus     = bonus * 824 // 1024
quiet_piece_to_bonus = bonus * 512 // 1024        # 輔助表，保守半額級
quiet_cont_bonus     = bonus * 820 // 1024        # 再進 cont 分層
quiet_lph_bonus      = bonus * 663 // 1024
quiet_pawn_bonus     = bonus * (1038 if bonus > -7 else 525) // 1024

cap_bonus            = bonus * 1280 // 1024       # SF 1366 → 1280
cap_malus            = malus * 1400 // 1024       # SF 1518 → 1400

quiet_malus0         = malus * 1136 // 1024
quiet_malus_decay    = 956 / 1024                 # 對齊 SF；廢除 990
refute_cont_malus    = malus * 680 // 1024        # SF 683
```

Cont 分層權重（完整 SF 比例，總歸一到 1024 風格整數）：

```text
# SF weights: 1→1040, 2→780, 3→300, 4→537, 5→129, 6→423
# 本引擎 offsets: idx0=ply-1,1=ply-2,2=ply-3,3=ply-4,4=ply-6（無 ply-5 槽則把 w5 併入鄰近或略）
CONT_WEIGHT = [1040, 780, 300, 537, 423]   # 對 idx 0..4；ply-5 權重 129 省略或併入 idx4
# 更新：
delta = cont_bonus * CONT_WEIGHT[i] * cmhc_mult // 131072
delta += 71 if i < 2 else 0
# 被將：僅 i in {0,1}（前 2 層）
# cmhc_mult：正歷史一致性完整表 [96,113,101,105,127,121,126]（SF）
```

> 註：CMHC 與 `+71` 偏置屬完整版；若單階段 Elo 為負，允許只關 CMHC（乘固定 1024）做 ablation，**不**刪分層權重。

### 3.4 迭代衰減（針對「5s 不比 1s 強」）

#### 假設檢驗

「每步 5 秒不比 1 秒強 ⇒ 歷史污染」**可能為真，但更常是多因**：

| 假說 | 機制 | 如何驗證 |
| :--- | :--- | :--- |
| H1 歷史污染 | 深搜錯誤 cutoff 寫入過大 / 衰減過慢或過快不均 | 同版本： soft decay vs `÷2` vs 不衰減；比 1s/5s Elo 與平均深度 |
| H2 時間管理 | 5s 未把時間轉成有效深度（aspiration 重搜、移動不穩） | 記錄 depth、score 抖動、bestmove 變更次數 |
| H3 剪枝尺度 | 深層 history/LMR/RFP 與 HCE 方差不匹配，深了反而誤剪 | 5s 對局關閉 history prune 對照 |
| H4 報酬遞減 | HCE 本身在固定 TC 下 1s→5s Elo 曲線平緩 | 與「僅清 history 每步」的引擎對打 |
| H5 TT / 重複局面 | 長 TC 圖歷史交互 | 已有 TT 護欄；看是否 5s 非法/重複加分異常 |

**現況 `÷2` 每 iteration：**  
偏兇會讓**長 TC 的前幾 iteration 知識在深 iteration 被腰斬**，理論上會**傷害** 5s 相對 1s 的優勢，而不是幫助。若你觀察到 5s≈1s，更可能是 **H2/H3/H4**，或「污染來自單次搜尋內的錯誤寫入」而非「跨 iteration 保留」。

#### 本計畫衰減策略（完整、可開關）

```text
# 每個 iterative deepening 的新 depth 開始前（非整盤 clear）：
main[c][f][t]     = (main + 4) * 800 // 1024      # ~78% 保留（SF 789；HCE 略多留）
piece_to[p][t]    = (piece_to + 4) * 820 // 1024
capture           = (capture + 4) * 850 // 1024
pawn_hist         = (pawn_hist + 4) * 900 // 1024 # 兵型跨 iteration 極穩
cont              = cont * 850 // 1024            # 大表：避免整表 +4 的開銷可只做 *850
low_ply           = fill(100)                     # 對齊 SF：根部表每 iteration 重置偏置

# ucinewgame / 新棋局：hard clear 全表
# 單次 go 內：不 ÷2
```

Phase 2 必須帶 **A/B 旗標**（僅測試用）：

* `HIST_AGE_MODE = DIV2 | SOFT800 | NONE`
* 同一 TC 矩陣：1s / 5s × 三模式 → 寫入階段報告，再鎖定預設。

---

## 4. 分階段實作（5 個階段）

> 原則：**結構債與公式債分階段還**；每階段結束更新文件中的「階段結論」表。  
> 每階段 PR 範圍寫死，避免「順便改 RFP」污染歸因。

---

### Phase 1 — 索引語意對齊（低公式風險）

**目的：** 先讓表「裝的東西」正確，bonus 曲線暫不大改（僅修正已確認的 bug 與明顯錯誤係數）。

#### 1.A 必做

| ID | 項目 | 細節 |
| :--- | :--- | :--- |
| P1-1 | **Capture History 受害者** | `[aggressor][to][victim_type]`，`victim_type ∈ 0..5`（`type % 6`）；EP=Pawn；升變吃子記實際受害者 type。所有讀寫、tried 快取一併改。 |
| P1-2 | **Counter Move** | `counter_moves[12][64]`：索引改 `prev_piece, prev_to`；寫入/讀取/初始化全改。 |
| P1-3 | **文件與程式一致性** | quiet malus 衰減改 **956/1024**；pawn 非對稱改 **1038 / 525**（與 SF 一致，文件同步）；修 `threats_computed` / `threat_computed` 威脅重排失效 bug。 |
| P1-4 | **Check bonus** | 對齊 SF：`check_squares` 候選且 **SEE ≥ −75** 才加分（現文件寫了、實作缺）。 |
| P1-5 | **常數與型別** | `engine_types` / 所有 `np.zeros` 配置點、測試夾具、UCI 建立 context 路徑。 |
| P1-6 | **文件** | 更新 `HISTORY_HEURISTICS_CN.md` Phase1 節；本計畫書勾選。 |

#### 1.B 明確不做

* 不改 main 為 color×from_to（留 Phase 4）
* 不改 cont 維度、不改 soft decay、不大改 bonus 二次式

#### 1.C 輕測

```text
python -m tests.test_search
# 若有 history 單元測則一併；無則新增最小測：
# - capture hist 寫入 victim%6
# - counter 用 piece,to 回讀
```

#### 1.D 對戰

* 預期：小正或中性（bugfix + 索引變準）。  
* 若失敗：多半是 counter 寫入時機用錯 piece（make 前後 side）。

#### 1.E 通過後

同步基線；進入 Phase 2。

---

### Phase 2 — 衰減制度 + Bonus/Malus 本體（保守完整）

**目的：** 解決寫入曲線與跨 iteration 老化；這是「5s vs 1s」假說的主實驗場。

#### 2.A 必做

| ID | 項目 | 細節 |
| :--- | :--- | :--- |
| P2-1 | **`stat_bonus` / `stat_malus`** | §3.3 二次式 + tt 加分 + parent stat + node-width |
| P2-2 | **分表縮放** | quiet main 824、cap 1280/1400、lph 663、pawn 1038/525、refute 680、malus0 1136、decay 956 |
| P2-3 | **統一更新入口** | `update_quiet_histories(...)` / `update_all_stats(...)` 對齊 SF 呼叫圖；避免 cutoff 區塊複製貼上五份 cont |
| P2-4 | **迭代衰減** | 預設 SOFT800；保留 DIV2/NONE 旗標供對照實驗 |
| P2-5 | **Low-ply** | 每 iteration `fill(100)`；更新仍 gravity + 663 |
| P2-6 | **Reverse quiet（SF11）** | 成功 quiet 且移動子非兵：對 **反向 from-to** 寫 `-quiet_main_bonus`（僅 main 通道） |
| P2-7 | **History 相依門檻 v1** | 見 §5；與新尺度同時改 |
| P2-8 | **文件** | 公式、常數表、衰減假說實驗表 |

#### 2.B 係數保守點（刻意）

* 二次式 cap 1800/2000，低於「深層瘋狂灌分」
* tt 加分 280 而非 382
* cap malus 1400 而非 1518
* 尚未引入完整 fail-low scale 引擎（Phase 5）

#### 2.C 實驗矩陣（請使用者執行）

| 實驗 | 控制 | 觀察 |
| :--- | :--- | :--- |
| E2-a | SOFT800 vs DIV2，TC=1s | 短時是否掉 Elo |
| E2-b | SOFT800 vs DIV2，TC=5s | 長時是否上升（驗證污染假說方向） |
| E2-c | SOFT800 × (1s vs 5s) 自比 | 深度與 Elo 是否拉開 |

解讀指引：

* 若 **5s 僅在 SOFT 下明顯強於 1s**，而 DIV2 下 5s≈1s → 支持「÷2 傷害長 TC」而非「保留=污染」。
* 若 **兩種衰減下 5s 皆不強** → 優先查時間管理 / 剪枝（H2/H3），衰減改為 SOFT 仍建議保留（樣本效率）。

#### 2.D 通過後

鎖定 `HIST_AGE_MODE=SOFT800`（或實驗優勝者）；進 Phase 3。

---

### Phase 3 — Continuation History 完整版（一次做完）

**目的：** 依你的要求，**將軍分離 + capture 分離 + 分層權重 + CMHC + 被將只更前兩層**，不分工期縮水。

#### 3.A 資料結構

```text
continuation_history[
  in_check,      # 0/1  — 產生「前手之後、當前要走時」的 inCheck 狀態：採用 SF 慣例
                 #         ss->continuationHistory 綁在「剛走出的那步之後的狀態」：
                 #         continuationHistory[ss->inCheck][capture][pc][to]
  is_capture,    # 0/1
  ply_index,     # 0..4  ↔  offsets (1,2,3,4,6)
  prev_piece,    # 0..11
  prev_to,       # 0..63
  curr_piece,    # 0..11
  curr_to,       # 0..63
]
# dtype: int16；總量約 22.4 MB
```

**索引約定（寫進文件，避免實作分叉）：**

1. 與 SF 一致：在 **make_move 之後** 設定「本手留下的 cont 槽位指標」為  
   `cont[in_check_after_move 對「對手而言」?]`  
   SF 存的是：走出 move 後，`ss->continuationHistory = &continuationHistory[ss->inCheck][capture][pc][to]`，其中 `ss->inCheck` 是**走完後輪到對手時，對手是否被將**？  
   實際 SF：`ss` 是當前節點；`do_move` 後更新的是 **子節點** 的 continuation 指針，鍵為 **該步是否在 check 中走出 / 是否 capture**（父狀態 inCheck + 本步 capture）。  
2. 實作時以 `stockfish_repo/src/search.cpp` `do_move` 附近為準，在本引擎對應到 `make_move` 前後明確註解，並加單元測固定局面。

#### 3.B 更新邏輯（完整）

* `update_continuation_histories(ss, pc, to, bonus)`：
  * 權重表完整（§3.3）
  * CMHC 完整
  * `inCheck` 時僅 ply 1–2
* 所有 quiet bonus/malus、refute、fail-low（若已接）、TT quiet hit 更新，**只走此入口**
* Pawn / main 更新仍在 `update_quiet_histories`

#### 3.C 讀取邏輯（完整）

排序 quiet：

```text
score = 2 * main[color][from][to]          # Phase 4 到位前暫用現有 main 通道
      + piece_to_weight * piece_to[pc][to] # Phase 4 前維持
      + 2 * pawn_hist[...]
      + cont[ic][cap][0][prev...][pc][to]  # 各層權重在「更新」已處理，讀取端 1:1 相加
      + cont[...][1]...
      + cont[...][2]...
      + cont[...][3]...
      + cont[...][4]...                   # ply-6
      + low_ply 項
```

**重要：** Phase 3 起廢除「讀取端 cont0 ×4」這類雙重放大；改為 **更新分層、讀取近似等權**（對齊 SF modern）。LMR `statScore` 同步為：

```text
statScore = 2 * main + cont0 + cont1
```

#### 3.D NPS 護欄

* 預先快取 `prev_piece/to` 與 `in_check/capture` 旗標於 stack，避免 malus 迴圈內重複算。
* 若 NPS 掉 &gt;5%：優化快取，**不砍維度**。

#### 3.E 對戰

* 預期：中等正 Elo（ordering 在 check / tactical 後更準）。
* 失敗時 ablation：關 CMHC → 關 capture 維度（僅 debug，預設仍完整合併）。

---

### Phase 4 — Main History 語意重構（對齊 §1.3）

**目的：** 落地 color-aware main；piece-to 降為輔助；刪除語意重複的無 color butterfly 或將其 alias 到 main。

#### 4.A 必做

| ID | 項目 | 細節 |
| :--- | :--- | :--- |
| P4-1 | `main_history[2][64][64]` | color = side_to_move；索引 from,to |
| P4-2 | `piece_to_history[12][64]` | 自舊 `history_table` 更名；更新 scale 512/1024 |
| P4-3 | 移除或停用獨立 `butterfly_history` | 避免與 main 雙計；grep 全清 |
| P4-4 | Killer / Counter 分數段 | 確保仍高於 hist 最大值（現 15000 級 vs hist 8k 級需重算安全邊界） |
| P4-5 | 排序與 LMR 全改讀 main | `get_quiet_stat_score` / `get_lmr_stat_score` / prune |
| P4-6 | 門檻 v2 | §5 依新 main 尺度再標一次 |
| P4-7 | Ablation 旗標 | `PIECE_TO_HISTORY_WEIGHT ∈ {0, 512, 768}` 便於確認是否可刪表 |

#### 4.B 設計討論凍結點

* **預設保留 piece-to 輔助**；僅當 ablation 顯示 weight=0 不損 Elo 時，於後續小 PR 刪表。
* 不採用 `move.raw()` 作為 main 索引（low-ply 已覆蓋 move 級根部偏置）。

#### 4.C 對戰

* 這階段變動面大，建議局數拉高（≥400）或 SPRT。

---

### Phase 5 — Fail-low / 對手著法回饋完整版 + 門檻收斂

**目的：** 補齊 SF 在 bestMove 缺失與 prior capture 上的回饋；係數保守；並做 history 剪枝門檻最終校準。

#### 5.A Fail-low 完整結構（係數保守）

對齊 SF 分支，但縮小 scale：

```text
# 無 bestMove、前手 quiet、prevSq 有效：
bonusScale  = 0
bonusScale -= parent_stat // 120              # SF /98 → 更弱
bonusScale += min(40 * depth, 320)            # SF min(59*d,430) → 保守
bonusScale += 120 if parent_move_count > 8 else 0
bonusScale += 90 if (not in_check and best <= static_eval - 90) else 0
bonusScale += 90 if (not parent_in_check and best <= -parent_static - 70) else 0
bonusScale  = max(bonusScale, 0)

# 去掉 SF 的巨大中間量，直接：
base = min(100 * depth - 60, 1000)            # SF min(141*d-82,1472) 再打折
scaled = base * bonusScale // 512             # 整體再壓

cont   << scaled * 200 // 16384               # SF 236
main   << scaled * 180 // 32768               # SF 234
pawn   << scaled * 260 // 8192   if not pawn/promotion

# 無 bestMove、前手 capture：
capture_history[prev_pc][prev_to][captured_type] << min(40*depth + 200, 700)  # SF 固定 901
```

廢除現況「`legal_moves_tried > 4` 且 `bonus//4` 全表灑」的簡化路徑。

#### 5.B 其他完整小項

* TT quiet fail-high 更新：使用同一 `stat_bonus`，勿另寫 `120*depth`。
* `ttMoveHistory`（可選完整）：若 Phase 5 對戰前 NPS 餘裕足夠，加入 SE/LMR 用的 TT 著歷史；否則單列 Phase 5b。
* Correction history：本階段**只檢查** outer scale / 更新條件是否仍與新 staticEval 流程一致；不大改公式除非對戰暴露 eval 漂移。

#### 5.C 門檻最終版

見 §5 表「v_final」。

#### 5.D 文件收尾

* `HISTORY_HEURISTICS_CN.md` 重寫為與程式 1:1
* `SEARCH_ANALYSIS.md` move ordering / LMR / history prune 節更新
* 本計畫書各 Phase 結論填表
* `README.md` 導覽若有狀態標記則更新

---

## 5. History 相依剪枝與 LMR 門檻（必須隨尺度走）

### 5.1 重標方法（不要抄 -4313）

1. 在固定 TC 下跑 50 局，dump 各 depth 的 `statScore` / `main` / `cont0+cont1+pawn` 分位數。
2. History prune 觸發率目標：非 PV quiet 約 **5–15%**（過高=誤殺，過低=廢除）。
3. LMR history 調整使「好 quiet」平均少減 0.5–1 ply、「壞 quiet」多減 0.5 ply。

### 5.2 門檻表（實作時填實測，下列為 HCE 起點）

| 參數 | 現況 | Phase2 v1 | Phase4 v2 | Phase5 v_final |
| :--- | ---: | ---: | ---: | ---: |
| `PRUNING_HISTORY_THRESHOLD` | -4000 | **-2500**（main 尺度變了要重測） | 依 p20(stat) 定 | 鎖定 |
| Cont+pawn prune | （現用 total hist） | `hist < -3000 * depth` 起點 | 重測 | 鎖定 |
| 附加條件 | `main < 0` | 保留 | 保留 | 保留 |
| `LMR_HISTORY_SCALE` | 150 | 120–150 試 | 對齊 2*main+c0+c1 | 鎖定 |
| LMR stat 組成 | 2*main+c0+c1 | 同左 | main 改 from-to 後重標除數 | 鎖定 |
| History prune 最大 depth | 12 | 10–12 | — | 鎖定 |

**HCE 原則：** 寧可少剪；Phase 2 若 Elo 不穩，**先鬆門檻再動 bonus**。

---

## 6. 檔案與 API 變更地圖

| 檔案 | 主要變更 |
| :--- | :--- |
| `constants.py` | 全歷史常數、衰減模式、cont 權重、CMHC、門檻 |
| `engine_types.py` | 表形狀、counter、cont 7 維、main/piece_to 命名 |
| `search_heuristics.py` | gravity 更新、score_quiets、LMR stat、統一 update_* |
| `search.py` | cutoff 統計、fail-low、迭代衰減、refute、stack 旗標 |
| `core.py` / UCI 入口 | 分配新表尺寸 |
| 測試 | `tests/test_search*`、新增 history 單元測 |
| `HISTORY_HEURISTICS_CN.md` | 與程式同步重寫 |
| `SEARCH_ANALYSIS.md` | ordering / prune 節 |
| `HISTORY_REFORM_PLAN.md` | 本檔；階段結論 |
| `README.md` | 文件導覽加入本計畫 |

`classical_old`：每階段**通過後**用專案既有 `tools/sync_classical_old.py`（若存在）同步 `.py` 基線，**不同步 md**。

---

## 7. 你未點名、但建議納入討論的項目

下列不強制綁死五階段，但**不討論會在實作中踩雷**：

### 7.1 建議納入本計畫（已寫入階段）

| 項 | 原因 | 階段 |
| :--- | :--- | :--- |
| Reverse quiet（SF11） | HCE 經典 Elo；成本低 | P2 |
| Check + SEE 門檻 | 文件已承諾；假 check 污染 ordering | P1 |
| 威脅重排 bug | `threat_computed` 失效 | P1 |
| 統一 `update_*` 入口 | 完整 cont 無法靠複製貼上維護 | P2–P3 |
| Killer/Counter 與 hist 分數段 | main ceiling 下降後可能交叉 | P4 |
| 1s vs 5s 實驗矩陣 | 驗證衰減假說 | P2 |

### 7.2 建議討論後決定（預設「觀察，不主動大改」）

| 項 | 問題 | 建議 |
| :--- | :--- | :--- |
| **Correction History 尺度** | 新 ordering 改變樹形後，correction 學習分布會變 | Phase 5 只做一致性檢查；大改另開計畫 |
| **ttMoveHistory** | SE/LMR 現代項 | Phase 5 可選；NPS 緊則 5b |
| **Cont 綁定時刻 inCheck 語意** | 極易與 SF 差一 ply | Phase 3 必寫死註解 + 測例 |
| **QSearch 是否更新 history** | 現況？SF 主搜更新為主 | 維持主搜更新；QS 不寫 quiet hist |
| **PV 節點是否寫 malus** | SF 在 update_all_stats 對 PV 仍更新 best；node-width 僅非 PV | 對齊 SF |
| **Multi-cut / SE 與 hist 交互** | 你 SE 已 HCE 調參 | 本計畫不改 SE 常數，除非對戰顯示 TT 著 ordering 崩潰 |
| **時間管理** | 5s≈1s 的最大嫌疑之一 | 若 P2 實驗排除歷史假說，應**另開** timeman 計畫，勿繼續在 hist 上疊 |
| **NNUE 樹是否共用** | `chess_engine/nnue` 有無平行 search | 本計畫**只改 classical**；NNUE 另跟或不跟需你定 |
| **Numba cache 失效** | 表形狀變更會大面積重 JIT | 每階段接受首次編譯數分鐘；對戰前暖機 |
| **對戰基線污染** | 舊引擎若也手改 | 嚴格只用 sync 的 `classical_old` |

### 7.3 建議不要做的事

* 不要在同一階段同時改 RFP/NMP/LMP 常數與 history（歸因死亡）。
* 不要用「對局分數不穩」直接再 `÷2` 一層；先看 depth 與 prune 率。
* 不要把 SF18 的 `bonusScale` 中間 2.3M 量級原樣搬進 Python 再靠除法救回（溢位與可讀性差）；用 §5.A 壓縮形。

---

## 8. 階段閘門總表（執行用 checklist）

| 階段 | 核心交付 | 輕測 | 對戰 | 狀態（2026-07-16） |
| :---: | :--- | :---: | :---: | :--- |
| **1 full** | +威脅重排 +check SEE +SEE-MG | ✓ | vs old | **失敗**（~37.5%）→ 拆 1a |
| **1a** | Cap victim6、Counter piece-to、malus/pawn/LPH 尺度；無威脅/check SEE | ✓ | 500@200k | 曾通過；已併入 full |
| **2** | 二次 bonus/malus、SOFT 衰減、reverse、統一入口、門檻 v1 | ✓ | 待測 | **已實作**（SOFT 不污染零槽） |
| **3** | Cont `[inCheck][cap][5]…` 完整更新/讀取 | ✓ | 待測 | **已實作** |
| **4** | Main color×from_to、piece-to 輔助、去 butterfly | ✓ | 待測 | **已實作** |
| **5** | Fail-low 完整保守、門檻 final + diag | ✓ | 待測 | **已實作** |

回滾策略：對戰失敗可調係數或 `HIST_AGE_MODE`/`HISTORY_WEIGHT_PIECE_TO`；結構以 full 為準。  
工廠：`create_search_context` / `allocate_history_tables`。診斷：`DIAG_HIST_*` + `format_history_table_stats`。

---

## 9. 回答你的十點看法（對照本計畫）

| # | 你的看法 | 計畫落點 |
| :---: | :--- | :--- |
| 1 | Main 要改語意，但照抄有好處嗎？ | **語意對齊 SF；結構採 main(from-to×color)+輔助 piece-to**，不字面照抄刪 piece-to（§1） |
| 2 | Cont 將軍分離一次做完 | Phase 3 **+ capture 維度 + 權重 + CMHC** |
| 3 | Cap 受害者與 SF 一致 | Phase 1 |
| 4 | Counter 索引 | Phase 1 |
| 5 | Bonus/Malus 保守改 | Phase 2 完整結構 + 保守係數 |
| 6 | 延續歷史一次完整 | Phase 3 |
| 7 | 5s≈1s 疑污染 | Phase 2 實驗；並指出 ÷2 更可能傷害長 TC（§3.4） |
| 8 | Fail-low 完整但保守 | Phase 5 |
| 9 | 剪枝門檻確實調 | §5 + P2/P4/P5 三次標定 |
| 10 | 文件更新 | 每階段強制；P5 收斂 |

---

## 10. 階段結論紀錄

| 階段 | 日期 | 對戰條件 | Elo / SPRT | NPS 變化 | 結論 | 基線 |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 full | 2026-07-14 | ~24 局 @200k（中止） | New ~37.5% | — | **失敗**：威脅+check SEE+SEE-MG 混包 | — |
| **1a** | 2026-07-14 | 500 games @ 200k nodes | **253–247**（+4 Elo ±21）；SPRT LLR −0.07 | 中性 | **通過（中性）**：結構保留；**現用程式基線** | **未 sync Old**（保留對照） |
| **2**（第一輪） | 2026-07-14 | 中途 | — | **NPS −41.5%** | soft-age 對 cont/pawn 全表 int64 拷貝；已修大表÷2 | — |
| **2**（修速後） | 2026-07-14 | ~164/500 @200k（人工中止） | **50.0% / 0 Elo** CI≈[−40,+40] | **NPS −2.4%**；mean depth −0.19 | **無收益 → 整段回退** | 回到 **1a** |
| 3–5 | — | — | — | — | **暫緩**（見 §11） | — |

---

## 11. 決策：History 線暫停；下一段研究方向

### 11.1 為什麼先停

1. **1a 已還清最重要結構債**（cap victim6、counter piece-to、malus/pawn/LPH 尺度），且 fixed-nodes 下中性可接受。  
2. **P2 在 fixed-nodes 無 Elo、略損 NPS/depth** → 再疊 P3/P4（更大表、更多維度）ROI 差、歸因難。  
3. 同時改 history + 剪枝會讓對戰結果無法歸因（§7.3）。

### 11.2 何時才重開 P3–P5

滿足任一且**單獨**開 PR、有明確 A/B：

* 時間控制對局（`--time`）證明 history 污染或衰減是瓶頸；或  
* 單項實驗（例如僅 reverse quiet、僅 soft-age 小表）有 ≥ +10 Elo 且 CI 不嚴重跨 0；或  
* 轉 NNUE 後需 SF 語意對齊以便移植係數。

**禁止**在未證明 1a 穩定前一次做 Cont 完整維度 + Main 雙通道。

### 11.3 建議下一段（優先序，非 history）

| 優先 | 方向 | 理由 | 入口文件 |
| :---: | :--- | :--- | :--- |
| **A** | **Time management** | fixed-nodes 改動與實戰 Elo 脫鉤；1s/5s 體感與 `time_manager` 仍可能是短板 | `time_manager.py`、`SEARCH_ANALYSIS.md`、`tools/tournament.py --time` |
| **B** | **LMR / RFP / NMP / FP 尺度** | HCE 分數尺度下剪枝過兇或過鬆；比 history 公式更常直接換 Elo | `search.py`、`constants.py`、`.agents/rules/search-constraints.md` |
| **C** | **HCE 剩餘缺口（單項）** | 評估主線未完；每次只開一個特徵 A/B | `HCE_SF11_GAP_AUDIT.md`、`.agents/rules/hce-evaluation.md` |
| **D** | History 單項微實驗（可選） | 僅在 A–C 暫無明顯收益時；**禁止**再開整包 P2/P3 | 本文件 + `HISTORY_HEURISTICS_CN.md` |

### 11.4 對戰協定（延續）

```bash
# 搜尋/結構改動主測（等化節點）
python tools/tournament.py -c 4 --games 500 --nodes 200000 --name1 New --name2 Old

# 時間管理 / 長 TC 體感
python tools/tournament.py -c 4 --games 200 --time 1000   # 或 5000

# 分析（引擎名必須對上 --name1/--name2）
python tournament_analysis/統計數據.py --target New
python tournament_analysis/parse_search_stats.py --name1 New --name2 Old
```

**勿**預設 sync `classical_old`；僅在相對 Old **顯著**優勢後用 `tools/sync_classical_old.py`。

---

*文件結束。實作以本計畫 + `stockfish_repo/src/{history.h,search.cpp,movepick.cpp}` + `stockfish_11/src/search.cpp` 為準；衝突時 **語意從 SF，係數從 HCE 保守表**。History 線目前 **凍結於 1a**。*
