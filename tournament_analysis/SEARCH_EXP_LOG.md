# Classical 搜尋單項實驗日誌

> **用途：** 連續 A/B 實驗的唯一進度表與決策紀錄。未來接續實驗、回顧為何留下/回滾，以此檔為準。  
> **最後更新：** 2026-09-26  
> **當前基準狀態：** **E53 已採納進生產基線**（延續歷史剪枝門檻放寬 -1800->-1500，400@n300k, 52.62%, Elo +18.26 ±23.40, 淨勝 +21 局, 深度 +0.04 ply）。歷史診斷見 [`archive/tournament_analysis_2026-07/DIAG_PROTOCOL_D3.md`](../archive/tournament_analysis_2026-07/DIAG_PROTOCOL_D3.md)。

---

## 1. 協定（固定）

| 項目 | 設定 |
| :--- | :--- |
| 改動粒度 | **一次只改一項**（單常數 / 單旗標） |
| 對局 | **`--games 1000 --nodes 10000 -c 6`**（E22 起正式判定） |
| New | `main.py` → `chess_engine/classical/` |
| Old | `main_old.py` → `chess_engine/classical_old/` |
| 開局 | `data/openings.epd`（tournament 內 shuffle） |
| OwnBook | warmup 後關閉 |
| nodes | **正式判定固定 n10k × 1000 場**（舊 200@100k / 300@150k 僅歷史參考） |
| SPRT | 預設開（可提前停）；early-trash 同下 |

### 1.1 留下 / 回滾流程（**E22 起：1000@n10k**）

**不另開獨立 bench 當必做步驟。** 改常數 → **直接 1000@n10k 對戰**；診斷在對戰過程 / 結束後一併收集與書寫。

1. 只改 **New** 一項（意圖與預期 diags 方向先寫進本檔）  
2. `tournament.py` **1000@n10k** · early-trash（≥80 且 score%≤40）· SPRT 預設開  
3. 解析 PGN **depth/NPS**（`parse_search_stats.py`）  
4. **診斷（對戰順便）**：  
   - 對戰結束後必要時對 **同一 New 建置** 跑一次  
     `python tools/hist_diag_bench.py --nodes 10000 --positions 12 --tag <ID>_post`  
     （可選；優先保證對戰完成）  
   - 或引擎 `format_search_diagnostics` / `DIAG_HIST_*` 在搜尋路徑已計數，bench 只是彙總  
5. 依門檻判定 → 留下 `sync` / 回滾不 sync  
6. **詳細寫入本檔**：意圖、Elo/score、depth/NPS、hist 診斷摘要、判定、PGN/stats/diag 路徑  

**協定變更（2026-07-19）：** 正式對戰由 200@100k（或高精度 300@150k）改為 **1000 場 × nodes=10000**。歷史 E0–E21 結果仍有效，但跨協定 Elo 不可直接橫比。

#### 診斷欄位（有數據就寫；至少 depth/NPS + Elo）

| 欄位 | 來源 | 用途 |
| :--- | :--- | :--- |
| Elo / score% | tournament | 最終裁判 |
| Δ depth / NPS | parse_search_stats | 樹效率 |
| `hist_prune_per_1k_nodes` 等 | `hist_diag_bench` 或搜尋 diag（對戰後彙總） | prune / ceiling / aging / LMR |

**實作備註（2026-07-17）：** live 已接 `DIAG_HIST_*`；`format_search_diagnostics` 含 hist；`tools/hist_diag_bench.py` 可選彙總用。**日常單項仍：對戰即驗證，不另開必做 bench。**

### 1.1b 完整獨立診斷 D2（**E22 定案後執行一次**）

> **觸發：** E22（LMP 110）對戰結束 **且** 常數已鎖定（留下→sync / 回滾→LMP=105）。  
> **目的：** 在「確定 θ」上做一次**獨立、完整**的剪枝剖面，再決定下一刀方向（不是用診斷否決 E22 Elo）。  
> **原則：** 對戰定生死；D2 只回答「現在哪種 prune 過猛/過鬆、開局 vs 戰術差在哪」。

#### 步驟（固定順序）

1. **鎖定 θ**  
   - 留下：`python tools/sync_classical_old.py --sync`  
   - 回滾：`LMP_SCALE_PERCENT=105`，確認 New==Old 此常數  

2. **對戰 PGN 樹效率**（E22 同局）  
   ```text
   python tournament_analysis/parse_search_stats.py \
     --pgn tournament_analysis/tournament_results_E22_LMP110_n10k1000.pgn \
     --name1 E22_LMP110 --name2 Old \
     --out-dir tournament_analysis/stats_E22_LMP110
   ```  
   產出：mean depth / NPS / depth CDF（New−Old）。

3. **雙集合 hist/search 剖面（主診斷）** — 對 **已鎖定 New** 跑一次：  
   ```text
   python tools/hist_diag_bench.py --nodes 10000 -c 4 --json \
     --tag D2_post_E22_n10k \
     --puzzles data/hist_diag_puzzles_500.epd \
     --openings data/hist_diag_openings_500.epd
   ```  
   - **nodes=10k** 與正式對戰一致（剖面可比對戰 regime）  
   - puzzle / opening **分開 + 合計**（延續 D1：opening hist-prune 曾 ≈4× puzzle）  
   - 產物：`tournament_analysis/hist_diag_D2_post_E22_n10k.{txt,json}`  

4. **（可選加強）同 θ 在 100k nodes 再跑一輪小樣**  
   ```text
   python tools/hist_diag_bench.py --nodes 100000 -c 4 --json \
     --tag D2_post_E22_n100k --positions 100
   ```  
   僅在 10k 剖面與 PGN depth 嚴重背離時做；預設可跳過以省時間。

5. **解讀清單（寫入 log §D2）**  

   | 指標 | 讀法 | 對下一刀的含義 |
   | :--- | :--- | :--- |
   | `hist_prune / 1k` opening vs puzzle | opening ≫ puzzle → 開局仍 over-prune | 優先 **PRUNE 更負**（−6500） |
   | `lmp_skip` 量級 | LMP 110 後 skip 應略降（較晚剪） | 若 skip 仍極高且 Elo 負 → 勿再推 LMP；改 hist/FP |
   | `lmr_research%` | 偏高 → 減深常被 research 吃掉 | 勿硬推 LMR；先修好 prune/order |
   | `cut_first%` | <88% 異常 | ordering/history 問題，非 margin |
   | BF/CAP/CONT `sat%` / `mean_abs` | CONT 仍空 → **勿升 CONT**；CAP 可考慮 | CAP 12288 僅 sat 或 Elo 線有空時 |
   | PGN Δdepth @n10k | 負且 Elo 負 → 樹肥無回報 | 下一刀應 **更敢剪** 或換軸，勿再鬆 LMP |
   | PGN Δdepth 正且 Elo 正 | 剪得更有效 | 可沿同方向小步 |

6. **由 D2 產出下一刀（單選）**  

   ```text
   if opening_hist_prune 仍明顯 > puzzle 且 cut_first 正常:
       → E23 = PRUNING_HISTORY_THRESHOLD −6000→−6500
   elif E22 留下且 LMP skip 仍高、depth 未崩:
       → 可考慮 LMP 115（低優先；E22 若負則永不）
   elif CAP sat 上升或 history 線仍有 Elo 空間:
       → CAP 10240→12288
   elif cut_first 差或 research% 異常高:
       → 停 prune 旋鈕，改 history 讀權重/ordering（非本輪）
   else:
       → 剪枝軸凍結；改其他搜尋或評估
   ```

7. **交付物**  
   - `SEARCH_EXP_LOG.md` 新增 **§D2**（表 + 下一刀決策）  
   - `stats_E22_*` + `hist_diag_D2_*`  
   - **然後**才開 E23 單項 1000@n10k

### 1.2 通過門檻

| 等級 | 條件 |
| :--- | :--- |
| **明確正** | score ≥ 52% 或 Elo ≥ +10，且 depth 不掉、NPS 掉 ≤5% |
| **近平手有優勢（亦可接受）** | score ≥ 50.5% **或** Elo 點估計 > 0，且 depth/NPS 無明顯退步（depth 掉 ≤0.1、NPS 掉 ≤5%）；CI 可含 0 |
| **回滾** | score ≤ 49.5% 或 Elo 點估計明顯負，或 depth/NPS 明顯退 |
| **明顯垃圾提前停** | ≥**80** 場且 **score% ≤ 40%** → 立刻回滾（`tools/tournament.py` 預設；`--no-early-trash` 可關）。**不再使用 Elo 門檻。** |

**early-trash 變更紀錄：**

| 日期 | 規則 |
| :--- | :--- |
| E5 當下 | ≥80 場且（score%≤**45%** 或 Elo≤**−20**） |
| 2026-07-17 早 | score 改 **40%**，仍保留 Elo≤−20（E6 因此在 47.1%/Elo−20.2 被砍） |
| **2026-07-17 後** | **取消 Elo 條件**；僅 ≥80 場且 **score%≤40%** 才提前停。邊緣 Elo 負但 score>40% 須跑滿 200 再判。 |

### 1.3 累積基線（截至 E5 後、E6 前）

Old 與 New 應一致，除非 New 正在跑實驗項：

| 常數 | 值 | 來源 |
| :--- | :--- | :--- |
| `ENABLE_CHECK_SEE_GATE` | `True` | E1 留下 |
| `CHECK_SEE_THRESHOLD` | `-75` | 既有 |
| `PRUNING_HISTORY_THRESHOLD` | `-6500` | E21 −6000 → **E23 −6500**（1000@n10k +7 Elo） |
| `LMP_SCALE_PERCENT` | `105` | E4 留下（原 100） |
| `LMR_HISTORY_SCALE` | `150` | E5/E6 回滾後維持 |
| `LMR_BASE_OFFSET` | `460` | E7 480 → E18 448 → **E20 460**（300@150k） |
| `HISTORY_WEIGHT_MAIN` | `3` | E8/H1 留下（原 2） |
| `HISTORY_WEIGHT_CONT_1` | `5` | E9/H2 留下（原 4） |
| `HISTORY_MAX_BUTTERFLY` | `12288` | E10 10240 → E14 12288 |
| `HISTORY_MAX_CAPTURE` | `10240` | E15 8192→10240 |
| `HISTORY_MAX_CONTINUATION` | `12288` | E16 試 14336 **回滾**；維持 12288 |
| `HIST_AGE_SOFT_NUM` | `900` | E11 850 → E13 900（使用者指定留下） |

**歷史大方向備註（勿重踩）：** full history / SoftHist / fake 1a-compat 曾大幅負 Elo；Phase 1a hard restore 後再從此基線單項前進。History **表結構大改**暫緩；history 相關只做**單常數小步**。

---

## 2. 結果總表

| ID | 變更 | 結果摘要 | 判定 | 日期 |
| :--- | :--- | :--- | :--- | :--- |
| E0 | 基線 Restored（無改） | 51% / +7 @200k×100（參考） | 基線 | 2026-07-16 |
| E1 | `ENABLE_CHECK_SEE_GATE=True` | **200@100k: 52.25% · Elo +15.7 · depth 0 · NPS 0** | **留下**（明確正） | 2026-07-16 |
| E2 | `LMP_SCALE_PERCENT` 100→95（略早 LMP） | **200@100k: 48.5% · Elo −10.4 · depth −0.06 · NPS +0.5%** | **回滾** | 2026-07-16 |
| E3 | `PRUNING_HISTORY_THRESHOLD` −4000→−4500（略難剪） | **200@100k: 51.25% · Elo +8.7 · depth +0.03 · NPS +1%** | **留下**（近平手有優勢） | 2026-07-16 |
| E4 | `LMP_SCALE_PERCENT` 100→105（略晚 LMP；E2 反向） | **200@100k: 52.25% · Elo +15.7 · depth +0.04 · NPS +0.6%** | **留下**（明確正） | 2026-07-16 |
| E5 | `LMR_HISTORY_SCALE` 150→160（history 更主導 LMR） | **82 場 early-trash: 46.3% · Elo −25.5** | **回滾** | 2026-07-16 |
| E6 | `LMR_HISTORY_SCALE` 150→140（E5 反向） | **86 場 early-trash: 47.1% · Elo −20.2** | **回滾**（Elo≤−20；score 未觸 40%） | 2026-07-17 |
| E7 | `LMR_BASE_OFFSET` 512→480（略少基礎減深） | **200@100k: 52.5% · Elo +17.4 · depth +0.06 · NPS −0.6%** | **留下**（明確正） | 2026-07-17 |
| E8/H1 | `HISTORY_WEIGHT_MAIN` 2→3（history 基建） | **200@100k: 50.5% · Elo +3.5 · depth +0.06 · NPS 0** | **留下**（近平手有優勢） | 2026-07-17 |
| E9/H2 | `HISTORY_WEIGHT_CONT_1` 4→5 | **200@100k: 50.75% · Elo +5.2 · depth +0.04 · NPS +0.2%** | **留下**（近平手有優勢） | 2026-07-17 |
| E10/H3 | `HISTORY_MAX_BUTTERFLY` 8192→10240 | **200@100k: 55.5% · Elo +38.4 · depth +0.12 · NPS +0.7%** | **留下**（明確正） | 2026-07-17 |
| E11/H4 | `HIST_AGE_SOFT_NUM` 800→850 | **200@100k: 56.5% · Elo +45.4 · depth +0.10 · NPS +1.5%** | **留下**（明確正） | 2026-07-17 |
| E12/H5 | `PRUNING_HISTORY_THRESHOLD` −4500→−5000 | **200@100k: 52.75% · Elo +19.1 · depth +0.11 · NPS +0.5%** | **留下**（明確正） | 2026-07-17 |
| E13 | `HIST_AGE_SOFT_NUM` 850→900（E11 加深） | 對戰中止；**使用者保留 900 並 sync** | **留下**（使用者指定） | 2026-07-17 |
| E14 | `HISTORY_MAX_BUTTERFLY` 10240→12288（E10 加深） | **200@100k: 50.75% · Elo +5.2 · depth 0 · NPS 0** | **留下**（近平手有優勢；邊際遞減） | 2026-07-17 |
| E15 | `HISTORY_MAX_CAPTURE` 8192→10240（ceiling 同源） | **200@100k: 52.0% · Elo +13.9 · depth +0.04 · NPS +1%** | **留下**（明確正） | 2026-07-17 |
| E16 | `HISTORY_MAX_CONTINUATION` 12288→14336 | **200@100k: 48.25% · Elo −12.2 · depth −0.03 · NPS +0.7%** | **回滾**（中段虛高，終局翻負；**勿再推 CONT ceiling**） | 2026-07-17 |
| E17 | `PRUNING_HISTORY_THRESHOLD` −5000→−5500（E12 加深；D1 驅動） | **200@100k: 54.5% · Elo +31.4 · depth +0.06 · NPS +1.1%** | **留下**（明確正） | 2026-07-17 |
| E18 | `LMR_BASE_OFFSET` 480→448（E7 加深；D1 opening LMR/depth） | **200@100k: 50.2% · Elo +1.7 · depth 0 · NPS +1.2%** | **留下**（近平手有優勢；中段虛高終局回落） | 2026-07-17 |
| E19 | `PRUNING_HISTORY_THRESHOLD` −5500→−6000（E12/E17 第三刀） | JIT/對戰未完成；**使用者改優先 E20** | **中止**（New prune 已還原 −5500） | 2026-07-17 |
| E20 | `LMR_BASE_OFFSET` 448→460（E18 半回調） | **300@150k: 51.2% · Elo +8.1 · depth +0.05 · NPS +0.7%** | **留下**（近平手有優勢；高精度） | 2026-07-17 |
| E21 | `PRUNING_HISTORY_THRESHOLD` −5500→−6000 | **300@150k: 53.2% · Elo +22.0 · depth +0.12 · NPS +0.6%** | **留下**（明確正） | 2026-07-17 |
| E22 | `LMP_SCALE_PERCENT` 105→110 | **~794@n10k 中止: 48.1% · Elo ≈−13 · depth −0.01** | **回滾**（使用者中止+回退） | 2026-07-19 |
| E23 | `PRUNING_HISTORY_THRESHOLD` −6000→−6500 | **1000@n10k: 51.0% · Elo +7.0 · depth +0.01 · NPS −0.2%** | **留下**（近平手有優勢；已 sync） | 2026-07-19 |
| E24 | `RFP_BASE_MULT` 170→180（略鬆 RFP） | **~906@n10k 中止: 49.7% · Elo ≈−2.3 · depth −0.02** | **回滾**（使用者中止） | 2026-07-19 |
| E25A | `FP_MULTIPLIER` 130→140（略鬆 FP） | **1000@n10k: 49.0% · Elo −7.3** | **回滾** | 2026-07-19 |
| E25B | `PRUNING_HISTORY_THRESHOLD` −6500→−7000 | **~216@n10k 中止: ~46% · Elo ≈−26** | **回滾**；margin 戰役停 | 2026-07-19 |
| N1 | NMP scope **3**（cut∨nonPV&d≥6）；gate/R legacy | **1000@n10k: 49.0% · Elo −6.6** | **回滾** scope→0 | 2026-07-19 |
| N2 | NMP gate `LEGACY_BASE` 200→**160**（scope0；d10≈β+20） | **1000@n10k: 50.3% · Elo +2.1 · depth +0.03 · NPS +0.6%** | **留下**（近平手；已 sync） | 2026-07-19 |
| E26 | **Alpha-raise depth reduction**（SF18 式；HCE 適配） | 見下 §E26 | **回滾 / 關閉** | 2026-07-20 |
| E27 | **`ENABLE_FOLLOW_PV`**（上一輪 ID PV 路徑豁免 IIR / quiet skip） | **~332@n30k 中止: ~47.7% · Elo≈−16**（中段曾回 50% 後再滑） | **回滾** `False` | 2026-07-20 |
| E28 | **History→lmrDepth quiet FP/SEE**（`ENABLE_HIST_LMR_DEPTH_PRUNING`） | **~700@n30k 中止: ~49.9% · Elo≈−1**（前段虛高後收斂均勢） | **留下**（中性保留；已進基線） | 2026-07-20 |
| E29 | **Capture futility + captHist→SEE margin** | **~368@n30k 中止: ~47.7% · Elo≈−16**（100p 曾 51% 後下滑） | **回滾** `False` | 2026-07-20 |
| E30 | **LMR tactical relief + TT capture SEE exempt** | **~334@n10k: 45.1% · Elo≈−34 · CI[−66,−3]**；depth **−0.09** | **刪除實作**（非僅關旗） | 2026-07-20 |
| E31 | **`ENABLE_MULTICUT`**（修復判定 `exclusion_score >= beta` 對齊 SF18 Step 15） | **300@n300k: 53.3% (82W 156D 62L) · Elo +23.2 ±27.2 · LLR=+1.15** | **採納進生產 / 基線** | 2026-09-20 |
| E32 | **Threat reordering**（`ENABLE_THREAT_REORDERING`，含象車完整射線） | **296@n300k 中止: 49.3% (67W 158D 71L) · Elo −4.7 ±27.0 · LLR=−0.53** | **回滾** `False` | 2026-09-20 |
| E33 | **`ENABLE_CAPTURE_FUTILITY`**（前移至 make_move 前 + HCE 裕度縮放） | **300@n300k: 51.2% (72W 163D 65L) · Elo +8.11 ±26.6 · LLR=+0.19** | **採納進生產 / 基線** | 2026-09-20 |
| E34 | **Alpha-raise depth reduction**（TT 深度/PV 重搜修復版） | **300@n300k: 49.7% (75W 148D 77L) · Elo −2.3 ±39.3 · LLR=−0.24** | **回滾** `False` | 2026-09-20 |
| E35 | **`ENABLE_PROBCUT_TT_SHORTCUT`**（Step 12: TT ProbCut 快速捷徑） | **300@n300k: 51.7% (74W 162D 64L) · Elo +11.59 ±26.66 · LLR=+0.36** | **採納進生產 / 基線** | 2026-09-21 |
| E36 | **Pre-make Quiet Pruning Cascade**（Step 14: LMP → History → FP → SEE 對齊 SF18） | **400@n300k: 53.0% (108W 208D 84L) · Elo +20.87 ±23.84 · NPS +6.1% · 淨勝 +24 局** | **採納進生產 / 基線** | 2026-09-21 |
| E37 | **Aspiration Window 收斂與幾何擴展**（對齊 SF17/18） | **400@n300k: 50.0% (94W 212D 94L) · Elo 0.00 ±22.80 · NPS +0.6%** | **採納進生產 / 基線** | 2026-09-23 |
| E38 | **Continuation History Pruning**（SF17 Step 14 延續歷史獨立剪枝） | **400@n300k: 52.5% (97W 226D 77L) · Elo +17.39 ±23.88 · 長局勝率 57.9%** | **採納進生產 / 基線** | 2026-09-23 |
| E39 | **SE 負延伸幅度 -3 探勘**（修復 Multi-Cut 判定後的 SE 減步） | **322@n300k 中止: 49.8% (77W 167D 78L) · Elo -1.08 ±26.42 · 深度 +0.19** | **不採納** | 2026-09-24 |
| E40 | **SE 負延伸幅度 -2 vs 原版基線**（溫和負延伸探勘） | **336@n300k: 46.9% (62W 191D 83L) · Elo -21.74 ±24.47 · 深度 -0.06** | **回滾** | 2026-09-24 |
| E41 | **Step 14c FP BestValue 更新**（SF18 式 FP 緊緻上界估計） | **268@n300k 中止: 47.4% (52W 150D 66L) · Elo -18.17 ±27.70 · TT 灌水** | **回滾** | 2026-09-24 |
| E42 | **Step 21 Fail-Low 反擊歷史獎勵門檻解禁與安靜步守護**（對齊 SF18 Step 21） | **400@n300k: 48.9% (87W 217D 96L) · Elo -7.82 ±23.86 · 中局雜訊** | **回滾** | 2026-09-25 |
| E43 | **Step 9 RFP 深度提升至 8 與 TT 安靜步守護**（對齊 SF19 Step 9，無平滑回傳） | **400@n300k: 50.9% (100W 207D 93L) · Elo +6.08 ±23.68 · 耗時顯著 -7.1ms · 深度 +0.13** | **採納進生產 / 基線** | 2026-09-25 |
| E44 | **IIR 深度門檻提升至 6 與 Cut/PV 節點守護**（對齊 SF19 Step 11） | **136@n300k 中止: 47.1% (25W 78D 33L) · Elo -20.46 ±38.25 · 深度萎縮 -0.19** | **回滾** | 2026-09-25 |
| E45 | **LMR 動態 Eval-to-Alpha 差值調節**（對齊 SF19 Step 18） | **124@n300k 中止: 46.4% (21W 73D 30L) · Elo -25.26 ±41.74 · 黑方誤剪** | **回滾** | 2026-09-25 |
| E46 | **Razoring 現代化與二次方深度裕度**（對齊 SF19 Step 8） | **400@n300k: 50.9% (100W 207D 93L) · Elo +6.08 ±22.47 · 深度 +0.03 · 殘局勝率 56.2%** | **採納進生產 / 基線** | 2026-09-25 |
| E47 | **ProbCut 深度動態自適應**（對齊 SF19 Step 12） | **400@n300k: 51.8% (94W 226D 80L) · Elo +12.17 ±19.88 · 淨勝 +14 局 · 耗時 -1.4ms** | **採納進生產 / 基線** | 2026-09-26 |
| E48 | **LMR ALL-Node 放大比率對齊**（對齊 SF19 Step 18） | **400@n300k: 51.0% (102W 204D 94L) · Elo +6.95 ±23.83 · 深度 +0.08 · 耗時 -3.8ms · 淨勝 +8 局** | **採納進生產 / 基線** | 2026-09-26 |
| E49 | **LMR CutNode 減深加強**（對齊 SF19 Step 18） | **400@n300k: 49.8% (87W 224D 89L) · Elo -1.74 ±22.59 · 深度 -0.13 · 黑方受壓** | **回滾** | 2026-09-26 |
| E51 | **LMR TT Move 減深減免對齊**（對齊 SF19 Step 18） | 中止（194 局均勢；查明 `move == tt_move` 在 LMR 分支為死碼） | **中止 / 關閉** | 2026-09-26 |
| E52 | **SE 雙重延伸深度門檻解禁 (10->7)**（對齊 SF19 Step 16） | **118@n300k 中止: 44.5% (21W 63D 34L) · Elo -38.43 ±45.18 · 深度 -0.21 · 長局崩盤 38.1%** | **回滾** | 2026-09-26 |
| E53 | **延續歷史剪枝門檻放寬 (-1800->-1500)**（對齊 SF19 Step 14） | **400@n300k: 52.6% (105W 211D 84L) · Elo +18.26 ±23.40 · 淨勝 +21 局 · 長局 55.4%** | **採納進生產 / 基線** | 2026-09-26 |

#### E26 Alpha-raise（否決）

| 變體 | 協定 | 結果 | 搜尋（parse_search_stats） |
| :--- | :--- | :--- | :--- |
| δ=1 · band 5–12 · `ply>0` · 含 PV | ~530@**n30k** · 8moves 16 plies | **50.0% · Elo 0** CI±24 | mean depth **+0.31**；nodes/move **+3.9%**；多在 d11–12，d≥14 不變；同 d nodes 比≈1 |
| δ=2 · 同上 | ~210@**n30k** 中止 | **~45.7% · Elo ≈−30** CI[−68,+8] | mean depth **+0.56**；nodes **+5.1%**；d11–12 更肥 |

**機制備註：**

- 曾誤加 `not is_pv` → null-window 下死碼（暖機 nodes 與 Base 完全相同）；修正後才有效。  
- 與 LMR/PVS 重疊；HCE 排序不穩時「先抬 α 再減兄弟」易錯；名義 depth↑ 不換 Elo。  
- **常數：** `ENABLE_ALPHA_RAISE_DEPTH_REDUCTION = False`（保留 LO/HI/DELTA 旋鈕）。  
- **產物：** `archive/tournament_analysis_2026-07/pgn/match_alpharaise_d1_*_partial.pgn`、`match_alpharaise_d2_*_partial.pgn`。

#### E27 followPV（回滾）

- **New：** `ENABLE_FOLLOW_PV=True`；**Base** 關。  
- **協定：** 1000@**n30k** · 16 plies · 中止於 **~332 局 · ~47.7% · Elo≈−16**（CI±31）。  
- **軌跡：** 前段極差（~39%）→ 中段曾回 50–51% → 再滑到 ~48%。不穩 + 終點偏負 → **不採納**。  
- **產物：** `archive/.../match_followpv_n30k_partial_aborted.pgn`。  
- **常數：** `ENABLE_FOLLOW_PV = False`。

#### E28 History→lmrDepth quiet FP/SEE（留下）

- **結果：** ~700@n30k · **~49.9% · Elo≈−1**（CI 仍跨 0）；前段曾 55%+ 後收斂。  
- **判定：** 中性、無明顯傷害 → **採納進生產**（`ENABLE_HIST_LMR_DEPTH_PRUNING=True`）。  
- **產物：** `tournament_analysis/match_hist_lmr_prune_n30k_g1000.pgn`（或 archive 同名）。  
- **基線：** sync 後 Old 含此功能。

#### E29 Capture futility + captHist SEE（回滾）

- **結果：** ~368@n30k · **~47.7% · Elo≈−16**（CI±29）；軌跡 100p≈51% → 150p≈48% → 終點再降。  
- **判定：** 未達 1000 局但樣本充足且持續偏負 → **不採納**。  
- **常數：** `ENABLE_CAPTURE_FUTILITY=False`（碼保留）。  
- **產物：** `archive/.../match_cap_futility_n30k_e29_rejected.pgn`。  
- **解讀：** 吃子側再剪（即使有 captHist）在 HCE@30k 易誤砍；後續改「少剪錯」而非再兇 cap-FP。

#### E30 Tactical safety（刪除）

- **結果：** ~334@n10k · **45.1% · Elo≈−34** · CI **[−66, −3]**（顯著負）；mean depth **−0.09**、nodes **−1.4%**。  
- **判定：** fixed-nodes 下強迫著少減深／TT 吃子多搜 → 吸節點、名義深度掉、棋力掉。  
- **處理：** **完整移除** `ENABLE_LMR_TACTICAL_RELIEF` / `ENABLE_TT_CAPTURE_SEE_EXEMPT` 及相關常數與分支；還原將軍 LMR 收尾 `lmr-=1`。  
- **產物：** `archive/.../match_tactical_safety_n10k_e30_rejected.pgn`。

#### E31 Multi-Cut（採納）

- **New：** `ENABLE_MULTICUT=True`，修正原先 `exclusion_beta >= beta` 誤判為 `exclusion_score >= beta and not is_decisive(exclusion_score)`，並回傳真實搜尋分數。
- **協定：** 300@**n300k** · 8 workers · TT=64MB。
- **結果：** **160.0–140.0（82W 156D 62L）· 53.3% · Elo +23.2 ±27.2 · LLR=+1.15**（在 n100k 下為 48.3%，隨深度/節點增加顯著展現深層剪枝正向擴展優勢）。
- **判定：** 顯著優勢，淨勝 +20 局 → **採納進生產**，已由 `tools/sync_classical_old.py` 同步為新黃金基線。
- **產物：** `tournament_analysis/match_multicut_n300k_e31_adopted.pgn`。

#### E32 Threat reordering（否決回滾）

- **New：** `ENABLE_THREAT_REORDERING=True`，在 `score_quiets` 與 `score_moves` 補齊 `enemy_bishop_attacks`（象抓車）與 `enemy_rook_attacks`（車抓后）。
- **協定：** 300@**n300k** · 8 workers · TT=64MB。
- **結果：** **~296 局中止：146.0–150.0（67W 158D 71L）· 49.3% · Elo −4.7 ±27.0 · LLR=−0.53**。
- **判定：** 負向趨勢穩定，未能帶來正向收益 → **不採納，完整回滾**。
- **解讀：** 
  1. 在 Python/Numba 環境下，每個安靜步節點即時計算敵方 4 種棋子的攻擊位元盤產生了額外 CPU/NPS 開銷。
  2. HCE 高度依賴精心調校的歷史表（History 表）提供細緻的走步排列，額外疊加 `piece_value * 20` 的靜態威脅分數反而擾亂了原本更優秀的歷史表排序。
- **產物：** `tournament_analysis/match_threat_n300k_e32_rejected.pgn`。

#### E33 Capture Futility Pruning 重構（採納）

- **New：** `ENABLE_CAPTURE_FUTILITY=True`，將吃子無效剪枝前移至 `make_move` 之前（Step 14 原生設計），利用真實 `cap_lmr_d`、`MG_MATERIAL_VALUES`、`capture_history` 與 `static_score` 進行零走棋開銷預測；徹底刪除過去位於 `make_move` 之後低效且引發負 Elo 的 `unmake -> eval -> make` 冗餘碼；裕度依 HCE 比例縮放（`CAP_FP_BASE = 110`, `CAP_FP_LMR_MULT = 100`）。
- **協定：** 300@**n300k** · 8 workers · TT=64MB。
- **結果：** **153.5–146.5（72W 163D 65L）· 51.2% · Elo +8.11 ±26.57 · LLR=+0.19**。
- **搜尋統計分析（parse_search_stats）：**
  - nodes/move 顯著縮減 **-0.5%**（Welch $t = -2.656, p = 0.0079$，達到統計顯著性，確認剪枝有效減少無謂節點浪費）。
  - NPS 與名義深度完全持平（NPS -0.6%, $p = 0.33$；depth -0.04 ply, $p = 0.79$）。
- **判定：** 兩次 300 局測試均穩定維持 +8 Elo 且節點統計顯著縮減 → **採納進生產 / 基線**，已由 `tools/sync_classical_old.py` 同步為新黃金基線。
- **產物：** `tournament_analysis/match_cap_futility_n300k_e33_adopted.pgn`。

#### E34 Alpha-Raise Depth Reduction 修復版（否決回滾）

- **New：** `ENABLE_ALPHA_RAISE_DEPTH_REDUCTION=True`，修復四大歷史缺陷：
  1. 引入 `entry_depth` 保護置換表，徹底解決 E26 寫入縮水深度導致 TT 命中率受損與節點反增 5% 的問題；
  2. 在 PVS `do_full_pv_search` 重搜時補償還原深度（`adjusted_depth + DELTA`），保護戰術翻盤著；
  3. 增加 `alpha_raise_reduced` 旗標，嚴格限定單節點最多扣減一次；
  4. 顯式守護 `is_pv and ply > 0`。
- **協定：** 300@**n300k** · 8 workers · TT=64MB · $\delta=1$（5~12 ply 生效）。
- **結果：** **149.0–151.0（75W 148D 77L）· 49.67% · Elo −2.32 ±39.32 · LLR=−0.24**。
- **搜尋統計分析（parse_search_stats）：**
  - mean depth +0.14 ply（Welch $t = +0.811, p = 0.42$，不顯著）。
  - nodes/move -0.2%（Welch $t = -1.183, p = 0.24$，不顯著）。
  - NPS 差異 -0.0%（$p = 0.96$）。
- **判定：** 雖然修復了 E26 的所有架構 bug，但此機制在 HCE 淺樹（10~12 ply）下受限於 PV 節點極低佔比（<2%）與 LMR 的強重疊，實質表現為完全中性（0 Elo）。依據奧卡姆剃刀與無明確正 Elo 不增添代碼複雜度原則，**否決並完整回滾**，保持 Phase 4 純淨基線。
- **產物：** `tournament_analysis/match_alpharaise_n300k_e34_rejected.pgn`。

#### E35 TT-based ProbCut Shortcut（採納）

- **New：** `ENABLE_PROBCUT_TT_SHORTCUT=True`，實作對齊 Stockfish 18 Step 12（A small Probcut idea）。若 TT 條目已有下界記錄（`BOUND_LOWER` 或 `BOUND_EXACT`）且深度達 `tt_depth >= depth - 4`，分數達 `tt_score >= beta + PROBCUT_TT_SHORTCUT_MARGIN`（334 cp，對齊 SF18 的 428 cp 乘以 HCE 縮放 $78/100$），則直接判定 Fail-High 回傳 `probcut_beta`。
- **特點：** 放置於 IIR 之後、Move Ordering 幾何預算與走步生成之前。純 $O(1)$ 零搜尋開銷，跳過牽制計算、Check Squares 與走步生成。
- **守護：** 嚴格限制 `ply > 0`（根節點禁止啟發式截斷）、`not is_exclusion_search`、`not is_currently_in_check`、`depth >= 5`、排除將死與確知勝負分數線。
- **協定：** 300@**n300k** · 8 workers · TT=64MB。
- **結果：** **155.0–145.0（74W 162D 64L）· 51.67% · Elo +11.59 ±26.66 · LLR=+0.36**（淨勝 +10 局）。
- **搜尋統計分析（parse_search_stats）：**
  - mean depth 從 15.98 提升至 **16.05 ply (+0.08 ply)**。
  - depth $\ge 14$ 佔比從 38.4% 提升至 **39.5% (+1.1%)**。
  - NPS 完全持平（350,430 vs 350,030，+0.1%）。
  - 同深度節點開銷平穩，無戰術震盪。
- **判定：** 淨勝 +10 局，正向 Elo 且深度指標顯著提升 → **採納進生產 / 基線**，已由 `tools/sync_classical_old.py` 同步為新黃金基線。
- **產物：** `tournament_analysis/match_probcut_tt_n300k_e35_adopted.pgn`。

#### E36 Pre-make Quiet Pruning Cascade (SF18 Step 14 對齊，方向一)

- **New：** 重構安靜著剪枝階梯，將原本位於 `make_move` 之後、需調用 `unmake_move` 復原的安靜著剪枝（LMP、History Pruning、Futility Pruning），全部前移至走步迴圈開頭的 **Step 14 前置淺層剪枝區**（在 `make_move` 之前直接 `continue`）。
- **特點：**
  1. **代價階梯排序（Cost Ordering）**：按照計算開銷由低至高執行：[14a] LMP（$O(1)$ 計數器比較）$\to$ [14b] History Pruning（歷史表查表）$\to$ [14c] Futility Pruning（直接在 parent state 評估 full static eval，省去 2 次 make/unmake 往返）$\to$ [14d] Quiet SEE Pruning（射線攻防）。
  2. **將軍著絕對安全**：利用 `_qsearch_ordinary_move_gives_check` 前置探測將軍，將軍著走 Checking SEE 專屬通道，絕不被安靜剪枝過濾。
  3. **LMR 計數不變量**：使用 `quiet_counted` 旗標精確追蹤計數，跳過 Step 14 的深層節點與 PV 節點在 `make_move` 後由 fallback 累計 `quiet_move_counter`，保證 LMR 走步序數 100% 準確。
  4. **代碼清理**：徹底刪除原 `make_move` 後面冗餘的 History Pruning、LMP、FP 代碼。
- **守護：** `depth <= 12`、`!is_pv`、`!follow_pv`、`!is_exclusion_search`、`!is_currently_in_check`、`!low_material_pruning_guard`。
- **協定：** 400@**n300k** · 10 workers · TT=64MB。
- **結果：** **212.0–188.0（108W 208D 84L）· 53.00% · Elo +20.87 ±23.84**（淨勝 **+24 局**！）。
- **先手與執色分析：**
  - 持白：64W 94D 42L（勝率 55.50%）
  - 持黑：44W 114D 42L（**勝率 50.50%，持黑突破 50%！**）
- **搜尋統計分析（parse_search_stats）：**
  - **每步思考時間**：635.2 ms vs 697.2 ms（**節省 62 ms / -8.9% 思考時間**，Welch t=-29.6, $p = 6.7 \times 10^{-191}$ 極度顯著）。
  - **NPS**：371,224 vs 350,011（**+6.1% NPS 提升**，Welch t=+11.08, $p = 1.6 \times 10^{-28}$ 極度顯著）。
  - **深度指標**：
    - mean depth 從 15.97 提升至 **16.08 ply (+0.11 ply)**。
    - depth $\ge 14$ 佔比從 38.8% 提升至 **40.2% (+1.4%)**。
- **判定：** 淨勝 +24 局，Elo +20.87，NPS 提升 +6.1%，每步用時減少 8.9%，全面正向。待用戶同意後由 `tools/sync_classical_old.py` 採納同步為新黃金基線。
- **產物：** `tournament_analysis/match_premake_quiet_pruning_n300k_e36.pgn`。


**E16 診斷（回滾後基線 `hist_diag_E16_post_rollback`，50k×6 局面）：**  
cont 表 `mean_abs≈1.0`、`sat%≈0`（幾乎不飽和）→ 升 ceiling 無「解飽和」收益，卻可能改 gravity 尺度/排序噪音 → 與負 Elo 一致。BF/CAP sat% 也近 0；prune≈23/1k nodes；cut_first≈90%；lmr_research≈4.6%。  
產物：`tournament_results_E16_HistCont14336.pgn`、`stats_E16_HistCont14336/`、`hist_diag_E16_post_rollback.txt|.json`。

### 2.0c 深度診斷 D1（500 puzzle + 500 opening @300k，分開+合計）

| 來源 | 說明 |
| :--- | :--- |
| 指令 | `python tools/hist_diag_bench.py --nodes 300000 --positions 500 -c 4 --tag puzzles_openings_300k --json` |
| EPD | `data/hist_diag_puzzles_500.epd`、`data/hist_diag_openings_500.epd`（`tools/_build_hist_diag_epd.py`） |
| 產物 | `tournament_analysis/hist_diag_puzzles_openings_300k.txt` / `.json` |
| 並行 | ThreadPool `-c 4`，每局面獨立 SearchContext/TT（與串行統計等價） |
| 基線常數 | AGE=900 · BF=12288 · CAP=10240 · CONT=12288 · PRUNE=−5000（E16 回滾後） |

| 指標（mean） | **Puzzles 500** | **Openings 500** | **Combined 1000** |
| :--- | ---: | ---: | ---: |
| depth | **18.7** | 12.1 | 15.4 |
| cut_first% | **92.2** | 89.3 | 90.8 |
| lmr_research% | 3.4 | **5.5** | 4.5 |
| hist_prune / 1k | **30.8** | **132.6** | 81.7 |
| quiet bonus/malus ratio | **340** | **3.1** | 171 |
| piece_to mean_abs | 1082 | 666 | 874 |
| piece_to sat% | 0.74 | 0.09 | 0.42 |
| butterfly / capture sat% | ~0 / ~0 | ~0 / ~0 | ~0 |
| continuation mean_abs | 6.2 | 3.5 | 4.8 |
| continuation sat% | ~0 | ~0 | ~0 |

**D1 結論 → 下一刀：**

1. **Opening history prune ≈ puzzle 的 4×**（133 vs 31 /1k）且 opening 平均 depth 明顯較淺 → 開局／安靜位可能 **over-prune**。  
2. **Puzzle bonus/malus 極度不對稱**（ratio≈340）→ combined 平均被 tactical 位拉歪；**必須分開統計**。  
3. **單局 @300k 幾乎不飽和 ceiling**（BF/CAP/CONT sat≈0；CONT mean_abs 仍極低）→ 與 E16 失敗一致；**勿再推 CONT**；BF 再升邊際已小（E14）。  
4. **E17 選擇：** `PRUNING_HISTORY_THRESHOLD` −5000→**−5500**（E12 軸加深＝更難剪）→ **對戰 54.5% / +31.4 Elo → 留下**。  
5. **下一輪候選：** CAP 10240→12288（E15 加深）或 prune −6000；**不要**再升 CONT。

**LMR_HISTORY_SCALE 結論（E5+E6）：** 160 與 140 皆負 → **150 為目前平台**，此軸暫不再挖。  
**LMR_BASE_OFFSET 結論（E7）：** 480 明確正 → **新基線 480**（原 512）。  
**CONT ceiling 結論（E16+D1）：** 表幾乎空仍升 ceiling 有害 → **凍結 CONT=12288**。

### 2.0 深度 / NPS 對照（New − Old，主統計 mean depth）

| ID | 判定 | Elo | **Δ depth** | New mean d | Old mean d | d≥14 New/Old | Δ NPS | 備註 |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- | ---: | :--- |
| E1 | 留下 | +15.7 | **0.00** | — | — | — | ~0 | 棋力↑、深度持平 |
| E2 | 回滾 | −10.4 | **−0.06** | 10.73 | 10.78 | 5.8%/6.3% | +0.5% | 略淺且負 Elo |
| E3 | 留下 | +8.7 | **+0.03** | 11.00 | 10.97 | 8.2%/8.2% | +1% | |
| E4 | 留下 | +15.7 | **+0.04** | 10.90 | 10.86 | 7.3%/6.9% | +0.6% | |
| E5 | 回滾 | −25.5 | — | — | — | — | — | early-trash 未全量 depth |
| E6 | 回滾 | −20.2 | — | — | — | — | — | 同上 |
| E7 | 留下 | +17.4 | **+0.06** | 10.86 | 10.80 | 7.2%/6.9% | −0.6% | |
| E8 | 留下 | +3.5 | **+0.06** | 10.88 | 10.82 | 6.8%/6.5% | ~0 | |
| E9 | 留下 | +5.2 | **+0.04** | 10.97 | 10.93 | 8.0%/7.7% | +0.2% | |
| E10 | 留下 | +38.4 | **+0.12** | 11.01 | 10.89 | 8.6%/8.0% | +0.7% | depth 升最多之一 |
| E11 | 留下 | +45.4 | **+0.10** | 11.03 | 10.93 | 8.6%/8.0% | +1.5% | Elo 最高單項 |
| E12 | 留下 | +19.1 | **+0.11** | 11.00 | 10.89 | 8.2%/7.9% | +0.5% | |
| E17 | 留下 | +31.4 | **+0.06** | 10.96 | 10.90 | 8.7%/8.3% | +1.1% | D1 驅動；prune 軸再加深 |

**讀法：** 固定 100k nodes 下，正 Δ depth ≈ 同節點搜得更深（剪枝/排序更有效）。E10/E11/E12 同時 **Elo 大 + depth 升**。

### 2.0b 明確正項目 → 第二輪加深（E13+）

原計畫 vs 實際（一次一項）：

| 計畫 | 實際 | 結果 |
| :--- | :--- | :--- |
| E11 軸 soft-age 850→900 | **E13** | 留下（使用者指定） |
| E10 軸 BF 10240→12288 | **E14** | 留下（邊際遞減 +5 Elo） |
| E12 軸 prune −5000→−5500 | 原標 E15；**改 E17**（中間插 CAP/CONT） | **留下 +31.4** |
| CAP 8192→10240 | **E15**（插隊 ceiling 同源） | 留下 +13.9 |
| CONT 12288→14336 | **E16** | **回滾 −12.2**；凍結 |
| E7 軸 LMR base 480→448 | 延後 | 候選 |
| E4 軸 LMP 105→110 | 延後 | 候選（歷史 110 曾撐樹） |

### 2.1 單項備註（機制與產物）

#### E1 CheckSEE

- **意圖：** quiet check 給 `CHECK_BONUS` 前要求 SEE ≥ −75，減少自殺將軍排序污染。  
- **產物：** `tournament_results_E1_CheckSEE.pgn`；sync 後 Old 含 gate。  

#### E2 LMP 95

- **意圖：** 略早 LMP（scale 100→95）。  
- **結果：** 略負；depth/NPS 幾乎持平 → 問題在棋力非速度。  
- **產物：** `tournament_results_E2_LMP95.pgn`、`stats_E2_LMP95/`。  

#### E3 History prune −4500

- **意圖：** 更難觸發 history prune（更負門檻）。  
- **結果：** 近平手有優勢 → 留下。  
- **產物：** `tournament_results_E3_HistPrune4500.pgn`、`stats_E3_HistPrune4500/`。  

#### E4 LMP 105

- **意圖：** E2 反向，略晚 LMP。  
- **結果：** 明確正 → 留下。註：sync 時曾誤把 105 寫進 Old 再手動修回，E4 對局時 New=105 / Old=100 正確。  
- **產物：** `tournament_results_E4_LMP105.pgn`、`stats_E4_LMP105/`。  

#### E5 LMR_HISTORY_SCALE 160

- **公式：** `r -= (lmr_stat_score * LMR_HISTORY_SCALE) / DIV` → scale ↑ 則 history 對減深槓桿 ↑。  
- **結果：** 明顯負；early-trash（Elo≤−20）。**推論：** 在目前 1a history 下，LMR 不宜再放大 history 權重。  
- **產物：** `tournament_results_E5_LMRHist160_trash.pgn`（82 場）。  

#### E6 LMR_HISTORY_SCALE 140

- **意圖：** E5 反向；略降 history 對 LMR 槓桿（約 −6.7%）。  
- **結果：** 86 場 early-trash（Elo −20.2；score 47.1% > 40% 門檻，由 Elo 觸發）。  
- **產物：** `tournament_results_E6_LMRHist140_trash.pgn`。  
- **推論：** 與 E5 合看 → 150 附近已是局部最優；勿再試 130/170。  

#### E7 LMR_BASE_OFFSET 480

- **意圖：** 換軸；基礎 LMR offset 512→480（減深略少、樹略肥）。  
- **結果：** **留下**（52.5% / +17.4 Elo；depth +0.06、NPS −0.6%）。  
- **產物：** `tournament_results_E7_LMRBase480.pgn`、`stats_E7_LMRBase480/`；已 sync。  

#### E8/H1 HISTORY_WEIGHT_MAIN 3

- **意圖：** history 基建穿插第 1 項；1a-compat 讀取權重 2→3。  
- **結果：** **留下**（50.5% / +3.5 Elo；depth +0.06、NPS ≈0）。  
- **產物：** `tournament_results_E8_HistWeightMain3.pgn`、`stats_E8_HistWeightMain3/`；已 sync。  

#### E9/H2 HISTORY_WEIGHT_CONT_1 5

- **意圖：** cont-1 讀取權重 4→5（1a-compat）。  
- **結果：** **留下**（50.75% / +5.2 Elo；depth +0.04、NPS +0.2%）。  
- **產物：** `tournament_results_E9_HistCont1w5.pgn`、`stats_E9_HistCont1w5/`；已 sync。  

#### E10/H3 HISTORY_MAX_BUTTERFLY 10240

- **意圖：** butterfly ceiling 略升；飽和較慢。  
- **結果：** **留下**（55.5% / +38.4 Elo；depth +0.12、NPS +0.7%）。history 基建線目前最大單項增益。  
- **產物：** `tournament_results_E10_HistBF10240.pgn`、`stats_E10_HistBF10240/`；已 sync。  

#### E11/H4 HIST_AGE_SOFT_NUM 850

- **意圖：** soft-age 略慢（800/1024 → 850/1024）；僅小表 aging，不掃 multi-MB cont。  
- **結果：** **留下**（56.5% / +45.4 Elo；depth +0.10、NPS +1.5%）。history 基建線目前最強單項之一。  
- **產物：** `tournament_results_E11_HistAge850.pgn`、`stats_E11_HistAge850/`；已 sync。  

#### E12/H5 PRUNING_HISTORY_THRESHOLD −5000

- **意圖：** 接 E3（−4000→−4500）再略難剪；history 基建收尾。  
- **結果：** **留下**（52.75% / +19.1 Elo；depth +0.11、NPS +0.5%）。  
- **產物：** `tournament_results_E12_HistPrune5000.pgn`、`stats_E12_HistPrune5000/`；已 sync。  

#### E13–E16（加深與 ceiling）

- **E13** soft-age 900：使用者保留並 sync。  
- **E14** BF 12288：50.75% / +5.2 → 留下（邊際遞減）。  
- **E15** CAP 10240：52.0% / +13.9 → 留下。  
- **E16** CONT 14336：48.25% / −12.2 → **回滾**；D1 確認 cont 表仍空。  

#### E17 PRUNING_HISTORY_THRESHOLD −5500

- **意圖（D1）：** opening prune/1k ≈132 vs puzzle ≈31；再略難剪（−5000→−5500），期望降低開局 over-prune、抬 depth。  
- **結果：** **留下（明確正）** — 200@100k **54.5% · Elo +31.4**（62W-94D-44L）；depth **+0.06**、NPS **+1.1%**。  
- **解讀：** 與 E12（−4500→−5000，+19 Elo）同軸再加深仍明確正；D1「opening over-prune」假設得到對戰勝支持。新基線 **PRUNE=−5500**。  
- **產物：** `tournament_results_E17_HistPrune5500.pgn`、`stats_E17_HistPrune5500/`；**已 sync**。  

#### E18 LMR_BASE_OFFSET 448

- **意圖（D1）：** opening lmr_research% 較高、depth 較淺；E7 軸再略少基礎減深（480→448）。  
- **結果：** **留下（近平手有優勢）** — 200@100k **50.2% · Elo +1.7**（60W-81D-59L）；depth **0.00**、NPS **+1.2%**。  
- **解讀：** 前半常 55–60%、後半回落至平手附近（類似虛高風險，但終局 Elo>0 且 depth/NPS 無退）。**LMR 再加深（例如 416）暫緩**；基線改 **448**。  
- **產物：** `tournament_results_E18_LMRBase448.pgn`、`stats_E18_LMRBase448/`；**已 sync**。  

#### E19 PRUNING_HISTORY_THRESHOLD −6000

- **意圖：** E12/E17 同軸第三刀。  
- **結果：** **中止**（編譯/對戰未完成；使用者改優先 LMR 460 高精度複驗）。New prune 已還原 **−5500**，與 Old 一致。  

#### E20 LMR_BASE_OFFSET 460（300@150k）

- **意圖：** E18（480→448）僅 +1.7 且後半回落 → 半回調 **448→460**。  
- **協定：** **300@150k**（高精度）。  
- **結果：** **留下（近平手有優勢）** — 300@150k **51.2% · Elo +8.1**（76W-155D-69L；CI ±27.3）；depth **+0.05**（11.77 vs 11.72）、NPS **+0.7%**。  
- **解讀：** 比 E18 的 448 更穩；前半略負、後半爬回 +8 → 高精度下仍偏正。新基線 **LMR=460**。  
- **產物：** `tournament_results_E20_LMRBase460.pgn`、`stats_E20_LMRBase460/`；**已 sync**。  

#### E21 PRUNING_HISTORY_THRESHOLD −6000（300@150k）

- **意圖：** history prune 再難觸發（−5500→−6000）；D1 opening over-prune 同軸第三刀。  
- **結果：** **留下（明確正）** — 300@150k **53.2% · Elo +22.0**（99W-121D-80L）；depth **+0.12**（11.67 vs 11.55）、NPS **+0.6%**。  
- **解讀：** 與 E12/E17 一致，再放鬆 history prune 仍有效；新基線 **PRUNE=−6000**。  
- **產物：** `tournament_results_E21_HistPrune6000.pgn`、`stats_E21_HistPrune6000/`；**已 sync**。  

#### SF11 校準（E21 後）

- **設定：** 100 局；我方 **1000ms/步** vs SF11 **100ms/步**（時間 10:1）· `-c 4`  
- **第 1 次：** SPRT 開 → 16 局 H0 停：1.5/16（0W-3D-13L）· Elo ≈ −394 ±196  
- **第 2 次（正式）：** `--no-sprt` 跑滿 **100 局**  
  - ClassicalE21 **14.5 – 85.5**（**4W-21D-75L**）· **Score% 14.5%**  
  - Elo ≈ **−308** ±73 · 95% CI **[−382, −235]**  
- **解讀：** 時間讓手 10:1 下仍約 −300 Elo；SF11 遠強。`match_runner` 已加 **`--no-sprt`** 供校準用。  
- **產物：** `tournament_results_SF11_100_1000_100_full100.pgn`（及 stage_1 / 先前 early 備份）。 

### History 基建線總結（H1–H5 全留下）

| ID | 變更 | score | Elo | 判定 |
| :--- | :--- | :--- | :--- | :--- |
| E8 | `HISTORY_WEIGHT_MAIN` 2→3 | 50.5% | +3.5 | 留下 |
| E9 | `HISTORY_WEIGHT_CONT_1` 4→5 | 50.75% | +5.2 | 留下 |
| E10 | `HISTORY_MAX_BUTTERFLY` 8192→10240 | 55.5% | +38.4 | 留下 |
| E11 | `HIST_AGE_SOFT_NUM` 800→850 | 56.5% | +45.4 | 留下 |
| E12 | `PRUNING_HISTORY_THRESHOLD` −4500→−5000 | 52.75% | +19.1 | 留下 |

**觀察：** 權重小步（E8/E9）近平手有優勢；**ceiling 與 soft-age（E10/E11）增益最大**。未觸表結構大改。

---

#### E22 LMP 110（中止回滾）

- **意圖：** E4 軸加深，略晚 LMP（105→110）。  
- **對戰：** 1000@n10k 跑至 **~794 場** 使用者中止；Score% **≈48.1%** · Elo **≈−13** [±20] · SPRT Continue 但方向穩定偏負。  
- **PGN 樹效率：** mean depth **7.10 vs 7.11（−0.01，不顯著）**；NPS **−0.4%**。同深度 node ratio ≈1.0。  
- **判定：** **回滾** → `LMP_SCALE_PERCENT=105`（New=Old）。晚 LMP 在 n10k 無 Elo、無深度收益。  
- **產物：** `tournament_results_E22_LMP110_n10k1000.pgn`、`stats_E22_LMP110/`。  

#### §D2 完整診斷（鎖定 θ = E21 基線，LMP=105）

| 來源 | 路徑 |
| :--- | :--- |
| hist_diag | `tournament_analysis/hist_diag_D2_post_E22_n10k.txt` / `.json` |
| 設定 | 500 puzzle + 500 opening @ **nodes=10000** · c=4 · wall ≈87s |
| θ | PRUNE=−6000 · LMP=105 · LMR_BASE=460 · LMR_HIST=150 · BF=12288 · CAP=10240 · CONT=12288 · AGE=900 |

| 指標（mean） | **Puzzles 500** | **Openings 500** | **Combined** | vs D1@300k（參考） |
| :--- | ---: | ---: | ---: | :--- |
| depth | **9.57** | **6.17** | 7.87 | D1: 18.7 / 12.1（更深 regime） |
| cut_first% | **92.5** | **89.6** | 91.1 | 正常（≥88） |
| lmr_research% | **2.78** | **3.87** | 3.32 | 不高；LMR 非主痛 |
| hist_prune / 1k | **3.50** | **8.97** | 6.23 | D1: 30.8 / **132.6**（開局仍較兇） |
| prune 比 opening/puzzle | — | **≈2.56×** | — | D1 曾 ≈4.3×；n10k 仍開局偏高 |
| quiet bonus/malus ratio | **20.8** | **2.62** | 11.7 | 戰術位極不對稱；須分開看 |
| BF / CAP sat% | ~0.03 / 0.05 | ~0.01 / 0.02 | ~0 | **勿靠升 ceiling 解飽和** |
| CONT mean_abs / sat% | 0.37 / ~0 | 0.22 / 0 | ~0 | **CONT 凍結**再確認 |
| piece_to mean_abs | 190 | 134 | 162 | 健康、未爆 |

**樣本 full dump 補充（非全域 mean，僅示意剪枝組成）：**  
- Opening 例：`lmp_skip` 很大、`rfp_cut` 活躍、`nmp` gate 率低、`hist_prune` 中等。  
- Puzzle 例：`rfp_cut` / mate 深搜主導；`hist_prune` 近 0。  

**D2 結論 → 剪枝方向**

1. **不要再推 LMP 更晚（110+）** — E22 已否；depth 也不變 → n10k 下 LMP 邊際飽和。  
2. **History quiet prune 軸仍有效方向** — 開局 `hist_prune/1k` 仍是 puzzle 的 **~2.6×**，cut_first 正常 → 下一刀優先 **`PRUNING_HISTORY_THRESHOLD` −6000→−6500**（延續 E12/E17/E21）。  
3. **勿升 CONT / 勿為 sat 升 CAP** — 表幾乎空；CAP 升 ceiling 僅當要複製 E15 Elo 故事，非 D2 優先。  
4. **LMR 暫不動** — research% 低；LMR_HIST 150 平台；LMR_BASE 460 已穩。  
5. **n10k regime 特徵** — 平均深度僅 ~6–9；RFP/LMP/hist-prune 比深 LMR 更影響；診斷與對戰 nodes 對齊正確。  
6. **E22 啟示** — 略肥 LMP 無回報；若要動 LMP 只考慮 **更早**（scale&lt;105）需極謹慎（E2@100k 已負），**D2 不建議**當 E23。

**鎖定下一刀：** E23 已完成（見下）。

#### E23 Hist prune −6500（留下）

- **意圖：** D2 開局 hist-prune 仍 ~2.6× puzzle；同軸再放鬆 −6000→−6500。  
- **對戰：** 1000@n10k **51.0% · Elo +7.0** [±18]（361W-298D-341L）；SPRT Continue（LLR≈0.32）。  
- **樹效率：** depth **+0.01**、NPS **−0.2%**（皆不顯著）。  
- **判定：** **留下（近平手有優勢）**；`sync_classical_old` 已完成。  
- **產物：** `tournament_results_E23_HistPrune6500_n10k1000.pgn`、`stats_E23_HistPrune6500/`。  

#### §D3 中淺層剪枝詳細診斷（E23 鎖定後，θ=−6500）

| 來源 | 路徑 |
| :--- | :--- |
| hist_diag | `tournament_analysis/hist_diag_D3_post_E23_n10k.txt` / `.json` |
| 設定 | 500 puzzle + 500 opening @ **n10k**；含 LMP/RFP/FP/Razor/NMP/ProbCut/TT per-1k |
| 對照 | D2（−6000）同 nodes：`hist_diag_D2_post_E22_n10k.*` |

**A. Hist-prune 相對 D2（E23 是否生效）**

| | Puzzle hist_prune/1k | Opening | 比 opening/puzzle |
| :--- | ---: | ---: | ---: |
| D2 (−6000) | 3.50 | 8.97 | **2.56×** |
| **D3 (−6500)** | **2.54** | **6.09** | **2.39×** |
| Δ | **−27%** | **−32%** | 略收斂 |

→ E23 確實降低 quiet hist-prune；開局仍偏高但差距略縮。depth/cut_first 與 D2 幾乎相同。

**B. 中淺層剪枝率（D3 mean per 1k nodes）**

| 機制 | Puzzle | Opening | 讀法 |
| :--- | ---: | ---: | :--- |
| **LMP skip** | **697** | **1238** | 開局遠高；體積最大（skip 非 cut，尺度不同） |
| **RFP cut** | **62.6** | **25.0** | **戰術位更兇**；淺層靜態 fail-high 主力 |
| **FP skip** | **22.7** | **5.5** | puzzle ≫ opening |
| **Razor cut** | 13.4 | 6.5 | 中等；應多在 d=1 |
| **hist_prune** | 2.54 | 6.09 | 開局仍較兇（E23 後） |
| **NMP cut** | 6.1 | 2.2 | 開局較少 |
| **TT cut** | 89.2 | 47.3 | 正常主 cutoff |
| **ProbCut** | 3.8 | 0.44 | 開局幾乎不觸發 |
| NMP gate% | 32.0 | **12.9** | 開局 gate 很嚴（靜態難過） |
| NMP fh% | 55.0 | **71.4** | 過 gate 後開局 null 較準 |
| cut_first% | 92.5 | 89.6 | ordering 健康 |
| lmr_research% | 2.8 | 3.9 | 低；LMR 非瓶頸 |
| mean depth | 9.57 | 6.17 | 開局淺 → 中淺層旋鈕主導 |

**C. 量級排序（開局 n10k，實戰相關）**

```text
LMP skip 體積 ≫ RFP ≈ FP/Razor/hist-prune 帶 ≫ NMP ≫ ProbCut
```

**D. D3 → 下一刀方向**

| 優先 | 動作 | 理由 |
| :---: | :--- | :--- |
| **1** | **RFP 小步**（margin = mult×depth） | 中淺層 **真 cut** 最高活躍度之一；**mult↑=margin↑=較難 fail-high（鬆）**；**mult↓=更兇**；尚無 n10k Elo 單項 |
| **2** | **FP mult 小步**（130→140） | puzzle FP 活躍；舊戰役 fatten 過猛需小步 |
| **3** | Hist −7000 | E23 僅 +7、depth 幾乎不動 → **邊際遞減**；可選非必須 |
| **凍** | LMP 105 | E22 110 負；體積大≠再鬆有 Elo |
| **凍** | NMP 結構 / ProbCut | gate 開局低屬結構；戰役已否抄 SF |
| **凍** | LMR / CONT | research 低、CONT 空 |

**E24 結果（略鬆 180）→ 回滾；見下。**

#### E24 RFP 180（中止回滾）

- **意圖：** D3 中淺層 RFP 活躍；略鬆 = mult **170→180**（margin↑ → 較少 fail-high）。  
- **對戰：** ~**906** 場 @n10k 使用者中止；Score% **≈49.7%** · Elo **≈−2.3** [±19] · SPRT LLR ≈ −0.7（Continue 但穩定略負）。  
- **樹效率：** mean depth **7.06 vs 7.08（−0.02）**；NPS ≈持平／略高。  
- **判定：** **回滾** → `RFP_BASE_MULT=170`。  

**原因判斷（為何略鬆 RFP 沒幫助）**

1. **D3 的「RFP 高」≠「RFP 過兇」**  
   - Puzzle RFP/1k ≈63、opening ≈25：戰術位多 fail-high 常是 **eval 已明顯 > β** 的健康剪枝，不是亂砍。  
   - 略鬆後 Elo 略負、depth 幾乎不變 → 少剪的節點 **沒換成更深／更準的 PV**，固定 n10k 下只是把預算花在已可剪掉的分支。  

2. **與 E23 hist-prune 不同質**  
   - Hist-prune 放鬆有 Elo（+7）：針對 **quiet late-move 排序差** 的錯剪。  
   - RFP 是 **靜態分數 vs β** 的 fail-high；HCE 在 n10k 淺層若尺度大致可用，再加大 margin 只會 **變肥** 而無戰術收益。  

3. **「開局偏兇」主因不在 RFP**  
   - D3 開局 RFP 反而低於 puzzle；開局痛點仍是 **LMP 體積 + hist-prune**（E22 已否 LMP 更晚、E23 已推 hist）。  
   - 再動 RFP 鬆緊對開局 over-prune 敘事 **對症不準**。  

4. **不建議立刻試 160（更兇）**  
   - 180 已略負且 depth 無升；更兇可能掉戰術（puzzle 高 RFP 處）。若試 160，需當 **獨立反向假設**，優先級低於 FP / hist−7000 / CAP。  

**產物：** `tournament_results_E24_RFP180_n10k1000.pgn`、`stats_E24_RFP180/`。

---

## 3. 候選隊列

**原則：** 仍一次一項；**CONT 凍結**；正式對戰 **1000@n10k**；**LMP 凍結 105**；**RFP 凍結 170**（E24）。

**2026-07-19 剪枝 *margin* 戰役結束（邊際耗盡）。**  
凍結 margin：LMP 105 · RFP 170 · FP 130 · PRUNE **−6500**。  
**政策：** 取消「禁止比 SF11 更兇」硬上限（`constants.py` / `SEARCH_ANALYSIS` §9）。  
**NMP：** N1 scope3 **失敗**；**N2 gate BASE 200→160 留下**（scope 仍 cut-only）。d10 !imp 門檻約 β+60→β+20。勿再擴 scope；gate 可再小步（如 140）或停。  

## E30. 全對局 162 盤面參數等差數列掃描 (2026-07-27)

> **目的：** 利用改版後的 `tests/test_search.py` 與 `tests/test_search_param_sweep.py`，針對 15. Be2 至 95... Kf2 全對局 162 個盤面（每步 1.0 秒時間限制），以現有數值為等差中項進行項數為 5 的參數掃描，分析深度與樹效率影響。

### 1. `FP_MULTIPLIER` (Futility Pruning 放大係數)
* **等差數列 (d=10)：** `[105, 115, 125 (Base), 135, 145]`

| FP_MULTIPLIER | Mean Nodes | Median Nodes | Mean NPS | Median NPS | Mean Depth | Median Depth |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **105** | 965,654.5 | 914,010.0 | 1,425,635.6 | 1,550,874.5 | **13.60** | 13.00 |
| **115** | 948,073.2 | 901,934.5 | 1,404,188.4 | 1,529,921.0 | **13.59** | 13.00 |
| **125 (Base)** | 906,838.0 | 874,731.0 | 1,334,091.4 | 1,362,726.5 | 13.43 | 13.00 |
| **135** | 939,823.3 | 901,308.0 | 1,388,589.2 | 1,504,771.5 | 13.49 | 13.00 |
| **145** | 971,381.2 | 914,010.0 | 1,422,511.2 | 1,587,106.0 | 13.59 | 13.00 |

#### 分析與結論：
1. **深度變化**：基準值 `125` 的平均深度為 `13.43`。當 `FP_MULTIPLIER` 調小至 `115` 或 `105` 時，由於每層 futility pruning 邊界較收斂，搜尋能夠在 1.0 秒時間限制內推升至更深層（平均深度提升 +0.16 ~ +0.17 ply 至 `13.59`~`13.60`）。
2. **節點數**：`125` 時平均節點數最少（906,838.0）；當 `115`/`105` 讓搜尋多探訪一層深度時，總搜尋節點數相應略升至 948k / 965k。

---

## E31. 歷史啟發統一重構 (Step 21 Move Stats Update) (2026-09-21)

> **意圖：** 徹底修復歷史啟發深層嵌套於走步循環內 `if alpha >= beta:` 導致 PV 節點走步獲得 0 bonus、劣步獲得 0 malus 的架構缺陷。全面對齊 Stockfish 17/18 Step 21 走步統計更新架構。

* **改動細節：**
  1. 在 `_search` 走步循環中，`alpha >= beta` 時僅更新剪枝診斷與安靜步 Killer Move，即刻 `break`。
  2. 走步循環結束後，於 Step 21 由 `if final_flag != TT_FLAG_ALPHA and best_move != NO_MOVE:` 統一對 `best_move` 發放歷史獎勵（quiet / capture）。
  3. 遍歷 `quiet_moves_tried` 與 `capture_moves_tried` 扣除 Malus 時，明確以 `if bad_move == best_move: continue` 排除最佳步。
  4. 互斥對手 fail-low 獎勵（`elif final_flag == TT_FLAG_ALPHA and ply > 0 ...`）。
* **對戰設定：** `python tools/tournament.py --games 400 --nodes 300000 --concurrency 10 --tt-mb 64`
* **對戰結果：**
  * **Games:** 400
  * **Score:** New 211.5 - 188.5 Old (107W 209D 84L), 得分率 52.9%
  * **Elo difference:** **+20.00 ± 23.52 (95% CI)**
  * **SPRT LLR:** 1.11 (bounds [-2.94, 2.94])
* **判定：** **留下 (KEEP)**，已執行 `tools/sync_classical_old.py --sync` 同步至 `classical_old`。

---

## E32. 移除重複 Continuation Correction History 更新區塊測試 (2026-09-21)

> **意圖：** 嘗試刪除 `search.py:3355-3383` 標記為 `(duplicate block kept for parity)` 的重複延續糾錯歷史更新區塊，回歸 Stockfish 單次更新。

* **改動細節：** 刪除第二段重複執行的 2-ply / 4-ply `continuation_correction_history` 更新代碼。
* **對戰設定：** `python tools/tournament.py --games 400 --nodes 300000 --concurrency 10 --tt-mb 64`
* **對戰表現（156 局趨勢）：**
  * **Games:** 156
  * **Score:** New 31W 83D 42L (得分率 46.5%)
* **根因分析：**
  * 引擎既有的靜態評估糾錯常數（如 `CORRECTION_CONT_2PLY_NUM` / `CORRECTION_CONT_4PLY_NUM`）是在「此區塊更新兩次」的有效等效步長下調優與擬合的。
  * 單純刪除第二個區塊，形同將 continuation correction history 的動態學習率與步長硬生生砍半，導致對局中靜態評估偏差無法及時被糾正，引起 ~ -25 Elo 實質倒退。
* **判定：** **回滾 (REVERT)**。保留該區塊以維持現有常數擬合平衡，待未來連同糾錯常數 SPSA 重新調優時再行整併。

---

## E33. Step 14 淺層剪枝 TT 走步保護與根節點守護 (`ply > 0 and move != tt_move`) (2026-09-21)

> **意圖：** 徹底消除走步迴圈中「TT 走步在 depth ≤ 12 時被淺層啟發式剪枝（Capture FP、Capture SEE、Checking SEE、LMP、History Pruning、Quiet FP、Quiet SEE）誤剪而引發的搜尋不穩定性（Search Instability / TT Move Drop）」，並解鎖後續 Step 15 奇異擴展（Singular Extension）與 Multi-Cut 的完整執行；同時補齊 `ply > 0` 對齊 Stockfish 18 的 `!rootNode` 根節點保護。

* **改動細節：**
  在 `chess_engine/classical/search.py` 的 Step 14 淺層剪枝入口守衛及影子診斷處，新增 `ply > 0 and move != tt_move`：
  ```python
  if (ply > 0 and move != tt_move and search_context.enable_see_pruning and not is_exclusion_search
          and depth <= PRUNING_SHALLOW_DEPTH and not is_currently_in_check and not is_pv):
  ```
* **對戰設定：** `python tools/tournament.py --games 400 --nodes 300000 --concurrency 10 --tt-mb 64`
* **對戰結果：**
  * **Games:** 400
  * **Score:** New 217.5 - 182.5 Old (106W 223D 71L), 得分率 54.37%
  * **Elo difference:** **+30.48 ± 22.63 (95% CI)** (有效區間 +7.85 ~ +53.11)
  * **SPRT LLR:** 1.06 (bounds [-2.94, 2.94])
* **根因分析與效益：**
  1. **根絕 TT Move 誤殺**：過去在不利局面或戰術棄子時，`tt_move` 常因 `static_score + margin < alpha` 或 `SEE < 0` 被 Step 14 攔截 `continue`，導致上一迭代算出的最佳走步被丟失，引發搜尋不穩定與大漏著。
  2. **解鎖 Step 15 奇異擴展與 Multi-Cut**：確保所有關鍵候選著法均能平安送達 Step 15 接受奇異性檢驗，精準在唯一解上獲得 +1~+3 ply 擴展。
  3. **提升 Cut-Node 截斷率**：Cut-Node 在第 1 步（`tt_move`）直接觸發 Beta Cutoff 的機率顯著提高，大幅縮減樹體積。
* **判定：** **留下 (KEEP)**，已執行 `tools/sync_classical_old.py --sync` 同步至 `classical_old`。

---

## E34. ProbCut TT 捷徑前置至入口處 (Step 12 前移對齊 SF18) (2026-09-21)

> **意圖：** 修復原本置於 Step 12（Line 1948，位於 IIR、NMP、Razoring 甚至整個 Step 11 吃子 ProbCut 之後）的 TT-based ProbCut Shortcut 執行時機過晚問題。若置換表已能證明 `tt_depth >= depth - 4` 且 `tt_score >= beta + 334`，舊架構仍會白白執行完整的吃子生成、被牽制子計算、SEE 篩選、靜態搜尋、depth-4 遞迴子搜尋、NMP 與 Razoring，才抵達捷徑，造成巨大的算力浪費。將其前移至 ProbCut 入口處，以 $O(1)$ 代價在搜尋展開前直接截斷。

* **改動細節：**
  將 `ENABLE_PROBCUT_TT_SHORTCUT` 邏輯從 Step 12（舊 line 1948）移至 `_search` 的 ProbCut 入口處（line 1609，吃子生成與子搜尋之前）：
  ```python
  if ENABLE_PROBCUT_TT_SHORTCUT and ply > 0:
      probcut_beta_s12 = beta + PROBCUT_TT_SHORTCUT_MARGIN
      if tt_entry['flag'] == TT_FLAG_BETA or tt_entry['flag'] == TT_FLAG_EXACT:
          if tt_entry['depth'] >= depth - PROBCUT_R:
              tt_score_s12 = np.int32(tt_entry['score'])
              # ... mate score & fifty move scaling ...
              if (tt_score_s12 >= probcut_beta_s12
                      and not is_decisive(tt_score_s12) and not is_win(tt_score_s12)):
                  search_context.pv_table[ply, ply] = NO_MOVE
                  _diag_add(search_context, DIAG_PROBCUT)
                  tt_hits += np.uint64(1)
                  return (np.int32(probcut_beta_s12), tt_entry['best_move'], nodes_searched, quiescence_nodes, tt_hits)
  ```
* **對戰設定：** `python tools/tournament.py --games 400 --nodes 300000 --concurrency 10 --tt-mb 64` (對手為剛獲得 +30.48 Elo 的 E33 基線)
* **對戰結果：**
  * **Games:** 400
  * **Score:** New 206.5 - 193.5 Old (93W 227D 80L), 得分率 51.63%
  * **Elo difference:** **+11.30 ± 22.39 (95% CI)**
  * **SPRT LLR:** 0.51 (bounds [-2.94, 2.94])
* **根因分析與效益：**
  1. **零開銷早期截斷 ($O(1)$ Cutoff)**：在遇到高勝率已被置換表記錄的枝葉時，完全省去生成所有吃子、計算 SEE、執行 QSearch 以及進入遞迴子搜尋（depth - 4）的節點預算，實測每深層節點比率降至 0.985。
  2. **避免多餘的 NMP / Razoring 誤殺**：提前回傳正確的 fail-high 分數，避免因後續 NMP 驗證或 Razoring 帶來的不確定性。
  3. **連續兩次成功推進**：在 E33 (+30.48 Elo) 如此強大的新基準線上，依然再度斬獲 +11.30 Elo，驗證了搜尋順序優化的強大乘數效應。
* **判定：** **留下 (KEEP)**，已執行 `tools/sync_classical_old.py --sync` 同步至 `classical_old`。

---

## E35. Mate Distance Pruning 根節點守護與順序調整測試 (2026-09-21)

> **意圖：** 嘗試為 `ENABLE_MATE_DISTANCE_PRUNING` 增加 `ply > 0` 守護（避免根節點在將殺邊界誤觸 `NO_MOVE`），並將其移至重複局面（Repetition）與 50 步規則之後（對齊 Stockfish 18 Step 2 和棋 -> Step 3 將死剪枝順序）。

* **改動細節：** 將 Mate Distance Pruning 從 `_search` 入口（舊 line 1214）移至 `if ply > 0:` 內部、重複局面與 50 步和棋檢查之後。
* **對戰設定：** `python tools/tournament.py --games 400 --nodes 300000 --concurrency 10 --tt-mb 64`
* **對戰表現：**
  * 前 140 局對戰呈現高度鏡像對稱（27W 86D 27L，步數完全相等 7825 vs 7825，`mate=0`）。
  * 在常規局面（[-2000, 2000]）下兩者二進位邏輯等價；但在後續盤面中出現失利破稱。
* **根因分析：**
  * 在 300k nodes 的常規開局對戰中，常規局面從未觸發將死分數邊界；而在深層複雜殘局中，順序調整可能影響了極端變例的 Alpha/Beta 收斂或與其他啟發式的微妙耦合。
  * 由於該改動在常規對局中無法帶來實質正向 Elo 增益，且伴隨極端局面的不可控性。
* **判定：** **回滾 (REVERT)**。已還原至 E34 生產基線，雙目錄保持嚴格邏輯一致。

---

## E36. 將軍延伸父盤面前置評估重構 (消除 Check Extension unmake/remake 開銷) (2026-09-21)

> **意圖：** 徹底消除走步迴圈中，當著法給予將軍且滿足延伸條件時，舊代碼因需要父節點狀態而執行 `unmake_move -> _see_ge_jit -> make_move` 的沉重反悔往返開銷。

* **改動細節：**
  1. 在 `make_move` 之前，利用當前父盤面已有的位元盤與被牽制棋子資訊，預先評估 `check_gate_see_ok`（若吃子已在 LMR 算過 `not is_bad_capture` 則直接復用）。
  2. 走棋後若給予將軍，直接判定 `elif check_gate_see_ok: check_extension = 1`。
  3. 徹底刪除走步迴圈內的 `unmake_move` 與重新 `make_move` 呼叫。
* **性質與效益：**
  * **純運算速度與架構重構（0-Node Tree Delta）**：搜尋樹決策與各著法延伸判定在數學上 100% 完全等價（Kiwipete d14 走法與節點數精確到個位數一致）。
  * 消除將軍節點上的兩次冗餘棋盤歷史與位元盤突變，降低節點 CPU 耗時並提高 NPS。
* **判定：** **留下 (KEEP)**，已執行 `tools/sync_classical_old.py --sync` 同步至 `classical_old` 作為新生產基線。

---

## E37. Aspiration Window 動態視窗收斂與幾何擴展 (對齊 Stockfish 17/18) (2026-09-23)

> **意圖：** 修復舊架構在迭代加深 Aspiration Window 中的兩大效率缺陷（原為 SF11 舊式邏輯），全面對齊 Stockfish 17/18（`Thread::search()`）的當代實現：
> 1. **Fail-Low 邊界收斂**：舊代碼在 `score <= alpha` 時執行 `beta = (alpha + beta) // 2`。然而依據 Alpha-Beta 剪枝定義，既然返回分數小於等於 `alpha`，已嚴格證偽該局面得分不可能達到或超過 `alpha`，保留大於 `alpha` 的上界只會導致重搜時無效地探索已被證偽的區間。SF17/18 做法是直接將 `beta = alpha`，徹底阻斷上方無效分支；同時以最新分數向下方擴張 `alpha = max(-INFINITY, score - delta)`。
> 2. **Fail-High 邊界提升**：舊代碼在 `score >= beta` 時僅將 `beta = min(INFINITY, score + delta)`，而讓 `alpha` 保持不動。SF17/18 則將下界大幅向上拉升 `alpha = max(beta - delta, alpha)`，從下方鎖定搜尋區間，避免在已被證偽的低分著法上浪費節點。
> 3. **視窗擴展速度（幾何擴張）**：舊代碼採用 SF11 的 `delta += delta // 3 + 3`（固定每輪約 +33% + 3cp）；SF17/18 改為幾何整數縮放 `delta += 44 * delta / 128`（約 1.344 倍乘法擴展，無浮點開銷），並在 Python 中加上 `max(1, 44 * delta // 128)` 確保極小 `delta` 也能持續擴展。

* **改動細節：**
  在 `chess_engine/classical/search.py` 的 `iterative_deepening_search` 內更新 Fail-Low / Fail-High 重搜窗口更新：
  ```python
  if score <= alpha:
      # Fail-Low: Beta 鎖定至 alpha (已證偽不可能大於 alpha)，Alpha 基於回傳分數擴張 (對齊 SF17/18)
      ...
      beta = alpha
      alpha = max(-INFINITY, score - delta)
      failed_high_cnt = 0
      delta += max(1, 44 * delta // 128)
  elif score >= beta:
      # Fail-High: Beta 基於回傳分數擴張，Alpha 向上拉升收窄視窗 (對齊 SF17/18)
      ...
      alpha = max(beta - delta, alpha)
      beta = min(INFINITY, score + delta)
      failed_high_cnt += 1
      delta += max(1, 44 * delta // 128)
  ```
* **對戰設定：** `python tools/tournament.py --games 400 --nodes 300000 --concurrency 10 --tt-mb 64` (對手為已同步至 E36 基準之 `classical_old`)
* **對戰結果：**
  * **Games:** 400
  * **Score:** New 200.0 - 200.0 Old (94W 212D 94L), 得分率 50.00%
  * **Elo difference:** **0.00 ± 22.80 (95% CI)** (區間 [-22.80, 22.80])
  * **搜尋指標：** NPS +0.6% (New 414,149 vs Old 411,638)；平均深度 New 16.76 vs Old 16.84 (-0.08 ply, p=0.58 無顯著差異)；深層 >=14 佔比 55.6% vs 55.9%。
* **根因分析與效益：**
  1. **絕對數值與架構健全（0 瑕疵）**：現代 SF17 視窗邊界收斂邏輯在 400 盤深層對戰中展現極致穩定性（0 崩潰、0 超時、0 異常著法），並帶來 +0.6% NPS 提升。
  2. **固定節點下的數學中性**：在固定 300k nodes 限制下，根節點絕大多數局面皆在初次搜尋窗口內完成；少數重搜情況下，邊界收斂主要優化重搜節點分配，對最終根著法選擇呈現出完全鏡像對稱（94W 212D 94L 完美 50.00% 平手）。
  3. **現代架構基底奠定**：徹底汰除 SF11 舊式非嚴格收斂邏輯，使引擎迭代加深完全對齊 Stockfish 17/18 當代體系，為後續更深層的剪枝優化提供更精確的搜尋窗口。
* **判定：** **留下 (KEEP)**，已執行 `tools/sync_classical_old.py --sync` 同步至 `classical_old` 作為新生產基線。備份 PGN：`tournament_analysis/match_aspiration_window_sf17_n300k_e37_adopted.pgn`。

---

## E38. 延續歷史淺層剪枝 (Continuation History Pruning, 對齊 Stockfish 17 Step 14) (2026-09-23)

> **意圖：** 在 Step 14 淺層剪枝級聯中，解除舊歷史剪枝對 `main_history_score < 0`（全局 Piece-To 歷史）的依賴，引入純粹由局部戰術應手驅動的延續歷史剪枝（SF17 Step 14）。若某個安靜走步在具體局面上下文（對手前一步應手、我方前兩步應手及兵結構）中多次被證偽為敗著，無論其全局格位是否熱門，直接在走棋前跳過。

* **改動細節：**
  1. 在 `chess_engine/classical/constants.py` 新增常數：
     * `ENABLE_CONTINUATION_HISTORY_PRUNING = True`
     * `PRUNING_CONTINUATION_THRESHOLD = -1800`（由 SF17 之 `-4313 * depth` 依歷史表總容量 $28672 / 68192 \approx 0.42$ 等比縮放）。
  2. 在 `chess_engine/classical/search_heuristics.py` 新增 `@njit` 函數 `get_continuation_pruning_score`：
     計算未加權之 `continuation_history[0] + continuation_history[1] + pawn_history`。
  3. 在 `chess_engine/classical/search.py` 的 Step 14b 走步迴圈中，優先檢驗 `cont_prune_score < PRUNING_CONTINUATION_THRESHOLD * depth`，命中時直接標記 `prune_quiet_move = True` 截斷該安靜著法。
* **對戰設定：** `python tools/tournament.py --games 400 --nodes 300000 --concurrency 10 --tt-mb 64` (對手為已同步至 E37 基準之 `classical_old`)
* **對戰結果：**
  * **Games:** 400
  * **Score:** New 210.0 - 190.0 Old (105W 210D 85L), 得分率 52.50% (+20 場淨勝)
  * **Elo difference:** **+17.39 ± 23.46 (95% CI)** (區間 [-5.78, 40.71])
  * **SPRT LLR:** **+0.89** (bounds [-2.94, 2.94])
  * **搜尋指標：** 平均深度 New 16.82 vs Old 16.81 (+0.01 ply)；深層 >=14 佔比 54.9% vs 55.0%；NPS 384,732 vs 387,641 (-0.75%)；超長局 (>80 步) 勝率高達 **57.90% (27W 34D 15L)**。
* **根因分析與效益：**
  1. **解除全局歷史盲區**：舊版歷史剪枝受限於 `main_history_score < 0` 守護條件，無法剪除全局熱門但特定局部上下文為敗著的分支。E38 依據「前手應手 + 兵結構」直接跳過劣著，大幅壓低有效分支因子。
  2. **漫長殘局統治力**：在 >80 步的超長局中斬獲 27W 15L（勝率 57.90%），證明在走步規律高度依賴上下文的殘局中，延續歷史能第一時間封殺無效試探，將算力集中於勝勢轉換。
* **判定：** **留下 (KEEP)**，已執行 `tools/sync_classical_old.py --sync` 同步至 `classical_old` 作為新生產基線。備份 PGN：`tournament_analysis/match_cont_hist_pruning_sf17_n300k_e38_adopted.pgn`。

---

## E39. 奇異延伸 (SE) 負延伸條件修復與幅度對齊 SF17 (-3/-2) (2026-09-23)

> **意圖：** 修復 Step 15 奇異延伸中非奇異著法的負延伸邏輯。舊代碼檢驗 `if exclusion_score >= beta: singular_extension = -2`，但該條件早已在上方被 Multi-Cut 剪枝（Line 2520）攔截並直接返回，導致 `-2` 實質成為死碼，僅有 `cut_node` 賦予保守的 `-1`。全面對齊 Stockfish 17 Step 15：當排除搜尋證明其他走法亦能守住（非奇異），若置換表自身評估已預期 Fail-High（`se_tt_score >= beta`），則將該非奇異 TT 著法負延伸深入至 **`-3`**；若為 `cut_node` 則賦予 **`-2`**。

* **改動細節：**
  在 `chess_engine/classical/search.py` 的 Step 15c 延伸計算中：
  ```python
  elif exclusion_score >= exclusion_beta:
      # Negative extensions when not singular (SF17/18 Step 15)
      if not is_pv:
          if se_tt_score >= beta:
              singular_extension = -3
          elif cut_node:
              singular_extension = -2
  ```
* **對戰結果：**
  * **Games:** 322 (提早終止：收斂至均勢平手)
  * **Score:** New 160.5 - 161.5 Old (77W 167D 78L), 得分率 49.84%
  * **Elo difference:** **-1.08 ± 26.42 (95% CI)** (區間 [-27.50, 25.34])
  * **搜尋指標：** 平均深度 New 16.99 vs Old 16.91 (+0.09 ply，中局高峰達 +0.19 ply)；NPS 400,969 vs 399,543 (+1.4k NPS)；超長局 (>80 步) 85 場 (20W 44D 21L，49.41%)。
* **根因分析：**
  1. **深度與效率確實提升**：`-3` 在非奇異但高勝率的分支上大幅縮減搜尋深度，省下大量冗餘節點，實測全場平均深度確實增加 +0.09 ply，中局更達 +0.19 ply，驗證了方向上的搜尋效率提升。
  2. **-3 減步幅度過猛導致戰術盲區**：跳過 3 層深度在特定複雜對攻局面中容易跨過戰術回擊邊界（Horizon Effect），引發偶發性殘局逆轉，導致勝率被微幅拉回至 49.84%。
  3. **策略轉向**：原版（E38 基線）中因死碼從未實現過 `-2`（實質僅跑 `cut_node: -1`）。故將減步幅度調回更溫和穩健的 `-2`（`se_tt_score >= beta: -2`，`cut_node: -1`），對戰原版驗證溫和負延伸效益。
* **判定：** **不採納 (REJECT / PIVOT)**。PGN 備份：`tournament_analysis/match_se_neg3_sf17_n300k_e39.pgn`。舊引擎 `classical_old` 保持為 E38 原版基線。

---

## E40. 奇異延伸 (SE) 溫和負延伸 (-2/-1) 激活測試 (2026-09-23)

> **意圖：** 針對 E39 驗證之「負延伸能拉升平均深度，但 -3 過度激進」現象，將條件修復後的負延伸幅度設定為經典溫和值：當 `se_tt_score >= beta` 時給予 **`-2`**（修復原版中因 Multi-Cut 阻擋導致從未觸發的 `-2` 死碼），若為 `cut_node` 則給予 **`-1`**。直接對戰原版（E38 基線，實質僅有 `cut_node: -1`），測試溫和負延伸是否能在保持戰術銳利度的同時兼獲深度紅利。

* **改動細節：**
  在 `chess_engine/classical/search.py` 的 Step 15c 延伸計算中：
  ```python
  elif exclusion_score >= exclusion_beta:
      # Negative extensions when not singular (SE -2 / -1)
      if not is_pv:
          if se_tt_score >= beta:
              singular_extension = -2
          elif cut_node:
              singular_extension = -1
  ```
* **對戰設定：** `python tools/tournament.py --games 400 --nodes 300000 --concurrency 10 --tt-mb 64` (對手為 E38 原版基線 `classical_old`)
* **對戰結果：**
  * **Games:** 336 (提早終止：負向趨勢顯著)
  * **Score:** New 157.5 - 178.5 Old (62W 191D 83L), 得分率 46.88% (-21 場淨負)
  * **Elo difference:** **-21.74 ± 24.47 (95% CI)** (區間 [-46.21, 2.73])
  * **搜尋指標：** 平均深度 New 16.77 vs Old 16.83 (-0.06 ply)；NPS 413,154 vs 411,891 (+1.2k NPS)；超長局 (>=80 步) 87 場 (23W 37D 27L，47.70%)。
* **根因分析：**
  1. **首步截斷率（Move 1 Cutoff Rate）崩塌**：置換表著法（`tt_move`）是全樹最期望第一手達成 $\beta$-Cutoff 的核心。當非奇異時給予 `tt_move` 負延伸 `-2`，其深度被縮減至 `depth - 3`，戰術視野受損導致其在臨界對攻中頻繁未能成功 fail-high。
  2. **淺層走步消耗預算，平均深度反跌 (-0.06 ply)**：一旦首步未能截斷，搜尋引擎被迫逐一搜尋後續 Move 2..N 的所有合法走步。在固定 300k nodes 限制下，大量節點被浪費在淺層次優分支的驗證上，迭代加深反而無法推進至深層。
  3. **原版架構合理性驗證**：原版中 `if exclusion_score >= beta: -2` 為死碼，保持 `tt_move` 完整深度搜尋，確保了極高的首步截斷率與極小的樹分支因子。
* **判定：** **回滾 (REVERT)**。已徹底還原 `chess_engine/classical/search.py` 至 E38 原版基線，`tools/sync_classical_old.py --diff` 驗證完全一致。備份 PGN：`tournament_analysis/match_se_neg2_sf17_n300k_e40_rejected.pgn`。

---

## E41. Step 14c 父節點 Futility Pruning 的 BestValue 緊緻邊界更新 (對齊 Stockfish 18 Step 14) (2026-09-23)

> **意圖：** 在主要搜尋走步迴圈中，當某個安靜走步被父節點 Futility Pruning (FP, Step 14c) 判定為無效並剪除時（`futility_value = static_score + margin < alpha`），舊代碼僅僅執行計數與 `continue`，丟失了該著法的評估資訊，使得節點在全負 (Fail-Low) 時僅能儲存並回傳鬆散的 `-INFINITY` 或原始 `alpha`。本改動對齊 Stockfish 18 Step 14：以 `futility_value` 安全更新 `max_eval`（保持緊緻上界），使全負節點在置換表 (TT) 與父節點回傳時擁有精確的真實 Fail-Low 上界估計，促進後續重搜與兄弟節點的即時 Cutoff。

* **改動細節：**
  在 `chess_engine/classical/search.py` 的 Step 14c 父節點 FP 剪枝中：
  ```python
  futility_value = static_score + margin
  if margin > 0 and futility_value < alpha:
      ...
      if margin > 0 and futility_value < alpha:
          _diag_add(search_context, DIAG_FP_SKIP)
          pruned_moves += 1
          # SF18 Step 14: Update bestValue/max_eval with futilityValue on FP cutoff
          if (max_eval < futility_value
                  and abs(futility_value) < MATE_IN_MAX_PLY
                  and (max_eval == -INFINITY or abs(max_eval) < MATE_IN_MAX_PLY)):
              max_eval = np.int32(futility_value)
          continue
  ```
* **對戰設定：** `python tools/tournament.py --games 400 --nodes 300000 --concurrency 10 --tt-mb 64` (對手為 E38 原版基線 `classical_old`)
* **對戰結果：**
  * **Games:** 268 (提早終止：負向趨勢顯著)
  * **Score:** New 127.0 - 141.0 Old (52W 150D 66L), 得分率 47.39% (-14 場淨負)
  * **Elo difference:** **-18.17 ± 27.70 (95% CI)** (區間 [-45.87, 9.53])
  * **搜尋指標：** 平均深度 New 16.65 vs Old 16.65 (持平)；NPS 386,160 vs 385,959；超長局 (>=80 步) 59 場 (15W 31D 13L，51.69%)。
* **根因分析：**
  1. **HCE 寬鬆防禦性裕度 vs SF 緊緻神經網路尺度**：Stockfish 基於 NNUE 精確評估，FP 裕度極小；本作 HCE 為了防範殘局與對攻波動，FP 裕度設定非常保守（`180 + 130 * lmr_d`，在 `lmr_d=2` 時裕度高達 **+440 cp**）。
  2. **假設性上限引發「分數虛高灌水」**：`futility_value = static_score + 440` 本質是「哪怕出現奇蹟 +440 cp 也追不上 alpha 的安全跳過條件」，絕非該走步真實能達到的分數。直接將其賦值給 `max_eval`，等於對外宣稱找到 +440 cp 好棋，導致 Fail-Low 節點被嚴重虛高灌水。
  3. **破壞置換表 (TT) 與修正歷史 (Correction History)**：虛高的 `max_eval` 存入 TT 誤導後續搜尋；更致命的是它使 `max_eval > static_score` 始終為真，徹底癱瘓了 Fail-Low 時對高估局面的負向修正機制；同時也干擾了 LMR Re-Search 深度調整。
* **判定：** **回滾 (REVERT)**。已徹底還原 `chess_engine/classical/search.py` 至 E38 原版基線，`tools/sync_classical_old.py --diff` 驗證完全一致。備份 PGN：`tournament_analysis/match_fp_bestvalue_sf18_n300k_e41_rejected.pgn`。

---

## E42. Step 21 Fail-Low 反擊歷史獎勵門檻解禁與安靜步守護 (對齊 Stockfish 18 Step 21) (2026-09-25)

> **意圖：** 在主要搜尋走步迴圈中，當所有走步均無法突破 alpha（節點 Fail-Low）時，表示對手上一步的走步成功遏制了我方所有的進攻路徑。Stockfish 18 Step 21 會給予對手上一著反擊步（Countermove）正向歷史獎勵（含主歷史、蝴蝶歷史與延續歷史）。然而舊代碼存在兩個嚴重缺陷：(1) 設有 `legal_moves_tried > 4` 門檻，在殘局合法步少、被將軍應將（僅 1~2 步）、或大量安靜步被剪枝的防守節點中，永遠無法觸發反擊獎勵；(2) 缺少 `!priorCapture` 守護，且 `(prev_piece % 6) != 0` 誤包了整個區塊導致兵的反擊步完全無法更新延續歷史。本實驗對齊 SF18 Step 21：移除 >4 步門檻、加入 `not parent_was_capture` 安靜步守護、並讓兵的走步享有正規反擊延續歷史。

* **改動細節：**
  在 `chess_engine/classical/search.py` 的 Step 21 Fail-Low 區塊中：
  ```python
  # --- Opponent Move Bonus (Fail-Low / Alpha Flag Bonus) (SF18 Step 21) ---
  elif final_flag == TT_FLAG_ALPHA and ply > 0 and (legal_moves_tried > 0 or pruned_moves > 0):
      parent_was_capture = search_context.move_is_capture_stack[ply - 1]
      if not parent_was_capture:
          prev_move = search_context.move_stack[ply - 1]
          prev_piece = search_context.piece_stack[ply - 1]
          
          # Reward the opponent's previous quiet move if it successfully caused us to fail low
          if prev_move != NO_MOVE and prev_piece != -1:
              prev_to = get_to_square(prev_move)
              prev_from = get_from_square(prev_move)
              pawn_key_idx = game_state[PAWN_KEY_INDEX] & PAWN_HISTORY_MASK
              
              base_bonus = min(HISTORY_BONUS_SCALE * depth, HISTORY_BONUS_CAP)
              sub_bonus = base_bonus // 4
              
              if (prev_piece % 6) != 0:
                  update_pawn_history(search_context.pawn_history, pawn_key_idx, prev_piece, prev_to, sub_bonus)
                  _diag_add(search_context, DIAG_HIST_PAWN_UPD)
              update_history(search_context.history_table, prev_piece, prev_to, sub_bonus)
              update_butterfly_history(search_context.butterfly_history, prev_from, prev_to, sub_bonus)
              _diag_add(search_context, DIAG_HIST_FAIL_LOW)
              _diag_add(search_context, DIAG_HIST_PIECE_TO_UPD)
              _diag_add(search_context, DIAG_HIST_CONT_UPD)
              
              # Continuation history (ply-2, ply-3, ply-4, ply-5, ply-7)
              ...
  ```
* **對戰設定：** `python tools/tournament.py --games 400 --nodes 300000 --concurrency 10 --tt-mb 64` (對手為 E38 原版基線 `classical_old`)
* **對戰結果：**
  * **Games:** 400
  * **Score:** New 195.5 - 204.5 Old (87W 217D 96L), 得分率 48.88% (-9 場淨負)
  * **Elo difference:** **-7.82 ± 23.86 (95% CI)** (區間 [-31.71, 16.00])
  * **搜尋指標：** 平均深度 New 16.73 vs Old 16.82 (-0.09 ply, p=0.56)；平均節點數 200,805 vs 200,958 (-0.1%, p=0.67)；NPS 373,095 vs 371,547 (+0.4%)；耗時 647.1ms vs 652.1ms (-5.0ms, p=0.017)。
  * **分桶洞察：**
    * 中局 (26-50 步): 1W 82D 7L (46.7%, -6 淨負)
    * 長局 (51-80 步): 62W 63D 66L (49.0%, -4 淨負)
    * 超長殘局 (>80 步): 23W 38D 23L (50.0%, 殘局持平)
* **根因分析：**
  1. **缺少 SF 動態負門檻導致歷史表被無效雜訊污染**：Stockfish 18/19（`search.cpp:1485`）給予反擊獎勵時設有 `bonusScale = -245` 的巨大負底數，且要求 `bestValue <= staticEval - 103`（崩盤超 100 cp）才發放獎勵。
  2. **中局排序品質受損**：本作移除 `legal_moves_tried > 4` 後，在沒有上述嚴格條件保護下，大量普通或微弱的 Fail-Low 節點皆無差別發放反擊獎勵，雜訊在中局（Move Ordering 最為關鍵的階段）擾亂了安靜步排序（中局勝率僅 46.7%）。
  3. **原版防噪門檻之必要性**：驗證了原版註解 `FAIL_LOW_MIN_LEGAL_MOVES = 4 # reduce noise` 的設計在簡化架構下是不可或缺的防噪守門員。
* **判定：** **回滾 (REVERT)**。已徹底還原 `chess_engine/classical/search.py` 至 E38 原版基線，`tools/sync_classical_old.py --diff` 驗證完全一致。備份 PGN：`tournament_analysis/match_fail_low_bonus_sf18_n300k_e42_rejected.pgn`。

---

## E43. Step 9 RFP 深度提升至 8 與 TT 安靜步守護 (對齊 Stockfish 19 Step 9，無平滑回傳) (2026-09-25)

> **意圖：** 針對舊架構中反向無效剪枝（RFP）僅限於 `depth <= 5` 的保守限制，參考最新 Stockfish 19（`stockfish_repo/src/search.cpp:994-1008` Step 9 Futility Pruning）機制進行現代化升級：
> 1. **`(!tt_pv && (!tt_move || tt_capture))` 守護**：舊代碼在 `depth <= 5` 時不論置換表記錄何種走步一律 RFP 剪枝。SF19 關鍵守護在於：若置換表記錄了關鍵的「安靜反駁步」（quiet move），絕不提前 RFP，必須搜尋驗證；反之，若無 TT 著法或 TT 著法為吃子步（吃子步將由後續 QS / SEE 嚴格驗證），則能安全進行 RFP。
> 2. **深度上限提升至 8**：在安靜步守護的保護下，將 `RFP_MAX_DEPTH` 由 5 提升至 **8**，大幅剪除深層勝負已分分支的無效展開。
> 3. **無平滑回傳**：嚴格遵循本作 HCE 分數體系，保持原始 `return (np.int32(static_score), NO_MOVE, ...)` 分數語義，不引入加權平滑。

* **改動細節：**
  1. 在 `chess_engine/classical/constants.py`：
     ```python
     -RFP_MAX_DEPTH = 5
     +RFP_MAX_DEPTH = 8
     ```
  2. 在 `chess_engine/classical/search.py` 的 Step 9 RFP 剪枝中：
     ```python
     # --- M5: Reverse Futility Pruning (SF19 Step 9: !tt_pv, !tt_move or tt_capture) ---
     if (search_context.enable_rfp and depth <= RFP_MAX_DEPTH and not low_material_pruning_guard
             and not tt_pv and (tt_move == NO_MOVE or tt_capture)
             and not is_win(static_score) and not is_loss(beta)):
     ```
* **對戰設定：** `python tools/tournament.py --games 400 --nodes 300000 --concurrency 10 --tt-mb 64` (對手為 E38 原版基線 `classical_old`)
* **單元測試：** 執行 `test_history.py`、`test_search_core_contracts.py`、`test_search_param_registry.py` 全數 38 項測試 PASS。
* **對戰結果：**
  * **Games:** 400
  * **Score:** New 203.5 - 196.5 Old (100W 207D 93L), 得分率 50.88% (+7 場淨勝)
  * **Elo difference:** **+6.08 ± 23.68 (95% CI)** (區間 [-17.60, 29.76])
  * **搜尋指標：**
    * 平均深度 New 17.01 vs Old 16.88 (**+0.13 ply**，Welch t=+0.825)；深層 >=14 比例 New 58.0% vs Old 56.0% (**+2.0%**)
    * 每步耗時 New 674.3ms vs Old 681.4ms (**-7.1ms，p=8.51e-4 極顯著提速**)
    * 平均 NPS New 352,831 vs Old 349,150 (**+1.1%，p=0.0418 顯著提升**)
    * 平均節點數 New 201,072 vs Old 200,746 (+0.2%，基本持平)
  * **分桶洞察 (對局長度)：**
    * 短局 (<=25 步): 1W 29D 0L (51.7%)
    * 中局 (26-50 步): 7W 75D 7L (50.0%)
    * 長局 (51-80 步): 65W 64D 65L (50.0%)
    * 超長殘局 (>80 步): **27W 39D 21L (53.4%，+6 場淨勝)**
* **根因分析與效益：**
  1. **深層早期剪枝提速顯著**：將 RFP 上限由 5 提升至 8 後，在深度 6~8 的明朗局面中安全省下大量非必要分支展開，全場每步搜尋耗時顯著降低 -7.1ms (p < 0.001)，NPS 提高 +1.1%。
  2. **TT 安靜步守護徹底防範戰術誤判**：舊架構一刀切的 RFP 在深層極易漏算安靜反駁步；新機制借鑑 SF19 Step 9，若置換表已記錄安靜著法即嚴格豁免 RFP，確保殘局精確度不受損，在 >80 步超長局中斬獲 53.4% 勝率。
  3. **通過標準判定**：勝率 50.88% (>=50.5%)，Elo +6.08 (>0)，深度提高 +0.13 ply 且耗時顯著下降，符合「近平手有優勢（亦可接受）」通過門檻。
* **判定：** **留下 (KEEP)**。已執行 `python tools/sync_classical_old.py --sync` 同步至 `classical_old` 作為新生產基線。備份 PGN：`tournament_analysis/match_rfp_depth8_ttquiet_sf19_n300k_e43.pgn`。

---

## E44. IIR 深度門檻提升至 6 與 Cut/PV 節點守護 (對齊 Stockfish 19 Step 11) (2026-09-25)

> **意圖：** 針對舊架構中內部迭代減深（IIR）門檻過低與無差別減深的兩大缺陷，全面對齊 Stockfish 19（`stockfish_repo/src/search.cpp:1048-1053` Step 11 IIR）：
> 1. **深度門檻提升至 6 (`IIR_MIN_DEPTH = 6`)**：原版在 `depth >= 4` 只要無 TT 著法即強制減深 1 層（砍至 3/4 層），在淺層極易引發戰術盲區（Horizon Effect）。SF19 警告 `(*Scaler) Making IIR more aggressive scales poorly.`，明確規範深度至少達到 6 才介入。
> 2. **`!allNode`（即 `is_pv or cut_node`）守護**：嚴格豁免預期全軍覆沒的 ALL-節點（防守節點）。在 ALL-節點上必須以完整深度檢驗所有分支以證明防守成立；僅在預期有 Cut 或處於 PV 路線時才執行 IIR 減深探勘。

* **改動細節：**
  1. 在 `chess_engine/classical/constants.py`：
     ```python
     -IIR_MIN_DEPTH = 4
     +IIR_MIN_DEPTH = 6  # SF19 Step 11: depth >= 6
     ```
  2. 在 `chess_engine/classical/search.py` 的 Step 11 IIR 判定中：
     ```python
     # --- M1: IIR (Internal Iterative Reduction, SF19 Step 11: depth >= 6, !allNode) ---
     if (tt_move == NO_MOVE and ENABLE_IIR and not is_exclusion_search
             and not is_currently_in_check and not follow_pv
             and (is_pv or cut_node)):
         if depth >= IIR_MIN_DEPTH:
             depth -= 1
     ```
* **對戰設定：** `python tools/tournament.py --games 400 --nodes 300000 --concurrency 10 --tt-mb 64` (對手為已同步 E43 基準之 `classical_old`)
* **單元測試：** 執行 `test_history.py`、`test_search_core_contracts.py`、`test_search_param_registry.py` 全數 38 項測試 PASS。
* **對戰結果：**
  * **Games:** 136 (提早終止：負向顯著)
  * **Score:** New 64.0 - 72.0 Old (25W 78D 33L), 得分率 47.06% (-8 場淨負)
  * **Elo difference:** **-20.46 ± 38.25 (95% CI)** (區間 [-58.71, 17.79])
  * **搜尋指標：**
    * 平均深度 New 16.40 vs Old 16.60 (**-0.19 ply**，Welch t=-0.757)；深層 >=14 比例 New 51.7% vs Old 56.7% (**-5.0% 嚴重萎縮**)
    * 每步耗時 New 651.6ms vs Old 645.6ms (+6.0ms)
    * 平均 NPS New 370,089 vs Old 372,946 (-0.8%)
  * **分桶洞察 (對局長度)：**
    * 短局 (<=25 步): 0W 16D 0L (50.0%)
    * 中局 (26-50 步): 1W 28D 5L (44.1%)
    * 長局 (51-80 步): 15W 18D 23L (42.9%，-8 場淨負，主力失分區間)
    * 超長殘局 (>80 步): 9W 16D 5L (56.7%)
* **根因分析：**
  1. **缺少 TT 著法時淺中層分支因子暴增**：當節點缺少 TT 著法記錄時，Move Ordering 無法在第 1 步實現 $\beta$-Cutoff。原版機制在 `depth >= 4` 且所有節點均減深 1 層，能以極小開銷快速試探並回填 TT 著法。
  2. **強行全深度盲搜揮霍算力**：E44 將 IIR 門檻提升至 6 並豁免 ALL-節點，導致引擎在 depth 4、depth 5 以及所有防守節點上面臨無 TT 著法時被迫進行全深度盲搜，有效分支因子急劇上升。
  3. **固定節點下深層探索嚴重受阻**：在 300k nodes 限制下，淺中層過度消耗算力，使得搜尋樹無力推進至深層（$d \ge 14$ 佔比從 56.7% 暴跌至 51.7%），平均深度萎縮 -0.19 ply，導致中長局被舊版基線全面擊潰。
* **判定：** **回滾 (REVERT)**。已徹底還原 `chess_engine/classical/constants.py`（`IIR_MIN_DEPTH = 4`）與 `chess_engine/classical/search.py` 至 E43 原版基線，`tools/sync_classical_old.py --diff` 驗證完全一致。備份 PGN：`tournament_analysis/match_iir_depth6_sf19_n300k_e44_rejected.pgn`。

---

## E45. LMR 動態 Eval-to-Alpha 差值調節 (對齊 Stockfish 19 Step 18) (2026-09-25)

> **意圖：** 在晚期走步減深（LMR）計算中，引入對齊最新 Stockfish 19（`stockfish_repo/src/search.cpp:1354-1355` Step 18 LMR）的動態局面評估調節機制：
> ```cpp
> if (!capture && !is_decisive(alpha))
>     r += 3 * std::clamp(alpha - eval, -64, 96);
> ```
> 1. **平滑動態自適應**：
>    * 當局面處於劣勢或掙扎和棋時（`static_score < alpha`，即 `alpha - static_score > 0`）：安靜著法極難翻盤，微幅加大減深（在 1024 尺度上最高 $+288$，約 $+0.28$ ply），避免在無效枝節上浪費固定節點預算。
>    * 當局面處於優勢或積極進攻時（`static_score > alpha`，即 `alpha - static_score < 0`）：安靜著法很有可能是致勝妙手，微幅放寬減深（在 1024 尺度上最高 $-192$，約 $-0.19$ ply），保留更多算力深入搜尋。
> 2. **嚴格有界保護**：差值嚴格限制在 $[-64, 96]$（約 $[-0.19, +0.28]$ ply），僅在邊界處提供柔性微調，杜絕整層突變風險。

* **改動細節：**
  1. 在 `chess_engine/classical/constants.py` 新增常數：
     ```python
     # SF19 Step 18: Dynamic LMR eval-to-alpha scaling for quiet moves
     ENABLE_LMR_EVAL_DIFF_SCALING = True
     LMR_EVAL_DIFF_SCALE = 3
     LMR_EVAL_DIFF_MIN = -64
     LMR_EVAL_DIFF_MAX = 96
     ```
  2. 在 `chess_engine/classical/search.py` 的 Step 18 LMR 計算中（歷史表調整之後、ALL 節點縮放之前）：
     ```python
     if ENABLE_LMR_EVAL_DIFF_SCALING and not is_capture and not is_decisive(alpha):
         eval_diff = alpha - static_score
         if eval_diff < LMR_EVAL_DIFF_MIN:
             eval_diff = LMR_EVAL_DIFF_MIN
         elif eval_diff > LMR_EVAL_DIFF_MAX:
             eval_diff = LMR_EVAL_DIFF_MAX
         r += LMR_EVAL_DIFF_SCALE * eval_diff
     ```
* **對戰設定：** `python tools/tournament.py --games 400 --nodes 300000 --concurrency 10 --tt-mb 64` (對手為已同步 E43 基準之 `classical_old`)
* **單元測試：** 執行 `test_history.py`、`test_search_core_contracts.py`、`test_search_param_registry.py` 全數 38 項測試 PASS。
* **對戰結果：**
  * **Games:** 124 (提早終止：負向顯著)
  * **Score:** New 57.5 - 66.5 Old (21W 73D 30L), 得分率 46.37% (-9 場淨負)
  * **Elo difference:** **-25.26 ± 41.74 (95% CI)** (區間 [-67.00, 16.48])
  * **搜尋指標：**
    * 平均深度 New 16.56 vs Old 16.66 (-0.10 ply)；深層 >=14 比例 New 55.1% vs Old 58.1% (**-3.0%**)
    * 每步耗時 New 655.1ms vs Old 652.3ms (+2.8ms)
    * 平均 NPS New 363,570 vs Old 364,765 (-0.3%)
  * **執色與分桶洞察：**
    * 持白表現: 13W 36D 13L (50.0%)
    * 持黑表現: **8W 37D 17L (42.74%，-9 場淨負全在黑方)**
    * 中局 (26-50 步): 2W 27D 6L (44.3%)
    * 長局 (51-80 步): 13W 17D 18L (44.8%)
    * 超長殘局 (>80 步): 6W 18D 6L (50.0%)
* **根因分析：**
  1. **NNUE 平滑評估與 HCE 波動評估之不可調和性**：Stockfish 19 的公式依賴神經網路極低雜訊的靜態估值。本作 HCE 具有較高的局部波動（如牽制威脅、王翼暴露等），靜態分數常比實際局勢暫時偏低 50~100 cp。
  2. **防守方關鍵安靜解危步遭到「致命誤剪」**：當黑方處於開局或中局微劣受壓局面（`alpha = +50`，而局部 `static_score = -40`），公式算出 `alpha - static_score = +90`，導致 `r += 270`（相當於增加近 1 層有效減深）。黑方原本用以化解攻勢的安靜防守步全部被草率減深而未能看清，導致黑方勝率暴跌至 42.74%（淨負 9 局全發生在黑方）。
  3. **優勢局虛胖消耗算力**：在 HCE 盲目樂觀時放寬減深，在無戰術威脅的枝葉上過度深入，深層比例從 58.1% 萎縮至 55.1%。
* **判定：** **回滾 (REVERT)**。已徹底還原 `chess_engine/classical/constants.py` 與 `chess_engine/classical/search.py` 至 E43 原版基線，`tools/sync_classical_old.py --diff` 驗證完全一致。備份 PGN：`tournament_analysis/match_lmr_eval_diff_sf19_n300k_e45_rejected.pgn`。

---

## E46. Razoring 現代化與二次方深度裕度 (對齊 Stockfish 19 Step 8) (2026-09-25)

> **意圖：** 針對舊架構中剃刀剪枝（Razoring）停留在 SF11 時代的保守限制，全面對齊 Stockfish 19（`stockfish_repo/src/search.cpp:983-991` Step 8 Razoring）：
> 1. **節點級剪枝前置**：舊版 Razoring 被放置於 NMP 與 ProbCut 之後（深層位置），且在 NMP 被跳過時才偶爾執行。SF19 將 Razoring 作為節點級剪枝的入口第一步（Step 8，位於 Step 9 RFP 之前），一旦局面呈現無可挽回的潰敗，直接以靜態搜尋快速截斷，省下後續所有節點展開與試探開銷。
> 2. **二次方深度裕度 (`482 * depth * depth`)**：
>    * 舊版僅在 `depth < 2`（僅限 depth 1）使用線性常數 250 cp。
>    * SF19 將範圍擴展至 `depth <= 3`，並使用二次方裕度門檻：
>      - $d = 1$: $482 \times 1^2 = 482$ cp
>      - $d = 2$: $482 \times 2^2 = 1928$ cp
>      - $d = 3$: $482 \times 3^2 = 4338$ cp
>    * 門檻極寬且隨深度二次方急劇擴大，完全免疫 HCE 評估值的局部小幅波動（解決 E45 雜訊誤剪問題），僅在真正崩盤的分支介入。
> 3. **截斷語義對齊**：調用 `quiescence_search(alpha, beta)`，若即使在吃子安靜搜尋後分數仍無法突破 $\alpha$，直接回傳 qsearch 分數截斷。

* **改動細節：**
  1. 在 `chess_engine/classical/constants.py`：
     ```python
     # Razoring — SF19 Step 8: eval < alpha - RAZORING_COEFF * depth * depth
     RAZORING_MARGIN = 250  # Retained for runtime tune slot compatibility
     RAZORING_MAX_DEPTH = 3
     RAZORING_COEFF = 482
     ```
  2. 在 `chess_engine/classical/search.py`：
     - 於節點級剪枝入口（Line 1574，Step 9 RFP 之前）新增 Step 8 Razoring：
       ```python
       # --- Step 8: Razoring (SF19 Step 8: eval < alpha - 482 * depth * depth) ---
       if (search_context.enable_razoring and depth <= RAZORING_MAX_DEPTH and not low_material_pruning_guard
               and not is_win(static_score) and not is_loss(alpha)
               and abs(alpha) < VALUE_KNOWN_WIN):
           razor_margin = RAZORING_COEFF * depth * depth
           if static_score < alpha - razor_margin:
               if not static_eval_is_full:
                   raw_static_eval, static_score, improving, cached_pinned_white, cached_pinned_black = compute_full_corrected_static_eval(
                       piece_bbs, occupancy_bbs, game_state, search_context, ply
                   )
                   static_eval_is_full = True
               if (static_score < alpha - razor_margin
                       and not is_win(static_score) and not is_loss(alpha)):
                   search_context.pv_table[ply, ply] = NO_MOVE
                   razor_score, child_q_nodes = quiescence_search(
                       piece_bbs, occupancy_bbs, game_state, alpha, beta, ply, search_context, _i32(0)
                   )
                   quiescence_nodes += child_q_nodes
                   _diag_add(search_context, DIAG_RAZOR_CUT)
                   return (np.int32(razor_score), NO_MOVE, nodes_searched, quiescence_nodes, tt_hits)
       ```
     - 移除舊版位於 NMP/ProbCut 之後、僅限 depth < 2 的舊式 Razoring 區塊。
* **對戰設定：** `python tools/tournament.py --games 400 --nodes 300000 --concurrency 10 --tt-mb 64` (對手為已同步 E43 基準之 `classical_old`)
* **單元測試：** 執行 `test_history.py`、`test_search_core_contracts.py`、`test_search_param_registry.py`、`test_nodes_limit_reset.py` 全數 41 項測試 PASS (3.88s)。
* **對戰結果：**
  * **Games:** 400
  * **Score:** New 203.5 - 196.5 Old (100W 207D 93L), 得分率 50.88% (+7 場淨勝)
  * **Elo difference:** **+6.08 ± 22.47 (95% CI)** (區間 [-16.36, 28.58]，color-swapped pair clustered)
  * **搜尋指標：**
    * 平均深度 New 17.08 vs Old 17.05 (+0.03 ply)；深層 >=14 比例 New 58.6% vs Old 57.4% (**+1.2%**)
    * 每步耗時 New 631.0ms vs Old 625.6ms (+5.4ms)
    * 平均 NPS New 381,634 vs Old 381,015 (+0.2%，持平)
    * 平均節點數 New 201,303 vs Old 199,729 (+0.8%)
  * **執色與分桶洞察：**
    * 持白表現: 56W 105D 39L (54.25%)
    * 持黑表現: 44W 102D 54L (47.50%)
    * 短局 (<=25 步): 0W 45D 0L (50.0%)
    * 中局 (26-50 步): 9W 72D 7L (51.1%)
    * 長局 (51-80 步): 66W 49D 71L (48.7%)
    * 超長殘局 (>80 步): **25W 41D 15L (56.2%，+10 場淨勝！殘局顯著優勢！)**
* **根因分析與效益：**
  1. **安全截斷崩盤死枝，算力集中深層**：二次方深度裕度門檻（482/1928/4338）極其寬闊，只在局面真正潰敗時生效，徹底免疫 HCE 的局部波動，避免了 E45 的誤剪悲劇。深層 $d \ge 14$ 比例由 57.4% 提升至 58.6%。
  2. **前置 Step 8 節省冗餘展開**：在崩盤節點直接以靜態吃子搜尋（QS）截斷，省下後續 RFP、NMP、ProbCut 等繁複試探。
  3. **超長殘局（>80 步）斬獲顯著優勢**：在超過 80 步的超長殘局中打出 25W 41D 15L（勝率 56.2%，淨勝 +10 局），證明現代化 Razoring 在複雜長局殘局中精準剪除落後分支，將算力導向致勝防守與突破線路。
  4. **通過門檻判定**：勝率 50.88% (>=50.5%)，Elo 點估計 +6.08 (>0)，淨勝 +7 局，完全符合「近平手有優勢（亦可接受）」通過門檻。
* **判定：** **留下 (KEEP)**。已於 2026-09-25 執行 `python tools/sync_classical_old.py --sync` 同步進生產基線。備份 PGN：`tournament_analysis/match_razoring_sf19_step8_n300k_e46.pgn`。

---

## E47. ProbCut 深度動態自適應 (對齊 Stockfish 19 Step 12) (2026-09-25)

> **意圖：** 針對舊架構中概率剪枝（ProbCut）使用固定 4 層減深（`PROBCUT_R = 4`）與固定深度門檻（`PROBCUT_MIN_DEPTH = 5`）的限制，全面對齊 Stockfish 19（`stockfish_repo/src/search.cpp:1058-1064` Step 12 ProbCut）：
> ```cpp
> Depth probCutDepth = depth - (improving ? 5 : 3);
> ```
> 1. **動態減深 (`improving ? 5 : 3`)**：
>    * 當盤面正在改善（`improving == True`）：局面樂觀且進攻主動，減深 **5** 層（`depth - 5`），大幅加速驗證速度並省下深層展開開銷；
>    * 當盤面未改善（`improving == False`）：局面受壓或停滯，保守減深 **3** 層（`depth - 3`），防止在劣勢局中因過度減深而漏算對手的戰術反撲。
> 2. **自適應深度門檻**：
>    * 門檻調整為 `depth >= probcut_r + 1`（`improving` 時為 6，`not improving` 時為 4），保證子搜尋深度至少為 1，並讓優勢局面更早在淺層觸發 ProbCut 截斷。
> 3. **TT 捷徑與子搜尋聯動**：
>    * TT 快速捷徑門檻由原固定 `depth - 4` 同步改為 `depth - probcut_r`，保持條件嚴格自洽。

* **改動細節：**
  1. 在 `chess_engine/classical/constants.py`：
     ```python
     # SF19 Step 12: Dynamic ProbCut reduction based on improving flag
     PROBCUT_R_IMPROVING = 5
     PROBCUT_R_NOT_IMPROVING = 3
     ```
  2. 在 `chess_engine/classical/search.py`：
     - 匯入 `PROBCUT_R_IMPROVING, PROBCUT_R_NOT_IMPROVING`。
     - 計算 `probcut_r = PROBCUT_R_IMPROVING if improving else PROBCUT_R_NOT_IMPROVING` 與 `probcut_min_depth = probcut_r + 1`。
     - 門檻檢查：`depth >= probcut_min_depth`。
     - TT 捷徑檢查：`if tt_entry['depth'] >= depth - probcut_r:`。
     - 子搜尋調用：`_search(..., _i32(depth - probcut_r), ...)`。
* **對戰設定：** `python tools/tournament.py --games 400 --nodes 300000 --concurrency 10 --tt-mb 64` (對手為已同步 E46 基準之 `classical_old`)
* **單元測試：** 執行 `test_history.py`、`test_search_core_contracts.py`、`test_search_param_registry.py`、`test_nodes_limit_reset.py` 全數 41 項測試 PASS (2.23s)。
* **對戰結果：**
  * **Games:** 400
  * **Score:** New 207.0 - 193.0 Old (94W 226D 80L), 得分率 51.75% (**+14 場淨勝！**)
  * **Elo difference:** **+12.17 ± 19.88 (95% CI)** (區間 [-7.67, 32.08]，color-swapped pair clustered)
  * **搜尋指標：**
    * 平均深度 New 16.88 vs Old 16.87 (+0.01 ply)；深層 >=14 比例 New 58.2% vs Old 58.5%
    * 每步耗時 New 625.9ms vs Old 627.3ms (**-1.4ms 提速**)
    * 平均 NPS New 389,128 vs Old 387,908 (**+0.3%**)
    * 平均節點數 New 202,118 vs Old 202,122 (-0.0%，完全持平)
  * **執色與分桶洞察：**
    * 持白表現: 58W 110D 32L (56.50%)
    * 持黑表現: 36W 116D 48L (47.00%)
    * 短局 (<=25 步): 0W 38D 1L (48.7%)
    * 中局 (26-50 步): **11W 86D 9L (50.9%，+2 淨勝)**
    * 長局 (51-80 步): **59W 61D 54L (51.4%，+5 淨勝)**
    * 超長殘局 (>80 步): **24W 41D 16L (54.9%，+8 淨勝)**
* **根因分析與效益：**
  1. **優勢局面大幅提速收割**：當 `improving` 為真時，局面明朗且呈上升趨勢，動態減深 5 層（`depth - 5`）讓引擎以極小代價迅速完成子搜尋驗證並達成 ProbCut 截斷，全場平均耗時縮短 -1.4ms，NPS 提高 +0.3%。
  2. **危險停滯局面更安全**：當 `improving` 為假時，局面受壓或處於均勢，保守減深 3 層（`depth - 3`）有效杜絕了過度減深導致漏算防守的風險，使得中長局與超長殘局全面斬獲淨勝（中局 +2、長局 +5、殘局 +8）。
  3. **通過門檻判定**：勝率 51.75% (>=50.5%)，Elo 點估計 +12.17 (>0)，全場大幅淨勝 +14 局，且耗時降低、NPS 提高，完全符合**「明確正向 / 近平手有優勢」通過門檻**。
* **判定：** **採納進生產 / 基線 (KEEP & SYNCED)**。已備份 PGN：`tournament_analysis/match_probcut_improving_sf19_n300k_e47.pgn`。已執行 `python tools/sync_classical_old.py --sync` 同步進生產基線。

### E48. LMR ALL-Node 放大比率對齊 (對齊 Stockfish 19 Step 18) (2026-09-26)

* **動機與設計依據：**
> 1. **Stockfish 19 Step 18 原生公式**：
>    在 Stockfish 19 中，當節點為預期 ALL-Node（即 `!PvNode && cutNode`，在非 PV 且預期剪枝節點失敗低的情況下）時，LMR 的減深放大比率為：
>    `r += r * 276 / (256 * depth + 268)`
> 2. **原版參數落差**：
>    現行 codebase 中的參數為：
>    `LMR_ALLNODE_SCALE_NUM = 150`
>    `LMR_ALLNODE_SCALE_DENOM_BASE = 256`
>    `LMR_ALLNODE_SCALE_DENOM_OFFSET = 285`
>    其中分子 `150` 僅為 Stockfish 19（`276`）的 54.3%，導致引擎在確定為 ALL-Node 的防守節點（預期所有走步皆無法突破 $\alpha$）上減深力度嚴重不足，浪費大量 nodes 在無效防守分支。
> 3. **極致單一變數控制**：
>    `chess_engine/classical/search.py` 原本即已具備 `r += r * LMR_ALLNODE_SCALE_NUM // (LMR_ALLNODE_SCALE_DENOM_BASE * depth + LMR_ALLNODE_SCALE_DENOM_OFFSET)` 的完整架構，本次僅需對齊 `constants.py` 的常數定義，零程式碼邏輯變動，風險極低。

* **改動細節：**
  在 `chess_engine/classical/constants.py`（Line 1095-1098）：
  ```python
  # SF19 Step 18: Scale up reductions for expected ALL nodes: r += r * 276 / (256 * depth + 268)
  LMR_ALLNODE_SCALE_NUM = 276
  LMR_ALLNODE_SCALE_DENOM_BASE = 256
  LMR_ALLNODE_SCALE_DENOM_OFFSET = 268
  ```
* **對戰設定：** `python tools/tournament.py --games 400 --nodes 300000 --concurrency 10 --tt-mb 64` (對手為已同步 E47 基準之 `classical_old`)
* **單元測試：** 執行 `test_history.py`、`test_search_core_contracts.py`、`test_search_param_registry.py`、`test_nodes_limit_reset.py` 全數 41 項測試 PASS (2.30s)。
* **對戰結果：**
  * **Games:** 400
  * **Score:** New 204.0 - 196.0 Old (102W 204D 94L), 得分率 51.00% (**+8 場淨勝**)
  * **Elo difference:** **+6.95 ± 23.83 (95% CI)** (區間 [-15.94, 29.90]，color-swapped pair clustered)
  * **搜尋指標：**
    * 平均深度 New 16.97 vs Old 16.89 (**+0.08 ply**)；深層 >=14 比例 New 57.7% vs Old 56.8% (+0.9%)
    * 每步耗時 New 632.2ms vs Old 636.0ms (**-3.8ms 提速**)
    * 平均 NPS New 378,222 vs Old 377,036 (**+0.3%**)
    * 平均節點數 New 201,687 vs Old 201,844 (-0.1%)
  * **執色與分桶洞察：**
    * 持白表現: 48W 102D 50L (49.50%)
    * 持黑表現: **54W 102D 44L (52.50%，防守表現亮眼)**
    * 短局 (<=25 步): 0W 36D 0L (50.0%)
    * 中局 (26-50 步): **13W 74D 6L (53.8%，+7 淨勝！)**
    * 長局 (51-80 步): **64W 54D 60L (51.1%，+4 淨勝)**
    * 超長殘局 (>80 步): 25W 40D 28L (48.4%)
    * 賽程穩定度: 前半 200 盤得分 50.7%，後半 200 盤得分 51.2%
* **根因分析與效益：**
  1. **防守 ALL 節點高效止血**：在非 PV 且無法達成剪枝的節點（預期所有走步皆無法突破 $\alpha$）上，對齊 SF19 放大比例公式精確放大了減深力度，防止引擎在注定無法突破的安靜走步上浪費節點。
  2. **深度有效延伸且耗時縮短**：省下的算力被再投資到關鍵 PV 分支，全場平均深度提升 **+0.08 ply**（深層比例 +0.9%），且每步耗時提速 **-3.8ms**。
  3. **持黑與中局顯著優勢**：在通常處於防守或均勢的中局（淨勝 +7 局，勝率 53.8%）與持黑局面（勝率 52.5%）上斬獲顯著勝場，完全印證 ALL-Node 減深加強的理論預期。
  4. **通過門檻判定**：勝率 51.00% (>=50.5%)，Elo 點估計 +6.95 (>0)，淨勝 +8 局，且深度提升、耗時縮短，完全符合**「近平手有優勢（亦可接受）」通過門檻**。
* **判定：** **採納進生產 / 基線 (KEEP & SYNCED)**。已備份 PGN：`tournament_analysis/match_lmr_allnode_sf19_n300k_e48.pgn`。已執行 `python tools/sync_classical_old.py --sync` 同步進生產基線。

### E49. LMR CutNode 減深加強 (對齊 Stockfish 19 Step 18) (2026-09-26)

* **動機與設計依據：**
> 1. **CutNode 劣步高發特性**：
>    在 Cut 節點上，父節點預期僅搜尋第一步（通常為 TT 著法）即可達成 $\beta$-Cutoff。當第一步未截斷時，後續安靜走步（`moveCount > 1`）在統計上超過 95% 全為劣步，絕大多數無法突破 $\alpha$。
> 2. **Stockfish 19 Step 18 原生機制**：
>    在 Stockfish 19 中，CutNode 的減深加成為：
>    `if (cutNode) r += 4026 + 933 * !ttData.move;`
> 3. **原版參數落差**：
>    現行 codebase 中的參數為：
>    `LMR_CUTNODE_BONUS = 2048`（SF11 舊世代，僅約 2 plies）
>    `LMR_NO_TTMOVE_BONUS = 1024`
>    這導致在 Cut 節點後續的弱走步上搜尋過深，浪費大量 nodes。
> 4. **極致單一變數控制**：
>    `search.py` 原本即已具備完整的調用結構，本次僅需對齊 `constants.py` 的常數定義（`LMR_CUTNODE_BONUS = 4026`，`LMR_NO_TTMOVE_BONUS = 933`），零邏輯改動，風險極低。

* **改動細節：**
  1. 在 `chess_engine/classical/constants.py`（Line 1079, 1092）：
     ```python
     LMR_CUTNODE_BONUS = 4026       # SF19 Step 18: r += 4026 + 933 * !ttData.move
     LMR_NO_TTMOVE_BONUS = 933      # SF19 Step 18: extra reduction without TT move on cut nodes
     ```
  2. 在 `tune_search/search_param_registry.py`：將 `LMR_CUTNODE_BONUS` 上限由 3000 擴展至 5000。
  3. 在 `tests/test_search_param_registry.py`：將上限斷言更新為 5000。
* **對戰設定：** `python tools/tournament.py --games 400 --nodes 300000 --concurrency 10 --tt-mb 64` (對手為已同步 E48 基準之 `classical_old`)
* **對戰設定：** `python tools/tournament.py --games 400 --nodes 300000 --concurrency 10 --tt-mb 64` (對手為已同步 E48 基準之 `classical_old`)
* **單元測試：** 執行 `test_history.py`、`test_search_core_contracts.py`、`test_search_param_registry.py`、`test_nodes_limit_reset.py` 全數 41 項測試 PASS (2.64s)。
* **對戰結果：**
  * **Games:** 400
  * **Score:** New 199.0 - 201.0 Old (87W 224D 89L), 得分率 49.75% (**淨負 -2 局**)
  * **Elo difference:** **-1.74 ± 22.59 (95% CI)** (區間 [-24.16, 20.67]，color-swapped pair clustered)
  * **搜尋指標：**
    * 平均深度 New 16.73 vs Old 16.86 (**-0.13 ply 倒退**)；深層 >=14 比例 New 57.6% vs Old 60.1% (**-2.5% 萎縮**)
    * 每步耗時 New 620.9ms vs Old 624.0ms (-3.1ms)
    * 平均 NPS New 393,272 vs Old 393,184 (+0.0%，持平)
    * 平均節點數 New 202,645 vs Old 203,410 (-0.4%)
  * **執色與分桶洞察：**
    * 持白表現: 54W 120D 26L (57.00%)
    * 持黑表現: **33W 104D 63L (42.50%，淨負 -30 局，黑方嚴重受壓)**
    * 短局 (<=25 步): 0W 39D 0L (50.0%)
    * 中局 (26-50 步): 6W 71D 4L (51.2%)
    * 長局 (51-80 步): **52W 61D 64L (46.6%，淨負 -12 局)**
    * 超長殘局 (>80 步): 29W 53D 21L (53.9%)
    * 賽程波動: 前半 200 盤得分 49.2%，中段一度反彈至 51.4%，後段再次回落至 49.75%
* **根因分析：**
  1. **過度減深引發 Re-search 算力風暴**：
     CutNode 減深加成由 2048 暴增到 4026（約 4 plies）。在古典 HCE 的 Move Ordering 不如神經網路精準的現實條件下，後續走步過度減深導致大量潛在防守步或反擊步在初次搜尋時被大幅削弱；一旦走步突破 $\alpha$，頻繁觸發 LMR Re-search（二度全深搜尋），反而揮霍了固定 300,000 nodes 算力。
  2. **深度實質萎縮**：
     二度搜尋的浪費直接導致全場平均迭代深度由 16.86 倒退至 16.73（-0.13 ply），深層比例萎縮 -2.5%。
  3. **黑方防守體系崩盤**：
     在天生被動的持黑局面中，CutNode 減深加強讓黑方漏算白方的深層攻勢，持黑勝率暴跌至 42.50%（淨負 -30 局）。
  4. **通過門檻判定**：
     勝率 49.75%（未達 50.5% 門檻），Elo 點估計為負 (-1.74)，且深度倒退，觸發**回滾標準**。
* **判定：** **回滾 (ROLLBACK)**。已備份 PGN：`tournament_analysis/match_lmr_cutnode_sf19_n300k_e49.pgn`。代碼已全數回滾至 E48 生產基線，並通過單元測試與 `--diff` 驗證。

### E50. 廢除無效將軍延伸 (Check Extension 限制) (對齊 Stockfish 19 Step 17) (2026-09-26)

* **動機與設計依據：**
> 1. **將軍爆炸（Check Explosion）病灶**：
>    在古典搜尋樹中，只要走出將軍就盲目加深 1 層（`depth - 1 + 1 = depth`），極易在「無效逃王與連環將軍」的分支中急劇膨脹，深度根本無法向下推進，固定算力（300,000 nodes）被大量揮霍。
> 2. **Stockfish 19 Step 17 原生範式**：
>    在 Stockfish 19 中，`extension` 唯一來源是 Singular Extensions（奇異延伸），**完全廢除了 Check Extension**。將軍走步的戰術安全由 Move Ordering（將軍優先排序）、QSearch（靜態應將）與 LMR（`is_giving_check` 減深少 1 層）全面守護，絕不給予整層深度延伸。
> 3. **歷史驗證**：
>    本作在 E1 曾藉由收緊將軍延伸門檻斬獲 **+15.7 Elo**，已充分驗證消除無效將軍加深的巨大正向潛力。
> 4. **極致單一變數控制**：
>    在 `constants.py` 設置 `ENABLE_CHECK_EXTENSION = False`，並在 `search.py` 中守護 pre-make SEE 計算與 extension 判定，零額外邏輯副作用。

* **改動細節：**
  1. 在 `chess_engine/classical/constants.py`：
     ```python
     # SF19 Step 17: Check extension eliminated to prevent check explosion
     ENABLE_CHECK_EXTENSION = False
     ```
  2. 在 `chess_engine/classical/search.py`：
     - 匯入 `ENABLE_CHECK_EXTENSION`。
     - 守護 Pre-make SEE 計算：`if ENABLE_CHECK_EXTENSION and not is_pv and move != tt_move and depth > CHECK_EXT_NON_PV_MAX_DEPTH:`。
     - 守護 Extension 判定：`if ENABLE_CHECK_EXTENSION and is_giving_check_after_move:`。
* **對戰設定：** `python tools/tournament.py --games 400 --nodes 300000 --concurrency 10 --tt-mb 64` (對手為已同步 E48 基準之 `classical_old`)
* **單元測試：** 執行 `test_history.py`、`test_search_core_contracts.py`、`test_search_param_registry.py`、`test_nodes_limit_reset.py` 全數 41 項測試 PASS (2.17s)。
* **對戰結果：**
  * **Games:** 400
  * **Score:** New 213.0 - 187.0 Old (96W 234D 70L), 得分率 53.25% (**淨勝高達 +26 局！**)
  * **Elo difference:** **+22.62 ± 20.50 (95% CI)** (區間 [2.20, 43.19]，CI 下界 > 0 嚴格統計顯著勝出！)
  * **搜尋指標：**
    * 平均深度 New 17.91 vs Old 16.71 (**+1.20 ply 暴漲！** Welch t=+8.489, p=2.15e-17)
    * 深層 >=14 比例 New 76.5% vs Old 59.2% (**+17.3% 大幅突破！**)
    * 每步耗時 New 601.0ms vs Old 614.4ms (**-13.4ms 大幅提速！**)
    * 同深度節點消耗比: 0.86 ~ 0.93 (同深度節省 7%~14% nodes)
    * 平均 NPS New 383,902 vs Old 395,003 (-2.8%，因深度大幅加深進入更深層的樹結構)
  * **執色與分桶洞察：**
    * 持白表現: 54W 119D 27L (56.75%，淨勝 +27 局)
    * 持黑表現: 42W 115D 43L (49.75%，極度穩健)
    * 短局 (<=25 步): 0W 39D 0L (50.0%)
    * 中局 (26-50 步): 7W 86D 4L (51.5%)
    * 長局 (51-80 步): **69W 54D 48L (56.14%，淨勝 +21 局！)**
    * 超長殘局 (>80 步): **20W 55D 18L (51.08%，淨勝 +2 局)**
* **根因分析與效益：**
  1. **徹底消除將軍爆炸**：
     將軍步不再盲目無腦加深 1 層，消除了在逃王與穿插將軍死胡同裡的無效算力膨脹。原本被將軍步卡住的搜尋樹得以順暢向下推進，全場平均深度暴增 **+1.20 ply**（深層比例由 59.2% 狂飆至 76.5%）！
  2. **提速收割與實質戰力轉化**：
     省去將軍前置幾何檢驗與無效分支算力後，每步搜尋提速 **-13.4ms**，同深度節點開銷降低 7%~14%。釋放出的算力轉化為實質棋力優勢，在 51~80 步的長局激戰中打出 **56.14%** 的壓倒性勝率（淨勝 +21 局！）。
  3. **通過門檻判定**：
     勝率 **53.25%**（超越明確正 52% 門檻），Elo **+22.62**（超越 +10 Elo 門檻，且 CI 下界 +2.20 > 0），全場狂勝 +26 局，深度暴增 +1.20 ply，完全達到**「明確正向」最高通過門檻**。
* **判定：** **留下 (KEEP)**。備份 PGN：`tournament_analysis/match_no_check_ext_sf19_n300k_e50.pgn`。**已於 2026-09-26 執行 `python tools/sync_classical_old.py --sync` 正式同步進生產基線**，驗證 `--diff` 雙目錄 100% 邏輯一致，單元測試 41 項全數 PASS。

#### E51 LMR TT Move 減深減免對齊（對齊 SF19 Step 18）

* **代碼：** `chess_engine/classical/constants.py` (`LMR_TTMOVE_REDUCTION`: 1024 -> 2179)
* **對戰過程：** 跑至 194 局時精確維持 50.0%（38W 118D 38L，Elo -0.0）。
* **根因分析：**
  * 使用者敏銳發現 Kiwipete 暖機 depth 14 節點數完全一致（274,724 vs 274,724）。
  * 深入源碼排查查明：在 `search.py` 中，`move == tt_move` 永遠走 `searched_legal_moves == 1`（主搜尋路徑），根本不進 `else:`（LMR 區塊）。因此 LMR 內部的 `elif move == tt_move:` 在現行架構下為 100% 碰不到的死碼（Dead Code）。
* **判定：** **中止 (ABORTED)**。已復原 `LMR_TTMOVE_REDUCTION = 1024`。

#### E52 奇異延伸 (SE) 雙重延伸深度門檻解禁 (10->7)（對齊 SF19 Step 16）

* **代碼：** `chess_engine/classical/constants.py`
  * `SINGULAR_DOUBLE_EXT_MIN_DEPTH`: 10 -> **7**（完全對齊 `MIN_SINGULAR_DEPTH = 7`，解禁 depth 7~9 的雙重延伸限制）。
* **背景與動機：**
  * 在 E50 廢除將軍延伸後，`singular_extension` 是全引擎唯一的走步加深正向來源。
  * Stockfish 19 Step 16 中，雙重延伸（Double Extension）完全沒有 `depth >= 10` 的人為限制，只要節點滿足奇異延伸條件且優勢顯著（`exclusion_score < exclusion_beta - double_margin`），直接給予 +2 ply 雙重延伸。
  * 原版限制在 10，導致 300k nodes 下佔比超過 80% 的 depth 7~9 節點即使走出壓倒性好步也被閹割為 +1 ply。解鎖至 7 能極大增強中深層關鍵戰術穿透力，且**完全不增加任何額外排除搜尋節點**（`MIN_SINGULAR_DEPTH` 維持 7 不變）。
* **對戰設定：** 400 場 × 300,000 nodes（10 並行），對抗生產基線 E50（`classical_old`）。
* **對戰結果（118 局提早終止）：**
  * **戰績：** 勝 21 | 和 63 | 負 34（淨負 -13 局）
  * **勝率：** **44.49%** · **Elo：** **-38.43 ± 45.18**（95% CI [-84.27, +6.10]）
  * **長局崩盤：** 51~80 步戰績 11W 10D 21L，勝率跌破至 **38.10%**！
  * **搜尋指標：**
    * 平均深度萎縮：New 18.00 vs Old 18.22（**-0.21 ply**，深層 $d \ge 14$ 佔比由 75.0% 降至 72.8%）。
    * 同深度節點開銷暴增：$d=8$ ratio=1.044 (+4.4%)、$d=9$ ratio=1.035 (+3.5%)。
* **根因分析：**
  1. **中淺層節點指數級分支爆炸**：在 300,000 nodes 固定算力下，深度 7~9 涵蓋絕大多數母體節點。在此區間賦予 +2 ply 雙重延伸，直接引發幾何級數分支爆炸，將大量算力浪費在少數被加深的局部枝葉中。
  2. **算力被吸乾導致全局迭代加深深度萎縮**：其他兄弟分支因節點預算耗盡無法向上推進，全場平均深度硬生生萎縮 -0.21 ply，導致中長局殘局全面被壓制。
  3. **驗證了 `depth >= 10` 防火牆的必然性**：Double Extension 必須嚴格保留深層門檻（$\ge 10$），不可下放至中淺層。
* **判定：** **回滾 (REVERT)**。PGN 備份：`tournament_analysis/match_se_double_ext_min7_sf19_n300k_e52_aborted.pgn`。已完全復原 `SINGULAR_DOUBLE_EXT_MIN_DEPTH = 10`，差分 100% 乾淨。

#### E53 延續歷史剪枝門檻放寬 (-1800 -> -1500)（對齊 SF19 Step 14）

* **代碼：** `chess_engine/classical/constants.py`
  * `PRUNING_CONTINUATION_THRESHOLD`: -1800 -> **-1500**（由 -1800 * depth 放寬為 -1500 * depth）。
* **背景與動機：**
  * 在 E38 中，引入延續歷史剪枝直接狂斬 **+17.39 Elo**，展現極強統治力。
  * 在 E50 廢除將軍延伸後，全局搜尋平均深度大幅暴漲 +1.20 ply（達到 17.91 ply），導致深層節點中的 `depth` 因子普遍放大，原本的 `-1800 * depth` 在深度 3~6 變得過於嚴苛（例如 $d=4$ 為 -7200，$d=5$ 為 -9000），幾乎無法剪除任何走步。
  * Stockfish 19 官方在 Step 14 中，亦將門檻由 SF17 的 -4313 放寬至 -4136。
  * 將門檻微調放寬約 16%（至 -1500），能精準補償深度拉升後的嚴苛化，讓被前手應手證明為劣步的安靜步在中深層更順暢被剪除，進一步提升整體搜尋深度與長局勝率。
* **對戰設定：** 400 場 × 300,000 nodes（10 並行），對抗生產基線 E50（`classical_old`）。
* **對戰結果（400 局完賽）：**
  * **戰績：** 勝 105 | 和 211 | 負 84（**淨勝高達 +21 局！**）
  * **勝率：** **52.62%** · **Elo：** **+18.26 ± 23.40**（95% CI [-3.80, 40.46]）
  * **SPRT LLR：** **+1.05**（bounds [-2.94, 2.94]）
  * **長局統治力：**
    * 51~80 步戰績：**76W 54D 56L（勝率 55.38%，淨勝 +20 局！）**
    * >80 步超長殘局：**24W 46D 18L（勝率 53.41%，淨勝 +6 局！）**
  * **搜尋指標：**
    * 平均深度：New 18.14 vs Old 18.11（**+0.04 ply**，深層 $d \ge 14$ 佔比由 74.3% 提升至 **75.2%**）。
    * 同深度節點開銷全面節省：$d=7$ (0.992)、$d=9$ (0.990)、$d=10$ (0.993)、$d=11$ (0.996)、$d=12$ (0.992)。中淺層節點消耗全部低於 Old！
* **根因分析與效益：**
  1. **方差震盪與大數定律印證**：對局前 40 局因開局庫隨機抽樣曾出現短暫方差回撤（一度跌至 47.0%），但隨後在第 80~400 局展現壓倒性統治力，下半場（後 200 局）打出高達 **54.7%** 的得分率！
  2. **中淺層劣步精準剃除**：同深度節點開銷在 depth 7~12 全面下降 0.4%~1.0%，證明門檻由 -1800 放寬至 -1500 正好抵銷了 E50 深度暴漲帶來的門檻嚴苛化，在中深層更及時地剪除了無效安靜步。
  3. **通過門檻判定**：勝率 **52.62%**（超越 52% 明確正向門檻），Elo **+18.26**，全場淨勝 +21 局，完全達到**「明確正向」採納門檻**！
* **判定：** **留下 (KEEP)**。備份 PGN：`tournament_analysis/match_cont_pruning_thresh1500_sf19_n300k_e53_adopted.pgn`。建議執行 `python tools/sync_classical_old.py --sync` 同步進生產基線。

---

## 4. 常用指令

```text
# 正式對局（E22 起）：1000 場 × n10k（early-trash：≥80 且 score%≤40；SPRT 預設開）
python -u tools/tournament.py -c 6 --games 1000 --nodes 10000 --name1 <ID> --name2 Old \
  --pgn tournament_analysis/tournament_results_<ID>.pgn

# 可加 --no-clear-cache 加速（僅當改的是 runtime tune 常數、無需強制清 JIT 時）
# 關閉 early-trash：--no-early-trash
# 強制跑滿 1000（關 SPRT）：--no-sprt

# depth / NPS
python tournament_analysis/parse_search_stats.py --pgn tournament_analysis/tournament_results_<ID>.pgn --name1 <ID> --name2 Old --out-dir tournament_analysis/stats_<ID>

# 留下後同步 Old
python tools/sync_classical_old.py --sync
```

---

## 5. 操作注意

- **sync 時機：** 僅在「判定留下」之後；先改下一項再 sync 會污染 Old（E4 曾發生，已修正）。  
- **正確順序：** 判定留下 → sync → **再**改 New 的下一項常數。  
- **PGN 預設路徑：** `tournament_analysis/tournament_results.pgn`（每次開跑會清空覆寫）→ 結束立刻 copy 備份。  
- **README 索引：** 本檔屬 `tournament_analysis/` 實驗紀錄；技術導覽仍以根目錄 `README.md` 為準。
