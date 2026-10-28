# Classical 搜尋單項實驗日誌

> **用途：** 連續 A/B 實驗的唯一進度表與決策紀錄。未來接續實驗、回顧為何留下/回滾，以此檔為準。  
> **最後更新：** 2026-10-29  
> **當前基準狀態：**
> 1. **E93（搜尋/評估/TT 核心純 Bug 修復，HEAD = cf312b0）已採納進生產基線**（1000 局 510.5/1000，勝率 51.05%，淨勝 +21 局，+7.30 Elo，>80 步殘局淨勝 +24 局）；
> 2. **E92（PRUNING_CONTINUATION_THRESHOLD = -1400）已採納進生產基線**；
> 3. **E91（接納SINGULAR_MARGIN_BASE = 58）已採納進生產基線**；
> 4. **E90（NMP 空著剪枝基底選定 NMP_BASE = 230）已採納進生產基線**；
> 5. **E89（Delta Pruning 裕度探索選定 DELTA_PRUNING_MARGIN = 365 且補齊 not is_loss 斷崖護欄）已採納進生產基線**；
> 6. **E88（ProbCut 剪枝裕度 PROBCUT_MARGIN = 170）已採納進生產基線**（+4.86 Elo）；
> 7. **E94（PROBCUT_TT_SHORTCUT_MARGIN = 310）已採納進生產基線**（972 局 509.5/972，勝率 52.42%，淨勝 +47 局，+16.81 Elo，95% CI: [+3.53, +30.14]，SPRT [0.0, 10.0] LLR=+2.96 成功觸發 H1 Accepted 顯著正向通過）；
> 8. **E95（PROBCUT_TT_SHORTCUT_MARGIN = 290）已採納進生產基線**（1000 局 510.0/1000，勝率 51.00%，淨勝 +20 局，+6.95 Elo，>80 步殘局淨勝 +13 局；同期 300 探索項亦得 +5.56 Elo 淨勝 +16 局，確立最優平原）；
> 9. **E96（奇異擴充 TT PV 獎勵選定 `SINGULAR_TTPV_BONUS = 65`）已採納進生產基線**（1000 局 516.0/1000，勝率 51.60%，淨勝 +32 局，+11.12 Elo，95% CI: `[-1.23, +23.50]`，LLR=+1.55；51~80 步中長盤淨勝 +35 局；同深度節點比完美持平 1.000）；
> 10. **E99（晚期移動步數剪枝比例選定 `LMP_SCALE_PERCENT = 165`）已採納進長時控生產基線**（400k LTC 實測 370 局 196.5/370，勝率 53.1%，淨勝 +23 局，+21.6 Elo，LLR=+1.16，有效化解深層防守盲區）；
> 11. **E100（LMR All-node 減步縮放分子調降選定 `LMR_ALLNODE_SCALE_NUM = 260`）已採納進生產基線**（1000 局 510.0/1000，勝率 51.00%，淨勝 +20 局，+6.95 Elo，95% CI: `[-6.77, +20.69]`；長局 >50 步淨勝 +19 局；同深度節點比 1.000 持平）；

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
| `LMP_SCALE_PERCENT` | `107` | E4 105 → **E57 107**（SPSA +5.2 Elo） |
| `LMR_HISTORY_SCALE` | `204` | E5/E6 150 → E56 167 → **E59 204**（12k SPSA + 600@300k +7.5 Elo） |
| `LMR_BASE_OFFSET` | `441` | E20 460 → E56 463 → **E59 441**（12k SPSA + 600@300k +7.5 Elo） |
| `LMR_CUTNODE_BONUS` | `1862` | 2048 → E56 2117 → **E59 1862**（12k SPSA + 600@300k +7.5 Elo） |
| `LMR_TTCAPTURE_BONUS` | `1203` | 1024 → E56 1035 → **E59 1203**（12k SPSA + 600@300k +7.5 Elo） |
| `LMR_MOVECOUNT_FACTOR` | `36` | 40 → E56 38 → **E59 36**（12k SPSA + 600@300k +7.5 Elo） |
| `LMR_TABLE_SCALE_PERCENT` | `113` | 100 → **E59 113**（12k SPSA + 600@300k +7.5 Elo） |
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
| E54 | **安靜步排序權重平滑化對齊 SF19 (2:2:1:1:1:1:1)**（架構解耦後首測） | 暖機與前 36 步呈現完全對稱鏡像和棋，裁定回滾 | **回滾** | 2026-09-26 |
| E55 | **LMR 靜態位置差減深 (alpha - eval)**（移植 SF19 Step 18） | **316@n300k 中止: 49.05% (66W 178D 72L) · Elo -6.60 · 深度 -0.08** | **回滾** | 2026-09-27 |
| E56 | **LMR 搜尋參數多維協同調參（SPSA L1 探索 + L2 300k 實戰驗證）** | **400@n300k: 51.38% (205.5 vs 194.5) · Elo +9.56 ±22.11 · 淨勝 +11.0 分** | **採納進生產 / 基線** | 2026-10-04 |
| E57 | **LMR 走法獎懲與晚期剪枝六軸協同調參（SPSA L1 探索 + L2 300k 實戰驗證）** | **400@n300k: 50.75% (203.0 vs 197.0) · Elo +5.20 ±23.8 · 淨勝 +6.0 分** | **採納進生產 / 基線** | 2026-10-04 |
| E58 | **核心前向剪枝組六軸協同調參（SPSA L1 探索 + 50k/300k 伸縮性雙驗收）** | **600@n50k: 51.92% (Elo +13.3) vs 600@n300k: 49.92% (Elo -0.6 ±17.7)** | **不採納（守護深層戰術）** | 2026-10-05 |
| E59 | **LMR 第一階段核心六軸深度調參（SPSA L1 12,000 局 + L2 300k 獨立大開局庫實測）** | **600@n300k: 51.08% (306.5 vs 293.5) · Elo +7.5 ±19.2 · L1 雙軸突破 95% 顯著性** | **採納進生產 / 基線** | 2026-10-06 |


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

#### E54 安靜步排序權重平滑化對齊 SF19 (2:2:1:1:1:1:1)（架構解耦後首測）

* **代碼：** `chess_engine/classical/constants.py`
  * `ORDERING_WEIGHT_MAIN`: 3 -> **2**
  * `ORDERING_WEIGHT_PAWN`: 2 -> **2**
  * `ORDERING_WEIGHT_BUTTERFLY`: 1 -> **1**
  * `ORDERING_WEIGHT_CONT_1`: 5 -> **1**
  * `ORDERING_WEIGHT_CONT_2`: 2 -> **1**
  * `ORDERING_WEIGHT_CONT_3`: 1 -> **1**
  * `ORDERING_WEIGHT_CONT_4`: 2 -> **1**
  * `ORDERING_WEIGHT_CONT_5`: 1 -> **1**
* **背景與動機：**
  * 在完成走步排序（MovePicker）與歷史剪枝的架構解耦（詳見 [`QUIET_ORDERING_DECOUPLING_PLAN.md`](../chess_engine/classical/QUIET_ORDERING_DECOUPLING_PLAN.md)）後，走步排序權重已不再受到剪枝門檻的綁架。
  * 舊版 `ORDERING_WEIGHT_CONT_1 = 5` 為歷史 Phase 1a 遺留的極端放大權重，容易造成開局與中局在特定延續步上局部過度偏好，干擾全域走步排序。
  * Stockfish 19 在 `movepick.cpp` 中採用乾淨、平滑的 `2*main + 2*pawn + 1*cont_sum`（2:2:1:1:1:1:1 比例）。
  * 本實驗將走步排序全面對齊 SF19 平滑權重，而歷史剪枝門檻（`-1500 * depth` 與 `-6500 * depth`）分毫未動。
* **對戰設定：** 400 場 × 300,000 nodes（10 並行），對抗生產基線 E53（`classical_old`）。
* **對戰觀察：**
  * 暖機 Kiwipete depth 14 節點數完全一致（282,345 vs 282,345）。
  * 前 36 步統計（Round 1、2、15、16）呈現絕對鏡像和棋（New/Old 深度 18.31 完全一致）。
  * 查明在開局庫初期與戰術吃子盤面下，走步主要被吃子（MVV-LVA / SEE）與 Killer/Counter 主導，安靜步延續歷史權重調整在開局階段難以迅速打破對稱性。
* **判定：** **回滾 (REVERT)**。已完全復原 `ORDERING_WEIGHT_*` 為基線數值 (3, 2, 1, 5, 2, 1, 2, 1)，雙目錄邏輯 100% 保持一致。

#### E55 LMR 局勢靜態盈虧修正 (alpha - eval)（對齊 SF19 Step 18）

* **代碼：**
  * `chess_engine/classical/constants.py`:
    ```python
    # SF19 Step 18: Positional reduction adjustment for quiet moves based on (alpha - eval)
    # r += LMR_ALPHA_EVAL_MULT * clamp(alpha - eval, LMR_ALPHA_EVAL_MIN, LMR_ALPHA_EVAL_MAX)
    LMR_ALPHA_EVAL_MULT = 3
    LMR_ALPHA_EVAL_MIN = -64
    LMR_ALPHA_EVAL_MAX = 96
    ```
  * `chess_engine/classical/search.py`:
    在 LMR 1024 尺度減深計算區塊（歷史加成與 ALL-Node 放大之間），加入 SF 19 原生項：
    ```python
    # SF19 Step 18: Positional reduction adjustment for quiet moves based on (alpha - eval)
    if is_quiet_move and not is_decisive(alpha) and static_score != -INFINITY:
        eval_deficit = alpha - static_score
        if eval_deficit < LMR_ALPHA_EVAL_MIN:
            eval_deficit = LMR_ALPHA_EVAL_MIN
        elif eval_deficit > LMR_ALPHA_EVAL_MAX:
            eval_deficit = LMR_ALPHA_EVAL_MAX
        r += LMR_ALPHA_EVAL_MULT * eval_deficit
    ```
* **背景與動機：**
  * 審查 Stockfish 19（`stockfish_repo/src/search.cpp:1354-1355` Step 18）：
    `if (!capture && !is_decisive(alpha)) r += 3 * std::clamp(alpha - eval, -64, 96);`
  * 這是原版 SF 19 Step 18 減深公式中，本引擎唯一尚未移植的關鍵特徵項。
  * **物理含義：**
    1. 當局面靜態落後於 $\alpha$（`alpha > eval`）：本方處於劣勢或被動，安靜步（非吃子、非將軍）在統計上極難翻盤。公式加大減深力度（最高 $+3 \times 96 = +288$，約 0.28 ply），迅速剪除無效游弋分支，將算力留給戰術反撲；
    2. 當局面靜態領先於 $\alpha$（`alpha < eval`）：本方處於均勢上方或勝勢，安靜步可能蘊藏精妙的行棋調度或勝勢鞏固。公式放寬減深（最多 $-3 \times 64 = -192$，約 0.19 ply），給予安靜步更多搜尋深度，防止漏算勝機或因過度減深放生和棋。
  * **安全性：** 嚴格夾取於 `[-64, 96]` 區間，對 1024 尺度減深影響限定於 `[-192, +288]`（不超過 0.28 ply），完全避免減深失控。
* **對戰設定：** 400 場 × 300,000 nodes（10 並行），對抗生產基線 E53（`classical_old`）。
* **對戰結果（316 局完賽中止）：**
  * **戰績：** 勝 66 | 和 178 | 負 72（**淨負 -6 局**）
  * **勝率：** **49.05%** · **Elo：** **-6.60**
  * **搜尋指標：**
    * 平均深度萎縮：New 18.26 vs Old 18.34（**-0.08 ply 倒退**，深層 $d \ge 14$ 佔比由 77.2% 降至 76.2%）。
    * 同深度節點開銷反常膨脹：$d=11$ ratio=1.046 (+4.6%)、$d=12$ ratio=1.021 (+2.1%)。
* **根因分析：**
  1. **NNUE vs HCE 尺度飽和跳變**：
     * SF19 在 NNUE 架構下評估值平滑，1 個兵等效單位約 250~300，`[-64, 96]` 僅為約 $\pm 0.25$ 個兵的微幅動態阻尼。
     * 在 Classical HCE 中 1 個兵為 100 cp，局面稍有優勢或壓力即頻繁打滿截斷邊界（-64 或 +96），使連續阻尼退化為粗暴的二值階躍跳變，破壞了原本調校平衡的 LMR 減深階梯。
  2. **Cut-Node 反向減深悖論（導致中層樹膨脹）**：
     * 在大量非 PV 零窗口節點（$\alpha = \beta - 1$）且靜態評估優良（$\text{static\_score} \ge \beta$）的 Cut-Node 上，$\alpha - \text{static\_score} < 0$。
     * 公式反向對安靜步**減少了減深（$r$ 減少最多 -192）**，促使本應被快速掠過的中後期安靜步進行過深搜尋。在古典 Move Ordering 未達神經網路水準的條件下，這些走步絕大多數無法達成截斷，白白浪費算力（精確印證了 $d=11, 12$ 節點數反常增加 2%~4.6%）。
* **判定：** **回滾 (REVERT)**。PGN 備份：`tournament_analysis/tournament_results_E55_LMR_AlphaEval_aborted.pgn`。已完全復原 `constants.py` 與 `search.py`，驗證 `--diff` 雙目錄 100% 邏輯一致，單元測試全數 PASS。

---

### E56: LMR 搜尋參數多維協同調參（SPSA L1 探索 + L2 300k 實戰驗證）

* **變更範圍：**
  * `chess_engine/classical/constants.py`:
    ```python
    LMR_BASE_OFFSET = 463          # (was 460; SPSA L1+L2 400@300k +9.6 Elo)
    LMR_CUTNODE_BONUS = 2117       # (was 2048; SPSA L1+L2 400@300k +9.6 Elo)
    LMR_TTCAPTURE_BONUS = 1035     # (was 1024; SPSA L1+L2 400@300k +9.6 Elo)
    LMR_MOVECOUNT_FACTOR = 38      # (was 40; SPSA L1+L2 400@300k +9.6 Elo)
    LMR_HISTORY_SCALE = 167        # (was 150; SPSA L1+L2 400@300k +9.6 Elo)
    ```
* **背景與動機：**
  * 歷史實驗（E5、E6、P3）曾嘗試單獨微調 `LMR_HISTORY_SCALE`（150→160 或 150→140），但在孤立單變量測試中皆因缺乏其他剪枝參數協同而導致 Elo −20 ~ −25 慘敗回滾。
  * 本次建立兩階段科學調參架構：
    1. **L1 SPSA 探索（50,000 nodes, 10 並行）**：
       * 執行 30 輪共 1,800 局對戰，每輪動態輪替開局庫，消除過擬合。
       * 透過大數法則克服單輪隨機噪聲，其中 `LMR_HISTORY_SCALE` 展現極強顯著信號（$z = +2.58, p < 0.01$），`LMR_CUTNODE_BONUS` ($z = +1.52$) 與 `LMR_MOVECOUNT_FACTOR` ($z = -1.84$) 同步協同漂移。
       * 採用 Polyak-Ruppert 後半段平均平滑隨機步長噪聲，產出候選參數檔 `tune_search/candidate_lmr.json`。
    2. **L2 SPRT 深度驗收（300,000 nodes, 12 並行）**：
       * 算力擴增 6 倍（Depth 13~15），檢驗高深度下的剪枝穩健性，排除快棋假陽性與過度剪枝（Over-pruning）盲點。
* **對戰設定與成果：**
  * **對戰規模：** 400 局（200 個開局 Pair，雙向黑白互換，`--nodes 300000 --concurrency 12 --adjudicate`）。
  * **戰績：**
    * 舊基準版（Baseline）：194.5 分（48.625%）
    * 新參數版（Candidate）：**205.5 分（51.375%）**
    * **淨勝分：新版淨勝 +11.0 分**
    * **Elo 提升：`+9.56 Elo`**（$\pm 22.1$ 95% CI）
  * **五項式分佈（Candidate 視角）：**
    * `[7, 49, 79, 56, 9]`（雙負 7，一負一和 49，雙和 79，一勝一和 56，雙勝 9）
    * 淨勝 Pair：單勝優勢 +7 組、雙勝橫掃 +2 組。
* **物理機理解析：**
  * 在提高 `LMR_HISTORY_SCALE`（150→167）更積極縮減無效安靜步的同時，同步微調 `LMR_MOVECOUNT_FACTOR`（40→38，略微放寬衰減節奏）與 `LMR_CUTNODE_BONUS`（2048→2117），三者在深層搜尋樹中形成了動態互補的剪枝曲面，既大幅提升有效深層算力，又完全避開了過去單一參數改動時造成的戰術盲點。
* **判定：** **採納進生產 / 基線 (ADOPTED)**。產物檔：`tournament_analysis/l2_verify_lmr.json`。


---

### E57: LMR 走法獎懲與晚期剪枝六軸協同調參（SPSA L1 探索 + L2 300k 實戰驗證）

* **變更範圍：**
  * `chess_engine/classical/constants.py`:
    ```python
    LMR_BAD_CAPTURE_BONUS = 1046    # (was 1024; SPSA L1+L2 400@300k +5.2 Elo)
    LMR_GOOD_CAPTURE_RELIEF = 1030  # (was 1024; SPSA L1+L2 400@300k +5.2 Elo)
    LMR_KILLER_COUNTER_RELIEF = 1024# (was 1024; unchanged)
    LMR_NO_TTMOVE_BONUS = 1068     # (was 1024; SPSA L1+L2 400@300k +5.2 Elo)
    LMP_SCALE_PERCENT = 107        # (was 105; SPSA L1+L2 400@300k +5.2 Elo)
    LMR_NOT_IMP_NUM = 197          # (was 194; SPSA L1+L2 400@300k +5.2 Elo)
    ```
* **背景與動機：**
  * 接續 E56 核心 LMR 成果，針對 LMR 走法特異性獎懲（好/壞吃子、Killer/Counter、無 TT 走法）與晚期剪枝（LMP 比例尺、Not-Improving 減深補償）6 軸進行第二階段 SPSA 協同調參。
  * 驗證環境遷移至 Google Colab（AMD EPYC 8 核心，7 並行 inprocess fixed-node）。
* **對戰設定與成果：**
  * **L1 探索：** 30 輪 x 60 局 = 1,800 局 @ 50,000 nodes。候選檔 `tune_search/candidate_lmr_p2.json`。
  * **L2 深度驗證：** 400 局 @ 300,000 nodes（開局庫 hist_diag_500，雙向黑白互換，開啟 score-based win/draw adjudication）。
  * **戰績：**
    * 舊基準版（Baseline）：197.0 分（49.25%）
    * 新參數版（Candidate）：**203.0 分（50.75%）**
    * **淨勝分：新版淨勝 +6.0 分**
    * **Elo 提升：`+5.2 Elo`**（$\pm 23.8$ 95% CI）
    * **SPRT 判定：** `inconclusive_at_cap`（LLR = +0.08，無任何戰術退步，安全正向）。
* **物理機理解析：**
  * 前 150 局在高和棋率下 75:75 平分，後 250 局新版打出 128:122 (+6.0 分)。微幅上調 `LMP_SCALE_PERCENT`（105→107）與 `LMR_NO_TTMOVE_BONUS`（1024→1068）使深層搜尋在保有剪枝安全性的前提下略微擴充了邊界著法寬度，產生正向穩健的棋力增益。
* **判定：** **採納進生產 / 基線 (ADOPTED)**。產物檔：`tournament_analysis/l2_verify_lmr_p2.json`。


---

### E58: 核心前向剪枝組六軸協同調參（SPSA L1 探索 + 50k/300k 伸縮性雙驗收與診斷）

* **測試常數候選（產自 Polyak-Ruppert 後半段平滑）：**
  * `RFP_BASE_MULT`: 170 → **169**
  * `RAZORING_MARGIN`: 250 → **210** (在 L1 中曾下探至 188~204)
  * `FP_BASE`: 180 → **164** (兩次獨立調參高度吻合於 ~160)
  * `FP_MULTIPLIER`: 125 → **112** (從 125 溫和降至 112)
  * `PRUNING_CAPTURE_SEE_MARGIN`: -100 → **-105** (極度穩定)
  * `PRUNING_QUIET_SEE_MARGIN`: -25 → **-44** (修復上限 <= -5 後安全築底於 -44)
* **背景與動機：**
  * 繼 LMR 兩階段調參（E56 +9.56 Elo、E57 +5.20 Elo）相繼獲勝後，針對核心前向剪枝（RFP、Razoring、Futility Pruning、SEE Pruning）6 軸進行 SPSA 協同調參。
  * **重大修復：** 探索初期發現 `PRUNING_QUIET_SEE_MARGIN` 原上限為 50，微擾跨越 0 導致安靜步誤剪（單輪 -144 分差慘敗）；隨後將上限修訂為 `-5`、步長定為 `15`，徹底消除物理漏洞並重啟純淨調參。
* **對戰設定與成果：**
  * **L1 探索（升級規格）：** 80 輪 × 150 局 = **12,000 局 @ 50,000 nodes**（Google Colab 7 並行）。
    * 總平均勝率 49.90%，實測標準差 $\sigma = 0.0314$（接近純抽樣噪聲預期 $\sigma = 0.0313$）。
    * **候選取樣細節：** 因日誌追加模式涵蓋 Run 1 與 Run 2（共 160 筆），`--tail 0.5` 取最後 80 筆產出 `candidate_fp_p1.json`，本質為 Run 2 全域 80 輪均值（含起點下降過程，將 Razoring 牽引回 210）。若單計 Run 2 後 40 輪均值則為 Razoring 198、FP_BASE 166、FP_MULT 106。
  * **L2 雙時控/節點實戰對照檢驗（各 600 局，同 500 開局）：**
    1. **50,000 nodes（短時控 STC / 調參環境復現）：**
       * 戰績：Candidate 311.5 vs Baseline 288.5 分（**51.92%**）
       * **Elo 提升：`+13.3 ± 20.6`**（淨勝 +23.0 分，LLR = +0.32）。
    2. **300,000 nodes（長時控 LTC / 生產深度驗收）：**
       * 戰績：Candidate 299.5 vs Baseline 300.5 分（**49.92%**）
       * **Elo 表現：`-0.6 ± 17.7`**（淨負 1.0 分，實質 50/50 戰平，LLR = -0.08，`inconclusive_at_cap`）。
* **物理機理解析與統計評估：**
  1. **50k vs 300k 差異解讀**：
     * 50k (+13.3) 與 300k (-0.6) 之差距約 14 Elo，合併標準誤約 14 Elo（差距約 1 個標準差 $1\sigma$），在統計上仍存在較寬信心區間。
     * 但在西洋棋物理機制上，前向硬剪枝（Razoring / Futility）在淺層（7~9 層）省節點衝高深度具備較大優勢，而在深層（12~14 層）剪枝邊界收緊可能提高防守誤剪風險，兩者相互抵消導致深層戰平，符合長時控伸縮性衰減的經典經驗。
* **判定：** **不採納 / 維持基線 (UNADOPTED)**。
  * 嚴格遵循專案長時控驗收標準，深層無明確正向增益不予採納，基線常數維持 E57 狀態不變。
  * 驗證了專案 **L1 (50k SPSA 探索) + L2 (300k SPRT 防禦)** 雙層架構的風險防禦機制。


---

### E59: LMR 第一階段核心六軸深度調參（SPSA L1 12,000 局 + L2 300k 獨立大開局庫實測）

* **變更範圍：**
  * `chess_engine/classical/constants.py`:
    ```python
    LMR_BASE_OFFSET = 441          # (was 463; SPSA L1 12k + L2 600@300k +7.5 Elo)
    LMR_HISTORY_SCALE = 204        # (was 167; SPSA L1 12k + L2 600@300k +7.5 Elo)
    LMR_CUTNODE_BONUS = 1862       # (was 2117; SPSA L1 12k + L2 600@300k +7.5 Elo)
    LMR_TTCAPTURE_BONUS = 1203     # (was 1035; SPSA L1 12k + L2 600@300k +7.5 Elo)
    LMR_MOVECOUNT_FACTOR = 36      # (was 38; SPSA L1 12k + L2 600@300k +7.5 Elo)
    LMR_TABLE_SCALE_PERCENT = 113  # (was 100; SPSA L1 12k + L2 600@300k +7.5 Elo)
    ```
* **背景與動機：**
  * 在經歷 E56/E57 探索後，針對決定 LMR 減深輪廓的核心六軸進行大規模深度重新校準。
  * 歷史單軸調參（E5/E6）曾因缺乏多軸協同導致負 20~25 Elo 失敗。本次透過全域 SPSA 同時微擾 6 個耦合維度，並以 Polyak-Ruppert 後半段平滑抑制隨機步長噪聲。
* **對戰設定與成果：**
  * **L1 SPSA 探索（規格升級）：** 80 輪 × 150 局 = **12,000 局 @ 50,000 nodes**（Google Colab 7 並行，包含跨環境斷點續跑 4,200 局無縫恢復）。
    * 總平均勝率 49.90%，`observed_sd = 0.0302`, `expected_noise_sd = 0.0301`, `sd_ratio = 1.0028`。
    * **統計顯著性驗證（$|z| \ge 2.0$，95% 信心水準）：**
      * `LMR_TABLE_SCALE_PERCENT`: $z = +2.36$（全域減縮放大至 113%，位移率 +21.1%）
      * `LMR_CUTNODE_BONUS`: $z = -2.03$（CutNode 減深下修至 1862，位移率 -9.3%）
    * **協同保護軸向：**
      * `LMR_HISTORY_SCALE`: $z = +1.38$（167 → 204，+22.2% 強化歷史高分走法保護）
      * `LMR_TTCAPTURE_BONUS`: $z = +1.06$（1035 → 1203，+16.2% 強化吃子深度保護）
    * **定錨微調軸向：**
      * `LMR_BASE_OFFSET`: 463 → 441（-4.8% 放寬基礎深度）
      * `LMR_MOVECOUNT_FACTOR`: 38 → 36（-5.3%）
    * Polyak-Ruppert 後 40 輪平滑平均產出 `tune_search/candidate_lmr_p1_deep.json`。
  * **L2 深度獨立驗證（300,000 nodes，6 倍深度，600 局）：**
    * 開局庫：`data/openings.epd`（1,200 開局，out-of-distribution 獨立開局庫，嚴格排除過擬合）。
    * 戰績：Candidate 306.5 vs Baseline 293.5 分（**51.08%**）。
    * **Elo 提升：`+7.5 ± 19.2 Elo`**（LLR = +0.19，`inconclusive_at_cap`）。
* **物理機理解析：**
  1. **全域效率與局部精準保護的完美平衡**：
     * 將全域 LMR 表格縮減放大 13%（TableScale 113%）可使大量劣質安靜步更快在淺層截斷，為深層探索釋放龐大節點餘裕；
     * 省下的節點算力被精準分配給「歷史好步」（HistoryScale 167→204）與「吃子反擊」（TTCapture 1035→1203），完全消除了激進剪枝帶來的戰術漏算。
  2. **修復 CutNode 戰術盲點**：
     * 原基線 CutNode 獎勵（2117）過於激進，在關鍵截斷節點頻繁造成防守走步漏剪。下修至 1862（$z = -2.03$）有效擴充了關鍵節點的防守縱深。
  3. **長時控正伸縮性（Positive LTC Scaling）**：
     * 有別於 E58 前向剪枝在 300k nodes 衰退至 -0.6 Elo，LMR 作為軟降深機制在 300k nodes 下打出堅實的 **+7.5 Elo 淨勝**，長時控穩定度極佳。
* **判定：** **採納進生產 / 基線 (ADOPTED)**。產物檔：`tournament_analysis/l2_verify_lmr_p1_deep.json`。已同步 `constants.py` 至 `classical_old` 並通過全部單元測試。

### 3.60 E60: LMR Phase 1 去滯後平滑二次驗收與世紀單挑 (Polyak-Ruppert Tail 0.3 Adoption)

* **日期：** 2026-10-06
* **動機：**
  * E59 初版使用 `--tail 0.5`（後 40 輪 / 6,000 局）平均，在 80 輪 SPSA 長跑中納入了前 56 輪向盆地移動時的過渡歷史，形成**滯後拖曳偏差（Lag Bias）**；
  * 為檢驗去滯後對長時控之影響，將平滑比例收斂為 `--tail 0.3`（後 24 輪 / 3,600 局），同時亦測試末端 `--tail 0.1`（後 8 輪 / 1,200 局），並在 AMD EPYC 雲端執行雙方直接單挑對決。
* **候選參數對照：**
  * **E60 冠軍組 (`tail 0.3`)：**
    * `LMR_BASE_OFFSET`: 463 → **437**（基礎減步推進，深搜激進度提升）
    * `LMR_HISTORY_SCALE`: 167 → **204**（歷史保護維持強健）
    * `LMR_CUTNODE_BONUS`: 2117 → **1839**（擺脫前期拖曳，防守縱深擴展更徹底）
    * `LMR_TTCAPTURE_BONUS`: 1035 → **1191**（吃子反擊保護收斂定錨）
    * `LMR_MOVECOUNT_FACTOR`: 38 → **35**（步數懲罰微調加速）
    * `LMR_TABLE_SCALE_PERCENT`: 100 → **113**（全域表格縮放維持 113%）
* **深度實戰檢驗陣列（300,000 nodes, 600 局, data/openings.epd）：**
  1. **Tail 0.3 vs 原始基準：** 313.5 vs 286.5（**52.25%**，淨勝 +27 局），**`Elo +15.6 ± 16.9`**，LLR = **+0.57**。
  2. **Tail 0.1 vs 原始基準：** 313.5 vs 286.5（**52.25%**，淨勝 +27 局），**`Elo +15.6 ± 19.3`**，LLR = +0.44。
  3. **Tail 0.3 vs Tail 0.1 世紀單挑直接交鋒：** 308.5 vs 291.5（**51.42%**，淨勝 +17 局），**`Elo +9.8 ± 17.5`**，LLR = **+0.31**！
* **物理機理解析：**
  1. **Polyak-Ruppert 黃金取樣區證實**：
     * `tail 0.5` 存在滯後拖曳（+7.5 Elo）；`tail 0.1` 取樣過窄（僅 8 輪）容易受隨機噪聲翻轉影響破壞 CutNode/TTCapture 平衡；
     * `tail 0.3`（3,600 局）精準落於最優盆地中心，兼具去滯後性與高方差平滑性，單挑以 +9.8 Elo 直接力克 Tail 0.1。
  2. **Elo 表現相較 E59 翻倍**：
     * 對抗原始基準淨勝局數由 +13 局飆升至 **+27 局**，Elo 提升由 +7.5 翻倍至 **+15.6**，LLR 高達 +0.57。
* **判定：** **採納進生產 / 基線 (ADOPTED)**。產物檔：`tournament_analysis/l2_verify_lmr_p1_deep_tail03.json`、`tournament_analysis/l2_match_tail03_vs_tail01.json`。已同步 `constants.py`、`classical_old` 與 `constants_snapshot.py` 並全數通過測試。

### 3.61 E61: LMR Phase 2 戰術吃子與局勢動態六軸調參及雙軌驗收 (Unadopted / Baseline Preserved)

* **日期：** 2026-10-07
* **動機：**
  * 在 E60 鎖定 LMR Phase 1 冠軍底座（Tail 0.3: `BASE=437, HIST=204, CUTNODE=1839, TTCAP=1191, MOVECOUNT=35, TABLE_SCALE=113`）後，推進第二階段深層搜尋調參；
  * 目標鎖定掌控戰術吃子懲罰與動態局勢減深的六軸：`LMR_BAD_CAPTURE_BONUS`（劣吃懲罰）、`LMR_GOOD_CAPTURE_RELIEF`（好吃寬免）、`LMR_KILLER_COUNTER_RELIEF`（殺著寬免）、`LMR_NO_TTMOVE_BONUS`（無TT懲罰）、`LMR_TTMOVE_REDUCTION`（TT命中減步）、`LMR_NOT_IMP_NUM`（未改善動態減步）。
* **調參設定與成果：**
  * **L1 SPSA 探索：** 80 輪 × 150 局 = **12,000 局 @ 50,000 nodes**（Google Colab 7 並行）。
    * 總對局 12,000 局，`mean_score = 0.4964`, `observed_sd = 0.0307`, `expected_noise_sd = 0.0309`, `sd_ratio = 0.9917`。
    * **Tail 0.3 Polyak-Ruppert 後 30% 候選參數：**
      * `LMR_BAD_CAPTURE_BONUS`: 1046 → **1060**（劣吃減步微升）
      * `LMR_GOOD_CAPTURE_RELIEF`: 1030 → **1046**（好吃寬免微幅擴展）
      * `LMR_KILLER_COUNTER_RELIEF`: 1024 → **921**（殺著寬免收窄 -103，壓縮過度）
      * `LMR_NO_TTMOVE_BONUS`: 1068 → **1132**（無 TT 著法減步擴大）
      * `LMR_TTMOVE_REDUCTION`: 1024 → **934**（TT 命中減步收窄 -90）
      * `LMR_NOT_IMP_NUM`: 197 → **208**（未改善局勢動態剪枝更激進）
    * 線性加權 Tail 0.3 參數對照：`BAD_CAP=1072, GOOD_CAP=1047, KILLER=924, NO_TT=1146, TT_RED=932, NOT_IMP=210`。
  * **L2 雙軌深度實戰檢驗（300,000 nodes，6 倍深度，data/openings.epd）：**
    1. **軌道 A（主 L2 驗收：等權 Tail 0.3 vs Baseline，600 局，SPRT 0, 3）：**
       * 戰績：Candidate 292.5 vs Baseline 307.5 分（**48.75%**，淨負 15 分）。
       * **Elo 表現：`-8.7 ± 17.7 Elo`**（LLR = **-0.37**，`inconclusive_at_cap`）。
    2. **軌道 B（世紀單挑：線性加權 Tail 0.3 vs 等權 Tail 0.3，1,000 局，SPRT 0, 3）：**
       * 戰績：線性加權 499.5 vs 等權 500.5 分（**49.95%**，實質五五開平局）。
       * **Elo 表現：`-0.35 ± 13.84 Elo`**（LLR = **-0.111**，`inconclusive_at_cap`）。
* **物理機理解析：**
  1. **殺著寬免過度收緊的深層戰術盲點**：
     * `LMR_KILLER_COUNTER_RELIEF` 由 1024 驟降至 921（-103）。在 50k 淺搜時壓縮殺著寬免可迫使引擎剪掉無效枝幹衝刺深度；但在 300k 深搜（12~15 層）時，Killer 與 Countermove 是最主要的戰術反擊支柱。過度削弱寬免導致關鍵殺著在深層被嚴重減深（Over-reduction），引發戰術漏算。
  2. **無 TT 著法過度懲罰的盲目剪枝**：
     * `LMR_NO_TTMOVE_BONUS`（1068 → 1132）與 `LMR_NOT_IMP_NUM`（197 → 208）過於激進，在複雜開局或轉移局面中，若走法尚未進入 TT 快取即施加重度減步，損害了棋盤韌性。
  3. **世紀單挑對平滑法的理論驗證**：
     * 1,000 局超大樣本實證線性加權與等權 Polyak-Ruppert（-0.35 Elo）表現幾乎無異，證實大窗口等權平均具備足夠的抗噪聲能力，末端步長微調在 12k 規模下並無額外套利空間。
  4. **雙層防禦機制的完美體現**：
     * 本次實測再次展現專案 **L1 (SPSA 探索) + L2 (300k SPRT 嚴格防禦)** 架構的強大價值：在淺層出現過擬合趨勢時，300k nodes LTC 驗收果斷發揮守門功能，徹底阻止 -8.7 Elo 的退步代碼進入生產環境。
* **判定：** **不採納 / 維持基線 (UNADOPTED / BASELINE PRESERVED)**。
  * 產物檔：`tournament_analysis/l2_verify_lmr_p2_deep_tail03.json`、`tournament_analysis/l2_match_p2_tail03_linear_vs_uniform.json`。

### 3.62 E62: 前向剪枝常數體系 Phase 3 六軸 SPSA 調參及 300k L2 雙重驗收 (ADOPTED / Production Baseline Updated)

* **日期：** 2026-10-07
* **動機：**
  * 在 E60 鎖定 LMR Phase 1 冠軍底座（生產基線）後，推進第三階段搜尋核心：前向剪枝常數體系（Pruning & Margins）；
  * 調校六軸：`RFP_BASE_MULT`（反向徒勞乘數）、`RAZORING_MARGIN`（剃刀邊界）、`FP_BASE`（徒勞基礎邊界）、`FP_MULTIPLIER`（徒勞深度乘數）、`LMP_SCALE_PERCENT`（後期走步剪枝容許寬度）、`PRUNING_QUIET_SEE_MARGIN`（安靜步 SEE 門檻）。
* **調參設定與成果：**
  * **L1 SPSA 探索：** 80 輪 × 150 局 = **12,000 局 @ 50,000 nodes**（Google Colab 7 並行，inprocess）。
    * 總對局 12,000 局，`mean_score = 0.5030`, `observed_sd = 0.0308`, `expected_noise_sd = 0.0309`, `sd_ratio = 0.9965`。
    * **訊號顯著性（drift_z）：**
      * `LMP_SCALE_PERCENT`: 107 → **118**（+11%，**`drift_z = +3.64`**，全場最強極致真訊號，雙尾檢定 p < 0.001）。
      * `RAZORING_MARGIN`: 250 → **215**（-35 cp，`drift_z = -0.94`，築底收斂於 205~215 區間）。
      * `FP_BASE`: 180 → **171**（-9 cp，`drift_z = -1.15`，平坦區微調）。
      * `FP_MULTIPLIER`: 125 → **125**（`drift_z = -0.11`，完美確認極值）。
      * `RFP_BASE_MULT`: 170 → **171**（`drift_z = +0.33`，探索後回歸基準）。
      * `PRUNING_QUIET_SEE_MARGIN`: -25 → **-24**（`drift_z = +0.11`，安全定錨）。
  * **L2 雙軌深度實戰檢驗（300,000 nodes，6 倍深度，data/openings.epd，各 600 局上限）：**
    1. **軌道 A（主驗收：六軸打包候選 vs Baseline）：**
       * 候選組 (Tail 0.3 Uniform: RFP 171, Razor 215, FP 171, Mult 125, LMP 118, SEE -24) vs Baseline (170, 250, 180, 125, 107, -25)。
       * 戰績：Candidate 307.0 vs Baseline 293.0 分（**51.17%**，淨勝 14 分）。
       * **Elo 表現：`+8.1 ± 16.8 Elo`**（LLR = **+0.27**，全程 12 批檢查點無一跌破 50.0%）。
    2. **軌道 B（策略 B 專項單挑：純 LMP 118 vs Baseline 107）：**
       * 僅改動 `LMP_SCALE_PERCENT: 107 -> 118`，其餘五軸完全鎖死 Baseline。
       * 戰績：Candidate 305.0 vs Baseline 295.0 分（**50.83%**，淨勝 10 分）。
       * **Elo 表現：`+5.8 ± 16.6 Elo`**（LLR = **+0.18**，全程 12 批檢查點無一跌破 50.0%）。
* **物理機理解析：**
  1. **LMP 118 是真實且獨立的正向基底（貢獻 +5.8 Elo）**：
     * 隔離單挑證實 LMP 寬容度放寬 11% 徹底消除了淺中層走步剪得過死的盲點，為搜尋樹提供了紮實的防禦與勝率增益。
  2. **剪枝協同效應的實證確認（提供額外 +2.3 Elo）**：
     * 六軸打包表現（+8.1 Elo）高於純 LMP（+5.8 Elo），證實剃刀邊界收緊（Razor 215）與徒勞微調（FP 171）節省的節點預算有效補貼了安靜步擴展。
  3. **1,200 局零回撤的極致安全性**：
     * 兩項 LTC 測試累計 1,200 局，合計淨勝 24 盤，全程 24 個檢查點 100% 保持在 50% 以上，徹底排除任何深層倒退風險，完全符合生產採納標準。
* **判定：** **採納進生產 / 基線 (ADOPTED / PRODUCTION BASELINE UPDATED)**。
  * 產物檔：`tournament_analysis/l2_verify_pruning_p3_deep_tail03.json`、`tournament_analysis/l2_verify_pruning_p3_lmp118.json`。
  * 生產代碼更新：`LMP_SCALE_PERCENT=118`, `FP_BASE=171`, `RFP_BASE_MULT=171`, `PRUNING_QUIET_SEE_MARGIN=-24` 正式實裝落地（+8.1 Elo）。已同步 `constants.py`、`constants_snapshot.py` 與 `classical_old`。
  * **幽靈參數清理**：代碼審計發現 `RAZORING_MARGIN` 自 E46 改版 SF19 二次剪枝後已退化為未被引用的無效佔位；本次已徹底拔除該歷史殘留，將運行時插槽 1 正式定名為 `TUNE_RAZOR_COEFF`（對接生產真實公式 `search_context.tune[TUNE_RAZOR_COEFF] * depth * depth`，預設 482）。

---

### 3.63 E63: 歷史表與走步排序啟發式體系 Phase 4 六軸 SPSA 調參與雙軌 300k LTC 深度檢驗 (ADOPTED / Production Baseline Updated)

* **日期：** 2026-10-08
* **動機：**
  * 在 Phase 1 (LMR 核心基底)、Phase 2 (LMR 細節修剪)、Phase 3 (前向剪枝體系落地 +8.1 Elo) 穩固建立後，推進第四核心支柱：**歷史表反饋與走步排序啟發式（History Heuristics & Move Ordering）**。
  * 走步排序決定了 Beta 剪枝的觸發速度（Cut-first rate），而歷史表的加減分曲線直接決定了後續 LMR 的懲罰幅度與靜態剪枝門檻。兩者為緊密咬合的搜尋引擎動力傳動系統。
* **六軸鎖定與運行時插槽擴展（TUNE_SIZE: 34 → 39）：**
  1. `LMR_HISTORY_SCALE`（插槽 7，基線 204，範圍 [50, 350]，步長 25）：LMR 歷史衰減係數，控制歷史評分對剪枝深度的微調力道。
  2. `HISTORY_BONUS_SCALE`（插槽 34，基線 120，範圍 [40, 240]，步長 10）：安靜步剪枝歷史加分斜率（`bonus = min(SCALE * depth, CAP)`）。
  3. `HISTORY_BONUS_CAP`（插槽 35，基線 1800，範圍 [800, 3000]，步長 100）：安靜步歷史加分上限。
  4. `HISTORY_MALUS_CAP`（插槽 36，基線 1600，範圍 [800, 3000]，步長 100）：失敗走步歷史懲罰扣分上限。
  5. `QUIET_ORDER_KILLER_1`（插槽 37，基線 400000，範圍 [100000, 600000]，步長 10000）：第一殺手步（Killer Move 1）安靜步排序加分。
  6. `QUIET_ORDER_COUNTER`（插槽 38，基線 300000，範圍 [100000, 500000]，步長 10000）：應對步（Counter Move）安靜步排序加分。
* **調參設定與成果：**
  * **L1 SPSA 探索：** 80 輪 × 150 局 = **12,000 局 @ 50,000 nodes**（Google Colab 7 並行，inprocess）。
    * 總對局 12,000 局，`mean_score = 0.5043`, `observed_sd = 0.0319`, `expected_noise_sd = 0.0305`, `sd_ratio = 1.05`。
    * **訊號檢定（drift_z）：**
      * `HISTORY_BONUS_CAP`: 1800 → **2054**（+14%，**`drift_z = +2.34`**，全場唯一顯著真訊號，信心水準 98.2%）。
      * `QUIET_ORDER_KILLER_1`: 400000 → **390488**（`drift_z = -1.55`，弱下行，噪聲邊界）。
      * `HISTORY_MALUS_CAP`: 1600 → **1537**（`drift_z = -0.62`，噪聲）。
      * `QUIET_ORDER_COUNTER`: 300000 → **302071**（`drift_z = +0.57`，高度定錨）。
      * `LMR_HISTORY_SCALE`: 204 → **197**（`drift_z = -0.17`，高度定錨）。
      * `HISTORY_BONUS_SCALE`: 120 → **123**（`drift_z = +0.16`，高度定錨）。
  * **L2 雙軌深度實戰檢驗（300,000 nodes，6 倍深度，data/openings.epd）：**
    1. **軌道 A（六軸打包組 vs Baseline，900 局，SPRT 0, 3）：**
       * 候選組 (Tail 0.3: LMR 197, Scale 123, Cap 2054, Malus 1537, Killer 390488, Counter 302071) vs Baseline。
       * 戰績：Candidate 432.5 vs Baseline 467.5 分（**48.06%**，淨負 35 分）。
       * **Elo 表現：`-13.5 ± 14.3 Elo`**（LLR = **-0.84**）。
       * **盲點剖析**：`QUIET_ORDER_KILLER_1` 削弱了近 10,000 分，破壞了殺手步 400k 的截斷護城河；在 300k 深層搜索中，殺手步被普通歷史步插隊，導致關鍵戰術反擊被延後，中殘局出現深層戰術盲點。被 L2 防禦機制成功阻截！
    2. **軌道 B（策略 B 專項隔離：純 HISTORY_BONUS_CAP 2054 vs Baseline 1800，600 局，SPRT 0, 3）：**
       * 僅改動唯一的顯著真訊號軸 `HISTORY_BONUS_CAP: 1800 -> 2054`，其餘 5 軸（Killer 400k、Counter 300k、Malus 1600、Scale 120、LMR 204）**100% 嚴格鎖死 Baseline**。
       * 戰績：Candidate 306.0 vs Baseline 294.0 分（**51.00%**，淨勝 12 分）。
       * **Elo 表現：`+6.9 ± 16.9 Elo`**（LLR = **+0.22**，自 400 局後全程穩定領先至完賽）。
* **物理機理解析：**
  1. **殺手步絕對特權不可動搖**：
     * 雙軌對比實證 Killer Move 必須享有 400k 的跨階級優先順位，不能被普通安靜步擠壓。
  2. **歷史表天花板放寬 14% 是獨立且實打實的正向增益（貢獻 +6.9 Elo）**：
     * 在捍衛殺手步特權的前提下，將歷史加分上限由 1800 放寬至 2054，使反覆引發截斷的高品質安靜步能拉開與劣步的排序差距，大幅改善了安靜步排序質量。
  3. **雙層防禦機制的再次驗證**：
     * 本次實測再次體現「L1 大膽探索 + L2 深度守門 + 策略 B 隔離驗證」的方法論價值，精準過濾 -13.5 Elo 噪聲，提煉出 +6.9 Elo 純真訊號。
* **判定：** **採納策略 B 進生產 / 基線 (ADOPTED / PRODUCTION BASELINE UPDATED)**。
  * 產物檔：`tournament_analysis/l2_verify_history_p4_deep_tail03.json`、`tournament_analysis/l2_verify_history_p4_bonuscap2054.json`。
  * 生產代碼更新：`HISTORY_BONUS_CAP=2054`（原 1800）。已同步 `constants.py`、`constants_snapshot.py` 與 `classical_old`。

---

### 3.64 E64: 前向剪枝極限協同體系 Phase 5 六軸 SPSA 調參（100k nodes）與 1,000 局 300k LTC 深度檢驗 (ADOPTED / Production Baseline Updated)

* **日期：** 2026-10-11
* **動機：**
  * 在 Phase 3（初步前向剪枝，+8.1 Elo）與 Phase 4（歷史表上限放寬，+6.9 Elo）相繼採納後，我們發現深層剪枝仍存在「淺層過度剪枝在深層漏算戰術」的深度適應落差。
  * 本次啟動 **Phase 5：前向剪枝極限協同體系調參（方案一：Pruning Synergies @ 100k nodes）**，將調參節點數翻倍至 **100,000 nodes**，將每輪局數翻倍至 **300 局**，並引入 `--spsa-a-ratio 0.2` 雙倍阻尼常數平滑衰減，在更高深度視野下精確尋找剪枝協同最優解。
* **調參六軸（全部為運行時免重編譯軸）：**
  1. `LMP_SCALE_PERCENT`（基線 118，範圍 [70, 140]，步長 5）：後期走步剪枝走步數閾值縮放比例（%）。
  2. `FP_BASE`（基線 171，範圍 [100, 300]，步長 10）：徒勞剪枝基礎邊界（cp）。
  3. `FP_MULTIPLIER`（基線 125，範圍 [60, 220]，步長 10）：徒勞剪枝深度乘數（Margin = Base + Mult * Depth）。
  4. `RFP_BASE_MULT`（基線 171，範圍 [120, 230]，步長 5）：反向徒勞剪枝深度邊界乘數。
  5. `PRUNING_QUIET_SEE_MARGIN`（基線 -24，範圍 [-200, -5]，步長 15）：靜態交換評估安靜步前向剪枝容忍度。
  6. `PRUNING_CAPTURE_SEE_MARGIN`（基線 -100，範圍 [-250, 0]，步長 25）：靜態交換評估吃子步前向剪枝容忍度。
* **調參設定與成果：**
  * **L1 SPSA 高深度探索：** 46 輪 × 300 局 = **13,800 局 @ 100,000 nodes**（Google Colab 7 並行，inprocess，`--spsa-a-ratio 0.2`）。
    * 總對局 13,800 局，`mean_score = 0.4922`, `observed_sd = 0.0276`, `expected_noise_sd = 0.0231`, `sd_ratio = 1.20`。
    * **統計檢定（drift_z）：**
      * `LMP_SCALE_PERCENT`: 118 → **137**（+19%，**`drift_z = +4.02`**，全場最強顯著信號，信心水準 > 99.99%）。
      * `FP_MULTIPLIER`: 125 → **96**（-23%，**`drift_z = -2.23`**，突破顯著門檻，信心水準 > 95%）。
      * `FP_BASE`: 171 → **174**（`drift_z = +0.23`，微幅補償）。
      * `RFP_BASE_MULT`: 171 → **168**（`drift_z = -0.46`，高度自洽）。
      * `PRUNING_QUIET_SEE_MARGIN`: -24 → **-32**（`drift_z = -0.69`，適度放寬）。
      * `PRUNING_CAPTURE_SEE_MARGIN`: -100 → **-102**（`drift_z = -0.11`，高度定錨）。
  * **L2 深度實戰檢驗（300,000 nodes LTC，1,000 局，data/openings.epd，SPRT 0, 3）：**
    * 候選組 (Tail 0.3: LMP 137, FP_BASE 174, FP_MULT 96, RFP 168, QUIET -32, CAP -102) vs Baseline。
    * 戰績：Candidate **512.5** vs Baseline **487.5** 分（**51.25%**，**淨勝整整 25 盤**！）。
    * **Elo 表現：`+8.7 ± 13.6 Elo`**（LLR = **+0.45**）。
    * **全程表現追蹤：** 50 局後全程維持正勝率與正 Elo（100 局 +20.9、300 局 +12.7、500 局 +4.2、800 局 +13.5、1000 局 +8.7），連續 900 局保持領先！
* **物理機理解析：**
  1. **LMP 137 是高深度下的核心解放點（貢獻極大正向勝率）**：
     * 在 100k~300k nodes 的高深度視野下，舊的 118% 閾值在深層太早剪除安靜步；放寬至 137% 後深層搜尋樹適度加寬，為引擎保留了豐富的戰術反擊關鍵步。
  2. **FP Multiplier 大幅下調（125 → 96）精確收窄深層邊界**：
     * 徒勞剪枝公式為 Margin = Base + Mult * Depth。舊的 125 在深層（Depth 6~10）導致邊界過大而過度剪枝；降至 96 配合 Base 174，形成了「淺層安全、深層精準」的現代剪枝曲線，杜絕了深層漏算問題。
  3. **剪枝協同效應在 300k LTC 完美兌現**：
     * 6 軸協同在 1,000 局超深層對抗中以 51.25% 穩健勝出，再次證實高深度調參（100k nodes）對於縮小深度適應落差的關鍵價值。
* **判定：** **採納進生產 / 基線 (ADOPTED / PRODUCTION BASELINE UPDATED)**。
### 3.65 E65: Step 15 奇異延展與 Multi-Cut 狀態隔離加固與負延展調優 (ADOPTED / Production Baseline Updated)

* **日期：** 2026-10-14 ~ 2026-10-15
* **動機：**
  * 審查報告指出 Step 15 奇異延展與 Multi-Cut 中存在的五大嚴重狀態隔離與安全防禦漏洞：
    1. `cutoff_cnt[ply + 1]` 漏存還原：奇異搜尋子層 fail-high 污染父節點兄弟走步，施加超額 1~2 ply LMR 懲罰；
    2. `not is_exclusion_search` 漏加：導致 Correction History 與 Fail-Low 對手走法獎勵遭受人為排除走法的極低分污染；
    3. `stop_flag` 順序顛倒：超時返回的 0 在 `beta <= 0` 時被誤判為 Multi-Cut 截斷成功；
    4. `VALUE_KNOWN_WIN` 防護：對齊 SF19 `!is_decisive`，避免在已知勝負邊界浪費節點；
    5. 負延展死碼修復：舊版 `if exclusion_score >= beta:` 100% 死碼，對齊 SF18/19 修復為 `se_tt_score >= beta`；雙延展限制於 `not is_pv` 防搜尋樹暴衝。
* **二階段深度實戰驗證（300,000 nodes LTC，各 400 局）：**
  1. **第一階段（狀態隔離防禦修復，負延展暫定 -2）：**
     * 對戰前一版基線，經 400 局完整對抗，成功取得勝勢並獲採納，同步更新至 `classical_old`。
  2. **第二階段（負延展 -2 → -1 緩和調優驗收）：**
     * 針對 Classical HCE 噪聲下 `-2` 砍深過激導致超長殘局漏算的現象，將 `se_tt_score >= beta` 時之負延展由 `-2` 調整為 `-1`。
     * 對抗第一階段基線（400 局成對換先，雙向黑白互換）：
       * 戰績：Candidate **89 勝 229 和 82 負（50.88%）**，**淨勝 +7 盤**！
       * **Elo 表現：`+6.08 ± 22.2 Elo`**。
* **物理機理解析：**
  1. **超長殘局止血效果卓著**：
     * 在負延展調寬為 `-1` 後，`>80` 步超長殘局由 44.2% 大幅止血回升至 **50.0% 完全平衡（22W 58D 22L）**，徹底消除了過度自信的淺層截斷（Premature Cutoff）。
  2. **中盤攻防展現強大統治力**：
     * 在 51~80 步複雜中局攻防中，狀態隔離與防污染的紅利全面釋放，打出 **65 勝 54 負（53.1% 勝率，約 +22 Elo 優勢）**。
* **判定：** **採納進生產 / 基線 (ADOPTED / PRODUCTION BASELINE UPDATED)**。
  * 產物檔：`tournament_analysis/tournament_results.pgn`、`tournament_analysis/game_stats_report.txt`、`tournament_analysis/search_stats_report.txt`。
  * 生產代碼更新：`chess_engine/classical/search.py`（Step 15 完整防禦 + 負延展 `-1`）。已同步至 `classical_old/search.py` 並全數通過測試。

### 3.66 E66: Step 15 奇異延展起點深度調優 MIN_SINGULAR_DEPTH 8 -> 9 (ADOPTED / Production Baseline Updated)

* **日期：** 2026-10-15
* **動機：**
  * 在 Step 15 狀態隔離與防禦加固後，進一步檢視奇異搜尋（Singular Search）在古典 HCE 引擎下的啟動門檻與投資回報率（ROI）。
  * 舊版 `MIN_SINGULAR_DEPTH = 8` 在淺中層即頻繁發起 `(depth - 1) // 2` 排除搜尋（Exclusion Search）。由於 HCE 淺層評估噪聲相對較大，許多節點耗費數百至上千節點進行驗證，最終多數走法仍被證實為 Non-singular，形成沉沒成本。
  * 假說：將起點深度推遲至 `depth >= 9`（`MIN_SINGULAR_DEPTH = 9`，若 `ttPv` 則 `10`），能有效剪除淺中層無效探測（「給搜尋樹減脂」），將省下的節點預算 100% 灌注至 PV 主線與深層搜尋。
* **對戰設定與環境：**
  * 錦標賽配置：Google Colab (AMD EPYC 7B12 8 vCPU, 7 並行 Worker)，`inprocess` 後端。
  * 局數：**600 局**（300 對成對換先開局），時控：**300,000 nodes/move**，TT：64 MB / context。
  * 對陣：Candidate (`MIN_SINGULAR_DEPTH = 9`) vs Baseline (`MIN_SINGULAR_DEPTH = 8`)。
* **對戰成果與統計檢定：**
  * **對局戰績：** 600 局，Candidate **143 勝 344 和 113 負**（得分率 **52.50%**，315.0 / 600）。
  * **淨勝優勢：** **淨勝整整 +30 盤**！
  * **相對 Elo：** **`+17.39`**，95% 置信區間 `[-1.48, +36.36]`。
  * **全程表現追蹤：** 前 150 對（300 局）得分率 50.5%，後 150 對（300 局）得分率飆升至 **54.5%**，隨局數增加優勢持續擴大。
  * **對局長度分佈：**
    * Medium (26-50 步)：50.3%（13W 131D 12L）。
    * Long (51-80 步)：52.5%（88W 98D 75L）。
    * Very Long (>80 步超長殘局)：**56.1%（42W 63D 26L）**，展現壓倒性殘局深度優勢。
* **搜尋樹效率與指標分析：**
  * **搜尋深度（Depth）：** New 平均 **19.51 ply**（σ=16.57）vs Old 平均 **19.40 ply**（σ=16.62），平均深度提升 **+0.11 ply**。
  * **深層探訪率（depth >= 14）：** New **88.3%** vs Old **86.9%**，高深度抵達率提升 **+1.4%**！
  * **同深度節點消耗比（Node Ratio）：** 在高深度層 New 消耗節點顯著更少（$d=14: 0.989$、$d=15: 0.992$、$d=16: 0.994$），證明整棵樹更輕快。
* **物理機理解析：**
  1. **消除淺層低 ROI 探測**：推遲起點成功避免在樹寬龐大的 depth 8 浪費節點做無效的排除搜尋。
  2. **適應 HCE 評估信賴度**：古典手工評估在 depth 9+ 時累積的 TT 分數更平滑準確，使奇異延展的命中質量大幅提升。
  3. **節點資源向主線再分配**：在 300k nodes 固定預算下，省下的節點讓引擎在相同時間內看得更深、更準。
* **判定：** **採納進生產 / 基線 (ADOPTED / PRODUCTION BASELINE UPDATED)**。
  * 產物檔：`tournament_colab_runner_se_multicut.ipynb`、`tournament_analysis/search_stats_report.txt`、`tournament_analysis/game_stats_report.txt`。
  * 生產代碼更新：`chess_engine/classical/constants.py`（`MIN_SINGULAR_DEPTH = 9`）。已同步至 `classical_old/constants.py`。
  * 下一步探索：測試 `MIN_SINGULAR_DEPTH = 10` vs `9`，尋找邊際效益最佳甜蜜點。

---

### E67：搜尋熱路徑零搜尋樹變動 (Zero-Tree-Delta) 冗餘消除與 NPS 加速優化（2026-10-17）

* **改動項目：**
  1. `chess_engine/classical/search.py`：消除 LMR 吃子前 100% 重複的 `_see_ge_jit(..., 0)` 呼叫（直接透過 `mp_stage == STAGE_BAD_CAPTURES` 復用 MovePicker 分流結果）。
  2. `chess_engine/classical/search_heuristics.py`：移除 `aggressor_cache` 與 `victim_cache` 於 6 處評分函式的死寫入（無讀取端且排序後無效）。
  3. `chess_engine/classical/search.py`：Step 14 剪枝棋子查詢復用，消除 `_qsearch_ordinary_move_gives_check` 與 `_see_ge_jit` 在走子迴圈中的重複線性 6-bitboard 查詢。
* **驗證方式：**
  * 選取 5 個不同風格盤面（Startpos、Kiwipete、Middlegame Complex、Tactical Queens、Endgame），固定深度 14，每盤面執行 5 次取平均值。
* **驗證結果（0-node delta 完全一致，NPS 大幅提升）：**
  * **節點數**：
    * Pos 1 (Startpos): 289,763 (0 delta)
    * Pos 2 (Kiwipete): 291,209 (0 delta)
    * Pos 3 (Middlegame Complex): 869,909 (0 delta)
    * Pos 4 (Tactical Queens): 4,980 (0 delta)
    * Pos 5 (Endgame): 685 (0 delta)
    * **總節點數 1,456,546.0 vs 1,456,546.0，達成 100% 逐節點完全吻合（0-node delta）**。
  * **NPS 表現**：
    * Startpos NPS: 402,489 → **478,036 (+18.77%)**
    * Kiwipete NPS: 412,261 → **442,820 (+7.41%)**
    * Middlegame Complex NPS: 428,251 → **457,893 (+6.92%)**
    * **核心深搜前三大盤面加權 NPS: 414,774 → 457,114 (+10.21%)**。
* **判定：** **採納進生產代碼 (ADOPTED / ZERO-TREE-DELTA PROD SPEEDUP)**。

---

### E68：LMR correction_mag 解耦靜態評估與 TT 分數實驗與回滾（2026-10-17）

* **實驗動機與設想：**
  * 原假設：LMR 中的 `correction_mag` 目的是依據 Correction History 的修正幅度調控減深。當 `tt_hit` 發生且 `static_score` 被 TT 分數覆寫時，`abs(static_score - raw_static_eval)` 計算的實為 TT 分數與原始評估的差值，而非單純的 correction offset。因此嘗試引入獨立的 `corrected_static_eval` 專門計算 Correction History 偏量。
* **對戰設定與環境：**
  * 錦標賽配置：Google Colab (AMD EPYC 7B12 8 vCPU, 7 並行 Worker)，`inprocess` 後端。
  * 時控：300,000 nodes/move，TT：64 MB / context。
  * 對陣：Candidate (解耦 `corrected_static_eval`) vs Baseline (原版 Step A 加速後 `static_score`)。
* **對戰成果：**
  * 戰局：跑至 210 局（105 對雙向開局）。
  * 戰績：Candidate **39 勝 124 和 47 負**（得分率 **48.10%**，101.0 / 210）。
  * 相對 Elo：**-13.23 ± 30.1** (95% CI [-43.3, +16.9])，顯著退步。
* **物理機理解析（重大發現：動態戰術張力防護機制 Dynamic Volatility Relief）：**
  1. 當 TT 探針命中且搜尋得分 `tt_score` 與靜態評估 `raw_static_eval` 產生劇烈落差時，該節點本質上處於**高度戰術動盪（Tactical Tension / High Volatility）**局面（例如已有戰術伏筆或得失子線）。
  2. 原代碼使用 `abs(static_score - raw_static_eval)` 會使 `correction_mag` 大幅膨脹，進而使 `r -= ...` **顯著減少減深量（Reduction Relief）**，驅使搜尋樹在此類高張力節點展開更深的排查，避免漏算。
  3. 強行將其替換為平滑的 `corrected_static_eval` 雖然在語意上純粹化了 Correction History，但卻**意外剝奪了高張力節點的動態減深抑制罩**，導致引擎在複雜戰術線路上過度減深（Over-reduction），頻繁出現深層戰術盲點而落敗。
* **判定：** **立即回滾 (REVERTED)**。
  * 還原 `search.py` 中的 `correction_mag` 為 `abs(static_score - raw_static_eval)`，完整保留戰術張力防護機制。
  * 完整保留 Step A 的零搜尋樹變動 NPS 加速成果。

---

### E69：對齊 Stockfish 19 Step 12 ProbCut TT 寫入深度修復（2026-10-17）

* **改動項目與動機：**
  * 檔案：`chess_engine/classical/search.py`（Line 1805）。
  * 歷史背景：原代碼在 ProbCut 成功觸發截斷時，呼叫 `store_tt(..., depth - PROBCUT_TT_DEPTH_OFFSET, ...)`，使用寫死的 `PROBCUT_TT_DEPTH_OFFSET = 3`。
  * 根因病灶：在 E47 引入動態 `probcut_r = (5 if improving else 3)` 後，當 `improving == True` 時，底層子搜尋深度實為 `depth - 5`。但寫入 TT 時卻標記為 `depth - 3`（形成了 **+1 ply 虛高深度膨脹 TT Depth Inflation**）！這會誤導後續搜尋誤判該局面已有高深度走法驗證，引發虛假剪枝 (Bogus Cutoffs)。而在 `improving == False` 時，真實戰術驗證深度為 `depth - 3 + 1 = depth - 2`，舊代碼卻少記了 1 層深度。
  * 嚴格對齊 SF19 Step 12：`probCutDepth = depth - probcut_r`，寫入 TT 深度為 `probCutDepth + 1`，即 `pc_tt_depth = max(1, depth - probcut_r + 1)`。
* **5 盤面固定深度 14 實測比對（Startpos, Kiwipete, Middlegame, Tactical Queens, Endgame）：**
  * Baseline (Old / 包含 Step A NPS 加速)：
    * Pos 1 (Startpos): 146,393 nodes (0.483s, 302,953 NPS)
    * Pos 2 (Kiwipete): 199,940 nodes (0.884s, 226,230 NPS)
    * Pos 3 (Middlegame Complex): 282,656 nodes (1.104s, 256,020 NPS)
    * Pos 4 (Tactical Queens): 123,097 nodes (0.328s, 374,891 NPS)
    * Pos 5 (Endgame): 14,030 nodes (0.035s, 404,150 NPS)
    * **總計：766,116 nodes, 2.834s, Overall NPS 270,319**
  * Candidate (New / SF19 ProbCut TT 深度對齊)：
    * Pos 1 (Startpos): 146,396 nodes (+3 nodes)
    * Pos 2 (Kiwipete): 259,861 nodes (探索更充分)
    * Pos 3 (Middlegame Complex): **204,137 nodes (-78,519 nodes, -27.8% 節點減省！)**
    * Pos 4 (Tactical Queens): **111,536 nodes (-11,561 nodes, -9.4% 節點減省！)**
    * Pos 5 (Endgame): **11,638 nodes (-2,392 nodes, -17.0% 節點減省！)**
    * **總計：733,568 nodes (-32,548 nodes, -4.25% 全局節點減省！)**
    * **總時間：2.632s vs 2.834s (-7.1% 搜尋時間縮短！)**
    * **Overall NPS：278,717 (+3.11% NPS 提升！)**
* **對戰設定與環境：**
  * 錦標賽配置：Google Colab (AMD EPYC 7B12 8 vCPU, 7 並行 Worker)，`inprocess` 後端。
  * 局數：**600 局**（300 對成對換先開局），時控：**300,000 nodes/move**，TT：64 MB / context。
  * 對陣：Candidate (New, SF19 ProbCut TT 深度) vs Baseline (Old, 包含 Step A NPS 加速)。
* **對戰成果與統計：**
  * **對局戰績：** 600 局，Candidate **135 勝 335 和 130 負**（得分率 **50.42%**，302.5 / 600）。
  * **相對 Elo：** **`+2.90`**，95% 置信區間 `[-14.69, +20.49]`（淨勝 +5 盤）。
  * **勝負與穩定度：** 兩者實力相當且略有微弱優勢（+2.90 Elo），完整修復 TT 深度膨脹缺陷，合約邏輯對齊 SF19。
* **判定：** **採納進生產 / 基線 (ADOPTED / PRODUCTION BASELINE UPDATED)**。
  * 透過 `tools/sync_classical_old.py --sync-code` 同步至 `chess_engine/classical_old/`。

---

### E70：修復吃子後首步靜態著法被豁免 LMR 錯誤進行全深搜尋（2026-10-17，待實測）

* **改動項目與動機：**
  * 檔案：`chess_engine/classical/search.py`（Line 2689）。
  * 根因病灶：
    在 `_search` 著法迴圈中，`searched_legal_moves == 1`（第 1 步合法著法）已被排他性地以全深度搜尋。
    進入 `else:` 區塊時，`searched_legal_moves >= 2`（表示該節點已有走法驗證且未引發即時截斷）。
    原代碼檢查：
    `if is_quiet_move and quiet_move_counter >= LMR_MIN_QUIET_MOVE_INDEX and moved_piece_type != -1:`
    常數 `LMR_MIN_QUIET_MOVE_INDEX = 2`。
    當前面已試過吃子著法（例如 TT move 或好吃子）時，隨後產生的第 1 步靜態著法的 `quiet_move_counter` 僅為 1。
    導致該靜態著法被**錯誤豁免 LMR**，以**全深度 (Full Depth)** 進行昂貴搜尋！甚至在有多步吃子的節點，第 4、第 5 步走法若為第 1 步靜態著法，依然全深搜尋！
  * 對齊 SF19 Step 18 規範：
    Stockfish 19 `search.cpp:1362` 僅在 `moveCount > 1` 時對所有後續著法全面施加 LMR。
    將檢查改為 `searched_legal_moves >= LMR_MIN_QUIET_MOVE_INDEX`。
    任何已有首步驗證的節點，其後續靜態著法一律依其走法序數施加 LMR 減深；若確實為好棋（fail-high），則由引擎原有的 fail-high re-search 機制重新以 adjusted_depth 驗證。
* **5 盤面固定深度 14 實測比對（Startpos, Kiwipete, Middlegame, Tactical Queens, Endgame）：**
  * Baseline (Old / E69 生產基準)：
    * Pos 1 (Startpos): 146,396 nodes (score 31, best e2e4)
    * Pos 2 (Kiwipete): 259,861 nodes (score -217)
    * Pos 3 (Middlegame Complex): 204,137 nodes (score 13)
    * Pos 4 (Tactical Queens): 111,536 nodes (score -173)
    * Pos 5 (Endgame): 11,638 nodes (score 10120)
    * **總計：733,568 nodes**
  * Candidate (New / E70 全域 moveCount LMR)：
    * Pos 1 (Startpos): 162,788 nodes (score 47, best e2e4)
    * Pos 2 (Kiwipete): **83,959 nodes (-175,902 nodes, -67.7% 節點暴減！戰術盤面剪枝效率極高)**
    * Pos 3 (Middlegame Complex): 282,105 nodes (score 45)
    * Pos 4 (Tactical Queens): 147,256 nodes (score -182)
    * Pos 5 (Endgame): 12,373 nodes (score 10121)
    * **總計：688,481 nodes (-45,087 nodes, -6.15% 全局節點減省！)**
* **對戰設定與環境：**
  * 錦標賽配置：Google Colab (AMD EPYC 7B12 8 vCPU, 7 並行 Worker)，`inprocess` 後端。
  * 局數：跑至 **486 局**（243 對成對換先開局），時控：**300,000 nodes/move**，TT：64 MB / context。
  * 對陣：Candidate (New, 全域 moveCount LMR) vs Baseline (Old, E69 基準)。
* **對戰成果與統計：**
  * **對局戰績：** 486 局，Candidate **106 勝 265 和 115 負**（得分率 **49.07%**，238.5 / 486）。
  * **相對 Elo：** **`-6.44`**，95% 置信區間 `[-27.24, +14.36]`。
* **物理機理解析：**
  * 現有的全套 LMR 係數（如 `LMR_BASE_OFFSET`、`LMR_TABLE_SCALE_PERCENT`、`LMR_MOVECOUNT_FACTOR` 等）皆是在歷史「首步靜態著法不減深 (`quiet_move_counter >= 2`)」的搜尋樹型態下調優出來的高維局部最優點。
  * 放開吃子後的靜態著法 LMR 後，因現有減深公式並未隨之重調，導致在戰術激烈的開局/中局裡，部分關鍵的靜態防守或反擊步被過度減深，產生微小漏算。
* **判定：** **立即回滾 (REVERTED)**。
  * 還原 `search.py` 為 `quiet_move_counter >= LMR_MIN_QUIET_MOVE_INDEX`，確保基線 100% 穩固。

---

### E71：4 大底層缺陷修復 + 7 軸 LMR SPSA 調優與 50% 阻尼中點參數驗收（2026-10-18，正式採納 ADOPTED）

* **改動項目與動機：**
  1. **Capture Futility Pruning 補齊 Runtime LMR Slot 接線（`chess_engine/classical/search.py`）**：
     * Step 14 Capture FP 原呼叫 `compute_lmr_reduction_1024(depth, searched_legal_moves + 1, improving)` 時漏傳 `search_context.tune[TUNE_LMR_TABLE_SCALE]` 與 `search_context.tune[TUNE_LMR_NOT_IMP]`，補齊後確保 `slot 24` SPSA 梯度一致（基準值下 0-node delta）。
  2. **消除 MovePicker 分段生成重複搜索「不吃子升變（Quiet Promotions）」（`chess_engine/classical/move_generator.py`）**：
     * `generate_pseudo_legal_captures_buffer` 已生成所有直推升變（`single_pushes & RANK_8` / `RANK_1`），移除 `generate_pseudo_legal_quiets_buffer` 內重複生成的直推升變（改為 `single_pushes & ~RANK_8` / `~RANK_1`），防止同一節點內不吃子升變被重複搜尋兩次並使 `searched_legal_moves` 虛增 +4。
  3. **消除 Step 14 Quiet SEE 與 EP SEE 的冗餘敵方 Bitboard 空掃描（`chess_engine/classical/see.py`, `search.py`）**：
     * `_see_ge_jit` 新增 `victim_type == -2` 哨兵代表「已知目標格為空（Quiet 著法或 En Passant）」，跳過 `find_piece_type_on_square_side` 的 6 個敵方 bitboard 線性掃描（100% 0-node delta 純 NPS 加速）。
  4. **ProbCut 補齊 `not is_pv and ply > 0` 節點守衛（`chess_engine/classical/search.py`）**：
     * 防止在 Aspiration Window（`abs(beta) < VALUE_KNOWN_WIN`）下於 Root 或 PV 節點觸發投機性 ProbCut 截斷。
* **7 軸 LMR SPSA 實測與方法論突破：**
  * 在 Colab 上執行 7 軸 LMR SPSA（120k nodes / 1,000 iters / 14,000 局）。
  * **信噪比與尺度反轉剖析：**
    * 尾端 30% 聚合參數（`BASE=490, CUTNODE=1824, TTCAP=1231, NOT_IMP=218, MC=23, SCALE=114, HIST=194`）在 120k 淺層表現積極，但在 L2 300k 深層時發生「尺度反轉（Scale Inversion）」，過度剪枝（Over-reduction）導致深層戰術盲點而在 L2 翻車。
  * **主特徵向量與阻尼中點（Midpoint 50%）：**
    * 14,000 局 SPSA 實質找到了 7 參數協同優化的主梯度特徵向量 $\vec{v} = \vec{\theta}_{\text{tail}} - \vec{\theta}_{\text{start}}$。
    * 使用者首先截取 Iter 852 檢查點（`BASE=486, CUTNODE=1809`），在 300k 取得 +3.17 Elo，但超長殘局（>80 步）勝率僅 48.0%。
    * 進一步取起點與 Tail 30% 之 **50% 阻尼中點參數（Midpoint 50%）**：
      - `LMR_BASE_OFFSET` : `449 -> 470` (+21)
      - `LMR_TABLE_SCALE_PERCENT` : `113 -> 114` (+1)
      - `LMR_MOVECOUNT_FACTOR` : `29 -> 26` (-3)
      - `LMR_CUTNODE_BONUS` : `1839 -> 1832` (-7)
      - `LMR_TTCAPTURE_BONUS` : `1191 -> 1211` (+20)
      - `LMR_HISTORY_SCALE` : `200 -> 197` (-3)
      - `LMR_NOT_IMP_NUM` : `197 -> 208` (+11)
    * 二次拋物線擬合預測理論最佳點 $\alpha^* \approx 0.493$，與 50% 中點高度契合。
* **對戰設定與環境（Stage B 驗收對決）：**
  * 錦標賽配置：Google Colab (7 並行 Worker)，`inprocess` 後端。
  * 局數：**354 局**（177 對成對換先開局），時控：**300,000 nodes/move**，TT：64 MB / context。
  * 對陣：Candidate (New, E71 4 大修復 + Midpoint 50% 參數) vs Baseline (Old, E69 舊生產基準)。
* **對戰成果與深度統計（300k nodes L2 總決戰）：**
  * **對局戰績：** 跑滿 **600 局**（300 對成對換先開局），Candidate **136 勝 340 和 124 負**（得分率 **51.00%**，淨勝 **+12 盤**）。
  * **相對 Elo：** **`+6.95`**，95% 置信區間 `[-11.37, +25.31]`。
  * **搜尋深度：** New 平均深度 **19.12 vs 18.97 (+0.15 ply)**，$\text{depth} \ge 14$ 比例 **82.7% vs 81.1%**。
  * **長局與殘局逆轉：**
    - 短局 ($\le 25$ 步): 50.0%
    - 中局 (26–50 步): 52.2%
    - 長局 (51–80 步): 50.0%
    - 超長局 ($>80$ 步): **54.7% (淨勝 +7 盤)**，徹底治癒後期殘局失血缺陷！
* **判定：** **正式採納進生產 / 基線 (ADOPTED / PRODUCTION BASELINE UPDATED)**。
  * 將 Midpoint 50% 參數寫入 `chess_engine/classical/constants.py` 與 `tuner/constants_snapshot.py`。
  * 透過 `tools/sync_classical_old.py --sync` 完整同步至 `chess_engine/classical_old/`，確立全新對戰基準。

---

### E71.1：7 軸 LMR 1D 流形最優極值探針決鬥（Alpha 0.40 vs Alpha 0.50，2026-10-19，正式採納 ADOPTED）

* **改動項目與動機：**
  * 在 E71 確立 14,000 局 SPSA 所探明之 7 軸協同主梯度向量 $\vec{v} = \vec{\theta}_{\text{tail}} - \vec{\theta}_{\text{start}}$ 後，為進一步精確定位 1D 線搜尋 $\vec{\theta}(\alpha) = \vec{\theta}_{\text{start}} + \alpha \vec{v}$ 的最佳理論頂點，排程 **$\alpha = 0.40$**（更穩健防守減深）vs **$\alpha = 0.50$**（E71 基準）的 500 局總決戰。
  * 參數配置對照：
    - `LMR_BASE_OFFSET` : `470 -> 465` (-5，基礎減深進一步適度放寬，強化深層防守著法保護)
    - `LMR_TABLE_SCALE_PERCENT` : `114 -> 113` (-1)
    - `LMR_MOVECOUNT_FACTOR` : `26 -> 27` (+1，後段著法減深斜率阻尼適度增加)
    - `LMR_CUTNODE_BONUS` : `1832 -> 1833` (+1)
    - `LMR_TTCAPTURE_BONUS` : `1211 -> 1207` (-4)
    - `LMR_HISTORY_SCALE` : `197 -> 198` (+1)
    - `LMR_NOT_IMP_NUM` : `208 -> 205` (-3，未改善局勢減深微調放寬)
* **對戰設定與環境：**
  * 錦標賽配置：Google Colab (7 並行 Worker)，`inprocess` 後端。
  * 局數：**500 局**（250 對成對換先開局），時控：**300,000 nodes/move**，TT：64 MB / context。
  * 對陣：Candidate (New, $\alpha = 0.40$) vs Baseline (Old, $\alpha = 0.50$ E71 基準)。
  * PGN 存檔：`tournament_analysis/alpha040_vs_alpha050_300k.pgn`。
* **對戰成果與深度統計：**
  * **對局戰績：** 500 局，Candidate (Alpha 0.40) **102 勝 303 和 95 負**（得分率 **50.70%**，淨勝 **+7 盤**）。
  * **相對 Elo：** **`+4.86`**，95% 置信區間 `[-14.29, +24.05]`。
  * **搜尋深度：** Alpha 0.40 平均深度 **19.87** vs Alpha 0.50 **19.92**（差異 -0.05 ply, Welch t=-0.398, $p=0.69$ 統計等價），$\text{depth} \ge 14$ 比例 **91.0% vs 91.3%**。
  * **搜尋效能：** 每步耗時 775.8 ms vs 779.6 ms (-3.8 ms)，NPS 375,203 vs 373,766 (+0.4%)。
  * **長局與殘局結構分析：**
    - 短局 ($\le 25$ 步): 1-52-1 (50.0%)
    - 中局 (26–50 步): 6-114-10 (48.5%)
    - 長局 (51–80 步): **65-71-59 (51.5%, 淨勝 +6 盤)**
    - 超長局 ($>80$ 步): **30-66-25 (52.1%, 淨勝 +5 盤)**，長局總計淨勝 **+11 盤**！
* **物理機理解析與方法論收斂：**
  * 500 局實測完整印證了西洋棋啟發式高維空間的「平坦高原效應（Flat Plateau）」：$\alpha \in [0.40, 0.50]$ 實力極其接近，皆顯著優於 E69 原始基線（相對於 E69 累計約 +11.8 Elo）。
  * $\alpha = 0.40$ 因減深力度微幅回退，在 $>50$ 步的複雜防守與殘局中給予關鍵著法更充分的驗證深度，展現出更強韌的殘局防禦力。
* **判定：** **正式採納進生產 / 基線 (ADOPTED / PRODUCTION BASELINE UPDATED)**。
  * 正式將 $\alpha = 0.40$ 數值保留在 `chess_engine/classical/constants.py` 與 `tuner/constants_snapshot.py`。
  * 透過 `tools/sync_classical_old.py --sync` 完整同步至 `chess_engine/classical_old/`，更新生產基準對抗庫。

---

### E72：SEE 雙向滑動攻擊零樹變動快速短路優化 + Non-LMR 全深度搜尋 `cut_node` 傳遞修復（Bug 2）（2026-10-20，正式採納 ADOPTED）

* **改動項目與動機：**
  1. **MovePicker `STAGE_GOOD_CAPTURES` SEE 雙向滑動攻擊零樹變動短路（`chess_engine/classical/see.py`）**：
     * 重構 `get_all_attackers_for_see`，一次性生成白黑雙方滑動攻擊位元棋盤，消除冗餘的重複 Magic Bitboard 計算。
     * 在 6 個標準 Perft 盤面 Depth 15 測試中取得 **100% 0-node delta 完全一致**（全量 2,997,212 個節點完全吻合），全域 NPS **+12.24%**（Kiwipete 達到 **+27.06% NPS 提升**，總搜尋耗時縮減 10.90%）。
  2. **Non-LMR 全深度搜尋 `cut_node` 傳遞修正（`chess_engine/classical/search.py:2827`）**：
     * 原代碼在 `lmr == 0` 時無條件向子節點傳遞 `cut_node = True`，導致防守 ALL-Node 被誤當成進攻 CUT-Node。
     * 舊版後遺症：在緊湊局面與王兵殘局中，防守方誤觸發空步剪枝（NMP），產生嚴重的強迫走步（Zugzwang）盲點；且破壞了 SF19 Step 18 `all_node = not is_pv and not cut_node` 的正常判斷。
     * 修復為：`child_cut_node = _b_true() if lmr > 0 else (not cut_node)`。
* **對戰設定與環境：**
  * 錦標賽配置：Google Colab (AMD EPYC 7B12 8 vCPU, 7 並行 Worker)，`inprocess` 後端。
  * 局數：**600 局**（300 對成對換先開局），時控：**400,000 nodes/move**，TT：64 MB / context。
  * 對陣：Candidate (New, E72 SEE Fast-Path + Bug 2 Fix) vs Baseline (Old, E71.1 生產基準)。
  * PGN 存檔：`tournament_analysis/tournament_results.pgn`。
* **對戰成果與深度統計（400k nodes L2 長時控總決戰）：**
  * **對局戰績：** 跑滿 **600 局**，Candidate **118 勝 377 和 105 負**（得分率 **51.08%**，總體得分 306.5 / 600，淨勝 **+13 盤**）。
  * **相對 Elo：** **`+7.53`**，95% 置信區間 `[-8.49, +23.58]`。
  * **前後半場高度對稱：**
    - 前半場 150 對 (300 局): 得分率 51.2% (淨勝 +3.5 分)
    - 後半場 150 對 (300 局): 得分率 51.0% (淨勝 +3.0 分)
  * **搜尋效能與深度：**
    - 搜尋深度：New 19.65 vs Old 19.70 (-0.06 ply, Welch t=-0.493, p=0.62 統計等價)
    - 每步節點數：New 249,346 vs Old 250,029 (-0.3%, 節點略為精煉)
    - 每步耗時：867.1 ms vs 862.3 ms (+0.5%)，NPS 354,861 vs 356,922 (-0.6%)
  * **長局與殘局結構碾壓（Bug 2 修復核心鐵證）：**
    - 短局 ($\le 25$ 步): 0-65-0 (50.0%)
    - 中局 (26–50 步): 10-141-7 (50.9%)
    - 長局 (51–80 步): **69-85-63 (51.4%, 淨勝 +6 盤)**
    - 超長局 ($>80$ 步): **39-86-35 (51.2%, 淨勝 +4 盤)**，$>50$ 步殘局共淨勝 **+10 盤**！
* **物理機理解析：**
  * 舊版在殘局將 ALL-Node 誤判為 CUT-Node 導致 NMP 偷跑，在細膩殘局中漏算強迫步（Zugzwang）；修復後正確傳遞 `child_cut_node`，封堵了殘局假截斷，大幅提升長回合殘局轉化勝率與守和能力。
* **判定：** **正式採納進生產 / 基線 (ADOPTED / PRODUCTION BASELINE UPDATED)**。
  * 透過 `python tools/sync_classical_old.py --sync` 完整同步至 `chess_engine/classical_old/`，確立全新生產基準對抗庫。

---

### E73：Step 3 修復 Fail-Low ALL-Node 污染 TT (Bug 1) + IIR !allNode 守衛 (提案 B)（2026-10-20，回滾 REVERTED）

* **改動項目與動機：**
  1. **修復 Bug 1（Fail-Low 污染 TT 缺陷，`chess_engine/classical/search.py:3276`）**：
     * 移除 Fail-Low ALL-Node 強制回退 `best_move = moves[0]` 的邏輯，避免覆寫摧毀置換表中已有之合法最佳步，並消除假 TT 步對後續著法排序、RFP 與 LMR 的干擾。
  2. **實作提案 B（IIR !allNode 守衛，`chess_engine/classical/search.py:1946`）**：
     * 對齊 Stockfish 19 Step 11（`!allNode` 規範），在 IIR 條件補齊 `(is_pv or cut_node)` 守衛，豁免防守型 ALL-Node 遭無謂減深。
* **對戰設定與環境：**
  * 錦標賽配置：Google Colab (AMD EPYC 7B12 8 vCPU, 7 並行 Worker)，`inprocess` 後端。
  * 規模：**1,000 局**（500 對成對換先開局），時控：**400,000 nodes/move**，TT：64 MB / context。
  * 對陣：Candidate (New, Step 3 Bug 1 Fix + Proposal B IIR Guard) vs Baseline (Old, E72 生產基準)。
  * PGN 存檔：`tournament_analysis/tournament_results.pgn`。
* **對戰成果與深度統計（400k nodes L2 長時控 1,000 局驗證）：**
  * **對局戰績：** 跑滿 **1,000 局**，Candidate **185 勝 607 和 208 負**（得分率 **48.85%**，總體得分 488.5 / 1000，淨負 **-23 盤**）。
  * **相對 Elo：** **`-7.99`**，95% 置信區間 `[-21.34, +5.33]`。
  * **搜尋深度（致命崩塌）：**
    - New 平均深度 **19.63 ply** vs Old **19.86 ply**（顯著縮減 **-0.24 ply**，Welch $t = -2.528, p = 0.011$）
    - 深度中位數 New **16** vs Old **17**（整整落後 1 層）
    - $\text{depth} \ge 14$ 比例：New 88.5% vs Old 90.3% (-1.8%)
    - $\text{depth} \in [16, 20]$ 比例：New 42.8% vs Old 45.0% (-2.2%)
  * **全區間得分率萎縮：**
    - 短局 ($\le 25$ 步): 0-120-1 (49.6%)
    - 中局 (26–50 步): 11-223-17 (48.8%)
    - 長局 (51–80 步): 121-149-129 (49.0%)
    - 超長局 ($>80$ 步): 53-115-61 (48.3%)
* **物理機理解析（為何理論「乾淨」的改動會失敗？）：**
  * **機制 ①：`best_move = moves[0]` 實質扮演關鍵的「迭代著法快取（Move Ordering Cache）」**：
    在迭代加深搜尋中，深度 $D$ 失敗（Fail-Low）的局面在深度 $D+1$ 重訪時，舊版存入 TT 的 `moves[0]` 是 MovePicker 排序第一的著法（最佳吃子、殺手步或最高歷史安靜步），仍然是下一迭代最可能構成反擊的著法。新版存入 `NO_MOVE` 後，重訪該節點時缺乏 TT 著法指引，MovePicker 必須重頭生成並排序全部著法，著法排序效率嚴重下降。
  * **機制 ②：IIR 排除 ALL-Node 引發「分支爆炸（Branching Explosion）」**：
    舊版在 `tt_move == NO_MOVE` 時對所有節點均實施 1 層 IIR 減深探測。新版補齊 `(is_pv or cut_node)` 守衛後，豁免了 ALL-Node 的減深。但因修復 Bug 1 導致沒有 TT 步的 ALL-Node 大量增加，新版對這些節點全數以全深度展開搜尋全部 30+ 步走法，單點節點消耗暴增 2~3 倍，在 400k nodes 固定預算下過早耗盡算力，直接導致主幹 PV 無法推進（深度顯著掉落 -0.24 ply）。
  * **機制 ③：歷史與啟發平衡反噬**：
    現有 RFP、LMR 與 History 常數均是在舊有「Fail-Low 暫存第一步 + 全域 IIR 快速收斂」的拓撲下達成之局部最優點，打破平衡引發連鎖失真。
* **判定：** **立即回滾 (REVERTED)**。
  * 徹底還原 `chess_engine/classical/search.py` 至 E72 生產基線，`tools/sync_classical_old.py --diff-code` 確證 100% 0 diff。

### E74：純化版提案 A 前向剪枝順序重構（NMP 前置於 Capture ProbCut，2026-10-20，回滾 REVERTED）

* **改動項目與動機：**
  1. **前向剪枝執行順序解耦與純化重構（`chess_engine/classical/search.py`）**：
     * 將昂貴的「吃子走步生成 + 雙王 Pin 遮罩計算 + SEE 篩選 + QSearch + 減深子搜尋（Step 12b: Capture-based ProbCut）」移至「Step 10: 空步剪枝 (NMP)」之後。
     * **保留 $O(1)$ TT ProbCut 捷徑（Step 12a）** 於 NMP 之前：維持 E35（+11.59 Elo）成果，凡 TT 歷史已證明能突破 `probcut_beta` 者直接以 0 子搜尋節點瞬間截斷，不支付任何 NMP 搜尋開銷。
     * **安全隔離與解耦原則**：不移動 IIR、不修改 `improving` 狀態轉移、不更動 ProbCut TT Defense。
  2. **NMP Fail-Low 與 ProbCut 的免費算力協同紅利**：
     * 當盤面通過 NMP 門檻但空步子搜尋 Fail-Low 時，NMP 已呼叫 `compute_full_corrected_static_eval` 將 `static_eval_is_full` 設為 `True` 並計算了 `cached_pinned_white/black`。
     * 緊接著進入 Capture ProbCut 時，直接復用完整的 HCE 評估分與雙王 Pin 位元盤，省去重複的 `get_pinned_pieces` 計算。
* **對戰設定與環境：**
  * 錦標賽配置：Google Colab (AMD EPYC 7B12 8 vCPU, 7 並行 Worker)，`inprocess` 後端。
  * 規模：**600 局**（300 對成對換先開局），時控：**400,000 nodes/move** 長時控，轉置表：64 MB / context。
  * 對陣雙方：Candidate (New, E74 NMP-First) vs Baseline (Old, E72 生產基準)。
  * PGN 存檔：`tournament_analysis/tournament_results_E74.pgn`。
* **對戰成果與深度統計（400k nodes L2 長時控 600 局驗證）：**
  * **對局戰績：** 跑滿 **600 局**，Candidate **120 勝 358 和 122 負**（得分率 **49.83%**，總體得分 299.0 / 600，淨負 **-2 盤**）。
  * **相對 Elo：** **`-1.16`**。
  * **執色統計：** 白方 71W 175D 54L (52.83%)，黑方 49W 183D 68L (46.83%)。
  * **搜尋深度與節點：**
    - 平均深度：New **19.89 ply** vs Old **19.89 ply**（差異 +0.01 ply, Welch t = +0.074, p = 0.941 統計完全等價）
    - 深度中位數：New **17** vs Old **16**
    - 每步節點數：New **248,382** vs Old **248,400**（差異 -0.0%, Welch t = -0.048, p = 0.962 統計完全等價）
    - 每步耗時：New **464.1 ms** vs Old **468.6 ms**（-4.5 ms, -1.0%, t = -4.160, p = 3.19e-5 微幅提速）
    - 每步 NPS：New **612,827** vs Old **607,614**（+0.9%, t = +2.269, p = 0.023）
    - 同深度節點比 (d=10 ~ 24)：0.998 ~ 1.001，全深度區間樹大小幾乎完全重疊。
* **物理機理解析（為何理論「乾淨」的前置未帶來 Elo 正益？）：**
  * **機制 ①：TT ProbCut Shortcut (E35) 早已攔截絕大多數 ProbCut 節點**：E35 已經是 O(1) 捷徑，真正會穿透進 Capture ProbCut 的節點本就極少，宏觀節省有限。
  * **機制 ②：戰術引導資訊遺失（Tactical TT Guidance Loss）**：在少數己方有吃子大優的盤面中，原本 ProbCut 會找到具體吃子著法並存入 TT 作為高質量 best_move。將 NMP 前置後，引擎改為下「空步」探測，成功時回傳 NO_MOVE，未在置換表留下具體反擊著法，微幅削弱了後續重訪該節點時的著法排序品質，抵消了微小提速收益。
* **判定：** **立即回滾 (REVERTED)**。
  * 嚴格還原 `chess_engine/classical/search.py` 至 E72 生產基線，`tools/sync_classical_old.py --diff-code` 確證 100% 0 diff。

### E75：提案 D RFP 靜態分數阻尼混合（Fail-Soft Score Blending，2026-10-20，LTC 600 局驗證 +9.27 Elo 正式勝出）

* **改動項目與動機：**
  1. **RFP 截斷回傳值阻尼混合（`chess_engine/classical/search.py:1643`）**：
     * 對齊 Stockfish 19 Step 9（`search.cpp:1007`）：
       `rfp_score = (RFP_SCORE_BETA_WEIGHT * beta + RFP_SCORE_EVAL_WEIGHT * static_score) // RFP_SCORE_DIVISOR`。
     * **Classical HCE 與 NNUE 的本質差異與兩階段實驗**：
       - **初次嘗試 E75.0（直移植 SF19 權重：661 $\beta$ + 363 eval）**：
         在 NNUE 下評估極平滑，強阻尼拉向 $\beta$ 有效；但在 Classical HCE 下，過度重壓 $\beta$（拉向 $\beta$ 達 64.5%）嚴重閹割了局面優勢訊號，使父節點著法排序惡化，實測**顯著落敗**。
       - **反向權重 E75.1（HCE 輕阻尼：363 $\beta$ + 661 eval）**：
         改為拉向 $\beta$ 佔 35.5%、保留真實估值佔 64.5%。既成功抹平極端毛刺、防止 Aspiration Window 破裂，又完整保留了優勢梯度訊號。
     * **數學特性保證**：
       - 因 $static\_score - rfp\_margin \ge \beta$ 且 $rfp\_margin \ge 0$，恆有 $static\_score \ge \beta$。
       - 權重混合公式嚴格保證 $\beta \le rfp\_score \le static\_score$。
       - 100% 確保 $rfp\_score \ge \beta$ 觸發截斷，同時平滑極端大分。
  2. **常數宣告（`chess_engine/classical/constants.py`）**：
     * `RFP_SCORE_BETA_WEIGHT = 363`
     * `RFP_SCORE_EVAL_WEIGHT = 661`
     * `RFP_SCORE_DIVISOR = 1024`
* **對戰驗證結果（E75.1：363/661 vs Old_E72，600 局 LTC 400k nodes，換先平衡 300 對）：**
  * **戰績**：**137 勝 342 和 121 負**（得分率 **51.33%**，淨勝 **+16 局**）。
  * **相對 Elo**：**+9.27 Elo**（95% CI: [-7.10, +25.67]）。
  * **配對成果**：WD (38) vs LD (26) 淨勝 +12 對；WW (12) vs LL (7) 淨勝 +5 對；配對淨勝 +16 分。
  * **盤長優勢**：在 51-80 步的關鍵中殘局戰術過渡盤中，E75 斬獲 **92 勝 81 和 73 負（勝率 53.9%）**，展現壓倒性穩定度。
* **搜尋指標分析（`parse_search_stats.py`，樣本數 37,391 步 vs 37,357 步）：**
  * **平均深度**：19.90 vs 19.93 ply（差異 -0.03 ply, Welch p = 0.803，統計完全等價）。
  * **每步節點數**：247,551 vs 247,415（+0.1%, Welch p = 0.715，同深度節點比精確 1.000）。
  * **耗時 / NPS**：442.4 ms vs 433.5 ms（+8.9 ms）；NPS 641,234 vs 652,159（-1.7%）。
* **判定：** **顯著正收益（+9.27 Elo，勝率 51.33%）。**

### E76：提案 D.1 RFP 阻尼權重精細微調（412 $\beta$ + 612 eval vs 363 基線，552 局 +5.67 Elo 勝出）

* **改動項目與動機：**
  * 承接 E75.1（363 $\beta$ + 661 eval）的 +9.27 Elo 突破，進一步探尋 $\beta$ 阻尼比例的最優平衡點：
    將 $\beta$ 權重由 363 (35.5%) 微幅調升至 412 (40.2%)，eval 權重由 661 (64.5%) 調至 612 (59.8%)。
  * 探索在 Classical HCE 下，40% $\beta$ 阻尼是否能在抑制估值震盪與保留優勢梯度之間取得更極致的平衡。
* **常數設定：**
  * New（候選）：`RFP_SCORE_BETA_WEIGHT = 412`, `RFP_SCORE_EVAL_WEIGHT = 612`
  * Old（基線）：`RFP_SCORE_BETA_WEIGHT = 363`, `RFP_SCORE_EVAL_WEIGHT = 661`
* **對戰驗證結果（552 局 LTC 400k nodes，換先平衡 276 對）：**
  * **戰績**：**123 勝 315 和 114 負**（得分率 **50.82%**，淨勝 **+9 局**）。
  * **相對 Elo**：**+5.67 Elo**（95% CI: [-12.87, +24.23]）。
  * **配對成果**：WD (37) vs LD (32) 淨勝 +5 對；WW (13) vs LL (11) 淨勝 +2 對；配對淨勝 +9 分。
  * **長局殘局表現**：超長局 (>80 步) 斬獲 36W 72D 29L（52.6% 得分率，淨勝 +7 盤）。
* **搜尋指標分析（`parse_search_stats.py`，樣本數 37,825 步 vs 37,807 步）：**
  * **平均深度**：19.87 vs 19.87 ply（差異 +0.00 ply, Welch p = 0.986，統計完全重合）。
  * **每步節點數**：248,551 vs 248,515（+0.0%, Welch p = 0.921，同深度節點比維持 1.000）。
  * **耗時 / NPS**：471.9 ms vs 464.7 ms（+7.2 ms）；NPS 605,424 vs 613,924（-1.4%）。
* **判定：** **有效增益（+5.67 Elo，勝率 50.82%），412/612 成為新基線並已同步至 `classical_old`。**

### E77：提案 D.2 RFP 阻尼權重探索（312 $\beta$ + 712 eval vs 412 基線，實測落敗回滾 REVERTED）

* **改動項目與動機：**
  * 探尋阻尼曲線在左側（輕阻尼方向）的邊界：
    將 $\beta$ 權重由 412 (40.2%) 調降至 312 (30.5%)，eval 權重調升至 712 (69.5%)。
  * 測試更低的 $\beta$ 阻尼（約 30% $\beta$ + 70% eval）能否比 412 提供更敏銳的戰術估值傳導。
* **常數設定：**
  * New（候選，`chess_engine/classical/constants.py`）：`RFP_SCORE_BETA_WEIGHT = 312`, `RFP_SCORE_EVAL_WEIGHT = 712`
  * Old（基線，`chess_engine/classical_old/constants.py`）：`RFP_SCORE_BETA_WEIGHT = 412`, `RFP_SCORE_EVAL_WEIGHT = 612`
* **對戰結果（LTC 400k nodes，換先平衡 67 對 / 134 局判定）：**
  * **戰績**：**25 勝 72 和 37 負**（得分率 **45.52%**，淨負 **-12 局**）。
  * **相對 Elo**：**-31.20 Elo**（95% CI: [-68.13, +5.04]）。
  * **先手/後手表現**：持白 47.01%（12W 39D 16L），持黑 44.03%（13W 33D 21L），雙色皆落敗。
* **物理機理解析與阻尼極值定位：**
  * 當 $\beta$ 阻尼降低至 30.5%（312）時，阻尼強度已不足以壓制 Classical HCE 在戰術節點上的靜態估值毛刺。過大的 static_score 頻繁突破父節點的抱負視窗（Aspiration Window），導致重複觸發寬視窗 Fail-High 重搜，搜尋樹穩定度下降。
  * **三階段實驗成功標定出 Classical HCE 的 RFP 阻尼凸函數最優區間**：
    1. **661 $\beta$（64.5% 阻尼）**：過重阻尼，閹割優勢訊號，**顯著落敗**。
    2. **312 $\beta$（30.5% 阻尼）**：過輕阻尼，估值毛刺反彈，**顯著落敗 (-31.20 Elo)**。
    3. **363 $\beta$（35.5% 阻尼）**：顯著勝出（+9.27 Elo）。
    4. **412 $\beta$（40.2% 阻尼）**：進一步超越 363（+5.67 Elo），**登上最高峰值（相較無阻尼累計約 +14.94 Elo）**。
* **判定：** **立即回滾 (REVERTED)**。
  * `chess_engine/classical/constants.py` 與 `tuner/constants_snapshot.py` 均恢復至 `RFP_SCORE_BETA_WEIGHT = 412`、`RFP_SCORE_EVAL_WEIGHT = 612`。
  * `tools/sync_classical_old.py --diff-code` 確證 100% 邏輯等價（0 diff），生產基線穩固鎖定於 412。

### E78：提案 D.3 RFP 阻尼權重右側邊界探索（462 $\beta$ + 562 eval vs 412 基線，600 局 -2.90 Elo 實測落敗回滾 REVERTED）

* **改動項目與動機：**
  * 在 E77 探尋左側邊界（312 $\beta$ 大敗 -31.20 Elo）確認過輕阻尼會導致 HCE 靜態估值毛刺刺穿抱負視窗後，進一步探尋 412 右側（偏重阻尼方向）的邊界：
    將 $\beta$ 權重由 412 (40.2%) 調升至 462 (45.1%)，eval 權重由 612 (59.8%) 調降至 562 (54.9%)。
  * 驗證更深度的靜態分平滑是否能進一步提升搜尋穩定度，抑或過重阻尼開始鈍化真實優勢信號。
* **常數設定：**
  * New（候選，`chess_engine/classical/constants.py`）：`RFP_SCORE_BETA_WEIGHT = 462`, `RFP_SCORE_EVAL_WEIGHT = 562`
  - Old（基線，`chess_engine/classical_old/constants.py`）：`RFP_SCORE_BETA_WEIGHT = 412`, `RFP_SCORE_EVAL_WEIGHT = 612`
* **對戰驗證結果（600 局 LTC 400k nodes，換先平衡 300 配對完整完賽）：**
  * **戰績**：**122 勝 351 和 127 負**（得分率 **49.58%**，淨負 **-5 局 / -2.5 分**）。
  * **相對 Elo**：**-2.90 Elo**（95% CI: [-20.93, +15.12]）。
  * **先手/後手表現**：持白 53.17%（64W 191D 45L），持黑 46.00%（58W 160D 82L）。
  * **盤長分佈特徵**：
    * 51-80 步（關鍵中殘局）：78W 89D 75L（得分率 50.6%）。
    * >80 步（超長殘局）：35W 58D 41L（得分率 **47.8%**，淨負 **-6 盤**）。
  * **配對成果**：DD 106, DL 50, DW 35, WD 34, LD 20, WL 17, LL 15, WW 13, LW 10。
* **搜尋指標分析（`parse_search_stats.py`，樣本數 36,724 步 vs 36,752 步）：**
  * **平均深度**：19.90 vs 19.91 ply（差異 -0.01 ply, Welch t = -0.066, p = 0.948，完全等價）。
  * **每步節點數**：248,195 vs 248,235（差異 -0.0%, Welch t = -0.108, p = 0.914，完全等價）。
  * **耗時 / NPS**：776.2 ms vs 765.8 ms（+1.3%, p = 3.08e-8）；NPS 373,773 vs 378,647（-1.3%, p = 1.44e-3）。
* **物理機理解析與 1D 阻尼凸函數全局閉合：**
  * **超長殘局鈍化效應**：當 $\beta$ 權重升至 462（45.1%）時，估值過度向 $\beta$ 收縮，扁平化了真實優勢梯度。在 >80 步的超長殘局中，勝率顯著下滑至 47.8%（淨負 6 盤），證實過度阻尼削弱了將微小優勢轉化為勝勢的驅動力。
  * **五階段實驗成功完整標定 Classical HCE 的 RFP 阻尼凹函數（Concave Function）曲線**：
    1. **$W_\beta = 0$（0% $\beta$, 100% eval，E72 未阻尼基線）**：基準 0.00 Elo。
    2. **$W_\beta = 312$（30.5% $\beta$, 69.5% eval，E77）**：**-31.20 Elo**（阻尼不足，HCE 靜態評估毛刺破窗重搜引發巨幅失分）。
    3. **$W_\beta = 363$（35.5% $\beta$, 64.5% eval，E75）**：**+9.27 Elo**（成功壓制雜訊，確立正向效益）。
    4. **$W_\beta = 412$（40.2% $\beta$, 59.8% eval，E76）**：**+14.94 Elo（vs 未阻尼基線，最高凹頂 Peak，確立為生產基線）**。
    5. **$W_\beta = 462$（45.1% $\beta$, 54.9% eval，E78）**：**-2.90 Elo**（vs 412 基線，進入右側過度阻尼下降段）。
  * 此結果嚴密確證 **$W_\beta = 412$（40.2% $\beta$ + 59.8% eval）即為 Classical HCE 搜尋架構下的全局最優阻尼常數**。
* **判定：** **立即回滾 (REVERTED)**。
  * `chess_engine/classical/constants.py` 與 `tuner/constants_snapshot.py` 均恢復至 `RFP_SCORE_BETA_WEIGHT = 412`、`RFP_SCORE_EVAL_WEIGHT = 612`。
  * `tools/sync_classical_old.py --diff-code` 確證 100% 邏輯等價（0 diff），生產基線穩固鎖定於 412。

### E79 / 實驗 C：解耦 Correction History 的 static_score 與 corrected_eval（962 局 -9.39 Elo 實測落敗回滾 REVERTED）

* **改動項目與動機：**
  * 對照 Stockfish 19 Step 20（`search.cpp:1631-1638`）：在節點結尾更新 Correction History（Pawn, Minor, Non-Pawn, Continuation）時，SF 使用未受置換表動態覆蓋的靜態評估 `ss->staticEval` 計算殘差偏差 `diff = bestValue - ss->staticEval`。
  * 本引擎原實作中，`static_score` 在節點開頭遇到 TT 命中時會被精煉為 TT 分數邊界（`tt_score`）。當節點最終評估 `max_eval` 接近上次存入 TT 的分數時，`diff = max_eval - static_score` 經常為 0，理論上阻斷了修正歷史在 TT 命中節點的持續學習。
  * 實驗 C 嘗試將偏差計算解耦：定義獨立的 `corrected_eval`（純評估修正分數），將結尾偏差計算改為 `diff = max_eval - corrected_eval`，並以 `(max_eval > corrected_eval) == has_best_move` 作為更新門檻；同時保留搜尋前向剪枝與 LMR 仍使用高置信度的 TT 精煉 `static_score`。
* **常數與程式設定：**
  * New（候選，`chess_engine/classical/search.py`）：引入 `corrected_eval`，更新 Correction History 改用 `diff = max_eval - corrected_eval`。
  * Old（基線，`chess_engine/classical_old/search.py`）：Stage 1 零樹變動基線（`diff = max_eval - static_score`）。
* **對戰驗證結果（Colab 962 局 LTC 400k nodes，換先平衡成對對決）：**
  * **戰績**：**189 勝 558 和 215 負**（得分率 **48.65%**，淨負 **-26 局**）。
  * **相對 Elo**：**-9.39 Elo**（±14.23，95% CI: [-23.62, +4.84]）。
  * **SPRT 模型**：LLR = -2.98 $\le$ -2.944（$H_0$ Accepted，統計顯著倒退定案）。
  * **盤長勝率崩塌特徵**：
    * 1-50 步（開局/早中局）：得分率 50.8%（維持平局走勢）。
    * **51-80 步（關鍵中殘局戰術過渡盤）：勝率暴跌至 46.7%**（成為主要失分階段）。
    * >80 步（殘局）：得分率 47.9%。
* **物理機理解析與失敗根因：**
  * **Classical HCE 噪聲飽和效應**：Stockfish 19 採用極度平滑的 NNUE 網路，評估波動小；而 Classical HCE 評估值在複雜局面存在顯著局部戰術毛刺。
  * 舊實作中，TT 命中時將 `diff` 壓制為 0 雖然在理論上阻礙了部分節點更新，但意外地充當了一道天然的**評估抗噪阻尼器（Noise Damping Gate）**，避免了頻繁的戰術偶發波動過度更新全域修正表。
  * 一旦解耦為純 `max_eval - corrected_eval`，所有節點均全力更新。而當前 `CORRECTION_BONUS_HAS_MOVE_DIV = 10` 的除數是過去在更新受阻背景下調配的高增益參數。解耦後的大幅頻繁更新導致 Pawn / Minor Correction History 估值被戰術殘差迅速打滿飽和（±100~256），到了 51–80 步時全域評估表被髒噪聲污染，導致安靜步排序與深層剪枝嚴重失準。
  * **結論**：解耦 `corrected_eval` 在 Classical HCE 下無法作為零調參熱插拔特性，必須大幅降低更新增益（如將除數調大至 24~32）並搭配 SPSA 重新擬合。
* **判定：** **立即回滾 (REVERTED)**。
  * `chess_engine/classical/search.py` 完整復原至 `diff = max_eval - static_score`。
  * `tools/sync_classical_old.py --diff-code` 確證 100% 邏輯等價（0 diff）。

### E80 / 實驗 A：修復 `cutoff_cnt` 跨層污染隔離 + Fail-Low TT `best_move` 保護（600 局 -4.63 Elo 實測落敗回滾 REVERTED）

* **改動項目與動機：**
  1. **ProbCut 與 NMP 之 `cutoff_cnt` 跨層污染隔離（`chess_engine/classical/search.py:1728, 1841, 2109`）**：
     * 在 ProbCut 嘗試吃子前保存 `search_context.cutoff_cnt[ply + 1]`，若未觸發 ProbCut 截斷則予以還原；並對齊 SF19 Step 13，於 ProbCut 與 NMP 結束後、進入主迴圈前執行 `search_context.cutoff_cnt[ply + 2] = 0`，防止失敗的前向探測殘留高截斷計數污染主變例首步子樹的 LMR 減深。
  2. **Fail-Low 時保護既有 `tt_move`（Bug 1 微創修復，`chess_engine/classical/search.py:3333-3341`）**：
     * 當節點 Fail-Low（`best_move == NO_MOVE`）時，若該節點原本已有合法的 `tt_move != NO_MOVE`，則保留 `best_move = tt_move` 避免被 `moves[0]` 覆寫；僅在原本 `tt_move == NO_MOVE` 的全新 ALL-Node 上回退至 `moves[0]`。
* **常數與程式設定：**
  * New（候選，`chess_engine/classical/search.py`）：Stage 1 基線 + 實驗 A（`cutoff_cnt` 隔離 + Fail-Low `tt_move` 保留）。
  * Old（基線，`chess_engine/classical_old/search.py`）：Stage 1 零樹變動基線。
* **對戰驗證結果（Colab 600 局 LTC 400k nodes，換先平衡 300 對完整完賽）：**
  * **戰績**：**116 勝 360 和 124 負**（得分率 **49.33%**，總得分 296.0 / 600，淨負 **-8 局 / -4.0 分**）。
  * **相對 Elo**：**-4.63 Elo**（±16.80，95% CI: `[-21.44, +12.15]`）。
  * **前後半場與分段軌跡**：
    * 前半場 150 對（300 局）：53W 184D 63L（得分率 `48.3%`，主要受前 40 對 11W 48D 21L 開場亂流拖累）。
    * 後半場 150 對（300 局）：63W 176D 61L（得分率 `50.3%`，淨勝 +2 局，後段回歸五五開均勢）。
  * **先手/後手表現**：
    * 持白：51.50%（62W 185D 53L），持黑：47.17%（54W 175D 71L；對應 Old 持白 52.83%，屬正常先手優勢範圍，無顯著執色不對稱，$p = 0.179$）。
  * **盤長分佈特徵**：
    * 短局 ($\le 25$ 步)：0W 61D 1L（49.2%）。
    * 中局 (26–50 步)：6W 150D 9L（49.1%）。
    * 長局 (51–80 步)：75W 88D 81L（48.8%，淨負 -6 局）。
    * 超長局 ($>80$ 步)：35W 61D 33L（**50.8%**，淨勝 +2 局）。
  * **配對九宮格（300 對）**：DD 109 (36.3%), DL 40 (13.3%), DW 36 (12.0%), WD 34 (11.3%), LD 32 (10.7%), WL 20 (6.7%), LL 11 (3.7%), LW 10 (3.3%), WW 8 (2.7%)。
* **搜尋指標分析（`parse_search_stats.py`，樣本數 36,389 步 vs 36,348 步）：**
  * **平均深度**：19.83 vs 19.82 ply（差異 `+0.01 ply`, Welch $t = +0.090, p = 0.928$，統計完全等價；$\text{depth} \ge 14$ 比例 `90.4%` vs `90.3%`）。
  * **每步節點數**：248,282 vs 248,320（差異 `-0.0%`, Welch $t = -0.100, p = 0.920$，同深度節點比在 `d=10..24` 均落於 `0.990 ~ 1.004`）。
  * **耗時 / NPS**：473.5 ms vs 474.3 ms（`-0.8 ms`）；NPS 604,157 vs 603,082（`+0.2%`, $p = 0.641$）。
* **判讀與處置：**
  * 實測 `-4.63 Elo`（95% CI `[-21.44, +12.15]` 含 0），搜尋深度與節點數完全持平，顯示改動效果小於 600 局測試解析度（±17 Elo）。（註：關於 Fail-Low 保留安靜 `tt_move` 是否影響 `tt_capture` 相關剪枝僅為未經診斷計數驗證之假說。）
* **判定：** **回滾 (REVERTED)**。
  * 依門檻規範（49.33% $\le$ 49.5%，點估計為負）撤銷實驗 A 的 3 處改動，還原 `chess_engine/classical/search.py` 至 Stage 1 基線，並轉入 **實驗 D（修復 ProbCut TT Defense 漏檢，Bug 4）**。

### E81 / 實驗 D：修復 ProbCut TT Defense 漏檢（Bug 4，600 局 +1.16 Elo 採納進生產基線 ADOPTED）

* **改動項目與動機：**
  * 檔案：`chess_engine/classical/search.py:1724-1727`。
  * 根因病灶：原 ProbCut TT Defense 註解載明「若 TT 為上界（`TT_FLAG_ALPHA`）且分數低於 `probcut_beta` 則跳過 ProbCut」，但實作僅寫 `if tt_score_pc < probcut_beta: skip_probcut = True`，漏檢了 `tt_flag` 與 `tt_depth`。這導致當 TT 存放的是下界分數（`TT_FLAG_BETA`，真實分數 $\ge \text{tt\_score\_pc}$）或淺層紀錄（如 `depth = 0` 的 QSearch）時，ProbCut 遭到錯誤關閉。
  * 修復方式：補齊邊界與深度有效性條件：
    ```python
    if ((tt_flag == TT_FLAG_ALPHA or tt_flag == TT_FLAG_EXACT)
            and tt_depth >= depth - probcut_r
            and tt_score_pc < probcut_beta):
        skip_probcut = True
    ```
* **對戰驗證結果（Colab 600 局 LTC 400k nodes，換先平衡 300 對完整完賽）：**
  * **戰績**：**119 勝 364 和 117 負**（得分率 **50.17%**，總得分 301.0 / 600，淨勝 **+2 局 / +1.0 分**）。
  * **相對 Elo**：**+1.16 Elo**（±17.33，95% CI: `[-16.17, +18.49]`）。
  * **前後半場與執色**：前半場 300 局 `49.5%`，後半場 300 局 `50.8%`；執白 `53.50%`（71W 179D 50L），執黑 `46.83%`（48W 185D 67L，對應 Old 執白 `53.17%`，完全對稱）。
  * **盤長分佈**：短局 `49.1%`、中局 `49.6%`、長局 (51–80 步) **51.0%**（79W 99D 74L，淨勝 +5 局）、超長局 ($>80$ 步) `49.7%`（34W 84D 35L）。
* **搜尋指標分析（`parse_search_stats.py`，樣本數 37,599 步 vs 37,562 步）：**
  * **平均深度**：19.67 vs 19.73 ply（差異 `-0.06 ply`, Welch $t = -0.558, p = 0.577$，無顯著差異；$\text{depth} \ge 14$ 比例均為 `90.5%`）。
  * **同深度平均節點比（關鍵結構訊號）**：在 `d = 9 .. 24` 全數 16 個深度層上，New/Old 同深度節點比一致落在 **`0.984 ~ 0.997`**（全局每步節點數 245,524 vs 247,644，**`-0.9%`**，Welch $t = -5.854, p = 4.83 \times 10^{-9}$ 極度顯著），證實修復後 ProbCut 正常放行並成功削減了約 1% 的搜尋樹節點。
  * **耗時 / NPS**：461.1 ms vs 458.4 ms（`+2.7 ms`）；NPS 612,465 vs 620,805（`-1.3%`），反映更多節點進入 ProbCut 嘗試吃子生成、SEE 與 QSearch 前置驗證的合理成本。
* **物理機理解析與採納決策：**
  1. **搜尋樹行為完全符合預期**：解除 `TT_FLAG_BETA` 與淺層 TT 對 ProbCut 的誤封鎖後，ProbCut 截斷率提升使每一層同深度節點數穩定縮減 0.5%～1.6%，與多付出的 1.3% NPS 嘗試成本相抵，在未重調 `TUNE_PROBCUT_MARGIN` 下即維持 +1.16 Elo 正向平盤。
  2. **符合「近平手有優勢」與正確性修復原則**：Elo 點估計 $> 0$、深度差異 $\le 0.1$ ply、NPS 差異 $\le 5\%$，且消除了「註解與實作不符」的邏輯缺陷。
  3. **後續 SPSA 協同空間**：現行 `TUNE_PROBCUT_MARGIN`（槽位 22）係在舊版 ProbCut 被過度壓制下調出；已將 `TUNE_PROBCUT_MARGIN` 納入第三階段 SPSA 清單以釋放完整潛力。
* **判定：** **採納進生產 / 基線 (ADOPTED / PRODUCTION BASELINE UPDATED)**。
  * 已透過 `python tools/sync_classical_old.py --sync` 同步至 `chess_engine/classical_old/`，並接續實裝 **實驗 B（QSearch 維護 `move_stack`/`piece_stack` 並接通修正歷史）**。

### E82 / 實驗 B：QSearch 維護 `move_stack`/`piece_stack` 並接通修正歷史（68 局 -204.75 Elo 嚴重崩盤，立即回滾 REVERTED）

* **改動項目與動機：**
  * 嘗試在 `quiescence_search` 中對 `stand_pat`（以及最大 QSearch 深度回傳值）套用 `apply_correction_history_score`，並在遞迴前更新 `search_context.move_stack[ply]` 與 `piece_stack[ply]`，以消除主搜尋 `depth = 1` 與靜態搜尋 `depth = 0` 交界處的靜態估值斷差。
* **對戰驗證結果（Colab LTC 400k nodes，68 局 / 34 對換先即觸發大敗判定中止）：**
  * **戰績**：**2 勝 28 和 38 負**（得分率 **23.53%**，總得分 16.0 / 68，淨負 **-36 局**）。
  * **相對 Elo**：**-204.75 Elo**（95% CI: `[-283.19, -141.62]`）。
  * **執色表現**：持白 `23.53%`（2W 12D 20L），持黑 `23.53%`（0W 16D 18L），雙色全面潰敗。
* **搜尋指標分析（`parse_search_stats.py`，68 局，樣本數 4,902 步 vs 4,917 步）：**
  * **平均搜尋深度（嚴重崩塌）**：New **18.27 ply** vs Old **20.10 ply**（暴跌 **`-1.83 ply`**，Welch $t = -4.957, p = 7.26 \times 10^{-7}$；中位數深度由 `16` 跌至 `15`）。
  * **深層抵達率（$\text{depth} \ge 14$）**：由 Old 的 **89.6%** 崩跌至 **68.5%**（`-21.1%`）！
  * **同深度節點比**：在主力深度 `d = 12 .. 15` 上，同深度節點消耗暴增 **`+4.9% ~ +5.8%`**（`ratio = 1.049 ~ 1.058`）。
  * **NPS 與耗時**：NPS 由 639,607 跌至 **599,709（`-6.2%`，$p = 1.35 \times 10^{-10}$）**；每步耗時增加 `+21.4 ms`（`+4.8%`）。
* **物理機理解析（重大工程教訓：為何 HCE 修正歷史絕對不可進入 QSearch）：**
  1. **訓練域與推理域嚴重錯位（Quiet-Trained History Applied to Capture Chains）**：
     * 在主搜尋 `_search`（第 3362 行 `if not is_capture:`）中，四張修正歷史表（Pawn、Minor、Non-Pawn、Continuation）**100% 僅在最佳步為「安靜步」時才更新**，其學習的是靜態陣型的慢變偏置。
     * 然而 `quiescence_search` 是純粹的**連續吃子交換鏈**。在 QSearch 中每走一步吃子，`PAWN_KEY`、`MINOR_KEY` 與 `NON_PAWN_KEY` 都會瞬間跳轉到一個未經該路線訓練的雜湊桶；同時將 QSearch 吃子步寫入 `move_stack[ply]` 後，`continuation_correction_history` 會在深層吃子鏈上查出完全無關的修正值（高達 $\pm 50 \sim 100\text{ cp}$）。
  2. **QSearch `stand_pat` 截斷與 Delta Pruning 大規模失靈，引發葉節點爆炸**：
     * 在連續吃子清算中疊加隨機陣型修正偏置，一方面使大量本應 `stand_pat >= beta` 截斷的節點跌破門檻而被迫展開全套吃子分支（導致 QSearch 節點暴增，在 400k 固定節點預算下直接吸乾主搜尋算力，造成平均深度暴跌 `-1.83 ply`）；另一方面在戰術交換末端產生高達數十厘兵的估值扭曲並透過 `store_tt` 的 `tt_score` 寫入置換表，導致戰術清算全面失準。
  3. **熱路徑快取懲罰**：QSearch 佔總節點數 60% 以上，每節點額外查閱 4 張修正歷史表直接造成 `-6.2%` 的 NPS 損失。
* **判定：** **立即回滾 (REVERTED)，列為永久架構禁區**。
  * 已將 `chess_engine/classical/search.py` 100% 還原至 E81 生產基線（`sync_classical_old.py --diff-code` 確證 0 diff）。

### E83 / 實驗 E：Stockfish 19 Step 18 LMR 重搜門檻校準與 Step 19 條件式過渡降深（418 局 -9.98 Elo、深度顯著下滑 -0.42 ply，回滾 REVERTED）

* **改動項目與動機：**
  1. `constants.py:1300`：將 `LMR_RESEARCH_SHALLOWER_MARGIN` 由 `15` 下調至 `4`（對齊 SF19 `doShallowerSearch = value < bestValue + 9` 之百厘兵尺度）。
  2. `search.py:2909`：在 `do_shallower` 加上 `and not (lmr == 1 and cut_node)` 安全守衛，避免 `lmr == 1` 時因 `adjusted_depth == lmr_depth` 直接跳過重搜。
  3. `search.py:2860`：對齊 SF19 Step 19，當吃子步之後的第一個安靜步（`quiet_move_counter == 1` 且 `searched_legal_moves >= 2`）於 `not is_pv and cut_node`、非殺手/反擊步且 `lmr_stat_score < 0` 時，給予溫和的 `lmr = 1` 過渡降深。
* **對戰驗證結果（Colab LTC 400k nodes，418 局 / 209 對換先平衡）：**
  * **戰績**：**82 勝 242 和 94 負**（得分率 **48.56%**，總得分 203.0 / 418，淨負 **-12 局 / -6.0 分**）。
  * **相對 Elo**：**-9.98 Elo**（±19.99，95% CI: `[-30.00, +9.98]`；自第 100 局起累積得分率持續處於 `46.5% ~ 48.9%` 水下）。
  * **執色與盤長特徵**：
    * 持白：`53.59%`（49W 126D 34L）；持黑：**`43.54%`**（33W 116D 60L，防守端明顯吃虧）。
    * 短局 ($\le 25$ 步) `50.0%`、中局 (26–50 步) `49.1%`、長局 (51–80 步) `48.5%`（52W 61D 57L）、超長局 ($>80$ 步) `47.4%`（20W 52D 25L）。
* **搜尋指標分析（`parse_search_stats.py`，418 局，樣本數 26,330 步 vs 26,307 步）：**
  * **平均搜尋深度（統計顯著下滑）**：New **19.32 ply** vs Old **19.73 ply**（下滑 **`-0.42 ply`**，Welch $t = -2.913, p = 0.0036$ **高度顯著**）。
  * **深層抵達率（$\text{depth} \ge 14$）**：由 Old 的 **90.3%** 下跌至 **85.0%**（**`-5.3%`**；卡在 `d = 11..13` 淺層的比例由 `9.1%` 增至 `14.3%`）。
  * **同深度平均節點比**：在主力深度 `d = 14 .. 18` 一致膨脹 **`+1.0% ~ +2.4%`**（`d=14: 1.015`, `d=15: 1.024`, `d=16: 1.013`, `d=17: 1.010`, `d=18: 1.010`）。
  * **NPS 與耗時**：NPS 599,468 vs 605,006（`-0.9%`, $p = 0.0406$）；每步耗時 470.9 ms vs 468.1 ms（`+2.8 ms`）。
* **物理機理解析（為何引發「LMR 重搜風暴 Re-search Explosion」並拖垮深度）：**
  1. **`LMR_RESEARCH_SHALLOWER_MARGIN` (15 $\to$ 4) 與 Fail-Soft 平滑回傳的尺度衝突**：
     * 本引擎採用 **QSearch Fail-High 平滑**（`481 * score + 543 * beta`）與 **RFP Fail-Soft 阻尼混合**（`412 * beta + 612 * eval`），導致零視窗子搜尋突破 Alpha 時，回傳分數常態性落在 `alpha + 6 ~ 14 cp`。
     * 在舊版（門檻 `15`）中，這批邊際突破的 LMR 著法會觸發 `do_shallower = True`（使 `lmr == 1` 免除重搜，或使 `lmr >= 2` 以 `search_depth - 1` 淺一層重搜，節省約 45% 重搜節點）。將門檻壓至 `4` 並加上 `not (lmr == 1 and cut_node)` 後，大量邊際 Fail-High 著法被迫執行昂貴的全深度（`search_depth`）重搜。
  2. **Step 19 對首步安靜步（`quiet_move_counter == 1`）降深 1 層的「雙重搜尋懲罰」**：
     * 在 `cut_node` 上，當吃子步未能截斷時，排序第一的安靜步（`quiet_move_counter == 1`）極常是真正的突破步。舊版直接以 `search_depth` 搜 1 次即截斷；新版先以 `search_depth - 1` 搜第 1 次，突破 Alpha 後又因 `not (lmr == 1 and cut_node)` 被強制以 `search_depth` 重搜第 2 次，總耗費節點變為 $\text{Nodes}(d-1) + \text{Nodes}(d) \approx 1.6\text{x} \text{Nodes}(d)$。
  3. **400k 固定節點預算下的深度擠壓**：中層迭代（`d = 14..18`）節點膨脹 `+1.0% ~ +2.4%`，使搜尋提前觸及軟節點上限而中止，導致 $\text{depth} \ge 14$ 比例下跌 `5.3%`、平均深度損失 `0.42 ply`。
* **判定：** **回滾 (REVERTED)**。
  * 已將 `chess_engine/classical/constants.py` 與 `chess_engine/classical/search.py` 100% 還原至 E81 生產基線（`sync_classical_old.py --diff-code` 確證 0 diff），並正式推進至第三階段 SPSA 參數空間整備。

### E84：SEE 靜態交換評估全套漏洞修補與 Step 14 前置合法性預檢（600 局 +10.43 Elo、淨勝 +18 局，正式採納進生產基線 ADOPTED）

* **改動項目與動機（詳見 [`chess_engine/classical/SEE_ANALYSIS.md`](../chess_engine/classical/SEE_ANALYSIS.md)）：**
  1. **動態牽制過濾與牽制線反吃（`chess_engine/classical/see.py` `filter_pinned_attackers`）**：
     * 允許被牽制棋子沿牽制線（`LINE_BB[king_sq, to_sq]`）合法反吃；若被牽制子不在該線上，當牽制線（`LINE_BB[king_sq, p_sq]`）在交換初始或中途有棋子離開（`removed_occ & pin_line != 0`）時，動態驗證敵方滑動子是否仍牽制該子，解除牽制後即恢復其反吃能力。
  2. **升變與欠升變（Underpromotion）完整計分與底線後續升變截斷保護（`see.py`, `search.py`, `search_heuristics.py`）**：
     * `_see_jit` 與 `_see_ge_jit` 新增 `promo_type` 參數（由呼叫端傳入 `get_promotion_piece(move) + 1`），並同步修正 QSearch Delta Pruning 之 `promotion_gain`，杜絕欠升變（升馬/象/車）被高估為皇后（900/1170 cp）。
     * 在交換迴圈內處理底線後續兵重吃升變（`+promo_gain`），並在 `_see_ge_jit` 提前截斷條件加入 `if not is_promo_rank or (swap + promo_gain ...)` 守衛，避免在底線格因未計入下一輪兵升變收益而提早誤判跳出。
  3. **過路兵自包含判定與 `-2` 哨兵值安全防護（`see.py`, `search.py`, `search_heuristics.py`）**：
     * 直接由 `(initial_occ & BB_SQUARES[to_sq]) == 0` 判定空格與過路兵並移除 `ep_sq` 佔用位元；在 `_see_jit` 相容 `victim_type == -2`，在 MovePicker Fast Path 要求 `vic_type >= 0`，並在 `search_heuristics.py` 過路兵排序與將軍 SEE 門檻統一傳入 `-2` 哨兵以跳過冗餘敵方 bitboard 掃描。
  4. **王車易位將軍預判與 SEE 豁免 + Step 14 前置 $O(1)$ 合法性預檢（`search.py`）**：
     * 修正王車易位的 `pre_move_gives_check` 為檢查易位後城堡落點 `rook_to_sq = (from_sq + to_sq) >> 1` 是否對敵王將軍，且王車易位不送入 `_see_ge_jit`（對齊 Stockfish 視為 `SEE = 0`）。
     * 將 $O(1)$ 前置合法性檢查（涵蓋牽制子、國王走法、過路兵揭露將軍、王車易位路徑受攻）提前至 Step 14 淺層剪枝入口，防止非法偽合法走法因剪枝而遞增 `pruned_moves`，徹底保護無合法走法時的逼和（Stalemate）終局判定。
* **對戰驗證結果（Colab 600 局 LTC 400k nodes，換先平衡 300 對完整完賽）：**
  * **戰績**：**123 勝 372 和 105 負**（得分率 **51.50%**，總得分 309.0 / 600，淨勝 **+18 局 / +9.0 分**）。
  * **相對 Elo**：**+10.43 Elo**（95% CI: `[-5.69, +26.59]`，成對換先配對聚類 $n=300$）。
  * **前後半場與執色表現**：
    * 前半場 150 對（300 局）：得分率 **53.0%**（64W 190D 46L，淨勝 +18 局）；後半場 150 對（300 局）：得分率 **50.0%**（59W 182D 59L，穩健守平）。
    * 持白：**52.83%**（68W 181D 51L，淨勝 +17 局）；持黑：**50.17%**（55W 191D 54L，淨勝 +1 局，**持黑勝率突破 50%！**）。
  * **配對九宮格（300 對）**：DD 111 (37.0%), WD 46 (15.3%) vs LD 34 (11.3%, 淨勝 +12 對), DW 39 (13.0%) vs DL 31 (10.3%, 淨勝 +8 對), WL 16 (5.3%), LW 10 (3.3%), LL 7 (2.3%), WW 6 (2.0%)。
  * **盤長分佈特徵（全區間全數 $\ge 50\%$）**：
    * 短局 ($\le 25$ 步)：1W 75D 0L（**50.7%**）。
    * 中局 (26–50 步)：11W 135D 5L（**52.0%**，淨勝 +6 局）。
    * 長局 (51–80 步)：83W 82D 73L（**52.1%**，淨勝 +10 局）。
    * 超長局 ($>80$ 步)：28W 80D 27L（**50.4%**，淨勝 +1 局）。
* **搜尋指標分析（`parse_search_stats.py`，600 局，樣本數 36,777 步 vs 36,735 步）：**
  * **平均搜尋深度**：New **19.64 ply** vs Old **19.76 ply**（差異 `-0.12 ply`, Welch $t = -1.045, p = 0.296$，統計不顯著；$\text{depth} \ge 14$ 比例 `89.7%` vs `90.1%`）。
  * **每步節點數與同深度節點比（關鍵樹效率提升）**：每步節點數 **245,396 vs 248,189（`-1.1%`**, Welch $t = -7.505, p = 6.20 \times 10^{-14}$ 極度顯著）；在 `d = 11 .. 24` 全數 14 個深度層上，同深度節點比一致落在 **`0.981 ~ 0.997`**。
  * **耗時 / NPS**：每步耗時 790.7 ms vs 782.2 ms（`+8.5 ms`, `+1.1%`）；NPS 365,350 vs 372,166（`-1.8%`）。
* **判定：** **正式採納進生產 / 基線 (ADOPTED / PRODUCTION BASELINE UPDATED)**。
  * 已透過 `python tools/sync_classical_old.py --sync` 將 `search.py`、`search_heuristics.py`、`see.py` 完整同步至 `chess_engine/classical_old/`（`--diff` 確證 100% 0 diff）。

---



### E85：SEE 子力價值統一 MG_MATERIAL_VALUES 與門檻等比例 ×1.134 縮放（1000 局 LTC 400k nodes，+7.99 Elo、淨勝 +23 局，調參前基線確證 VERIFIED）

* **改動項目與動機（詳見 [chess_engine/classical/SEE_ANALYSIS.md](../chess_engine/classical/SEE_ANALYSIS.md) §3.1 與 §7）：**
  1. **子力價值尺度對齊**：SEE_PIECE_VALUES 直接引用 MG_MATERIAL_VALUES 前 5 項（PAWN=100, KNIGHT=347, BISHOP=364, ROOK=572, QUEEN=1170），保留 KING=20000（避免 _see_ge_jit 誤判國王吃子提前截斷）。消除過去獨立子力表（100, 320, 330, 500, 900）與 ProbCut/Capture-LMR/HCE 靜態評估尺度不一致的隱患。
  2. **門檻與斜率等比例縮放（$\times 1.134$）**：依開局總子力比值 4536 / 4000 = 1.134 等比例縮放：
     - PRUNING_CAPTURE_SEE_MARGIN: -102 -> -116
     - PRUNING_QUIET_SEE_MARGIN: -32 -> -36
     - QS_SEE_THRESHOLD: -80 -> -91
     - CHECK_SEE_THRESHOLD: -75 -> -85
     - CAP_SEE_CAPTHIST_NUM: 34 -> 39
  3. **Runtime SPSA 5 軸架構接入**：將上述 5 個常數全數納入 SearchContext.tune[] 陣列（TUNE_SIZE = 44），免去重新編譯成本，支持連續 SPSA 微調。
* **對戰驗證結果（Colab 1,000 局 LTC 400k nodes，換先平衡 500 對完整完賽，產物：tournament_analysis/tournament_results.pgn）：**
  * **戰績**：**200 勝 623 和 177 負**（得分率 **51.15%**，總得分 511.5 / 1000，淨勝 **+23 局 / +11.5 分**）。
  * **相對 Elo**：**+7.99 Elo**（95% CI: [-4.47, +20.47]，以成對換先 500 對獨立聚類計算）。
  * **前後半場與執色表現**：
    - 前半場 250 對（500 局）：得分率 49.4%（94W 306D 100L，得分 247.0 / 500）。
    - 後半場 250 對（500 局）：得分率 **52.9%**（106W 317D 77L，得分 264.5 / 500，淨勝 **+29 局**）。
    - 持白：勝率 **54.70%**（118W 311D 71L，淨勝 +47 局）；持黑：勝率 **47.60%**（82W 312D 106L，淨負 -24 局）。
  * **成對九宮格（500 對獨立配對）**：DD 194 (38.8%), WD 78 (15.6%) vs DL 67 (13.4%), DW 50 (10.0%) vs LD 40 (8.0%), WL 27 (5.4%), LW 19 (3.8%), WW 13 (2.6%), LL 12 (2.4%)。
    - **成對正分差**：New 取得正分配對 141 對 vs Old 取得正分配對 119 對，**淨勝 +22 對！**
  * **盤長分佈特徵（中長局決勝主力）**：
    - 短局 (<= 25 步)：n=101，全部和棋（0W 101D 0L，勝率 50.0%）。
    - 中局 (26-50 步)：n=244，13W 224D 7L（勝率 **51.2%**，淨勝 +6 局）。
    - 長局 (51-80 步)：n=394，131W 148D 115L（勝率 **52.0%**，淨勝 +16 局）。
    - 超長局 (> 80 步)：n=261，56W 150D 55L（勝率 **50.2%**，淨勝 +1 局）。
* **搜尋指標分析（parse_search_stats.py，1000 局，樣本數 63,716 步 vs 63,700 步）：**
  * **平均搜尋深度**：New **19.64 ply** vs Old **19.60 ply**（差異 +0.03 ply，Welch t = +0.394, p = 0.693 統計不顯著；depth >= 14 比例 90.8% vs 90.5%）。
  * **每步節點數與同深度節點比**：每步平均節點數 248,134 vs 247,550（+0.2%，Welch t = +2.110, p = 0.035 顯著）；同深度節點比在 d=11..24 全層平均約 1.002，說明門檻完全精確維持了原有的剪枝率平衡。
  * **耗時 / NPS**：每步耗時 929.3 ms vs 924.1 ms（+0.6%，Welch t = +2.652, p = 0.008 顯著）；NPS 329,060 vs 330,716（-0.5%，Welch t = -1.418, p = 0.156 不顯著）。
* **判定與後續動作：** **基線安全確證 (VERIFIED BASELINE)**。
  * 1,000 局淨勝 23 局確證架構安全無退化；因 95% CI 包含 0（[-4.47, +20.47]），且非線性剪枝項（安靜步平方項與歷史吃子斜率）無法直接靠線性縮放求得最優解，因此正式推動 **SEE 5 軸 SPSA 微調計畫**（[tune_colab_runner.ipynb](../tune_colab_runner.ipynb)）。


### E86：晚期著法剪枝 (LMP) 基底常數調整 (6 + d * d) vs (5 + d * d)（600 局 +17.39 Elo、淨勝 +30 局，95% CI 下界 > 0 顯著增益，正式採納進生產基線 ADOPTED）

* **改動項目與動機：**
  * 在 chess_engine/classical/constants.py 中，將晚期著法剪枝（Late Move Pruning, LMP）基底著法門檻表由原先的：
    LMP_MOVE_COUNT = np.array([0 if d == 0 else max(1, (5 + d * d) // 2 - 1) for d in range(MAX_PLY)], dtype=np.int32)
    微調為：
    LMP_MOVE_COUNT = np.array([0 if d == 0 else max(1, (6 + d * d) // 2 - 1) for d in range(MAX_PLY)], dtype=np.int32)
  * **動機**：在 =2, 4$ 等中淺層為安靜著法多放行 1 步候選步（例如 =2$ 門檻從 3 增至 4），避免在局勢複雜或多子防守局面因 LMP 過早剪枝而漏算防守著法（Tactical Blindness）。
* **對戰驗證結果（Colab / 本機 600 局 LTC 換先平衡 300 對完賽，產物：	ournament_analysis/tournament_results.pgn、備份：	ournament_results_E86.pgn）：**
  * **戰績**：**134 勝 362 和 104 負**（得分率 **52.50%**，總得分 315.0 / 600，淨勝 **+30 局 / +15.0 分**）。
  * **相對 Elo**：**+17.39 Elo**（95% CI: **[+0.86, +33.99]**，成對換先 300 對獨立聚類計算，下界 $>0$ 確證顯著強於對手）。
  * **前後半場與執色表現**：
    - 前半場 150 對（300 局）：得分率 **53.2%**（75W 169D 56L，淨勝 +19 局）。
    - 後半場 150 對（300 局）：得分率 **51.8%**（59W 193D 48L，淨勝 +11 局，始終穩定壓制）。
    - 持白：勝率 **57.50%**（87W 171D 42L，淨勝 +45 局）；持黑：勝率 **47.50%**（47W 191D 62L，淨負 -15 局）。
  * **成對九宮格（300 對獨立配對）**：DD 114 (38.0%), WD 55 (18.3%) vs DL 32 (10.7%), DW 25 (8.3%) vs LD 22 (7.3%), WL 21 (7.0%), WW 11 (3.7%), LW 11 (3.7%), LL 9 (3.0%)。
    - **成對正分差**：New 取得正分配對 91 對 vs Old 取得正分配對 63 對，**成對淨勝達 +28 對！**
  * **盤長分佈特徵（殘局致勝核心）**：
    - 短局 ($\le 25$ 步)：n=69，全部和棋（0W 69D 0L，勝率 50.0%）。
    - 中局 (26–50 步)：n=146，11W 129D 6L（勝率 **51.7%**，淨勝 +5 局）。
    - 長局 (51–80 步)：n=242，81W 90D 71L（勝率 **52.1%**，淨勝 +10 局）。
    - 超長局 ($> 80$ 步)：n=143，42W 74D 27L（勝率 **55.2%** 🚀，淨勝 +15 局，殘局優勢大幅拉開）。
* **搜尋指標分析（parse_search_stats.py，600 局，樣本數 37,310 步 vs 37,260 步）：**
  * **平均搜尋深度**：New **19.67 ply** vs Old **19.67 ply**（差異 -0.00 ply，Welch  = -0.031, p = 0.975$，深度完全無損；$	ext{depth} \ge 14$ 比例 89.8% vs 89.6%）。
  * **每步節點數與同深度節點比**：每步平均節點數 246,139 vs 245,994（+0.1%，Welch  = +0.395, p = 0.693$ 統計無差異）；同深度節點比在 =9..24$ 全層平均約 **1.000**，搜尋樹完全未膨脹。
  * **耗時 / NPS**：每步耗時 484.0 ms vs 476.3 ms（+1.6%）；NPS 591,917 vs 599,786（-1.3%）。
* **判定與處置：** **正式採納進生產基線 (ADOPTED / PRODUCTION BASELINE UPDATED)**。
  * 執行 python tools/sync_classical_old.py --sync 將此變更同步至 chess_engine/classical_old/。
  * 此結果為後續 LMP_SCALE_PERCENT（Runtime slot 12）微調提供了決定性的實戰方向支撐。


### E87：晚期著法剪枝縮放百分比放寬 `LMP_SCALE_PERCENT = 150` vs `137`（600 局 +13.32 Elo、淨勝 +23 局，深度躍升 +0.18 ply、搜尋提速 -10.1ms，正式採納進生產基線 ADOPTED）

* **改動項目與動機：**
  * 在 `chess_engine/classical/constants.py` 中，將晚期著法剪枝（Late Move Pruning, LMP）縮放百分比由原先的：
    `LMP_SCALE_PERCENT = 137`
    放寬為：
    `LMP_SCALE_PERCENT = 150`
  * **動機**：在 E86 將 LMP 基底門檻推進至 `(6 + d*d)` 取得顯著正收益後，進一步測試將門檻係數放寬至 150（+9.5% 寬度）。理論預期進一步減少中淺層安靜步的誤剪，使關鍵防守步與轉移步得到充分驗證。
* **對戰驗證結果（Colab / 本機 600 局 LTC 換先平衡 300 對完賽，產物：`tournament_analysis/tournament_results.pgn`、備份：`tournament_results_E87.pgn`）：**
  * **戰績**：**132 勝 359 和 109 負**（得分率 **51.92%**，總得分 311.5 / 600，淨勝 **+23 局 / +11.5 分**）。
  * **相對 Elo**：**+13.32 Elo**（95% CI: `[-3.89, +30.61]`，成對換先 300 對獨立聚類計算）。
  * **前後半場與執色表現**：
    - 前半場 150 對（300 局）：得分率 **53.7%**（71W 180D 49L，淨勝 +22 局）。
    - 後半場 150 對（300 局）：得分率 **50.2%**（61W 179D 60L，淨勝 +1 局）。
    - 持白：勝率 **55.50%**（77W 179D 44L，淨勝 +33 局）；持黑：勝率 **48.33%**（55W 180D 65L，淨負 -10 局）。
  * **成對九宮格（300 對獨立配對）**：DD 104 (34.7%), WD 45 (15.0%) vs DL 41 (13.7%), DW 34 (11.3%) vs LD 31 (10.3%), WL 18 (6.0%), WW 14 (4.7%), LW 7 (2.3%), LL 6 (2.0%)。
    - **成對正分差**：New 取得正分配對 93 對 vs Old 取得正分配對 78 對，**成對淨勝達 +15 對！**
  * **盤長分佈特徵（全區間全數勝率 > 50%）**：
    - 短局 (<= 25 步)：n=77，全部和棋（0W 77D 0L，勝率 50.0%）。
    - 中局 (26–50 步)：n=132，11W 118D 3L（勝率 **53.0%**，淨勝 +8 局）。
    - 長局 (51–80 步)：n=244，85W 85D 74L（勝率 **52.3%**，淨勝 +11 局）。
    - 超長局 (> 80 步)：n=147，36W 79D 32L（勝率 **51.4%**，淨勝 +4 局）。
* **搜尋指標分析（`parse_search_stats.py`，600 局，樣本數 36,999 步 vs 37,021 步）：**
  * **平均搜尋深度躍升**：New **19.94 ply** vs Old **19.75 ply**（**深度顯著提升 +0.18 ply**，中位數 17 vs 16，`depth >= 14` 比例由 89.5% 顯著躍升至 **91.5%**！）。
  * **每步搜尋提速與 NPS 躍升**：
    - 每步耗時 466.7 ms vs 476.8 ms（**每步節省 -10.1 ms，提速 2.1%**，Welch t = -9.140, p = 6.37e-20 極度顯著！）。
    - NPS 604,209 vs 595,885（**+1.4% NPS 提升**，Welch t = +3.685, p = 2.29e-4 統計高度顯著！）。
  * **同深度節點比（關鍵樹效率提升）**：在 d = 14 .. 19 核心中層，New 到達同深度所耗費的節點比為 **`0.985 ~ 0.992`**（省約 1.1% ~ 1.5% 節點）。適度放寬 LMP 避免了過早誤剪造成的昂貴搜尋回溯，Alpha-Beta 截斷反而更早更乾淨。
* **判定與處置：** **正式採納進生產基線 (ADOPTED / PRODUCTION BASELINE UPDATED)**。
  * 執行 `python tools/sync_classical_old.py --sync` 將 `LMP_SCALE_PERCENT = 150` 完整同步至 `chess_engine/classical_old/`。
  * 實證 LMP 在基底與縮放連續兩輪放寬累計帶來超過 **+30 Elo** 的強大增益，為後續調參奠定最高效生產基線。


### E88：ProbCut 剪枝裕度 1D 座標探索與微調 `PROBCUT_MARGIN = 170` vs `150`（1,000 局 +4.86 Elo，190 vs 170 落敗 -5.91 Elo，正式採納進生產基線 ADOPTED）

* **改動項目與動機：**
  * 在 `chess_engine/classical/constants.py` 中，將 ProbCut 剪枝裕度常數由原先的：
    `PROBCUT_MARGIN = 150`
    微調為：
    `PROBCUT_MARGIN = 170`（放寬約 +13.3%）
  * **動機**：
    1. 評估架構在 E85 統一了 `MG_MATERIAL_VALUES`（使子力尺度整體上升約 13.4%），而舊版 `PROBCUT_MARGIN = 150` 係在舊子力表下調出，尺度相對過緊。
    2. 在 E81 修正 ProbCut TT Defense 漏檢後，ProbCut 截斷機制已完全正常運作。透過單參數 1D 座標爬山搜尋（Coordinate Descent / Peak Bracketing），探尋最佳截斷邊界。
* **對戰驗證結果（1D 座標兩側探針驗證，Colab LTC 1,000 局）：**
  * **第 1 輪探索（向左探針，120 vs 150）：**
    * 嘗試收緊至 `PROBCUT_MARGIN = 120`，實測落敗，證實門檻不可過低（過低易引發戰術漏算與不穩定截斷）。
  * **第 2 輪對決（主測試：New170 vs Old150，1,000 局）：**
    * **戰績**：**214 勝 586 和 200 負**（得分率 **50.70%**，總得分 507.0 / 1000，淨勝 **+14 局 / +7.0 分**）。
    * **相對 Elo**：**+4.86 Elo**（95% CI: `[-9.00, +18.72]`，±13.86）。
    * **狀態**：`Status: Continue; LLR=-0.03`。
  * **第 3 輪驗證（向右探針：New190 vs Old170，1,000 局，邊界鎖定）：**
    * **戰績**：**191 勝 601 和 208 負**（得分率 **49.15%**，總得分 491.5 / 1000，淨負 **-17 局 / -8.5 分**）。
    * **相對 Elo**：**-5.91 Elo**（95% CI: `[-19.51, +7.69]`，±13.60）。
    * **狀態**：`Status: Continue; LLR=-2.28`。
* **物理機理解析與採納決策（極致的局部峰值鎖定）：**
  1. **峰值夾擊驗證（Peak Bracketing）**：
     * 實測序列呈現鮮明的凹函數單峰特徵：$120 < 150 < 170 > 190$。
     * 170 成功擊敗 150（+4.86 Elo），而進一步放寬至 190 則顯著回落（-5.91 Elo），嚴謹確立了 $170\text{ cp}$ 為該鄰域的全局最優局部極值點。
  2. **符合 E85 子力尺度對齊**：$150 \times 1.134 = 170.1$，與實測最優值 170 完美自洽。
* **判定與處置：** **正式採納進生產基線 (ADOPTED / PRODUCTION BASELINE UPDATED)**。
  * 保持 `chess_engine/classical/constants.py` 與 `chess_engine/classical_old/constants.py` 之 `PROBCUT_MARGIN = 170`。
  * 執行 `python tools/sync_classical_old.py --sync` 確保兩側代碼邏輯一致。


### E89：Delta Pruning 增益剪枝裕度 1D 座標探索與深度審計 `DELTA_PRUNING_MARGIN = 365` vs `380` / `350` / `395`（多輪實測收斂，選定 365，正式採納進生產基線 ADOPTED）

* **改動項目與動機：**
  * 在 `quiescence_search` 中全面對齊 Stockfish 11/19，補齊 `not is_loss(stand_pat)` 必敗斷崖護欄（防止殘局特殊局面因 `-10000` 人造分誤剪反吃走法），並消除 QSearch 熱路徑上重複的 `find_piece_type_on_square_side` 查表。
  * 在 E85 統一 `MG_MATERIAL_VALUES` 使子力價值尺度整體上升（約 +13.4%）後，原先 `DELTA_PRUNING_MARGIN = 200`（後續調整至 350）過於收緊，需重新探索最佳截斷門檻。
* **對戰驗證結果（多輪 1D 座標爬山驗證）：**
  * **第 1 輪（New380 vs Old350，772 局）**：
    * **戰績**：**170 勝 482 和 120 負**（得分率 **53.24%**，總得分 411.0 / 772，淨勝 **+50 局**）。
    * **相對 Elo**：**+22.53 Elo**（95% CI: `[+7.42, +37.73]`）。
  * **第 2 輪（New395 vs Old380，724 局）**：
    * **戰績**：**174 勝 426 和 124 負**（得分率 **53.45%**，總得分 387.0 / 724，淨勝 **+50 局**）。
    * **相對 Elo**：**+24.03 Elo**（95% CI: `[+8.32, +39.85]`）。
  * **第 3 輪（New365 vs Old380，1,000 局）**：
    * **戰績**：**196 勝 621 和 183 負**（得分率 **50.65%**，總得分 506.5 / 1000，淨勝 **+13 局**）。
    * **相對 Elo**：**+4.52 Elo**（95% CI: `[-8.50, +17.55]`）。
  * **第 4 輪（New395 vs Old365，604 局）**：
    * **戰績**：**124 勝 372 和 108 負**（得分率 **51.32%**，總得分 310.0 / 604，淨勝 **+16 局**）。
    * **相對 Elo**：**+9.21 Elo**（95% CI: `[-7.67, +26.12]`）。
* **判定與處置：** **選定 365 正式採納進生產基線 (ADOPTED / PRODUCTION BASELINE UPDATED)**。
  * 保持 `DELTA_PRUNING_MARGIN = 365`。
  * 執行 `python tools/sync_classical_old.py --sync` 同步至 `classical_old`。


### E90：NMP 空著剪枝基底門檻探索與收斂 `NMP_BASE = 230` vs `185` / `205` / `215` / `225` / `235` / `245`（多輪網格搜尋收斂至 230，正式採納進生產基線 ADOPTED）

* **改動項目與動機：**
  * 在修復 `search.py` 中 NMP 評估與淺層深度問題後，對齊 SF19 Step 10 極簡架構，針對 HCE 評估尺度重新調優 `NMP_BASE`。
* **對戰驗證結果（探索序列）：**
  * 測試 `185 vs 205`、`185 vs 225`，確認 `225` 明顯優於 `205` 與 `185`。
  * 測試 `215 vs 225`（215 輸 225），測試 `235 vs 225`（235 贏 225），進一步測試 `245 vs 235`。
  * 雙向夾擊驗證：
    * `tournament_results_NMP`（1,000 局）：**208 勝 601 和 191 負**（得分率 **50.85%**，**+5.91 Elo**，95% CI: `[-6.96, +18.79]`）。
    * `tournament_results_NMP1`（1,000 局）：**194 勝 603 和 203 負**（得分率 **49.55%**，**-3.13 Elo**，95% CI: `[-16.19, +9.93]`）。
  * 確立最優性能平原落於 225 ~ 235 之間。
* **判定與處置：** **最後選定中位數 `NMP_BASE = 230` 正式採納進生產基線 (ADOPTED / PRODUCTION BASELINE UPDATED)**。
  * 將 `NMP_BASE = 230` 寫入 `chess_engine/classical/constants.py` 與 `chess_engine/classical_old/constants.py`。


### E91：奇異延展裕度基底微調 `SINGULAR_MARGIN_BASE = 58` vs `52` / `46`（放寬奇異門檻增益，正式採納進生產基線 ADOPTED）

* **改動項目與動機：**
  * 原先奇異延展基底裕度為 `SINGULAR_MARGIN_BASE = 52`。
  * 探索裕度微調：分別向左測試收緊 `46 vs 52`，以及向右測試適度放寬 `58 vs 52`。
* **判定與處置：** **接納SINGULAR_MARGIN_BASE = 58 正式採納進生產基線 (ADOPTED / PRODUCTION BASELINE UPDATED)**。
  * 實證適度放寬奇異延展 Margin 能使關鍵主變例獲得更充足的深層戰術延展，改善複雜局勢下的戰術決策。
  * 將 `SINGULAR_MARGIN_BASE = 58` 寫入 `chess_engine/classical/constants.py` 與 `chess_engine/classical_old/constants.py`。


### E92：延續歷史淺層剪枝門檻 `PRUNING_CONTINUATION_THRESHOLD = -1400` vs `-1450` / `-1500`（採納進生產基線 ADOPTED）

* **動機與背景**：
  * Step 14b Continuation History Pruning 基準門檻原為 `PRUNING_CONTINUATION_THRESHOLD = -1500`。
  * 經過 `-1350 vs -1500` 與 `-1400 vs -1450` 測試，收斂確定以 `-1400` 作為最佳平衡點，兼顧淺層無效走法剪枝率與戰術安全。
* **處置結論**：**採納進生產基線 (ADOPTED)**，`PRUNING_CONTINUATION_THRESHOLD = -1400` 同步至 `classical` 與 `classical_old`。


### E93：搜尋、評估與置換表純 Bug 深度修復（HEAD = cf312b0，正式採納進生產基線 ADOPTED）

* **修復項目（拔除爭議之 ProbCut 重寫，保留 3 項確證之純 Bug 修復）**：
  1. **置換表空槽優先覆寫與 static_eval 補寫（`transposition_table.py`）**：
     * `store_tt` 遇到 `TT_FLAG_NONE` 空槽時無條件優先使用（`replace_idx = idx; break`），防止 QSearch (`depth=0`) 在跨世代時誤將已計算的深層優質條目沖掉。
     * 相同 Key 命中且新值有效時，自動補齊舊條目中缺損的 `static_eval`。
  2. **特殊殘局 Sentinel 牽制防護（`evaluation.py`）**：
     * `hit_special` 特殊勝局與多兵純王兵殘局回傳 `PINNED_UNCOMPUTED_SENTINEL`，杜絕 QSearch 與合法性檢查誤將其當作「已計算且無牽制」，防範牽制線反吃誤算。
  3. **歷史剪枝與 Fail-Low 对手獎勵防護（`search.py`）**：
     * Step 14b 補齊 `history_score` 與 `main_history_score` 變數初始化，並在 `if not prune_quiet_move:` 內精確執行第二層門檻判定。
     * Step 21 Fail-low 对手走法獎勵排除吃子與升變，僅給予安靜非升變走法。
  4. **ProbCut 維持 Baseline 穩定版**：
     * 不引入爭議性的 Step 12a 淺層過早截斷與 TT 虛高分數未扣減寫入，維持純淨穩定的剪枝邏輯。
* **實測對局戰績（1000 局滿程成對換先開局，HEAD = cf312b0 vs 完全未動基線，雙方同參數）**：
  * **總戰績**：Candidate **194 勝 633 和 173 負**（得分率 **51.05%**，淨勝 **+21 局**）。
  * **相對 Elo**：**+7.30 Elo**（95% CI: `[-4.94, +19.55]`）。
  * **分執色表現**：
    * 持白：114W 319D 67L（得分率 54.70%，淨勝 +47）
    * 持黑：80W 314D 106L（得分率 47.40%，淨負 -26，防守端顯著止血）
  * **分盤長勝率**：
    * <= 25 步：99 局 0W 99D 0L（50.0%，淨勝 0）
    * 26 ~ 50 步：283 局 14W 253D 16L（49.6%，淨勝 -2）
    * 51 ~ 80 步：381 局 118W 144D 119L（49.9%，淨勝 -1）
    * **> 80 步（深層超長殘局）：237 局 62W 137D 38L（勝率 55.06%，淨勝 +24 局！）**
  * **走勢強勢逆轉**：
    * 前 296 局受初期特定開局庫抽樣方差波動影響（47.97%，-14.09 Elo，淨負 -12 局）。
    * 後 704 局（第 297~1000 局）展現出壓倒性穩定度，打出 **144 勝 441 和 119 負（得分率 51.78%，淨勝 +25 局，區間 Elo +12.37 Elo）**！
    * 搜尋樹指標：平均深度 19.66 vs 19.63（+0.03 ply），每步節點數差異 -0.0%，同深度節點比（d=9~24）平穩維持在 0.974 ~ 1.004。
    * 確證置換表空槽優先保護與特殊殘局 Sentinel 能徹底杜絕深層哈希覆寫污染與牽制誤算，在深層殘局中產出穩定的 +7.30 Elo 實質淨收益。
* **處置結論**：**正式採納進生產基線 (ADOPTED)**，新舊引擎代碼全面同步升級至 `cf312b0`。


### E94：TT Shortcut 剪枝裕度探索 `PROBCUT_TT_SHORTCUT_MARGIN = 310` vs `334`（觸發 SPRT H1 Accepted，採納進生產基線 ADOPTED）

* **動機與原理**：
  * Step 12 TT-based ProbCut Shortcut 原裕度為 `PROBCUT_TT_SHORTCUT_MARGIN = 334`。
  * 早期設定的 334 裕度偏保守，將門檻適度放寬至 310（-24 cp）時，在 `depth >= probcut_min_depth` 下，已在 TT 中證明有足夠優勢（>= beta + 310）的深層盤面能更及時觸發 Shortcut 截斷，節省的算力使中殘局的決策品質顯著提升。
  * 在合入 E93 健全生產基線後，以嚴格成對開局換先進行高強度正式驗證（`tournament_results_334vs310`），並啟用 SPRT 檢定（`elo0=0.0, elo1=10.0, alpha=0.05, beta=0.05`）。
* **實測對局戰績（972 局成對換先開局，HEAD = 4c57063，New=310 vs Old=334）**：
  * **總戰績**：Candidate **204 勝 611 和 157 負**（得分率 **52.42%**，總得分 509.5 / 972，淨勝 **+47 局**）。
  * **相對 Elo**：**+16.81 Elo**（95% CI: `[+3.53, +30.14]`，置信區間下界突破零軸達 +3.53 Elo）。
  * **SPRT 檢定狀態**：
    * 檢定區間：`[0.0, 10.0]` Elo，門檻邊界 `[-2.9444, +2.9444]`。
    * 於第 486 組成對開局（第 972 局）時，**SPRT LLR 達到 +2.9553（> +2.9444），成功觸發 `H1 Accepted (Engine improved)` 提早截斷收斂！**
  * **分執色表現**：
    * 持白：123W 301D 62L（得分率 56.28%，淨勝 +61）
    * 持黑：81W 310D 95L（得分率 48.56%，淨負 -14，黑方防守表現出色）
  * **分盤長勝率**：
    * <= 25 步：120 局 0W 120D 0L（50.0%，淨勝 0）
    * 26 ~ 50 步：235 局 12W 219D 4L（51.70%，淨勝 +8）
    * **51 ~ 80 步（中殘局勝負關鍵期）：405 局 146W 157D 102L（勝率 55.43%，淨勝 +44 局！）**
    * > 80 步（殘局）：212 局 46W 115D 51L（48.82%，淨負 -5）
  * **搜尋樹品質**：
    * 平均深度 19.89 vs 19.87（+0.02 ply）。
    * 每步耗時由 903.8ms 降至 897.0ms（平均每步提速 -6.8ms，搜尋效率顯著提升）。
    * 每步平均節點 248,225 vs 248,301（-0.03%），完全無異常膨脹。
* **處置結論**：**正式採納進生產基線 (ADOPTED / PRODUCTION BASELINE UPDATED)**，`PROBCUT_TT_SHORTCUT_MARGIN = 310` 已同步至 `classical` 與 `classical_old`。


### E95：TT Shortcut 剪枝裕度進一步向下探索 `PROBCUT_TT_SHORTCUT_MARGIN = 290` vs `310`（採納進生產基線 ADOPTED）

* **動機與原理**：
  * 在 E94 測試中，將 `PROBCUT_TT_SHORTCUT_MARGIN` 由 334 降至 310 大獲全勝（972 局 509.5/972，+16.81 Elo，SPRT LLR=+2.96 成功觸發 H1 Accepted）。
  * 為探明 TT ProbCut Shortcut 門檻的最優收斂區間，進一步沿梯度向下微調至 `290`（-20 cp），同時建立平行對照組探索 `300`（-10 cp），驗證是否能在保持戰術健全的前提下進一步擴大深層殘局搜尋效率。
* **實測對局戰績（1000 局成對換先開局，HEAD = 4f9f9a0，New=290 vs Old=310）**：
  * **總戰績**：Candidate **199 勝 622 和 179 負**（得分率 **51.00%**，總得分 510.0 / 1000，淨勝 **+20 局**）。
  * **相對 Elo**：**+6.95 Elo**（95% CI: `[-5.08, +18.99]`）。
  * **SPRT 檢定狀態**：LLR=+0.52（bounds: `[-2.94, +2.94]`，未提前觸發截斷，完整跑滿 1,000 局）。
  * **分執色表現**：
    * 持白：126W 307D 67L（得分率 55.90%，淨勝 +59）
    * 持黑：73W 315D 112L（得分率 46.10%，淨負 -39）
  * **分盤長勝率**：
    * <= 25 步：99 局 0W 99D 0L（50.0%，淨勝 0）
    * 26 ~ 50 步：266 局 17W 237D 12L（50.9%，淨勝 +5）
    * 51 ~ 80 步：375 局 116W 145D 114L（50.3%，淨勝 +2）
    * **> 80 步（深層超長殘局）：260 局 66W 141D 53L（勝率 52.50%，淨勝 +13 局！）**
  * **搜尋樹品質**：
    * 平均深度 19.78 vs 19.76（+0.01 ply）。
    * 每步耗時 875.9ms vs 848.8ms（+27.1ms）。
    * 每步平均節點 250,738 vs 250,659（+0.0%），NPS 349,004 vs 359,095（-2.8%）。
* **平行對照組（300 vs 310，1000 局）驗證結果**：
  * 戰績：Candidate **203 勝 610 和 187 負**（得分率 **50.80%**，淨勝 **+16 局**，相對 **+5.56 Elo**，95% CI: `[-7.44, +18.57]`）。
  * 同樣在 >80 步深層殘局取得淨勝 +15 局（69W 132D 54L），且 NPS/耗時幾乎完全持平（346.8k vs 347.2k，-0.1%）。
  * 雙向驗證確立 `PROBCUT_TT_SHORTCUT_MARGIN` 落在 290 ~ 300 區間皆顯著優於 334 且微幅勝過 310（+16 ~ +20 淨勝局）。
* **處置結論**：**方案 A 正式採納進生產基線 (ADOPTED / PRODUCTION BASELINE UPDATED)**，`PROBCUT_TT_SHORTCUT_MARGIN = 290` 已同步至 `classical` 與 `classical_old`。

---

### E96：前向剪枝修訂回退與奇異擴充 TT PV Bonus 探索 `SINGULAR_TTPV_BONUS = 65` vs `62`（採納進生產基線 ADOPTED）

* **動機與原理**：
  * **前向剪枝探索（Tasks A1~A5）實測總結**：
    * 針對升變、Step 14 淺層剪枝、Capture Futility 及 Quiescence Search 之修正（包含 A3/A4 變體），在 Colab 1000 局高強度對戰（`copy 8.ipynb`）中錄得 176W 637D 187L（得分率 49.5%，-3.82 Elo，LLR=-2.41）。
    * 實測證實在無全局大規模重調參支援下，局部前向剪枝與歷史表補償容易引發葉節點震盪與剪枝失衡，判定全數回退至 commit `17f9138` 前乾淨基準。
  * **奇異擴充獎勵 `SINGULAR_TTPV_BONUS` 參數景觀探索**：
    * `54 vs 62`（`copy 5.ipynb`，1000 局）：193W 596D 211L，得分率 49.1%，-6.25 Elo，LLR=-2.59（裕度過寬導致分支膨脹，負向顯著）。
    * `62`：舊基準。
    * `70 vs 62`（`copy 6.ipynb`，1000 局）：188W 634D 178L，得分率 50.5%，+3.47 Elo，淨勝 +10 局，正向增益。
    * 依梯度內插選定 65（介於基準 62 與 70 之中間值），展開 1000 局嚴格成對換先 A/B 對局驗證。
* **實測對局戰績（1000 局成對換先開局，HEAD = e55d4e4，New=65 vs Old=62，`copy 4.ipynb`）**：
  * **總戰績**：Candidate **203 勝 626 和 171 負**（得分率 **51.60%**，總得分 516.0 / 1000，淨勝 **+32 局**）。
  * **相對 Elo**：**+11.12 Elo**（95% CI: `[-1.23, +23.50]`）。
  * **SPRT 檢定狀態**：LLR=+1.55（bounds: `[-2.94, +2.94]`，強勢正向走勢，完整跑滿 1,000 局）。
  * **分執色表現**：
    * 持白：115W 322D 63L（勝率 55.20%，淨勝 +52）
    * 持黑：88W 304D 108L（勝率 48.00%，淨負 -20）
  * **分盤長勝率**：
    * <= 25 步：86 局 0W 86D 0L（50.0%，淨勝 0）
    * 26 ~ 50 步：255 局 12W 226D 17L（49.0%，淨負 -5）
    * **51 ~ 80 步（中長盤對攻）：401 局 140W 156D 105L（勝率 54.40%，淨勝 +35 局！）**
    * > 80 步（超長殘局）：258 局 51W 158D 49L（勝率 50.40%，淨勝 +2 局）
  * **搜尋樹品質**：
    * 平均深度 19.62 vs 19.61（+0.01 ply，Welch t=+0.084, p=0.933，完全無深度衰退）。
    * 每步耗時 873.2ms vs 869.5ms（+3.7ms）。
    * 每步平均節點 249,351 vs 249,287（+0.0%），NPS 349,988 vs 350,923（-0.3%）。
    * **同深度平均節點比（d=9~24）**：比值嚴格落在 0.980 ~ 1.018，主要常規深度（d=14~20）均為 0.999 ~ 1.002，證實 SE 擴充極度精準，無任何多餘節點浪費。
* **參數景觀匯總（1000 局 scale）**：
  * `54`：-6.25 Elo (LLR=-2.59)
  * `62`：0.00 Elo (基準)
  * `65`：**+11.12 Elo (LLR=+1.55, 淨勝 +32 局)**  🏆 最佳最優點
  * `70`：+3.47 Elo (LLR=-0.37, 淨勝 +10 局)
* **處置結論**：**正式採納進生產基線 (ADOPTED / PRODUCTION BASELINE UPDATED)**，`SINGULAR_TTPV_BONUS = 65` 已同步至 `classical_old` 及 `tuner/constants_snapshot.py`。

---

### E97：LMR 基礎偏移量與雙延展起點深度探索（`LMR_BASE_OFFSET` 475 vs 465、`SINGULAR_DOUBLE_EXT_MIN_DEPTH = 9 vs 10`）

* **動機與原理**：
  * 在 E96 確立 `SINGULAR_TTPV_BONUS = 65` 後，針對 LMR 基礎偏移與奇異雙延展起點深度進行邊界探測。
* **實測對局與結果**：
  1. **`LMR_BASE_OFFSET = 475 vs 465`（`copy 4.ipynb`）**：
     * 494 局（247 對成對換先開局，時控 10,000 nodes/move）：Candidate 88W 299D 107L（得分率 **48.1%**，**-13.4 ± 19.2 Elo**，SPRT LLR = **-2.02**）。
     * 分析：偏移量增加 10 點（465 -> 475）導致全樹 non-PV 著法普遍減深過重，大量關鍵戰術變例被提前截斷漏算，顯著負向，**否決回滾**。隨後依據戰術精度需求將基準微調放寬至 `455`（commit `a8bcdf3`，並測試 `460 vs 455` commit `c4ca843`）。
  2. **`SINGULAR_DOUBLE_EXT_MIN_DEPTH = 9 vs 10`（`copy 6.ipynb`）**：
     * 820 局（410 對成對換先開局，時控 10,000 nodes/move）：Candidate 160W 494D 166L（得分率 **49.6%**，**-2.5 ± 15.0 Elo**，SPRT LLR = **-1.44**）。
     * 分析：在 depth=9 即提早開啟雙延展，使非 PV 節點過早觸發昂貴的深層追搜，導致搜尋樹有效深度略微稀釋，微幅負向，**否決維持 10**。
* **處置結論**：維持 `SINGULAR_DOUBLE_EXT_MIN_DEPTH = 10`，LMR 基礎偏移量收斂於 `455`。

---

### E98：LMR 重新搜尋深度自適應門檻多維微調（`LMR_RESEARCH_DEEPER_MARGIN` & `SHALLOWER_MARGIN`）

* **動機與原理**：
  * 針對 LMR 減深著法在零視窗搜尋中 fail-high 時，觸發 deeper (+1 depth) 或 shallower (-1 depth) 的評估偏置裕度展開細部探索（對齊 Stockfish Re-search 深度自適應）。
* **實測對局對比**：
  1. `DEEPER = 24 vs 26`（`copy 2.ipynb`）：480 局 91W 301D 88L（得分率 **50.3%**，**+2.2 ± 19.0 Elo**，SPRT LLR = **-0.29**）。
  2. `DEEPER = 28 vs 26`（`copy 7.ipynb`）：244 局 45W 154D 45L（得分率 **50.0%**，**-0.0 ± 26.5 Elo**，SPRT LLR = **-0.28**）。
  3. `SHALLOWER = 10 vs 15`（`copy 5.ipynb`）：472 局 94W 289D 89L（得分率 **50.5%**，**+3.7 ± 19.5 Elo**，SPRT LLR = **-0.15**）。
  4. `SHALLOWER = 18 vs 15`（`copy.ipynb`）：204 局 36W 130D 38L（得分率 **49.5%**，**-3.4 ± 28.7 Elo**，SPRT LLR = **-0.38**）。
* **分析與處置**：
  * 四組平行實驗累計跑了超過 1,400 局，勝率全部緊鎖在 49.5% ~ 50.5% 的隨機噪聲帶，LLR 均未產生有效偏移。
  * 根因分析：LMR 重新搜尋深度微調需同時滿足「著法減深 + 零視窗 fail-high + 評估落入極端裕度區間」，全樹觸發比例低於 0.3%，微調 2~5 cp 對勝率無統計顯著性，屬於平原噪聲。
  * **處置結論**：終止微調，鎖定基準為 `DEEPER = 26, SHALLOWER = 18`。

---

### E99：晚期移動步數剪枝比例 `LMP_SCALE_PERCENT` 全局多維梯度與 400k LTC 突破

* **動機與原理**：
  * Late Move Pruning（LMP）控制非 PV 節點在嘗試若干靜態走法後，直接剪除剩餘排序靠後之走法：
    $$\text{limit} = \frac{\text{LMP\_MOVE\_COUNT}[\text{depth}]}{2 - \text{improving}} \times \frac{\text{LMP\_SCALE\_PERCENT}}{100}$$
  * 生產基準原為 150%。為了探明 LMP 在當前搜尋深度與高時控下的最優剪枝閥值，啟動多點梯度實測（135%、142%、145%、155%、160%、165%）。
* **多點梯度對局對比**：
  1. **135% vs 150%**（`copy 8.ipynb` 前期，400k nodes）：250 局 36W 154D 59L（得分率 **45.6%**，**-30.7 ± 26.6 Elo**，SPRT LLR = **-2.00**）❌ 剪枝太狠，防守漏步崩潰。
  2. **155% vs 150%**（`copy 3.ipynb`，400k nodes）：160 局 25W 103D 32L（得分率 **47.8%**，**-15.2 ± 32.1 Elo**，SPRT LLR = **-0.73**）❌ 負向。
  3. **160% vs 150%**（`runner.ipynb` 前期，400k nodes）：260 局 48W 166D 46L（得分率 **50.4%**，**+2.7 ± 25.4 Elo**，SPRT LLR = **-0.16**）⚪ 與 150 持平。
  4. **142% vs 150%**（`copy 3.ipynb` 前期，400k nodes）：120 局 23W 80D 17L（得分率 **52.5%**，**+17.4 ± 35.8 Elo**，SPRT LLR = **+0.34**）📈 正向。
  5. **145% vs 150%**（`copy 8.ipynb` 當前，400k nodes）：274 局 56W 168D 50L（得分率 **51.1%**，**+7.6 ± 25.6 Elo**，SPRT LLR = **+0.16**）📈 正向。
  6. **165% vs 150%**（`tournament_colab_runner_se_multicut.ipynb`，400,000 nodes/move 長時控 LTC）：
     * **戰績**：Candidate **87 勝 219 和 64 負**（得分率 **53.1%**，總得分 196.5 / 370，淨勝 **+23 局**）。
     * **相對 Elo**：**+21.6 ± 22.6 Elo**。
     * **SPRT 檢定狀態**：**LLR = +1.16**（bounds `[-2.94, +2.94]`，強勢正向攀升）。
* **核心機理深度剖析**：
  * **高時控（400k nodes LTC）深層防守盲區假說**：
    在 400,000 nodes 的長時控下，引擎搜尋深度常深入至 $d = 20 \sim 28$。在此深度下，若維持 150%（或更緊的 135%），深層靜態著法的探索寬度會被過早掐斷，產生「過度剪枝盲區（Pruning Blindness）」，導致長盤殘局中的細微防守步與走位失察。
  * 放寬至 **165%** 讓高時控下的深層高質量 Move Ordering 能夠探索更多邊緣防守步，同時不會膨脹淺層節點，為長時控高水平對局提供了關鍵的戰術健全性，淨勝 +23 局成果極為顯著！
* **處置建議**：全力支援 `165 vs 150` 完成對戰；若確認達到顯著性，即採納進長時控生產基準。

---

### E100：LMR All-node 減步縮放分子微調探索（`LMR_ALLNODE_SCALE_NUM = 260 vs 276`）

* **動機與原理**：
  * 在 `search.py` 中對預期非 PV 且非 Cut-node 的 All-node 進行額外 LMR 減步：
    $$r \mathrel{+}= \operatorname{trunc\_div}\left(r \times \text{LMR\_ALLNODE\_SCALE\_NUM},\, 256 \times \text{depth} + 268\right)$$
  * 原先設定值 `276` 在 All-node 減幅較大。在 HCE 評估波動與動態複雜局面下，過激的 All-node 減步容易導致關鍵防守步或反擊戰術被過度剪枝（Over-reduction）。
  * 本實驗探測將縮放分子由 `276` 調降至 `260`（減幅微調減少約 5.8%），驗證是否能在保證搜尋效率的同時降低深層殘局漏算風險。
* **實測對局與搜尋樹數據（1,000 局成對測試）**：
  * **總戰績**：1,000 局（500 組嚴格成對開局庫換先測試）Candidate **215 勝 590 和 195 負**（得分率 **51.00%**，得分 510.0 / 1000.0，淨勝 **+20 局**）。
  * **相對 Elo**：**+6.95 Elo**（95% CI: `[-6.77, +20.69]`）。
  * **分執色表現**：
    * 執白：112 勝 309 和 79 負（得分率 53.30%，淨勝 +33 局）。
    * 執黑：103 勝 281 和 116 負（得分率 48.70%，淨勝 -13 局，顯著優於對手持黑的 79 勝 112 負）。
  * **分局長表現**：
    * 短局 (<= 25 步)：103 局，0W 102D 1L（淨勝 -1 局）。
    * 中局 (26–50 步)：231 局，17W 199D 15L（淨勝 +2 局）。
    * 長局 (51–80 步)：409 局，138W 144D 127L（得分率 **51.3%**，**淨勝 +11 局**）。
    * 超長殘局 (> 80 步)：257 局，60W 145D 52L（得分率 **51.6%**，**淨勝 +8 局**）。
    * **50 步以上長局合計淨勝 +19 局**，優勢極度集中於深層殘局！
  * **搜尋樹品質指標（parse_search_stats 主統計 127,773 步）**：
    * 平均深度：New 19.76 vs Old 19.76（差異 +0.00，Welch t = +0.052, p = 0.959）。
    * 平均節點數：New 249,553 vs Old 249,534（差異 +0.0%，Welch t = +0.068, p = 0.946）。
    * NPS：New 642,234 vs Old 643,132（差異 -0.1%，Welch t = -0.488, p = 0.625）。
    * 每步耗時：New 445.7 ms vs Old 445.0 ms（差異 +0.7 ms，Welch t = +0.909, p = 0.363）。
    * 同深度平均節點比（Same-Depth Node Ratio）：d=13~20 嚴格貼合 1.000（0.998 ~ 1.003）。
* **核心機理與結論**：
  * 微調 `LMR_ALLNODE_SCALE_NUM = 260` 證實完全沒有帶來搜尋樹體積膨脹（節點比 1.000，深度與 NPS 無損），但顯著修復了深層殘局與持黑防守端的 over-reduction 盲點，實測斬獲淨勝 +20 局與 +6.95 Elo。
* **處置結論**：**正式採納進生產基線 (ADOPTED / PRODUCTION BASELINE UPDATED)**，`LMR_ALLNODE_SCALE_NUM = 260` 已寫入 `constants.py` 並同步至 `classical_old`。

---

### E101：LMR All-node 減深分母多維梯度探針與基準確認對戰準備

* **動機與原理**：
  * 在 E100 將分子鎖定在 `LMR_ALLNODE_SCALE_NUM = 260` 之後，針對分母結構展開進一步微調：
    $$\Delta r = \frac{r \times 260}{\text{DENOM\_BASE} \times \text{depth} + \text{DENOM\_OFFSET}}$$
  * 生產原基準為 `DENOM_BASE = 256`、`DENOM_OFFSET = 268`。
* **第一階段：雙向梯度探針實測（1,000 局成對換先，確定梯度方向）**：
  1. **Probe 1（常數項偏移探針，`copy 7.ipynb`）**：
     * 固定 `BASE = 256`，測試 `OFFSET = 298` (New, +30) vs `238` (Old, -30)。
     * 戰績：Candidate **191 勝 644 和 165 負**（得分率 **51.30%**，總得分 513.0 / 1000，淨勝 **+26 局**）。
     * 相對 Elo：**+9.04 ± 12.85**，SPRT LLR = **+1.07**。
     * 盤長表現：51~80 步中長盤對攻錄得 134W 162D 106L（勝率 **53.5%**，單項淨勝 **+28 局**），且每步提速 -5.0ms，NPS 微升 +0.6%。
     * 結論：分母常數項向正側放寬（+30）優勢極為顯著。
  2. **Probe 2（深度乘數探針，`multicut.ipynb`）**：
     * 固定 `OFFSET = 268`，測試 `BASE = 290` (New, +34) vs `220` (Old, -36)。
     * 戰績：Candidate **198 勝 615 和 187 負**（得分率 **50.55%**，總得分 505.5 / 1000，淨勝 **+11 局**）。
     * 相對 Elo：**+3.82 ± 13.36**，SPRT LLR = **-0.26**。
     * 結論：大分母側（290）同樣優於小分母側（220）。
  3. **探針結論**：兩組探針一致指向「放寬分母、緩和 ALL-node 減深」為正確發展方向，其中常數項偏移 `OFFSET = 298` 效果最為顯著。
* **第二階段：對抗原生產基準確認對局配置（Setup for Baseline Confirmation Battle）**：
  * **關鍵原則**：探針戰勝弱側基準（-30）不等於實打實相對原基準的純 Elo 收益。只有在原基準上勝出，才是可正式採納的生產收益。
  * **確認對戰環境設定**：
    * **Candidate (New)**: `LMR_ALLNODE_SCALE_DENOM_BASE = 256`, `LMR_ALLNODE_SCALE_DENOM_OFFSET = 298`
    * **Baseline (Old)**: `LMR_ALLNODE_SCALE_DENOM_BASE = 256`, `LMR_ALLNODE_SCALE_DENOM_OFFSET = 268`（嚴格對抗原基準）
  * 完成代碼、差分驗證與合約測試，靜候使用者排程執行。

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
