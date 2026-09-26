# 搜尋 / 評估戰役報告（2026-07-15～16）

**範圍：** Classical HCE 搜尋樹效率、剪枝係數、結構 A/B、LMR 表係數、eval 對照、New vs Old 固定節點對戰。  
**對戰控制：** `--nodes 200000`（每步固定節點）。  
**統計快照：** 雙對戰約 **66 局** 時完整統計；其後續打至約 72–74 局後**手動中止**（SPRT 仍 Continue、方向無優勢）。

---

## 1. 一句話結論

| 問題 | 答案 |
| :--- | :--- |
| 剪枝 **margin 座標下降** 還有空間嗎？ | **幾乎沒有**（R1–R3 平台） |
| 抄 **SF11 NMP/ProbCut 公式** 能瘦樹嗎？ | **不能**（pass↓ 或更肥） |
| 更軟 **LMR 表 scale=70/85** 有 Elo 嗎？ | **對戰未顯示**；深度變差 |
| **scale=100** 比 85/70 好嗎？ | **深度最接近 Old**；棋力仍略遜、不顯著 |
| 同 depth 仍比 SF11 肥？ | **是**（~2.5×）；主因偏 **eval 地形**，非再調 LMR |
| **今天該接納什麼？** | **凍結 R2 剪枝鎖 + FP_MULT=210；LMR 表 scale 回到 100；不採 70/85** |
| **下一步？** | **Eval（Material/PST → Mobility）**，搜尋旋鈕先停 |

---

## 2. 今日與連日實驗時間線

```text
[背景凍結] SF11-cap 搜尋 + P1 bad-cap 1536 + 曾同步 classical_old
     │
     ├─ 消融 / 掃描：LMR/LMP 主省節點；razor/FP 弱；ProbCut 關了略瘦
     ├─ Dual：LMR_BASE=768 + FP_MULT=160（後被 full 座標下降改寫）
     ├─ R1 full@d10 座標下降：pass 448→458；LMR_BASE 回 512 等
     ├─ R2：僅 FP_MULT 190→210；3s half 246/250
     ├─ R3 結構 LMR tune：10/10 鎖原值（平台）
     ├─ S1/S2 結構 A/B（NMP/ProbCut）：baseline 最優，全部不採
     ├─ E2 LMR 表 scale：70 勝出（pass+6，nodes+25%）→ 曾寫入生產 70
     ├─ E1 eval vs SF11：ratio 散布大，非純 ×0.78
     ├─ 對戰 scale=70 vs Old @200k nodes：depth 明顯↓（先前報告）
     └─ 雙對戰 scale=100 / 85 vs Old @200k：無 Elo、無深度優勢 → 中止
```

---

## 3. 實驗結果總表

### 3.1 閘道定義（本戰役）

| 閘道 | 用途 |
| :--- | :--- |
| full 500 puzzles @ **depth 10** | 主閘：pass + sum_nodes |
| 3s half（250 題 stride 2） | 副閘：timed；有噪（±2～4） |
| node_bench / ablation d10 | 樹肥與剪枝消融 |
| New vs Old **nodes=200000** | 棋力 + 實戰深度 |

選法（座標下降）：**先 pass**；pass−1 僅當 nodes≤0.95×；否則 min nodes。

### 3.2 座標下降 R1 / R2 / R3

| 輪次 | 內容 | full@d10 | 3s half | 採納 |
| :--- | :--- | ---: | ---: | :--- |
| **R1** | 14 參數 ×~5 值，從 dual 出發 | **458/500 · 20.5M** | **246/250** | **是**（鎖 LMR_BASE=512, hist=210, LMP=110, FP 160/190…） |
| **R2** | 反向細格 + QS_SEE | 458 · 20.43M（0.994×） | 246/250 | **僅 FP_MULT→210** |
| **R3** | 結構 LMR 5 槽 + volume re-lock | 458 · 20.43M，**無參數移動** | 242/250（噪） | **否（維持 R2）** |

**解讀：** 在 R2 鎖附近，**1D 細掃與結構 LMR 邊角已無正邊際**。

### 3.3 結構 A/B（S1 NMP / S2 ProbCut）— 一次 JIT，12 packs

| 結果 | 判定 |
| :--- | :--- |
| **baseline（R2 鎖）458 · 1.00×** | **採納** |
| non-PV NMP | nodes 0.986× 但 pass 451 → **不採** |
| SF NMP gate/R 全開 | pass↓ 且 nodes↑ → **不採** |
| ProbCut 改/關 | 皆差 → **不採** |

**解讀：** 在現行 HCE 上，**不能靠抄 SF11 NMP/ProbCut 公式瘦樹**。

### 3.4 E2：LMR 表係數 scale

| scale% | full@d10 pass | nodes | 對戰 depth vs Old | 對戰 Elo |
| ---: | ---: | ---: | :--- | :--- |
| **70** | **464** | **+25%** | 明顯變淺（先前 200k 局） | 無正證 |
| **85** | （未單獨 full 重測） | 介於中間 | **−0.66 ply**（66 局） | **−32** CI 含 0 |
| **100** | 458 | 基準 | **−0.22 ply**（66 局） | **−21** CI 含 0 |
| 115–130 | pass 掉 | 更瘦 | — | — |

3s half（scale=70）：244/250（持平級）。

**解讀：** puzzle 上「更軟 LMR」可換 pass；**固定 nodes 對戰上換不到 Elo，還掉深度** → **E2 生產採納 70 應撤回**。

### 3.5 E1：與 SF11 靜態評估對照（詳細）

| 指標 | 數值 |
| :--- | :--- |
| mean our/sf（white cp） | **~1.55**（期望純尺度 ~0.78） |
| ratio 範圍 | **0.72 – 2.78** |
| mean \|diff\| | **~23 cp**（root 已如此） |
| 最大殘差 bundle | **Material+pieces/PST** → **Mobility** → King safety → Threats |

**解讀：** 「HCE 結構像 SF」≠「分數地形像 SF」。剪枝觸發吃的是 **eval 相對 α/β**，故 **eval 失配可解釋同 depth 樹肥與搜尋公式抄不準**。

### 3.6 與 SF11 樹肥（背景，仍有效）

| 測試 | 約略比值 |
| :--- | ---: |
| node_bench d10 SUM | **~2.5×** SF11 |
| NPS | **~2× 慢** |
| 消融最大槓桿 | 關 LMR ~2.9×、關 LMP ~2.5× |

排序正常（cut_first ~91%）→ **不是 ordering 崩壞**。

---

## 4. 兩場對戰結果（2026-07-16，統計 @66 局）

**設定：** New（`CHESS_LMR_TABLE_SCALE`=100 或 85）vs Old（`classical_old` / `main_old.py`），`--nodes 200000`，每場 2 worker，雙場並行、先 4 New JIT 再 4 Old JIT。  
**棋譜：**

- `tournament_results_lmr100.pgn`
- `tournament_results_lmr85.pgn`

**報告目錄：**

- `stats_lmr100/` · `stats_lmr85/`（game + search + 圖）

### 4.1 棋力

| | New100 | New85 |
| :--- | ---: | ---: |
| 局數（統計快照） | 66 | 66 |
| W–D–L | 14–34–18 | 14–32–20 |
| 得分率 | **47.0%** | **45.5%** |
| Elo vs Old | **−21** [−81, +38] | **−32** [−94, +29] |
| 顯著？ | **否** | **否** |
| 中止時約 | 74 局 · 35.0–39.0 · LLR≈−0.27 | 72 局 · 33.5–38.5 · LLR≈−0.34 |

兩者皆 **略遜 Old、未達 SPRT 邊界**；再打完 100 局也 **不值得** 為「可能翻盤」而燒 CPU（方向穩定偏負）。

### 4.2 搜尋深度（主統計，排除將殺與和棋深搜）

| | New100 | Old | New85 | Old |
| :--- | ---: | ---: | ---: | ---: |
| mean depth | **12.34** | 12.55 | **11.83** | 12.49 |
| Δ | **−0.22**（顯著但小） | | **−0.66**（顯著） | |
| depth≥14 | 20.8% / 23.6% | | 15.6% / 23.7% | |
| NPS | ≈ −1% | | ≈ −1% | |

- **100 ≈ Old 深度**（實務可接受的微小差）。  
- **85 明顯更淺**（與「較軟 LMR → 樹肥」一致）。  
- NPS 無實質差 → **不是變慢，是每層更貴**。

### 4.3 與 scale=70 對戰（同日稍早）的銜接

先前 New（生產 scale=70）@200k：mean depth **11.43 vs Old 12.22（−0.79）**，depth≥14 約半。  
與 85/100 形成單調趨勢：

```text
scale 70  → depth 最差
scale 85  → 中等
scale 100 → 最接近 Old
```

---

## 5. 該怎麼接納（決策表）

### 5.1 採納（保留）

| 項目 | 理由 |
| :--- | :--- |
| **R2 剪枝鎖**（LMR_BASE=512, hist=210, LMP_SCALE=110, FP_BASE=160, **FP_MULT=210**, bad-cap=1536, 其餘 SF11-cap 預設） | full@d10 458、3s 246；R3/S 結構掃不動 |
| **P1 LMR_BAD_CAPTURE=1536** | puzzle 正邊際 |
| **diag `tune[]` 基礎設施** | 一次 JIT 多配置，避免重編譯 |
| **SF11-cap 政策** | 勿比 SF11 更兇的激進 pack（歷史回歸） |
| **E1 結論：eval 地形是主線** | 有數據（ratio 散布、term MAD） |
| **「LMR 表 scale 軟化換 Elo」假設** | **否定**（對戰證據） |

### 5.2 不採納 / 撤回

| 項目 | 理由 |
| :--- | :--- |
| **LMR_TABLE_SCALE=70（E2）** | puzzle pass↑ 但對戰 depth↓、無 Elo → **生產改回 100** |
| **scale=85** | 深度與得分皆不如 100 |
| S1/S2 NMP·ProbCut 對齊 SF | 全輸 baseline |
| R3 結構 LMR 邊角改值 | 全鎖原值 |
| P0/P2–P6/soft LMR packs | 已歷史回退 |
| Dual LMR_BASE=768 | 已被 R1 以 pass 覆寫回 512 |

### 5.3 生產常數凍結建議（「今日終點」）

```text
LMR_TABLE_SCALE_PERCENT = 100     # 撤回 70
LMR_NOT_IMP_NUM         = 194
FP_BASE                 = 160
FP_MULTIPLIER           = 210
LMR_BASE_OFFSET         = 512
LMR_HISTORY_SCALE       = 210
LMP_SCALE_PERCENT       = 110
LMR_BAD_CAPTURE_BONUS   = 1536
# 結構 TUNE_* modes = 0
# 其餘 R2 / SF11-cap 預設
```

**對戰 Old：** 若要以「今日凍結」為新 Old，需另跑 `tools/sync_classical_old.py --sync`（僅在確認 New 優於當前 Old 時；**本次 scale 實驗未證明優於 Old，不必為 70/85 同步**）。

---

## 6. 因果圖（為什麼改了很多卻減不了節點、也沒 Elo）

```text
                    ┌─────────────────────┐
                    │  HCE 結構「像」SF11  │
                    │  但 root static 比值  │
                    │  0.7～2.8（E1）      │
                    └──────────┬──────────┘
                               │
         eval 驅動剪枝觸發集合 ≠ SF
                               │
         ┌─────────────────────┼─────────────────────┐
         ▼                     ▼                     ▼
   同 depth 樹 ~2.5×      抄 SF 公式失效         軟 LMR 可換
   （node_bench）         （S1/S2）              puzzle pass
                                                    │
                                                    ▼
                                          固定 nodes 深度↓
                                          對戰無 Elo（今日）
```

**Margin 掃不動** = 已在 pass 邊界局部最優。  
**要 Elo / 真瘦樹** = 改 **eval 決策邊界** 或 **NPS**，不是再 ±LMR 常數。

---

## 7. 下一步（優先序）

### P0 — 立刻（收尾）

1. **`LMR_TABLE_SCALE_PERCENT = 100`** 寫回 constants / engine 預設。  
2. 本報告歸檔；`SEARCH_ANALYSIS.md` 標註 E2 對戰否定。  
3. **搜尋超參凍結**：不再開 margin / LMR scale / NMP 抄公式大掃。

### P1 — 下一戰役主線：Eval

依 E1 MAD：

| 順位 | 目標 | 方法 |
| ---: | :--- | :--- |
| 1 | **Material + PST** | 對 SF term；縮小 root \|our−sf\| 與 ratio 散布 |
| 2 | **Mobility** | 同上 |
| 3 | King safety / Threats | 次之 |

**閘道建議：**  
- full@d10 pass ≥ **458**（勿為瘦樹犧牲戰術底線）  
- 3s half ≥ **244**  
- New vs Old @200k：**得分率 ≥ 0.50 或 Elo CI 下界 > −10** 才考慮 sync old  
- 可選：node_bench d10 對 SF 比值是否下降（次要）

### P2 — 中長期

| 項目 | 說明 |
| :--- | :--- |
| NPS / Numba 熱路徑 | 同 nodes 更深；與 eval 正交 |
| 搜尋結構再啟 | **僅在 eval 對齊後** 重試 NMP 觸發率；勿現在重抄 |
| SF18 節奏 | 仍是長線目標；本凍結仍是 HCE 階段 |

### 明確不做

- 再一輪 LMR/RFP/razor 座標下降  
- 再掃 scale 70/85/115「碰運氣對戰」  
- 並行雙場長對戰而無明確假設  

---

## 8. 產物索引

| 類型 | 路徑 |
| :--- | :--- |
| 本報告 | `tournament_analysis/REPORT_2026-07-15_16_search_campaign.md` |
| 搜尋分析累積 | `chess_engine/classical/SEARCH_ANALYSIS.md` §9 |
| R1/R2/R3 | `scratch/coord_descent_*` |
| S1/S2 | `scratch/struct_ab_once.*` |
| E2 | `scratch/lmr_table_sweep.*` |
| E1 | `scratch/eval_compare_sf11_detail.*` |
| 雙對戰 log | `tournament_analysis/dual_lmr_match.log` |
| PGN | `tournament_results_lmr100.pgn` · `tournament_results_lmr85.pgn` |
| 統計 | `stats_lmr100/` · `stats_lmr85/` |

---

## 9. 給決策者的三句話

1. **今天搜尋側主線已證完：再調 LMR/結構公式換不到固定節點 Elo。**  
2. **接納 R2 剪枝鎖 + FP_MULT=210；撤回 LMR scale=70，回到 100。**  
3. **下一步只做 eval 對齊（Material/PST → Mobility），用同一套 puzzle + 200k 對戰閘驗證。**
