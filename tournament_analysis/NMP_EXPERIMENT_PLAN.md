# NMP 節點效率診斷與實驗計畫

**日期：** 2026-07-16  
**範圍：** Classical HCE 搜尋 — Null Move Pruning  
**相關文件：** `SEARCH_ANALYSIS.md` §9.3、`struct_ab_once_results.json`、`REPORT_2026-07-15_16_search_campaign.md`  
**狀態：** 診斷同意（有條件）+ 實驗計畫待執行（**尚未改生產常數**）

---

## 0. 執行摘要

| 問題 | 結論 |
| :--- | :--- |
| NMP 只貢獻 ~15% 節點削減，是否偏低？ | **是**（相對強引擎中局常見 25–40% 貢獻）。消融「關 NMP → ~1.18× nodes」已證實。 |
| 主因是「gate 太嚴」嗎？ | **部分正確**，但**不充分**。Legacy 確實偏難觸發；然而**單純放寬 gate / 抄 SF11 已在 S1 結構 A/B 中失敗**。 |
| 該不該立刻把 `TUNE_NMP_*` 設成 SF 全開？ | **不該。** `S1_full_raw` 曾使 nodes **↑6.7%** 且 pass 掉。 |
| 真正目標是什麼？ | 不是「提高 nmp_cut 計數」 alone，而是 **在 pass/Elo 不掉的前提下** 提高 NMP 的**淨節點節省**（cut 收益 − null/verify 開銷）。 |
| 建議路線 | **Phase 0 漏斗診斷 → Phase 1 單軸中間態 → Phase 2 聯合微調 → Phase 3 品質/Elo 閘**；禁止再盲抄 SF 二元開關。 |

**一句話：** 同意「NMP 效率偏低、值得專案優化」；**不同意**「放寬 gate + 加深 reduction 就能到 25%+」——現有資料顯示那條路已經踩過坑。

---

## 1. 對診斷意見的逐條回應

### 1.1 同意的部分

1. **~15% 貢獻偏低**  
   消融定義：`nodes(no_nmp) / nodes(baseline) ≈ 1.18` ⇒ 節省約 `1 − 1/1.18 ≈ 15%`。  
   與 LMR（關 → ~2.9×）、LMP（關 → ~2.5×）相比，NMP 是**第三大**但仍明顯偏軟的槓桿。

2. **Legacy gate 在中深層偏嚴**（`TUNE_NMP_GATE=0` 生產預設）  

   ```text
   threshold = beta - 14*d - 45*improving + 200
   d=10, !improving → static >= beta + 60   # 難
   d=10, improving  → static >= beta + 15
   ```

   SF11 風格約為：

   ```text
   margin = -32*d + 292 - 30*improving
   need: static >= beta  AND  static >= beta + margin
   d=10, !improving → static >= beta 且 static >= beta - 28  → 實質 static >= beta
   ```

3. **多重守衛壓縮觸發面**  
   `cut_node only`（scope=0）、`side_non_pawn ≥ 2`、前步非 null、`abs(beta) < VALUE_KNOWN_WIN`、in-check 禁止、exclusion 禁止——這些大多**正確且必要**，但會把「可省節點」的樣本縮小。  
   另：部分 FEN（如 pos3）歷史上 `nmp_cut=0`，支持**觸發不均**。

4. **Verification 有成本**  
   `NMP_VERIFICATION_DEPTH = 8`，且 verification 深度為 `depth - R` 的 full-window 再搜。若 verify 多數通過，理論上可延後 verify 門檻；若多數失敗，則應**先修 gate/R** 而不是關 verify。

5. **Eval 準確性決定 NMP 上限**  
   E1 對照：root static 相對 SF11 的 ratio 散布 **0.72–2.78**，mean ~1.55，**不是純 ×0.78 尺度**。  
   任何「eval 對 β」的剪枝在錯地形上都會系統性偏觸發或假觸發。

### 1.2 必須修正 / 反對的部分

#### A. 「直接設 TUNE_NMP_GATE=1/2、TUNE_NMP_R=1、SCOPE=1」已被實驗否決

`scratch/struct_ab_once.py`（full 500 @ d10，baseline = 458 pass · 20.43M nodes）：

| pack | scope | gate | R | pass | nodes×base | 解讀 |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| **baseline** | 0 | 0 | 0 | **458** | **1.000** | **生產最優** |
| S1a_nonpv | 1 | 0 | 0 | 451 | **0.986** | 唯一下 nodes；**pass −7** |
| S1b_gate_sf | 0 | 1 | 0 | 450 | **1.039** | 放寬 gate → **更肥** |
| S1b_gate_sf78 | 0 | 2 | 0 | 448 | 1.032 | 同上 |
| S1c_r_sf | 0 | 0 | 1 | 446 | **1.045** | 改 SF R（更淺）→ 更肥 |
| S1_full_raw | 1 | 1 | 1 | 455 | **1.067** | 全開 **最肥級** |
| S1_full_sf78 | 1 | 2 | 1 | 454 | 1.057 | 同上 |

**關鍵機制（與直覺相反）：**

| 現象 | 機制 |
| :--- | :--- |
| 放寬 gate → nodes↑ | 更多 null-search **嘗試**，但 fail-high 率不夠 → **白付 null 子樹** |
| 改 SF R（R≈5@d10）→ nodes↑ | Legacy R（`7+d//3`≈10@d10）**觸發後砍更深**；SF R 更溫和 → 單次 cut 省得少；若 cut 率沒同比例上升則淨虧 |
| non-PV 略瘦但掉 pass | ALL-node NMP 多砍到需要的防禦/戰術線 |

因此：**問題不是「沒抄 SF」**，而是 **「難觸發 + 觸發後猛砍」在現行 HCE 上恰好比「易觸發 + 溫和砍」更划算**（至少在 puzzle@d10 指標上）。  
優化方向應是 **找中間態**，不是把三個 tune 旋鈕打到 SF。

#### B. 常數方向建議有誤

原文：「降低 `NMP_SF_MARGIN_*`（例如把 `NMP_SF_MARGIN_BASE` **調高** 50–100）」——兩句互相矛盾，且：

```text
_sf_margin = -32*d + BASE - 30*imp
threshold  = beta + _sf_margin
```

**提高 BASE → margin 變大（更不負 / 更正）→ threshold 更高 → gate 更嚴。**  
若目標是放寬 SF gate，應 **降低 BASE** 或 **提高 DEPTH_COEF**（使 margin 更負），不是提高 BASE。

#### C. 「提高 reduction 到 3–5+ plies」對 legacy 不適用

| 公式 | d=10 近似 R（不計 eval_margin） |
| :--- | :--- |
| Legacy `7 + d//3` | **10** |
| SF `(854+68d)//258` | **≈5.9** |

現行 legacy **已經比 SF 砍得深**。再加深 R 的空間有限，且更容易在 zugzwang / 戰術位假 cut（需靠 verify 或更嚴 gate 補償）。

#### D. 「nmp_cut↑ 即成功」是錯誤代理指標

`DIAG_NMP_CUT` 只計**最終通過 verify 的 cut**。可能出現：

- nmp_cut↑ 但 null 嘗試暴增 → **總 nodes 仍↑**
- nmp_cut 持平但 R 更深 → nodes↓（高風險）
- nmp_cut↓ 但每次 cut 更深 / 更準 → nodes↓ 且 Elo↑

**成功定義必須是：nodes↓ + pass 不掉 +（後段）Elo↑。**

#### E. 15% → 25% 的期望值需重標

在 **HCE 地形與 SF 不同**、且 S1 全開已失敗的前提下，短期可達成的現實目標建議：

| 階段 | 節點節省（相對 baseline，固定 d10） | 備註 |
| :--- | :--- | :--- |
| 地板 | 維持 ≥15%（不退化） | 消融 no_nmp ≥1.18× |
| 近程目標 | **18–22%**（no_nmp ≥1.22–1.28×） | 不傷 pass |
| 遠程期望 | 22–28% | 可能需 eval 改善配合 |
| 不宜承諾 | 「一定 25–40% 像 SF 中局」 | SF 有不同 eval / 整體剪枝生態 |

---

## 2. 現行實作快照（生產預設）

| 旋鈕 | 生產值 | 意義 |
| :--- | :--- | :--- |
| `TUNE_NMP_SCOPE` | **0** | 僅 `cut_node` |
| `TUNE_NMP_GATE` | **0** | Legacy 門檻 |
| `TUNE_NMP_R` | **0** | Legacy `7+d//3` |
| `NMP_MIN_DEPTH` | 3 | |
| `NMP_MIN_SIDE_NON_PAWNS` | 2 | |
| `NMP_VERIFICATION_DEPTH` | 8 | d≥8 做 verification |
| `LOW_MATERIAL_PRUNING_PIECE_COUNT` | 7 | 低子力剪枝守衛（NMP 本身看 non-pawn） |

**已具備的實驗基建：**

- `SearchContext.enable_nmp` 消融
- `tune[TUNE_NMP_SCOPE/GATE/R]` 執行期切換（**免重 JIT 換公式族**）
- `diag_stats[DIAG_NMP_CUT]` + `format_search_diagnostics`
- 腳本：`scratch/node_ablation.py`、`scratch/struct_ab_once.py`、`scratch/node_bench.py`、`tests/test_puzzle` 路徑、`tools/tournament.py`

**缺口（本計畫要補）：** NMP **漏斗計數**（嘗試 / gate 過 / null fail-high / verify fail / cut），目前只有末端 `NMP_CUT`。

---

## 3. 成功指標與閘道（不可妥協）

沿用戰役標準並加 NMP 專用 KPI。

### 3.1 品質閘（硬）

| 閘 | 標準 | 失敗處理 |
| :--- | :--- | :--- |
| **G1** full 500 @ depth 10 | pass ≥ **456**（baseline 458，允許 −2） | 丟棄該配置 |
| **G2** full@d10 nodes | ≤ **0.95×** baseline（≤ ~19.4M）為「明確勝利」；≤ **0.98×** 為「弱勝利」可進 Phase 2 | 僅 nodes↑ → 丟棄 |
| **G3** 3s half（stride2, 250） | pass ≥ **240**（允許 timed 噪 ±2–4） | 再跑一次確認 |
| **G4** 戰術安全 | 固定題庫上 bestmove 與 baseline 重大分歧率有記錄（不必 0） | 分歧大 → 人工抽樣 |

### 3.2 效率 KPI（主）

在 **同一 10-FEN node_bench 套件**（與 `node_ablation` 同步）上：

| KPI | 定義 | 目標 |
| :--- | :--- | :--- |
| **NMP_SAVE** | `1 − nodes_base / nodes_no_nmp` | 由 ~15% → **≥18%**，理想 ≥22% |
| **NMP_OVERHEAD** | `(null_try_nodes + verify_nodes) / total`（需新 diag） | 相對 cut 節省下降 |
| **NMP_HIT** | `nmp_cut / nmp_try` | 診斷用；不單獨當目標 |
| **VERIFY_FAIL** | `verify_fail / verify_try` | 過高 → gate 太松或 R 太深 |
| **lmr_research%** | 既有 | 不爆炸（&lt;12%） |
| **cut_first%** | 既有 | 不崩（&gt;85%） |

### 3.3 Elo 閘（Phase 3，需使用者核准長跑）

| 設定 | 內容 |
| :--- | :--- |
| 對局 | New(NMP 候選) vs Old / 或 vs 凍結 baseline 二進位 |
| 控制 | **`--nodes 200000`** 固定節點（與 7/16 戰役一致）+ 可選 **固定時間** 一場 |
| 規模 | 至少 **100 局** 或 SPRT（α=β=0.05，elo0=0, elo1=5 等級） |
| 通過 | Elo ≥ **0** 且 95% CI 下界 &gt; −15；或 SPRT accept H1 |
| 附加 | mean depth 不應明顯變淺（|Δdepth| &lt; 0.3 於 nodes 控制） |

**重要：** 固定 nodes 下「更激進 NMP」若只換來錯誤剪枝，**Elo 會掉且 depth 假高**；固定時間下才直接對應實戰。兩種都要看。

---

## 4. Phase 0 — 漏斗診斷（1–2 天，必先做）

**目標：** 在改任何公式前，量化「錢花在哪」。

### 4.1 新增 diag 槽位（建議）

在 `constants.py` / `diag_stats` 擴充（名稱示意）：

| 索引名 | 何時 +1 |
| :--- | :--- |
| `DIAG_NMP_ELIGIBLE` | 通過 scope/check/npm/beta 等硬守衛、準備算 gate |
| `DIAG_NMP_GATE_PASS` | `nmp_gate_ok` |
| `DIAG_NMP_NULL_TRY` | 實際 `make_null_move` + 子搜尋 |
| `DIAG_NMP_NULL_FH` | `null_move_score >= beta` |
| `DIAG_NMP_VERIFY_TRY` | 進入 verification |
| `DIAG_NMP_VERIFY_FAIL` | verify 後 `verify_score < beta` |
| `DIAG_NMP_CUT` | 既有：最終 cut |

可選（更重）：累加 **null 子樹 nodes**、**verify 子樹 nodes**（兩個 `uint64` 累加器）。

### 4.2 基準報告表（固定 d=10，10 FEN + full 500 各出一張）

| 局面類型 | eligible | gate% | null_fh% | verify_fail% | cut | nmp_save 估計 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| start / mid / tact / end / pos3 | … | … | … | … | … | … |

**決策樹：**

```text
gate% 很低（<15% eligible）
  → 優先中間態放寬 gate（Phase 1A），不要先改 R

gate% 中高，但 null_fh% 很低
  → eval/β 不一致或 R 太淺導致 null 搜完仍 <β；查 correction/dissonance

null_fh% 高，verify_fail% 高
  → 假優勢：gate 太松或 R 太深；應收緊 gate 或加深 R 前先修 eval 信任

null_fh% 高，verify_fail% 低，但 NMP_SAVE 仍低
  → cut 太少發生在「貴」的節點 / scope 太窄；試有限 non-PV 或調 R
  → 或 verification 本身太貴（Phase 1C）
```

### 4.3 產物

- 腳本：`scratch/nmp_funnel_bench.py`（或擴充 `node_ablation.py`）
- 輸出：`scratch/nmp_funnel_baseline.json` + 文字摘要
- **凍結 baseline 數字** 後再進 Phase 1

**通過條件：** 漏斗數字可重現（兩次跑相對誤差 &lt;3%）。

---

## 5. Phase 1 — 單軸實驗（一次 JIT 原則）

**原則：**

1. **一次 Numba 編譯**，用 `tune[]` + 少量新 tune 槽覆蓋參數族。  
2. **每次只動一軸**（scope / gate / R / verify），避免 S1_full 式混淆。  
3. 禁止重測已失敗的極端點（`S1_full_raw/sf78`）除非漏斗顯示環境已變（例如 eval 大改後）。

### 5.1 軸 A — Gate 中間態（最高優先）

**假設：** Legacy（β+60@d10）太嚴，SF（≈β）太松且與 R 不匹配；存在 **β + k(d)** 的最優帶。

建議參數化（取代二元 gate mode）：

```text
# 建議新公式（可 tune）
nmp_threshold = beta - G_DEPTH * depth - G_IMP * improving + G_BASE
# 可選：nmp_need_ge_beta 作為 0/1 旗標
```

**掃描網格（示例，可依 Phase 0 收斂）：**

| 參數 | 候選 |
| :--- | :--- |
| 有效 d=10 !imp 要求（static−β） | +60（legacy）、+40、+20、+0、−20 |
| improving 放寬 | 維持 45 或 30/20 |
| `need_ge_beta` | off（legacy）/ on（SF） |

對應到現有常數：調 `NMP_LEGACY_*` 或新增 `TUNE_NMP_G_BASE/DEPTH/IMP`。

**期望：** 找到 pass≥456 且 nodes 最小的 gate；gate% 上升但 verify_fail 不爆。

### 5.2 軸 B — Reduction R 中間態

**假設：** Legacy R 過深限制了「敢觸發」；SF R 過淺浪費 cut。中間 R 配合中間 gate 才有協同。

| d | Legacy R | SF R | 建議掃描 R≈ |
| ---: | ---: | ---: | :--- |
| 6 | 9 | ~4.9 | 5–8 |
| 8 | 9 | ~5.4 | 6–9 |
| 10 | 10 | ~5.9 | 6–9 |
| 12 | 11 | ~6.4 | 7–10 |

參數化建議：

```text
R = R_BASE + depth // R_DIV + min((static-beta)//EM_DIV, EM_MAX)
# 例：R_BASE ∈ {5,6,7,8}；R_DIV ∈ {3,4,5}
```

**注意：** 軸 B **必須在軸 A 的優勝 gate 上**再掃，或做小的 2D 網格（見 Phase 2），避免單獨改 R 重現 `S1c_r_sf` 失敗。

### 5.3 軸 C — Verification 成本

| 變體 | 描述 | 風險 |
| :--- | :--- | :--- |
| C0 | 現狀：verify @ d≥8，depth=`d−R` | 基準 |
| C1 | verify @ d≥10 或 12 | 低（略激進） |
| C2 | verify 用 null-window `(β−1,β)` 且 depth 再 −1 | 中 |
| C3 | 暫時 `if False` 關 verify（**僅診斷**） | 高；只能量測成本，不可生產 |

**判定：**

- C3 相對 C0：若 nodes 大降且 pass 幾乎不掉 → verify 是主負擔，做 C1/C2  
- 若 C3 pass 大掉 → verify 在救假 cut，應 **收緊 gate** 而非關 verify

### 5.4 軸 D — Scope（謹慎）

| 變體 | 描述 | 已知 |
| :--- | :--- | :--- |
| D0 | cut_node only | 生產 |
| D1 | all non-PV | S1a：−1.4% nodes，pass 451 |
| D2 | non-PV 且 `depth ≥ 6` 或 `static ≥ beta` | **未測**；可能保留省節點、降戰術傷害 |
| D3 | non-PV 僅 `improving` | 未測 |

**D1 不直接採納**；D2/D3 是本計畫相對 S1 的新增價值。

### 5.5 軸 E — 材料 / 端局守衛（次優先）

| 變體 | 描述 |
| :--- | :--- |
| E0 | `side_non_pawn ≥ 2` |
| E1 | `≥ 1`（更激進，zugzwang 風險） |
| E2 | 端局（總非兵子少）強制禁用 NMP 更嚴 |

只在中局 funnel 健康後再動。

### 5.6 Phase 1 執行清單

```text
[ ] 0. 實作 funnel diag + nmp_funnel_bench
[ ] 1. 跑 baseline funnel，存檔
[ ] 2. 軸 A gate 網格（~6–10 configs），full@d10
[ ] 3. 取 A* = 過 G1 且 nodes 最優
[ ] 4. 在 A* 上跑軸 B R 網格
[ ] 5. 軸 C verify（C1/C2；C3 僅 funnel）
[ ] 6. 軸 D2/D3 小規模
[ ] 7. 產出 Phase1 排行榜 JSON + 淘汰表
```

**預估工作量：** 診斷 0.5–1 日；單軸掃描 1–2 日（一次 JIT + 每 pack ~30–40s full@d10）。

---

## 6. Phase 2 — 聯合微調（小網格 / 座標下降）

僅對 Phase 1 存活的 **2–3 個旋鈕** 做聯合：

```text
例如：G_BASE × R_BASE × VERIFY_DEPTH
每維 3 值 → 27 點；或座標下降 2 輪
```

**選法（與 R1–R3 一致）：**

1. 先 **pass**  
2. pass 相同（或 −1 且 nodes≤0.95×）→ **min nodes**  
3. nodes 接近時 → 較高 `NMP_SAVE`、較低 `VERIFY_FAIL`

**禁止：** 再次 14 參數大海掃；NMP 外的 LMR/FP 鎖死在目前生產值。

**產出：** `NMP_CANDIDATE_V1` 參數組 + funnel 對照表。

---

## 7. Phase 3 — 品質與 Elo 確認

### 7.1 輕量（可自動）

```bash
python -m tests.test_search
python scratch/node_ablation.py --depth 10          # NMP_SAVE
python scratch/nmp_funnel_bench.py                 # 漏斗
# full 500 @ d10（沿用 puzzle 腳本 / struct_ab 路徑）
```

### 7.2 中量（建議執行）

- 3s half ×2（降噪）  
- node_bench 與 SF11 比值是否從 ~2.5× 改善（期望小步，例如 2.5→2.3）

### 7.3 重量（需使用者明確同意）

```text
tools/tournament.py
  New(NMP_CANDIDATE_V1) vs classical_old
  --nodes 200000
  ≥100 games 或 SPRT
可選第二場：固定 movetime 對照實戰深度
```

**只有 Phase 3 通過才寫入生產 `constants.py` 預設 / 關閉實驗 mode。**

---

## 8. 實作設計要點（給實作者）

### 8.1 保持正確性約束（`search-constraints.md`）

- in-check **禁止** NMP  
- root（ply==0）不靠 NMP 決定唯一 root move 的合法性（NMP 在內部節點）  
- mate / known-win 不把 unproven 分當 cut（現有 `is_win` / `is_decisive` cap 保留）  
- null move 後 GHI：確認 halfmove / key 還原（現有 save/restore 路徑已存在，回歸測即可）

### 8.2 建議新增的 runtime tune（一次編譯）

| tune 槽 | 含義 |
| :--- | :--- |
| 既有 SCOPE/GATE/R | 保留相容 |
| `TUNE_NMP_G_BASE` | 中間 gate base（覆蓋 legacy 常數） |
| `TUNE_NMP_G_DEPTH` | depth 係數 |
| `TUNE_NMP_R_BASE` / `R_DIV` | 中間 R |
| `TUNE_NMP_VERIFY_DEPTH` | verify 門檻（8/10/12/999=off 診斷） |
| `TUNE_NMP_SCOPE_MIN_DEPTH` | non-PV 時最小 depth（D2） |

生產預設必須 **bit-exact 重現現有 legacy 行為**。

### 8.3 腳本規劃

| 腳本 | 職責 |
| :--- | :--- |
| `scratch/nmp_funnel_bench.py` | 漏斗 + 10 FEN + 可選 full |
| `scratch/nmp_grid_once.py` | 一次 JIT，讀 JSON 網格，輸出 pass/nodes/funnel |
| 擴充 `format_search_diagnostics` | 印出 funnel 比例 |

### 8.4 與 eval 工作的介面

若 Phase 1–2 **過不了 G1+G2 聯合**（無配置同時瘦樹且保 pass）：

→ **停止搜尋側硬調 NMP**，優先 E1 已指出的 **Material/PST → Mobility** eval 對齊。  
理由：S1 已證明「SF 公式 + 錯地形 = 更肥」；NMP 的理論天花板由 static 相對 β 的可靠性決定。

```text
搜尋 NMP 優化  ┐
               ├─ 若漏斗顯示 gate 合理但 null_fh 系統性偏低
eval 地形修復  ┘     → 轉 eval 專案，NMP 參數凍結
```

---

## 9. 時程與決策節點

```text
Day 0–1   Phase 0 漏斗；凍結 baseline KPI
Day 1–3   Phase 1 軸 A→B→C→D
Day 3–4   Phase 2 小聯合網格 → NMP_CANDIDATE_V1
Day 4     輕/中量閘 G1–G3
Day 5+    Phase 3 對戰（需核准）→ 生產合併或丟棄
```

**決策節點：**

| 節點 | 問題 | 是 | 否 |
| :--- | :--- | :--- | :--- |
| N0 | funnel 顯示 verify 白燒？ | 做 C1/C2 | 跳過 verify 軸 |
| N1 | 軸 A 有 pass 安全的 nodes↓？ | 進 B | 轉 eval / 結束 NMP 大改 |
| N2 | A×B 達 NMP_SAVE≥18% 且 G1？ | 進 Phase 3 | 接受局部最優或 eval |
| N3 | Elo 閘過？ | **合併生產** | 回退，保留 diag 基建 |

---

## 10. 風險與反模式

| 反模式 | 為何有害 |
| :--- | :--- |
| 只看 nmp_cut 數字 | 可與 nodes 反向 |
| 一次改 scope+gate+R | S1_full 已證明混淆且失敗 |
| 關 verification 當生產 | 戰術假 cut → Elo 地雷 |
| 用 puzzle pass 唯一導向 | 過軟 LMR 曾 +pass/+nodes 無 Elo |
| 承諾「一定到 SF 25–40%」 | 生態不同；應以 NMP_SAVE 相對自身 baseline |
| 與 LMR 同時大改 | 無法歸因 |
| 把 BASE 調高當「放寬」 | 公式上是收緊 |

---

## 11. 對原建議的最終對照表

| 原建議 | 本計畫立場 |
| :--- | :--- |
| TUNE_NMP_GATE=1/2 | **拒絕作為預設**；僅作掃描對照點（已知差） |
| TUNE_NMP_R=1 | **拒絕單獨採用**；改中間 R 網格 |
| TUNE_NMP_SCOPE=1 | **拒絕全開**；改 D2/D3 條件 non-PV |
| 提高 SF MARGIN_BASE | **方向錯誤**；放寬應降 BASE / 調 DEPTH 係數 |
| 加深 reduction 到 3–5+ | Legacy 已 ~10@d10；應探索 **6–9 中間帶** 而非更深 |
| verify 提到 10–12 | **採納為軸 C 候選** |
| 暫時關 VERIFY 測 nodes | **僅診斷**（C3），不進生產 |
| 印 diag nmp_cut | **採納並擴充為完整漏斗** |
| 關其他剪枝只留 NMP | **採納為消融協議**（已有 `node_ablation`） |
| 目標 25%+ 貢獻 | **改為階梯目標 18→22→28**；25% 不保證 |

---

## 12. 建議的立即下一步（若批准本計畫）

1. **實作 Phase 0 funnel diag**（最小侵入：+6 個計數器 + format 輸出）。  
2. 跑 `nmp_funnel_bench` + 既有 `node_ablation`，把「15%」拆成漏斗故事。  
3. 依 N0/N1 決策進軸 A 中間 gate 網格（**不要**先改生產預設）。  
4. 任何 full@d10 / 對戰長跑前，維持 workflow：輕測可跑；tournament 需你點頭。

---

## 附錄 A — 既有關鍵數據（勿重複踩坑）

**消融（SEARCH_ANALYSIS）：** 關 NMP → ~1.18×；關 LMR → ~2.89×；關 LMP → ~2.45×。  

**結構 A/B：** 見 §1.2 表；baseline 全勝。  

**戰役結論（2026-07-16）：** 剪枝 margin 座標下降已平台；搜尋側下一刀應是**有診斷的結構中間態**或 **eval**，不是再 ±256 LMR。

## 附錄 B — 公式速查

**Legacy gate：** `static >= beta - 14d - 45·imp + 200`  

**SF-style gate：** `static >= beta` 且 `static >= beta + (-32d + 292 - 30·imp)`（gate=2 時 margin×78/100）  

**Legacy R：** `7 + d//3 + clamp((static-β)//150, 0..3)`  

**SF R：** `(854 + 68d)//258 + same margin`  

**Verification：** `d >= 8` 時以 `max(1, d-R)`、窗口 `(β-1, β)` 重搜且暫時 `enable_nmp=False`。

---

*本文件為實驗計畫，不修改生產預設。合併條件見 §3 與 §9 N3。*
