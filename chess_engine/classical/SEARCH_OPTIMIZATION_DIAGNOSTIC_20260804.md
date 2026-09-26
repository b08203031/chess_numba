# Classical 搜尋優化診斷與交接報告（2026-08-04）

## 1. 範圍與結論

本報告先完成 **搜尋診斷、SF11/SF18 對照與交接**，後續曾依報告實作 P0、P1、P2 與 P3。
使用者的固定節點新舊引擎對戰沒有接納 P0/P1/P3 組合，因此 production 已回到同步前管線：
qsearch 在 scorer 後才做 evasion legality、main quiet legality 在 shallow SEE 後執行、main
in-check picker 不壓縮 captures/quiets。P0/P1/P3 的 classifier、counter 與報表欄位全部保留為
`diag_enabled` 的 shadow diagnostics，不改 production 候選集合、排序或剪枝。使用者在回退後
重新對戰確認新版占優，並已自行同步 `classical_old`；本輪 P4 沒有再次同步；
`tests/test_search.py` 已依使用者允許改成固定深度 13，但正式搜尋診斷統一採用
`hist_diag_bench.py --nodes 30000`。

結論按優先順序如下：

1. **P0 評分前穩定壓縮已由對戰否決並回退。** 修改前的 30k baseline 在
   `make_move` 前排除 5,439,840 個可證明非法候選，但它們仍先支付 capture/history 評分與
   selection sort 成本；診斷模式仍在 scorer 前 shadow 掃描整個 pseudo-evasion 集，歷史實驗的
   完整掃描計數為 12,234,379。production 不壓縮 buffer；兩者計數定義不同，詳見 P0 驗收段。
2. **P1 production 前移與 geometry reuse 已回退，診斷保留。** n30k 1000 盤面曾顯示
   quiet SEE 前可證明非法率為 4.44%，因此將既有 pin/king legality fast path 前移；同時把
   `_search` 已計算的五個 check bitboard 傳入 `score_quiets()`，移除每次 scorer 的重建。
   固定節點總量、完成深度、cut-first 與 LMR re-search 維持基線，主 quiet SEE 呼叫平均由
   51,567.6 降至 49,626.5（-3.77%），geometry rebuild 由 3,898.8 降至 0；但這些局部效率
   指標沒有轉化為對戰收益。production 已恢復 scorer 內 lazy geometry 與 SEE 後 legality；
   shadow counters 仍可重現成本漏斗。
3. **目前沒有證據支持先加重 LMR、改 NMP/ProbCut，或重做 history。** LMR re-search 率只有
   2.35%（戰術）與 4.84%（開局），cut-first 為 93.04%／89.54%；history 表幾乎沒有飽和，
   而這些方向先前已有負面 A/B／對戰紀錄。
4. **P2 aspiration diagnostics 已完成。** BASE 20 參考線合計每個完成迭代平均 1.356 次額外窗口重搜，
   失敗窗口約占 41.66% nodes；先前曾選定 BASE 30 作為穩定性／效率折衷，現因固定節點對戰結果回退至
   BASE 20。完整 A/B 與檔案見 §6 的 P2 段落。
5. P3 診斷確認 **main in-check picker 的非法候選**事件密度很高。正式 n30k
   1000 盤面平均每盤有 1,383.8 個 in-check main nodes，幾何檢查的 25,485 個候選中
   22,117（84.06%）可在評分前證明非法；但 stable captures/quiets compaction 未通過對戰，
   production 已回退。分類器現在只在 diagnostics context shadow 執行，EP 仍列為 fallback。
6. P3 同時確認 qsearch cap 與 TT deep verify 不應盲改：q cap 平均 352.5/盤，只有
   73.85% cap 事件帶有 pseudo forcing candidate，且 check-cap 為 0；TT verify 平均
   4.07/盤、reject 0.105%、saved-cut 32.40%，保留 SF18 同款保護。
7. 第二優先（P3 前的舊排序）是 **main search quiet SEE 的剩餘成本診斷**。1000 盤面
   在 30k nodes 共呼叫 main quiet SEE 51,613,220 次，其中 13,285,254 次實際造成略過。它是有效剪枝，不能
   直接移除；應先區分 pinned/king illegal、checking-route、history gate 與真正 SEE 失敗。
8. 目前瓶頸主要是 **QS 結構性 CPU 成本**，其次是 main quiet SEE 熱路徑；不像是 move ordering
   已失效或 history 表損壞。這次不宣稱 Elo 提升，任何改動仍須由固定深度、戰術 gate 與使用者
   的新舊引擎對戰決定。
9. **P4 decision-quality shadow diagnostics 已完成。** 500+500 n30k 顯示排序來源分層有效：
   TT／good capture／good quiet 的 cutoff precision 為 71.16%／56.16%／24.09%，bad capture／
   bad quiet 只有 2.72%／0.70%。LMR reduced fail-high 經較深 zero-window 驗證後有 37.36% 被
   推翻，而現有 butterfly／LMR statScore 對 keep 與 reject 的正號率只相差 8.43／7.38 個百分點。
   因此先把 SF11/SF18 類型的 post-LMR continuation-history feedback 列為單軸候選；後續
   production A/B 已完成，但四個候選都未通過 n30k，詳見 §10.4，現行 history 未改。

## 2. 現行基線與文件漂移

### 2.1 實際生效的搜尋設定

以 `chess_engine/classical/constants.py` 為準：

| 項目 | 現值 | 判讀 |
| :--- | :--- | :--- |
| NMP scope | `NMP_SCOPE_MODE = 0` | 只在 cut node；不是舊文件所寫的 scope 3 |
| LMR | scale 100、base offset 460、history scale 150 | 已經歷座標下降與對戰；不可只看 nodes 再加重 |
| LMP | scale 105 | 觸發量很大，但目前屬已驗證基線 |
| ProbCut | depth ≥ 5、margin 150 | 開局集觸發稀少；直接抄 SF 公式曾失敗 |
| qsearch | depth cap 5（check 最多 10）、SEE -80 | P3 已加入 non-check/check cap 與 pseudo forcing hit counter；目前不改 cap |
| TT cutoff | rule50 < 96、deep verify depth 7 | P3 try/pass/reject/saved-cut 診斷仍支持保留 SF18 護欄 |
| stop polling | mask 4095 | 固定 10k nodes 會有明顯 overshoot，適合看比率、不適合當精確 10k 樹大小 |
| 關閉的實驗 | multicut、alpha-raise reduction、follow-PV、capture futility | 都有回歸或不穩定紀錄，不應順手重開 |

原 `SEARCH_ANALYSIS.md` 的 NMP scope 3 敘述已過時：scope 3 的對戰結果不足以保留，現行值已回到
scope 0。本輪同步修正文檔，但不改常數。

### 2.2 已保留且不應拆除的正確性護欄

- root 不吃一般 TT cutoff，mate distance、repetition、50-move 與 ply 邊界處理保留。
- RFP、razor、NMP、ProbCut 等 node-level 靜態剪枝維持 non-PV / not-in-check / exclusion 等條件。
- decisive / known-win 分數不進入不安全的靜態剪枝或 depth reduction。
- checking quiet 不套一般 quiet `lmrDepth²` SEE gate；目前專用 route 是前一輪已驗證的保護。
- qsearch 的 double-check、single-check interposition/capture、pin ray 與 king destination 檢查保留；
  EP 等難以純靜態證明的案例仍必須經過 post-make legality。
- depth ≥ 7 的 TT child verification 是 SF18 也使用的保護，不是本引擎自行增加的多餘成本。

## 3. 診斷方法

### 3.1 快速固定深度 gate

執行：

```text
python scratch/node_bench.py --depth 10 --diag --puzzles --puzzle-depth 12 \
  --label sync-baseline-report-20260804
```

結果完整重現同步後基線：

| 指標 | 結果 |
| :--- | ---: |
| 13 positions d10 total nodes | 870,909 |
| main / qsearch nodes | 466,056 / 404,853（QS 46.49%） |
| cut-first / average cutoff move | 90.1% / 1.21 |
| LMR re-search | 5.1% |
| TT / NMP / RFP / razor / ProbCut cut | 37,617 / 2,219 / 31,982 / 5,433 / 1,151 |
| LMP / futility skips | 1,057,878 / 8,816 |
| tactical d12 | 11/11，44,532 nodes |
| 搜尋失敗 | 0 |

這組資料說明同步基線沒有退化，也顯示 move ordering 的首著 cutoff 仍健康。

### 3.2 使用者指定的混合診斷集

使用 `tools/hist_diag_bench.py`：

- 500 個戰術盤面；
- 500 個 opening／quiet middlegame 盤面；
- 每盤 `--nodes 10000`，4 個 worker；
- 每盤配置獨立 context，收集剪枝、SEE、QS、history 與表飽和診斷。

10k 與使用者追加的 30k 命令：

```text
python tools/hist_diag_bench.py --nodes 10000 --positions 500 -c 4 \
  --tag search_audit_20260804_n10k_fixed --json

python tools/hist_diag_bench.py --nodes 30000 --positions 500 -c 4 \
  --tag search_audit_20260804_n30k_fixed --json
```

本輪先修正該**診斷工具**的兩個過時問題，沒有改搜尋：

1. 已移除的 `HIST_AGE_SOFT_NUM` import 改為現行 `HISTORY_COMPAT_1A`，報告正確顯示
   `PHASE1A_DIV2`。
2. `iterative_deepening_search()` 回傳的 `result[2]` 是 main-search nodes；原工具錯把它當 total，
   導致 qnodes 可能大於 total、所有 per-1k 比率被放大。現改用
   `total_nodes = search_nodes + q_nodes`，並另外輸出 `search_nodes`。

權威基線輸出是帶 `_fixed` 的
[`n10k JSON`](../../archive/tournament_analysis_2026-08/hist_diag_search_audit_20260804_n10k_fixed.json) 與
[`n30k JSON`](../../archive/tournament_analysis_2026-08/hist_diag_search_audit_20260804_n30k_fixed.json)。P0 修改後的
正式輸出是 [`n30k P0 v2 JSON`](../../archive/tournament_analysis_2026-08/hist_diag_search_audit_20260804_n30k_p0_v2.json)，
另保留 [`n30k P0 v1 JSON`](../../archive/tournament_analysis_2026-08/hist_diag_search_audit_20260804_n30k_p0.json) 與
[`n30k P0 v1 repeat JSON`](../../archive/tournament_analysis_2026-08/hist_diag_search_audit_20260804_n30k_p0_repeat.json) 作為交叉檢查，另有同名文字報告；
不帶 `_fixed` 的第一次輸出分母錯誤，已刪除且禁止拿來比較。

固定 nodes 因 `STOP_CHECK_MASK=4095`、終局提早結束與 iteration 邊界會 overshoot，因此下表以
事件率與兩資料集的相對差異為主。4-worker 的 wall time / aggregate NPS 同時受 JIT、ThreadPool 與
Windows 排程污染，**不能**當引擎單執行緒 NPS 基線。

## 4. 500 + 500 診斷結果

以下 §4.1～§4.4 保留原始 10k 結果，§4.5 用 30k 驗證結論是否只是淺層假象。

### 4.1 樹形與 move ordering

| 指標 | 戰術 500 | 開局／安靜 500 | 合計 |
| :--- | ---: | ---: | ---: |
| total nodes | 5,755,985 | 5,897,836 | 11,653,821 |
| qsearch nodes | 1,872,873（32.54%） | 2,839,560（48.15%） | 4,712,433（40.44%） |
| 平均完成深度 | 9.72 | 6.12 | — |
| cut-first | 93.34% | 90.33% | 92.03% |
| LMR re-search / reduced search | 2.11% | 3.44% | 2.60% |

判讀：cut-first 高而 LMR re-search 低，未呈現「排序差導致大量錯誤縮減後重搜」的症狀。這也與
過往 LMR scale 70/85、alpha-raise reduction、follow-PV 等實驗沒有穩定收益相符。

### 4.2 qsearch 熱路徑

| 指標 | 戰術 500 | 開局／安靜 500 | 合計 |
| :--- | ---: | ---: | ---: |
| generated q moves | 3,908,466 | 6,193,386 | 10,101,852 |
| post/pre legality 判非法 | 1,137,450 | 856,423 | 1,993,873 |
| 既有 evasion prefilter 命中 | 1,129,495 | 845,694 | 1,975,189 |
| prefilter / illegal | 99.30% | 98.75% | 99.06% |
| prefilter / q moves | 28.90% | 13.65% | 19.55% |
| prefilter / 1k total nodes | 196.23 | 143.39 | 169.49 |

現行 `quiescence_search()` 的順序是：

```text
產生所有 pseudo moves
  → score_captures_with_tt_lazy（含 quiet-evasion history）
  → 建 checker / pin / evasion masks
  → selection sort 取下一手
  → 靜態 evasion prefilter
  → make/unmake fallback legality
```

因此上一輪的 prefilter 已省掉大量 make/unmake，卻還沒省掉被拒候選的評分與排序。尤其 quiet
evasion 的評分會讀取 piece-to、butterfly、pawn 及多層 continuation history；selection sort 又是
候選數平方型比較。把相同判定移到 scoring 前，理論上可在**不改變存活著法順序與搜尋樹**的前提
下降低 CPU 成本。

### 4.3 main search SEE 與剪枝分布

| 指標 | 戰術 500 | 開局／安靜 500 | 判讀 |
| :--- | ---: | ---: | :--- |
| main quiet SEE calls | 6,627,810 | 9,458,515 | 合計 1.61 千萬，是第二大明確熱點 |
| quiet SEE skips | 2,040,161（30.78%） | 2,054,046（21.72%） | 剪枝有效，不能直接刪除 |
| TT cut / 1k nodes | 91.28 | 48.66 | 戰術 transposition 較多 |
| NMP cut / 1k nodes | 5.67 | 1.48 | 不是目前最大節點來源 |
| RFP cut / 1k nodes | 59.44 | 21.81 | 活躍且有既有保護 |
| razor cut / 1k nodes | 13.05 | 6.11 | 戰術集較多 |
| ProbCut / 1k nodes | 3.42 | 0.31 | 開局集非常稀少 |
| LMP skips / 1k nodes | 659.82 | 1,149.27 | 最大樹剪枝來源，改動風險高 |
| futility skips / 1k nodes | 29.23 | 5.75 | 次要 |
| history prune / 1k nodes | 1.90 | 2.96 | 目前不是主瓶頸 |

main search 已在 node 前段計算 own/their king、check squares 與 blockers，但 `score_quiets()` 又為排序
重建一部分 check geometry；另外 pinned/king pseudo-legal 候選可能先付出 shallow SEE，再到後段合法性
檢查。這些都值得量測與重用，但比 QS pre-score compaction 更容易碰到 gives-check、EP、discovered
attack 與 pin 語義，因此排在第二階段。

### 4.4 history 是否出錯

表飽和率是每盤診斷的平均值：

| history 表 | 戰術 500 | 開局／安靜 500 |
| :--- | ---: | ---: |
| piece-to | 0.1029% | 0.0154% |
| butterfly | 0.0265% | 0.0050% |
| capture | 0.0451% | 0.0113% |
| continuation | 0.000089% | 0.000005% |
| pawn | 0.000066% | 0.000019% |

沒有全表飽和、符號爆炸或更新停滯的證據。清除記憶體後偶爾走出更佳著法，較合理的解釋是：

- history 是跨節點／跨 `go` 的非對稱 move-ordering 先驗；清除後同分候選的排序與 LMR 路徑會改變；
- TT、counter move、low-ply/continuation history 一起改變，可能在固定時間／固定 nodes 下到達不同分支；
- 這是有限資源搜尋的路徑依賴，不等同 history index 寫錯；
- 只有能重現「特定更新造成錯誤索引、非法著、表極端飽和或固定深度分數不一致」時，才應判成 bug。

現行 Phase 1a 每個 root 將表值除 2，UCI context 也已改為跨手持久化並在 `ucinewgame` 清理。過去 full
history reform / soft-age 實驗沒有通過，因此本輪建議保留。若要繼續研究，先增加「乾淨／暖 history
的固定深度 PV 分歧」記錄，而不是再改公式。

### 4.5 30k nodes 深層複驗

使用者指出 10k 的 opening median depth 只有 6，可能低估深層機制；因此同一批 500+500 以 30k
重跑。結果如下：

| 指標 | 10k 合計 | 30k 合計 | 判讀 |
| :--- | ---: | ---: | :--- |
| total nodes | 11,653,821 | 31,612,003 | 完整 1000/1000 |
| qsearch nodes | 40.44% | 39.51% | QS 佔比穩定 |
| puzzle depth mean / median | 9.72 / 8 | 12.92 / 10 | 30k 確實更深；含少量已解而回報 depth 64 的題目 |
| opening depth mean / median | 6.12 / 6 | 7.91 / 8 | 深約 2 ply，但仍不是 depth 13 強度測試 |
| cut-first | 92.03% | 91.53% | 排序仍健康 |
| LMR re-search | 2.59% | 3.19% | 隨深度合理上升，未顯示過度縮減 |
| q evasion prefilter | 1,975,189 | 5,439,840 | 30k 仍是百萬級熱點 |
| prefilter / q moves | 19.55% | 19.67% | 幾乎不變，證明不是淺層偏差 |
| prefilter / 1k total nodes | 169.49 | 172.08 | 密度反而微升 |
| main quiet SEE calls | 16,086,325 | 51,613,220 | 深層第二熱點成立 |
| quiet SEE skip rate | 25.45% | 25.74% | 有效剪枝，不可移除 |
| LMP skips / 1k | 907.52 | 1,052.67 | 深層更活躍，仍是高風險調參軸 |
| history prune / 1k | 2.44 | 9.29 | 深層開始活躍，但遠低於 LMP |
| ProbCut / 1k | 1.84 | 2.83 | 取樣改善，仍非最大來源 |

分資料集看，30k 的 prefilter 仍占戰術 q moves 29.05%、開局 q moves 14.45%；每 1k total nodes
分別命中 184.69 與 159.86 次。P0 因此不只在 depth 6–7 有效，優先級維持第一。

30k 也補足了 10k 幾乎看不到的深層事件：

- NMP verification：4,202 次嘗試、7 次拒絕 null fail-high（10k 為 379/0）。這證明 verification
  護欄確實會救回少量節點，不能為了 NPS 移除；但 7/4,202 也不支持先重寫整套 NMP。
- singular：22,746 次嘗試、9,069 次 extension、607 次 double extension。深層有足夠樣本，未見
  搜尋失敗，但若未來調整必須單獨 A/B。
- history 平均飽和率：piece-to 0.1500%、butterfly 0.0398%、capture 0.0908%、continuation
  0.000232%、pawn 0.000126%。比 10k 提高但仍極低，不支持「history 表飽和造成 mistake」。

所以 30k **加強而非推翻**原診斷：P0 的事件密度幾乎完全重現；P1 quiet SEE 也持續成立；唯一
需要調整的是，NMP verification 與 singular 不再被視為「未取樣」，而是已有樣本且應保留現行護欄。

## 5. SF11 / SF18 對照後的優化清單

| 區域 | 本引擎現況 | SF11 / SF18 可參考處 | 建議 |
| :--- | :--- | :--- | :--- |
| qsearch move pipeline | production 在 scorer／排序後做 legality；P0 classifier 只做 shadow diagnostics | 兩版皆以 staged MovePicker 逐步取著並配 legality/SEE gate | P0 production 已回退；cap forcing 只做診斷，不盲目放寬 |
| main move picker | staged captures/quiets、partial sort、SEE；in-check classifier 只做 shadow diagnostics | SF11/18 同樣重視 staged picking 與 check evasions | P3 compaction 已回退；EP/未知 geometry 仍列為 fallback |
| aspiration | `20 + abs(score)/120`，失敗後固定擴窗 | SF11 用前分數尺度；SF18 再納入 score variance / mean-squared | 目前回退 BASE 20；保留 per-depth diagnostics |
| NMP | scope 0、legacy gate/R、verification | SF11/18 公式與 statScore/improving 結合更深 | 直接搬運已失敗；暫不改 |
| ProbCut | margin 150、觸發偏少 | SF18 有 improving 與 TT gate、raised beta | 先保留；不是本次主瓶頸 |
| LMR | table + history + node/move/context 修正 | SF11/18 公式更多但高度耦合 | re-search 健康且舊實驗負面，暫不加重 |
| LMP / futility | LMP 觸發量最大，有 checking/known-win 保護 | SF 也將 move count、history、SEE 耦合 | 高收益也高風險；只做單軸對戰實驗 |
| history | Phase 1a、數個表與 correction history | SF18 correction histories 更豐富 | 未見飽和；先診斷暖／冷分歧，不重構 |
| singular / double extension | 已實作且觸發受控 | SF11/18 皆有多條件 singular 邏輯 | 保留；不要與其他 pruning 同時改 |
| TT deep verify | depth 7 + rule50 96；P3 已有 try/pass/reject/saved-cut/skip | SF18 同樣有 child-TT direction verification；SF11 較簡單 | **保留保護**；reject 少但非零、saved-cut 有樣本 |
| qsearch depth cap | 非 check 5、check 10；P3 已有 cap-hit/pseudo-forcing counter | SF 更依賴 MAX_PLY 與 search semantics | check-cap 未命中；不改 cap，後續若改須獨立合法性 A/B |
| correction history | 已接到 static eval | SF18 使用更多 pawn/non-pawn/continuation correction | 屬搜尋↔評估交界；凍結 HCE 時只做診斷，不擴表 |

## 6. 調整後的優先級

### P0：qsearch 評分前 evasion stable compaction（production 已回退；診斷保留）

**價值：高；樹風險：低；Numba 風險：中。** 這是唯一同時具備百萬級實測事件、可保留搜尋語義、
且沒有歷史負面 A/B 的方向。其密度在 10k 與 30k 分別為 169.49／172.08 次每 1k nodes，已通過
不同搜尋深度的複驗。

以下是已完成但後來由對戰否決的歷史實作；現行 production 不再執行 1–5 的 buffer 壓縮／跳過：

1. qsearch 產生 moves 後、呼叫 `score_captures_with_tt_lazy()` 前，計算 side/king/checkers/pins。
2. 對 `moves[0:move_count]` 做 **in-place、stable** compaction，不配置新陣列：
   - in-check castling 直接拒絕；
   - king destination 受攻擊則拒絕；
   - double check 拒絕所有非王著；
   - single check 的 ordinary 非王著只保留 capture checker / interposition，且必須通過 pin ray；
   - EP 與無法靜態確定的邊角保留給 post-make fallback；
   - checker count 不一致或資訊不足時採保守保留。
3. 保持存活 move 的原始相對順序，之後只評分／排序 compacted count。
4. 既有 `DIAG_Q_ILLEGAL_SKIP` 與 `DIAG_Q_EVASION_PREFILTER` 對拒絕著各加一次，避免統計語義改變。
5. 第一版保留後段 legality branch 作 safety net；P0 repeat 已證明新版本 deterministic，後續再用
   n30k 量測是否值得消除重複判定。
6. 不同步 `classical_old`。

### P0 n30k 驗收結果

同一批 500 戰術 + 500 開局盤面、每盤 30k nodes、4 workers：

| 指標 | 修改前 `n30k_fixed` | P0 v2 `n30k_p0_v2` | 判讀 |
| :--- | ---: | ---: | :--- |
| total nodes | 31,612,003 | 31,620,299（+0.026%） | 量級穩定；固定節點邊界下有少量樹差異 |
| qsearch nodes | 12,491,325 | 12,514,334 | QS 比例僅小幅變化 |
| mean completed depth | 10.416 | 10.385 | 沒有深度提升證據 |
| cut-first | 91.529% | 91.548% | ordering 沒有崩壞 |
| LMR re-search | 3.193% | 3.194% | 非目標機制穩定 |
| q moves generated | 27,649,003 | 27,811,271 | 固定節點路徑略有分歧 |
| q illegal diagnostic | 5,485,760 | 12,278,285 | 計數語義改變，見下方說明 |
| q evasion prefilter | 5,439,840 | 12,234,379 | 不可直接與舊值比較 |
| main quiet SEE | 51,613,220 | 51,567,607 | 約 -0.09% |
| mean per-position elapsed | 0.08631 s | 0.08686 s | v2 4-worker 結果未證明 NPS 提升 |

`DIAG_Q_EVASION_PREFILTER` 在舊版只會統計「排序後實際走到、且尚未 beta cut 的候選」；P0
則在 scorer 前完整掃描該 qsearch node 的 pseudo evasions，因此所有會被 beta cut 遮蔽的非法候選
也會被計入。這是計數定義的改變，不是非法著增加。P0 repeat 的 1000 個 FEN 結果在 total nodes、
qnodes、prefilter、SEE 與每盤 nodes 上完全一致，表示新版本本身是 deterministic。v2 只再移除
已由 compaction 證明的 ordinary/king 後段重複 legality check；v1/v2 的 1000 個 FEN raw metrics
完全一致，因此當時曾保留 v2 作為工作版本；後續已依對戰結果回退。

這些數據只能確認歷史版本「前置壓縮曾接線且合法性契約通過」，不能證明 Elo。後續固定節點對戰
判定組合版本較弱，因此 P0 production 已回退；`DIAG_Q_EVASION_PREFILTER` 現在只做 scorer 前
shadow 分類，不改 `moves`、`move_count` 或後段 baseline legality。

### P1：main quiet SEE／legality／check geometry（production 已回退；診斷保留）

先新增 opt-in counters，再依事件密度做過兩個局部優化；下列第 2、3 項是歷史實作，現已回退：

1. `DIAG_MAIN_QUIET_*` 分類 quiet SEE 前的 pinned/king 非法、SEE 通過後的 legality 拒絕、
   fallback 次數；`DIAG_MAIN_GEOMETRY_NODE` 與 `DIAG_MAIN_SCORE_GEOMETRY_REBUILD` 量測
   `_search` 與 `score_quiets()` 的重複 geometry。
2. `_search` 已計算的 `check_sq_pawn/knight/bishop/rook/queen` 直接傳給 `score_quiets()`，
   不再於 scorer 重建 slider attacks。這不改 move score、順序或 discovered-check 判定。
3. 在 shallow quiet SEE 前沿用相同 pin/king fast path。可證明非法的 ordinary quiet 直接跳過 SEE；
   EP、castling、captures、in-check 與 post-make safety net 維持原路徑。

### P1 n30k 驗收

同一批 500 戰術 + 500 開局盤面、每盤 30k nodes、4 workers：

| 指標 | P1 診斷前移前 | P1 前移後 | 判讀 |
| :--- | ---: | ---: | ---: |
| total nodes / position | 31,620.299 | 31,620.307 | 固定節點邊界僅 0.008 節點差 |
| mean completed depth | 10.385 | 10.385 | 無深度退化 |
| cut-first / LMR re-search | 90.8692% / 4.0841% | 90.8692% / 4.0842% | 非目標排序穩定 |
| quiet SEE calls / position | 51,567.607 | 49,626.518 | **-3.77%** |
| quiet SEE skips / position | 13,259.850 | 11,805.520 | 非法候選不再支付 SEE |
| pre-SEE illegal rate | 4.4376% | 4.4378% | 事件分類一致 |
| score geometry rebuild / position | 3,898.775 | 0 | **完整移除重建** |
| fallback rejects | 0 | 0 | safety fallback 未失效 |

P1 診斷與前移後的完整輸出分別為
`hist_diag_search_audit_20260804_n30k_p1_diag.json`、
`hist_diag_search_audit_20260804_n30k_p1_geometry.json` 與最終驗收的
`hist_diag_search_audit_20260804_n30k_p1_final.json`。4-worker wall time 受排程與 JIT
影響，只作參考；後續對戰未接納組合版本。現行 `score_quiets()` 恢復 scorer 內 lazy geometry，
quiet legality 恢復到 shallow SEE 後；pre-SEE illegal 與 geometry rebuild counters 仍為 opt-in
shadow diagnostics。若要再壓 main SEE，應另開單軸微基準，不與 LMR/NMP/history 合併。

### P2：aspiration diagnostics 與窗口 A/B（診斷完成；公式維持基線）

P2 已新增按完成深度保存的 `DIAG_ASPIRATION_*` counter：

1. aspiration iteration 數；
2. fail-low、fail-high 與額外 re-search 數；
3. 每深度最大 `delta`；
4. 失敗窗口所消耗的 main+qsearch nodes（`wasted_nodes`）。

`wasted_nodes` 是重搜開銷的診斷名稱，不宣稱那些 nodes 完全無用；它們仍可能暖化 TT。資料由
`collect_hist_diag_dict()` 同時輸出 aggregate rate 與 `aspiration_by_depth`，且只在
`diag_enabled=True` 時寫入，生產 UCI 搜尋不支付 counter array write。

#### P2 n30k BASE 20 參考線診斷

輸出：`archive/tournament_analysis_2026-08/hist_diag_search_audit_20260804_n30k_p2_diag.json`。
同一批 500 戰術 + 500 開局、每盤 30k nodes、4 workers：

| 指標 | 戰術 | 開局 | 合計 |
| :--- | ---: | ---: | ---: |
| aspiration iterations / position | 12.824 | 7.914 | 10.369 |
| fail-low / iteration | 0.386 | 0.324 | 0.355 |
| fail-high / iteration | 1.611 | 0.391 | 1.001 |
| re-search / iteration | 1.997 | 0.715 | 1.356 |
| wasted nodes / total nodes | 51.70% | 31.62% | 41.66% |
| mean max delta | 192.2 | 45.4 | 118.8 |

事件確實偏高，且主要集中在 depth 8–12，不只是 mate/decisive score。因此進行兩個單軸候選：

* 先前候選 `ASPIRATION_WINDOW_BASE=30`：完整 n30k 合計 re-search/iteration `1.356→0.971`、
  wasted nodes `41.66%→30.21%`；平均完成深度 `10.385→10.294`，開局集 `7.914→7.910`。
  單執行緒 100+100 n30k 的平均 elapsed 約 `0.0597→0.0599 s/position`（本次量測近似持平）。
* `ASPIRATION_WINDOW_BASE 20→40`：完整 n30k 將合計 re-search/iteration 降至 0.714、wasted
  nodes 降至 22.70%，但平均完成深度 `10.385→10.294`（-0.09 ply），單執行緒 200 盤面
  elapsed 約慢 1%；不接納。
* SF18-inspired score-square selective widening：只對高上一層 cp 放寬，開局盤面幾乎不變；
  100+100 單執行緒 re-search 僅降約 8.9%，elapsed 約慢 3.0%，不接納。

先前 BASE 30 已通過完整 n30k 與單執行緒 smoke，但固定節點對戰顯示 New 的完成深度與戰績不如基線，
因此目前 production 已回退至 `ASPIRATION_WINDOW_BASE=20`，並保留原有 fail-high adjusted-depth 政策。
本次回退只代表搜尋窗口設定變更，不宣稱 Elo 提升，仍須由同條件新舊引擎對戰確認。若日後要再測，應以
固定 depth/單執行緒重複量測，而不是以 `wasted_nodes` 單一指標決策。
回退後 30k 回歸（20 tactical + 20 opening，4 workers）為 40/40 完成、無 failure；合計平均完成深度
28.05，re-search/iteration 0.489，輸出為 `archive/tournament_analysis_2026-08/hist_diag_p2_base20_recheck_20260804.json`。

### P3：qsearch cap、main in-check、TT verify 成本診斷

P3 先加 opt-in counters，再依 20+20 n30k 事件密度決定唯一的生產修改；diagnostics disabled
時不寫 counter array。三條漏斗的正式 500+500 n30k 輸出為
[`P3 JSON`](../../archive/tournament_analysis_2026-08/hist_diag_p3_final_20260804_n30k.json) 與
[`P3 文字報告`](../../archive/tournament_analysis_2026-08/hist_diag_p3_final_20260804_n30k.txt)。

#### P3 正式 n30k 結果

同一批 500 戰術 + 500 opening/quiet、每盤 30k nodes、4 workers，1,000/1,000 完成，沒有
零節點或 engine failure；wall time 94.9997 秒只作流程參考：

| 指標 | 戰術 500 | 開局／安靜 500 | 合計 | 判讀 |
| :--- | ---: | ---: | ---: | :--- |
| total nodes / position | 31,187.1 | 31,947.1 | 31,567.1 | 固定節點 overshoot 正常 |
| mean completed depth | 12.668 | 7.896 | 10.282 | 相對 BASE30 約 -0.012 ply，無實質退化 |
| main in-check nodes | 2,294.4 | 473.2 | 1,383.8 | 事件量足夠 |
| picker geometry candidates | 40,958.4 | 10,011.6 | 25,485.0 | pre-score 分類總量 |
| provably illegal before scoring | 35,647.2 (82.89%) | 8,587.0 (85.24%) | 22,117.1 (84.06%) | 高價值壓縮軸 |
| q non-check cap / position | 132.9 | 572.1 | 352.5 | 約 1% nodes；check-cap = 0 |
| q cap with pseudo forcing | 58.13% | 89.56% | 73.85% | 只是假合法上界，不足以改 cap |
| TT verify try / position | 8.02 | 0.12 | 4.07 | SF18 保護確實被觸發 |
| TT verify reject / try | 0.209% | 0% | 0.105% | 少量但非零，不能移除 |
| TT verify saved-cut / try | 58.19% | 6.60% | 32.40% | 保留 cutoff 的有效樣本 |

#### P3 歷史實作與回退決策

1. **main in-check picker 前 stable compaction（已回退）**：歷史版本由 `_search` 計算 current king、checker
   count、single-check capture/interposition targets 與 pinned mask；`get_next_move()` 在
   `score_captures_lazy()`、`score_quiets()` 前分別壓縮 capture/quiet segment，保持每個 segment
   的 generator 相對順序。double check 僅保留 king move，single check 僅保留吃將軍子／擋線，
   king destination 仍以 occupancy-without-king 攻擊測試；EP、checker geometry 不可用的情形保留
   原有 post-make fallback。TT move 也先經同一分類，非法時回到一般 generation。這與 P0 的
   幾何判定同源，沒有加入新的 heuristic prune。固定節點對戰沒有接納 P0/P1/P3 組合，因此
   現行 `get_next_move()` 只在 `diag_enabled` 時分類並累加 funnel，不再拒絕 TT move 或壓縮
   capture/quiet segment。
2. **qsearch cap 不改**：cap counter 的 forcing 欄位是 pseudo capture/evasion 候選數，沒有支付
   make/unmake 來證明合法；正式資料沒有任何 check-cap hit，也沒有 false-negative 證據。直接把
   cap 5/10 放寬會擴張樹且不能由此診斷宣稱改善。
3. **TT deep verify 不改**：`try/pass/reject/saved-cut/skip` 均已接線。reject 雖少，但非零；
   saved-cut 樣本足以證明 child-TT direction verification 正在保護 cutoff。維持 SF18 同款
   depth-7、rule50 與 child direction guard。
4. **非目標機制穩定**：P3 前後完整 n30k 合計平均深度 10.294→10.282、total nodes
   31,591.9→31,567.1；沒有改 HCE、LMR、NMP、ProbCut、history 或 aspiration BASE。這種固定
   nodes 的微小差異不作 Elo/NPS 宣稱，最終仍由使用者新舊引擎對戰決定。

### 暫不執行

- 再縮 LMR、scale 70/85、alpha-raise reduction、follow-PV。
- full history reform、soft-age、無證據增加更多 history 表。
- 直接搬 SF11/SF18 NMP 或 ProbCut 公式。
- capture futility、multicut。
- 同時改多個 margin，或只用固定 nodes 深度判定 Elo。

## 7. P0/P1/P3 production 回退後的測試與後續順序

1. P0 qsearch compaction、P1 geometry reuse／legality-before-SEE、P3 main in-check compaction 已從
   production 回退；P2 diagnostics、P0/P1/P3 counters、q cap 與 TT verify diagnostics 保留。
   沒有改 HCE、LMR/NMP/ProbCut/history 參數；此回退當時沒有同步 `classical_old`，使用者已在
   後續重新對戰確認新版占優後自行同步。
2. `tests/test_search_core_contracts.py` + `tests/test_history.py` 現為 **22/22 通過**，包含 P0/P1/P3
   shadow-only、P4 decision-quality shadow-only、不得壓縮 production buffer、TT legacy cutoff 與
   baseline legality 順序契約。
3. 保留這些 legality FEN 契約：double check only king、單滑子將軍的吃將軍子／擋將、pinned
   interposer、king destination attacked、EP discovered-check fallback。
4. 正式搜尋診斷固定使用：

   ```text
   python tools/hist_diag_bench.py --nodes 30000 --positions 500 -c 4 \
     --tag p3_final_20260804_n30k --json
   ```

   比較時使用 raw sums；4-worker wall time 僅作參考，不能單獨宣稱 NPS。
5. P3 歷史正式 n30k 已確認 1000/1000 完成、無 engine failure，main in-check pre-illegal 約
   84.06%；此事件密度保留作診斷證據，不再視為 production 接納依據。
6. P2 aspiration diagnostics 已通過 smoke 與完整 n30k；BASE 40、score-square 兩個窗口候選均未通過
   單執行緒速度／深度接納。BASE 30 曾被選用，現已因實戰結果回退 BASE 20；`aspiration_by_depth` 保留在
   P2 JSON，後續可針對單一深度或 score band 做更小的 A/B。
7. `tests/test_search.py --engine new` 已完成 162/162 個固定 depth 13 位置，平均深度與中位數
   皆為 13，沒有 `NO_MOVE` 或 early-stop；後續正式搜尋比較仍以 hist_diag n30k 為主。
8. P3 後續只保留單軸：若要改 qsearch cap，必須先增加合法 forcing-line 分類與 fixed-depth gate；
   TT verify 不應因省 NPS 移除。不要與 NMP/LMR/history 同時修改，也不要直接搬 SF11/SF18 公式。

### Production 回退 n30k 驗收（2026-08-04）

輸出：`archive/tournament_analysis_2026-08/hist_diag_p013_prod_rollback_20260804_n30k.json`／`.txt`。同一批
500 戰術 + 500 opening/quiet、每盤 30k nodes、4 workers，1000/1000 完成且沒有 engine
failure。與同步前 `hist_diag_search_audit_20260804_n30k_fixed.json` 逐盤比較：

| 契約 | 結果 |
| :--- | :--- |
| 每盤 total nodes | 1000/1000 完全一致 |
| 每盤 qsearch nodes | 1000/1000 完全一致 |
| 每盤完成深度 | 1000/1000 完全一致 |
| 合計平均 nodes / depth | 31,612.003 / 10.416，與 baseline 完全一致 |
| cut-first / LMR re-search | 90.8469% / 4.0720%，與 baseline 完全一致 |
| P1 scorer geometry rebuild | 3,906.653/盤，證明 production lazy local geometry 已恢復 |
| P3 shadow classifier | 25,816.902 candidates/盤、84.268% pre-illegal，診斷仍有資料 |

這是 production 語義精確還原的證據，不是 Elo 宣稱。診斷模式仍支付 shadow classifier 成本，
因此其 wall time 不可當作 `diag_enabled=False` 的 production NPS。

## 8. 接納標準

P0/P1/P2/P3/P4 只有同時滿足以下條件才值得進入對戰：

- n30k 500+500 全部完成，沒有 engine failure；
- P0 repeat 在同一資料集的每盤 nodes 與 aggregate counters 完全一致；
- legality / core contracts 全通過；
- P1 n30k 前後搜尋樹與深度穩定，geometry rebuild 為 0，quiet SEE 呼叫量下降；這是歷史微基準，
  已被後續對戰的否決覆蓋；
- P2 aspiration diagnostics 完成 per-depth fail/re-search/wasted nodes；窗口公式候選需另通過單執行緒
  elapsed、固定 depth 與對戰，不能只因 re-search 下降就接納；
- P3 main in-check compaction 即使在 500+500 n30k 無 engine failure，仍須由對戰接納；本輪已否決
  並回退，qsearch cap/TT verify 維持只診斷不改語義；
- fixed-depth 13 的 `test_search.py --engine new` 162/162 通過；
- NPS 要以同一執行緒設定、同一資料集的重複 n30k 量測判定，不能以單次 4-worker wall time 宣稱；
- `classical_old` 保持使用者剛同步的基線；
- 最終棋力只由使用者的新舊引擎對戰接納，不以單次 NPS 或 nodes 代替 Elo。

## 9. 本輪檔案範圍

- 修正：`tools/hist_diag_bench.py`（過時 import、node denominator、P2 aspiration summary）。
- 修改：`chess_engine/classical/search.py`（回退 P0 qsearch compaction、P1 quiet legality-before-SEE、P3 main picker 前壓縮；保留 P0/P1/P2/P3 shadow diagnostics、q cap 與 TT verify 診斷）。
- 修改：`chess_engine/classical/search_heuristics.py`（回退 P1 geometry 傳入，恢復 `score_quiets` 內 lazy geometry；保留 rebuild counter）。
- 修改：`chess_engine/classical/constants.py`（P1/P2/P3 diagnostics counters；production aspiration BASE=20）。
- 修改：`tests/test_search_core_contracts.py`（P0/P1/P3 shadow-only 與 baseline production 順序、P2/P3 contracts）。
- 修改：`tests/test_search.py`（每步 fixed depth 13）。
- 新增：本報告與 500+500 的 10k／30k baseline、P0 v1/v2、P1 diagnostics/geometry/legality、P3 q cap/main in-check/TT verify，以及 P0/P1/P3 production 回退驗收輸出。
- 更新：`README.md` 文件導覽、`SEARCH_ANALYSIS.md` 的 NMP 現況與 P3 狀態、本輪 P0/P3 狀態與報告連結。
- 本輪沒有再次同步使用者已更新的 `classical_old`；未修改 HCE 或其他 pruning 常數。

## 10. P4：move ordering、LMR 驗證與 post-LMR history 診斷

P4 的目的，是把「每節點品質」拆成可以否證的事件，而不是再用總 NPS 或平均完成深度猜原因。
所有 counter 只在 `diag_enabled=True` 執行；沒有改 move score、picker stage、reduction、re-search
條件或 history update。正式輸出：
[`P4 JSON`](../../archive/tournament_analysis_2026-08/hist_diag_p4_decision_quality_diag_20260804_n30k.json) 與
[`P4 文字報告`](../../archive/tournament_analysis_2026-08/hist_diag_p4_decision_quality_diag_20260804_n30k.txt)。

### 10.1 診斷定義

- move source：TT、good capture、good quiet、bad capture、bad quiet、other；記錄 try、beta cutoff
  與 precision。`other=0` 且 source cutoff coverage=100%，驗證 stage classifier 沒漏算。
- quiet shortcut：killer1、killer2、countermove 的 try/cut；quiet cutoff 在「實際進入搜尋的 quiet」
  中的 rank（`quiet_moves_tried_count`），以及 cutoff 前已支付的 node-work proxy。
- LMR：按實際 reduction 1／2／3+ ply 與 quiet/capture 分桶，追蹤 reduced fail-high；再記錄
  verification keep、reject、deeper、shallower 與因 adjusted depth 不大於 reduced depth 而未驗證。
- post-LMR shadow sample：quiet reduced fail-high 經驗證仍成立記 bonus，遭推翻記 malus；只觀察
  當下 butterfly 與既有 LMR statScore 的符號，不實際呼叫 `update_continuation_history()`。
- `hist_diag_bench.py` 的 aggregate percentage 改由 raw count 總和重算；避免事件很少的盤面與
  事件很多的盤面獲得相同權重。舊報告的 90.8469% cut-first／4.0720% LMR re-search 是逐盤
  percentage 的算術平均；同一搜尋資料改用 raw-weighted 定義為 91.5291%／3.1930%。

### 10.2 正式 500+500 n30k 結果

1000/1000 完成、無 engine failure。與
`hist_diag_p013_prod_rollback_20260804_n30k.json` 逐盤比較：total nodes、search nodes、qsearch
nodes、完成 depth 四個欄位全部 **1000/1000 完全一致**。合計平均仍為 31,612.003 nodes、
depth 10.416；這證明 P4 是 shadow-only，不是棋力或 NPS 優化。

| 指標（raw-weighted） | 戰術 500 | 開局／安靜 500 | 合計 1000 |
| :--- | ---: | ---: | ---: |
| cut-first | 93.039% | 89.539% | 91.529% |
| TT cutoff precision | 72.377% | 68.902% | 71.155% |
| good capture precision | 58.024% | 54.565% | 56.162% |
| good quiet precision | 22.020% | 27.002% | 24.086% |
| bad capture precision | 2.281% | 3.021% | 2.715% |
| bad quiet precision | 0.643% | 0.777% | 0.701% |
| killer1 / killer2 / counter precision | 65.25 / 41.66 / 41.92% | 50.31 / 27.91 / 31.87% | 57.39 / 34.70 / 38.62% |
| quiet cutoff 平均 rank | 1.237 | 1.402 | 1.304 |
| LMR R1 fail-high | 5.113% | 7.385% | 6.162% |
| LMR R2 fail-high | 3.268% | 6.173% | 4.456% |
| LMR R3+ fail-high | 0.864% | 4.313% | 1.448% |
| verified reduced fail-high reject | 37.213% | 37.506% | 37.363% |

每盤平均 post-LMR quiet shadow samples 為 bonus 50.919、malus 35.026、unverified 23.221。
butterfly 非負率在 bonus/malus 為 93.321%／84.891%，既有 LMR statScore 為
92.698%／85.317%。兩者都能分出一些方向，但 rejected move 仍大多呈正值；這不是「history
表出錯」，而是目前 history 更新主要等待最終 beta cutoff，沒有把 reduced fail-high 的驗證結果
立即回饋到 continuation context。

### 10.3 診斷後決策

1. 不先調大 LMR、改 R1/R2/R3 表或降低 re-search；fail-high 隨 reduction 增大而下降，沒有全域
   under-search 證據，且較深縮減已有歷史負面實驗。
2. 不重做整套 move ordering。TT、good capture、good quiet 與 bad buckets 的 precision 呈清楚單調
   分層；quiet cutoff 在實際進入搜尋的 quiet 中平均 rank 只有 1.304。
3. **post-LMR continuation-history feedback 已依此執行並回退**：雙向強／弱、reject-only 與
   LMR statScore 的 from-to 對齊都沒有通過完整 n30k。現行 production 不在 verification 後寫
   history，也不把 butterfly 混入 LMR statScore；完整失敗矩陣見 §10.4。
4. 接納順序：契約與 legality → 500+500 n30k tree/事件比較 → fixed-depth gate → 使用者固定節點
   新舊引擎對戰。診斷本身不構成 Elo 證據。

### 10.4 依診斷執行的 production A/B（全部回退）

使用者同意後，依序測了四個互斥單軸候選。每版都先通過語法／history／search contracts，再跑
同一批 500 戰術 + 500 opening/quiet、每盤 30k nodes。四版都是 1000/1000 完成、無 engine
failure，但沒有任何一版同時改善配對深度與尾部風險：

| 候選 | 合計 mean nodes / depth | 配對 depth 提升／退化 | 最差戰術 delta | 決策 |
| :--- | ---: | ---: | ---: | :--- |
| P4 純診斷基準 | 31,612.003 / 10.416 | — | — | baseline |
| keep/reject 對稱 `120×depth, cap 900` | 31,564.878 / 10.390 | 141 / 148 | **−23** | 回退 |
| keep `80×d, cap 600`；reject `40×d, cap 300` | 31,590.152 / 10.386 | 99 / 125 | −3 | 回退 |
| reject-only `40×d, cap 300` | 31,589.903 / 10.396 | 85 / 103 | −2 | 回退 |
| LMR statScore 加完整 butterfly from-to | 31,543.727 / 10.357 | 163 / 190 | **−16** | 回退 |

回退後另重跑 `p4_production_ab_rollback_final_20260804_n30k`。與 P4 純診斷基準逐盤比對，
puzzle 500/500 與 opening 500/500 的 nodes、search/qsearch nodes、depth 及全部診斷計數都
完全一致；唯一差異是 `elapsed_s`，確認 production 搜尋語意已精確還原。

第一版暴露的保護盤面 `4r1k1/5p1p/3Q4/p4B2/2P5/8/P1q2PPP/3R1K2 b - - 0 35`
由 depth 30 降到 7，history-prune 102→579、LMR re-search 0.66%→10.81%。降低為 600/300
與 reject-only 後該盤恢復 depth 30，證明強 malus 會經第一層 continuation 讀取權重 5 放大，進而
誤觸 history pruning；但兩個保守版在 opening/quiet 配對仍分別為 53/71、40/59，不能接納。

最後測試 `2×piece-to + butterfly(from,to) + cont1 + cont2`，是因為 SF18 main history 使用完整
move key，而本引擎 piece-to 缺 from-square；但完整 butterfly 經現行 `LMR_HISTORY_SCALE=150`
足以改變整 ply reduction，保護盤又降到 depth 14，戰術集 63 提升／90 退化。因此也回退。

結論不是「post-LMR 理念錯誤」，而是本引擎的 continuation 讀取權重、history pruning 門檻與
整 ply LMR 量化高度耦合；沒有單獨搬運 SF 更新的安全空間。production 維持 P4 前搜尋樹，保留
shadow counters 與以上正式輸出供後續研究；下一次若再研究，應先增加暖 history／連續走步模式
與 history-prune attribution，不再直接加寫入或把完整 butterfly 塞進 reduction。

## 11. P5：暖 history 連續局面與 history-prune attribution

### 11.1 實作與語意護欄

P5 仍是 opt-in shadow diagnostics，沒有修改 production 的排序、剪枝、LMR 或 history
寫入。每次 production history prune 已經成立後，才重建完全相同的 8 個加總項：

1. `3×piece-to`；
2. butterfly；
3. `2×pawn history`；
4. continuation offset 1/2/3/4/6，讀取權重為 5/2/1/2/1。

針對每層記錄負值次數與幅度、拿掉該層是否能救回該著（decisive）、以及最負的
dominant layer；另記錄移除全部 continuation 後是否救回、門檻距離、depth、quiet
move index 與該著若未剪除是否具備 LMR 資格。`get_quiet_stat_components()` 有單元契約：
8 項總和必須精確等於 production `get_quiet_stat_score()`。

`hist_diag_bench.py --sequence-plies N --paired-cold` 會讓同一 context 沿每個 seed 的最佳著
前進，並在每個實際到達的局面另跑 fresh context。預設每步清 TT，只保留 history、
correction 與 counter/killers 等啟發狀態；只有顯式 `--keep-sequence-tt` 才測完整 context。
這避免下一個根局面直接命中上一次搜尋子樹，把 TT 誤當成 history 效果。

完整 cold shadow 閡門 `hist_diag_p5_attribution_shadow_final_20260804_n30k.json`
為 500+500 n30k。與 P4 最終回退基準逐盤比對所有既有語意欄位，puzzle 500/500
與 opening 500/500 都是 0 mismatch；mean nodes/depth 同為 31,612.003/10.416。
這證明 P5 counters 與工具擴充沒有改變 production 搜尋樹。

### 11.2 n30k 暖／fresh 配對結果

正式輸出：`hist_diag_p5_warm_history_sequence_diag_20260804_n30k.json`。使用前 100 戰術
seed + 100 opening/quiet seed，每個沿最佳著搜 3 ply，warm/fresh 各 30k nodes，最多
1200 次搜尋。ply 0 內部控制為 puzzle 100/100、opening 100/100 的 nodes/depth/score/
best move 完全一致，確認配對架構未自我污染。

| 集合 / sequence ply | warm depth | fresh depth | up / same / down | best move 相同 | mean |score delta| |
| :--- | ---: | ---: | ---: | ---: | ---: |
| puzzle ply 1 | 32.37 | 31.96 | 26 / 53 / 21 | 90% | 616.60* |
| puzzle ply 2 | 31.61 | 31.84 | 25 / 51 / 24 | 92% | 1531.98* |
| opening ply 1 | 8.06 | 7.96 | 32 / 45 / 23 | 56% | 20.43 |
| opening ply 2 | 8.00 | 7.86 | 37 / 38 / 25 | 46% | 18.86 |

`*` 戰術集含將死分數與 depth 64 提早結束，score delta 不可當作普通 cp 平均。

opening 的 history prune 在 ply 1 為 warm/fresh **6,706 / 42,730**，ply 2 為
**2,082 / 45,020**。這證明每個 FEN 重建 fresh context 的舊 benchmark 不能代表實戰搜尋
狀態；實戰中表會持續並每步 ÷2，`pawn_history=-1238` 的 fresh prior 也會快速變弱。

### 11.3 attribution 結論與下一個候選

將 sequence ply 1+ 的 raw counts 加總：

| 集合 / warm 層 | 單層 decisive | dominant |
| :--- | ---: | ---: |
| puzzle cont1 | **74.6%** | **77.5%** |
| puzzle main | 14.6% | 6.7% |
| opening cont1 | **56.1%** | **40.9%** |
| opening main | 70.2% | 53.8% |
| opening pawn | 26.7% | 0.0% |

warm prune 中 continuation 合計是必要條件的比例，puzzle ply 1/2 為 95.2%/99.9%，
opening 為 73.6%/93.5%。特別是 puzzle 的負 cont1 加權平均幅度約 19.8k，足以單獨
跨過多個 depth 的剪枝門檻。opening 的 prune 平均 depth 約 1.02，所以 P4 退化不是
「同一著直接被 LMR 縮深」，而是 history 寫入後在後續節點改變淺層剪枝。
full cold 500+500 也只有 1.16% 的 history prune 若未被剪掉會具備 LMR 資格。

下一個單軸 production 候選改為：**只在 history-prune 讀取路徑限制 cont1 的負值尾部**，
例如先 A/B `max(weighted_cont1, -4096)`。排序、LMR statScore、history 寫入與其他 continuation
層全部不動。這比「拿掉全部 continuation」更小，也直接針對 P4 暴露的負尾風險；
但目前尚未改 production，必須另做 cold 500+500 與 warm sequence 雙閡門。

## 12. P6：history-prune cont1 負尾保護

### 12.1 production 作用範圍

依 P5 結果實作 `HISTORY_PRUNE_CONT1_FLOOR = -4096`。只有 history-prune 呼叫
`get_quiet_stat_score()` 時顯式傳入此 floor；函式預設仍不限幅，因此 quiet ordering、
qsearch evasion ordering、LMR statScore、history 更新與 continuation table 本身均不變。
P6 counters 另記錄：raw weighted cont1 低於 floor 的次數、限幅增加的分數，以及 raw
本來會 prune、限幅後不 prune 的 rescue 次數。

### 12.2 cold 500+500 n30k

正式輸出：`hist_diag_p6_cont1_floor4096_cold_final_20260804_n30k.json`。

| 指標 | P5 基準 | P6 −4096 | 差異 |
| :--- | ---: | ---: | ---: |
| mean nodes | 31,612.003 | 31,600.384 | −11.619 |
| mean depth | 10.416 | 10.411 | −0.005 |
| history prune 總數 | 293,646 | 280,398 | −13,248（−4.51%） |
| cap hit / rescue | — | 653,259 / 11,489 | rescue/hit 1.76% |

puzzle depth up/same/down 為 3/494/3，最佳著 **500/500 不變**。opening 為
2/494/4，最佳著 496/500 不變；4 個改變的局面 score 差均在 41 cp 內。全體深度
提升 5、相同 988、降低 7，沒有重現 P4 的極端戰術 depth collapse。

### 12.3 warm sequence n30k

正式輸出：`hist_diag_p6_cont1_floor4096_warm_sequence_final_20260804_n30k.json`。
規格仍是 100 puzzle + 100 opening seeds、3 ply、warm/fresh 各 n30k，且每步清 TT。

| 集合 / ply | P6 warm/fresh depth | P6 best move 相同 | P5 best move 相同 |
| :--- | :---: | ---: | ---: |
| puzzle 1 | 32.37 / 31.96 | 90% | 90% |
| puzzle 2 | 31.59 / 31.84 | 92% | 92% |
| opening 1 | 8.06 / 7.98 | 56% | 56% |
| opening 2 | 7.99 / 7.87 | 47% | 46% |

與 P5 production 直接配對、且前序走步仍相同的位置中，puzzle 最佳著 300/300 不變；
opening 為 292/298（97.99%）。opening warm/fresh mean |score delta| 由 P5 的
20.43/18.86 cp 降為 19.38/17.98 cp，未顯示 history 狀態敏感度惡化。puzzle 的
history prune 在 ply 0/1/2 由 21,016/8,969/13,004 降為
14,935/4,276/5,260；opening 則由 34,707/6,706/2,082 降為
33,809/6,536/1,732。

結論：P6 通過 cold 戰術完整保護與 warm-history 雙閡門，production 候選保留，交由固定
節點新舊引擎對戰決定 Elo。它是刻意的小範圍防誤剪，不應與新的 history 寫入或 LMR 調整
同時合併測試。
