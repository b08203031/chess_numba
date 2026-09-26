# 搜尋歷史啟發 (History Heuristics)

| 項目 | 現況（2026-09-21） |
| :--- | :--- |
| **程式基線** | **Phase 1a 歷史啟發統一重構**（對齊 Stockfish 17/18 Step 21 走步統計更新架構） |
| **Step 21 重構** | **通過**（移出 move loop，PV 節點與 Fail-High 均正確獲得 Bonus/Malus；對戰 +20.00 Elo） |
| **full history 實驗** | 失敗（見下表）；檔案備份於 `search_heuristics_fullhist_backup.py` 等 |
| **工程計畫** | [`HISTORY_REFORM_PLAN.md`](HISTORY_REFORM_PLAN.md)（結構債暫緩，先恢復強度） |
| **P4 post-LMR** | 雙向／非對稱／reject-only n30k 均未通過，production 已回退；只保留 shadow diagnostics |
| **P5 暖 history** | 連續局面 warm/fresh n30k 配對已完成；cont1 負尾是 prune 主要風險 |
| **P6 cont1 保護** | production 僅在 history-prune 讀取時將 weighted cont1 下限設為 −4096；排序、LMR、寫入不變 |

### 對戰摘要

| 變體 | 對局設定 | Score | Elo | 結果 |
| :--- | :--- | ---: | ---: | :--- |
| FullHist | 200k nodes × 100 局 | 45.5% | ~−31 | 淘汰 |
| SoftHist（降係數） | 200k nodes × 100 局 | 45.0% | ~−35 | 淘汰 |
| Hist1a-compat（語意映射） | 200k nodes × 100 局 | **39%** | **~−78** | 淘汰 |
| **Step 21 統一更新重構** | **300k nodes × 400 局** | **52.9% (107W 209D 84L)** | **+20.00 ± 23.52** | **留下 (KEEP, 已同步)** |

---

## 歷史啟發統一重構（2026-09-21，對齊 Stockfish 17/18 Step 21）

### 1. 舊架構缺陷與根因
* 舊版程式碼將所有歷史表更新（History Bonus、Malus、Refutation Penalty、Continuation History、Capture History 等）深層嵌套在走步循環內的 `if alpha >= beta:`。
* **致命缺陷**：在 PV 節點（包含根節點 `ply == 0`，以及任何 `alpha < score < beta` 的節點），搜尋永遠不會觸發 `alpha >= beta`。導致 PV 最佳步獲得 **0 歷史獎勵**，而該節點嘗試過的劣步也 **0 懲罰 (Malus)**，嚴重阻礙後續迭代的走步排序收斂。

### 2. 重構後設計
1. **Beta Cutoff 精簡**：在走步循環內部發生 `alpha >= beta` 時，僅更新剪枝診斷計數器與安靜步 Killer Move，隨即 `break`。
2. **走步循環外統一更新（Step 21）**：
   * 在循環結束後由 `if final_flag != TT_FLAG_ALPHA and best_move != NO_MOVE:` 守護，涵蓋所有改善 alpha 的節點（PV 節點與 Fail-high 節點）。
   * **Best Move 獎勵**：安靜最佳步獲得 main / butterfly / pawn / continuation / low-ply 獎勵及 counter move 更新；吃子最佳步獲得 capture history 獎勵。
   * **劣步懲罰（Malus）**：遍歷 `quiet_moves_tried` 與 `capture_moves_tried`，明確以 `if move == best_move: continue` 排除最佳步，正確懲罰非最佳走步。
   * **Fail-Low 互斥**：以 `elif final_flag == TT_FLAG_ALPHA and ply > 0 and legal_moves_tried > FAIL_LOW_MIN_LEGAL_MOVES:` 獎勵造成對手 fail-low 的上一步走步。

---

## Phase 1a 表結構

| 類型 | 形狀 | 角色 |
| :--- | :--- | :--- |
| Piece-To | `history_table[12][64]` | 主排序 / LMR |
| Butterfly | `butterfly_history[64][64]` | from×to |
| Capture | `capture_history[12][64][6]` | victim type 0..5 |
| Counter | `counter_moves[12][64]` | prev_piece × prev_to |
| Continuation | `[5][12][64][12][64]` | offsets 1,2,3,4,6 |
| Pawn | `[8192][12][64]` | |
| Low-ply | `[5][65536]` | |

### 寫入

* 線性 `bonus/malus = min(120*depth, 1800/1600)`
* Gravity + 分級 ceiling（16384 / 8192 / 12288 / 4096）
* 迭代 aging：各表 **÷2**（Phase 1a）
* Quiet malus 尺度 1136/956；pawn 非對稱 1038/525

### 排序 quiet

`2*history + butterfly + 2*pawn + cont×(4,2,1,2,1) + LPH + killers/counter/check`

---

## 教訓

1. 大表每步 soft-age 會把 NPS 砍半 → 若再做 full cont，**禁止**每步掃 22MB。  
2. 僅調係數無法救 full history 結構 Elo。  
3. 「語意兼容 1a」≠ 真 1a；半成品映射可更差。  
4. 重加結構必須 **單項 A/B**，通過才合併。
5. post-LMR continuation feedback 會被 cont-1 讀取權重 5 與 history pruning 放大；即使 cap 300
   能消除極端戰術退化，opening/quiet 配對仍偏負。不要在未做暖 history attribution 前重啟。
6. P5 證明 cold-per-FEN 會大幅高估實戰 warm context 的 history prune；以後 history 改動
   必須同時通過 cold 500+500 與清 TT 的 warm-sequence/fresh 配對。
7. warm ply1+ 中 cont1 在 puzzle/opening 的 dominant 比例為 77.5%/40.9%。下一步只可在
   history-prune 讀取路徑限制負尾，不可改全局排序權重或再加 post-LMR 寫入。
8. P6 的 −4096 floor 在 cold 500+500 n30k 救回 11,489 次原本會被剪掉的著；500 個戰術
   根局面最佳著全數不變。warm sequence 的戰術可比位置也為 300/300 不變，因此保留供對戰。
9. **歷史啟發不可只在 Beta Cutoff 更新**：PV 節點獲得 0 獎勵、劣步獲得 0 懲罰會直接導致迭代深層主要變例排序鈍化；對齊 SF17/18 Step 21 統一更新後，在 400 局 300k nodes 對戰實測帶來 +20.00 Elo 顯著增益。
