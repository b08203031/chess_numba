# 搜尋參數調優器 (Search Parameter Tuner) 使用說明

本目錄 (`tune_search/`) 包含了一套專為搜尋參數設計的自動化調優系統。與靜態評估調參不同，搜尋參數（如剪枝邊界、LMR 深度等）無法通過靜態數據集 (MSE) 進行優化，必須通過**實戰對局 (Match Playing)** 來驗證優劣。

本系統採用 **SPSA (Simultaneous Perturbation Stochastic Approximation)** 演算法，透過讓引擎自我對弈來不斷調整參數，以提升棋力。參數定義現在集中在
`search_param_registry.py`：目前盤點 **114 個實際搜尋標量**，其中 **34 個**已映射到
`SearchContext.tune[]`，其餘是需要重啟/JIT 的 compile-time 候選。這避免把不存在的
舊名稱（例如早期的 `FP_MARGIN_D1`）誤當成可調參數。

---

## 1. 核心原理 (Principles)

### 為什麼需要獨立的搜尋調優器？
*   **評估 (Evaluation)**：目標是讓靜態分數接近真實勝率。可以使用大師對局數據計算 MSE (均方誤差) 來訓練，速度極快。
*   **搜尋 (Search)**：目標是修剪無效分支並保留關鍵路徑。參數（例如 "RFP Margin = 250"）直接影響搜尋樹的形狀。唯一的驗證方法是**實際下棋**，看是否能贏過舊參數的引擎。

### SPSA for Match Playing
1.  **基準 (Base)**：當前最佳的一組參數。
2.  **擾動 (Perturbation)**：隨機生成一個擾動向量 $\Delta$（例如 +50, -1, +25...）。
3.  **對弈 (Match)**：
    *   生成兩組參數：$\theta_{plus} = \theta + c\Delta$ 與 $\theta_{minus} = \theta - c\Delta$。
    *   讓 $\theta_{plus}$ 與 $\theta_{minus}$ 互相比賽（例如 10 局）。
4.  **更新 (Update)**：
    *   如果 $\theta_{plus}$ 勝率高，則參數往 $+\Delta$ 方向移動。
    *   如果 $\theta_{minus}$ 勝率高，則參數往 $-\Delta$ 方向移動。
    *   移動步長由學習率 (Learning Rate) 決定，隨迭代次數衰減。

---

## 2. 系統架構 (Architecture)

為了優化執行效率並支援 Numba JIT 的特性，本系統採用以下架構：

### 1. 進程重用與生命週期管理
*   **舊版問題**：每盤棋重新啟動引擎 -> 觸發 Numba 重新編譯 -> 浪費大量時間。
*   **runtime 新路徑 (in-process)**：
    *   **一次性啟動**：整個 SPSA campaign 開始時建立固定數量的 `SearchContext` 對（Pair）。
    *   **持續運行**：每個 SPSA iteration 只改寫 `SearchContext.tune[]`，所有 iteration 共用同一個 Python process/JIT。
    *   **狀態重置**：每局棋前清除 TT、history、killer/counter 等狀態，參數陣列則保留在各自的 A/B context。
*   **compile-time 路徑 (UCI)**：選到尚未映射到 `tune[]` 的常數時，仍由 adapter 建立新進程；這是必要成本，不應拿來做高迭代 SPSA。

### 2. 智能暖機 (Smart Warm-up)
*   引擎啟動後，會自動執行一次 **Kiwipete 深度 1 (Depth 1)** 的搜尋。
*   **目的**：Kiwipete 局面極其複雜，能強制 Numba 編譯絕大多數的搜尋路徑（如進攻、防禦、叫殺檢查等）。這確保了正式對局開始時，JIT 編譯已完成，引擎能立即全速運作。
*   **使用者體驗**：啟動時會顯示 `Warming up...` 提示；依 cache 狀態可能需要數十秒到數分鐘，之後的對局才會回到穩定速度。

### 3. 輸出緩衝處理
*   強制使用 Python 的 `-u` (Unbuffered) 模式啟動引擎，防止在 Windows 環境下因 I/O 緩衝導致的通訊死鎖 (Hang)。

---

## 3. 參數清冊與分階段調參

不要把 114 個軸一次塞進 SPSA；高維度會讓每個回合的勝負訊號過於稀薄。調參器預設
只選 runtime profile 的 34 個軸，並提供依搜尋機制拆分的 profile：

```bash
# 列出預設 runtime profile（不啟動引擎）
python -m tune_search.search_tuner --list-params

# 先做一個小的 runtime LMR block（只選可 live mutate 的 slot）
python -m tune_search.search_tuner \
    --params LMR_BASE_OFFSET,LMR_HISTORY_SCALE,LMR_CUTNODE_BONUS,LMR_TTCAPTURE_BONUS,LMR_MOVECOUNT_FACTOR,LMR_TABLE_SCALE_PERCENT \
    --backend inprocess --scale-mode integer --iter 20 --games 20 --nodes 30000 --concurrency 6 \
    --log tune_search/logs/lmr_l1_30k_20260804.jsonl

# 明確指定一組 runtime 參數
python -m tune_search.search_tuner --params LMR_BASE_OFFSET,LMR_HISTORY_SCALE,LMR_TABLE_SCALE_PERCENT \
    --backend inprocess --scale-mode integer --iter 20 --games 20 --nodes 30000 --concurrency 6
```

`runtime`、`pruning`、`nmp`、`lmr`、`lmr_runtime`、`lmr_compile`、`qsearch`、
`history`、`ordering`、`safeguards`、`aspiration` 和明確的 `all` 都來自同一份 registry。
其中 `lmr_runtime` 只選 12 個 live slot，`lmr_compile` 只選 10 個需要重啟/JIT 的
LMR 常數；`lmr` 才是兩者合併的完整 22 軸。每個條目都記錄預設值、邊界、最小步長、
連續/離散型別與 SPSA 的 `r_end`。離散模式（例如 `NMP_GATE_MODE`）適合單獨做 A/B
或小 block；不應和大量連續軸盲目混合。

調參器現在按參數使用自己的 `c_end` 與 `r_end`，而不是舊版對所有數值共用
`c=20/a=200`。每個批次的對局數會換算成 paired games，並依照完整排程的 pair
數衰減 `c_k=c_end*(N/k)^gamma`、`a_k=a_end*(A+N)^alpha/(A+k)^alpha`，其中
`a_end=r_end*c_end²`、`A=A_ratio*N`。這與 Fishtest classic SPSA 的 worker 更新
`theta += (a_k/c_k)*(wins-losses)*delta` 一致；JSONL 會保留 `c_k/a_k/r_k`、pair
進度與原始更新，方便查核。舊 registry 的 `spsa_r_end` 是 0.4--0.8 的歷史相對增益，
canonical tuner 透過明確的 `canonical_r_end` 轉換係數 0.01 後使用，舊 checkpoint 不會
被誤當成新排程續跑。

`--scale-mode integer` 只在寫入 `SearchContext.tune[]` 的邊界做一單位整數 rounding；
內部 theta 仍保留浮點累積，避免每回合把小於 1 的證據捨棄。`registry` 則保留舊的粗網格
輸出。runtime 參數已接到單進程 fixed-node match；compile-time 參數仍由 adapter 在
import 前 patch，必須用新進程驗證。

### L1 的細粒度整數尺度

`--scale-mode integer` 的 plus/minus 仍以 registry `step` 作為可觀測的探測半徑，
但實際套用值是一單位整數網格；不再有額外的 1% 人工 cap。這讓更新大小完全由
Fishtest 的 `c_end/r_end/A/alpha/gamma` 排程決定，且大、小數值軸自然使用不同尺度。

若要重現舊的粗粒度實驗，明確加入 `--scale-mode registry`；不同 scale mode 必須使用
不同的 state/log，不能互相 resume。

目前 `--backend auto` 已完成這個分流：runtime-only profile 會建立一次 persistent
`SearchContext` worker pool，整個 SPSA campaign 共用一次 JIT；只要選到 compile-time 候選，
則自動回到 UCI adapter（每個 SPSA 參數組合需要新進程）。runtime 路徑使用 fixed nodes，
以免時間控制與 Windows 排程噪音污染參數訊號。調參狀態預設寫入
`tune_search/search_spsa_state.json`，可用 `--resume` 接續同一組參數。

## 4. 本機硬體與推薦調參流程

目前工作站辨識到的 CPU 是 **12th Gen Intel Core i5-12500H**：4 個 P-core + 8 個
E-core，共 **12 個實體核心 / 16 個硬體執行緒**。因此 `--concurrency` 應視為同時
搜尋的 context-pair 數，而不是單純的執行緒數；每個 pair 有 A/B 兩個 context。建議
依目前決定先用 `--concurrency 6`（12 個搜尋 context），再觀察溫度與實際 nodes/s；若
長時間降頻或結果噪音變大，退回 4。不要開到 8 或 16，因為 E-core、記憶體頻寬與散熱
會讓每節點速度及結果噪音變差。

專案的 JIT 報告顯示，冷啟動約 158 秒、暖啟動仍約 78 秒，而同一 process 的後續搜尋
才會回到毫秒級。因此最重要的節省不是把編譯器再並行化，而是讓 runtime campaign
只編譯一次：不要在每個 SPSA iteration 清 cache 或重啟引擎；只有 compile-time 參數
才另開 UCI process。

推薦採用四層流程：

1. **L0 基線**：固定 registry 預設值，JIT 暖機一次；記錄 nps、平均深度、nodes、
   root move 與勝和負和局，不把暖機時間當成引擎 NPS。
2. **L1 方向搜尋**：一次只選同一機制的 4–6 個 runtime 軸。使用每局 **30k nodes**、
   每 iteration 20 局、先跑 12–20 iterations；30k 已經足以比 3k 更可靠地分辨 LMR
   方向，但仍保留足夠吞吐量。第一個候選從 LMR block 開始，維持目前邊界。
3. **L2 晉級驗證**：固定 L1 找到的中心值，改用 `hist_diag_bench.py` 的戰術 + 中局/安靜
   盤面，使用 **200k nodes**、40–80 局，並用另一個 seed/另一半盤面重跑。L2 主要是
   驗證候選，不建議在 200k 上重新跑大型 SPSA。
4. **最終確認**：保留通過 L2 的一組，才進行你安排的新舊引擎 100 局對戰；不要為
   每一組候選反覆刪除 JIT disk cache，因為每局 matcher 已經會重置 TT/history，公平性
   不依賴冷啟動。

runtime L1 範例：

```bash
python -m tune_search.search_tuner \
  --params LMR_BASE_OFFSET,LMR_HISTORY_SCALE,LMR_CUTNODE_BONUS,LMR_TTCAPTURE_BONUS,LMR_MOVECOUNT_FACTOR,LMR_TABLE_SCALE_PERCENT \
  --backend inprocess --scale-mode integer \
  --iter 24 --schedule-iterations 24 --games 50 --nodes 30000 --concurrency 6 \
  --spsa-a-ratio 0.1 \
  --seed 20260804 \
  --state tune_search/states/lmr_l1_30k_fishtest_20260804.json \
  --log tune_search/logs/lmr_l1_30k_fishtest_20260804.jsonl
```

每 iteration 完成後會寫入 `tune_search/search_spsa_state.json`；中斷後用相同參數、
nodes、games、concurrency、backend 和 `--resume` 接續。checkpoint 會保存 SPSA/開局
所用的 Python random state，所以 seed 只需在第一次啟動時指定；若改變 match 配置，
請另用新的 `--state`。`--iter` 在 resume 模式表示「再增加幾次」，不是覆蓋原有次數。
每次啟動會建立或延續一個 JSONL log；每個 iteration 會記錄 plus/minus 參數、delta、
`c_k`/`a_k`、plus score、signal 與更新後參數。若沒有指定 `--log`，會自動建立 timestamp
檔案；resume 會沿用 checkpoint 裡的 log 路徑。
若是修正 SPSA 更新公式前建立的舊 checkpoint，必須使用新的 `--state`，不能直接 resume。
調參器的速度提升主要是移除重複 JIT 成本，不代表被調參引擎本身的單執行緒
NPS 必然上升；棋力是否改善仍以 L2 與最終對戰為準。

compile-time 候選採用另一種節奏：先做單軸或 2–3 軸 coordinate sweep，每個點只做
短 match，確認值得投入後才安排一次 cold compile；不要直接對所有 compile-time 軸
跑 100 次 SPSA。長期若某個常數反覆被調整，優先把它安全地映射到 `tune[]`，一次冷
編譯後即可納入 persistent runtime campaign。

## 5. 操作步驟 (Operation Steps)

### 步驟 1: 配置參數 (可選)
請編輯 `tune_search/search_param_registry.py` 的 registry，而不是在
`param_config.py` 另加一份清單。`param_config.py` 只保留舊腳本的相容 facade；測試會檢查
每個 source constant、runtime slot、邊界與預設值都一致。

### 步驟 2: 準備開局庫
建議準備一個 PGN 或 Polyglot (.bin) 開局庫。系統預設尋找 `tune_search/polyglot.bin` 或 `tuner/data/pgn/twic1613.pgn`。
*   若使用 Polyglot 書，系統會隨機漫步產生開局。
*   若使用 PGN 書，系統會隨機選取對局的前 8 半步 (4 回合) 作為開局。

### 步驟 3: 開始調優
在專案根目錄執行以下指令：

```bash
# runtime profile 可列出 34 個 live slot；實際 SPSA 先限制在同一機制的 block
python -m tune_search.search_tuner --params \
    LMR_BASE_OFFSET,LMR_HISTORY_SCALE,LMR_CUTNODE_BONUS,LMR_TTCAPTURE_BONUS,LMR_MOVECOUNT_FACTOR,LMR_TABLE_SCALE_PERCENT \
    --backend inprocess --scale-mode integer --iter 20 --games 20 --nodes 30000 --concurrency 6 \
    --log tune_search/logs/lmr_l1_30k_20260804.jsonl

# compile-time 參數才使用 UCI/time-control adapter；不要用大量 iteration
python -m tune_search.search_tuner --params LMR_MIN_DEPTH \
    --backend uci --iter 3 --games 18 --concurrency 2 --tc "0.1"
```

**參數說明：**
*   `--iter <N>`: 總迭代次數。
*   `--games <N>`: 每次迭代的對局數。
*   `--concurrency <N>`: runtime inprocess 時是同時使用的 context-pair 數；本階段先用 6。UCI backend 時則是引擎進程 pair 數，成本較高。
*   `--nodes <N>`: runtime inprocess 的每步固定節點數；L1 使用 30k，L2/晉級使用 200k。
*   `--scale-mode integer`: 只在引擎參數邊界做一單位整數 rounding；這是目前建議模式。
*   `--schedule-iterations <N>`: 完整 Fishtest 衰減排程的批次總數；新 campaign 預設等於
    `--iter`，resume 時若省略會讀 checkpoint 的原值。
*   `--spsa-a-ratio <R>`: `A = R * paired_games`，預設 `0.1`；更換時必須使用新的 state。
*   `--log <Path>`: 每次啟動與每次 iteration 的 JSONL 紀錄檔。
*   `--tc <Time>`: 只供 UCI backend 使用的時間控制，格式為 `秒+增量` (例如 `0.1+0.01`)。

### 步驟 4: 查看結果
系統會提供詳細的即時輸出：
*   **暖機進度**：顯示引擎啟動與 Kiwipete 暖機狀態。
*   **單局結果**：每當一對引擎完成一組對局（互換先後手），會顯示 `Game Pair Finished: Test Won | Base Won`。
*   **累計分數**：即時顯示當前迭代的測試組勝率與勝/負/和局數。
*   **參數更新**：迭代結束後，顯示參數的調整方向與新數值。

結果會即時保存至 `tune_search/current_params.txt`。

---

## 6. 常見問題

1.  **為什麼一開始要等很久？**
    *   這是正常的 JIT 編譯過程（冷啟動可能超過兩分鐘）。runtime campaign 只需等待一次；看到 warmup 完成後，後續 iteration 不應再次出現同等級等待。

2.  **如何終止調優？**
    *   直接按 `Ctrl+C`。由於使用了進程池，系統會嘗試優雅地關閉所有引擎進程。

3.  **如何應用最佳參數？**
    *   手動將 `tune_search/current_params.txt` 中的數值更新回 `chess_engine/classical/constants.py`。
