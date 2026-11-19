# 引擎對戰與 Stockfish Match 指南

本專案有兩個對戰入口：

- `tools/tournament.py`：現行 `classical` 對基準 `classical_old`。
- `tools/match_runner.py`：現行引擎對外部 Stockfish UCI 執行檔。

兩者共用 `tools/match_core.py` 的裁判、PGN、SPRT、開局配對與工作池實作。預設搜尋限制是每步固定 `50,000` nodes；固定深度與固定時間仍保留。

兩支程式在正式對局前會依序執行兩層暖機：先讓每個 Python/Numba backend 搜尋一次 Kiwipete、固定 `depth 14`；再從標準初始局面進行一場完整的 depth-1 暖機局。d1 暖機局的雙方每步都固定 `d1`。所有暖機結果與 PGN 都不列入比分、SPRT 或正式輸出。in-process backend 的 Kiwipete 搜尋與 d1 對局只需使用第一組共享 JIT 的 context；若明確使用 Python UCI backend，彼此不共享 JIT 的 worker 子行程會各自執行相同暖機。外部 native Stockfish 不額外執行 Kiwipete 搜尋，但會參與 d1 暖機對局。

第一組 worker 的 Kiwipete 暖機會即時列出雙方每個完成深度的 UCI `info`（depth、score、nodes、nps、PV）；後續平行 worker 仍保持安靜，避免暖機訊息重複刷屏。這些診斷行只供觀察，不會改變搜尋限制或正式對局結果。

## 1. 為什麼改成 in-process

舊版 `tournament.py -c 8` 會建立 8 個 New 與 8 個 Old Python 子行程。歷史上 Numba 的遞迴 `_search` / quiescence 曾必須使用 `cache=False`，因此主要搜尋函數會在 16 個行程內分別編譯；即使在已啟用 disk cache 的版本中，16 個子行程仍須各自載入執行時期、重複配置並執行暖機。

現在預設架構是：

```text
單一 Python 行程
├─ chess_engine.classical：JIT 一次
├─ chess_engine.classical_old：JIT 一次
├─ worker 1：New context 1 ↔ Old context 1
├─ worker 2：New context 2 ↔ Old context 2
└─ 主執行緒：python-chess 裁判、PGN、SPRT、統計
```

每個 context 仍有獨立的 TT、pawn cache、history、correction history 與搜尋狀態；只有已編譯的機器碼被同一 namespace 的 workers 共用。Numba 搜尋函數會釋放 GIL，所以 worker threads 可以真正同時搜尋。

`classical_old` 的 runtime `eval_weights` 會明確從 `classical_old.constants` 初始化，不會混用現行評估參數。

### `InProcessEngine` 的 Lazy SMP Helper-Context 生命週期

當 `InProcessEngineFactory(..., threads=N)` 指定 `threads > 1` 時，`tools/match_core.py` 會在單一行程內管理每個引擎實例的 Helper Context 生命週期：

1. **相容性驗證**：初始化時透過 `inspect.signature` 與模組屬性檢查 `iterative_deepening_search` 是否支援 `num_threads` 與 `helper_contexts` 關鍵字參數，以及 `engine_types` 是否提供 `create_smp_helper_context` 與 `reset_search_context`；若目標命名空間不支援（例如 `chess_engine.nnue` 或未同步 SMP 的舊快照），立即拋出明確 `ValueError`。
2. **配置與設定同步（`_make_contexts`）**：先依 `env_extra`、namespace `eval_weights` 與 runtime 調參覆寫建立主 `SearchContext`，再呼叫 `create_smp_helper_context(main_context)` 配置 `threads - 1` 個 Helper Context（與主 Context 零拷貝共享實體 `transposition_table` 與 `stop_flag`，各自擁有獨立 `NRT_MemInfo` 標頭與 thread-local 歷史表/堆疊），並將 `tune`、`eval_weights` 與 `enable_*` 開關同步至所有 Helper。
3. **暖機（`prepare`）**：僅 `worker_id == 0` 執行；當 `threads > 1` 時會先呼叫 `warmup_search_jit(search_context=engine.context)` 預熱 `_lazy_smp_helper_loop_jit`，隨後在 Kiwipete `depth=14` 上執行一次多執行緒暖機搜尋，最後進入 `startpos` `d1` 暖機局。
4. **開新局重置（`new_game`）**：對主 Context 執行重置並**只清空一次**共享置換表，接著逐一以 `reset_search_context(helper_ctx)` 重置各 Helper 的私有歷史表與堆疊（不重複清空共享 TT）。
5. **下棋呼出（`get_move`）**：當 `threads > 1` 時，自動將 `num_threads=self.threads` 與 `helper_contexts=self.helper_contexts` 轉傳給 `iterative_deepening_search`。

### 純評估參數 A/B 基準

若目標是驗證 Texel 常數本身，New 與 Old 必須共用完全相同的搜尋與評估程式，只讓 `constants.py` 不同。同步與檢查使用：

```bash
python tools/sync_classical_old.py --sync-code
python tools/sync_classical_old.py --diff-code
```

`--sync-code` 會同步所有 Python 引擎程式、namespace imports 與 `constants.py` 的搜尋常數區，但刻意保留 `chess_engine/classical_old/constants.py` 前半部的舊評估常數；未變動的檔案不會重寫。`--diff-code` 應輸出 `Engine code and search constants are logically identical`。不要在 tournament 執行途中同步；應先完成並封存當期 PGN。

舊參數快照仍須具備現行公式需要的結構常數。對於舊版沒有的 king-danger rookless 分支，Old 令 `KING_DANGER_NO_QUEEN_ROOKLESS = KING_DANGER_NO_QUEEN`，等價保留原本「有后／無后」單一係數的行為；New 才使用分開調出的兩個值。phase 上下限則由各自的 MG 材質值套用同一公式推導，因此材質與 phase 尺度保持耦合。

## 2. tournament.py

### 固定節點（建議）

```bash
python tools/tournament.py --games 200 --nodes 100000 -c 8 --no-sprt
```

未指定 `--nodes`、`--depth` 或 `--time` 時，預設等同 `--nodes 50000`。

固定節點適合比較棋力，因為 CPU 排程或瞬間 NPS 波動不會改變每一步的搜尋工作量。每個開局會交換黑白各下一盤。

### 固定深度

```bash
python tools/tournament.py --games 100 --depth 12 -c 4
```

固定深度的實際節點量可能相差很大，主要適合功能診斷，不建議直接拿不同搜尋版本的 depth 當等成本棋力比較。

### 固定時間

```bash
python tools/tournament.py --games 100 --time 1000 -c 4
```

`--time 1000` 表示每步 1,000 ms。多盤同時執行時，wall-clock 搜尋會受核心排程影響；若重視 NPS 穩定度，使用固定節點，或以 `-c 1` 另做 NPS benchmark。

### 多階段（含非對稱 stage 限制）

```bash
python tools/tournament.py -c 8 --no-sprt \
  --stages "200:n20000,400:n100000,200:d12"
```

也支援 `games:limit1:limit2` 為雙方指定非對稱限制（例如 `--stages "100:n100000:n50000"` 或 `"100:1000:500"`）。`--stages` 不可與 `--nodes/--depth/--time` 或 `--nodes1/2/--depth1/2/--time1/2` 同時使用。

限制 token：

- `n50000`：固定 50,000 nodes。
- `d12`：固定 depth 12。
- `1000`：固定 1,000 ms/move。

所有階段共用同一批 contexts，階段切換不會重新 JIT 或重新配置工作池；每盤開始仍會清空該 context 的 TT/history，維持對局獨立性。

### 非對稱限制（Per-Side Limits）

單階段對戰也可直接用個別旗標覆寫雙方的搜尋上限（每方 `--nodes{1,2}` / `--depth{1,2}` / `--time{1,2}` 三選一互斥）：

```bash
python tools/tournament.py --games 100 -c 4 --no-sprt \
  --nodes1 100000 --nodes2 50000
```

若只指定其中一方（例如 `--time1 1500 --time 1000`），未單獨指定的一方沿用共用限制（或預設 50,000 nodes）。當黑白雙方實際限制不同時，PGN 會寫入 `WhiteLimit` 與 `BlackLimit` header；多階段且雙邊限制不同時，階段 PGN 檔名會以 `<limit1>_vs_<limit2>` 命名。

### 多執行緒對戰與同引擎 `Threads=N` vs `Threads=1` 比較

`tournament.py` 支援對稱與非對稱執行緒設定（皆須為 `>= 1` 的整數；`run_tournament(..., threads1=1, threads2=1)` 亦提供相同參數）：

- `--threads N`：同時設定雙方搜尋執行緒數（除非該方另有 `--threads1` / `--threads2` 覆寫）。
- `--threads1 N`、`--threads2 N`：分別指定引擎 1 與引擎 2 的搜尋執行緒數（預設皆為 `1`）。
- 若任一方使用 `> 1` 執行緒，每盤 PGN 會依實際執白／執黑自動寫入 `WhiteThreads` 與 `BlackThreads` header。

測試單一引擎的多核心 Lazy SMP 擴展強度時，應使用**固定時間（`--time`）**並設定清晰的 `--name1` / `--name2`（若雙邊名稱相同，`tournament.py` 會印出提醒避免 PGN 無法區分）：

1. **In-process 模式（推薦：同一命名空間只 JIT 一次並跨雙方共享）**  
   注意 `--backend auto` 僅在預設 `main.py` 對 `main_old.py` 時自動選 `inprocess`；同引擎比較請明確指定 `--backend inprocess` 與 `--engine1-module` / `--engine2-module`：

   ```bash
   python tools/tournament.py --backend inprocess \
     --engine1-module chess_engine.classical --engine2-module chess_engine.classical \
     --name1 SMP4 --name2 SMP1 --threads1 4 --threads2 1 \
     --time 1000 --games 100 -c 1 --no-sprt
   ```

2. **UCI 模式（適用 `main.py` vs `main.py` 或外部 UCI 執行檔）**：

   ```bash
   python tools/tournament.py --backend uci \
     --engine1 main.py --engine2 main.py \
     --name1 SMP4 --name2 SMP1 --threads1 4 --threads2 1 \
     --time 1000 --games 100 -c 1 --no-sprt
   ```

### Backend 選擇與 UCI `Hash`/`Threads` 轉發

預設 `--backend auto`：

- `main.py` 對 `main_old.py`：選擇 `inprocess`。
- 自訂任意 UCI 路徑（含 `main.py` vs `main.py`）：選擇 `uci` 相容模式。

可以明確指定：

```bash
python tools/tournament.py --backend inprocess
python tools/tournament.py --backend uci
```

in-process backend 使用：

- `--engine1-module chess_engine.classical`
- `--engine2-module chess_engine.classical_old`

在 `uci` 模式下，`tournament.py` 會自動將 `{"OwnBook": own_book1/2, "Hash": tt_mb, "Threads": threads1/2}` 傳給 `UciEngineFactory`（`match_runner.py` 則傳遞 `{"OwnBook": False, "Hash": tt_mb, "Threads": engine_threads}`）。`UciEngine` 啟動時會先記錄引擎於 `uciok` 前宣告的 `option name ...`，並刻意將 `Hash`、`OwnBook` 等選項排在 **`Threads` 之前** 送出，讓引擎先調整 TT 大小再配置 SMP Helper；若要求 `Threads > 1` 而引擎未宣告 `Threads` 選項，則印出非致命警告。此外，`UciEngine.get_move()` 會忽略 `info string` 與非主變例（`multipv != 1`）行，並能正確解析 SF19 風格帶 `lowerbound` / `upperbound` 尾註的 `info ... score cp/mate ...` 輸出。

`--own-book1/--own-book2` 只支援 UCI backend。一般 A/B 測試應關閉引擎自己的隨機 book，改用裁判提供的成對開局。

## 3. 記憶體與 TT

`--tt-mb` 預設維持 `128`，而且是「每個引擎 context」128 MB：

```bash
python tools/tournament.py -c 8 --tt-mb 128 --nodes 100000
```

`-c 8` 會建立 8 個 New + 8 個 Old contexts，單計 TT 約 2 GB；另外還有 continuation history、pawn history、pawn cache 等大型陣列。16 GB 系統可先使用 `-c 8`，若出現大量換頁、整機記憶體逼近上限或 NPS 明顯下降，再降低 concurrency，而不是縮小 TT。

`-c` 與執行緒數建議不要超過實體核心數（**CPU Oversubscription 防護**）：每盤對局同一時間只有輪到走棋的一方正在搜尋，因此 `tournament.py` 的峰值活躍搜尋執行緒數為 `actual_concurrency * max(threads1, threads2)`（`match_runner.py` 則檢查 `actual_concurrency * max(engine_threads, sf_threads)`，並同時列印總配置執行緒數 `actual_concurrency * (engine_threads + sf_threads)`）。只要該峰值超過 `os.cpu_count()`，兩支程式都會印出非致命的 `[warning]` 提示 CPU 競爭可能影響時間控制與 NPS；跑多執行緒測試（如 `--threads1 4 --threads2 1`）時建議將 `-c` 壓低（固定時間擴展測試建議 `-c 1`）。

## 4. 開局與重現性

預設讀取 `data/openings.epd`，並用 `--seed 0` 做可重現的洗牌。

使用 PGN 開局庫：

```bash
python tools/tournament.py --openings-pgn data/8moves_v3.pgn \
  --opening-moves 8 --seed 123 --nodes 100000
```

只從初始局面開始：

```bash
python tools/tournament.py --no-openings --nodes 100000
```

## 5. SPRT、early trash 與 PGN

- SPRT 預設啟用，假設區間可用 `--sprt-elo0`、`--sprt-elo1` 修改。
- `--no-sprt` 會完整跑完排定局數。
- `tournament.py` 預設保留 early trash；可用 `--no-early-trash` 關閉。
- PGN 每步註解包含 depth、side-to-move evaluation、nodes、time 與 NPS；當任一方使用 `> 1` 執行緒時寫入 `WhiteThreads` / `BlackThreads` header，當雙方限制不同時寫入 `WhiteLimit` / `BlackLimit` header。
- engine crash、timeout、無 bestmove、非法著或 backend 狀態不同步都會明確判負，不會偷偷改走第一個合法著。
- failure PGN 會附帶 `Failure` 與 `FailurePly` header，保存 bounded exception 訊息與發生回合；統計時應剔除該 failure 所在的整組換色配對。
- 預設最長 400 plies；可用 `--max-plies` 修改，超過時裁定和棋並寫入 PGN Termination。
- **提前裁決（Early Adjudication）**：`tournament.py` 與 `match_runner.py` 皆預設啟用（`--adjudicate`），對標 Cutechess / CCRL 標準，可省去約 10%~15% 的無效垃圾殘局時間；支援一鍵開關（`--no-adjudicate`）：
  - **勝負裁定（Win/Resign Adjudication）**：雙方連續 5 回合（10 個半步/plies）認定差距 $\ge 500\text{ cp}$ 時立即判勝（歷史 894 局實測 100% 準確率、0 翻盤）。
    - 參數：`--adjudicate-win-score 500`、`--adjudicate-win-moves 5`
  - **和棋裁定（Draw Adjudication）**：第 40 回合以後，雙方連續 6 回合（12 個半步/plies）認定分數在 $8\text{ cp}$ 以內時立即判和（具備開局/中局保護期，避免誤判戰術搏殺局）。
    - 參數：`--adjudicate-draw-score 8`、`--adjudicate-draw-min-move 40`、`--adjudicate-draw-moves 6`

自訂輸出：

```bash
python tools/tournament.py --pgn tournament_analysis/eval_ab.pgn --nodes 100000
```

多階段會自動加入 `_stageN_limit` 後綴。

## 6. match_runner.py：對 Stockfish

預設架構：

```text
單一 Python 行程
├─ classical JIT 一次
├─ worker 1：local context 1 ↔ Stockfish process 1
├─ ...
└─ worker N：local context N ↔ Stockfish process N
```

Stockfish 是外部 native UCI 引擎，因此每個平行 worker 仍需要一個長駐 Stockfish 行程；我方 Numba 引擎不再因 `-c` 重複編譯。

等節點對戰：

```bash
python tools/match_runner.py --sf external/stockfish/stockfish-windows-x86-64-avx2.exe \
  --games 100 --engine-nodes 50000 --sf-nodes 50000 -c 8 --no-sprt
```

不指定限制時，雙方都預設 50,000 nodes。

非對稱校準或混合限制：

```bash
python tools/match_runner.py --stages \
  "20:n50000:n10000,20:n100000:n20000,20:d12:d10" -c 4 --no-sprt
```

整批 stages 共用同一個 local context 池和同一批 Stockfish 行程。

資源、執行緒與裁決選項：

- `--tt-mb 128`：我方每 context 的 TT，預設 128 MB。
- `--sf-hash-mb 128`：需要時明確設定每個 Stockfish 行程的 Hash；未指定時維持 Stockfish 自己的預設值。
- `--engine-threads 1`：我方引擎搜尋執行緒數（預設 `1`，亦提供 `run_match_stages(..., engine_threads=1)`）；`inprocess` 模式傳給 `InProcessEngineFactory(threads=...)`，`uci` 模式轉發 `Threads` 與 `Hash` 選項。
- `--sf-threads 1`：每個 Stockfish 行程的 threads；`-c > 1` 時建議維持 1，避免觸發 CPU 超額訂閱警告並干擾計時／NPS。
- `--engine-backend uci`：自訂我方 UCI 程式時的相容模式。
- `--adjudicate / --no-adjudicate`、`--adjudicate-win-score 500`、`--adjudicate-win-moves 5`（10 plies）、`--adjudicate-draw-score 8`、`--adjudicate-draw-min-move 40`、`--adjudicate-draw-moves 6`（12 plies）：與 `tournament.py` 完全對齊的提前裁決旗標（亦支援 `run_match_stages(...)` 參數），啟動橫幅會列出目前裁決設定。

多執行緒對外部 Stockfish 固定時間測試範例：

```bash
python tools/match_runner.py --sf external/stockfish/stockfish-windows-x86-64-avx2.exe \
  --games 40 --engine-threads 4 --sf-threads 4 --engine-time 1000 --sf-time 1000 -c 1 --no-sprt
```

Stockfish 校準通常應加 `--no-sprt`，因為測量目標常是完整比分而不是判斷「我方是否比 Stockfish +10 Elo」。`match_runner.py` 的 early trash 預設關閉，需要時才加 `--early-trash`。

## 7. Cache 與 timeout

正常執行不會自動刪除 Numba cache。只有確認原始碼／依賴快取失效或要做冷啟動實驗時才使用：

```bash
python tools/tournament.py --clear-cache --nodes 50000
```

UCI backend 使用非阻塞 stdout reader，並用 `--search-timeout` 防止外部引擎永久卡住。in-process 固定時間則由搜尋本身的 stop flag 控制。

## 8. 實務建議

1. 棋力 A/B：固定 nodes、成對開局、固定 seed。
2. 最大吞吐量：逐步提高 `-c`，直到 games/hour 不再增加。
3. 純 NPS：另用 `-c 1`、固定 position/節點，多次取中位數。
4. 時間制：保留至少一個實體核心給作業系統，避免背景程式與 CPU 降頻。
5. 不要把固定 depth 的勝率直接解讀成等成本棋力差。
