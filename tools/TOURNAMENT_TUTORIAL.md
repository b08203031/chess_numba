# 引擎對戰與 Stockfish Match 指南

本專案有兩個對戰入口：

- `tools/tournament.py`：現行 `classical` 對基準 `classical_old`。
- `tools/match_runner.py`：現行引擎對外部 Stockfish UCI 執行檔。

兩者共用 `tools/match_core.py` 的裁判、PGN、SPRT、開局配對與工作池實作。預設搜尋限制是每步固定 `50,000` nodes；固定深度與固定時間仍保留。

兩支程式在正式對局前會依序執行兩層暖機：先讓每個 Python/Numba backend 搜尋一次 Kiwipete、固定 `depth 14`；再從標準初始局面進行一場完整的 depth-1 暖機局。d1 暖機局的雙方每步都固定 `d1`。所有暖機結果與 PGN 都不列入比分、SPRT 或正式輸出。in-process backend 的 Kiwipete 搜尋與 d1 對局只需使用第一組共享 JIT 的 context；若明確使用 Python UCI backend，彼此不共享 JIT 的 worker 子行程會各自執行相同暖機。外部 native Stockfish 不額外執行 Kiwipete 搜尋，但會參與 d1 暖機對局。

第一組 worker 的 Kiwipete 暖機會即時列出雙方每個完成深度的 UCI `info`（depth、score、nodes、nps、PV）；後續平行 worker 仍保持安靜，避免暖機訊息重複刷屏。這些診斷行只供觀察，不會改變搜尋限制或正式對局結果。

## 1. 為什麼改成 in-process

舊版 `tournament.py -c 8` 會建立 8 個 New 與 8 個 Old Python 子行程。Numba 的遞迴 `_search` / quiescence 目前必須使用 `cache=False`，因此主要搜尋函數會在 16 個行程內分別編譯。

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

### 多階段

```bash
python tools/tournament.py -c 8 --no-sprt \
  --stages "200:n20000,400:n100000,200:d12"
```

限制 token：

- `n50000`：固定 50,000 nodes。
- `d12`：固定 depth 12。
- `1000`：固定 1,000 ms/move。

所有階段共用同一批 contexts，階段切換不會重新 JIT 或重新配置工作池；每盤開始仍會清空該 context 的 TT/history，維持對局獨立性。

### Backend 選擇

預設 `--backend auto`：

- `main.py` 對 `main_old.py`：選擇 `inprocess`。
- 自訂任意 UCI 路徑：退回 `uci` 相容模式。

可以明確指定：

```bash
python tools/tournament.py --backend inprocess
python tools/tournament.py --backend uci
```

in-process backend 使用：

- `--engine1-module chess_engine.classical`
- `--engine2-module chess_engine.classical_old`

`--own-book1/--own-book2` 只支援 UCI backend。一般 A/B 測試應關閉引擎自己的隨機 book，改用裁判提供的成對開局。

## 3. 記憶體與 TT

`--tt-mb` 預設維持 `128`，而且是「每個引擎 context」128 MB：

```bash
python tools/tournament.py -c 8 --tt-mb 128 --nodes 100000
```

`-c 8` 會建立 8 個 New + 8 個 Old contexts，單計 TT 約 2 GB；另外還有 continuation history、pawn history、pawn cache 等大型陣列。16 GB 系統可先使用 `-c 8`，若出現大量換頁、整機記憶體逼近上限或 NPS 明顯下降，再降低 concurrency，而不是縮小 TT。

`-c` 建議不要超過實體核心數。每盤只有輪到走棋的一方搜尋，因此 `-c N` 最多約有 N 個同時進行的搜尋。

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
- PGN 每步註解包含 depth、side-to-move evaluation、nodes、time 與 NPS。
- engine crash、timeout、無 bestmove、非法著或 backend 狀態不同步都會明確判負，不會偷偷改走第一個合法著。
- failure PGN 會附帶 `Failure` 與 `FailurePly` header，保存 bounded exception 訊息與發生回合；統計時應剔除該 failure 所在的整組換色配對。
- 預設最長 400 plies；可用 `--max-plies` 修改，超過時裁定和棋並寫入 PGN Termination。

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

資源選項：

- `--tt-mb 128`：我方每 context 的 TT，預設 128 MB。
- `--sf-hash-mb 128`：需要時明確設定每個 Stockfish 行程的 Hash；未指定時維持 Stockfish 自己的預設值。
- `--sf-threads 1`：每個 Stockfish 行程的 threads；`-c > 1` 時建議維持 1，避免超額使用核心。
- `--engine-backend uci`：自訂我方 UCI 程式時的相容模式。

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
