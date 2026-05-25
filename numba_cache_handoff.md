# Chess Numba Engine — JIT 快取持久化開發交接說明書 (Handoff & Technical Report)

此文檔旨在為接下來接手的 AI 助手提供完整的專案現況、核心研究發現、解決方案設計，以及後續的執行步驟，以實現西洋棋引擎啟動時間從 **5 分鐘縮短至 0.1 秒** 的目標。

---

## 1. 目前現況與問題背景 (Context & Background)

* **核心痛點**：每次啟動 `main.py`、`assistant.py` 或進行搜尋測試時，引擎都需要花費將近 **5 分鐘** 進行 Numba JIT 編譯，即便在 `boundscheck=False` 與 `fastmath=True` 的情況下，每次重啟 Python 進程仍會重新編譯，完全無法重用快取。
* **原先的嘗試**：先前曾嘗試過將所有 `@numba.njit` 函數加上 `cache=True`，並在啟動時執行一次 Depth-10 的搜尋來暖機，但依然無效，每次重啟仍然強制全編譯。
* **本次任務目標**：徹底修復 Numba 快取失效的問題，使得第二次啟動時能實現**亞秒級 (sub-second) 瞬間載入**，同時保證 NPS (每秒搜尋節點數) 與搜尋正確性完全不受損害。

---

## 2. 核心研究與發現 (Key Findings & Root Cause)

透過使用 Numba 內部除錯環境變數 `NUMBA_DEBUG_CACHE=1` 進行追蹤，我們發現了以下深層原因：

### 根本原因：Numba JITClass 跨進程型態不一致 (Type Invalidation)
1. 引擎核心將預計算的靜態攻擊表封裝在一個名為 `AttackTables` 的 JITClass 中（定義於 `engine_types.py`，實例化為 `GLOBAL_ATTACK_TABLES`）。
2. 在 Python 每次啟動並導入模組時，`@jitclass` 裝飾器都會在記憶體中**動態創建一個新的類別型態對象**。
3. 因為這個型態對象在每一次 Python 進程重啟後的內部識別碼 (Identity/Hash) 都不同，Numba 在載入磁碟上的 JIT 快取檔案 (`.nbi` / `.nbc`) 時，會認為輸入參數 `attack_tables` 的型態發生了改變，進而判定快取失效 (Type Mismatch)。
4. 這會觸發**連鎖重新編譯**：所有接收 `attack_tables` 作為參數的 JIT 函數（涵蓋 `move_generator.py`、`evaluation.py`、`see.py` 等 50 多個核心運算函數）都必須重新編譯。這就是 5 分鐘編譯時間的萬惡之源！

### 實驗驗證與突破
我們在專案根目錄下建立了測試腳本 `scratch_test_numba_namedtuple.py` 進行了對比測試：
* **使用 NamedTuple / JITClass**：每次導入皆會生成新的動態型態，導致 `[cache] index saved to` 與編譯新重載 (Overload)。
* **使用標準 Python `tuple`**：Numba 將其識別為穩定的內建型態 `Tuple(Array(uint64, 2, 'C'), ...)`。在跨進程重啟後，Numba 能 **100% 成功命中快取**，輸出：
  ```
  [cache] index loaded from ...
  [cache] data loaded from ...
  ```
  載入時間從編譯的數百毫秒降到近乎為零。

---

## 3. 解決方案設計 (Proposed Design)

為了讓所有核心計算函數能完美命中快取，我們必須將 `AttackTables` 從動態的 JITClass 重構為**靜態的標準元組 (Tuple)**。

### A. 索引常量化 (Index Constants)
在 `engine_types.py` 中，將 `AttackTables` 的 17 個屬性對應到 tuple 的固定索引：
```python
PAWN_ATTACKS_IDX = 0
KNIGHT_ATTACKS_IDX = 1
KING_ATTACKS_IDX = 2
BISHOP_MASKS_IDX = 3
ROOK_MASKS_IDX = 4
BISHOP_MAGIC_NUMBERS_IDX = 5
ROOK_MAGIC_NUMBERS_IDX = 6
BISHOP_RELEVANT_BITS_IDX = 7
ROOK_RELEVANT_BITS_IDX = 8
BISHOP_ATTACKS_IDX = 9
ROOK_ATTACKS_1_IDX = 10
ROOK_ATTACKS_2_IDX = 11
ROOK_ATTACKS_3_IDX = 12
ROOK_ATTACKS_4_IDX = 13
SQUARES_BETWEEN_IDX = 14
ROOK_RAYS_IDX = 15
BISHOP_RAYS_IDX = 16
```

### B. 靜態元組類型定義
定義一個穩定的 `attack_tables_type` 以供 JIT 靜態簽章使用：
```python
attack_tables_type = numba.types.Tuple((
    numba.uint64[:, :],  # pawn_attacks
    numba.uint64[:],     # knight_attacks
    numba.uint64[:],     # king_attacks
    numba.uint64[:],     # bishop_masks
    numba.uint64[:],     # rook_masks
    numba.uint64[:],     # bishop_magic_numbers
    numba.uint64[:],     # rook_magic_numbers
    numba.uint8[:],      # bishop_relevant_bits
    numba.uint8[:],      # rook_relevant_bits
    numba.uint64[:, :],  # bishop_attacks
    numba.uint64[:, :],  # rook_attacks_1
    numba.uint64[:, :],  # rook_attacks_2
    numba.uint64[:, :],  # rook_attacks_3
    numba.uint64[:, :],  # rook_attacks_4
    numba.uint64[:, :],  # squares_between
    numba.uint64[:],     # rook_rays
    numba.uint64[:]      # bishop_rays
))
```

### C. 訪問改寫 (Access Refactoring)
在所有 JIT 函數中，將原先對 `attack_tables.field` 的點運算子訪問，改為對元組的索引訪問，例如：
* `attack_tables.pawn_attacks` ➡️ `attack_tables[PAWN_ATTACKS_IDX]`
* `attack_tables.knight_attacks` ➡️ `attack_tables[KNIGHT_ATTACKS_IDX]`

此改動對機器碼生成**完全透明**，LLVM 會在編譯時將其優化為直接指針偏移動作，性能完全相同。

---

## 4. 下一步執行計劃 (Next Steps)

接手的 AI 助手請按照以下步驟進行修改與驗證：

### 步驟 1：修改 `chess_engine/classical/engine_types.py`
1. 移除 `@jitclass` 的 `class AttackTables` 定義。
2. 加入 17 個 `_IDX` 索引常量（如上所示）。
3. 定義並匯出 `attack_tables_type`。

### 步驟 2：修改 `chess_engine/classical/move_generator.py`
1. 將 `GLOBAL_ATTACK_TABLES` 修改為直接定義的 Python 元組：
   ```python
   GLOBAL_ATTACK_TABLES = (
       PAWN_ATTACKS, KNIGHT_ATTACKS, KING_ATTACKS,
       BISHOP_MASKS, ROOK_MASKS, BISHOP_MAGIC_NUMBERS, ROOK_MAGIC_NUMBERS,
       BISHOP_RELEVANT_BITS, ROOK_RELEVANT_BITS,
       BISHOP_ATTACKS, ROOK_ATTACKS_1, ROOK_ATTACKS_2, ROOK_ATTACKS_3, ROOK_ATTACKS_4,
       SQUARES_BETWEEN, ROOK_RAYS, BISHOP_RAYS
   )
   ```
2. 匯入 17 個 `_IDX` 常量。
3. 將此檔案中所有的 `attack_tables.field` 替換為 `attack_tables[FIELD_IDX]`。

### 步驟 3：修改 `evaluation.py`、`see.py` 與 `search_heuristics.py`
1. 匯入所需的 `_IDX` 常量。
2. 替換這些檔案中的 `attack_tables.field` 為對應的索引訪問（總計約 51 處，可使用指令進行正則替換或手動替換）。

### 步驟 4：執行快取驗證
1. 執行搜尋測試（第一次編譯快取）：
   ```bash
   python -m tests.test_search
   ```
2. 第二次執行搜尋測試，觀察載入時間。如果成功，引擎將在 **1 秒內** 完成暖機與測試。
3. 您可以開啟偵錯選項來確認快取 hit 狀態：
   ```bash
   # Windows PowerShell
   $env:NUMBA_DEBUG_CACHE="1"
   python -m tests.test_search
   ```

### 步驟 5：運行基準測試驗證 NPS
1. 執行 `python -m tests.benchmark` 確保重構前後的 NPS (Nodes Per Second) 保持一致，沒有任何效能衰退。
