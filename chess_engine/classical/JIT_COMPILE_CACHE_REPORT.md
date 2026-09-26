# Classical 引擎 Numba JIT 編譯與快取實驗報告

| 項目 | 內容 |
| :--- | :--- |
| 範圍 | `chess_engine/classical` only |
| 狀態 | **進行中的技術主文件**；後續 JIT / cache 工作請**參考並擴寫本檔** |
| 最後更新 | 2026-07-17 |
| 相關規則 | [`.agents/rules/numba-jit.md`](../../.agents/rules/numba-jit.md)、[`.agents/rules/workflow.md`](../../.agents/rules/workflow.md) |
| 量測腳本 | `scratch/jit_cache_experiment.py`、`scratch/_jit_warm_cold_once.py`、`scratch/_ctx_structref_smoke.py` |

---

## 1. 問題陳述

### 1.1 症狀

啟動 classical 引擎（`main.py` / 任何會 import `search` 並觸發搜尋的路徑）時：

* **首次**（或 cache 失效後）等待約 **數分鐘～近 10 分鐘**，才進入可下達 UCI `go` 的狀態。
* 使用者觀感：每次重開 Python process 都像「重新編譯整個引擎」。
* 同 process 內第二次搜尋通常很快（已在記憶體中）。

### 1.2 目標

1. 找出 **disk cache 失效 / 無法寫入** 的根因並修復可修部分。
2. 降低 **cold compile**（清 cache 後）與 **warm start**（新 process、有 cache）時間。
3. 留下可重跑的量測方法與可擴寫的結論，供後續工作延續。

### 1.3 名詞

| 名詞 | 定義 |
| :--- | :--- |
| **Cold** | 刪除 `chess_engine/classical/__pycache__` 內 `.nbi` / `.nbc` 後，新 process 第一次編譯 |
| **Warm** | 不刪 cache，**新 process** 再跑一次（測 disk cache 是否命中） |
| **In-process 2nd** | 同一 process 內第二次搜尋（測記憶體內是否已就緒） |
| **Eager compile** | `@njit(signature, ...)` 在 **import / 裝飾時** 立即編譯 |
| **Lazy compile** | 無完整 signature 時，於**第一次呼叫**才編譯 |

---

## 2. Numba 快取機制（與本專案相關）

### 2.1 寫入條件

* `@numba.njit(cache=True)`（或 `jit(nopython=True, cache=True)`）。
* 產物位於模組旁 `__pycache__/`：
  * `*.nbi`：overload 索引（signature → data file）。
  * `*.nbc`：序列化後的編譯結果（可達數 MB / 函式）。

### 2.2 無法 cache：dynamic globals（含「過大全域陣列」）

Numba 源碼（`numba/core/base.py` → `make_constant_array`）規則摘要：

```text
size_limit = 10**6   # 1_000_000 bytes
若陣列非 C/F 連續 或  nbytes > size_limit:
    → 改以 runtime 指標嵌入 (numba.dynamic.globals.*)
    → library.has_dynamic_globals = True
    → cache 拒絕寫入，並警告:
      Cannot cache compiled function "..." as it uses dynamic globals
      (such as ctypes pointers and large global arrays)
```

**後果：** 不只該函式不能 cache；任何 **transitive 依賴** 它、且同樣 `cache=True` 的函式，在 finalize 時也可能帶上 dynamic globals，導致 **整條 movegen / eval / search 鏈** 無法落盤。

### 2.3 可以寫入但跨 process 打不中

Cache key 含 **參數型別字串**。若型別名稱含 **process 專屬 ID**（見 §4.2 jitclass），則：

* Process A 寫入 `.nbc`；
* Process B 的型別字串不同 → **lookup miss** → 視為未編譯 → 再編一次。

磁碟上會累積多份幾乎相同的 `.nbc`（不同 key）。

### 2.4 Eager signature 與啟動時間的關係

```python
@njit(ret_type(arg_types...), cache=True)
def f(...):
    ...
```

有完整 signature 時，Numba 在 **裝飾時（通常即 import）** 就編譯。  
因此本專案中 **`import chess_engine.classical.search` 可佔用整段 cold/warm 的絕大部分時間**，而不是第一次 `iterative_deepening_search` 呼叫。

---

## 3. 根因總覽（按影響）

| # | 根因 | 影響 | 狀態（2026-07-17） |
| :--- | :--- | :--- | :--- |
| R1 | Rook magic 表固定 `(64, 4096)` → **2 097 152 B > 1e6** | 全鏈 `Cannot cache` | **已修**（packed） |
| R2 | `SearchContext` 為 **jitclass**，type 名含 `#id(class)` | 跨 process search cache 必 miss | **已改 StructRef** |
| R3 | `_search` 對 `Literal[bool]` / `int64` 等 **特化爆炸**（每份 ~5.5 MB IR） | Cold 時間 ×N | **已壓到 1 overload** |
| R4 | `get_rook_attacks` 等對 `Literal[int](sq)` 特化 | 多餘 `.nbc`、編譯量增加 | **已修**（顯式 signature） |
| R5 | 遞迴函式 + 大 StructRef signature：**disk cache 載入 LLVM unresolved** | `_search` / `quiescence` 無法安全 `cache=True` | **緩解**：該對設 `cache=False` |
| R6 | 單一 `_search` IR 仍巨大（~5 MB 級） | 即使 1 overload，cold 仍要數分鐘 | **未消**（結構性） |

---

## 4. 詳細根因與修復

### 4.1 R1：Rook attack 表過大（2 MB）

#### 現象

```text
ROOK_ATTACKS_FLAT.shape = (64 * 4096,)   # 或 (64, 4096)
nbytes = 64 * 4096 * 8 = 2_097_152 > 10**6
```

編譯日誌（歷史）出現大量：

```text
Cannot cache compiled function "get_rook_attacks" as it uses dynamic globals ...
Cannot cache compiled function "is_square_attacked" ...
Cannot cache compiled function "_search" ...
```

`__pycache__` 中往往只有小工具函式的 `.nbc`，沒有完整 search/eval 鏈。

#### 修復：Packed magic table

檔案：`chess_engine/classical/move_generator.py`

* 每格只配置 `2 ** ROOK_RELEVANT_BITS[sq]` 個 slot。
* 全盤 `sum(2^bits) = 102400` entries → **819 200 B < 1e6**。
* 查表：

```text
index = ROOK_ATTACK_OFFSETS[sq] + magic_index
attacks = ROOK_ATTACKS[index]
```

| 項目 | 舊 | 新 |
| :--- | ---: | ---: |
| 佈局 | 固定 64×4096 | packed + offsets |
| 大小 | 2 097 152 B | **819 200 B** |
| 對 on-the-fly 正確性 | — | 隨機 occupancy **0 mismatch** |
| `get_rook_attacks` cannot_cache | 是 | **否** |

**注意：** 不要再 flatten 回 2 MB 的 dense 表；任何「為了微優化 NPS」而恢復 dense 表都會再次摧毀 cache。

#### 順帶：顯式攻擊函式簽名

```python
@njit(uint64(int64, uint64), cache=True, ...)
def get_rook_attacks(sq, occ): ...
```

避免呼叫點傳入常數格（王車易位等）時產生 `Literal[int](4)`、`Literal[int](60)` 等 **十餘個** 同義 overload。

---

### 4.2 R2：jitclass `SearchContext` 跨 process 不可 cache

#### 現象

Cache index 中的 signature 片段：

```text
instance.jitclass._SearchContextJIT#19cc94acc70<transposition_table:...>
instance.jitclass._SearchContextJIT#1e06f4c94e0<...>   # 另一 process
```

`#` 後為 **該 process 的 class 物件 id**。  
故 **disk cache 對所有「參數含 SearchContext」的函式在跨 process 時幾乎永遠 miss**。

量測（修復 rook 後、仍用 jitclass 時）：

| 階段 | 時間 | 說明 |
| :--- | ---: | :--- |
| Cold profile | ~463 s | 其中 `search_depth1` ~419 s |
| Warm profile | ~369 s | **幾乎不降** → 非「沒寫 cache」，是 **key 打不中** |
| In-process 2nd search | ~0.01–0.3 s | 同 process 正常 |

Import 拆解（有舊 cache 時）曾出現：

```text
evaluation      ~2.5 s
search import ~350–460 s   ← 幾乎全是 eager 重編 search 圖
```

#### 修復：StructRef

檔案：`chess_engine/classical/engine_types.py`

* `SearchContext` 改為 `numba.experimental.structref`。
* `typeof(ctx)` 跨 process 字串 **穩定**（以欄位名 + 型別表達，**不含** process id）。
* 實驗（`scratch/_structref_cache_probe.py`）：兩 process 的 `typeof` 字串一致。
* 純 Python 端屬性（UCI `stop_flag`、history aging 的 `np.divide(ctx.history_table, ...)`）需 **手動 getter/setter**（StructRef 不會自動 mirror 到 Python）；以 `exec` 產生 `@njit(cache=False)` 小 helper 掛在 proxy 上。

欄位 array 一律用 **C-contiguous** 型別（`[:, ::1]` 等），使 `typeof(實例) == search_context_type`（簽名用固定 type），避免 layout `A` vs `C` 再特化。

#### 副作用與決策

將 `SearchContext` 改 StructRef 後，對 **遞迴** `_search` / `quiescence_search` 啟用 `cache=True` 時，warm load 出現：

```text
LLVM ERROR: Symbol not found: .numba.unresolved$...quiescence_search...SearchContextType(...)
```

判定為 **Numba 對「遞迴 + 極大 StructRef 簽名」disk cache 反序列化 / 符號連結** 的問題（見 R5）。  
決策：

* **`_search`、`quiescence_search`：`cache=False`**（每 process 仍重編這一對，但只 1 份 overload）。
* 其餘吃 `search_context_type` 的 helper（`get_next_move`、heuristics、部分 eval 路徑等）維持 **`cache=True`**，warm 可命中。

---

### 4.3 R3 / R4：特化爆炸

#### `_search` 的 Literal / 寬整數

即使裝飾器寫了：

```python
@njit(search_return_type(..., int32, int32, int32, search_context_type, int32, uint16, boolean, boolean), ...)
```

遞迴呼叫若直接傳 `True` / `False` / `depth - 1`（推成 int64），Numba 仍可能為 **Literal[bool]** 與 **int64** 再生出多份完整 IR（曾見 **6～18** 個 overload，單份 ~5.5 MB）。

#### 修復模式（必須 `inline='never'`）

```python
@njit(boolean(), cache=True, inline='never')
def _b_false():
    return False

@njit(boolean(), cache=True, inline='never')
def _b_true():
    return True

@njit(int32(int64), cache=True, inline='never')
def _i32(x):
    return np.int32(x)

@njit(uint16(int64), cache=True, inline='never')
def _u16(x):
    return np.uint16(x)
```

**關鍵教訓：** `inline='always'` 會把 `return False` 再內聯回呼叫點 → **Literal 復現** → 特化爆炸回來。

所有遞迴 `_search(...)` 與 root entry 改為：

```text
_search(..., _i32(depth-…), _i32(-beta), …, _i32(ply+1), _u16(NO_MOVE), _b_false(), not cut_node)
```

並去掉依賴 default 參數的呼叫 arity 差異。

#### 回傳型別

`search_return_type` 第一元為 `int32`。`max_eval = -INFINITY`（Python int）會讓整條路徑變成 int64，與 signature 衝突（StructRef 切換後更容易在 lowering 爆出）。  
統一：

* `max_eval` 以 `np.int32(-INFINITY)` 初始化；
* 出口 `return (np.int32(max_eval), ...)` / `np.int32(tt_score)` 等。

#### 結果

```text
search._search-*.nbi  n_overloads = 1
  [0] … int32 … uint16 bool bool …   # 無 Literal[bool]
```

---

### 4.4 R5：遞迴 + disk cache + 大 StructRef

| 嘗試 | 結果 |
| :--- | :--- |
| StructRef + `_search`/`quiescence` `cache=True` | Cold 可寫入；**Warm load 進程直接 LLVM ERROR 崩潰** |
| 同上但 `cache=False` | Cold/Warm 皆穩定；Warm 仍需重編此對 |

**現行策略（务必写入 AGENTS / 本檔）：**

```text
_search, quiescence_search  →  cache=False
其餘 classical 熱路徑       →  cache=True（預設）
```

未來若要重新開啟二者 cache，必須先有 **可重現的 warm load 測試**（見 §7），通過後才能合併。

---

## 5. 實驗方法論

### 5.1 環境

* Python **3.10**，Numba **0.61.2**（量測時）。
* OS：Windows；CPU feature string 會寫進 cache key（換機器 / 換 CPU flags 會整包失效）。
* 工作目錄：倉庫根；`PYTHONPATH=.`。

### 5.2 腳本

| 腳本 | 用途 |
| :--- | :--- |
| `scratch/jit_cache_experiment.py` | light / profile / suite：cold-warm 子行程、cannot_cache 收集、large globals |
| `scratch/_jit_warm_cold_once.py` | 單次 cold 或 warm：import / ctx / search1 / search2、overload 數 |
| `scratch/_ctx_structref_smoke.py` | StructRef 建構、`typeof == search_context_type`、njit 突變 |
| `scratch/_structref_cache_probe.py` | 迷你 StructRef 跨 process `typeof` 穩定性 |

清除 classical numba cache（勿整庫亂刪除非有意）：

```text
刪除 chess_engine/classical/__pycache__ 下 *.nbi / *.nbc
# 或使用 jit_cache_experiment.clear_classical_numba_cache()
```

### 5.3 建議標準量測

```bash
# Cold
python scratch/_jit_warm_cold_once.py --clear --label cold

# Warm（緊接上一指令，新 process）
python scratch/_jit_warm_cold_once.py --label warm
```

紀錄欄位至少包含：

* `import_s`、`ctx_s`、`search1_s`、`search2_s`、`total1_s`
* `cannot_cache_count` / `uncacheable_funcs`（若開 warnings）
* `n_overloads`（`search._search-*.nbi`）
* `ROOK_ATTACKS.nbytes`、是否 `< 10**6`
* Numba / CPU 版本

### 5.4 判讀注意

1. **Eager compile：** 時間多半在 `import search`，不要只看 `search()` 計時。
2. **Warm ≠ 同 process 第二次：** 必須 **新 process**。
3. **Source stamp：** 改 `search.py` / `engine_types.py` 任一處會使該檔相關 cache 失效。
4. **CPU features：** cache key 含 feature 字串；CI 與本機可能不共用。

---

## 6. 量測結果時間線（摘要）

量測：startpos，`iterative_deepening_search(..., depth=1, no time limit)`，以「import + 第一次 depth1」為主。

### 6.1 基線（修復前，歷史）

| 狀態 | 量級 | 備註 |
| :--- | :--- | :--- |
| 使用者體感 | ~5–10 min | 含 cannot_cache 全鏈重編 |
| 日誌特徵 | 大量 Cannot cache + rook 2 MB | `coord_descent_*.log` 等 |

### 6.2 僅 packed rook 後（jitclass 仍在）

| 階段 | 時間 | cannot_cache | 備註 |
| :--- | ---: | ---: | :--- |
| Cold light（eval） | ~37 s | 0 | movegen+eval cache 恢復 |
| Warm light | ~4 s | 0 | **跨 process eval 命中** |
| Cold full profile | ~463 s | 0 | search 仍巨 |
| Warm full profile | ~369 s | 0 | **search 幾乎無收益**（jitclass id） |

### 6.3 + Literal bool 收斂（`_b_*` + `inline='never'`）

| 階段 | 時間 | `_search` overloads |
| :--- | ---: | ---: |
| Cold | ~245 s | 3（無 Literal，仍有 int 寬度差） |

### 6.4 + StructRef + `_i32` 統一 + 遞迴 `cache=False`（2026-07-17 午後）

| 階段 | import | search1 | total1 | search2 | overloads |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Cold** | ~158 s | ~0.8 s | **~161 s** | ~0.01 s | **1** |
| **Warm** | ~86 s | ~0.8 s | **~89 s** | ~0.01 s | （遞迴不落盤） |

功能 smoke：depth 3 startpos → 可正常出招（例：`e2e3` score 22）。

### 6.5 續優化實驗（同日晚間）

| 嘗試 | 結果 | 結論 |
| :--- | :--- | :--- |
| `NUMBA_OPT=1/2/3` 對 cold import_compile | 164–170 s，幾乎無差；d7 NPS 同量級 | **OPT 降級無益** |
| 遞迴 `cache=True`（含僅 QS self-rec） | Warm load **LLVM unresolved abort** | StructRef + **任何遞迴** 目前不可 cache |
| Lazy（無 decorate signature）+ 遞迴 `cache=False` | import ~4–12 s；first go ~115–180 s；warm total **更差** (~120 s) | 多 overload / 較慢特化；**總時間劣於 explicit sig** |
| Lazy helper（`get_next_move` 等去 signature，`cache=True`）+ 遞迴 **explicit sig + cache=False** | Cold import 仍被 QS/_search eager 主導 | helper lazy  alone 不夠；遞迴仍 dominate |

### 6.6 現行配置與數字（2026-07-17 晚，**採用**）

| 項目 | 設定 |
| :--- | :--- |
| Rook | packed &lt; 1 MB |
| SearchContext | StructRef |
| `_search` / `quiescence_search` | **explicit signature + `cache=False`** |
| `get_next_move` / `compute_full_corrected_*` / `apply_correction_*` | **lazy + `cache=True`**（首次搜尋時與遞迴一併編譯／命中） |
| UCI | `isready` 呼叫 `warmup_search_jit()` 一次（若 import 已 eager 編過則幾乎 no-op） |

| 階段 | import | search1 | total1 | search2 |
| :--- | ---: | ---: | ---: | ---: |
| **Cold** | ~156 s | ~0.8 s | **~158 s** | ~0.01 s |
| **Warm** | ~84 s | ~0.9 s | **~86 s** | ~0.01 s |

### 6.7 解讀

```text
原始 ~6–10 min
  → packed rook：eval/movegen warm 恢復；search 仍 ~6 min（jitclass）
  → 特化收斂：cold ~4 min
  → StructRef + 單 overload：cold ~2.5 min；warm ~1.5 min
  → 同 process 第二次：毫秒～數百毫秒
  → OPT / 遞迴 cache / 純 lazy 遞迴：未再壓低 total（部分反而變差）
```

Warm 剩餘 **~85 s** 幾乎全是 **每 process 重編 `_search` + `quiescence_search`**。  
下一步要再砍，必須 **非遞迴化** 或 **修好 Numba 遞迴+StructRef cache**（見 §8）。

---

## 7. 現行程式約束（寫 code 時的 checklist）

### 7.1 全域表

- [ ] 任何 `@njit` 會關閉的 **全域 ndarray** 必須 **`nbytes ≤ 1_000_000`** 且 C/F contiguous。
- [ ] Rook magic 維持 **packed**；禁止無打包的 64×4096 dense 表。
- [ ] 新增大表前先算 `shape.numel() * itemsize`，必要時拆表 / packed / 改參數傳入。

### 7.2 SearchContext

- [ ] 使用 **StructRef**（`engine_types.SearchContext` / `search_context_type`），**不要** 回復 jitclass。
- [ ] 新增欄位：同步 `search_context_spec`、factory 參數順序、proxy 欄位名列表；保持 C-contiguous。
- [ ] 純 Python 若需讀寫新 **純量** 欄位：加入 `_SCALAR_FIELDS`（或等效 getter/setter）。

### 7.3 遞迴與 cache

- [ ] `_search` / `quiescence_search` 維持 **`cache=False`**，除非完成 §7.4 驗證。
- [ ] 其他熱路徑預設 **`cache=True`**。
- [ ] 遞迴呼叫禁止直接傳 Python `True`/`False` 字面量進會特化的參數；用 `_b_true`/`_b_false` 或已是 `bool` 的變數（且 **不要** 把 false 常量 helper `inline='always'`）。
- [ ] 深度 / αβ / ply 進 `_search` 前經 **`_i32`**；走子編碼經 **`_u16`**。

### 7.4 若要重新開啟 `_search` disk cache（未來工作閘門）

必須全部通過：

1. `cache=True` 寫入 cold。
2. **新 process** warm load：**無** LLVM ERROR、無 abort。
3. `n_overloads` 仍為 1（或有文件化理由的小數目）。
4. depth≥3 smoke 與基線分數/走法合理。
5. 在本檔 §6 追加一列量測。

### 7.5 量測與文件

- [ ] 改動 JIT/cache 相關後，跑 §5.3 cold+warm，把數字 **追加** 到 §6。
- [ ] 重大結論同步本檔；索引見根目錄 `README.md`「專案文件導覽」。

---

## 8. 後續工作路線圖（優先序）

| 優先 | 項目 | 預期收益 | 風險 / 難度 |
| :---: | :--- | :--- | :--- |
| P0 | 維持本檔 checklist；禁止 rook dense / jitclass 回歸 | 守住現有 4× 量級收益 | 低 |
| P1 | 修復遞迴 + StructRef 的 disk cache（或升 Numba 驗證） | Warm 從 ~90 s → 數秒～十幾秒 | 中高 |
| P1b | 非遞迴 search（顯式 stack）以便安全 cache | 同上 + 可測性 | 高（大重構） |
| P2 | 拆 `_search` 為較小 njit 單元 | 縮短單次 LLVM、利於增量編譯 | 高 |
| P2 | 減少 eager 大簽名 import 成本（延遲到 first go） | UCI `uci` 回應更快；總編譯量不變 | 中（產品體驗） |
| P3 | 常駐 prewarm daemon / CI 預熱 cache artifact | 開發與對戰啟動體驗 | 運維 |
| P3 | AOT（`pycc` 等）可行性調研 | 部署形態 | 高（生態限制多） |

---

## 9. 關鍵檔案索引

| 檔案 | JIT 相關職責 |
| :--- | :--- |
| `move_generator.py` | Packed rook/bishop magic、`get_*_attacks` 簽名 |
| `engine_types.py` | StructRef `SearchContext`、`search_context_type` |
| `search.py` | `_search` / `quiescence`、`_i32`/`_b_*`、遞迴 cache 策略 |
| `evaluation.py` 等 | 依賴 attack tables / 部分 context；應維持可 cache |
| `scratch/jit_cache_experiment.py` | 系統性 cold/warm / cannot_cache |
| `scratch/_jit_warm_cold_once.py` | 快速 cold/warm 一鍵量測 |

---

## 10. 經驗法則（給後續 Agent / 開發者）

1. **先問「能不能 cache」再問「NPS」**——不能 cache 時每次重開都付滿額編譯稅。
2. **1 MB 全域表是硬上限**（Numba `10**6`），不是建議值。
3. **jitclass 不適合當跨 process 快取邊界上的參數型別**；長期狀態袋優先 StructRef / 純陣列參數。
4. **Literal 特化會複製整份巨型 IR**；遞迴 + 布林旗標尤其致命。
5. **`inline='always'` 與「去 Literal helper」互斥**（會把常量再內聯回去）。
6. **Eager `@njit(sig)` 把成本綁在 import**；看啟動時間請 profile import 圖。
7. **Warm 必須用新 process 驗證**；同 process 第二次搜尋永遠偏樂觀。
8. **遞迴 + cache 在部分 Numba 版本/簽名下會直接崩潰**——失敗時用 `cache=False` 止血，並記在本檔。
9. **改動後用 cold/warm 數字說話**，並把列追加進 §6。

---

## 11. 修訂紀錄

| 日期 | 作者 / 情境 | 變更 |
| :--- | :--- | :--- |
| 2026-07-17 | JIT 啟動時間專案（本報告初版） | R1–R5 診斷；packed rook；StructRef；特化收斂；遞迴 cache=False；cold≈161s / warm≈89s |
| 2026-07-17 | 續優化輪 | OPT 實驗無效；遞迴 cache 全面確認崩潰；lazy 遞迴總時間變差；UCI `isready` warmup；現行 cold≈158s / warm≈86s |
| 2026-07-18 | 大工程：iterative QS | 備份 `archive/jit_backup_20260718_001523/`；QS 改非遞迴 + `cache=True` + stack 在 SearchContext；d15 NPS 基線 794k → 792k（≈中性）；warm JIT ~78s（`_search` 仍遞迴 `cache=False`） |
| 2026-07-18 | **回退 iterative QS** | 對戰持平但 **NPS 偏低**；`classical` / `classical_old` / `main` / `main_old` **統一還原**為 `archive/jit_backup_20260718_001523/`（遞迴 QS + `cache=False`） |
| 2026-07-30 | 搜尋診斷 opt-in | `SearchContext` 尾端新增 `diag_enabled`；正式搜尋不再寫診斷 counters，工具明確啟用。StructRef schema 變更會使既有相關 cache 失效一次；遞迴 `_search` / QS 仍為 `cache=False`。 |

### 6.8 Iterative QSearch（2026-07-18）— 已回退

| 項目 | 內容 |
| :--- | :--- |
| 備份 | `archive/jit_backup_20260718_001523/`（search / engine_types / move_generator / main 等） |
| 作法 | `quiescence_search` 改 **stack 狀態機**（phase 0/1/2/3），**無自遞迴** |
| stack 儲存 | `SearchContext.qs_i32/u64/u16/bool/i8/u8`（勿用 module global：Numba 會 readonly） |
| decorator | `@njit(..., cache=True)` — 可落盤 |
| d15 NPS | baseline **794 086** → after **792 219**（差 **&lt;0.3%**） |
| d15 best | 皆 `e2e4`；d1–d12 info nodes **與基線一致** |
| JIT cold/warm | cold total≈158s；warm≈78s（仍由 **`_search` 遞迴重編** 主導） |

#### 對戰與回退決策

| 對戰 | 結果 | NPS（主統計 mean） |
| :--- | :--- | :--- |
| New_IterQS vs Old_classical · 300@150k · c8 | New **154.5–145.5**（+10 Elo, CI 跨 0） | New **435k** vs Old **456k**（**−4.6%**，顯著） |
| New_IterQS vs Mid_RecQS · 300@150k · c6 | New **150.5–149.5**（+1 Elo, 實質持平） | New **526k** vs Mid **539k**（**−2.4%**） |

**結論：** 強度無顯著提升，實戰 NPS 穩定偏低 → **放棄 iterative QS**，工作樹還原為備份（遞迴 `quiescence_search`，`cache=False`）。  
`classical` 與 `classical_old` 已邏輯一致（後者僅 import 路徑為 `classical_old`）。

**未完成（仍有效）：** `_search` 非遞迴化（或修好遞迴 disk cache）。完成前 warm 難以再壓到數秒。

量測腳本：`scratch/nps_d15_bench.py`、結果 JSON：`scratch/nps_d15_baseline.json`、`scratch/nps_d15_after_qs_ctx_stack.json`。  
對戰 PGN：`tournament_analysis/match_iterqs_vs_old_300_150k.pgn`、`match_iterqs_vs_mid_300_150k.pgn`。

（後續請在表格**追加**列，勿覆寫歷史數字；若方法變更請在 §5 註明。）
