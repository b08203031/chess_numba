---
description: "Numba nopython bans, bitboard array types, JIT decorator defaults, cache limits, and NNUE/search exceptions."
trigger: always_on
---

# Numba JIT Constraints

Hot-path code under `chess_engine/**` compiles with `@numba.njit` / `@numba.jit(nopython=True)`. Violations fail compile or silently destroy NPS.

**Full experiment log, measurements, and future roadmap:**  
[`chess_engine/classical/JIT_COMPILE_CACHE_REPORT.md`](../../chess_engine/classical/JIT_COMPILE_CACHE_REPORT.md)  
（後續 JIT / disk cache 工作請讀該檔並擴寫量測。）

---

## 1. Forbidden inside `njit` (MUST NOT)

* Custom Python classes / dynamic attributes (except supported **StructRef** / existing patterns already in the project)
* **New jitclass** for types passed into cached hot paths — jitclass type names embed process `id(class)` and **break cross-process disk cache** (see JIT report §4.2)
* Heterogeneous lists/dicts/sets; dict/set/generator comprehensions
* Nested “reflected” Python lists — use NumPy arrays
* `try` / `except` / `finally`
* Rebinding the same name to different types across branches
* Non-Numba libraries (pandas, network I/O, pickle, etc.)

---

## 2. Project data model (MUST)

* `piece_bbs`: `np.uint64` length **12**
* `occupancy_bbs`: `np.uint64` length **3**
* Bit shifts: `np.uint64(1) << square` (avoid default int32 overflow)
* Index game-state fields via named constants (`PAWN_KEY_INDEX`, etc.), not magic numbers
* **Search context:** `SearchContext` / `search_context_type` is a **StructRef** (`engine_types.py`), not jitclass

---

## 3. Disk cache & global arrays (MUST)

Numba refuses to cache functions that freeze globals as **dynamic addresses** when:

* a closed-over ndarray is **not C/F contiguous**, or
* **`nbytes > 1_000_000`** (hard limit in Numba `make_constant_array`)

Then every transitive `cache=True` caller may also fail with  
`Cannot cache ... dynamic globals (such as ... large global arrays)`.

**Rules:**

* Rook magic tables stay **packed** (`ROOK_ATTACKS` + `ROOK_ATTACK_OFFSETS`, ~819 KiB). **Do not** restore dense `64×4096` (~2 MiB) layouts.
* Before adding a module-level table used inside `njit`, compute `size = numel * itemsize` and keep **≤ 1e6**, or pass the buffer as an argument / split tables.
* Prefer explicit signatures on small table lookups (e.g. `get_rook_attacks(int64, uint64)`) to avoid `Literal[int]` specialization storms.

---

## 4. Decorators

**Default for performance-critical classical / shared hot paths:**

```python
@numba.njit(cache=True, boundscheck=False, fastmath=True)
```

* Use `inline='always'` on tiny helpers (e.g. piece-on-square lookups).
* **Do not** `inline='always'` helpers whose sole job is to **kill literals** (`_b_false` / `_i32` style) — inlining re-materializes `Literal[...]` and multiplies IR.
* **Recursive search & Lazy SMP** (`chess_engine/classical/search.py`: `_search`, `quiescence_search`, `_lazy_smp_helper_loop_jit`): use **`cache=True`** with explicit signatures and strict call-site signature convergence (`_i32` / `_u16` / `_b_true` / `_b_false` + `SearchContextType.preprocess_fields` `types.unliteral`) so recursive lowering emits `call_internal` with `0` `.numba.unresolved$` symbols. Whenever recursive signatures or call sites change, re-verify `0` `.numba.unresolved$` and a clean cross-process warm reload before merging (Numba `CPUCodeLibrary._unserialize` skips `RuntimeLinker` scanning on cached `'object'` reloads, so any signature divergence re-introducing `.numba.unresolved$` aborts on warm load).
* **Exception — NNUE weight-backed inference** (`chess_engine/nnue/ml_eval/inference.py` and similar): prefer `cache=False` when functions close over or bind reloadable global weight arrays, so stale disk caches cannot pin old nets. Document any new exception next to the decorator.

---

## 5. Layering (MUST)

* Keep UCI / GUI / PGN / file I/O in pure Python.
* Keep make/unmake, movegen, search, eval, SEE, NNUE forward under `njit`.
* **Do not return structured-array records across the Python/JIT boundary:** Calling a `@njit` function that returns a structured NumPy element (e.g. `probe_tt` returning `tt[idx]` / `_EMPTY_TT_ENTRY`) from pure Python triggers `np.record` boxing issues. Pure-Python callers must use a scalar-tuple `@njit` wrapper (e.g. `probe_tt_best_move`).
* **Decouple non-search unit tests from `search.py`:** Even though `classical/` search, QS, and Lazy SMP now use `cache=True` (~8.9 s warm import), cold compilation after invalidating `search.py` or `engine_types.py` still takes several minutes (and `classical_old/` / `nnue/` paths differ). Construct `SearchContext` directly from `engine_types` in fast eval/unit tests instead of pulling in `search.py` via `tools.match_core` or `tune_eval_match.inprocess_match`.
* After JIT/cache-related changes: run cold+warm timing (see JIT report §5) and append numbers to that report §6 when results change.
