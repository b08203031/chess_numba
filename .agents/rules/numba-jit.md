---
description: "Numba nopython bans, bitboard array types, JIT decorator defaults and NNUE cache exceptions."
trigger: always_on
---

# Numba JIT Constraints

Hot-path code under `chess_engine/**` compiles with `@numba.njit` / `@numba.jit(nopython=True)`. Violations fail compile or silently destroy NPS.

---

## 1. Forbidden inside `njit` (MUST NOT)

* Custom Python classes / dynamic attributes (except supported `jitclass` patterns already in the project)
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

---

## 3. Decorators

**Default for performance-critical classical / shared hot paths:**

```python
@numba.njit(cache=True, boundscheck=False, fastmath=True)
```

* Use `inline='always'` on tiny helpers (e.g. piece-on-square lookups).
* **Exception — NNUE weight-backed inference** (`chess_engine/nnue/ml_eval/inference.py` and similar): prefer `cache=False` when functions close over or bind reloadable global weight arrays, so stale disk caches cannot pin old nets. Document any new exception next to the decorator.

---

## 4. Layering (MUST)

* Keep UCI / GUI / PGN / file I/O in pure Python.
* Keep make/unmake, movegen, search, eval, SEE, NNUE forward under `njit`.
