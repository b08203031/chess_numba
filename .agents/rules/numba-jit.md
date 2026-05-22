---
description: "Standard and project-specific Numba JIT constraints and best practices for high-performance code compilation."
trigger: always_on
---

# Numba JIT & Performance Constraints

To ensure that this high-performance chess engine compiles successfully without runtime errors, all AI assistants must strictly adhere to the following Numba JIT compilation rules and performance invariants.

---

## 1. Standard JIT Restrictions (nopython=True)

Numba's `@numba.njit` compiles Python functions directly to machine code, completely bypassing the Python interpreter. The following Python features are **STRICTLY PROHIBITED** inside JIT-compiled functions:

*   **No OOP or Dynamic Objects**:
    *   Do **NOT** instantiate custom Python classes inside `njit` functions.
    *   Do **NOT** read or write dynamic object attributes that are not explicitly supported by Numba.
*   **Container Restrictions (Lists & Dicts)**:
    *   **No Heterogeneous Containers**: Lists, sets, and dicts must have completely homogeneous element types (e.g., `{1, 2.5}` will cause compilation failure).
    *   **No Advanced Comprehensions**: Basic list comprehensions are supported, but dict, set, and generator comprehensions are **strictly forbidden**.
    *   **No Reflected Lists (List-of-lists)**: Passing arbitrary nested Python lists into JIT code triggers extremely slow reflection logic. Use multi-dimensional NumPy arrays instead.
*   **Language & Library Exclusions**:
    *   **No Exception Handling**: `try...except` and `try...finally` blocks are unsupported and must be completely omitted.
    *   **No Dynamic Typing**: All variable types must be statically inferred at compilation time. Do **NOT** assign values of different types to the same variable name across different branches.
    *   **No Unsupported External Libraries**: Third-party libraries like pandas, standard pickle, or network-bound modules cannot be called inside JIT functions.

---

## 2. Project-Specific Performance Standards

This engine utilizes highly optimized static arrays and bitwise operations. The following data models and compilation options must be maintained without exception:

*   **Board & State Representation**:
    *   The bitboard representing the pieces (`piece_bbs`) **must** be a `np.uint64` array of size 12.
    *   The occupancy bitboard (`occupancy_bbs`) **must** be a `np.uint64` array of size 3.
    *   Always cast bitwise shifts safely using `np.uint64(1) << square` to prevent default `int32` overflows.
    *   Indices of state arrays (like `game_state`) **must** be accessed using pre-defined constants (e.g., `PAWN_KEY_INDEX`, `MINOR_KEY_INDEX`) to ensure static type inference.
*   **JIT Compilation Decorators & Caching**:
    *   All performance-critical functions (move generation, board operations, minimax search, evaluation helpers) **must** be decorated with:
        `@numba.njit(cache=True, boundscheck=False, fastmath=True)`
    *   `cache=True` is required to avoid expensive re-compilation overhead upon startup.
    *   `inline='always'` should be applied to high-frequency micro-helpers (e.g., square-to-piece lookups).
*   **Architectural Separation of Concerns**:
    *   Completely decouple high-performance calculation logic (board state mutations, search trees, NNUE inference) from the control and UI layers (UCI protocol parsing, GUI assistant, PGN loaders).
    *   The control layer remains pure Python for dynamic flexibility, while the computational core must run entirely under `njit`.
