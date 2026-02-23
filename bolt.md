# Bolt's Journal - Python Chess Engine Performance

## 2025-05-22 - Initial Baseline
**Learning:** The engine uses Numba for JIT compilation. Initial search on Start Position is heavily impacted by JIT overhead (~27k NPS). Subsequent searches like Kiwipete show true performance (~1M NPS).
**Action:** Always use a warmed-up benchmark for measuring optimizations.

**Baseline NPS (Kiwipete):** ~1,000,000

## 2025-05-22 - Redundant Piece Lookups in Move Scoring
**Learning:** `score_moves` and `see_ge` were both independently looking up the aggressor and victim piece types using `find_piece_type_on_square_side`, which loops over 6 bitboards. By looking them up once in `score_moves` and passing them to a refactored `see_ge`, we save redundant work in the search hot path.
**Action:** Always reuse piece type information when multiple functions need it for the same move.
**Impact:** ~3% NPS improvement on Kiwipete benchmark.
