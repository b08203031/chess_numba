# Phase 4: Aggressor & Victim Cache Optimization

To further optimize the classical chess engine's Nodes Per Second (NPS), Phase 4 replaces the linear lookups of piece types (`find_piece_type_on_square_side`) with a cached approach.

## Design and Analysis

In the hot search path:
1. `score_captures` and `score_captures_with_tt` query the piece type of the victim and the aggressor for every generated capture.
2. `score_quiets` queries the piece type of the aggressor for every quiet move.
3. During search, quiescence search delta pruning queries the victim type of a capture move to estimate the material gain.
4. Capture history updates query the victim piece type after a beta cutoff.

Previously, these queries performed linear bitboard scans over 6 bitboards via `find_piece_type_on_square_side`. Because this is called on the hot path, it is a significant bottleneck.

### O(1) Cache Architecture

We introduced two arrays in `SearchContext` JIT class:
- `aggressor_cache: numba.int8[:, :]` of size `(MAX_PLY, 65536)`
- `victim_cache: numba.int8[:, :]` of size `(MAX_PLY, 65536)`

#### Why index by `move` instead of move index?
A typical search stores moves in a buffer for each ply. However, these moves are sorted (swapped) frequently by selection sort and insertion sort algorithms. Indexing the cache by the move's buffer index (e.g. `[ply, i]`) would require matching swaps during move sorting, adding overhead and complexity.

By indexing the cache directly using the 16-bit move integer (`[ply, move]`), the cache entries are fully decoupled from move sorting. A move's piece type is looked up in $O(1)$ time regardless of its sorted index.

#### Memory and CPU Cache Friendliness
Each row of the cache takes exactly $65,536 \times 1 \text{ byte} = 64 \text{ KB}$.
- For a given ply, the search only accesses the row corresponding to that ply.
- A single search node generates up to ~256 moves, so only up to ~256 random indices are accessed at that ply.
- These accessed cache lines fit entirely within the CPU's L1/L2 data cache, ensuring near-instantaneous memory retrieval.

## Implementation Details

1. **`SearchContext` Initialization**:
   - Spec fields: `aggressor_cache` and `victim_cache`.
   - Initialized to `np.zeros((MAX_PLY, 65536), dtype=np.int8)`.

2. **Population during Scoring**:
   - Inside `score_captures` and `score_captures_with_tt`, the aggressor and victim types are computed once and stored:
     ```python
     search_context.aggressor_cache[ply, move] = aggressor_type
     search_context.victim_cache[ply, move] = victim_type
     ```
   - Inside `score_quiets` and `score_moves`, the aggressor type is stored:
     ```python
     search_context.aggressor_cache[ply, move] = aggressor_type
     ```

3. **Cache Lookup**:
   - In `quiescence_search` delta pruning, the victim piece type lookup is replaced with:
     ```python
     victim_type = search_context.victim_cache[ply, move]
     ```
   - In Capture History update, since `unmake_move` is already called and the board is restored, we can bypass `find_piece_type_on_square_side` completely by extracting `captured_piece_type` from the existing `unmake_info` return value:
     ```python
     victim_type = unmake_info[1]
     if victim_type != -1:
         enemy_side = 1 - original_side
         victim_type += enemy_side * 6
     ```
