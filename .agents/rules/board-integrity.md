---
description: "Chess board integrity, state representation, and Make/Unmake correctness invariants."
trigger: glob
glob: "**/{board_operations,move_generator,board,zobrist}.py"
---

# Board Integrity & State Invariants

When designing or modifying the board state representation, move generator, or move execution scripts, the AI assistant **MUST** ensure the mathematical and logical consistency of all board invariants.

---

## 1. Board Representation Invariants

*   **Perfect Synchronization of Bitboards and Occupancy**:
    *   `piece_bbs` (length 12) stores the 6 piece types for White (0-5) and Black (6-11) independently.
    *   `occupancy_bbs[0]` (WHITE) **must** be the bitwise OR union of `piece_bbs[0:6]`.
    *   `occupancy_bbs[1]` (BLACK) **must** be the bitwise OR union of `piece_bbs[6:12]`.
    *   `occupancy_bbs[2]` (ALL) **must** be exactly `occupancy_bbs[0] | occupancy_bbs[1]`.
    *   **This relationship must hold perfectly after any move execution (Make) or retraction (Unmake). Otherwise, sliding piece attack masks and move generation will completely fail.**

---

## 2. In-Place Modifications & Reversibility of Unmake

The engine utilizes highly optimized in-place state modifications rather than cloning the board structure.

*   **Responsibilities of `make_move`**:
    1.  Update the piece placements within `piece_bbs` and `occupancy_bbs`.
    2.  Handle specialized moves: **Captures** (remove opponent piece, update occupancy), **Promotions** (remove pawn, place new piece), **En Passant** (remove target pawn on adjacent square), and **Castling** (move both King and Rook atomically).
    3.  Compute new state attributes: toggle side-to-move, update castling rights, set new en passant target squares, and update the halfmove clock.
    4.  **Preserve Irreversible State**: Package pre-move metadata (castling rights, en passant file, halfmove clock, old Zobrist key, evaluation keys) into an `unmake_info` tuple and return it.
*   **Responsibilities of `unmake_move`**:
    *   `unmake_move` **must** represent the mathematically exact inverse of `make_move`.
    *   Use the preserved metadata in `unmake_info` to restore all irreversible states.
    *   Reverse sliding operations (e.g., castling, promotion, en passant) using precise bitwise masks to ensure no orphaned bits or missing pieces remain.

---

## 3. Zobrist Hash & Incremental Key Invariants

To prevent catastrophic collision errors and missed moves in the Transposition Table (TT):

*   **Strictly Incremental Zobrist Updates**:
    *   **Never** recalculate the entire Zobrist key from scratch during search.
    *   Use bitwise XOR operations to incrementally update the hash key:
        *   Piece movement: `key ^= PIECE_SQUARE_KEYS[piece, from] ^ PIECE_SQUARE_KEYS[piece, to]`
        *   Capture: `key ^= PIECE_SQUARE_KEYS[captured_piece, to]`
        *   Castling rights, en passant, or active color changes: XOR with the corresponding feature key.
*   **Incremental Evaluation Keys**:
    *   The pawn structure key (`pawn_key`), minor piece key (`minor_key`), and non-pawn keys (`non_pawn_key_white/black`) **must** also be incrementally updated via XOR operations during make/unmake.
    *   `unmake_move` **must** restore these keys directly from `unmake_info` rather than running any full-board scanning.

---

## 4. Perft Verification Mandatory Recommendation

*   **Verification Safeguard**: Any modification touching `make_move`, `unmake_move`, `move_generator.py`, or bitwise helper routines **must** recommend that the user executes a Perft (performance test) suite.
*   The AI assistant **must** supply specific terminal commands to run a Perft up to Depth 5 on multiple positions (including the Start Position, Kiwipete, and highly tactical en-passant/promotion setups) to be manually run by the user.
