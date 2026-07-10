---
description: "Bitboard occupancy sync, make/unmake reversibility, incremental Zobrist; perft recommendation."
trigger: glob
glob: "**/chess_engine/**/{board_operations,move_generator,zobrist,bitboard_utils,move,fen_parser}.py"
---

# Board Integrity Invariants

When changing board state, movegen, or hashing, preserve these invariants.

Details: `chess_engine/README.md`, classical `board_operations.py` / `zobrist.py`.

---

## 1. Occupancy sync (MUST hold after every make/unmake)

* `piece_bbs[0:6]` = White pieces; `piece_bbs[6:12]` = Black
* `occupancy_bbs[0]` = OR of White piece BBs
* `occupancy_bbs[1]` = OR of Black piece BBs
* `occupancy_bbs[2]` = `occupancy_bbs[0] | occupancy_bbs[1]`

Broken occupancy → wrong sliders, legality, SEE, and eval.

---

## 2. Make / unmake (MUST)

**`make_move`:**

1. Update `piece_bbs` + `occupancy_bbs` in place
2. Handle captures, promotions, en passant (remove pawn on **capture** square), castling (king + rook)
3. Toggle STM; update castling rights, EP, halfmove clock
4. Return `unmake_info` with all irreversible fields (rights, EP, halfmove, old Zobrist / eval keys, captured piece, etc.)

**`unmake_move`:**

* Exact inverse of make; restore irreversible state **only** from `unmake_info`
* No orphan bits, no full-board rebuild

---

## 3. Zobrist & eval keys (MUST)

* **Incremental XOR only** during search — never full rehash on the hot path
* Piece move / capture / rights / EP / color: XOR corresponding keys
* `pawn_key`, `minor_key`, non-pawn keys: incremental on make; restore from `unmake_info` on unmake

---

## 4. After changes (recommend to user)

Do not auto-run heavy perft. Suggest:

```bash
python -m tests.perft perft --depth 5
python -m tests.perft perft --depth 4 --fen "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
```

Include EP/promotion positions if those paths changed.
