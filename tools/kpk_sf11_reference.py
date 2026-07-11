"""
Independent SF11-faithful KPK bitbase generator (pure Python).

Transcribed from stockfish_11/src/bitbase.cpp for golden comparison against
chess_engine.classical.endgame.KPK_BITBASE. Not used at runtime by the engine.
"""
from __future__ import annotations

import numpy as np

MAX_INDEX = 2 * 24 * 64 * 64  # 196608
INVALID, UNKNOWN, DRAW, WIN = 0, 1, 2, 4


def _chebyshev(a: int, b: int) -> int:
    return max(abs((a // 8) - (b // 8)), abs((a & 7) - (b & 7)))


def _king_attacks(sq: int) -> list[int]:
    r, f = sq // 8, sq & 7
    out = []
    for dr in (-1, 0, 1):
        for df in (-1, 0, 1):
            if dr == 0 and df == 0:
                continue
            rr, ff = r + dr, f + df
            if 0 <= rr <= 7 and 0 <= ff <= 7:
                out.append(rr * 8 + ff)
    return out


def _white_pawn_attacks(psq: int) -> set[int]:
    r, f = psq // 8, psq & 7
    out = set()
    if r < 7:
        if f > 0:
            out.add((r + 1) * 8 + (f - 1))
        if f < 7:
            out.add((r + 1) * 8 + (f + 1))
    return out


def index(us: int, bksq: int, wksq: int, psq: int) -> int:
    # SF11: RANK_7 - rank  →  for 0-based rank: 6 - rank
    return (
        int(wksq)
        | (int(bksq) << 6)
        | (int(us) << 12)
        | ((int(psq) & 7) << 13)
        | ((6 - (int(psq) // 8)) << 15)
    )


def decode(idx: int):
    wksq = idx & 0x3F
    bksq = (idx >> 6) & 0x3F
    us = (idx >> 12) & 1
    pfile = (idx >> 13) & 3
    prank = 6 - ((idx >> 15) & 7)
    psq = prank * 8 + pfile
    return us, bksq, wksq, psq, prank


def build_sf11_kpk_bitbase() -> np.ndarray:
    """Return uint32 packed bitbase identical in layout to SF11 / our engine."""
    king_att = [_king_attacks(s) for s in range(64)]
    db = [UNKNOWN] * MAX_INDEX

    for idx in range(MAX_INDEX):
        us, bksq, wksq, psq, prank = decode(idx)
        if prank < 1 or prank > 6:
            db[idx] = INVALID
            continue
        if (
            _chebyshev(wksq, bksq) <= 1
            or wksq == psq
            or bksq == psq
            or (us == 0 and bksq in _white_pawn_attacks(psq))
        ):
            db[idx] = INVALID
            continue

        if us == 0 and prank == 6:
            promo = psq + 8
            if wksq != promo and (
                _chebyshev(bksq, promo) > 1 or promo in king_att[wksq]
            ):
                db[idx] = WIN
                continue

        if us == 1:
            # stalemate or capture undefended pawn
            safe = [
                t
                for t in king_att[bksq]
                if t not in king_att[wksq] and t not in _white_pawn_attacks(psq)
            ]
            can_take_pawn = psq in king_att[bksq] and psq not in king_att[wksq]
            if not safe or can_take_pawn:
                db[idx] = DRAW
                continue

        db[idx] = UNKNOWN

    changed = True
    while changed:
        changed = False
        for idx in range(MAX_INDEX):
            if db[idx] != UNKNOWN:
                continue
            us, bksq, wksq, psq, prank = decode(idx)
            r = INVALID
            if us == 0:
                good, bad = WIN, DRAW
                for to in king_att[wksq]:
                    r |= db[index(1, bksq, to, psq)]
                if prank < 6:
                    r |= db[index(1, bksq, wksq, psq + 8)]
                if prank == 1 and (psq + 8) != wksq and (psq + 8) != bksq:
                    r |= db[index(1, bksq, wksq, psq + 16)]
            else:
                good, bad = DRAW, WIN
                for to in king_att[bksq]:
                    r |= db[index(0, to, wksq, psq)]

            if r & good:
                db[idx] = good
                changed = True
            elif not (r & UNKNOWN):
                db[idx] = bad
                changed = True

    bitbase = np.zeros(MAX_INDEX // 32, dtype=np.uint32)
    for idx in range(MAX_INDEX):
        if db[idx] == WIN:
            bitbase[idx // 32] |= np.uint32(1) << np.uint32(idx & 31)
    return bitbase


def probe_ref(bitbase: np.ndarray, wksq: int, psq: int, bksq: int, us: int) -> bool:
    idx = index(us, bksq, wksq, psq)
    return bool(bitbase[idx // 32] & (np.uint32(1) << np.uint32(idx & 31)))


def bitbase_stats(bitbase: np.ndarray) -> dict:
    wins = 0
    for word in bitbase:
        wins += int(word).bit_count()
    return {"max_index": MAX_INDEX, "win_entries": wins, "words": len(bitbase)}
