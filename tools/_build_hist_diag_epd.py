"""Build hist-diag EPD sets: 500 puzzles + 500 openings (separate files)."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def unique_append(out: list[str], seen: set[str], fen: str) -> None:
    fen = fen.strip()
    if not fen or fen in seen:
        return
    # strip EPD ops after 4th/6th field roughly: take first 4+ tokens min for board side castling ep
    seen.add(fen)
    out.append(fen)


def load_puzzle_fens(limit: int = 500) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    t = (ROOT / "tests" / "test_puzzle.py").read_text(encoding="utf-8")
    for fen in re.findall(r'"fen":\s*"([^"]+)"', t):
        unique_append(out, seen, fen)
        if len(out) >= limit:
            return out[:limit]
    fp_path = ROOT / "failed_puzzles.json"
    if fp_path.exists():
        fp = json.loads(fp_path.read_text(encoding="utf-8"))
        if isinstance(fp, list):
            for x in fp:
                if isinstance(x, dict) and "fen" in x:
                    unique_append(out, seen, x["fen"])
                    if len(out) >= limit:
                        return out[:limit]
    return out[:limit]


def load_opening_fens(limit: int = 500) -> list[str]:
    """Evenly sample unique FENs from openings.epd up to limit."""
    op = ROOT / "data" / "openings.epd"
    lines = [ln.strip() for ln in op.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return []
    # stride so we get ~limit across the file
    stride = max(1, len(lines) // limit)
    out: list[str] = []
    seen: set[str] = set()
    for i in range(0, len(lines), stride):
        fen = lines[i].split(" bm ")[0].strip()
        unique_append(out, seen, fen)
        if len(out) >= limit:
            break
    # if still short (duplicates), fill sequentially
    if len(out) < limit:
        for line in lines:
            fen = line.split(" bm ")[0].strip()
            unique_append(out, seen, fen)
            if len(out) >= limit:
                break
    return out[:limit]


def main() -> None:
    puzzles = load_puzzle_fens(500)
    openings = load_opening_fens(500)
    p_path = ROOT / "data" / "hist_diag_puzzles_500.epd"
    o_path = ROOT / "data" / "hist_diag_openings_500.epd"
    p_path.write_text("\n".join(puzzles) + "\n", encoding="utf-8")
    o_path.write_text("\n".join(openings) + "\n", encoding="utf-8")
    print(f"puzzles:  {len(puzzles)} -> {p_path}")
    print(f"openings: {len(openings)} -> {o_path}")


if __name__ == "__main__":
    main()
