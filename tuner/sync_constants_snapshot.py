"""
Sync tuner/constants_snapshot.py from chess_engine/classical/constants.py.

Run after any HCE constant change before SPSA:

    python -m tuner.sync_constants_snapshot
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "chess_engine" / "classical" / "constants.py"
DST = ROOT / "tuner" / "constants_snapshot.py"
SHIM = ROOT / "tuner" / "constants_to_be_tuned.py"
CLASSICAL_CACHE_DIR = ROOT / "chess_engine" / "classical" / "__pycache__"

HEADER = """# AUTO-GENERATED snapshot of chess_engine/classical/constants.py for HCE SPSA.
# Do not edit by hand. Regenerate with: python -m tuner.sync_constants_snapshot
# Source of truth remains classical/constants.py

"""

SHIM_TEXT = """# Backward-compatible alias. Prefer tuner.constants_snapshot.
from tuner.constants_snapshot import *  # noqa: F403
"""


def clear_classical_numba_cache(cache_dir: Path = CLASSICAL_CACHE_DIR) -> int:
    """Remove rebuildable Numba caches that may have frozen old HCE globals."""
    if not cache_dir.is_dir():
        return 0
    removed = 0
    for path in cache_dir.iterdir():
        if path.is_file() and path.suffix in {".nbi", ".nbc"}:
            path.unlink()
            removed += 1
    return removed


def main() -> None:
    body = SRC.read_text(encoding="utf-8")
    snapshot = HEADER + body
    snapshot_changed = not DST.exists() or DST.read_text(encoding="utf-8") != snapshot

    if snapshot_changed:
        DST.write_text(snapshot, encoding="utf-8")
        removed = clear_classical_numba_cache()
        print(
            f"Wrote {DST.relative_to(ROOT)} from {SRC.relative_to(ROOT)}; "
            f"removed {removed} stale classical Numba cache file(s)"
        )
    else:
        print(f"Unchanged {DST.relative_to(ROOT)}")

    if not SHIM.exists() or SHIM.read_text(encoding="utf-8") != SHIM_TEXT:
        SHIM.write_text(SHIM_TEXT, encoding="utf-8")


if __name__ == "__main__":
    main()
