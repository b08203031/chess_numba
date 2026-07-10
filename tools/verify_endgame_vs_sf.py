#!/usr/bin/env python3
"""
Optional Stockfish search oracle for specialized endgame FENs.

Compares this engine's evaluate_special_endgame / evaluate_position against
Stockfish deep search (win/draw/loss direction). Not for daily CI — use:

  python -m unittest tests.test_endgame_conformance -v

Usage (repo root):
  python tools/verify_endgame_vs_sf.py
  python tools/verify_endgame_vs_sf.py --depth 22 --write-md
  python tools/verify_endgame_vs_sf.py --sf path/to/stockfish.exe
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chess_engine.classical.fen_parser import parse_fen
from chess_engine.classical.endgame import evaluate_special_endgame
from chess_engine.classical.evaluation import evaluate_position

DEFAULT_SF = ROOT / "external" / "stockfish" / "stockfish-windows-x86-64-avx2.exe"
DOC_PATH = ROOT / "chess_engine" / "classical" / "ENDGAME_SF_VERIFICATION.md"

POSITIONS = [
    ("KBNK white", "4k3/8/8/8/3BN3/8/8/7K w - - 0 1"),
    ("KBNK black", "7k/8/8/3bn3/8/8/8/4K3 w - - 0 1"),
    ("KRK", "4k3/8/8/8/3R4/8/8/7K w - - 0 1"),
    ("KQK", "4k3/8/8/8/3Q4/8/8/7K w - - 0 1"),
    ("KPK draw opposition", "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1"),
    ("KPK win h2 corner", "8/8/8/8/8/8/7P/5k1K w - - 0 1"),
    ("KPK draw front", "8/8/8/8/4k3/8/4P3/4K3 w - - 0 1"),
    ("KPK draw a-pawn", "8/8/8/8/8/k7/P7/K7 w - - 0 1"),
    ("KPK win advanced", "4k3/8/4K3/4P3/8/8/8/8 w - - 0 1"),
    ("KRKP", "4k3/8/8/8/3R4/8/4p3/4K3 w - - 0 1"),
    ("KQKP e7", "4k3/4p3/8/8/8/8/8/3QK3 w - - 0 1"),
    ("KBK", "4k3/8/8/8/3B4/8/8/4K3 w - - 0 1"),
]


class Stockfish:
    def __init__(self, path: str):
        self.proc = subprocess.Popen(
            [path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        self._wait_for("uciok")
        self._send("setoption name Hash value 64")
        self._send("isready")
        self._wait_for("readyok")

    def _send(self, cmd: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _wait_for(self, token: str, timeout: float = 30.0) -> list[str]:
        assert self.proc.stdout is not None
        lines: list[str] = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                break
            line = line.rstrip("\n")
            lines.append(line)
            if token in line:
                return lines
        raise TimeoutError(f"timeout waiting for {token!r}; last lines: {lines[-5:]}")

    def eval_and_search(self, fen: str, depth: int = 22):
        self._send("ucinewgame")
        self._send("isready")
        self._wait_for("readyok")
        self._send(f"position fen {fen}")
        self._send("eval")
        eval_lines = self._wait_for("Final evaluation", timeout=10)
        self._send("isready")
        self._wait_for("readyok")

        self._send(f"go depth {depth}")
        search_lines = self._wait_for("bestmove", timeout=120)

        final_eval = None
        for line in eval_lines:
            if "Final evaluation" in line:
                final_eval = line.strip()

        scores = []
        for line in search_lines:
            m = re.search(r"info depth (\d+).*score (cp|mate) (-?\d+)", line)
            if m:
                scores.append((int(m.group(1)), m.group(2), int(m.group(3))))
        deepest = scores[-1] if scores else None
        best = next((l for l in search_lines if l.startswith("bestmove")), None)
        return final_eval, deepest, best, search_lines

    def close(self) -> None:
        try:
            self._send("quit")
        except Exception:
            pass
        try:
            self.proc.kill()
        except Exception:
            pass


def classify_verdict(name: str, hit: bool, our_w: int, our_stm: int, deep):
    if deep is None:
        return "⚠️ SF無分數"
    d, kind, s = deep
    if "KBNK white" in name or name in ("KRK", "KQK"):
        if kind == "mate" and s > 0:
            return "✅ SF殺王" if hit and our_w > 10000 else "⚠️ SF殺王但本引擎分數異常"
        if kind == "cp" and s > 150:
            return "✅ SF白優(深搜可殺)" if hit and our_w > 10000 else "⚠️ SF白優/引擎異常"
        return f"❌ 預期白勝但 SF={kind} {s}"
    if "KBNK black" in name:
        if kind == "mate" and s < 0:
            return "✅ SF被殺" if hit and our_w < -10000 else "⚠️ SF被殺但本引擎異常"
        if kind == "cp" and s < -150:
            return "✅ SF白劣(深搜被殺)" if hit and our_w < -10000 else "⚠️ SF白劣/引擎異常"
        return f"❌ 預期白劣但 SF={kind} {s}"
    if "KPK" in name:
        if "draw" in name.lower():
            if kind == "cp" and abs(s) < 80 and hit and our_w == 0:
                return "✅ 和棋一致"
            return f"⚠️ 預期和棋 SF={kind} {s} our_W={our_w}"
        if "win" in name.lower():
            if (kind == "mate" and s > 0) or (kind == "cp" and s > 200):
                return "✅ 白勝一致" if hit and our_w > 10000 else f"⚠️ SF白勝 our_W={our_w}"
            return f"⚠️ 預期白勝 SF={kind} {s} our_W={our_w}"
        return f"⚠️ SF={kind} {s}, our_W={our_w}"
    if "KRKP" in name or "KQKP" in name:
        if kind == "mate" and s > 0:
            return "✅ 白勝" if hit and our_w > 0 else f"⚠️ SF白勝 our={our_w}"
        if kind == "cp" and s > 200:
            return "✅ 白優" if hit and our_w > 0 else f"⚠️ SF白優 our={our_w}"
        return f"⚠️ SF={kind} {s}, our={our_w}"
    if "KBK" in name:
        if not hit and kind == "cp" and abs(s) < 100:
            return "✅ 理論和(不短路)"
        return f"⚠️ hit={hit} SF={kind} {s}"
    return "—"


def write_method_doc() -> None:
    """Refresh method-only doc (no frozen score table)."""
    text = """# 專用殘局 vs Stockfish 驗證說明

**範圍：** `chess_engine/classical` 專用殘局評估（`endgame.py`）與 Stockfish 搜尋對照  
**目的：** 用可重跑的 oracle 與單元測試，確認 KXK / KPK / KRKP / KQKP / KBNK 等勝負方向正確。

本文件**不寫死**某一輪的 cp 數值表；以「怎麼驗證」為主。最新一輪表格可用：

```bash
python tools/verify_endgame_vs_sf.py --write-md
```

寫入的是**當次** oracle 結果表（可選）；日常回歸以單元測試為準。

---

## 1. 驗證工具

| 工具 | 路徑 | 用途 |
| :--- | :--- | :--- |
| SF 搜尋 oracle | `tools/verify_endgame_vs_sf.py` | 外部 Stockfish deep search + 本引擎 special/eval |
| 殘局單元測試 | `tests/test_endgame_conformance.py` | HCE 尺度勝負回歸（`VALUE_KNOWN_WIN ≈ 10000`） |
| HCE 階段 A 測試 | `tests/test_phase_a_eval.py` | A1–A4、KPK 邊角等 |

Stockfish 預設：`external/stockfish/stockfish-windows-x86-64-avx2.exe`（可用 `--sf` 覆寫）。

---

## 2. 執行方式

```bash
# 日常 / CI 友好
python -m unittest tests.test_endgame_conformance tests.test_phase_a_eval -v

# 偶爾對照 SF（需本機 SF；勿在 CI 自動跑）
python tools/verify_endgame_vs_sf.py
python tools/verify_endgame_vs_sf.py --depth 22 --write-md
```

---

## 3. 判定注意

1. 現代 Stockfish **NNUE** 靜態 `eval` 與 SF11 HCE **尺度不同**，不可直接比 `10000`。
2. Oracle 以 **搜尋** 的 mate/cp **方向** 為準，不是 HCE 分數對齊。
3. 新增殘局：先寫 `test_endgame_conformance.py`，再跑 oracle 交叉確認。

## 4. 相關原始碼

| 路徑 | 角色 |
| :--- | :--- |
| `chess_engine/classical/endgame.py` | 專用殘局 + KPK bitbase |
| `chess_engine/classical/evaluation.py` | 先 special EG，再一般 HCE |
| `chess_engine/classical/HCE_SF11_GAP_AUDIT.md` | HCE 對齊主文件 |
| `stockfish_11/src/endgame.cpp` / `bitbase.cpp` | 參考 |
"""
    DOC_PATH.write_text(text, encoding="utf-8")


def write_run_table(sf_path: str, depth: int, rows: list) -> None:
    """Append a concrete run table under the method doc (overwrites file with method + table)."""
    lines = [
        "# 專用殘局 vs Stockfish — 本次 oracle 跑分",
        "",
        f"Stockfish：`{sf_path}`",
        f"搜尋深度：{depth}；`Final evaluation` 為靜態 NNUE（僅供參考）。",
        "搜尋分數以 **行棋方** 視角；`our_W` 白方視角、`our_STM` 行棋方視角。",
        "",
        "方法說明（常態文件）見本目錄歷史維護；日常請跑 `tests/test_endgame_conformance.py`。",
        "重新產生方法文件：`python tools/verify_endgame_vs_sf.py --write-method-doc`",
        "",
        "| 名稱 | hit | our_W | our_STM | SF search | SF final | bestmove | 判定 | FEN |",
        "| :--- | :---: | ---: | ---: | :--- | ---: | :--- | :--- | :--- |",
    ]
    for name, fen, hit, sc, val, sf_s, fe_s, bm_s, verdict in rows:
        lines.append(
            f"| {name} | {hit} | {sc:+d} | {val:+d} | `{sf_s}` | {fe_s} | `{bm_s}` | {verdict} | `{fen}` |"
        )
    lines.extend(
        [
            "",
            "## 判定規則（摘要）",
            "",
            "- 殺王殘局：SF mate 或大正分；本引擎 hit 且 |score| > 10000。",
            "- KPK：和棋 SF≈0 且 our_W=0；白勝 SF mate/大正分且 our_W>10000。",
            "- KRKP/KQKP：方向一致即可。",
            "- KBK：理論和，不應 special-hit KXK。",
            "",
        ]
    )
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="SF oracle for specialized endgames")
    parser.add_argument("--sf", type=Path, default=DEFAULT_SF, help="Stockfish executable")
    parser.add_argument("--depth", type=int, default=22, help="SF search depth")
    parser.add_argument(
        "--write-md",
        action="store_true",
        help="Write this run's result table to ENDGAME_SF_VERIFICATION.md",
    )
    parser.add_argument(
        "--write-method-doc",
        action="store_true",
        help="Rewrite ENDGAME_SF_VERIFICATION.md as method-only doc (no SF run)",
    )
    args = parser.parse_args(argv)

    if args.write_method_doc:
        write_method_doc()
        print(f"Wrote method doc: {DOC_PATH}")
        return 0

    sf_path = args.sf
    if not sf_path.exists():
        print(f"Error: Stockfish not found: {sf_path}", file=sys.stderr)
        print("Install under external/stockfish/ or pass --sf", file=sys.stderr)
        return 1

    print(f"Stockfish: {sf_path}  depth={args.depth}")
    sf = Stockfish(str(sf_path))
    print(
        f"{'name':24} | hit | {'our_W':>8} | {'our_STM':>8} | SF_search | SF_final | bestmove | verdict"
    )
    print("-" * 140)
    rows = []
    try:
        for name, fen in POSITIONS:
            pb, ob, gs = parse_fen(fen)
            hit, sc = evaluate_special_endgame(pb, ob, gs, 0)
            val, _, _ = evaluate_position(pb, ob, gs)
            fe, deep, bm, raw = sf.eval_and_search(fen, depth=args.depth)
            if deep:
                d, kind, s = deep
                sf_s = f"d{d} {kind} {s:+d}"
            else:
                sf_s = "NO_SCORE"
            fe_s = "?"
            if fe:
                m = re.search(r"([+-]?\d+\.\d+)", fe)
                fe_s = m.group(1) if m else fe[:30]
            bm_s = bm.replace("bestmove ", "").split()[0] if bm else "?"
            verdict = classify_verdict(name, bool(hit), int(sc), int(val), deep)
            print(
                f"{name:24} | {str(bool(hit)):4} | {int(sc):+8d} | {int(val):+8d} | "
                f"{sf_s:14} | {fe_s:>8} | {bm_s:6} | {verdict}"
            )
            rows.append((name, fen, bool(hit), int(sc), int(val), sf_s, fe_s, bm_s, verdict))
    finally:
        sf.close()

    if args.write_md:
        write_run_table(str(sf_path), args.depth, rows)
        print(f"\nWrote run table: {DOC_PATH}")
    else:
        print("\n(Tip: pass --write-md to save this table; default does not overwrite the doc.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
