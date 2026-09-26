"""Generate IDX_* constants from ParameterManager into new_indices.txt."""
from __future__ import annotations

from pathlib import Path

import tuner.parameters as parameters


def generate_indices_text() -> str:
    pm = parameters.ParameterManager()
    lines = ["# Generated indices — do not edit; run generate_indices.py / generate_tunable_eval.py"]
    for p in pm.param_map:
        lines.append(f"IDX_{p['name']} = {p['start']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    text = generate_indices_text()
    out = Path(__file__).resolve().parent / "new_indices.txt"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
