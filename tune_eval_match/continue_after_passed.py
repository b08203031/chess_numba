#!/usr/bin/env python3
"""After block passed finishes: sync constants, run material, sync again."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tune_eval_match.pipeline_after_a import (  # noqa: E402
    load_best_int,
    log,
    run_spsa,
    sync_classical_old,
    sync_material_to_constants,
    sync_passed_to_constants,
)

PASSED_DIR = ROOT / "tournament_analysis" / "spsa_eval_passed"
MATERIAL_DIR = ROOT / "tournament_analysis" / "spsa_eval_material"
STATE = ROOT / "tournament_analysis" / "spsa_eval_a123" / "pipeline_state.json"


def passed_running() -> bool:
    if sys.platform == "win32":
        r = subprocess.run(
            [
                "powershell", "-NoProfile", "-Command",
                "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
                "Select-Object -ExpandProperty CommandLine",
            ],
            capture_output=True, text=True, timeout=30,
        )
        for line in (r.stdout or "").splitlines():
            if "spsa_match" in line and ("passed" in line or "spsa_eval_passed" in line):
                return True
    return False


def main() -> int:
    poll = 120
    log("waiting for passed block to finish...")
    while passed_running():
        time.sleep(poll)
        log("passed still running")
    log("passed finished")

    if not (PASSED_DIR / "best_params.json").exists():
        log("ERROR: no passed best_params.json — abort material")
        return 1

    best = load_best_int(PASSED_DIR)
    sync_passed_to_constants(best)
    sync_classical_old()

    st = {}
    if STATE.exists():
        st = json.loads(STATE.read_text(encoding="utf-8"))
    st["phase"] = "run_material"
    st["passed_best"] = best
    STATE.write_text(json.dumps(st, indent=2), encoding="utf-8")

    code = run_spsa("material", MATERIAL_DIR)
    st["material_exit"] = code
    if (MATERIAL_DIR / "best_params.json").exists():
        mb = load_best_int(MATERIAL_DIR)
        sync_material_to_constants(mb)
        sync_classical_old()
        st["material_best"] = mb
    st["phase"] = "done"
    st["finished_at"] = datetime.now(timezone.utc).isoformat()
    STATE.write_text(json.dumps(st, indent=2), encoding="utf-8")
    log("pipeline DONE (passed→material)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
