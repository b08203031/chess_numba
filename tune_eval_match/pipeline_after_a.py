#!/usr/bin/env python3
"""
Autonomous continuation after block A match-SPSA finishes.

Watches tournament_analysis/spsa_eval_a123 for A completion, then:
  1) sync A best_params theta_int into classical/constants.py (+ classical_old)
  2) run --block passed
  3) sync passed best into constants
  4) run --block material

Does not kill a live A process; only acts when A has stopped.

Usage (repo root):
  python -u tune_eval_match/pipeline_after_a.py
  python -u tune_eval_match/pipeline_after_a.py --poll 120 --once-check
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
A_DIR = ROOT / "tournament_analysis" / "spsa_eval_a123"
PASSED_DIR = ROOT / "tournament_analysis" / "spsa_eval_passed"
MATERIAL_DIR = ROOT / "tournament_analysis" / "spsa_eval_material"
CONST = ROOT / "chess_engine" / "classical" / "constants.py"
STATE = A_DIR / "pipeline_state.json"

# Map best_params names -> constants.py assignment patterns
A_CONST_MAP = {
    "KING_FLANK_ATTACK_NUM": "KING_FLANK_ATTACK_NUM",
    "KING_FLANK_DEFENSE_MULT": "KING_FLANK_DEFENSE_MULT",
    "KING_SHELTER_FEEDBACK_NUM": "KING_SHELTER_FEEDBACK_NUM",
    "PAWNLESS_FLANK_MG": None,  # array
    "PAWNLESS_FLANK_EG": None,
    "FLANK_ATTACKS_MG": None,  # array
    "KING_DANGER_OUT_MG_NUM": "KING_DANGER_OUT_MG_NUM",
    "EG_SAFETY_SCALE": None,  # float 1.0 in constants; weights use 100
    "SHELTER_BASE_MG": "SHELTER_BASE_MG",
    "SHELTER_BASE_EG": "SHELTER_BASE_EG",
    "BLOCKED_STORM": "BLOCKED_STORM",
    "BLOCKED_STORM_EG": "BLOCKED_STORM_EG",
}

PASSED_CONST_MAP = {
    "PASSED_PATH_SAFE_NONE_ATTACK": "PASSED_PATH_SAFE_NONE_ATTACK",
    "PASSED_PATH_SAFE_EDGE_ATTACK": "PASSED_PATH_SAFE_EDGE_ATTACK",
    "PASSED_PATH_SAFE_BLOCK_ONLY": "PASSED_PATH_SAFE_BLOCK_ONLY",
    "PASSED_PATH_SUPPORT_BONUS": "PASSED_PATH_SUPPORT_BONUS",
    "PASSED_DYNAMICS_MULT": "PASSED_DYNAMICS_MULT",
    "PASSED_DYNAMICS_OFFSET": "PASSED_DYNAMICS_OFFSET",
    "KING_PROX_ENEMY_MULT": "KING_PROX_ENEMY_MULT",
    "KING_PROX_FRIENDLY_MULT": "KING_PROX_FRIENDLY_MULT",
    "PASSED_FILE_BONUS_MG": None,
    "PASSED_FILE_BONUS_EG": None,
}

MAT_CONST_MAP = {
    "MAT_N_MG": ("MG_MATERIAL_VALUES", 1),
    "MAT_N_EG": ("EG_MATERIAL_VALUES", 1),
    "MAT_B_MG": ("MG_MATERIAL_VALUES", 2),
    "MAT_B_EG": ("EG_MATERIAL_VALUES", 2),
    "MAT_R_MG": ("MG_MATERIAL_VALUES", 3),
    "MAT_R_EG": ("EG_MATERIAL_VALUES", 3),
    "MAT_Q_MG": ("MG_MATERIAL_VALUES", 4),
    "MAT_Q_EG": ("EG_MATERIAL_VALUES", 4),
    "IMBALANCE_SCALE_MG": "IMBALANCE_SCALE_MG",
    "IMBALANCE_SCALE_EG": "IMBALANCE_SCALE_EG",
}


def log(msg: str) -> None:
    print(f"[pipeline {datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"phase": "wait_a", "history": []}


def save_state(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=2), encoding="utf-8")


def a_spsa_running() -> bool:
    """True if a python process is still running spsa_match with a123 out-dir."""
    try:
        import psutil  # type: ignore
        for p in psutil.process_iter(["cmdline"]):
            cmd = p.info.get("cmdline") or []
            s = " ".join(cmd)
            if "spsa_match" in s and "spsa_eval_a123" in s:
                return True
    except Exception:
        pass
    # Windows fallback: wmic / tasklist unreliable for cmdline; use PowerShell
    if sys.platform == "win32":
        try:
            r = subprocess.run(
                [
                    "powershell", "-NoProfile", "-Command",
                    "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
                    "Select-Object -ExpandProperty CommandLine",
                ],
                capture_output=True, text=True, timeout=30,
            )
            for line in (r.stdout or "").splitlines():
                if "spsa_match" in line and "spsa_eval_a123" in line:
                    return True
        except Exception:
            pass
    return False


def a_finished() -> bool:
    """A is done if not running and (final_params / stopped_reason / high iter idle)."""
    if a_spsa_running():
        return False
    final = A_DIR / "final_params.json"
    ck = A_DIR / "checkpoint.json"
    if final.exists():
        return True
    if ck.exists():
        c = json.loads(ck.read_text(encoding="utf-8"))
        if c.get("stopped_reason"):
            return True
        # process dead with checkpoint — treat as finished if best exists and no live process
        if (A_DIR / "best_params.json").exists():
            return True
    return False


def load_best_int(dir_path: Path) -> dict:
    bp = dir_path / "best_params.json"
    if not bp.exists():
        raise FileNotFoundError(bp)
    b = json.loads(bp.read_text(encoding="utf-8"))
    return {k: int(v) for k, v in (b.get("theta_int") or b["theta"]).items()}


def _replace_scalar(text: str, name: str, value: int) -> str:
    # Matches: NAME = 128 | NAME = np.int32(128) | NAME = np.int32(_MG_SCALE_DEN) ...
    pat = re.compile(
        rf"^({re.escape(name)}\s*=\s*)(.+)$",
        re.M,
    )
    m = pat.search(text)
    if not m:
        log(f"WARN: could not find scalar {name} in constants.py")
        return text
    rhs = m.group(2).strip()
    # Preserve trailing comment
    comment = ""
    if "  #" in rhs:
        rhs, _, cmt = rhs.partition("  #")
        comment = "  #" + cmt
    elif " #" in rhs:
        rhs, _, cmt = rhs.partition(" #")
        comment = " #" + cmt
    rhs = rhs.strip()
    if rhs.startswith("np.int32("):
        new_rhs = f"np.int32({value})"
    else:
        new_rhs = str(value)
    return pat.sub(rf"\g<1>{new_rhs}{comment}", text, count=1)


def _replace_array2(text: str, name: str, v0: int, v1: int) -> str:
    # NAME = np.array([a, b], dtype=np.int32)
    pat = re.compile(
        rf"({re.escape(name)}\s*=\s*np\.array\(\[)\s*-?\d+\s*,\s*-?\d+(\]\s*,\s*dtype=np\.int32\))",
        re.M,
    )
    if not pat.search(text):
        log(f"WARN: could not find array {name}")
        return text
    return pat.sub(rf"\g<1>{v0}, {v1}\2", text, count=1)


def _replace_material_idx(text: str, arr_name: str, idx: int, value: int) -> str:
    # MG_MATERIAL_VALUES = np.array([100, 347, 373, 578, 1177, 0], ...)
    pat = re.compile(
        rf"({re.escape(arr_name)}\s*=\s*np\.array\(\[)([^\]]+)(\]\s*,\s*dtype=np\.int32\))",
        re.M,
    )
    m = pat.search(text)
    if not m:
        log(f"WARN: could not find {arr_name}")
        return text
    parts = [p.strip() for p in m.group(2).split(",")]
    if idx >= len(parts):
        log(f"WARN: index {idx} out of range for {arr_name}")
        return text
    parts[idx] = str(value)
    return pat.sub(m.group(1) + ", ".join(parts) + m.group(3), text, count=1)


def sync_a_to_constants(best: dict) -> None:
    text = CONST.read_text(encoding="utf-8")
    text = _replace_scalar(text, "KING_FLANK_ATTACK_NUM", best["KING_FLANK_ATTACK_NUM"])
    text = _replace_scalar(text, "KING_FLANK_DEFENSE_MULT", best["KING_FLANK_DEFENSE_MULT"])
    text = _replace_scalar(text, "KING_SHELTER_FEEDBACK_NUM", best["KING_SHELTER_FEEDBACK_NUM"])
    text = _replace_array2(text, "PAWNLESS_FLANK", best["PAWNLESS_FLANK_MG"], best["PAWNLESS_FLANK_EG"])
    # FLANK_ATTACKS = np.array([-2, 0] — only MG tuned
    text = _replace_array2(text, "FLANK_ATTACKS", best["FLANK_ATTACKS_MG"], 0)
    text = _replace_scalar(text, "KING_DANGER_OUT_MG_NUM", best["KING_DANGER_OUT_MG_NUM"])
    # EG_SAFETY_SCALE is float 1.0 in constants; weight uses 100 = 1.0 — leave float unless ~100
    if abs(best.get("EG_SAFETY_SCALE", 100) - 100) >= 1:
        scale = best["EG_SAFETY_SCALE"] / 100.0
        text = re.sub(
            r"EG_SAFETY_SCALE\s*=\s*[0-9.]+",
            f"EG_SAFETY_SCALE = {scale}",
            text,
            count=1,
        )
    text = _replace_scalar(text, "SHELTER_BASE_MG", best["SHELTER_BASE_MG"])
    text = _replace_scalar(text, "SHELTER_BASE_EG", best["SHELTER_BASE_EG"])
    text = _replace_scalar(text, "BLOCKED_STORM", best["BLOCKED_STORM"])
    text = _replace_scalar(text, "BLOCKED_STORM_EG", best["BLOCKED_STORM_EG"])
    CONST.write_text(text, encoding="utf-8")
    log("synced A best into constants.py")


def sync_passed_to_constants(best: dict) -> None:
    text = CONST.read_text(encoding="utf-8")
    for k, cname in PASSED_CONST_MAP.items():
        if cname:
            text = _replace_scalar(text, cname, best[k])
    text = _replace_array2(
        text, "PASSED_FILE_BONUS",
        best["PASSED_FILE_BONUS_MG"], best["PASSED_FILE_BONUS_EG"],
    )
    CONST.write_text(text, encoding="utf-8")
    log("synced Passed best into constants.py")


def sync_material_to_constants(best: dict) -> None:
    text = CONST.read_text(encoding="utf-8")
    for k, spec in MAT_CONST_MAP.items():
        if isinstance(spec, tuple):
            text = _replace_material_idx(text, spec[0], spec[1], best[k])
        else:
            text = _replace_scalar(text, spec, best[k])
    CONST.write_text(text, encoding="utf-8")
    log("synced Material best into constants.py")


def sync_classical_old() -> None:
    r = subprocess.run(
        [sys.executable, "tools/sync_classical_old.py", "--sync"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    log(f"sync classical_old exit={r.returncode}")
    if r.stdout:
        log(r.stdout[-500:])


def run_spsa(block: str, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "console_pipeline.log"
    cmd = [
        sys.executable, "-u", "tune_eval_match/spsa_match.py",
        "--block", block,
        "--out-dir", str(out_dir.relative_to(ROOT) if out_dir.is_relative_to(ROOT) else out_dir),
        "-c", "10",
        "--tt-mb", "4",
        "--games", "250",
        "--nodes", "20000",
        "--l2-every", "25",
        "--l2-games", "250",
        "--l2-nodes", "200000",
        "--l2-stop-after-fails", "2",
        "--iters", "500",
        "--ckpt-every", "5",
        "--seed", "42",
        "--no-verify",
        "--no-phase2-after-l2",
    ]
    log(f"starting SPSA: {' '.join(cmd)}")
    with open(log_path, "w", encoding="utf-8") as f:
        p = subprocess.Popen(
            cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    log(f"SPSA pid={p.pid} block={block} log={log_path}")
    return p.wait()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poll", type=int, default=180, help="Seconds between A-completion checks")
    ap.add_argument("--once-check", action="store_true", help="Single poll then exit if A not done")
    ap.add_argument("--force-start-passed", action="store_true", help="Skip wait; assume A done")
    args = ap.parse_args()

    st = load_state()
    log(f"phase={st.get('phase')}")

    while st.get("phase") == "wait_a":
        if args.force_start_passed or a_finished():
            log("A finished (or forced)")
            st["phase"] = "sync_a"
            save_state(st)
            break
        log("A still running or not finished; sleep")
        if args.once_check:
            return 0
        time.sleep(max(30, args.poll))

    if st.get("phase") == "sync_a":
        best = load_best_int(A_DIR)
        sync_a_to_constants(best)
        sync_classical_old()
        st["phase"] = "run_passed"
        st["a_best"] = best
        save_state(st)

    if st.get("phase") == "run_passed":
        code = run_spsa("passed", PASSED_DIR)
        st["passed_exit"] = code
        st["phase"] = "sync_passed"
        save_state(st)

    if st.get("phase") == "sync_passed":
        if (PASSED_DIR / "best_params.json").exists():
            best = load_best_int(PASSED_DIR)
            sync_passed_to_constants(best)
            sync_classical_old()
            st["passed_best"] = best
        else:
            log("WARN: no passed best_params; skip const sync")
        st["phase"] = "run_material"
        save_state(st)

    if st.get("phase") == "run_material":
        code = run_spsa("material", MATERIAL_DIR)
        st["material_exit"] = code
        st["phase"] = "sync_material"
        save_state(st)

    if st.get("phase") == "sync_material":
        if (MATERIAL_DIR / "best_params.json").exists():
            best = load_best_int(MATERIAL_DIR)
            sync_material_to_constants(best)
            sync_classical_old()
            st["material_best"] = best
        st["phase"] = "done"
        save_state(st)
        log("pipeline DONE")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
