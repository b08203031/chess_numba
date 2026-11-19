#!/usr/bin/env python3
"""Export Match-SPSA eval_weights candidates into classical/constants.py format.

Preserves all existing comments and formatting in ``chess_engine/classical/constants.py``
by reusing ``tuner.apply_tuned.apply_tuned_source`` AST-span replacement.

Usage:
  python -m tune_eval_match.export_constants \
    --params-json tournament_analysis/spsa_eval_threats_mg/best_params.json \
    --out-constants /content/drive/MyDrive/chess_numba_data/eval_match_spsa/constants_best.py \
    --out-snippet /content/drive/MyDrive/chess_numba_data/eval_match_spsa/tuned_constants_best.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tune_eval_match.eval_weights import EVAL_PARAM_INDEX  # noqa: E402
from tuner.apply_tuned import (  # noqa: E402
    DEFAULT_TARGET,
    _eval_const_module,
    _format_assignment,
    _is_np_int32_scalar_call,
    _locate_target_nodes,
    apply_tuned_source,
)

# Maps array-backed EVAL_PARAM_INDEX names -> (constants_attr_name, index_tuple)
_ARRAY_PARAM_MAP: Dict[str, Tuple[str, Tuple[int, ...]]] = {
    "PAWNLESS_FLANK_MG": ("PAWNLESS_FLANK", (0,)),
    "PAWNLESS_FLANK_EG": ("PAWNLESS_FLANK", (1,)),
    "FLANK_ATTACKS_MG": ("FLANK_ATTACKS", (0,)),
    "FLANK_ATTACKS_EG": ("FLANK_ATTACKS", (1,)),
    "PASSED_FILE_BONUS_MG": ("PASSED_FILE_BONUS", (0,)),
    "PASSED_FILE_BONUS_EG": ("PASSED_FILE_BONUS", (1,)),
    "MAT_N_MG": ("MG_MATERIAL_VALUES", (1,)),
    "MAT_N_EG": ("EG_MATERIAL_VALUES", (1,)),
    "MAT_B_MG": ("MG_MATERIAL_VALUES", (2,)),
    "MAT_B_EG": ("EG_MATERIAL_VALUES", (2,)),
    "MAT_R_MG": ("MG_MATERIAL_VALUES", (3,)),
    "MAT_R_EG": ("EG_MATERIAL_VALUES", (3,)),
    "MAT_Q_MG": ("MG_MATERIAL_VALUES", (4,)),
    "MAT_Q_EG": ("EG_MATERIAL_VALUES", (4,)),
    "PASSED_RANK3_MG": ("PASSED_PAWN_BONUS", (3, 0)),
    "PASSED_RANK3_EG": ("PASSED_PAWN_BONUS", (3, 1)),
    "PASSED_RANK4_MG": ("PASSED_PAWN_BONUS", (4, 0)),
    "PASSED_RANK4_EG": ("PASSED_PAWN_BONUS", (4, 1)),
    "PASSED_RANK5_MG": ("PASSED_PAWN_BONUS", (5, 0)),
    "PASSED_RANK5_EG": ("PASSED_PAWN_BONUS", (5, 1)),
    "PASSED_RANK6_MG": ("PASSED_PAWN_BONUS", (6, 0)),
    "PASSED_RANK6_EG": ("PASSED_PAWN_BONUS", (6, 1)),
    "KING_ATTACK_WEIGHT_N": ("KING_ATTACK_WEIGHTS", (2,)),
    "KING_ATTACK_WEIGHT_B": ("KING_ATTACK_WEIGHTS", (3,)),
    "KING_ATTACK_WEIGHT_R": ("KING_ATTACK_WEIGHTS", (4,)),
    "KING_ATTACK_WEIGHT_Q": ("KING_ATTACK_WEIGHTS", (5,)),
    "THREAT_SAFE_PAWN_MG": ("THREAT_SAFE_PAWN", (0,)),
    "THREAT_SAFE_PAWN_EG": ("THREAT_SAFE_PAWN", (1,)),
    "THREAT_KNIGHT_ON_QUEEN_MG": ("THREAT_KNIGHT_ON_QUEEN", (0,)),
    "THREAT_KNIGHT_ON_QUEEN_EG": ("THREAT_KNIGHT_ON_QUEEN", (1,)),
    "THREAT_SLIDER_ON_QUEEN_MG": ("THREAT_SLIDER_ON_QUEEN", (0,)),
    "THREAT_SLIDER_ON_QUEEN_EG": ("THREAT_SLIDER_ON_QUEEN", (1,)),
    "THREAT_BY_KING_MG": ("THREAT_BY_KING", (0,)),
    "THREAT_BY_KING_EG": ("THREAT_BY_KING", (1,)),
    "THREAT_HANGING_MG": ("THREAT_HANGING", (0,)),
    "THREAT_HANGING_EG": ("THREAT_HANGING", (1,)),
    "THREAT_RESTRICTED_PIECE_MG": ("THREAT_RESTRICTED_PIECE", (0,)),
    "THREAT_RESTRICTED_PIECE_EG": ("THREAT_RESTRICTED_PIECE", (1,)),
    "THREAT_PAWN_PUSH_MG": ("THREAT_PAWN_PUSH", (0,)),
    "THREAT_PAWN_PUSH_EG": ("THREAT_PAWN_PUSH", (1,)),
}

for _pt, _sym in enumerate(("P", "N", "B", "R", "Q")):
    _ARRAY_PARAM_MAP[f"THREAT_BY_MINOR_{_sym}_MG"] = ("THREAT_BY_MINOR", (_pt, 0))
    _ARRAY_PARAM_MAP[f"THREAT_BY_MINOR_{_sym}_EG"] = ("THREAT_BY_MINOR", (_pt, 1))
    _ARRAY_PARAM_MAP[f"THREAT_BY_ROOK_{_sym}_MG"] = ("THREAT_BY_ROOK", (_pt, 0))
    _ARRAY_PARAM_MAP[f"THREAT_BY_ROOK_{_sym}_EG"] = ("THREAT_BY_ROOK", (_pt, 1))


def load_params_dict(path: Path) -> Dict[str, int]:
    """Load integer parameter dict from best_params.json, latest_params.json, or flat JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        keys = (
            ("theta_int", "theta", "best_theta_int", "best_theta")
            if path.name.startswith("latest")
            else ("best_theta_int", "theta_int", "theta", "best_theta")
        )
        for key in keys:
            if isinstance(data.get(key), dict):
                data = data[key]
                break
    out: Dict[str, int] = {}
    for k, v in data.items():
        name = str(k)
        if name not in EVAL_PARAM_INDEX:
            raise ValueError(f"Unknown eval parameter {name!r} in {path}")
        out[name] = int(round(float(v)))
    return out


def build_constants_patch(
    params: Dict[str, int],
    target_path: Path = DEFAULT_TARGET,
    verbose: bool = False,
) -> Tuple[str, str, list[str]]:
    """Return ``(updated_constants_src, snippet_src, changed_const_names)``."""
    target_src = target_path.read_text(encoding="utf-8")
    target_ns = _eval_const_module(target_path)
    target_nodes = _locate_target_nodes(target_src.replace("\r\n", "\n").replace("\r", "\n"))

    tuned_ns: dict = {}
    ordered_const_names: list[str] = []

    def _ensure_const(cname: str) -> None:
        if cname in tuned_ns:
            return
        if cname not in target_ns:
            raise KeyError(f"Constant {cname!r} not found in {target_path}")
        val = target_ns[cname]
        tuned_ns[cname] = np.array(val, copy=True) if isinstance(val, np.ndarray) else int(val)
        ordered_const_names.append(cname)

    for pname, raw_val in params.items():
        if pname not in EVAL_PARAM_INDEX:
            raise ValueError(f"Unknown eval parameter: {pname!r}")
        iv = int(round(float(raw_val)))
        if pname in _ARRAY_PARAM_MAP:
            cname, idx = _ARRAY_PARAM_MAP[pname]
            _ensure_const(cname)
            tuned_ns[cname][idx] = iv
        else:
            _ensure_const(pname)
            tuned_ns[pname] = iv

    tuned_assigns: Dict[str, str] = {}
    for cname in ordered_const_names:
        node = target_nodes.get(cname)
        wrap = _is_np_int32_scalar_call(getattr(node, "value", None)) if node else False
        tuned_assigns[cname] = _format_assignment(cname, tuned_ns[cname], wrap_np_int32=wrap)

    updated_src, changed, _ = apply_tuned_source(
        target_src=target_src,
        tuned_ns=tuned_ns,
        target_ns=target_ns,
        tuned_assigns=tuned_assigns,
        names=ordered_const_names,
        verbose=verbose,
    )

    snippet_lines = [
        "# Auto-generated by tune_eval_match.export_constants",
        "import numpy as np",
        "",
    ]
    for cname in changed:
        snippet_lines.append(tuned_assigns[cname])
    if not changed:
        snippet_lines.append("# (no numeric differences vs baseline constants.py)")
    snippet_src = "\n".join(snippet_lines) + "\n"
    return updated_src, snippet_src, changed


def export_params_to_files(
    params: Dict[str, int],
    *,
    target_path: Path = DEFAULT_TARGET,
    out_constants: Path | None = None,
    out_snippet: Path | None = None,
    apply_inplace: bool = False,
    verbose: bool = False,
) -> list[str]:
    """Export updated constants.py and/or snippet file, optionally applying in place."""
    updated_src, snippet_src, changed = build_constants_patch(
        params, target_path=target_path, verbose=verbose
    )
    if out_constants is not None:
        out_constants.parent.mkdir(parents=True, exist_ok=True)
        with out_constants.open("w", encoding="utf-8", newline="\n") as f:
            f.write(updated_src)
    if out_snippet is not None:
        out_snippet.parent.mkdir(parents=True, exist_ok=True)
        with out_snippet.open("w", encoding="utf-8", newline="\n") as f:
            f.write(snippet_src)
    if apply_inplace and changed:
        with target_path.open("w", encoding="utf-8", newline="\n") as f:
            f.write(updated_src)
    return changed


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params-json", type=Path, required=True, help="best_params.json / latest_params.json")
    ap.add_argument("--target", type=Path, default=DEFAULT_TARGET, help="Target constants.py")
    ap.add_argument("--out-constants", type=Path, default=None, help="Write full patched constants.py copy here")
    ap.add_argument("--out-snippet", type=Path, default=None, help="Write changed assignments snippet here")
    ap.add_argument("--apply", action="store_true", help="Also write changes back into --target in place")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)

    params = load_params_dict(args.params_json)
    changed = export_params_to_files(
        params,
        target_path=args.target,
        out_constants=args.out_constants,
        out_snippet=args.out_snippet,
        apply_inplace=args.apply,
        verbose=not args.quiet,
    )
    print(f"[export_constants] {len(changed)} constant symbol(s) changed: {changed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
