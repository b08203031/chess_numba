"""
Generate tuner/tunable_eval.py from classical evaluation stack.

Merges material.py + pawns.py + evaluation.py, injects theta getters for
parameters registered in tuner.parameters.ParameterManager, and renames the
entry point to evaluate_position_tunable(..., theta) -> score only.
"""
from __future__ import annotations

import re
from pathlib import Path

import tuner.parameters as parameters
from tuner.codegen.generate_indices import generate_indices_text

ROOT = Path(__file__).resolve().parents[2]
EVAL_PATH = ROOT / "chess_engine" / "classical" / "evaluation.py"
PAWNS_PATH = ROOT / "chess_engine" / "classical" / "pawns.py"
MATERIAL_PATH = ROOT / "chess_engine" / "classical" / "material.py"
OUT_PATH = ROOT / "tuner" / "tunable_eval.py"
INDICES_PATH = Path(__file__).resolve().parent / "new_indices.txt"


def _strip_module_header(text: str) -> str:
    """Remove leading comments/imports up to first non-import code of interest."""
    lines = text.splitlines(True)
    out = []
    past_imports = False
    for line in lines:
        if not past_imports:
            s = line.strip()
            if (
                s.startswith("import ")
                or s.startswith("from ")
                or s.startswith("#")
                or s.startswith('"""')
                or s.startswith("'''")
                or s == ""
            ):
                # keep only blank / comment after we started? skip all header imports
                continue
            past_imports = True
        out.append(line)
    return "".join(out)


def _strip_imports_only(text: str) -> str:
    """Remove import statements, including multi-line ``from x import ( ... )``."""
    lines = text.splitlines(True)
    out = []
    skipping_paren_import = False
    for line in lines:
        s = line.strip()
        if skipping_paren_import:
            if ")" in s:
                skipping_paren_import = False
            continue
        if s.startswith("import ") or s.startswith("from "):
            # Multi-line: from foo import (
            if "(" in s and ")" not in s:
                skipping_paren_import = True
            continue
        out.append(line)
    return "".join(out)


def _param_meta():
    pm = parameters.ParameterManager()
    meta = {}
    for p in pm.param_map:
        meta[p["name"]] = p
    return meta


def _apply_theta_rewrites(text: str, meta: dict) -> str:
    # Longer names first to avoid partial replacements
    names = sorted(meta.keys(), key=len, reverse=True)

    for name in names:
        shape = meta[name]["shape"]
        # 2D array access NAME[r, c] or NAME[r][c]
        if len(shape) == 2:
            rows, cols = shape
            # NAME[a, b]
            text = re.sub(
                rf"\b{name}\[([^,\]]+),\s*([^\]]+)\]",
                rf"get_2d_val(theta, IDX_{name}, \1, {cols}, \2)",
                text,
            )
        if len(shape) == 1 and shape[0] > 1:
            text = re.sub(
                rf"\b{name}\[([^\]]+)\]",
                rf"get_array_val(theta, IDX_{name}, \1)",
                text,
            )
        elif shape == (1,) or (len(shape) == 1 and shape[0] == 1):
            # scalar stored as 1-element or true scalar count==1
            text = re.sub(rf"\b{name}\b", f"get_int(theta, IDX_{name})", text)
        elif len(shape) == 1:
            text = re.sub(
                rf"\b{name}\[([^\]]+)\]",
                rf"get_array_val(theta, IDX_{name}, \1)",
                text,
            )

    # True scalars (count==1 registered as shape (1,)) already handled.
    # Also handle bare name uses for multi-element arrays (rare) — skip.

    # Re-process pure scalars: parameters with count 1 were given shape (1,)
    # Some code uses bare scalar names without indexing — already get_int.

    return text


def _inject_theta_arg(text: str, func_names: list[str]) -> str:
    """
    For each registered function name, append ``theta`` to every call and
    definition argument list. Handles multi-line argument lists via paren depth.
    """
    names = set(func_names)
    # Sort longer first so e.g. evaluate_pawn_structure_cached before evaluate_pawn_structure
    ordered = sorted(names, key=len, reverse=True)
    # Build regex that finds name( with word boundary; not preceded by .
    pattern = re.compile(
        r"(?<![\w.])(" + "|".join(re.escape(n) for n in ordered) + r")\s*\("
    )

    out = []
    i = 0
    n = len(text)
    while i < n:
        m = pattern.search(text, i)
        if not m:
            out.append(text[i:])
            break
        out.append(text[i : m.start()])
        name = m.group(1)
        # Find matching close paren
        j = m.end()  # position after '('
        depth = 1
        while j < n and depth > 0:
            ch = text[j]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            j += 1
        # args text is text[m.end():j-1]
        args = text[m.end() : j - 1]
        # Skip numba type signatures in decorators: look like piece_bbs_signature without real args?
        # Always inject unless theta already present as a bare argument token.
        if re.search(r"(?<![\w])theta(?![\w])", args):
            new_call = text[m.start() : j]
        else:
            args_stripped = args.rstrip()
            if args_stripped:
                # preserve trailing whitespace/newlines inside parens roughly
                if args_stripped.endswith(","):
                    new_args = args + " theta"
                else:
                    # insert before final whitespace
                    trailing_ws = args[len(args.rstrip()) :]
                    new_args = args_stripped + ", theta" + trailing_ws
            else:
                new_args = "theta"
            new_call = f"{name}({new_args})"
        out.append(new_call)
        i = j
    return "".join(out)


def _fix_numba_signatures(text: str) -> str:
    """Append numba.float64[:] for theta to common njit signatures when present."""
    # Broad approach: for @numba.njit(...(...)) lines that decorate our tuned
    # functions, inject theta type if missing. Heuristic: any signature tuple
    # that ends with ')' before ', cache' — inject before final ')'.
    def inject(sig_line: str) -> str:
        if "float64[:]" in sig_line or "theta" in sig_line:
            return sig_line
        # Only touch signatures that look like function type specs
        if "numba." not in sig_line or "(" not in sig_line:
            return sig_line
        # Insert before the last ')' that closes the parameter type list of the
        # function signature (not the njit call). Pattern: ...)(TYPES)
        # Example: @numba.njit(numba.int32(...types...), cache=True)
        m = re.match(
            r"(@numba\.njit\()(.*)(\).*)$",
            sig_line.strip(),
        )
        if not m:
            return sig_line
        # Find the innermost parameter type list of the return-type(args) form
        core = m.group(2)
        # Inject before the last ')' of core if core has a function type
        if "cache" in sig_line and "(" in core:
            # Find last ')' before ', cache' or end inside njit args
            # Simpler: replace ", cache=" / ", boundscheck" — inject into the
            # type signature preceding first keyword arg.
            kw = re.search(r",\s*cache\s*=", sig_line)
            if not kw:
                return sig_line
            head = sig_line[: kw.start()]
            tail = sig_line[kw.start() :]
            if head.rstrip().endswith(")"):
                # insert before last )
                idx = head.rfind(")")
                head = head[:idx] + ", numba.float64[:]" + head[idx:]
                return head + tail
        return sig_line

    lines = text.splitlines(True)
    return "".join(inject(line) if line.lstrip().startswith("@numba.njit") else line for line in lines)


def build() -> str:
    meta = _param_meta()
    indices_text = generate_indices_text()
    INDICES_PATH.write_text(indices_text, encoding="utf-8")

    material = _strip_imports_only(MATERIAL_PATH.read_text(encoding="utf-8"))
    pawns = _strip_imports_only(PAWNS_PATH.read_text(encoding="utf-8"))
    evaluation = EVAL_PATH.read_text(encoding="utf-8")

    # Drop evaluation imports that we inline / replace
    evaluation = re.sub(
        r"from chess_engine\.classical\.constants import \*\r?\n", "", evaluation
    )
    evaluation = re.sub(
        r"from chess_engine\.classical\.pawns import \([\s\S]*?\)\r?\n",
        "",
        evaluation,
    )
    evaluation = re.sub(
        r"from chess_engine\.classical\.material import compute_imbalance\r?\n",
        "",
        evaluation,
    )

    # Keep endgame + bitboard + move_generator imports from evaluation header
    # Rebuild clean header
    header = f'''# AUTO-GENERATED by tuner/codegen/generate_tunable_eval.py — do not edit by hand.
# Regenerate after classical eval / parameters.py changes:
#   python -m tuner.codegen.generate_tunable_eval

import numba
import numpy as np

from chess_engine.classical.constants import *  # non-tuned constants + bitboards
from chess_engine.classical.endgame import get_endgame_scale_factor, evaluate_special_endgame_with_material
from chess_engine.classical.bitboard_utils import (
    get_lsb_index, get_msb_index, count_bits, WHITE_KING_ZONES, BLACK_KING_ZONES,
    FILE_MASKS, SQUARES_BETWEEN, ROOK_RAYS, BISHOP_RAYS,
)
from chess_engine.classical.engine_types import (
    piece_bbs_signature, occupancy_bbs_signature, game_state_signature, piece_counts_signature
)
from chess_engine.classical.move_generator import (
    get_bishop_attacks, get_rook_attacks, get_queen_attacks, KNIGHT_ATTACKS, PAWN_ATTACKS, KING_ATTACKS,
    get_pinned_pieces
)

{indices_text}

@numba.njit(numba.int32(numba.float64[:], numba.int32), cache=True, inline='always')
def get_int(theta, idx):
    return numba.int32(np.rint(theta[idx]))

@numba.njit(numba.int32(numba.float64[:], numba.int32, numba.int32), cache=True, inline='always')
def get_array_val(theta, base_idx, offset):
    return numba.int32(np.rint(theta[base_idx + offset]))

@numba.njit(numba.int32(numba.float64[:], numba.int32, numba.int32, numba.int32, numba.int32), cache=True, inline='always')
def get_2d_val(theta, base_idx, row, col_size, col):
    return numba.int32(np.rint(theta[base_idx + row * col_size + col]))

'''

    # evaluation body without its original import block
    eval_body = _strip_imports_only(evaluation)

    combined = material + "\n\n" + pawns + "\n\n" + eval_body

    # Function list that need theta parameter
    funcs = [
        "compute_imbalance",
        "_material_mg_eg",
        "_passed_rank_bonus",
        "evaluate_shelter_aligned",
        "evaluate_pawn_structure",
        "evaluate_pawn_structure_cached",
        "evaluate_piece_coordination",
        "evaluate_passed_pawns",
        "evaluate_king_safety",
        "evaluate_attacks_mobility_threats",
        "evaluate_space",
        "_evaluate_king_pawn_endgame",
        "_process_piece_score_and_count",
        "_evaluate_king_attackers",
        "_evaluate_pawn_shield_for_color",
        "_compute_initiative",
        "_evaluate_position_jit",
    ]
    # Only add theta to functions that exist in the combined source
    existing = [f for f in funcs if re.search(rf"def {f}\(", combined)]
    combined = _inject_theta_arg(combined, existing)
    combined = _apply_theta_rewrites(combined, meta)
    combined = _fix_numba_signatures(combined)

    # Entry points (njit so SPSA compute_mse can call it from nopython)
    combined = combined.replace(
        "def evaluate_position(piece_bbs, occupancy_bbs, game_state, lazy: bool = False):",
        "@numba.njit(cache=True, boundscheck=False, fastmath=True)\n"
        "def evaluate_position_tunable(piece_bbs, occupancy_bbs, game_state, theta, lazy=False):",
    )
    # Return score only for SPSA cost function
    combined = re.sub(
        r"return _evaluate_position_jit\(piece_bbs, occupancy_bbs, game_state, lazy(, theta)?\)",
        "score, _pw, _pb = _evaluate_position_jit(piece_bbs, occupancy_bbs, game_state, lazy, None, theta)\n"
        "    return score",
        combined,
    )
    # _evaluate_position_jit may need search_context=None + theta in signature
    # After theta injection, signature should be (... lazy, search_context=None, theta)
    # Fix call if still wrong
    if "def evaluate_position_tunable" in combined and "return score" not in combined:
        combined = combined.replace(
            "return _evaluate_position_jit(piece_bbs, occupancy_bbs, game_state, lazy, theta)",
            "score, _pw, _pb = _evaluate_position_jit(piece_bbs, occupancy_bbs, game_state, lazy, None, theta)\n"
            "    return score",
        )

    # Ensure _evaluate_position_jit accepts search_context before theta
    combined = re.sub(
        r"def _evaluate_position_jit\(piece_bbs, occupancy_bbs, game_state, lazy: bool, search_context=None, theta\):",
        "def _evaluate_position_jit(piece_bbs, occupancy_bbs, game_state, lazy, search_context, theta):",
        combined,
    )
    combined = re.sub(
        r"def _evaluate_position_jit\(piece_bbs, occupancy_bbs, game_state, lazy: bool, theta\):",
        "def _evaluate_position_jit(piece_bbs, occupancy_bbs, game_state, lazy, search_context, theta):",
        combined,
    )

    # Fix double theta or broken signatures from naive replace
    combined = combined.replace(", theta, theta)", ", theta)")
    combined = combined.replace("(theta, theta)", "(theta)")

    text = header + combined

    # Remove evaluate_pawn_structure_cached theta issues: force non-cached path is already
    # used when search_context is None.

    return text


def main() -> None:
    text = build()
    OUT_PATH.write_text(text, encoding="utf-8")
    print(f"Wrote {OUT_PATH} ({len(text)} bytes)")


if __name__ == "__main__":
    main()
