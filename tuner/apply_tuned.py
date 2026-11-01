"""
Merge tuner/tuned_constants.py into chess_engine/classical/constants.py.

SPSA always dumps the full parameter set (all registered names). This tool
compares **numeric values** (not text formatting) and only rewrites constants
that actually differ, using exact AST statement spans so surrounding comments,
inline comments, and blank lines in constants.py are preserved intact.
Optional --only restricts to named params.

Usage (from repo root)::

    python -m tuner.apply_tuned --dry-run
    python -m tuner.apply_tuned --only MG_MATERIAL_VALUES EG_MATERIAL_VALUES
    python -m tuner.apply_tuned
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TUNED = ROOT / "tuner" / "tuned_constants.py"
DEFAULT_TARGET = ROOT / "chess_engine" / "classical" / "constants.py"


def _extract_assignments(source: str) -> dict[str, str]:
    """Map name -> full assignment statement text (possibly multi-line)."""
    tree = ast.parse(source)
    lines = source.splitlines(True)
    out: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Name):
                start = node.lineno - 1
                end = node.end_lineno or node.lineno
                out[t.id] = "".join(lines[start:end])
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            start = node.lineno - 1
            end = node.end_lineno or node.lineno
            out[node.target.id] = "".join(lines[start:end])
    return out


def _is_np_int32_scalar_call(expr: ast.AST | None) -> bool:
    """Detect ``np.int32(x)`` scalar wrappers in target constants.py."""
    if not isinstance(expr, ast.Call) or len(expr.args) != 1:
        return False
    func = expr.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "int32"
        and isinstance(func.value, ast.Name)
        and func.value.id in ("np", "numpy")
    )


def _locate_target_nodes(source: str) -> dict[str, ast.Assign | ast.AnnAssign]:
    """Map top-level constant name -> AST assignment node (last wins if repeated)."""
    tree = ast.parse(source)
    nodes: dict[str, ast.Assign | ast.AnnAssign] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Name):
                nodes[t.id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            nodes[node.target.id] = node
    return nodes


def _eval_const_module(path: Path) -> dict:
    """Exec a constants-like file and return its globals (numpy available)."""
    ns: dict = {"np": np, "numpy": np}
    code = path.read_text(encoding="utf-8")
    # classical constants may call helpers; exec full file
    try:
        exec(compile(code, str(path), "exec"), ns)
    except Exception:
        # Fallback: only evaluate simple assignments from tuned dump style
        ns = {"np": np, "numpy": np}
        for name, stmt in _extract_assignments(code).items():
            try:
                exec(stmt, ns)
            except Exception:
                pass
    return ns


def _values_equal(a, b) -> bool:
    try:
        aa = np.asarray(a)
        bb = np.asarray(b)
        if aa.shape != bb.shape:
            # scalar vs 0-d
            if aa.size == 1 and bb.size == 1:
                return int(np.round(float(aa.ravel()[0]))) == int(np.round(float(bb.ravel()[0])))
            return False
        if aa.dtype.kind in "fc" or bb.dtype.kind in "fc":
            return np.allclose(aa.astype(np.float64), bb.astype(np.float64), rtol=0, atol=0.51)
        return np.array_equal(np.round(aa).astype(np.int64), np.round(bb).astype(np.int64))
    except Exception:
        return a == b


def _format_assignment(name: str, value, wrap_np_int32: bool = False) -> str:
    """Emit a constants.py-style assignment (without trailing newline)."""
    arr = np.asarray(value)
    if arr.ndim == 0 or (
        arr.size == 1 and arr.ndim <= 1 and not isinstance(value, (list, np.ndarray))
    ):
        v = int(np.round(float(np.asarray(value).ravel()[0])))
        if wrap_np_int32:
            return f"{name} = np.int32({v})"
        return f"{name} = {v}"
    arr = np.round(arr).astype(int)
    if arr.ndim == 1:
        body = ", ".join(str(int(x)) for x in arr.tolist())
        return f"{name} = np.array([{body}], dtype=np.int32)"
    if arr.ndim == 2:
        lines = [f"{name} = np.array(["]
        for row in arr:
            row_str = ", ".join(str(int(x)) for x in row.tolist())
            lines.append(f"    [{row_str}],")
        lines.append("], dtype=np.int32)")
        return "\n".join(lines)
    return f"{name} = np.array({arr.tolist()}, dtype=np.int32)"


def apply_tuned_source(
    target_src: str,
    tuned_ns: dict,
    target_ns: dict,
    tuned_assigns: dict[str, str],
    names: list[str],
    verbose: bool = True,
) -> tuple[str, list[str], int]:
    """Rewrite numerically changed constants in ``target_src`` preserving all comments."""
    normalized_src = target_src.replace("\r\n", "\n").replace("\r", "\n")
    target_nodes = _locate_target_nodes(normalized_src)
    lines = normalized_src.splitlines(True)

    changed: list[str] = []
    skipped_same = 0
    edits: list[tuple[int, int, str, str, str]] = []

    for name in names:
        if name not in target_ns:
            if verbose:
                print(f"Skip (not in target module): {name}")
            continue
        if name not in tuned_ns:
            if verbose:
                print(f"Skip (not evaluable in tuned): {name}")
            continue
        if _values_equal(tuned_ns[name], target_ns[name]):
            skipped_same += 1
            continue

        node = target_nodes.get(name)
        if node is None:
            if verbose:
                print(f"Skip (AST assignment not found in target): {name}")
            continue

        wrap_np_int32 = _is_np_int32_scalar_call(getattr(node, "value", None))
        stmt = tuned_assigns.get(name)
        if stmt is None or wrap_np_int32:
            replacement_body = _format_assignment(
                name, tuned_ns[name], wrap_np_int32=wrap_np_int32
            )
        else:
            replacement_body = stmt.strip()

        start_idx = node.lineno - 1
        end_idx = (node.end_lineno or node.lineno) - 1
        start_line_bytes = lines[start_idx].encode("utf-8")
        leading_prefix = start_line_bytes[: node.col_offset].decode("utf-8")
        end_line_bytes = lines[end_idx].encode("utf-8")
        end_col = node.end_col_offset if node.end_col_offset is not None else len(end_line_bytes)
        trailing_suffix = end_line_bytes[end_col:].decode("utf-8")
        if not trailing_suffix.endswith("\n"):
            trailing_suffix += "\n"

        spliced = leading_prefix + replacement_body + trailing_suffix

        old_val = target_ns[name]
        new_val = tuned_ns[name]
        try:
            o = np.asarray(old_val).ravel()[:6].tolist()
            n = np.asarray(new_val).ravel()[:6].tolist()
            detail = f"  {name}: {o} -> {n}"
        except Exception:
            detail = f"  {name}"

        edits.append((start_idx, end_idx, name, detail, spliced))

    for _, _, name, detail, _ in edits:
        changed.append(name)
        if verbose:
            print(detail)

    # Apply in reverse line order so earlier AST line spans stay exact
    for start_idx, end_idx, _, _, spliced in sorted(
        edits, key=lambda item: item[0], reverse=True
    ):
        lines[start_idx : end_idx + 1] = [spliced]

    return "".join(lines), changed, skipped_same


def apply_tuned(
    tuned_path: Path,
    target_path: Path,
    dry_run: bool = False,
    only: list[str] | None = None,
) -> list[str]:
    tuned_ns = _eval_const_module(tuned_path)
    target_ns = _eval_const_module(target_path)
    tuned_assigns = _extract_assignments(tuned_path.read_text(encoding="utf-8"))

    names = list(tuned_assigns.keys())
    if only:
        only_set = set(only)
        names = [n for n in names if n in only_set]
        missing = only_set - set(names)
        for m in sorted(missing):
            print(f"Skip (not in tuned file): {m}")

    target_src = target_path.read_text(encoding="utf-8")
    new_src, changed, skipped_same = apply_tuned_source(
        target_src=target_src,
        tuned_ns=tuned_ns,
        target_ns=target_ns,
        tuned_assigns=tuned_assigns,
        names=names,
        verbose=True,
    )

    print(
        f"Summary: {len(changed)} numeric change(s), "
        f"{skipped_same} identical (skipped), "
        f"{len(tuned_assigns)} names in tuned dump."
    )

    if dry_run:
        print(f"[dry-run] Would update {len(changed)} constant(s) in {target_path.name}")
        return changed

    if not changed:
        print("No numeric changes to apply.")
        return changed

    with target_path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(new_src)
    try:
        shown = target_path.resolve().relative_to(ROOT.resolve())
    except ValueError:
        shown = target_path
    print(f"Updated {len(changed)} constant(s) in {shown}")
    print("Next: python -m tuner.sync_constants_snapshot, light tests, then optional tournament vs classical_old.")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply numerically changed constants from tuned_constants.py into classical/constants.py"
    )
    parser.add_argument("--tuned", type=Path, default=DEFAULT_TUNED)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Only consider these parameter names (e.g. MG_MATERIAL_VALUES EG_MATERIAL_VALUES)",
    )
    args = parser.parse_args()
    if not args.tuned.exists():
        raise SystemExit(f"Missing {args.tuned}. Run SPSA first.")
    apply_tuned(args.tuned, args.target, dry_run=args.dry_run, only=args.only)


if __name__ == "__main__":
    main()
