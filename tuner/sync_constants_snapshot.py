"""
Sync tuner/constants_snapshot.py from chess_engine/classical/constants.py.

Run after any HCE constant change before SPSA:

    python -m tuner.sync_constants_snapshot
"""
from __future__ import annotations

import argparse
import ast
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, Optional, Set

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLASSICAL_DIR = ROOT / "chess_engine" / "classical"
SRC = CLASSICAL_DIR / "constants.py"
DST = ROOT / "tuner" / "constants_snapshot.py"
SHIM = ROOT / "tuner" / "constants_to_be_tuned.py"
CLASSICAL_CACHE_DIR = CLASSICAL_DIR / "__pycache__"

HEADER = """# AUTO-GENERATED snapshot of chess_engine/classical/constants.py for HCE SPSA.
# Do not edit by hand. Regenerate with: python -m tuner.sync_constants_snapshot
# Source of truth remains classical/constants.py

"""

SHIM_TEXT = """# Backward-compatible alias. Prefer tuner.constants_snapshot.
from tuner.constants_snapshot import *  # noqa: F403
"""


def _is_jit_decorator(dec: ast.expr) -> bool:
    target = dec.func if isinstance(dec, ast.Call) else dec
    if isinstance(target, ast.Name):
        return target.id in {"jit", "njit"}
    if isinstance(target, ast.Attribute):
        return target.attr in {"jit", "njit"}
    return False


def _load_constants_ns(path: Path) -> Dict[str, object]:
    ns: Dict[str, object] = {}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), ns)
    return {
        k: v
        for k, v in ns.items()
        if not k.startswith("__") and not callable(v) and type(v).__name__ != "module"
    }


def _values_equal(a: object, b: object) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (
            isinstance(a, np.ndarray)
            and isinstance(b, np.ndarray)
            and a.dtype == b.dtype
            and np.array_equal(a, b)
        )
    if type(a) is not type(b):
        return False
    return bool(a == b)


def diff_constant_names(old_path: Path, new_path: Path) -> Optional[Set[str]]:
    """Return symbol names whose runtime values differ between ``old_path`` and ``new_path``."""
    if not old_path.is_file():
        return None
    old_ns = _load_constants_ns(old_path)
    new_ns = _load_constants_ns(new_path)
    changed: Set[str] = set()
    for name in old_ns.keys() | new_ns.keys():
        if name not in old_ns or name not in new_ns:
            changed.add(name)
        elif not _values_equal(old_ns[name], new_ns[name]):
            changed.add(name)
    return changed


def _collect_names_and_attrs(node: ast.AST) -> tuple[Set[str], Set[tuple[str, str]]]:
    names: Set[str] = set()
    attrs: Set[tuple[str, str]] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            names.add(child.id)
        elif isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name):
            attrs.add((child.value.id, child.attr))
    return names, attrs


def _resolve_classical_module(node: ast.ImportFrom) -> Optional[str]:
    mod = node.module or ""
    if node.level == 0 and mod.startswith("chess_engine.classical."):
        return mod.split(".", 3)[2]
    if node.level == 1 and mod:
        return mod.split(".", 1)[0]
    return None


def _parse_module_deps(path: Path, module_names: Set[str]) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    const_imports: Dict[str, str] = {}
    const_star = False
    const_aliases: Set[str] = set()
    sibling_imports: Dict[str, str] = {}
    sibling_aliases: Dict[str, str] = {}
    local_assigns: Dict[str, Set[str]] = {}
    jit_body_names: Set[str] = set()
    jit_body_attrs: Set[tuple[str, str]] = set()

    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            target = _resolve_classical_module(node)
            if target == "constants":
                for alias in node.names:
                    if alias.name == "*":
                        const_star = True
                    else:
                        const_imports[alias.asname or alias.name] = alias.name
            elif target in module_names:
                for alias in node.names:
                    if alias.name != "*":
                        sibling_imports[alias.asname or alias.name] = target
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "chess_engine.classical.constants":
                    const_aliases.add(alias.asname or alias.name)
                elif alias.name.startswith("chess_engine.classical."):
                    target = alias.name.split(".", 3)[2]
                    if target in module_names:
                        sibling_aliases[alias.asname or alias.name] = target
        elif isinstance(node, ast.FunctionDef) and any(_is_jit_decorator(d) for d in node.decorator_list):
            for stmt in node.body:
                n, a = _collect_names_and_attrs(stmt)
                jit_body_names |= n
                jit_body_attrs |= a
        elif isinstance(node, ast.Assign) and node.value is not None:
            rhs_names, _ = _collect_names_and_attrs(node.value)
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    local_assigns.setdefault(tgt.id, set()).update(rhs_names)
        elif isinstance(node, ast.AnnAssign) and node.value is not None and isinstance(node.target, ast.Name):
            rhs_names, _ = _collect_names_and_attrs(node.value)
            local_assigns.setdefault(node.target.id, set()).update(rhs_names)

    # Expand module-level globals (e.g. SEE_PIECE_VALUES) referenced inside @njit bodies.
    expanded_jit_names = set(jit_body_names)
    queue = deque(jit_body_names)
    while queue:
        sym = queue.popleft()
        for dep in local_assigns.get(sym, ()):
            if dep not in expanded_jit_names:
                expanded_jit_names.add(dep)
                queue.append(dep)

    used_consts = {const_imports[n] for n in expanded_jit_names if n in const_imports}
    if const_star:
        used_consts |= expanded_jit_names
    used_consts |= {attr for mod, attr in jit_body_attrs if mod in const_aliases}

    callees = {sibling_imports[n] for n in expanded_jit_names if n in sibling_imports}
    callees |= {sibling_aliases[mod] for mod, _ in jit_body_attrs if mod in sibling_aliases}

    return {"used_consts": used_consts, "callees": callees}


def affected_classical_modules(
    changed_consts: Iterable[str], classical_dir: Path = CLASSICAL_DIR
) -> Set[str]:
    """Return classical module stems whose Numba cache may bake any symbol in ``changed_consts``.

    Numba ``cache=True`` embeds callee LLVM bitcode into caller ``.nbc`` archives,
    so any transitive ``@njit`` caller of a directly affected module is included.
    Pure-Python modules/call sites (e.g. ``engine_types.SearchContext.__init__``)
    read ``constants.py`` at runtime and do not require cache invalidation.
    """
    changed = set(changed_consts)
    if not changed:
        return set()

    py_files = {
        p.stem: p
        for p in classical_dir.glob("*.py")
        if p.stem not in {"constants", "__init__"}
    }
    deps = {stem: _parse_module_deps(path, set(py_files)) for stem, path in py_files.items()}

    reverse_edges: Dict[str, Set[str]] = {stem: set() for stem in py_files}
    dirty: Set[str] = set()
    for stem, info in deps.items():
        if info["used_consts"] & changed:
            dirty.add(stem)
        for callee in info["callees"]:
            reverse_edges.setdefault(callee, set()).add(stem)

    queue = deque(dirty)
    while queue:
        mod = queue.popleft()
        for caller in reverse_edges.get(mod, ()):
            if caller not in dirty:
                dirty.add(caller)
                queue.append(caller)
    return dirty


def clear_classical_numba_cache(
    cache_dir: Path = CLASSICAL_CACHE_DIR,
    modules: Optional[Iterable[str]] = None,
) -> int:
    """Remove rebuildable Numba caches that may have frozen old HCE globals.

    When ``modules`` is ``None``, all ``*.nbi`` / ``*.nbc`` files in ``cache_dir``
    are removed.  When ``modules`` is given, only cache files whose leading
    ``<module>.<func>-...`` prefix matches ``modules`` are removed.
    """
    if not cache_dir.is_dir():
        return 0
    mod_set = None if modules is None else set(modules)
    if mod_set is not None and not mod_set:
        return 0
    removed = 0
    for path in cache_dir.iterdir():
        if not path.is_file() or path.suffix not in {".nbi", ".nbc"}:
            continue
        if mod_set is not None and path.name.split(".", 1)[0] not in mod_set:
            continue
        path.unlink()
        removed += 1
    return removed


def main(argv: Optional[Iterable[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--clear-all",
        action="store_true",
        help="Delete every classical Numba cache file instead of selective invalidation",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    body = SRC.read_text(encoding="utf-8")
    snapshot = HEADER + body
    snapshot_changed = not DST.exists() or DST.read_text(encoding="utf-8") != snapshot

    if args.clear_all:
        if snapshot_changed:
            with DST.open("w", encoding="utf-8", newline="\n") as fh:
                fh.write(snapshot)
        removed = clear_classical_numba_cache(modules=None)
        print(
            f"Synced {DST.relative_to(ROOT)} (--clear-all); "
            f"removed {removed} classical Numba cache file(s)"
        )
    elif snapshot_changed:
        changed_consts = diff_constant_names(DST, SRC)
        with DST.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(snapshot)
        modules = None if changed_consts is None else affected_classical_modules(changed_consts)
        removed = clear_classical_numba_cache(modules=modules)
        scope = "all modules" if modules is None else (
            "none (no @njit constant changes)" if not modules else ", ".join(sorted(modules))
        )
        print(
            f"Wrote {DST.relative_to(ROOT)} from {SRC.relative_to(ROOT)}; "
            f"invalidated [{scope}], removed {removed} stale classical Numba cache file(s)"
        )
    else:
        print(f"Unchanged {DST.relative_to(ROOT)}")

    if not SHIM.exists() or SHIM.read_text(encoding="utf-8") != SHIM_TEXT:
        with SHIM.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(SHIM_TEXT)


if __name__ == "__main__":
    main()
