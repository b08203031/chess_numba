"""Fixed-node match backend for runtime search-parameter SPSA.

This is a thin adapter over the already-tested ``tune_eval_match`` worker
pool.  It keeps one Python process and one Numba compilation alive while SPSA
iterations mutate the 34 live search slots.  Compile-time registry entries
are intentionally rejected here and must use the UCI adapter backend.
"""

from __future__ import annotations

from typing import Sequence

from tune_eval_match.inprocess_match import (
    expand_worker_pool,
    load_openings,
    make_search_context,
    make_worker_pool,
    run_match as _run_match,
    warmup_jit,
)
from tune_search.runtime_apply import apply_runtime_params
from tune_search.search_param_registry import get_spec


def set_search_params(ctx, params: dict, *, scale_mode: str = "registry") -> None:
    """Apply a runtime-only parameter dictionary to one live context."""
    apply_runtime_params(ctx, params, scale_mode=scale_mode)


def validate_runtime_params(params: dict) -> None:
    """Fail before a long match if a compile-time axis was selected."""
    for name in params:
        if not get_spec(name).is_runtime:
            raise ValueError(
                f"{name} is compile-time; use --backend uci for this parameter"
            )


def run_match(
    params_a: dict,
    params_b: dict,
    openings: Sequence[str],
    nodes: int,
    games: int,
    *,
    log_every: int = 10,
    worker_pool=None,
    ctx_a=None,
    ctx_b=None,
    concurrency: int = 1,
    early_trash_min_games: int = 0,
    early_trash_max_score: float = 0.45,
    scale_mode: str = "registry",
    opening_offset: int = 0,
    adjudicate: dict | None = None,
) -> dict:
    """Run a fixed-node search-parameter match without spawning UCI engines."""
    validate_runtime_params(params_a)
    validate_runtime_params(params_b)
    if scale_mode not in ("registry", "integer"):
        raise ValueError(f"unknown runtime parameter scale mode: {scale_mode!r}")

    def _set_params(ctx, params):
        set_search_params(ctx, params, scale_mode=scale_mode)

    return _run_match(
        params_a,
        params_b,
        openings=openings,
        nodes=int(nodes),
        games=int(games),
        log_every=int(log_every),
        worker_pool=worker_pool,
        ctx_a=ctx_a,
        ctx_b=ctx_b,
        concurrency=int(concurrency),
        early_trash_min_games=int(early_trash_min_games),
        early_trash_max_score=float(early_trash_max_score),
        param_setter=_set_params,
        opening_offset=int(opening_offset),
        adjudicate=adjudicate,
    )


__all__ = [
    "expand_worker_pool",
    "load_openings",
    "make_search_context",
    "make_worker_pool",
    "run_match",
    "set_search_params",
    "validate_runtime_params",
    "warmup_jit",
]
