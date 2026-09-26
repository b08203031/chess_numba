"""Apply registry runtime axes to a live ``SearchContext``.

This tiny Python-only bridge is intended for the upcoming in-process,
fixed-node matcher.  It deliberately refuses compile-time candidates instead
of silently pretending that mutating a context can change a module constant.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from tune_search.search_param_registry import (
    get_spec,
    quantize_integer_value,
    quantize_value,
)


def apply_runtime_params(
    search_context,
    params: Mapping[str, float],
    *,
    scale_mode: str = "registry",
) -> None:
    """Mutate ``search_context.tune`` for supplied runtime parameters.

    ``registry`` preserves the historical coarse registry grid.  The
    ``integer`` mode is used by the fine SPSA campaign: values remain bounded
    but are written on a one-unit integer grid so an update such as 460 -> 459
    is not rounded back to the 25-unit LMR registry step.
    """
    if scale_mode not in ("registry", "integer"):
        raise ValueError(f"unknown runtime parameter scale mode: {scale_mode!r}")
    quantizer = quantize_integer_value if scale_mode == "integer" else quantize_value
    tune = search_context.tune
    for name, value in params.items():
        spec = get_spec(name)
        if not spec.is_runtime:
            raise ValueError(
                f"{name} is compile-time; restart the engine instead of mutating SearchContext"
            )
        tune[spec.runtime_index] = np.int32(quantizer(name, value))


def read_runtime_params(search_context, names=None) -> dict:
    """Read live runtime values, useful for match checkpoints and audits."""
    if names is None:
        from tune_search.search_param_registry import profile_names

        names = profile_names("runtime")
    out = {}
    for name in names:
        spec = get_spec(name)
        if not spec.is_runtime:
            raise ValueError(f"{name} is not a runtime SearchContext parameter")
        out[name] = int(search_context.tune[spec.runtime_index])
    return out
