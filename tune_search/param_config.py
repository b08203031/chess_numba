"""Compatibility facade for the search-parameter registry.

Keep this module small because older scripts import ``SEARCH_PARAMS`` and the
four helper functions directly.  New code should use
``search_param_registry`` so it can select a staged profile and distinguish
runtime-backed values from compile-time constants.
"""

from tune_search.search_param_registry import (
    DEFAULT_PROFILE,
    PROFILE_NAMES,
    SEARCH_PARAM_BY_NAME,
    SEARCH_PARAM_SPECS,
    SearchParamSpec,
    get_bounds,
    get_default_params as _get_default_params,
    get_spec,
    get_step,
    legacy_search_params,
    profile_names,
    quantize_integer_value,
    quantize_value,
    resolve_names,
    validate_registry,
)


# Legacy shape: name -> (default, min, max, step).  This now contains the
# complete registry rather than the previous stale 32-entry list.
SEARCH_PARAMS = legacy_search_params()


def get_param_names(profile: str = DEFAULT_PROFILE):
    return list(profile_names(profile))


def get_default_params(names=None):
    """Return defaults for explicit names, or all registry entries."""
    return _get_default_params(names=names)


__all__ = [
    "DEFAULT_PROFILE",
    "PROFILE_NAMES",
    "SEARCH_PARAMS",
    "SEARCH_PARAM_BY_NAME",
    "SEARCH_PARAM_SPECS",
    "SearchParamSpec",
    "get_bounds",
    "get_default_params",
    "get_param_names",
    "get_spec",
    "get_step",
    "legacy_search_params",
    "profile_names",
    "quantize_integer_value",
    "quantize_value",
    "resolve_names",
    "validate_registry",
]
