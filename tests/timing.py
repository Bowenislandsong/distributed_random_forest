"""Test timing helpers: GitHub macOS/Windows runners are often much slower than dev laptops."""

from __future__ import annotations

import os

_ON_CI = bool(os.environ.get("CI", "").lower() in ("1", "true", "yes"))


def max_wall_seconds(
    local_limit: float,
    *,
    ci_scale: float = 5.0,
) -> float:
    """Stricter on developer machines, relaxed in CI to avoid false failures."""
    if not _ON_CI:
        return local_limit
    return max(local_limit * ci_scale, local_limit + 2.0)


def max_per_sample_seconds(
    local_limit: float,
    *,
    ci_scale: float = 4.0,
) -> float:
    if not _ON_CI:
        return local_limit
    return local_limit * ci_scale
