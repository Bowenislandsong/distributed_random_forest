"""CPU parallelism helpers (same ``n_jobs`` rules as scikit-learn / joblib).

- ``n_jobs is None or n_jobs == 0`` → **1** (no parallel pool).
- ``n_jobs == -1`` → all logical CPUs.
- A positive int → that many jobs (capped to available hardware).

Set ``n_jobs=1`` for deterministic, single-threaded runs (e.g. tiny tests, debugging).
Set ``n_jobs=-1`` for best throughput (defaults on
:class:`~distributed_random_forest.models.random_forest.RandomForest` and
:class:`~distributed_random_forest.federation.aggregator.FederatedAggregator`).
"""

from __future__ import annotations

from typing import Optional

from joblib import effective_n_jobs

__all__ = ["resolve_n_jobs"]


def resolve_n_jobs(n_jobs: Optional[int]) -> int:
    """
    Return a positive worker count for :class:`joblib.Parallel` and ``n_jobs`` kwargs.

    Args:
        n_jobs: If ``None`` or ``0``, returns ``1`` (sequential). Otherwise
            :func:`joblib.effective_n_jobs` is used (``-1`` = all cores).
    """
    if n_jobs in (None, 0):
        return 1
    return int(effective_n_jobs(n_jobs))
