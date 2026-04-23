"""Pytest configuration and shared fixtures."""

import numpy as np
import pytest

from distributed_random_forest.datasets.public import BreastCancerBenchmark, load_breast_cancer_bench


@pytest.fixture
def breast_cancer_split() -> BreastCancerBenchmark:
    """UCI WDBC stratified train/val/test (fixed seed for reproducible tests)."""
    return load_breast_cancer_bench(as_frame=True, random_state=42)


@pytest.fixture
def random_xy_small():
    """Small synthetic 2D dataset for quick checks."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    return X, y
