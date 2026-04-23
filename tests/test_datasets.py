"""Unit tests for public reference datasets."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from distributed_random_forest.datasets import load_breast_cancer_bench, summarize_split
from distributed_random_forest.datasets.public import BreastCancerBenchmark


class TestBreastCancerLoader:
    """Tests for :func:`load_breast_cancer_bench`."""

    def test_row_counts_sum_to_569(self):
        split = load_breast_cancer_bench(random_state=0)
        n = len(split.y_train) + len(split.y_val) + len(split.y_test)
        assert n == 569
        assert split.n_features == 30
        assert split.n_classes == 2

    def test_no_nan_in_features(self):
        split = load_breast_cancer_bench()
        for arr in (split.X_train, split.X_val, split.X_test):
            assert not np.isnan(arr).any()

    def test_frame_populated_when_requested(self):
        split = load_breast_cancer_bench(as_frame=True)
        assert split.data_frame is not None
        assert split.data_frame.shape[0] == 569
        assert "target" in split.data_frame.columns

    def test_summarize_split(self):
        s = load_breast_cancer_bench()
        d = summarize_split(s)
        assert d["n_train"] + d["n_val"] + d["n_test"] == 569
        assert d["n_features"] == 30

    def test_feature_names_length(self):
        s = load_breast_cancer_bench()
        assert len(s.feature_names) == 30
        for name in s.feature_names:
            assert isinstance(name, str) and name


class TestBreastCancerBenchmarkType:
    """Type-level checks on the dataclass."""

    def test_is_frozen(self):
        s = load_breast_cancer_bench()
        with pytest.raises(FrozenInstanceError):
            s.name = "x"  # type: ignore[misc]
        assert isinstance(s, BreastCancerBenchmark)
