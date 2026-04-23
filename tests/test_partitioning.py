"""Tests for distributed partitioning strategies."""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from distributed_random_forest import create_partitions, summarize_partitions
from distributed_random_forest.distributed.partitioning import (
    partition_dirichlet,
    partition_label_skew,
    partition_stratified,
)


@pytest.fixture
def synthetic_data():
    """Create a moderately sized multiclass dataset."""
    X, y = make_classification(
        n_samples=300,
        n_features=12,
        n_classes=3,
        n_informative=8,
        random_state=42,
    )
    return X, y


def test_partition_stratified_preserves_total_size(synthetic_data):
    """Stratified partitioning should preserve all samples."""
    X, y = synthetic_data
    partitions = partition_stratified(X, y, n_clients=4, random_state=42)

    assert sum(len(y_part) for _, y_part in partitions) == len(y)
    assert all(len(y_part) > 0 for _, y_part in partitions)


def test_partition_dirichlet_preserves_total_size(synthetic_data):
    """Dirichlet partitioning should allocate every sample once."""
    X, y = synthetic_data
    partitions = partition_dirichlet(X, y, n_clients=5, alpha=0.3, random_state=42)

    assert sum(len(y_part) for _, y_part in partitions) == len(y)
    assert all(len(y_part) > 0 for _, y_part in partitions)


def test_partition_label_skew_creates_heterogeneous_clients(synthetic_data):
    """Label-skew partitioning should produce client imbalance."""
    X, y = synthetic_data
    partitions = partition_label_skew(X, y, n_clients=4, classes_per_client=2, random_state=42)

    unique_class_counts = [len(np.unique(y_part)) for _, y_part in partitions]
    assert any(count < len(np.unique(y)) for count in unique_class_counts)


def test_create_partitions_unknown_strategy_raises(synthetic_data):
    """Unknown partitioning strategies should fail fast."""
    X, y = synthetic_data

    with pytest.raises(ValueError):
        create_partitions(X, y, strategy='unknown', n_clients=3)


def test_summarize_partitions_returns_client_metadata(synthetic_data):
    """Partition summaries should expose per-client sample counts."""
    X, y = synthetic_data
    partitions = create_partitions(X, y, strategy='uniform', n_clients=3, random_state=42)
    summary = summarize_partitions(partitions, classes=np.unique(y))

    assert len(summary) == 3
    assert summary[0]['client_id'] == 0
    assert sum(item['n_samples'] for item in summary) == len(y)
