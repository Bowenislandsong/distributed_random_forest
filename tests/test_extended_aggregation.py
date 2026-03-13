"""Tests for extended federated aggregation strategies."""

import pytest
from sklearn.datasets import make_classification

from distributed_random_forest import ClientRF, FederatedAggregator, aggregate_trees


@pytest.fixture
def trained_clients():
    """Train several clients for aggregation tests."""
    X, y = make_classification(
        n_samples=360,
        n_features=10,
        n_classes=3,
        n_informative=6,
        random_state=42,
    )
    X_train, y_train = X[:240], y[:240]
    X_val, y_val = X[240:], y[240:]

    clients = []
    for client_id in range(4):
        start = client_id * 60
        end = start + 60
        client = ClientRF(
            client_id=client_id,
            rf_params={'n_estimators': 8, 'random_state': 42 + client_id},
        )
        client.train(X_train[start:end], y_train[start:end])
        clients.append(client)

    return clients, X_val, y_val


def test_proportional_weighted_accuracy_strategy(trained_clients):
    """Proportional strategies should honor the global tree budget."""
    clients, X_val, y_val = trained_clients
    trees = aggregate_trees(
        [client.get_trees() for client in clients],
        X_val,
        y_val,
        strategy='proportional_weighted_accuracy',
        n_total_trees=10,
        client_sample_counts=[client.n_train_samples for client in clients],
    )

    assert len(trees) == 10


def test_threshold_strategy_selects_at_least_one_tree(trained_clients):
    """Threshold strategies should still return a fallback tree when needed."""
    clients, X_val, y_val = trained_clients
    trees = aggregate_trees(
        [client.get_trees() for client in clients],
        X_val,
        y_val,
        strategy='threshold_weighted_accuracy',
        n_total_trees=5,
        min_score=1.5,
    )

    assert len(trees) >= 1


def test_federated_aggregator_exposes_summary(trained_clients):
    """Aggregators should keep a structured summary after selection."""
    clients, X_val, y_val = trained_clients
    aggregator = FederatedAggregator(
        strategy='top_k_global_balanced_accuracy',
        n_total_trees=12,
    )
    aggregator.aggregate(clients, X_val, y_val, classes=clients[0].rf.classes_)
    summary = aggregator.get_summary()

    assert summary is not None
    assert summary['n_selected'] == 12
    assert summary['selection_metric'] == 'balanced_accuracy'
