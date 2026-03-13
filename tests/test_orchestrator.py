"""Tests for high-level federated orchestration."""

import json

import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from distributed_random_forest import FederatedRandomForest


@pytest.fixture
def dataset():
    """Create train and test splits for orchestration tests."""
    X, y = make_classification(
        n_samples=400,
        n_features=16,
        n_classes=3,
        n_informative=10,
        random_state=42,
    )
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def test_federated_random_forest_fit_and_evaluate(dataset):
    """The orchestrator should train, select a strategy, and evaluate."""
    X_train, X_test, y_train, y_test = dataset
    model = FederatedRandomForest(
        n_clients=4,
        rf_params={'n_estimators': 15, 'random_state': 42},
        aggregation_strategy='auto',
        execution_backend='thread',
        max_workers=2,
        random_state=42,
    )
    model.fit(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)

    assert model.selected_strategy is not None
    assert len(model.clients) == 4
    assert 'accuracy' in metrics
    assert 0 <= metrics['accuracy'] <= 1
    assert model.report is not None
    assert len(model.report.strategy_reports) >= 1


def test_federated_random_forest_export_report(dataset, tmp_path):
    """Training reports should export as JSON for auditability."""
    X_train, X_test, y_train, y_test = dataset
    model = FederatedRandomForest(
        n_clients=3,
        rf_params={'n_estimators': 10, 'random_state': 42},
        aggregation_strategy='rf_s_dts_wa_all',
        execution_backend='sequential',
        random_state=42,
    )
    model.fit(X_train, y_train)
    model.evaluate(X_test, y_test)

    report_path = tmp_path / 'report.json'
    model.export_report(report_path)

    payload = json.loads(report_path.read_text())
    assert payload['selected_strategy'] == 'rf_s_dts_wa_all'
    assert 'global_metrics' in payload
    assert len(payload['client_summaries']) == 3


def test_federated_random_forest_supports_dp_training(dataset):
    """The orchestrator should also support DP client training."""
    X_train, X_test, y_train, y_test = dataset
    model = FederatedRandomForest(
        n_clients=3,
        rf_params={'n_estimators': 8, 'random_state': 42},
        aggregation_strategy='top_k_global_balanced_accuracy',
        use_differential_privacy=True,
        epsilon=2.0,
        execution_backend='sequential',
        random_state=42,
    )
    model.fit(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)

    assert metrics['accuracy'] >= 0
    assert all('epsilon' in client.summary()['train_metrics'] for client in model.clients)
