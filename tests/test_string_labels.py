"""Regression tests for string labels across local and federated forests."""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from distributed_random_forest import ClientRF, FederatedAggregator, RandomForest


def _string_label_dataset():
    X, y = make_classification(
        n_samples=240,
        n_features=12,
        n_classes=3,
        n_informative=8,
        random_state=42,
    )
    labels = np.array(['benign', 'dos', 'probe'])[y]
    return train_test_split(X, labels, test_size=0.2, random_state=42, stratify=labels)


def test_random_forest_handles_string_labels_with_weighted_voting():
    """Weighted forests should keep string labels intact."""
    X_train, X_test, y_train, y_test = _string_label_dataset()
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.25,
        random_state=42,
        stratify=y_train,
    )

    model = RandomForest(n_estimators=12, voting='weighted', random_state=42)
    model.fit(X_fit, y_fit, X_val, y_val)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    assert set(np.unique(predictions)).issubset(set(model.classes_))
    assert probabilities.shape == (len(X_test), len(model.classes_))
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert model.score(X_test, y_test) >= 0


def test_federated_aggregation_preserves_string_label_probabilities():
    """Aggregated forests should align probabilities to global string classes."""
    X_train, X_test, y_train, y_test = _string_label_dataset()
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train,
    )

    client_partitions = np.array_split(np.arange(len(X_train)), 3)
    clients = []
    for client_id, indices in enumerate(client_partitions):
        client = ClientRF(
            client_id=client_id,
            rf_params={'n_estimators': 8, 'random_state': 42 + client_id},
        )
        client.train(X_train[indices], y_train[indices])
        clients.append(client)

    aggregator = FederatedAggregator(
        strategy='top_k_global_balanced_accuracy',
        n_total_trees=12,
    )
    aggregator.aggregate(clients, X_val, y_val, classes=np.unique(y_train))
    global_rf = aggregator.build_global_rf(classes=np.unique(y_train), voting='weighted')

    probabilities = global_rf.predict_proba(X_test)
    assert probabilities.shape == (len(X_test), len(global_rf.classes_))
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert global_rf.score(X_test, y_test) >= 0
