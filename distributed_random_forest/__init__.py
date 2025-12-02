"""Distributed Random Forest with Differential Privacy.

This package provides a federated learning framework for Random Forest
classifiers with optional differential privacy support.

Example usage:
    from distributed_random_forest import (
        RandomForest,
        ClientRF,
        DPRandomForest,
        DPClientRF,
        FederatedAggregator,
    )

    # Train a standard Random Forest
    rf = RandomForest(n_estimators=100, criterion='gini')
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)

    # Train federated clients
    client = ClientRF(client_id=0, rf_params={'n_estimators': 20})
    client.train(X_client, y_client)

    # Aggregate trees from multiple clients
    aggregator = FederatedAggregator(strategy='rf_s_dts_a')
    aggregator.aggregate(clients, X_val, y_val)
    global_rf = aggregator.build_global_rf(classes)
"""

from distributed_random_forest.models.random_forest import RandomForest, ClientRF
from distributed_random_forest.models.dp_rf import DPRandomForest, DPClientRF
from distributed_random_forest.models.tree_utils import (
    compute_accuracy,
    compute_weighted_accuracy,
    compute_f1_score,
    evaluate_tree,
    rank_trees_by_metric,
)
from distributed_random_forest.federation.aggregator import (
    FederatedAggregator,
    aggregate_trees,
    rf_s_dts_a,
    rf_s_dts_wa,
    rf_s_dts_a_all,
    rf_s_dts_wa_all,
)
from distributed_random_forest.federation.voting import (
    simple_voting,
    weighted_voting,
    compute_tree_weights_from_accuracy,
    compute_tree_weights_from_weighted_accuracy,
)

__version__ = "0.1.0"

__all__ = [
    # Core models
    "RandomForest",
    "ClientRF",
    "DPRandomForest",
    "DPClientRF",
    # Federation
    "FederatedAggregator",
    "aggregate_trees",
    "rf_s_dts_a",
    "rf_s_dts_wa",
    "rf_s_dts_a_all",
    "rf_s_dts_wa_all",
    # Voting
    "simple_voting",
    "weighted_voting",
    "compute_tree_weights_from_accuracy",
    "compute_tree_weights_from_weighted_accuracy",
    # Utilities
    "compute_accuracy",
    "compute_weighted_accuracy",
    "compute_f1_score",
    "evaluate_tree",
    "rank_trees_by_metric",
]
