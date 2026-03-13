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

from distributed_random_forest.distributed import (
    FederatedRandomForest,
    FederatedRunReport,
    create_partitions,
    partition_by_feature,
    partition_dirichlet,
    partition_label_skew,
    partition_random_with_sizes,
    partition_stratified,
    partition_uniform_random,
    summarize_partitions,
)
from distributed_random_forest.federation.aggregator import (
    AVAILABLE_STRATEGIES,
    AggregationSummary,
    FederatedAggregator,
    aggregate_trees,
    rf_s_dts_a,
    rf_s_dts_a_all,
    rf_s_dts_wa,
    rf_s_dts_wa_all,
)
from distributed_random_forest.federation.voting import (
    compute_tree_weights_from_accuracy,
    compute_tree_weights_from_weighted_accuracy,
    simple_voting,
    weighted_voting,
)
from distributed_random_forest.models.dp_rf import DPClientRF, DPRandomForest
from distributed_random_forest.models.random_forest import ClientRF, RandomForest
from distributed_random_forest.models.tree_utils import (
    compute_accuracy,
    compute_balanced_accuracy,
    compute_class_distribution,
    compute_f1_score,
    compute_weighted_accuracy,
    evaluate_predictions,
    evaluate_tree,
    rank_trees_by_metric,
)

__version__ = "0.2.0"

__all__ = [
    # Core models
    "RandomForest",
    "ClientRF",
    "DPRandomForest",
    "DPClientRF",
    # Federation
    "AggregationSummary",
    "AVAILABLE_STRATEGIES",
    "FederatedAggregator",
    "FederatedRandomForest",
    "FederatedRunReport",
    "aggregate_trees",
    "rf_s_dts_a",
    "rf_s_dts_wa",
    "rf_s_dts_a_all",
    "rf_s_dts_wa_all",
    "create_partitions",
    "partition_by_feature",
    "partition_dirichlet",
    "partition_label_skew",
    "partition_random_with_sizes",
    "partition_stratified",
    "partition_uniform_random",
    "summarize_partitions",
    # Voting
    "simple_voting",
    "weighted_voting",
    "compute_tree_weights_from_accuracy",
    "compute_tree_weights_from_weighted_accuracy",
    # Utilities
    "compute_accuracy",
    "compute_balanced_accuracy",
    "compute_class_distribution",
    "compute_weighted_accuracy",
    "compute_f1_score",
    "evaluate_tree",
    "evaluate_predictions",
    "rank_trees_by_metric",
]
