"""Aggregation strategies for federated Random Forest."""

import numpy as np
from models.tree_utils import evaluate_tree, rank_trees_by_metric


def rf_s_dts_a(client_trees_list, X_val, y_val, n_trees_per_client, classes=None):
    """RF_S_DTs_A: Sort DTs by accuracy within each client RF.

    Selects top performing trees from each client based on accuracy.

    Args:
        client_trees_list: List of lists, where each inner list contains
            trees from one client.
        X_val: Validation features.
        y_val: Validation labels.
        n_trees_per_client: Number of trees to select from each client.
        classes: Optional class labels.

    Returns:
        list: Selected trees for the global RF.
    """
    selected_trees = []

    for client_trees in client_trees_list:
        ranked = rank_trees_by_metric(
            client_trees, X_val, y_val, metric='accuracy', classes=classes
        )
        top_trees = [tree for tree, _ in ranked[:n_trees_per_client]]
        selected_trees.extend(top_trees)

    return selected_trees


def rf_s_dts_wa(client_trees_list, X_val, y_val, n_trees_per_client, classes=None):
    """RF_S_DTs_WA: Sort DTs by weighted accuracy within each client RF.

    Selects top performing trees from each client based on weighted accuracy.

    Args:
        client_trees_list: List of lists of trees from each client.
        X_val: Validation features.
        y_val: Validation labels.
        n_trees_per_client: Number of trees to select from each client.
        classes: Optional class labels.

    Returns:
        list: Selected trees for the global RF.
    """
    selected_trees = []

    for client_trees in client_trees_list:
        ranked = rank_trees_by_metric(
            client_trees, X_val, y_val, metric='weighted_accuracy', classes=classes
        )
        top_trees = [tree for tree, _ in ranked[:n_trees_per_client]]
        selected_trees.extend(top_trees)

    return selected_trees


def rf_s_dts_a_all(client_trees_list, X_val, y_val, n_total_trees, classes=None):
    """RF_S_DTs_A_All: Sort all DTs globally by accuracy.

    Collects all trees from all clients, sorts globally by accuracy,
    and selects the best N trees.

    Args:
        client_trees_list: List of lists of trees from each client.
        X_val: Validation features.
        y_val: Validation labels.
        n_total_trees: Total number of trees to select.
        classes: Optional class labels.

    Returns:
        list: Selected trees for the global RF.
    """
    all_trees = []
    for client_trees in client_trees_list:
        all_trees.extend(client_trees)

    ranked = rank_trees_by_metric(
        all_trees, X_val, y_val, metric='accuracy', classes=classes
    )
    selected_trees = [tree for tree, _ in ranked[:n_total_trees]]

    return selected_trees


def rf_s_dts_wa_all(client_trees_list, X_val, y_val, n_total_trees, classes=None):
    """RF_S_DTs_WA_All: Sort all DTs globally by weighted accuracy.

    Collects all trees from all clients, sorts globally by weighted accuracy,
    and selects the best N trees.

    Args:
        client_trees_list: List of lists of trees from each client.
        X_val: Validation features.
        y_val: Validation labels.
        n_total_trees: Total number of trees to select.
        classes: Optional class labels.

    Returns:
        list: Selected trees for the global RF.
    """
    all_trees = []
    for client_trees in client_trees_list:
        all_trees.extend(client_trees)

    ranked = rank_trees_by_metric(
        all_trees, X_val, y_val, metric='weighted_accuracy', classes=classes
    )
    selected_trees = [tree for tree, _ in ranked[:n_total_trees]]

    return selected_trees


def aggregate_trees(
    client_trees_list,
    X_val,
    y_val,
    strategy='rf_s_dts_a',
    n_trees_per_client=None,
    n_total_trees=None,
    classes=None,
):
    """Aggregate trees from multiple clients using specified strategy.

    Args:
        client_trees_list: List of lists of trees from each client.
        X_val: Validation features.
        y_val: Validation labels.
        strategy: Aggregation strategy name:
            - 'rf_s_dts_a': Sort by accuracy within each client
            - 'rf_s_dts_wa': Sort by weighted accuracy within each client
            - 'rf_s_dts_a_all': Sort all trees globally by accuracy
            - 'rf_s_dts_wa_all': Sort all trees globally by weighted accuracy
        n_trees_per_client: Trees to select per client (for non-global strategies).
        n_total_trees: Total trees to select (for global strategies).
        classes: Optional class labels.

    Returns:
        list: Selected trees for the global RF.
    """
    strategy_map = {
        'rf_s_dts_a': rf_s_dts_a,
        'rf_s_dts_wa': rf_s_dts_wa,
        'rf_s_dts_a_all': rf_s_dts_a_all,
        'rf_s_dts_wa_all': rf_s_dts_wa_all,
    }

    if strategy not in strategy_map:
        valid_strategies = ', '.join(strategy_map.keys())
        raise ValueError(f"Unknown strategy: {strategy}. Must be one of: {valid_strategies}")

    if strategy in ['rf_s_dts_a', 'rf_s_dts_wa']:
        if n_trees_per_client is None:
            n_trees_per_client = 10
        return strategy_map[strategy](
            client_trees_list, X_val, y_val, n_trees_per_client, classes
        )
    else:
        if n_total_trees is None:
            n_total_trees = 100
        return strategy_map[strategy](
            client_trees_list, X_val, y_val, n_total_trees, classes
        )


class FederatedAggregator:
    """Aggregator for federated Random Forest.

    Manages the aggregation of decision trees from multiple clients
    into a global Random Forest.
    """

    def __init__(self, strategy='rf_s_dts_a', n_trees_per_client=10, n_total_trees=100):
        """Initialize aggregator.

        Args:
            strategy: Aggregation strategy name.
            n_trees_per_client: Trees to select per client (for non-global strategies).
            n_total_trees: Total trees to select (for global strategies).
        """
        self.strategy = strategy
        self.n_trees_per_client = n_trees_per_client
        self.n_total_trees = n_total_trees
        self.global_trees = None
        self.global_rf = None

    def aggregate(self, client_rfs, X_val, y_val, classes=None):
        """Aggregate trees from client RFs.

        Args:
            client_rfs: List of trained ClientRF or DPClientRF instances.
            X_val: Validation features.
            y_val: Validation labels.
            classes: Optional class labels.

        Returns:
            list: Aggregated trees.
        """
        client_trees_list = [client.get_trees() for client in client_rfs]

        self.global_trees = aggregate_trees(
            client_trees_list,
            X_val,
            y_val,
            strategy=self.strategy,
            n_trees_per_client=self.n_trees_per_client,
            n_total_trees=self.n_total_trees,
            classes=classes,
        )

        return self.global_trees

    def build_global_rf(self, classes, voting='simple'):
        """Build global RF from aggregated trees.

        Args:
            classes: Class labels.
            voting: Voting method ('simple' or 'weighted').

        Returns:
            RandomForest: Global RF instance.
        """
        from models.random_forest import RandomForest

        self.global_rf = RandomForest(voting=voting)
        self.global_rf.set_trees(self.global_trees, classes)
        return self.global_rf

    def evaluate(self, X_test, y_test):
        """Evaluate global RF.

        Args:
            X_test: Test features.
            y_test: Test labels.

        Returns:
            dict: Evaluation metrics.
        """
        if self.global_rf is None:
            raise RuntimeError("Global RF not built")

        from models.tree_utils import compute_accuracy, compute_weighted_accuracy

        y_pred = self.global_rf.predict(X_test)
        return {
            'accuracy': compute_accuracy(y_test, y_pred),
            'weighted_accuracy': compute_weighted_accuracy(
                y_test, y_pred, self.global_rf.classes_
            ),
        }
