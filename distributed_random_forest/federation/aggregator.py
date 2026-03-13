"""Aggregation strategies for federated Random Forest."""

from dataclasses import dataclass, field

import numpy as np

from distributed_random_forest.models.random_forest import RandomForest
from distributed_random_forest.models.tree_utils import evaluate_predictions, evaluate_tree

LEGACY_STRATEGIES = {
    'rf_s_dts_a': ('top_k_per_client', 'accuracy'),
    'rf_s_dts_wa': ('top_k_per_client', 'weighted_accuracy'),
    'rf_s_dts_a_all': ('top_k_global', 'accuracy'),
    'rf_s_dts_wa_all': ('top_k_global', 'weighted_accuracy'),
}

EXTENDED_STRATEGIES = {
    'top_k_per_client_accuracy': ('top_k_per_client', 'accuracy'),
    'top_k_per_client_weighted_accuracy': ('top_k_per_client', 'weighted_accuracy'),
    'top_k_global_accuracy': ('top_k_global', 'accuracy'),
    'top_k_global_weighted_accuracy': ('top_k_global', 'weighted_accuracy'),
    'top_k_global_balanced_accuracy': ('top_k_global', 'balanced_accuracy'),
    'top_k_global_f1': ('top_k_global', 'f1_score'),
    'proportional_weighted_accuracy': ('proportional', 'weighted_accuracy'),
    'proportional_balanced_accuracy': ('proportional', 'balanced_accuracy'),
    'threshold_weighted_accuracy': ('threshold', 'weighted_accuracy'),
}

AVAILABLE_STRATEGIES = {**LEGACY_STRATEGIES, **EXTENDED_STRATEGIES}


@dataclass
class ScoredTree:
    """Validation metrics for a single tree."""

    tree: object
    client_id: int
    tree_id: int
    metrics: dict
    client_sample_count: int = 0


@dataclass
class AggregationSummary:
    """Structured report for a tree aggregation run."""

    strategy: str
    selection_mode: str
    selection_metric: str
    n_candidates: int
    n_selected: int
    per_client_tree_counts: dict = field(default_factory=dict)
    selected_tree_scores: list = field(default_factory=list)
    validation_metrics: dict = field(default_factory=dict)

    def to_dict(self):
        """Convert the summary to a JSON-serializable dictionary."""
        return {
            'strategy': self.strategy,
            'selection_mode': self.selection_mode,
            'selection_metric': self.selection_metric,
            'n_candidates': self.n_candidates,
            'n_selected': self.n_selected,
            'per_client_tree_counts': self.per_client_tree_counts,
            'selected_tree_scores': self.selected_tree_scores,
            'validation_metrics': self.validation_metrics,
        }


def _resolve_strategy(strategy):
    """Map a strategy name to its selection mode and metric."""
    if strategy not in AVAILABLE_STRATEGIES:
        valid = ', '.join(sorted(AVAILABLE_STRATEGIES))
        raise ValueError(f"Unknown strategy: {strategy}. Must be one of: {valid}")
    return AVAILABLE_STRATEGIES[strategy]


def _score_client_trees(client_trees_list, X_val, y_val, classes=None, client_sample_counts=None):
    """Evaluate all candidate trees on a common validation set."""
    scored = []
    client_sample_counts = client_sample_counts or [0] * len(client_trees_list)

    for client_id, client_trees in enumerate(client_trees_list):
        sample_count = client_sample_counts[client_id]
        for tree_id, tree in enumerate(client_trees):
            scored.append(
                ScoredTree(
                    tree=tree,
                    client_id=client_id,
                    tree_id=tree_id,
                    metrics=evaluate_tree(tree, X_val, y_val, classes),
                    client_sample_count=sample_count,
                )
            )

    return scored


def _sort_scored_trees(scored_trees, metric):
    """Sort scored trees with deterministic tie-breaking."""
    return sorted(
        scored_trees,
        key=lambda item: (
            item.metrics[metric],
            item.metrics['weighted_accuracy'],
            item.metrics['balanced_accuracy'],
            -item.client_id,
            -item.tree_id,
        ),
        reverse=True,
    )


def _build_selection_report(selected_trees):
    """Summarize selected trees for reporting."""
    per_client_tree_counts = {}
    selected_tree_scores = []

    for item in selected_trees:
        per_client_tree_counts[item.client_id] = per_client_tree_counts.get(item.client_id, 0) + 1
        selected_tree_scores.append(
            {
                'client_id': item.client_id,
                'tree_id': item.tree_id,
                'client_sample_count': item.client_sample_count,
                'metrics': item.metrics,
            }
        )

    return per_client_tree_counts, selected_tree_scores


def _select_top_k_per_client(scored_trees, metric, n_trees_per_client):
    """Pick the top K trees inside each client."""
    grouped = {}
    for item in scored_trees:
        grouped.setdefault(item.client_id, []).append(item)

    selected = []
    for client_id in sorted(grouped):
        ranked = _sort_scored_trees(grouped[client_id], metric)
        selected.extend(ranked[:n_trees_per_client])

    return selected


def _select_top_k_global(scored_trees, metric, n_total_trees):
    """Pick the globally best trees under a chosen metric."""
    ranked = _sort_scored_trees(scored_trees, metric)
    return ranked[:n_total_trees]


def _select_proportional(scored_trees, metric, n_total_trees, client_sample_counts=None):
    """Allocate tree budget proportionally to client data volume."""
    if client_sample_counts is None:
        client_sample_counts = []

    grouped = {}
    for item in scored_trees:
        grouped.setdefault(item.client_id, []).append(item)

    if not client_sample_counts:
        client_sample_counts = [
            len(grouped.get(client_id, []))
            for client_id in range(len(grouped))
        ]

    total_samples = max(sum(max(count, 0) for count in client_sample_counts), 1)
    quotas = {}
    active_clients = [client_id for client_id in grouped if grouped[client_id]]

    if not active_clients:
        return []

    for client_id in active_clients:
        requested = int(np.floor(n_total_trees * client_sample_counts[client_id] / total_samples))
        quotas[client_id] = min(len(grouped[client_id]), requested)

    assigned = sum(quotas.values())
    remaining_budget = min(n_total_trees, len(scored_trees)) - assigned

    if remaining_budget > 0:
        ranked_clients = sorted(
            active_clients,
            key=lambda client_id: client_sample_counts[client_id],
            reverse=True,
        )
        for client_id in ranked_clients:
            if remaining_budget <= 0:
                break
            if quotas[client_id] < len(grouped[client_id]):
                quotas[client_id] += 1
                remaining_budget -= 1

    selected = []
    leftovers = []
    for client_id in active_clients:
        ranked = _sort_scored_trees(grouped[client_id], metric)
        selected.extend(ranked[:quotas[client_id]])
        leftovers.extend(ranked[quotas[client_id]:])

    if len(selected) < min(n_total_trees, len(scored_trees)):
        leftovers = _sort_scored_trees(leftovers, metric)
        needed = min(n_total_trees, len(scored_trees)) - len(selected)
        selected.extend(leftovers[:needed])

    return selected


def _select_threshold(scored_trees, metric, min_score, n_total_trees=None):
    """Pick all trees above a threshold, optionally capped by total budget."""
    if min_score is None:
        min_score = 0.5

    ranked = _sort_scored_trees(scored_trees, metric)
    selected = [item for item in ranked if item.metrics[metric] >= min_score]

    if n_total_trees is not None:
        selected = selected[:n_total_trees]

    if not selected and ranked:
        fallback_budget = 1 if n_total_trees is None else max(1, min(n_total_trees, len(ranked)))
        return ranked[:fallback_budget]

    return selected


def _select_scored_trees(
    scored_trees,
    strategy,
    n_trees_per_client=None,
    n_total_trees=None,
    min_score=None,
    client_sample_counts=None,
):
    """Dispatch tree selection across supported strategies."""
    selection_mode, metric = _resolve_strategy(strategy)

    if selection_mode == 'top_k_per_client':
        if n_trees_per_client is None:
            n_trees_per_client = 10
        selected = _select_top_k_per_client(scored_trees, metric, n_trees_per_client)
    elif selection_mode == 'top_k_global':
        if n_total_trees is None:
            n_total_trees = len(scored_trees)
        selected = _select_top_k_global(scored_trees, metric, n_total_trees)
    elif selection_mode == 'proportional':
        if n_total_trees is None:
            n_total_trees = len(scored_trees)
        selected = _select_proportional(
            scored_trees,
            metric,
            n_total_trees,
            client_sample_counts=client_sample_counts,
        )
    else:
        selected = _select_threshold(
            scored_trees,
            metric,
            min_score=min_score,
            n_total_trees=n_total_trees,
        )

    return selected, selection_mode, metric


def rf_s_dts_a(client_trees_list, X_val, y_val, n_trees_per_client, classes=None):
    """RF_S_DTs_A: sort trees by validation accuracy within each client."""
    return aggregate_trees(
        client_trees_list,
        X_val,
        y_val,
        strategy='rf_s_dts_a',
        n_trees_per_client=n_trees_per_client,
        classes=classes,
    )


def rf_s_dts_wa(client_trees_list, X_val, y_val, n_trees_per_client, classes=None):
    """RF_S_DTs_WA: sort trees by weighted accuracy within each client."""
    return aggregate_trees(
        client_trees_list,
        X_val,
        y_val,
        strategy='rf_s_dts_wa',
        n_trees_per_client=n_trees_per_client,
        classes=classes,
    )


def rf_s_dts_a_all(client_trees_list, X_val, y_val, n_total_trees, classes=None):
    """RF_S_DTs_A_All: sort all trees globally by validation accuracy."""
    return aggregate_trees(
        client_trees_list,
        X_val,
        y_val,
        strategy='rf_s_dts_a_all',
        n_total_trees=n_total_trees,
        classes=classes,
    )


def rf_s_dts_wa_all(client_trees_list, X_val, y_val, n_total_trees, classes=None):
    """RF_S_DTs_WA_All: sort all trees globally by weighted accuracy."""
    return aggregate_trees(
        client_trees_list,
        X_val,
        y_val,
        strategy='rf_s_dts_wa_all',
        n_total_trees=n_total_trees,
        classes=classes,
    )


def aggregate_trees(
    client_trees_list,
    X_val,
    y_val,
    strategy='rf_s_dts_a',
    n_trees_per_client=None,
    n_total_trees=None,
    classes=None,
    min_score=None,
    client_sample_counts=None,
    return_summary=False,
):
    """Aggregate trees from multiple clients using the specified strategy.

    Args:
        client_trees_list: List of lists of trees from each client.
        X_val: Validation features.
        y_val: Validation labels.
        strategy: Aggregation strategy name.
        n_trees_per_client: Trees to select per client.
        n_total_trees: Total trees to select globally.
        classes: Optional class labels.
        min_score: Minimum metric value for threshold strategies.
        client_sample_counts: Optional client sample counts.
        return_summary: Whether to return a structured ``AggregationSummary``.

    Returns:
        list | tuple: Selected trees, optionally paired with a summary.
    """
    scored_trees = _score_client_trees(
        client_trees_list,
        X_val,
        y_val,
        classes=classes,
        client_sample_counts=client_sample_counts,
    )
    selected, selection_mode, metric = _select_scored_trees(
        scored_trees,
        strategy=strategy,
        n_trees_per_client=n_trees_per_client,
        n_total_trees=n_total_trees,
        min_score=min_score,
        client_sample_counts=client_sample_counts,
    )

    selected_trees = [item.tree for item in selected]

    if not return_summary:
        return selected_trees

    per_client_tree_counts, selected_tree_scores = _build_selection_report(selected)
    summary = AggregationSummary(
        strategy=strategy,
        selection_mode=selection_mode,
        selection_metric=metric,
        n_candidates=len(scored_trees),
        n_selected=len(selected_trees),
        per_client_tree_counts=per_client_tree_counts,
        selected_tree_scores=selected_tree_scores,
    )
    return selected_trees, summary


class FederatedAggregator:
    """Aggregator for federated Random Forest."""

    def __init__(
        self,
        strategy='rf_s_dts_a',
        n_trees_per_client=10,
        n_total_trees=100,
        min_score=None,
    ):
        """Initialize the aggregator."""
        self.strategy = strategy
        self.n_trees_per_client = n_trees_per_client
        self.n_total_trees = n_total_trees
        self.min_score = min_score
        self.global_trees = None
        self.global_rf = None
        self.summary = None

    def aggregate(self, client_rfs, X_val, y_val, classes=None):
        """Aggregate trees from client RFs."""
        client_trees_list = [client.get_trees() for client in client_rfs]
        client_sample_counts = [getattr(client, 'n_train_samples', 0) for client in client_rfs]

        self.global_trees, self.summary = aggregate_trees(
            client_trees_list,
            X_val,
            y_val,
            strategy=self.strategy,
            n_trees_per_client=self.n_trees_per_client,
            n_total_trees=self.n_total_trees,
            classes=classes,
            min_score=self.min_score,
            client_sample_counts=client_sample_counts,
            return_summary=True,
        )
        return self.global_trees

    def build_global_rf(self, classes=None, voting='simple'):
        """Build a global RF from the aggregated trees."""
        if self.global_trees is None:
            raise RuntimeError("No aggregated trees available. Call aggregate() first.")

        self.global_rf = RandomForest(voting=voting)
        self.global_rf.set_trees(self.global_trees, classes)
        return self.global_rf

    def evaluate(self, X_test, y_test):
        """Evaluate the global RF on held-out data."""
        if self.global_rf is None:
            raise RuntimeError("Global RF not built")

        metrics = evaluate_predictions(
            y_test,
            self.global_rf.predict(X_test),
            self.global_rf.classes_,
        )
        if self.summary is not None:
            self.summary.validation_metrics = metrics
        return metrics

    def get_summary(self):
        """Return the latest aggregation summary."""
        if self.summary is None:
            return None
        return self.summary.to_dict()
