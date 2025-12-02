"""EXP 3 — Global RF from Federated Aggregation.

Independent client RFs are merged using 4 strategies:
- RF_S_DTs_A
- RF_S_DTs_WA
- RF_S_DTs_A_All
- RF_S_DTs_WA_All
"""

import numpy as np

from distributed_random_forest.models.random_forest import RandomForest
from distributed_random_forest.models.tree_utils import compute_accuracy, compute_weighted_accuracy
from distributed_random_forest.federation.aggregator import (
    aggregate_trees,
    FederatedAggregator,
    rf_s_dts_a,
    rf_s_dts_wa,
    rf_s_dts_a_all,
    rf_s_dts_wa_all,
)


def run_exp3_federated_aggregation(
    client_rfs,
    X_val,
    y_val,
    X_test,
    y_test,
    n_trees_per_client=10,
    n_total_trees=100,
    voting='simple',
    verbose=True,
):
    """Run federated aggregation experiment.

    Args:
        client_rfs: List of trained ClientRF instances.
        X_val: Validation features for tree selection.
        y_val: Validation labels.
        X_test: Test features.
        y_test: Test labels.
        n_trees_per_client: Trees to select per client.
        n_total_trees: Total trees for global strategies.
        voting: Voting method for global RF.
        verbose: Whether to print progress.

    Returns:
        dict: Results for all aggregation strategies.
    """
    strategies = ['rf_s_dts_a', 'rf_s_dts_wa', 'rf_s_dts_a_all', 'rf_s_dts_wa_all']

    client_trees_list = [client.get_trees() for client in client_rfs]
    classes = client_rfs[0].rf.classes_

    if verbose:
        print(f"\nEXP 3: Federated Aggregation")
        print(f"Number of clients: {len(client_rfs)}")
        print(f"Total trees across clients: {sum(len(t) for t in client_trees_list)}")

    results = {}

    for strategy in strategies:
        if verbose:
            print(f"\nTesting strategy: {strategy}")

        if strategy in ['rf_s_dts_a', 'rf_s_dts_wa']:
            n_param = n_trees_per_client
        else:
            n_param = n_total_trees

        selected_trees = aggregate_trees(
            client_trees_list,
            X_val,
            y_val,
            strategy=strategy,
            n_trees_per_client=n_trees_per_client,
            n_total_trees=n_total_trees,
            classes=classes,
        )

        global_rf = RandomForest(voting=voting)
        global_rf.set_trees(selected_trees, classes)

        y_pred = global_rf.predict(X_test)
        accuracy = compute_accuracy(y_test, y_pred)
        weighted_accuracy = compute_weighted_accuracy(y_test, y_pred, classes)

        results[strategy] = {
            'n_trees': len(selected_trees),
            'accuracy': accuracy,
            'weighted_accuracy': weighted_accuracy,
            'global_rf': global_rf,
        }

        if verbose:
            print(f"  Trees selected: {len(selected_trees)}")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Weighted accuracy: {weighted_accuracy:.4f}")

    best_strategy = max(results.keys(), key=lambda s: results[s]['accuracy'])

    if verbose:
        print(f"\nBest strategy: {best_strategy} with accuracy {results[best_strategy]['accuracy']:.4f}")

    return {
        'strategy_results': results,
        'best_strategy': best_strategy,
        'best_accuracy': results[best_strategy]['accuracy'],
        'best_global_rf': results[best_strategy]['global_rf'],
    }


def compare_with_baseline(
    global_rf_results,
    client_results,
    baseline_rf=None,
    X_test=None,
    y_test=None,
    verbose=True,
):
    """Compare global RF with baseline and client performance.

    Args:
        global_rf_results: Results from run_exp3_federated_aggregation.
        client_results: Results from EXP 2.
        baseline_rf: Optional centralized RF for comparison.
        X_test: Test features (needed if baseline_rf provided).
        y_test: Test labels.
        verbose: Whether to print comparison.

    Returns:
        dict: Comparison metrics.
    """
    best_client_acc = max(r['global_test_metrics']['accuracy'] for r in client_results)
    avg_client_acc = np.mean([r['global_test_metrics']['accuracy'] for r in client_results])
    global_acc = global_rf_results['best_accuracy']

    comparison = {
        'global_rf_accuracy': global_acc,
        'best_client_accuracy': best_client_acc,
        'avg_client_accuracy': avg_client_acc,
        'global_vs_best_client': global_acc - best_client_acc,
        'global_vs_avg_client': global_acc - avg_client_acc,
    }

    if baseline_rf is not None and X_test is not None and y_test is not None:
        baseline_acc = baseline_rf.score(X_test, y_test)
        comparison['baseline_accuracy'] = baseline_acc
        comparison['global_vs_baseline'] = global_acc - baseline_acc

    if verbose:
        print("\n=== Performance Comparison ===")
        print(f"Global RF accuracy: {global_acc:.4f}")
        print(f"Best client accuracy: {best_client_acc:.4f}")
        print(f"Average client accuracy: {avg_client_acc:.4f}")
        print(f"Global vs Best client: {comparison['global_vs_best_client']:+.4f}")
        print(f"Global vs Avg client: {comparison['global_vs_avg_client']:+.4f}")
        if 'baseline_accuracy' in comparison:
            print(f"Baseline accuracy: {comparison['baseline_accuracy']:.4f}")
            print(f"Global vs Baseline: {comparison['global_vs_baseline']:+.4f}")

    return comparison
