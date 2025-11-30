"""EXP 4 — Federated RF with Differential Privacy.

Each client trains a DP-Random Forest using per-client differential privacy.

Tested ε values: 0.1, 0.5, 1, 5
"""

import numpy as np
from sklearn.model_selection import train_test_split

from models.dp_rf import DPRandomForest, DPClientRF
from models.random_forest import RandomForest
from models.tree_utils import compute_accuracy, compute_weighted_accuracy
from federation.aggregator import aggregate_trees


def run_exp4_dp_federation(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    rf_params,
    partitions=None,
    epsilon_values=None,
    n_clients=5,
    aggregation_strategy='rf_s_dts_a',
    n_trees_per_client=10,
    n_total_trees=100,
    validation_split=0.2,
    random_state=42,
    verbose=True,
):
    """Run DP federated RF experiment.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features for aggregation.
        y_val: Validation labels.
        X_test: Test features.
        y_test: Test labels.
        rf_params: RF parameters from EXP 1.
        partitions: Pre-computed client partitions (from EXP 2).
        epsilon_values: List of epsilon values to test.
        n_clients: Number of clients (if partitions not provided).
        aggregation_strategy: Strategy from EXP 3.
        n_trees_per_client: Trees per client for aggregation.
        n_total_trees: Total trees for global strategies.
        validation_split: Fraction for validation if partitions not given.
        random_state: Random seed.
        verbose: Whether to print progress.

    Returns:
        dict: Results for all epsilon values.
    """
    if epsilon_values is None:
        epsilon_values = [0.1, 0.5, 1.0, 5.0]

    if partitions is None:
        from experiments.exp2_clients import partition_uniform_random
        partitions = partition_uniform_random(X_train, y_train, n_clients, random_state)

    if verbose:
        print(f"\nEXP 4: Federated RF with Differential Privacy")
        print(f"Epsilon values: {epsilon_values}")
        print(f"Number of clients: {len(partitions)}")
        print(f"Aggregation strategy: {aggregation_strategy}")

    results = {}

    for epsilon in epsilon_values:
        if verbose:
            print(f"\n--- Testing epsilon = {epsilon} ---")

        dp_client_rfs = []
        client_results = []

        for i, (X_client, y_client) in enumerate(partitions):
            if verbose:
                print(f"\nTraining DP client {i + 1}/{len(partitions)} (ε={epsilon})")

            try:
                X_tr, X_val_client, y_tr, y_val_client = train_test_split(
                    X_client, y_client,
                    test_size=validation_split,
                    random_state=random_state,
                    stratify=y_client if len(np.unique(y_client)) > 1 else None,
                )
            except ValueError as e:
                # Fallback if stratified split fails (e.g., too few samples per class)
                X_tr, y_tr = X_client, y_client
                X_val_client, y_val_client = X_val, y_val

            dp_params = dict(rf_params)
            dp_client = DPClientRF(client_id=i, epsilon=epsilon, rf_params=dp_params)
            dp_client.train(X_tr, y_tr, X_val_client, y_val_client)
            dp_client_rfs.append(dp_client)

            global_metrics = dp_client.evaluate(X_test, y_test)
            client_results.append({
                'client_id': i,
                'epsilon': epsilon,
                'n_samples': len(X_client),
                'global_test_metrics': global_metrics,
            })

            if verbose:
                print(f"  Global accuracy: {global_metrics['accuracy']:.4f}")

        client_trees_list = [client.get_trees() for client in dp_client_rfs]
        classes = dp_client_rfs[0].rf.classes_

        selected_trees = aggregate_trees(
            client_trees_list,
            X_val,
            y_val,
            strategy=aggregation_strategy,
            n_trees_per_client=n_trees_per_client,
            n_total_trees=n_total_trees,
            classes=classes,
        )

        global_rf = RandomForest(voting=rf_params.get('voting', 'simple'))
        global_rf.set_trees(selected_trees, classes)

        y_pred = global_rf.predict(X_test)
        global_accuracy = compute_accuracy(y_test, y_pred)
        global_weighted_accuracy = compute_weighted_accuracy(y_test, y_pred, classes)

        avg_client_accuracy = np.mean([r['global_test_metrics']['accuracy'] for r in client_results])
        best_client_accuracy = max(r['global_test_metrics']['accuracy'] for r in client_results)

        results[epsilon] = {
            'dp_client_rfs': dp_client_rfs,
            'client_results': client_results,
            'global_rf': global_rf,
            'global_accuracy': global_accuracy,
            'global_weighted_accuracy': global_weighted_accuracy,
            'avg_client_accuracy': avg_client_accuracy,
            'best_client_accuracy': best_client_accuracy,
            'n_trees': len(selected_trees),
        }

        if verbose:
            print(f"\n  Federated DP Global RF:")
            print(f"    Trees: {len(selected_trees)}")
            print(f"    Global accuracy: {global_accuracy:.4f}")
            print(f"    Avg client accuracy: {avg_client_accuracy:.4f}")

    return results


def compare_dp_vs_non_dp(
    dp_results,
    non_dp_global_results,
    verbose=True,
):
    """Compare DP and non-DP federated RF performance.

    Args:
        dp_results: Results from run_exp4_dp_federation.
        non_dp_global_results: Results from EXP 3.
        verbose: Whether to print comparison.

    Returns:
        dict: Comparison across epsilon values.
    """
    non_dp_accuracy = non_dp_global_results['best_accuracy']

    comparison = {
        'non_dp_accuracy': non_dp_accuracy,
        'dp_accuracies': {},
        'degradation': {},
    }

    for epsilon, result in dp_results.items():
        dp_acc = result['global_accuracy']
        comparison['dp_accuracies'][epsilon] = dp_acc
        comparison['degradation'][epsilon] = non_dp_accuracy - dp_acc

    if verbose:
        print("\n=== DP vs Non-DP Comparison ===")
        print(f"Non-DP Global RF accuracy: {non_dp_accuracy:.4f}")
        print("\nDP Global RF accuracies:")
        for epsilon in sorted(comparison['dp_accuracies'].keys()):
            dp_acc = comparison['dp_accuracies'][epsilon]
            deg = comparison['degradation'][epsilon]
            print(f"  ε={epsilon}: {dp_acc:.4f} (degradation: {deg:+.4f})")

    return comparison


def get_dp_degradation_curve(dp_results):
    """Extract DP degradation curve data.

    Args:
        dp_results: Results from run_exp4_dp_federation.

    Returns:
        tuple: (epsilon_values, accuracies) for plotting.
    """
    epsilon_values = sorted(dp_results.keys())
    accuracies = [dp_results[eps]['global_accuracy'] for eps in epsilon_values]
    return np.array(epsilon_values), np.array(accuracies)
