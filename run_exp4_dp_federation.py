#!/usr/bin/env python
"""Run EXP 4: Federated RF with Differential Privacy.

Train DP-RF on each client and evaluate privacy-utility tradeoff.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from distributed_random_forest import ClientRF
from distributed_random_forest.experiments.exp1_hparams import get_default_best_params
from distributed_random_forest.experiments.exp2_clients import partition_uniform_random
from distributed_random_forest.experiments.exp3_global_rf import run_exp3_federated_aggregation
from distributed_random_forest.experiments.exp4_dp_rf import (
    run_exp4_dp_federation,
    compare_dp_vs_non_dp,
    get_dp_degradation_curve,
)


def main():
    """Run EXP 4 with synthetic data."""
    print("=" * 60)
    print("EXP 4: Federated RF with Differential Privacy")
    print("=" * 60)

    print("\nGenerating synthetic dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=5,
        n_clusters_per_class=1,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")

    rf_params = get_default_best_params()
    rf_params['n_estimators'] = 21
    rf_params['random_state'] = 42

    n_clients = 5
    partitions = partition_uniform_random(X_train, y_train, n_clients, random_state=42)

    print("\nTraining non-DP federated RF for comparison...")
    non_dp_clients = []
    for i, (X_p, y_p) in enumerate(partitions):
        client = ClientRF(client_id=i, rf_params=rf_params)
        X_tr, X_v, y_tr, y_v = train_test_split(
            X_p, y_p, test_size=0.2, random_state=42,
            stratify=y_p if len(np.unique(y_p)) > 1 else None
        )
        client.train(X_tr, y_tr, X_v, y_v)
        non_dp_clients.append(client)

    non_dp_results = run_exp3_federated_aggregation(
        client_rfs=non_dp_clients,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        n_trees_per_client=10,
        n_total_trees=50,
        verbose=False,
    )
    print(f"Non-DP federated accuracy: {non_dp_results['best_accuracy']:.4f}")

    print("\nRunning DP federated experiments...")
    dp_results = run_exp4_dp_federation(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        rf_params=rf_params,
        partitions=partitions,
        epsilon_values=[0.1, 0.5, 1.0, 5.0],
        aggregation_strategy=non_dp_results['best_strategy'],
        n_trees_per_client=10,
        n_total_trees=50,
        verbose=True,
    )

    print("\n" + "-" * 40)
    comparison = compare_dp_vs_non_dp(dp_results, non_dp_results)

    epsilons, accuracies = get_dp_degradation_curve(dp_results)
    print("\n" + "=" * 60)
    print("DP Degradation Curve")
    print("=" * 60)
    print("Epsilon\t\tAccuracy")
    for eps, acc in zip(epsilons, accuracies):
        print(f"{eps:.1f}\t\t{acc:.4f}")

    return {
        'non_dp_results': non_dp_results,
        'dp_results': dp_results,
        'comparison': comparison,
    }


if __name__ == '__main__':
    main()
