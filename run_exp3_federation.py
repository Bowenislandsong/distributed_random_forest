#!/usr/bin/env python
"""Run EXP 3: Global RF from Federated Aggregation.

Merge client RFs using different aggregation strategies.
"""

import sys
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

sys.path.insert(0, '.')

from experiments.exp1_hparams import get_default_best_params
from experiments.exp2_clients import run_exp2_2_uniform_partitioning
from experiments.exp3_global_rf import (
    run_exp3_federated_aggregation,
    compare_with_baseline,
)
from models.random_forest import RandomForest


def main():
    """Run EXP 3 with synthetic data."""
    print("=" * 60)
    print("EXP 3: Global RF from Federated Aggregation")
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
    rf_params['random_state'] = 42

    print("\nTraining baseline centralized RF...")
    baseline_rf = RandomForest(**rf_params)
    baseline_rf.fit(X_train, y_train, X_val, y_val)
    baseline_acc = baseline_rf.score(X_test, y_test)
    print(f"Baseline accuracy: {baseline_acc:.4f}")

    print("\nTraining independent client RFs...")
    exp2_results = run_exp2_2_uniform_partitioning(
        X_train, y_train, X_test, y_test,
        rf_params=rf_params,
        n_clients=5,
        random_state=42,
        verbose=False,
    )

    print("\nRunning federated aggregation...")
    exp3_results = run_exp3_federated_aggregation(
        client_rfs=exp2_results['client_rfs'],
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        n_trees_per_client=20,
        n_total_trees=100,
        verbose=True,
    )

    print("\n" + "-" * 40)
    comparison = compare_with_baseline(
        exp3_results,
        exp2_results['client_results'],
        baseline_rf=baseline_rf,
        X_test=X_test,
        y_test=y_test,
    )

    print("\n" + "=" * 60)
    print("Best Aggregation Strategy")
    print("=" * 60)
    print(f"Strategy: {exp3_results['best_strategy']}")
    print(f"Accuracy: {exp3_results['best_accuracy']:.4f}")

    return {
        'exp2': exp2_results,
        'exp3': exp3_results,
        'comparison': comparison,
        'baseline_accuracy': baseline_acc,
    }


if __name__ == '__main__':
    main()
