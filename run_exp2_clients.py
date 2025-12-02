#!/usr/bin/env python
"""Run EXP 2: Independent RFs Per Client.

Train RFs on partitioned data with different partitioning strategies.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from distributed_random_forest.experiments.exp1_hparams import get_default_best_params
from distributed_random_forest.experiments.exp2_clients import (
    run_exp2_1_feature_partitioning,
    run_exp2_2_uniform_partitioning,
    run_exp2_3_sized_partitioning,
)


def main():
    """Run EXP 2 with synthetic data."""
    print("=" * 60)
    print("EXP 2: Independent RFs Per Client")
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

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    rf_params = get_default_best_params()
    rf_params['random_state'] = 42
    print(f"\nUsing RF params: {rf_params}")

    print("\n" + "-" * 40)
    print("EXP 2.2: Uniform Random Partitioning")
    print("-" * 40)
    results_uniform = run_exp2_2_uniform_partitioning(
        X_train, y_train, X_test, y_test,
        rf_params=rf_params,
        n_clients=5,
        random_state=42,
    )

    print("\n" + "-" * 40)
    print("EXP 2.1: Feature-based Partitioning")
    print("-" * 40)
    results_feature = run_exp2_1_feature_partitioning(
        X_train, y_train, X_test, y_test,
        rf_params=rf_params,
        feature_idx=0,
        n_clients=5,
        random_state=42,
    )

    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"Uniform partitioning - Avg accuracy: {results_uniform['avg_accuracy']:.4f}")
    print(f"Feature partitioning - Avg accuracy: {results_feature['avg_accuracy']:.4f}")

    return {
        'uniform': results_uniform,
        'feature': results_feature,
    }


if __name__ == '__main__':
    main()
