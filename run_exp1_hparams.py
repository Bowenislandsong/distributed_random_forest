#!/usr/bin/env python
"""Run EXP 1: Hyperparameter Selection.

Grid search over RF hyperparameters to find the best configuration.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from distributed_random_forest.experiments.exp1_hparams import (
    run_exp1_hyperparameter_selection,
    quick_hyperparameter_selection,
)


def main():
    """Run EXP 1 with synthetic data."""
    print("=" * 60)
    print("EXP 1: RF Hyperparameter Selection")
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

    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")

    print("\nRunning quick hyperparameter selection...")
    results = quick_hyperparameter_selection(
        X, y,
        n_estimators_candidates=[11, 21, 51],
        random_state=42,
    )

    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"Best parameters: {results['best_params']}")
    print(f"Best validation accuracy: {results['best_score']:.4f}")

    return results


if __name__ == '__main__':
    main()
