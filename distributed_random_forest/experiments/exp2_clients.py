"""EXP 2 — Independent RFs Per Client.

Each client trains RFs independently using the best configuration from EXP 1.

Three data-partitioning strategies:
- EXP 2.1: Feature-based Partitioning
- EXP 2.2: Uniform Random Partitioning
- EXP 2.3: Random Partitioning with EXP 2.1 Sample Counts
"""

import numpy as np
from sklearn.model_selection import train_test_split

from distributed_random_forest.distributed.partitioning import (
    create_partitions,
    partition_by_feature,
    partition_random_with_sizes,
    partition_uniform_random,
)
from distributed_random_forest.models.random_forest import ClientRF

__all__ = [
    'partition_by_feature',
    'partition_uniform_random',
    'partition_random_with_sizes',
    'run_exp2_independent_clients',
    'run_exp2_1_feature_partitioning',
    'run_exp2_2_uniform_partitioning',
    'run_exp2_3_sized_partitioning',
]


def run_exp2_independent_clients(
    X_train,
    y_train,
    X_test,
    y_test,
    rf_params,
    partitioning='uniform',
    n_clients=5,
    feature_idx=0,
    validation_split=0.2,
    random_state=42,
    verbose=True,
):
    """Run independent client RF experiment.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        rf_params: RF parameters from EXP 1.
        partitioning: Partitioning strategy ('feature', 'uniform', 'sized').
        n_clients: Number of clients (for uniform partitioning).
        feature_idx: Feature index (for feature-based partitioning).
        validation_split: Fraction of client data for validation.
        random_state: Random seed.
        verbose: Whether to print progress.

    Returns:
        dict: Results including trained client RFs.
    """
    if partitioning == 'sized':
        sizes = [len(X_train) // n_clients] * n_clients
        partitions = partition_random_with_sizes(X_train, y_train, sizes, random_state)
    else:
        partitions = create_partitions(
            X_train,
            y_train,
            strategy=partitioning,
            n_clients=n_clients,
            random_state=random_state,
            feature_idx=feature_idx,
        )

    if verbose:
        print(f"\nEXP 2: Training {len(partitions)} clients with {partitioning} partitioning")
        print(f"Partition sizes: {[len(p[0]) for p in partitions]}")

    client_rfs = []
    client_results = []

    for i, (X_client, y_client) in enumerate(partitions):
        if verbose:
            print(f"\nTraining client {i + 1}/{len(partitions)}")

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_client, y_client,
            test_size=validation_split,
            random_state=random_state,
            stratify=y_client if len(np.unique(y_client)) > 1 else None,
        )

        client = ClientRF(client_id=i, rf_params=rf_params)
        client.train(X_tr, y_tr, X_val, y_val)
        client_rfs.append(client)

        local_metrics = client.evaluate(X_client, y_client)
        global_metrics = client.evaluate(X_test, y_test)

        result = {
            'client_id': i,
            'n_samples': len(X_client),
            'train_metrics': client.train_metrics,
            'val_metrics': client.val_metrics,
            'local_test_metrics': local_metrics,
            'global_test_metrics': global_metrics,
        }
        client_results.append(result)

        if verbose:
            print(f"  Local accuracy: {local_metrics['accuracy']:.4f}")
            print(f"  Global accuracy: {global_metrics['accuracy']:.4f}")

    best_client_idx = np.argmax([r['global_test_metrics']['accuracy'] for r in client_results])
    avg_accuracy = np.mean([r['global_test_metrics']['accuracy'] for r in client_results])

    if verbose:
        print(
            f"\nBest client: {best_client_idx} with accuracy "
            f"{client_results[best_client_idx]['global_test_metrics']['accuracy']:.4f}"
        )
        print(f"Average accuracy: {avg_accuracy:.4f}")

    return {
        'client_rfs': client_rfs,
        'client_results': client_results,
        'best_client_idx': best_client_idx,
        'avg_accuracy': avg_accuracy,
        'partitions': partitions,
    }


def run_exp2_1_feature_partitioning(
    X_train, y_train, X_test, y_test, rf_params, feature_idx=0, n_clients=None, **kwargs
):
    """EXP 2.1 — Feature-based Partitioning."""
    return run_exp2_independent_clients(
        X_train, y_train, X_test, y_test, rf_params,
        partitioning='feature', feature_idx=feature_idx, n_clients=n_clients, **kwargs
    )


def run_exp2_2_uniform_partitioning(
    X_train, y_train, X_test, y_test, rf_params, n_clients=5, **kwargs
):
    """EXP 2.2 — Uniform Random Partitioning."""
    return run_exp2_independent_clients(
        X_train, y_train, X_test, y_test, rf_params,
        partitioning='uniform', n_clients=n_clients, **kwargs
    )


def run_exp2_3_sized_partitioning(
    X_train, y_train, X_test, y_test, rf_params, sizes=None, n_clients=5, **kwargs
):
    """EXP 2.3 — Random Partitioning with specified sizes."""
    if sizes is None:
        sizes = [len(X_train) // n_clients] * n_clients

    return run_exp2_independent_clients(
        X_train, y_train, X_test, y_test, rf_params,
        partitioning='sized', n_clients=len(sizes), **kwargs
    )
