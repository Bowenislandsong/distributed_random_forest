"""EXP 2 — Independent RFs Per Client.

Each client trains RFs independently using the best configuration from EXP 1.

Three data-partitioning strategies:
- EXP 2.1: Feature-based Partitioning
- EXP 2.2: Uniform Random Partitioning
- EXP 2.3: Random Partitioning with EXP 2.1 Sample Counts
"""

import numpy as np
from sklearn.model_selection import train_test_split

from models.random_forest import ClientRF
from models.tree_utils import compute_accuracy, compute_weighted_accuracy


def partition_by_feature(X, y, feature_idx, n_partitions=None):
    """Partition data based on feature values.

    Args:
        X: Features array.
        y: Labels array.
        feature_idx: Index of feature to partition by.
        n_partitions: Optional number of partitions (uses quantiles).

    Returns:
        list: List of (X_partition, y_partition) tuples.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    feature_values = X[:, feature_idx]

    if n_partitions is None:
        unique_values = np.unique(feature_values)
        partitions = []
        for val in unique_values:
            mask = feature_values == val
            if np.sum(mask) > 0:
                partitions.append((X[mask], y[mask]))
    else:
        quantiles = np.percentile(feature_values, np.linspace(0, 100, n_partitions + 1))
        partitions = []
        for i in range(n_partitions):
            if i == n_partitions - 1:
                mask = (feature_values >= quantiles[i]) & (feature_values <= quantiles[i + 1])
            else:
                mask = (feature_values >= quantiles[i]) & (feature_values < quantiles[i + 1])
            if np.sum(mask) > 0:
                partitions.append((X[mask], y[mask]))

    return partitions


def partition_uniform_random(X, y, n_clients, random_state=42):
    """Partition data uniformly at random across clients.

    Args:
        X: Features array.
        y: Labels array.
        n_clients: Number of clients.
        random_state: Random seed.

    Returns:
        list: List of (X_partition, y_partition) tuples.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(X))

    splits = np.array_split(indices, n_clients)

    partitions = []
    for split_indices in splits:
        partitions.append((X[split_indices], y[split_indices]))

    return partitions


def partition_random_with_sizes(X, y, sizes, random_state=42):
    """Partition data randomly with specified sizes.

    Args:
        X: Features array.
        y: Labels array.
        sizes: List of sizes for each partition.
        random_state: Random seed.

    Returns:
        list: List of (X_partition, y_partition) tuples.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(X))

    partitions = []
    start = 0
    for size in sizes:
        end = min(start + size, len(indices))
        split_indices = indices[start:end]
        partitions.append((X[split_indices], y[split_indices]))
        start = end

    return partitions


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
    if partitioning == 'feature':
        partitions = partition_by_feature(X_train, y_train, feature_idx, n_clients)
    elif partitioning == 'uniform':
        partitions = partition_uniform_random(X_train, y_train, n_clients, random_state)
    elif partitioning == 'sized':
        sizes = [len(X_train) // n_clients] * n_clients
        partitions = partition_random_with_sizes(X_train, y_train, sizes, random_state)
    else:
        raise ValueError(f"Unknown partitioning: {partitioning}")

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
        print(f"\nBest client: {best_client_idx} with accuracy {client_results[best_client_idx]['global_test_metrics']['accuracy']:.4f}")
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
