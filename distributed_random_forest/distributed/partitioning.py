"""Partitioning utilities for distributed and federated training."""

import numpy as np

from distributed_random_forest.models.tree_utils import compute_class_distribution


def partition_by_feature(X, y, feature_idx, n_partitions=None):
    """Partition data based on feature values or feature quantiles."""
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
        return partitions

    quantiles = np.percentile(feature_values, np.linspace(0, 100, n_partitions + 1))
    partitions = []
    for idx in range(n_partitions):
        left = quantiles[idx]
        right = quantiles[idx + 1]
        if idx == n_partitions - 1:
            mask = (feature_values >= left) & (feature_values <= right)
        else:
            mask = (feature_values >= left) & (feature_values < right)
        if np.sum(mask) > 0:
            partitions.append((X[mask], y[mask]))

    return partitions


def partition_uniform_random(X, y, n_clients, random_state=42):
    """Partition data uniformly at random across clients."""
    X = np.asarray(X)
    y = np.asarray(y)

    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(X))
    splits = np.array_split(indices, n_clients)
    return [(X[split], y[split]) for split in splits if len(split) > 0]


def partition_stratified(X, y, n_clients, random_state=42):
    """Partition data so each client roughly preserves class balance."""
    X = np.asarray(X)
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)

    client_indices = [[] for _ in range(n_clients)]
    for cls in np.unique(y):
        class_indices = np.where(y == cls)[0]
        rng.shuffle(class_indices)
        class_splits = np.array_split(class_indices, n_clients)
        for client_id, split in enumerate(class_splits):
            client_indices[client_id].extend(split.tolist())

    partitions = []
    for indices in client_indices:
        shuffled = np.array(indices, dtype=int)
        rng.shuffle(shuffled)
        partitions.append((X[shuffled], y[shuffled]))
    return partitions


def partition_random_with_sizes(X, y, sizes, random_state=42):
    """Partition data randomly with specified per-client sizes."""
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

    if start < len(indices) and partitions:
        remainder = indices[start:]
        last_X, last_y = partitions[-1]
        partitions[-1] = (
            np.concatenate([last_X, X[remainder]], axis=0),
            np.concatenate([last_y, y[remainder]], axis=0),
        )

    return partitions


def partition_dirichlet(X, y, n_clients, alpha=0.5, min_size=1, random_state=42):
    """Create non-IID partitions using a Dirichlet label-allocation process."""
    X = np.asarray(X)
    y = np.asarray(y)

    rng = np.random.default_rng(random_state)
    classes = np.unique(y)

    while True:
        client_indices = [[] for _ in range(n_clients)]
        for cls in classes:
            cls_indices = np.where(y == cls)[0]
            rng.shuffle(cls_indices)
            proportions = rng.dirichlet(np.full(n_clients, alpha))
            split_points = (np.cumsum(proportions)[:-1] * len(cls_indices)).astype(int)
            class_splits = np.split(cls_indices, split_points)
            for client_id, split in enumerate(class_splits):
                client_indices[client_id].extend(split.tolist())

        if all(len(indices) >= min_size for indices in client_indices):
            break

    partitions = []
    for indices in client_indices:
        shuffled = np.array(indices, dtype=int)
        rng.shuffle(shuffled)
        partitions.append((X[shuffled], y[shuffled]))
    return partitions


def partition_label_skew(X, y, n_clients, classes_per_client=2, random_state=42):
    """Create partitions where each client sees only a subset of labels."""
    X = np.asarray(X)
    y = np.asarray(y)

    rng = np.random.default_rng(random_state)
    classes = np.unique(y)
    if classes_per_client <= 0:
        raise ValueError("classes_per_client must be positive")

    client_choices = {}
    for client_id in range(n_clients):
        client_choices[client_id] = rng.choice(
            classes,
            size=min(classes_per_client, len(classes)),
            replace=False,
        )

    eligible_clients_by_class = {}
    for cls in classes:
        eligible_clients = [
            client_id
            for client_id, chosen_classes in client_choices.items()
            if cls in chosen_classes
        ]
        if not eligible_clients:
            eligible_clients = [int(rng.integers(0, n_clients))]
        eligible_clients_by_class[cls] = eligible_clients

    client_indices = [[] for _ in range(n_clients)]
    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        rng.shuffle(cls_indices)
        eligible_clients = eligible_clients_by_class[cls]
        class_splits = np.array_split(cls_indices, len(eligible_clients))
        for client_id, split in zip(eligible_clients, class_splits):
            client_indices[client_id].extend(split.tolist())

    partitions = []
    for indices in client_indices:
        shuffled = np.array(indices, dtype=int)
        rng.shuffle(shuffled)
        partitions.append((X[shuffled], y[shuffled]))
    return partitions


def create_partitions(X, y, strategy='uniform', n_clients=5, random_state=42, **kwargs):
    """Dispatch partition creation across supported strategies."""
    if strategy == 'feature':
        feature_idx = kwargs.pop('feature_idx', 0)
        n_partitions = kwargs.pop('n_partitions', n_clients)
        return partition_by_feature(X, y, feature_idx=feature_idx, n_partitions=n_partitions)
    if strategy == 'uniform':
        return partition_uniform_random(X, y, n_clients=n_clients, random_state=random_state)
    if strategy == 'stratified':
        return partition_stratified(X, y, n_clients=n_clients, random_state=random_state)
    if strategy == 'sized':
        sizes = kwargs.pop('sizes', [len(X) // n_clients] * n_clients)
        return partition_random_with_sizes(X, y, sizes=sizes, random_state=random_state)
    if strategy == 'dirichlet':
        alpha = kwargs.pop('alpha', 0.5)
        min_size = kwargs.pop('min_size', 1)
        return partition_dirichlet(
            X,
            y,
            n_clients=n_clients,
            alpha=alpha,
            min_size=min_size,
            random_state=random_state,
        )
    if strategy == 'label_skew':
        classes_per_client = kwargs.pop('classes_per_client', 2)
        return partition_label_skew(
            X,
            y,
            n_clients=n_clients,
            classes_per_client=classes_per_client,
            random_state=random_state,
        )

    raise ValueError(
        "Unknown partitioning strategy: "
        f"{strategy}. Must be one of: feature, uniform, stratified, sized, dirichlet, label_skew"
    )


def summarize_partitions(partitions, classes=None):
    """Summarize partition sizes and label balance for reporting."""
    summary = []
    for client_id, (_, y_part) in enumerate(partitions):
        summary.append(
            {
                'client_id': client_id,
                'n_samples': int(len(y_part)),
                'class_distribution': compute_class_distribution(y_part, classes),
            }
        )
    return summary
