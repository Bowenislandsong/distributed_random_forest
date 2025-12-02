"""Voting methods for Random Forest ensemble predictions."""

import numpy as np


def simple_voting(predictions, classes=None):
    """Perform simple majority voting across predictions.

    Args:
        predictions: Array of shape (n_trees, n_samples) with predictions.
        classes: Optional array of class labels.

    Returns:
        ndarray: Final predictions via majority vote (n_samples,).
    """
    predictions = np.asarray(predictions)
    n_trees, n_samples = predictions.shape

    result = []
    for i in range(n_samples):
        sample_preds = predictions[:, i]
        unique, counts = np.unique(sample_preds, return_counts=True)
        result.append(unique[np.argmax(counts)])

    return np.array(result)


def weighted_voting(predictions, weights, classes):
    """Perform weighted voting across predictions.

    Args:
        predictions: Array of shape (n_trees, n_samples) with predictions.
        weights: Array of shape (n_trees,) with tree weights.
        classes: Array of class labels.

    Returns:
        ndarray: Final predictions via weighted vote (n_samples,).
    """
    predictions = np.asarray(predictions)
    weights = np.asarray(weights)
    classes = np.asarray(classes)

    n_trees, n_samples = predictions.shape
    n_classes = len(classes)

    class_votes = np.zeros((n_samples, n_classes))

    for tree_idx in range(n_trees):
        weight = weights[tree_idx]
        for sample_idx in range(n_samples):
            pred = predictions[tree_idx, sample_idx]
            class_idx = np.where(classes == pred)[0]
            if len(class_idx) > 0:
                class_votes[sample_idx, class_idx[0]] += weight

    return classes[np.argmax(class_votes, axis=1)]


def compute_tree_weights_from_accuracy(trees, X_val, y_val):
    """Compute weights for trees based on validation accuracy.

    Args:
        trees: List of fitted decision tree estimators.
        X_val: Validation features.
        y_val: Validation labels.

    Returns:
        ndarray: Normalized weights for each tree.
    """
    from distributed_random_forest.models.tree_utils import compute_accuracy

    weights = []
    for tree in trees:
        y_pred = tree.predict(X_val)
        acc = compute_accuracy(y_val, y_pred)
        weights.append(max(acc, 1e-6))

    weights = np.array(weights)
    return weights / weights.sum()


def compute_tree_weights_from_weighted_accuracy(trees, X_val, y_val, classes=None):
    """Compute weights for trees based on weighted accuracy.

    Args:
        trees: List of fitted decision tree estimators.
        X_val: Validation features.
        y_val: Validation labels.
        classes: Optional array of class labels.

    Returns:
        ndarray: Normalized weights for each tree.
    """
    from distributed_random_forest.models.tree_utils import compute_weighted_accuracy

    weights = []
    for tree in trees:
        y_pred = tree.predict(X_val)
        wa = compute_weighted_accuracy(y_val, y_pred, classes)
        weights.append(max(wa, 1e-6))

    weights = np.array(weights)
    return weights / weights.sum()
