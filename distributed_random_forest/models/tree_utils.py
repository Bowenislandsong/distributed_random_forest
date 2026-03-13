"""Tree utility functions for distributed and federated forests."""

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def _map_tree_predictions(y_pred, tree_classes, target_classes):
    """Map tree predictions to target class labels.

    When a RandomForestClassifier is trained with string labels, the individual
    trees return numeric indices (0, 1, 2, ...) instead of the original class
    labels. This function maps those indices back to the original classes.

    Args:
        y_pred: Raw predictions from a tree.
        tree_classes: Classes known to the tree (typically numeric indices).
        target_classes: Target class labels to map to.

    Returns:
        ndarray: Predictions mapped to target class labels.
    """
    y_pred = np.asarray(y_pred)
    tree_classes = np.asarray(tree_classes)
    target_classes = np.asarray(target_classes)

    # No mapping needed if classes have same representation
    if np.array_equal(tree_classes, target_classes):
        return y_pred

    # Check if tree classes are numeric indices that need mapping
    # This happens when RF is trained with non-numeric labels
    # Tree classes will be floats like [0., 1., 2.] while target_classes are strings
    if (tree_classes.dtype != target_classes.dtype and
            len(tree_classes) == len(target_classes)):
        # Verify tree_classes are sequential indices starting from 0
        expected_indices = np.arange(len(tree_classes), dtype=float)
        if np.allclose(tree_classes, expected_indices):
            # Map predictions using index lookup with bounds checking
            indices = y_pred.astype(int)
            if np.all((indices >= 0) & (indices < len(target_classes))):
                return target_classes[indices]

    return y_pred


def _map_tree_probabilities(tree_proba, tree_classes, target_classes):
    """Align per-tree probabilities to a global class ordering.

    This is important when aggregating trees trained on heterogeneous clients,
    because some clients may not observe every class during local training.

    Args:
        tree_proba: Array of shape (n_samples, n_tree_classes).
        tree_classes: Class labels known to the tree.
        target_classes: Desired global class ordering.

    Returns:
        ndarray: Probabilities aligned to ``target_classes``.
    """
    tree_proba = np.asarray(tree_proba)
    tree_classes = np.asarray(tree_classes)
    target_classes = np.asarray(target_classes)

    if np.array_equal(tree_classes, target_classes):
        return tree_proba

    aligned = np.zeros((tree_proba.shape[0], len(target_classes)), dtype=float)
    if (
        tree_classes.dtype != target_classes.dtype
        and len(tree_classes) == len(target_classes)
        and np.allclose(tree_classes, np.arange(len(tree_classes), dtype=float))
    ):
        aligned[:, :len(tree_classes)] = tree_proba[:, :len(tree_classes)]
    else:
        for source_idx, cls in enumerate(tree_classes):
            target_idx = np.where(target_classes == cls)[0]
            if len(target_idx) > 0:
                aligned[:, target_idx[0]] = tree_proba[:, source_idx]

    row_sums = aligned.sum(axis=1, keepdims=True)
    nonzero_rows = row_sums.squeeze(axis=1) > 0
    aligned[nonzero_rows] /= row_sums[nonzero_rows]
    return aligned


def compute_accuracy(y_true, y_pred):
    """Compute overall accuracy.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        float: Accuracy score between 0 and 1.
    """
    return float(accuracy_score(y_true, y_pred))


def compute_balanced_accuracy(y_true, y_pred):
    """Compute balanced accuracy.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        float: Balanced accuracy score between 0 and 1.
    """
    return float(balanced_accuracy_score(y_true, y_pred))


def compute_weighted_accuracy(y_true, y_pred, classes=None):
    """Compute weighted accuracy (A × mean per-class accuracy).

    Weighted accuracy prioritizes trees that perform consistently
    across multiple classes.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        classes: Optional list of class labels. If None, inferred from y_true.

    Returns:
        float: Weighted accuracy score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if classes is None:
        classes = np.unique(y_true)

    overall_accuracy = accuracy_score(y_true, y_pred)

    per_class_accuracies = []
    for cls in classes:
        mask = y_true == cls
        if np.sum(mask) > 0:
            class_acc = accuracy_score(y_true[mask], y_pred[mask])
            per_class_accuracies.append(class_acc)

    if len(per_class_accuracies) == 0:
        return 0.0

    mean_per_class_acc = np.mean(per_class_accuracies)
    return float(overall_accuracy * mean_per_class_acc)


def compute_f1_score(y_true, y_pred, average='macro'):
    """Compute F1 score.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        average: Averaging strategy ('macro', 'micro', 'weighted').

    Returns:
        float: F1 score.
    """
    return float(f1_score(y_true, y_pred, average=average, zero_division=0))


def compute_class_distribution(y, classes=None):
    """Summarize class counts for a label vector.

    Args:
        y: Label vector.
        classes: Optional explicit class order.

    Returns:
        dict: Mapping of class label to count.
    """
    y = np.asarray(y)

    if classes is None:
        classes = np.unique(y)

    distribution = {}
    for cls in classes:
        distribution[str(cls)] = int(np.sum(y == cls))
    return distribution


def evaluate_predictions(y_true, y_pred, classes=None):
    """Compute a standard metric bundle for predictions.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        classes: Optional explicit class ordering.

    Returns:
        dict: Accuracy, weighted accuracy, balanced accuracy, and F1 score.
    """
    return {
        'accuracy': float(compute_accuracy(y_true, y_pred)),
        'weighted_accuracy': float(compute_weighted_accuracy(y_true, y_pred, classes)),
        'balanced_accuracy': float(compute_balanced_accuracy(y_true, y_pred)),
        'f1_score': float(compute_f1_score(y_true, y_pred)),
    }


def evaluate_tree(tree, X_val, y_val, classes=None):
    """Evaluate a single decision tree on validation data.

    Args:
        tree: A fitted decision tree estimator.
        X_val: Validation features.
        y_val: Validation labels.
        classes: Optional list of class labels. If provided, tree predictions
            will be mapped to these classes (useful when tree uses numeric
            indices but labels are strings).

    Returns:
        dict: Dictionary with 'accuracy' and 'weighted_accuracy' keys.
    """
    y_pred = tree.predict(X_val)

    # Map tree predictions to target classes if needed
    if classes is not None and hasattr(tree, 'classes_'):
        y_pred = _map_tree_predictions(y_pred, tree.classes_, classes)

    return evaluate_predictions(y_val, y_pred, classes)


def rank_trees_by_metric(trees, X_val, y_val, metric='accuracy', classes=None):
    """Rank trees by a specified metric.

    Args:
        trees: List of fitted decision tree estimators.
        X_val: Validation features.
        y_val: Validation labels.
        metric: Metric to rank by ('accuracy' or 'weighted_accuracy').
        classes: Optional list of class labels.

    Returns:
        list: List of (tree, score) tuples sorted by score descending.
    """
    valid_metrics = {
        'accuracy',
        'weighted_accuracy',
        'balanced_accuracy',
        'f1_score',
    }
    if metric not in valid_metrics:
        valid_metrics_str = ', '.join(sorted(valid_metrics))
        raise ValueError(f"Unknown ranking metric: {metric}. Must be one of: {valid_metrics_str}")

    scored_trees = []
    for tree in trees:
        metrics = evaluate_tree(tree, X_val, y_val, classes)
        score = metrics[metric]
        scored_trees.append((tree, score))

    scored_trees.sort(key=lambda x: x[1], reverse=True)
    return scored_trees
