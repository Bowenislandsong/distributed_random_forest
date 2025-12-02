"""Tree utility functions for computing accuracy metrics."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_accuracy(y_true, y_pred):
    """Compute overall accuracy.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        float: Accuracy score between 0 and 1.
    """
    return accuracy_score(y_true, y_pred)


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
    return overall_accuracy * mean_per_class_acc


def compute_f1_score(y_true, y_pred, average='macro'):
    """Compute F1 score.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        average: Averaging strategy ('macro', 'micro', 'weighted').

    Returns:
        float: F1 score.
    """
    return f1_score(y_true, y_pred, average=average, zero_division=0)


def evaluate_tree(tree, X_val, y_val, classes=None):
    """Evaluate a single decision tree on validation data.

    Args:
        tree: A fitted decision tree estimator.
        X_val: Validation features.
        y_val: Validation labels.
        classes: Optional list of class labels.

    Returns:
        dict: Dictionary with 'accuracy' and 'weighted_accuracy' keys.
    """
    y_pred = tree.predict(X_val)
    return {
        'accuracy': compute_accuracy(y_val, y_pred),
        'weighted_accuracy': compute_weighted_accuracy(y_val, y_pred, classes),
        'f1_score': compute_f1_score(y_val, y_pred),
    }


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
    scored_trees = []
    for tree in trees:
        metrics = evaluate_tree(tree, X_val, y_val, classes)
        score = metrics[metric]
        scored_trees.append((tree, score))

    scored_trees.sort(key=lambda x: x[1], reverse=True)
    return scored_trees
