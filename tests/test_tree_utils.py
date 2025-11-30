"""Unit tests for tree utility functions."""

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification

import sys
sys.path.insert(0, '.')

from models.tree_utils import (
    compute_accuracy,
    compute_weighted_accuracy,
    compute_f1_score,
    evaluate_tree,
    rank_trees_by_metric,
)


class TestComputeAccuracy:
    """Tests for compute_accuracy function."""

    def test_perfect_accuracy(self):
        """Test accuracy when all predictions are correct."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        assert compute_accuracy(y_true, y_pred) == 1.0

    def test_zero_accuracy(self):
        """Test accuracy when all predictions are wrong."""
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        assert compute_accuracy(y_true, y_pred) == 0.0

    def test_partial_accuracy(self):
        """Test accuracy with mixed correct/incorrect predictions."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        assert compute_accuracy(y_true, y_pred) == 0.5

    def test_binary_classification(self):
        """Test accuracy on binary classification."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0, 1])
        assert compute_accuracy(y_true, y_pred) == 5 / 6


class TestComputeWeightedAccuracy:
    """Tests for compute_weighted_accuracy function."""

    def test_perfect_weighted_accuracy(self):
        """Test weighted accuracy when all predictions are correct."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        # Perfect accuracy: 1.0 * 1.0 = 1.0
        assert compute_weighted_accuracy(y_true, y_pred) == 1.0

    def test_zero_weighted_accuracy(self):
        """Test weighted accuracy when all predictions are wrong."""
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        # Overall accuracy is 0, so weighted accuracy is 0
        assert compute_weighted_accuracy(y_true, y_pred) == 0.0

    def test_with_explicit_classes(self):
        """Test weighted accuracy with explicit class labels."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        classes = np.array([0, 1, 2])  # Include extra class
        wa = compute_weighted_accuracy(y_true, y_pred, classes)
        assert wa == 1.0  # All predictions correct

    def test_imbalanced_classes(self):
        """Test weighted accuracy with imbalanced classes."""
        y_true = np.array([0, 0, 0, 0, 1])  # 4 class 0, 1 class 1
        y_pred = np.array([0, 0, 0, 0, 0])  # Predicts all as class 0
        wa = compute_weighted_accuracy(y_true, y_pred)
        # Overall acc = 4/5 = 0.8
        # Class 0 acc = 4/4 = 1.0
        # Class 1 acc = 0/1 = 0.0
        # Mean per-class acc = (1.0 + 0.0) / 2 = 0.5
        # Weighted acc = 0.8 * 0.5 = 0.4
        assert abs(wa - 0.4) < 1e-10


class TestComputeF1Score:
    """Tests for compute_f1_score function."""

    def test_perfect_f1(self):
        """Test F1 when all predictions are correct."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        assert compute_f1_score(y_true, y_pred) == 1.0

    def test_zero_f1(self):
        """Test F1 when all predictions are wrong."""
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        assert compute_f1_score(y_true, y_pred) == 0.0

    def test_different_averaging(self):
        """Test F1 with different averaging methods."""
        y_true = np.array([0, 0, 1, 1, 2])
        y_pred = np.array([0, 1, 1, 1, 0])
        
        macro_f1 = compute_f1_score(y_true, y_pred, average='macro')
        weighted_f1 = compute_f1_score(y_true, y_pred, average='weighted')
        
        assert 0 <= macro_f1 <= 1
        assert 0 <= weighted_f1 <= 1


class TestEvaluateTree:
    """Tests for evaluate_tree function."""

    @pytest.fixture
    def fitted_tree(self):
        """Create a fitted decision tree for testing."""
        X, y = make_classification(n_samples=100, n_features=10, 
                                   n_classes=3, n_informative=5,
                                   random_state=42)
        tree = DecisionTreeClassifier(random_state=42)
        tree.fit(X, y)
        return tree, X, y

    def test_returns_dict_with_expected_keys(self, fitted_tree):
        """Test that evaluate_tree returns expected metrics."""
        tree, X, y = fitted_tree
        result = evaluate_tree(tree, X, y)
        
        assert 'accuracy' in result
        assert 'weighted_accuracy' in result
        assert 'f1_score' in result

    def test_metrics_in_valid_range(self, fitted_tree):
        """Test that all metrics are in valid range [0, 1]."""
        tree, X, y = fitted_tree
        result = evaluate_tree(tree, X, y)
        
        assert 0 <= result['accuracy'] <= 1
        assert 0 <= result['weighted_accuracy'] <= 1
        assert 0 <= result['f1_score'] <= 1


class TestRankTreesByMetric:
    """Tests for rank_trees_by_metric function."""

    @pytest.fixture
    def multiple_trees(self):
        """Create multiple fitted trees for testing."""
        X, y = make_classification(n_samples=200, n_features=10,
                                   n_classes=3, n_informative=5,
                                   random_state=42)
        trees = []
        for i in range(5):
            tree = DecisionTreeClassifier(random_state=i, max_depth=i + 1)
            tree.fit(X, y)
            trees.append(tree)
        return trees, X, y

    def test_returns_sorted_list(self, multiple_trees):
        """Test that trees are sorted by metric in descending order."""
        trees, X, y = multiple_trees
        ranked = rank_trees_by_metric(trees, X, y, metric='accuracy')
        
        scores = [score for _, score in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_returns_all_trees(self, multiple_trees):
        """Test that all trees are returned."""
        trees, X, y = multiple_trees
        ranked = rank_trees_by_metric(trees, X, y, metric='accuracy')
        
        assert len(ranked) == len(trees)

    def test_rank_by_weighted_accuracy(self, multiple_trees):
        """Test ranking by weighted accuracy."""
        trees, X, y = multiple_trees
        ranked = rank_trees_by_metric(trees, X, y, metric='weighted_accuracy')
        
        scores = [score for _, score in ranked]
        assert scores == sorted(scores, reverse=True)
