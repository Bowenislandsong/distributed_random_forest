"""Unit tests for voting methods."""

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification

from distributed_random_forest import (
    simple_voting,
    weighted_voting,
    compute_tree_weights_from_accuracy,
    compute_tree_weights_from_weighted_accuracy,
)


class TestSimpleVoting:
    """Tests for simple_voting function."""

    def test_unanimous_predictions(self):
        """Test voting when all trees agree."""
        predictions = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ])  # 3 trees, 3 samples, all predict class 0
        
        result = simple_voting(predictions)
        assert np.array_equal(result, np.array([0, 0, 0]))

    def test_majority_wins(self):
        """Test voting returns majority class."""
        predictions = np.array([
            [0, 1, 2],
            [0, 1, 0],
            [0, 0, 1],
        ])  # 3 trees, 3 samples
        
        result = simple_voting(predictions)
        # Sample 0: all 0s -> 0
        # Sample 1: two 1s, one 0 -> 1
        # Sample 2: one each of 0, 1, 2 -> first with max (tie-breaker)
        assert result[0] == 0
        assert result[1] == 1

    def test_correct_output_shape(self):
        """Test output shape matches number of samples."""
        predictions = np.array([
            [0, 1, 2, 0, 1],
            [0, 1, 0, 1, 1],
        ])  # 2 trees, 5 samples
        
        result = simple_voting(predictions)
        assert result.shape == (5,)

    def test_binary_classification(self):
        """Test voting with binary classification."""
        predictions = np.array([
            [0, 1, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 1],
        ])  # 3 trees, 4 samples
        
        result = simple_voting(predictions)
        assert result[0] == 0  # 0, 0, 1 -> 0
        assert result[3] == 1  # 1, 1, 1 -> 1


class TestWeightedVoting:
    """Tests for weighted_voting function."""

    def test_equal_weights_same_as_simple(self):
        """Test weighted voting with equal weights matches simple voting for unanimous."""
        predictions = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ])
        weights = np.array([1/3, 1/3, 1/3])
        classes = np.array([0, 1, 2])
        
        result = weighted_voting(predictions, weights, classes)
        assert np.array_equal(result, np.array([0, 0, 0]))

    def test_high_weight_tree_dominates(self):
        """Test that high-weight tree dominates prediction."""
        predictions = np.array([
            [0, 0],  # Tree 0 predicts 0
            [1, 1],  # Tree 1 predicts 1
            [1, 1],  # Tree 2 predicts 1
        ])
        weights = np.array([0.8, 0.1, 0.1])  # Tree 0 has much higher weight
        classes = np.array([0, 1])
        
        result = weighted_voting(predictions, weights, classes)
        assert np.array_equal(result, np.array([0, 0]))

    def test_correct_output_shape(self):
        """Test output shape matches number of samples."""
        predictions = np.array([
            [0, 1, 2, 0, 1],
            [0, 1, 0, 1, 1],
        ])
        weights = np.array([0.5, 0.5])
        classes = np.array([0, 1, 2])
        
        result = weighted_voting(predictions, weights, classes)
        assert result.shape == (5,)

    def test_normalized_weights(self):
        """Test voting works with unnormalized weights."""
        predictions = np.array([
            [0, 0],
            [1, 1],
        ])
        weights = np.array([2.0, 1.0])  # Unnormalized
        classes = np.array([0, 1])
        
        result = weighted_voting(predictions, weights, classes)
        assert result[0] == 0  # Tree 0 has higher weight


class TestComputeTreeWeightsFromAccuracy:
    """Tests for compute_tree_weights_from_accuracy function."""

    @pytest.fixture
    def trees_and_data(self):
        """Create trees and validation data."""
        X, y = make_classification(n_samples=100, n_features=10, 
                                   n_classes=3, n_informative=5,
                                   random_state=42)
        trees = []
        for i in range(3):
            tree = DecisionTreeClassifier(random_state=i, max_depth=i + 1)
            tree.fit(X, y)
            trees.append(tree)
        return trees, X, y

    def test_returns_normalized_weights(self, trees_and_data):
        """Test that weights sum to 1."""
        trees, X, y = trees_and_data
        weights = compute_tree_weights_from_accuracy(trees, X, y)
        
        assert np.isclose(weights.sum(), 1.0)

    def test_correct_number_of_weights(self, trees_and_data):
        """Test returns one weight per tree."""
        trees, X, y = trees_and_data
        weights = compute_tree_weights_from_accuracy(trees, X, y)
        
        assert len(weights) == len(trees)

    def test_weights_positive(self, trees_and_data):
        """Test all weights are positive."""
        trees, X, y = trees_and_data
        weights = compute_tree_weights_from_accuracy(trees, X, y)
        
        assert all(w > 0 for w in weights)


class TestComputeTreeWeightsFromWeightedAccuracy:
    """Tests for compute_tree_weights_from_weighted_accuracy function."""

    @pytest.fixture
    def trees_and_data(self):
        """Create trees and validation data."""
        X, y = make_classification(n_samples=100, n_features=10,
                                   n_classes=3, n_informative=5,
                                   random_state=42)
        trees = []
        for i in range(3):
            tree = DecisionTreeClassifier(random_state=i, max_depth=i + 1)
            tree.fit(X, y)
            trees.append(tree)
        return trees, X, y

    def test_returns_normalized_weights(self, trees_and_data):
        """Test that weights sum to 1."""
        trees, X, y = trees_and_data
        weights = compute_tree_weights_from_weighted_accuracy(trees, X, y)
        
        assert np.isclose(weights.sum(), 1.0)

    def test_with_explicit_classes(self, trees_and_data):
        """Test with explicit class labels."""
        trees, X, y = trees_and_data
        classes = np.unique(y)
        weights = compute_tree_weights_from_weighted_accuracy(trees, X, y, classes)
        
        assert np.isclose(weights.sum(), 1.0)
        assert len(weights) == len(trees)
