"""Unit tests for Differentially Private Random Forest."""

import numpy as np
import pytest
from sklearn.datasets import make_classification

import sys
sys.path.insert(0, '.')

from models.dp_rf import DPRandomForest, DPClientRF


class TestDPRandomForest:
    """Tests for DPRandomForest class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample classification data."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_classes=3,
            n_informative=5,
            random_state=42,
        )
        X_train, y_train = X[:120], y[:120]
        X_val, y_val = X[120:160], y[120:160]
        X_test, y_test = X[160:], y[160:]
        return X_train, y_train, X_val, y_val, X_test, y_test

    def test_initialization_default_params(self):
        """Test DP-RF initializes with default parameters."""
        dp_rf = DPRandomForest()
        assert dp_rf.n_estimators == 100
        assert dp_rf.epsilon == 1.0
        assert dp_rf.dp_mechanism == 'laplace'

    def test_initialization_custom_epsilon(self):
        """Test DP-RF initializes with custom epsilon."""
        dp_rf = DPRandomForest(epsilon=0.5)
        assert dp_rf.epsilon == 0.5

    def test_initialization_gaussian_mechanism(self):
        """Test DP-RF initializes with gaussian mechanism."""
        dp_rf = DPRandomForest(dp_mechanism='gaussian')
        assert dp_rf.dp_mechanism == 'gaussian'

    def test_fit_laplace(self, sample_data):
        """Test fitting DP-RF with Laplace mechanism."""
        X_train, y_train, _, _, _, _ = sample_data
        dp_rf = DPRandomForest(n_estimators=10, epsilon=1.0, 
                               dp_mechanism='laplace', random_state=42)
        dp_rf.fit(X_train, y_train)
        
        assert dp_rf._trees is not None
        assert len(dp_rf._trees) == 10

    def test_fit_gaussian(self, sample_data):
        """Test fitting DP-RF with Gaussian mechanism."""
        X_train, y_train, _, _, _, _ = sample_data
        dp_rf = DPRandomForest(n_estimators=10, epsilon=1.0,
                               dp_mechanism='gaussian', random_state=42)
        dp_rf.fit(X_train, y_train)
        
        assert dp_rf._trees is not None
        assert len(dp_rf._trees) == 10

    def test_predict_returns_correct_shape(self, sample_data):
        """Test predict returns correct shape."""
        X_train, y_train, _, _, X_test, _ = sample_data
        dp_rf = DPRandomForest(n_estimators=10, epsilon=1.0, random_state=42)
        dp_rf.fit(X_train, y_train)
        
        predictions = dp_rf.predict(X_test)
        assert predictions.shape == (len(X_test),)

    def test_predict_returns_valid_classes(self, sample_data):
        """Test predict returns valid class labels."""
        X_train, y_train, _, _, X_test, _ = sample_data
        dp_rf = DPRandomForest(n_estimators=10, epsilon=1.0, random_state=42)
        dp_rf.fit(X_train, y_train)
        
        predictions = dp_rf.predict(X_test)
        unique_predictions = np.unique(predictions)
        assert all(p in dp_rf._classes for p in unique_predictions)

    def test_get_privacy_budget(self):
        """Test get_privacy_budget returns epsilon."""
        dp_rf = DPRandomForest(epsilon=0.5)
        assert dp_rf.get_privacy_budget() == 0.5

    def test_different_epsilon_values(self, sample_data):
        """Test DP-RF with different epsilon values."""
        X_train, y_train, _, _, X_test, y_test = sample_data
        
        results = {}
        for epsilon in [0.1, 1.0, 5.0]:
            dp_rf = DPRandomForest(n_estimators=10, epsilon=epsilon, random_state=42)
            dp_rf.fit(X_train, y_train)
            results[epsilon] = dp_rf.score(X_test, y_test)
        
        # Generally, higher epsilon should give better accuracy (less noise)
        # But we just check they all produce valid scores
        for epsilon, score in results.items():
            assert 0 <= score <= 1

    def test_weighted_voting_with_validation(self, sample_data):
        """Test DP-RF with weighted voting."""
        X_train, y_train, X_val, y_val, X_test, _ = sample_data
        dp_rf = DPRandomForest(n_estimators=10, epsilon=1.0, 
                               voting='weighted', random_state=42)
        dp_rf.fit(X_train, y_train, X_val, y_val)
        
        assert dp_rf._tree_weights is not None
        predictions = dp_rf.predict(X_test)
        assert predictions.shape == (len(X_test),)

    def test_reproducibility_with_random_state(self, sample_data):
        """Test DP-RF produces same results with same random_state."""
        X_train, y_train, _, _, X_test, _ = sample_data
        
        dp_rf1 = DPRandomForest(n_estimators=10, epsilon=1.0, random_state=42)
        dp_rf1.fit(X_train, y_train)
        pred1 = dp_rf1.predict(X_test)
        
        dp_rf2 = DPRandomForest(n_estimators=10, epsilon=1.0, random_state=42)
        dp_rf2.fit(X_train, y_train)
        pred2 = dp_rf2.predict(X_test)
        
        assert np.array_equal(pred1, pred2)


class TestDPClientRF:
    """Tests for DPClientRF class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample classification data."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_classes=3,
            n_informative=5,
            random_state=42,
        )
        X_train, y_train = X[:120], y[:120]
        X_val, y_val = X[120:160], y[120:160]
        X_test, y_test = X[160:], y[160:]
        return X_train, y_train, X_val, y_val, X_test, y_test

    def test_initialization(self):
        """Test DPClientRF initialization."""
        client = DPClientRF(client_id=0, epsilon=0.5)
        assert client.client_id == 0
        assert client.epsilon == 0.5
        assert client.rf is None

    def test_initialization_with_params(self):
        """Test DPClientRF initialization with RF params."""
        params = {'n_estimators': 20, 'criterion': 'entropy'}
        client = DPClientRF(client_id=1, epsilon=1.0, rf_params=params)
        assert client.client_id == 1
        assert client.rf_params == params

    def test_train_creates_dp_rf(self, sample_data):
        """Test train creates and fits DP-RF."""
        X_train, y_train, X_val, y_val, _, _ = sample_data
        
        client = DPClientRF(client_id=0, epsilon=1.0, 
                           rf_params={'n_estimators': 10, 'random_state': 42})
        client.train(X_train, y_train, X_val, y_val)
        
        assert client.rf is not None
        assert isinstance(client.rf, DPRandomForest)

    def test_train_metrics_include_epsilon(self, sample_data):
        """Test that training metrics include epsilon."""
        X_train, y_train, X_val, y_val, _, _ = sample_data
        
        client = DPClientRF(client_id=0, epsilon=0.5,
                           rf_params={'n_estimators': 10, 'random_state': 42})
        client.train(X_train, y_train, X_val, y_val)
        
        assert client.train_metrics['epsilon'] == 0.5

    def test_get_trees_returns_list(self, sample_data):
        """Test get_trees returns trees from DP-RF."""
        X_train, y_train, _, _, _, _ = sample_data
        
        client = DPClientRF(client_id=0, epsilon=1.0,
                           rf_params={'n_estimators': 10, 'random_state': 42})
        client.train(X_train, y_train)
        
        trees = client.get_trees()
        assert isinstance(trees, list)
        assert len(trees) == 10

    def test_evaluate_includes_epsilon(self, sample_data):
        """Test evaluate returns epsilon."""
        X_train, y_train, _, _, X_test, y_test = sample_data
        
        client = DPClientRF(client_id=0, epsilon=0.5,
                           rf_params={'n_estimators': 10, 'random_state': 42})
        client.train(X_train, y_train)
        
        metrics = client.evaluate(X_test, y_test)
        assert 'epsilon' in metrics
        assert metrics['epsilon'] == 0.5

    def test_evaluate_before_training_raises(self, sample_data):
        """Test evaluate before training raises error."""
        _, _, _, _, X_test, y_test = sample_data
        
        client = DPClientRF(client_id=0, epsilon=1.0)
        with pytest.raises(RuntimeError):
            client.evaluate(X_test, y_test)

    def test_train_returns_self(self, sample_data):
        """Test train returns self for chaining."""
        X_train, y_train, _, _, _, _ = sample_data
        
        client = DPClientRF(client_id=0, epsilon=1.0,
                           rf_params={'n_estimators': 10, 'random_state': 42})
        result = client.train(X_train, y_train)
        
        assert result is client
