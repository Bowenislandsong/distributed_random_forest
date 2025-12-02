"""Unit tests for Random Forest implementation."""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from distributed_random_forest import RandomForest, ClientRF


class TestRandomForest:
    """Tests for RandomForest class."""

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
        # Split into train/val/test
        X_train, y_train = X[:120], y[:120]
        X_val, y_val = X[120:160], y[120:160]
        X_test, y_test = X[160:], y[160:]
        return X_train, y_train, X_val, y_val, X_test, y_test

    def test_initialization_default_params(self):
        """Test RF initializes with default parameters."""
        rf = RandomForest()
        assert rf.n_estimators == 100
        assert rf.criterion == 'gini'
        assert rf.voting == 'simple'
        assert rf.max_depth is None

    def test_initialization_custom_params(self):
        """Test RF initializes with custom parameters."""
        rf = RandomForest(
            n_estimators=50,
            criterion='entropy',
            voting='weighted',
            max_depth=10,
            random_state=42,
        )
        assert rf.n_estimators == 50
        assert rf.criterion == 'entropy'
        assert rf.voting == 'weighted'
        assert rf.max_depth == 10
        assert rf.random_state == 42

    def test_fit_simple_voting(self, sample_data):
        """Test fitting RF with simple voting."""
        X_train, y_train, X_val, y_val, _, _ = sample_data
        rf = RandomForest(n_estimators=10, voting='simple', random_state=42)
        rf.fit(X_train, y_train)
        
        assert rf._forest is not None
        assert len(rf._trees) == 10
        assert rf._classes is not None

    def test_fit_weighted_voting(self, sample_data):
        """Test fitting RF with weighted voting."""
        X_train, y_train, X_val, y_val, _, _ = sample_data
        rf = RandomForest(n_estimators=10, voting='weighted', random_state=42)
        rf.fit(X_train, y_train, X_val, y_val)
        
        assert rf._tree_weights is not None
        assert len(rf._tree_weights) == 10
        assert np.allclose(rf._tree_weights.sum(), 1.0)

    def test_predict_returns_correct_shape(self, sample_data):
        """Test predict returns correct shape."""
        X_train, y_train, _, _, X_test, _ = sample_data
        rf = RandomForest(n_estimators=10, random_state=42)
        rf.fit(X_train, y_train)
        
        predictions = rf.predict(X_test)
        assert predictions.shape == (len(X_test),)

    def test_predict_returns_valid_classes(self, sample_data):
        """Test predict returns valid class labels."""
        X_train, y_train, _, _, X_test, _ = sample_data
        rf = RandomForest(n_estimators=10, random_state=42)
        rf.fit(X_train, y_train)
        
        predictions = rf.predict(X_test)
        unique_predictions = np.unique(predictions)
        assert all(p in rf._classes for p in unique_predictions)

    def test_predict_proba_returns_correct_shape(self, sample_data):
        """Test predict_proba returns correct shape."""
        X_train, y_train, _, _, X_test, _ = sample_data
        rf = RandomForest(n_estimators=10, random_state=42)
        rf.fit(X_train, y_train)
        
        proba = rf.predict_proba(X_test)
        assert proba.shape == (len(X_test), len(rf._classes))

    def test_predict_proba_sums_to_one(self, sample_data):
        """Test predict_proba probabilities sum to 1."""
        X_train, y_train, _, _, X_test, _ = sample_data
        rf = RandomForest(n_estimators=10, random_state=42)
        rf.fit(X_train, y_train)
        
        proba = rf.predict_proba(X_test)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_score_returns_valid_range(self, sample_data):
        """Test score returns value in [0, 1]."""
        X_train, y_train, _, _, X_test, y_test = sample_data
        rf = RandomForest(n_estimators=10, random_state=42)
        rf.fit(X_train, y_train)
        
        score = rf.score(X_test, y_test)
        assert 0 <= score <= 1

    def test_get_trees_returns_list(self, sample_data):
        """Test get_trees returns list of trees."""
        X_train, y_train, _, _, _, _ = sample_data
        rf = RandomForest(n_estimators=10, random_state=42)
        rf.fit(X_train, y_train)
        
        trees = rf.get_trees()
        assert isinstance(trees, list)
        assert len(trees) == 10

    def test_set_trees(self, sample_data):
        """Test set_trees properly sets trees."""
        X_train, y_train, _, _, X_test, _ = sample_data
        rf1 = RandomForest(n_estimators=10, random_state=42)
        rf1.fit(X_train, y_train)
        trees = rf1.get_trees()[:5]  # Take first 5 trees
        
        rf2 = RandomForest()
        rf2.set_trees(trees, classes=rf1._classes)
        
        assert len(rf2._trees) == 5
        assert rf2.n_estimators == 5
        
        # Should be able to predict
        predictions = rf2.predict(X_test)
        assert predictions.shape == (len(X_test),)

    def test_gini_criterion(self, sample_data):
        """Test RF with gini criterion."""
        X_train, y_train, _, _, X_test, y_test = sample_data
        rf = RandomForest(n_estimators=10, criterion='gini', random_state=42)
        rf.fit(X_train, y_train)
        
        score = rf.score(X_test, y_test)
        assert score > 0  # Should do better than random

    def test_entropy_criterion(self, sample_data):
        """Test RF with entropy criterion."""
        X_train, y_train, _, _, X_test, y_test = sample_data
        rf = RandomForest(n_estimators=10, criterion='entropy', random_state=42)
        rf.fit(X_train, y_train)
        
        score = rf.score(X_test, y_test)
        assert score > 0  # Should do better than random

    def test_reproducibility_with_random_state(self, sample_data):
        """Test RF produces same results with same random_state."""
        X_train, y_train, _, _, X_test, _ = sample_data
        
        rf1 = RandomForest(n_estimators=10, random_state=42)
        rf1.fit(X_train, y_train)
        pred1 = rf1.predict(X_test)
        
        rf2 = RandomForest(n_estimators=10, random_state=42)
        rf2.fit(X_train, y_train)
        pred2 = rf2.predict(X_test)
        
        assert np.array_equal(pred1, pred2)


class TestClientRF:
    """Tests for ClientRF class."""

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
        """Test ClientRF initialization."""
        client = ClientRF(client_id=0)
        assert client.client_id == 0
        assert client.rf_params == {}
        assert client.rf is None

    def test_initialization_with_params(self):
        """Test ClientRF initialization with RF params."""
        params = {'n_estimators': 20, 'criterion': 'entropy'}
        client = ClientRF(client_id=1, rf_params=params)
        assert client.client_id == 1
        assert client.rf_params == params

    def test_train_creates_rf(self, sample_data):
        """Test train creates and fits RF."""
        X_train, y_train, X_val, y_val, _, _ = sample_data
        
        client = ClientRF(client_id=0, rf_params={'n_estimators': 10, 'random_state': 42})
        client.train(X_train, y_train, X_val, y_val)
        
        assert client.rf is not None
        assert client.train_metrics != {}
        assert client.val_metrics != {}

    def test_train_metrics_populated(self, sample_data):
        """Test that training metrics are populated."""
        X_train, y_train, X_val, y_val, _, _ = sample_data
        
        client = ClientRF(client_id=0, rf_params={'n_estimators': 10, 'random_state': 42})
        client.train(X_train, y_train, X_val, y_val)
        
        assert 'accuracy' in client.train_metrics
        assert 'weighted_accuracy' in client.train_metrics
        assert 'accuracy' in client.val_metrics
        assert 'weighted_accuracy' in client.val_metrics

    def test_get_trees_returns_list(self, sample_data):
        """Test get_trees returns trees from RF."""
        X_train, y_train, _, _, _, _ = sample_data
        
        client = ClientRF(client_id=0, rf_params={'n_estimators': 10, 'random_state': 42})
        client.train(X_train, y_train)
        
        trees = client.get_trees()
        assert isinstance(trees, list)
        assert len(trees) == 10

    def test_get_trees_before_training(self):
        """Test get_trees before training returns empty list."""
        client = ClientRF(client_id=0)
        trees = client.get_trees()
        assert trees == []

    def test_evaluate(self, sample_data):
        """Test evaluate returns metrics."""
        X_train, y_train, _, _, X_test, y_test = sample_data
        
        client = ClientRF(client_id=0, rf_params={'n_estimators': 10, 'random_state': 42})
        client.train(X_train, y_train)
        
        metrics = client.evaluate(X_test, y_test)
        assert 'accuracy' in metrics
        assert 'weighted_accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1

    def test_evaluate_before_training_raises(self, sample_data):
        """Test evaluate before training raises error."""
        _, _, _, _, X_test, y_test = sample_data
        
        client = ClientRF(client_id=0)
        with pytest.raises(RuntimeError):
            client.evaluate(X_test, y_test)

    def test_train_returns_self(self, sample_data):
        """Test train returns self for chaining."""
        X_train, y_train, _, _, _, _ = sample_data
        
        client = ClientRF(client_id=0, rf_params={'n_estimators': 10, 'random_state': 42})
        result = client.train(X_train, y_train)
        
        assert result is client
