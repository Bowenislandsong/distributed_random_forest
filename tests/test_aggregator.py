"""Unit tests for aggregation strategies."""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from distributed_random_forest import (
    RandomForest,
    ClientRF,
    rf_s_dts_a,
    rf_s_dts_wa,
    rf_s_dts_a_all,
    rf_s_dts_wa_all,
    aggregate_trees,
    FederatedAggregator,
)


class TestAggregationStrategies:
    """Tests for individual aggregation strategy functions."""

    @pytest.fixture
    def client_trees_and_data(self):
        """Create client trees and validation data."""
        X, y = make_classification(
            n_samples=300,
            n_features=10,
            n_classes=3,
            n_informative=5,
            random_state=42,
        )
        X_train, y_train = X[:200], y[:200]
        X_val, y_val = X[200:], y[200:]
        
        # Create 3 clients with 10 trees each
        client_trees_list = []
        for i in range(3):
            rf = RandomForest(n_estimators=10, random_state=i)
            # Use subset of data for each client
            start = i * 66
            end = start + 66
            rf.fit(X_train[start:end], y_train[start:end])
            client_trees_list.append(rf.get_trees())
        
        return client_trees_list, X_val, y_val

    def test_rf_s_dts_a_selects_correct_count(self, client_trees_and_data):
        """Test RF_S_DTs_A selects correct number of trees per client."""
        client_trees_list, X_val, y_val = client_trees_and_data
        n_trees_per_client = 5
        
        selected = rf_s_dts_a(client_trees_list, X_val, y_val, n_trees_per_client)
        
        # Should have 5 trees per client * 3 clients = 15 trees
        assert len(selected) == 15

    def test_rf_s_dts_wa_selects_correct_count(self, client_trees_and_data):
        """Test RF_S_DTs_WA selects correct number of trees per client."""
        client_trees_list, X_val, y_val = client_trees_and_data
        n_trees_per_client = 3
        
        selected = rf_s_dts_wa(client_trees_list, X_val, y_val, n_trees_per_client)
        
        assert len(selected) == 9  # 3 trees * 3 clients

    def test_rf_s_dts_a_all_selects_correct_count(self, client_trees_and_data):
        """Test RF_S_DTs_A_All selects correct total number of trees."""
        client_trees_list, X_val, y_val = client_trees_and_data
        n_total_trees = 20
        
        selected = rf_s_dts_a_all(client_trees_list, X_val, y_val, n_total_trees)
        
        assert len(selected) == 20

    def test_rf_s_dts_wa_all_selects_correct_count(self, client_trees_and_data):
        """Test RF_S_DTs_WA_All selects correct total number of trees."""
        client_trees_list, X_val, y_val = client_trees_and_data
        n_total_trees = 15
        
        selected = rf_s_dts_wa_all(client_trees_list, X_val, y_val, n_total_trees)
        
        assert len(selected) == 15

    def test_rf_s_dts_a_all_respects_max_available(self, client_trees_and_data):
        """Test RF_S_DTs_A_All handles request for more trees than available."""
        client_trees_list, X_val, y_val = client_trees_and_data
        n_total_trees = 100  # More than available (30 total)
        
        selected = rf_s_dts_a_all(client_trees_list, X_val, y_val, n_total_trees)
        
        # Should return all available trees (30)
        assert len(selected) == 30


class TestAggregateTrees:
    """Tests for aggregate_trees function."""

    @pytest.fixture
    def client_trees_and_data(self):
        """Create client trees and validation data."""
        X, y = make_classification(
            n_samples=300,
            n_features=10,
            n_classes=3,
            n_informative=5,
            random_state=42,
        )
        X_train, y_train = X[:200], y[:200]
        X_val, y_val = X[200:], y[200:]
        
        client_trees_list = []
        for i in range(3):
            rf = RandomForest(n_estimators=10, random_state=i)
            start = i * 66
            end = start + 66
            rf.fit(X_train[start:end], y_train[start:end])
            client_trees_list.append(rf.get_trees())
        
        return client_trees_list, X_val, y_val

    def test_aggregate_rf_s_dts_a(self, client_trees_and_data):
        """Test aggregate_trees with rf_s_dts_a strategy."""
        client_trees_list, X_val, y_val = client_trees_and_data
        
        selected = aggregate_trees(
            client_trees_list, X_val, y_val,
            strategy='rf_s_dts_a',
            n_trees_per_client=5,
        )
        
        assert len(selected) == 15

    def test_aggregate_rf_s_dts_wa(self, client_trees_and_data):
        """Test aggregate_trees with rf_s_dts_wa strategy."""
        client_trees_list, X_val, y_val = client_trees_and_data
        
        selected = aggregate_trees(
            client_trees_list, X_val, y_val,
            strategy='rf_s_dts_wa',
            n_trees_per_client=4,
        )
        
        assert len(selected) == 12

    def test_aggregate_rf_s_dts_a_all(self, client_trees_and_data):
        """Test aggregate_trees with rf_s_dts_a_all strategy."""
        client_trees_list, X_val, y_val = client_trees_and_data
        
        selected = aggregate_trees(
            client_trees_list, X_val, y_val,
            strategy='rf_s_dts_a_all',
            n_total_trees=20,
        )
        
        assert len(selected) == 20

    def test_aggregate_rf_s_dts_wa_all(self, client_trees_and_data):
        """Test aggregate_trees with rf_s_dts_wa_all strategy."""
        client_trees_list, X_val, y_val = client_trees_and_data
        
        selected = aggregate_trees(
            client_trees_list, X_val, y_val,
            strategy='rf_s_dts_wa_all',
            n_total_trees=18,
        )
        
        assert len(selected) == 18

    def test_aggregate_invalid_strategy_raises(self, client_trees_and_data):
        """Test aggregate_trees raises error for invalid strategy."""
        client_trees_list, X_val, y_val = client_trees_and_data
        
        with pytest.raises(ValueError) as exc_info:
            aggregate_trees(
                client_trees_list, X_val, y_val,
                strategy='invalid_strategy',
            )
        
        assert 'Unknown strategy' in str(exc_info.value)

    def test_aggregate_default_n_trees_per_client(self, client_trees_and_data):
        """Test aggregate_trees uses default n_trees_per_client."""
        client_trees_list, X_val, y_val = client_trees_and_data
        
        selected = aggregate_trees(
            client_trees_list, X_val, y_val,
            strategy='rf_s_dts_a',
        )
        
        # Default is 10 trees per client
        assert len(selected) == 30  # All trees (10 per client, but only 10 available)


class TestFederatedAggregator:
    """Tests for FederatedAggregator class."""

    @pytest.fixture
    def clients_and_data(self):
        """Create trained clients and test data."""
        X, y = make_classification(
            n_samples=400,
            n_features=10,
            n_classes=3,
            n_informative=5,
            random_state=42,
        )
        X_train, y_train = X[:240], y[:240]
        X_val, y_val = X[240:320], y[240:320]
        X_test, y_test = X[320:], y[320:]
        
        # Create 3 clients
        clients = []
        for i in range(3):
            client = ClientRF(
                client_id=i,
                rf_params={'n_estimators': 10, 'random_state': i}
            )
            start = i * 80
            end = start + 80
            client.train(X_train[start:end], y_train[start:end])
            clients.append(client)
        
        return clients, X_val, y_val, X_test, y_test

    def test_initialization(self):
        """Test FederatedAggregator initialization."""
        aggregator = FederatedAggregator(
            strategy='rf_s_dts_a',
            n_trees_per_client=5,
            n_total_trees=50,
        )
        
        assert aggregator.strategy == 'rf_s_dts_a'
        assert aggregator.n_trees_per_client == 5
        assert aggregator.n_total_trees == 50

    def test_aggregate_returns_trees(self, clients_and_data):
        """Test aggregate returns list of trees."""
        clients, X_val, y_val, _, _ = clients_and_data
        
        aggregator = FederatedAggregator(
            strategy='rf_s_dts_a',
            n_trees_per_client=5,
        )
        trees = aggregator.aggregate(clients, X_val, y_val)
        
        assert isinstance(trees, list)
        assert len(trees) == 15  # 5 trees * 3 clients

    def test_build_global_rf(self, clients_and_data):
        """Test build_global_rf creates RF."""
        clients, X_val, y_val, X_test, _ = clients_and_data
        
        aggregator = FederatedAggregator(strategy='rf_s_dts_a', n_trees_per_client=5)
        aggregator.aggregate(clients, X_val, y_val)
        
        classes = clients[0].rf._classes
        global_rf = aggregator.build_global_rf(classes)
        
        assert global_rf is not None
        predictions = global_rf.predict(X_test)
        assert predictions.shape == (len(X_test),)

    def test_evaluate(self, clients_and_data):
        """Test evaluate returns metrics."""
        clients, X_val, y_val, X_test, y_test = clients_and_data
        
        aggregator = FederatedAggregator(strategy='rf_s_dts_a', n_trees_per_client=5)
        aggregator.aggregate(clients, X_val, y_val)
        
        classes = clients[0].rf._classes
        aggregator.build_global_rf(classes)
        
        metrics = aggregator.evaluate(X_test, y_test)
        
        assert 'accuracy' in metrics
        assert 'weighted_accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1

    def test_evaluate_before_build_raises(self, clients_and_data):
        """Test evaluate raises error if global RF not built."""
        clients, X_val, y_val, X_test, y_test = clients_and_data
        
        aggregator = FederatedAggregator(strategy='rf_s_dts_a', n_trees_per_client=5)
        aggregator.aggregate(clients, X_val, y_val)
        
        with pytest.raises(RuntimeError):
            aggregator.evaluate(X_test, y_test)

    def test_different_strategies(self, clients_and_data):
        """Test different aggregation strategies work."""
        clients, X_val, y_val, X_test, y_test = clients_and_data
        
        strategies = ['rf_s_dts_a', 'rf_s_dts_wa', 'rf_s_dts_a_all', 'rf_s_dts_wa_all']
        
        for strategy in strategies:
            aggregator = FederatedAggregator(
                strategy=strategy,
                n_trees_per_client=5,
                n_total_trees=15,
            )
            aggregator.aggregate(clients, X_val, y_val)
            
            classes = clients[0].rf._classes
            aggregator.build_global_rf(classes)
            
            metrics = aggregator.evaluate(X_test, y_test)
            assert 0 <= metrics['accuracy'] <= 1
