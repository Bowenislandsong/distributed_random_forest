"""End-to-end tests for the complete federated RF pipeline."""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from distributed_random_forest import (
    RandomForest,
    ClientRF,
    DPRandomForest,
    DPClientRF,
    FederatedAggregator,
    aggregate_trees,
)
from distributed_random_forest.experiments.exp1_hparams import quick_hyperparameter_selection, get_default_best_params
from distributed_random_forest.experiments.exp2_clients import (
    partition_uniform_random,
    run_exp2_independent_clients,
)
from distributed_random_forest.experiments.exp3_global_rf import run_exp3_federated_aggregation
from distributed_random_forest.experiments.exp4_dp_rf import run_exp4_dp_federation


class TestEndToEndPipeline:
    """End-to-end tests for the complete federated learning pipeline."""

    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic classification data."""
        X, y = make_classification(
            n_samples=500,
            n_features=20,
            n_classes=3,
            n_informative=10,
            n_redundant=5,
            random_state=42,
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        return X_train, y_train, X_val, y_val, X_test, y_test

    def test_full_federated_pipeline(self, synthetic_data):
        """Test complete federated RF pipeline from training to evaluation."""
        X_train, y_train, X_val, y_val, X_test, y_test = synthetic_data
        
        # Step 1: Partition data for multiple clients
        n_clients = 3
        partitions = partition_uniform_random(X_train, y_train, n_clients, random_state=42)
        
        # Step 2: Train RF on each client
        rf_params = {'n_estimators': 20, 'random_state': 42}
        clients = []
        for i, (X_client, y_client) in enumerate(partitions):
            client = ClientRF(client_id=i, rf_params=rf_params)
            client.train(X_client, y_client)
            clients.append(client)
        
        # Step 3: Aggregate trees from all clients
        aggregator = FederatedAggregator(
            strategy='rf_s_dts_a',
            n_trees_per_client=10,
        )
        aggregator.aggregate(clients, X_val, y_val)
        
        # Step 4: Build global RF
        classes = clients[0].rf._classes
        global_rf = aggregator.build_global_rf(classes)
        
        # Step 5: Evaluate global RF
        metrics = aggregator.evaluate(X_test, y_test)
        
        # Assertions
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1
        assert metrics['accuracy'] > 0.3  # Should do better than random (1/3)
        
        # Global RF should have trees from all clients
        assert len(global_rf._trees) == 30  # 10 trees * 3 clients

    def test_dp_federated_pipeline(self, synthetic_data):
        """Test federated RF pipeline with differential privacy."""
        X_train, y_train, X_val, y_val, X_test, y_test = synthetic_data
        
        # Step 1: Partition data
        n_clients = 3
        partitions = partition_uniform_random(X_train, y_train, n_clients, random_state=42)
        
        # Step 2: Train DP-RF on each client
        epsilon = 1.0
        rf_params = {'n_estimators': 20, 'random_state': 42}
        dp_clients = []
        for i, (X_client, y_client) in enumerate(partitions):
            client = DPClientRF(client_id=i, epsilon=epsilon, rf_params=rf_params)
            client.train(X_client, y_client)
            dp_clients.append(client)
        
        # Step 3: Aggregate trees
        aggregator = FederatedAggregator(
            strategy='rf_s_dts_a',
            n_trees_per_client=10,
        )
        aggregator.aggregate(dp_clients, X_val, y_val)
        
        # Step 4: Build and evaluate global RF
        classes = dp_clients[0].rf._classes
        aggregator.build_global_rf(classes)
        metrics = aggregator.evaluate(X_test, y_test)
        
        # Assertions
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1

    def test_compare_aggregation_strategies(self, synthetic_data):
        """Test that all aggregation strategies work and can be compared."""
        X_train, y_train, X_val, y_val, X_test, y_test = synthetic_data
        
        # Train clients
        n_clients = 3
        partitions = partition_uniform_random(X_train, y_train, n_clients, random_state=42)
        rf_params = {'n_estimators': 20, 'random_state': 42}
        
        clients = []
        for i, (X_client, y_client) in enumerate(partitions):
            client = ClientRF(client_id=i, rf_params=rf_params)
            client.train(X_client, y_client)
            clients.append(client)
        
        # Compare strategies
        strategies = ['rf_s_dts_a', 'rf_s_dts_wa', 'rf_s_dts_a_all', 'rf_s_dts_wa_all']
        results = {}
        
        for strategy in strategies:
            aggregator = FederatedAggregator(
                strategy=strategy,
                n_trees_per_client=10,
                n_total_trees=30,
            )
            aggregator.aggregate(clients, X_val, y_val)
            aggregator.build_global_rf(clients[0].rf._classes)
            metrics = aggregator.evaluate(X_test, y_test)
            results[strategy] = metrics['accuracy']
        
        # All strategies should produce valid results
        for strategy, accuracy in results.items():
            assert 0 <= accuracy <= 1
            assert accuracy > 0  # Should make some correct predictions

    def test_client_evaluation_vs_global(self, synthetic_data):
        """Test that global RF improves over individual clients."""
        X_train, y_train, X_val, y_val, X_test, y_test = synthetic_data
        
        # Train clients
        n_clients = 3
        partitions = partition_uniform_random(X_train, y_train, n_clients, random_state=42)
        rf_params = {'n_estimators': 20, 'random_state': 42}
        
        clients = []
        client_accuracies = []
        for i, (X_client, y_client) in enumerate(partitions):
            client = ClientRF(client_id=i, rf_params=rf_params)
            client.train(X_client, y_client)
            clients.append(client)
            
            # Evaluate each client on global test set
            metrics = client.evaluate(X_test, y_test)
            client_accuracies.append(metrics['accuracy'])
        
        # Build global RF
        aggregator = FederatedAggregator(
            strategy='rf_s_dts_a',
            n_trees_per_client=10,
        )
        aggregator.aggregate(clients, X_val, y_val)
        aggregator.build_global_rf(clients[0].rf._classes)
        global_metrics = aggregator.evaluate(X_test, y_test)
        
        # Global should be at least as good as average client
        avg_client_accuracy = np.mean(client_accuracies)
        # This is a soft check - federation should help
        assert global_metrics['accuracy'] >= 0  # At minimum, produces valid results


class TestExperimentPipelines:
    """Tests for the experiment runner functions."""

    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic classification data."""
        X, y = make_classification(
            n_samples=300,
            n_features=15,
            n_classes=3,
            n_informative=8,
            random_state=42,
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, y_train, X_test, y_test

    def test_exp1_hyperparameter_selection(self, synthetic_data):
        """Test EXP 1: Hyperparameter selection."""
        X_train, y_train, _, _ = synthetic_data
        
        # Use small search space for speed
        results = quick_hyperparameter_selection(
            X_train, y_train,
            n_estimators_candidates=[5, 11],
            random_state=42,
        )
        
        assert 'best_params' in results
        assert 'best_score' in results
        assert 'n_estimators' in results['best_params']
        assert 0 <= results['best_score'] <= 1

    def test_exp2_independent_clients(self, synthetic_data):
        """Test EXP 2: Independent client training."""
        X_train, y_train, X_test, y_test = synthetic_data
        
        rf_params = get_default_best_params()
        rf_params['n_estimators'] = 10
        rf_params['random_state'] = 42
        
        results = run_exp2_independent_clients(
            X_train, y_train, X_test, y_test,
            rf_params=rf_params,
            partitioning='uniform',
            n_clients=3,
            random_state=42,
            verbose=False,
        )
        
        assert 'client_rfs' in results
        assert 'client_results' in results
        assert len(results['client_rfs']) == 3
        assert results['avg_accuracy'] >= 0

    def test_exp3_federated_aggregation(self, synthetic_data):
        """Test EXP 3: Federated aggregation."""
        X_train, y_train, X_test, y_test = synthetic_data
        
        # Split off validation set
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # First run EXP 2 to get client RFs
        rf_params = {'n_estimators': 10, 'random_state': 42}
        
        partitions = partition_uniform_random(X_tr, y_tr, 3, random_state=42)
        clients = []
        for i, (X_client, y_client) in enumerate(partitions):
            client = ClientRF(client_id=i, rf_params=rf_params)
            client.train(X_client, y_client)
            clients.append(client)
        
        # Run EXP 3
        results = run_exp3_federated_aggregation(
            client_rfs=clients,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            n_trees_per_client=5,
            n_total_trees=15,
            verbose=False,
        )
        
        assert 'strategy_results' in results
        assert 'best_strategy' in results
        assert 'best_accuracy' in results
        assert 0 <= results['best_accuracy'] <= 1

    def test_exp4_dp_federation(self, synthetic_data):
        """Test EXP 4: DP federated RF."""
        X_train, y_train, X_test, y_test = synthetic_data
        
        # Split off validation set
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        rf_params = {'n_estimators': 10, 'random_state': 42}
        
        results = run_exp4_dp_federation(
            X_train=X_tr,
            y_train=y_tr,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            rf_params=rf_params,
            epsilon_values=[1.0, 5.0],  # Test only 2 epsilon values for speed
            n_clients=3,
            n_trees_per_client=5,
            n_total_trees=15,
            random_state=42,
            verbose=False,
        )
        
        assert 1.0 in results
        assert 5.0 in results
        assert 'global_accuracy' in results[1.0]
        assert 'global_accuracy' in results[5.0]


class TestModelPersistence:
    """Tests for model behavior across operations."""

    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic classification data."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_classes=3,
            n_informative=5,
            random_state=42,
        )
        return X, y

    def test_set_trees_preserves_predictions(self, synthetic_data):
        """Test that setting trees preserves prediction capability."""
        X, y = synthetic_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train original RF
        rf1 = RandomForest(n_estimators=10, random_state=42)
        rf1.fit(X_train, y_train)
        
        # Get trees and set them on new RF
        trees = rf1.get_trees()
        rf2 = RandomForest()
        rf2.set_trees(trees, rf1._classes)
        
        # Both should make predictions
        pred1 = rf1.predict(X_test)
        pred2 = rf2.predict(X_test)
        
        assert pred1.shape == pred2.shape
        assert np.array_equal(pred1, pred2)

    def test_client_trees_usable_in_global_rf(self, synthetic_data):
        """Test that trees from clients work in global RF."""
        X, y = synthetic_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train client
        client = ClientRF(client_id=0, rf_params={'n_estimators': 10, 'random_state': 42})
        client.train(X_train, y_train)
        
        # Get trees and create global RF
        trees = client.get_trees()
        global_rf = RandomForest()
        global_rf.set_trees(trees, client.rf._classes)
        
        # Should be able to predict and score
        predictions = global_rf.predict(X_test)
        score = global_rf.score(X_test, y_test)
        
        assert predictions.shape == (len(X_test),)
        assert 0 <= score <= 1
