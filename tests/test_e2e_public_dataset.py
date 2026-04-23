"""End-to-end tests on a real public (UCI, via scikit-learn) holdout — not only synthetic data."""

import pytest

from distributed_random_forest import (
    ClientRF,
    DPClientRF,
    DPRandomForest,
    FederatedAggregator,
)
from distributed_random_forest.datasets import load_breast_cancer_bench
from distributed_random_forest.experiments.exp1_hparams import quick_hyperparameter_selection
from distributed_random_forest.experiments.exp2_clients import (
    partition_uniform_random,
    run_exp2_independent_clients,
)
from distributed_random_forest.experiments.exp3_global_rf import run_exp3_federated_aggregation
from distributed_random_forest.experiments.exp4_dp_rf import run_exp4_dp_federation


class TestE2EOnBreastCancer:
    """EXP-style runners on the Wisconsin breast cancer split."""

    @pytest.fixture
    def split(self):
        return load_breast_cancer_bench(random_state=42)

    def test_full_federated_merges_and_scores(self, split):
        n_clients = 3
        partitions = partition_uniform_random(
            split.X_train, split.y_train, n_clients, random_state=0
        )
        rf_params = {"n_estimators": 20, "random_state": 1}
        clients = []
        for i, (xc, yc) in enumerate(partitions):
            c = ClientRF(client_id=i, rf_params=rf_params)
            c.train(xc, yc)
            clients.append(c)
        ag = FederatedAggregator(strategy="rf_s_dts_a", n_trees_per_client=8)
        ag.aggregate(clients, split.X_val, split.y_val)
        ag.build_global_rf(clients[0].rf._classes)
        m = ag.evaluate(split.X_test, split.y_test)
        assert 0.75 <= m["accuracy"] <= 1.0
        n_global = len(ag.global_trees) if ag.global_trees is not None else 0
        assert n_global == 8 * n_clients

    def test_all_aggregation_strategies_run(self, split):
        partitions = partition_uniform_random(
            split.X_train, split.y_train, 2, random_state=3
        )
        clients = []
        for i, (xc, yc) in enumerate(partitions):
            c = ClientRF(client_id=i, rf_params={"n_estimators": 16, "random_state": 0})
            c.train(xc, yc)
            clients.append(c)
        for strategy in [
            "rf_s_dts_a",
            "rf_s_dts_wa",
            "rf_s_dts_a_all",
            "rf_s_dts_wa_all",
        ]:
            ag = FederatedAggregator(
                strategy=strategy,
                n_trees_per_client=5,
                n_total_trees=10,
            )
            ag.aggregate(clients, split.X_val, split.y_val)
            ag.build_global_rf(clients[0].rf._classes)
            m = ag.evaluate(split.X_test, split.y_test)
            assert 0.0 < m["accuracy"] <= 1.0

    def test_exp1_exp2_on_public_data(self, split):
        h = quick_hyperparameter_selection(
            split.X_train,
            split.y_train,
            n_estimators_candidates=[7, 11],
            random_state=0,
        )
        assert 0.0 <= h["best_score"] <= 1.0
        p = h["best_params"]
        p["n_estimators"] = min(11, p.get("n_estimators", 11))
        p["random_state"] = 0
        r2 = run_exp2_independent_clients(
            split.X_train,
            split.y_train,
            split.X_test,
            split.y_test,
            rf_params=p,
            partitioning="uniform",
            n_clients=2,
            random_state=0,
            verbose=False,
        )
        assert r2["avg_accuracy"] > 0.5

    def test_exp3_federated_aggregation(self, split):
        parts = partition_uniform_random(
            split.X_train, split.y_train, 2, random_state=5
        )
        clients = []
        for i, (xc, yc) in enumerate(parts):
            c = ClientRF(client_id=i, rf_params={"n_estimators": 12, "random_state": 0})
            c.train(xc, yc)
            clients.append(c)
        r3 = run_exp3_federated_aggregation(
            client_rfs=clients,
            X_val=split.X_val,
            y_val=split.y_val,
            X_test=split.X_test,
            y_test=split.y_test,
            n_trees_per_client=5,
            n_total_trees=10,
            verbose=False,
        )
        assert 0.0 < r3["best_accuracy"] <= 1.0
        assert r3["best_strategy"] in r3["strategy_results"]

    def test_exp4_dp_federation_runs(self, split):
        r4 = run_exp4_dp_federation(
            X_train=split.X_train,
            y_train=split.y_train,
            X_val=split.X_val,
            y_val=split.y_val,
            X_test=split.X_test,
            y_test=split.y_test,
            rf_params={"n_estimators": 10, "random_state": 0},
            epsilon_values=(2.0, 5.0),
            n_clients=2,
            n_trees_per_client=4,
            n_total_trees=8,
            random_state=0,
            verbose=False,
        )
        assert 2.0 in r4 and 5.0 in r4
        for eps in (2.0, 5.0):
            assert "global_accuracy" in r4[eps]


class TestDPModesOnRealData:
    """DP model classes accept the UCI feature matrix and produce finite metrics."""

    def test_dp_random_forest_fit_predict(self, breast_cancer_split):
        split = breast_cancer_split
        dp = DPRandomForest(
            n_estimators=12,
            epsilon=2.0,
            dp_mechanism="laplace",
            random_state=0,
        )
        dp.fit(split.X_train, split.y_train)
        assert dp.score(split.X_test, split.y_test) >= 0.0
        p = dp.predict(split.X_test)
        assert p.shape == (len(split.y_test),)

    def test_dp_client_in_federated_list(self, breast_cancer_split):
        split = breast_cancer_split
        (xc, yc), = partition_uniform_random(
            split.X_train, split.y_train, 1, random_state=0
        )
        c = DPClientRF(client_id=0, epsilon=2.0, rf_params={"n_estimators": 8})
        c.train(xc, yc)
        ag = FederatedAggregator(strategy="rf_s_dts_a", n_trees_per_client=4)
        ag.aggregate([c], split.X_val, split.y_val)
        ag.build_global_rf(c.rf._classes)
        m = ag.evaluate(split.X_test, split.y_test)
        assert 0.0 < m["accuracy"] <= 1.0
