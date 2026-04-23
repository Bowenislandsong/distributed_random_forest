"""
Accuracy and latency checks on a public, stratified UCI holdout (breast cancer).

These are *regression* bounds (not load tests): if they fail, investigate model or
data-loader changes, not the runner hardware.
"""

from __future__ import annotations

import time

import pytest

from distributed_random_forest import ClientRF, FederatedAggregator, RandomForest
from distributed_random_forest.datasets import load_breast_cancer_bench
from distributed_random_forest.experiments.exp2_clients import partition_uniform_random
from tests.timing import max_per_sample_seconds, max_wall_seconds

pytestmark = pytest.mark.performance


class TestPublicDatasetPerformance:
    """Wisconsin breast cancer: realistic accuracy; prediction stays fast on CPU."""

    @pytest.mark.parametrize("random_state", [0, 42])
    def test_central_rf_accuracy_breast_cancer(self, random_state: int) -> None:
        split = load_breast_cancer_bench(random_state=random_state)
        rf = RandomForest(
            n_estimators=48,
            criterion="gini",
            voting="simple",
            random_state=random_state,
        )
        rf.fit(split.X_train, split.y_train, split.X_val, split.y_val)
        acc = rf.score(split.X_test, split.y_test)
        # Strong linear signal; a healthy RF should land well above chance.
        assert acc >= 0.88, f"unexpected central accuracy {acc:.4f}"

    def test_central_predict_latency(self, breast_cancer_split) -> None:
        split = breast_cancer_split
        rf = RandomForest(
            n_estimators=32,
            random_state=99,
        )
        rf.fit(split.X_train, split.y_train, split.X_val, split.y_val)
        t0 = time.perf_counter()
        rf.predict(split.X_test)
        dt = time.perf_counter() - t0
        n = len(split.X_test)
        assert dt < max_wall_seconds(2.0), f"full-batch predict too slow: {dt:.2f}s for n={n}"
        per_sample = dt / n
        assert per_sample < max_per_sample_seconds(0.01), (
            f"per-sample predict too slow: {per_sample*1000:.1f} ms"
        )

    def test_federated_global_accuracy(self, breast_cancer_split) -> None:
        split = breast_cancer_split
        n_clients = 3
        n_trees_per = 12
        partitions = partition_uniform_random(
            split.X_train, split.y_train, n_clients, random_state=1
        )
        rf_params = {"n_estimators": 40, "random_state": 7}
        clients = []
        for i, (xc, yc) in enumerate(partitions):
            c = ClientRF(client_id=i, rf_params=rf_params)
            c.train(xc, yc)
            clients.append(c)
        ag = FederatedAggregator(
            strategy="rf_s_dts_a",
            n_trees_per_client=n_trees_per,
        )
        ag.aggregate(clients, split.X_val, split.y_val)
        ag.build_global_rf(clients[0].rf._classes)
        m = ag.evaluate(split.X_test, split.y_test)
        assert m["accuracy"] >= 0.80, f"global RF weak on holdout: {m['accuracy']:.4f}"

    def test_federated_predict_not_prohibitively_slow(self, breast_cancer_split) -> None:
        split = breast_cancer_split
        clients = []
        for i, (xc, yc) in enumerate(
            partition_uniform_random(split.X_train, split.y_train, 2, random_state=3)
        ):
            c = ClientRF(client_id=i, rf_params={"n_estimators": 24, "random_state": 0})
            c.train(xc, yc)
            clients.append(c)
        ag = FederatedAggregator(strategy="rf_s_dts_a", n_trees_per_client=8)
        ag.aggregate(clients, split.X_val, split.y_val)
        ag.build_global_rf(clients[0].rf._classes)
        assert ag.global_rf is not None
        t0 = time.perf_counter()
        ag.global_rf.predict(split.X_test)
        dt = time.perf_counter() - t0
        assert dt < max_wall_seconds(3.0), f"global predict {dt:.2f}s for n={len(split.y_test)}"


class TestNJobsPredictDeterminism:
    """``n_jobs=1`` vs ``-1`` for merged:custom RF must agree on ``predict`` / ``score``."""

    def test_merged_forest_n_jobs_1_and_neg1(
        self,
        breast_cancer_split,
    ) -> None:
        split = breast_cancer_split
        a = RandomForest(
            n_estimators=16,
            random_state=0,
            n_jobs=1,
        )
        a.fit(
            split.X_train,
            split.y_train,
            split.X_val,
            split.y_val,
        )
        t1, t2 = a.get_trees(), a._classes
        r1 = RandomForest(n_jobs=1)
        r1.set_trees(t1, t2)
        r2 = RandomForest(n_jobs=-1)
        r2.set_trees(t1, t2)
        p1, p2 = r1.predict(split.X_test), r2.predict(split.X_test)
        assert (p1 == p2).all()
        assert r1.score(split.X_test, split.y_test) == r2.score(
            split.X_test, split.y_test
        )


class TestArrayAllocationOverhead:
    """Micro-check that repeated small predicts stay bounded (sanity, not a benchmark)."""

    def test_many_small_predicts(self, random_xy_small) -> None:
        X, y = random_xy_small
        Xtr, ytr, Xte, yte = X[:32], y[:32], X[32:], y[32:]
        rf = RandomForest(n_estimators=12, random_state=0)
        rf.fit(Xtr, ytr)
        t0 = time.perf_counter()
        for _ in range(50):
            p = rf.predict(Xte)
            assert p.shape == (len(yte),)
        assert time.perf_counter() - t0 < max_wall_seconds(2.0)
