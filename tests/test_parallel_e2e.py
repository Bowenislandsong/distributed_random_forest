"""
End-to-end checks that parallel ``n_jobs`` paths match sequential ones.

Where validation scores can tie, tree order *could* differ; we assert metrics and
test-set predictions are identical for fixed seeds in practice.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from distributed_random_forest import (
    ClientRF,
    FederatedAggregator,
    RandomForest,
    resolve_n_jobs,
)
from distributed_random_forest.experiments.exp2_clients import partition_uniform_random
from distributed_random_forest.experiments.exp3_global_rf import run_exp3_federated_aggregation

pytestmark = [pytest.mark.e2e, pytest.mark.parallel_parity]


def _synthetic_federated_split(
    n_samples: int = 800,
    random_state: int = 0,
) -> tuple[np.ndarray, ...]:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=16,
        n_classes=3,
        n_informative=8,
        random_state=random_state,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


class TestFederatedAggregatorNJobsParity:
    def test_same_accuracies_and_preds_synthetic(
        self,
    ) -> None:
        X_train, y_train, X_val, y_val, X_test, y_test = _synthetic_federated_split()
        n_clients = 3
        rf_params = {"n_estimators": 24, "random_state": 1}
        clients = []
        for i, (xc, yc) in enumerate(
            partition_uniform_random(
                X_train, y_train, n_clients, random_state=2
            )
        ):
            c = ClientRF(client_id=i, rf_params=rf_params)
            c.train(xc, yc)
            clients.append(c)
        for strategy in (
            "rf_s_dts_a",
            "rf_s_dts_wa",
            "rf_s_dts_a_all",
            "rf_s_dts_wa_all",
        ):
            a1 = FederatedAggregator(
                strategy=strategy,
                n_trees_per_client=6,
                n_total_trees=12,
                n_jobs=1,
            )
            a1.aggregate(clients, X_val, y_val)
            a1.build_global_rf(clients[0].rf._classes)
            m1 = a1.evaluate(X_test, y_test)
            p1 = a1.global_rf.predict(X_test)

            a2 = FederatedAggregator(
                strategy=strategy,
                n_trees_per_client=6,
                n_total_trees=12,
                n_jobs=-1,
            )
            a2.aggregate(clients, X_val, y_val)
            a2.build_global_rf(clients[0].rf._classes)
            m2 = a2.evaluate(X_test, y_test)
            p2 = a2.global_rf.predict(X_test)

            assert m1["accuracy"] == m2["accuracy"]
            assert m1["weighted_accuracy"] == m2["weighted_accuracy"]
            assert np.array_equal(p1, p2), f"mismatch for strategy {strategy}"


class TestFederatedNJobsOnPublicData:
    def test_parity_breast_cancer(
        self,
        breast_cancer_split,
    ) -> None:
        split = breast_cancer_split
        clients = []
        for i, (xc, yc) in enumerate(
            partition_uniform_random(
                split.X_train, split.y_train, 2, random_state=5
            )
        ):
            c = ClientRF(
                client_id=i,
                rf_params={"n_estimators": 18, "random_state": 0},
            )
            c.train(xc, yc)
            clients.append(c)
        for nj in (1, resolve_n_jobs(-1)):
            a = FederatedAggregator(
                strategy="rf_s_dts_wa",
                n_trees_per_client=6,
                n_jobs=nj,
            )
            a.aggregate(clients, split.X_val, split.y_val)
            a.build_global_rf(clients[0].rf._classes)
            m = a.evaluate(split.X_test, split.y_test)
            assert 0.0 < m["accuracy"] <= 1.0
        a1 = FederatedAggregator(
            "rf_s_dts_wa", n_trees_per_client=6, n_jobs=1
        )
        a1.aggregate(clients, split.X_val, split.y_val)
        a1.build_global_rf(clients[0].rf._classes)
        a2 = FederatedAggregator(
            "rf_s_dts_wa", n_trees_per_client=6, n_jobs=-1
        )
        a2.aggregate(clients, split.X_val, split.y_val)
        a2.build_global_rf(clients[0].rf._classes)
        e1, e2 = a1.evaluate(split.X_test, split.y_test), a2.evaluate(
            split.X_test, split.y_test
        )
        assert e1["accuracy"] == e2["accuracy"]
        assert np.array_equal(
            a1.global_rf.predict(split.X_test),
            a2.global_rf.predict(split.X_test),
        )


class TestExp3NJobsParity:
    def test_exp3_best_accuracy_unchanged(
        self,
    ) -> None:
        X_tr, y_tr, X_v, y_v, X_te, y_te = _synthetic_federated_split(
            n_samples=600, random_state=3
        )
        clients = []
        for i, (xc, yc) in enumerate(
            partition_uniform_random(X_tr, y_tr, 2, random_state=4)
        ):
            c = ClientRF(
                client_id=i,
                rf_params={"n_estimators": 14, "random_state": 0},
            )
            c.train(xc, yc)
            clients.append(c)
        r1 = run_exp3_federated_aggregation(
            clients,
            X_v,
            y_v,
            X_te,
            y_te,
            n_trees_per_client=5,
            n_total_trees=8,
            n_jobs=1,
            verbose=False,
        )
        r2 = run_exp3_federated_aggregation(
            clients,
            X_v,
            y_v,
            X_te,
            y_te,
            n_trees_per_client=5,
            n_total_trees=8,
            n_jobs=-1,
            verbose=False,
        )
        assert r1["best_accuracy"] == r2["best_accuracy"]
        assert r1["best_strategy"] == r2["best_strategy"]


class TestRandomForestPredictNJobs:
    def test_merged_forest_predict_parity(
        self,
    ) -> None:
        X, y = make_classification(
            n_samples=300,
            n_features=12,
            n_classes=2,
            random_state=0,
        )
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.3, random_state=0, stratify=y
        )
        rf = RandomForest(n_estimators=20, random_state=0, n_jobs=-1)
        rf.fit(Xtr, ytr)
        trees, classes = rf.get_trees(), rf._classes
        a = RandomForest(n_jobs=1)
        a.set_trees(trees, classes)
        b = RandomForest(n_jobs=-1)
        b.set_trees(trees, classes)
        p1, p2 = a.predict(Xte), b.predict(Xte)
        assert np.array_equal(p1, p2)
        s1, s2 = a.score(Xte, yte), b.score(Xte, yte)
        assert s1 == s2
