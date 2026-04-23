"""
Stress / scale checks: larger client + tree counts with ``n_jobs=-1`` must finish
and return sane metrics. Not machine-load benchmarks; CI-safe runtimes.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from distributed_random_forest import ClientRF, FederatedAggregator, RandomForest
from distributed_random_forest.datasets import load_breast_cancer_bench
from distributed_random_forest.experiments.exp2_clients import partition_uniform_random
from distributed_random_forest.models.tree_utils import rank_trees_by_metric
from tests.timing import max_wall_seconds

pytestmark = pytest.mark.stress


def _make_split(
    n_samples: int, random_state: int
) -> tuple[np.ndarray, ...]:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=24,
        n_informative=10,
        n_classes=3,
        random_state=random_state,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=random_state, stratify=y_train
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


class TestScaleManyClientsAndTrees:
    def test_8_clients_20_trees_each_aggregation(
        self,
    ) -> None:
        X_tr, y_tr, X_v, y_v, X_te, y_te = _make_split(2400, random_state=1)
        n_clients = 8
        clients = []
        for i, (xc, yc) in enumerate(
            partition_uniform_random(
                X_tr, y_tr, n_clients, random_state=2
            )
        ):
            c = ClientRF(
                client_id=i,
                rf_params={"n_estimators": 20, "random_state": i + 1},
            )
            c.train(xc, yc)
            clients.append(c)
        t0 = time.perf_counter()
        a = FederatedAggregator(
            strategy="rf_s_dts_a_all",
            n_total_trees=48,
            n_jobs=-1,
        )
        a.aggregate(clients, X_v, y_v)
        a.build_global_rf(clients[0].rf._classes)
        m = a.evaluate(X_te, y_te)
        wall = time.perf_counter() - t0
        assert wall < max_wall_seconds(60.0), f"aggregation too slow: {wall:.1f}s"
        assert 0.0 < m["accuracy"] <= 1.0
        assert len(a.global_trees) == 48

    def test_rank_hundreds_of_trees_breast_cancer(
        self,
    ) -> None:
        """Many candidate trees: ranking + parallel should complete."""
        split = load_breast_cancer_bench(random_state=0)
        X_tr, y_tr, X_v, y_v, X_te, y_te = (
            split.X_train,
            split.y_train,
            split.X_val,
            split.y_val,
            split.X_test,
            split.y_test,
        )
        clients = []
        for i, (xc, yc) in enumerate(
            partition_uniform_random(
                X_tr, y_tr, 4, random_state=3
            )
        ):
            c = ClientRF(
                client_id=i,
                rf_params={"n_estimators": 32, "random_state": 7 + i},
            )
            c.train(xc, yc)
            clients.append(c)
        all_trees: list = []
        for c in clients:
            all_trees.extend(c.get_trees())
        assert len(all_trees) == 4 * 32
        t0 = time.perf_counter()
        ranked = rank_trees_by_metric(
            all_trees,
            X_v,
            y_v,
            metric="weighted_accuracy",
            n_jobs=-1,
        )
        assert time.perf_counter() - t0 < max_wall_seconds(45.0)
        assert len(ranked) == len(all_trees)
        accs = [s for _, s in ranked]
        assert accs == sorted(accs, reverse=True)
        a = RandomForest(n_jobs=-1)
        a.set_trees([t for t, _ in ranked[:60]], clients[0].rf._classes)
        s = a.score(X_te, y_te)
        assert 0.0 < s <= 1.0


class TestCentralRFLarge:
    def test_fit_200_estimators_breast_cancer(
        self,
    ) -> None:
        split = load_breast_cancer_bench(random_state=42)
        t0 = time.perf_counter()
        rf = RandomForest(
            n_estimators=200,
            n_jobs=-1,
            random_state=42,
        )
        rf.fit(
            split.X_train,
            split.y_train,
            split.X_val,
            split.y_val,
        )
        s = rf.score(split.X_test, split.y_test)
        assert time.perf_counter() - t0 < max_wall_seconds(30.0)
        assert 0.85 < s <= 1.0
