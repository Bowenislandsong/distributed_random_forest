#!/usr/bin/env python3
"""
Run a *central* and *federated* Random Forest on a public, pandas-friendly dataset.

Dataset: **Wisconsin Breast Cancer (UCI)** — via :func:`sklearn.datasets.load_breast_cancer`
(see :func:`distributed_random_forest.datasets.load_breast_cancer_bench`).

From the repository root, after ``pip install -e .``::

    python examples/benchmark_public_dataset.py
    python examples/benchmark_public_dataset.py --quick
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Allow `python examples/benchmark_public_dataset.py` from a clone without `pip install -e .`
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402

from distributed_random_forest import (  # noqa: E402
    ClientRF,
    FederatedAggregator,
    RandomForest,
)
from distributed_random_forest.datasets import (  # noqa: E402
    load_breast_cancer_bench,
    summarize_split,
)
from distributed_random_forest.experiments.exp2_clients import (  # noqa: E402
    partition_uniform_random,
)


def _time_predict(model: Any, X: np.ndarray, *, n_warmup: int = 1) -> float:
    """Return seconds for ``predict`` on *X* (single batch) after a short warmup."""
    for _ in range(n_warmup):
        model.predict(X)
    t0 = time.perf_counter()
    model.predict(X)
    return time.perf_counter() - t0


def run_benchmark(
    *,
    n_estimators: int,
    n_clients: int,
    n_trees_per_client: int,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train central and federated models; return metrics and timings."""
    split = load_breast_cancer_bench(
        as_frame=True,
        random_state=random_state,
    )
    assert split.data_frame is not None
    frame = split.data_frame

    out: Dict[str, Any] = {
        "dataset": split.name,
        "description": split.description,
        "n_rows": int(len(frame)),
        "split": summarize_split(split),
    }

    X_train, y_train = split.X_train, split.y_train
    X_val, y_val = split.X_val, split.y_val
    X_test, y_test = split.X_test, split.y_test
    n_test = len(y_test)

    # ---- Central (single-site) model ----
    t_fit0 = time.perf_counter()
    central = RandomForest(
        n_estimators=n_estimators,
        criterion="gini",
        voting="simple",
        random_state=random_state,
    )
    central.fit(X_train, y_train, X_val, y_val)
    fit_central = time.perf_counter() - t_fit0
    acc_central = float(central.score(X_test, y_test))
    lat_central = _time_predict(central, X_test)

    out["central"] = {
        "test_accuracy": acc_central,
        "fit_time_s": fit_central,
        "predict_latency_s": lat_central,
        "predict_ms_per_sample": 1000.0 * lat_central / n_test,
    }

    # ---- Federated: one RF per client, then global merge ----
    partitions = partition_uniform_random(
        X_train, y_train, n_clients=n_clients, random_state=random_state
    )
    rf_params = {
        "n_estimators": n_estimators,
        "random_state": random_state,
    }
    clients: list = []
    t_fed_local = 0.0
    for i, (X_c, y_c) in enumerate(partitions):
        t0 = time.perf_counter()
        c = ClientRF(client_id=i, rf_params=rf_params)
        c.train(X_c, y_c)
        t_fed_local += time.perf_counter() - t0
        clients.append(c)

    t_agg0 = time.perf_counter()
    ag = FederatedAggregator(
        strategy="rf_s_dts_a",
        n_trees_per_client=n_trees_per_client,
    )
    ag.aggregate(clients, X_val, y_val)
    ag.build_global_rf(clients[0].rf._classes)
    t_agg = time.perf_counter() - t_agg0
    m_global = ag.evaluate(X_test, y_test)
    global_rf = ag.global_rf
    acc_fed = float(m_global["accuracy"])
    lat_fed = _time_predict(global_rf, X_test) if global_rf is not None else float("nan")

    out["federated"] = {
        "n_clients": n_clients,
        "test_accuracy": acc_fed,
        "client_train_time_s": t_fed_local,
        "aggregate_time_s": t_agg,
        "predict_latency_s": lat_fed,
        "predict_ms_per_sample": 1000.0 * lat_fed / n_test,
    }
    return out


def _print_results(res: Dict[str, Any]) -> None:
    print("Public dataset:", res["dataset"])
    print(" ", res["description"])
    sp = res["split"]
    print(
        f"  rows total={res['n_rows']}; "
        f"train/val/test={sp['n_train']}/{sp['n_val']}/{sp['n_test']}; "
        f"features={sp['n_features']}"
    )
    c = res["central"]
    print("\n--- Central (single) RF ---")
    print(f"  test accuracy     : {c['test_accuracy']:.4f}")
    print(f"  fit time          : {c['fit_time_s']:.3f} s")
    print(f"  predict (full test): {c['predict_latency_s']*1000:.2f} ms  "
          f"({c['predict_ms_per_sample']:.3f} ms / sample)")

    f = res["federated"]
    print(f"\n--- Federated ({f['n_clients']} clients) ---")
    print(f"  test accuracy     : {f['test_accuracy']:.4f}")
    print(f"  client train (sum): {f['client_train_time_s']:.3f} s")
    print(f"  aggregate+build   : {f['aggregate_time_s']:.3f} s")
    print(f"  predict (full test): {f['predict_latency_s']*1000:.2f} ms  "
          f"({f['predict_ms_per_sample']:.3f} ms / sample)")

    print(
        "\n(Latency is one forward pass on the full test set; "
        "ms/sample divides total time by the number of test rows.)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark accuracy and prediction latency on UCI breast cancer data.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="smaller models for smoke testing / CI (faster, slightly noisier accuracy).",
    )
    args = parser.parse_args()
    n_est = 20 if args.quick else 64
    n_cl = 2 if args.quick else 3
    n_tpc = 6 if args.quick else 20
    res = run_benchmark(
        n_estimators=n_est,
        n_clients=n_cl,
        n_trees_per_client=n_tpc,
    )
    _print_results(res)


if __name__ == "__main__":
    main()
