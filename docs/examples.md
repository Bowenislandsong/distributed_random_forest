# Code examples

This page lists **runnable** patterns: the snippets stand alone (with imports), and a **UCI** benchmark is available as a script under `examples/`.

| Topic | When to use |
|--------|----------------|
| [Single-site RF](#train-one-random-forest) | One machine, full data. |
| [Public UCI data + accuracy & latency](#public-dataset-uci-breast-cancer) | Reproducible numbers on a [pandas](https://pandas.pydata.org/)-friendly table. |
| [Federated + aggregation](#federated-learning) | Many clients, merge trees with `FederatedAggregator`. |
| [Differential privacy](#differential-privacy) | `DPRandomForest` / `DPClientRF` with a privacy budget. |
| [Compare merge strategies (EXP3)](#compare-aggregation-strategies) | `run_exp3_federated_aggregation` helper. |

Run any fragment from the **repository root** after `pip install -e .` (or use the same `sys.path` trick as in `examples/benchmark_public_dataset.py`).

## Train one Random Forest

```python
from distributed_random_forest import RandomForest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=1000, n_features=20, n_classes=3, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForest(
    n_estimators=100,
    criterion="gini",
    voting="simple",
    random_state=42,
)
rf.fit(X_train, y_train)
print(f"Accuracy: {rf.score(X_test, y_test):.4f}")
```

## Public dataset: UCI breast cancer

**Dataset:** *Wisconsin Diagnostic Breast Cancer* (UCI), loaded with scikit-learn’s `as_frame` path, exposed as
`load_breast_cancer_bench()` on a stratified **train / validation / test** split. You get a **NumPy** matrix for the models and an optional `pandas` `DataFrame` for inspection when `as_frame=True`.

### Minimal train + test accuracy

```python
from time import perf_counter

from distributed_random_forest import RandomForest
from distributed_random_forest.datasets import load_breast_cancer_bench

split = load_breast_cancer_bench(as_frame=True, random_state=42)
assert split.data_frame is not None  # pandas: 569×31 with a "target" column

X_train, y_train = split.X_train, split.y_train
X_val, y_val = split.X_val, split.y_val
X_test, y_test = split.X_test, split.y_test

t0 = perf_counter()
rf = RandomForest(n_estimators=64, random_state=42)
rf.fit(X_train, y_train, X_val, y_val)  # weighted voting uses val when fitted with X_val, y_val
fit_s = perf_counter() - t0
acc = rf.score(X_test, y_test)

t1 = perf_counter()
_ = rf.predict(X_test)  # one forward pass on the full test set
lat_s = perf_counter() - t1
per_ms = 1000 * lat_s / len(y_test)

print(f"test accuracy: {acc:.4f}  |  fit: {fit_s:.3f}s  |  predict: {1000*lat_s:.1f} ms full batch ({per_ms:.3f} ms / sample)")
```

### Full benchmark (central + federated + table output)

The repository ships a small CLI that prints **test accuracy** and **prediction latency** (full batch and per test row) for a **single-site** model and a **federated** merge:

```bash
python examples/benchmark_public_dataset.py
python examples/benchmark_public_dataset.py --quick  # smaller forests for a fast smoke test
```

This is the same path exercised by the **performance** and **example smoke** tests in CI.

## Federated learning

**Flow:** (1) load a split, (2) partition the training set with `partition_uniform_random`, (3) one `ClientRF` per client, (4) `FederatedAggregator` to merge, (5) `evaluate` on the test set.

```python
from distributed_random_forest import ClientRF, FederatedAggregator
from distributed_random_forest.datasets import load_breast_cancer_bench
from distributed_random_forest.experiments.exp2_clients import partition_uniform_random

split = load_breast_cancer_bench(random_state=42)
X_train, y_train = split.X_train, split.y_train
X_val, y_val = split.X_val, split.y_val
X_test, y_test = split.X_test, split.y_test

n_clients = 3
partitions = partition_uniform_random(
    X_train, y_train, n_clients=n_clients, random_state=0
)
rf_params = {"n_estimators": 40, "random_state": 1}

clients = []
for i, (Xc, yc) in enumerate(partitions):
    c = ClientRF(client_id=i, rf_params=rf_params)
    c.train(Xc, yc)
    clients.append(c)

ag = FederatedAggregator(strategy="rf_s_dts_a", n_trees_per_client=12)
ag.aggregate(clients, X_val, y_val)
ag.build_global_rf(clients[0].rf._classes)
metrics = ag.evaluate(X_test, y_test)
print(f"Global test accuracy: {metrics['accuracy']:.4f}")
```

## Differential privacy

`DPRandomForest` and `DPClientRF` add noise consistent with a chosen **ε** (and mechanism such as **Laplace**).

```python
from distributed_random_forest import DPRandomForest, DPClientRF
from distributed_random_forest.datasets import load_breast_cancer_bench

split = load_breast_cancer_bench(random_state=0)

dp_rf = DPRandomForest(
    n_estimators=40,
    epsilon=2.0,
    dp_mechanism="laplace",
    random_state=0,
)
dp_rf.fit(split.X_train, split.y_train, split.X_val, split.y_val)
print("ε ≈", dp_rf.get_privacy_budget())
print("Test accuracy (DP, single):", float(dp_rf.score(split.X_test, split.y_test)))

# One federated client (illustration only: production uses several clients)
xc, yc = split.X_train[:200], split.y_train[:200]
dpc = DPClientRF(
    client_id=0, epsilon=2.0, rf_params={"n_estimators": 20, "random_state": 0}
)
dpc.train(xc, yc, split.X_val, split.y_val)
```

## Compare aggregation strategies

`run_exp3_federated_aggregation` ranks all four merge strategies and returns the best on your validation split (mirrors the EXP3 driver).

```python
from distributed_random_forest import ClientRF
from distributed_random_forest.datasets import load_breast_cancer_bench
from distributed_random_forest.experiments.exp2_clients import partition_uniform_random
from distributed_random_forest.experiments.exp3_global_rf import run_exp3_federated_aggregation

split = load_breast_cancer_bench(random_state=0)
X_train, y_train = split.X_train, split.y_train

clients = []
for i, (Xc, yc) in enumerate(
    partition_uniform_random(X_train, y_train, n_clients=2, random_state=1)
):
    c = ClientRF(client_id=i, rf_params={"n_estimators": 24, "random_state": 0})
    c.train(Xc, yc)
    clients.append(c)

results = run_exp3_federated_aggregation(
    client_rfs=clients,
    X_val=split.X_val,
    y_val=split.y_val,
    X_test=split.X_test,
    y_test=split.y_test,
    n_trees_per_client=8,
    n_total_trees=16,
    verbose=False,
)
print("Best strategy:", results["best_strategy"])
print("Best accuracy:", f"{results['best_accuracy']:.4f}")
```

## Tests

- **Unit:** `tests/test_datasets.py` (loader invariants)
- **Performance / accuracy bounds:** `tests/test_performance.py` (marked `performance`, uses the same UCI holdout)
- **E2E on real data:** `tests/test_e2e_public_dataset.py` (EXP1–4-style runners on the public split)
- **Example script smoke:** `tests/test_examples_run.py` (runs `examples/benchmark_public_dataset.py --quick` in a subprocess)

See [Getting started](getting-started.md) for the full `pytest` command line.
