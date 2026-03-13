# Distributed Random Forest

[![PyPI version](https://img.shields.io/pypi/v/distributed-random-forest)](https://pypi.org/project/distributed-random-forest/)
[![Python Versions](https://img.shields.io/pypi/pyversions/distributed-random-forest)](https://pypi.org/project/distributed-random-forest/)
[![Tests](https://github.com/Bowenislandsong/distributed_random_forrest/actions/workflows/ci.yml/badge.svg)](https://github.com/Bowenislandsong/distributed_random_forrest/actions/workflows/ci.yml)
[![Docs](https://github.com/Bowenislandsong/distributed_random_forrest/actions/workflows/docs.yml/badge.svg)](https://github.com/Bowenislandsong/distributed_random_forrest/actions/workflows/docs.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Distributed Random Forest is a Python package for federated and distributed tree
ensembles. It is designed for people who need more than "train a few local
forests and concatenate the trees":

- realistic client partitioning, including non-IID splits
- multiple aggregation strategies instead of one hard-coded merge rule
- parallel client training backends
- structured reports for benchmarking and audits
- runnable examples, docs, tests, CI, and PyPI packaging

This README is used on both GitHub and PyPI. The top half is package-user
focused so the PyPI page answers "why should I install this?" before it dives
into repo internals.

## Why This Implementation Is Different

Most distributed RF repositories are really experiment scripts. This one is a
reusable package with a benchmarkable orchestration layer.

| Area | Typical paper-style repo | This implementation |
| --- | --- | --- |
| Distributed workflow | manual scripts | `FederatedRandomForest` orchestration API |
| Client heterogeneity | mostly uniform splits | `uniform`, `stratified`, `feature`, `sized`, `dirichlet`, `label_skew` |
| Tree aggregation | one or two ranking rules | classic paper rules plus balanced, proportional, threshold, and auto search |
| Execution | sequential only | sequential, thread, and process backends |
| Reporting | print statements | JSON run reports with partition, client, and strategy summaries |
| Privacy | often omitted | built-in DP RF support for experimentation and comparison |
| Packaging | code snapshot | PyPI package, CLI, docs site, CI, release workflow |

## What Is Special About This Package

The main differentiator is that the package separates three concerns cleanly:

1. `models`
   Local RF and DP-RF training.
2. `federation`
   Tree ranking, voting, and aggregation.
3. `distributed`
   Partitioning, parallel client training, strategy search, and report export.

That makes it useful for both:

- researchers comparing aggregation strategies under non-IID data
- engineers who want a callable library instead of notebook-only code

## Good Use Cases

- Network intrusion detection where traffic distributions differ by site.
- Fraud or risk scoring across branches, regions, or subsidiaries.
- Edge/IoT classification where each site owns a small, skewed local dataset.
- Privacy-sensitive health or security workflows that need federated baselines.
- Benchmarking how aggregation strategies behave under controlled heterogeneity.

## Install

```bash
pip install distributed-random-forest
```

From source:

```bash
git clone https://github.com/Bowenislandsong/distributed_random_forrest
cd distributed_random_forrest
python -m pip install -e ".[dev,docs]"
```

## Quick Examples

### 1. End-to-end federated training

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from distributed_random_forest import FederatedRandomForest

X, y = make_classification(
    n_samples=1200,
    n_features=20,
    n_classes=3,
    n_informative=10,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = FederatedRandomForest(
    n_clients=4,
    rf_params={"n_estimators": 24, "random_state": 42, "voting": "weighted"},
    partition_strategy="dirichlet",
    partition_kwargs={"alpha": 0.8},
    aggregation_strategy="auto",
    execution_backend="thread",
    max_workers=4,
    random_state=42,
)

model.fit(X_train, y_train)
metrics = model.evaluate(X_test, y_test)
print(model.selected_strategy)
print(metrics)
```

### 2. Quick CLI smoke test

```bash
drf-quickstart --clients 4 --partition-strategy dirichlet --backend thread
```

### 3. Differential privacy baseline

```python
from distributed_random_forest import FederatedRandomForest

model = FederatedRandomForest(
    n_clients=5,
    rf_params={"n_estimators": 20, "random_state": 13},
    partition_strategy="stratified",
    aggregation_strategy="top_k_global_balanced_accuracy",
    use_differential_privacy=True,
    epsilon=10.0,
    random_state=13,
)
```

More runnable examples:

- [basic_federated_training.py](https://github.com/Bowenislandsong/distributed_random_forrest/blob/main/examples/basic_federated_training.py)
- [non_iid_dirichlet.py](https://github.com/Bowenislandsong/distributed_random_forrest/blob/main/examples/non_iid_dirichlet.py)
- [dp_enterprise_workflow.py](https://github.com/Bowenislandsong/distributed_random_forrest/blob/main/examples/dp_enterprise_workflow.py)
- [performance_benchmark.py](https://github.com/Bowenislandsong/distributed_random_forrest/blob/main/examples/performance_benchmark.py)

## Performance Snapshot

The table below comes from a local single-run benchmark on a synthetic
multiclass dataset with 6,000 samples, 40 features, and 4 classes. It is meant
to show the relative behavior of this implementation, not to claim a universal
leaderboard. You can reproduce it with
[examples/performance_benchmark.py](https://github.com/Bowenislandsong/distributed_random_forrest/blob/main/examples/performance_benchmark.py).

| Scenario | Accuracy | Balanced Acc. | Weighted Acc. | F1 | Time (s) | Strategy |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Centralized RF | 0.8642 | 0.8641 | 0.7467 | 0.8640 | 0.35 | n/a |
| Federated uniform | 0.7842 | 0.7840 | 0.6148 | 0.7833 | 1.44 | `proportional_weighted_accuracy` |
| Federated dirichlet | 0.7642 | 0.7642 | 0.5840 | 0.7599 | 1.42 | `proportional_weighted_accuracy` |
| Federated dirichlet + DP | 0.5125 | 0.5129 | 0.2629 | 0.4950 | 0.74 | `top_k_global_balanced_accuracy` |

What this shows:

- the package preserves a large share of centralized accuracy under realistic federated splits
- non-IID partitions are supported as first-class workflows, not afterthought scripts
- DP support is available, with the expected privacy/utility tradeoff clearly visible

## Supported Distributed RF Patterns

### Partitioning

- `uniform`
- `stratified`
- `feature`
- `sized`
- `dirichlet`
- `label_skew`

### Aggregation

- `rf_s_dts_a`
- `rf_s_dts_wa`
- `rf_s_dts_a_all`
- `rf_s_dts_wa_all`
- `top_k_global_balanced_accuracy`
- `top_k_global_f1`
- `proportional_weighted_accuracy`
- `proportional_balanced_accuracy`
- `threshold_weighted_accuracy`
- automatic strategy search through `FederatedRandomForest(aggregation_strategy="auto")`

## What You Get In The Package

- `RandomForest` and `DPRandomForest` for local models
- `ClientRF` and `DPClientRF` for client-scoped training and evaluation
- `FederatedAggregator` for explicit tree-selection experiments
- `FederatedRandomForest` for end-to-end orchestration
- partitioning utilities and JSON run report export
- a CLI, examples, tests, docs, and release automation

## Documentation

- Docs site: [bowenislandsong.github.io/distributed_random_forrest](https://bowenislandsong.github.io/distributed_random_forrest/)
- Docs source: [docs/](https://github.com/Bowenislandsong/distributed_random_forrest/tree/main/docs)

## Development

```bash
make lint
make test
make docs
make build
```

## CI/CD

The repository includes workflows for:

- linting and tests on pushes and pull requests
- package build validation
- GitHub Pages deployment
- PyPI publishing on GitHub releases

## Project Structure

```text
distributed_random_forest/
  distributed/   # orchestration and partitioning
  experiments/   # experiment pipelines
  federation/    # aggregation and voting
  models/        # local RF implementations
docs/            # GitHub Pages / MkDocs site
examples/        # runnable use cases and benchmark scripts
tests/           # regression and end-to-end coverage
```
