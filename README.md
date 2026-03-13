# Distributed Random Forest

[![PyPI version](https://img.shields.io/pypi/v/distributed-random-forest)](https://pypi.org/project/distributed-random-forest/)
[![Python Versions](https://img.shields.io/pypi/pyversions/distributed-random-forest)](https://pypi.org/project/distributed-random-forest/)
[![Tests](https://github.com/Bowenislandsong/distributed_random_forest/actions/workflows/ci.yml/badge.svg)](https://github.com/Bowenislandsong/distributed_random_forest/actions/workflows/ci.yml)
[![Docs](https://github.com/Bowenislandsong/distributed_random_forest/actions/workflows/docs.yml/badge.svg)](https://github.com/Bowenislandsong/distributed_random_forest/actions/workflows/docs.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Distributed Random Forest is a Python toolkit for federated and distributed tree
ensembles. It started from a paper-inspired baseline and has been expanded into
a more complete platform with:

- federated client orchestration
- parallel training backends
- non-IID partitioning strategies
- richer aggregation strategies
- optional differential privacy
- structured JSON reports
- PyPI packaging
- CI, release, and docs automation
- GitHub Pages documentation

## What Makes It Better

The original implementation already supported the paper-style workflow of:

1. train client forests
2. rank trees
3. merge them into a global forest

This repo now also supports the broader engineering work around that core:

- `FederatedRandomForest` for end-to-end orchestration
- `uniform`, `stratified`, `feature`, `sized`, `dirichlet`, and `label_skew` partitioning
- legacy aggregation strategies plus `balanced_accuracy`, `proportional`, and `threshold` selectors
- sequential, thread, and process execution backends
- client summaries and exportable run reports
- documentation site generation with MkDocs Material

## Installation

### Users

```bash
pip install distributed-random-forest
```

### Contributors

```bash
git clone https://github.com/Bowenislandsong/distributed_random_forest
cd distributed_random_forest
python -m pip install -e ".[dev,docs]"
```

## Quickstart

### CLI

```bash
drf-quickstart --clients 4 --partition-strategy dirichlet --backend thread
```

### Python API

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
    partition_kwargs={"alpha": 0.5},
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
- `auto` strategy search via `FederatedRandomForest`

## Differential Privacy

Built-in DP training is available through `DPRandomForest`, `DPClientRF`, and
`FederatedRandomForest(..., use_differential_privacy=True, epsilon=...)`.

## Examples

- [`examples/basic_federated_training.py`](examples/basic_federated_training.py)
- [`examples/non_iid_dirichlet.py`](examples/non_iid_dirichlet.py)
- [`examples/dp_enterprise_workflow.py`](examples/dp_enterprise_workflow.py)

## Documentation

- GitHub Pages site: [bowenislandsong.github.io/distributed_random_forest](https://bowenislandsong.github.io/distributed_random_forest/)
- Docs source: [`docs/`](docs/)

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
examples/        # runnable use cases
tests/           # regression and end-to-end tests
```

## Status

The test suite currently covers the legacy APIs and the new orchestration,
partitioning, aggregation, and string-label behaviors.
