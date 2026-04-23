# Distributed Random Forest

[![PyPI version](https://img.shields.io/pypi/v/distributed-random-forest)](https://pypi.org/project/distributed-random-forest/)
[![Python Versions](https://img.shields.io/pypi/pyversions/distributed-random-forest)](https://pypi.org/project/distributed-random-forest/)
[![CI](https://github.com/Bowenislandsong/distributed_random_forest/actions/workflows/ci.yml/badge.svg)](https://github.com/Bowenislandsong/distributed_random_forest/actions/workflows/ci.yml)
[![Docs](https://github.com/Bowenislandsong/distributed_random_forest/actions/workflows/docs.yml/badge.svg)](https://github.com/Bowenislandsong/distributed_random_forest/actions/workflows/docs.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/distributed-random-forest?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/distributed-random-forest)

Federated and distributed **Random Forest** training: multiple clients each learn a local forest, you merge **decision trees** into a **global** model, and you can add **differential privacy** (DP) at the client. The design is inspired by research on *Random Forest with Differential Privacy in Federated Learning*; this codebase is general-purpose and ships as a **reusable package** (not just experiment scripts).

## At a glance

- Train RFs on many clients; aggregate with **tree-ranking strategies** (per-client, global, proportional, threshold, and more).
- **Gini** or **entropy** splits; **simple** or **weighted** voting; parallel **`n_jobs`** for scoring and merged prediction.
- **DP** random forests and federated DP vs non-DP comparisons.
- **CRF orchestration** (`FederatedRandomForest`), **CLI** (`drf-quickstart`), **scripts** for EXP 1–4-style pipelines, **docs**, and **CI**.

## Creator

Maintained by
[Bowen Song](https://bowenislandsong.github.io/#/personal) (USC Viterbi): health AI, federated learning, explainable AI, and scalable ML systems.

- Site: [bowenislandsong.github.io/#/personal](https://bowenislandsong.github.io/#/personal)
- CV: [Bowen_Song_Resume.pdf](https://bowenislandsong.github.io/img/resume/Bowen_Song_Resume.pdf)
- ORCID: [0000-0002-5071-3880](https://orcid.org/0000-0002-5071-3880)

## Why this implementation is different

| Area | Typical paper-style repo | This implementation |
| --- | --- | --- |
| Distributed workflow | manual scripts | `FederatedRandomForest` orchestration API |
| Client heterogeneity | mostly uniform splits | `uniform`, `stratified`, `feature`, `sized`, `dirichlet`, `label_skew` |
| Tree aggregation | one or two rules | paper baselines plus balanced, proportional, threshold, and `auto` search |
| Execution | sequential only | sequential, thread, and process backends |
| Parallelism | ad hoc | `n_jobs` for aggregation scoring and merged RF (parity-tested vs sequential) |
| Reporting | print statements | JSON run reports with partition, client, and strategy summaries |
| Privacy | often omitted | built-in DP RF for experimentation and comparison |
| Packaging | code snapshot | PyPI, CLI, docs, CI, release workflow |

## Good use cases

- Network/security analytics with site-specific data distributions.
- Fraud or risk scoring across branches or regions.
- Edge/IoT with small, skewed local datasets.
- Health or privacy-sensitive federated baselines.
- Benchmarking aggregation strategies under controlled heterogeneity.

## Documentation

- **Online:** [GitHub Pages](https://bowenislandsong.github.io/distributed_random_forest/) — ensure Pages is set to “GitHub Actions” as the source if the site is not live.
- **Local:** `pip install -e ".[docs]"` and `mkdocs serve`.

**Contents:** [concepts](https://bowenislandsong.github.io/distributed_random_forest/concepts/) · [patterns](https://bowenislandsong.github.io/distributed_random_forest/patterns/) · [pipeline](https://bowenislandsong.github.io/distributed_random_forest/pipeline/) · [getting started](https://bowenislandsong.github.io/distributed_random_forest/getting-started/) · [examples](https://bowenislandsong.github.io/distributed_random_forest/examples/) · [repository](https://bowenislandsong.github.io/distributed_random_forest/repository/) · [citing](https://bowenislandsong.github.io/distributed_random_forest/citing/)

## Install

```bash
pip install distributed-random-forest
```

From source:

```bash
git clone https://github.com/Bowenislandsong/distributed_random_forest.git
cd distributed_random_forest
python -m pip install -e ".[dev,docs]"
```

## Quick examples

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
    X, y, test_size=0.2, random_state=42, stratify=y,
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

### 2. Single-site `RandomForest`

```python
from distributed_random_forest import RandomForest
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=1000, n_features=20, n_classes=3, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForest(
    n_estimators=100, criterion="gini", voting="simple", random_state=42
)
rf.fit(X_train, y_train)
print(f"Accuracy: {rf.score(X_test, y_test):.4f}")
```

More snippets (federation, DP, EXP3) are in the [examples](https://bowenislandsong.github.io/distributed_random_forest/examples/) page.

### 3. CLI

```bash
drf-quickstart --clients 4 --partition-strategy dirichlet --backend thread
```

### 4. Differential privacy (orchestrated)

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

## Runnable scripts in the repo

- [basic_federated_training.py](https://github.com/Bowenislandsong/distributed_random_forest/blob/main/examples/basic_federated_training.py)
- [non_iid_dirichlet.py](https://github.com/Bowenislandsong/distributed_random_forest/blob/main/examples/non_iid_dirichlet.py)
- [dp_enterprise_workflow.py](https://github.com/Bowenislandsong/distributed_random_forest/blob/main/examples/dp_enterprise_workflow.py)
- [performance_benchmark.py](https://github.com/Bowenislandsong/distributed_random_forest/blob/main/examples/performance_benchmark.py)

## Experiment drivers

```bash
python run_exp1_hparams.py
python run_exp2_clients.py
python run_exp3_federation.py
python run_exp4_dp_federation.py
```

**UCI benchmark** (Wisconsin breast cancer via scikit-learn): test accuracy and prediction latency (central vs federated):

```bash
python examples/benchmark_public_dataset.py
python examples/benchmark_public_dataset.py --quick
```

## Performance snapshot

Reproduce with
[examples/performance_benchmark.py](https://github.com/Bowenislandsong/distributed_random_forest/blob/main/examples/performance_benchmark.py).

| Scenario | Accuracy | Balanced Acc. | Weighted Acc. | F1 | Time (s) | Strategy |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Centralized RF | 0.8642 | 0.8641 | 0.7467 | 0.8640 | 0.35 | n/a |
| Federated uniform | 0.7842 | 0.7840 | 0.6148 | 0.7833 | 1.44 | `proportional_weighted_accuracy` |
| Federated dirichlet | 0.7642 | 0.7642 | 0.5840 | 0.7599 | 1.42 | `proportional_weighted_accuracy` |
| Federated dirichlet + DP | 0.5125 | 0.5129 | 0.2629 | 0.4950 | 0.74 | `top_k_global_balanced_accuracy` |

## Supported distributed RF patterns

**Partitioning:** `uniform`, `stratified`, `feature`, `sized`, `dirichlet`, `label_skew`.

**Aggregation (high level):** `rf_s_dts_a`, `rf_s_dts_wa`, `rf_s_dts_a_all`, `rf_s_dts_wa_all`, `top_k_global_balanced_accuracy`, `top_k_global_f1`, `proportional_weighted_accuracy`, `proportional_balanced_accuracy`, `threshold_weighted_accuracy`, and `aggregation_strategy="auto"` in `FederatedRandomForest`.

## What you get in the package

- `RandomForest` and `DPRandomForest` for local models
- `ClientRF` and `DPClientRF` for client-scoped training
- `FederatedAggregator` for explicit tree-selection experiments
- `FederatedRandomForest` for end-to-end orchestration
- partitioning utilities, JSON run report export, CLI, tests, and docs

## Development

```bash
make lint
make test
make docs
make build
```

(`python -m pytest`, `python -m ruff check .`, etc., if you do not use `make`.)

## CI/CD

Workflows cover lint, tests, package build, GitHub Pages (docs), and optional PyPI publishing on releases.

## Project structure

```text
distributed_random_forest/
  distributed/   # orchestration and partitioning
  experiments/   # experiment pipelines
  federation/    # aggregation and voting
  models/        # local RF implementations
docs/            # MkDocs site
examples/        # benchmarks and use cases
tests/           # regression and e2e coverage
```

## Run tests

```bash
pytest tests/ -v
pytest tests/ --cov=distributed_random_forest
```

## Cite

BibTeX and APA: [License & citation](https://bowenislandsong.github.io/distributed_random_forest/citing/) (or `docs/citing.md` in the repo).

## License

Apache License 2.0 — see [LICENSE](LICENSE).
