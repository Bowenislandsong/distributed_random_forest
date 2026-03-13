## Installation

### End users

```bash
pip install distributed-random-forest
```

### Contributors

```bash
git clone https://github.com/Bowenislandsong/distributed_random_forest
cd distributed_random_forest
python -m pip install -e ".[dev,docs]"
```

## First Federated Run

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

## Local Quality Checks

```bash
make test
make lint
make docs
make build
```

If you do not use `make`, the equivalent commands are:

```bash
python -m pytest tests -q
python -m ruff check .
python -m mkdocs build --strict
python -m build
```

## Differential Privacy

Differential privacy is optional. The built-in DP mode works without extra packages:

```python
model = FederatedRandomForest(
    n_clients=5,
    rf_params={"n_estimators": 16, "random_state": 7},
    use_differential_privacy=True,
    epsilon=2.0,
)
```

If you want external privacy tooling as well, install the optional extra:

```bash
python -m pip install -e ".[privacy]"
```

## Reports

Every orchestrated run can export a JSON report:

```python
model.export_report("artifacts/federated-run.json")
```

That report includes:

- client sample counts and training metrics
- partition summaries
- evaluated aggregation strategies
- validation and final test metrics
