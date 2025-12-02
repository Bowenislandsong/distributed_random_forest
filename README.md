# Distributed Random Forest with Differential Privacy

### Enhancement Report & README

This repository implements a **Distributed / Federated Random Forest (RF)** framework inspired by:

> **"Random Forest with Differential Privacy in Federated Learning Framework for Network Attack Detection and Classification."**

The implementation includes:

* RF training on multiple distributed clients
* Aggregation of decision trees into a global RF
* Differential Privacy (DP) support
* Extensive evaluation pipelines and hyperparameter selection

---

## 1. Core Ideas

### **Splitting Rules**

We support the two classical RF impurity measures:

* **Gini index** — favors isolating the largest homogeneous class.
* **Entropy** — aims to minimize within-node class diversity.

### **Ensemble Voting Methods**

For local RF inference:

* **Simple Voting (SV):** majority vote across decision trees.
* **Weighted Voting (WV):** majority vote weighted by each DT's class-specific accuracy.

---

## 2. Federated Aggregation of Trees

After each client trains its own RF, decision trees (DTs) are merged into a global RF using four strategies:

### **Sorting DTs Within Each RF**

1. **RF_S_DTs_A** — Sort DTs by validation accuracy within each client RF and select the top performers.
2. **RF_S_DTs_WA** — Same as above, but sort by *weighted accuracy* (WA).

### **Sorting DTs Across All Clients**

3. **RF_S_DTs_A_All** — Collect all DTs from all clients, sort globally by accuracy, select best N.
4. **RF_S_DTs_WA_All** — Global sorting of all DTs by weighted accuracy.

These merging strategies allow the global RF to retain the strongest trees from heterogeneous local models.

---

## 3. Evaluation Metrics

### **Accuracy (A)**

Overall DT accuracy on the validation set.

### **Weighted Accuracy (WA)**

DT accuracy × (mean per-class accuracy).
Prioritizes trees that perform consistently across multiple classes.

### **Other metrics**

* **F1 Score** (macro or weighted depending on experiment)
* **Client-to-global performance gap**
* **DP degradation curves**

---

## 4. Experimental Pipeline

### **EXP 1 — RF Hyperparameter Selection**

Performed *before* federated splitting.
Grid search over:

* Number of trees (odd numbers 1–100)
* Splitting rule (gini, entropy)
* Ensemble rule (SV, WV)

The best configuration is used for all remaining experiments.

---

### **EXP 2 — Independent RFs Per Client**

Each client trains RFs independently using the best configuration from EXP 1.

Three data-partitioning strategies are evaluated:

#### **EXP 2.1 — Feature-based Partitioning**

Subsets created based on a specific feature criterion.
Testing:

* Only on the client's own subset
* On the full global test set

#### **EXP 2.2 — Uniform Random Partitioning**

Clients receive equal amounts of random samples.
Testing on the full test set.

#### **EXP 2.3 — Random Partitioning with EXP 2.1 Sample Counts**

Mimics the subset sizes from EXP 2.1 but randomizes the samples.
Testing on the full test set.

---

### **EXP 3 — Global RF from Federated Aggregation**

Independent client RFs are merged using the 4 strategies:

* RF_S_DTs_A
* RF_S_DTs_WA
* RF_S_DTs_A_All
* RF_S_DTs_WA_All

The global RF is evaluated on the full test set and compared to:

* Independent RF performance
* Best‐client performance
* Baseline centralized RF (if provided)

---

### **EXP 4 — Federated RF with Differential Privacy**

Each client trains a **DP-Random Forest** using per-client differential privacy.

Tested ε values:

* **0.1, 0.5, 1, 5**

Pipeline:

1. Train DP-RF per client
2. Evaluate each DP-RF on the full test set
3. Merge using the best aggregation strategy determined in EXP 3
4. Compare:

   * DP-client RF
   * Federated DP Global RF
   * Non-DP Global RF

---

## 5. Summary of Enhancements in This Implementation

* Clean modular design of RF, client trainers, and federated aggregator
* Support for **Gini**, **Entropy**, **SV**, **WV**
* Four global aggregation algorithms implemented
* Weighted accuracy for tree ranking
* Full experiment pipeline (EXP 1 → EXP 4) implemented in code
* Differential privacy integrated at client training level
* Extensible API for additional DP mechanisms (Gaussian, Laplace, tree-level clipping, etc.)

---

## 6. Getting Started

### Installation

#### Install from PyPI (coming soon)

```bash
pip install distributed-random-forest
```

#### Install from source (development mode)

```bash
git clone https://github.com/Bowenislandsong/distributed_random_forrest
cd distributed_random_forrest
pip install -e .
```

For development with test dependencies:

```bash
pip install -e ".[dev]"
```

### Run Experiments

```bash
python run_exp1_hparams.py    # Hyperparameter selection
python run_exp2_clients.py    # Independent client training
python run_exp3_federation.py # Federated aggregation
python run_exp4_dp_federation.py # DP federation
```

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=models --cov=federation --cov=experiments

# Run specific test suites
pytest tests/test_tree_utils.py -v      # Unit tests for utilities
pytest tests/test_random_forest.py -v   # Unit tests for RF
pytest tests/test_dp_rf.py -v           # Unit tests for DP-RF
pytest tests/test_aggregator.py -v      # Unit tests for aggregation
pytest tests/test_e2e.py -v             # End-to-end tests
```

---

## 7. Usage Examples

### Basic Random Forest Training

```python
from models.random_forest import RandomForest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create sample data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RF with Gini criterion and simple voting
rf = RandomForest(n_estimators=100, criterion='gini', voting='simple', random_state=42)
rf.fit(X_train, y_train)

# Evaluate
accuracy = rf.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
```

### Federated Learning with Multiple Clients

```python
from models.random_forest import ClientRF
from federation.aggregator import FederatedAggregator
from experiments.exp2_clients import partition_uniform_random

# Partition data for 5 clients
partitions = partition_uniform_random(X_train, y_train, n_clients=5, random_state=42)

# Train RF on each client
clients = []
for i, (X_client, y_client) in enumerate(partitions):
    client = ClientRF(client_id=i, rf_params={'n_estimators': 20, 'random_state': i})
    client.train(X_client, y_client)
    clients.append(client)

# Aggregate trees using RF_S_DTs_A strategy
aggregator = FederatedAggregator(strategy='rf_s_dts_a', n_trees_per_client=10)
aggregator.aggregate(clients, X_val, y_val)

# Build and evaluate global RF
global_rf = aggregator.build_global_rf(clients[0].rf._classes)
metrics = aggregator.evaluate(X_test, y_test)
print(f"Global RF Accuracy: {metrics['accuracy']:.4f}")
```

### Differential Privacy Training

```python
from models.dp_rf import DPRandomForest, DPClientRF

# Train DP-RF with epsilon=1.0 (Laplace mechanism)
dp_rf = DPRandomForest(
    n_estimators=50,
    epsilon=1.0,
    dp_mechanism='laplace',
    random_state=42
)
dp_rf.fit(X_train, y_train)
print(f"Privacy budget: ε={dp_rf.get_privacy_budget()}")

# DP client for federated learning
dp_client = DPClientRF(client_id=0, epsilon=0.5, rf_params={'n_estimators': 20})
dp_client.train(X_client, y_client)
```

### Comparing Aggregation Strategies

```python
from experiments.exp3_global_rf import run_exp3_federated_aggregation

results = run_exp3_federated_aggregation(
    client_rfs=clients,
    X_val=X_val,
    y_val=y_val,
    X_test=X_test,
    y_test=y_test,
    n_trees_per_client=10,
    verbose=True
)

print(f"Best strategy: {results['best_strategy']}")
print(f"Best accuracy: {results['best_accuracy']:.4f}")
```

---

## 8. Repository Structure

```
distributed_random_forrest/
│
├── data/                     # Raw and processed datasets
├── models/
│   ├── random_forest.py      # Core RF implementation
│   ├── dp_rf.py              # Differentially private RF
│   └── tree_utils.py         # Utility functions for metrics
├── federation/
│   ├── aggregator.py         # DT aggregation strategies (A, WA, All)
│   └── voting.py             # SV, WV methods
├── experiments/
│   ├── exp1_hparams.py       # Hyperparameter selection
│   ├── exp2_clients.py       # Independent client training
│   ├── exp3_global_rf.py     # Federated aggregation
│   └── exp4_dp_rf.py         # DP federation
├── tests/
│   ├── test_tree_utils.py    # Unit tests for utilities
│   ├── test_random_forest.py # Unit tests for RF
│   ├── test_dp_rf.py         # Unit tests for DP-RF
│   ├── test_voting.py        # Unit tests for voting
│   ├── test_aggregator.py    # Unit tests for aggregation
│   └── test_e2e.py           # End-to-end tests
├── .github/workflows/
│   └── tests.yml             # CI/CD workflow
├── requirements.txt          # Python dependencies
└── README.md                 # You are here
```

---

## License

This project is provided as-is for educational and research purposes.
