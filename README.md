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

```bash
git clone <your-repo>
cd distributed_random_forrest
pip install -r requirements.txt
```

### Run Experiments

```bash
python run_exp1_hparams.py
python run_exp2_clients.py
python run_exp3_federation.py
python run_exp4_dp_federation.py
```

---

## 7. Repository Structure

```
distributed_random_forrest/
│
├── data/                     # Raw and processed datasets
├── models/
│   ├── random_forest.py      # Core RF implementation
│   ├── dp_rf.py              # Differentially private RF
│   └── tree_utils.py
├── federation/
│   ├── aggregator.py         # DT aggregation strategies (A, WA, All)
│   └── voting.py             # SV, WV methods
│
├── experiments/
│   ├── exp1_hparams.py
│   ├── exp2_clients.py
│   ├── exp3_global_rf.py
│   └── exp4_dp_rf.py
│
└── README.md                 # You are here
```

---

## License

This project is provided as-is for educational and research purposes.
