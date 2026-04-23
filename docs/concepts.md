# Core concepts

For **how the training set is split across clients**, how **per-client** forests interact with **federated tree aggregation** (and DP), and when to use each **partition** or **merge** strategy, see [Supported distributed RF patterns](patterns.md).

## Splitting rules (node impurity)

| Criterion | Role |
|-----------|------|
| **Gini** | Tends to isolate the largest pure class in a split. |
| **Entropy** | Reduces mixed-class diversity in child nodes. |

## Local ensemble voting

After each tree predicts a class, the forest combines votes:

| Method | Behavior |
|--------|------------|
| **Simple voting (SV)** | Plain majority over trees. |
| **Weighted voting (WV)** | Votes weighted by each tree’s class-wise accuracy. |

## Federated aggregation (merging trees)

Each client finishes with its own RF. You select and merge **decision trees (DTs)** into one global model. The library offers four named strategies (accuracy vs. weighted accuracy; per-client top-*K* vs. global top-*K*). Full descriptions, parameter names (`n_trees_per_client` vs. `n_total_trees`), and when to use each pattern are in [Supported distributed RF patterns](patterns.md) (this section is the short version).

| Strategy name | Ranks by | Tree budget |
|---------------|-----------|------------|
| **rf_s_dts_a** | Validation accuracy (A) | *N* **per client** |
| **rf_s_dts_wa** | Weighted accuracy (WA) | *N* **per client** |
| **rf_s_dts_a_all** | A on pooled trees | *K* **total** |
| **rf_s_dts_wa_all** | WA on pooled trees | *K* **total** |

## Metrics

| Symbol | Meaning |
|--------|--------|
| **Accuracy (A)** | A tree’s accuracy on the holdout/validation set. |
| **Weighted accuracy (WA)** | Accuracy times mean per-class accuracy; rewards balance across classes. |
| **F1** | Macro or weighted, depending on the experiment. |

Analysis may also report client-to-global gaps and **DP degradation** curves (utility vs. ε).

## Implementation highlights

* Modular **RF**, **client trainers**, and **FederatedAggregator**.
* Gini, entropy, SV, WV, and all four global merge strategies.
* Privacy hooks for mechanisms such as **Laplace** / **Gaussian** and future extensions (e.g. tree-level clipping).

For the scripted evaluation flow, see [Experiment pipeline](pipeline.md).
