# Supported distributed RF patterns

This page explains **how data is split across clients**, how **local forests** are trained, and how **trees are merged** into a single global model. The implementation lives mainly under `distributed_random_forest/experiments/` (for partitioning) and `distributed_random_forest/federation/aggregator.py` (for strategies).

For tree **splitting rules** (Gini / entropy) and **voting** (SV / WV), see [Core concepts](concepts.md). For the scripted EXP1–4 flow, see [Experiment pipeline](pipeline.md).

---

## 1. Data partitioning (federation layout)

*Partitioning* answers: *which rows of the training set does each client see?* The package supports three complementary patterns, matching EXP 2.1–2.3 in the pipeline.

### Uniform random (`uniform`)

**Idea:** Shuffle row indices, then split them into the requested number of clients so each shard is approximately the same size. Every client still sees the **full set of feature columns**; the split is on **samples** only.

**When it helps:** Baseline for **IID-style** simulation—clients differ only by which examples they hold, not by which features or regions of the input space. This is the default in `run_exp2_independent_clients(..., partitioning='uniform')` and the helper `partition_uniform_random` used in the docs examples.

**Code:** `partition_uniform_random`, or `run_exp2_2_uniform_partitioning` for the full EXP2 wrapper.

### Feature-based (`feature`)

**Idea:** Pick one feature index (e.g. a specific sensor or field). **Rows are grouped** by that feature: either by **discrete** values, or for continuous features by **quantile bins** when you pass a client count, so each client’s data live in a different “slice” of that dimension.

**When it helps:** Simulates **non-IID, vertically separated** conditions—each client’s distribution over *other* features can differ strongly because their samples come from different regions. Useful when you need to study **heterogeneous** clients, not just random subsampling.

**Code:** `partition_by_feature` with a chosen `feature_idx`, or `run_exp2_1_feature_partitioning` (EXP 2.1). You can tune `n_clients` to control the number of quantile bins in the continuous case.

### Size-matched random (`sized`)

**Idea:** You specify **target shard sizes** (e.g. the same list of counts you got from a feature-based run). A **random permutation** of training indices is drawn, and consecutive blocks of that size are assigned to each client. So the **client sample counts** are controlled, but **which** points go where is random—unlike feature partitioning, the split is not driven by a single feature’s value.

**When it helps:** Isolates the effect of **class imbalance or sample counts** (from EXP 2.1) from the effect of **feature-driven separation**. EXP 2.3 uses this to mirror EXP 2.1’s shard sizes with a random sample assignment, then test on the full test set for comparability with uniform splits.

**Code:** `partition_random_with_sizes`, or `run_exp2_3_sized_partitioning` with a `sizes` list.

---

## 2. Local training (one random forest per client)

After partitioning, each client `i` receives `(X_i, y_i)` and trains a **ClientRF**—a standard `RandomForest` in this library with a fixed hyperparameter set (ideally the best configuration from EXP 1).

**Validation inside the client:** For **weighted voting** (WV) across trees, each client can hold out part of its local data to score trees and set weights. The experiment driver uses a stratified `train_test_split` on each partition when you use `run_exp2_independent_clients`.

**Output:** A list of trained clients, each with its own set of **decision trees** and, internally, the class order needed later for fusion.

This step does **not** build a global model yet; it only produces **independent** forests to be combined in the next step (unless you stop at a best-client baseline).

---

## 3. Aggregation (building one global random forest)

*Aggregation* answers: *which trees from which clients are kept, and in what order, to form a single `RandomForest`?* The library never retrains a joint forest on all data; it **selects and concatenates** trees and wires them for prediction.

**Shared validation for ranking:** All strategies need a **common validation set** `X_val`, `y_val` (not owned by a single client) to score every candidate tree—accuracy (A) or weighted accuracy (WA) as in [Core concepts](concepts.md). That score drives sorting and trimming.

**Two families:**

| Family | What is ranked | API hint |
|--------|------------------|----------|
| **Per-client then merge** | Within each client, take the best *K* trees; concatenate. | `n_trees_per_client` |
| **Global pool** | Drop client boundaries, pool every tree, sort once, take best *K* total. | `n_total_trees` in the lower-level `aggregate_trees` / aggregators for “_All” strategies |

**The four built-in strategies (string names in `FederatedAggregator`):**

1. **`rf_s_dts_a` (RF_S_DTs_A)**  
   For each client, sort its trees by **validation accuracy (A)**, then keep the top *N* per client. Merges the chosen trees into the global set. Favors the **strongest local trees** under plain accuracy, with an equal per-client "quota" of trees.

2. **`rf_s_dts_wa` (RF_S_DTs_WA)**  
   Same as above, but sorting uses **weighted accuracy (WA)** so trees that do well **across classes** rank higher. Better when you care about **balanced** performance, not a single majority class.

3. **`rf_s_dts_a_all` (RF_S_DTs_A_All)**  
   Pool **all** trees from all clients, sort by **A**, then keep the best **K** **globally** (total tree budget, not per client). Can concentrate the global model on a few “star” clients if their trees score highest on the shared validation set.

4. **`rf_s_dts_wa_all` (RF_S_DTs_WA_All)**  
   Pool all trees, sort by **WA**, then keep the best **K** overall. Like (3), but with **WA** for a multi-class–friendly ranking.

**Choosing parameters:** For `rf_s_dts_a` and `rf_s_dts_wa`, set **`n_trees_per_client`**. For `rf_s_dts_a_all` and `rf_s_dts_wa_all`, set **`n_total_trees`** (or rely on the helper defaults in `FederatedAggregator`). The EXP3 driver `run_exp3_federated_aggregation` can compare all four and pick the best on your test split.

**After selection:** The aggregator builds a **new** `RandomForest` instance, injects the selected trees and class labels, and you evaluate on the global test set like any other classifier (same API as a centrally trained `RandomForest`).

---

## 4. Differential privacy in the same layout

**Pattern:** The partition step is unchanged. Instead of `ClientRF`, each site trains **`DPClientRF`** with a per-client **ε** (and optional mechanism such as Laplace in `DPRandomForest`). **Aggregation** then reuses the same **four** strategies, typically using the best non-DP strategy from EXP 3 as a policy for EXP 4.

**Interpretation:** Privacy noise is applied **during local training**; the merger still operates on the **resulting** trees, so you report both **per-client** DP model performance and **federated** global performance for each ε. See [Experiment pipeline](pipeline.md) (EXP 4) and [Code examples](examples.md).

---

## 5. CPU parallelism (`n_jobs`)

The package uses the same **`n_jobs` convention as scikit-learn and joblib** (see
`distributed_random_forest.parallelism.resolve_n_jobs` in code):

* **`1`** (or `None` / `0` resolved to 1) — *no* parallel pool: scoring and custom prediction run in a single process (good for debugging or tiny problems).
* **`-1`** — use all logical CPUs (this is the **default** on
  `FederatedAggregator`, `RandomForest`, and `DPRandomForest` where it applies).
* A **positive int** — at most that many workers (capped to available cores).

**Where it applies**

1. **`RandomForest` / `DPRandomForest`**  
   *Training:* scikit-learn’s `RandomForestClassifier(..., n_jobs=…)` inside `fit`.  
   *After fit:* per-tree **weighted-voting weights** and **inference** on the extracted tree list run in parallel when `n_jobs>1` and you have more than one tree.  
2. **Federation** — `aggregate_trees(..., n_jobs=...)` and `FederatedAggregator(n_jobs=...)` parallelize the **per-tree validation scoring** in `rank_trees_by_metric` (independent for each tree). The **global** `RandomForest` built in `build_global_rf` reuses the same `n_jobs` for prediction.  
3. **EXP3 / EXP4** — `run_exp3_federated_aggregation(..., n_jobs=...)` and
   `run_exp4_dp_federation(..., n_jobs=...)` forward to `aggregate_trees` and the merged `RandomForest`.

Results do **not** change with `n_jobs`; only wall time does (subject to the usual
floating-point and thread-order caveats, which are negligible here).

---

## 6. Quick reference

| Concern | Pattern / knob |
|--------|-----------------|
| IID-like clients, equal size | `partition_uniform_random` or `partitioning='uniform'` |
| Heterogeneous clients by a feature | `partition_by_feature` or `partitioning='feature'` |
| Same client sizes as (feature), but random mix of rows | `partition_random_with_sizes` or `partitioning='sized'` |
| Strong trees per site, per-client cap | `rf_s_dts_a` or `rf_s_dts_wa` with `n_trees_per_client` |
| Best trees anywhere, global cap | `rf_s_dts_a_all` or `rf_s_dts_wa_all` with `n_total_trees` |
| Calibrate on validation | All aggregators: pass shared `X_val`, `y_val` |
| Use all CPU cores for scoring + merged predict | `n_jobs=-1` on `FederatedAggregator`, `RandomForest`, or `aggregate_trees` |

This table is a map to the code; the sections above are the full explanations the library assumes when you read “partition,” “client RF,” and “federated aggregation” in the rest of the docs.
