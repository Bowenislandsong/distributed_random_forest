# Experiment pipeline

The repo ships scripts that follow a four-stage **EXP 1 → EXP 4** flow. Stages build on the best settings from the previous step where applicable. For a fuller account of each **partition** style (EXP 2) and each **aggregation** style (EXP 3), read [Supported distributed RF patterns](patterns.md).

## EXP 1 — Hyperparameters

Runs **before** simulating federated split data.

* **Search:** number of trees (odd counts in a configured range, e.g. 1–100), **gini** vs **entropy**, **SV** vs **WV**.
* **Output:** One best configuration reused in later experiments.

**Script:** `run_exp1_hparams.py`

## EXP 2 — Independent client forests

Each client trains with the EXP 1 configuration. **Data split** style matters:

| Sub-exp | Split idea |
|---------|------------|
| **2.1** | Feature-based partitions; test on the client’s slice and on the full test set. |
| **2.2** | Uniform random shards of equal size; evaluate on the full test set. |
| **2.3** | Same **sizes** as 2.1 but randomize which samples go to which client. |

**Script:** `run_exp2_clients.py`

## EXP 3 — Global RF via federation

Merge client RFs with **RF_S_DTs_A**, **RF_S_DTs_WA**, **RF_S_DTs_A_All**, or **RF_S_DTs_WA_All**; compare the global model to independent RFs, best single client, and (if run) a centralized baseline.

**Script:** `run_exp3_federation.py`

## EXP 4 — Differential privacy

* Clients train **DP** random forests (example ε values: **0.1, 0.5, 1, 5**).
* Each DP model is scored on the full test set; then trees are merged using the **best strategy from EXP 3**.
* Compare: DP per-client, federated DP global, and non-DP global.

**Script:** `run_exp4_dp_federation.py`

---

See [Getting started](getting-started.md) for one-line commands to run each script.
