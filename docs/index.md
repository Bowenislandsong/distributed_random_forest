# Distributed Random Forest

**Federated and distributed [Random Forest](https://en.wikipedia.org/wiki/Random_forest) training** with **optional differential privacy**, inspired by work on *Random Forest with Differential Privacy in Federated Learning for Network Attack Detection*.

## What you get

| Capability | Description |
|------------|-------------|
| **Client-side RF** | Each site trains its own forest (Gini or entropy, simple or weighted voting). |
| **Global RF** | Merge decision trees from clients with several ranking strategies. |
| **Differential privacy** | Train DP random forests and compare privacy–utility trade-offs. |
| **Experiments** | Scripts for hyperparameter search, federation, and DP end-to-end runs. |

## Where to go next

1. [Core concepts](concepts.md) — splits, voting, tree aggregation, and metrics.
2. [Supported distributed RF patterns](patterns.md) — data partitioning, aggregation strategies, and DP layout.
3. [Experiment pipeline](pipeline.md) — how EXP 1–4 are organized.
4. [Getting started](getting-started.md) — install, run experiments, and tests.
5. [Code examples](examples.md) — copy-paste training and federated patterns.

## Install (quick)

```bash
pip install distributed-random-forest
```

For building this documentation locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

Then open the URL printed in the terminal (usually `http://127.0.0.1:8000`).

## Links

* **PyPI:** [distributed-random-forest](https://pypi.org/project/distributed-random-forest/)
* **Source:** [GitHub](https://github.com/Bowenislandsong/distributed_random_forest)
