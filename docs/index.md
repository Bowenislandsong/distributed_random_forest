<div class="hero">
  <h1>Distributed Random Forest</h1>
  <p>
    A Python toolkit for teams that want more than a notebook demo: federated training,
    parallel orchestration, non-IID partitioning, aggregation strategy search, optional
    differential privacy, structured run reports, and CI/CD-ready packaging.
  </p>
</div>

## Why This Project Exists

Most "distributed random forest" repos stop at "train a few local forests and concatenate trees."
This project goes further:

<div class="grid-cards">
  <div class="grid-card">
    <strong>Real federated orchestration</strong>
    Parallel client training with sequential, thread, or process backends.
  </div>
  <div class="grid-card">
    <strong>Non-IID simulation</strong>
    Uniform, stratified, feature-based, Dirichlet, sized, and label-skew partitioning.
  </div>
  <div class="grid-card">
    <strong>Aggregation strategy search</strong>
    Compare classic paper baselines with balanced and proportional selectors.
  </div>
  <div class="grid-card">
    <strong>Enterprise reporting</strong>
    Export structured JSON reports for audits, benchmarks, and dashboards.
  </div>
</div>

## What You Can Build

- Privacy-sensitive security analytics where data must remain on client sites.
- Multi-region fraud, risk, or IoT classifiers with heterogeneous data quality.
- Research benchmarks for tree-selection strategies under non-IID client splits.
- CI-verified Python packages that ship clean docs, examples, and release workflows.

## Quick Install

```bash
pip install distributed-random-forest
```

Or from source:

```bash
git clone https://github.com/Bowenislandsong/distributed_random_forest
cd distributed_random_forest
python -m pip install -e ".[dev,docs]"
```

## Fastest Way To Try It

```bash
drf-quickstart --clients 4 --partition-strategy dirichlet --backend thread
```

## Core Building Blocks

- `RandomForest` and `DPRandomForest` for local training.
- `ClientRF` and `DPClientRF` for client-scoped model ownership and metrics.
- `FederatedAggregator` for explicit tree selection strategies.
- `FederatedRandomForest` for end-to-end orchestration, validation, and reporting.

## Next Steps

- Follow [Getting Started](getting-started.md) for installation and a first training run.
- Read [Distributed Strategies](distributed-strategies.md) to choose partitioning and aggregation.
- Use [Operations](operations.md) to enable CI, release publishing, and GitHub Pages deployment.
