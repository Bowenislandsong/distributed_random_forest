<div class="hero">
  <h1>Distributed Random Forest</h1>
  <p>
    A Python toolkit for teams that want more than a notebook demo: federated training,
    parallel orchestration, non-IID partitioning, aggregation strategy search, optional
    differential privacy, structured run reports, and CI/CD-ready packaging.
  </p>
</div>

**Federated and distributed [Random Forest](https://en.wikipedia.org/wiki/Random_forest) training** with **optional differential privacy**, inspired by work on *Random Forest with Differential Privacy in Federated Learning for Network Attack Detection*.

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

## Maintainer

This project is maintained by
[Bowen Song](https://bowenislandsong.github.io/#/personal), an AI Scientist and
PhD candidate at USC Viterbi working across health AI, federated learning,
explainable AI, and scalable machine-learning systems. The project’s
positioning is intentionally research-aware but package-first: it should be
useful for reproducible experiments, demos, and real engineering evaluation.

- Personal site: [bowenislandsong.github.io/#/personal](https://bowenislandsong.github.io/#/personal)
- CV: [Bowen_Song_Resume.pdf](https://bowenislandsong.github.io/img/resume/Bowen_Song_Resume.pdf)
- ORCID: [0000-0002-5071-3880](https://orcid.org/0000-0002-5071-3880)

## What You Can Build

- Privacy-sensitive security analytics where data must remain on client sites.
- Multi-region fraud, risk, or IoT classifiers with heterogeneous data quality.
- Research benchmarks for tree-selection strategies under non-IID client splits.
- CI-verified Python packages that ship clean docs, examples, and release workflows.

## Why This Implementation Stands Out

| Area | Baseline script repo | This project |
| --- | --- | --- |
| Federated workflow | ad hoc experiments | reusable orchestration API |
| Heterogeneity | mostly uniform data splits | uniform, stratified, feature, sized, Dirichlet, label skew |
| Aggregation | one or two selection rules | classic paper rules plus balanced, proportional, threshold, and auto search |
| Operational maturity | code only | package, CLI, CI, docs, GitHub Pages, release workflow |

## Performance Snapshot

Single-run local benchmark on a synthetic 4-class dataset. Reproduce with
`python examples/performance_benchmark.py` (when the script is present in the repo).

| Scenario | Accuracy | Time (s) | Strategy |
| --- | ---: | ---: | --- |
| Centralized RF | 0.8642 | 0.35 | n/a |
| Federated uniform | 0.7842 | 1.44 | `proportional_weighted_accuracy` |
| Federated dirichlet | 0.7642 | 1.42 | `proportional_weighted_accuracy` |
| Federated dirichlet + DP | 0.5125 | 0.74 | `top_k_global_balanced_accuracy` |

## Guides

1. [Core concepts](concepts.md) — splits, voting, tree aggregation, and metrics.
2. [Patterns (parallel, DP, sharding)](patterns.md) — `n_jobs`, data partitioning, aggregation strategies, and DP layout.
3. [Experiment pipeline](pipeline.md) — how EXP 1–4 are organized.
4. [Getting started](getting-started.md) — install, run experiments, and tests.
5. [Code examples](examples.md) — copy-paste training and federated patterns.

## Quick install

```bash
pip install distributed-random-forest
```

Or from source:

```bash
git clone https://github.com/Bowenislandsong/distributed_random_forest
cd distributed_random_forest
python -m pip install -e ".[dev,docs]"
```

## Fastest way to try it

```bash
drf-quickstart --clients 4 --partition-strategy dirichlet --backend thread
```

## Build this documentation locally

```bash
pip install -e ".[docs]"
mkdocs serve
```

Then open the URL printed in the terminal (usually `http://127.0.0.1:8000`).

## Core building blocks

- `RandomForest` and `DPRandomForest` for local training.
- `ClientRF` and `DPClientRF` for client-scoped model ownership and metrics.
- `FederatedAggregator` for explicit tree selection strategies.
- `FederatedRandomForest` for end-to-end orchestration, validation, and reporting.

## Next steps

- Follow [Getting started](getting-started.md) for installation and a first training run.
- Read [Distributed strategies & DP](distributed-strategies.md) to choose partitioning and aggregation.
- Use [Operations](operations.md) to enable CI, release publishing, and GitHub Pages deployment.

## Links

- **PyPI:** [distributed-random-forest](https://pypi.org/project/distributed-random-forest/)
- **Source:** [GitHub](https://github.com/Bowenislandsong/distributed_random_forest)
