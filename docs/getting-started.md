# Getting started

## Install from PyPI

```bash
pip install distributed-random-forest
```

## Editable install (source)

```bash
git clone https://github.com/Bowenislandsong/distributed_random_forest.git
cd distributed_random_forest
pip install -e .
```

## Development dependencies

```bash
pip install -e ".[dev]"
```

## Build the documentation

```bash
pip install -e ".[docs]"
mkdocs serve    # local preview
# mkdocs build  # static site in ./site
```

## Run experiment scripts

| Stage | Command |
|-------|---------|
| EXP 1 — hyperparameters | `python run_exp1_hparams.py` |
| EXP 2 — per-client RFs | `python run_exp2_clients.py` |
| EXP 3 — federation | `python run_exp3_federation.py` |
| EXP 4 — DP federation | `python run_exp4_dp_federation.py` |
| UCI example (accuracy & latency) | `python examples/benchmark_public_dataset.py` — use `--quick` for a short run |

## Run tests

```bash
pytest tests/ -v
```

With coverage of the `distributed_random_forest` package:

```bash
pytest tests/ -v --cov=distributed_random_forest
```

Targeted suites (examples):

| File | Focus |
|------|--------|
| `tests/test_tree_utils.py` | Utilities and tree metrics |
| `tests/test_random_forest.py` | Core RF |
| `tests/test_dp_rf.py` | DP random forest |
| `tests/test_voting.py` | Voting |
| `tests/test_aggregator.py` | Aggregation |
| `tests/test_e2e.py` | End-to-end (synthetic) |
| `tests/test_e2e_public_dataset.py` | End-to-end (UCI breast cancer) |
| `tests/test_datasets.py` | Public dataset loader |
| `tests/test_performance.py` | Accuracy / latency bounds (marked `performance`) |
| `tests/test_examples_run.py` | Example script smoke test |
| `tests/test_parallel_e2e.py` | E2E: ``n_jobs=1`` vs ``-1`` parity (federated, EXP3) |
| `tests/test_parallel_stress.py` | Stress: many clients/trees, ranking load |
| `tests/test_parallelism.py` | :func:`resolve_n_jobs` |

## Next steps

* [Supported distributed RF patterns](patterns.md) for partitioning, aggregation, and DP layout.
* [Code examples](examples.md) for API usage.
* [Core concepts](concepts.md) for design detail.
