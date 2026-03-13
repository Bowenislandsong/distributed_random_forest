## System Overview

The library is organized into four layers:

1. `models`
   Local Random Forest and DP Random Forest implementations.
2. `federation`
   Tree ranking, aggregation, and voting behavior.
3. `distributed`
   Partitioning, parallel execution, orchestration, and run reporting.
4. `experiments`
   Paper-style experiment flows for benchmarking and reproduction.

## Training Flow

1. Prepare a centralized dataset or explicit client partitions.
2. Hold out a shared validation set or provide one directly.
3. Partition the remaining data across clients.
4. Train local client forests in parallel.
5. Score candidate trees on the shared validation set.
6. Build one or more global forests from selected trees.
7. Choose the best strategy by validation metric and evaluate on test data.
8. Export a structured report for downstream review.

## Key APIs

| Layer | Primary API | Purpose |
| --- | --- | --- |
| Local models | `RandomForest`, `DPRandomForest` | Train single-site forests |
| Client wrappers | `ClientRF`, `DPClientRF` | Capture metrics and metadata per client |
| Aggregation | `FederatedAggregator` | Apply explicit tree-selection strategies |
| Orchestration | `FederatedRandomForest` | End-to-end distributed workflow |

## Design Goals

### Backward compatibility

The original paper-inspired functions such as `rf_s_dts_a` and `rf_s_dts_wa_all`
still exist and still work.

### Operational clarity

The orchestration layer returns structured strategy reports rather than only
printing scores to stdout.

### Extension points

You can extend the project in a few predictable places:

- add new partitioning methods in `distributed/partitioning.py`
- add new selection modes or metrics in `federation/aggregator.py`
- add new client types by mirroring `ClientRF` or `DPClientRF`
- add new workflows or benchmarks in `examples/` and `experiments/`

## Enterprise Readiness Checklist

- typed, importable Python package
- PyPI packaging metadata
- GitHub Actions for tests, linting, package build, and docs deployment
- JSON run reports for reproducibility
- examples for baseline, non-IID, and DP workflows
- docs site deployable to GitHub Pages
