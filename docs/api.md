## Primary Imports

```python
from distributed_random_forest import (
    RandomForest,
    DPRandomForest,
    ClientRF,
    DPClientRF,
    FederatedAggregator,
    FederatedRandomForest,
    create_partitions,
)
```

## Local Training

### `RandomForest`

Train a local forest with simple or weighted voting.

### `DPRandomForest`

Train a local forest with built-in differential privacy noise at the tree level.

## Federated Training

### `ClientRF` and `DPClientRF`

Client-scoped wrappers that store:

- training metrics
- validation metrics
- class distributions
- sample counts

### `FederatedAggregator`

Use this when you want direct control over tree selection:

```python
aggregator = FederatedAggregator(
    strategy="top_k_global_balanced_accuracy",
    n_total_trees=24,
)
```

### `FederatedRandomForest`

Use this when you want an end-to-end workflow with partitioning, parallel client
training, strategy evaluation, and JSON reports.

## Utilities

### Partitioning

- `create_partitions`
- `partition_uniform_random`
- `partition_stratified`
- `partition_dirichlet`
- `partition_label_skew`
- `partition_by_feature`
- `partition_random_with_sizes`

### Metrics

- `compute_accuracy`
- `compute_weighted_accuracy`
- `compute_balanced_accuracy`
- `compute_f1_score`
- `evaluate_predictions`

### Aggregation Constants

- `AVAILABLE_STRATEGIES`
- `AggregationSummary`
