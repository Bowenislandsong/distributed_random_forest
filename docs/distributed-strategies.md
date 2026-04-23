## Partitioning Strategies

The project now supports a broader range of client-data topologies.

| Strategy | Best for | Notes |
| --- | --- | --- |
| `uniform` | balanced client splits | Fast default for smoke tests |
| `stratified` | class-balance preservation | Good baseline for fair comparisons |
| `feature` | rule-based site segmentation | Useful when client boundaries depend on a feature |
| `sized` | capacity planning | Simulate known site volumes |
| `dirichlet` | non-IID research | Standard benchmark for heterogeneous federated learning |
| `label_skew` | site specialization | Simulate clients that only observe a subset of labels |

## Aggregation Strategies

### Legacy paper-style strategies

| Strategy | Behavior |
| --- | --- |
| `rf_s_dts_a` | top-K trees per client by accuracy |
| `rf_s_dts_wa` | top-K trees per client by weighted accuracy |
| `rf_s_dts_a_all` | global top-K trees by accuracy |
| `rf_s_dts_wa_all` | global top-K trees by weighted accuracy |

### Extended strategies

| Strategy | Behavior |
| --- | --- |
| `top_k_global_balanced_accuracy` | global top-K by balanced accuracy |
| `top_k_global_f1` | global top-K by macro F1 |
| `proportional_weighted_accuracy` | allocate tree budget by client data volume, then rank by weighted accuracy |
| `proportional_balanced_accuracy` | proportional budget with balanced-accuracy ranking |
| `threshold_weighted_accuracy` | include trees above a quality floor, with fallback selection |

## When To Use Which Strategy

- Use `rf_s_dts_wa_all` when you want a strong general-purpose baseline.
- Use `top_k_global_balanced_accuracy` when class imbalance matters.
- Use `proportional_weighted_accuracy` when larger clients should contribute more trees.
- Use `aggregation_strategy="auto"` in `FederatedRandomForest` when you want validation-driven strategy selection.

## Example

```python
from distributed_random_forest import FederatedRandomForest

model = FederatedRandomForest(
    n_clients=6,
    partition_strategy="dirichlet",
    partition_kwargs={"alpha": 0.3},
    aggregation_strategy="auto",
    selection_metric="weighted_accuracy",
    execution_backend="thread",
)
```

## Metric Bundle

Validation and test reports include:

- `accuracy`
- `weighted_accuracy`
- `balanced_accuracy`
- `f1_score`
