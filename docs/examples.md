## Included Examples

The repository ships with runnable examples in
[the `examples/` directory](https://github.com/Bowenislandsong/distributed_random_forrest/tree/main/examples).

### `basic_federated_training.py`

Centralized data, uniform client partitioning, automatic strategy selection.

### `non_iid_dirichlet.py`

Dirichlet non-IID client simulation plus explicit report export.

### `dp_enterprise_workflow.py`

Differentially private client training with balanced aggregation and JSON output.

## Example Use Cases

### Network intrusion detection

Different sites collect different traffic mixes. Use `dirichlet` or `label_skew`
partitioning to simulate realistic heterogeneity, then compare `rf_s_dts_wa_all`
and `top_k_global_balanced_accuracy`.

### Multi-branch fraud scoring

Branches may differ dramatically in volume. Use `sized` partitioning and
`proportional_weighted_accuracy` aggregation to preserve stronger representation
from high-volume sites without ignoring smaller branches.

### Privacy-constrained healthcare classification

Enable DP mode with `use_differential_privacy=True` and track the privacy/utility
trade-off across `epsilon` values.

## Running Examples

```bash
python examples/basic_federated_training.py
python examples/non_iid_dirichlet.py
python examples/dp_enterprise_workflow.py
```

## Quick CLI Demo

```bash
drf-quickstart --clients 5 --partition-strategy dirichlet --alpha 0.4
```
