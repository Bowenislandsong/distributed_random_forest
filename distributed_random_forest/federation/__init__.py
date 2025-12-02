"""Federation package for aggregation strategies and voting methods."""

from distributed_random_forest.federation.aggregator import (
    rf_s_dts_a,
    rf_s_dts_wa,
    rf_s_dts_a_all,
    rf_s_dts_wa_all,
    aggregate_trees,
)
from distributed_random_forest.federation.voting import simple_voting, weighted_voting

__all__ = [
    'rf_s_dts_a',
    'rf_s_dts_wa',
    'rf_s_dts_a_all',
    'rf_s_dts_wa_all',
    'aggregate_trees',
    'simple_voting',
    'weighted_voting',
]
