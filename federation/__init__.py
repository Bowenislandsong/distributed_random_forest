"""Federation package for aggregation strategies and voting methods."""

from federation.aggregator import (
    rf_s_dts_a,
    rf_s_dts_wa,
    rf_s_dts_a_all,
    rf_s_dts_wa_all,
    aggregate_trees,
)
from federation.voting import simple_voting, weighted_voting

__all__ = [
    'rf_s_dts_a',
    'rf_s_dts_wa',
    'rf_s_dts_a_all',
    'rf_s_dts_wa_all',
    'aggregate_trees',
    'simple_voting',
    'weighted_voting',
]
