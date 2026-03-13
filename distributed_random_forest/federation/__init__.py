"""Federation package for aggregation strategies and voting methods."""

__all__ = [
    'AVAILABLE_STRATEGIES',
    'AggregationSummary',
    'rf_s_dts_a',
    'rf_s_dts_wa',
    'rf_s_dts_a_all',
    'rf_s_dts_wa_all',
    'aggregate_trees',
    'simple_voting',
    'weighted_voting',
]


def __getattr__(name):
    """Lazily expose federation helpers without creating import cycles."""
    if name in {
        'AVAILABLE_STRATEGIES',
        'AggregationSummary',
        'rf_s_dts_a',
        'rf_s_dts_wa',
        'rf_s_dts_a_all',
        'rf_s_dts_wa_all',
        'aggregate_trees',
    }:
        from distributed_random_forest.federation import aggregator

        return getattr(aggregator, name)

    if name in {'simple_voting', 'weighted_voting'}:
        from distributed_random_forest.federation import voting

        return getattr(voting, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
