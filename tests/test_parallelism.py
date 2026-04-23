"""Tests for :mod:`distributed_random_forest.parallelism`."""

from distributed_random_forest import resolve_n_jobs


class TestResolveNJobs:
    def test_none_is_one(self) -> None:
        assert resolve_n_jobs(None) == 1

    def test_zero_is_one(self) -> None:
        assert resolve_n_jobs(0) == 1

    def test_pos_int(self) -> None:
        assert resolve_n_jobs(2) == 2

    def test_neg_one_uses_effective_cores(self) -> None:
        n = resolve_n_jobs(-1)
        assert n >= 1
