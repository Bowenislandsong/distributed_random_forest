# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-04-23

### Added

- `FederatedRandomForest` orchestration, partitioning, strategy search, and JSON run reports (`distributed_random_forest.distributed`).
- CLI entry point `drf-quickstart` (`drf-quickstart` / `python -m distributed_random_forest.cli`).
- Extended tree aggregation: additional strategies, `AggregationSummary`, and structured reporting in `FederatedAggregator`.
- Parallel `n_jobs` for aggregation scoring, `RandomForest` merged-tree prediction, and `rank_trees_by_metric` (joblib, parity-tested).
- `CHANGELOG.md` and a **Changelog** page in the documentation site.
- `tests/timing.py` helpers so time-bounded performance tests pass on slow CI runners (e.g. `macos-latest`).

### Changed

- CI: `ci.yml` (lint, multi-OS / multi-Python tests, package smoke); `tests.yml` removed in favor of unified CI.
- Docs: Material theme, merged guides (`patterns`, `concepts`, `pipeline`, examples); MkDocs emoji config uses valid `!!python/name` tags.
- `pyproject.toml`: dependencies include `joblib`, `pandas`, and `diffprivlib`; optional `privacy` extra.

### Fixed

- Documentation build: `pymdownx.emoji` no longer received quoted placeholder strings (restore working `mkdocs build --strict`).

[0.4.0]: https://github.com/Bowenislandsong/distributed_random_forest/compare/v0.3.1...v0.4.0

## [0.3.1] - earlier

Initial numbered baseline for this changelog file; see [GitHub releases](https://github.com/Bowenislandsong/distributed_random_forest/releases) for prior tags and notes.
