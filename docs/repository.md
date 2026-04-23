# Repository layout

```
distributed_random_forest/
├── distributed_random_forest/   # installable package
│   ├── __init__.py              # public exports
│   ├── data/                    # ad-hoc data assets
│   ├── datasets/                # UCI / sklearn public loaders (benchmarks, docs)
│   ├── models/
│   │   ├── random_forest.py     # RF and clients
│   │   ├── dp_rf.py              # DP RF
│   │   └── tree_utils.py
│   ├── federation/
│   │   ├── aggregator.py
│   │   └── voting.py
│   └── experiments/             # EXP 1–4 entry points
├── docs/                        # this documentation (MkDocs)
├── examples/                    # runnable scripts (UCI benchmark CLI)
├── tests/
├── .github/workflows/           # CI (tests) and doc deploy
├── mkdocs.yml
├── pyproject.toml
├── requirements.txt
└── README.md
```

| Path | Role |
|------|------|
| `run_exp*.py` (repo root) | Runnable experiment drivers matching [Experiment pipeline](pipeline.md). |
| `tests/` | `pytest` suite (unit, integration, and end-to-end). |
| `mkdocs.yml` + `docs/` | Static site built with MkDocs. |

Continuous integration runs the test suite, lint checks, and experiment smoke tests on each push/PR. Documentation can be published to **GitHub Pages** when the workflow in `.github/workflows/docs.yml` is enabled on the default branch.
