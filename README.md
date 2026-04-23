# Distributed Random Forest (with differential privacy)

[![PyPI version](https://img.shields.io/pypi/v/distributed_random_forest)](https://pypi.org/project/distributed_random_forest/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/distributed-random-forest?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/distributed-random-forest)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python versions](https://img.shields.io/pypi/pyversions/distributed_random_forest)](https://pypi.org/project/distributed_random_forest/)
[![Tests](https://github.com/Bowenislandsong/distributed_random_forest/actions/workflows/tests.yml/badge.svg)](https://github.com/Bowenislandsong/distributed_random_forest/actions/workflows/tests.yml)

Federated and distributed **Random Forest** training: multiple clients each learn a local forest, you merge **decision trees** into a **global** model, and you can add **differential privacy** (DP) at the client. The design is inspired by research on *Random Forest with Differential Privacy in Federated Learning* (network attack use cases in the original paper; this codebase is general-purpose).

## At a glance

- Train RFs on many clients, aggregate with **four tree-ranking strategies** (per-client and global).
- **Gini** or **entropy** splits; **simple** or **weighted** voting.
- **DP** random forests and federated DP vs non-DP comparisons.
- **Scripts** for hyperparameter search, partition experiments, and EXP 1–4-style pipelines (see the docs).

## Documentation

- **Online:** [full documentation (GitHub Pages)](https://bowenislandsong.github.io/distributed_random_forest/) — [enable Pages from Actions](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site#publishing-with-a-custom-github-actions-workflow) on the repository if the site is not live yet, then run the *Documentation* workflow (or push to `main`).
- **Local:** `pip install -e ".[docs]"` and `mkdocs serve`, then open the local URL in your browser.

**Contents:** [concepts & metrics](https://bowenislandsong.github.io/distributed_random_forest/concepts/) · [distributed RF patterns](https://bowenislandsong.github.io/distributed_random_forest/patterns/) · [experiment pipeline](https://bowenislandsong.github.io/distributed_random_forest/pipeline/) · [install & tests](https://bowenislandsong.github.io/distributed_random_forest/getting-started/) · [code examples](https://bowenislandsong.github.io/distributed_random_forest/examples/) · [repository layout](https://bowenislandsong.github.io/distributed_random_forest/repository/) · [license & citation](https://bowenislandsong.github.io/distributed_random_forest/citing/)

## Quick install

```bash
pip install distributed-random-forest
```

**From source (editable):**

```bash
git clone https://github.com/Bowenislandsong/distributed_random_forest.git
cd distributed_random_forest
pip install -e .
# optional: tests — pip install -e ".[dev]"
# optional: docs  — pip install -e ".[docs]"
```

## Quick start (API)

```python
from distributed_random_forest import RandomForest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForest(n_estimators=100, criterion="gini", voting="simple", random_state=42)
rf.fit(X_train, y_train)
print(f"Accuracy: {rf.score(X_test, y_test):.4f}")
```

Federated, DP, and aggregation examples live in the **[examples](https://bowenislandsong.github.io/distributed_random_forest/examples/)** page (full snippets, imports, and EXP3 helper usage).

## Run experiment drivers

```bash
python run_exp1_hparams.py
python run_exp2_clients.py
python run_exp3_federation.py
python run_exp4_dp_federation.py
```

**UCI (public) benchmark** — [Wisconsin breast cancer](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+\(Diagnostic\)) via scikit-learn, with **test accuracy** and **prediction latency** (central vs federated):

```bash
python examples/benchmark_public_dataset.py
python examples/benchmark_public_dataset.py --quick
```

## Run tests

```bash
pytest tests/ -v
pytest tests/ --cov=distributed_random_forest
```

## Cite

BibTeX and APA are on the [License & citation](https://bowenislandsong.github.io/distributed_random_forest/citing/) page (same text as in `docs/citing.md` if you are offline).

## License

Apache License 2.0 — see [LICENSE](LICENSE).
