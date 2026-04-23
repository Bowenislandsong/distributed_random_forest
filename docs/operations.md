## CI/CD

The repository includes GitHub Actions workflows for:

- linting with `ruff`
- unit and end-to-end testing across Python versions and operating systems
- package build validation
- documentation build and GitHub Pages deployment
- PyPI publishing on release

## Local Release Checklist

```bash
make lint
make test
make docs
make build
```

## GitHub Pages

The docs workflow builds the MkDocs site and deploys it automatically on pushes
to `main` (and on manual dispatch).

## PyPI Publishing

The **Publish** workflow (`.github/workflows/publish.yml`) runs when a **GitHub Release is published**. It builds an sdist and wheel, then uploads to PyPI using the repository secret **`PYPI_ALL`** (a [PyPI API token](https://pypi.org/manage/account/token/); the workflow uses the username `__token__`).

1. In the GitHub repository: **Settings → Secrets and variables → Actions**, add **`PYPI_ALL`** with the token value (`pypi-…`).
2. On PyPI, ensure the `distributed-random-forest` project exists and the token is scoped to that project (or the whole account, if you use a user-wide token with care).
3. Tag the release commit and push the tag, for example: `git tag v0.4.0` and `git push origin v0.4.0`. The **Publish** workflow runs on that tag and uploads sdist and wheel to PyPI.
4. (Optional) Add a [GitHub Release](https://github.com/Bowenislandsong/distributed_random_forest/releases) for the same tag for release notes in the web UI. Use **Releases → Draft a new release**, pick the tag, and publish (this does not need to re-upload to PyPI).

The canonical list of what changed in each version is in [`CHANGELOG.md`](https://github.com/Bowenislandsong/distributed_random_forest/blob/main/CHANGELOG.md) and on the docs site under **Changelog**.

## Recommended Repository Settings

- require pull requests before merge
- require the CI workflow to pass
- protect `main`
- require signed or verified release tags if your team uses them

## Reporting And Auditability

`FederatedRandomForest.export_report()` emits a JSON report that can be archived
per benchmark run, attached to CI artifacts, or fed into dashboards.
