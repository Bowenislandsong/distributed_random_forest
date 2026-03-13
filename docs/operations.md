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

The publish workflow is configured for trusted publishing via GitHub Actions.
To activate it end-to-end:

1. Create the PyPI project or reserve the name.
2. Add GitHub as a trusted publisher for this repository.
3. Cut a GitHub release.

## Recommended Repository Settings

- require pull requests before merge
- require the CI workflow to pass
- protect `main`
- require signed or verified release tags if your team uses them

## Reporting And Auditability

`FederatedRandomForest.export_report()` emits a JSON report that can be archived
per benchmark run, attached to CI artifacts, or fed into dashboards.
