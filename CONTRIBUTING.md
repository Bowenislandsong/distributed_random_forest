# Contributing

## Setup

```bash
python -m pip install -e ".[dev,docs]"
```

## Before Opening A PR

```bash
make lint
make test
make docs
make build
```

## Pull Request Expectations

- keep changes focused
- include tests for behavioral changes
- update docs or examples when public APIs change
- avoid breaking the legacy paper-style experiment scripts unless the change is intentional
