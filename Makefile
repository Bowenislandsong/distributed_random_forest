PYTHON ?= python

.PHONY: install install-dev lint test build docs quickstart

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev,docs]"

lint:
	$(PYTHON) -m ruff check .

test:
	$(PYTHON) -m pytest tests -q

build:
	$(PYTHON) -m build

docs:
	$(PYTHON) -m mkdocs build --strict

quickstart:
	$(PYTHON) -m distributed_random_forest.cli --json
