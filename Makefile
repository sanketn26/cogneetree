.PHONY: help setup venv install test package clean clean-env lock

POETRY ?= poetry
PYTHON ?= python3.11

help:
	@echo "Available targets:"
	@echo "  setup      Create in-project .venv via Poetry and install dev dependencies"
	@echo "  venv       Create/select local .venv with Poetry"
	@echo "  install    Install project dependencies in Poetry env (including dev extras)"
	@echo "  test       Run test suite via Poetry"
	@echo "  package    Build distribution artifacts via Poetry"
	@echo "  lock       Regenerate poetry.lock without updating dependency versions"
	@echo "  clean      Remove build and test artifacts"
	@echo "  clean-env  Remove Poetry virtual environments and run clean"

setup: venv install

venv:
	$(POETRY) config virtualenvs.in-project true --local
	$(POETRY) env use $(PYTHON)

install:
	$(POETRY) install --extras "dev"

test:
	$(POETRY) run pytest -q

package:
	$(POETRY) build

lock:
	$(POETRY) lock --no-update

clean:
	rm -rf build dist .pytest_cache .mypy_cache .coverage htmlcov
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-env: clean
	$(POETRY) env remove --all || true
	rm -rf .venv
