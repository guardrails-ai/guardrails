# Guardrails AI Makefile
# For full task management, use mise: `mise tasks` or `mise run <task>`
# This Makefile provides compatibility for common workflows.

MKDOCS_SERVE_ADDR ?= localhost:8000

.PHONY: all install dev full lint type test test-cov docs-serve docs-deploy \
        autoformat precommit refresh clean build check nox

# =============================================================================
# Installation
# =============================================================================

install:
	uv sync

dev:
	uv sync --group dev
	uv run pre-commit install

full:
	uv sync --group dev --all-extras

# =============================================================================
# Code Quality
# =============================================================================

autoformat:
	ruff check guardrails/ tests/ --fix
	ruff format guardrails/ tests/

lint:
	ruff check guardrails/ tests/
	ruff format guardrails/ tests/ --check

type:
	uv run pyright guardrails/

# =============================================================================
# Testing
# =============================================================================

test:
	uv run pytest tests/ -v

test-basic:
	uv run python -c "import guardrails as gd"
	uv run python -c "import guardrails.version as mversion"

test-cov:
	uv run pytest tests/ --cov=./guardrails/ --cov-report=xml --cov-report=term

view-test-cov:
	uv run pytest tests/ --cov=./guardrails/ --cov-report=html && open htmlcov/index.html

test-unit:
	uv run pytest tests/unit_tests/ -v

test-integration:
	uv run pytest tests/integration_tests/ -v

nox:
	uv run nox

# =============================================================================
# Documentation
# =============================================================================

docs-serve:
	uv run mkdocs serve -a $(MKDOCS_SERVE_ADDR)

docs-build:
	uv run mkdocs build

docs-deploy:
	uv run mkdocs gh-deploy

docs-gen:
	uv run python ./docs/pydocs/generate_pydocs.py
	cp -r docs/src/* docs/dist
	uv run nbdoc_build --force_all True --srcdir ./docs/dist

# =============================================================================
# Build & Release
# =============================================================================

build:
	uv build

clean:
	rm -rf dist/ build/ *.egg-info .pytest_cache .coverage htmlcov/ .ruff_cache/

# =============================================================================
# CI / Validation
# =============================================================================

precommit:
	uv run pyright guardrails/
	$(MAKE) lint
	./.github/workflows/scripts/update_notebook_matrix.sh

check: lint type test

liccheck:
	uv run liccheck -s pyproject.toml

# =============================================================================
# Environment Management
# =============================================================================

refresh:
	@echo "Removing old virtual environment and lock file"
	rm -rf .venv uv.lock
	@echo "Reinstalling dependencies"
	uv sync --group dev

update-lock:
	uv lock

all: autoformat type lint test
