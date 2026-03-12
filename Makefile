.PHONY: autoformat type lint test test-basic test-cov view-test-cov view-test-cov-file dev full install docs-gen self-install all precommit refresh update-lock

autoformat:
	ruff check guardrails/ tests/ --fix
	ruff format guardrails/ tests/
	docformatter --in-place --recursive guardrails tests

type:
	pyright guardrails/

lint:
	ruff check guardrails/ tests/
	ruff format guardrails/ tests/ --check

test:
	pytest tests/

test-basic:
	set -e
	python -c "import guardrails as gd"
	python -c "import guardrails.version as mversion"

test-cov:
	pytest tests/ --cov=./guardrails/ --cov-report=xml

view-test-cov:
	pytest tests/ --cov=./guardrails/ --cov-report html && open htmlcov/index.html

view-test-cov-file:
	pytest tests/unit_tests/test_logger.py --cov=./guardrails/ --cov-report html && open htmlcov/index.html

dev:
	poetry install --extras "dev"
	poetry run pre-commit install

full:
	poetry install --all-extras

install:
	poetry install

docs-gen:
	poetry run python ./docs/pydocs/generate_pydocs.py

self-install:
	pip install -e .

all: autoformat type lint docs-gen test

precommit:
	# pytest -x -q --no-summary
	pyright guardrails/
	make lint
	./.github/workflows/scripts/update_notebook_matrix.sh

refresh:
	echo "Removing old virtual environment"
	rm -rf ./.venv;
	echo "Creating new virtual environment"
	python3 -m venv ./.venv;
	echo "Sourcing and installing"
	source ./.venv/bin/activate && make full;

update-lock:
	poetry lock --no-update

