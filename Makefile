MKDOCS_SERVE_ADDR ?= localhost:8000 # Default address for mkdocs serve, format: <host>:<port>, override with `make docs-serve MKDOCS_SERVE_ADDR=<host>:<port>`

autoformat:
	ruff check guardrails/ tests/ --fix
	ruff format guardrails/ tests/
	docformatter --in-place --recursive guardrails tests

.PHONY: type
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

docs-serve:
	poetry run mkdocs serve -a $(MKDOCS_SERVE_ADDR)

docs-deploy:
	poetry run mkdocs gh-deploy

dev:
	poetry install --extras "dev"
	poetry run pre-commit install

full:
	poetry install --all-extras

docs-gen:
	poetry run python ./docs/pydocs/generate_pydocs.py
	cp -r docs/src/* docs/build
	poetry run nbdoc_build --force_all True --srcdir ./docs/build

self-install:
	pip install -e .

all: autoformat type lint docs test

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

