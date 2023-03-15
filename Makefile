autoformat:
	black guardrails/ tests/
	isort --atomic guardrails/ tests/
	docformatter --in-place --recursive guardrails tests

lint:
	isort -c guardrails/ tests/
	black guardrails/ tests/ --check
	flake8 guardrails/ tests/

test:
	pytest tests/

test-basic:
	set -e
	python -c "import guardrails as gd"
	python -c "import guardrails.version as mversion"

test-cov:
	pytest tests/ --cov=./ --cov-report=xml

docs-serve:
	mkdocs serve

docs-deploy:
	mkdocs gh-deploy

dev:
	pip install -e ".[dev]"

all: autoformat lint docs test