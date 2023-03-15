autoformat:
	black guardrails/ tests/ demo/
	isort --atomic guardrails/ tests/ demo/
	docformatter --in-place --recursive guardrails tests demo

lint:
	isort -c guardrails/ tests/ demo/
	black guardrails/ tests/ demo/ --check
	flake8 guardrails/ tests/ demo/

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