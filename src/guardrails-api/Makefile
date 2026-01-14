
# Installs production dependencies
install:
	uv sync

# Installs development dependencies
install-dev:
	uv sync --group dev

dev:
	uv sync --group dev
	uv run pre-commit install

full:
	uv sync --group dev

lock:
	uv lock

install-lock:
	uv sync

.PHONY: build
build:
	make install
	
	cp "$$(python -c "import guardrails_api_client as _; print(_.__path__[0])")/openapi-spec.json" ./guardrails_api/open-api-spec.json
	

start:
	make build
	bash ./guardrails_api/start.sh

start-dev:
	make dev
	make build
	bash ./guardrails_api/start-dev.sh

infra:
	docker compose --profile infra up --build

env:
	@echo "Use 'uv sync' to manage the environment. Use 'uv run <cmd>' to run commands."

refresh:
	uv sync


format:
	uv run ruff check guardrails_api/ tests/ --fix
	uv run ruff format guardrails_api/ tests/


lint:
	uv run ruff check guardrails_api/ tests/
	uv run ruff format guardrails_api/ tests/ --check

type:
	uv run ty check guardrails_api/

qa:
	make build
	make lint
	make type
	make test-cov

# This doesn't actually work, but it's nice to be able to just copy/paste instead of typing this out in the terminal.
source:
	@echo "No-op: uv manages the environment. Use 'uv run <cmd>' instead."

test:
	uv run pytest ./tests

test-cov:
	uv run coverage run --source=./guardrails_api -m pytest ./tests
	uv run coverage report --fail-under=45

view-test-cov:
	uv run coverage run --source=./guardrails_api -m pytest ./tests
	uv run coverage html
	open htmlcov/index.html