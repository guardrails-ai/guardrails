# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

This project uses **uv** for Python dependency management. Use either `make` or `mise` for task execution.

```bash
# Install dependencies
make dev                    # Install with dev dependencies + pre-commit hooks
make full                   # Install with all optional extras

# Code quality
make autoformat             # Fix lint issues and format code
make lint                   # Check linting (ruff check + format --check)
make type                   # Run type checker (ty check)

# Testing
make test                   # Run all tests
make test-unit              # Run unit tests only
make test-integration       # Run integration tests only
uv run pytest tests/unit_tests/test_file.py -v              # Run specific test file
uv run pytest tests/unit_tests/test_file.py::test_func -v   # Run specific test function

# Multi-version testing
make nox                    # Run tests across Python 3.10-3.14

# Build
make build                  # Build package with uv
```

## Architecture Overview

### Core Components

**Guard** (`guardrails/guard.py`) - The main entry point for validation. Guards can be created via:
- `Guard().use(validator)` / `Guard().use_many(...)` - Add validators to a guard
- `Guard.for_pydantic(Model)` - Create from Pydantic model
- `Guard.for_string(validators)` - Create for string validation
- `Guard.for_rail(file)` - Create from .rail file (legacy XML format)

Guards support both local execution and server mode (`settings.use_server`).

**Validator** (`guardrails/validator_base.py`) - Base class for all validators. Validators return `PassResult`, `FailResult`, or `ValidationResult`. Key attributes:
- `rail_alias` - Unique identifier for the validator
- `on_fail` - Action to take on failure (EXCEPTION, NOOP, FIX, REASK, etc.)
- `required_metadata_keys` - Keys that must be present in metadata

**Runner** (`guardrails/run/runner.py`, `stream_runner.py`, `async_runner.py`) - Orchestrates LLM calls and validation loops. Handles reasks when validation fails.

**ValidatorService** (`guardrails/validator_service/`) - Executes validators against values. Has async and sequential implementations. Uses `AsyncValidatorService` by default unless `GUARDRAILS_RUN_SYNC=true`.

### Workspace Structure

This is a **uv workspace** with multiple packages:
- Root package: `guardrails-ai` (the main library)
- `packages/guardrails-api/` - The Guardrails API server

### Key Directories

- `guardrails/` - Main library source
  - `classes/` - Core data classes (history, validation results, schema)
  - `cli/` - CLI commands (`guardrails configure`, `guardrails hub install`, etc.)
  - `integrations/` - LangChain, LlamaIndex integrations
  - `telemetry/` - OpenTelemetry tracing
- `tests/unit_tests/` - Unit tests (fast, mocked)
- `tests/integration_tests/` - Integration tests (may require external services)

### Validation Flow

1. `Guard.__call__()` or `Guard.parse()` is invoked
2. If `settings.use_server` is True and model is supported, calls server API
3. Otherwise, `Runner` is instantiated with validation config
4. Runner calls LLM (if provided), then validates output
5. `validator_service.validate()` runs validators against the output
6. On failure with reask enabled, generates reask prompt and loops
7. Returns `ValidationOutcome` with validated output or error info

### Type System

- `OT` (OutputType) - Generic type parameter for Guard output
- `OutputTypes` enum - STRING, LIST, DICT for schema types
- `ValidatorMap` - Dict mapping JSON paths to validators

## Testing Notes

- Tests use `pytest` with `pytest-asyncio` for async tests
- Hub telemetry is mocked by default via `tests/conftest.py` fixtures
- Use `@pytest.mark.no_hub_telemetry_mock` to disable telemetry mocking
- Environment: `OPENAI_API_KEY=mocked` is set automatically in tests

## Configuration

- `.guardrailsrc` - User config file (API keys, telemetry settings)
- `guardrails configure` - CLI to set up configuration
- `settings.use_server` - Toggle between local and server execution
