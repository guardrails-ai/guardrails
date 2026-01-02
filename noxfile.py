"""Nox configuration for multi-version testing.

Run with: `uv run nox` or `mise run nox`
"""

import nox

# Test against all supported Python versions
PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13", "3.14"]

# Default sessions to run
nox.options.sessions = ["tests"]


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run the test suite across Python versions."""
    session.install("-e", ".")
    session.install("pytest", "pytest-mock", "pytest-cov", "pytest-asyncio")
    session.run(
        "pytest",
        "tests/unit_tests/",
        "-v",
        "--tb=short",
        *session.posargs,
    )


@nox.session(python=["3.12"])
def lint(session: nox.Session) -> None:
    """Run linting checks."""
    session.install("ruff")
    session.run("ruff", "check", "guardrails/", "tests/")
    session.run("ruff", "format", "--check", "guardrails/", "tests/")


@nox.session(python=["3.12"])
def typecheck(session: nox.Session) -> None:
    """Run type checking."""
    session.install("-e", ".")
    session.install("ty")
    session.run("ty", "check", "guardrails/")


@nox.session(python=["3.12"])
def docs(session: nox.Session) -> None:
    """Build documentation."""
    session.install("-e", ".[docs]")
    session.run("mkdocs", "build", "--strict")


@nox.session(python=["3.12"])
def liccheck(session: nox.Session) -> None:
    """Check license compliance."""
    session.install("-e", ".")
    session.install("liccheck")
    session.run("liccheck", "-s", "pyproject.toml")
