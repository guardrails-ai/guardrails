"""Nox configuration for multi-version testing.

Run with: `uv run nox`
"""

import nox

PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13", "3.14"]

nox.options.sessions = ["tests"]


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run the test suite across Python versions."""
    session.install("-e", ".")
    session.install("pytest", "pytest-mock", "coverage")
    session.run("pytest", "tests/", "-v", *session.posargs)


@nox.session(python=["3.12"])
def lint(session: nox.Session) -> None:
    """Run linting checks."""
    session.install("ruff")
    session.run("ruff", "check", "guardrails_api/", "tests/")
    session.run("ruff", "format", "--check", "guardrails_api/", "tests/")


@nox.session(python=["3.12"])
def typecheck(session: nox.Session) -> None:
    """Run type checking."""
    session.install("-e", ".")
    session.install("ty")
    session.run("ty", "check", "guardrails_api/")
