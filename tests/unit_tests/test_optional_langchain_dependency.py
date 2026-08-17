"""guardrails must import and construct a Guard without langchain-core installed.

langchain-core moved from a core dependency to the `langchain` extra (see
`Guard.to_runnable` / `Validator.to_runnable`), so `import guardrails` cannot
require it. These checks run in a subprocess with langchain_core's import
blocked, since the package is genuinely installed in the test environment
(via the `langchain` extra) and any eager top-level import would otherwise
go unnoticed here.
"""

import subprocess
import sys
import textwrap
from typing import TypeVar

import guardrails.classes as classes
from guardrails.integrations.langchain.base_runnable import InputType


def _run_with_langchain_core_blocked(script: str) -> subprocess.CompletedProcess:
    setup = """
        import builtins

        _real_import = builtins.__import__

        def _blocked_import(name, *args, **kwargs):
            if name == "langchain_core" or name.startswith("langchain_core."):
                raise ModuleNotFoundError(f"No module named {name!r}")
            return _real_import(name, *args, **kwargs)

        builtins.__import__ = _blocked_import
    """
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(setup) + textwrap.dedent(script)],
        capture_output=True,
        text=True,
    )


def test_guard_import_and_construction_without_langchain_core():
    result = _run_with_langchain_core_blocked(
        """
        from guardrails import Guard
        Guard()
        print("OK")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_to_runnable_still_raises_clearly_without_langchain_core():
    result = _run_with_langchain_core_blocked(
        """
        from guardrails import Guard
        try:
            Guard().to_runnable()
        except ModuleNotFoundError:
            print("RAISED")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "RAISED" in result.stdout


def test_input_type_no_longer_exported_from_classes():
    assert not hasattr(classes, "InputType")
    assert "InputType" not in classes.__all__


def test_input_type_lives_with_its_only_consumer():
    assert isinstance(InputType, TypeVar)
    assert InputType.__constraints__[0] is str
