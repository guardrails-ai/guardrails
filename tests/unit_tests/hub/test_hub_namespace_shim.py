"""Tests for the `guardrails.hub` back-compat namespace shim (WS-G.3).

After a validator is installed as a public `guardrails-ai-<name>` package
(importable as `guardrails_ai.<name>`), `from guardrails.hub import X` must keep
working by falling back to scanning the `guardrails_ai.*` namespace, while
emitting a one-time DeprecationWarning.
"""

import sys
import warnings

import pytest

import guardrails.hub as hub_module


@pytest.fixture
def fake_namespace_package(tmp_path, monkeypatch):
    """Create an installed `guardrails_ai.faketest` namespace package on disk."""
    ns_dir = tmp_path / "guardrails_ai" / "faketest"
    ns_dir.mkdir(parents=True)
    # PEP 420 namespace: no guardrails_ai/__init__.py
    (ns_dir / "__init__.py").write_text("class FakeValidator:\n    pass\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    # Ensure a clean import + reset the shim's caches/flags.
    for mod in list(sys.modules):
        if mod == "guardrails_ai" or mod.startswith("guardrails_ai."):
            del sys.modules[mod]
    hub_module._export_map_cache = None
    hub_module._namespace_deprecation_warned = False

    yield

    for mod in list(sys.modules):
        if mod == "guardrails_ai" or mod.startswith("guardrails_ai."):
            del sys.modules[mod]
    hub_module._export_map_cache = None
    hub_module._namespace_deprecation_warned = False


def test_shim_resolves_from_guardrails_ai_namespace(fake_namespace_package):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved = hub_module.__getattr__("FakeValidator")

    assert resolved.__name__ == "FakeValidator"
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_shim_unknown_export_raises_attribute_error(fake_namespace_package):
    with pytest.raises(AttributeError):
        hub_module.__getattr__("DoesNotExistValidator")


def test_shim_returns_none_without_namespace(monkeypatch):
    """If no guardrails_ai namespace is installed, lookups raise AttributeError."""
    for mod in list(sys.modules):
        if mod == "guardrails_ai" or mod.startswith("guardrails_ai."):
            del sys.modules[mod]
    monkeypatch.setitem(sys.modules, "guardrails_ai", None)
    hub_module._export_map_cache = None
    with pytest.raises(AttributeError):
        hub_module.__getattr__("AnythingAtAll")
