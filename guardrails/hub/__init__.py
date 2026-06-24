"""guardrails.hub - Dynamic import resolution for hub validators.

Validators registered in ``.guardrails/hub_registry.json`` are resolved lazily
on first attribute access and cached for subsequent imports.

DEPRECATED: ``from guardrails.hub import X`` is deprecated. Validators are now
published to public PyPI as ``guardrails-ai-<name>`` and should be imported
directly, e.g. ``from guardrails_ai.detect_pii import DetectPII``. As a
back-compat shim, if an export is not found in the local hub registry this
module falls back to scanning the installed ``guardrails_ai.*`` namespace
packages for it (emitting a one-time deprecation warning).
"""

import importlib
import pkgutil
import warnings

from guardrails.hub.registry import get_registry


_export_map_cache = None
_namespace_deprecation_warned = False


def _build_export_map() -> dict:
    """Build mapping from export name to import path.

    Returns a dict mapping export names (e.g. "DetectPII") to their
    module import paths (e.g. "guardrails_ai.detect_pii").
    """
    registry = get_registry()
    export_map = {}
    for entry in registry.validators.values():
        import_path = entry.import_path
        for export_name in entry.exports:
            export_map[export_name] = import_path
    return export_map


def _get_export_map() -> dict:
    """Return cached export map, building it on first access."""
    global _export_map_cache
    if _export_map_cache is None:
        _export_map_cache = _build_export_map()
    return _export_map_cache


def _resolve_from_namespace(name: str):
    """Fallback resolver: scan installed ``guardrails_ai.*`` packages.

    Returns the resolved attribute, or ``None`` if no installed
    ``guardrails_ai`` namespace package exports ``name``.
    """
    try:
        import guardrails_ai  # type: ignore
    except ImportError:
        return None

    namespace_path = getattr(guardrails_ai, "__path__", None)
    if namespace_path is None:
        return None

    for mod in pkgutil.iter_modules(namespace_path, "guardrails_ai."):
        try:
            module = importlib.import_module(mod.name)
        except Exception:
            continue
        if hasattr(module, name):
            return getattr(module, name)
    return None


def _warn_namespace_shim_once(name: str) -> None:
    global _namespace_deprecation_warned
    if _namespace_deprecation_warned:
        return
    _namespace_deprecation_warned = True
    warnings.warn(
        "Importing validators from `guardrails.hub` is deprecated and will be "
        "removed in a future major release. Import directly from the "
        f"`guardrails_ai` namespace instead, e.g. "
        f"`from guardrails_ai.<name> import {name}`.",
        DeprecationWarning,
        stacklevel=3,
    )


def __getattr__(name: str):
    export_map = _get_export_map()
    if name in export_map:
        import_path = export_map[name]
        try:
            module = importlib.import_module(import_path)
            attr = getattr(module, name)
            globals()[name] = attr
            return attr
        except (ModuleNotFoundError, AttributeError) as e:
            raise ImportError(
                f"Cannot import '{name}' from hub registry. "
                f"Module '{import_path}' not found. "
                f"Try installing it: pip install guardrails-ai-<validator>"
            ) from e

    # Back-compat fallback: resolve from the installed guardrails_ai namespace.
    attr = _resolve_from_namespace(name)
    if attr is not None:
        _warn_namespace_shim_once(name)
        globals()[name] = attr
        return attr

    raise AttributeError(f"module 'guardrails.hub' has no attribute '{name}'")


def __dir__():
    base = list(globals().keys())
    base.extend(_get_export_map().keys())
    return base
