"""guardrails.hub - Dynamic import resolution from hub_registry.json.

Validators registered in .guardrails/hub_registry.json are resolved lazily
on first attribute access and cached for subsequent imports.
"""

import importlib

from guardrails.hub.registry import get_registry


_export_map_cache = None


def _build_export_map() -> dict:
    """Build mapping from export name to import path.

    Returns a dict mapping export names (e.g. "DetectPII") to their
    module import paths (e.g. "guardrails_grhub_detect_pii").
    """
    registry = get_registry()
    if not registry:
        return {}
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
                f"Try reinstalling: guardrails hub install "
                f"hub://<org>/<validator>"
            ) from e
    raise AttributeError(f"module 'guardrails.hub' has no attribute '{name}'")


def __dir__():
    base = list(globals().keys())
    base.extend(_get_export_map().keys())
    return base
