"""Shared deprecation helpers for the `guardrails hub` CLI command group.

The hub CLI and the private validator registry are deprecated. Validators are
now published to public PyPI as ``guardrails-ai-<name>`` and should be managed
with ``pip`` directly. These helpers emit a ``DeprecationWarning`` and print a
user-facing pointer to the equivalent ``pip`` command.
"""

import warnings
from typing import Optional

from guardrails.hub.validator_package_service import ValidatorPackageService


def pip_name_for_uri(package_uri: str) -> Optional[str]:
    """Translate a ``hub://<namespace>/<name>`` URI to its public PyPI dist name.

    Returns ``None`` if the URI can't be parsed into a validator id.
    """
    try:
        validator_id, _ = ValidatorPackageService.get_validator_id(package_uri)
        return ValidatorPackageService.get_normalized_package_name(validator_id)
    except Exception:
        return None


def warn_hub_cli_deprecated(console=None, pip_hint: Optional[str] = None) -> None:
    """Emit the standard hub-CLI deprecation warning.

    Args:
        console: Optional rich console for a user-facing message.
        pip_hint: Optional ``pip install ...`` command to recommend.
    """
    message = (
        "The `guardrails hub` CLI and the private Guardrails validator registry "
        "are deprecated and will be removed in a future major release. "
        "Validators are now published to public PyPI as `guardrails-ai-<name>`; "
        "manage them with `pip` directly."
    )
    if pip_hint:
        message = f"{message} Use `{pip_hint}` instead."

    warnings.warn(message, DeprecationWarning, stacklevel=2)
    if console is not None:
        console.print(f"[yellow]DeprecationWarning:[/yellow] {message}")
