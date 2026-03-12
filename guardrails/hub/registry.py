import json
import os
from pathlib import Path

from guardrails.types.validator_registry import ValidatorRegistry
from guardrails.logger import logger


def get_registry_path() -> Path:
    """Return the project-level registry path."""
    return Path(os.getcwd()) / ".guardrails" / "hub_registry.json"


def get_registry() -> ValidatorRegistry | None:
    registry_file = get_registry_path()
    try:
        registry_str = registry_file.read_text()
        registry_dict = json.loads(registry_str)
        registry = ValidatorRegistry.model_validate(registry_dict)
        return registry
    except (json.JSONDecodeError, OSError):
        logger.warning("Failed to read hub registry at %s", registry_file)
        return None
