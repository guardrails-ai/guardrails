import json

from guardrails.cli.hub.hub import hub_command
from guardrails.hub_telemetry.hub_tracing import trace
from .console import console


@hub_command.command(name="list")
@trace(name="guardrails-cli/hub/list")
def list():
    """List all installed validators."""
    from guardrails.hub.validator_package_service import ValidatorPackageService

    registry_path = ValidatorPackageService.get_registry_path()

    if not registry_path.exists():
        console.print("No validators installed.")
        return

    try:
        registry = json.loads(registry_path.read_text())
    except (json.JSONDecodeError, OSError):
        console.print("No validators installed.")
        return

    validators = registry.get("validators", {})
    if not validators:
        console.print("No validators installed.")
        return

    console.print("Installed Validators:")
    for validator_id, entry in sorted(validators.items()):
        exports = ", ".join(entry.get("exports", []))
        console.print(f"- {validator_id} ({exports})")
