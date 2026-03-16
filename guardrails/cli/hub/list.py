from guardrails.cli.hub.hub import hub_command
from guardrails.hub.registry import get_registry
from guardrails.hub_telemetry.hub_tracing import trace
from .console import console


@hub_command.command(name="list")
@trace(name="guardrails-cli/hub/list")
def list():
    """List all installed validators."""
    registry = get_registry()

    validators = registry.validators
    if not validators:
        console.print("No validators installed.")
        return

    console.print("Installed Validators:")
    for validator_id, entry in sorted(validators.items()):
        exports = ", ".join(entry.exports)
        console.print(f"- {validator_id} ({exports})")
