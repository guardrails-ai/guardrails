import os
import re

from guardrails.cli.hub.hub import hub_command
from guardrails.cli.hub.utils import get_site_packages_location
from guardrails.cli.telemetry import trace_if_enabled
from .console import console


@hub_command.command(name="list")
def list():
    """List all installed validators."""
    trace_if_enabled("hub/list")
    site_packages = get_site_packages_location()
    hub_init_file = os.path.join(site_packages, "guardrails", "hub", "__init__.py")

    installed_validators = []

    if os.path.isfile(hub_init_file):
        with open(hub_init_file, "r") as file:
            content = file.read()
            matches = re.findall(r"from .* import (\w+)", content)
            installed_validators.extend(matches)

    if installed_validators:
        console.print("Installed Validators:")
        for validator in installed_validators:
            console.print(f"- {validator}")
    else:
        console.print("No validators installed.")
