import json
import os
import re
import sysconfig
import sys
from datetime import datetime, timezone
from pathlib import Path

import typer

from guardrails.cli.hub.hub import hub_command
from guardrails.cli.logger import logger
from guardrails.hub_telemetry.hub_tracing import trace
from guardrails.cli.hub.console import console
from guardrails.cli.version import version_warnings_if_applicable


@hub_command.command(name="migrate")
@trace(name="guardrails-cli/hub/migrate")
def migrate_cli(
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Run the command in quiet mode to reduce output verbosity.",
    ),
):
    """Migrate hub validators from __init__.py barrel file to project-level registry.

    Reads the current hub/__init__.py from site-packages, parses its import
    statements, and writes them to .guardrails/hub_registry.json in the
    project root.
    """
    try:
        version_warnings_if_applicable(console)
        registry = migrate_registry(quiet=quiet)
        if registry:
            count = len(registry.get("validators", {}))
            if not quiet:
                console.print(
                    f"[green]Migrated {count} validator(s) to "
                    f".guardrails/hub_registry.json[/green]"
                )
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)


def migrate_registry(quiet: bool = False) -> dict:
    """Migrate barrel imports from hub/__init__.py to project-level registry.

    Reads the current hub/__init__.py from site-packages, parses its import
    statements, and writes them to .guardrails/hub_registry.json in the
    project root.

    Returns:
        The registry dict, or empty dict if nothing to migrate.
    """
    site_packages = sysconfig.get_paths()["purelib"]
    hub_init = Path(site_packages) / "guardrails" / "hub" / "__init__.py"

    if not hub_init.exists():
        if not quiet:
            logger.info(
                "No hub/__init__.py found in site-packages. Nothing to migrate."
            )
        return {}

    content = hub_init.read_text()
    registry = {"version": 1, "validators": {}}

    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        match = re.match(r"from\s+(\w+_grhub_\w+)\s+import\s+(.+)", line)
        if match:
            import_path = match.group(1)
            exports = [e.strip() for e in match.group(2).split(",")]

            # Reverse the naming convention to get the validator_id:
            # guardrails_grhub_detect_pii -> guardrails/detect_pii
            # guardrails_ai_grhub_id -> guardrails-ai/id
            grhub_idx = import_path.find("_grhub_")
            if grhub_idx == -1:
                continue
            namespace_part = import_path[:grhub_idx].replace("_", "-")
            name_part = import_path[grhub_idx + len("_grhub_") :]
            validator_id = f"{namespace_part}/{name_part}"
            package_name = import_path.replace("_", "-")

            registry["validators"][validator_id] = {
                "import_path": import_path,
                "exports": exports,
                "installed_at": datetime.now(timezone.utc).isoformat(),
                "package_name": package_name,
            }

    if not registry["validators"]:
        if not quiet:
            logger.info(
                "No hub validator imports found in __init__.py. Nothing to migrate."
            )
        return {}

    registry_dir = Path(os.getcwd()) / ".guardrails"
    registry_dir.mkdir(parents=True, exist_ok=True)
    registry_file = registry_dir / "hub_registry.json"
    registry_file.write_text(json.dumps(registry, indent=2))

    return registry
