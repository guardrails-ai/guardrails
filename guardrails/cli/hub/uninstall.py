import os
import sys
from typing import List, Literal

import typer

from guardrails.cli.hub.hub import hub_command
from guardrails.cli.logger import LEVELS, logger
from guardrails.cli.server.hub_client import get_validator_manifest
from guardrails_hub_types import Manifest

from guardrails.cli.hub.utils import pip_process
from guardrails.hub_telemetry.hub_tracing import trace

from .console import console

json_format: Literal["json"] = "json"
string_format: Literal["string"] = "string"


def remove_line(file_path: str, line_content: str):
    with open(file_path, "r+") as file:
        lines = file.readlines()
        file.seek(0)
        lines = [line for line in lines if line.strip() != line_content.strip()]
        file.writelines(lines)
        file.truncate()
        return lines


def remove_from_hub_inits(manifest: Manifest, site_packages: str):
    from guardrails.hub.validator_package_service import ValidatorPackageService

    exports: List[str] = manifest.exports or []
    sorted_exports = sorted(exports, reverse=True)

    validator_id = manifest.id
    import_path = ValidatorPackageService.get_import_path_from_validator_id(
        validator_id
    )
    import_line = f"from {import_path} import {', '.join(sorted_exports)}"

    # Remove import line from main __init__.py
    hub_init_location = os.path.join(site_packages, "guardrails", "hub", "__init__.py")
    remove_line(hub_init_location, import_line)


def uninstall_hub_module(manifest: Manifest):
    from guardrails.hub.validator_package_service import ValidatorPackageService

    validator_id = manifest.id
    package_name = ValidatorPackageService.get_normalized_package_name(validator_id)
    pip_process("uninstall", package_name, flags=["-y"], quiet=True)


@hub_command.command()
@trace(name="guardrails-cli/hub/uninstall")
def uninstall(
    package_uri: str = typer.Argument(
        help="URI to the package to uninstall. Example: hub://guardrails/regex_match."
    ),
):
    """Uninstall a validator from the Hub."""
    from guardrails.hub.validator_package_service import ValidatorPackageService

    if not package_uri.startswith("hub://"):
        logger.error("Invalid URI!")
        sys.exit(1)

    console.print(f"\nUninstalling {package_uri}...\n")
    logger.log(
        level=LEVELS.get("SPAM", 0),
        msg=f"Uninstalling {package_uri}...",
    )

    # Validation
    module_name = package_uri.replace("hub://", "")

    # Prep
    with console.status("Fetching manifest", spinner="bouncingBar"):
        module_manifest = get_validator_manifest(module_name)
        site_packages = ValidatorPackageService.get_site_packages_location()

    # Uninstall
    with console.status("Removing module", spinner="bouncingBar"):
        uninstall_hub_module(module_manifest)

    # Cleanup
    with console.status("Cleaning up", spinner="bouncingBar"):
        remove_from_hub_inits(module_manifest, site_packages)

    console.print("✅ Successfully uninstalled!")  # type: ignore
    logger.log(level=LEVELS.get("SPAM"), msg="✅ Successfully uninstalled!")  # type: ignore
