import os
import subprocess
import sys
from typing import List

import typer
from pydash.strings import snake_case

from guardrails.cli.logger import logger
from guardrails.cli.server.module_manifest import ModuleManifest
from .hub import hub_command
from .console import console

def pip_process(action: str, package: str, flags: List[str] = []):
    try:
        command = [sys.executable, "-m", "pip", action] + flags + [package]
        output = subprocess.check_output(command, stderr=subprocess.STDOUT)
        return output.decode()
    except subprocess.CalledProcessError as exc:
        logger.error(f"Failed to {action} {package}\nExit code: {exc.returncode}\nOutput: {exc.output.decode()}")
        sys.exit(exc.returncode)

def uninstall_validator(package_uri: str):
    """Function to handle the uninstallation of a validator."""
    try:
        # Assuming the package is installed and can be directly referenced by its name
        output = pip_process("uninstall", package_uri, ["--yes"])
        logger.info(f"Uninstalled {package_uri}: {output}")
    except Exception as e:
        logger.error(f"An error occurred while uninstalling {package_uri}: {str(e)}")
        sys.exit(1)

@hub_command.command()
def uninstall(package_uri: str = typer.Argument(..., help="URI of the package to uninstall.")):
    """Uninstall a validator from the Hub."""
    if not package_uri.startswith("hub://"):
        logger.error("Invalid URI!")
        sys.exit(1)

    package_uri_clean = package_uri.replace("hub://", "")
    logger.log(level=logger.LEVELS.get("SPAM"), msg=f"Uninstalling {package_uri_clean}...")

    with console.status(f"Uninstalling {package_uri_clean}", spinner="bouncingBar"):
        uninstall_validator(package_uri_clean)

    console.print(f"\n Successfully uninstalled {package_uri_clean}!\n")

if __name__ == "__main__":
    typer.run(uninstall)
