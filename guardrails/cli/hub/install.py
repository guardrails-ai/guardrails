import sys
from typing import Optional, List

import typer

from guardrails.cli.hub.hub import hub_command
from guardrails.cli.logger import logger
from guardrails.cli.telemetry import trace_if_enabled


@hub_command.command()
def install(
    package_uris: List[str] = typer.Argument(
        ...,
        help="URIs to the packages to install. Example: hub://guardrails/regex_match hub://guardrails/toxic_language",
    ),
    local_models: Optional[bool] = typer.Option(
        None,
        "--install-local-models/--no-install-local-models",
        help="Install local models",
    ),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Run the command in quiet mode to reduce output verbosity.",
    ),
    upgrade: bool = typer.Option(
        False, "--upgrade", help="Upgrade the package to the latest version."
    ),
):
    try:
        trace_if_enabled("hub/install")
        from guardrails.hub.install import install_multiple

        def confirm():
            return typer.confirm(
                "This validator has a Guardrails AI inference endpoint available. "
                "Would you still like to install the"
                " local models for local inference?",
            )

        install_multiple(
            package_uris,
            install_local_models=local_models,
            quiet=quiet,
            upgrade=upgrade,
            install_local_models_confirm=confirm,
        )
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)
