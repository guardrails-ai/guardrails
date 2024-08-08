import sys
from typing import Optional

import typer

from guardrails.cli.hub.hub import hub_command
from guardrails.cli.logger import logger


@hub_command.command()
def install(
    package_uri: str = typer.Argument(
        help="URI to the package to install.\
Example: hub://guardrails/regex_match."
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
):
    try:
        from guardrails.hub.install import install

        def confirm():
            return typer.confirm(
                "This validator has a Guardrails AI inference endpoint available. "
                "Would you still like to install the"
                " local models for local inference?",
            )

        install(
            package_uri,
            install_local_models=local_models,
            quiet=quiet,
            install_local_models_confirm=confirm,
        )
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)
