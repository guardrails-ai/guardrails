import sys
from typing import Optional, List

import typer

from guardrails.cli.hub.hub import hub_command
from guardrails.cli.logger import logger
from guardrails.hub_telemetry.hub_tracing import trace
from guardrails.cli.hub.console import console
from guardrails.cli.version import version_warnings_if_applicable


# Quick note: This is the command for `guardrails hub install`.  We change the name of
# the function def to prevent confusion, lest people import it directly and calling it
# with a string for package_uris instead of a list, which behaves oddly. If you need to
# call install from a script, please consider importing install from guardrails,
# not guardrails.cli.hub.install.
@hub_command.command(name="install")
@trace(name="guardrails-cli/hub/install")
def install_cli(
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
        if isinstance(package_uris, str):
            logger.error(
                f"`install` in {__file__} was called with a string instead of "
                "a list! This can happen if it is invoked directly instead of "
                "being run via the CLI. Did you mean to import `from guardrails import "
                "install` instead?  Recovering..."
            )
            package_uris = [
                package_uris,
            ]

        from guardrails.hub.install import install_multiple

        def confirm():
            return typer.confirm(
                "This validator has a Guardrails AI inference endpoint available. "
                "Would you still like to install the"
                " local models for local inference?",
            )

        version_warnings_if_applicable(console)

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
