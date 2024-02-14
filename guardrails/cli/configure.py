import os
import sys
import uuid
from os.path import expanduser
from typing import Optional

import typer

from guardrails.cli.guardrails import guardrails
from guardrails.cli.logger import LEVELS, logger
from guardrails.cli.server.hub_client import AuthenticationError, get_auth


def save_configuration_file(
    client_id: str, client_secret: str, no_metrics: bool
) -> None:
    home = expanduser("~")
    guardrails_rc = os.path.join(home, ".guardrailsrc")
    with open(guardrails_rc, "w") as rc_file:
        lines = [
            f"id={str(uuid.uuid4())}{os.linesep}",
            f"client_id={client_id}{os.linesep}",
            f"client_secret={client_secret}{os.linesep}",
            f"no_metrics={str(no_metrics).lower()}{os.linesep}",
        ]
        rc_file.writelines(lines)
        rc_file.close()


@guardrails.command()
def configure(
    client_id: Optional[str] = typer.Option(
        help="Your Guardrails Hub client ID.", default=""
    ),
    client_secret: Optional[str] = typer.Option(
        help="Your Guardrails Hub client secret.", hide_input=True, default=""
    ),
    no_metrics: Optional[str] = typer.Option(
        help="Opt out of anonymous metrics collection.", default=False
    ),
):
    """Set the global configuration for the Guardrails CLI and Hub."""
    try:
        notice_message = """

    You can find your tokens at https://hub.guardrailsai.com/tokens
    """
        logger.log(level=LEVELS.get("NOTICE"), msg=notice_message)  # type: ignore
        if not client_id:
            client_id = typer.prompt("Client ID")
        if not client_secret:
            client_secret = typer.prompt("Client secret", hide_input=True)
        logger.info("Configuring...")
        save_configuration_file(client_id, client_secret, no_metrics)  # type: ignore

        logger.info("Validating credentials...")
        get_auth()
        success_message = """

    Login successful.

    Get started by installing our RegexMatch validator:
    https://hub.guardrailsai.com/validator/guardrails_ai/regex_match

    You can install it by running:
    guardrails hub install hub://guardrails/regex_match

    Find more validators at https://hub.guardrailsai.com
    """
        logger.log(level=LEVELS.get("SUCCESS"), msg=success_message)  # type: ignore
    except AuthenticationError as auth_error:
        logger.error(auth_error)
        logger.error(
            """
            Check that your Client ID and Client secret are correct and try again.

            If you don't have your token credentials you can find them here:

            https://hub.guardrailsai.com/tokens
            """
        )
        sys.exit(1)
    except Exception as e:
        logger.error("An unexpected error occurred!")
        logger.error(e)
        sys.exit(1)
