import os
import sys
import uuid
from os.path import expanduser
from typing import Optional

import typer

from guardrails.cli.guardrails import guardrails
from guardrails.cli.logger import LEVELS, logger
from guardrails.cli.server.hub_client import AuthenticationError, get_auth


def save_configuration_file(token: str, no_metrics: bool) -> None:
    home = expanduser("~")
    guardrails_rc = os.path.join(home, ".guardrailsrc")
    with open(guardrails_rc, "w") as rc_file:
        lines = [
            f"id={str(uuid.uuid4())}{os.linesep}",
            f"token={token}{os.linesep}",
            f"no_metrics={str(no_metrics).lower()}",
        ]
        rc_file.writelines(lines)
        rc_file.close()


def get_existing_config() -> dict:
    """Get the configuration from the file if it exists."""
    home = expanduser("~")
    guardrails_rc = os.path.join(home, ".guardrailsrc")
    config = {}

    # If the file exists
    if os.path.exists(guardrails_rc):
        with open(guardrails_rc, "r") as rc_file:
            lines = rc_file.readlines()
            for line in lines:
                key, value = line.strip().split("=")
                config[key] = value
    return config


@guardrails.command()
def configure(
    token: Optional[str] = typer.Option(
        None,
        help="Your Guardrails Hub auth token.",
        hide_input=True,
        prompt="Token (optional) [None]",
    ),
    no_metrics: Optional[str] = typer.Option(
        None,
        help="Opt out of anonymous metrics collection.",
        prompt="Disable anonymous metrics reporting? [Yes/No]",
    ),
):
    """Set the global configuration for the Guardrails CLI and Hub."""
    # Get the existing configuration if present

    headless = token is not None or no_metrics is not None
    existing_config = get_existing_config()

    existing_token = existing_config.get("token", "")
    existing_no_metrics = existing_config.get("no_metrics", "false")

    if not headless:
        notice_message = """

    You can find your token at https://hub.guardrailsai.com/tokens
    """
        logger.log(level=LEVELS.get("NOTICE"), msg=notice_message)  # type: ignore

    # Prompt for token if not provided
    if not token and not headless:
        token = typer.prompt(
            "> Token (optional) [None]",
            default=existing_token,
            hide_input=True,
        )

    # Prompt for no_metrics if not provided and not running headless
    if no_metrics is None and not headless:
        no_metrics = typer.prompt(
            "> Disable anonymous metrics reporting?",
            default=existing_no_metrics,
        )

    try:
        # If token or no_metrics was updated, save the configuration
        token_was_updated = token and token != existing_token
        no_metrics_was_updated = no_metrics != existing_no_metrics
        if token_was_updated or no_metrics_was_updated:
            logger.info("Configuring...")
            save_configuration_file(token, no_metrics)

            # Authenticate with the Hub if token was updated
        if token_was_updated:
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
            logger.log(level=LEVELS.get("SUCCESS"), msg=success_message)  # type: ignore`
        elif not token:
            print("No token provided. Skipping authentication.")

        if not headless and not token_was_updated:
            print("Existing token found. Skipping re-authentication.")
    except AuthenticationError as auth_error:
        logger.error(auth_error)
        logger.error(
            """
            Check that your token is correct and try again.

            If you don't have your token credentials you can find them here:

            https://hub.guardrailsai.com/tokens
            """
        )
        sys.exit(1)
    except Exception as e:
        logger.error("An unexpected error occurred!")
        logger.error(e)
        sys.exit(1)
