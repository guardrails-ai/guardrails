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
    existing_config = get_existing_config()

    # Normalize no_metrics to bool
    if no_metrics is not None:
        no_metrics_bool = no_metrics.lower() == 'yes'
    else:
        no_metrics_bool = existing_config.get("no_metrics", "false") == "true"

    # Fetch existing token if None provided
    if token is None:
        token = existing_config.get("token", "")

    # Only save configuration if both token and no_metrics are valid
    if token and no_metrics is not None:
        save_configuration_file(token, no_metrics_bool)
        logger.info("Configuration saved.")

        # Authenticate with the Hub if token was updated
        if token != existing_config.get("token", ""):
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
            logger.log(level=LEVELS.get("SUCCESS", 25), msg=success_message)  # Assuming 25 is the SUCCESS level

    else:
        if not token:
            print("No token provided. Skipping authentication.")
        if no_metrics is None:
            print("No metrics preference provided. Skipping configuration update.")

    # Log an information message if neither token nor no_metrics provided
    if not token and no_metrics is None:
        logger.info("No updates to configuration required.")