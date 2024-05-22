import os
import sys
import uuid
from os.path import expanduser
from typing import Optional

from guardrails.classes.credentials import Credentials
from guardrails.cli.server.hub_client import AuthenticationError, get_auth
import typer

from guardrails.cli.guardrails import guardrails
from guardrails.cli.logger import LEVELS, logger


DEFAULT_TOKEN = ""
DEFAULT_ENABLE_METRICS = True


def save_configuration_file(
    token: Optional[str], enable_metrics: Optional[bool]
) -> None:
    if token is None:
        token = DEFAULT_TOKEN
    if enable_metrics is None:
        enable_metrics = DEFAULT_ENABLE_METRICS

    home = expanduser("~")
    guardrails_rc = os.path.join(home, ".guardrailsrc")
    with open(guardrails_rc, "w") as rc_file:
        lines = [
            f"id={str(uuid.uuid4())}{os.linesep}",
            f"token={token}{os.linesep}",
            f"enable_metrics={str(enable_metrics).lower()}",
        ]
        rc_file.writelines(lines)
        rc_file.close()


def _get_default_token() -> str:
    """Get the default token from the configuration file."""
    file_token = Credentials.from_rc_file(logger).token
    if file_token is None:
        return ""
    return file_token


@guardrails.command()
def configure(
    token: Optional[str] = typer.Option(
        default_factory=_get_default_token,
        help="Your Guardrails Hub auth token.",
        hide_input=True,
        prompt="Token (optional)",
    ),
    enable_metrics: Optional[bool] = typer.Option(
        DEFAULT_ENABLE_METRICS,
        "--enable-metrics/--disable-metrics",
        help="Opt out of anonymous metrics collection.",
        prompt="Enable anonymous metrics reporting?",
    ),
    clear_token: Optional[bool] = typer.Option(
        False,
        "--clear-token",
        help="Clear the existing token from the configuration file.",
    ),
):
    if clear_token is True:
        token = DEFAULT_TOKEN
    try:
        save_configuration_file(token, enable_metrics)
        logger.info("Configuration saved.")

        if not token:
            logger.info("No token provided. Skipping authentication.")
    except Exception as e:
        logger.error("An unexpected error occured!")
        logger.error(e)
        sys.exit(1)

        # Authenticate with the Hub if token is not empty
    if token != "" and token is not None:
        logger.info("Validating credentials...")
        try:
            get_auth()
            success_message = """
            Login successful.

            Get started by installing our RegexMatch validator:
            https://hub.guardrailsai.com/validator/guardrails_ai/regex_match

            You can install it by running:
            guardrails hub install hub://guardrails/regex_match

            Find more validators at https://hub.guardrailsai.com
            """
            logger.log(level=LEVELS.get("SUCCESS", 25), msg=success_message)
        except AuthenticationError as e:
            logger.error(e)
            # We do not want to exit the program if the user fails to authenticate
            # instead, save the token and other configuration options
