import os
import uuid
from os.path import expanduser
from typing import Optional

from guardrails.cli.server.hub_client import get_auth
import typer

from guardrails.cli.guardrails import guardrails
from guardrails.cli.logger import LEVELS, logger


DEFAULT_TOKEN = ""
DEFAULT_NO_METRICS = False


def save_configuration_file(token: Optional[str], no_metrics: Optional[bool]) -> None:
    if token is None:
        token = DEFAULT_TOKEN
    if no_metrics is None:
        no_metrics = DEFAULT_NO_METRICS

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


def _get_default_token() -> str:
    """Get the default token from the configuration file."""
    return get_existing_config().get("token", DEFAULT_TOKEN)


@guardrails.command()
def configure(
    token: Optional[str] = typer.Option(
        default_factory=_get_default_token,
        help="Your Guardrails Hub auth token.",
        hide_input=True,
        prompt="Token (optional)",
    ),
    no_metrics: Optional[bool] = typer.Option(
        DEFAULT_NO_METRICS,
        "--no-metrics/--metrics",
        help="Opt out of anonymous metrics collection.",
        prompt="Disable anonymous metrics reporting?",
    ),
    clear_token: Optional[bool] = typer.Option(
        False,
        "--clear-token",
        help="Clear the existing token from the configuration file.",
    ),
):
    if clear_token is True:
        token = DEFAULT_TOKEN

    # Authenticate with the Hub if token is not empty
    if token != "" and token is not None:
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
        logger.log(level=LEVELS.get("SUCCESS", 25), msg=success_message)

    save_configuration_file(token, no_metrics)
    logger.info("Configuration saved.")

    if not token:
        print("No token provided. Skipping authentication.")
