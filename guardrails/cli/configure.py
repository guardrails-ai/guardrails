import os
import sys
import uuid
from os.path import expanduser
from typing import Optional

import typer

from guardrails.settings import settings
from guardrails.cli.guardrails import guardrails
from guardrails.cli.logger import LEVELS, logger
from guardrails.cli.hub.console import console
from guardrails.cli.server.hub_client import AuthenticationError, get_auth
from guardrails.cli.telemetry import trace_if_enabled
from guardrails.cli.version import version_warnings_if_applicable

DEFAULT_TOKEN = ""
DEFAULT_ENABLE_METRICS = True
DEFAULT_USE_REMOTE_INFERENCING = True


def save_configuration_file(
    token: Optional[str],
    enable_metrics: Optional[bool],
    use_remote_inferencing: Optional[bool] = DEFAULT_USE_REMOTE_INFERENCING,
) -> None:
    if token is None:
        token = DEFAULT_TOKEN
    if enable_metrics is None:
        enable_metrics = DEFAULT_ENABLE_METRICS
    if use_remote_inferencing is None:
        use_remote_inferencing = DEFAULT_USE_REMOTE_INFERENCING

    home = expanduser("~")
    guardrails_rc = os.path.join(home, ".guardrailsrc")
    with open(guardrails_rc, "w", encoding="utf-8") as rc_file:
        lines = [
            f"id={str(uuid.uuid4())}{os.linesep}",
            f"token={token}{os.linesep}",
            f"enable_metrics={str(enable_metrics).lower()}{os.linesep}",
            f"use_remote_inferencing={str(use_remote_inferencing).lower()}",
        ]
        rc_file.writelines(lines)
        rc_file.close()


def _get_default_token() -> str:
    """Get the default token from the configuration file."""
    file_token = settings.rc.token
    if file_token is None:
        return ""
    return file_token


@guardrails.command()
def configure(
    enable_metrics: Optional[bool] = typer.Option(
        DEFAULT_ENABLE_METRICS,
        "--enable-metrics/--disable-metrics",
        help="Opt out of anonymous metrics collection.",
        prompt="Enable anonymous metrics reporting?",
    ),
    token: Optional[str] = typer.Option(
        None,
        "--token",
        help="API Key for Guardrails. If not provided, you will be prompted for it.",
    ),
    remote_inferencing: Optional[bool] = typer.Option(
        DEFAULT_USE_REMOTE_INFERENCING,
        "--enable-remote-inferencing/--disable-remote-inferencing",
        help="Opt in to remote inferencing. "
        "If not provided, you will be prompted for it.",
        prompt="Do you wish to use remote inferencing?",
    ),
    clear_token: Optional[bool] = typer.Option(
        False,
        "--clear-token",
        help="Clear the existing token from the configuration file.",
    ),
):
    version_warnings_if_applicable(console)
    if settings.rc.exists():
        trace_if_enabled("configure")
    existing_token = _get_default_token()
    last4 = existing_token[-4:] if existing_token else ""

    if not clear_token and token is None:
        console.print("\nEnter API Key below", style="bold", end=" ")

        if last4:
            console.print(
                "[dim]leave empty if you want to keep existing token[/dim]",
                style="italic",
                end=" ",
            )
            console.print(f"[{last4}]", style="italic")

        console.print(
            ":backhand_index_pointing_right: You can find your API Key at https://hub.guardrailsai.com/keys"
        )

        token = typer.prompt("\nAPI Key", existing_token, show_default=False)

    else:
        token = token or DEFAULT_TOKEN

    try:
        save_configuration_file(token, enable_metrics, remote_inferencing)
        logger.info("Configuration saved.")
    except Exception as e:
        logger.error("An unexpected error occured saving configuration!")
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
    else:
        logger.info("No token provided. Skipping authentication.")
