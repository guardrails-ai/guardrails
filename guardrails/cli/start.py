from typing import Optional
import typer

from guardrails.cli.guardrails import guardrails
from guardrails.cli.hub.utils import pip_process
from guardrails.cli.logger import logger
from guardrails.cli.telemetry import trace_if_enabled
from guardrails.cli.version import version_warnings_if_applicable
from guardrails.cli.hub.console import console
from guardrails.settings import settings


def api_is_installed() -> bool:
    try:
        import guardrails_api  # type: ignore  # noqa

        return True
    except ImportError:
        return False


@guardrails.command()
def start(
    env: Optional[str] = typer.Option(
        default="",
        help="An env file to load environment variables from.",
    ),
    config: Optional[str] = typer.Option(
        default="",
        help="A config file to load Guards from.",
    ),
    port: Optional[int] = typer.Option(
        default=8000,
        help="The port to run the server on.",
    ),
    watch: bool = typer.Option(
        default=False, is_flag=True, help="Enable watch mode for logs."
    ),
):
    logger.debug("Checking for prerequisites...")
    if not api_is_installed():
        package_name = "guardrails-api>=0.2.1"
        pip_process("install", package_name)

    from guardrails_api.cli.start import start as start_api  # type: ignore

    logger.info("Starting Guardrails server")

    if watch:
        settings._watch_mode_enabled = True

    version_warnings_if_applicable(console)
    trace_if_enabled("start")
    start_api(env, config, port)
