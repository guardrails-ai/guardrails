from typing import Optional
import typer

from guardrails.cli.guardrails import guardrails
from guardrails.cli.hub.utils import pip_process
from guardrails.cli.logger import logger
from guardrails.cli.telemetry import trace_if_enabled
from guardrails.cli.version import version_warnings_if_applicable
from guardrails.cli.hub.console import console


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
):
    logger.debug("Checking for prerequisites...")
    if not api_is_installed():
        package_name = 'guardrails-api>="^0.0.0a0"'
        pip_process("install", package_name)

    from guardrails_api.cli.start import start  # type: ignore

    logger.info("Starting Guardrails server")
    version_warnings_if_applicable(console)
    trace_if_enabled("start")
    start(env, config, port)
