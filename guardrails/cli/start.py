import typer
from importlib.metadata import version

from guardrails.cli.guardrails import guardrails
from guardrails.cli.hub.utils import installer_process
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
    env: str = typer.Option(
        default="",
        help="An env file to load environment variables from.",
    ),
    config: str = typer.Option(
        default="",
        help="A config file to load Guards from.",
    ),
    port: int = typer.Option(
        default=8000,
        help="The port to run the server on.",
    ),
    watch: bool = typer.Option(
        default=False, is_flag=True, help="Enable watch mode for logs."
    ),
    env_override: bool = typer.Option(
        default=False,
        help="Override existing environment variables with values from the env file.",
    ),
):
    logger.debug("Checking for prerequisites...")
    if not api_is_installed():
        package_name = "guardrails-api>=0.2.1"
        installer_process("install", package_name)

    from guardrails_api.cli.start import start as start_api  # type: ignore

    guardrails_api_version = version("guardrails_api")

    major, minor, *_ = guardrails_api_version.split(".")

    logger.info("[INFO]: Starting Guardrails server")

    if watch:
        settings._watch_mode_enabled = True

    version_warnings_if_applicable(console)
    trace_if_enabled("start")

    if major == "0" and int(minor) < 3:
        if env_override:
            logger.warning(
                "[WARNING]: 'env_override' is only supported for guardrails-api>=0.3.0."
                f"  You have guardrails-api=={guardrails_api_version}."
                "  'env_override' will be ignored."
            )

        start_api(env, config, port)
    else:
        start_api(env, config, port, env_override)  # type: ignore
