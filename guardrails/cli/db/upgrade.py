import typer
from typing import Annotated
from importlib.metadata import PackageNotFoundError, version
from guardrails.cli.db.db import db_command
from guardrails.cli.logger import logger


@db_command.command(name="upgrade")
def upgrade(
    revision: Annotated[str, typer.Argument()] = "head",
    env: str = typer.Option(
        default=".env",
        help="An env file to load environment variables from.",
    ),
    env_override: bool = typer.Option(
        default=False,
        help="Override existing environment variables with values from the env file.",
    ),
):
    try:
        guardrails_api_version = version("guardrails_api")
        major, minor, *_ = guardrails_api_version.split(".")
        if major == "0" and int(minor) < 3:
            logger.error(
                "[ERROR]: 'db upgrade' is only supported for guardrails-api>=0.3.0."
                f"  You have guardrails-api=={guardrails_api_version}."
            )
        else:
            from guardrails_api.cli.db.downgrade import downgrade  # type: ignore

            downgrade(revision, env, env_override)
    except PackageNotFoundError:
        logger.error("[ERROR]: 'db upgrade' requires guardrails-api to be installed.")
