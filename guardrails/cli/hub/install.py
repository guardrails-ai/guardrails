import typer

from guardrails.cli.hub.hub import hub
from guardrails.cli.logger import logger

@hub.command()
def install(
    package_uri: str = typer.Argument(
        ..., help="URI to the package to install. Example: hub://guardrails/regex-match." #, exists=True, file_okay=True, dir_okay=False
    )
):
    """Install a validator from the Hub."""
    logger.info(f"Installing {package_uri}...")