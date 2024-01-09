import typer

from typing import Optional
from guardrails.cli.guardrails import guardrails
from guardrails.cli.logger import logger


def save_configuration_file(client_id: str, client_secret: str, no_metrics: bool) -> None:
    return


@guardrails.command()
def configure(
    client_id: Optional[str] = typer.Option(help="Your Guardrails Hub client ID.", default=""),
    client_secret: Optional[str] = typer.Option(help="Your Guardrails Hub client secret.", hide_input=True, default=""),
    no_metrics: Optional[str] = typer.Option(help="Opt out of anonymous metrics collection.", default=False),
):
    """Set the global configuration for the Guardrails CLI and Hub"""
    if not client_id:
        client_id = typer.prompt("Client ID")
    if not client_secret:
        client_secret = typer.prompt("Client secret", hide_input=True)
    logger.info("Configuring...")
    save_configuration_file(client_id, client_secret, no_metrics)
