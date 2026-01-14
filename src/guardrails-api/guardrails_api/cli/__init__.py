import typer
import guardrails_api.cli.start  # noqa
from typing import Optional
from guardrails_api import __version__
from guardrails_api.cli.cli import cli


def version_callback(value: bool):
    if value:
        print(f"guardrails-api CLI Version: {__version__}")
        raise typer.Exit()


@cli.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", callback=version_callback, is_eager=True
    ),
):
    pass


if __name__ == "__main__":
    cli()
