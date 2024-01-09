import typer

from guardrails.cli.cli import cli

def compile_rail(rail: str, out: str) -> None:
    """Compile guardrails from the guardrails.yml file."""
    raise NotImplementedError("Currently compiling rail is not supported.")


@cli.command()
def compile(
    rail: str = typer.Argument(
        ..., help="Path to the rail spec.", exists=True, file_okay=True, dir_okay=False
    ),
    out: str = typer.Option(
        default=".rail_output",
        help="Path to the compiled output directory.",
        file_okay=False,
        dir_okay=True,
    ),
):
    """Compile guardrails from a `rail` spec."""
    print("Not supported yet. Use `validate` instead.")