import guardrails.cli.configure  # noqa
import guardrails.cli.start  # noqa
import guardrails.cli.validate  # noqa
from guardrails.cli.guardrails import guardrails as cli
from guardrails.cli.hub import hub_command

cli.add_typer(
    hub_command, name="hub", help="Manage validators installed from the Guardrails Hub."
)


if __name__ == "__main__":
    cli()
