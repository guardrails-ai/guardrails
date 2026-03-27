import guardrails.cli.configure  # noqa
import guardrails.cli.start  # noqa
import guardrails.cli.validate  # noqa
from guardrails.cli.create import create_command  # noqa: F401
from guardrails.cli.guardrails import guardrails as cli
from guardrails.cli.hub import hub_command
from guardrails.cli.watch import watch_command  # noqa: F401
from guardrails.cli.db import db_command


cli.add_typer(
    hub_command, name="hub", help="Manage validators installed from the Guardrails Hub."
)

cli.add_typer(db_command, name="db", help="Manage database migrations.")


if __name__ == "__main__":
    cli()
