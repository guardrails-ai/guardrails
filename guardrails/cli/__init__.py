import guardrails.cli.compile  # noqa
import guardrails.cli.configure  # noqa
import guardrails.cli.validate  # noqa
from guardrails.cli.guardrails import guardrails as cli
from guardrails.cli.hub import hub

cli.add_typer(hub, name="hub")


if __name__ == "__main__":
    cli()
