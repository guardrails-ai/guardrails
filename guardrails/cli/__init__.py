import guardrails.cli.compile
import guardrails.cli.validate

from guardrails.cli.guardrails import guardrails as cli
from guardrails.cli.hub import hub


cli.add_typer(hub, name="hub")


if __name__ == "__main__":
    cli()
