import os
import sys
from pydash import snake_case, pascal_case
import typer
from guardrails.cli.hub.hub import hub
from guardrails.cli.logger import LEVELS, logger
from string import Template

from guardrails.cli.server.hub_client import HttpError, post_validator_submit


@hub.command(name='submit')
def submit(
    package_name: str = typer.Argument(
        help="The package name for your validator."
    ),
    filepath: str = typer.Argument(
        help="The location to your validator file.",
        default="./{package_name}.py"
    )
):
    try:
        if not filepath or filepath == "./{validator_name}.py":
            filepath = f"./{package_name}.py"
        
        target = os.path.abspath(filepath)
        with open(target, 'r') as validator_file:
            content = validator_file.read()

            post_validator_submit(package_name, content)

            validator_file.close()

        success_message = Template(
            """

        Successfully submitted validator!

        Once your submission is reviewed and published you will be able to install it via:

        guardrails hub install hub://guardrails/${package_name}
        """
        ).safe_substitute(
            {"package_name": snake_case(package_name)}
        )
        logger.log(level=LEVELS.get("SUCCESS"), msg=success_message)  # type: ignore

    except HttpError:
        logger.error(f"Failed to submit {package_name}!")
        sys.exit(1)
    except Exception as e:
        logger.error("An unexpected error occurred!", e)
        sys.exit(1)
    