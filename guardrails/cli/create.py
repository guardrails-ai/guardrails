import importlib
#import pkgutil
import inspect
import subprocess
from typing import Optional

import rich
import typer

from guardrails.cli.guardrails import guardrails as gr_cli


template = """
from guardrails import Guard
from guardrails.hub import (
    DetectPII,
    CompetitorCheck
)


input_guards = Guard()

output_guards = Guard()
output_guards.name = "Output Guard"
output_guards.use_many(
    DetectPII(
        pii_entities='pii'
    ),
    CompetitorCheck(
        competitors=['OpenAI', 'Anthropic']
    )
)
"""


@gr_cli.command(name="create")
def create_command(
    validators: str = typer.Option(
        help="A comma-separated list of validator hub URIs. ",
    ),
    name: Optional[str] = typer.Option(
        default=None,
        help="The name of the guard to define in the file."
    ),
    filepath: str = typer.Option(
        default="config.py",
        help="The path to which the configuration file should be saved."
    ),
    dry_run: bool = typer.Option(
        default=False,
        is_flag=True,
        help="Print out the validators to be installed without making any changes."
    )
):
    installed_validators = split_and_process_validators(validators, dry_run)
    new_config_file = generate_config_file(installed_validators, name)
    if dry_run:
        rich.print(f"Not actually saving output to {filepath}")
        rich.print(f"The following would have been written:\n{new_config_file}")
    else:
        with open(filepath, 'wt') as fout:
            fout.write(new_config_file)
        rich.print(f"Saved configuration to {filepath}")


def split_and_process_validators(validators: str, dry_run: bool = False):
    """Given a comma-separated list of validators, check the hub to make sure all of
    them exist, then install each one via pip.  """
    # Quick sanity check after split:
    validators = validators.split(",")
    checked_validators = list()
    for v in validators:
        if not v.strip().startswith("hub://"):
            rich.print(f"WARNING: Validator {v} does not appear to be a valid URI.")
        checked_validators.append(v.strip())
    validators = checked_validators

    # We should make sure they exist.
    for v in validators:
        rich.print(f"Installing {v}")
        try:
            if not dry_run:
                # TODO: When we have the programmatic hub install tool, switch to that.
                subprocess.run(
                    ["guardrails", "hub", "install", v],
                    capture_output=True,
                    check=True
                )
        except subprocess.CalledProcessError as cpe:
            rich.print(f"ERROR: Failed to install guard {v}.{cpe.stdout}\n{cpe.stderr}")
            raise cpe

    # Pull the hub information from each of the installed validators and return it.
    return [v for v in validators]


def generate_config_file(validators: str, name: Optional[str] = None) -> str:
    return "asdf"


def _reload_hub():
    import guardrails.hub
    importlib.invalidate_caches()
    return importlib.reload(guardrails.hub)