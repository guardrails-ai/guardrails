import os
import sys
import uuid
from os.path import expanduser
from typing import Optional

import typer

from guardrails.settings import settings
from guardrails.cli.guardrails import guardrails
from guardrails.cli.logger import LEVELS, logger
from guardrails.cli.console import console
from guardrails.cli.version import version_warnings_if_applicable

DEFAULT_USE_REMOTE_INFERENCING = True


def save_configuration_file(
    use_remote_inferencing: Optional[bool] = DEFAULT_USE_REMOTE_INFERENCING,
) -> None:
    if use_remote_inferencing is None:
        use_remote_inferencing = DEFAULT_USE_REMOTE_INFERENCING

    home = expanduser("~")
    guardrails_rc = os.path.join(home, ".guardrailsrc")
    with open(guardrails_rc, "w", encoding="utf-8") as rc_file:
        lines = [
            f"id={str(uuid.uuid4())}{os.linesep}",
            f"use_remote_inferencing={str(use_remote_inferencing).lower()}",
        ]
        rc_file.writelines(lines)
        rc_file.close()

    settings._initialize()


@guardrails.command()
def configure(
    remote_inferencing: Optional[bool] = typer.Option(
        DEFAULT_USE_REMOTE_INFERENCING,
        "--enable-remote-inferencing/--disable-remote-inferencing",
        help="Opt in to remote inferencing. "
        "If not provided, you will be prompted for it.",
        prompt="Do you wish to use remote inferencing?",
    ),
):
    version_warnings_if_applicable(console)
    try:
        save_configuration_file(remote_inferencing)
        success_message = """
        Configuration successful.

        Get started by installing our RegexMatch validator:
        https://guardrailsai.com/hub/validator/guardrails_ai/regex_match

        You can install it by running:
        pip install guardrails-ai-regex-match

        Find more validators at https://guardrailsai.com/hub
        """
        logger.log(level=LEVELS.get("SUCCESS", 25), msg=success_message)
    except Exception as e:
        logger.error("An unexpected error occurred saving configuration!")
        logger.error(e)
        sys.exit(1)
