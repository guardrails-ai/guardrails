import time
from typing import Optional

import rich
import typer

from guardrails.cli.guardrails import guardrails as gr_cli
from guardrails.guard_call_logging import (
    GuardLogEntry,
    SyncStructuredLogHandlerSingleton,
)


def _print_and_format_log_message(log_msg: GuardLogEntry):
    rich.print(log_msg)


@gr_cli.command(name="watch")
def watch_command(
    num_lines: int = typer.Option(
        default=0,
        help="Print the last n most recent lines. If omitted, will print all history."
    ),
    refresh_frequency: Optional[float] = typer.Option(
        default=1.0,
        help="How long (in seconds) should the watch command wait between outputs."
    ),
    follow: bool = typer.Option(
        False,
        "--follow",
        help="Continuously read the last output commands",
    ),
    log_path_override: Optional[str] = typer.Option(
        default=None,
        help="Specify a path to the log output file."
    ),
):
    if log_path_override is not None:
        log_reader = SyncStructuredLogHandlerSingleton.get_reader(log_path_override)
    else:
        log_reader = SyncStructuredLogHandlerSingleton.get_reader()

    while True:
        for log_msg in log_reader.tail_logs(-num_lines):
            _print_and_format_log_message(log_msg)
        if not follow:
            return
        time.sleep(refresh_frequency)
