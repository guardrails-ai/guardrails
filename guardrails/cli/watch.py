import json
import sqlite3
import sys
import time
from dataclasses import asdict

import rich
import typer

from guardrails.cli.guardrails import guardrails as gr_cli
from guardrails.call_tracing import GuardTraceEntry, TraceHandler


@gr_cli.command(name="watch")
def watch_command(
    plain: bool = typer.Option(
        default=False,
        is_flag=True,
        help="Do not use any rich formatting, instead printing each entry on a line.",
    ),
    num_lines: int = typer.Option(
        default=0,
        help="Print the last n most recent lines. If omitted, will print all history.",
    ),
    follow: bool = typer.Option(
        default=True,
        help="Continuously read the last output commands",
    ),
    clear: bool = typer.Option(
        default=False, is_flag=True, help="Clear all log outputs and exit."
    ),
):
    if clear:
        _clear_and_quit()
        return

    # Open a reader for the log path:
    log_reader = _wait_for_logfile()

    # If we are using fancy outputs, grab a console ref and prep a table.
    output_fn = _print_and_format_plain
    if not plain:
        output_fn = _print_fancy

    # Spin while tailing, breaking if we aren't continuously tailing.
    for log_msg in log_reader.tail_logs(-num_lines, follow):
        output_fn(log_msg)


def _wait_for_logfile():
    log_reader = None
    while log_reader is None:
        try:
            log_reader = TraceHandler.get_reader()
        except sqlite3.OperationalError:
            print("Logfile not found. Retrying.")
            time.sleep(1)
    print(f"Reading log from {log_reader.log_path}")
    return log_reader


def _print_fancy(log_msg: GuardTraceEntry):
    rich.print(log_msg)


def _print_and_format_plain(log_msg: GuardTraceEntry) -> None:
    print(json.dumps(asdict(log_msg)))


def _clear_and_quit():
    log_reader = TraceHandler()
    log_reader.clear_logs()
    sys.exit(0)  # Force an exit here so we don't accidentally do something afterwards.
