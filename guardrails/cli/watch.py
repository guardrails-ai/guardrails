import sqlite3
import time
from dataclasses import asdict
from typing import Optional

import rich
import rich.console
import rich.table
import typer

from guardrails.cli.guardrails import guardrails as gr_cli
from guardrails.guard_call_logging import (
    GuardLogEntry,
    SyncStructuredLogHandlerSingleton,
)


@gr_cli.command(name="watch")
def watch_command(
    plain: bool = typer.Option(
        default=False,
        is_flag=True,
        help="Do not use any rich formatting, instead printing each entry on a line."
    ),
    num_lines: int = typer.Option(
        default=0,
        help="Print the last n most recent lines. If omitted, will print all history."
    ),
    follow: bool = typer.Option(
        default=False,
        help="Continuously read the last output commands",
    ),
    log_path_override: Optional[str] = typer.Option(
        default=None,
        help="Specify a path to the log output file."
    ),
):
    # Open a reader for the log path:
    log_reader = None
    while log_reader is None:
        try:
            if log_path_override is not None:
                log_reader = SyncStructuredLogHandlerSingleton.get_reader(log_path_override)
            else:
                log_reader = SyncStructuredLogHandlerSingleton.get_reader()
        except sqlite3.OperationalError:
            print("Logfile not found. Retrying.")
            time.sleep(1)

    # If we are using fancy outputs, grab a console ref and prep a table.
    if not plain:
        console, table = _setup_console_table()

    # Spin while tailing, breaking if we aren't continuously tailing.
    for log_msg in log_reader.tail_logs(-num_lines, follow):
        if plain:
            _print_and_format_plain(log_msg)
        else:
            _update_table(log_msg, console, table)


def _setup_console_table():
    console = rich.console.Console()
    table = rich.table.Table(
        show_header=True,
        header_style="bold",
    )
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Start Time")
    table.add_column("End Time")
    table.add_column("Time Delta")
    table.add_column("Prevalidate Text")
    table.add_column("Postvalidate Text")
    table.add_column("Result Text")
    return console, table


def _update_table(log_msg: GuardLogEntry, console, table):
    table.add_row(
        str(log_msg.id),
        str(log_msg.guard_name),
        str(log_msg.start_time),
        str(log_msg.end_time),
        str(log_msg.timedelta),
        str(log_msg.prevalidate_text),
        str(log_msg.postvalidate_text),
        str(log_msg.exception_message),
    )
    console.print(table)


def _print_and_format_plain(log_msg: GuardLogEntry) -> None:
    str_builder = list()
    for k, v in asdict(log_msg).items():
        str_builder.append(f"{k}: {v}")
    print("\t ".join(str_builder))
