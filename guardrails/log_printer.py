from logging import LogRecord
import os
from typing import Dict, List


"""
This class is responsible for sorting incoming logs
by guard/execution, then writing them to the appropriate files.

Those files will then be consumed by the guardrails watch command.

In the future, this class will also be responsible for enforcing the TTL
of log files, to avoid filling up the disk with logs.
"""


class LogPrinter:
    log_files: Dict[str, List[str]] = {}
    log_dir: str

    def __init__(self, log_dir):
        # pull existing guard logs into internal map
        self.log_dir = log_dir
        if os.path.exists(self.log_dir) is False:
            os.mkdir(self.log_dir)
        guard_folders = os.listdir(self.log_dir)
        for guard_folder in guard_folders:
            execution_files = os.listdir(os.path.join(self.log_dir, guard_folder))
            self.log_files[guard_folder] = [execution_files]

    def emit(self, record: LogRecord):
        if len(record.args) < 2:
            return None
        if isinstance(record.args[0], str) is False:
            return None
        if isinstance(record.args[1], str) is False:
            return None
        guard_name = record.args[0]
        execution_name = record.args[1]

        if guard_name is None or execution_name is None:
            return
        existing_executions = self.log_files.get(guard_name)
        if existing_executions is None:
            # create guard folder and execution file
            os.mkdir(os.path.join(self.log_dir, guard_name))
            self.log_files[guard_name] = []
            existing_executions = []
            pass
        path = os.path.join(self.log_dir, guard_name, execution_name)
        with open(path, "w") as f:
            # write execution file to internal filemap
            # write log to execution file
            f.write(record.msg)
            pass
        if execution_name not in existing_executions:
            # write execution file to internal filemap
            # write log to execution file
            self.log_files[guard_name].append(execution_name)
