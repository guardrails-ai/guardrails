from logging import LogRecord
import os
from typing import Dict, List


LOG_DIR = "guard_logs"

"""
This class is responsible for sorting incoming logs
by guard/execution, then writing them to the appropriate files.

Those files will then be consumed by the guardrails watch command.

In the future, this class will also be responsible for enforcing the TTL
of log files, to avoid filling up the disk with logs.
"""


class LogPrinter:
    log_files: Dict[str, List[str]] = {}

    def __init__(self):
        # pull existing guard logs into internal map
        guard_folders = os.listdir(LOG_DIR)
        if os.path.exists(LOG_DIR) is False:
            os.mkdir(LOG_DIR)
        for guard_folder in guard_folders:
            execution_files = os.listdir(os.path.join(LOG_DIR, guard_folder))
            self.log_files[guard_folder] = [execution_files]

    def guard_from_logrecord(record: LogRecord):
        pass

    def execution_name_from_logrecord(record: LogRecord):
        pass

    def emit(self, record: LogRecord):
        guard_name = self.guard_from_logrecord(record)
        execution_name = self.execution_name_from_logrecord(record)

        if guard_name is None or execution_name is None:
            return
        existing_executions = self.log_files.get(guard_name)
        if existing_executions is None:
            # create guard folder and execution file
            os.mkdir(os.path.join(LOG_DIR, guard_name))
            self.log_files[guard_name] = []
            existing_executions = []
            pass
        with os.open(os.path.join(LOG_DIR, guard_name, execution_name), "w") as f:
            # write execution file to internal filemap
            # write log to execution file
            f.write(record.getMessage())
            pass
        if execution_name not in existing_executions:
            # write execution file to internal filemap
            # write log to execution file
            self.log_files[guard_name].append(execution_name)
