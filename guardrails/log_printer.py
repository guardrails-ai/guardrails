from logging import LogRecord
import os
from typing import Dict, List


LOG_DIR = "guard_logs"


class LogPrinter:
    log_files: Dict[str, List[str]] = {}

    def __init__(self):
        # pull existing guard logs into internal map
        guard_folders = os.listdir(LOG_DIR)
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

        existing_executions = self.log_files.get(guard_name)
        if existing_executions is None:
            # create guard folder and execution file
            # write execution file to internal filemap
            # write log to execution file
            pass
        else:
            if execution_name not in existing_executions:
                pass
                # create execution file
                # write execution file to internal filemap
                # write log to execution file
            else:
                # write log to execution file
                pass
