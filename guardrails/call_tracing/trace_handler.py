"""trace_handler.py.

A set of tools to track the behavior of guards, specifically with the intent of
collating the pre/post validation text and timing of guard calls.  Uses a singleton to
share write access to a SQLite database across threads.

By default, logs will be created in a temporary directory.  This can be overridden by
setting GUARDRAILS_LOG_FILE_PATH in the environment.  tracehandler.log_path will give
the full path of the current log file.

# Reading logs (basic):
>>> reader = TraceHandler.get_reader()
>>> for t in reader.tail_logs():
>>>    print(t)

# Reading logs (advanced):
>>> reader = TraceHandler.get_reader()
>>> reader.db.execute("SELECT * FROM guard_logs;")  # Arbitrary SQL support.

# Saving logs
>>> writer = TraceHandler()
>>> writer.log(
>>>    "my_guard_name", 0.0, 1.0, "Raw LLM Output Text", "Sanitized", "exception?"
>>> )
"""

import os
import tempfile
import threading

from guardrails.call_tracing.sqlite_trace_handler import SQLiteTraceHandler
from guardrails.call_tracing.tracer_mixin import TracerMixin

# TODO: We should read this from guardrailsrc.
LOG_FILENAME = "guardrails_calls.db"
LOGFILE_PATH = os.environ.get(
    "GUARDRAILS_LOG_FILE_PATH",  # Document this environment variable.
    os.path.join(tempfile.gettempdir(), LOG_FILENAME),
)


class TraceHandler(TracerMixin):
    """TraceHandler wraps the internal _SQLiteTraceHandler to make it multi-
    thread safe.

    Coupled with some write ahead journaling in the _SyncTrace internal,
    we have a faux-multi-write multi-read interface for SQLite.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            # We run two 'if None' checks so we don't have to call the mutex check for
            # the cases where there's obviously no handler.  Only do a check if there
            # MIGHT not be a handler instantiated.
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls._create()
        return cls._instance

    @classmethod
    def _create(cls) -> TracerMixin:  # type: ignore
        return SQLiteTraceHandler(LOGFILE_PATH, read_mode=False)  # type: ignore
        # To disable logging:
        # return _BaseTraceHandler(LOGFILE_PATH, read_mode=False)

    @classmethod
    def get_reader(cls) -> TracerMixin:  # type: ignore
        return SQLiteTraceHandler(LOGFILE_PATH, read_mode=True)  # type: ignore
