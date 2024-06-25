"""
guard_call_logging.py

A set of tools to track the behavior of guards, specifically with the intent of
collating the pre/post validation text and timing of guard calls.  Uses a singleton to
share write access to a SQLite database across threads.

# Reading logs (basic):
reader = SyncStructuredLogHandlerSingleton.get_reader()
for t in reader.tail_logs():
    print(t)

# Reading logs (advanced):
reader = SyncStructuredLogHandlerSingleton.get_reader()
reader.db.execute("SELECT * FROM guard_logs;")  # Arbitrary SQL support.

# Saving logs
writer = SynbcStructuredLogHandlerSingleton()
writer.log(
  "my_guard_name", start, end, "Raw LLM Output Text", "Sanitized", "exception?", 0
)

"""

import os
import sqlite3
import threading
import time
from dataclasses import dataclass, asdict
from typing import Iterator

from guardrails.classes import ValidationOutcome
from guardrails.classes.history import Call


LOG_FILENAME = "guardrails_calls.db"


@dataclass
class GuardLogEntry:
    guard_name: str
    start_time: float
    end_time: float
    log_level: int
    id: int = -1
    prevalidate_text: str = ""
    postvalidate_text: str = ""
    exception_message: str = ""


class _SyncStructuredLogHandler:
    CREATE_COMMAND = """
        CREATE TABLE IF NOT EXISTS guard_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            guard_name TEXT,
            start_time REAL,
            end_time REAL,
            prevalidate_text TEXT,
            postvalidate_text TEXT,
            exception_message TEXT,
            log_level INTEGER
        );
    """
    INSERT_COMMAND = """
        INSERT INTO guard_logs (
            guard_name, start_time, end_time, prevalidate_text, postvalidate_text,
            exception_message, log_level
        ) VALUES (
            :guard_name, :start_time, :end_time, :prevalidate_text, :postvalidate_text,
            :exception_message, :log_level
        );
    """

    def __init__(self, log_path: os.PathLike, read_mode: bool):
        self.readonly = read_mode
        if read_mode:
            self.db = _SyncStructuredLogHandler._get_read_connection(log_path)
        else:
            self.db = _SyncStructuredLogHandler._get_write_connection(log_path)

    @classmethod
    def _get_write_connection(cls, log_path: os.PathLike) -> sqlite3.Connection:
        try:
            db = sqlite3.connect(log_path, isolation_level=None)
            db.execute('PRAGMA journal_mode = wal')
            db.execute('PRAGMA synchronous = OFF')
            # isolation_level = None and pragma WAL means we can READ from the DB
            # while threads using it are writing.  Synchronous off puts us on the
            # highway to the danger zone, depending on how willing we are to lose log
            # messages in the event of a guard crash.
        except sqlite3.OperationalError as e:
            #logging.exception("Unable to connect to guard log handler.")
            raise e
        with db:
            db.execute(_SyncStructuredLogHandler.CREATE_COMMAND)
        return db

    @classmethod
    def _get_read_connection(cls, log_path: os.PathLike) -> sqlite3.Connection:
        # A bit of a hack to open in read-only mode...
        db = sqlite3.connect(
            "file:" + log_path + "?mode=ro",
            isolation_level=None,
            uri=True
        )
        db.row_factory = sqlite3.Row
        return db

    def log(
            self,
            guard_name: str,
            start_time: float,
            end_time: float,
            prevalidate_text: str,
            postvalidate_text: str,
            exception_text: str,
            log_level: int,
    ):
        assert not self.readonly
        with self.db:
            self.db.execute(_SyncStructuredLogHandler.INSERT_COMMAND, dict(
                guard_name=guard_name,
                start_time=start_time,
                end_time=end_time,
                prevalidate_text=prevalidate_text,
                postvalidate_text=postvalidate_text,
                exception_message=exception_text,
                log_level=log_level
            ))

    def log_entry(self, guard_log_entry):
        assert not self.readonly
        with self.db:
            self.db.execute(
                _SyncStructuredLogHandler.INSERT_COMMAND,
                asdict(guard_log_entry)
            )

    def tail_logs(self, start_offset_idx: int = 0) -> Iterator[GuardLogEntry]:
        last_idx = start_offset_idx
        cursor = self.db.cursor()
        sql = """
            SELECT 
                id, guard_name, start_time, end_time, prevalidate_text, 
                postvalidate_text, exception_message, log_level 
            FROM guard_logs 
            WHERE id > ?
            ORDER BY start_time;
        """
        cursor.execute("SELECT 1 LIMIT 0;")
        while True:
            for row in cursor:
                last_entry = GuardLogEntry(**row)
                last_idx = last_entry.id
                yield last_entry
            # If we're here we've run out of entries to tail.
            cursor.execute(sql, (last_idx,))


class SyncStructuredLogHandlerSingleton(_SyncStructuredLogHandler):
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:  # Yes, two 'if' checks to avoid mutex contention.
            # This only runs if we definitely need to lock.
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls._create()
        return cls._instance

    @classmethod
    def _create(cls) -> _SyncStructuredLogHandler:
        return _SyncStructuredLogHandler(LOG_FILENAME, read_mode=False)

    @classmethod
    def get_reader(cls) -> _SyncStructuredLogHandler:
        return _SyncStructuredLogHandler(LOG_FILENAME, read_mode=True)


class LoggedCall:
    def __init__(self, call_log: Call):
        # Have to wait until the actual call to get the sync/async method.
        self.log_handler = None
        self.called_fn = call_log
        self.start_time = None
        self.end_time = None

    def report_outcome(self, result: ValidationOutcome):
        pass

    def __enter__(self):
        self.log_handler = SyncStructuredLogHandlerSingleton()
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        # Log the outcome of the operation.
        self.log_handler.log()

    def __aenter__(self):
        self.start_time = time.time()

    def __aexit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
