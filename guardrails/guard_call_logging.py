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

import datetime
import os
import sqlite3
import threading
from dataclasses import dataclass, asdict
from typing import Iterator

from guardrails.classes import ValidationOutcome
from guardrails.utils.casting_utils import to_string
from guardrails.classes.history import Call
from guardrails.classes.validation.validator_logs import ValidatorLogs


LOG_FILENAME = "guardrails_calls.db"


# These adapters make it more convenient to add data into our log DB:
# Handle timestamp -> sqlite map:
def adapt_datetime(val):
    """Adapt datetime.datetime to Unix timestamp."""
    # return val.isoformat()  # If we want to go to datetime/isoformat...
    return int(val.timestamp())


sqlite3.register_adapter(datetime.datetime, adapt_datetime)


def convert_timestamp(val):
    """Convert Unix epoch timestamp to datetime.datetime object."""
    # To go to datetime.datetime:
    # return datetime.datetime.fromisoformat(val.decode())
    return datetime.datetime.fromtimestamp(int(val))


sqlite3.register_converter("timestamp", convert_timestamp)


# This class makes it slightly easier to be selective about how we pull data.
# While it's not the ultimate contract/DB schema, it helps with typing and improves dx.
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

    @property
    def timedelta(self):
        return self.end_time - self.start_time


class _NoopTraceHandler:
    def __init__(self, log_path: os.PathLike, read_mode: bool):
        pass

    def log(self, *args, **kwargs):
        pass

    def log_entry(self, guard_log_entry: GuardLogEntry):
        pass

    def log_validator(self, vlog: ValidatorLogs):
        pass

    def tail_logs(
            self,
            start_offset_idx: int = 0,
            follow: bool = False,
    ) -> Iterator[GuardLogEntry]:
        return []



# This structured handler shouldn't be used directly, since it's touching a SQLite db.
# Instead, use the singleton or the async singleton.
class _SyncTraceHandler:
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
            self.db = _SyncTraceHandler._get_read_connection(log_path)
        else:
            self.db = _SyncTraceHandler._get_write_connection(log_path)

    @classmethod
    def _get_write_connection(cls, log_path: os.PathLike) -> sqlite3.Connection:
        try:
            db = sqlite3.connect(
                log_path,
                isolation_level=None,
                check_same_thread=False,
            )
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
            db.execute(_SyncTraceHandler.CREATE_COMMAND)
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
            self.db.execute(_SyncTraceHandler.INSERT_COMMAND, dict(
                guard_name=guard_name,
                start_time=start_time,
                end_time=end_time,
                prevalidate_text=prevalidate_text,
                postvalidate_text=postvalidate_text,
                exception_message=exception_text,
                log_level=log_level
            ))

    def log_entry(self, guard_log_entry: GuardLogEntry):
        assert not self.readonly
        with self.db:
            self.db.execute(
                _SyncTraceHandler.INSERT_COMMAND,
                asdict(guard_log_entry)
            )

    def log_validator(self, vlog: ValidatorLogs):
        assert not self.readonly
        maybe_outcome = str(vlog.validation_result.outcome) \
            if hasattr(vlog.validation_result, "outcome") else ""
        with self.db:
            self.db.execute(_SyncTraceHandler.INSERT_COMMAND, dict(
                guard_name=vlog.validator_name,
                start_time=vlog.start_time if vlog.start_time else None,
                end_time=vlog.end_time if vlog.end_time else 0.0,
                prevalidate_text=to_string(vlog.value_before_validation),
                postvalidate_text=to_string(vlog.value_after_validation),
                exception_message=maybe_outcome,
                log_level=0
            ))

    def tail_logs(
            self,
            start_offset_idx: int = 0,
            follow: bool = False
    ) -> Iterator[GuardLogEntry]:
        """Returns an iterator to generate GuardLogEntries.
        @param start_offset_idx int : Start printing entries after this IDX. If
        negative, this will instead start printing the LAST start_offset_idx entries.
        @param follow : If follow is True, will re-check the database for new entries
        after the first batch is complete.  If False (default), will return when entries
        are exhausted.
        """
        last_idx = start_offset_idx
        cursor = self.db.cursor()
        if last_idx < 0:
            # We're indexing from the end, so do a quick check.
            cursor.execute(
                "SELECT id FROM guard_logs ORDER BY id DESC LIMIT 1 OFFSET ?;",
                (-last_idx,)
            )
            for row in cursor:
                last_idx = row['id']
        sql = """
            SELECT 
                id, guard_name, start_time, end_time, prevalidate_text, 
                postvalidate_text, exception_message, log_level 
            FROM guard_logs 
            WHERE id > ?
            ORDER BY start_time;
        """
        cursor.execute(sql, (last_idx,))
        while True:
            for row in cursor:
                last_entry = GuardLogEntry(**row)
                last_idx = last_entry.id
                yield last_entry
            if not follow:
                return
            # If we're here we've run out of entries to tail. Fetch more:
            cursor.execute(sql, (last_idx,))


class SyncTraceHandler(_SyncTraceHandler):
    """SyncTraceHandler wraps the internal _SyncTraceHandler to make it multi-thread
    safe.  Coupled with some write ahead journaling in the _SyncTrace internal, we have
    a faux-multi-write multi-read interface for SQLite."""
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
    def _create(cls, path: os.PathLike = LOG_FILENAME) -> _SyncTraceHandler:
        return _SyncTraceHandler(path, read_mode=False)

    @classmethod
    def get_reader(cls, path: os.PathLike = LOG_FILENAME) -> _SyncTraceHandler:
        return _SyncTraceHandler(path, read_mode=True)
