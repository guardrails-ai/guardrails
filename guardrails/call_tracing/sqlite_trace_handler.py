"""sqlite_trace_handler.py.

This is the metaphorical bread and butter of our tracing implementation,
or at least the butter.  It wraps a SQLite database and configures it to
be 'agreeable' in multithreaded situations.  Normally, when sharing
across threads and instances one should consider using a larger database
solution like Postgres, but in this case we only care about _supporting_
writing from multiple places.  We don't expect it will be the norm. We
care about (1) not negatively impacting performance, (2) not crashing
when used in unusual ways, and (3) not losing data when possible.

The happy path should be reasonably performant.  The unhappy path should
not crash.

The other part of the multithreaded support comes from the public
trace_handler, which uses a singleton pattern to only have a single
instance of the database per-thread. If we _do_ somehow end up shared
across threads, the journaling settings and writeahead should protect us
from odd behavior.
"""

import datetime
import os
import sqlite3
import time
from dataclasses import asdict
from typing import Iterator

from guardrails.call_tracing.trace_entry import GuardTraceEntry
from guardrails.call_tracing.tracer_mixin import TracerMixin
from guardrails.classes.validation.validator_logs import ValidatorLogs
from guardrails.utils.casting_utils import to_string


LOG_RETENTION_LIMIT = 100000
TIME_BETWEEN_CLEANUPS = 10.0  # Seconds


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


# This structured handler shouldn't be used directly, since it's touching a SQLite db.
# Instead, use the singleton or the async singleton.
class SQLiteTraceHandler(TracerMixin):
    CREATE_COMMAND = """
        CREATE TABLE IF NOT EXISTS guard_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            guard_name TEXT,
            start_time REAL,
            end_time REAL,
            prevalidate_text TEXT,
            postvalidate_text TEXT,
            exception_message TEXT
        );
    """
    INSERT_COMMAND = """
        INSERT INTO guard_logs (
            guard_name, start_time, end_time, prevalidate_text, postvalidate_text,
            exception_message
        ) VALUES (
            :guard_name, :start_time, :end_time, :prevalidate_text, :postvalidate_text,
            :exception_message
        );
    """

    def __init__(self, log_path: os.PathLike, read_mode: bool):
        self._log_path = log_path  # Read-only value.
        self.last_cleanup = time.time()
        self.readonly = read_mode
        if read_mode:
            self.db = SQLiteTraceHandler._get_read_connection(log_path)
        else:
            self.db = SQLiteTraceHandler._get_write_connection(log_path)

    @classmethod
    def _get_write_connection(cls, log_path: os.PathLike) -> sqlite3.Connection:
        try:
            db = sqlite3.connect(
                log_path,
                isolation_level=None,
                check_same_thread=False,
            )
            db.execute("PRAGMA journal_mode = wal")
            db.execute("PRAGMA synchronous = OFF")
            # isolation_level = None and pragma WAL means we can READ from the DB
            # while threads using it are writing.  Synchronous off puts us on the
            # highway to the danger zone, depending on how willing we are to lose log
            # messages in the event of a guard crash.
        except sqlite3.OperationalError as e:
            # logging.exception("Unable to connect to guard log handler.")
            raise e
        with db:
            db.execute(SQLiteTraceHandler.CREATE_COMMAND)
        return db

    @classmethod
    def _get_read_connection(cls, log_path: os.PathLike) -> sqlite3.Connection:
        # A bit of a hack to open in read-only mode...
        db = sqlite3.connect(
            "file:" + str(log_path) + "?mode=ro", isolation_level=None, uri=True
        )
        db.row_factory = sqlite3.Row
        return db

    def _truncate(self, force: bool = False, keep_n: int = LOG_RETENTION_LIMIT):
        assert not self.readonly
        now = time.time()
        if force or (now - self.last_cleanup > TIME_BETWEEN_CLEANUPS):
            self.last_cleanup = now
        self.db.execute(
            """
            DELETE FROM guard_logs 
            WHERE id < (
                SELECT id FROM guard_logs ORDER BY id DESC LIMIT 1 OFFSET ?  
            );
            """,
            (keep_n,),
        )

    def log(
        self,
        guard_name: str,
        start_time: float,
        end_time: float,
        prevalidate_text: str,
        postvalidate_text: str,
        exception_text: str,
    ):
        assert not self.readonly
        with self.db:
            self.db.execute(
                SQLiteTraceHandler.INSERT_COMMAND,
                dict(
                    guard_name=guard_name,
                    start_time=start_time,
                    end_time=end_time,
                    prevalidate_text=prevalidate_text,
                    postvalidate_text=postvalidate_text,
                    exception_message=exception_text,
                ),
            )
        self._truncate()

    def log_entry(self, guard_log_entry: GuardTraceEntry):
        assert not self.readonly
        with self.db:
            self.db.execute(SQLiteTraceHandler.INSERT_COMMAND, asdict(guard_log_entry))
        self._truncate()

    def log_validator(self, vlog: ValidatorLogs):
        assert not self.readonly
        maybe_outcome = (
            str(vlog.validation_result.outcome)
            if (
                vlog.validation_result is not None
                and hasattr(vlog.validation_result, "outcome")
            )
            else ""
        )
        with self.db:
            self.db.execute(
                SQLiteTraceHandler.INSERT_COMMAND,
                dict(
                    guard_name=vlog.validator_name,
                    start_time=vlog.start_time if vlog.start_time else None,
                    end_time=vlog.end_time if vlog.end_time else 0.0,
                    prevalidate_text=to_string(vlog.value_before_validation),
                    postvalidate_text=to_string(vlog.value_after_validation),
                    exception_message=maybe_outcome,
                ),
            )
        self._truncate()

    def clear_logs(self):
        self.db.execute("DELETE FROM guard_logs;")

    def tail_logs(
        self, start_offset_idx: int = 0, follow: bool = False
    ) -> Iterator[GuardTraceEntry]:
        """Returns an iterator to generate GuardLogEntries.

        @param start_offset_idx : Start printing entries after this IDX.
        If negative, this will instead start printing the LAST
        start_offset_idx entries.

        @param follow : If follow is True, will re-check the database
        for new entries after the first batch is complete.  If False
        (default), will return when entries are exhausted.
        """
        last_idx = start_offset_idx
        cursor = self.db.cursor()
        if last_idx < 0:
            # We're indexing from the end, so do a quick check.
            cursor.execute(
                "SELECT id FROM guard_logs ORDER BY id DESC LIMIT 1 OFFSET ?;",
                (-last_idx,),
            )
            for row in cursor:
                last_idx = row["id"]
        sql = """
            SELECT 
                id, guard_name, start_time, end_time, prevalidate_text, 
                postvalidate_text, exception_message 
            FROM guard_logs 
            WHERE id > ?
            ORDER BY start_time;
        """
        cursor.execute(sql, (last_idx,))
        while True:
            for row in cursor:
                last_entry = GuardTraceEntry(**row)
                last_idx = last_entry.id
                yield last_entry
            if not follow:
                return
            # If we're here we've run out of entries to tail. Fetch more:
            cursor.execute(sql, (last_idx,))
