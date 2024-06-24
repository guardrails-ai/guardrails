import os
import sqlite3
import threading
from dataclasses import dataclass, fields


@dataclass
class GuardCallLogEntry:
    # Keep in sync with the table creation.
    guardname: str
    start_time: float
    end_time: float
    prevalidate_text: str
    postvalidate_text: str
    exception_text: str
    log_level: int


class _SyncStructuredLogHandler:
    LOG_TABLE = "guard_logs"
    CREATE_COMMAND = """"""
    INSERT_COMMAND = """
        INSERT INTO ? VALUES (
            :guard_name, :start_time, :end_time, :prevalidate_text, :exception_text, 
            :log_level
        )
    """

    def __init__(self, default_log_path: os.PathLike):
        self.db = sqlite3.connect(default_log_path, )
        cursor = self.db.cursor()
        # Generate table rows from GuardCallLogEntry.
        create_fields = ""
        for field in fields(GuardCallLogEntry):
            create_fields += field.name
            create_fields += " "
            if field.type == int:
                create_fields += "INTEGER"
            elif field.type == float:
                create_fields += "REAL"
            elif field.type == str:
                create_fields += "TEXT"
            create_fields += ","
        create_fields.removesuffix(",")  # Remove the spurious trailing ','.
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS " +
            _SyncStructuredLogHandler.LOG_TABLE +
            f"({fields});"
        )

    def log_entry(self, entry: GuardCallLogEntry):
        cursor = self.db.cursor()
        cursor.execute("""INSERT""")

    def log(
            self,
            guard_name: str,
            start_time: float,
            end_time: float,
            prevalidate_text: str,
            exception_text: str,
            log_level: int,
    ):
        cursor = self.db.cursor()
        cursor.execute(INSERT_COMMAND, (
            _SyncStructuredLogHandler.LOG_TABLE,
            guard_name,
            start_time,
            end_time,
            prevalidate_text,
            exception_text,
            log_level
        ))


class SyncStructuredLogHandlerSingleton(_SyncStructuredLogHandler):
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:  # Yes, two 'if' checks to avoid mutex contention.
            # This only runs if we definitely need to lock.
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls.create()
        return cls._instance

    @classmethod
    def create(cls) -> _SyncStructuredLogHandler:
        return _SyncStructuredLogHandler()

