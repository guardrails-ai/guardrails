"""trace_entry.py.

GuardTraceEntry is a dataclass which doesn't explicitly define the
schema of our logs, but serves as a nice, easy-to-use dataclass for when
we want to manipulate things programmatically.  If performance and
filtering is a concern, it's probably worth writing the SQL directly
instead of filtering these in a for-loop.
"""

from dataclasses import dataclass


@dataclass
class GuardTraceEntry:
    id: int = -1
    guard_name: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    prevalidate_text: str = ""
    postvalidate_text: str = ""
    exception_message: str = ""

    @property
    def timedelta(self):
        return self.end_time - self.start_time
