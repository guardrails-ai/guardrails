from dataclasses import dataclass


@dataclass
class GuardTraceEntry:
    guard_name: str
    start_time: float
    end_time: float
    id: int = -1
    prevalidate_text: str = ""
    postvalidate_text: str = ""
    exception_message: str = ""

    @property
    def timedelta(self):
        return self.end_time - self.start_time
