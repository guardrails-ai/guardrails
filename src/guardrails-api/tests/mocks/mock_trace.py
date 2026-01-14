from contextlib import AbstractContextManager
import random
import string
from types import TracebackType
from typing import Optional, Type


class MockSpan(AbstractContextManager):
    def __exit__(
        self,
        __exc_type: Optional[Type[BaseException]],
        __exc_value: Optional[BaseException],
        __traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        return super().__exit__(__exc_type, __exc_value, __traceback)

    def add_event(self, *args, **kwargs):
        pass

    def set_status(self, *args, **kwargs):
        pass

    def set_attribute(self, *args, **kwargs):
        pass


class MockTrace:
    @staticmethod
    def get_current_span(*args, **kwargs):
        return MockSpan()


class MockTracer:
    span: MockSpan

    def __init__(self, span: Optional[MockSpan] = None):
        self.span = span or MockSpan()

    def start_as_current_span(self, *args, **kwargs):
        return self.span


class MockContext:
    _id: str

    def __init__(self):
        self._id = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
