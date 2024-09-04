from contextlib import AbstractContextManager
from types import TracebackType
from typing import Optional, Type
from unittest.mock import MagicMock


class MockSpan(AbstractContextManager):
    def __exit__(
        self,
        __exc_type: Optional[Type[BaseException]],
        __exc_value: Optional[BaseException],
        __traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        return super().__exit__(__exc_type, __exc_value, __traceback)

    def __init__(self):
        super().__init__()
        self.set_attribute = MagicMock()
