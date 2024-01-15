from contextlib import AbstractContextManager
from types import TracebackType
from typing import Optional, Type


class MockFile(AbstractContextManager):
    def __exit__(
        self,
        __exc_type: Optional[Type[BaseException]],
        __exc_value: Optional[BaseException],
        __traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        return super().__exit__(__exc_type, __exc_value, __traceback)
    def readlines(self):
        pass
    def writelines(self, *args):
        pass
    def close(self):
        pass
    def read(self, *args):
        pass
    def write(self, *args):
        pass
    def seek(self, *args):
        pass