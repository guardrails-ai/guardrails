import threading
from typing import Optional

from guardrails.classes.rc import RC


class Settings:
    _instance = None
    _lock = threading.Lock()
    _rc: RC
    _watch_mode_enabled: bool
    """Whether to use a local server for running Guardrails."""
    use_server: Optional[bool]
    """Whether to disable tracing.

    Traces are only ever sent to a telemetry sink you specify via
    environment variables or by instantiating a TracerProvider.
    """
    disable_tracing: Optional[bool]

    def __new__(cls) -> "Settings":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(Settings, cls).__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.use_server = None
        self.disable_tracing = None
        self._rc = RC.load()
        self._watch_mode_enabled = False

    @property
    def rc(self) -> RC:
        if self._rc is None:
            self._rc = RC.load()
        return self._rc

    @rc.setter
    def rc(self, value: RC):
        self._rc = value

    @property
    def watch_mode_enabled(self) -> bool:
        return self._watch_mode_enabled


settings = Settings()
