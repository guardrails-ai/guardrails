import threading
from typing import Optional


class Settings:
    _instance = None
    _lock = threading.Lock()
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


settings = Settings()
