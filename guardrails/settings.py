import threading
from typing import Optional


class Settings:
    _instance = None
    _lock = threading.Lock()
    """Whether to use a local server for running Guardrails."""
    use_server: Optional[bool]

    def __new__(cls) -> "Settings":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(Settings, cls).__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.use_server = None


settings = Settings()
