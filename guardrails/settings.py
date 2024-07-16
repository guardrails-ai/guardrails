from typing import Self


class Settings:
    _instance = None

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super(Settings, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        self.use_server = None


settings = Settings()
