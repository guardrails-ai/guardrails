import logging
import logging.config
from logging import Handler, LogRecord
from typing import Dict, List, Optional

# from src.modules.otel_logger import handler as otel_handler

name = "guardrails-ai"
base_scope = "base"
all_scopes = "all"


class ScopeHandler(Handler):
    scope: str
    scoped_logs: Dict[str, List[LogRecord]]

    def __init__(self, level=logging.NOTSET, scope=base_scope):
        super().__init__(level)
        self.scope = scope
        self.scoped_logs = {}

    def emit(self, record: LogRecord) -> None:
        logs = self.scoped_logs.get(self.scope, [])
        logs.append(record)
        self.scoped_logs[self.scope] = logs

    def set_scope(self, scope: str = base_scope):
        self.scope = scope

    def get_all_logs(self) -> List[LogRecord]:
        all_logs = []
        for key in self.scoped_logs:
            logs = self.scoped_logs.get(key, [])
            all_logs.extend(logs)
        return all_logs

    def get_logs(self, scope: Optional[str] = None) -> List[LogRecord]:
        scope = scope or self.scope
        if scope == all_scopes:
            return self.get_all_logs()
        logs = self.scoped_logs.get(scope, [])
        return logs


class LoggerConfig:
    def __init__(self, config={}, level=logging.NOTSET, scope=base_scope):
        self.config = config
        self.level = level
        self.scope = scope


_logger = logging.getLogger(name)
handler = ScopeHandler()
scoped_logs: Dict[str, List[LogRecord]] = {}
logger_config = LoggerConfig()


def get_logger():
    _setup_handler(logger_config.level, logger_config.scope)

    if logger_config.config:
        logging.config.dictConfig(logger_config.config)

    _logger.setLevel(logger_config.level)

    get_scope_handler()

    return _logger


def set_config(config=None):
    if config is not None:
        logger_config.config = config
        logging.config.dictConfig(logger_config.config)


def set_level(level=None):
    if level is not None:
        logger_config.level = level
        _logger.setLevel(level)


def set_scope(scope: str = base_scope):
    logger_config.scope = scope
    scope_handler = get_scope_handler()
    scope_handler.set_scope(scope)


def _setup_handler(log_level=logging.NOTSET, scope=base_scope) -> ScopeHandler:
    global handler
    if not handler:
        handler = ScopeHandler(log_level, scope)
    return handler


def get_scope_handler() -> ScopeHandler:
    global _logger
    try:
        scope_handler: ScopeHandler = [
            h for h in _logger.handlers if isinstance(h, ScopeHandler)
        ][0]
        return scope_handler
    except IndexError:
        hdlr = _setup_handler()
        _logger.addHandler(hdlr)
        return hdlr


logger = get_logger()
