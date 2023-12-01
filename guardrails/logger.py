import os
import logging
import logging.config
from typing import Dict, List
# from src.modules.otel_logger import handler as otel_handler

name = "guardrails-ai"
config = {}
log_level = logging.NOTSET
scoped_logs: Dict[str, List[str]] = {}
current_scope: str = None

def get_logger():
    logging.config.dictConfig(config)
    _logger = logging.getLogger(name)
    logger.setLevel(log_level)
    # _logger.addHandler(otel_handler)
    # TODO: Create custom log handler that appends logs to the current scope if it exists
    #       and also logs to stdout
    return _logger

def set_config(logging_config=None):
    if logging_config is not None:
        config = logging_config
    
def set_level(level=None):
    if level is not None:
        log_level = level


logger = get_logger()
