import os
import logging


def get_logger():
    log_level = os.environ.get("LOGLEVEL", logging.INFO)
    logging.basicConfig(level=log_level)
    _logger = logging.getLogger("guardrails-api")
    # _logger.addHandler(otel_handler)
    return _logger


logger = get_logger()
