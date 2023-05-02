import logging
import logging.config


def configure_logging(logging_config=None, log_level=None):
    if logging_config is not None:
        logging.config.dictConfig(logging_config)

    if log_level is not None:
        logger = logging.getLogger(__name__)
        logger.setLevel(log_level)
