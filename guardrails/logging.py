import logging
import logging.config

LOGGING_CONFIG = {
    "version": 1,
    "formatters": {
        "default": {
            "format": "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "default",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "your_package": {
            "level": "DEBUG",
            "handlers": ["console"],
            "propagate": False,
        },
    },
    "root": {
        "level": "WARNING",
        "handlers": ["console"],
    },
}


def configure_logging(logging_config=None, log_level=None):
    if logging_config is None:
        logging_config = LOGGING_CONFIG

    logging.config.dictConfig(logging_config)

    if log_level is not None:
        logger = logging.getLogger(__name__)
        logger.setLevel(log_level)
