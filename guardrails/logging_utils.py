from guardrails.logger import set_config, set_level


def configure_logging(logging_config=None, log_level=None):
    set_config(logging_config)
    set_level(log_level)
