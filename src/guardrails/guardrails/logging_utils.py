from guardrails.logger import set_config, set_level


# TODO: Support a sink for logs so that they are not solely held in memory
def configure_logging(logging_config=None, log_level=None):
    set_config(logging_config)
    set_level(log_level)
