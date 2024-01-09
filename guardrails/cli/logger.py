import coloredlogs, logging


logger = logging.getLogger("guardrails-cli")
coloredlogs.install(level='DEBUG', logger=logger)