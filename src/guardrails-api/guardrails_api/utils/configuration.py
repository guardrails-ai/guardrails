from guardrails_api.clients.postgres_client import postgres_is_enabled
from typing import Optional
import os


def valid_configuration(config: Optional[str] = ""):
    default_config_file = os.path.join(os.getcwd(), "./config.py")

    default_config_file_path = os.path.abspath(default_config_file)
    # If config.py is not present and
    # if a config filepath is not passed and
    # if postgres is not there (i.e. we’re using in-mem db)
    # then raise ConfigurationError
    has_default_config_file = os.path.isfile(default_config_file_path)

    has_config_file = (config != "" and config is not None) and os.path.isfile(
        os.path.abspath(config)
    )
    if not has_default_config_file and not has_config_file and not postgres_is_enabled():
        raise ConfigurationError(
            "Can not start. Configuration not provided and default"
            " configuration not found and postgres is not enabled."
        )
    return True


class ConfigurationError(Exception):
    pass
