import platform
from typing import Optional
from guardrails.classes.credentials import Credentials
from guardrails.utils.hub_telemetry_utils import HubTelemetry
from guardrails.version import GUARDRAILS_VERSION
from guardrails.cli.logger import logger

config: Optional[Credentials] = None


def load_config_file() -> Credentials:
    global config
    if not config:
        config = Credentials.from_rc_file(logger)
    return config


def trace_if_enabled(command_name: str):
    config = load_config_file()
    if config.enable_metrics is True:
        telemetry = HubTelemetry()
        telemetry.create_new_span(
            f"guardrails-cli/{command_name}",
            [
                ("guardrails-version", GUARDRAILS_VERSION),
                ("python-version", platform.python_version()),
                ("system", platform.system()),
                ("platform", platform.platform()),
                ("arch", platform.architecture()[0]),
                ("machine", platform.machine()),
                ("processor", platform.processor()),
            ],
            True,
            False,
        )
