import os
from guardrails_api.otel.constants import none


def logs_are_disabled() -> bool:
    otel_logs_exporter = os.environ.get("OTEL_LOGS_EXPORTER", none)
    return otel_logs_exporter == none
