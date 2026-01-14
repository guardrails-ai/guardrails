import os
from guardrails_api.otel.logs import logs_are_disabled
from guardrails_api.otel.metrics import (
    initialize_metrics_collector,
    metrics_are_disabled,
    get_meter,  # noqa
)
from guardrails_api.otel.traces import (
    traces_are_disabled,
    initialize_tracer,
    get_tracer,  # noqa
)


def otel_is_disabled() -> bool:
    sdk_is_disabled = os.environ.get("OTEL_SDK_DISABLED") == "true"

    all_signals_disabled = traces_are_disabled() and metrics_are_disabled() and logs_are_disabled()
    return sdk_is_disabled or all_signals_disabled


def initialize():
    initialize_tracer()
    initialize_metrics_collector()
    # Logs are supported yet in the Python SDK
    # initialize_logs_collector()
