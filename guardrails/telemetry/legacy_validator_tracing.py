from operator import attrgetter
from typing import Any, List
from guardrails_api_client.models import Reask
from guardrails.actions.filter import Filter
from guardrails.actions.refrain import Refrain
from guardrails.call_tracing.trace_handler import TraceHandler
from guardrails.classes.validation.validator_logs import ValidatorLogs
from guardrails.telemetry.common import get_span
from guardrails.utils.casting_utils import to_string


def get_result_type(before_value: Any, after_value: Any, outcome: str):
    try:
        if isinstance(after_value, (Filter, Refrain, Reask)):
            name = after_value.__class__.__name__.lower()
        elif after_value != before_value:
            name = "fix"
        else:
            name = outcome
        return name
    except Exception:
        return type(after_value)


# TODO: How do we depreciate this?
# We want to encourage users to utilize the validator spans
#   instead of the events on the step span
def trace_validator_result(
    current_span, validator_log: ValidatorLogs, attempt_number: int, **kwargs
):
    (
        validator_name,
        value_before_validation,
        validation_result,
        value_after_validation,
        start_time,
        end_time,
        instance_id,
    ) = attrgetter(
        "registered_name",
        "value_before_validation",
        "validation_result",
        "value_after_validation",
        "start_time",
        "end_time",
        "instance_id",
    )(validator_log)
    result = (
        validation_result.outcome
        if hasattr(validation_result, "outcome")
        and validation_result.outcome is not None
        else "unknown"
    )
    result_type = get_result_type(
        value_before_validation, value_after_validation, result
    )

    event = {
        "validator_name": validator_name,
        "attempt_number": attempt_number,
        "result": result,
        "result_type": result_type,
        "input": to_string(value_before_validation),
        "output": to_string(value_after_validation),
        "start_time": start_time.isoformat() if start_time else None,
        "end_time": end_time.isoformat() if end_time else None,
        "instance_id": instance_id,
        **kwargs,
    }

    TraceHandler().log_validator(validator_log)

    current_span.add_event(
        f"{validator_name}_result",
        {k: v for k, v in event.items() if v is not None},
    )


# TODO: How do we depreciate this?
# We want to encourage users to utilize the validator spans
#   instead of the events on the step span
def trace_validation_result(
    validation_logs: List[ValidatorLogs],
    attempt_number: int,
    current_span=None,
):
    _current_span = get_span(current_span)
    if _current_span is not None:
        for log in validation_logs:
            trace_validator_result(_current_span, log, attempt_number)
