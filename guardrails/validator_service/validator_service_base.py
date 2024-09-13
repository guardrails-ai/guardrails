from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Dict, Optional, Union

from guardrails.actions.filter import Filter
from guardrails.actions.refrain import Refrain
from guardrails.classes.history import Iteration
from guardrails.classes.validation.validation_result import (
    FailResult,
    ValidationResult,
)
from guardrails.errors import ValidationError
from guardrails.merge import merge
from guardrails.types import OnFailAction
from guardrails.utils.hub_telemetry_utils import HubTelemetry
from guardrails.classes.validation.validator_logs import ValidatorLogs
from guardrails.actions.reask import FieldReAsk
from guardrails.telemetry import trace_validator
from guardrails.utils.serialization_utils import deserialize, serialize
from guardrails.validator_base import Validator

ValidatorResult = Optional[Union[ValidationResult, Awaitable[ValidationResult]]]


@dataclass
class ValidatorRun:
    value: Any
    metadata: Dict
    on_fail_action: Union[str, OnFailAction]
    validator_logs: ValidatorLogs


class ValidatorServiceBase:
    """Base class for validator services."""

    def __init__(self, disable_tracer: Optional[bool] = True):
        self._disable_tracer = disable_tracer

    # NOTE: This is avoiding an issue with multiprocessing.
    #       If we wrap the validate methods at the class level or anytime before
    #       loop.run_in_executor is called, multiprocessing fails with a Pickling error.
    #       This is a well known issue without any real solutions.
    #       Using `fork` instead of `spawn` may alleviate the symptom for POSIX systems,
    #       but is relatively unsupported on Windows.
    def execute_validator(
        self,
        validator: Validator,
        value: Any,
        metadata: Optional[Dict],
        stream: Optional[bool] = False,
        *,
        validation_session_id: str,
        **kwargs,
        # TODO: Make this just Optional[ValidationResult]
        #       Also maybe move to SequentialValidatorService
    ) -> ValidatorResult:
        validate_func = validator.validate_stream if stream else validator.validate
        traced_validator = trace_validator(
            validator_name=validator.rail_alias,
            obj_id=id(validator),
            on_fail_descriptor=validator.on_fail_descriptor,
            validation_session_id=validation_session_id,
            **validator._kwargs,
        )(validate_func)
        if stream:
            result = traced_validator(value, metadata, **kwargs)
        else:
            result = traced_validator(value, metadata)
        return result

    def perform_correction(
        self,
        result: FailResult,
        value: Any,
        validator: Validator,
        rechecked_value: Optional[ValidationResult] = None,
    ):
        on_fail_descriptor = validator.on_fail_descriptor
        if on_fail_descriptor == OnFailAction.FIX:
            # FIXME: Should we still return fix_value if it is None?
            # I think we should warn and return the original value.
            return result.fix_value
        elif on_fail_descriptor == OnFailAction.FIX_REASK:
            # FIXME: Same thing here
            fixed_value = result.fix_value

            if isinstance(rechecked_value, FailResult):
                return FieldReAsk(
                    incorrect_value=fixed_value,
                    fail_results=[result],
                )

            return fixed_value
        if on_fail_descriptor == OnFailAction.CUSTOM:
            if validator.on_fail_method is None:
                raise ValueError("on_fail is 'custom' but on_fail_method is None")
            return validator.on_fail_method(value, result)
        if on_fail_descriptor == OnFailAction.REASK:
            return FieldReAsk(
                incorrect_value=value,
                fail_results=[result],
            )
        if on_fail_descriptor == OnFailAction.EXCEPTION:
            raise ValidationError(
                "Validation failed for field with errors: "
                + ", ".join([result.error_message])
            )
        if on_fail_descriptor == OnFailAction.FILTER:
            return Filter()
        if on_fail_descriptor == OnFailAction.REFRAIN:
            return Refrain()
        if on_fail_descriptor == OnFailAction.NOOP:
            return value
        else:
            raise ValueError(
                f"Invalid on_fail_descriptor {on_fail_descriptor}, "
                f"expected 'fix' or 'exception'."
            )

    def before_run_validator(
        self,
        iteration: Iteration,
        validator: Validator,
        value: Any,
        absolute_property_path: str,
    ) -> ValidatorLogs:
        validator_class_name = validator.__class__.__name__
        validator_logs = ValidatorLogs(
            validator_name=validator_class_name,
            value_before_validation=value,
            registered_name=validator.rail_alias,
            property_path=absolute_property_path,
            # If we ever re-use validator instances across multiple properties,
            #   this will have to change.
            instance_id=id(validator),
        )
        iteration.outputs.validator_logs.append(validator_logs)

        start_time = datetime.now()
        validator_logs.start_time = start_time

        return validator_logs

    def after_run_validator(
        self,
        validator: Validator,
        validator_logs: ValidatorLogs,
        result: Optional[ValidationResult],
    ) -> ValidatorLogs:
        end_time = datetime.now()
        validator_logs.validation_result = result
        validator_logs.end_time = end_time

        if not self._disable_tracer:
            # Get HubTelemetry singleton and create a new span to
            # log the validator usage
            _hub_telemetry = HubTelemetry()
            _hub_telemetry.create_new_span(
                span_name="/validator_usage",
                attributes=[
                    ("validator_name", validator.rail_alias),
                    ("validator_on_fail", validator.on_fail_descriptor),
                    (
                        "validator_result",
                        result.outcome
                        if isinstance(result, ValidationResult)
                        else None,
                    ),
                ],
                is_parent=False,  # This span will have no children
                has_parent=True,  # This span has a parent
            )

        return validator_logs

    def run_validator(
        self,
        iteration: Iteration,
        validator: Validator,
        value: Any,
        metadata: Dict,
        absolute_property_path: str,
        stream: Optional[bool] = False,
        **kwargs,
    ) -> ValidatorRun:
        raise NotImplementedError

    def merge_results(self, original_value: Any, new_values: list[Any]) -> Any:
        new_vals = deepcopy(new_values)
        current = new_values.pop()
        while len(new_values) > 0:
            nextval = new_values.pop()
            current = merge(
                serialize(current), serialize(nextval), serialize(original_value)
            )
            current = deserialize(original_value, current)
        if current is None and original_value is not None:
            # QUESTION: How do we escape hatch
            #    for when deserializing the merged value fails?

            # Should we return the original value?
            # return original_value

            # Or just pick one of the new values?
            return new_vals[0]
        return current
