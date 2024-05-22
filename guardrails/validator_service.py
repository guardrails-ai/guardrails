import asyncio
import itertools
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from guardrails.actions.filter import Filter, apply_filters
from guardrails.actions.refrain import Refrain, apply_refrain
from guardrails.classes.history import Iteration
from guardrails.classes.output_type import OutputTypes
from guardrails.classes.validation.validation_result import (
    FailResult,
    PassResult,
    ValidationResult,
)
from guardrails.datatypes import FieldValidation
from guardrails.errors import ValidationError
from guardrails.logger import logger
from guardrails.types import ValidatorMap, OnFailAction
from guardrails.utils.hub_telemetry_utils import HubTelemetry
from guardrails.classes.validation.validator_logs import ValidatorLogs
from guardrails.utils.reask_utils import FieldReAsk, ReAsk
from guardrails.utils.telemetry_utils import trace_validation_result, trace_validator
from guardrails.validator_base import Validator


def key_not_empty(key: str) -> bool:
    return key is not None and len(str(key)) > 0


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
        self, validator: Validator, value: Any, metadata: Optional[Dict]
    ) -> ValidationResult:
        traced_validator = trace_validator(
            validator_name=validator.rail_alias,
            obj_id=id(validator),
            # TODO - re-enable once we have namespace support
            # namespace=validator.namespace,
            on_fail_descriptor=validator.on_fail_descriptor,
            **validator._kwargs,
        )(validator.validate)
        result = traced_validator(value, metadata)
        return result

    def perform_correction(
        self,
        results: List[FailResult],
        value: Any,
        validator: Validator,
        on_fail_descriptor: Union[OnFailAction, str],
    ):
        if on_fail_descriptor == OnFailAction.FIX:
            # FIXME: Should we still return fix_value if it is None?
            # I think we should warn and return the original value.
            return results[0].fix_value
        elif on_fail_descriptor == OnFailAction.FIX_REASK:
            # FIXME: Same thing here
            fixed_value = results[0].fix_value
            result = self.execute_validator(
                validator, fixed_value, results[0].metadata or {}
            )

            if isinstance(result, FailResult):
                return FieldReAsk(
                    incorrect_value=fixed_value,
                    fail_results=results,
                )

            return fixed_value
        if on_fail_descriptor == "custom":
            if validator.on_fail_method is None:
                raise ValueError("on_fail is 'custom' but on_fail_method is None")
            return validator.on_fail_method(value, results)
        if on_fail_descriptor == OnFailAction.REASK:
            return FieldReAsk(
                incorrect_value=value,
                fail_results=results,
            )
        if on_fail_descriptor == OnFailAction.EXCEPTION:
            raise ValidationError(
                "Validation failed for field with errors: "
                + ", ".join([result.error_message for result in results])
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

    def run_validator(
        self,
        iteration: Iteration,
        validator: Validator,
        value: Any,
        metadata: Dict,
        absolute_property_path: str,
    ) -> ValidatorLogs:
        validator_class_name = validator.__class__.__name__
        validator_logs = ValidatorLogs(
            validator_name=validator_class_name,
            value_before_validation=value,
            registered_name=validator.rail_alias,
            property_path=absolute_property_path,
        )
        iteration.outputs.validator_logs.append(validator_logs)

        start_time = datetime.now()
        result = self.execute_validator(validator, value, metadata)
        end_time = datetime.now()

        result = validator.validate(value, metadata)
        if result is None:
            result = PassResult()

        validator_logs.validation_result = result
        validator_logs.start_time = start_time
        validator_logs.end_time = end_time
        # If we ever re-use validator instances across multiple properties,
        #   this will have to change.
        validator_logs.instance_id = id(validator)

        if not self._disable_tracer:
            # Get HubTelemetry singleton and create a new span to
            # log the validator usage
            _hub_telemetry = HubTelemetry()
            _hub_telemetry.create_new_span(
                span_name="/validator_usage",
                attributes=[
                    ("validator_name", validator.rail_alias),
                    ("validator_on_fail", validator.on_fail_descriptor),
                    ("validator_result", result.outcome),
                ],
                is_parent=False,  # This span will have no children
                has_parent=True,  # This span has a parent
            )

        return validator_logs


class SequentialValidatorService(ValidatorServiceBase):
    def run_validators(
        self,
        iteration: Iteration,
        validator_map: ValidatorMap,
        value: Any,
        metadata: Dict[str, Any],
        absolute_property_path: str,
        reference_property_path: str,
    ) -> Tuple[Any, Dict[str, Any]]:
        # Validate the field
        validators = validator_map.get(reference_property_path, [])
        for validator in validators:
            validator_logs = self.run_validator(
                iteration, validator, value, metadata, absolute_property_path
            )

            result = validator_logs.validation_result
            if isinstance(result, FailResult):
                value = self.perform_correction(
                    [result], value, validator, validator.on_fail_descriptor
                )
            elif isinstance(result, PassResult):
                if (
                    validator.override_value_on_pass
                    and result.value_override is not result.ValueOverrideSentinel
                ):
                    value = result.value_override
            else:
                raise RuntimeError(f"Unexpected result type {type(result)}")

            validator_logs.value_after_validation = value
            if result.metadata is not None:
                metadata = result.metadata

            if isinstance(value, (Refrain, Filter, ReAsk)):
                return value, metadata
        return value, metadata

    def validate(
        self,
        value: Any,
        metadata: dict,
        validator_map: ValidatorMap,
        iteration: Iteration,
        absolute_path: str = "$",
        reference_path: str = "$",
    ) -> Tuple[Any, dict]:
        ###
        # NOTE: The way validation can be executed now is fundamentally wide open.
        #   Since validators are tracked against the JSONPaths for the
        #       properties they should be applied to, we have the following options:
        #       1. Keep performing a Deep-First-Search
        #           - This is useful for backwards compatibility.
        #           - Is there something we gain by validating inside out?
        #       2. Swith to a Breadth-First-Search
        #           - Possible, no obvious advantages
        #       3. Run un-ordered
        #           - This would allow for true parallelism
        #           - Also means we're not unnecessarily iterating down through
        #               the object if there aren't any validations applied there.
        ###

        child_ref_path = reference_path.replace(".*", "")
        # Validate children first
        if isinstance(value, List):
            for index, child in enumerate(value):
                abs_child_path = f"{absolute_path}.{index}"
                ref_child_path = f"{child_ref_path}.*"
                child_value, metadata = self.validate(
                    child,
                    metadata,
                    validator_map,
                    iteration,
                    abs_child_path,
                    ref_child_path,
                )
                value[index] = child_value
        elif isinstance(value, Dict):
            for key in value:
                child = value.get(key)
                abs_child_path = f"{absolute_path}.{key}"
                ref_child_path = f"{child_ref_path}.{key}"
                child_value, metadata = self.validate(
                    child,
                    metadata,
                    validator_map,
                    iteration,
                    abs_child_path,
                    ref_child_path,
                )
                value[key] = child_value

        # Then validate the parent value
        value, metadata = self.run_validators(
            iteration, validator_map, value, metadata, absolute_path, reference_path
        )

        return value, metadata


class MultiprocMixin:
    multiprocessing_executor: Optional[ProcessPoolExecutor] = None
    process_count = int(os.environ.get("GUARDRAILS_PROCESS_COUNT", 10))

    def __init__(self):
        if MultiprocMixin.multiprocessing_executor is None:
            MultiprocMixin.multiprocessing_executor = ProcessPoolExecutor(
                max_workers=MultiprocMixin.process_count
            )


class AsyncValidatorService(ValidatorServiceBase, MultiprocMixin):
    def group_validators(self, validators: List[Validator]):
        groups = itertools.groupby(
            validators, key=lambda v: (v.on_fail_descriptor, v.override_value_on_pass)
        )
        # NOTE: This isn't ordering anything.
        #   If we want to yield fix-like valiators first,
        #       then we need to extract them outside of the loop.
        for (on_fail_descriptor, override_on_pass), group in groups:
            if override_on_pass or on_fail_descriptor in [
                OnFailAction.FIX,
                OnFailAction.FIX_REASK,
                "custom",
            ]:
                for validator in group:
                    yield on_fail_descriptor, [validator]
            else:
                yield on_fail_descriptor, list(group)

    async def run_validators(
        self,
        iteration: Iteration,
        validator_map: ValidatorMap,
        value: Any,
        metadata: Dict,
        absolute_property_path: str,
        reference_property_path: str,
    ):
        loop = asyncio.get_running_loop()
        validators = validator_map.get(reference_property_path, [])
        for on_fail, validator_group in self.group_validators(validators):
            parallel_tasks = []
            validators_logs: List[ValidatorLogs] = []
            for validator in validator_group:
                if validator.run_in_separate_process:
                    # queue the validators to run in a separate process
                    parallel_tasks.append(
                        loop.run_in_executor(
                            self.multiprocessing_executor,
                            self.run_validator,
                            iteration,
                            validator,
                            value,
                            metadata,
                            absolute_property_path,
                        )
                    )
                else:
                    # run the validators in the current process
                    result = self.run_validator(
                        iteration, validator, value, metadata, absolute_property_path
                    )
                    validators_logs.append(result)

            # wait for the parallel tasks to finish
            if parallel_tasks:
                parallel_results = await asyncio.gather(*parallel_tasks)
                validators_logs.extend(parallel_results)

            # process the results, handle failures
            fails = [
                logs
                for logs in validators_logs
                if isinstance(logs.validation_result, FailResult)
            ]
            if fails:
                fail_results = [logs.validation_result for logs in fails]
                value = self.perform_correction(
                    fail_results, value, validator_group[0], on_fail
                )

            # handle overrides
            if (
                len(validator_group) == 1
                and validator_group[0].override_value_on_pass
                and isinstance(validators_logs[0].validation_result, PassResult)
                and validators_logs[0].validation_result.value_override
                is not PassResult.ValueOverrideSentinel
            ):
                value = validators_logs[0].validation_result.value_override

            for logs in validators_logs:
                logs.value_after_validation = value

            # return early if we have a filter, refrain, or reask
            if isinstance(value, (Filter, Refrain, FieldReAsk)):
                return value, metadata

        return value, metadata

    async def validate_children(
        self,
        value: Any,
        metadata: Dict,
        validator_map: ValidatorMap,
        iteration: Iteration,
        abs_parent_path: str,
        ref_parent_path: str,
    ):
        async def validate_child(
            child_value: Any, *, key: Optional[str], index: Optional[int]
        ):
            child_key = key or index
            abs_child_path = f"{abs_parent_path}.{child_key}"
            ref_child_path = ref_parent_path
            if key is not None:
                ref_child_path = f"{ref_child_path}.{key}"
            elif index is not None:
                ref_child_path = f"{ref_child_path}.*"
            new_child_value, new_metadata = await self.async_validate(
                child_value,
                metadata,
                validator_map,
                iteration,
                abs_child_path,
                ref_child_path,
            )
            return key, new_child_value, new_metadata

        tasks = []
        if isinstance(value, List):
            for index, child in enumerate(value):
                tasks.append(validate_child(child, index=index))
        elif isinstance(value, Dict):
            for key in value:
                child = value.get(key)
                tasks.append(validate_child(child, key=key))

        results = await asyncio.gather(*tasks)

        for key, child_value, child_metadata in results:
            value[key] = child_value
            # TODO address conflicting metadata entries
            metadata = {**metadata, **child_metadata}

        return value, metadata

    async def async_validate(
        self,
        value: Any,
        metadata: dict,
        validator_map: ValidatorMap,
        iteration: Iteration,
        absolute_path: str = "$",
        reference_path: str = "$",
    ) -> Tuple[Any, dict]:
        child_ref_path = reference_path.replace(".*", "")
        # Validate children first
        if isinstance(value, List) or isinstance(value, Dict):
            self.validate_children(
                value, metadata, validator_map, iteration, absolute_path, child_ref_path
            )

        # Then validate the parent value
        value, metadata = await self.run_validators(
            iteration, validator_map, value, metadata, absolute_path, reference_path
        )

        return value, metadata

    def validate(
        self,
        value: Any,
        metadata: dict,
        validator_setup: FieldValidation,
        iteration: Iteration,
        path: Optional[str] = None,
    ) -> Tuple[Any, dict]:
        # Run validate_async in an async loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            raise RuntimeError(
                "Async event loop found, please call `validate_async` instead."
            )
        value, metadata = loop.run_until_complete(
            self.async_validate(value, metadata, validator_setup, iteration, path)
        )
        return value, metadata


def validate(
    value: Any,
    metadata: dict,
    validator_map: ValidatorMap,
    iteration: Iteration,
    disable_tracer: Optional[bool] = True,
    path: Optional[str] = None,
):
    process_count = int(os.environ.get("GUARDRAILS_PROCESS_COUNT", 10))

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None

    if process_count == 1:
        logger.warning(
            "Process count was set to 1 via the GUARDRAILS_PROCESS_COUNT"
            "environment variable."
            "This will cause all validations to run synchronously."
            "To run asynchronously, specify a process count"
            "greater than 1 or unset this environment variable."
        )
        validator_service = SequentialValidatorService(disable_tracer)
    elif loop is not None and not loop.is_running():
        validator_service = AsyncValidatorService(disable_tracer)
    else:
        validator_service = SequentialValidatorService(disable_tracer)
    return validator_service.validate(value, metadata, validator_map, iteration, path)


async def async_validate(
    value: Any,
    metadata: dict,
    validator_map: ValidatorMap,
    iteration: Iteration,
    disable_tracer: Optional[bool] = True,
    path: Optional[str] = None,
):
    validator_service = AsyncValidatorService(disable_tracer)
    return await validator_service.async_validate(
        value, metadata, validator_map, iteration, path
    )


def post_process_validation(
    validation_response: Any,
    attempt_number: int,
    iteration: Iteration,
    output_type: OutputTypes,
) -> Any:
    validated_response = apply_refrain(validation_response, output_type)

    # Remove all keys that have `Filter` values.
    validated_response = apply_filters(validated_response)

    trace_validation_result(
        validation_logs=iteration.validator_logs, attempt_number=attempt_number
    )

    return validated_response
