import asyncio
import itertools
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Tuple

from guardrails.datatypes import FieldValidation
from guardrails.utils.logs_utils import FieldValidationLogs, ValidatorLogs
from guardrails.utils.reask_utils import FieldReAsk, ReAsk
from guardrails.validators import (
    FailResult,
    Filter,
    PassResult,
    PydanticReAsk,
    Refrain,
    Validator,
    ValidatorError,
)

logger = logging.getLogger(__name__)


class ValidatorServiceBase:
    """Base class for validator services."""

    def perform_correction(
        self,
        results: List[FailResult],
        value: Any,
        validator: Validator,
        on_fail_descriptor: str,
    ):
        if on_fail_descriptor == "fix":
            return results[0].fix_value
        elif on_fail_descriptor == "fix_reask":
            fixed_value = results[0].fix_value
            result = validator.validate(fixed_value, results[0].metadata)
            if result.metadata is None:
                result.metadata = result.metadata

            if isinstance(result, FailResult):
                return FieldReAsk(
                    incorrect_value=fixed_value,
                    fail_results=results,
                )

            return fixed_value
        if on_fail_descriptor == "custom":
            return validator.on_fail_method(value, results[0])
        if on_fail_descriptor == "reask":
            return FieldReAsk(
                incorrect_value=value,
                fail_results=results,
            )
        if on_fail_descriptor == "exception":
            raise ValidatorError(
                "Validation failed for field with errors: "
                + ", ".join([result.error_message for result in results])
            )
        if on_fail_descriptor == "filter":
            return Filter()
        if on_fail_descriptor == "refrain":
            return Refrain()
        if on_fail_descriptor == "noop":
            return value
        else:
            raise ValueError(
                f"Invalid on_fail_descriptor {on_fail_descriptor}, "
                f"expected 'fix' or 'exception'."
            )

    def run_validator(
        self, validation_logs, validator, value, metadata
    ) -> ValidatorLogs:
        validator_class_name = validator.__class__.__name__
        validator_logs = ValidatorLogs(
            validator_name=validator_class_name,
            value_before_validation=value,
        )
        validation_logs.validator_logs.append(validator_logs)

        result = validator.validate(value, metadata)
        if result is None:
            result = PassResult()

        validator_logs.validation_result = result
        return validator_logs


class SequentialValidatorService(ValidatorServiceBase):
    def run_validators(
        self,
        validation_logs: FieldValidationLogs,
        validator_setup: FieldValidation,
        value: Any,
        metadata: Dict[str, Any],
    ):
        # Validate the field
        for validator in validator_setup.validators:
            validator_logs = self.run_validator(
                validation_logs, validator, value, metadata
            )
            validation_logs.validator_logs.append(validator_logs)

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
            metadata = validator_logs.validation_result.metadata

            if isinstance(value, (Refrain, Filter, ReAsk, PydanticReAsk)):
                return value, metadata
        return value, metadata

    def validate_dependents(self, value, metadata, validator_setup, validation_logs):
        for child_setup in validator_setup.children:
            child_schema = value[child_setup.key]
            child_validation_logs = FieldValidationLogs()
            validation_logs.children[child_setup.key] = child_validation_logs
            child_schema, metadata = self.validate(
                child_schema,
                metadata,
                child_setup,
                child_validation_logs,
            )
            value[child_setup.key] = child_schema

    def validate(
        self,
        value: Any,
        metadata: dict,
        validator_setup: FieldValidation,
        validation_logs: FieldValidationLogs,
    ) -> Tuple[Any, dict]:
        # Validate children first
        if validator_setup.children:
            self.validate_dependents(value, metadata, validator_setup, validation_logs)

        # Validate the field
        value, metadata = self.run_validators(
            validation_logs, validator_setup, value, metadata
        )

        return value, metadata


class MultiprocMixin:
    multiprocessing_executor: ProcessPoolExecutor = None
    process_count = int(os.environ.get("GUARDRAILS_PROCESS_COUNT", 10))

    def __init__(self):
        if MultiprocMixin.multiprocessing_executor is None:
            MultiprocMixin.multiprocessing_executor = ProcessPoolExecutor(
                max_workers=MultiprocMixin.process_count
            )


class AsyncValidatorService(ValidatorServiceBase, MultiprocMixin):
    def group_validators(self, validators):
        groups = itertools.groupby(
            validators, key=lambda v: (v.on_fail_descriptor, v.override_value_on_pass)
        )
        for (on_fail_descriptor, override_on_pass), group in groups:
            if override_on_pass or on_fail_descriptor in ["fix", "fix_reask", "custom"]:
                for validator in group:
                    yield on_fail_descriptor, [validator]
            else:
                yield on_fail_descriptor, list(group)

    async def run_validators(self, validation_logs, validator_setup, value, metadata):
        loop = asyncio.get_running_loop()
        for on_fail, validator_group in self.group_validators(
            validator_setup.validators
        ):
            parallel_tasks = []
            validators_logs = []
            for validator in validator_group:
                if validator.run_in_separate_process:
                    # queue the validators to run in a separate process
                    parallel_tasks.append(
                        loop.run_in_executor(
                            self.multiprocessing_executor,
                            self.run_validator,
                            validation_logs,
                            validator,
                            value,
                            metadata,
                        )
                    )
                else:
                    # run the validators in the current process
                    result = self.run_validator(
                        validation_logs, validator, value, metadata
                    )
                    validators_logs.append(result)

            # wait for the parallel tasks to finish
            if parallel_tasks:
                parallel_results = await asyncio.gather(*parallel_tasks)
                validation_logs.validator_logs.extend(parallel_results)
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
            if isinstance(value, (Filter, Refrain, FieldReAsk, PydanticReAsk)):
                return value, metadata

        return value, metadata

    async def validate_dependents(
        self, value, metadata, validator_setup, validation_logs
    ):
        async def process_child(child_setup):
            child_value = value[child_setup.key]
            child_validation_logs = FieldValidationLogs()
            validation_logs.children[child_setup.key] = child_validation_logs
            new_child_value, new_metadata = await self.async_validate(
                child_value,
                metadata,
                child_setup,
                child_validation_logs,
            )
            return child_setup.key, new_child_value, new_metadata

        tasks = [process_child(child_setup) for child_setup in validator_setup.children]
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
        validator_setup: FieldValidation,
        validation_logs: FieldValidationLogs,
    ) -> Tuple[Any, dict]:
        # Validate children first
        if validator_setup.children:
            await self.validate_dependents(
                value, metadata, validator_setup, validation_logs
            )

        # Validate the field
        value, metadata = await self.run_validators(
            validation_logs, validator_setup, value, metadata
        )

        return value, metadata

    def validate(
        self,
        value: Any,
        metadata: dict,
        validator_setup: FieldValidation,
        validation_logs: FieldValidationLogs,
    ) -> Tuple[Any, dict]:
        # Run validate_async in an async loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            raise RuntimeError(
                "Async event loop found, please call `validate_async` instead."
            )
        value, metadata = loop.run_until_complete(
            self.async_validate(
                value,
                metadata,
                validator_setup,
                validation_logs,
            )
        )
        return value, metadata


def validate(
    value: Any,
    metadata: dict,
    validator_setup: FieldValidation,
    validation_logs: FieldValidationLogs,
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
        validator_service = SequentialValidatorService()
    elif loop is not None and not loop.is_running():
        validator_service = AsyncValidatorService()
    else:
        validator_service = SequentialValidatorService()
    return validator_service.validate(
        value,
        metadata,
        validator_setup,
        validation_logs,
    )


async def async_validate(
    value: Any,
    metadata: dict,
    validator_setup: FieldValidation,
    validation_logs: FieldValidationLogs,
):
    validator_service = AsyncValidatorService()
    return await validator_service.async_validate(
        value,
        metadata,
        validator_setup,
        validation_logs,
    )
