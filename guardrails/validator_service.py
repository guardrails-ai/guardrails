import asyncio
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Tuple

from guardrails.datatypes import FieldValidation
from guardrails.utils.logs_utils import FieldValidationLogs, ValidatorLogs

logger = logging.getLogger(__name__)


class SequentialValidatorService:
    def run_validators(self, validation_logs, validator_setup, value, metadata):
        # Validate the field
        for validator in validator_setup.validators:
            validator_class_name = validator.__class__.__name__
            validator_logs = ValidatorLogs(
                validator_name=validator_class_name,
                value_before_validation=value,
            )
            validation_logs.validator_logs.append(validator_logs)
            logger.debug(
                f"Validating field {validator_setup.key} "
                f"with validator {validator_class_name}..."
            )

            # if validator.run_in_separate_process:
            #     logger.warning(
            #         "Running validators in a separate processes "
            #         "is not supported in synchronously, "
            #         "try invoking `guard` asynchronously instead."
            #     )
            value = validator.validate_with_correction(value, metadata)

            validator_logs.value_after_validation = value
            logger.debug(
                f"Validator {validator_class_name} finished, "
                f"key {validator_setup.key} has value {value}."
            )
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
    process_count = os.environ.get("GUARDRAILS_PROCESS_COUNT", 10)

    def __init__(self):
        if MultiprocMixin.multiprocessing_executor is None:
            MultiprocMixin.multiprocessing_executor = ProcessPoolExecutor(
                max_workers=MultiprocMixin.process_count
            )


class AsyncValidatorService(MultiprocMixin):
    def process_entrypoint(self, validator, value, metadata) -> tuple[Any, dict]:
        value = validator.validate_with_correction(value, metadata)
        return value, metadata

    async def run_validators(self, validation_logs, validator_setup, value, metadata):
        # Validate the field
        for validator in validator_setup.validators:
            validator_class_name = validator.__class__.__name__
            validator_logs = ValidatorLogs(
                validator_name=validator_class_name,
                value_before_validation=value,
            )
            validation_logs.validator_logs.append(validator_logs)
            logger.debug(
                f"Validating field {validator_setup.key} "
                f"with validator {validator_class_name}..."
            )

            if validator.run_in_separate_process:
                loop = asyncio.get_running_loop()
                task = loop.run_in_executor(
                    self.multiprocessing_executor,
                    self.process_entrypoint,
                    validator,
                    value,
                    metadata,
                )
                completed, pending = await asyncio.wait([task])
                if any(pending):
                    raise RuntimeError("Pending futures left?")
                value, metadata = [t.result() for t in completed][0]
            else:
                value = validator.validate_with_correction(value, metadata)

            validator_logs.value_after_validation = value
            logger.debug(
                f"Validator {validator_class_name} finished, "
                f"key {validator_setup.key} has value {value}."
            )
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
    loop = asyncio.get_event_loop()
    if loop.is_running():
        logger.warning("Async event loop found, but guard was invoked synchronously."
                       "For validator parallelization, please call `validate_async` instead.")
        validator_service = SequentialValidatorService()
    else:
        validator_service = AsyncValidatorService()
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
