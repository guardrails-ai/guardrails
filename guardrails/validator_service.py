import logging
from typing import Any, Callable, Tuple

import pydantic

from guardrails.datatypes import FieldValidation
from guardrails.utils.logs_utils import ValidatorLogs, FieldValidationLogs

logger = logging.getLogger(__name__)


class ValidatorService:
    def validate(
        self,
        value: Any,
        metadata: dict,
        validator_setup: FieldValidation,
        validation_logs: FieldValidationLogs,
    ) -> Tuple[Any, dict]:
        # Validate children first
        for child_setup in validator_setup.children:
            child_schema = value[child_setup.key]
            child_schema, metadata = self.validate(
                child_schema,
                metadata,
                child_setup,
                validation_logs,
            )
            value[child_setup.key] = child_schema

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
            value = validator.validate_with_correction(value, metadata)
            validator_logs.value_after_validation = value
            logger.debug(
                f"Validator {validator_class_name} finished, "
                f"key {validator_setup.key} has value {value}."
            )

        return value, metadata
