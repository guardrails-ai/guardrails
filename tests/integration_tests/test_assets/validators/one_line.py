from typing import Any, Dict

from guardrails.logger import logger
from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)


@register_validator(name="one-line", data_type="string")
class OneLine(Validator):
    """Validates that a value is a single line, based on whether or not the
    output has a newline character (\\n).

    **Key Properties**

    | Property                      | Description                            |
    | ----------------------------- | -------------------------------------- |
    | Name for `format` attribute   | `one-line`                             |
    | Supported data types          | `string`                               |
    | Programmatic fix              | Keep the first line, delete other text |
    """

    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        logger.debug(f"Validating {value} is a single line...")

        if len(value.splitlines()) > 1:
            return FailResult(
                error_message=f"Value {value} is not a single line.",
                fix_value=value.splitlines()[0],
            )

        return PassResult()
