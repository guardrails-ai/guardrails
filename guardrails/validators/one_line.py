from typing import Any, Dict, Optional, Callable
from warnings import warn

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

    def __init__(self, on_fail: Optional[Callable] = None):
        warn(
            """
            Using this validator from `guardrails.validators` is deprecated.
            Please install and import this validator from Guardrails Hub instead. 
            This validator would be removed from this module in the next major release.
            """,
            FutureWarning,
        )
        super().__init__(on_fail=on_fail)

    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        logger.debug(f"Validating {value} is a single line...")

        if len(value.splitlines()) > 1:
            return FailResult(
                error_message=f"Value {value} is not a single line.",
                fix_value=value.splitlines()[0],
            )

        return PassResult()
