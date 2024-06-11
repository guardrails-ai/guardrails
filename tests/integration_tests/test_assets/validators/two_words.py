from typing import Any, Dict

from pydash.strings import words as _words

from guardrails.logger import logger
from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)


@register_validator(name="two-words", data_type="string")
class TwoWords(Validator):
    """Validates that a value is two words.

    **Key Properties**

    | Property                      | Description                       |
    | ----------------------------- | --------------------------------- |
    | Name for `format` attribute   | `two-words`                       |
    | Supported data types          | `string`                          |
    | Programmatic fix              | Pick the first two words.         |
    """

    def _get_fix_value(self, value: str) -> str:
        words = value.split()
        if len(words) == 1:
            words = _words(value)

        if len(words) == 1:
            value = f"{value} {value}"
            words = value.split()

        return " ".join(words[:2])

    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        logger.debug(f"Validating {value} is two words...")

        if len(value.split()) != 2:
            return FailResult(
                error_message="must be exactly two words",
                fix_value=self._get_fix_value(str(value)),
            )

        return PassResult()
