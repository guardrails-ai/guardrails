from typing import Any, Callable, Dict, Optional
from warnings import warn

from pydash.strings import words as _words

from guardrails.logger import logger
from guardrails.validator_base import (
    VALIDATOR_IMPORT_WARNING,
    VALIDATOR_NAMING,
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

    def __init__(self, on_fail: Optional[Callable] = None):
        class_name = self.__class__.__name__
        if class_name not in VALIDATOR_NAMING:
            warn(
                f"""Validator {class_name} is deprecated and
                will be removed after version 0.5.x.
                """,
                FutureWarning,
            )
        else:
            warn(
                VALIDATOR_IMPORT_WARNING.format(
                    validator_name=class_name,
                    hub_validator_name=VALIDATOR_NAMING.get(class_name)[0],
                    hub_validator_url=VALIDATOR_NAMING.get(class_name)[1],
                ),
                FutureWarning,
            )
        super().__init__(on_fail=on_fail)

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
