from typing import Any, Callable, Dict, List, Optional

from guardrails.logger import logger
from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)


@register_validator(name="valid-choices", data_type="all")
class ValidChoices(Validator):
    """Validates that a value is within the acceptable choices.

    **Key Properties**

    | Property                      | Description                       |
    | ----------------------------- | --------------------------------- |
    | Name for `format` attribute   | `valid-choices`                   |
    | Supported data types          | `all`                             |
    | Programmatic fix              | None                              |

    Args:
        choices: The list of valid choices.
    """

    def __init__(self, choices: List[Any], on_fail: Optional[Callable] = None):
        super().__init__(on_fail=on_fail, choices=choices)
        self._choices = choices

    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        """Validates that a value is within a range."""
        logger.debug(f"Validating {value} is in choices {self._choices}...")

        if value not in self._choices:
            return FailResult(
                error_message=f"Value {value} is not in choices {self._choices}.",
            )

        return PassResult()
