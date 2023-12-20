from typing import Any, Callable, Dict, Optional

from guardrails.logger import logger
from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)


@register_validator(name="valid-range", data_type=["integer", "float", "percentage"])
class ValidRange(Validator):
    """Validates that a value is within a range.

    **Key Properties**

    | Property                      | Description                       |
    | ----------------------------- | --------------------------------- |
    | Name for `format` attribute   | `valid-range`                     |
    | Supported data types          | `integer`, `float`, `percentage`  |
    | Programmatic fix              | Closest value within the range.   |

    Args:
        min: The inclusive minimum value of the range.
        max: The inclusive maximum value of the range.
    """

    def __init__(
        self,
        min: Optional[int] = None,
        max: Optional[int] = None,
        on_fail: Optional[Callable] = None,
    ):
        super().__init__(on_fail=on_fail, min=min, max=max)

        self._min = min
        self._max = max

    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        """Validates that a value is within a range."""
        logger.debug(f"Validating {value} is in range {self._min} - {self._max}...")

        val_type = type(value)

        if self._min is not None and value < val_type(self._min):
            return FailResult(
                error_message=f"Value {value} is less than {self._min}.",
                fix_value=self._min,
            )

        if self._max is not None and value > val_type(self._max):
            return FailResult(
                error_message=f"Value {value} is greater than {self._max}.",
                fix_value=self._max,
            )

        return PassResult()
