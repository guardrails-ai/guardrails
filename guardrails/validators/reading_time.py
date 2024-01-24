from typing import Any, Dict

from guardrails.logger import logger
from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)


@register_validator(name="reading-time", data_type="string")
class ReadingTime(Validator):
    """Validates that the a string can be read in less than a certain amount of
    time.

    **Key Properties**

    | Property                      | Description                         |
    | ----------------------------- | ----------------------------------- |
    | Name for `format` attribute   | `reading-time`                      |
    | Supported data types          | `string`                            |
    | Programmatic fix              | None                                |

    Args:

        reading_time: The maximum reading time.
    """

    def __init__(self, reading_time: int, on_fail: str = "fix"):
        super().__init__(on_fail=on_fail, reading_time=reading_time)
        self._max_time = reading_time

    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        logger.debug(
            f"Validating {value} can be read in less than {self._max_time} seconds..."
        )

        # Estimate the reading time of the string
        reading_time = len(value.split()) / 200 * 60
        logger.debug(f"Estimated reading time {reading_time} seconds...")

        if abs(reading_time - self._max_time) > 1:
            logger.error(f"{value} took {reading_time} to read")
            return FailResult(
                error_message=f"String should be readable "
                f"within {self._max_time} minutes.",
                fix_value=value,
            )

        return PassResult()
