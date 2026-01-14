from typing import Any, Callable, Dict, Optional

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

        reading_time: The maximum reading time in minutes.
    """

    def __init__(self, reading_time: int, on_fail: Optional[Callable] = None):
        super().__init__(
            on_fail=on_fail,
            reading_time=reading_time,
        )
        self._max_time = reading_time

    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        logger.debug(f"Validating {value} can be read in less than {self._max_time} minutes...")

        # Estimate the reading time of the string
        reading_time = len(value.split()) / 200
        logger.debug(f"Estimated reading time {reading_time} minutes...")

        if (reading_time - self._max_time) > 0:
            logger.error(f"{value} took {reading_time} minutes to read")
            return FailResult(
                error_message=f"String should be readable within {self._max_time} minutes."
            )

        return PassResult()
