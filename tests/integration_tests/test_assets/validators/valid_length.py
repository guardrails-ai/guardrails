import string
from typing import Callable, Dict, List, Optional, Union

import rstr

from guardrails.logger import logger
from guardrails.utils.casting_utils import to_int
from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)


@register_validator(name="length", data_type=["string", "list"])
class ValidLength(Validator):
    """Validates that the length of value is within the expected range.

    **Key Properties**

    | Property                      | Description                       |
    | ----------------------------- | --------------------------------- |
    | Name for `format` attribute   | `length`                          |
    | Supported data types          | `string`, `list`, `object`        |
    | Programmatic fix              | If shorter than the minimum, pad with empty last elements. If longer than the maximum, truncate. |

    Args:
        min: The inclusive minimum length.
        max: The inclusive maximum length.
    """  # noqa

    def __init__(
        self,
        min: Optional[int] = None,
        max: Optional[int] = None,
        on_fail: Optional[Callable] = None,
    ):
        super().__init__(
            on_fail=on_fail,
            min=min,
            max=max,
        )
        self._min = to_int(min)
        self._max = to_int(max)

    def validate(self, value: Union[str, List], metadata: Dict) -> ValidationResult:
        """Validates that the length of value is within the expected range."""
        logger.debug(f"Validating {value} is in length range {self._min} - {self._max}...")

        if self._min is not None and len(value) < self._min:
            logger.debug(f"Value {value} is less than {self._min}.")

            # Repeat the last character to make the value the correct length.
            if isinstance(value, str):
                if not value:
                    last_val = rstr.rstr(string.ascii_lowercase, 1)
                else:
                    last_val = value[-1]
                corrected_value = value + last_val * (self._min - len(value))
            else:
                if not value:
                    last_val = [rstr.rstr(string.ascii_lowercase, 1)]
                else:
                    last_val = [value[-1]]
                # extend value by padding it out with last_val
                corrected_value = value.extend([last_val] * (self._min - len(value)))

            return FailResult(
                error_message=f"Value has length less than {self._min}. "
                f"Please return a longer output, "
                f"that is shorter than {self._max} characters.",
                fix_value=corrected_value,
            )

        if self._max is not None and len(value) > self._max:
            logger.debug(f"Value {value} is greater than {self._max}.")
            return FailResult(
                error_message=f"Value has length greater than {self._max}. "
                f"Please return a shorter output, "
                f"that is shorter than {self._max} characters.",
                fix_value=value[: self._max],
            )

        return PassResult()
