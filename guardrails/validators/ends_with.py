from typing import Any, Dict

from guardrails.logger import logger
from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)


@register_validator(name="ends-with", data_type=["string", "list"])
class EndsWith(Validator):
    """Validates that a list ends with a given value.

    **Key Properties**

    | Property                      | Description                       |
    | ----------------------------- | --------------------------------- |
    | Name for `format` attribute   | `ends-with`                       |
    | Supported data types          | `list`, `string                   |
    | Programmatic fix              | Append the given value to the list or string |

    Args:
        end: The required last element.
    """

    def __init__(self, end: str, on_fail: str = "fix"):
        super().__init__(on_fail=on_fail, end=end)
        self._end = end

    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        logger.debug(f"Validating {value} ends with {self._end}...")

        end = self._end
        if isinstance(value, list) and not isinstance(self._end, list):
            end = [self._end]

        ending_idxs = len(end)
        if not value[-ending_idxs:] == end:
            return FailResult(
                error_message=f"{value} must end with {end}",
                fix_value=value + end,
            )

        return PassResult()
