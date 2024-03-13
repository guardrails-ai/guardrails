from typing import Any, Dict
from warnings import warn

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


@register_validator(name="ends-with", data_type="list")
class EndsWith(Validator):
    """Validates that a list ends with a given value.

    **Key Properties**

    | Property                      | Description                       |
    | ----------------------------- | --------------------------------- |
    | Name for `format` attribute   | `ends-with`                       |
    | Supported data types          | `list`                            |
    | Programmatic fix              | Append the given value to the list. |

    Args:
        end: The required last element.
    """

    def __init__(self, end: str, on_fail: str = "fix"):
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
                    hub_validator_name=VALIDATOR_NAMING[class_name][0],
                    hub_validator_url=VALIDATOR_NAMING[class_name][1],
                ),
                FutureWarning,
            )
        super().__init__(on_fail=on_fail, end=end)
        self._end = end

    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        logger.debug(f"Validating {value} ends with {self._end}...")

        if not value[-1] == self._end:
            return FailResult(
                error_message=f"{value} must end with {self._end}",
                fix_value=value + [self._end],
            )

        return PassResult()
