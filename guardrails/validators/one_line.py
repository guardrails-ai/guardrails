from typing import Any, Callable, Dict, Optional
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

    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        logger.debug(f"Validating {value} is a single line...")

        if len(value.splitlines()) > 1:
            return FailResult(
                error_message=f"Value {value} is not a single line.",
                fix_value=value.splitlines()[0],
            )

        return PassResult()
