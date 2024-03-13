import ast
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


@register_validator(name="bug-free-python", data_type="string")
class BugFreePython(Validator):
    """Validates that there are no Python syntactic bugs in the generated code.

    This validator checks for syntax errors by running `ast.parse(code)`,
    and will raise an exception if there are any.
    Only the packages in the `python` environment are available to the code snippet.

    **Key Properties**

    | Property                      | Description                       |
    | ----------------------------- | --------------------------------- |
    | Name for `format` attribute   | `bug-free-python`                 |
    | Supported data types          | `string`                          |
    | Programmatic fix              | None                              |
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
        logger.debug(f"Validating {value} is not a bug...")

        # The value is a Python code snippet. We need to check for syntax errors.
        try:
            ast.parse(value)
        except SyntaxError as e:
            return FailResult(
                error_message=f"Syntax error: {e.msg}",
            )

        return PassResult()
