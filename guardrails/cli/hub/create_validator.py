import os
from pydash import snake_case
import typer
from os.path import expanduser
from guardrails.cli.hub.hub import hub
from string import Template

validator_template = Template("""from typing import Any, Dict

from guardrails.logger import logger
from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)


@register_validator(name="guardrails/${package_name}", data_type=["string", "list"])
class ${class_name}(Validator):
    \"""Validates that a string or list ends with a given value.

    **Key Properties**

    | Property                      | Description                        |
    | ----------------------------- | ---------------------------------  |
    | Name for `format` attribute   | `hub://guardrails/${package_name}` |
    | Supported data types          | `string`, `list`                   |
    | Programmatic fix              | Append the given value to the end. |

    Args:
        end: The required last element.
    \"""

    def __init__(self, end: str, on_fail: str = "fix"):
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

                              
class Test${class_name}:
    def test_success_case(self):
        validator = ${class_name}("s")
        result = validator.validate("pass")
        assert isintance(result, PassResult) is True

    def test_failure_case(self):
        validator = ${class_name}("s")
        result = validator.validate("fail")
        assert isintance(result, FailResult) is True
        assert result.error_message == "fail must end with s"
        assert result.fix_value == "fails"
""")


@hub.command(name='create-validator')
def create_validator(
    name: str = typer.Argument(
        help="The name for your validator."
    ),
    filepath: str = typer.Argument(
        help="The location to write your validator template to",
        default="./{validator_name}.py"
    )
):
    file_name = snake_case(name)
    target = os.path.join(os.getcwd(), file_name)
    