import os
from pydash import snake_case, pascal_case
import typer
from guardrails.cli.hub.hub import hub
from guardrails.cli.logger import LEVELS, logger
from string import Template

validator_template = Template("""from typing import Any, Callable, Dict, Optional

from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)

# List any additional dependencies here.
\"""
dependencies = [                              
    "guardrails-ai>=0.3.2"
]
                              
[project.optional-dependencies]
dev = [
    "pytest"
] 
\"""
                              

@register_validator(name="guardrails/${package_name}", data_type="string")
class ${class_name}(Validator):
    \"""Validates that {fill in how you validator interacts with the passed value}.

    **Key Properties**

    | Property                      | Description                       |
    | ----------------------------- | --------------------------------- |
    | Name for `format` attribute   | `guardrails/${package_name}`   |
    | Supported data types          | `string`                          |
    | Programmatic fix              | {If you support programmatic fixes, explain it here. Otherwise `None`} |

    Args:
        arg_1 (string): {Description of the argument here}
        arg_2 (string): {Description of the argument here}
    \"""  # noqa

    # If you don't have any init args, you can omit the __init__ method.
    def __init__(
        self,
        arg_1: str,
        arg_2: str,
        on_fail: Optional[Callable] = None,
    ):
        super().__init__(on_fail=on_fail, arg_1=arg_1, arg_2=arg_2)
        self._arg_1 = arg_1
        self._arg_2 = arg_2

    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        \"""Validates that {fill in how you validator interacts with the passed value}.\"""
        # Add your custom validator logic here and return a PassResult or FailResult accordingly.
        if value != "pass": # FIXME
            return FailResult(
                error_message="{A descriptive but concise error message about why validation failed}",
                fix_value="{The programmtic fix if applicable, otherwise remove this kwarg.}",
            )
        return PassResult()


# Run tests via `pytest -rP ${filepath}`
class Test${class_name}:
    def test_success_case(self):
        validator = ${class_name}("s")
        result = validator.validate("pass", {})
        assert isinstance(result, PassResult) is True

    def test_failure_case(self):
        validator = ${class_name}("s")
        result = validator.validate("fail", {})
        assert isinstance(result, FailResult) is True
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
    package_name = snake_case(name)
    class_name = pascal_case(name)
    if not filepath or filepath == "./{validator_name}.py":
        filepath = f"./{package_name}.py"

    template = validator_template.safe_substitute({
        "package_name": package_name,
        "class_name": class_name,
        "filepath": filepath
    })
    
    target = os.path.abspath(filepath)
    with open(target, 'w') as validator_file:
        validator_file.write(template)
        validator_file.close()

    success_message = Template(
        """

    Successfully created validator template at ${filepath}!

    Make any necessary changes then submit for review with the following command:

    guardrails hub submit ${package_name} ${filepath}
    """
    ).safe_substitute(
        {"filepath": filepath, "package_name": package_name}
    )
    logger.log(level=LEVELS.get("SUCCESS"), msg=success_message)  # type: ignore
    