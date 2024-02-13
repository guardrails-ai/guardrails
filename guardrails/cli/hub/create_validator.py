import os
from datetime import date
from string import Template

import typer
from pydash import pascal_case, snake_case

from guardrails.cli.hub.hub import hub
from guardrails.cli.logger import LEVELS, logger

validator_template = Template(
    """from typing import Any, Callable, Dict, Optional

from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)


@register_validator(name="guardrails/${package_name}", data_type="string")
class ${class_name}(Validator):
    \"""# Overview

    | Developed by | {Your organization name} |
    | Date of development | ${dev_date} |
    | Validator type | Format |
    | Blog |  |
    | License | Apache 2 |
    | Input/Output | Output |

    # Description

    This validator ensures that a generated output is the literal \"pass\".

    # Installation

    ```bash
    $ guardrails hub install hub://guardrails/${package_name}
    ```

    # Usage Examples

    ## Validating string output via Python

    In this example, we'll test that a generated word is `pass`.

    ```python
    # Import Guard and Validator
    from guardrails.hub import ${class_name}
    from guardrails import Guard

    # Initialize Validator
    val = ${class_name}()

    # Setup Guard
    guard = Guard.from_string(
        validators=[val, ...],
    )

    guard.parse(\"pass\")  # Validator passes
    guard.parse(\"fail\")  # Validator fails
    ```

    ## Validating JSON output via Python

    In this example, we verify that a processes's status is specified as `pass`.

    ```python
    # Import Guard and Validator
    from pydantic import BaseModel
    from guardrails.hub import ${class_name}
    from guardrails import Guard

    val = ${class_name}()

    # Create Pydantic BaseModel
    class Process(BaseModel):
    		process_name: str
    		status: str = Field(validators=[val])

    # Create a Guard to check for valid Pydantic output
    guard = Guard.from_pydantic(output_class=Process)

    # Run LLM output generating JSON through guard
    guard.parse(\"""
    {
    		"process_name": "templating",
    		"status": "pass"
    }
    \""")
    ```

    # API Reference

    `__init__`
    - `arg_1`: A placeholder argument to demonstrate how to use init arguments.
    - `arg_2`: Another placeholder argument to demonstrate how to use init arguments.
    - `on_fail`: The policy to enact when a validator fails.

    # Dependencies

    ## Production
    guardrails-ai >= 0.3.2

    ## Development
    pytest
    pyright
    ruff
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
        assert result.error_message == "{A descriptive but concise error message about why validation failed}"
        assert result.fix_value == "fails"
"""
)


@hub.command(name="create-validator")
def create_validator(
    name: str = typer.Argument(help="The name for your validator."),
    filepath: str = typer.Argument(
        help="The location to write your validator template to",
        default="./{validator_name}.py",
    ),
):
    package_name = snake_case(name)
    class_name = pascal_case(name)
    if not filepath or filepath == "./{validator_name}.py":
        filepath = f"./{package_name}.py"

    template = validator_template.safe_substitute(
        {
            "package_name": package_name,
            "class_name": class_name,
            "filepath": filepath,
            "dev_date": date.today().strftime("%b %d, %Y"),
        }
    )

    target = os.path.abspath(filepath)
    with open(target, "w") as validator_file:
        validator_file.write(template)
        validator_file.close()

    success_message = Template(
        """

    Successfully created validator template at ${filepath}!

    Make any necessary changes then submit for review with the following command:

    guardrails hub submit ${package_name} ${filepath}
    """
    ).safe_substitute({"filepath": filepath, "package_name": package_name})
    logger.log(level=LEVELS.get("SUCCESS"), msg=success_message)  # type: ignore
