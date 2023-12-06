from typing import Any, Callable, Dict, Union

from guardrails import Validator, register_validator
from guardrails.validators import FailResult, PassResult, ValidationResult


def create_mock_validator(
    name: str,
    on_fail: Union[str, Callable] = None,
    should_pass: bool = True,
    return_value: Any = None,
):
    def validate(self, value: Any, metadata: Dict[str, Any]) -> ValidationResult:
        if self.should_pass:
            return self.return_value if self.return_value is not None else PassResult()
        else:
            return FailResult(
                error_message="Value is not valid.",
            )

    validator_type = type(
        name,
        (Validator,),
        {
            "validate": validate,
            "name": name,
            "on_fail": on_fail,
            "should_pass": should_pass,
            "return_value": return_value,
        },
    )
    register_validator(name=name, data_type=["string"])(validator_type)
    return validator_type
