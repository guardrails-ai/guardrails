from typing import Any, Callable, Dict, Union
from guardrails import Validator, register_validator
from guardrails.validators import FailResult, PassResult, ValidationResult

class MockValidator(Validator):
    def __init__(
            self,
            name: str,
            on_fail: Union[str, Callable] = None,
            should_pass: bool = True
        ):
        register_validator(name=name, data_type=["string"])(self)
        super().__init__(on_fail=on_fail)
        self.name = name
        self.on_fail = on_fail
        self.should_pass = should_pass

    def validate(self, value: Any, metadata: Dict[str, Any]) -> ValidationResult:
        if self.should_pass:
            return PassResult()
        else:
            return FailResult(
                error_message=f"Value is not valid.",
            )
    
