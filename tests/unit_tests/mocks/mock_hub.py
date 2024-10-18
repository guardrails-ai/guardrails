from typing import Any, Dict

from guardrails.validator_base import (
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)


@register_validator(name="mock-validator", data_type="string")
class MockValidator(Validator):
    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        return PassResult()
