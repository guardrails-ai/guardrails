from typing import Any, Callable, Dict, Optional
from warnings import warn

from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)


@register_validator(name="pydantic_field_validator", data_type="all")
class PydanticFieldValidator(Validator):
    """Validates a specific field in a Pydantic model with the specified
    validator method.

    **Key Properties**

    | Property                      | Description                       |
    | ----------------------------- | --------------------------------- |
    | Name for `format` attribute   | `pydantic_field_validator`        |
    | Supported data types          | `Any`                             |
    | Programmatic fix              | Override with return value from `field_validator`.   |

    Args:

        field_validator (Callable): A validator for a specific field in a Pydantic model.
    """  # noqa

    override_value_on_pass = True

    def __init__(
        self,
        field_validator: Callable,
        on_fail: Optional[Callable[..., Any]] = None,
        **kwargs,
    ):
        warn(
            """
            PydanticFieldValidator is deprecated (v0.3.3); will be removed (v0.4.0).
            Instead, use a custom Guardrails validator as shown here:
            https://www.guardrailsai.com/docs/concepts/validators#custom-validators
            """,
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(on_fail, field_validator=field_validator, **kwargs)
        self.field_validator = field_validator

    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        try:
            validated_field = self.field_validator(value)
        except Exception as e:
            return FailResult(
                error_message=str(e),
                fix_value=None,
            )
        return PassResult(
            value_override=validated_field,
        )

    def to_prompt(self, with_keywords: bool = True) -> str:
        return self.field_validator.__name__
