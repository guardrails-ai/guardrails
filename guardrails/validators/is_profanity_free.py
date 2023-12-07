from typing import Any, Dict

from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)


@register_validator(name="is-profanity-free", data_type="string")
class IsProfanityFree(Validator):
    """Validates that a translated text does not contain profanity language.

    This validator uses the `alt-profanity-check` package to check if a string
    contains profanity language.

    **Key Properties**

    | Property                      | Description                       |
    | ----------------------------- | --------------------------------- |
    | Name for `format` attribute   | `is-profanity-free`               |
    | Supported data types          | `string`                          |
    | Programmatic fix              | None                              |
    """

    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        try:
            from profanity_check import predict  # type: ignore
        except ImportError:
            raise ImportError(
                "`is-profanity-free` validator requires the `alt-profanity-check`"
                "package. Please install it with `poetry add profanity-check`."
            )

        prediction = predict([value])
        if prediction[0] == 1:
            return FailResult(
                error_message=f"{value} contains profanity. "
                f"Please return a profanity-free output.",
                fix_value="",
            )
        return PassResult()
