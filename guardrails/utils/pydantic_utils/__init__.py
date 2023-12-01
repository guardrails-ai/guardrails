import pydantic.version

PYDANTIC_VERSION = pydantic.version.VERSION

if PYDANTIC_VERSION.startswith("1"):
    from .v1 import (
        ArbitraryModel,
        add_pydantic_validators_as_guardrails_validators,
        add_validator,
        convert_pydantic_model_to_datatype,
        convert_pydantic_model_to_openai_fn,
    )
else:
    from .v2 import (
        ArbitraryModel,
        add_pydantic_validators_as_guardrails_validators,
        add_validator,
        convert_pydantic_model_to_datatype,
        convert_pydantic_model_to_openai_fn,
    )


__all__ = [
    "add_validator",
    "add_pydantic_validators_as_guardrails_validators",
    "convert_pydantic_model_to_openai_fn",
    "convert_pydantic_model_to_datatype",
    "ArbitraryModel",
]
