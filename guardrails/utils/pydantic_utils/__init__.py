import pydantic.version

PYDANTIC_VERSION = pydantic.version.VERSION

if PYDANTIC_VERSION.startswith("1"):
    from .v1 import (
        convert_pydantic_model_to_openai_fn,
    )
else:
    from .v2 import (
        convert_pydantic_model_to_openai_fn,
    )


__all__ = [
    "convert_pydantic_model_to_openai_fn",
]
