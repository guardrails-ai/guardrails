from openai.version import VERSION

OPENAI_VERSION = VERSION

if OPENAI_VERSION.startswith("0"):
    from .v0 import AsyncOpenAIClientV0 as AsyncOpenAIClient
    from .v0 import OpenAIClientV0 as OpenAIClient
    from .v0 import (
        OpenAIServiceUnavailableError,
        get_static_openai_acreate_func,
        get_static_openai_chat_acreate_func,
        get_static_openai_chat_create_func,
        get_static_openai_create_func,
    )
else:
    from .v1 import AsyncOpenAIClientV1 as AsyncOpenAIClient
    from .v1 import OpenAIClientV1 as OpenAIClient
    from .v1 import (
        OpenAIServiceUnavailableError,
        get_static_openai_acreate_func,
        get_static_openai_chat_acreate_func,
        get_static_openai_chat_create_func,
        get_static_openai_create_func,
    )


__all__ = [
    "OPENAI_VERSION",
    "AsyncOpenAIClient",
    "OpenAIClient",
    "get_static_openai_create_func",
    "get_static_openai_chat_create_func",
    "get_static_openai_acreate_func",
    "get_static_openai_chat_acreate_func",
    "OpenAIServiceUnavailableError",
]
