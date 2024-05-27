from openai.version import VERSION
from .v1 import AsyncOpenAIClientV1 as AsyncOpenAIClient
from .v1 import OpenAIClientV1 as OpenAIClient
from .v1 import (
    OpenAIServiceUnavailableError,
    get_static_openai_acreate_func,
    get_static_openai_chat_acreate_func,
    get_static_openai_chat_create_func,
    get_static_openai_create_func,
)

OPENAI_VERSION = VERSION

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
