from typing import Iterable, Optional, AsyncIterable

from guardrails_api_client import LLMResponse as ILLMResponse
from pydantic.config import ConfigDict


# TODO: We might be able to delete this
class LLMResponse(ILLMResponse):
    # Pydantic Config
    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_token_count: Optional[int] = None
    response_token_count: Optional[int] = None
    output: str
    stream_output: Optional[Iterable] = None
    async_stream_output: Optional[AsyncIterable] = None
